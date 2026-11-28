#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/hostStats.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace Libs::Graphics {

struct StreamBufferTestAccess {
	static bool CheckWatchRetention(GraphicContext& graphics, CommandScheduler& scheduler) {
		const uint64_t stride = std::max<uint64_t>(4, graphics.physical_device_properties.limits.nonCoherentAtomSize);
		const uint64_t size = 1024 * stride;
		StreamBuffer ring(graphics, scheduler, MemoryUsage::Upload, size);
		for (uint32_t value = 0; value < 1024; ++value) {
			if (ring.Copy(&value, sizeof(value), stride) != value * stride) return false;
			scheduler.Flush();
		}
		const auto peak = ring.m_current_watches.size();
		std::vector<uint8_t> pattern(size, 0x96);
		(void)ring.Copy(pattern.data(), size);
		scheduler.Flush();
		(void)ring.Copy(pattern.data(), size);
		return peak == 1024 && ring.m_lap == 2 && ring.m_current_watches.size() == 256 &&
		       std::memcmp(ring.Mapped().data(), pattern.data(), size) == 0;
	}
};

struct VramBudgetTestAccess {
	static void SetReportedMemory(GraphicContext& graphics, uint64_t budget, uint64_t usage) {
		graphics.m_memory_override_enabled = true;
		graphics.m_memory_budget_override  = budget;
		graphics.m_memory_usage_override   = usage;
	}

	static bool Collecting(const TextureCache& cache) { return cache.m_collecting; }

	static bool CheckParkedDepth(TextureCache& cache, uint64_t address) {
		ImageInfo info;
		info.data.address    = address;
		info.pixel_format    = vk::Format::eD16Unorm;
		info.guest_format    = Prospero::BufferFormat::k16UNorm;
		info.extent          = vk::Extent3D {64, 64, 1};
		info.pitch           = 64;
		info.bytes_per_block = sizeof(uint16_t);
		info.UpdateSize();

		Common::ScopedLock lock {cache.m_lock};
		const auto baseline = MemoryTelemetry::parked_local.load();
		ImageId second;
		uint64_t bytes = 0;
		uint64_t local = 0;
		for (size_t i = 0; i < 257; ++i) {
			const auto id = cache.InsertImage(info);
			if (i == 1) second = id;
			const auto& image = cache.m_slot_images[id];
			bytes = image.AccountedSize();
			if (i != 0 && image.backing.charge.local) local += image.backing.charge.bytes;
			cache.ParkImage(id);
		}
		const bool bounded = bytes != 0 && cache.m_parked_images.size() == 256 &&
		                     cache.m_parked_bytes == 256 * bytes && MemoryTelemetry::parked_local.load() == baseline + local;
		const auto restored = cache.UnparkImage(info);
		const bool reused = restored == second && cache.m_parked_bytes == 255 * bytes;
		cache.DiscardParkedImages(address, info.data.size);
		if (restored) cache.FreeImage(restored);
		return bounded && reused && cache.m_parked_images.empty() && cache.m_parked_bytes == 0 &&
		       MemoryTelemetry::parked_local.load() == baseline;
	}

	static bool CheckScratch(TextureCache& cache) {
		constexpr uint64_t MiB = 1024 * 1024;
		auto& tiler = cache.m_tiler;
		const auto large = tiler.GetScratchBuffer(68 * MiB);
		const auto small = tiler.GetScratchBuffer(4 * MiB, large.buffer);
		auto& scheduler = tiler.m_scheduler;
		Buffer download(tiler.m_graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, 8);
		const auto copy = [&](uint32_t value, uint64_t offset) {
			tiler.m_scratch[0]->Fill(0, 4, value);
			download.CopyFrom(scheduler.Current(), *tiler.m_scratch[0], 0, offset, 4);
		};
		copy(0x77115aae, 0);
		for (unsigned i = 0; i < 120; ++i) tiler.TrimScratch();
		const bool kept = tiler.m_scratch[0] && tiler.m_scratch[0]->Handle() == large.buffer;
		for (unsigned i = 0; i < 120; ++i) {
			(void)tiler.GetScratchBuffer(MiB);
			tiler.TrimScratch();
		}
		const bool shrank = !tiler.m_scratch[0] && tiler.m_scratch[1]->Handle() == small.buffer;
		(void)tiler.GetScratchBuffer(MiB);
		const bool small_again = tiler.m_scratch[0]->Size() == MiB;
		(void)tiler.GetScratchBuffer(68 * MiB);
		copy(0x22441234, 4);
		scheduler.Wait(scheduler.CurrentTick());
		download.Invalidate(0, 8);
		const uint32_t expected[] = {0x77115aae, 0x22441234};
		return kept && shrank && small_again && tiler.m_scratch[0]->Size() == 68 * MiB &&
		       std::memcmp(download.Mapped().data(), expected, sizeof(expected)) == 0;
	}

	static bool CheckCommandPool(GraphicContext& graphics, CommandScheduler& scheduler) {
		MasterSemaphore timeline(graphics);
		CommandScheduler::CommandPool pool(graphics, timeline);
		constexpr size_t Words = 16384;
		std::vector<uint32_t> words(Words, 0xa55a4321);
		Buffer download(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, Words * 4);
		vk::CommandBuffer held;
		for (size_t i = 0; i < 12; ++i) {
			auto [command, slot] = pool.CommitDeferred();
			vk::CommandBufferBeginInfo begin {};
			EXIT_IF(command.begin(&begin) != vk::Result::eSuccess);
			command.updateBuffer(download.Handle(), 0, Words * 4, words.data());
			vk::MemoryBarrier barrier {.srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eHostRead};
			command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
			                        {}, 1, &barrier, 0, nullptr, 0, nullptr);
			EXIT_IF(command.end() != vk::Result::eSuccess);
			if (slot == 6) held = command;
			else pool.Retire(slot, 1);
			if (slot != i) return false;
		}
		const auto before = pool.m_buffers.size();
		Common::HostStatsSampler stats;
		const auto rss_before = stats.Sample().process_resident.value_or(0);
		vk::SemaphoreSignalInfo signal {.semaphore = timeline.Handle(), .value = 1};
		EXIT_IF(graphics.device.signalSemaphore(&signal) != vk::Result::eSuccess);
		pool.TrimCompleted();
		const auto partial = pool.m_buffers.size();
		vk::SubmitInfo submit {.commandBufferCount = 1, .pCommandBuffers = &held};
		EXIT_IF(graphics.queue.submit(1, &submit, {}) != vk::Result::eSuccess);
		EXIT_IF(graphics.queue.waitIdle() != vk::Result::eSuccess);
		pool.Retire(6, 0);
		pool.TrimCompleted();
		const auto rss_after = stats.Sample().process_resident.value_or(0);
		std::printf("command buffers %zu -> %zu -> %zu, process resident bytes %llu -> %llu\n",
		            before, partial, pool.m_buffers.size(), static_cast<unsigned long long>(rss_before),
		            static_cast<unsigned long long>(rss_after));
		download.Invalidate(0, Words * 4);
		return before == 12 && partial == 7 && pool.m_buffers.size() == 4 &&
		       std::memcmp(download.Mapped().data(), words.data(), Words * 4) == 0;
	}

	static bool CheckReadbackReserve(GraphicContext& graphics, CommandScheduler& scheduler) {
		constexpr uint64_t MiB = 1024 * 1024;
		BufferCache::ReadbackQueue queue;
		vk::CommandPoolCreateInfo info {.queueFamilyIndex = graphics.queue_family};
		EXIT_IF(graphics.device.createCommandPool(&info, nullptr, &queue.pool) != vk::Result::eSuccess);
		const auto allocate = [&](uint64_t size) {
			queue.staging = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
			                                         vk::BufferUsageFlagBits::eTransferDst, size);
		};
		allocate(16 * MiB);
		for (size_t i = 0; i < 256; ++i) queue.Finish(graphics);
		const bool retained = queue.staging && queue.completed == 0;
		allocate(17 * MiB);
		queue.Finish(graphics);
		graphics.device.destroyCommandPool(queue.pool);
		return retained && !queue.staging && queue.completed == 0;
	}

	static bool MeasureArena(RenderContext& renderer) {
		auto& graphics = renderer.GetGraphics();
		auto& scheduler = renderer.GetCommandScheduler();
		constexpr uint64_t MiB = 1024 * 1024;
		RayTracing::Arena arena(graphics, scheduler);
		const auto a = arena.Allocate(24 * MiB, 256);
		const auto b = arena.Allocate(24 * MiB, 256);
		const auto c = arena.Allocate(24 * MiB, 256);
		const auto d = arena.Allocate(40 * MiB, 256);
		const auto used = arena.Used();
		const auto reserved = arena.Reserved();
		arena.Free(d);
		arena.Free(c);
		arena.Free(b);
		arena.Free(a);
		std::printf("BLAS arena MiB used/reserved %llu/%llu -> %llu/%llu\n",
		            static_cast<unsigned long long>(used / MiB), static_cast<unsigned long long>(reserved / MiB),
		            static_cast<unsigned long long>(arena.Used() / MiB), static_cast<unsigned long long>(arena.Reserved() / MiB));
		RayTracing::TlasObject tlas {};
		renderer.GetRayTracing()->CreateTlasObject(tlas, 65536);
		std::printf("native TLAS storage for 65536 instances: %llu bytes\n",
		            static_cast<unsigned long long>(tlas.storage->Size()));
		graphics.device.destroyAccelerationStructureKHR(tlas.handle);
		return used == 112 * MiB && reserved == 168 * MiB && arena.Used() == 0 && arena.Reserved() == 64 * MiB;
	}

	static uint64_t TextureStop(const TextureCache& cache) { return cache.m_stop_gc_memory; }
	static uint64_t TexturePressure(const TextureCache& cache) { return cache.m_pressure_gc_memory; }
	static uint64_t TextureCritical(const TextureCache& cache) { return cache.m_critical_gc_memory; }

	static std::vector<vk::DescriptorPool> PendingDescriptorPools(const DescriptorHeap& heap) {
		std::vector<vk::DescriptorPool> pools;
		for (const auto& [pool, tick]: heap.m_pending_pools) {
			pools.push_back(pool.pool);
		}
		return pools;
	}

	static ImageId CreateImage(TextureCache& cache, const ImageInfo& info) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.InsertImage(info);
	}

	static void MarkMixedOwnership(TextureCache& cache, ImageId id) {
		Common::ScopedLock lock {cache.m_lock};
		cache.m_slot_images[id].MarkGpuModified();
		cache.m_slot_images[id].MarkBufferModified();
	}

	static uint64_t Serial(TextureCache& cache, ImageId id) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.m_slot_images[id].serial;
	}

	static bool IsImageRegistered(TextureCache& cache, ImageId id) {
		Common::ScopedLock lock {cache.m_lock};
		const auto*        image = cache.m_slot_images.try_get(id);
		return image != nullptr && image->registered;
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t GiB          = 1024ull * 1024 * 1024;
constexpr uint64_t GuestBytes   = 0x20000;
constexpr uint64_t LazyCaseOffset = 192 * 1024;
constexpr uint64_t SpillGuestBytes = LazyCaseOffset + 2 * 256 * 256 * sizeof(uint32_t);
constexpr int      CpuReadWrite = 0x03;

const char* CheckDescriptorPoolRetirement(GraphicContext& graphics) {
	MasterSemaphore timeline(graphics);
	DescriptorHeap heap(graphics, timeline);
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                               vk::ShaderStageFlagBits::eCompute};
	vk::DescriptorSetLayoutCreateInfo create {};
	create.bindingCount = 1;
	create.pBindings = &binding;
	vk::DescriptorSetLayout layout;
	if (graphics.device.createDescriptorSetLayout(&create, nullptr, &layout) != vk::Result::eSuccess) {
		return "could not create the test descriptor layout";
	}
	DescriptorHeap::RegisterLayout(layout, std::span {&binding, 1});
	constexpr uint32_t SetsPerPool = 1024;
	for (uint32_t i = 0; i < 3 * SetsPerPool + 1; i++) {
		(void)heap.Commit(layout);
	}
	const auto first = timeline.NextTick();
	for (uint32_t i = 0; i < SetsPerPool; i++) {
		(void)heap.Commit(layout);
	}
	const auto last = timeline.NextTick();
	const auto pending = VramBudgetTestAccess::PendingDescriptorPools(heap);
	heap.TrimCompletedPools();
	const auto in_flight = VramBudgetTestAccess::PendingDescriptorPools(heap);

	vk::SemaphoreSignalInfo signal {};
	signal.semaphore = timeline.Handle();
	signal.value = first;
	EXIT_IF(graphics.device.signalSemaphore(&signal) != vk::Result::eSuccess);
	heap.TrimCompletedPools();
	const auto partial = VramBudgetTestAccess::PendingDescriptorPools(heap);

	signal.value = last;
	EXIT_IF(graphics.device.signalSemaphore(&signal) != vk::Result::eSuccess);
	heap.TrimCompletedPools();
	const auto ready = VramBudgetTestAccess::PendingDescriptorPools(heap);
	graphics.device.destroyDescriptorSetLayout(layout, nullptr);

	if (pending.size() != 4 || in_flight != pending) {
		return "descriptor trimming did not preserve pools awaiting GPU completion";
	}
	if (partial.size() != 2 || partial[0] != pending[2] || partial[1] != pending[3] ||
	    ready.size() != 1 || ready[0] != pending[3]) {
		return "descriptor trimming failed to retain one completed pool and all pending pools";
	}
	return nullptr;
}

const char* CheckBufferCharges(GraphicContext& graphics, CommandScheduler& scheduler) {
	using namespace MemoryTelemetry;
	for (const auto& [usage, category]: {std::pair {MemoryUsage::DeviceLocal, Kind::Buffer},
	                                    std::pair {MemoryUsage::Download, Kind::Download},
	                                    std::pair {MemoryUsage::Upload, Kind::Upload},
	                                    std::pair {MemoryUsage::Stream, Kind::Stream}}) {
		const auto& heaps = totals[static_cast<size_t>(category)];
		const auto bytes = [&] { return heaps[0].bytes.load() + heaps[1].bytes.load(); };
		const auto before = bytes();
		{
			Buffer buffer(graphics, scheduler, usage, 0, vk::BufferUsageFlagBits::eTransferSrc, 4096);
			if (bytes() < before + 4096) return "buffer allocation not charged to its usage";
		}
		if (bytes() != before) return "destroyed buffer retained a charge";
	}
	return nullptr;
}

const char* CheckVramBudget(GraphicContext& graphics, TextureCache& textures, BufferCache& buffers) {
	if (const auto* failure = CheckDescriptorPoolRetirement(graphics)) {
		return failure;
	}
	const auto tex_stop_large  = VramBudgetTestAccess::TextureStop(textures);
	const auto tex_pressure_large = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_large = VramBudgetTestAccess::TextureCritical(textures);

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 0);
	textures.RunGarbageCollector();
	buffers.RunGarbageCollector();

	const auto tex_stop_small  = VramBudgetTestAccess::TextureStop(textures);
	const auto tex_pressure_small = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_small = VramBudgetTestAccess::TextureCritical(textures);

	if (tex_stop_small >= tex_stop_large || tex_pressure_small >= tex_pressure_large ||
	    tex_critical_small >= tex_critical_large) {
		return "texture cache GC did not refresh thresholds when reported budget decreased";
	}
	if (tex_stop_small != 3 * GiB || tex_pressure_small != (18 * GiB + 4) / 5) {
		return "unexpected eviction thresholds under the smaller budget";
	}

	const auto check_pressure = [&](uint64_t used, bool expected) {
		VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, used);
		textures.RunGarbageCollector();
		return VramBudgetTestAccess::Collecting(textures) == expected;
	};
	if (!check_pressure(tex_pressure_small - 1, false) || !check_pressure(tex_pressure_small, true) ||
	    !check_pressure(tex_stop_small + 1, true) || !check_pressure(tex_stop_small, false) ||
	    !check_pressure(tex_stop_small + 1, false)) {
		return "eviction hysteresis failed at a budget boundary";
	}
	vk::ImageCreateInfo recycle_info {};
	recycle_info.imageType     = vk::ImageType::e2D;
	recycle_info.format        = vk::Format::eR8G8B8A8Unorm;
	recycle_info.extent        = vk::Extent3D {64, 64, 1};
	recycle_info.mipLevels     = 1;
	recycle_info.arrayLayers   = 1;
	recycle_info.samples       = vk::SampleCountFlagBits::e1;
	recycle_info.tiling        = vk::ImageTiling::eOptimal;
	recycle_info.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	recycle_info.sharingMode   = vk::SharingMode::eExclusive;
	recycle_info.initialLayout = vk::ImageLayout::eUndefined;

	graphics.TrimRecycledImages();
	const auto& charges = MemoryTelemetry::totals[static_cast<size_t>(MemoryTelemetry::Kind::Texture)];
	const auto texture_bytes = [&] { return charges[0].bytes.load() + charges[1].bytes.load(); };
	const auto baseline = texture_bytes();
	VulkanImage recycled_image {};
	if (!graphics.CreateImage(recycle_info, recycled_image)) {
		return "could not create recyclable image for recycle pool check";
	}
	const auto bytes = recycled_image.charge.bytes;
	if (bytes == 0 || texture_bytes() != baseline + bytes) return "image allocation was not counted";
	graphics.DeleteImage(recycled_image);
	if (texture_bytes() != baseline + bytes) return "recycling released a live image charge";
	if (!graphics.CreateImage(recycle_info, recycled_image) || texture_bytes() != baseline + bytes) {
		return "reusing a recycled image changed its charge";
	}
	const auto start = std::chrono::steady_clock::now();
	graphics.LogMemoryMinute("sample", 0);
	const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start);
	std::printf("memory sample including log write: %.1f us\n", elapsed.count());
	graphics.DeleteImage(recycled_image);
	graphics.TrimRecycledImages();
	if (texture_bytes() != baseline) return "trim left a dead image charge";
	if (!graphics.CreateImage(recycle_info, recycled_image)) return "could not recreate image for GC";
	graphics.DeleteImage(recycled_image);
	if (graphics.RecycledImageCount() == 0) {
		return "deleted recyclable image did not enter the recycle pool";
	}

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 6 * GiB);
	textures.RunGarbageCollector();
	if (graphics.RecycledImageCount() != 0) {
		return "pressure GC pass did not drain the recycled image pool";
	}
	return nullptr;
}

bool CanSpill(const GraphicContext& graphics, const VulkanImage& backing) {
	if (!graphics.spill_allocator) return false;
	const auto types = graphics.device.getImageMemoryRequirements(backing.image).memoryTypeBits;
	for (uint32_t type = 0; type < graphics.physical_device_memory_properties.memoryTypeCount; ++type) {
		if ((types & (uint32_t {1} << type)) && !graphics.IsLocalMemoryType(type)) return true;
	}
	return false;
}

const char* CheckSpillContent(GraphicContext& graphics, CommandScheduler& scheduler, TextureCache& textures,
                             uint64_t address, uint32_t width, uint32_t levels) {
	TextureCache::ImageDesc desc {};
	desc.info.data.address = address;
	desc.info.pixel_format = vk::Format::eR32Uint;
	desc.info.guest_format = Prospero::BufferFormat::k32UInt;
	desc.info.extent = vk::Extent3D {width, width, 1};
	desc.info.pitch = width;
	desc.info.resources.levels = levels;
	desc.info.UpdateSize();
	desc.type = TextureCache::BindingType::Storage;
	desc.view_info.format = desc.info.pixel_format;
	desc.view_info.usage = vk::ImageUsageFlagBits::eStorage;
	const auto bytes = desc.info.data.size;
	std::vector<uint8_t> expected(2 * bytes, 0xa5);
	Libs::LibKernel::Memory::WriteBacking(address, expected.data(), expected.size());
	for (size_t i = 0; i < bytes; ++i) expected[i] = static_cast<uint8_t>((i * 37) ^ (i >> 7));

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 0);
	const auto source = textures.FindImage(desc);
	if (!textures.FindTexture(source, desc)) return "could not bind storage before the spill";
	auto& image = textures.GetImage(source);
	const bool source_local = image.backing.charge.local;
	if (image.backing.spilled) return "source image unexpectedly spilled below threshold";
	Buffer upload(graphics, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc, bytes);
	std::memcpy(upload.Mapped().data(), expected.data(), bytes);
	upload.Flush(0, bytes);
	image.Upload(desc.info.BufferCopies(), upload.Handle(), 0, bytes);

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 5 * GiB);
	desc.type = TextureCache::BindingType::Texture;
	desc.info.data.size = 0;
	desc.info.resources.layers = 2;
	desc.info.UpdateSize();
	desc.view_info.usage = vk::ImageUsageFlagBits::eSampled;
	desc.view_info.type = vk::ImageViewType::e2DArray;
	desc.view_info.layer_count = 2;
	const auto replacement = textures.FindImage(desc);
	const auto view = textures.FindTexture(replacement, desc);
	auto& expanded = textures.GetImage(replacement);
	const bool placed = !CanSpill(graphics, expanded.backing) ||
	                    (expanded.backing.spilled && !expanded.backing.charge.local);
	std::printf("texture with %u levels, %llu bytes per layer: %s -> %s\n", levels,
	            static_cast<unsigned long long>(bytes), source_local ? "local" : "system",
	            expanded.backing.charge.local ? "local" : "system");
	Buffer download(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, expected.size());
	expanded.Download(desc.info.BufferCopies(), download.Handle(), 0, expected.size());
	textures.ProcessDownloadImages();
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
	download.Invalidate(0, expected.size());
	if (source == replacement || !view || !placed || VramBudgetTestAccess::IsImageRegistered(textures, source)) {
		return "cache did not replace the image with the requested backing";
	}
	if (std::memcmp(download.Mapped().data(), expected.data(), expected.size()) != 0) {
		return "replacement image lost texels or changed the added layer";
	}
	textures.FlushLazyReadback(address, bytes);
	std::vector<uint8_t> guest(expected.size());
	if (!Libs::LibKernel::Memory::TryReadBacking(address, guest.data(), guest.size()) || guest != expected) {
		return "replacement lost guest readback of GPU-written texels";
	}
	return nullptr;
}

const char* CheckGuestEviction(GraphicContext& graphics, TextureCache& textures, uint64_t address) {
	if (!VramBudgetTestAccess::CheckParkedDepth(textures, address)) return "parked depth image budget or reuse";
	ImageInfo info;
	info.data.address = address;
	info.pixel_format = vk::Format::eR8G8B8A8Unorm;
	info.guest_format = Prospero::BufferFormat::k8_8_8_8UNorm;
	info.extent       = vk::Extent3D {64, 64, 1};
	info.pitch        = 64;
	info.UpdateSize();

	TextureCache::ImageDesc desc {.info = info};
	desc.view_info.format = info.pixel_format;
	const auto id = textures.FindImage(desc);
	if (textures.GetImage(id).backing.spilled) {
		return "ordinary texture spilled below the pressure threshold";
	}
	if (!VramBudgetTestAccess::IsImageRegistered(textures, id)) {
		return "test image was not registered";
	}

	info.data.address += info.data.size;
	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 5 * GiB);
	desc.info = info;
	const auto mixed = textures.FindImage(desc);
	const auto& crowded = textures.GetImage(mixed).backing;
	const bool can_spill = CanSpill(graphics, crowded);
	if (crowded.charge.kind != MemoryTelemetry::Kind::Texture || (can_spill && (!crowded.spilled || crowded.charge.local))) {
		return "ordinary texture did not use compatible system memory under pressure";
	}
	VramBudgetTestAccess::MarkMixedOwnership(textures, mixed);
	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 0);
	textures.RunGarbageCollector();
	textures.RunGarbageCollector();
	info.data.address += info.data.size;
	const auto unread = VramBudgetTestAccess::CreateImage(textures, info);
	info.data.address += info.data.size;
	const auto unread_mixed = VramBudgetTestAccess::CreateImage(textures, info);
	VramBudgetTestAccess::MarkMixedOwnership(textures, unread_mixed);
	info.data.address += info.data.size;
	const auto renewed = VramBudgetTestAccess::CreateImage(textures, info);
	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 6 * GiB);
	info.data.address += info.data.size;
	desc.info = info;
	desc.type = TextureCache::BindingType::RenderTarget;
	desc.view_info.usage = vk::ImageUsageFlagBits::eColorAttachment;
	const auto target = textures.FindImage(desc);
	if (textures.GetImage(target).backing.spilled) return "new color target spilled under pressure";

	desc.type = TextureCache::BindingType::Texture;
	desc.heap = true;
	desc.info.resources.layers = 2;
	desc.info.UpdateSize();
	desc.view_info.usage = vk::ImageUsageFlagBits::eSampled;
	const auto expanded = textures.FindImage(desc);
	const auto& backing = textures.GetImage(expanded).backing;
	if (expanded == target || backing.charge.kind != MemoryTelemetry::Kind::HeapTexture ||
	    (can_spill && (!backing.spilled || backing.charge.local))) {
		return "expanded heap texture lost its allocation policy under pressure";
	}

	textures.EvictUnread(unread, VramBudgetTestAccess::Serial(textures, unread));
	textures.EvictUnread(unread_mixed, VramBudgetTestAccess::Serial(textures, unread_mixed));
	textures.EvictUnread(renewed, VramBudgetTestAccess::Serial(textures, renewed) + 1u);
	textures.RunGarbageCollector();
	if (VramBudgetTestAccess::IsImageRegistered(textures, unread)) {
		return "an unread texture handed over by the image heap was not evicted in the same pass";
	}
	if (!VramBudgetTestAccess::IsImageRegistered(textures, unread_mixed)) {
		return "the unread eviction discarded an image with GPU writes and mixed ownership";
	}
	if (!VramBudgetTestAccess::IsImageRegistered(textures, renewed)) {
		return "the unread eviction freed an image whose serial no longer matches";
	}
	for (uint32_t i = 0; i < 4; i++) {
		textures.RunGarbageCollector();
	}
	if (VramBudgetTestAccess::IsImageRegistered(textures, id)) {
		return "critical memory pressure did not evict a recent unused texture";
	}
	if (!VramBudgetTestAccess::IsImageRegistered(textures, mixed)) {
		return "GC discarded an image with GPU writes and mixed ownership";
	}
	return nullptr;
}

bool CheckScheduledPoolTrim(RenderContext& renderer, CommandProcessor& producer, bool deferred) {
	auto& graphics = renderer.GetGraphics();
	CommandScheduler scheduler(renderer, graphics, deferred);
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	const uint32_t expected[] = {0x2761548c, 0xd328cbe1};
	Buffer output(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, sizeof(expected));
	output.Fill(0, 4, expected[0]);
	scheduler.TrimCommandPool();
	output.Fill(4, 4, expected[1]);
	scheduler.Wait(scheduler.CurrentTick());
	output.Invalidate(0, sizeof(expected));
	return std::memcmp(output.Mapped().data(), expected, sizeof(expected)) == 0;
}

bool CheckLocalMemoryTarget() {
	constexpr uint64_t MiB = 1ull << 20;
	constexpr uint64_t size = 16 * GiB;
	const auto target = [&](uint64_t used) {
		return LocalMemoryBudget::Target(13 * GiB, 10 * GiB, size, BoardMemory {size, used});
	};
	return target(10 * GiB) == 13 * GiB && target(13 * GiB) == 11776 * MiB &&
	       target(14 * GiB) == 10752 * MiB && target(9 * GiB) == 13 * GiB &&
	       LocalMemoryBudget::Target(13 * GiB, 0, size, BoardMemory {size, size}) == 0 &&
	       LocalMemoryBudget::Target(13 * GiB, 10 * GiB, size, std::nullopt) == 11776 * MiB;
}

const char* CheckVramCase(GraphicContext& graphics, uint64_t address, bool acceleration_storage, bool spill_content) {
	if (!CheckLocalMemoryTarget()) return "incorrect pressure target for shared device memory";
	if (graphics.memory_budget_ext_enabled && graphics.GetLocalMemoryUsage().budget == 0) {
		return "the driver reported an empty device-local memory budget";
	}
	VramBudgetTestAccess::SetReportedMemory(graphics, (address == 0 ? 16 : 6) * GiB, 0);

	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	if (const auto* failure = CheckBufferCharges(graphics, scheduler)) return failure;
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	if (address == 0) {
		for (const bool deferred: {false, true}) {
			if (!CheckScheduledPoolTrim(renderer, producer, deferred)) return "scheduled command pool trim lost a GPU write";
		}
		if (!VramBudgetTestAccess::CheckCommandPool(graphics, scheduler)) return "command pool retirement lost a recording";
		if (!StreamBufferTestAccess::CheckWatchRetention(graphics, scheduler)) return "stream ring did not release idle timeline watches";
		if (!VramBudgetTestAccess::CheckReadbackReserve(graphics, scheduler)) return "readback retained an oversized buffer";
		if (!VramBudgetTestAccess::CheckScratch(textures)) return "scratch did not shrink and regrow after a burst";
		if (renderer.GetBufferCache().GetUtilityBuffer(MemoryUsage::Upload).Size() != (256ull << 20)) return "upload ring capacity";
		if (acceleration_storage && !VramBudgetTestAccess::MeasureArena(renderer)) return "acceleration storage retention";
	}
	if (spill_content) {
		if (const auto* failure = CheckSpillContent(graphics, scheduler, textures, address, 128, 2)) return failure;
		return CheckSpillContent(graphics, scheduler, textures, address + LazyCaseOffset, 256, 1);
	}
	const auto* failure = address == 0 ? CheckVramBudget(graphics, textures, renderer.GetBufferCache())
	                                  : CheckGuestEviction(graphics, textures, address);
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	return failure;
}

int RunVramTest(bool mapped, bool wait_for_minute, bool spill_content) {
	const uint64_t guest_bytes = spill_content ? SpillGuestBytes : GuestBytes;
	const char* name = spill_content ? "kyty_shim_texture_spill_content" :
	                   mapped ? "kyty_shim_vram_budget_guest_eviction" : "kyty_shim_vram_budget";
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize(false, true, true, !mapped)) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s)\n",
		             name, required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", name);
		return 1;
	}

	uint64_t address = 0;
	if (mapped) {
		void* guest = nullptr;
		if (sceKernelMapFlexibleMemory(&guest, guest_bytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
			std::fprintf(stderr, "%s: FAILED: could not map guest backing\n", name);
			return 1;
		}
		address = reinterpret_cast<uint64_t>(guest);
	}
	const bool invalid_range = mapped && !GuestRange {address, guest_bytes}.Valid();
	gpu.graphics.StartMemoryMinutes();
	const char* failure = invalid_range ? nullptr : CheckVramCase(gpu.graphics, address, gpu.acceleration_storage, spill_content);
	gpu.graphics.TrimRecycledImages();
	if (wait_for_minute) std::this_thread::sleep_for(std::chrono::seconds(61));
	gpu.graphics.StopMemoryMinutes();
	if (!failure) {
		for (const auto& heaps: MemoryTelemetry::totals) {
			for (const auto& total: heaps) {
				if (total.bytes != 0 || total.count != 0) failure = "destroyed renderer retained an allocation charge";
			}
		}
		if (MemoryTelemetry::resident != 0 || MemoryTelemetry::imported != 0 || MemoryTelemetry::pipeline_ram != 0 ||
		    MemoryTelemetry::live_objects<vk::Pipeline> != 0 || MemoryTelemetry::live_objects<vk::PipelineCache> != 0 ||
		    MemoryTelemetry::live_objects<vk::DescriptorPool> != 0 || MemoryTelemetry::live_objects<vk::CommandPool> != 0) {
			failure = "destroyed renderer retained a memory or object charge";
		}
	}
	if (mapped && sceKernelMunmap(address, guest_bytes) != 0) {
		std::fprintf(stderr, "%s: FAILED: could not unmap guest backing\n", name);
		return 1;
	}
	if (invalid_range) {
		std::fprintf(stderr, "%s: skipped, guest range 0x%llx is outside Kyty bounds\n",
		             name, static_cast<unsigned long long>(address));
		return 77;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "%s: FAILED: %s\n", name, failure);
		return 1;
	}
	std::printf("%s: ok\n", name);
	return 0;
}

}

extern "C" int RunKytyVramBudgetTest(bool guest_eviction, bool wait_for_minute) {
	return RunVramTest(guest_eviction, wait_for_minute, false);
}

extern "C" int RunKytyTextureSpillContentTest() {
	return RunVramTest(true, false, true);
}

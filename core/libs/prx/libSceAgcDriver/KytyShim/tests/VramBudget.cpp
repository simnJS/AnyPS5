#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

struct VramBudgetTestAccess {
	static void SetReportedMemory(GraphicContext& graphics, uint64_t budget, uint64_t usage) {
		graphics.m_memory_override_enabled = true;
		graphics.m_memory_budget_override  = budget;
		graphics.m_memory_usage_override   = usage;
	}

	static uint64_t TextureTrigger(const TextureCache& cache) { return cache.m_trigger_gc_memory; }
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
	const auto tex_trigger_large  = VramBudgetTestAccess::TextureTrigger(textures);
	const auto tex_pressure_large = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_large = VramBudgetTestAccess::TextureCritical(textures);

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 0);
	textures.RunGarbageCollector();
	buffers.RunGarbageCollector();

	const auto tex_trigger_small  = VramBudgetTestAccess::TextureTrigger(textures);
	const auto tex_pressure_small = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_small = VramBudgetTestAccess::TextureCritical(textures);

	if (tex_trigger_small >= tex_trigger_large || tex_pressure_small >= tex_pressure_large ||
	    tex_critical_small >= tex_critical_large) {
		return "texture cache GC did not refresh thresholds when reported budget decreased";
	}
	if (tex_trigger_small != 0) {
		return "texture cache trigger threshold was non-zero at <= 8 GiB budget";
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
	graphics.LogMemoryMinute("sample", 0);
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

const char* CheckGuestEviction(GraphicContext& graphics, TextureCache& textures, uint64_t address) {
	ImageInfo info;
	info.data.address = address;
	info.pixel_format = vk::Format::eR8G8B8A8Unorm;
	info.guest_format = Prospero::BufferFormat::k8_8_8_8UNorm;
	info.extent       = vk::Extent3D {64, 64, 1};
	info.pitch        = 64;
	info.UpdateSize();

	const auto id = VramBudgetTestAccess::CreateImage(textures, info);
	if (!VramBudgetTestAccess::IsImageRegistered(textures, id)) {
		return "test image was not registered";
	}

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 6 * GiB);
	for (uint32_t i = 0; i < 4; i++) {
		textures.RunGarbageCollector();
	}
	if (VramBudgetTestAccess::IsImageRegistered(textures, id)) {
		return "critical memory pressure did not evict a recent unused texture";
	}
	return nullptr;
}

const char* CheckVramCase(GraphicContext& graphics, uint64_t address) {
	VramBudgetTestAccess::SetReportedMemory(graphics, (address == 0 ? 16 : 6) * GiB, 0);

	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	if (const auto* failure = CheckBufferCharges(graphics, scheduler)) return failure;
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const auto* failure = address == 0 ? CheckVramBudget(graphics, textures, renderer.GetBufferCache())
	                                  : CheckGuestEviction(graphics, textures, address);
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	return failure;
}

}

extern "C" int RunKytyVramBudgetTest(bool guest_eviction) {
	const char* name = guest_eviction ? "kyty_shim_vram_budget_guest_eviction" : "kyty_shim_vram_budget";
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize(false, true, true)) {
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
	if (guest_eviction) {
		void* guest = nullptr;
		if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
			std::fprintf(stderr, "%s: FAILED: could not map guest backing\n", name);
			return 1;
		}
		address = reinterpret_cast<uint64_t>(guest);
	}
	const bool invalid_range = guest_eviction && !GuestRange {address, GuestBytes}.Valid();
	const char* failure = invalid_range ? nullptr : CheckVramCase(gpu.graphics, address);
	if (guest_eviction && sceKernelMunmap(address, GuestBytes) != 0) {
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

#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics {

struct UploadBatchFlushTestAccess {
	static uint64_t StagingSize(const BufferCache& cache) { return cache.m_staging_buffer.Size(); }

	static uint64_t StagingLap(const BufferCache& cache) { return cache.m_staging_buffer.Lap(); }

	static bool AdvanceStagingToRemaining(BufferCache& cache, uint64_t remaining) {
		auto& staging = cache.m_staging_buffer;
		if (staging.Size() <= remaining) {
			return false;
		}
		const uint64_t fill = staging.Size() - remaining;
		const auto [mapped, offset] = staging.Map(fill, 16, true);
		if (mapped == nullptr) {
			return false;
		}
		staging.Commit();
		return true;
	}

	static void UploadChunked(BufferCache& cache, Buffer& buffer, std::span<vk::BufferCopy> copies,
	                          uint64_t total_size, uint64_t max_chunk, bool batched) {
		cache.m_upload_batching   = batched;
		cache.m_upload_batch_tick = cache.m_scheduler.CurrentTick();
		cache.UploadAndRecordCopies(buffer, copies, total_size, max_chunk);
		if (batched) {
			cache.FlushUploadBatch();
		}
	}
};

}

namespace {

using namespace Libs::Graphics;
using Access = UploadBatchFlushTestAccess;

constexpr uint64_t MiB          = 1024ull * 1024;
constexpr uint64_t Region       = TRACKER_REGION_SIZE;
constexpr size_t   RegionCount  = 4;
constexpr uint64_t GuestBytes   = (RegionCount + 1) * Region;
constexpr int      CpuReadWrite = 0x03;

uint32_t TotalHeapAllocations(const GraphicContext& graphics) {
	std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
	const auto count = graphics.GetHeapUsage(heaps);
	uint32_t   total = 0;
	for (uint32_t heap = 0; heap < count; ++heap) {
		total += heaps[heap].allocations;
	}
	return total;
}

uint32_t PatternWord(uint32_t round, size_t region_index, size_t dword_index) {
	return (round << 24u) ^ (static_cast<uint32_t>(region_index + 1) * 0x9e3779b1u) ^
	       static_cast<uint32_t>(dword_index * 0x85ebca6bu);
}

void FillRegion(uint64_t address, uint32_t round, size_t region_index) {
	const size_t          count = static_cast<size_t>(Region / sizeof(uint32_t));
	std::vector<uint32_t> words(count);
	for (size_t i = 0; i < count; ++i) {
		words[i] = PatternWord(round, region_index, i);
	}
	Libs::LibKernel::Memory::WriteBacking(address, words.data(), Region);
}

const char* VerifyBuffers(CommandScheduler& scheduler, BufferCache& buffers, Buffer& download,
                          uint64_t base, uint32_t round, size_t count) {
	for (size_t r = 0; r < count; ++r) {
		const auto id = buffers.RegisteredBuffer(base + r * Region, Region);
		if (!id) {
			return "uploaded buffer was not registered";
		}
		auto& target = buffers.GetBuffer(id);
		download.CopyFrom(scheduler.Current(), target, target.Offset(base + r * Region), 0, Region);
		scheduler.Finish();
		download.Invalidate(0, Region);
		const auto* words = reinterpret_cast<const uint32_t*>(download.Mapped().data());
		const size_t dwords = static_cast<size_t>(Region / sizeof(uint32_t));
		for (size_t i = 0; i < dwords; ++i) {
			if (words[i] != PatternWord(round, r, i)) {
				return "uploaded GPU buffer bytes did not match guest memory after staging wrap";
			}
		}
	}
	return nullptr;
}

const char* CheckUploadBatchFlush(GraphicContext& graphics, uint64_t base) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& buffers   = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	if (Access::StagingSize(buffers) < 2 * Region) {
		return "staging buffer is smaller than expected";
	}

	Buffer download(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	                Region);
	RangeSet ranges;
	std::array<std::pair<uint64_t, uint64_t>, RegionCount> spans {};
	for (size_t r = 0; r < RegionCount; ++r) {
		const uint64_t addr = base + r * Region;
		FillRegion(addr, 1u, r);
		(void)buffers.FindBuffer(addr, Region);
		ranges.Add(addr, Region);
		spans[r] = {addr, addr + Region};
	}

	if (!Access::AdvanceStagingToRemaining(buffers, 2 * Region)) {
		return "could not advance staging buffer to pre-wrap boundary";
	}
	const uint64_t lap_before    = Access::StagingLap(buffers);
	const uint32_t allocs_before = TotalHeapAllocations(graphics);

	buffers.SynchronizeBuffersInRanges(ranges, false);

	const uint64_t lap_after    = Access::StagingLap(buffers);
	const uint32_t allocs_after = TotalHeapAllocations(graphics);
	if (lap_after <= lap_before) {
		return "SynchronizeBuffersInRanges did not wrap the staging buffer";
	}
	if (allocs_after > allocs_before) {
		return "SynchronizeBuffersInRanges allocated temporary Upload buffers instead of flushing the batch";
	}
	if (const char* fail = VerifyBuffers(scheduler, buffers, download, base, 1u, RegionCount)) {
		return fail;
	}

	for (size_t r = 0; r < RegionCount; ++r) {
		const uint64_t addr = base + r * Region;
		FillRegion(addr, 2u, r);
		buffers.InvalidateMemory(addr, Region);
	}
	if (!Access::AdvanceStagingToRemaining(buffers, Region)) {
		return "could not advance staging buffer before span sync";
	}
	scheduler.Flush();
	if (!Access::AdvanceStagingToRemaining(buffers, Region)) {
		return "could not wrap staging buffer before span sync";
	}
	const uint64_t span_lap_before    = Access::StagingLap(buffers);
	const uint32_t span_allocs_before = TotalHeapAllocations(graphics);

	buffers.SynchronizeBuffersInSpans(spans);

	if (Access::StagingLap(buffers) <= span_lap_before) {
		return "SynchronizeBuffersInSpans did not wrap the staging buffer";
	}
	if (TotalHeapAllocations(graphics) > span_allocs_before) {
		return "SynchronizeBuffersInSpans allocated temporary Upload buffers";
	}
	if (const char* fail = VerifyBuffers(scheduler, buffers, download, base, 2u, RegionCount)) {
		return fail;
	}

	BufferCache::SetRegions set_state;
	for (size_t r = 0; r < RegionCount; ++r) {
		const uint64_t addr = base + r * Region;
		FillRegion(addr, 3u, r);
		buffers.InvalidateMemory(addr, Region);
	}
	if (!Access::AdvanceStagingToRemaining(buffers, Region)) {
		return "could not advance staging buffer before set sync";
	}
	const uint64_t set_lap_before    = Access::StagingLap(buffers);
	const uint32_t set_allocs_before = TotalHeapAllocations(graphics);

	buffers.SynchronizeBuffersInSet(ranges, set_state);

	if (Access::StagingLap(buffers) <= set_lap_before) {
		return "SynchronizeBuffersInSet did not wrap the staging buffer";
	}
	if (TotalHeapAllocations(graphics) > set_allocs_before) {
		return "SynchronizeBuffersInSet allocated temporary Upload buffers";
	}
	if (const char* fail = VerifyBuffers(scheduler, buffers, download, base, 3u, RegionCount)) {
		return fail;
	}

	for (const bool batched: {false, true}) {
		const uint32_t round = batched ? 5u : 4u;
		FillRegion(base, round, 0);
		buffers.InvalidateMemory(base, Region);
		const auto id = buffers.RegisteredBuffer(base, Region);
		if (!id) {
			return "target buffer missing for chunked upload check";
		}
		auto& target = buffers.GetBuffer(id);
		constexpr uint64_t c0 = 1536ull * 1024;
		constexpr uint64_t c1 = 2048ull * 1024;
		static_assert(Region > c0 + c1);
		const uint64_t c2 = Region - c0 - c1;
		std::array<vk::BufferCopy, 3> copies {
		    vk::BufferCopy {0, target.Offset(base), c0},
		    vk::BufferCopy {c0, target.Offset(base + c0), c1},
		    vk::BufferCopy {c0 + c1, target.Offset(base + c0 + c1), c2},
		};
		const uint32_t chunk_allocs_before = TotalHeapAllocations(graphics);
		Access::UploadChunked(buffers, target, copies, Region, 1 * MiB, batched);
		if (TotalHeapAllocations(graphics) > chunk_allocs_before) {
			return "chunked upload allocated temporary Upload buffers";
		}
		if (const char* fail = VerifyBuffers(scheduler, buffers, download, base, round, 1)) {
			return fail;
		}
	}

	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	(void)buffers.DropFreedRange(base, RegionCount * Region, nullptr, nullptr);
	return nullptr;
}

}

extern "C" int RunKytyUploadBatchFlushTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize()) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_upload_batch_flush: %s, no usable Vulkan device (%s)\n",
		             required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fputs("kyty_shim_upload_batch_flush: FAILED: could not create the allocator\n", stderr);
		return 1;
	}
	void* guest = nullptr;
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fputs("kyty_shim_upload_batch_flush: FAILED: could not map guest backing\n", stderr);
		return 1;
	}
	const uint64_t raw         = reinterpret_cast<uint64_t>(guest);
	const uint64_t base        = (raw + Region - 1) / Region * Region;
	const bool     valid_guest = GuestRange {base, RegionCount * Region}.Valid();
	const char*    fail        = valid_guest ? CheckUploadBatchFlush(gpu.graphics, base) : nullptr;
	if (sceKernelMunmap(raw, GuestBytes) != 0) {
		std::fputs("kyty_shim_upload_batch_flush: FAILED: could not unmap guest backing\n", stderr);
		return 1;
	}
	if (!valid_guest) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_upload_batch_flush: %s: guest range 0x%llx is outside Kyty bounds\n",
		             required ? "FAILED" : "skipped", static_cast<unsigned long long>(base));
		return required ? 1 : 77;
	}
	if (fail != nullptr) {
		std::fprintf(stderr, "kyty_shim_upload_batch_flush: FAILED: %s\n", fail);
		return 1;
	}
	std::puts("kyty_shim_upload_batch_flush: ok");
	return 0;
}

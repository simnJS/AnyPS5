#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "gpu_tiler_shaders/buffer_scatter_spv.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/spscQueue.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/scout.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/guestMirror.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/execHelpers.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write,
                                        bool wait = true) {
	const auto [mapped, offset] = staging.Map(size, alignment, wait);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferHostOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

constexpr uint64_t ReadAheadLimit = 256 * 1024;

struct ReadbackStats {
	std::atomic<uint64_t> cp_reads {0};
	std::atomic<uint64_t> cp_ns {0};
	std::atomic<uint64_t> cp_ordered {0};
	std::atomic<uint64_t> cp_ordered_ns {0};
	std::atomic<uint64_t> cp_not_waited {0};
	std::atomic<uint64_t> other_reads {0};
	std::atomic<uint64_t> other_ns {0};
	std::atomic<uint64_t> other_gpu_bytes {0};
	std::atomic<uint64_t> other_false_sharing {0};
	std::atomic<uint64_t> cp_staged {0};
	std::atomic<uint64_t> other_replayed {0};
	std::atomic<uint64_t> other_dispatch_ns {0};
	std::atomic<uint64_t> other_wait_ns {0};
	std::atomic<uint64_t> other_release_ns {0};
	std::atomic<uint64_t> released_async {0};
	std::atomic<uint64_t> replayed_after_copy {0};
	std::atomic<uint64_t> waited_after_copy {0};
	std::atomic<bool>     line_overflow {false};
	std::atomic<int64_t>  last_line {0};
};

ReadbackStats g_readback_stats;

struct WriteBlockStats {
	std::atomic<uint64_t> faults {0};
	std::atomic<uint64_t> blocks {0};
	std::atomic<uint64_t> pages {0};
	std::atomic<uint64_t> uploaded {0};
	std::atomic<uint64_t> command {0};
	std::atomic<uint64_t> segment {0};
	std::atomic<uint64_t> gpu {0};
	std::atomic<uint64_t> image {0};
	std::atomic<uint64_t> buffer {0};
	std::atomic<uint64_t> other {0};
	std::atomic<uint64_t> already_open {0};
	std::atomic<uint64_t> command_image {0};
	std::atomic<int64_t>  last_line {0};
};

WriteBlockStats g_write_block_stats;


struct ReadbackReplay {
	bool     allowed   = false;
	bool     published = false;
	uint64_t begin     = 0;
	uint64_t end       = 0;
};

thread_local ReadbackReplay g_readback_replay;

int64_t SteadyNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

bool MirrorCheck() {
	static const bool enabled = [] {
		const char*            value = std::getenv("KYTY_MIRROR_CHECK");
		const std::string_view text {value == nullptr ? "" : value};
		if (!text.empty() && text != "0" && text != "1") {
			EXIT("invalid KYTY_MIRROR_CHECK: %s (0 or 1)\n", value);
		}
		return text == "1";
	}();
	return enabled;
}

std::vector<vk::BufferMemoryBarrier> CopyBarriers(std::span<const vk::BufferCopy> copies, bool source,
                                                  const vk::BufferMemoryBarrier& model) {
	constexpr size_t   MaxBarriers = 16;
	constexpr uint64_t FirstGap    = 64 * 1024;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	ranges.reserve(copies.size());
	for (const auto& copy: copies) {
		const auto offset = source ? copy.srcOffset : copy.dstOffset;
		ranges.emplace_back(offset, offset + copy.size);
	}
	std::sort(ranges.begin(), ranges.end());
	for (uint64_t gap = 0;; gap = gap == 0 ? FirstGap : gap * 4) {
		size_t count = 0;
		for (size_t index = 0; index < ranges.size(); index++) {
			if (count != 0 && ranges[index].first <= ranges[count - 1].second + gap) {
				ranges[count - 1].second = std::max(ranges[count - 1].second, ranges[index].second);
			} else {
				ranges[count++] = ranges[index];
			}
		}
		ranges.resize(count);
		if (count <= MaxBarriers) break;
	}
	std::vector<vk::BufferMemoryBarrier> barriers(ranges.size(), model);
	for (size_t index = 0; index < ranges.size(); index++) {
		barriers[index].offset = ranges[index].first;
		barriers[index].size   = ranges[index].second - ranges[index].first;
	}
	return barriers;
}

}

void BufferCache::CollectDownloadCopies(uint64_t vaddr, uint64_t size, std::vector<vk::BufferCopy>& copies,
                                        uint64_t& total_size) {
	const auto& buffer = *m_storage;
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(buffer.Offset(start), total_size, end - start);
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    if (m_gpu_modified_ranges.Subtract(address, bytes)) {
			    NoteGpuModifiedChange(address, bytes, false);
		    }
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (!copies.empty()) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);
	}
}

template <bool async>
bool BufferCache::DownloadMemory(uint64_t vaddr, uint64_t size) {
	const auto&                 buffer = *m_storage;
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.HandleOrigin();
	CollectDownloadCopies(vaddr, size, copies, total_size);
	if (copies.empty()) {
		return false;
	}
	CheckMirrorCopies(copies);

	if constexpr (!async) {
		if (DownloadOnReadbackQueue(copies, total_size)) {
			return true;
		}
	}
	auto download = m_download_buffer.MapDownload(total_size, 64);
	for (auto& copy: copies) {
		copy.dstOffset += download.offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	const auto barriers        = CopyBarriers(copies, true, before);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr,
	                       static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
	native.copyBuffer(buffer.Handle(), download.buffer->Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.buffer->Handle();
	after.offset        = download.offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	uint64_t written_begin = UINT64_MAX;
	uint64_t written_end   = 0;
	for (const auto& copy: copies) {
		written_begin = std::min(written_begin, buffer_address + copy.srcOffset);
		written_end   = std::max(written_end, buffer_address + copy.srcOffset + copy.size);
	}
	auto publish = [total_size, buffer_address, copies = std::move(copies),
	                download = std::move(download)] {
		download.buffer->Invalidate(download.offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(
			    buffer_address + copy.srcOffset,
			    download.mapped + (copy.dstOffset - download.offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish), written_begin,
		                                   written_end - written_begin);
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache, RenderContext* resources)
    : m_graphics(graphics), m_scheduler(scheduler), m_resources(resources),
      m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_null_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	m_scheduler.BeginCommand();
	m_fault_manager.PrepareFaultBuffer();
	SetVulkanObjectNameF(m_graphics.device, m_null_buffer.Handle(), "Kyty.NullBuffer");
	m_null_buffer.Fill(0, 16, 0);
	if (const auto refusal = GuestMirror::Unsupported(m_graphics); !refusal.empty()) {
		EXIT("[kyty-mirror] no mirror: %s; AnyKyty keeps the guest memory the GPU reads in a sparse buffer and needs "
		     "sparseBinding, sparseResidencyBuffer and a queue that binds sparse memory\n",
		     refusal.c_str());
	}
	const std::array<uint64_t, 1> chosen {m_graphics.mirror_size};
	m_guest_mirror = m_graphics.mirror_size != 0
	                     ? std::make_unique<GuestMirror>(m_graphics, chosen, m_graphics.mirror_base)
	                     : std::make_unique<GuestMirror>(m_graphics, GuestMirror::DefaultSizes, m_graphics.mirror_base);
	::Log::WriteToConsoleAndLog(fmt::format("[kyty-mirror] {}\n", m_guest_mirror->Description()));
	if (!m_guest_mirror->Available()) {
		EXIT("[kyty-mirror] %s; AnyKyty keeps the guest memory the GPU reads in a sparse buffer of 64 GiB or more and "
		     "cannot run on this GPU\n",
		     m_guest_mirror->Description().c_str());
	}
	SetVulkanObjectNameF(m_graphics.device, m_guest_mirror->Handle(), "Kyty.GuestMirror[size=0x{:x}]",
	                     m_guest_mirror->Size());
	m_storage = std::make_unique<Buffer>(
	    m_graphics, m_scheduler, m_guest_mirror->Base(), m_guest_mirror->Size(),
	    MirrorWindow {m_guest_mirror->Handle(), m_guest_mirror->DeviceAddress(), m_guest_mirror->Base()});
	m_host_view = std::make_unique<GuestHostView>(m_graphics, m_scheduler, m_staging_buffer, m_guest_mirror->Base(),
	                                              m_guest_mirror->Size());
	const uint64_t metadata[] {m_guest_mirror->DeviceAddress() - m_guest_mirror->Base(), m_guest_mirror->Base(),
	                           m_guest_mirror->Size(), 0,
	                           FAULT_BITMAP_OFFSET / sizeof(uint32_t) | (USE_BITMAP_OFFSET / sizeof(uint32_t)) << 32u,
	                           WRITE_BITMAP_OFFSET / sizeof(uint32_t), LibKernel::Memory::kExtendedMemoryBase,
	                           LOWER_ADDRESS_SIZE, m_host_view->TableAddress()};
	static_assert(sizeof(metadata) == HOST_TABLE_OFFSET + sizeof(uint64_t) && sizeof(metadata) <= RESIDENCY_OFFSET);
	const auto source = m_staging_buffer.Copy(metadata, sizeof(metadata), alignof(uint64_t));
	GetFaultBuffer()->CopyFrom(m_scheduler.Current(), m_staging_buffer, source, 0, sizeof(metadata));
	m_residency_words.resize((m_guest_mirror->Size() / MIRROR_BLOCK_SIZE + 31) / 32);
	m_scheduler.SetSubmitHook(
	    [](void* self, CommandBuffer& command, SubmitInfo& submit) {
		    auto* cache = static_cast<BufferCache*>(self);
		    cache->RecordMirrorCopies(command);
		    cache->SubmitGuestMirror(submit);
	    },
	    this);
}

BufferCache::~BufferCache() {
	m_scheduler.SetSubmitHook(nullptr, nullptr);
	m_guest_mirror->ForEachRequested(m_guest_mirror->Base(), m_guest_mirror->Size(), [this](uint64_t begin, uint64_t end) {
		if (m_memory_tracker.IsRegionGpuModified(begin, end - begin)) {
			EXIT("BufferCache: destroyed with GPU-modified memory in the mirror\n");
		}
	});
	m_storage.reset();
	m_host_view.reset();
	m_guest_mirror.reset();
	if (m_scatter_pipeline) {
		m_graphics.device.destroyPipeline(m_scatter_pipeline);
	}
	if (m_scatter_layout) {
		m_graphics.device.destroyPipelineLayout(m_scatter_layout);
	}
	if (m_readback.pool) {
		m_graphics.device.destroyFence(m_readback.fence);
		m_graphics.device.destroyCommandPool(m_readback.pool);
	}
	for (auto& slot: m_readback_slots) {
		if (slot.queue.pool) {
			(void)m_graphics.device.waitForFences(1, &slot.queue.fence, VK_TRUE, UINT64_MAX);
			m_graphics.device.destroyFence(slot.queue.fence);
			m_graphics.device.destroyCommandPool(slot.queue.pool);
		}
	}
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size, bool guest_fault) {
	for (;;) {
		bool gpu_modified = false;
		m_memory_tracker.InvalidateRegion(vaddr, size, [&gpu_modified] { gpu_modified = true; });
		if (!gpu_modified) {
			return;
		}
		ReadMemory(vaddr, size, true, guest_fault);
	}
}

void BufferCache::InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	bool requested = false;
	m_memory_tracker.InvalidateRegion(vaddr, size, [&] {
		if (std::exchange(requested, true)) return;
		if (GuestGpu::IsGpuThread()) {
			DiscardGpuWrites(vaddr, size);
			return;
		}
		m_scheduler.Context().GetGpu().SendUrgentCommandSync([this, vaddr, size] { DiscardGpuWrites(vaddr, size); });
	});
}

void BufferCache::DiscardGpuWrites(uint64_t vaddr, uint64_t size) {
	const auto end   = vaddr + size;
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(end, TRACKER_PAGE_SIZE);
	if (!IsRegionRegistered(vaddr, size)) {
		return;
	}
	if (const auto inflight = InFlightReadbackTick(first, last); inflight != 0) {
		m_scheduler.Wait(inflight);
		m_scheduler.WaitPriorityOperations(inflight);
		std::erase_if(m_inflight_readbacks, [inflight](const InFlightReadback& readback) { return readback.tick <= inflight; });
	}
	for (const auto& [begin, stop]: {std::pair {first, vaddr}, std::pair {end, last}}) {
		if (begin >= stop || !m_gpu_modified_ranges.Intersects(begin, stop - begin)) continue;
		(void)DownloadMemory<false>(begin, stop - begin);
	}
	if (m_gpu_modified_ranges.Subtract(first, last - first)) {
		NoteGpuModifiedChange(first, last - first, false);
	}
	ForgetColorMetadataChecks(first, last - first);
	UnmarkGpuModified(first, last - first);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
}

void BufferCache::InvalidateUnlessGpuModified(uint64_t vaddr, uint64_t size) noexcept {
	if (GuestRange {vaddr, size}.Valid()) {
		m_memory_tracker.MarkRegionAsCpuModifiedUnlessGpuDirty(vaddr, size);
	}
}

BufferCache::WriteFaultOwner BufferCache::InvalidateGuestWriteFault(uint64_t vaddr) {
	static constexpr uint64_t FaultBlockSize = CommandBlocks::BlockSize;
	const auto page       = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto block      = Common::AlignDown(page, FaultBlockSize);
	const bool command    = m_memory_tracker.IsCommandBlock(page);
	const bool segment    = !command && m_memory_tracker.IsSegmentBlock(page);
	const bool open_block = command || segment;
	const auto owner_of   = [&](bool gpu, bool tracked) {
		return command   ? WriteFaultOwner::CommandBuffer
		       : segment ? WriteFaultOwner::CommandSegment
		       : gpu     ? WriteFaultOwner::GpuNeighbor
		       : tracked ? WriteFaultOwner::Buffer
		                 : WriteFaultOwner::Other;
	};
	const auto end  = block + FaultBlockSize;
	uint64_t   mark = TRACKER_PAGE_SIZE;
	if (!open_block && page >= TRACKER_PAGE_SIZE && GuestRange {page - TRACKER_PAGE_SIZE, end - page}.Valid() &&
	    m_memory_tracker.IsRegionCpuModified(page - TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE) &&
	    !m_memory_tracker.IsRegionGpuModified(page, end - page) && MayMarkAhead()) {
		mark = end - page;
	}
	auto& stats   = g_write_block_stats;
	bool  gpu_any = false;
	if (m_memory_tracker.WriteFaultFindsOpen(page, FaultBlockSize, open_block, mark, &gpu_any)) {
		stats.already_open.fetch_add(1, std::memory_order_relaxed);
		return owner_of(gpu_any, false);
	}
	const auto probe = m_memory_tracker.FaultWrite(page, FaultBlockSize, open_block, mark);
	const auto owner = owner_of(probe.gpu, probe.tracked);
	if (probe.opened != 0) {
		stats.blocks.fetch_add(1, std::memory_order_relaxed);
		stats.pages.fetch_add(probe.opened, std::memory_order_relaxed);
	}
	if (!probe.done) {
		if (mark == TRACKER_PAGE_SIZE) {
			InvalidateMemory(vaddr, 1, true);
		} else {
			InvalidateMemory(page, mark, true);
		}
	}
	return owner;
}

BufferCache::WriteFaultOwner BufferCache::InvalidateFaultedPage(uint64_t vaddr) {
	static constexpr uint64_t FaultBlockSize = CommandBlocks::BlockSize;
	static_assert(TRACKER_REGION_SIZE % FaultBlockSize == 0);
	const auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	if (!GuestGpu::IsGpuThread() && !GuestGpu::IsCommandProcessorThread() &&
	    GuestRange {Common::AlignDown(page, FaultBlockSize), FaultBlockSize}.Valid()) {
		return InvalidateGuestWriteFault(vaddr);
	}
	m_memory_tracker.NoteGuestWriteFault(page);
	const auto end = Common::AlignDown(page, FaultBlockSize) + FaultBlockSize;
	if (page >= TRACKER_PAGE_SIZE && GuestRange {page - TRACKER_PAGE_SIZE, end - page}.Valid() &&
	    m_memory_tracker.IsRegionCpuModified(page - TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE) &&
	    !m_memory_tracker.IsRegionGpuModified(page, end - page) && MayMarkAhead()) {
		InvalidateMemory(page, end - page, true);
		return WriteFaultOwner::Other;
	}
	InvalidateMemory(vaddr, 1, true);
	return WriteFaultOwner::Other;
}

bool BufferCache::MayMarkAhead() const noexcept {
	return m_indirect_writes_pending.load(std::memory_order_acquire) == 0;
}

void BufferCache::CountWriteFault(WriteFaultOwner owner, bool image) {
	auto& stats = g_write_block_stats;
	stats.faults.fetch_add(1, std::memory_order_relaxed);
	if (image && (owner == WriteFaultOwner::CommandBuffer || owner == WriteFaultOwner::CommandSegment)) {
		stats.command_image.fetch_add(1, std::memory_order_relaxed);
	}
	auto& counter = owner == WriteFaultOwner::CommandBuffer    ? stats.command
	                : owner == WriteFaultOwner::CommandSegment ? stats.segment
	                : owner == WriteFaultOwner::GpuNeighbor    ? stats.gpu
	                : image                                 ? stats.image
	                : owner == WriteFaultOwner::Buffer      ? stats.buffer
	                                                        : stats.other;
	counter.fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write, bool guest_fault) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	if (!GuestGpu::IsGpuThread()) {
		ReadMemoryAsync(vaddr, size, is_write, guest_fault);
		return;
	}
	m_scheduler.Context().GetGpu().SendUrgentCommandSync([this, vaddr, size, is_write, guest_fault] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		if (TryGuestReadFromFeedback(vaddr, size, is_write)) {
			return;
		}
		const auto read_size = guest_fault ? ReadAheadSize(vaddr, size) : size;

		const auto pages_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		const auto pages_end   = Common::AlignUp(vaddr + read_size, TRACKER_PAGE_SIZE);
		if (const auto inflight = InFlightReadbackTick(pages_begin, pages_end); inflight != 0) {
			m_scheduler.Wait(inflight);
			m_scheduler.WaitPriorityOperations(inflight);
			std::erase_if(m_inflight_readbacks, [inflight](const InFlightReadback& readback) {
				return readback.tick <= inflight;
			});
		}
		if (DownloadMemory<false>(vaddr, read_size) ||
		    !m_gpu_modified_ranges.Intersects(pages_begin, pages_end - pages_begin)) {
			UnmarkGpuModified(vaddr, read_size);
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
	}, false);
}

void BufferCache::ReadMemoryAsync(uint64_t vaddr, uint64_t size, bool is_write, bool guest_fault) {
	const bool command_processor = GuestGpu::IsCommandProcessorThread();
	const auto started           = SteadyNs();
	ReadMemoryAsyncTimed(vaddr, size, is_write, guest_fault, command_processor);
	auto&      stats   = g_readback_stats;
	const auto elapsed = static_cast<uint64_t>(SteadyNs() - started);
	(command_processor ? stats.cp_reads : stats.other_reads).fetch_add(1, std::memory_order_relaxed);
	(command_processor ? stats.cp_ns : stats.other_ns).fetch_add(elapsed, std::memory_order_relaxed);
}

void BufferCache::AllowReplayAfterReadback() noexcept {
	g_readback_replay = {.allowed = true};
}

bool BufferCache::TakeReplayAfterReadback(uint64_t* begin, uint64_t* end) noexcept {
	const auto replay = std::exchange(g_readback_replay, {});
	*begin            = replay.begin;
	*end              = replay.end;
	return replay.published;
}

void BufferCache::NoteReplayAfterReadback(bool replayed) noexcept {
	(replayed ? g_readback_stats.replayed_after_copy : g_readback_stats.waited_after_copy)
	    .fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::WaitReadbackRelease() {
	m_scheduler.Context().GetGpu().SendReadbackCommandSync([] {});
}

void BufferCache::NoteReplayedRead() {
	g_readback_stats.other_replayed.fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::NoteStagedCommandProcessorRead() {
	g_readback_stats.cp_staged.fetch_add(1, std::memory_order_relaxed);
}

void BufferCache::LogWriteBlockSummary() {
	auto&      stats = g_write_block_stats;
	const auto now   = SteadyNs();
	auto       last  = stats.last_line.load(std::memory_order_relaxed);
	if (last == 0) {
		stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000 || !stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto faults   = stats.faults.exchange(0, std::memory_order_relaxed);
	const auto blocks   = stats.blocks.exchange(0, std::memory_order_relaxed);
	const auto pages    = stats.pages.exchange(0, std::memory_order_relaxed);
	const auto uploaded = stats.uploaded.exchange(0, std::memory_order_relaxed);
	const auto command  = stats.command.exchange(0, std::memory_order_relaxed);
	const auto segment  = stats.segment.exchange(0, std::memory_order_relaxed);
	const auto gpu      = stats.gpu.exchange(0, std::memory_order_relaxed);
	const auto image    = stats.image.exchange(0, std::memory_order_relaxed);
	const auto buffer   = stats.buffer.exchange(0, std::memory_order_relaxed);
	const auto other    = stats.other.exchange(0, std::memory_order_relaxed);
	const auto open     = stats.already_open.exchange(0, std::memory_order_relaxed);
	const auto cmd_img  = stats.command_image.exchange(0, std::memory_order_relaxed);
	const auto opened   = RegionManager::TakeOpenedUploadBytes();
	const auto written  = MemoryTracker::TakeWrittenUploadStats();
	const auto tracker  = TrackingSpinLock::TakeContention(SpinLockClass::Tracker);
	const auto pagelock = TrackingSpinLock::TakeContention(SpinLockClass::Pages);
	const auto textures = TrackingSpinLock::TakeContention(SpinLockClass::Textures);
	const auto others   = TrackingSpinLock::TakeContention(SpinLockClass::Other);
	const auto seconds  = static_cast<double>(now - last) / 1e9;
	if (tracker.waits + pagelock.waits + textures.waits + others.waits != 0) {
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-locks] summary of the last {:.0f} s: contended acquisitions of a TrackingSpinLock and time spent waiting: "
		    "{} on tracker regions ({:.1f} ms), {} on page manager regions ({:.1f} ms), {} on the texture cache ({:.1f} ms), {} "
		    "on others ({:.1f} ms)\n",
		    seconds, tracker.waits, static_cast<double>(tracker.ns) / 1e6, pagelock.waits,
		    static_cast<double>(pagelock.ns) / 1e6, textures.waits, static_cast<double>(textures.ns) / 1e6, others.waits,
		    static_cast<double>(others.ns) / 1e6));
	}
	if (faults == 0 && uploaded == 0 && written.uploads == 0) {
		return;
	}
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-write-blocks] summary of the last {:.0f} s: {} guest write faults, {} opened their 64 KiB block ({} clean pages "
	    "marked written with them), {} left to the page path; {:.1f} MiB of buffer uploads; faults by owner, first match: {} in "
	    "a command buffer the CP read, {} elsewhere in a command buffer segment, {} in a block with a page the GPU wrote, {} "
	    "watched by an image or texture, {} in a buffer of the cache, {} other; {:.1f} MiB of pages opened with a block handed "
	    "to uploads; {} found their page already open, {} in command memory also watched by an image or table; {} uploads of "
	    "buffers the GPU writes, {} copies made again for pages the guest wrote during the copy, {} uploads that kept the "
	    "region locks through the copy\n",
	    seconds, faults, blocks, pages, faults - std::min(faults, blocks),
	    static_cast<double>(uploaded) / (1024.0 * 1024.0), command, segment, gpu, image, buffer, other,
	    static_cast<double>(opened) / (1024.0 * 1024.0), open, cmd_img, written.uploads, written.again, written.locked));
}

void BufferCache::LogReadbackSummary() {
	LogWriteBlockSummary();
	auto&      stats = g_readback_stats;
	const auto now   = SteadyNs();
	auto       last  = stats.last_line.load(std::memory_order_relaxed);
	if (last == 0) {
		stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000 || !stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto cp_reads    = stats.cp_reads.exchange(0, std::memory_order_relaxed);
	const auto other_reads = stats.other_reads.exchange(0, std::memory_order_relaxed);
	const auto cp_ns       = stats.cp_ns.exchange(0, std::memory_order_relaxed);
	const auto other_ns    = stats.other_ns.exchange(0, std::memory_order_relaxed);
	const auto ordered     = stats.cp_ordered.exchange(0, std::memory_order_relaxed);
	const auto ordered_ns  = stats.cp_ordered_ns.exchange(0, std::memory_order_relaxed);
	const auto not_waited  = stats.cp_not_waited.exchange(0, std::memory_order_relaxed);
	const auto gpu_bytes   = stats.other_gpu_bytes.exchange(0, std::memory_order_relaxed);
	const auto false_share = stats.other_false_sharing.exchange(0, std::memory_order_relaxed);
	const auto staged      = stats.cp_staged.exchange(0, std::memory_order_relaxed);
	const auto replayed    = stats.other_replayed.exchange(0, std::memory_order_relaxed);
	const auto dispatch_ns = stats.other_dispatch_ns.exchange(0, std::memory_order_relaxed);
	const auto wait_ns     = stats.other_wait_ns.exchange(0, std::memory_order_relaxed);
	const auto release_ns  = stats.other_release_ns.exchange(0, std::memory_order_relaxed);
	const auto async_rel   = stats.released_async.exchange(0, std::memory_order_relaxed);
	const auto replayed_cp = stats.replayed_after_copy.exchange(0, std::memory_order_relaxed);
	const auto waited_cp   = stats.waited_after_copy.exchange(0, std::memory_order_relaxed);
	if (cp_reads + other_reads + staged + replayed == 0) {
		return;
	}
	const auto seconds = [](uint64_t ns) { return static_cast<double>(ns) / 1e9; };
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-readback] summary of the last {:.0f} s: command processor {} readbacks in {:.3f} s ({} after the "
	    "operations it is ordered after only, {:.3f} s waiting for them, {} operations not waited for), {} register "
	    "tables read without a fault; other threads {} readbacks in {:.3f} s ({} of faulted reads on bytes the GPU "
	    "wrote, {} on other bytes of a GPU written page, {} on other bytes replayed on the backing view without a readback; "
	    "{:.3f} s handing them to the execution thread, {:.3f} s waiting for the GPU copy, {:.3f} s releasing them through "
	    "the execution thread; {} released without waiting, the faulted instruction replayed on the copy {} times and "
	    "waiting for the release {} times){}\n",
	    seconds(static_cast<uint64_t>(now - last)), cp_reads, seconds(cp_ns), ordered, seconds(ordered_ns), not_waited,
	    staged, other_reads, seconds(other_ns), gpu_bytes, false_share, replayed, seconds(dispatch_ns), seconds(wait_ns),
	    seconds(release_ns), async_rel, replayed_cp, waited_cp,
	    stats.line_overflow.load(std::memory_order_relaxed) ? "; GPU line table overflowed" : ""));
}

void BufferCache::ReadMemoryAsyncTimed(uint64_t vaddr, uint64_t size, bool is_write, bool guest_fault,
                                       bool command_processor) {
	auto&             gpu         = m_scheduler.Context().GetGpu();
	uint64_t          tick        = 0;
	uint64_t          read_size   = size;
	uint64_t          pages_begin = 0;
	uint64_t          pages_end   = 0;
	ReadbackQueueRead queue_read;
	const bool        ordered        = command_processor;
	const bool        readback_queue = !command_processor || ordered;
	if (ordered) {
		const auto started = SteadyNs();
		const auto skipped = gpu.WaitForOrderedOperations(
		    Common::AlignDown(vaddr, TRACKER_PAGE_SIZE), Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE) + ReadAheadLimit);
		auto& stats = g_readback_stats;
		stats.cp_ordered.fetch_add(1, std::memory_order_relaxed);
		stats.cp_ordered_ns.fetch_add(static_cast<uint64_t>(SteadyNs() - started), std::memory_order_relaxed);
		stats.cp_not_waited.fetch_add(skipped, std::memory_order_relaxed);
	}
	auto&      split          = g_readback_stats;
	const auto dispatch_start = SteadyNs();
	const auto note_split     = [&](std::atomic<uint64_t>& counter, int64_t since) {
		if (!command_processor) {
			counter.fetch_add(static_cast<uint64_t>(SteadyNs() - since), std::memory_order_relaxed);
		}
	};
	gpu.SendReadbackCommandSync([&] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		if (TryGuestReadFromFeedback(vaddr, size, is_write)) {
			return;
		}
		read_size    = guest_fault ? ReadAheadSize(vaddr, size) : size;
		pages_begin  = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		pages_end    = Common::AlignUp(vaddr + read_size, TRACKER_PAGE_SIZE);
		tick         = InFlightReadbackTick(pages_begin, pages_end);
		NoteReadbackWait(pages_begin, pages_end, false,
		                 m_write_ticks.Latest(pages_begin, pages_end) >= m_scheduler.CurrentTick());
		const bool guest_read = guest_fault && !is_write;
		if (guest_read && !command_processor) {
			(m_gpu_modified_ranges.Intersects(vaddr, size) ? g_readback_stats.other_gpu_bytes
			                                                : g_readback_stats.other_false_sharing)
			    .fetch_add(1, std::memory_order_relaxed);
		}
		if (tick == 0 && readback_queue &&
		    StartReadbackQueueRead(vaddr, read_size, pages_begin, pages_end, queue_read, guest_read)) {
			tick = queue_read.recorded;
		} else if (DownloadMemory<true>(vaddr, read_size)) {
			const auto recorded = m_scheduler.CurrentTick();
			m_inflight_readbacks.push_back({pages_begin, pages_end, recorded});
			tick = std::max(tick, recorded);
		}
		if (guest_read) {
			NoteGuestReadPages(pages_begin, pages_end);
		}
		if (tick == 0) {
			ReleaseReadback(vaddr, size, pages_begin, pages_end, 0, is_write);
			return;
		}
		if (queue_read.slot == nullptr && tick >= m_scheduler.CurrentTick()) {
			m_scheduler.Flush();
		}
	});
	note_split(split.other_dispatch_ns, dispatch_start);
	if (tick == 0) {
		return;
	}
	const bool replay_after = g_readback_replay.allowed && !is_write && !command_processor;
	const auto release      = [&](Common::UniqueFunction<void>&& operation) {
		const auto release_start = SteadyNs();
		if (replay_after && gpu.PostReadbackCommand(std::move(operation))) {
			g_readback_replay.published = true;
			g_readback_replay.begin     = vaddr;
			g_readback_replay.end       = vaddr + read_size;
			split.released_async.fetch_add(1, std::memory_order_relaxed);
		} else {
			gpu.SendReadbackCommandSync(std::move(operation));
		}
		note_split(split.other_release_ns, release_start);
	};
	const auto wait_start = SteadyNs();
	if (queue_read.slot != nullptr) {
		FinishReadbackQueueRead(queue_read);
		note_split(split.other_wait_ns, wait_start);
		release([this, vaddr, size, pages_begin, pages_end, is_write, recorded = queue_read.recorded,
		         followers = std::move(queue_read.followers)] {
			const auto own = std::find_if(m_inflight_readbacks.begin(), m_inflight_readbacks.end(),
			                              [&](const InFlightReadback& readback) {
				                              return readback.begin == pages_begin &&
				                                     readback.end == pages_end &&
				                                     readback.tick == recorded;
			                              });
			if (own != m_inflight_readbacks.end()) {
				m_inflight_readbacks.erase(own);
			}
			ReleaseReadback(vaddr, size, pages_begin, pages_end, 0, is_write);
			if (!followers.empty()) {
				std::vector<uint64_t> pages;
				pages.reserve(followers.size());
				for (const auto& follower: followers) {
					pages.push_back(follower.page);
				}
				std::sort(pages.begin(), pages.end());
				std::erase_if(m_inflight_readbacks, [&](const InFlightReadback& readback) {
					return readback.tick == recorded &&
					       readback.end - readback.begin == TRACKER_PAGE_SIZE &&
					       std::binary_search(pages.begin(), pages.end(), readback.begin);
				});
				for (const auto& follower: followers) {
					if (InFlightReadbackTick(follower.page, follower.page + TRACKER_PAGE_SIZE) == 0 &&
					    !m_gpu_modified_ranges.Intersects(follower.page, TRACKER_PAGE_SIZE)) {
						UnmarkGpuModified(follower.page, TRACKER_PAGE_SIZE);
					}
				}
			}
		});
	} else {
		m_scheduler.GetMasterSemaphore().Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		note_split(split.other_wait_ns, wait_start);
		release([this, vaddr, size, pages_begin, pages_end, tick, is_write] {
			ReleaseReadback(vaddr, size, pages_begin, pages_end, tick, is_write);
		});
	}
}

void BufferCache::NoteGuestReadPages(uint64_t pages_begin, uint64_t pages_end) {
	const auto read = ++m_guest_reads;
	for (auto page = pages_begin; page < pages_end; page += TRACKER_PAGE_SIZE) {
		auto found = std::ranges::find(m_guest_read_pages, page, &GuestReadPage::page);
		if (found == m_guest_read_pages.end()) {
			if (m_guest_read_pages.size() >= MaxGuestReadPages) {
				found = std::ranges::min_element(m_guest_read_pages, {}, &GuestReadPage::read);
			} else {
				found = m_guest_read_pages.emplace(m_guest_read_pages.end());
			}
			found->page = page;
		}
		found->read = read;
	}
}

bool BufferCache::StartReadbackQueueRead(uint64_t vaddr, uint64_t size, uint64_t pages_begin, uint64_t pages_end,
                                         ReadbackQueueRead& read, bool followers) {
	if (!m_graphics.readback_queue || m_device_address_pending) {
		return false;
	}
	uint64_t needed  = m_device_address_tick;
	bool     unknown = false;
	bool     written = false;
	m_gpu_modified_ranges.ForEachInRange(pages_begin, pages_end - pages_begin, [&](uint64_t start,
	                                                                              uint64_t end) {
		const auto latest = m_write_ticks.Latest(start, end);
		unknown |= latest == UINT64_MAX;
		needed  = std::max(needed, latest);
		written = true;
		for (const auto& [pending, bytes]: m_pending_writes) {
			unknown |= pending < end && start < pending + bytes;
		}
	});
	if (!written || unknown || needed > m_scheduler.SubmittedTick()) {
		return false;
	}
	ReadbackQueueSlot* slot = nullptr;
	for (auto& candidate: m_readback_slots) {
		if (!candidate.busy.load(std::memory_order_acquire)) {
			slot = &candidate;
			break;
		}
	}
	if (slot == nullptr) {
		return false;
	}
	read.copies.clear();
	read.followers.clear();
	read.total_size = 0;
	CollectDownloadCopies(vaddr, size, read.copies, read.total_size);
	if (read.copies.empty()) {
		return false;
	}
	if (followers) {
		std::erase_if(m_guest_read_pages, [&](const GuestReadPage& entry) {
			return entry.read + GuestReadPageLifetime < m_guest_reads;
		});
		for (const auto& entry: m_guest_read_pages) {
			const auto page = entry.page;
			const auto end  = page + TRACKER_PAGE_SIZE;
			if ((page < pages_end && pages_begin < end) ||
			    !m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) ||
			    InFlightReadbackTick(page, end) != 0 ||
			    m_texture_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
				continue;
			}
			if (!Holds(page, TRACKER_PAGE_SIZE)) {
				continue;
			}
			uint64_t latest       = m_device_address_tick;
			bool     page_unknown = false;
			m_gpu_modified_ranges.ForEachInRange(page, TRACKER_PAGE_SIZE,
			                                     [&](uint64_t start, uint64_t finish) {
				                                     const auto tick = m_write_ticks.Latest(start, finish);
				                                     page_unknown |= tick == UINT64_MAX;
				                                     latest = std::max(latest, tick);
				                                     for (const auto& [pending, bytes]: m_pending_writes) {
					                                     page_unknown |=
					                                         pending < finish && start < pending + bytes;
				                                     }
			                                     });
			if (page_unknown || latest > m_scheduler.SubmittedTick() ||
			    (latest > needed && !m_scheduler.IsFree(latest))) {
				continue;
			}
			auto& follower = read.followers.emplace_back();
			CollectDownloadCopies(page, TRACKER_PAGE_SIZE, follower.copies, read.total_size);
			if (follower.copies.empty()) {
				read.followers.pop_back();
				continue;
			}
			follower.page = page;
			needed        = std::max(needed, latest);
		}
	}
	slot->busy.store(true, std::memory_order_release);
	auto& queue  = slot->queue;
	auto& device = m_graphics.device;
	if (!queue.pool) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
		                             vk::CommandPoolCreateFlagBits::eTransient;
		pool_info.queueFamilyIndex = m_graphics.readback_family;
		vk::CommandBufferAllocateInfo allocate {};
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		vk::FenceCreateInfo fence_info {};
		fence_info.flags = vk::FenceCreateFlagBits::eSignaled;
		if (device.createCommandPool(&pool_info, nullptr, &queue.pool) != vk::Result::eSuccess ||
		    (allocate.commandPool = queue.pool,
		     device.allocateCommandBuffers(&allocate, &queue.command) != vk::Result::eSuccess) ||
		    device.createFence(&fence_info, nullptr, &queue.fence) != vk::Result::eSuccess) {
			EXIT("BufferCache: could not create the readback queue resources\n");
		}
	}
	if (!queue.staging || queue.staging->Size() < read.total_size) {
		queue.staging = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Download, 0,
		    vk::BufferUsageFlagBits::eTransferDst,
		    std::max<uint64_t>(Common::AlignUp(read.total_size, 1024 * 1024), 1024 * 1024));
	}
	const auto command = queue.command;
	(void)device.resetFences(1, &queue.fence);
	(void)command.reset({});
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	(void)command.begin(&begin_info);
	CheckMirrorCopies(read.copies);
	std::vector<vk::BufferCopy> regions = read.copies;
	for (const auto& follower: read.followers) {
		CheckMirrorCopies(follower.copies);
		regions.insert(regions.end(), follower.copies.begin(), follower.copies.end());
	}
	command.copyBuffer(m_storage->Handle(), queue.staging->Handle(), static_cast<uint32_t>(regions.size()), regions.data());
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = queue.staging->Handle();
	after.offset              = 0;
	after.size                = read.total_size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &after, 0, nullptr);
	(void)command.end();
	const auto                      semaphore  = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &needed;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &semaphore;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	if (m_graphics.readback_queue.submit(1, &submit, queue.fence) != vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue submission failed\n");
	}
	read.slot           = slot;
	read.needed         = needed;
	read.recorded       = m_scheduler.CurrentTick();
	read.published      = std::make_shared<std::atomic<bool>>(false);
	m_inflight_readbacks.push_back({pages_begin, pages_end, read.recorded});
	if (read.followers.empty()) {
		m_scheduler.DeferPriorityOperation(
		    [published = read.published] { published->wait(false, std::memory_order_acquire); },
		    pages_begin, pages_end - pages_begin);
		return true;
	}
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	ranges.reserve(read.followers.size() + 1);
	ranges.emplace_back(pages_begin, pages_end - pages_begin);
	for (const auto& follower: read.followers) {
		m_inflight_readbacks.push_back({follower.page, follower.page + TRACKER_PAGE_SIZE, read.recorded});
		ranges.emplace_back(follower.page, TRACKER_PAGE_SIZE);
	}
	m_scheduler.DeferPriorityOperation(
	    [published = read.published] { published->wait(false, std::memory_order_acquire); }, ranges);
	return true;
}

void BufferCache::WaitReadbackQueueCopies() {
	for (auto& slot: m_readback_slots) {
		if (slot.busy.load(std::memory_order_acquire)) {
			(void)m_graphics.device.waitForFences(1, &slot.queue.fence, VK_TRUE, UINT64_MAX);
		}
	}
}

void BufferCache::FinishReadbackQueueRead(ReadbackQueueRead& read) {
	auto& queue = read.slot->queue;
	if (m_graphics.device.waitForFences(1, &queue.fence, VK_TRUE, UINT64_MAX) !=
	    vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue wait failed\n");
	}
	m_scheduler.WaitPriorityOperations(read.needed);
	queue.staging->Invalidate(0, read.total_size);
	const auto* mapped = queue.staging->Mapped().data();
	const auto  origin = m_storage->HandleOrigin();
	for (const auto& copy: read.copies) {
		Libs::LibKernel::Memory::WriteBacking(origin + copy.srcOffset, mapped + copy.dstOffset, copy.size);
	}
	for (const auto& follower: read.followers) {
		for (const auto& copy: follower.copies) {
			Libs::LibKernel::Memory::WriteBacking(origin + copy.srcOffset, mapped + copy.dstOffset, copy.size);
		}
	}
	read.slot->busy.store(false, std::memory_order_release);
	read.published->store(true, std::memory_order_release);
	read.published->notify_all();
}

uint64_t BufferCache::InFlightReadbackTick(uint64_t begin, uint64_t end) const {
	uint64_t tick = 0;
	for (const auto& readback: m_inflight_readbacks) {
		if (readback.begin < end && begin < readback.end) {
			tick = std::max(tick, readback.tick);
		}
	}
	return tick;
}

struct BufferCache::CopyFeedback {
	static constexpr uint64_t SlotSize  = 64 * 1024;
	static constexpr size_t   SlotCount = 512;
	struct Slot {
		uint64_t address = 0, size = 0, tick = 0, mapping_epoch = 0;
	};
	Buffer                      download;
	std::array<Slot, SlotCount> slots {};
	std::vector<std::pair<uint64_t, size_t>> index;
	size_t                                   cursor = 0;
	[[nodiscard]] auto UpperBound(uint64_t address) {
		return std::upper_bound(index.begin(), index.end(), address,
		                        [](uint64_t value, const auto& entry) { return value < entry.first; });
	}
	[[nodiscard]] auto LowerBound(uint64_t address) {
		return std::lower_bound(index.begin(), index.end(), address,
		                        [](const auto& entry, uint64_t value) { return entry.first < value; });
	}
	bool Erase(uint64_t address) {
		const auto it = LowerBound(address);
		if (it == index.end() || it->first != address) return false;
		index.erase(it);
		return true;
	}
	bool Insert(uint64_t address, size_t slot) {
		const auto it = LowerBound(address);
		if (it != index.end() && it->first == address) return false;
		index.insert(it, {address, slot});
		return true;
	}
	std::array<uint16_t, 4096> granules {};
	void                       Count(const Slot& slot, int delta) {
		for (uint64_t g = slot.address >> 24u; g <= (slot.address + slot.size - 1) >> 24u; ++g)
			granules[g & 4095u] = static_cast<uint16_t>(granules[g & 4095u] + delta);
	}
	[[nodiscard]] bool MayOverlap(uint64_t vaddr, uint64_t size) const {
		const uint64_t first = vaddr >> 24u, last = (vaddr + size - 1) >> 24u;
		if (size == 0 || last - first >= granules.size()) return true;
		for (uint64_t g = first; g <= last; ++g)
			if (granules[g & 4095u] != 0) return true;
		return false;
	}
	CopyFeedback(GraphicContext& graphics, CommandScheduler& scheduler)
	    : download(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	               SlotCount * SlotSize) {
		const auto atom = graphics.physical_device_properties.limits.nonCoherentAtomSize;
		EXIT_IF(atom == 0 || SlotSize % atom != 0);
	}
};

void BufferCache::InvalidateCopyFeedback(uint64_t vaddr, uint64_t size) noexcept {
	if (!m_copy_feedback || m_copy_feedback->index.empty()) return;
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size)) return;
	auto it = feedback.LowerBound(vaddr);
	if (it != feedback.index.begin()) {
		const auto  prior = std::prev(it);
		const auto& slot  = feedback.slots[prior->second];
		if (slot.address + slot.size > vaddr) it = prior;
	}
	auto last = it;
	for (; last != feedback.index.end() && last->first < vaddr + size; ++last) {
		feedback.Count(feedback.slots[last->second], -1);
		feedback.slots[last->second].address = 0;
	}
	feedback.index.erase(it, last);
	m_copy_feedback_entries = feedback.index.size();
}

void BufferCache::ScheduleCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (m_resources == nullptr) return;
	const auto mapping_epoch = m_resources->MappingEpoch();
	if (size == 0 || !GuestRange {vaddr, size}.Valid() || ((vaddr | size) & 3u) != 0 ||
	    size > CopyFeedback::SlotSize || !m_gpu_modified_ranges.Contains(vaddr, size) ||
	    m_texture_cache.IsRegionRegistered(vaddr, size) || !m_resources->IsMapped(vaddr, size)) {
		return;
	}
	if (!Holds(vaddr, size)) {
		return;
	}
	auto* owner = m_storage.get();
	if (!m_copy_feedback) m_copy_feedback = std::make_unique<CopyFeedback>(m_graphics, m_scheduler);
	auto&  feedback = *m_copy_feedback;
	auto&  master   = m_scheduler.GetMasterSemaphore();
	size_t selected = CopyFeedback::SlotCount;
	for (unsigned refresh = 0; refresh < 2 && selected == CopyFeedback::SlotCount; ++refresh) {
		if (refresh != 0) master.Refresh();
		const auto completed = master.KnownGpuTick();
		for (unsigned evict = 0; evict < 2 && selected == CopyFeedback::SlotCount; ++evict) {
			for (size_t n = 0; n < CopyFeedback::SlotCount; ++n) {
				const auto  i    = (feedback.cursor + n) % CopyFeedback::SlotCount;
				const auto& slot = feedback.slots[i];
				if (slot.tick <= completed && (evict != 0 || slot.address == 0)) {
					selected = i;
					break;
				}
			}
		}
	}
	if (selected == CopyFeedback::SlotCount) {
		return;
	}
	if (m_resources->MappingEpoch() != mapping_epoch) return;
	auto& slot = feedback.slots[selected];
	if (slot.address != 0 && feedback.Erase(slot.address)) {
		feedback.Count(slot, -1);
	}
	InvalidateCopyFeedback(vaddr, size);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = owner->Handle();
	before.offset              = owner->Offset(vaddr);
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                       nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {owner->Offset(vaddr), selected * CopyFeedback::SlotSize, size};
	native.copyBuffer(owner->Handle(), feedback.download.Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = feedback.download.Handle();
	after.offset        = selected * CopyFeedback::SlotSize;
	after.size          = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                       1, &after, 0, nullptr);
	slot = {vaddr, size, m_scheduler.CurrentTick(), mapping_epoch};
	if (feedback.Insert(vaddr, selected)) feedback.Count(slot, 1);
	m_copy_feedback_entries = feedback.index.size();
	feedback.cursor         = (selected + 1) % CopyFeedback::SlotCount;
}

bool BufferCache::TryReadCopyFeedback(uint64_t vaddr, uint64_t size) {
	if (!m_copy_feedback || m_resources == nullptr) return false;
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (!m_storage->IsInBounds(begin, end - begin) || end - begin > CopyFeedback::SlotSize ||
	    m_texture_cache.IsRegionRegistered(begin, end - begin)) {
		return false;
	}
	std::vector<std::pair<uint64_t, uint64_t>> copies;
	m_memory_tracker.ForEachDownloadRange<false>(begin, end - begin, [&](uint64_t address, uint64_t bytes) noexcept {
		m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes, "copy feedback");
		m_gpu_modified_ranges.ForEachInRange(address, bytes,
		                                     [&](uint64_t first, uint64_t last) { copies.emplace_back(first, last - first); });
	});
	if (copies.empty()) return false;
	struct Part {
		uint64_t address, size, offset;
	};
	std::vector<Part> parts;
	auto&             feedback      = *m_copy_feedback;
	const auto        mapping_epoch = m_resources->MappingEpoch();
	uint64_t          latest_tick   = 0;
	for (const auto& [address, bytes_in]: copies) {
		auto cursor = address;
		while (cursor < address + bytes_in) {
			auto it = feedback.UpperBound(cursor);
			if (it == feedback.index.begin()) return false;
			--it;
			const auto& slot = feedback.slots[it->second];
			if (cursor >= slot.address + slot.size) return false;
			if (m_resources->UnmappedSince(slot.mapping_epoch, slot.address, slot.size) ||
			    !m_resources->IsMapped(slot.address, slot.size)) {
				return false;
			}
			const auto bytes = std::min(address + bytes_in, slot.address + slot.size) - cursor;
			parts.push_back({cursor, bytes, it->second * CopyFeedback::SlotSize + cursor - slot.address});
			latest_tick = std::max(latest_tick, slot.tick);
			cursor += bytes;
		}
	}
	if (!m_scheduler.IsFree(latest_tick)) return false;
	m_scheduler.WaitPriorityOperations(latest_tick);
	if (m_resources->MappingEpoch() != mapping_epoch) return false;
	for (const auto& part: parts) feedback.download.Invalidate(part.offset, part.size);
	for (const auto& part: parts) {
		LibKernel::Memory::WriteBacking(part.address, feedback.download.Mapped().data() + part.offset, part.size);
	}
	for (const auto& [address, bytes]: copies) {
		if (m_gpu_modified_ranges.Subtract(address, bytes)) {
			NoteGpuModifiedChange(address, bytes, false);
		}
		ForgetColorMetadataChecks(address, bytes);
	}
	UnmarkGpuModified(begin, end - begin);
	return true;
}

bool BufferCache::TryGuestReadFromFeedback(uint64_t vaddr, uint64_t size, bool is_write) {
	if (m_copy_feedback_entries == 0 || !m_copy_feedback) return false;
	auto& feedback = *m_copy_feedback;
	if (!feedback.MayOverlap(vaddr, size) || !IsRegionRegistered(vaddr, size)) return false;
	auto first = vaddr, last = vaddr + size;
	if (auto it = feedback.UpperBound(vaddr); it != feedback.index.begin()) {
		const auto& slot = feedback.slots[std::prev(it)->second];
		if (slot.address <= vaddr && vaddr < slot.address + slot.size) {
			first = std::min(first, slot.address);
			last  = std::max(last, slot.address + slot.size);
		}
	}
	if (InFlightReadbackTick(Common::AlignDown(first, TRACKER_PAGE_SIZE), Common::AlignUp(last, TRACKER_PAGE_SIZE)) != 0)
		return false;
	if (last - first == size || !m_storage->IsInBounds(first, last - first) || !TryReadCopyFeedback(first, last - first)) {
		if (!TryReadCopyFeedback(vaddr, size)) return false;
	}
	if (is_write) m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	return true;
}

void BufferCache::ReleaseReadback(uint64_t vaddr, uint64_t size, uint64_t pages_begin,
                                  uint64_t pages_end, uint64_t tick, bool is_write) {
	std::erase_if(m_inflight_readbacks,
	              [tick](const InFlightReadback& readback) { return readback.tick <= tick; });
	if (InFlightReadbackTick(pages_begin, pages_end) != 0 ||
	    m_gpu_modified_ranges.Intersects(pages_begin, pages_end - pages_begin)) {
		return;
	}
	UnmarkGpuModified(pages_begin, pages_end - pages_begin);
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

uint64_t BufferCache::ReadAheadSize(uint64_t vaddr, uint64_t size) const {
	static constexpr uint64_t MaxReadAhead = ReadAheadLimit;
	const auto                faulting_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto                faulting_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	const auto                limit =
	    std::min(m_storage->CpuAddress() + m_storage->Size(), faulting_end + MaxReadAhead);
	const auto latest_write = [&](uint64_t begin, uint64_t end) {
		uint64_t latest = 0;
		m_gpu_modified_ranges.ForEachInRange(begin, end - begin, [&](uint64_t first, uint64_t last) {
			latest = std::max(latest, m_write_ticks.Latest(first, last));
			for (const auto& [pending, bytes]: m_pending_writes) {
				if (pending < last && first < pending + bytes) {
					latest = UINT64_MAX;
				}
			}
		});
		return latest;
	};
	const auto needed = latest_write(faulting_begin, faulting_end);
	if (needed == 0) {
		return size;
	}
	const auto current = m_scheduler.CurrentTick();
	const auto bound   = needed >= current ? UINT64_MAX : current - 1;
	auto end    = faulting_end;
	auto cursor = faulting_end;
	while (cursor + TRACKER_PAGE_SIZE <= limit &&
	       !m_texture_cache.IsRegionGpuModified(cursor, TRACKER_PAGE_SIZE)) {
		if (!m_gpu_modified_ranges.Intersects(cursor, TRACKER_PAGE_SIZE)) {
			cursor += TRACKER_PAGE_SIZE;
			continue;
		}
		const auto tick = latest_write(cursor, cursor + TRACKER_PAGE_SIZE);
		if (tick == 0 || tick > bound) {
			break;
		}
		cursor += TRACKER_PAGE_SIZE;
		end = cursor;
	}
	return end > faulting_end ? end - vaddr : size;
}

void BufferCache::CacheFaultedMemory(uint64_t vaddr, uint64_t size) {
	for (size_t i = 0; i < StaleRingSize; i++) {
		const auto address = m_stale_ring[i * 2].load(std::memory_order_relaxed);
		const auto length  = m_stale_ring[i * 2 + 1].load(std::memory_order_relaxed);
		if (length != 0 && address < vaddr + size && vaddr < address + length) {
			Common::HackNotice::Report(
			    "mirror block evicted by age reached again by a device-address fault",
			    "BufferCache::CacheFaultedMemory", 0, 0,
			    "a shader read that memory before its block was bound again (first read may be zero)");
			break;
		}
	}
	EnsureRangeContents(vaddr, size);
	m_guest_mirror->NoteDevice(vaddr, size);
}

bool BufferCache::PrepareHeapPublication(HeapProducer producer, uint64_t serial, uint64_t version,
                                         uint64_t address, uint64_t size) {
	if (m_heap_tables.Empty() || !m_heap_tables.Intersects(address, size)) return true;
	const Publication publication {producer, serial, version, address, size};
	if (std::ranges::find(m_heap_publications, publication) != m_heap_publications.end()) {
		return !m_scheduler.Context().GetImageHeap().FrozenReplay();
	}
	m_heap_publications.push_back(publication);
	return true;
}

void BufferCache::CopyHeapTable(Buffer& destination, uint64_t offset, uint64_t address, uint64_t size) {
	destination.CopyFrom(m_scheduler.Current(), *m_storage, m_storage->Offset(address), offset, size);
}

void BufferCache::BeginHeapDraw(bool replay) {
	if (replay) {
		m_heap_shader_writes.Clear();
		m_heap_bitmap_pending.Clear();
		return;
	}
	m_heap_publications.clear();
	if (!m_heap_bitmap_pending.Empty()) {
		m_heap_bitmap_pending.ForEach([&](uint64_t begin, uint64_t end) { m_heap_bitmap_ready.Add(begin, end - begin); });
		m_heap_bitmap_pending.Clear();
	}
	if (m_heap_shader_writes.Empty()) return;
	m_heap_shader_writes.ForEach([&](uint64_t begin, uint64_t end) { m_heap_table_writes.Add(begin, end - begin); });
	m_heap_shader_writes.Clear();
}

void BufferCache::CollectHeapWritePages(RangeSet& written) {
	if (m_heap_bitmap_ready.Empty()) return;
	if (m_heap_tables.Empty()) {
		m_heap_bitmap_ready.Clear();
		return;
	}
	RangeSet words;
	m_heap_bitmap_ready.ForEach([&](uint64_t begin, uint64_t end) {
		const auto packed = begin >= LibKernel::Memory::kExtendedMemoryBase
		                        ? begin - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE : begin;
		const auto first = packed / (32 * CACHING_PAGESIZE);
		const auto last = (packed + (end - begin) - 1u) / (32 * CACHING_PAGESIZE);
		words.Add(first * 4u, (last - first + 1u) * 4u);
	});
	m_heap_bitmap_ready.Clear();
	uint64_t bytes = 0;
	std::vector<vk::BufferCopy> copies;
	words.ForEach([&](uint64_t begin, uint64_t end) {
		copies.emplace_back(WRITE_BITMAP_OFFSET + begin, bytes, end - begin);
		bytes += end - begin;
	});
	m_heap_bitmap_bytes += bytes;
	auto download = m_download_buffer.MapDownload(bytes, 4);
	auto& command = m_scheduler.Current();
	auto* bitmap = GetFaultBuffer();
	for (auto& copy: copies) copy.dstOffset += download.offset;
	command.EndRendering();
	const auto native = command.Handle();
	const vk::MemoryBarrier before {.srcAccessMask = vk::AccessFlagBits::eMemoryWrite, .dstAccessMask = vk::AccessFlagBits::eTransferRead};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                       {}, 1, &before, 0, nullptr, 0, nullptr);
	native.copyBuffer(bitmap->Handle(), download.buffer->Handle(), static_cast<uint32_t>(copies.size()), copies.data());
	const vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eTransferWrite, .dstAccessMask = vk::AccessFlagBits::eHostRead};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &after, 0, nullptr, 0, nullptr);
	m_scheduler.Wait(m_scheduler.CurrentTick());
	++m_heap_bitmap_waits;
	download.buffer->Invalidate(download.offset, bytes);
	std::vector<uint64_t> consumed;
	for (const auto& copy: copies) {
		for (uint64_t at = 0; at < copy.size; at += 4u) {
			uint32_t bits;
			std::memcpy(&bits, download.mapped + (copy.dstOffset - download.offset) + at, sizeof(bits));
			const auto page = ((copy.srcOffset - WRITE_BITMAP_OFFSET + at) / 4u) * 32u;
			const auto before = consumed.size();
			bool hit = false;
			while (bits != 0u) {
				const auto address = GuestAddress((page + std::countr_zero(bits)) * CACHING_PAGESIZE);
				consumed.push_back(address);
				m_heap_tables.ForEachInRange(address, CACHING_PAGESIZE, [&](uint64_t begin, uint64_t end) {
					hit = true;
					written.Add(begin, end - begin);
				});
				bits &= bits - 1u;
			}
			if (hit) {
				bitmap->Fill(copy.srcOffset + at, 4, 0);
			} else {
				consumed.resize(before);
			}
		}
	}
	if (!consumed.empty()) m_scheduler.Context().GetIndirectWriteTables().NoteWrites(consumed, true);
}

void BufferCache::EnsureRangeContents(uint64_t vaddr, uint64_t size, RangeSet* uploaded) {
	RequestRange(vaddr, size);
	(void)SynchronizeRange(vaddr, size, false, false, uploaded);
}

std::pair<uint64_t, uint64_t> BufferCache::OwnedRun(uint64_t vaddr, uint64_t size, uint64_t low, uint64_t high) const {
	std::pair<uint64_t, uint64_t> found {vaddr, vaddr + size};
	m_guest_mirror->ForEachOwned(low, high - low, [&](uint64_t begin, uint64_t end) {
		if (begin <= vaddr && vaddr + size <= end) found = {begin, end};
	});
	return found;
}

void BufferCache::ImportGuestMemory(uint64_t vaddr, uint64_t size) {
	m_host_view->Import(vaddr, size);
}

void BufferCache::PrepareFaultBuffer(bool may_write) {
	m_fault_manager.PrepareFaultBuffer();
	m_host_view->Publish();
	m_host_view->NoteReach();
	m_guest_stores = m_guest_stores || (may_write && m_host_view->Enabled());
}

uint64_t BufferCache::ImportGeneration() {
	m_host_view->Publish();
	return m_host_view->Generation();
}

void BufferCache::RequestUnimportedRange(uint64_t vaddr, uint64_t size) {
	if (size == 0 || UINT64_MAX - vaddr < size) return;
	const auto begin = std::max(vaddr, m_guest_mirror->Base());
	const auto end   = std::min(vaddr + size, m_guest_mirror->Base() + m_guest_mirror->Size());
	if (begin >= end) return;
	m_host_view->Publish();
	m_host_view->ForEachGap(begin, end - begin, [this](uint64_t from, uint64_t to) { RequestDeviceRange(from, to - from); });
}

void BufferCache::RequestUnimportedRanges(const RangeSet& mapped, uint64_t mapping_epoch) {
	const std::array layout {mapping_epoch, RegistrationEpoch(), m_host_view->Generation()};
	if (layout == m_unimported_layout) return;
	uint64_t bytes = 0;
	mapped.ForEachInRange(m_guest_mirror->Base(), m_guest_mirror->Size(), [&](uint64_t begin, uint64_t end) {
		m_host_view->ForEachGap(begin, end - begin, [&](uint64_t from, uint64_t to) {
			RequestDeviceRange(from, to - from);
			bytes += to - from;
		});
	});
	m_unimported_bytes  = bytes;
	m_unimported_layout = layout;
}

bool BufferCache::IsResident(uint64_t vaddr) const noexcept {
	const auto block = (vaddr - m_guest_mirror->Base()) / MIRROR_BLOCK_SIZE;
	return vaddr >= m_guest_mirror->Base() && block / 32 < m_residency_words.size() &&
	       (m_residency_words[block / 32] >> (block % 32) & 1u) != 0;
}

void BufferCache::NotePromotion(uint64_t begin, uint64_t end) {
	m_memory_tracker.MarkRegionAsCpuModified(begin, end - begin);
	if (m_guest_stores) m_promoted.push_back({begin, end, m_scheduler.CurrentTick()});
}

void BufferCache::TakeOrderedCopies(std::vector<vk::BufferCopy>& copies, uint64_t& total_size) {
	if (m_promoted.empty()) return;
	std::erase_if(m_promoted, [this](const Promoted& run) { return m_scheduler.GetMasterSemaphore().IsFree(run.tick); });
	if (m_promoted.empty()) return;
	const auto& buffer = *m_storage;
	m_ordered_staged.clear();
	m_ordered_direct.clear();
	uint64_t   staged = 0;
	const auto stage  = [&](uint64_t address, uint64_t bytes) {
		if (bytes == 0) return;
		if (!m_ordered_staged.empty() && m_ordered_staged.back().dstOffset + m_ordered_staged.back().size == buffer.Offset(address)) {
			m_ordered_staged.back().size += bytes;
		} else {
			m_ordered_staged.emplace_back(staged, buffer.Offset(address), bytes);
		}
		staged += bytes;
	};
	const auto promoted = [this](uint64_t address) {
		return std::ranges::any_of(m_promoted, [address](const Promoted& run) { return address >= run.begin && address < run.end; });
	};
	uint64_t ordered = 0;
	for (const auto& copy: copies) {
		const auto end = buffer.HandleOrigin() + copy.dstOffset + copy.size;
		for (auto cursor = buffer.HandleOrigin() + copy.dstOffset; cursor < end;) {
			const auto stop = std::min(end, Common::AlignDown(cursor, MIRROR_BLOCK_SIZE) + MIRROR_BLOCK_SIZE);
			auto       done = cursor;
			if (promoted(cursor)) {
				m_host_view->ForEachSource(cursor, stop - cursor, [&](uint64_t begin, uint64_t bytes, vk::Buffer source, uint64_t offset) {
					stage(done, begin - done);
					m_ordered_direct.emplace_back(source, vk::BufferCopy {offset, buffer.Offset(begin), bytes});
					NoteHeapTransfer(begin, bytes);
					ordered += bytes;
					done = begin + bytes;
				});
			}
			stage(done, stop - done);
			cursor = stop;
		}
	}
	if (m_ordered_direct.empty()) return;
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferRead |
	                       vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	std::vector<vk::BufferCopy> regions;
	for (size_t first = 0; first < m_ordered_direct.size();) {
		regions.clear();
		auto last = first;
		for (; last < m_ordered_direct.size() && m_ordered_direct[last].first == m_ordered_direct[first].first; last++) {
			regions.push_back(m_ordered_direct[last].second);
		}
		auto                    barriers = CopyBarriers(regions, false, before);
		const vk::MemoryBarrier written {.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eMemoryWrite,
		                                 .dstAccessMask = vk::AccessFlagBits::eTransferRead};
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1, &written,
		                       static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
		native.copyBuffer(m_ordered_direct[first].first, buffer.Handle(), static_cast<uint32_t>(regions.size()), regions.data());
		for (auto& after: barriers) {
			after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		}
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data(),
		                       0, nullptr);
		first = last;
	}
	m_host_view->NoteReach();
	m_host_view->NoteOrderedCopy(ordered);
	copies.assign(m_ordered_staged.begin(), m_ordered_staged.end());
	total_size = staged;
}

void BufferCache::ForgetGuestMemory(uint64_t vaddr, uint64_t size) {
	m_host_view->Forget(vaddr, size);
}

void BufferCache::RequestDeviceRange(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0 || size == 0 || !GuestRange {vaddr, size}.Valid()) return;
	RequestRange(vaddr, size);
	m_guest_mirror->NoteDevice(vaddr, size);
}

uint64_t BufferCache::Frame() const noexcept {
	return m_resources != nullptr ? m_resources->FlipNumber() : m_gc_tick;
}

void BufferCache::RequestRange(uint64_t vaddr, uint64_t size) {
	static_assert(GuestMirror::PageBits == CACHING_PAGEBITS);
	if (!m_guest_mirror->Holds(vaddr, size)) {
		EXIT("[kyty-mirror] the GPU uses 0x%" PRIx64 " + 0x%" PRIx64 ", outside the mirror of 0x%" PRIx64
		     " bytes at guest address 0x%" PRIx64 "\n",
		     vaddr, size, m_guest_mirror->Size(), m_guest_mirror->Base());
	}
	m_guest_mirror->Use(vaddr, size);
	if (m_guest_mirror->Owned(vaddr, size)) {
		return;
	}
	m_fresh.clear();
	m_guest_mirror->NoteRange(vaddr, size, &m_fresh);
	if (!m_fresh.empty() && vaddr < m_fresh.front().first && vaddr + size <= m_fresh.back().second) {
		auto last = m_fresh.back().second;
		if (std::find(m_growth_ends.begin(), m_growth_ends.end(), m_fresh.front().first) != m_growth_ends.end()) {
			const auto leap = std::min(StreamLeap, m_guest_mirror->Base() + m_guest_mirror->Size() - last);
			m_guest_mirror->NoteRange(last, leap, &m_fresh);
			last += leap;
		}
		m_growth_ends[m_growth_next++ % m_growth_ends.size()] = last;
	}
	for (const auto& [begin, end]: m_fresh) {
		NoteBlockRegions(begin, end, true);
	}
	m_fresh.clear();
	const auto begin = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	m_guest_mirror->Own(begin, Common::AlignUp(vaddr + size, CACHING_PAGESIZE) - begin, &m_fresh);
	const auto block = m_guest_mirror->BlockSize();
	for (const auto& [first, last]: m_fresh) {
		SetResident(Common::AlignDown(first, block), Common::AlignUp(last, block), true);
	}
	if (!m_fresh.empty()) {
		m_register_epoch++;
	}
}

void BufferCache::SetResident(uint64_t begin, uint64_t end, bool resident) {
	const auto first = (begin - m_guest_mirror->Base()) / MIRROR_BLOCK_SIZE;
	const auto last = (end - m_guest_mirror->Base() - 1) / MIRROR_BLOCK_SIZE;
	uint64_t changed = UINT64_MAX, finish = 0;
	for (auto word = first / 32; word <= last / 32; ++word) {
		const auto low = word == first / 32 ? first % 32 : 0;
		const auto high = word == last / 32 ? last % 32 : 31;
		const uint32_t mask = (~uint32_t {0} << low) & (~uint32_t {0} >> (31 - high));
		auto& bits = m_residency_words[word];
		const auto next = resident ? bits | mask : bits & ~mask;
		for (auto fresh = next & ~bits; fresh != 0;) {
			const auto start  = static_cast<uint32_t>(std::countr_zero(fresh));
			const auto length = static_cast<uint32_t>(std::countr_one(fresh >> start));
			const auto from   = m_guest_mirror->Base() + (word * 32 + start) * MIRROR_BLOCK_SIZE;
			NotePromotion(from, from + length * MIRROR_BLOCK_SIZE);
			fresh = start + length >= 32 ? 0 : fresh & ~((uint32_t {1} << (start + length)) - 1);
		}
		if (bits != next) {
			bits = next;
			changed = std::min(changed, word);
			finish = word + 1;
		}
	}
	if (changed == UINT64_MAX) return;
	const auto bytes = (finish - changed) * sizeof(uint32_t);
	const auto source = m_staging_buffer.Copy(m_residency_words.data() + changed, bytes, sizeof(uint32_t));
	GetFaultBuffer()->CopyFrom(m_scheduler.Current(), m_staging_buffer, source,
	                           RESIDENCY_OFFSET + changed * sizeof(uint32_t), bytes);
}

void BufferCache::NoteBlockRegions(uint64_t begin, uint64_t end, bool insert) {
	const auto block = m_guest_mirror->BlockSize();
	GrowRegionTables((end - 1) / TRACKER_REGION_SIZE);
	for (auto index = begin / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < end; index++) {
		const auto from   = std::max(begin, index * TRACKER_REGION_SIZE);
		const auto to     = std::min(end, (index + 1) * TRACKER_REGION_SIZE);
		const auto blocks = static_cast<uint32_t>(std::max<uint64_t>((to - from) / block, 1));
		const auto bit    = uint64_t {1} << (index % 64);
		if (insert) {
			m_region_buffer_count[index] += blocks;
			m_buffer_region_bits[index / 64] |= bit;
			m_forced_region_bits[index / 64] |= bit;
			m_region_forces[index]++;
		} else {
			m_region_buffer_count[index] -= std::min(m_region_buffer_count[index], blocks);
			if (m_region_buffer_count[index] == 0) {
				m_buffer_region_bits[index / 64] &= ~bit;
			}
		}
	}
}


bool BufferCache::NoImageStartsAt(uint64_t vaddr) const noexcept {
	TextureCache::ImagePageTable::PageRange pages {};
	return TextureCache::ImagePageTable::TryGetPageRange(vaddr, 1, pages) &&
	       !m_texture_cache.m_registered_pages.Any(pages.first, pages.last_exclusive);
}

bool BufferCache::SynchronizeRange(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer, RangeSet* uploaded) {
	if (!is_written && !m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		if (is_texel_buffer && !NoImageStartsAt(vaddr)) {
			return SynchronizeRangeFromImage(vaddr, size);
		}
		return false;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	const MemoryTracker::UploadScope upload_scope;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, m_storage->Offset(address), bytes);
		    total_size += bytes;
		    NoteUpload(address, bytes);
	    },
	    [&]() noexcept {
		    const auto source = UploadCopies(copies, total_size);
		    if (uploaded != nullptr) {
			    for (const auto& copy: copies) {
				    const auto address = m_storage->HandleOrigin() + copy.dstOffset;
				    const auto first = std::max(address, vaddr);
				    const auto last = std::min<uint64_t>(address + copy.size, vaddr + size);
				    if (first < last) uploaded->Add(first, last - first);
			    }
		    }
		    RecordUpload(copies, source);
		    copies.clear();
		    total_size = 0;
	    },
	    m_device_sync_pass);
	if (is_texel_buffer && !is_written && !NoImageStartsAt(vaddr)) {
		return SynchronizeRangeFromImage(vaddr, size);
	}
	return false;
}

void BufferCache::RecordUpload(std::span<const vk::BufferCopy> copies, vk::Buffer source) {
	const auto& buffer = *m_storage;
	if (source && m_upload_batching && source == m_staging_buffer.Handle()) {
		m_upload_batch_counts.push_back(static_cast<uint32_t>(copies.size()));
		m_upload_batch_copies.insert(m_upload_batch_copies.end(), copies.begin(), copies.end());
	} else if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		auto barriers              = CopyBarriers(copies, false, before);
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr,
		                       static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		for (auto& after: barriers) {
			after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		}
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr,
		                       static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
	}
}

void BufferCache::SynchronizeRanges(std::span<const std::pair<uint64_t, uint64_t>> ranges) {
	if (m_span_copies.size() < ranges.size()) {
		m_span_copies.resize(ranges.size());
	}
	m_span_sizes.assign(ranges.size(), 0);
	for (size_t index = 0; index < ranges.size(); index++) {
		m_span_copies[index].clear();
	}
	const MemoryTracker::UploadScope upload_scope;
	m_memory_tracker.ForEachUploadRanges(
	    ranges,
	    [&](size_t index, uint64_t address, uint64_t bytes) noexcept {
		    m_span_copies[index].emplace_back(m_span_sizes[index], m_storage->Offset(address), bytes);
		    m_span_sizes[index] += bytes;
		    NoteUpload(address, bytes);
	    },
	    [&]() noexcept {
		    for (size_t index = 0; index < ranges.size(); index++) {
			    const auto source = UploadCopies(m_span_copies[index], m_span_sizes[index]);
			    RecordUpload(m_span_copies[index], source);
		    }
	    },
	    m_device_sync_pass);
}

vk::Buffer BufferCache::UploadCopies(std::vector<vk::BufferCopy>& copies, uint64_t total_size) {
	const auto& buffer = *m_storage;
	if (copies.empty()) {
		return nullptr;
	}
	TakeOrderedCopies(copies, total_size);
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);
	g_write_block_stats.uploaded.fetch_add(total_size, std::memory_order_relaxed);

	auto*      host_copies = m_scheduler.HostCopies();
	const auto ring        = m_staging_buffer.Mapped();
	const bool hand_over   = host_copies != nullptr && m_staging_buffer.IsCoherent() &&
	                         GuestGpu::IsGpuThread();
	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		const bool in_ring =
		    hand_over && mapped >= ring.data() && mapped < ring.data() + ring.size();
		if (auto& deferred = MemoryTracker::DeferredProtects(); !deferred.empty()) {
			bool queued = in_ring;
			for (const auto& copy: copies) {
				queued = queued && LibKernel::Memory::FindBackingPointer(buffer.HandleOrigin() + copy.dstOffset, copy.size) != nullptr;
			}
			for (const auto& item: deferred) {
				if (queued) {
					host_copies->Call(
					    [](void* manager, uint64_t address, uint64_t size) {
						    static_cast<RegionManager*>(manager)->ApplyDeferredProtection(address, size);
					    },
					    item.manager, item.address, item.size);
				} else {
					item.manager->ApplyDeferredProtection(item.address, item.size);
				}
			}
			deferred.clear();
			if (queued) m_protect_jobs = host_copies->Flush();
		}
		for (auto& copy: copies) {
			const auto  address = buffer.HandleOrigin() + copy.dstOffset;
			const void* backing =
			    in_ring ? LibKernel::Memory::FindBackingPointer(address, copy.size) : nullptr;
			if (backing != nullptr) {
				host_copies->Copy(mapped + copy.srcOffset, backing, copy.size);
			} else {
				if (m_protect_jobs != 0 && host_copies != nullptr) {
					host_copies->WaitFor(m_protect_jobs);
					m_protect_jobs = 0;
				}
				if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
					EXIT("BufferCache: failed to read sparse guest buffer memory\n");
				}
			}
			copy.srcOffset += base_offset;
		}
	}, !m_upload_batching);
	return source.first->Handle();
}

void BufferCache::StreamGuestData(uint8_t* destination, uint64_t vaddr, uint64_t size) {
	auto* copies = m_scheduler.HostCopies();
	if (copies != nullptr && m_stream_buffer.IsCoherent() && GuestGpu::IsGpuThread()) {
		if (const void* source = Libs::LibKernel::Memory::FindBackingPointer(vaddr, size)) {
			if (m_stream_in_place) {
				copies->CopyInline(destination, source, size);
			} else {
				copies->Copy(destination, source, size);
			}
			return;
		}
	}
	if (!LibKernel::Memory::TryReadBacking(vaddr, destination, size) &&
	    !LibKernel::Memory::CopySparseMemory(vaddr, destination, size)) {
		EXIT("BufferCache: failed to read sparse guest buffer memory at 0x%" PRIx64 " size 0x%" PRIx64
		     "\n",
		     vaddr, size);
	}
}

std::optional<uint64_t> BufferCache::StreamGuestRange(uint64_t vaddr, uint64_t size) {
	if (size == 0 || m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return std::nullopt;
	}
	const uint64_t base   = vaddr & ~uint64_t {255};
	const uint64_t bytes  = vaddr + size - base;
	const void*    source = Libs::LibKernel::Memory::FindBackingPointer(base, bytes);
	if (source == nullptr) {
		return std::nullopt;
	}
	auto [mapped, offset] = m_stream_buffer.Map(bytes, 256, false);
	if (mapped == nullptr) {
		return std::nullopt;
	}
	auto* copies = m_scheduler.HostCopies();
	if (copies != nullptr && m_stream_buffer.IsCoherent() && GuestGpu::IsGpuThread()) {
		if (m_stream_in_place) {
			copies->CopyInline(mapped, source, bytes);
		} else {
			copies->Copy(mapped, source, bytes);
		}
	} else {
		std::memcpy(mapped, source, bytes);
	}
	m_stream_buffer.Commit();
	return offset + (vaddr - base);
}

std::optional<uint64_t> BufferCache::FindStreamCopy(uint64_t vaddr, uint64_t size) {
	if (m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return std::nullopt;
	}
	const auto& stream = (*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) % StreamMemoSlots];
	if (stream.vaddr == vaddr && stream.size == size &&
	    stream.window == m_scheduler.Context().GetRenderExecutor().DrawWindow() &&
	    stream.lap == m_stream_buffer.Lap() && stream.write_generation == m_gpu_write_generation) {
		return stream.offset;
	}
	return std::nullopt;
}

void BufferCache::NoteStreamCopy(uint64_t vaddr, uint64_t size, uint64_t offset) {
	(*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) % StreamMemoSlots] = {
	    .vaddr            = vaddr,
	    .size             = size,
	    .window           = m_scheduler.Context().GetRenderExecutor().DrawWindow(),
	    .lap              = m_stream_buffer.Lap(),
	    .write_generation = m_gpu_write_generation,
	    .offset           = offset};
}

StreamBuffer& BufferCache::GetTableUploadBuffer() {
	if (!m_table_upload) {
		m_table_upload = std::make_unique<StreamBuffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 32 * MiB);
	}
	return *m_table_upload;
}

vk::DeviceAddress BufferCache::TableUpload(const void* data, uint64_t size) {
	if (!m_table_upload) {
		m_table_upload = std::make_unique<StreamBuffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 32 * MiB);
	}
	return m_table_upload->BufferDeviceAddress() + m_table_upload->Copy(data, size, 16);
}

std::pair<Buffer*, uint64_t> BufferCache::UploadHeapData(
    uint64_t size, const std::function<void(uint8_t*, uint64_t)>& write) {
	return UploadMemory(m_graphics, m_scheduler, m_stream_buffer, size, 16,
	                    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress |
	                        vk::BufferUsageFlagBits::eStorageBuffer, write, false);
}

uint64_t BufferCache::StreamUpload(const void* data, uint64_t size, uint64_t alignment) {
	auto* copies = m_scheduler.HostCopies();
	if (copies == nullptr || !m_stream_buffer.IsCoherent() || !GuestGpu::IsGpuThread()) {
		return m_stream_buffer.Copy(data, size, alignment);
	}
	const auto [mapped, offset] = m_stream_buffer.Map(size, alignment);
	EXIT_IF(mapped == nullptr);
	copies->CopyInline(mapped, data, size);
	m_stream_buffer.Commit();
	return offset;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainRange(uint64_t vaddr, uint64_t size, bool is_written,
                                                      bool is_texel_buffer, ObtainFrom from, HeapWriteSource writer) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));
	if (is_written) {
		m_scheduler.Context().CollectHotPageWrites(vaddr, size);
	}
	return ObtainSynchronizedRange(vaddr, size, is_written, is_texel_buffer, from, writer);
}

void BufferCache::CheckMirrorRange(uint64_t vaddr, uint64_t size) noexcept {
	if (m_guest_mirror->Requested(vaddr, size)) return;
	m_guest_mirror->NoteUnboundRead();
	if (MirrorCheck()) {
		EXIT("[kyty-mirror] 0x%" PRIx64 " + 0x%" PRIx64 " is used from the mirror while a block of it is not bound\n",
		     vaddr, size);
	}
}

void BufferCache::CheckMirrorCopies(std::span<const vk::BufferCopy> copies) noexcept {
	for (const auto& copy: copies) {
		CheckMirrorRange(m_storage->HandleOrigin() + copy.srcOffset, copy.size);
	}
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainSynchronizedRange(uint64_t vaddr, uint64_t size, bool is_written,
                                                                  bool is_texel_buffer, ObtainFrom from, HeapWriteSource writer) {
	if (!is_written && size <= CACHING_PAGESIZE && from == ObtainFrom::Any &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		const auto window = m_scheduler.Context().GetRenderExecutor().DrawWindow();
		auto&      stream = (*m_stream_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) % StreamMemoSlots];
		if (stream.vaddr == vaddr && stream.size == size && stream.window == window && stream.lap == m_stream_buffer.Lap() &&
		    stream.write_generation == m_gpu_write_generation) {
			return {&m_stream_buffer, stream.offset};
		}
		if (m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
			const auto alignment =
			    std::max<uint64_t>(m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
			auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
			if (mapped != nullptr) {
				StreamGuestData(mapped, vaddr, size);
				m_stream_buffer.Commit();
				stream = {.vaddr            = vaddr,
				          .size             = size,
				          .window           = window,
				          .lap              = m_stream_buffer.Lap(),
				          .write_generation = m_gpu_write_generation,
				          .offset           = offset};
				return {&m_stream_buffer, offset};
			}
		}
	}
	RequestRange(vaddr, size);
	if (!is_written) {
		(void)SynchronizeRange(vaddr, size, false, is_texel_buffer);
		return {m_storage.get(), m_storage->Offset(vaddr)};
	}
	const WrittenMemo now {.vaddr        = vaddr,
	                       .size         = size,
	                       .cpu_epoch    = RegionManager::CpuDirtyEpoch(),
	                       .gpu_epoch    = RegionManager::GpuCleanEpoch(),
	                       .region_epoch = CpuModificationEpoch(vaddr, size),
	                       .registration = RegistrationEpoch()};
	auto& memo = (*m_written_memo)[((vaddr ^ (size << 40u)) * 0x9e3779b97f4a7c15ull >> 32u) % WrittenMemoSlots];
	if (memo.Covers(now)) {
		if (MirrorCheck()) CheckMirrorRange(vaddr, size);
	} else {
		(void)SynchronizeRange(vaddr, size, true, is_texel_buffer);
		memo = now;
	}
	m_gpu_write_generation++;
	if (m_gpu_modified_ranges.Add(vaddr, size)) {
		NoteGpuModifiedChange(vaddr, size, true);
	}
	ForgetColorMetadataChecks(vaddr, size);
	NoteWrite(vaddr, size, writer);
	return {m_storage.get(), m_storage->Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainRangeWithProvenWrites(uint64_t vaddr, uint64_t size, std::span<const GuestRange> writes) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()) || writes.empty());
	if ((vaddr & 3u) != 0) {
		return ObtainRange(vaddr, size, true, false, ObtainFrom::Any, HeapWriteSource::Shader);
	}
	RequestRange(vaddr, size);
	(void)SynchronizeRange(vaddr, size, false, false);

	++m_gpu_write_generation;
	for (const auto& range: writes) {
		m_scheduler.Context().CollectHotPageWrites(range.address, range.size);
		(void)SynchronizeRange(range.address, range.size, true, false);
		if (m_gpu_modified_ranges.Add(range.address, range.size)) {
			NoteGpuModifiedChange(range.address, range.size, true);
		}
		ForgetColorMetadataChecks(range.address, range.size);
		NoteWrite(range.address, range.size, HeapWriteSource::Shader);
	}
	return {m_storage.get(), m_storage->Offset(vaddr)};
}

bool BufferCache::PrepareDeviceAddressRead(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return true;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		Common::HackNotice::Report("device address read outside guest memory skipped with the ray tracing data that "
		                           "needed it",
		                           "BufferCache::PrepareDeviceAddressRead", 0, 0,
		                           fmt::format("0x{:x}, {} bytes", vaddr, size));
		return false;
	}
	RequestRange(vaddr, size);
	m_guest_mirror->NoteDevice(vaddr, size);
	(void)SynchronizeRange(vaddr, size, false, false);
	return true;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainRangeForImage(uint64_t vaddr, uint64_t size) {
	if (m_guest_mirror->Holds(vaddr, size) && Holds(vaddr, size)) {
		m_guest_mirror->Use(vaddr, size);
		(void)SynchronizeRange(vaddr, size, false, false);
		return {m_storage.get(), m_storage->Offset(vaddr)};
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainRange(vaddr, size, false, false);
	}
	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size) &&
		    !Libs::LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::DrainHostCopies() {
	if (auto* copies = m_scheduler.HostCopies(); copies != nullptr && GuestGpu::IsGpuThread()) {
		copies->Drain();
	}
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	m_gpu_write_generation++;
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		InvalidateMemory(vaddr, size);
		m_texture_cache.InvalidateMemory(vaddr, size);
		DrainHostCopies();
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		LibKernel::Memory::NoteEmulatorWrite(vaddr, size);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainRange(vaddr, size, true, true);
	m_texture_cache.ProtectPendingImageWriteBacks(vaddr, size);
	dst->Fill(dst_offset, size, value);
}

bool BufferCache::ReplicateBuffer(uint64_t destination, uint64_t count, uint64_t source,
                                  uint64_t pattern) {
	const auto size = count * sizeof(uint32_t);
	const auto from = pattern * sizeof(uint32_t);
	if (destination == 0 || source == 0 || ((destination | source) & 3u) != 0 || count == 0 ||
	    pattern == 0 || (destination < source + from && source < destination + size) ||
	    IsRegionGpuModified(destination, size) || IsRegionGpuModified(source, from) ||
	    m_texture_cache.FindImageFromRange(source, from)) {
		return false;
	}
	m_gpu_write_generation++;
	(void)m_texture_cache.ClearMeta(destination);
	InvalidateMemory(destination, size);
	m_texture_cache.InvalidateMemory(destination, size);
	DrainHostCopies();
	auto*      target = reinterpret_cast<uint8_t*>(destination);
	const auto first  = std::min(size, from);
	std::memcpy(target, reinterpret_cast<const void*>(source), first);
	for (uint64_t written = first; written < size;) {
		const auto chunk = std::min(written, size - written);
		std::memcpy(target + written, target, chunk);
		written += chunk;
	}
	LibKernel::Memory::NoteEmulatorWrite(destination, size);
	return true;
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	m_gpu_write_generation++;
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		DrainHostCopies();
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		LibKernel::Memory::NoteEmulatorWrite(dst_vaddr, size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	auto [src, src_offset] = src_memory ? ObtainRange(src_vaddr, size, false, true)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainRange(dst_vaddr, size, true, true)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	if (dst_memory) {
		m_texture_cache.ProtectPendingImageWriteBacks(dst_vaddr, size);
	}
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
	if (dst_memory) {
		m_texture_cache.ApplyPartialBufferCopy(dst_vaddr, size, *dst, dst_offset);
	}
}

void BufferCache::WriteTicks::Assign(uint64_t begin, uint64_t end, uint64_t tick) {
	if (begin >= end) {
		return;
	}
	auto& recent = m_recent[((begin ^ (end << 24u)) * 0x9e3779b97f4a7c15ull) >> 56u];
	if (recent.begin == begin && recent.end == end && recent.tick == tick) {
		return;
	}
	recent = {begin, end, tick};
	auto it = m_ranges.lower_bound(begin);
	if (it != m_ranges.end() && it->first == begin && it->second.end == end) {
		it->second.tick = tick;
		return;
	}
	if (it != m_ranges.begin()) {
		const auto previous = std::prev(it);
		if (previous->second.end > begin) {
			const auto right       = previous->second;
			previous->second.end = begin;
			it                     = m_ranges.emplace_hint(it, begin, right);
		}
	}
	while (it != m_ranges.end() && it->first < end) {
		if (it->second.end > end) {
			const auto tail = it->second;
			m_ranges.erase(it);
			m_ranges.emplace(end, tail);
			break;
		}
		it = m_ranges.erase(it);
	}
	m_ranges.emplace(begin, Range {end, tick});
}

uint64_t BufferCache::WriteTicks::Latest(uint64_t begin, uint64_t end) const {
	auto it = m_ranges.upper_bound(begin);
	if (it != m_ranges.begin()) {
		--it;
	}
	uint64_t latest = 0;
	uint64_t cursor = begin;
	for (; it != m_ranges.end() && it->first < end && cursor < end; ++it) {
		if (it->second.end <= cursor) {
			continue;
		}
		if (it->first > cursor) {
			return UINT64_MAX;
		}
		latest = std::max(latest, it->second.tick);
		cursor = it->second.end;
	}
	return cursor >= end ? latest : UINT64_MAX;
}

void BufferCache::NoteReadbackWait(uint64_t begin, uint64_t end, bool device_address,
                                   bool drained) {
	const auto count = ++m_readback_count;
	if (device_address && (drained || m_device_address_hot != 0)) {
		m_device_address_hot = count;
	}
	std::erase_if(m_readback_hot, [count](const HotReadback& hot) {
		return count - hot.last_hit >= HotReadbackLifetime;
	});
	begin = Common::AlignDown(begin, TRACKER_PAGE_SIZE);
	end   = Common::AlignUp(end, TRACKER_PAGE_SIZE);
	for (auto& hot: m_readback_hot) {
		if (begin < hot.end && hot.begin < end) {
			hot.begin    = std::min(hot.begin, begin);
			hot.end      = std::max(hot.end, end);
			hot.last_hit = count;
			return;
		}
	}
	if (drained && !device_address) {
		if (m_readback_hot.size() == MaxHotReadbacks) {
			m_readback_hot.erase(std::ranges::min_element(m_readback_hot, {}, &HotReadback::last_hit));
		}
		m_readback_hot.push_back({begin, end, count});
	}
}

bool BufferCache::TakeFlushRequest() {
	if (!m_flush_requested) {
		return false;
	}
	if (m_flush_deferred_tick != 0 && m_scheduler.CurrentTick() != m_flush_deferred_tick) {
		m_flush_requested     = false;
		m_flush_deferred_tick = 0;
		return false;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - m_last_requested_flush < MinRequestedFlushInterval) {
		m_flush_deferred_tick = m_scheduler.CurrentTick();
		return false;
	}
	m_last_requested_flush = now;
	m_flush_requested      = false;
	m_flush_deferred_tick  = 0;
	return true;
}

bool BufferCache::CommitWriteTicks() {
	if (m_pending_writes.empty() && !m_device_address_pending) {
		return TakeFlushRequest();
	}
	const auto tick = m_scheduler.CurrentTick();
	for (const auto& [vaddr, size]: m_pending_writes) {
		m_write_ticks.Assign(vaddr, vaddr + size, tick);
	}
	if (!m_mirror_lines.empty()) {
		for (const auto& [vaddr, size]: m_pending_writes) {
			MarkMirrorLinesWritten(vaddr, size, tick);
		}
	}
	m_pending_writes.clear();
	if (m_device_address_pending) {
		m_device_address_pending = false;
		m_device_address_tick    = tick;
	}
	return TakeFlushRequest();
}

void BufferCache::MarkMirrorLinesWritten(uint64_t vaddr, uint64_t size, uint64_t tick) {
	const auto end = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
	for (auto it = m_mirror_lines.lower_bound(Common::AlignDown(vaddr, MirrorLineSize));
	     it != m_mirror_lines.end() && it->first < end; ++it) {
		auto& line = it->second;
		line.tick  = 0;
		if (m_mirror_uses - line.last_use < MirrorFlushLifetime) {
			m_flush_requested = true;
		}
		if (line.dirty_tick != tick) {
			line.dirty_tick = tick;
			m_mirror_dirty.push_back(it->first);
		}
	}
}

void BufferCache::AddMirrorLine(uint64_t line) {
	if (m_mirror_free_slots.empty()) {
		if (m_mirror_lines.size() < MaxMirrorLines) {
			m_mirror_free_slots.push_back(static_cast<uint32_t>(m_mirror_lines.size()));
		} else {
			std::vector<uint64_t> uses;
			uses.reserve(m_mirror_lines.size());
			for (const auto& [address, entry]: m_mirror_lines) {
				uses.push_back(entry.last_use);
			}
			std::nth_element(uses.begin(), uses.begin() + uses.size() / 2, uses.end());
			const auto oldest = uses[uses.size() / 2];
			for (auto it = m_mirror_lines.begin(); it != m_mirror_lines.end();) {
				if (it->second.last_use <= oldest) {
					m_mirror_free_slots.push_back(it->second.slot);
					it = m_mirror_lines.erase(it);
				} else {
					++it;
				}
			}
		}
	}
	MirrorLine entry;
	entry.slot = m_mirror_free_slots.back();
	m_mirror_free_slots.pop_back();
	entry.last_use = ++m_mirror_uses;
	entry.dirty_tick = m_scheduler.CurrentTick();
	m_mirror_lines.emplace(line, entry);
	m_mirror_dirty.push_back(line);
}

bool BufferCache::TryReadMirroredGpuWrites(uint64_t vaddr, void* data, uint64_t size) {
	if (size == 0 || size > MirrorMaxRead || m_device_address_pending ||
	    !GuestGpu::IsGpuThread()) {
		return false;
	}
	if (InFlightReadbackTick(vaddr, vaddr + size) != 0) {
		return false;
	}
	for (const auto& [address, bytes]: m_pending_writes) {
		if (address < vaddr + size && vaddr < address + bytes) {
			return false;
		}
	}
	uint64_t needed  = m_device_address_tick;
	bool     unknown = false;
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
		const auto latest = m_write_ticks.Latest(start, end);
		unknown |= latest == UINT64_MAX;
		needed = std::max(needed, latest);
	});
	if (unknown) {
		return false;
	}
	const auto first = Common::AlignDown(vaddr, MirrorLineSize);
	const auto last  = Common::AlignUp(vaddr + size, MirrorLineSize);
	uint64_t   wait  = 0;
	bool       held  = true;
	std::array<uint32_t, MirrorMaxRead / MirrorLineSize + 1> slots {};
	for (auto line = first; line < last; line += MirrorLineSize) {
		const auto it = m_mirror_lines.find(line);
		if (it == m_mirror_lines.end()) {
			AddMirrorLine(line);
			held = false;
			continue;
		}
		auto& entry    = it->second;
		entry.last_use = ++m_mirror_uses;
		slots[(line - first) / MirrorLineSize] = entry.slot;
		if (entry.tick == 0 || entry.tick < needed) {
			held = false;
			if (entry.dirty_tick != m_scheduler.CurrentTick()) {
				entry.dirty_tick = m_scheduler.CurrentTick();
				m_mirror_dirty.push_back(line);
			}
		}
		wait = std::max(wait, entry.tick);
	}
	if (!held || wait >= m_scheduler.CurrentTick()) {
		return false;
	}
	m_scheduler.GetMasterSemaphore().Wait(wait);
	m_mirror_flush_use = m_mirror_uses;
	m_scheduler.WaitPriorityOperations(needed);
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, data, size)) {
		return false;
	}
	const auto* mapped = m_mirror->Mapped().data();
	m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t start, uint64_t end) {
		while (start < end) {
			const auto line   = Common::AlignDown(start, MirrorLineSize);
			const auto chunk  = std::min(end, line + MirrorLineSize) - start;
			const auto slot   = slots[(line - first) / MirrorLineSize];
			const auto offset = uint64_t {slot} * MirrorLineSize + (start - line);
			m_mirror->Invalidate(offset, chunk);
			std::memcpy(static_cast<uint8_t*>(data) + (start - vaddr), mapped + offset, chunk);
			Libs::LibKernel::Memory::WriteBacking(start, mapped + offset, chunk);
			start += chunk;
		}
	});
	if (m_gpu_modified_ranges.Subtract(vaddr, size)) {
		NoteGpuModifiedChange(vaddr, size, false);
		UnmarkPagesWithoutGpuBytes(vaddr, size);
	}
	return true;
}

void BufferCache::UnmarkPagesWithoutGpuBytes(uint64_t vaddr, uint64_t size) {
	const auto end = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE); page < end; page += TRACKER_PAGE_SIZE) {
		if (m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE) &&
		    !m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
		    InFlightReadbackTick(page, page + TRACKER_PAGE_SIZE) == 0) {
			UnmarkGpuModified(page, TRACKER_PAGE_SIZE);
		}
	}
}

void BufferCache::RecordMirrorCopies(CommandBuffer& command) {
	if (m_mirror_dirty.empty() || !GuestGpu::IsGpuThread()) {
		return;
	}
	if (!m_mirror) {
		m_mirror = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                    vk::BufferUsageFlagBits::eTransferDst,
		                                    uint64_t {MaxMirrorLines} * MirrorLineSize);
	}
	std::sort(m_mirror_dirty.begin(), m_mirror_dirty.end());
	m_mirror_dirty.erase(std::unique(m_mirror_dirty.begin(), m_mirror_dirty.end()),
	                     m_mirror_dirty.end());
	const auto tick = m_scheduler.CurrentTick();
	command.EndRendering();
	const auto native = command.Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	std::vector<vk::BufferCopy> copies;
	for (const auto line: m_mirror_dirty) {
		const auto it = m_mirror_lines.find(line);
		if (it == m_mirror_lines.end() || !Holds(line, MirrorLineSize)) {
			continue;
		}
		copies.push_back({m_storage->Offset(line), uint64_t {it->second.slot} * MirrorLineSize, MirrorLineSize});
		it->second.tick = tick;
	}
	if (!copies.empty()) {
		native.copyBuffer(m_storage->Handle(), m_mirror->Handle(), static_cast<uint32_t>(copies.size()), copies.data());
	}
	m_mirror_dirty.clear();
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &after, 0, nullptr, 0, nullptr);
}

bool BufferCache::DownloadOnReadbackQueue(std::span<const vk::BufferCopy> copies, uint64_t total_size) {
	if (!m_graphics.readback_queue || m_device_address_pending) {
		return false;
	}
	const auto& buffer         = *m_storage;
	const auto  buffer_address = buffer.HandleOrigin();
	uint64_t needed = m_device_address_tick;
	const auto readback_begin = buffer_address + copies.front().srcOffset;
	const auto readback_end   = buffer_address + copies.back().srcOffset + copies.back().size;
	for (const auto& copy: copies) {
		const auto begin = buffer_address + copy.srcOffset;
		const auto end   = begin + copy.size;
		for (const auto& [vaddr, size]: m_pending_writes) {
			if (vaddr < end && begin < vaddr + size) {
				NoteReadbackWait(readback_begin, readback_end, false, true);
				return false;
			}
		}
		needed = std::max(needed, m_write_ticks.Latest(begin, end));
	}
	const bool device_address = needed != 0 && needed == m_device_address_tick;
	if (needed >= m_scheduler.CurrentTick()) {
		NoteReadbackWait(readback_begin, readback_end, device_address, true);
		return false;
	}
	NoteReadbackWait(readback_begin, readback_end, device_address, false);
#if defined(ANYPS5_KYTY)
	const auto submit_from = std::chrono::steady_clock::now();
	bool       submit_reported = false;
	for (uint32_t spins = 0; m_scheduler.SubmittedTick() < needed; ++spins) {
		if (!submit_reported && (spins & 1023u) == 1023u &&
		    std::chrono::steady_clock::now() - submit_from >= std::chrono::seconds(10)) {
			submit_reported = true;
			std::fprintf(stderr,
			             "[kyty] a readback of 0x%" PRIx64 "-0x%" PRIx64 " has waited 10 s for tick %" PRIu64
			             " to be submitted (submitted %" PRIu64 ")\n",
			             readback_begin, readback_end, needed, m_scheduler.SubmittedTick());
		}
		Common::SpinPause();
	}
#else
	while (m_scheduler.SubmittedTick() < needed) {
		Common::SpinPause();
	}
#endif
	auto& device = m_graphics.device;
	if (!m_readback.pool) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
		                             vk::CommandPoolCreateFlagBits::eTransient;
		pool_info.queueFamilyIndex = m_graphics.readback_family;
		vk::CommandBufferAllocateInfo allocate {};
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		const vk::FenceCreateInfo fence_info {};
		if (device.createCommandPool(&pool_info, nullptr, &m_readback.pool) != vk::Result::eSuccess ||
		    (allocate.commandPool = m_readback.pool,
		     device.allocateCommandBuffers(&allocate, &m_readback.command) != vk::Result::eSuccess) ||
		    device.createFence(&fence_info, nullptr, &m_readback.fence) != vk::Result::eSuccess) {
			EXIT("BufferCache: could not create the readback queue resources\n");
		}
	}
	if (!m_readback.staging || m_readback.staging->Size() < total_size) {
		m_readback.staging = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Download, 0,
		    vk::BufferUsageFlagBits::eTransferDst,
		    std::max<uint64_t>(Common::AlignUp(total_size, 1024 * 1024), 4 * MiB));
	}
	auto&                       staging = *m_readback.staging;
	const auto command = m_readback.command;
	(void)command.reset({});
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	(void)command.begin(&begin_info);
	command.copyBuffer(buffer.Handle(), staging.Handle(), static_cast<uint32_t>(copies.size()),
	                   copies.data());
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = staging.Handle();
	after.offset              = 0;
	after.size                = total_size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &after, 0, nullptr);
	(void)command.end();
	const auto                      semaphore = m_scheduler.GetMasterSemaphore().Handle();
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	timeline.pWaitSemaphoreValues    = &needed;
	vk::SubmitInfo submit {};
	submit.pNext              = &timeline;
	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores    = &semaphore;
	submit.pWaitDstStageMask  = &wait_stage;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	if (m_graphics.readback_queue.submit(1, &submit, m_readback.fence) != vk::Result::eSuccess) {
		EXIT("BufferCache: readback queue submission failed\n");
	}
#if defined(ANYPS5_KYTY)
	constexpr auto     SpinLimit = std::chrono::seconds(10);
	constexpr uint64_t StallNs   = 10'000'000'000ull;
	const auto         spin_from = std::chrono::steady_clock::now();
	for (uint32_t spins = 0;; ++spins) {
		const auto status = device.getFenceStatus(m_readback.fence);
		if (status == vk::Result::eSuccess) {
			break;
		}
		if (status != vk::Result::eNotReady) {
			EXIT("BufferCache: readback queue wait failed: %s\n", vk::to_string(status).c_str());
		}
		if ((spins & 1023u) == 1023u && std::chrono::steady_clock::now() - spin_from >= SpinLimit) {
			const auto report = [&] {
				auto& master = m_scheduler.GetMasterSemaphore();
				master.Refresh();
				if (needed != 0 && !master.IsFree(needed)) {
					master.ReportStall(needed);
					return;
				}
				static std::atomic_flag reported = ATOMIC_FLAG_INIT;
				if (!reported.test_and_set()) {
					std::fprintf(stderr,
					             "[kyty] a readback copy of 0x%" PRIx64 "-0x%" PRIx64 " on the readback queue has not "
					             "finished in 10 s although the GPU reached its tick %" PRIu64 " (GPU at %" PRIu64
					             "): the readback queue is stuck\n",
					             readback_begin, readback_end, needed, master.KnownGpuTick());
				}
			};
			report();
			auto result = device.waitForFences(1, &m_readback.fence, VK_TRUE, StallNs);
			while (result == vk::Result::eTimeout) {
				report();
				result = device.waitForFences(1, &m_readback.fence, VK_TRUE, StallNs);
			}
			if (result != vk::Result::eSuccess) {
				EXIT("BufferCache: readback queue wait failed: %s\n", vk::to_string(result).c_str());
			}
			break;
		}
		Common::SpinPause();
	}
#else
	for (;;) {
		const auto status = device.getFenceStatus(m_readback.fence);
		if (status == vk::Result::eSuccess) {
			break;
		}
		if (status != vk::Result::eNotReady) {
			EXIT("BufferCache: readback queue wait failed: %s\n", vk::to_string(status).c_str());
		}
		Common::SpinPause();
	}
#endif
	(void)device.resetFences(1, &m_readback.fence);
	m_scheduler.WaitPriorityOperations(needed);
	staging.Invalidate(0, total_size);
	const auto* mapped = staging.Mapped().data();
	for (const auto& copy: copies) {
		Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
		                                      mapped + copy.dstOffset, copy.size);
	}
	return true;
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	return m_guest_mirror->AnyOwned(vaddr, size);
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
	m_texture_cache.NoteMetadataWrite(vaddr, size);
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::IsRegionTrackedGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::WriteAroundGpuWrites(uint64_t vaddr, const void* data, uint64_t size) {
	constexpr uint64_t MaxBytes = 4096;
	if (size == 0 || size > MaxBytes || (vaddr & 3u) != 0 || (size & 3u) != 0 ||
	    !GuestRange {vaddr, size}.Valid() || !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	Buffer* buffer = Holds(vaddr, size) ? m_storage.get() : nullptr;
	if (buffer == nullptr && IsRegionRegistered(vaddr, size)) {
		return false;
	}
	if (buffer == nullptr && HasGpuDirtyBytes(vaddr, size)) {
		return false;
	}
	if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	if (buffer == nullptr) {
		return true;
	}
	auto&      stream  = GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset  = StreamUpload(data, size, 4);
	auto&      command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer->Handle();
	before.offset              = buffer->Offset(vaddr);
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {offset, buffer->Offset(vaddr), size};
	native.copyBuffer(stream.Handle(), buffer->Handle(), 1, &copy);
	NoteWrite(vaddr, size);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	m_gpu_write_generation++;
	if (m_inflight_readbacks.empty() || InFlightReadbackTick(vaddr, vaddr + size) == 0) {
		if (m_gpu_modified_ranges.Subtract(vaddr, size)) {
			NoteGpuModifiedChange(vaddr, size, false);
			UnmarkPagesWithoutGpuBytes(vaddr, size);
		}
	}
	return true;
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return HasTrackedGpuDirtyBytes(vaddr, size);
}

bool BufferCache::HasTrackedGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t PageSize = 4096;
	const auto         page     = vaddr / PageSize;
	if (!m_inflight_readbacks.empty() && size != 0 &&
	    InFlightReadbackTick(vaddr, vaddr + size) != 0) {
		return true;
	}
	if (size == 0 || (vaddr + size - 1) / PageSize != page) {
		return m_gpu_modified_ranges.Intersects(vaddr, size);
	}
	auto& mask = m_dirty_masks[page % m_dirty_masks.size()];
	if (mask.page != page) {
		mask.page = page;
		mask.bits.fill(0);
		const auto base = page * PageSize;
		m_gpu_modified_ranges.ForEachInRange(base, PageSize, [&](uint64_t start, uint64_t end) {
			const auto low  = std::max(start, base) - base;
			const auto high = std::min(end, base + PageSize) - base - 1;
			for (auto word = low / 64; word <= high / 64; word++) {
				const auto from = word == low / 64 ? low % 64 : 0;
				const auto to   = word == high / 64 ? high % 64 : 63;
				mask.bits[word] |= (~uint64_t {0} >> (63 - to)) & (~uint64_t {0} << from);
			}
		});
	}
	const auto first = vaddr % PageSize;
	const auto last  = first + size - 1;
	for (auto word = first / 64; word <= last / 64; word++) {
		const auto low  = word == first / 64 ? first % 64 : 0;
		const auto high = word == last / 64 ? last % 64 : 63;
		const auto bits = (~uint64_t {0} >> (63 - high)) & (~uint64_t {0} << low);
		if ((mask.bits[word] & bits) != 0) {
			return true;
		}
	}
	return false;
}

namespace {

constexpr uint64_t GpuLinePageBits = 12;
constexpr uint64_t GpuLineBits     = 6;
constexpr uint64_t GpuLineMaxPages = uint64_t {1} << 16u;
constexpr uint64_t GpuLineMemoPages = 256;
constexpr uint64_t GpuLineChunk     = 512;

uint64_t GpuLineMask(uint64_t page, uint64_t vaddr, uint64_t size) {
	const auto base  = page << GpuLinePageBits;
	const auto low   = (std::max(vaddr, base) - base) >> GpuLineBits;
	const auto high  = (std::min(vaddr + size, base + (uint64_t {1} << GpuLinePageBits)) - base - 1) >> GpuLineBits;
	return (~uint64_t {0} >> (63 - high)) & (~uint64_t {0} << low);
}

size_t GpuLineHash(uint64_t page) {
	return static_cast<size_t>((page * 0x9e3779b97f4a7c15ull) >> 40u);
}

}

void BufferCache::NoteGpuLines(uint64_t vaddr, uint64_t size) noexcept {
	const auto first = vaddr >> GpuLinePageBits;
	const auto last  = (vaddr + size - 1) >> GpuLinePageBits;
	if (last - first >= GpuLineMaxPages) {
		const auto end = vaddr + size;
		for (auto* list: {&m_gpu_line_current_ranges, &m_gpu_line_ever_ranges}) {
			bool kept = false;
			for (auto& range: *list) {
				if (range.begin.load(std::memory_order_relaxed) <= vaddr && end <= range.end.load(std::memory_order_relaxed)) {
					kept = true;
					break;
				}
			}
			for (auto& range: *list) {
				if (kept) {
					break;
				}
				if (range.end.load(std::memory_order_relaxed) == 0) {
					range.begin.store(vaddr, std::memory_order_relaxed);
					range.end.store(end, std::memory_order_release);
					kept = true;
				}
			}
			if (!kept) {
				m_gpu_line_overflow.store(true, std::memory_order_release);
				g_readback_stats.line_overflow.store(true, std::memory_order_relaxed);
			}
		}
		return;
	}
	const bool memo = last - first >= GpuLineMemoPages;
	if (memo) {
		for (const auto& range: m_gpu_line_memo) {
			if (range.begin.load(std::memory_order_relaxed) <= vaddr && vaddr + size <= range.end.load(std::memory_order_relaxed)) {
				return;
			}
		}
	}
	for (auto page = first; page <= last; page++) {
		const auto mask   = GpuLineMask(page, vaddr, size);
		const auto key    = page + 1;
		bool       stored = false;
		for (size_t probe = 0; probe < GpuLineProbes && !stored; probe++) {
			auto&    slot    = m_gpu_line_slots[(GpuLineHash(page) + probe) % GpuLineSlots];
			uint64_t current = slot.page.load(std::memory_order_acquire);
			if (current == 0 && slot.page.compare_exchange_strong(current, key, std::memory_order_acq_rel)) {
				current = key;
			}
			if (current == key) {
				slot.ever.fetch_or(mask, std::memory_order_release);
				slot.current.fetch_or(mask, std::memory_order_release);
				stored = true;
			}
		}
		if (!stored) {
			m_gpu_line_overflow.store(true, std::memory_order_release);
			g_readback_stats.line_overflow.store(true, std::memory_order_relaxed);
			return;
		}
	}
	if (memo) {
		auto& range = m_gpu_line_memo[m_gpu_line_memo_next++ % GpuLineMemo];
		range.end.store(0, std::memory_order_relaxed);
		range.begin.store(vaddr, std::memory_order_relaxed);
		range.end.store(vaddr + size, std::memory_order_release);
	}
}

void BufferCache::ForgetCurrentGpuLines(uint64_t vaddr, uint64_t size) noexcept {
	if (size == 0) {
		return;
	}
	const auto end = vaddr + size;
	for (auto& range: m_gpu_line_memo) {
		if (vaddr < range.end.load(std::memory_order_relaxed) && range.begin.load(std::memory_order_relaxed) < end) {
			range.end.store(0, std::memory_order_release);
		}
	}
	for (auto& range: m_gpu_line_current_ranges) {
		const auto range_end   = range.end.load(std::memory_order_relaxed);
		const auto range_begin = range.begin.load(std::memory_order_relaxed);
		if (range_end != 0 && vaddr <= range_begin && range_end <= end &&
		    !m_gpu_modified_ranges.Intersects(range_begin, range_end - range_begin)) {
			range.end.store(0, std::memory_order_release);
		}
	}
	const auto first = vaddr >> GpuLinePageBits;
	const auto last  = (end - 1) >> GpuLinePageBits;
	if (last - first >= GpuLineMaxPages) {
		return;
	}
	std::array<GpuLineSlot*, GpuLineChunk> slots {};
	std::array<uint64_t, GpuLineChunk>     kept {};
	for (auto chunk = first; chunk <= last; chunk += GpuLineChunk) {
		const auto chunk_last = std::min(last, chunk + GpuLineChunk - 1);
		bool       any        = false;
		for (auto page = chunk; page <= chunk_last; page++) {
			auto&      found = slots[page - chunk];
			const auto key   = page + 1;
			found            = nullptr;
			for (size_t probe = 0; probe < GpuLineProbes; probe++) {
				auto&      slot    = m_gpu_line_slots[(GpuLineHash(page) + probe) % GpuLineSlots];
				const auto current = slot.page.load(std::memory_order_acquire);
				if (current == key) {
					if (slot.current.load(std::memory_order_relaxed) != 0) {
						found = &slot;
						any   = true;
					}
					break;
				}
				if (current == 0) {
					break;
				}
			}
		}
		if (!any) {
			continue;
		}
		std::fill_n(kept.begin(), chunk_last - chunk + 1, 0);
		const auto base = chunk << GpuLinePageBits;
		m_gpu_modified_ranges.ForEachInRange(base, (chunk_last - chunk + 1) << GpuLinePageBits, [&](uint64_t from, uint64_t to) {
			const auto from_page = std::max(from >> GpuLinePageBits, chunk);
			const auto to_page   = std::min((to - 1) >> GpuLinePageBits, chunk_last);
			for (auto page = from_page; page <= to_page; page++) {
				kept[page - chunk] |= GpuLineMask(page, from, to - from);
			}
		});
		for (auto page = chunk; page <= chunk_last; page++) {
			if (auto* slot = slots[page - chunk]; slot != nullptr) {
				slot->current.store(kept[page - chunk], std::memory_order_release);
			}
		}
	}
}

void BufferCache::UnmarkGpuModified(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
	ForgetCurrentGpuLines(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
	                      Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE) - Common::AlignDown(vaddr, TRACKER_PAGE_SIZE));
}

bool BufferCache::GpuLinesClean(uint64_t vaddr, uint64_t size, bool ever) const noexcept {
	if (size == 0 || m_gpu_line_overflow.load(std::memory_order_acquire)) {
		return false;
	}
	const auto end = vaddr + size;
	for (const auto& range: ever ? m_gpu_line_ever_ranges : m_gpu_line_current_ranges) {
		const auto range_end = range.end.load(std::memory_order_acquire);
		if (range_end != 0 && vaddr < range_end && range.begin.load(std::memory_order_relaxed) < end) {
			return false;
		}
	}
	const auto first = vaddr >> GpuLinePageBits;
	const auto last  = (end - 1) >> GpuLinePageBits;
	if (last - first >= GpuLineMaxPages) {
		return false;
	}
	for (auto page = first; page <= last; page++) {
		const auto key   = page + 1;
		bool       found = false;
		for (size_t probe = 0; probe < GpuLineProbes; probe++) {
			const auto& slot    = m_gpu_line_slots[(GpuLineHash(page) + probe) % GpuLineSlots];
			const auto  current = slot.page.load(std::memory_order_acquire);
			if (current == key) {
				const auto lines = (ever ? slot.ever : slot.current).load(std::memory_order_acquire);
				if ((lines & GpuLineMask(page, vaddr, size)) != 0) {
					return false;
				}
				found = true;
				break;
			}
			if (current == 0) {
				found = true;
				break;
			}
		}
		if (!found) {
			return false;
		}
	}
	return true;
}

bool BufferCache::NeverGpuWritten(uint64_t vaddr, uint64_t size) const noexcept {
	return GpuLinesClean(vaddr, size, true);
}

bool BufferCache::NoGpuBytesNow(uint64_t vaddr, uint64_t size) const noexcept {
	return GpuLinesClean(vaddr, size, false);
}

void BufferCache::NoteGpuModifiedChange(uint64_t vaddr, uint64_t size, bool dirty) {
	constexpr uint64_t PageSize = 4096;
	if (size == 0) {
		return;
	}
	if (dirty) {
		NoteGpuLines(vaddr, size);
	}
	const auto first = vaddr / PageSize;
	const auto last  = (vaddr + size - 1) / PageSize;
	const auto apply = [&](DirtyPageMask& mask) {
		const auto base = mask.page * PageSize;
		const auto low  = std::max(vaddr, base) - base;
		const auto high = std::min(vaddr + size, base + PageSize) - base - 1;
		for (auto word = low / 64; word <= high / 64; word++) {
			const auto from = word == low / 64 ? low % 64 : 0;
			const auto to   = word == high / 64 ? high % 64 : 63;
			const auto bits = (~uint64_t {0} >> (63 - to)) & (~uint64_t {0} << from);
			mask.bits[word] = dirty ? mask.bits[word] | bits : mask.bits[word] & ~bits;
		}
	};
	if (last - first >= m_dirty_masks.size()) {
		for (auto& mask: m_dirty_masks) {
			if (mask.page >= first && mask.page <= last) {
				apply(mask);
			}
		}
		return;
	}
	for (auto page = first; page <= last; page++) {
		auto& mask = m_dirty_masks[page % m_dirty_masks.size()];
		if (mask.page == page) {
			apply(mask);
		}
	}
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

bool BufferCache::DropFreedRange(uint64_t vaddr, uint64_t size) {
	if (size == 0 || UINT64_MAX - vaddr < size) return false;
	ForgetCommandBuffers(vaddr, size);
	if (!IsRegionRegistered(vaddr, size)) return false;
	InvalidateMemory(vaddr, size);
	return true;
}

void BufferCache::SubmitGuestMirror(SubmitInfo& submit) {
	const auto wait = m_guest_mirror->Submit();
	if (wait.semaphore) submit.AddWait(wait.semaphore, wait.value);
}

void BufferCache::TickGuestMirror() {
	auto&      mirror = *m_guest_mirror;
	const auto now    = static_cast<uint64_t>(SteadyNs());
	if (m_guest_mirror_line == 0) {
		m_guest_mirror_line   = now;
		m_guest_mirror_frames = Frame();
		return;
	}
	if (now - m_guest_mirror_line < 10'000'000'000) return;
	uint64_t used = 0, budget = 0;
	if (m_graphics.CanReportMemoryUsage()) {
		std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
		const auto count = m_graphics.GetHeapUsage(heaps);
		for (uint32_t heap = 0; heap < count; heap++) {
			if (!heaps[heap].device_local) continue;
			used += heaps[heap].usage;
			budget += heaps[heap].budget;
		}
	}
	const auto seconds    = static_cast<double>(now - m_guest_mirror_line) / 1e9;
	const auto frames     = Frame() - std::exchange(m_guest_mirror_frames, Frame());
	const auto evictions  = std::exchange(m_guest_mirror_evictions, 0);
	const auto stats      = mirror.TakeStats();
	const auto per        = [frames](uint64_t value) { return frames != 0 ? static_cast<double>(value) / static_cast<double>(frames) : 0.0; };
	const auto block      = mirror.BlockSize();
	m_guest_mirror_line   = now;
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-mirror] summary of the last {:.1f} s: {} frames; {} blocks of {} KiB bound ({:.1f} MiB, {} chunks of {} MiB) "
	    "in a mirror of {} GiB; {} blocks bound in {} vkQueueBindSparse calls ({:.3f} calls and {:.2f} blocks per frame), {:.2f} "
	    "ms in the calls, {:.1f} ms until they were done (longest {:.2f} ms); {} submissions waited for a bind; {} blocks unbound "
	    "in {} evictions by age under memory pressure; {} blocks noted outside the mirror; {} reads of unbound blocks; "
	    "device-local memory {} MiB used of a {} MiB budget; {} bind requests, {:.3f} ms from the request to the bound block on "
	    "average (longest {:.2f} ms)\n",
	    seconds, frames, stats.bound_blocks, block >> 10u, static_cast<double>(stats.bound_blocks * block) / static_cast<double>(MiB),
	    stats.chunks, std::max(GuestMirror::ChunkBytes, block) / MiB, mirror.Size() / GuestMirror::GiB, stats.blocks, stats.calls,
	    per(stats.calls), per(stats.blocks), static_cast<double>(stats.call_ns) / 1e6, static_cast<double>(stats.done_ns) / 1e6,
	    static_cast<double>(stats.longest_ns) / 1e6, stats.waits, stats.unbound_blocks, evictions, mirror.OutsideBlocks(),
	    stats.unbound_reads, used / MiB, budget / MiB, stats.requests,
	    stats.requests != 0 ? static_cast<double>(stats.request_ns) / 1e6 / static_cast<double>(stats.requests) : 0.0,
	    static_cast<double>(stats.request_longest_ns) / 1e6));
	if (m_host_view->Enabled()) {
		const auto host = m_host_view->Totals();
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-host-view] {} imports of guest memory ({:.1f} MiB), {} refused; {} unmaps waited for the GPU; {:.1f} MiB "
		    "copied in command order into blocks bound behind a dispatch that stores by device address; {:.1f} MiB of mapped "
		    "memory without an import bound for the dispatches whose reach is not known; {} shader accesses went to guest "
		    "memory through an import ({:.2f} per frame)\n",
		    host.imports, static_cast<double>(host.bytes) / static_cast<double>(MiB), host.refused, host.waits,
		    static_cast<double>(host.ordered) / static_cast<double>(MiB),
		    static_cast<double>(m_unimported_bytes) / static_cast<double>(MiB), m_host_accesses, per(m_host_accesses)));
		m_host_accesses = 0;
		if (!m_host_access_read) {
			if (!m_host_access_download) {
				m_host_access_download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
				                                                  vk::BufferUsageFlagBits::eTransferDst, 16);
			}
			m_host_access_download->CopyFrom(m_scheduler.Current(), *GetFaultBuffer(), HOST_ACCESS_OFFSET, 0, sizeof(uint32_t));
			m_host_access_read = true;
			m_scheduler.DeferHostOperation([this] {
				m_host_access_download->Invalidate(0, sizeof(uint32_t));
				uint32_t value = 0;
				std::memcpy(&value, m_host_access_download->Mapped().data(), sizeof(value));
				m_host_accesses += static_cast<uint32_t>(value - m_host_access_seen);
				m_host_access_seen = value;
				m_host_access_read = false;
			});
		}
	}
}

void BufferCache::RunGarbageCollector() {
	m_gc_tick++;
	m_guest_mirror->Age(static_cast<uint32_t>(Frame()));
	TickGuestMirror();
	EvictOldBlocks();
}

void BufferCache::EvictOldBlocks() {
	static constexpr uint64_t PressureMargin = 256 * MiB;
	static constexpr uint64_t MaxEvictBytes  = 32 * MiB;
	static constexpr uint64_t YoungFrames    = 80;
	const auto                now            = SteadyNs();
	const auto                frame          = Frame();
	if (now - m_evict_check < 1'000'000'000 || !m_graphics.CanReportMemoryUsage() || frame <= YoungFrames) {
		return;
	}
	m_evict_check = now;
	uint64_t                                                   used = 0, budget = 0;
	std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
	const auto                                                 count = m_graphics.GetHeapUsage(heaps);
	for (uint32_t heap = 0; heap < count; heap++) {
		if (!heaps[heap].device_local) continue;
		used += heaps[heap].usage;
		budget += heaps[heap].budget;
	}
	if (budget == 0 || used + PressureMargin < budget) {
		return;
	}
	const auto block  = m_guest_mirror->BlockSize();
	const auto wanted = std::min(used + PressureMargin - budget, MaxEvictBytes);
	m_guest_mirror->PickOldest((wanted + block - 1) / block, static_cast<uint32_t>(frame - YoungFrames), m_victims);
	if (m_victims.empty()) {
		return;
	}
	for (const auto& [begin, end]: m_victims) {
		if (m_memory_tracker.IsRegionGpuModified(begin, end - begin)) {
			(void)DownloadMemory<true>(begin, end - begin);
		}
	}
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	WaitReadbackQueueCopies();
	for (const auto& [begin, end]: m_victims) {
		UnmarkGpuModified(begin, end - begin);
		m_memory_tracker.MarkRegionAsCpuModified(begin, end - begin);
		if (m_record_released) {
			m_released_ranges.Add(begin, end - begin);
		}
		const auto slot = (m_stale_ring_next.fetch_add(1, std::memory_order_relaxed) % StaleRingSize) * 2;
		m_stale_ring[slot].store(begin, std::memory_order_relaxed);
		m_stale_ring[slot + 1].store(end - begin, std::memory_order_relaxed);
		SetResident(begin, end, false);
		NoteBlockRegions(begin, end, false);
	}
	const auto epoch = m_registration_epoch.load(std::memory_order_relaxed) + 1;
	for (const auto& [begin, end]: m_victims) {
		for (auto region = begin / TRACKER_REGION_SIZE; region * TRACKER_REGION_SIZE < end; ++region) {
			m_region_registrations[region].store(epoch, std::memory_order_release);
		}
	}
	m_registration_epoch.store(epoch, std::memory_order_release);
	m_guest_mirror->Release(m_victims);
	m_guest_mirror_evictions++;
	m_gpu_write_generation++;
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::ProcessUseBuffer() {
	m_fault_manager.ProcessUseBuffer([this](std::span<const uint64_t> pages) {
		for (const auto page: pages) {
			m_guest_mirror->Use(page, MIRROR_BLOCK_SIZE);
		}
	});
}

void BufferCache::NoteGpuWrites(uint64_t vaddr, uint64_t size, HeapWriteSource writer) {
	m_gpu_write_generation++;
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid GPU write range\n");
	}
	m_scheduler.Context().CollectHotPageWrites(vaddr, size);
	const auto written = [&](uint64_t start, uint64_t finish) {
		m_guest_mirror->Use(start, finish - start);
		NoteWrite(start, finish - start, writer);
		m_memory_tracker.MarkRegionAsGpuModifiedUnlessCpuDirty(start, finish - start, [&](uint64_t page, uint64_t bytes) {
			const auto first = std::max(page, start);
			const auto last  = std::min(page + bytes, finish);
			if (m_gpu_modified_ranges.Add(first, last - first)) {
				NoteGpuModifiedChange(first, last - first, true);
			}
		});
	};
	m_guest_mirror->ForEachRequested(vaddr, size, [&](uint64_t start, uint64_t finish) {
		uint64_t run = UINT64_MAX;
		for (auto block = start; block < finish; block = std::min(finish, Common::AlignDown(block, MIRROR_BLOCK_SIZE) + MIRROR_BLOCK_SIZE)) {
			if (IsResident(block)) {
				if (run == UINT64_MAX) run = block;
			} else if (run != UINT64_MAX) {
				written(run, block);
				run = UINT64_MAX;
			}
		}
		if (run != UINT64_MAX) written(run, finish);
	});
}

void BufferCache::FlushUploadBatch() {
	m_upload_batching = false;
	if (m_upload_batch_counts.empty()) {
		return;
	}
	EXIT_IF(m_scheduler.CurrentTick() != m_upload_batch_tick);
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	if (m_upload_batch_counts.size() > 1 && ScatterUploadBatch(native)) {
		m_upload_batch_counts.clear();
		m_upload_batch_copies.clear();
		return;
	}
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion, 1,
	                       &before, 0, nullptr, 0, nullptr);
	native.copyBuffer(m_staging_buffer.Handle(), m_storage->Handle(), static_cast<uint32_t>(m_upload_batch_copies.size()),
	                  m_upload_batch_copies.data());
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eByRegion,
	                       1, &after, 0, nullptr, 0, nullptr);
	m_upload_batch_counts.clear();
	m_upload_batch_copies.clear();
}

bool BufferCache::ScatterUploadBatch(const CommandRecorder& native) {
	struct Job {
		uint64_t source;
		uint64_t destination;
		uint32_t size;
		uint32_t pad0;
		uint64_t pad1;
	};
	static_assert(sizeof(Job) == 32);
	constexpr uint64_t ScatterJobBytes = 4 * 1024;
	constexpr uint32_t MaxJobs         = 65535;
	const auto         staging         = m_staging_buffer.BufferDeviceAddress();
	const auto         destination     = m_storage->BufferDeviceAddress();
	uint64_t           job_count       = 0;
	if (staging == 0) {
		return false;
	}
	for (const auto& copy: m_upload_batch_copies) {
		if (((copy.srcOffset | copy.dstOffset | copy.size) & 3u) != 0) {
			return false;
		}
		job_count += (copy.size + ScatterJobBytes - 1) / ScatterJobBytes;
	}
	if (job_count == 0 || job_count > MaxJobs) {
		return false;
	}
	const auto bytes            = job_count * sizeof(Job);
	const auto [mapped, offset] = m_staging_buffer.Map(bytes, 16, false);
	if (mapped == nullptr) {
		return false;
	}
	uint8_t* job = mapped;
	for (const auto& copy: m_upload_batch_copies) {
		for (uint64_t done = 0; done < copy.size; done += ScatterJobBytes) {
			const Job value {staging + copy.srcOffset + done, destination + copy.dstOffset + done,
			                 static_cast<uint32_t>(std::min(ScatterJobBytes, copy.size - done)), 0, 0};
			std::memcpy(job, &value, sizeof(value));
			job += sizeof(value);
		}
	}
	m_staging_buffer.Commit();
	if (!m_scatter_pipeline) {
		const auto                   device = m_graphics.device;
		const vk::PushConstantRange  push {vk::ShaderStageFlagBits::eCompute, 0, 16};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(device.createPipelineLayout(&layout_info, nullptr, &m_scatter_layout),
		                     "create buffer scatter pipeline layout");
		const auto                        module = CompileSPV(BUFFER_SCATTER_SPV, device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = m_scatter_layout;
		const auto result    = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_scatter_pipeline);
		device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create buffer scatter pipeline");
		SetVulkanObjectNameF(device, m_scatter_pipeline, "Kyty.BufferScatter");
	}
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlagBits::eByRegion, 1,
	                       &before, 0, nullptr, 0, nullptr);
	struct Push {
		uint64_t jobs;
		uint32_t job_count;
		uint32_t pad;
	} push {staging + offset, static_cast<uint32_t>(job_count), 0};
	native.bindPipeline(vk::PipelineBindPoint::eCompute, m_scatter_pipeline);
	native.pushConstants(m_scatter_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	native.dispatch(push.job_count, 1, 1);
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                       vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlagBits::eByRegion,
	                       1, &after, 0, nullptr, 0, nullptr);
	return true;
}

struct BufferCache::UploadBatch {
	BufferCache& cache;
	explicit UploadBatch(BufferCache& owner): cache(owner) {
		cache.m_upload_batching   = true;
		cache.m_upload_batch_tick = cache.m_scheduler.CurrentTick();
	}
	~UploadBatch() { cache.FlushUploadBatch(); }
	UploadBatch(const UploadBatch&)            = delete;
	UploadBatch& operator=(const UploadBatch&) = delete;
};

void BufferCache::SynchronizeBuffersInSpans(std::span<const std::pair<uint64_t, uint64_t>> spans) {
	const UploadBatch batch {*this};
	m_sync_spans.assign(spans.begin(), spans.end());
	std::ranges::sort(m_sync_spans);
	size_t merged = 0;
	for (const auto& span: m_sync_spans) {
		if (merged != 0 && span.first <= m_sync_spans[merged - 1].second) {
			m_sync_spans[merged - 1].second = std::max(m_sync_spans[merged - 1].second, span.second);
		} else {
			m_sync_spans[merged++] = span;
		}
	}
	m_sync_spans.resize(merged);
	m_span_ranges.clear();
	RegionBits dirty;
	uint64_t   dirty_region = UINT64_MAX;
	bool       region_dirty = false;
	for (const auto& [first, last]: m_sync_spans) {
		for (auto at = first; at < last;) {
			const auto index = at / TRACKER_REGION_SIZE;
			const auto base  = index * TRACKER_REGION_SIZE;
			const auto end   = std::min(last, base + TRACKER_REGION_SIZE);
			if (index != dirty_region) {
				dirty_region = index;
				region_dirty = m_memory_tracker.CopyRegionCpuDirty(index, dirty);
			}
			const auto end_page = (end - base + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
			for (auto page = (at - base) / TRACKER_PAGE_SIZE; region_dirty && page < end_page;) {
				const auto [run, after] = dirty.FirstRangeFrom(page);
				if (run >= end_page) {
					break;
				}
				const auto begin  = std::max(base + run * TRACKER_PAGE_SIZE, at);
				const auto finish = std::min(base + std::min<uint64_t>(after, end_page) * TRACKER_PAGE_SIZE, end);
				CollectSpanRanges(begin, finish - begin, m_span_ranges);
				page = after;
			}
			at = end;
		}
	}
	if (!m_span_ranges.empty()) {
		SynchronizeRanges(m_span_ranges);
	}
}

void BufferCache::GrowRegionTables(uint64_t region) {
	if (region < m_region_buffer_count.size()) {
		return;
	}
	const auto words = region / 64 + 1;
	m_region_buffer_count.resize(words * 64, 0);
	m_synced_region_epochs.resize(words * 64, 0);
	m_buffer_region_bits.resize(words, 0);
	m_forced_region_bits.resize(words, 0);
	m_region_forces.resize(words * 64, 0);
}

void BufferCache::ForceSynchronization(uint64_t vaddr, uint64_t size) {
	if (size == 0 || m_buffer_region_bits.empty()) {
		return;
	}
	const auto first = vaddr / TRACKER_REGION_SIZE;
	const auto last  = std::min<uint64_t>((vaddr + size - 1) / TRACKER_REGION_SIZE, m_region_buffer_count.size() - 1);
	for (auto index = first; index <= last; index++) {
		m_forced_region_bits[index / 64] |= uint64_t {1} << (index % 64);
		m_region_forces[index]++;
	}
}

void BufferCache::AddToSet(RangeSet& set, SetRegions& state, uint64_t address, uint64_t size) {
	const bool covered = state.version == set.Version();
	if (!set.Add(address, size)) {
		return;
	}
	const auto end = address + size;
	for (auto index = address / TRACKER_REGION_SIZE; index <= (end - 1) / TRACKER_REGION_SIZE; index++) {
		if (index < state.regions.size()) {
			state.regions[index] = {};
		}
	}
	if (!covered) {
		return;
	}
	for (auto at = address; at < end;) {
		const auto index    = at / TRACKER_REGION_SIZE;
		const auto base     = index * TRACKER_REGION_SIZE;
		const auto stop     = std::min(end, base + TRACKER_REGION_SIZE);
		const auto found    = std::ranges::lower_bound(state.covered, index);
		const auto position = static_cast<size_t>(found - state.covered.begin());
		if (found == state.covered.end()) {
			state.covered.push_back(index);
			state.covered_pages.emplace_back();
		} else if (*found != index) {
			return;
		}
		state.covered_pages[position].SetRange((at - base) / TRACKER_PAGE_SIZE,
		                                       (stop - base + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE);
		at = stop;
	}
	state.version = set.Version();
}

void BufferCache::SynchronizeBuffersInSet(const RangeSet& set, SetRegions& state) {
	const UploadBatch batch {*this};
	const auto sync = ++m_device_syncs;
	m_device_sync_pass = sync;
	struct EndPass {
		BufferCache& cache;
		~EndPass() {
			cache.m_device_sync_pass = 0;
			cache.m_kept_pages       = cache.m_memory_tracker.TakeKeptPages() || cache.m_kept_pages;
		}
	} end_pass {*this};
	auto& regions = state.regions;
	if (regions.size() < m_region_buffer_count.size()) {
		regions.resize(m_region_buffer_count.size());
	}
	if (state.version != set.Version()) {
		state.version = set.Version();
		state.covered.clear();
		state.covered_pages.clear();
		set.ForEach([&](uint64_t begin, uint64_t end) {
			for (auto at = begin; at < end;) {
				const auto index = at / TRACKER_REGION_SIZE;
				const auto base  = index * TRACKER_REGION_SIZE;
				const auto stop  = std::min(end, base + TRACKER_REGION_SIZE);
				if (state.covered.empty() || state.covered.back() != index) {
					state.covered.push_back(index);
					state.covered_pages.emplace_back();
				}
				state.covered_pages.back().SetRange((at - base) / TRACKER_PAGE_SIZE,
				                                    (stop - base + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE);
				at = stop;
			}
		});
	}
	m_set_visits.clear();
	size_t covered = 0;
	for (size_t word = 0; word < m_buffer_region_bits.size(); word++) {
		for (auto bits = m_buffer_region_bits[word]; bits != 0; bits &= bits - 1) {
			const auto index = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
			while (covered < state.covered.size() && state.covered[covered] < index) {
				covered++;
			}
			if (covered != state.covered.size() && state.covered[covered] == index) {
				m_set_visits.emplace_back(index, covered);
			}
		}
	}
	const auto visit = [&](const std::pair<uint64_t, size_t>& region, std::vector<std::pair<uint64_t, uint64_t>>& chunk) {
		const auto index = region.first;
		const auto epoch  = m_memory_tracker.RegionCpuDirtyEpoch(index);
		const auto forces = m_region_forces[index] + 1;
		auto&      synced = regions[index];
		const auto kept   = m_memory_tracker.RegionKeptSync(index);
		if (synced.second == forces && (epoch == 0 || synced.first == epoch) &&
		    (kept == 0 || kept < state.pass)) {
			return;
		}
		RegionBits dirty;
		if (m_memory_tracker.CopyRegionCpuDirty(index, dirty)) {
			dirty &= state.covered_pages[region.second];
			const auto base = index * TRACKER_REGION_SIZE;
			for (size_t page = 0; page < TRACKER_REGION_PAGES;) {
				const auto [first, after] = dirty.FirstRangeFrom(page);
				if (first >= TRACKER_REGION_PAGES) {
					break;
				}
				set.ForEachInRange(base + first * TRACKER_PAGE_SIZE, (after - first) * TRACKER_PAGE_SIZE,
				                   [&](uint64_t start, uint64_t end) {
					                   CollectSpanRanges(start, end - start, chunk);
				                   });
				page = after;
			}
		}
		synced = {epoch, forces};
	};
	constexpr uint32_t Grain  = 8;
	const auto         count  = static_cast<uint32_t>(m_set_visits.size());
	const auto         chunks = (count + Grain - 1) / Grain;
	if (m_set_chunks.size() < chunks) {
		m_set_chunks.resize(chunks);
	}
	GetExecHelpers().ParallelFor(count, Grain, [&](uint32_t begin, uint32_t end) {
		auto& chunk = m_set_chunks[begin / Grain];
		chunk.clear();
		for (auto visited = begin; visited < end; visited++) {
			visit(m_set_visits[visited], chunk);
		}
	});
	m_span_ranges.clear();
	for (uint32_t chunk = 0; chunk < chunks; chunk++) {
		const auto& collected = m_set_chunks[chunk];
		m_span_ranges.insert(m_span_ranges.end(), collected.begin(), collected.end());
	}
	if (!m_span_ranges.empty()) {
		SynchronizeRanges(m_span_ranges);
	}
	state.pass = sync;
}

void BufferCache::SynchronizeBuffersInRanges(const RangeSet& ranges, bool all) {
	const UploadBatch batch {*this};
	const auto sync    = ++m_device_syncs;
	m_device_sync_pass = sync;
	(void)m_memory_tracker.TakeKeptPages();
	struct EndPass {
		BufferCache& cache;
		~EndPass() {
			cache.m_device_sync_pass = 0;
			cache.m_kept_pages       = cache.m_memory_tracker.TakeKeptPages();
		}
	} end_pass {*this};
	m_sync_regions.clear();
	for (size_t word = 0; word < m_buffer_region_bits.size(); word++) {
		const auto forced = m_forced_region_bits[word];
		m_forced_region_bits[word] = 0;
		for (auto bits = m_buffer_region_bits[word]; bits != 0; bits &= bits - 1) {
			const auto index = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
			const auto epoch = m_memory_tracker.RegionCpuDirtyEpoch(index);
			const auto kept = m_memory_tracker.RegionKeptSync(index);
			if (!all && (forced & (uint64_t {1} << (index % 64))) == 0 &&
			    (epoch == 0 || m_synced_region_epochs[index] == epoch) &&
			    (kept == 0 || kept < m_device_full_pass)) {
				continue;
			}
			m_sync_regions.emplace_back(index, epoch);
		}
	}
	for (const auto& [index, epoch]: m_sync_regions) {
		ranges.ForEachInRange(index * TRACKER_REGION_SIZE, TRACKER_REGION_SIZE,
		                      [this](uint64_t start, uint64_t end) {
			                      SynchronizeBuffersInRange(start, end - start);
		                      });
		m_synced_region_epochs[index] = epoch;
	}
	m_device_full_pass = sync;
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t MaxGap = 32 * TRACKER_PAGE_SIZE;
	m_memory_tracker.ForEachCpuDirtySpan(vaddr, size, MaxGap, [&](uint64_t span, uint64_t bytes) {
		m_span_ranges.clear();
		CollectSpanRanges(span, bytes, m_span_ranges);
		if (m_span_ranges.size() == 1) {
			(void)SynchronizeRange(m_span_ranges[0].first, m_span_ranges[0].second, false, false);
		} else if (!m_span_ranges.empty()) {
			SynchronizeRanges(m_span_ranges);
		}
	});
}

void BufferCache::CollectSpanRanges(uint64_t span, uint64_t bytes, std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	m_guest_mirror->ForEachRequested(span, bytes, [&](uint64_t start, uint64_t finish) { ranges.emplace_back(start, finish - start); });
}

}

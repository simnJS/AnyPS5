#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTMIRROR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTMIRROR_H_

#include "graphics/host_gpu/memoryTelemetry.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/mirrorWindowTable.h"
#include "common/threadSafety.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaAllocation)

namespace Libs::Graphics {

struct GraphicContext;

struct MirrorWait {
	vk::Semaphore semaphore = nullptr;
	uint64_t      value     = 0;
};

class GuestMirror {
public:
	static constexpr uint64_t GiB             = uint64_t {1} << 30u;
	static constexpr uint64_t LargestSize     = 864 * GiB;
	static constexpr uint64_t Headroom        = 96 * GiB;
	static constexpr uint32_t MinBlockBits    = 16;
	static constexpr uint64_t ChunkBytes      = uint64_t {32} << 20u;
	static constexpr uint32_t PageBits        = 14;
	static constexpr size_t   MaxBindsPerCall = 256;

	static constexpr std::array<uint64_t, 10> DefaultSizes {864 * GiB, 832 * GiB, 800 * GiB, 768 * GiB, 704 * GiB,
	                                                        640 * GiB, 512 * GiB, 256 * GiB, 128 * GiB, 64 * GiB};

	struct ProbeResult {
		uint64_t    size = 0;
		double      ms   = 0;
		std::string trials;
		std::string refusal;
	};
	[[nodiscard]] static ProbeResult Probe(const GraphicContext& graphics, std::span<const uint64_t> sizes);
	[[nodiscard]] static std::string Unsupported(const GraphicContext& graphics);

	struct Stats {
		uint64_t bound_blocks       = 0;
		uint64_t chunks             = 0;
		uint64_t calls              = 0;
		uint64_t blocks             = 0;
		uint64_t unbound_blocks     = 0;
		uint64_t call_ns            = 0;
		uint64_t done_ns            = 0;
		uint64_t longest_ns         = 0;
		uint64_t requests           = 0;
		uint64_t request_ns         = 0;
		uint64_t request_longest_ns = 0;
		uint64_t waits              = 0;
		uint64_t unbound_reads      = 0;
	};

	using Wait = MirrorWait;
	using Runs = std::vector<std::pair<uint64_t, uint64_t>>;

	GuestMirror(GraphicContext& graphics, std::span<const uint64_t> sizes = DefaultSizes, uint64_t base = 0, uint64_t guest_span = 0);
	~GuestMirror();
	KYTY_CLASS_NO_COPY(GuestMirror);

	[[nodiscard]] bool               Available() const noexcept { return m_size != 0; }
	[[nodiscard]] uint64_t           Size() const noexcept { return m_size; }
	static constexpr uint32_t WindowBits = MirrorWindowTable::WindowBits;
	static constexpr uint64_t WindowSize = MirrorWindowTable::WindowSize;
	[[nodiscard]] uint64_t GuestSpan() const noexcept { return m_guest_span; }
	[[nodiscard]] uint64_t Offset(uint64_t address) const noexcept;
	[[nodiscard]] uint64_t Address(uint64_t offset) const noexcept;
	[[nodiscard]] uint64_t WindowRemaining(uint64_t address) const noexcept {
		return WindowSize - ((address - m_base) & (WindowSize - 1));
	}
	[[nodiscard]] bool InOneWindow(uint64_t address, uint64_t bytes) const noexcept {
		return Holds(address, bytes) && bytes <= WindowRemaining(address) && bytes <= m_size && Offset(address) <= m_size - bytes;
	}
	[[nodiscard]] bool MapWindows(uint64_t address, uint64_t size);
	void FreeWindow(uint64_t address);
	[[nodiscard]] uint64_t ContiguousSize(uint64_t address, uint64_t bytes) const noexcept;
	[[nodiscard]] uint64_t           Base() const noexcept { return m_base; }
	[[nodiscard]] uint64_t           BlockSize() const noexcept { return uint64_t {1} << m_block_bits; }
	[[nodiscard]] uint64_t           DeviceAddress() const noexcept { return m_device_address; }
	[[nodiscard]] const std::string& Description() const noexcept { return m_description; }
	[[nodiscard]] vk::Buffer         Handle() const noexcept { return m_buffer; }
	[[nodiscard]] bool               Holds(uint64_t vaddr, uint64_t size) const noexcept {
		return size != 0 && vaddr >= m_base && vaddr - m_base < m_guest_span && size <= m_guest_span - (vaddr - m_base);
	}
	[[nodiscard]] bool     IsBound(uint64_t vaddr) const noexcept;
	[[nodiscard]] bool     Requested(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] bool     Ready(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] bool     AnyRequested(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] bool     Owned(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] bool     AnyOwned(uint64_t vaddr, uint64_t size) const noexcept;
	template <typename Func>
	void ForEachRequested(uint64_t vaddr, uint64_t size, Func&& func) const {
		ForEachSet(m_bound.get(), m_block_bits, vaddr, size, std::forward<Func>(func));
	}
	template <typename Func>
	void ForEachOwned(uint64_t vaddr, uint64_t size, Func&& func) const {
		ForEachSet(m_owned.get(), PageBits, vaddr, size, std::forward<Func>(func));
	}
	[[nodiscard]] uint64_t OutsideBlocks() const noexcept { return m_outside_blocks; }
	[[nodiscard]] bool     SharedQueue() const noexcept { return m_shared_queue; }

	void               NoteRange(uint64_t vaddr, uint64_t size, Runs* fresh = nullptr);
	void               Own(uint64_t vaddr, uint64_t size, Runs* fresh = nullptr);
	void               Use(uint64_t vaddr, uint64_t size) noexcept;
	void               NoteDevice(uint64_t vaddr, uint64_t size) noexcept;
	void               Age(uint32_t frame);
	void               PickOldest(uint64_t max_blocks, uint32_t older_than, Runs& victims) const;
	void               Release(const Runs& runs);
	[[nodiscard]] Wait Submit();
	void               Flush();
	void               WaitIdle();
	void               WaitBound(uint64_t value);
	void               NoteUnboundRead() noexcept { m_unbound_reads.fetch_add(1, std::memory_order_relaxed); }
	[[nodiscard]] Stats TakeStats();
	[[nodiscard]] Stats Totals() const;

private:
	struct Run {
		uint64_t first = 0;
		uint64_t count = 0;
	};
	enum class Kind : uint8_t { Bind, Unbind };
	struct Command {
		Kind             kind = Kind::Bind;
		std::vector<Run> runs;
		uint64_t         value     = 0;
		Wait             after     = {};
		uint64_t         requested = 0;
	};
	struct Chunk {
		MemoryTelemetry::Allocation charge;
		vk::DeviceMemory memory     = nullptr;
		VmaAllocator     allocator  = nullptr;
		VmaAllocation    allocation = nullptr;
		uint32_t         used   = 0;
		uint32_t         next   = 0;
	};
	struct Slot {
		uint32_t chunk = 0;
		uint32_t index = 0;
	};
	struct Grant {
		uint32_t chunk = 0;
		uint32_t index = 0;
		uint32_t count = 0;
	};

	[[nodiscard]] bool CreateBuffer();
	void               DestroyBuffer();
	void               BinderThread(std::stop_token stop);
	void               Bind(const Command& command);
	void               Unbind(const Command& command);
	void               Execute(std::vector<vk::SparseMemoryBind>& binds, const char* what, uint64_t signal);
	[[nodiscard]] Grant Allocate(uint64_t wanted);
	void               Free(Slot slot);
	void               Post(Command&& command);
	[[nodiscard]] bool AllSet(const std::atomic<uint64_t>* bits, uint32_t shift, uint64_t vaddr, uint64_t size) const noexcept;
	void               Disown(uint64_t begin, uint64_t end) noexcept;
	template <typename Func>
	void ForEachSet(const std::atomic<uint64_t>* map, uint32_t shift, uint64_t vaddr, uint64_t size, Func&& func) const {
		if (m_size == 0 || size == 0 || vaddr >= m_base + m_guest_span || vaddr + size <= m_base) return;
		const uint64_t begin = std::max(vaddr, m_base);
		const uint64_t end   = std::min(vaddr + size, m_base + m_guest_span);
		const uint64_t first = (begin - m_base) >> shift;
		const uint64_t last  = (end - m_base - 1) >> shift;
		uint64_t       run   = UINT64_MAX;
		for (uint64_t word = first / 64; word <= last / 64; word++) {
			const auto low  = word == first / 64 ? first % 64 : 0;
			const auto high = word == last / 64 ? last % 64 : 63;
			const auto mask = (high == 63 ? ~uint64_t {0} : (uint64_t {2} << high) - 1) & ~((uint64_t {1} << low) - 1);
			const auto bits = map[word].load(std::memory_order_acquire) & mask;
			if (bits == mask) {
				if (run == UINT64_MAX) run = word * 64 + low;
				continue;
			}
			for (uint64_t bit = low; bit <= high; bit++) {
				const bool set = (bits >> bit & 1u) != 0;
				if (set && run == UINT64_MAX) {
					run = word * 64 + bit;
				} else if (!set && run != UINT64_MAX) {
					func(std::max(begin, m_base + (run << shift)), m_base + ((word * 64 + bit) << shift));
					run = UINT64_MAX;
				}
			}
		}
		if (run != UINT64_MAX) func(std::max(begin, m_base + (run << shift)), end);
	}
	[[nodiscard]] uint32_t AgeOf(uint64_t block) const noexcept;
	static constexpr uint64_t AgeChunkBlocks = 16384;

	GraphicContext& m_graphics;
	uint64_t        m_guest_span = 0;
	std::optional<MirrorWindowTable> m_windows;
	uint64_t        m_size           = 0;
	uint64_t        m_base           = 0;
	uint32_t        m_block_bits     = MinBlockBits;
	uint32_t        m_memory_types   = 0;
	uint64_t        m_device_address = 0;
	std::string     m_description;
	bool            m_shared_queue = false;

	std::unique_ptr<std::atomic<uint64_t>[]> m_bound;
	std::unique_ptr<std::atomic<uint64_t>[]> m_ready;
	std::unique_ptr<std::atomic<uint64_t>[]> m_owned;
	uint64_t                                 m_words          = 0;
	std::vector<uint64_t>                    m_used;
	std::vector<uint64_t>                    m_used_words;
	std::vector<uint64_t>                    m_device;
	std::vector<std::unique_ptr<uint32_t[]>> m_ages;
	std::vector<Run>                         m_pending;
	uint64_t                                 m_pending_blocks = 0;
	uint64_t                                 m_first_pending  = 0;
	uint64_t                                 m_outside_blocks = 0;
	uint64_t                                 m_bind_posted    = 0;
	std::atomic<uint64_t>                    m_bind_signaled {0};

	vk::Buffer                         m_buffer    = nullptr;
	vk::Fence                          m_fence     = nullptr;
	vk::Semaphore                      m_semaphore = nullptr;
	std::vector<Chunk>                 m_chunks;
	uint32_t                           m_current_chunk = UINT32_MAX;
	std::vector<Slot>                  m_free_slots;
	std::vector<uint32_t>              m_free_chunks;
	std::unordered_map<uint64_t, Slot> m_block_slots;
	std::vector<vk::SparseMemoryBind>  m_binds;

	Common::AnnotatedMutex      m_mutex;
	std::condition_variable_any m_changed;
	std::deque<Command>         m_commands KYTY_GUARDED_BY(m_mutex);
	bool                        m_busy KYTY_GUARDED_BY(m_mutex) = false;

	std::atomic<uint64_t> m_bound_blocks {0};
	std::atomic<uint64_t> m_chunk_count {0};
	std::atomic<uint64_t> m_calls {0};
	std::atomic<uint64_t> m_blocks {0};
	std::atomic<uint64_t> m_unbound_blocks {0};
	std::atomic<uint64_t> m_call_ns {0};
	std::atomic<uint64_t> m_done_ns {0};
	std::atomic<uint64_t> m_longest_ns {0};
	std::atomic<uint64_t> m_requests {0};
	std::atomic<uint64_t> m_request_ns {0};
	std::atomic<uint64_t> m_request_longest_ns {0};
	std::atomic<uint64_t> m_waits {0};
	std::atomic<uint64_t> m_unbound_reads {0};
	Stats                 m_taken;

	std::jthread m_binder;
};

}

#endif

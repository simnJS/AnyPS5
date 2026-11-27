#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTHOSTVIEW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTHOSTVIEW_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/memoryTelemetry.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class CommandScheduler;
class StreamBuffer;

class GuestHostView {
public:
	static constexpr uint32_t PageBits        = 14;
	static constexpr uint64_t PageSize        = uint64_t {1} << PageBits;
	static constexpr uint32_t ChunkBits       = 30;
	static constexpr uint64_t EntriesPerChunk = uint64_t {1} << (ChunkBits - PageBits);
	static constexpr uint64_t PieceBytes      = uint64_t {64} << 20u;
	static constexpr uint64_t JoinBytes       = uint64_t {1} << 20u;

	struct Stats {
		uint64_t imports  = 0;
		uint64_t bytes    = 0;
		uint64_t refused  = 0;
		uint64_t waits    = 0;
		uint64_t ordered  = 0;
		uint64_t pending  = 0;
		uint64_t holes    = 0;
	};

	GuestHostView(GraphicContext& graphics, CommandScheduler& scheduler, StreamBuffer& staging, uint64_t base, uint64_t size);
	~GuestHostView();
	KYTY_CLASS_NO_COPY(GuestHostView);

	[[nodiscard]] bool     Enabled() const noexcept { return m_enabled; }
	[[nodiscard]] uint64_t TableAddress() const noexcept;
	[[nodiscard]] uint64_t Lookup(uint64_t vaddr) const;
	[[nodiscard]] Stats    Totals() const;
	[[nodiscard]] bool     Overlaps(uint64_t vaddr, uint64_t size) const;

	void Import(uint64_t vaddr, uint64_t size);
	void Refresh(uint64_t vaddr, uint64_t size);
	void Claim(uint64_t vaddr, uint64_t size);
	void Publish();
	void Forget(uint64_t vaddr, uint64_t size);
	void NoteReach() noexcept { m_reached = true; }
	[[nodiscard]] uint64_t Generation() const noexcept { return m_generation; }
	void NoteOrderedCopy(uint64_t bytes) noexcept { m_stats.ordered += bytes; }

	template <typename Func>
	void ForEachSource(uint64_t vaddr, uint64_t size, Func&& func) const {
		auto at = m_imports.upper_bound(vaddr);
		if (at != m_imports.begin() && std::prev(at)->second.guest + std::prev(at)->second.bytes > vaddr) --at;
		for (; at != m_imports.end() && at->first < vaddr + size; ++at) {
			const auto& import = at->second;
			const auto  begin  = std::max(vaddr, import.guest);
			const auto  end    = std::min(vaddr + size, import.guest + import.bytes);
			func(begin, end - begin, import.buffer, begin - import.guest);
		}
	}

	template <typename Func>
	void ForEachGap(uint64_t vaddr, uint64_t size, Func&& func) const {
		auto done = vaddr;
		ForEachSource(vaddr, size, [&](uint64_t begin, uint64_t bytes, vk::Buffer, uint64_t) {
			if (done < begin) func(done, begin);
			done = begin + bytes;
		});
		if (done < vaddr + size) func(done, vaddr + size);
	}

	template <typename Func>
	void ForEachUnheldGap(uint64_t vaddr, uint64_t size, Func&& func) const {
		std::vector<std::pair<uint64_t, uint64_t>> holes;
		HolesIn(vaddr, size, holes);
		size_t next = 0;
		ForEachGap(vaddr, size, [&](uint64_t from, uint64_t to) {
			auto at = from;
			while (next < holes.size() && holes[next].second <= at) next++;
			for (auto index = next; index < holes.size() && holes[index].first < to; index++) {
				if (at < holes[index].first) func(at, holes[index].first);
				at = std::max(at, holes[index].second);
			}
			if (at < to) func(at, to);
		});
	}

private:
	struct Hole {
		uint64_t end    = 0;
		uint8_t* host   = nullptr;
		uint64_t blocks = UINT64_MAX;
		uint64_t epoch  = UINT64_MAX;
	};

	struct HoleRun {
		uint64_t guest = 0;
		uint64_t end   = 0;
		uint8_t* host  = nullptr;
	};

	struct Imported {
		MemoryTelemetry::Allocation charge;
		vk::Buffer       buffer  = nullptr;
		vk::DeviceMemory memory  = nullptr;
		uint64_t         address = 0;
		uint64_t         guest   = 0;
		uint64_t         bytes   = 0;
		uint8_t*         host    = nullptr;
	};

	void               ImportRun(uint8_t* host, uint64_t guest, uint64_t bytes, std::vector<Imported>& made);
	void               ImportPopulated(uint8_t* host, uint64_t guest, uint64_t bytes, std::vector<Imported>& made,
	                                   std::vector<HoleRun>& holes);
	void               KeepHolesLocked(const std::vector<HoleRun>& holes, uint64_t blocks, uint64_t epoch);
	void               DropHolesLocked(uint64_t begin, uint64_t end);
	void               HolesIn(uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>& holes) const;
	[[nodiscard]] bool ImportPiece(uint8_t* host, uint64_t guest, uint64_t bytes, Imported& import);
	void               Destroy(Imported& import);
	void               WriteEntries(const Imported& import, bool present);
	[[nodiscard]] Buffer& Chunk(uint64_t index);
	void               WaitReaders();
	void               Refuse(uint64_t guest, uint64_t bytes, const char* why);

	GraphicContext&                      m_graphics;
	CommandScheduler&                    m_scheduler;
	StreamBuffer&                        m_staging;
	uint64_t                             m_base       = 0;
	uint64_t                             m_size       = 0;
	uint64_t                             m_alignment  = 0;
	std::unique_ptr<Buffer>              m_root;
	bool                                 m_enabled    = false;
	std::vector<std::unique_ptr<Buffer>> m_chunks;
	std::map<uint64_t, Imported>         m_imports;
	std::vector<uint64_t>                m_words;
	uint64_t                             m_generation = 0;
	bool                                 m_reached    = false;
	Stats                                m_stats;

	mutable std::mutex    m_mutex;
	std::vector<Imported> m_arrivals;
	std::map<uint64_t, Hole> m_holes;
	std::atomic<bool>     m_arrived {false};
	uint64_t              m_refused       = 0;
	bool                  m_refusal_logged = false;
};

}

#endif

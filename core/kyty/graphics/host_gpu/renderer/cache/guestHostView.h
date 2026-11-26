#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTHOSTVIEW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTHOSTVIEW_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

class GuestHostView {
public:
	static constexpr uint32_t PageBits        = 14;
	static constexpr uint64_t PageSize        = uint64_t {1} << PageBits;
	static constexpr uint32_t ChunkBits       = 30;
	static constexpr uint64_t EntriesPerChunk = uint64_t {1} << (ChunkBits - PageBits);
	static constexpr uint64_t SafeTail        = 256;

	struct Stats {
		uint64_t imports  = 0;
		uint64_t bytes    = 0;
		uint64_t refused  = 0;
		uint64_t retiring = 0;
	};

	GuestHostView(GraphicContext& graphics, uint64_t base, uint64_t size);
	~GuestHostView();
	KYTY_CLASS_NO_COPY(GuestHostView);

	[[nodiscard]] bool     Enabled() const noexcept { return m_table.address != 0; }
	[[nodiscard]] uint64_t TableAddress() const noexcept { return m_table.address; }
	[[nodiscard]] uint64_t Lookup(uint64_t vaddr) const;
	[[nodiscard]] Stats    Totals() const;

	void Import(uint64_t vaddr, uint64_t size);
	void Forget(uint64_t vaddr, uint64_t size);
	void Retire(uint64_t completed, uint64_t current);

private:
	struct Area {
		vk::Buffer       buffer  = nullptr;
		vk::DeviceMemory memory  = nullptr;
		uint64_t         address = 0;
		uint64_t*        entries = nullptr;
	};
	struct Imported {
		uint64_t end  = 0;
		uint64_t live = 0;
		Area     area;
	};

	[[nodiscard]] bool      MakeTable(Area& area, uint64_t bytes);
	[[nodiscard]] bool      ImportPiece(const uint8_t* host, uint64_t guest, uint64_t bytes);
	[[nodiscard]] uint64_t* Entry(uint64_t vaddr, bool create);
	void                    ForgetLocked(uint64_t begin, uint64_t end);
	void                    Release(Area& area);

	GraphicContext&                         m_graphics;
	uint64_t                                m_base      = 0;
	uint64_t                                m_size      = 0;
	uint64_t                                m_alignment = 0;
	Area                                    m_table;
	std::vector<Area>                       m_chunks;
	std::map<uint64_t, Imported>            m_imports;
	std::vector<std::pair<uint64_t, Area>>  m_retired;
	Stats                                   m_stats;
	bool                                    m_refusal_logged = false;
	mutable std::mutex                      m_mutex;
};

}

#endif

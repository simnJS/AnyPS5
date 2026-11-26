#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

class MirrorWindowTable {
public:
	static constexpr uint32_t WindowBits = 24;
	static constexpr uint64_t WindowSize = uint64_t{1} << WindowBits;

	MirrorWindowTable(uint64_t span, uint64_t capacity);
	[[nodiscard]] bool Map(uint64_t relative, uint64_t size);
	void Free(uint64_t relative);
	[[nodiscard]] uint64_t Offset(uint64_t relative) const noexcept;
	[[nodiscard]] uint64_t GuestOffset(uint64_t offset) const noexcept;

private:
	uint64_t m_span;
	uint64_t m_capacity;
	std::vector<uint32_t> m_windows;
	std::vector<uint32_t> m_guest_windows;
	std::mutex m_mutex;
};

}

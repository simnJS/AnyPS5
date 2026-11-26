#pragma once

#include "graphics/host_gpu/shaders/guest_apertures.inc"

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

enum class GuestMappingSpace { Gpu, CpuOnly, Invalid };

inline constexpr uint64_t GuestLowerEnd = uint64_t{1} << GUEST_LOWER_SIZE_BITS;
inline constexpr uint64_t GuestExtendedBase = uint64_t{1} << GUEST_EXTENDED_BASE_BITS;
inline constexpr uint64_t GuestExtendedEnd = GuestExtendedBase + (uint64_t{1} << GUEST_EXTENDED_SIZE_BITS);

constexpr GuestMappingSpace ClassifyGuestMapping(uint64_t address, uint64_t size) noexcept {
	if (address == 0 || size == 0 || size > UINT64_MAX - address) return GuestMappingSpace::Invalid;
	if (address < GuestLowerEnd) return size <= GuestLowerEnd - address ? GuestMappingSpace::Gpu : GuestMappingSpace::Invalid;
	if (address < GuestExtendedBase) return size <= GuestExtendedBase - address ? GuestMappingSpace::CpuOnly : GuestMappingSpace::Invalid;
	if (address < GuestExtendedEnd) return size <= GuestExtendedEnd - address ? GuestMappingSpace::Gpu : GuestMappingSpace::Invalid;
	return GuestMappingSpace::CpuOnly;
}

template<typename Reserve>
bool AdmitGuestMapping(uint64_t address, uint64_t size, Reserve&& reserve) {
	const auto space = ClassifyGuestMapping(address, size);
	return space == GuestMappingSpace::CpuOnly || (space == GuestMappingSpace::Gpu && reserve(address, size));
}

template<typename Unmap>
void ForEachGpuUnmap(uint64_t address, uint64_t size, Unmap&& unmap) {
	if (address == 0 || size == 0 || size > UINT64_MAX - address) return;
	const auto end = address + size;
	if (address < GuestLowerEnd) unmap(address, std::min(end, GuestLowerEnd) - address);
	const auto start = std::max(address, GuestExtendedBase);
	const auto finish = std::min(end, GuestExtendedEnd);
	if (start < finish) unmap(start, finish - start);
}

}

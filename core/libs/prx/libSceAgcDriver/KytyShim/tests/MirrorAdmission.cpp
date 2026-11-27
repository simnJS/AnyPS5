#include "graphics/host_gpu/guestMappingSpace.h"
#include "graphics/host_gpu/renderer/cache/mirrorWindowTable.h"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;
constexpr uint64_t Window = MirrorWindowTable::WindowSize;
constexpr uint64_t LowEnd = uint64_t{1} << 40;
constexpr uint64_t Extended = uint64_t{8} << 40;
constexpr uint64_t End = Extended + (uint64_t{1} << 39);
unsigned checks = 0;

void Check(bool ok, const char* message) {
	++checks;
	if (ok) return;
	std::fprintf(stderr, "FAILED: %s\n", message);
	std::exit(1);
}

void CheckApertures() {
	using enum GuestMappingSpace;
	struct Case { uint64_t address, size; GuestMappingSpace expected; };
	for (const auto& c : std::array{
	     Case{0, 1, Invalid}, Case{1, 0, Invalid}, Case{1, LowEnd - 1, Gpu}, Case{LowEnd - 1, 2, Invalid},
	     Case{LowEnd, Extended - LowEnd, CpuOnly}, Case{Extended - 1, 2, Invalid}, Case{Extended, End - Extended, Gpu},
	     Case{End - 1, 1, Gpu}, Case{End - 1, 2, Invalid}, Case{End, 1, CpuOnly}, Case{UINT64_MAX - 1, 1, CpuOnly}, Case{UINT64_MAX, 1, Invalid},
	     Case{UINT64_MAX, 2, Invalid}, Case{Extended - 1, UINT64_MAX, Invalid}, Case{LowEnd - 1, End - LowEnd + 2, Invalid}}) {
		Check(ClassifyGuestMapping(c.address, c.size) == c.expected, "guest aperture classification");
	}
}

void CheckUnmap() {
	using Span = std::pair<uint64_t, uint64_t>;
	auto intersections = [](uint64_t address, uint64_t size) {
		std::vector<Span> result;
		ForEachGpuUnmap(address, size, [&](uint64_t start, uint64_t bytes) { result.emplace_back(start, bytes); });
		return result;
	};
	Check(intersections(LowEnd - 1, Extended - LowEnd + 2) == std::vector<Span>{{LowEnd - 1, 1}, {Extended, 1}},
	      "unmap reports both live GPU intersections independently");
	Check(intersections(Extended - 1, 2) == std::vector<Span>{{Extended, 1}}, "unmap clips a leading gap");
	Check(intersections(End - 1, 2) == std::vector<Span>{{End - 1, 1}}, "unmap clips a trailing gap");
	Check(intersections(LowEnd, Extended - LowEnd).empty() && intersections(UINT64_MAX, 1).empty(), "unmap ignores holes and overflow");
}

void CheckWindows() {
	MirrorWindowTable empty(8 * Window, 0);
	Check(!empty.Map(0, 1), "empty mirror");
	MirrorWindowTable map(5 * Window + 8, 2 * Window + 4);
	Check(map.Map(2 * Window, 4), "exact last four bytes of the mirror");
	Check(map.Offset(2 * Window + 3) == 2 * Window + 3, "last translated byte");
	Check(!map.Map(2 * Window + 3, 2) && map.Offset(2 * Window + 4) == UINT64_MAX, "truncated physical window");
	Check(!map.Map(UINT64_MAX, 2) && !map.Map(1, UINT64_MAX) && !map.Map(0, 0), "overflow and empty requests");
	Check(map.Map(5 * Window, 8), "partial last guest window");
	auto slot = map.Offset(5 * Window);
	Check(map.GuestOffset(slot + 7) == 5 * Window + 7 && map.GuestOffset(slot + 8) == UINT64_MAX, "truncated guest window");
	Check(!map.Map(5 * Window + 7, 2), "guest end plus one");
	map.Free(UINT64_MAX);
	Check(map.Offset(5 * Window) == slot, "invalid release preserves assignment");
	map.Free(5 * Window);
	Check(map.Offset(5 * Window) == UINT64_MAX && map.GuestOffset(slot) == UINT64_MAX, "bidirectional release");
	MirrorWindowTable anchored(8 * Window, 3 * Window);
	Check(anchored.Map(3 * Window, 1), "anchor in physical slot zero");
	Check(!anchored.Map(3 * Window - 1, 2) && anchored.Offset(3 * Window - 1) == UINT64_MAX, "anchor underflow is atomic");
	Check(anchored.Map(3 * Window, 3 * Window) && anchored.Map(4 * Window - 1, 2), "exact capacity and duplicate mapping");
	Check(!anchored.Map(0, 1) && !anchored.Map(3 * Window, 3 * Window + 1), "capacity plus one");
	anchored.Free(4 * Window);
	Check(anchored.Map(Window, 1) && !anchored.Map(3 * Window, 3 * Window), "conflicting middle window");
	Check(anchored.Offset(3 * Window) == 0 && anchored.Offset(5 * Window) == 2 * Window, "conflict preserves both anchors");
}

void CheckSequences() {
	constexpr unsigned guests = 9, slots = 4;
	constexpr uint64_t capacity = (slots - 1) * Window + Window / 2;
	MirrorWindowTable table(guests * Window, capacity);
	std::mt19937 random(153);
	for (unsigned iteration = 0; iteration < 10000; ++iteration) {
		const unsigned first = random() % guests;
		if (random() % 4 == 0) {
			table.Free(first * Window);
			Check(table.Offset(first * Window) == UINT64_MAX, "release removes the guest window");
			continue;
		}
		const unsigned count = 1 + random() % (guests - first);
		const uint64_t start = first * Window + random() % Window;
		const uint64_t finish = count == 1 ? start + 1 : (first + count - 1) * Window + 1 + random() % Window;
		std::array<uint64_t, guests> before;
		for (unsigned g = 0; g < guests; ++g) before[g] = table.Offset(g * Window);
		bool possible = false;
		for (unsigned s = 0; s + count <= slots; ++s) {
			bool fits = s * Window + (finish - first * Window) <= capacity;
			for (unsigned n = 0; n < count; ++n) {
				const auto offset = (s + n) * Window;
				const auto owner = table.GuestOffset(offset);
				fits &= before[first + n] == UINT64_MAX || before[first + n] == offset;
				fits &= owner == UINT64_MAX || owner == (first + n) * Window;
			}
			possible |= fits;
		}
		const bool accepted = table.Map(start, finish - start);
		Check(accepted == possible, "admission agrees with exhaustive slot placements");
		for (unsigned g = 0; g < guests; ++g) {
			const auto offset = table.Offset(g * Window);
			if (!accepted || g < first || g >= first + count || before[g] != UINT64_MAX)
				Check(offset == before[g], "failed or unrelated mapping preserves its assignment");
			if (offset != UINT64_MAX) Check(table.GuestOffset(offset) == g * Window, "reverse lookup after a sequence");
		}
		if (accepted) Check(table.Offset(finish - 1) - table.Offset(start) == finish - start - 1, "whole admitted span is contiguous");
	}
}

MirrorWindowTable syscall_windows(End, Window);

bool Admit(uintptr_t address, size_t bytes) {
	return AdmitGuestMapping(address, bytes, [](uint64_t start, uint64_t size) { return syscall_windows.Map(start, size); });
}

uintptr_t cancelled_address = 0;
size_t cancelled_bytes = 0;

void Cancel(uintptr_t address, size_t bytes) {
	cancelled_address = address;
	cancelled_bytes = bytes;
	syscall_windows.Free(address);
}

void CheckKernelAdmission() {
	DirectMemorySetMapCancellation_nid_no_patch(Cancel);
	DirectMemorySetMapAdmission_nid_no_patch(Admit);
	constexpr auto invalid_placement = (uint64_t{960} << 30) + 1;
	auto misaligned = reinterpret_cast<void*>(invalid_placement);
	bool rejected = false;
	try {
		sceKernelMapFlexibleMemory(&misaligned, PS5_PAGE_SIZE, 3, 0x10);
	} catch (const std::invalid_argument&) {
		rejected = true;
	}
	Check(rejected && cancelled_address == invalid_placement && cancelled_bytes == PS5_PAGE_SIZE, "failed host placement cancels its admission");
	Check(syscall_windows.Offset(invalid_placement) == UINT64_MAX, "cancelled guest window returns to the free pool");
	auto crossing = reinterpret_cast<void*>(LowEnd - PS5_PAGE_SIZE);
	Check(sceKernelMapFlexibleMemory(&crossing, 2 * PS5_PAGE_SIZE, 3, 0x10) == SCE_KERNEL_ERROR_ENOMEM,
	      "kernel refuses a mixed GPU and CPU range before host placement");
	auto first = reinterpret_cast<void*>((uint64_t{960} << 30) - PS5_PAGE_SIZE);
	const auto mapped = first;
	Check(sceKernelMapFlexibleMemory(&first, PS5_PAGE_SIZE, 3, 0x90) == 0 && first == mapped, "kernel maps last page of a window");
	auto* data = static_cast<volatile uint32_t*>(first);
	*data = 0x12345678;
	Check(sceKernelMapFlexibleMemory(&first, 2 * PS5_PAGE_SIZE, 3, 0x10) == SCE_KERNEL_ERROR_ENOMEM,
	      "fixed replacement across the full window is refused");
	Check(first == mapped && *data == 0x12345678, "refusal preserves original mapping and contents");
	int64_t physical = -1;
	Check(sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), 2 * PS5_PAGE_SIZE, 0, 0, &physical) == 0,
	      "allocate direct memory for a fixed replacement");
	Check(sceKernelMapDirectMemory(&first, 2 * PS5_PAGE_SIZE, 3, 0x10, physical, 0) == SCE_KERNEL_ERROR_ENOMEM,
	      "direct fixed replacement propagates capacity refusal");
	Check(first == mapped && *data == 0x12345678, "direct refusal preserves existing flexible memory");
	Check(sceKernelReleaseDirectMemory(physical, 2 * PS5_PAGE_SIZE) == 0, "release direct memory fixture");
	auto other = reinterpret_cast<void*>(Extended);
	Check(sceKernelMapFlexibleMemory(&other, PS5_PAGE_SIZE, 3, 0x90) == SCE_KERNEL_ERROR_ENOMEM, "kernel propagates capacity refusal");
	Check(sceKernelMunmap(reinterpret_cast<uint64_t>(first), PS5_PAGE_SIZE) == 0, "kernel releases fixture");
	DirectMemorySetMapAdmission_nid_no_patch(nullptr);
	DirectMemorySetMapCancellation_nid_no_patch(nullptr);
}

}

int main() {
	CheckApertures();
	CheckUnmap();
	CheckWindows();
	CheckSequences();
	CheckKernelAdmission();
	std::printf("mirror admission: %u checks passed\n", checks);
}

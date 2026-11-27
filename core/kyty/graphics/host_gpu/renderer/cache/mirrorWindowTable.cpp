#include "graphics/host_gpu/renderer/cache/mirrorWindowTable.h"

#include <atomic>

namespace Libs::Graphics {

MirrorWindowTable::MirrorWindowTable(uint64_t span, uint64_t capacity):
    m_span(span), m_capacity(capacity),
    m_windows(span / WindowSize + (span % WindowSize != 0)),
    m_guest_windows(capacity / WindowSize + (capacity % WindowSize != 0)) {}

uint64_t MirrorWindowTable::Offset(uint64_t relative) const noexcept {
	if (relative >= m_span) return UINT64_MAX;
	const auto entry = std::atomic_ref<const uint32_t>(m_windows[relative >> WindowBits]).load(std::memory_order_acquire);
	if (entry == 0) return UINT64_MAX;
	const auto offset = (uint64_t{entry - 1} << WindowBits) + (relative & (WindowSize - 1));
	return offset < m_capacity ? offset : UINT64_MAX;
}

uint64_t MirrorWindowTable::GuestOffset(uint64_t offset) const noexcept {
	if (offset >= m_capacity) return UINT64_MAX;
	const auto entry = std::atomic_ref<const uint32_t>(m_guest_windows[offset >> WindowBits]).load(std::memory_order_acquire);
	if (entry == 0) return UINT64_MAX;
	const auto relative = (uint64_t{entry - 1} << WindowBits) + (offset & (WindowSize - 1));
	return relative < m_span ? relative : UINT64_MAX;
}

void MirrorWindowTable::Free(uint64_t relative) {
	if (relative >= m_span) return;
	std::lock_guard lock(m_mutex);
	auto& window = m_windows[relative >> WindowBits];
	const auto entry = window;
	if (entry == 0) return;
	std::atomic_ref<uint32_t>(window).store(0, std::memory_order_release);
	std::atomic_ref<uint32_t>(m_guest_windows[entry - 1]).store(0, std::memory_order_release);
}

bool MirrorWindowTable::Map(uint64_t relative, uint64_t size) {
	if (size == 0 || relative >= m_span || size > m_span - relative || size > m_capacity) return false;
	const uint32_t first = relative >> WindowBits;
	const uint32_t last = (relative + (size - 1)) >> WindowBits;
	if (first == last && Offset(relative) <= m_capacity - size) return true;
	const uint32_t count = last - first + 1;
	const uint32_t slots = m_guest_windows.size();
	if (count > slots) return false;

	std::lock_guard lock(m_mutex);
	const auto fits = [&](uint32_t slot) {
		if (slot > slots - count || ((uint64_t{slot} << WindowBits) + (relative & (WindowSize - 1))) > m_capacity - size) return false;
		for (uint32_t n = 0; n < count; ++n) {
			if (m_guest_windows[slot + n] != 0 && m_guest_windows[slot + n] != first + n + 1) return false;
		}
		return true;
	};
	uint32_t slot = first % slots;
	bool anchored = false;
	for (uint32_t n = 0; n < count; ++n) {
		const auto entry = m_windows[first + n];
		if (entry == 0) continue;
		if (entry <= n || (anchored && entry - 1 - n != slot)) return false;
		slot = entry - 1 - n;
		anchored = true;
	}
	if (!anchored) {
		if (first > 0 && m_windows[first - 1] != 0 && fits(m_windows[first - 1])) {
			slot = m_windows[first - 1];
		} else if (last + 1 < m_windows.size() && m_windows[last + 1] > count && fits(m_windows[last + 1] - count - 1)) {
			slot = m_windows[last + 1] - count - 1;
		}
		if (!fits(slot)) {
			for (slot = 0; slot <= slots - count && !fits(slot); ++slot) {}
		}
	}
	if (!fits(slot)) return false;
	for (uint32_t n = 0; n < count; ++n) {
		if (m_windows[first + n] != 0) continue;
		std::atomic_ref<uint32_t>(m_guest_windows[slot + n]).store(first + n + 1, std::memory_order_release);
		std::atomic_ref<uint32_t>(m_windows[first + n]).store(slot + n + 1, std::memory_order_release);
	}
	return true;
}

}

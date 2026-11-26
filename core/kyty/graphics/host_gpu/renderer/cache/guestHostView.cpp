#include "graphics/host_gpu/renderer/cache/guestHostView.h"

#include "common/alignment.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstring>

namespace Libs::Graphics {

namespace {

uint32_t PickHostType(const vk::PhysicalDeviceMemoryProperties& memory, uint32_t bits, bool system_only) {
	constexpr auto wanted = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	for (uint32_t type = 0; type < memory.memoryTypeCount; type++) {
		const auto& candidate = memory.memoryTypes[type];
		if ((bits & (1u << type)) == 0 || (candidate.propertyFlags & wanted) != wanted) continue;
		if (system_only && (memory.memoryHeaps[candidate.heapIndex].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) continue;
		return type;
	}
	return UINT32_MAX;
}

}

GuestHostView::GuestHostView(GraphicContext& graphics, uint64_t base, uint64_t size)
    : m_graphics(graphics), m_base(base), m_size(size), m_alignment(graphics.host_pointer_alignment) {
	if (!graphics.external_memory_host_enabled || m_alignment == 0 || m_alignment > PageSize || size == 0) return;
	m_chunks.resize((size + (uint64_t {1} << ChunkBits) - 1) >> ChunkBits);
	if (!MakeTable(m_table, m_chunks.size() * sizeof(uint64_t))) {
		::Log::WriteToConsoleAndLog("[kyty-host-view] no host-visible memory for the table of guest pages: misses read zero\n");
	}
}

GuestHostView::~GuestHostView() {
	for (auto& [start, import]: m_imports) Release(import.area);
	for (auto& [tick, area]: m_retired) Release(area);
	for (auto& chunk: m_chunks) Release(chunk);
	Release(m_table);
}

bool GuestHostView::MakeTable(Area& area, uint64_t bytes) {
	const vk::BufferCreateInfo info {
	    .size  = bytes,
	    .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer};
	if (m_graphics.device.createBuffer(&info, nullptr, &area.buffer) != vk::Result::eSuccess) return false;
	vk::MemoryRequirements requirements;
	m_graphics.device.getBufferMemoryRequirements(area.buffer, &requirements);
	auto type = PickHostType(m_graphics.physical_device_memory_properties, requirements.memoryTypeBits, true);
	if (type == UINT32_MAX) type = PickHostType(m_graphics.physical_device_memory_properties, requirements.memoryTypeBits, false);
	const vk::MemoryAllocateFlagsInfo flags {.flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
	const vk::MemoryAllocateInfo      allocate {.pNext = &flags, .allocationSize = requirements.size, .memoryTypeIndex = type};
	void*                             mapped = nullptr;
	if (type == UINT32_MAX || m_graphics.device.allocateMemory(&allocate, nullptr, &area.memory) != vk::Result::eSuccess ||
	    m_graphics.device.bindBufferMemory(area.buffer, area.memory, 0) != vk::Result::eSuccess ||
	    m_graphics.device.mapMemory(area.memory, 0, VK_WHOLE_SIZE, {}, &mapped) != vk::Result::eSuccess) {
		Release(area);
		return false;
	}
	area.entries = static_cast<uint64_t*>(mapped);
	std::memset(area.entries, 0, bytes);
	const vk::BufferDeviceAddressInfo address {.buffer = area.buffer};
	area.address = m_graphics.device.getBufferAddress(&address);
	return true;
}

void GuestHostView::Release(Area& area) {
	if (area.entries != nullptr) m_graphics.device.unmapMemory(area.memory);
	if (area.buffer) m_graphics.device.destroyBuffer(area.buffer);
	if (area.memory) m_graphics.device.freeMemory(area.memory);
	area = {};
}

uint64_t* GuestHostView::Entry(uint64_t vaddr, bool create) {
	const auto offset = vaddr - m_base;
	auto&      chunk  = m_chunks[offset >> ChunkBits];
	if (chunk.entries == nullptr) {
		if (!create || !MakeTable(chunk, EntriesPerChunk * sizeof(uint64_t))) return nullptr;
		m_table.entries[offset >> ChunkBits] = chunk.address;
	}
	return chunk.entries + ((offset >> PageBits) & (EntriesPerChunk - 1));
}

uint64_t GuestHostView::Lookup(uint64_t vaddr) const {
	std::lock_guard lock(m_mutex);
	if (!Enabled() || vaddr < m_base || vaddr - m_base >= m_size) return 0;
	const auto& chunk = m_chunks[(vaddr - m_base) >> ChunkBits];
	if (chunk.entries == nullptr) return 0;
	const auto entry = chunk.entries[((vaddr - m_base) >> PageBits) & (EntriesPerChunk - 1)];
	return entry != 0 ? entry + (vaddr & (PageSize - 1)) : 0;
}

GuestHostView::Stats GuestHostView::Totals() const {
	std::lock_guard lock(m_mutex);
	auto stats     = m_stats;
	stats.imports  = m_imports.size();
	stats.retiring = m_retired.size();
	return stats;
}

void GuestHostView::Import(uint64_t vaddr, uint64_t size) {
	if (!Enabled() || size == 0 || vaddr >= m_base + m_size || vaddr + size <= m_base) return;
	const auto begin = Common::AlignDown(std::max(vaddr, m_base), PageSize);
	const auto end   = Common::AlignUp(std::min(vaddr + size, m_base + m_size), PageSize);
	std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
	if (!LibKernel::Memory::TryGetBackingPieces(begin, end - begin, pieces)) {
		pieces.assign(1, {reinterpret_cast<const uint8_t*>(begin), end - begin});
	}
	std::lock_guard lock(m_mutex);
	ForgetLocked(begin, end);
	auto guest = begin;
	for (const auto& [host, bytes]: pieces) {
		if (!ImportPiece(host, guest, bytes)) {
			m_stats.refused++;
			if (!m_refusal_logged) {
				m_refusal_logged = true;
				::Log::WriteToConsoleAndLog(fmt::format(
				    "[kyty-host-view] the driver refused to import guest memory 0x{:x} + 0x{:x}: misses there read zero\n", guest,
				    bytes));
			}
		}
		guest += bytes;
	}
}

bool GuestHostView::ImportPiece(const uint8_t* host, uint64_t guest, uint64_t bytes) {
	const auto pointer = reinterpret_cast<uintptr_t>(host);
	if (host == nullptr || bytes == 0 || pointer % m_alignment != 0 || bytes % m_alignment != 0 || guest % PageSize != 0 ||
	    bytes % PageSize != 0) {
		return false;
	}
	auto*                           writable = const_cast<uint8_t*>(host);
	vk::MemoryHostPointerPropertiesEXT properties {};
	if (m_graphics.device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, writable,
	                                                        &properties) != vk::Result::eSuccess) {
		return false;
	}
	Imported                                 import {.end = guest + bytes, .live = bytes / PageSize};
	const vk::ExternalMemoryBufferCreateInfo external {.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT};
	const vk::BufferCreateInfo               info {
	                  .pNext = &external,
	                  .size  = bytes,
	                  .usage = vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer};
	if (m_graphics.device.createBuffer(&info, nullptr, &import.area.buffer) != vk::Result::eSuccess) return false;
	vk::MemoryRequirements requirements;
	m_graphics.device.getBufferMemoryRequirements(import.area.buffer, &requirements);
	const auto type = PickHostType(m_graphics.physical_device_memory_properties,
	                               requirements.memoryTypeBits & properties.memoryTypeBits, false);
	const vk::ImportMemoryHostPointerInfoEXT imported {
	    .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, .pHostPointer = writable};
	const vk::MemoryAllocateFlagsInfo flags {.pNext = &imported, .flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
	const vk::MemoryAllocateInfo      allocate {.pNext = &flags, .allocationSize = bytes, .memoryTypeIndex = type};
	if (type == UINT32_MAX || m_graphics.device.allocateMemory(&allocate, nullptr, &import.area.memory) != vk::Result::eSuccess ||
	    m_graphics.device.bindBufferMemory(import.area.buffer, import.area.memory, 0) != vk::Result::eSuccess) {
		Release(import.area);
		return false;
	}
	const vk::BufferDeviceAddressInfo address {.buffer = import.area.buffer};
	import.area.address = m_graphics.device.getBufferAddress(&address);
	for (uint64_t page = 0; page < bytes; page += PageSize) {
		auto* entry = Entry(guest + page, true);
		if (entry == nullptr) {
			for (uint64_t undo = 0; undo < page; undo += PageSize) *Entry(guest + undo, false) = 0;
			Release(import.area);
			return false;
		}
		*entry = import.area.address + page;
	}
	m_stats.bytes += bytes;
	m_imports.emplace(guest, std::move(import));
	return true;
}

void GuestHostView::Forget(uint64_t vaddr, uint64_t size) {
	if (!Enabled() || size == 0 || vaddr >= m_base + m_size || vaddr + size <= m_base) return;
	std::lock_guard lock(m_mutex);
	ForgetLocked(Common::AlignDown(std::max(vaddr, m_base), PageSize),
	             Common::AlignUp(std::min(vaddr + size, m_base + m_size), PageSize));
}

void GuestHostView::ForgetLocked(uint64_t begin, uint64_t end) {
	auto at = m_imports.upper_bound(begin);
	if (at != m_imports.begin() && std::prev(at)->second.end > begin) --at;
	while (at != m_imports.end() && at->first < end) {
		auto& import = at->second;
		for (auto page = std::max(begin, at->first); page < std::min(end, import.end); page += PageSize) {
			auto* entry = Entry(page, false);
			if (entry != nullptr && *entry != 0) {
				*entry = 0;
				import.live--;
			}
		}
		if (import.live != 0) {
			++at;
			continue;
		}
		m_stats.bytes -= import.end - at->first;
		m_retired.emplace_back(UINT64_MAX, import.area);
		at = m_imports.erase(at);
	}
}

void GuestHostView::Retire(uint64_t completed, uint64_t current) {
	std::lock_guard lock(m_mutex);
	std::erase_if(m_retired, [&](auto& retired) {
		auto& [tick, area] = retired;
		if (tick == UINT64_MAX) {
			tick = current;
			return false;
		}
		if (tick > completed) return false;
		Release(area);
		return true;
	});
}

}

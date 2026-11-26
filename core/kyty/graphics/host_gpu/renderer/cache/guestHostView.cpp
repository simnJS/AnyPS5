#include "graphics/host_gpu/renderer/cache/guestHostView.h"

#include "common/alignment.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "kernel/memory.h"

#include <fmt/format.h>

#include <algorithm>
#include <bit>

namespace Libs::Graphics {

namespace {

constexpr auto ImportHandle = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
constexpr auto ImportUsage  = vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer |
                             vk::BufferUsageFlagBits::eTransferSrc;
constexpr auto TableUsage   = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;

uint32_t PickType(const vk::PhysicalDeviceMemoryProperties& memory, uint32_t bits) {
	constexpr auto wanted = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	for (uint32_t type = 0; type < memory.memoryTypeCount; type++) {
		if ((bits & (1u << type)) != 0 && (memory.memoryTypes[type].propertyFlags & wanted) == wanted) return type;
	}
	return UINT32_MAX;
}

}

GuestHostView::GuestHostView(GraphicContext& graphics, CommandScheduler& scheduler, StreamBuffer& staging, uint64_t base,
                             uint64_t size)
    : m_graphics(graphics), m_scheduler(scheduler), m_staging(staging), m_base(base), m_size(size),
      m_alignment(graphics.host_pointer_alignment) {
	if (size == 0) return;
	const char* missing = nullptr;
	if (!graphics.external_memory_host_enabled) {
		missing = "the device has no VK_EXT_external_memory_host";
	} else if (m_alignment == 0 || !std::has_single_bit(m_alignment)) {
		missing = "minImportedHostPointerAlignment is not a power of two";
	} else {
		const vk::PhysicalDeviceExternalBufferInfo info {.usage = ImportUsage, .handleType = ImportHandle};
		vk::ExternalBufferProperties               properties {};
		graphics.physical_device.getExternalBufferProperties(&info, &properties);
		const auto features = properties.externalMemoryProperties.externalMemoryFeatures;
		if (!(features & vk::ExternalMemoryFeatureFlagBits::eImportable)) {
			missing = "the device cannot import host memory for a buffer read by device address";
		} else if (features & vk::ExternalMemoryFeatureFlagBits::eDedicatedOnly) {
			missing = "the device only takes such a buffer as a dedicated allocation, which a host pointer cannot be";
		}
	}
	if (missing != nullptr) {
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-host-view] guest memory is not imported: {}; a dispatch whose reach is not known binds all mapped "
		    "guest memory in the mirror first\n",
		    missing));
		return;
	}
	m_chunks.resize((size + (uint64_t {1} << ChunkBits) - 1) >> ChunkBits);
	m_root = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, TableUsage,
	                                  m_chunks.size() * sizeof(uint64_t));
	m_root->Fill(0, m_root->Size(), 0);
}

GuestHostView::~GuestHostView() {
	for (auto& [guest, import]: m_imports) Destroy(import);
	for (auto& import: m_arrivals) Destroy(import);
}

uint64_t GuestHostView::TableAddress() const noexcept {
	return m_root != nullptr ? m_root->BufferDeviceAddress() : 0;
}

uint64_t GuestHostView::Lookup(uint64_t vaddr) const {
	const auto at = m_imports.upper_bound(vaddr);
	if (at == m_imports.begin()) return 0;
	const auto& import = std::prev(at)->second;
	return vaddr - import.guest < import.bytes ? import.address + (vaddr - import.guest) : 0;
}

GuestHostView::Stats GuestHostView::Totals() const {
	auto stats    = m_stats;
	stats.imports = m_imports.size();
	std::lock_guard lock(m_mutex);
	stats.refused = m_refused;
	return stats;
}

bool GuestHostView::Overlaps(uint64_t vaddr, uint64_t size) const {
	if (!Enabled() || size == 0) return false;
	std::lock_guard lock(m_mutex);
	const auto      within = [&](const Imported& import) { return import.guest < vaddr + size && vaddr < import.guest + import.bytes; };
	auto            at     = m_imports.upper_bound(vaddr);
	if (at != m_imports.begin() && within(std::prev(at)->second)) return true;
	if (at != m_imports.end() && within(at->second)) return true;
	return std::ranges::any_of(m_arrivals, within);
}

void GuestHostView::Refuse(uint64_t guest, uint64_t bytes, const char* why) {
	std::lock_guard lock(m_mutex);
	m_refused++;
	if (m_refusal_logged) return;
	m_refusal_logged = true;
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-host-view] guest memory 0x{:x} + 0x{:x} is not imported: {}; a dispatch whose reach is not known binds it "
	    "in the mirror first\n",
	    guest, bytes, why));
}

void GuestHostView::Destroy(Imported& import) {
	if (import.buffer) m_graphics.device.destroyBuffer(import.buffer);
	if (import.memory) m_graphics.device.freeMemory(import.memory);
	import = {};
}

bool GuestHostView::ImportPiece(uint8_t* host, uint64_t guest, uint64_t bytes, Imported& import) {
	const auto pointer = reinterpret_cast<uintptr_t>(host);
	auto       first   = pointer;
	auto       last    = pointer + bytes;
	if (m_alignment > PageSize) {
		first = Common::AlignUp(first, m_alignment);
		last  = Common::AlignDown(last, m_alignment);
	}
	if (first >= last || first % m_alignment != 0 || (last - first) % m_alignment != 0 || (first - pointer) % PageSize != 0 ||
	    (last - first) % PageSize != 0 || guest % PageSize != 0) {
		Refuse(guest, bytes, "its host pages do not line up with minImportedHostPointerAlignment");
		return false;
	}
	host += first - pointer;
	guest += first - pointer;
	bytes = last - first;
	const auto                         device = m_graphics.device;
	vk::MemoryHostPointerPropertiesEXT properties {};
	if (device.getMemoryHostPointerPropertiesEXT(ImportHandle, host, &properties) != vk::Result::eSuccess) {
		Refuse(guest, bytes, "the driver gives no memory type for its host pages");
		return false;
	}
	const vk::ExternalMemoryBufferCreateInfo external {.handleTypes = ImportHandle};
	const vk::BufferCreateInfo               info {.pNext = &external, .size = bytes, .usage = ImportUsage};
	if (device.createBuffer(&info, nullptr, &import.buffer) != vk::Result::eSuccess) {
		import = {};
		Refuse(guest, bytes, "the driver refused the buffer");
		return false;
	}
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(import.buffer, &requirements);
	const auto type = PickType(m_graphics.physical_device_memory_properties, requirements.memoryTypeBits & properties.memoryTypeBits);
	if (type == UINT32_MAX || requirements.size > bytes) {
		Destroy(import);
		Refuse(guest, bytes,
		       type == UINT32_MAX ? "no memory type the host can see takes both the buffer and its host pages"
		                          : "the buffer does not fit the size of its host pages");
		return false;
	}
	const vk::ImportMemoryHostPointerInfoEXT imported {.handleType = ImportHandle, .pHostPointer = host};
	const vk::MemoryAllocateFlagsInfo        flags {.pNext = &imported, .flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
	const vk::MemoryAllocateInfo allocate {.pNext = &flags, .allocationSize = bytes, .memoryTypeIndex = type};
	if (device.allocateMemory(&allocate, nullptr, &import.memory) != vk::Result::eSuccess ||
	    device.bindBufferMemory(import.buffer, import.memory, 0) != vk::Result::eSuccess) {
		Destroy(import);
		Refuse(guest, bytes, "the driver refused to import its host pages");
		return false;
	}
	const vk::BufferDeviceAddressInfo address {.buffer = import.buffer};
	import.address = device.getBufferAddress(&address);
	import.guest   = guest;
	import.bytes   = bytes;
	import.host    = host;
	return true;
}

void GuestHostView::ImportRun(uint8_t* host, uint64_t guest, uint64_t bytes, std::vector<Imported>& made) {
	while (bytes != 0) {
		const auto piece = std::min(bytes, PieceBytes - (guest & (PieceBytes - 1)));
		Imported   import;
		if (ImportPiece(host, guest, piece, import)) made.push_back(import);
		host += piece;
		guest += piece;
		bytes -= piece;
	}
}

void GuestHostView::Import(uint64_t vaddr, uint64_t size) {
	if (!Enabled() || size == 0 || vaddr >= m_base + m_size || UINT64_MAX - vaddr < size || vaddr + size <= m_base) return;
	const auto begin = Common::AlignUp(std::max(vaddr, m_base), PageSize);
	const auto end   = Common::AlignDown(std::min(vaddr + size, m_base + m_size), PageSize);
	if (begin >= end) return;
	std::vector<Imported>                            made;
	std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
	if (LibKernel::Memory::TryGetBackingPieces(begin, end - begin, pieces)) {
		auto guest = begin;
		for (const auto& [host, bytes]: pieces) {
			ImportRun(const_cast<uint8_t*>(host), guest, bytes, made);
			guest += bytes;
		}
	} else {
		uint8_t* run       = nullptr;
		uint64_t run_guest = 0;
		uint64_t run_bytes = 0;
		for (auto page = begin; page < end; page += PageSize) {
			auto* host = const_cast<uint8_t*>(static_cast<const uint8_t*>(LibKernel::Memory::FindBackingPointer(page, PageSize)));
			if (host != nullptr && run != nullptr && host == run + run_bytes) {
				run_bytes += PageSize;
				continue;
			}
			if (run != nullptr) ImportRun(run, run_guest, run_bytes, made);
			run       = host;
			run_guest = page;
			run_bytes = PageSize;
		}
		if (run != nullptr) ImportRun(run, run_guest, run_bytes, made);
	}
	if (made.empty()) return;
	std::lock_guard lock(m_mutex);
	m_arrivals.insert(m_arrivals.end(), made.begin(), made.end());
	m_arrived.store(true, std::memory_order_release);
}

Buffer& GuestHostView::Chunk(uint64_t index) {
	auto& chunk = m_chunks[index];
	if (!chunk) {
		chunk = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, TableUsage,
		                                 EntriesPerChunk * sizeof(uint64_t));
		chunk->Fill(0, chunk->Size(), 0);
		const uint64_t address = chunk->BufferDeviceAddress();
		const auto     source  = m_staging.Copy(&address, sizeof(address), sizeof(address));
		m_root->CopyFrom(m_scheduler.Current(), m_staging, source, index * sizeof(uint64_t), sizeof(address));
	}
	return *chunk;
}

void GuestHostView::WriteEntries(const Imported& import, bool present) {
	const auto end = import.guest + import.bytes;
	for (auto at = import.guest; at < end;) {
		const auto offset = at - m_base;
		const auto first  = (offset >> PageBits) & (EntriesPerChunk - 1);
		const auto count  = std::min<uint64_t>(EntriesPerChunk - first, (end - at) >> PageBits);
		auto&      chunk  = Chunk(offset >> ChunkBits);
		if (present) {
			m_words.resize(count);
			for (uint64_t index = 0; index < count; index++) {
				m_words[index] = import.address + (at - import.guest) + (index << PageBits);
			}
			const auto source = m_staging.Copy(m_words.data(), count * sizeof(uint64_t), sizeof(uint64_t));
			chunk.CopyFrom(m_scheduler.Current(), m_staging, source, first * sizeof(uint64_t), count * sizeof(uint64_t));
		} else {
			chunk.Fill(first * sizeof(uint64_t), count * sizeof(uint64_t), 0);
		}
		at += count << PageBits;
	}
}

void GuestHostView::Publish() {
	if (!m_arrived.load(std::memory_order_acquire)) return;
	std::vector<Imported> taken;
	std::vector<Imported> late;
	{
		std::lock_guard lock(m_mutex);
		for (auto& import: m_arrivals) {
			bool held = false;
			ForEachSource(import.guest, import.bytes, [&](uint64_t, uint64_t, vk::Buffer, uint64_t) { held = true; });
			if (held) {
				late.push_back(import);
			} else {
				m_imports.emplace(import.guest, import);
				taken.push_back(import);
			}
		}
		m_arrivals.clear();
		m_arrived.store(false, std::memory_order_release);
	}
	for (auto& import: late) {
		Refuse(import.guest, import.bytes, "an import of an earlier mapping still holds its range");
		Destroy(import);
	}
	for (const auto& import: taken) {
		WriteEntries(import, true);
		m_stats.bytes += import.bytes;
	}
	if (!taken.empty()) m_generation++;
}

void GuestHostView::WaitReaders() {
	if (!m_reached) return;
	if (m_scheduler.Active()) {
		m_scheduler.Finish();
	} else if (const auto tick = m_scheduler.CurrentTick() - 1; tick != 0 && !m_scheduler.IsFree(tick)) {
		m_scheduler.Wait(tick);
	}
	m_stats.waits++;
}

void GuestHostView::Forget(uint64_t vaddr, uint64_t size) {
	if (!Enabled() || size == 0 || vaddr >= m_base + m_size || UINT64_MAX - vaddr < size || vaddr + size <= m_base) return;
	Publish();
	const auto            begin = Common::AlignDown(std::max(vaddr, m_base), PageSize);
	const auto            end   = Common::AlignUp(std::min(vaddr + size, m_base + m_size), PageSize);
	std::vector<Imported> doomed;
	{
		auto at = m_imports.upper_bound(begin);
		if (at != m_imports.begin() && std::prev(at)->second.guest + std::prev(at)->second.bytes > begin) --at;
		std::lock_guard lock(m_mutex);
		while (at != m_imports.end() && at->first < end) {
			doomed.push_back(at->second);
			at = m_imports.erase(at);
		}
	}
	if (doomed.empty()) return;
	for (const auto& import: doomed) {
		WriteEntries(import, false);
		m_stats.bytes -= import.bytes;
	}
	m_generation++;
	WaitReaders();
	std::vector<Imported> kept;
	for (auto& import: doomed) {
		auto* const host  = import.host;
		const auto  guest = import.guest;
		const auto  bytes = import.bytes;
		Destroy(import);
		if (guest < begin) ImportRun(host, guest, begin - guest, kept);
		if (guest + bytes > end) ImportRun(host + (end - guest), end, guest + bytes - end, kept);
	}
	{
		std::lock_guard lock(m_mutex);
		for (const auto& import: kept) m_imports.emplace(import.guest, import);
	}
	for (const auto& import: kept) {
		WriteEntries(import, true);
		m_stats.bytes += import.bytes;
	}
}

}

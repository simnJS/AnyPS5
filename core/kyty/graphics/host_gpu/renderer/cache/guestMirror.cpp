#include "graphics/host_gpu/renderer/cache/guestMirror.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <fmt/format.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <memory>

namespace Libs::Graphics {

namespace {

constexpr auto     BufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress |
                             vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eUniformTexelBuffer |
                             vk::BufferUsageFlagBits::eStorageTexelBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
                             vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndirectBuffer;
constexpr auto     BufferFlags = vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency;
constexpr uint64_t ProbeWaitNs = 5'000'000'000;

std::atomic<VkBuffer> g_mirror_buffer {VK_NULL_HANDLE};
PFN_vkCmdUpdateBuffer g_update_buffer = nullptr;

void VKAPI_CALL UpdateOutsideMirror(VkCommandBuffer command, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                    const void* data) {
	if (buffer == g_mirror_buffer.load(std::memory_order_relaxed)) {
		EXIT("[kyty-mirror] vkCmdUpdateBuffer into the mirror at 0x%" PRIx64 " (%" PRIu64
		     " bytes): write through a stream buffer and vkCmdCopyBuffer\n",
		     static_cast<uint64_t>(offset), static_cast<uint64_t>(size));
	}
	EXIT_IF(g_update_buffer == nullptr);
	g_update_buffer(command, buffer, offset, size, data);
}

uint64_t Now() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string Gib(uint64_t bytes) {
	return bytes % GuestMirror::GiB == 0 ? fmt::format("{} GiB", bytes / GuestMirror::GiB)
	                                     : fmt::format("{:.2f} GiB", static_cast<double>(bytes) / GuestMirror::GiB);
}

uint32_t PickMemoryType(const vk::PhysicalDeviceMemoryProperties& memory, uint32_t bits) {
	for (uint32_t pass = 0; pass < 2; pass++) {
		for (uint32_t type = 0; type < memory.memoryTypeCount; type++) {
			if ((bits & (1u << type)) == 0) continue;
			if (pass == 1 || (memory.memoryTypes[type].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
				return type;
			}
		}
	}
	return UINT32_MAX;
}

enum class Outcome { Bound, Lost, Failed, Stuck };

Outcome ProbeOnce(const GraphicContext& graphics, uint64_t size, std::string& detail) {
	const float                     priority = 1.0f;
	const vk::DeviceQueueCreateInfo queue_info {
	    .queueFamilyIndex = graphics.sparse_family, .queueCount = 1, .pQueuePriorities = &priority};
	vk::PhysicalDeviceVulkan12Features features12 {.bufferDeviceAddress = VK_TRUE};
	vk::PhysicalDeviceFeatures         features {};
	features.sparseBinding         = VK_TRUE;
	features.sparseResidencyBuffer = VK_TRUE;
	const vk::DeviceCreateInfo device_info {.pNext                = &features12,
	                                        .queueCreateInfoCount = 1,
	                                        .pQueueCreateInfos    = &queue_info,
	                                        .pEnabledFeatures     = &features};
	vk::Device device;
	if (const auto made = graphics.physical_device.createDevice(&device_info, nullptr, &device);
	    made != vk::Result::eSuccess) {
		detail = fmt::format("probe device refused ({})", vk::to_string(made));
		return Outcome::Failed;
	}
	auto dispatch = std::make_unique<vk::detail::DispatchLoaderDynamic>(
	    static_cast<VkInstance>(graphics.instance), VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
	    static_cast<VkDevice>(device));
	const auto&  d       = *dispatch;
	vk::Queue    queue   = device.getQueue(graphics.sparse_family, 0, d);
	vk::Buffer   buffer  = nullptr;
	vk::DeviceMemory memory = nullptr;
	vk::Fence    fence   = nullptr;
	Outcome      outcome = Outcome::Failed;
	const auto   cleanup = [&](bool device_usable) {
		if (!device_usable) return;
		if (fence) device.destroyFence(fence, nullptr, d);
		if (buffer) device.destroyBuffer(buffer, nullptr, d);
		if (memory) device.freeMemory(memory, nullptr, d);
		device.destroy(nullptr, d);
	};
	const vk::BufferCreateInfo buffer_info {.flags = BufferFlags, .size = size, .usage = BufferUsage};
	if (const auto made = device.createBuffer(&buffer_info, nullptr, &buffer, d); made != vk::Result::eSuccess) {
		detail = fmt::format("buffer refused ({})", vk::to_string(made));
		buffer = nullptr;
		cleanup(true);
		return Outcome::Failed;
	}
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(buffer, &requirements, d);
	const vk::BufferDeviceAddressInfo address_info {.buffer = buffer};
	const uint64_t                    address = device.getBufferAddress(&address_info, d);
	const auto                        where   = fmt::format("at 0x{:x}, end 0x{:x}", address, address + size);
	const auto type = PickMemoryType(graphics.physical_device_memory_properties, requirements.memoryTypeBits);
	const vk::MemoryAllocateFlagsInfo allocate_flags {.flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
	const vk::MemoryAllocateInfo      allocate {
	         .pNext = &allocate_flags, .allocationSize = requirements.alignment, .memoryTypeIndex = type};
	const vk::FenceCreateInfo    fence_info {};
	if (type == UINT32_MAX || device.allocateMemory(&allocate, nullptr, &memory, d) != vk::Result::eSuccess ||
	    device.createFence(&fence_info, nullptr, &fence, d) != vk::Result::eSuccess) {
		detail = "no memory for one page";
		cleanup(true);
		return Outcome::Failed;
	}
	vk::SparseMemoryBind bind {.resourceOffset = size - requirements.alignment,
	                           .size           = requirements.alignment,
	                           .memory         = memory,
	                           .memoryOffset   = 0};
	const vk::SparseBufferMemoryBindInfo buffer_bind {.buffer = buffer, .bindCount = 1, .pBinds = &bind};
	const vk::BindSparseInfo             bind_info {.bufferBindCount = 1, .pBufferBinds = &buffer_bind};
	auto result = queue.bindSparse(1, &bind_info, fence, d);
	if (result == vk::Result::eSuccess) {
		result = device.waitForFences(1, &fence, VK_TRUE, ProbeWaitNs, d);
	}
	if (result == vk::Result::eSuccess) {
		(void)device.resetFences(1, &fence, d);
		bind.memory = nullptr;
		result      = queue.bindSparse(1, &bind_info, fence, d);
		if (result == vk::Result::eSuccess) {
			result = device.waitForFences(1, &fence, VK_TRUE, ProbeWaitNs, d);
		}
	}
	const uint64_t space = graphics.physical_device_properties.limits.sparseAddressSpaceSize;
	const uint64_t room_size =
	    std::min(GuestMirror::Headroom, space > size ? space - size : 0) & ~(requirements.alignment - 1);
	vk::Buffer room = nullptr;
	if (result == vk::Result::eSuccess && room_size >= requirements.alignment) {
		(void)device.resetFences(1, &fence, d);
		const vk::BufferCreateInfo room_info {.flags = BufferFlags, .size = room_size, .usage = BufferUsage};
		if (device.createBuffer(&room_info, nullptr, &room, d) != vk::Result::eSuccess) {
			room = nullptr;
			cleanup(true);
			detail = fmt::format("{}: bound, but no room for {} of other memory beside it", where, Gib(room_size));
			return Outcome::Failed;
		}
		const vk::SparseMemoryBind room_bind {.resourceOffset = room_size - requirements.alignment,
		                                      .size           = requirements.alignment,
		                                      .memory         = memory,
		                                      .memoryOffset   = 0};
		const vk::SparseBufferMemoryBindInfo room_buffer {.buffer = room, .bindCount = 1, .pBinds = &room_bind};
		const vk::BindSparseInfo             room_bind_info {.bufferBindCount = 1, .pBufferBinds = &room_buffer};
		result = queue.bindSparse(1, &room_bind_info, fence, d);
		if (result == vk::Result::eSuccess) {
			result = device.waitForFences(1, &fence, VK_TRUE, ProbeWaitNs, d);
		}
		if (result != vk::Result::eTimeout) device.destroyBuffer(room, nullptr, d);
	}
	switch (result) {
		case vk::Result::eSuccess:
			outcome = Outcome::Bound;
			detail  = fmt::format("{}: bound, {} of room beside it", where, Gib(room_size));
			break;
		case vk::Result::eErrorDeviceLost:
			outcome = Outcome::Lost;
			detail  = where + ": device lost";
			break;
		case vk::Result::eTimeout:
			outcome = Outcome::Stuck;
			detail  = fmt::format("no answer after {} s", ProbeWaitNs / 1'000'000'000);
			break;
		default:
			outcome = Outcome::Failed;
			detail  = vk::to_string(result);
			break;
	}
	cleanup(outcome != Outcome::Stuck);
	return outcome;
}

}

std::string GuestMirror::Unsupported(const GraphicContext& graphics) {
	if (!graphics.sparse_buffers_enabled) {
		return "the device has no sparseBinding or no sparseResidencyBuffer";
	}
	if (graphics.sparse_family == static_cast<uint32_t>(-1) || !graphics.sparse_queue) {
		return "no queue family can bind sparse memory";
	}
	return {};
}

GuestMirror::ProbeResult GuestMirror::Probe(const GraphicContext& graphics, std::span<const uint64_t> sizes) {
	ProbeResult result;
	result.refusal = Unsupported(graphics);
	if (!result.refusal.empty()) return result;
	const auto     start = Now();
	const uint64_t space = graphics.physical_device_properties.limits.sparseAddressSpaceSize;
	for (const auto size: sizes) {
		if (size == 0 || size > space) {
			result.trials += fmt::format("{}{}: above the {} of sparse address space", result.trials.empty() ? "" : ", ",
			                             Gib(size), Gib(space));
			continue;
		}
		std::string detail;
		const auto  outcome = ProbeOnce(graphics, size, detail);
		result.trials += fmt::format("{}{}: {}", result.trials.empty() ? "" : ", ", Gib(size), detail);
		if (outcome == Outcome::Bound) {
			result.size = size;
			break;
		}
		if (outcome == Outcome::Stuck) break;
	}
	result.ms = static_cast<double>(Now() - start) / 1e6;
	if (result.size == 0) {
		result.refusal = fmt::format("no size bound on a probe device ({})", result.trials);
	}
	return result;
}

GuestMirror::GuestMirror(GraphicContext& graphics, std::span<const uint64_t> sizes, uint64_t base)
    : m_graphics(graphics), m_base(base) {
	const auto probe = Probe(graphics, sizes);
	if (probe.size == 0) {
		m_description = fmt::format("no mirror: {}", probe.refusal);
		return;
	}
	m_size = probe.size;
	if (!CreateBuffer()) {
		m_size = 0;
		return;
	}
	m_shared_queue = graphics.sparse_queue == graphics.queue;
	m_words        = ((m_size >> m_block_bits) + 63) / 64;
	m_bound        = std::make_unique<std::atomic<uint64_t>[]>(m_words);
	m_ready        = std::make_unique<std::atomic<uint64_t>[]>(m_words);
	m_owned        = std::make_unique<std::atomic<uint64_t>[]>(((m_size >> PageBits) + 63) / 64);
	m_used.assign(m_words, 0);
	m_device.assign(m_words, 0);
	m_ages.resize(((m_size >> m_block_bits) + AgeChunkBlocks - 1) / AgeChunkBlocks);
	vk::PhysicalDeviceVulkan13Properties properties13 {};
	vk::PhysicalDeviceProperties2        properties {.pNext = &properties13};
	graphics.physical_device.getProperties2(&properties);
	m_description = fmt::format(
	    "sparse mirror of {} at guest address 0x{:x}, device address 0x{:x} to 0x{:x} (probe {:.1f} ms: {}), blocks of {} KiB, "
	    "{} queue of family {}, memory type {}, every usage of the cache's buffers, shared with the readback family{}",
	    Gib(m_size), m_base, m_device_address, m_device_address + m_size, probe.ms, probe.trials, BlockSize() >> 10u,
	    m_shared_queue ? "the work" : fmt::format("its own (index {})", graphics.sparse_queue_index),
	    graphics.sparse_family, m_memory_type,
	    m_size > properties13.maxBufferSize ? fmt::format(", above maxBufferSize ({})", Gib(properties13.maxBufferSize))
	                                        : std::string {});
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (dispatch.vkCmdUpdateBuffer != UpdateOutsideMirror) {
		g_update_buffer            = dispatch.vkCmdUpdateBuffer;
		dispatch.vkCmdUpdateBuffer = UpdateOutsideMirror;
	}
	g_mirror_buffer.store(static_cast<VkBuffer>(m_buffer), std::memory_order_relaxed);
	m_binder = std::jthread([this](std::stop_token stop) { BinderThread(stop); });
}

GuestMirror::~GuestMirror() {
	if (m_binder.joinable()) {
		m_binder.request_stop();
		m_changed.notify_all();
		m_binder.join();
	}
	auto expected = static_cast<VkBuffer>(m_buffer);
	if (expected != VK_NULL_HANDLE) g_mirror_buffer.compare_exchange_strong(expected, VK_NULL_HANDLE);
	DestroyBuffer();
}

bool GuestMirror::CreateBuffer() {
	const auto                    device = m_graphics.device;
	const std::array<uint32_t, 2> families {m_graphics.queue_family, m_graphics.readback_family};
	vk::BufferCreateInfo          info {.flags = BufferFlags, .size = m_size, .usage = BufferUsage};
	if (m_graphics.readback_family != static_cast<uint32_t>(-1) && m_graphics.readback_family != m_graphics.queue_family) {
		info.sharingMode           = vk::SharingMode::eConcurrent;
		info.queueFamilyIndexCount = static_cast<uint32_t>(families.size());
		info.pQueueFamilyIndices   = families.data();
	}
	if (const auto made = device.createBuffer(&info, nullptr, &m_buffer); made != vk::Result::eSuccess) {
		m_buffer      = nullptr;
		m_description = fmt::format("no mirror: the device refused a sparse buffer of {} ({})", Gib(m_size),
		                            vk::to_string(made));
		return false;
	}
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(m_buffer, &requirements);
	const vk::BufferDeviceAddressInfo address_info {.buffer = m_buffer};
	m_device_address = device.getBufferAddress(&address_info);
	m_block_bits     = std::max<uint32_t>(
        MinBlockBits, static_cast<uint32_t>(std::bit_width(std::max<uint64_t>(requirements.alignment, 1)) - 1));
	m_memory_type = PickMemoryType(m_graphics.physical_device_memory_properties, requirements.memoryTypeBits);
	const auto& memory = m_graphics.physical_device_memory_properties;
	for (uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
		if ((requirements.memoryTypeBits & (1u << type)) != 0 &&
		    (memory.memoryTypes[type].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
		    !(memory.memoryHeaps[memory.memoryTypes[type].heapIndex].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
			m_host_type = type;
			break;
		}
	}
	const vk::FenceCreateInfo         fence_info {};
	vk::SemaphoreTypeCreateInfo       timeline {.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
	const vk::SemaphoreCreateInfo     semaphore_info {.pNext = &timeline};
	if (!std::has_single_bit(requirements.alignment) || m_memory_type == UINT32_MAX ||
	    device.createFence(&fence_info, nullptr, &m_fence) != vk::Result::eSuccess ||
	    device.createSemaphore(&semaphore_info, nullptr, &m_semaphore) != vk::Result::eSuccess) {
		m_description = fmt::format("no mirror: page of {} bytes, memory types 0x{:x}, or no timeline semaphore",
		                            requirements.alignment, requirements.memoryTypeBits);
		DestroyBuffer();
		return false;
	}
	return true;
}

void GuestMirror::DestroyBuffer() {
	const auto device = m_graphics.device;
	if (m_buffer) device.destroyBuffer(m_buffer);
	for (auto& chunk: m_chunks) {
		if (chunk.memory) device.freeMemory(chunk.memory);
	}
	if (m_fence) device.destroyFence(m_fence);
	if (m_semaphore) device.destroySemaphore(m_semaphore);
	m_buffer    = nullptr;
	m_fence     = nullptr;
	m_semaphore = nullptr;
	m_chunks.clear();
}

bool GuestMirror::IsBound(uint64_t vaddr) const noexcept {
	const auto block = (vaddr - m_base) >> m_block_bits;
	return vaddr >= m_base && block / 64 < m_words && (m_bound[block / 64].load(std::memory_order_relaxed) & (uint64_t {1} << (block % 64))) != 0;
}

bool GuestMirror::AllSet(const std::atomic<uint64_t>* bits, uint32_t shift, uint64_t vaddr, uint64_t size) const noexcept {
	if (!Holds(vaddr, size)) return false;
	const auto first = (vaddr - m_base) >> shift;
	const auto last  = (vaddr - m_base + size - 1) >> shift;
	for (auto word = first / 64; word <= last / 64; word++) {
		const auto low  = word == first / 64 ? first % 64 : 0;
		const auto high = word == last / 64 ? last % 64 : 63;
		const auto mask = (high == 63 ? ~uint64_t {0} : (uint64_t {2} << high) - 1) & ~((uint64_t {1} << low) - 1);
		if ((bits[word].load(std::memory_order_acquire) & mask) != mask) return false;
	}
	return true;
}

bool GuestMirror::Requested(uint64_t vaddr, uint64_t size) const noexcept {
	return AllSet(m_bound.get(), m_block_bits, vaddr, size);
}

bool GuestMirror::Ready(uint64_t vaddr, uint64_t size) const noexcept {
	return AllSet(m_ready.get(), m_block_bits, vaddr, size);
}

bool GuestMirror::Owned(uint64_t vaddr, uint64_t size) const noexcept {
	return AllSet(m_owned.get(), PageBits, vaddr, size);
}

bool GuestMirror::AnyOwned(uint64_t vaddr, uint64_t size) const noexcept {
	bool any = false;
	ForEachOwned(vaddr, size, [&any](uint64_t, uint64_t) { any = true; });
	return any;
}

void GuestMirror::Own(uint64_t vaddr, uint64_t size, Runs* fresh) {
	if (!Holds(vaddr, size)) return;
	const auto first = (vaddr - m_base) >> PageBits;
	const auto last  = (vaddr - m_base + size - 1) >> PageBits;
	for (auto word = first / 64; word <= last / 64; word++) {
		const auto low   = word == first / 64 ? first % 64 : 0;
		const auto high  = word == last / 64 ? last % 64 : 63;
		const auto mask  = (high == 63 ? ~uint64_t {0} : (uint64_t {2} << high) - 1) & ~((uint64_t {1} << low) - 1);
		const auto value = m_owned[word].load(std::memory_order_relaxed);
		auto       added = mask & ~value;
		if (added == 0) continue;
		m_owned[word].store(value | mask, std::memory_order_release);
		while (fresh != nullptr && added != 0) {
			const auto start  = static_cast<uint64_t>(std::countr_zero(added));
			const auto length = static_cast<uint64_t>(std::countr_one(added >> start));
			const auto begin  = m_base + ((word * 64 + start) << PageBits);
			const auto end    = begin + (length << PageBits);
			if (!fresh->empty() && fresh->back().second == begin) {
				fresh->back().second = end;
			} else {
				fresh->emplace_back(begin, end);
			}
			added = start + length >= 64 ? 0 : added & ~((uint64_t {1} << (start + length)) - 1);
		}
	}
}

void GuestMirror::Disown(uint64_t begin, uint64_t end) noexcept {
	if (begin >= end || !Holds(begin, end - begin)) return;
	const auto first = Common::AlignDown(begin - m_base, BlockSize()) >> PageBits;
	const auto last  = (Common::AlignUp(end - m_base, BlockSize()) - 1) >> PageBits;
	for (auto word = first / 64; word <= last / 64; word++) {
		const auto low  = word == first / 64 ? first % 64 : 0;
		const auto high = word == last / 64 ? last % 64 : 63;
		const auto mask = (high == 63 ? ~uint64_t {0} : (uint64_t {2} << high) - 1) & ~((uint64_t {1} << low) - 1);
		m_owned[word].fetch_and(~mask, std::memory_order_release);
	}
}

bool GuestMirror::AnyRequested(uint64_t vaddr, uint64_t size) const noexcept {
	bool any = false;
	ForEachRequested(vaddr, size, [&any](uint64_t, uint64_t) { any = true; });
	return any;
}

void GuestMirror::Use(uint64_t vaddr, uint64_t size) noexcept {
	if (!Holds(vaddr, size)) return;
	const auto first = (vaddr - m_base) >> m_block_bits;
	const auto last  = (vaddr - m_base + size - 1) >> m_block_bits;
	for (auto word = first / 64; word <= last / 64; word++) {
		const auto low  = word == first / 64 ? first % 64 : 0;
		const auto high = word == last / 64 ? last % 64 : 63;
		const auto mask = (high == 63 ? ~uint64_t {0} : (uint64_t {2} << high) - 1) & ~((uint64_t {1} << low) - 1);
		if (m_used[word] == 0) m_used_words.push_back(word);
		m_used[word] |= mask;
	}
}

void GuestMirror::NoteDevice(uint64_t vaddr, uint64_t size) noexcept {
	if (!Holds(vaddr, size)) return;
	const auto first = (vaddr - m_base) >> m_block_bits;
	const auto last  = (vaddr - m_base + size - 1) >> m_block_bits;
	for (auto block = first; block <= last; block++) {
		m_device[block / 64] |= uint64_t {1} << (block % 64);
	}
}

uint32_t GuestMirror::AgeOf(uint64_t block) const noexcept {
	const auto& chunk = m_ages[block / AgeChunkBlocks];
	return chunk ? chunk[block % AgeChunkBlocks] : 0;
}

void GuestMirror::Age(uint32_t frame) {
	for (const auto word: m_used_words) {
		for (auto bits = m_used[word]; bits != 0; bits &= bits - 1) {
			const auto block = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
			auto&      chunk = m_ages[block / AgeChunkBlocks];
			if (!chunk) chunk = std::make_unique<uint32_t[]>(AgeChunkBlocks);
			chunk[block % AgeChunkBlocks] = frame;
		}
		m_used[word] = 0;
	}
	m_used_words.clear();
}

void GuestMirror::PickOldest(uint64_t max_blocks, uint32_t older_than, Runs& victims) const {
	victims.clear();
	if (m_size == 0 || max_blocks == 0) return;
	struct Candidate {
		bool     device = false;
		uint32_t age    = 0;
		uint64_t block  = 0;
	};
	std::vector<Candidate> candidates;
	for (uint64_t word = 0; word < m_words; word++) {
		for (auto bits = m_bound[word].load(std::memory_order_acquire) & ~m_used[word]; bits != 0; bits &= bits - 1) {
			const auto block = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
			const auto age   = AgeOf(block);
			if (age >= older_than) continue;
			candidates.push_back({(m_device[word] >> (block % 64) & 1u) != 0, age, block});
		}
	}
	const auto count = std::min<uint64_t>(max_blocks, candidates.size());
	std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(count), candidates.end(),
	                  [](const Candidate& a, const Candidate& b) {
		                  return a.device != b.device ? !a.device : a.age != b.age ? a.age < b.age : a.block < b.block;
	                  });
	candidates.resize(count);
	std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.block < b.block; });
	for (const auto& candidate: candidates) {
		const auto begin = m_base + (candidate.block << m_block_bits);
		if (!victims.empty() && victims.back().second == begin) {
			victims.back().second += BlockSize();
		} else {
			victims.emplace_back(begin, begin + BlockSize());
		}
	}
}

void GuestMirror::Release(const Runs& runs) {
	if (m_size == 0 || runs.empty()) return;
	Flush();
	std::vector<Run> blocks;
	for (const auto& [begin, end]: runs) {
		if (begin >= end || !Holds(begin, end - begin)) continue;
		const auto first = (begin - m_base) >> m_block_bits;
		const auto last  = (end - m_base - 1) >> m_block_bits;
		for (auto block = first; block <= last; block++) {
			const auto bit = uint64_t {1} << (block % 64);
			if ((m_bound[block / 64].load(std::memory_order_relaxed) & bit) == 0) continue;
			m_ready[block / 64].fetch_and(~bit, std::memory_order_acq_rel);
			m_bound[block / 64].fetch_and(~bit, std::memory_order_relaxed);
			m_device[block / 64] &= ~bit;
			if (!blocks.empty() && blocks.back().first + blocks.back().count == block) {
				blocks.back().count++;
			} else {
				blocks.push_back({block, 1});
			}
		}
	}
	for (const auto& [begin, end]: runs) Disown(begin, end);
	if (!blocks.empty()) Post({.kind = Kind::Unbind, .runs = std::move(blocks)});
}

void GuestMirror::NoteRange(uint64_t vaddr, uint64_t size, Runs* fresh, bool prefer_host) {
	if (m_size == 0 || size == 0 || UINT64_MAX - vaddr < size) return;
	const uint64_t blocks = m_size >> m_block_bits;
	if (vaddr < m_base) {
		m_outside_blocks += ((std::min(vaddr + size, m_base) - vaddr + BlockSize() - 1) >> m_block_bits);
		if (vaddr + size <= m_base) return;
		size -= m_base - vaddr;
		vaddr = m_base;
	}
	const uint64_t first  = (vaddr - m_base) >> m_block_bits;
	uint64_t       last   = (vaddr - m_base + size - 1) >> m_block_bits;
	if (first >= blocks) {
		m_outside_blocks += last - first + 1;
		return;
	}
	if (last >= blocks) {
		m_outside_blocks += last - blocks + 1;
		last = blocks - 1;
	}
	for (auto block = first; block <= last; block++) {
		auto&      word  = m_bound[block / 64];
		const auto bit   = uint64_t {1} << (block % 64);
		const auto value = word.load(std::memory_order_relaxed);
		if ((value & bit) != 0) continue;
		word.store(value | bit, std::memory_order_relaxed);
		if (fresh != nullptr) {
			const auto begin = m_base + (block << m_block_bits);
			if (!fresh->empty() && fresh->back().second == begin) {
				fresh->back().second += BlockSize();
			} else {
				fresh->emplace_back(begin, begin + BlockSize());
			}
		}
		if (m_pending.empty()) m_first_pending = Now();
		m_pending_blocks++;
		if (!m_pending.empty() && m_pending.back().first + m_pending.back().count == block &&
		    m_pending.back().prefer_host == prefer_host) {
			m_pending.back().count++;
		} else {
			m_pending.push_back({block, 1, prefer_host});
		}
	}
}

void GuestMirror::Post(Command&& command) {
	{
		Common::ScopedLock lock(m_mutex);
		if (command.kind == Kind::Bind && !m_commands.empty() && m_commands.back().kind == Kind::Bind) {
			auto& queued = m_commands.back();
			queued.runs.insert(queued.runs.end(), command.runs.begin(), command.runs.end());
			queued.value     = std::max(queued.value, command.value);
			queued.requested = std::min(queued.requested, command.requested);
		} else {
			m_commands.push_back(std::move(command));
		}
	}
	m_changed.notify_all();
}

GuestMirror::Wait GuestMirror::Submit() {
	if (m_size == 0) return {};
	Flush();
	if (m_bind_signaled.load(std::memory_order_acquire) >= m_bind_posted) return {};
	m_waits.fetch_add(1, std::memory_order_relaxed);
	if (m_shared_queue) {
		WaitBound(m_bind_posted);
		return {};
	}
	return {m_semaphore, m_bind_posted};
}

void GuestMirror::Flush() {
	if (m_pending.empty()) return;
	m_pending_blocks = 0;
	Post({.kind = Kind::Bind, .runs = std::move(m_pending), .value = ++m_bind_posted, .requested = m_first_pending});
	m_pending.clear();
}

void GuestMirror::WaitIdle() {
	if (m_size == 0) return;
	Common::UniqueLock lock(m_mutex);
	m_changed.wait(lock.Native(), [this]() KYTY_REQUIRES(m_mutex) { return m_commands.empty() && !m_busy; });
}

void GuestMirror::WaitBound(uint64_t value) {
	if (m_size == 0) return;
	Common::UniqueLock lock(m_mutex);
	m_changed.wait(lock.Native(), [this, value]() KYTY_REQUIRES(m_mutex) {
		return m_bind_signaled.load(std::memory_order_acquire) >= value;
	});
}

void GuestMirror::BinderThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("Thread_GuestMirror");
	for (;;) {
		Command command;
		{
			Common::UniqueLock lock(m_mutex);
			m_changed.wait(lock.Native(), stop, [this]() KYTY_REQUIRES(m_mutex) { return !m_commands.empty(); });
			if (m_commands.empty()) return;
			command = std::move(m_commands.front());
			m_commands.pop_front();
			m_busy = true;
		}
		switch (command.kind) {
			case Kind::Bind: Bind(command); break;
			case Kind::Unbind: Unbind(command); break;
		}
		{
			Common::ScopedLock lock(m_mutex);
			if (command.kind == Kind::Bind) m_bind_signaled.store(command.value, std::memory_order_release);
			m_busy = false;
		}
		m_changed.notify_all();
	}
}

GuestMirror::Grant GuestMirror::Allocate(uint64_t wanted, bool host) {
	auto& current = m_current_chunks[host];
	auto& free_slots = m_free_slots[host];
	const auto slots = static_cast<uint32_t>(std::max<uint64_t>(ChunkBytes >> m_block_bits, 1));
	if (current != UINT32_MAX && m_chunks[current].next < slots) {
		auto&      chunk = m_chunks[current];
		const auto count = static_cast<uint32_t>(std::min<uint64_t>(wanted, slots - chunk.next));
		const auto first = chunk.next;
		chunk.next += count;
		chunk.used += count;
		return {current, first, count};
	}
	if (!free_slots.empty()) {
		const auto slot = free_slots.back();
		free_slots.pop_back();
		m_chunks[slot.chunk].used++;
		return {slot.chunk, slot.index, 1};
	}
	const vk::MemoryAllocateFlagsInfo flags {.flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
	const vk::MemoryAllocateInfo      info {
	         .pNext = &flags, .allocationSize = uint64_t {slots} << m_block_bits, .memoryTypeIndex = host ? m_host_type : m_memory_type};
	vk::DeviceMemory             memory;
	const auto                   made = m_graphics.device.allocateMemory(&info, nullptr, &memory);
	if (made != vk::Result::eSuccess) {
		EXIT("[kyty-mirror] could not allocate %" PRIu64 " MiB for the mirror: %s\n", info.allocationSize >> 20u,
		     vk::to_string(made).c_str());
	}
	if (!m_free_chunks.empty()) {
		current = m_free_chunks.back();
		m_free_chunks.pop_back();
		m_chunks[current] = {memory, 0, 0, host};
	} else {
		current = static_cast<uint32_t>(m_chunks.size());
		m_chunks.push_back({memory, 0, 0, host});
	}
	m_chunk_count.fetch_add(1, std::memory_order_relaxed);
	return Allocate(wanted, host);
}

void GuestMirror::Bind(const Command& command) {
	m_binds.clear();
	uint64_t blocks = 0;
	for (const auto& run: command.runs) {
		auto block = run.first;
		auto left  = run.count;
		while (left != 0) {
			if (m_block_slots.contains(block)) {
				block++;
				left--;
				continue;
			}
			uint64_t fresh = 1;
			while (fresh < left && !m_block_slots.contains(block + fresh)) fresh++;
			const auto [chunk, index, count] = Allocate(fresh, run.prefer_host && m_host_type != UINT32_MAX);
			m_binds.push_back({.resourceOffset = block << m_block_bits,
			                   .size           = uint64_t {count} << m_block_bits,
			                   .memory         = m_chunks[chunk].memory,
			                   .memoryOffset   = uint64_t {index} << m_block_bits});
			for (uint32_t i = 0; i < count; i++) m_block_slots[block + i] = {chunk, index + i};
			block += count;
			left -= count;
			blocks += count;
		}
	}
	Execute(m_binds, "bind", command.value);
	for (const auto& run: command.runs) {
		for (auto block = run.first; block < run.first + run.count; block++) {
			if (m_bound[block / 64].load(std::memory_order_relaxed) & (uint64_t {1} << (block % 64))) {
				m_ready[block / 64].fetch_or(uint64_t {1} << (block % 64), std::memory_order_acq_rel);
			}
		}
	}
	if (command.requested != 0) {
		const auto waited = Now() - command.requested;
		m_requests.fetch_add(1, std::memory_order_relaxed);
		m_request_ns.fetch_add(waited, std::memory_order_relaxed);
		auto longest = m_request_longest_ns.load(std::memory_order_relaxed);
		while (waited > longest && !m_request_longest_ns.compare_exchange_weak(longest, waited, std::memory_order_relaxed)) {
		}
	}
	m_blocks.fetch_add(blocks, std::memory_order_relaxed);
	m_bound_blocks.fetch_add(blocks, std::memory_order_relaxed);
}

void GuestMirror::Unbind(const Command& command) {
	m_binds.clear();
	uint64_t blocks = 0;
	for (const auto& run: command.runs) {
		m_binds.push_back({.resourceOffset = run.first << m_block_bits, .size = run.count << m_block_bits});
	}
	if (command.after.semaphore) {
		const vk::SemaphoreWaitInfo wait {
		    .semaphoreCount = 1, .pSemaphores = &command.after.semaphore, .pValues = &command.after.value};
		if (const auto waited = m_graphics.device.waitSemaphores(&wait, UINT64_MAX); waited != vk::Result::eSuccess) {
			EXIT("[kyty-mirror] waiting for tick %" PRIu64 " before an unbind failed: %s\n", command.after.value,
			     vk::to_string(waited).c_str());
		}
	}
	Execute(m_binds, "unbind", 0);
	for (const auto& run: command.runs) {
		for (auto block = run.first; block < run.first + run.count; block++) {
			const auto found = m_block_slots.find(block);
			if (found == m_block_slots.end()) continue;
			Free(found->second);
			m_block_slots.erase(found);
			blocks++;
		}
	}
	std::vector<bool> emptied(m_chunks.size(), false);
	for (uint32_t index = 0; index < m_chunks.size(); index++) {
		auto& chunk = m_chunks[index];
		if (chunk.memory && chunk.used == 0) {
			m_graphics.device.freeMemory(chunk.memory);
			auto& current = m_current_chunks[chunk.host];
			chunk          = {};
			emptied[index] = true;
			m_free_chunks.push_back(index);
			m_chunk_count.fetch_sub(1, std::memory_order_relaxed);
			if (current == index) current = UINT32_MAX;
		}
	}
	for (auto& slots: m_free_slots) {
		std::erase_if(slots, [&](const Slot& slot) { return emptied[slot.chunk]; });
	}
	m_unbound_blocks.fetch_add(blocks, std::memory_order_relaxed);
	m_bound_blocks.fetch_sub(std::min(blocks, m_bound_blocks.load(std::memory_order_relaxed)), std::memory_order_relaxed);
}

void GuestMirror::Free(Slot slot) {
	m_chunks[slot.chunk].used--;
	m_free_slots[m_chunks[slot.chunk].host].push_back(slot);
}

void GuestMirror::Execute(std::vector<vk::SparseMemoryBind>& binds, const char* what, uint64_t signal) {
	const auto device = m_graphics.device;
	const auto calls  = std::max<size_t>((binds.size() + MaxBindsPerCall - 1) / MaxBindsPerCall, 1);
	for (size_t call = 0; call < calls; call++) {
		const auto begin = call * MaxBindsPerCall;
		const auto count = std::min(MaxBindsPerCall, binds.size() - begin);
		const vk::SparseBufferMemoryBindInfo buffer_bind {
		    .buffer = m_buffer, .bindCount = static_cast<uint32_t>(count), .pBinds = binds.data() + begin};
		const bool                      signals = call + 1 == calls && signal != 0;
		vk::TimelineSemaphoreSubmitInfo timeline {.signalSemaphoreValueCount = signals ? 1u : 0u,
		                                          .pSignalSemaphoreValues    = &signal};
		const vk::BindSparseInfo info {.pNext                = &timeline,
		                               .bufferBindCount      = count != 0 ? 1u : 0u,
		                               .pBufferBinds         = &buffer_bind,
		                               .signalSemaphoreCount = signals ? 1u : 0u,
		                               .pSignalSemaphores    = &m_semaphore};
		const auto start = Now();
		vk::Result result;
		if (m_shared_queue) {
			Common::LockGuard lock(m_graphics.queue_mutex);
			result = m_graphics.sparse_queue.bindSparse(1, &info, m_fence);
		} else {
			result = m_graphics.sparse_queue.bindSparse(1, &info, m_fence);
		}
		const auto called = Now();
		if (result == vk::Result::eSuccess) {
			result = device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX);
		}
		if (result != vk::Result::eSuccess) {
			EXIT("[kyty-mirror] vkQueueBindSparse (%s of %zu ranges at 0x%" PRIx64 ") failed: %s\n", what, count,
			     count != 0 ? binds[begin].resourceOffset : 0, vk::to_string(result).c_str());
		}
		(void)device.resetFences(1, &m_fence);
		const auto done = Now();
		if (count == 0) continue;
		m_calls.fetch_add(1, std::memory_order_relaxed);
		m_call_ns.fetch_add(called - start, std::memory_order_relaxed);
		m_done_ns.fetch_add(done - start, std::memory_order_relaxed);
		auto longest = m_longest_ns.load(std::memory_order_relaxed);
		while (done - start > longest &&
		       !m_longest_ns.compare_exchange_weak(longest, done - start, std::memory_order_relaxed)) {
		}
	}
}

GuestMirror::Stats GuestMirror::Totals() const {
	return {.bound_blocks       = m_bound_blocks.load(std::memory_order_relaxed),
	        .chunks             = m_chunk_count.load(std::memory_order_relaxed),
	        .calls              = m_calls.load(std::memory_order_relaxed),
	        .blocks             = m_blocks.load(std::memory_order_relaxed),
	        .unbound_blocks     = m_unbound_blocks.load(std::memory_order_relaxed),
	        .call_ns            = m_call_ns.load(std::memory_order_relaxed),
	        .done_ns            = m_done_ns.load(std::memory_order_relaxed),
	        .longest_ns         = m_longest_ns.load(std::memory_order_relaxed),
	        .requests           = m_requests.load(std::memory_order_relaxed),
	        .request_ns         = m_request_ns.load(std::memory_order_relaxed),
	        .request_longest_ns = m_request_longest_ns.load(std::memory_order_relaxed),
	        .waits              = m_waits.load(std::memory_order_relaxed),
	        .unbound_reads      = m_unbound_reads.load(std::memory_order_relaxed)};
}

GuestMirror::Stats GuestMirror::TakeStats() {
	const auto totals = Totals();
	Stats      delta  = totals;
	delta.calls              = totals.calls - m_taken.calls;
	delta.blocks             = totals.blocks - m_taken.blocks;
	delta.unbound_blocks     = totals.unbound_blocks - m_taken.unbound_blocks;
	delta.call_ns            = totals.call_ns - m_taken.call_ns;
	delta.done_ns            = totals.done_ns - m_taken.done_ns;
	delta.requests           = totals.requests - m_taken.requests;
	delta.request_ns         = totals.request_ns - m_taken.request_ns;
	delta.waits              = totals.waits - m_taken.waits;
	delta.unbound_reads      = totals.unbound_reads - m_taken.unbound_reads;
	m_taken                  = totals;
	delta.longest_ns         = m_longest_ns.exchange(0, std::memory_order_relaxed);
	delta.request_longest_ns = m_request_longest_ns.exchange(0, std::memory_order_relaxed);
	return delta;
}

}

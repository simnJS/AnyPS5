#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/submitStats.h"

#include <array>
#include <cinttypes>
#include <cstring>
#include <numeric>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

namespace {

constexpr size_t WATCHES_INITIAL_RESERVE = 0x4000;
constexpr size_t WATCHES_RESERVE_CHUNK   = 0x1000;

constexpr vk::AccessFlags MEMORY_ACCESS =
    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;

[[nodiscard]] VmaAllocationCreateFlags AllocationFlags(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::Upload:
		case MemoryUsage::Stream:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT |
			       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		case MemoryUsage::Download:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		case MemoryUsage::DeviceLocal: return {};
	}
	return {};
}

[[nodiscard]] VmaMemoryUsage AllocationUsage(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::DeviceLocal:
		case MemoryUsage::Stream: return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		case MemoryUsage::Upload:
		case MemoryUsage::Download: return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

[[nodiscard]] bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result) {
	if (alignment == 0) {
		result = value;
		return true;
	}
	const auto aligned = Common::AlignUp(value, alignment);
	if (aligned < value) {
		return false;
	}
	result = aligned;
	return true;
}

}

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(usage), m_cpu_address(cpu_address),
      m_size(size) {
	Create(flags);
}

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, SparseResident,
               vk::BufferUsageFlags flags, uint64_t size)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(MemoryUsage::DeviceLocal), m_size(size) {
	if (!graphics.partial_buffers_enabled || (flags & vk::BufferUsageFlagBits::eShaderDeviceAddress)) {
		Create(flags);
		return;
	}
	vk::BufferCreateInfo buffer_info {};
	buffer_info.flags = vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency;
	buffer_info.size  = size;
	buffer_info.usage = flags;
	const std::array<uint32_t, 2> families {graphics.queue_family, graphics.readback_family};
	if (graphics.readback_family != static_cast<uint32_t>(-1)) {
		buffer_info.sharingMode           = vk::SharingMode::eConcurrent;
		buffer_info.queueFamilyIndexCount = 2;
		buffer_info.pQueueFamilyIndices   = families.data();
	}
	if (graphics.device.createBuffer(&buffer_info, nullptr, &m_buffer) != vk::Result::eSuccess) {
		m_buffer = nullptr;
		Create(flags);
		return;
	}
	const auto requirements = graphics.device.getBufferMemoryRequirements(m_buffer);
	m_sparse_block          = requirements.alignment;
	m_sparse_types          = requirements.memoryTypeBits;
	m_sparse_size           = requirements.size;
	m_sparse_blocks.assign((m_sparse_size + m_sparse_block - 1) / m_sparse_block, nullptr);
#if defined(ANYPS5_KYTY)
	m_device_local = true;
#endif
}

void Buffer::Commit(uint64_t offset, uint64_t size) {
	if (m_sparse_block == 0 || size == 0) {
		return;
	}
	EXIT_IF(offset > m_size || size > m_size - offset);
	auto&                 graphics = *m_graphics;
	const auto            first    = offset / m_sparse_block;
	const auto            last     = (offset + size - 1) / m_sparse_block;
	std::vector<uint64_t> blocks;
	for (auto block = first; block <= last; block++) {
		if (m_sparse_blocks[block] == nullptr) {
			blocks.push_back(block);
		}
	}
	if (blocks.empty()) {
		return;
	}
	vk::BufferCreateInfo zero_info {};
	zero_info.size  = m_sparse_block;
	zero_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	std::vector<vk::Buffer>           zero_buffers(blocks.size(), nullptr);
	std::vector<vk::SparseMemoryBind> binds;
	binds.reserve(blocks.size());
	for (size_t i = 0; i < blocks.size(); i++) {
		const auto block = blocks[i];
		EXIT_IF(graphics.device.createBuffer(&zero_info, nullptr, &zero_buffers[i]) != vk::Result::eSuccess);
		const auto           zero_requirements = graphics.device.getBufferMemoryRequirements(zero_buffers[i]);
		VkMemoryRequirements memory {};
		memory.size           = std::max(m_sparse_block, zero_requirements.size);
		memory.alignment      = std::max(m_sparse_block, zero_requirements.alignment);
		memory.memoryTypeBits = m_sparse_types & zero_requirements.memoryTypeBits;
		VmaAllocationCreateInfo alloc_info {};
		alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		VmaAllocationInfo allocated {};
		const auto        result =
		    vmaAllocateMemory(graphics.allocator, &memory, &alloc_info, &m_sparse_blocks[block], &allocated);
		if (result != VK_SUCCESS) {
			std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
			const auto heap_count = graphics.GetHeapUsage(heaps);
			for (uint32_t heap = 0; heap < heap_count; heap++) {
				LOGF("Buffer: heap %u%s: %" PRIu64 " MiB used of a %" PRIu64 " MiB budget\n", heap,
				     heaps[heap].device_local ? " (device local)" : "", heaps[heap].usage >> 20u,
				     heaps[heap].budget >> 20u);
			}
			EXIT("Buffer: could not allocate sparse block %" PRIu64 " of %" PRIu64 " bytes: %s\n", block,
			     m_sparse_block, vk::to_string(static_cast<vk::Result>(result)).c_str());
		}
		EXIT_IF(graphics.device.bindBufferMemory(zero_buffers[i], allocated.deviceMemory, allocated.offset) !=
		        vk::Result::eSuccess);
		vk::SparseMemoryBind bind {};
		bind.resourceOffset = block * m_sparse_block;
		bind.size           = std::min(m_sparse_block, m_sparse_size - block * m_sparse_block);
		bind.memory         = allocated.deviceMemory;
		bind.memoryOffset   = allocated.offset;
		binds.push_back(bind);
	}
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient;
	pool_info.queueFamilyIndex = graphics.queue_family;
	vk::CommandPool pool       = nullptr;
	EXIT_IF(graphics.device.createCommandPool(&pool_info, nullptr, &pool) != vk::Result::eSuccess);
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::CommandBuffer command   = nullptr;
	EXIT_IF(graphics.device.allocateCommandBuffers(&allocate, &command) != vk::Result::eSuccess);
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	EXIT_IF(command.begin(&begin_info) != vk::Result::eSuccess);
	for (const auto zero_buffer: zero_buffers) {
		command.fillBuffer(zero_buffer, 0, VK_WHOLE_SIZE, 0);
	}
	const vk::MemoryBarrier zero_visibility {.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                                         .dstAccessMask = vk::AccessFlagBits::eMemoryRead};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eAllCommands, {},
	                        1, &zero_visibility, 0, nullptr, 0, nullptr);
	EXIT_IF(command.end() != vk::Result::eSuccess);
	vk::SemaphoreCreateInfo semaphore_info {};
	vk::Semaphore           zeroed = nullptr;
	EXIT_IF(graphics.device.createSemaphore(&semaphore_info, nullptr, &zeroed) != vk::Result::eSuccess);
	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence = nullptr;
	EXIT_IF(graphics.device.createFence(&fence_info, nullptr, &fence) != vk::Result::eSuccess);
	vk::SubmitInfo submit {};
	submit.commandBufferCount   = 1;
	submit.pCommandBuffers      = &command;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores    = &zeroed;
	vk::SparseBufferMemoryBindInfo buffer_bind {m_buffer, static_cast<uint32_t>(binds.size()), binds.data()};
	vk::BindSparseInfo             bind_info {};
	bind_info.waitSemaphoreCount = 1;
	bind_info.pWaitSemaphores    = &zeroed;
	bind_info.bufferBindCount    = 1;
	bind_info.pBufferBinds       = &buffer_bind;
	const bool stats      = SubmitStats::Enabled();
	const auto lock_start = stats ? Common::SteadyNs() : 0;
	uint64_t   call_start = 0;
	uint64_t   call_end   = 0;
	{
		Common::LockGuard lock {graphics.queue_mutex};
		call_start = stats ? Common::SteadyNs() : 0;
		EXIT_IF(graphics.queue.submit(1, &submit, nullptr) != vk::Result::eSuccess);
		call_end = stats ? Common::SteadyNs() : 0;
		const auto result = graphics.queue.bindSparse(1, &bind_info, fence);
		if (result != vk::Result::eSuccess) {
			EXIT("Buffer: sparse bind of %zu blocks at 0x%" PRIx64 " failed: %s (block %" PRIu64
			     ", memory size %" PRIu64 ", buffer size %" PRIu64 ")\n",
			     binds.size(), binds.front().resourceOffset, vk::to_string(result).c_str(), m_sparse_block,
			     m_sparse_size, m_size);
		}
	}
	if (stats) {
		SubmitStats::Record({.queue           = SubmitStats::Queue::Sparse,
		                     .command_buffers = submit.commandBufferCount,
		                     .lock_ns         = call_start - lock_start,
		                     .call_ns         = call_end - call_start,
		                     .where           = std::source_location::current()});
	}
	EXIT_IF(graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX) != vk::Result::eSuccess);
	graphics.device.destroyFence(fence, nullptr);
	graphics.device.destroySemaphore(zeroed, nullptr);
	graphics.device.destroyCommandPool(pool, nullptr);
	for (const auto zero_buffer: zero_buffers) {
		graphics.device.destroyBuffer(zero_buffer, nullptr);
	}
}

void Buffer::Create(vk::BufferUsageFlags flags) {
	KYTY_PROFILER_FUNCTION();
	auto&      graphics = *m_graphics;
	const auto usage    = m_usage;
	const auto size     = m_size;
	EXIT_IF(graphics.allocator == nullptr || size == 0);

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = flags;
	const std::array<uint32_t, 2> families {graphics.queue_family, graphics.readback_family};
	if (graphics.readback_family != static_cast<uint32_t>(-1)) {
		buffer_info.sharingMode           = vk::SharingMode::eConcurrent;
		buffer_info.queueFamilyIndexCount = 2;
		buffer_info.pQueueFamilyIndices   = families.data();
	}

	const bool with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags =
	    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | AllocationFlags(usage);
	allocation_info.usage = AllocationUsage(usage);
	allocation_info.preferredFlags = usage == MemoryUsage::DeviceLocal
	                                     ? VkMemoryPropertyFlags {}
	                                     : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VmaAllocationInfo allocation_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	auto              result        = static_cast<vk::Result>(vmaCreateBuffer(
	    graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info), &allocation_info,
	    &native_buffer, &m_allocation, &allocation_result));
	if (result == vk::Result::eErrorOutOfDeviceMemory) {
		LOGF("Buffer allocation ran out of device memory with budget enforcement: size=%" PRIu64
		     ", memory_usage=%u; retrying without the budget limit\n",
		     size, static_cast<unsigned>(usage));
		graphics.LogMemoryBudget();
		allocation_info.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
		result = static_cast<vk::Result>(
		    vmaCreateBuffer(graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info),
		                    &allocation_info, &native_buffer, &m_allocation, &allocation_result));
	}
	m_allocator = graphics.allocator;
	if (result != vk::Result::eSuccess && graphics.spill_allocator != nullptr) {
		VmaAllocationCreateInfo spill_info = allocation_info;
		spill_info.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
		if (usage == MemoryUsage::DeviceLocal) {
			spill_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		}
		result = static_cast<vk::Result>(
		    vmaCreateBuffer(graphics.spill_allocator, static_cast<const VkBufferCreateInfo*>(buffer_info), &spill_info,
		                    &native_buffer, &m_allocation, &allocation_result));
		if (result == vk::Result::eSuccess) {
			m_allocator = graphics.spill_allocator;
			LOGF("Buffer allocation past the heap limit placed by the spill allocator: size=%" PRIu64 "\n", size);
		}
	}
	if (result != vk::Result::eSuccess) {
		graphics.LogMemoryBudget();
		EXIT("Buffer allocation failed: %s, size=%" PRIu64 ", memory_usage=%u, flags=0x%x\n",
		     vk::to_string(result).c_str(), size, static_cast<unsigned>(usage),
		     static_cast<unsigned>(static_cast<VkBufferUsageFlags>(flags)));
	}

	m_buffer = native_buffer;
	if (with_bda) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
	}

	VkMemoryPropertyFlags properties = 0;
	vmaGetAllocationMemoryProperties(m_allocator, m_allocation, &properties);
	m_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
#if defined(ANYPS5_KYTY)
	m_device_local = (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
#endif
	if (allocation_result.pMappedData != nullptr) {
		m_mapped = {static_cast<uint8_t*>(allocation_result.pMappedData),
		            static_cast<size_t>(size)};
	}
}

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t cpu_address, uint64_t size,
               vk::Buffer imported, vk::DeviceAddress device_address)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(MemoryUsage::DeviceLocal), m_cpu_address(cpu_address),
      m_device_address(device_address), m_buffer(imported), m_size(size), m_coherent(true), m_in_place(true) {
	EXIT_IF(imported == nullptr || size == 0 || device_address == 0);
}

Buffer::~Buffer() {
	if (m_buffer == nullptr) {
		return;
	}
	if (m_sparse_block != 0) {
		m_graphics->device.destroyBuffer(m_buffer);
		for (auto* block: m_sparse_blocks) {
			if (block != nullptr) {
				vmaFreeMemory(m_graphics->allocator, block);
			}
		}
		return;
	}
	if (m_in_place) {
		m_graphics->device.destroyBuffer(m_buffer);
	} else {
		vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
	}
}

void Buffer::Flush(uint64_t offset, uint64_t size) {
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaFlushAllocation(m_allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

void Buffer::Invalidate(uint64_t offset, uint64_t size) {
	EXIT_IF(m_usage != MemoryUsage::Download || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaInvalidateAllocation(m_allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

vk::BufferMemoryBarrier Buffer::Barrier(uint64_t offset, uint64_t size, vk::AccessFlags source,
                                        vk::AccessFlags destination) const {
	if (Handle() == nullptr || size == 0 || offset > Size() || size > Size() - offset) {
		EXIT("Buffer: invalid DMA barrier, handle=%p offset=0x%016" PRIx64 " size=0x%016" PRIx64
		     " capacity=0x%016" PRIx64 "\n",
		     static_cast<const void*>(Handle()), offset, size, Size());
	}
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = source;
	barrier.dstAccessMask       = destination;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = Handle();
	barrier.offset              = offset;
	barrier.size                = size;
	return barrier;
}

void Buffer::CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
                      uint64_t destination_offset, uint64_t size) {
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	command.EndRendering();
	const vk::BufferMemoryBarrier before[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eMemoryWrite,
	                   vk::AccessFlagBits::eTransferRead),
	    Barrier(destination_offset, size, MEMORY_ACCESS, vk::AccessFlagBits::eTransferWrite),
	};
	const auto native = command.Handle();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 2, before, 0, nullptr);
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	native.copyBuffer(source.Handle(), Handle(), 1, &copy);
	const vk::BufferMemoryBarrier after[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eTransferRead, MEMORY_ACCESS),
	    Barrier(destination_offset, size, vk::AccessFlagBits::eTransferWrite, MEMORY_ACCESS),
	};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, after, 0, nullptr);
}

void Buffer::Fill(uint64_t offset, uint64_t size, uint32_t value) {
	if (((offset | size) & 3u) != 0) {
		EXIT("Buffer: fill range must be dword aligned\n");
	}
	auto& command = Scheduler().Current();
	command.EndRendering();
	const auto before = Barrier(offset, size, MEMORY_ACCESS, vk::AccessFlagBits::eTransferWrite);
	const auto native = command.Handle();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.fillBuffer(Handle(), offset, size, value);
	const auto after = Barrier(offset, size, vk::AccessFlagBits::eTransferWrite, MEMORY_ACCESS);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

StreamBuffer::StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                           uint64_t size)
    : Buffer(graphics, scheduler, usage, 0,
             AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size),
      m_current_watches(WATCHES_INITIAL_RESERVE), m_previous_watches(WATCHES_INITIAL_RESERVE) {}

bool StreamBuffer::NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
                                        uint64_t& alignment) {
	if (coherent) {
		return true;
	}
	if (!AlignUp(size, atom, size)) {
		return false;
	}
	const auto divisor = std::gcd(alignment, atom);
	if (alignment != 0 && alignment / divisor > UINT64_MAX / atom) {
		return false;
	}
	alignment = alignment == 0 ? atom : alignment / divisor * atom;
	return true;
}

std::pair<uint8_t*, uint64_t> StreamBuffer::Map(uint64_t size, uint64_t alignment,
                                                bool allow_wait) {
	if (Mapped().empty()) {
		return {nullptr, 0};
	}
	uint64_t   mapped_size = size;
	const auto atom        = Graphics().physical_device_properties.limits.nonCoherentAtomSize;
	if (!NormalizeReservation(IsCoherent(), atom, mapped_size, alignment)) {
		return {nullptr, 0};
	}
	if (mapped_size > Size()) {
		return {nullptr, 0};
	}

	uint64_t aligned_offset = 0;
	if (!AlignUp(m_offset, alignment, aligned_offset)) {
		return {nullptr, 0};
	}

	const bool wrap = aligned_offset > Size() - mapped_size;
	if (wrap) {
		aligned_offset = 0;
	}

	auto wait_cursor = wrap ? size_t {0} : m_wait_cursor;
	auto wait_bound  = wrap ? uint64_t {0} : m_wait_bound;
	auto invalidation_mark =
	    wrap ? std::optional<size_t> {m_current_watch_cursor} : m_invalidation_mark;
	auto& pending_watches = wrap ? m_current_watches : m_previous_watches;
	if (!WaitPendingOperations(pending_watches, invalidation_mark, aligned_offset + mapped_size,
	                           allow_wait, wait_cursor, wait_bound)) {
		return {nullptr, 0};
	}

	if (wrap) {
		m_invalidation_mark    = invalidation_mark;
		m_current_watch_cursor = 0;
		std::swap(m_previous_watches, m_current_watches);
		m_lap++;
	}
	m_wait_cursor = wait_cursor;
	m_wait_bound  = wait_bound;
	m_offset      = aligned_offset;
	m_mapped_size = mapped_size;
	return {Mapped().data() + m_offset, m_offset};
}

void StreamBuffer::Commit() {
	if (Usage() != MemoryUsage::Download && m_mapped_size != 0) {
		Flush(m_offset, m_mapped_size);
	}

	m_offset += m_mapped_size;
	const auto tick = Scheduler().CurrentTick();
	if (m_current_watch_cursor != 0 && m_current_watches[m_current_watch_cursor - 1].tick == tick) {
		m_current_watches[m_current_watch_cursor - 1].upper_bound = m_offset;
		return;
	}
	if (m_current_watch_cursor + 1 >= m_current_watches.size()) {
		m_current_watches.resize(m_current_watches.size() + WATCHES_RESERVE_CHUNK);
	}
	auto& watch       = m_current_watches[m_current_watch_cursor++];
	watch.upper_bound = m_offset;
	watch.tick        = tick;
}

StreamBuffer::Download StreamBuffer::MapDownload(uint64_t size, uint64_t alignment) {
	if (const auto [mapped, offset] = Map(size, alignment); mapped != nullptr) {
		Commit();
		return {this, mapped, offset, nullptr};
	}
	auto owner = std::make_unique<Buffer>(Graphics(), Scheduler(), Usage(), 0, AllFlags, size);
	return {owner.get(), owner->Mapped().data(), 0, std::move(owner)};
}

uint64_t StreamBuffer::Copy(const void* source, uint64_t size, uint64_t alignment) {
	EXIT_IF(source == nullptr);
	const auto [data, offset] = Map(size, alignment);
	EXIT_IF(data == nullptr);
	std::memcpy(data, source, static_cast<size_t>(size));
	Commit();
	return offset;
}

bool StreamBuffer::WaitPendingOperations(const std::vector<Watch>& watches,
                                         std::optional<size_t>     invalidation_mark,
                                         uint64_t requested_upper_bound, bool allow_wait,
                                         size_t& wait_cursor, uint64_t& wait_bound) {
	if (!invalidation_mark.has_value()) {
		return true;
	}
	while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
		const auto& watch = watches[wait_cursor];
		if (!Scheduler().IsFree(watch.tick) && !allow_wait) {
			return false;
		}
		Scheduler().Wait(watch.tick);
		if (Usage() == MemoryUsage::Download) {
			Scheduler().WaitPriorityOperations(watch.tick);
		}
		wait_bound = watch.upper_bound;
		++wait_cursor;
	}
	return true;
}

}

#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

namespace {

struct RecycledImage {
	vk::ImageCreateInfo create;
	VkImage             image      = VK_NULL_HANDLE;
	VmaAllocation       allocation = nullptr;
	uint64_t            size       = 0;
};

constexpr size_t   RecycledImagesMax     = 32;
constexpr uint64_t RecycledImageBytesMax = 256ull * 1024 * 1024;

std::mutex                 g_recycled_mutex;
std::vector<RecycledImage> g_recycled;
uint64_t                   g_recycled_bytes = 0;

bool Recyclable(const vk::ImageCreateInfo& create) {
	return create.pNext == nullptr && create.tiling == vk::ImageTiling::eOptimal &&
	       create.sharingMode == vk::SharingMode::eExclusive &&
	       create.initialLayout == vk::ImageLayout::eUndefined;
}

bool SameCreateInfo(const vk::ImageCreateInfo& a, const vk::ImageCreateInfo& b) {
	return a.flags == b.flags && a.imageType == b.imageType && a.format == b.format &&
	       a.extent == b.extent && a.mipLevels == b.mipLevels && a.arrayLayers == b.arrayLayers &&
	       a.samples == b.samples && a.usage == b.usage;
}

class ImageGraveyard {
public:
	ImageGraveyard() = default;
	~ImageGraveyard() {
		Flush();
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_one();
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}
	ImageGraveyard(const ImageGraveyard&)            = delete;
	ImageGraveyard& operator=(const ImageGraveyard&) = delete;

	void Push(VmaAllocator allocator, VkImage image, VmaAllocation allocation) {
		{
			std::lock_guard lock(m_mutex);
			if (!m_thread.joinable()) {
				m_thread = std::thread([this] { Run(); });
			}
			m_items.push_back({allocator, image, allocation});
		}
		m_wake.notify_one();
	}

	void Flush() {
		std::unique_lock lock(m_mutex);
		m_idle.wait(lock, [this] { return m_items.empty() && !m_busy; });
	}

private:
	struct Item {
		VmaAllocator  allocator;
		VkImage       image;
		VmaAllocation allocation;
	};

	void Run() {
		std::vector<Item> batch;
		std::unique_lock  lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [this] { return m_stop || !m_items.empty(); });
			if (m_items.empty()) {
				return;
			}
			batch.swap(m_items);
			m_busy = true;
			lock.unlock();
			for (const auto& item: batch) {
				vmaDestroyImage(item.allocator, item.image, item.allocation);
			}
			batch.clear();
			lock.lock();
			m_busy = false;
			m_idle.notify_all();
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_wake;
	std::condition_variable m_idle;
	std::vector<Item>       m_items;
	bool                    m_busy = false;
	bool                    m_stop = false;
	std::thread             m_thread;
};

ImageGraveyard& Graveyard() {
	static ImageGraveyard graveyard;
	return graveyard;
}

void DestroyRecycledImages(VmaAllocator allocator) {
	Graveyard().Flush();
	std::lock_guard lock(g_recycled_mutex);
	for (const auto& entry: g_recycled) {
		vmaDestroyImage(allocator, entry.image, entry.allocation);
	}
	g_recycled.clear();
	g_recycled_bytes = 0;
}

}

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}
	info.preferredLargeHeapBlockSize = 64ull * 1024 * 1024;

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	DestroyRecycledImages(allocator);
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
	std::lock_guard lock(m_sparse_mutex);
	m_sparse_pending.clear();
	if (m_sparse_semaphore != nullptr) {
		device.destroySemaphore(m_sparse_semaphore, nullptr);
		m_sparse_semaphore = nullptr;
	}
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = physical_device_memory_properties;
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

static uint64_t AllocatorReserve(uint64_t budget) {
	return std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
}

GraphicContext::DeviceLocalBudget GraphicContext::GetDeviceLocalBudget() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return {};
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	DeviceLocalBudget result;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			result.usage += budgets[heap].usage;
			result.budget += budgets[heap].budget;
		}
	}
	result.reserve = AllocatorReserve(result.budget);
	return result;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - AllocatorReserve(budget);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	if (Recyclable(image_info)) {
		std::lock_guard lock(g_recycled_mutex);
		const auto      found =
		    std::find_if(g_recycled.begin(), g_recycled.end(), [&](const auto& entry) {
			    return SameCreateInfo(entry.create, image_info);
		    });
		if (found != g_recycled.end()) {
			native_image     = found->image;
			image.allocation = found->allocation;
			g_recycled_bytes -= found->size;
			g_recycled.erase(found);
		}
	}
	if (native_image == VK_NULL_HANDLE) {
		auto result = static_cast<vk::Result>(vmaCreateImage(
		    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
		    &native_image, &image.allocation, nullptr));
		if (result != vk::Result::eSuccess) {
			DestroyRecycledImages(allocator);
			result = static_cast<vk::Result>(vmaCreateImage(
			    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
			    &alloc_info, &native_image, &image.allocation, nullptr));
		}
		if (result != vk::Result::eSuccess) {
			image.image = native_image;
			LogMemoryBudget();
			return false;
		}
	}
	image.image = native_image;

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.recyclable = Recyclable(image_info);
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

bool GraphicContext::QueueLevelBinds(VulkanImage& image, uint32_t first_level, uint32_t end_level,
                                     bool bind_tail, bool within_budget, VmaAllocation& allocation) {
	const auto requirements = device.getImageMemoryRequirements(image.image);
	const auto sparse       = device.getImageSparseMemoryRequirements(image.image);
	const vk::SparseImageMemoryRequirements* color = nullptr;
	for (const auto& entry: sparse) {
		if (entry.formatProperties.aspectMask & vk::ImageAspectFlagBits::eMetadata) {
			return false;
		}
		if (entry.formatProperties.aspectMask & vk::ImageAspectFlagBits::eColor) {
			color = &entry;
		}
	}
	if (color == nullptr) {
		return false;
	}
	const auto  block       = requirements.alignment;
	const auto& granularity = color->formatProperties.imageGranularity;
	const bool  single_tail = static_cast<bool>(color->formatProperties.flags &
	                                           vk::SparseImageFormatFlagBits::eSingleMiptail);
	const auto  tail_first  = std::min(color->imageMipTailFirstLod, image.mip_levels);
	SparseBind  request {.image = image.image};
	uint64_t    total = 0;
	for (uint32_t layer = 0; layer < image.layers; layer++) {
		for (uint32_t level = first_level; level < std::min(end_level, tail_first); level++) {
			const vk::Extent3D extent {std::max(1u, image.extent.width >> level),
			                           std::max(1u, image.extent.height >> level), 1u};
			const uint64_t     blocks =
			    static_cast<uint64_t>((extent.width + granularity.width - 1) / granularity.width) *
			    ((extent.height + granularity.height - 1) / granularity.height);
			vk::SparseImageMemoryBind bind {};
			bind.subresource  = {vk::ImageAspectFlagBits::eColor, level, layer};
			bind.extent       = extent;
			bind.memoryOffset = total;
			request.levels.push_back(bind);
			total += blocks * block;
		}
	}
	if (bind_tail && tail_first < image.mip_levels) {
		const uint32_t tails     = single_tail ? 1u : image.layers;
		const uint64_t tail_size = Common::AlignUp(color->imageMipTailSize, block);
		for (uint32_t tail = 0; tail < tails; tail++) {
			vk::SparseMemoryBind bind {};
			bind.resourceOffset = color->imageMipTailOffset + tail * color->imageMipTailStride;
			bind.size           = color->imageMipTailSize;
			bind.memoryOffset   = total;
			request.tails.push_back(bind);
			total += tail_size;
		}
	}
	if (total == 0) {
		return false;
	}

	VkMemoryRequirements memory {};
	memory.size           = total;
	memory.alignment      = block;
	memory.memoryTypeBits = requirements.memoryTypeBits;
	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags         = within_budget ? VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT : 0;
	alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VmaAllocationInfo allocated {};
	if (vmaAllocateMemory(allocator, &memory, &alloc_info, &allocation, &allocated) != VK_SUCCESS) {
		allocation = nullptr;
		return false;
	}
	for (auto& bind: request.levels) {
		bind.memory = allocated.deviceMemory;
		bind.memoryOffset += allocated.offset;
	}
	for (auto& bind: request.tails) {
		bind.memory = allocated.deviceMemory;
		bind.memoryOffset += allocated.offset;
	}
	std::lock_guard lock(m_sparse_mutex);
	m_sparse_pending.push_back(std::move(request));
	return true;
}

bool GraphicContext::CreatePartialImage(const vk::ImageCreateInfo& image_info, uint32_t first_level,
                                        VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);
	if (!partial_textures_enabled || first_level == 0 || first_level >= image_info.mipLevels ||
	    image_info.imageType != vk::ImageType::e2D || image_info.samples != vk::SampleCountFlagBits::e1 ||
	    image_info.tiling != vk::ImageTiling::eOptimal) {
		return false;
	}
	auto create = image_info;
	create.flags |= vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency;
	const auto sparse_formats = physical_device.getSparseImageFormatProperties(
	    create.format, create.imageType, create.samples, create.usage, create.tiling);
	if (sparse_formats.empty()) {
		return false;
	}
	vk::Image native_image = nullptr;
	if (device.createImage(&create, nullptr, &native_image) != vk::Result::eSuccess) {
		return false;
	}
	const auto sparse = device.getImageSparseMemoryRequirements(native_image);
	const auto color  = std::ranges::find_if(sparse, [](const auto& entry) {
		return static_cast<bool>(entry.formatProperties.aspectMask & vk::ImageAspectFlagBits::eColor);
	});
	if (color == sparse.end() || color->imageMipTailFirstLod <= first_level) {
		device.destroyImage(native_image, nullptr);
		return false;
	}

	image.image                = native_image;
	image.format               = create.format;
	image.image_type           = create.imageType;
	image.extent               = create.extent;
	image.layers               = create.arrayLayers;
	image.mip_levels           = create.mipLevels;
	image.samples              = static_cast<uint32_t>(create.samples);
	image.usage                = create.usage;
	image.recyclable           = false;
	image.flags                = create.flags;
	image.first_resident_level = first_level;
	image.state                = {.layout = create.initialLayout};
	image.subresource_states.clear();
	if (!QueueLevelBinds(image, first_level, create.mipLevels, true, true, image.allocation)) {
		device.destroyImage(native_image, nullptr);
		image.image                = nullptr;
		image.allocation           = nullptr;
		image.first_resident_level = 0;
		return false;
	}
	return true;
}

bool GraphicContext::BindOmittedLevels(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	if (image.first_resident_level == 0) {
		return true;
	}
	EXIT_IF(image.omitted_allocation != nullptr);
	if (!QueueLevelBinds(image, 0, image.first_resident_level, false, false,
	                     image.omitted_allocation)) {
		return false;
	}
	image.first_resident_level = 0;
	return true;
}

uint64_t GraphicContext::FlushSparseBinds() {
	std::lock_guard lock(m_sparse_mutex);
	if (m_sparse_pending.empty()) {
		return m_sparse_value;
	}
	KYTY_PROFILER_FUNCTION();
	if (m_sparse_semaphore == nullptr) {
		vk::SemaphoreTypeCreateInfo type {};
		type.semaphoreType = vk::SemaphoreType::eTimeline;
		vk::SemaphoreCreateInfo create {};
		create.pNext = &type;
		if (device.createSemaphore(&create, nullptr, &m_sparse_semaphore) != vk::Result::eSuccess) {
			EXIT("could not create the sparse bind semaphore\n");
		}
	}
	std::vector<vk::SparseImageMemoryBindInfo>       levels;
	std::vector<vk::SparseImageOpaqueMemoryBindInfo> tails;
	levels.reserve(m_sparse_pending.size());
	tails.reserve(m_sparse_pending.size());
	for (const auto& request: m_sparse_pending) {
		if (!request.levels.empty()) {
			levels.push_back({request.image, static_cast<uint32_t>(request.levels.size()),
			                  request.levels.data()});
		}
		if (!request.tails.empty()) {
			tails.push_back({request.image, static_cast<uint32_t>(request.tails.size()),
			                 request.tails.data()});
		}
	}
	const uint64_t                  value = m_sparse_value + 1;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.signalSemaphoreValueCount = 1;
	timeline.pSignalSemaphoreValues    = &value;
	vk::BindSparseInfo info {};
	info.pNext                = &timeline;
	info.imageBindCount       = static_cast<uint32_t>(levels.size());
	info.pImageBinds          = levels.data();
	info.imageOpaqueBindCount = static_cast<uint32_t>(tails.size());
	info.pImageOpaqueBinds    = tails.data();
	info.signalSemaphoreCount = 1;
	info.pSignalSemaphores    = &m_sparse_semaphore;
	vk::Result result;
	{
		Common::LockGuard queue_lock {queue_mutex};
		result = queue.bindSparse(1, &info, nullptr);
	}
	if (result != vk::Result::eSuccess) {
		EXIT("vkQueueBindSparse failed: %s\n", vk::to_string(result).c_str());
	}
	m_sparse_value = value;
	m_sparse_pending.clear();
	return value;
}

void GraphicContext::WaitSparseBinds(vk::Image image) {
	uint64_t value = 0;
	{
		std::lock_guard lock(m_sparse_mutex);
		std::erase_if(m_sparse_pending,
		              [image](const SparseBind& request) { return request.image == image; });
		value = m_sparse_value;
	}
	if (value == 0) {
		return;
	}
	vk::SemaphoreWaitInfo wait {};
	wait.semaphoreCount = 1;
	wait.pSemaphores    = &m_sparse_semaphore;
	wait.pValues        = &value;
	if (device.waitSemaphores(&wait, UINT64_MAX) != vk::Result::eSuccess) {
		EXIT("waiting for the sparse binds failed\n");
	}
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);
	if (image.flags & vk::ImageCreateFlagBits::eSparseBinding) {
		WaitSparseBinds(image.image);
		if (image.omitted_allocation != nullptr) {
			Graveyard().Push(allocator, VK_NULL_HANDLE, image.omitted_allocation);
			image.omitted_allocation = nullptr;
		}
	}

	vk::ImageCreateInfo create {};
	create.flags         = image.flags;
	create.imageType     = image.image_type;
	create.format        = image.format;
	create.extent        = image.extent;
	create.mipLevels     = image.mip_levels;
	create.arrayLayers   = image.layers;
	create.samples       = static_cast<vk::SampleCountFlagBits>(image.samples);
	create.tiling        = vk::ImageTiling::eOptimal;
	create.usage         = image.usage;
	create.sharingMode   = vk::SharingMode::eExclusive;
	create.initialLayout = vk::ImageLayout::eUndefined;
	if (image.recyclable) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		std::lock_guard lock(g_recycled_mutex);
		g_recycled.push_back({create, image.image, image.allocation, allocation_info.size});
		g_recycled_bytes += allocation_info.size;
		while (g_recycled.size() > RecycledImagesMax || g_recycled_bytes > RecycledImageBytesMax) {
			const auto& oldest = g_recycled.front();
			Graveyard().Push(allocator, oldest.image, oldest.allocation);
			g_recycled_bytes -= oldest.size;
			g_recycled.erase(g_recycled.begin());
		}
	} else {
		Graveyard().Push(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
}

}

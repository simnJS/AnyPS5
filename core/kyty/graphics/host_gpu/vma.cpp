#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

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

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <string_view>
#include <charconv>
#include <atomic>
#include <array>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

namespace {

struct RecycledImage {
	vk::ImageCreateInfo create;
	VkImage             image      = VK_NULL_HANDLE;
	VmaAllocation       allocation = nullptr;
	MemoryTelemetry::Allocation charge;
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

	void Push(VmaAllocator allocator, VkImage image, VmaAllocation allocation, MemoryTelemetry::Allocation charge) {
		{
			std::lock_guard lock(m_mutex);
			if (!m_thread.joinable()) {
				m_thread = std::thread([this] { Run(); });
			}
			m_items.push_back({allocator, image, allocation, charge});
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
		MemoryTelemetry::Allocation charge;
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
			for (auto& item: batch) {
				vmaDestroyImage(item.allocator, item.image, item.allocation);
				item.charge.Release();
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
	for (auto& entry: g_recycled) {
		vmaDestroyImage(allocator, entry.image, entry.allocation);
		entry.charge.Release();
	}
	g_recycled.clear();
	g_recycled_bytes = 0;
}

}

static uint64_t MemoryBudgetCap() {
	static const uint64_t cap = []() -> uint64_t {
		const char* value = std::getenv("KYTY_MEMORY_BUDGET_MB");
		if (value == nullptr || value[0] == '\0') {
			return UINT64_MAX;
		}
		const std::string_view text(value);
		uint64_t               mib = 0;
		const auto [end, error]    = std::from_chars(text.data(), text.data() + text.size(), mib, 10);
		if (error != std::errc() || end != text.data() + text.size() || mib == 0 || mib > (UINT64_MAX >> 20u)) {
			std::fprintf(stderr, "KYTY_MEMORY_BUDGET_MB: '%s' is not a positive number of MiB; the automatic limit applies\n",
			             value);
			return UINT64_MAX;
		}
		return mib << 20u;
	}();
	return cap;
}

static std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> HeapSizeLimits(const GraphicContext& graphics) {
	constexpr uint64_t DesktopReserve = 3072ull << 20u;
	constexpr uint64_t LargeHeap      = 4ull << 30u;
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS> limits {};
	limits.fill(VK_WHOLE_SIZE);
	if (graphics.physical_device_properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu) {
		return limits;
	}
	vk::PhysicalDeviceMemoryBudgetPropertiesEXT budgets {};
	vk::PhysicalDeviceMemoryProperties2         memory {};
	if (graphics.memory_budget_ext_enabled) {
		memory.pNext = &budgets;
	}
	graphics.physical_device.getMemoryProperties2(&memory);
	const auto& heaps = memory.memoryProperties;
	for (uint32_t heap = 0; heap < heaps.memoryHeapCount; heap++) {
		const uint64_t size = heaps.memoryHeaps[heap].size;
		if (!(heaps.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) || size < LargeHeap) {
			continue;
		}
		const uint64_t available = graphics.memory_budget_ext_enabled && budgets.heapBudget[heap] != 0
		                               ? std::min<uint64_t>(budgets.heapBudget[heap], size)
		                               : size;
		const uint64_t forced  = MemoryBudgetCap();
		const uint64_t reserve = std::min(DesktopReserve, available / 4);
		const uint64_t limit   = forced != UINT64_MAX ? std::min(forced, size) : available - reserve;
		limits[heap] = limit;
		std::fprintf(stderr, "Vulkan memory: device-local heap %u of %" PRIu64 " MiB limited to %" PRIu64 " MiB (%s)\n",
		             heap, size >> 20u, limit >> 20u,
		             forced != UINT64_MAX ? "KYTY_MEMORY_BUDGET_MB" : "budget less the desktop reserve");
	}
	return limits;
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
	const auto heap_limits = HeapSizeLimits(*this);
	info.pHeapSizeLimit    = heap_limits.data();

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	if (std::any_of(heap_limits.begin(), heap_limits.end(), [](VkDeviceSize limit) { return limit != VK_WHOLE_SIZE; })) {
		info.pHeapSizeLimit = nullptr;
		const auto spill    = static_cast<vk::Result>(vmaCreateAllocator(&info, &spill_allocator));
		if (spill != vk::Result::eSuccess) {
			std::fprintf(stderr, "Vulkan memory: no spill allocator (%s): an allocation past the heap limits fails\n",
			             vk::to_string(spill).c_str());
			spill_allocator = nullptr;
		}
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (GpuBreadcrumbs::Requested()) GpuBreadcrumbs::Shutdown(*this);
	if (allocator == nullptr) {
		return;
	}
	DestroyRecycledImages(allocator);
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
	if (spill_allocator != nullptr) {
		vmaDestroyAllocator(spill_allocator);
		spill_allocator = nullptr;
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
	if (m_memory_override_enabled) {
		return m_memory_usage_override;
	}
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

GraphicContext::HeapUsage GraphicContext::GetLocalMemoryUsage() const {
	if (m_memory_override_enabled) {
		return {.usage = m_memory_usage_override, .budget = m_memory_budget_override};
	}
	HeapUsage local {};
	if (memory_budget_ext_enabled) {
		vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget {};
		vk::PhysicalDeviceMemoryProperties2 properties {.pNext = &budget};
		physical_device.getMemoryProperties2(&properties);
		for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
			if (properties.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				local.usage += budget.heapUsage[i];
				local.budget += budget.heapBudget[i];
			}
		}
		local.budget = std::min(local.budget, MemoryBudgetCap());
	}
	return local;
}

uint64_t GraphicContext::LocalBlockSpace() const {
	if (m_memory_override_enabled) {
		return 0;
	}
	uint64_t space = 0;
	for (VmaAllocator vma: {allocator, spill_allocator}) {
		if (vma == nullptr) {
			continue;
		}
		VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
		vmaGetHeapBudgets(vma, budgets);
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			const auto& statistics = budgets[heap].statistics;
			if ((physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) &&
			    statistics.blockBytes > statistics.allocationBytes) {
				space += statistics.blockBytes - statistics.allocationBytes;
			}
		}
	}
	return space;
}

bool GraphicContext::LocalMemoryCrowded() const {
	const auto space = LocalBlockSpace();
	if (!m_memory_override_enabled && allocator != nullptr) {
		VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
		vmaGetHeapBudgets(allocator, budgets);
		uint64_t usage  = 0;
		uint64_t budget = 0;
		for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
			if (physical_device_memory_properties.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				usage += budgets[heap].usage;
				budget += budgets[heap].budget;
			}
		}
		if (usage - std::min(usage, space) < HeapImageThreshold(budget)) {
			return false;
		}
	}
	const auto local = GetLocalMemoryUsage();
	return local.budget != 0 && local.usage - std::min(local.usage, space) >= HeapImageThreshold(local.budget);
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (m_memory_override_enabled) {
		return m_memory_budget_override;
	}
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
		return std::min(budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024), MemoryBudgetCap());
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::min(std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0}),
	                MemoryBudgetCap());
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	return CreateImage(image_info, image, ImageAllocation::PreferDevice);
}

static uint32_t TextureSpillTypes(const GraphicContext& graphics) {
	if (graphics.spill_allocator == nullptr) {
		return 0;
	}
	const VkPhysicalDeviceMemoryProperties* limited = nullptr;
	vmaGetMemoryProperties(graphics.allocator, &limited);
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(graphics.allocator, budgets);
	const auto& physical = graphics.physical_device_memory_properties;
	bool pressure = false;
	for (uint32_t heap = 0; heap < limited->memoryHeapCount; ++heap) {
		const auto limit = limited->memoryHeaps[heap].size;
		if (limit < physical.memoryHeaps[heap].size &&
		    budgets[heap].statistics.blockBytes >= limit - limit / 4) {
			pressure = true;
			break;
		}
	}
	if (!pressure && !graphics.LocalMemoryCrowded()) {
		return 0;
	}
	uint32_t types = 0;
	for (uint32_t type = 0; type < limited->memoryTypeCount; ++type) {
		const auto heap = limited->memoryTypes[type].heapIndex;
		if (!(limited->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
			types |= uint32_t {1} << type;
		}
	}
	return types;
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image,
                                 ImageAllocation allocation) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	const bool heap_texture = allocation == ImageAllocation::HeapTexture;
	const auto category = (image_info.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment)
	                          ? MemoryTelemetry::Kind::Depth
	                          : heap_texture ? MemoryTelemetry::Kind::HeapTexture : MemoryTelemetry::Kind::Texture;
	if (Recyclable(image_info)) {
		std::lock_guard lock(g_recycled_mutex);
		const auto      found =
		    std::find_if(g_recycled.begin(), g_recycled.end(), [&](const auto& entry) {
			    return SameCreateInfo(entry.create, image_info);
		    });
		if (found != g_recycled.end()) {
			native_image     = found->image;
			image.allocation = found->allocation;
			image.charge = found->charge;
			image.charge.Reclassify(category);
			g_recycled_bytes -= found->charge.bytes;
			g_recycled.erase(found);
		}
	}
	if (native_image == VK_NULL_HANDLE) {
		VmaAllocationInfo info {};
		const auto* create = static_cast<const vk::ImageCreateInfo::NativeType*>(image_info);
		auto result = vk::Result::eErrorOutOfDeviceMemory;
		const bool texture = heap_texture ||
		                     (allocation == ImageAllocation::Texture && category != MemoryTelemetry::Kind::Depth);
		if (const uint32_t types = texture ? TextureSpillTypes(*this) : 0; types != 0) {
			VmaAllocationCreateInfo spill_info {};
			spill_info.flags          = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
			spill_info.usage          = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			spill_info.memoryTypeBits = types;
			result = static_cast<vk::Result>(vmaCreateImage(
			    spill_allocator, create, &spill_info, &native_image, &image.allocation, &info));
			image.spilled = result == vk::Result::eSuccess;
		}
		if (result != vk::Result::eSuccess) {
			result = static_cast<vk::Result>(vmaCreateImage(
			    allocator, create, &alloc_info, &native_image, &image.allocation, &info));
		}
		if (result != vk::Result::eSuccess) {
			DestroyRecycledImages(allocator);
			result = static_cast<vk::Result>(vmaCreateImage(
			    allocator, create, &alloc_info, &native_image, &image.allocation, &info));
		}
		if (result != vk::Result::eSuccess && spill_allocator != nullptr) {
			VmaAllocationCreateInfo spill_info {};
			spill_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			result           = static_cast<vk::Result>(vmaCreateImage(
			    spill_allocator, create, &spill_info, &native_image, &image.allocation, &info));
			if (result == vk::Result::eSuccess) {
				image.spilled = true;
				static std::atomic<uint32_t> reported {0};
				if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
					LOGF("Vulkan memory: an image of %ux%u past the heap limit is placed by the spill allocator\n",
					     image_info.extent.width, image_info.extent.height);
				}
			}
		}
		if (result != vk::Result::eSuccess && spill_allocator != nullptr) {
			VmaAllocationCreateInfo spill_info {};
			spill_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
			result           = static_cast<vk::Result>(vmaCreateImage(
                spill_allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &spill_info,
                &native_image, &image.allocation, &info));
			if (result == vk::Result::eSuccess) {
				image.spilled = true;
				static std::atomic<uint32_t> reported {0};
				if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
					LOGF("Vulkan memory: an image of %ux%u past the heap limit is placed by the spill allocator\n",
					     image_info.extent.width, image_info.extent.height);
				}
			}
		}
		if (result != vk::Result::eSuccess) {
			image.image = native_image;
			LogMemoryBudget();
			return false;
		}
		image.charge.Add(category, IsLocalMemoryType(info.memoryType), info.size);
	}
	image.image = native_image;

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.recyclable = Recyclable(image_info) && !image.spilled;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

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
		std::lock_guard lock(g_recycled_mutex);
		g_recycled.push_back({create, image.image, image.allocation, image.charge});
		g_recycled_bytes += image.charge.bytes;
		while (g_recycled.size() > RecycledImagesMax || g_recycled_bytes > RecycledImageBytesMax) {
			const auto& oldest = g_recycled.front();
			Graveyard().Push(allocator, oldest.image, oldest.allocation, oldest.charge);
			g_recycled_bytes -= oldest.charge.bytes;
			g_recycled.erase(g_recycled.begin());
		}
	} else {
		Graveyard().Push(image.spilled ? spill_allocator : allocator, image.image, image.allocation, image.charge);
	}
	image.charge = {};
	image.image      = nullptr;
	image.allocation = nullptr;
	image.spilled    = false;
}

void GraphicContext::TrimRecycledImages() {
	if (allocator == nullptr) {
		return;
	}
	DestroyRecycledImages(allocator);
}

size_t GraphicContext::RecycledImageCount() const {
	std::lock_guard lock(g_recycled_mutex);
	return g_recycled.size();
}

}

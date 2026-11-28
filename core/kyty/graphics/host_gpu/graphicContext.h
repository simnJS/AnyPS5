#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include "graphics/host_gpu/driverMemoryReport.h"
#include "graphics/host_gpu/memoryTelemetry.h"
#include "graphics/host_gpu/localMemoryBudget.h"

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

enum class ImageAllocation : uint8_t { PreferDevice, Texture, HeapTexture };

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

[[nodiscard]] constexpr bool NeedsMeshLdsPadding(uint32_t vendor, uint32_t device, vk::DriverId driver) {
	return vendor == 0x1002u && device == 0x73bfu && driver == vk::DriverId::eAmdProprietary;
}

struct GraphicContext {
	friend struct ImageHeapReadsTestAccess;
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::DriverId                       driver_id                             = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	VmaAllocator                       spill_allocator                       = nullptr;
	std::unique_ptr<DriverMemoryReport> driver_memory_report;
	bool                               memory_budget_ext_enabled             = false;
	bool                               external_memory_host_enabled          = false;
	uint64_t                           host_pointer_alignment                = 0;
	bool                               compute_subgroup_size_control_enabled = false;
	bool                               shader_image_int64_atomics_enabled    = false;
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	bool                               present_fences_enabled                = false;
	bool                               checkpoints_enabled                   = false;
	bool                               buffer_marker_enabled                 = false;
	bool                               unified_image_layouts_enabled         = false;
	bool                               maintenance8_enabled                  = false;
	bool                               shader_resource_min_lod_enabled       = false;
	bool                               robust_buffer_access_enabled          = false;
	bool                               ray_query_enabled                     = false;
	bool                               compute_derivative_quads_enabled      = false;
	bool                               compute_derivative_linear_enabled     = false;
	bool                               r8_srgb_views                         = false;
	bool                               r8g8_srgb_views                       = false;
	bool                               r4g4b4a4_render_targets               = false;
	bool                               conditional_rendering_enabled         = false;
	bool                               sampler_filter_minmax_enabled         = false;
	bool                               occlusion_queries_enabled             = false;
	bool                               occlusion_query_precise_enabled       = false;
	bool                               indirect_draws_enabled                = false;
#if defined(ANYPS5_KYTY)
	bool                               fragment_shader_pixel_interlock_enabled = false;
#endif
	bool                               supports_block_texel_view              = false;
	bool                               pipeline_cache_lookup_enabled          = false;
#if defined(ANYPS5_KYTY)
	bool                               pipeline_library_enabled               = false;
	bool                               pipeline_cache_persistent              = true;
	bool                               compute_pipeline_binaries              = false;
	std::filesystem::path              compute_pipeline_binary_directory;
#endif
	bool                                      mesh_shader_enabled                   = false;
	bool                                      shader_float16_enabled                = false;
	vk::ShaderStageFlags                      subgroup_supported_stages             = {};
	vk::SubgroupFeatureFlags                  subgroup_supported_operations         = {};
	bool                                      native_ray_tracing_enabled            = false;
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	PFN_vkGetShaderInfoAMD                    fn_get_shader_info                    = nullptr;
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	uint32_t                           device_clock_period_ps                = 0;
	vk::DescriptorSetLayout            image_heap_layout                     = nullptr;
	Common::Mutex                      queue_mutex;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;
	vk::Queue                          readback_queue  = nullptr;
	uint32_t                           readback_family = static_cast<uint32_t>(-1);
	bool                               sparse_buffers_enabled = false;
	vk::Queue                          sparse_queue           = nullptr;
	uint32_t                           sparse_family          = static_cast<uint32_t>(-1);
	uint32_t                           sparse_queue_index     = 0;
	uint64_t                           mirror_base            = 0;
	uint64_t                           mirror_size            = 0;

	[[nodiscard]] bool IsLocalMemoryType(uint32_t type) const {
		const auto& memory = physical_device_memory_properties;
		const auto heap = memory.memoryTypes[type].heapIndex;
		return bool(memory.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
	}

	[[nodiscard]] const vk::PhysicalDeviceProperties& GetPhysicalDeviceProperties() const {
		return physical_device_properties;
	}

	[[nodiscard]] vk::FormatProperties GetFormatProperties(vk::Format format) const {
		std::scoped_lock lock(m_format_properties_mutex);
		auto [it, inserted] = m_format_properties.try_emplace(format);
		if (inserted) {
			physical_device.getFormatProperties(format, &it->second);
		}
		return it->second;
	}

	[[nodiscard]] bool SamplesFiltered(vk::Format format) const {
		constexpr vk::FormatFeatureFlags features =
		    vk::FormatFeatureFlagBits::eSampledImage |
		    vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
		return (GetFormatProperties(format).optimalTilingFeatures & features) == features;
	}

	[[nodiscard]] vk::Result GetImageFormatProperties(vk::Format format, vk::ImageType type,
	                                                  vk::ImageTiling            tiling,
	                                                  vk::ImageUsageFlags        usage,
	                                                  vk::ImageCreateFlags       flags,
	                                                  vk::ImageFormatProperties* properties) const {
		using Key = std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
		                       vk::ImageCreateFlags>;
		std::scoped_lock lock(m_image_format_properties_mutex);
		auto [it, inserted] =
		    m_image_format_properties.try_emplace(Key {format, type, tiling, usage, flags});
		if (inserted) {
			it->second.first = physical_device.getImageFormatProperties(format, type, tiling, usage,
			                                                            flags, &it->second.second);
		}
		if (properties != nullptr) {
			*properties = it->second.second;
		}
		return it->second.first;
	}

	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
		return subgroup_size == 64u || compute_subgroup_size_control_enabled;
	}

	[[nodiscard]] vk::DeviceSize StorageMinAlignment() const {
		const auto alignment = physical_device_properties.limits.minStorageBufferOffsetAlignment;
		return alignment != 0 ? alignment : 1;
	}

	[[nodiscard]] bool CreateAllocator();
	void               DestroyAllocator();
	void               LogMemoryBudget() const;
	void               LogMemoryMinute(const char* reason, uint64_t seconds, bool pressure_only = false) const;
	void               StartMemoryMinutes();
	void               StopMemoryMinutes();
	[[nodiscard]] bool CanReportMemoryUsage() const noexcept {
		return memory_budget_ext_enabled || m_memory_override_enabled;
	}
	[[nodiscard]] uint64_t GetDeviceMemoryUsage() const;
	[[nodiscard]] uint64_t GetTotalMemoryBudget() const;

	struct HeapUsage {
		uint64_t usage        = 0;
		uint64_t budget       = 0;
		uint32_t allocations  = 0;
		bool     device_local = false;
	};
	[[nodiscard]] static uint64_t HeapImageThreshold(uint64_t budget, bool low = false) noexcept {
		const auto share = [budget](uint64_t percent) { return budget / 100u * percent + budget % 100u * percent / 100u; };
		const auto reserve = std::min<uint64_t>(share(15), 2ull << 30u);
		return share(low ? 80 : 85) - reserve;
	}
	[[nodiscard]] LocalMemorySample GetLocalMemoryUsage() const;
	[[nodiscard]] bool      LocalMemoryCrowded() const;
	[[nodiscard]] uint64_t  LocalBlockSpace() const;
	[[nodiscard]] uint32_t GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const;
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image,
	                                   ImageAllocation allocation);
	void                   DeleteImage(VulkanImage& image);
	void                   TrimRecycledImages();
	[[nodiscard]] size_t   RecycledImageCount() const;

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	friend struct VramBudgetTestAccess;
	LocalMemoryBudget m_local_memory_budget;
	bool     m_memory_override_enabled = false;
	uint64_t m_memory_budget_override  = 0;
	uint64_t m_memory_usage_override   = 0;
	mutable std::mutex                                 m_format_properties_mutex;
	mutable std::map<vk::Format, vk::FormatProperties> m_format_properties;
	mutable std::mutex                                 m_image_format_properties_mutex;
	mutable std::map<std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
	                            vk::ImageCreateFlags>,
	                 std::pair<vk::Result, vk::ImageFormatProperties>>
	    m_image_format_properties;
};

struct VulkanImageState {
	vk::PipelineStageFlags2 pl_stage    = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        access_mask = vk::AccessFlagBits2::eNone;
	vk::ImageLayout         layout      = vk::ImageLayout::eUndefined;
};

struct VulkanImage {
	VulkanImage() = default;
	KYTY_CLASS_NO_COPY(VulkanImage);

	vk::Format                    format      = vk::Format::eUndefined;
	vk::ImageType                 image_type  = vk::ImageType::e2D;
	vk::Extent3D                  extent      = {1, 1, 1};
	uint32_t                      layers      = 1;
	uint32_t                      mip_levels  = 1;
	uint32_t                      samples     = 1;
	vk::ImageUsageFlags           usage       = {};
	vk::ImageCreateFlags          flags       = {};
	vk::Image                     image       = nullptr;
	VulkanImageState              state;
	std::vector<VulkanImageState> subresource_states;
	MemoryTelemetry::Allocation   charge;
	VmaAllocation                allocation = nullptr;
	bool recyclable = false;
	bool spilled    = false;
};



}

#endif

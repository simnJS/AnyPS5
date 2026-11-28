#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;

constexpr const char* Name = "kyty_shim_gpl_generated_module_lifetime";

constexpr std::array<uint32_t, 288> IndirectMeshSpirv {
	0x07230203u, 0x00010600u, 0x00070000u, 0x00000030u, 0x00000000u, 0x00020011u, 0x000014a3u, 0x00020011u,
	0x00000001u, 0x0006000au, 0x5f565053u, 0x5f545845u, 0x6873656du, 0x6168735fu, 0x00726564u, 0x0003000eu,
	0x00000000u, 0x00000001u, 0x0009000fu, 0x000014f5u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u,
	0x00000003u, 0x00000004u, 0x00000005u, 0x00030010u, 0x00000001u, 0x000014b2u, 0x00040010u, 0x00000001u,
	0x0000001au, 0x00000003u, 0x00040010u, 0x00000001u, 0x00001496u, 0x00000001u, 0x00060010u, 0x00000001u,
	0x00000011u, 0x00000020u, 0x00000001u, 0x00000001u, 0x00040047u, 0x00000002u, 0x0000000bu, 0x00000000u,
	0x00040047u, 0x00000003u, 0x0000001eu, 0x00000000u, 0x00040047u, 0x00000004u, 0x0000000bu, 0x000014b0u,
	0x00020013u, 0x00000006u, 0x00030021u, 0x00000007u, 0x00000006u, 0x00020014u, 0x00000008u, 0x00040015u,
	0x00000009u, 0x00000020u, 0x00000000u, 0x00030016u, 0x0000000au, 0x00000020u, 0x00040017u, 0x0000000bu,
	0x0000000au, 0x00000004u, 0x00040017u, 0x0000000cu, 0x00000009u, 0x00000003u, 0x0004002bu, 0x00000009u,
	0x0000000du, 0x00000000u, 0x0004002bu, 0x00000009u, 0x0000000eu, 0x00000001u, 0x0004002bu, 0x00000009u,
	0x0000000fu, 0x00000002u, 0x0004002bu, 0x00000009u, 0x00000010u, 0x00000003u, 0x0004002bu, 0x00000009u,
	0x00000011u, 0x00000006u, 0x0004002bu, 0x0000000au, 0x00000012u, 0xbf800000u, 0x0004002bu, 0x0000000au,
	0x00000013u, 0x00000000u, 0x0004002bu, 0x0000000au, 0x00000014u, 0x3f800000u, 0x0004002bu, 0x0000000au,
	0x00000015u, 0x40400000u, 0x0004001cu, 0x00000016u, 0x0000000bu, 0x00000010u, 0x0004001cu, 0x00000017u,
	0x0000000cu, 0x0000000eu, 0x0004001cu, 0x00000018u, 0x00000009u, 0x00000011u, 0x00040020u, 0x00000019u,
	0x00000003u, 0x00000016u, 0x00040020u, 0x0000001au, 0x00000003u, 0x0000000bu, 0x00040020u, 0x0000001bu,
	0x00000003u, 0x00000017u, 0x00040020u, 0x0000001cu, 0x00000003u, 0x0000000cu, 0x00040020u, 0x0000001du,
	0x0000151au, 0x00000018u, 0x00040020u, 0x0000001eu, 0x0000151au, 0x00000009u, 0x0004003bu, 0x00000019u,
	0x00000002u, 0x00000003u, 0x0004003bu, 0x00000019u, 0x00000003u, 0x00000003u, 0x0004003bu, 0x0000001bu,
	0x00000004u, 0x00000003u, 0x0004003bu, 0x0000001du, 0x00000005u, 0x0000151au, 0x00050036u, 0x00000006u,
	0x00000001u, 0x00000000u, 0x00000007u, 0x000200f8u, 0x0000001fu, 0x000314afu, 0x00000010u, 0x0000000eu,
	0x00050041u, 0x0000001eu, 0x00000020u, 0x00000005u, 0x0000000du, 0x0004003du, 0x00000009u, 0x00000021u,
	0x00000020u, 0x000500aau, 0x00000008u, 0x00000022u, 0x00000021u, 0x00000010u, 0x000600a9u, 0x0000000au,
	0x00000023u, 0x00000022u, 0x00000014u, 0x00000013u, 0x00070050u, 0x0000000bu, 0x00000024u, 0x00000023u,
	0x00000023u, 0x00000023u, 0x00000014u, 0x00070050u, 0x0000000bu, 0x00000025u, 0x00000012u, 0x00000012u,
	0x00000013u, 0x00000014u, 0x00070050u, 0x0000000bu, 0x00000026u, 0x00000015u, 0x00000012u, 0x00000013u,
	0x00000014u, 0x00070050u, 0x0000000bu, 0x00000027u, 0x00000012u, 0x00000015u, 0x00000013u, 0x00000014u,
	0x00050041u, 0x0000001au, 0x00000028u, 0x00000002u, 0x0000000du, 0x0003003eu, 0x00000028u, 0x00000025u,
	0x00050041u, 0x0000001au, 0x00000029u, 0x00000002u, 0x0000000eu, 0x0003003eu, 0x00000029u, 0x00000026u,
	0x00050041u, 0x0000001au, 0x0000002au, 0x00000002u, 0x0000000fu, 0x0003003eu, 0x0000002au, 0x00000027u,
	0x00050041u, 0x0000001au, 0x0000002bu, 0x00000003u, 0x0000000du, 0x0003003eu, 0x0000002bu, 0x00000024u,
	0x00050041u, 0x0000001au, 0x0000002cu, 0x00000003u, 0x0000000eu, 0x0003003eu, 0x0000002cu, 0x00000024u,
	0x00050041u, 0x0000001au, 0x0000002du, 0x00000003u, 0x0000000fu, 0x0003003eu, 0x0000002du, 0x00000024u,
	0x00060050u, 0x0000000cu, 0x0000002eu, 0x0000000du, 0x0000000eu, 0x0000000fu, 0x00050041u, 0x0000001cu,
	0x0000002fu, 0x00000004u, 0x0000000du, 0x0003003eu, 0x0000002fu, 0x0000002eu, 0x000100fdu, 0x00010038u,
};

constexpr std::array<uint32_t, 112> PerVertexPixelSpirv {
	0x07230203u, 0x00010600u, 0x00070000u, 0x00000012u, 0x00000000u, 0x00020011u, 0x00000001u, 0x00020011u,
	0x000014a4u, 0x000a000au, 0x5f565053u, 0x5f52484bu, 0x67617266u, 0x746e656du, 0x6168735fu, 0x5f726564u,
	0x79726162u, 0x746e6563u, 0x00636972u, 0x0003000eu, 0x00000000u, 0x00000001u, 0x0007000fu, 0x00000004u,
	0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u, 0x00000003u, 0x00030010u, 0x00000001u, 0x00000007u,
	0x00030047u, 0x00000002u, 0x000014a5u, 0x00040047u, 0x00000002u, 0x0000001eu, 0x00000000u, 0x00040047u,
	0x00000003u, 0x0000001eu, 0x00000000u, 0x00020013u, 0x00000004u, 0x00030021u, 0x00000005u, 0x00000004u,
	0x00040015u, 0x00000006u, 0x00000020u, 0x00000000u, 0x00030016u, 0x00000007u, 0x00000020u, 0x00040017u,
	0x00000008u, 0x00000007u, 0x00000004u, 0x0004002bu, 0x00000006u, 0x00000009u, 0x00000000u, 0x0004002bu,
	0x00000006u, 0x0000000au, 0x00000003u, 0x0004001cu, 0x0000000bu, 0x00000008u, 0x0000000au, 0x00040020u,
	0x0000000cu, 0x00000001u, 0x0000000bu, 0x00040020u, 0x0000000du, 0x00000001u, 0x00000008u, 0x00040020u,
	0x0000000eu, 0x00000003u, 0x00000008u, 0x0004003bu, 0x0000000cu, 0x00000002u, 0x00000001u, 0x0004003bu,
	0x0000000eu, 0x00000003u, 0x00000003u, 0x00050036u, 0x00000004u, 0x00000001u, 0x00000000u, 0x00000005u,
	0x000200f8u, 0x0000000fu, 0x00050041u, 0x0000000du, 0x00000010u, 0x00000002u, 0x00000009u, 0x0004003du,
	0x00000008u, 0x00000011u, 0x00000010u, 0x0003003eu, 0x00000003u, 0x00000011u, 0x000100fdu, 0x00010038u,
};

constexpr std::array<uint32_t, 91> RectListVertexSpirv {
	0x07230203u, 0x00010600u, 0x00070000u, 0x00000011u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0003000eu,
	0x00000000u, 0x00000001u, 0x0006000fu, 0x00000000u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u,
	0x00050048u, 0x00000003u, 0x00000000u, 0x0000000bu, 0x00000000u, 0x00030047u, 0x00000003u, 0x00000002u,
	0x00020013u, 0x00000004u, 0x00030021u, 0x00000005u, 0x00000004u, 0x00040015u, 0x00000006u, 0x00000020u,
	0x00000001u, 0x00030016u, 0x00000007u, 0x00000020u, 0x00040017u, 0x00000008u, 0x00000007u, 0x00000004u,
	0x0004002bu, 0x00000006u, 0x00000009u, 0x00000000u, 0x0004002bu, 0x00000007u, 0x0000000au, 0x00000000u,
	0x0004002bu, 0x00000007u, 0x0000000bu, 0x3f800000u, 0x0003001eu, 0x00000003u, 0x00000008u, 0x00040020u,
	0x0000000cu, 0x00000003u, 0x00000003u, 0x00040020u, 0x0000000du, 0x00000003u, 0x00000008u, 0x0004003bu,
	0x0000000cu, 0x00000002u, 0x00000003u, 0x00050036u, 0x00000004u, 0x00000001u, 0x00000000u, 0x00000005u,
	0x000200f8u, 0x0000000eu, 0x00070050u, 0x00000008u, 0x0000000fu, 0x0000000au, 0x0000000au, 0x0000000au,
	0x0000000bu, 0x00050041u, 0x0000000du, 0x00000010u, 0x00000002u, 0x00000009u, 0x0003003eu, 0x00000010u,
	0x0000000fu, 0x000100fdu, 0x00010038u,
};

constexpr std::array<uint32_t, 66> SolidPixelSpirv {
	0x07230203u, 0x00010600u, 0x00070000u, 0x0000000bu, 0x00000000u, 0x00020011u, 0x00000001u, 0x0003000eu,
	0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u,
	0x00030010u, 0x00000001u, 0x00000007u, 0x00040047u, 0x00000002u, 0x0000001eu, 0x00000000u, 0x00020013u,
	0x00000003u, 0x00030021u, 0x00000004u, 0x00000003u, 0x00030016u, 0x00000005u, 0x00000020u, 0x00040017u,
	0x00000006u, 0x00000005u, 0x00000004u, 0x0004002bu, 0x00000005u, 0x00000007u, 0x3f800000u, 0x00040020u,
	0x00000008u, 0x00000003u, 0x00000006u, 0x0004003bu, 0x00000008u, 0x00000002u, 0x00000003u, 0x00050036u,
	0x00000003u, 0x00000001u, 0x00000000u, 0x00000004u, 0x000200f8u, 0x00000009u, 0x00070050u, 0x00000006u,
	0x0000000au, 0x00000007u, 0x00000007u, 0x00000007u, 0x00000007u, 0x0003003eu, 0x00000002u, 0x0000000au,
	0x000100fdu, 0x00010038u,
};

struct IndirectMeshParams {
	uint64_t records;
	uint64_t index_address;
	uint32_t stride;
	uint32_t index_size;
	uint32_t index_limit;
	uint32_t primitive_size;
	uint32_t group_step;
	uint32_t max_groups;
	uint32_t max_instances;
	uint32_t max_total;
};

struct ModuleSpyState {
	std::mutex                         mutex;
	PFN_vkCreateShaderModule           real_create  = nullptr;
	PFN_vkDestroyShaderModule          real_destroy = nullptr;
	PFN_vkDestroyPipeline              real_destroy_pipeline = nullptr;
	bool                               active       = false;
	uint32_t                           created      = 0;
	uint32_t                           destroyed    = 0;
	uint32_t                           destroyed_before_pipeline = 0;
	uint32_t                           pipelines_destroyed       = 0;
	std::unordered_set<VkShaderModule> live;
};

ModuleSpyState g_spy;

VKAPI_ATTR VkResult VKAPI_CALL SpyCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* pCreateInfo,
                                                     const VkAllocationCallbacks* pAllocator,
                                                     VkShaderModule* pShaderModule) {
	const auto result = g_spy.real_create(device, pCreateInfo, pAllocator, pShaderModule);
	if (result == VK_SUCCESS && pShaderModule != nullptr && *pShaderModule != VK_NULL_HANDLE) {
		std::lock_guard lock(g_spy.mutex);
		if (g_spy.active) {
			g_spy.created++;
			g_spy.live.insert(*pShaderModule);
		}
	}
	return result;
}

VKAPI_ATTR void VKAPI_CALL SpyDestroyShaderModule(VkDevice device, VkShaderModule shaderModule,
                                                  const VkAllocationCallbacks* pAllocator) {
	if (shaderModule != VK_NULL_HANDLE) {
		std::lock_guard lock(g_spy.mutex);
		if (g_spy.active && g_spy.live.erase(shaderModule) != 0) {
			g_spy.destroyed++;
			if (g_spy.pipelines_destroyed == 0) {
				g_spy.destroyed_before_pipeline++;
			}
		}
	}
	g_spy.real_destroy(device, shaderModule, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL SpyDestroyPipeline(VkDevice device, VkPipeline pipeline,
                                              const VkAllocationCallbacks* pAllocator) {
	if (pipeline != VK_NULL_HANDLE) {
		std::lock_guard lock(g_spy.mutex);
		if (g_spy.active) {
			g_spy.pipelines_destroyed++;
		}
	}
	g_spy.real_destroy_pipeline(device, pipeline, pAllocator);
}

void Require(bool condition, const std::string& message) {
	if (condition) return;
	std::fprintf(stderr, "%s: FAILED: %s\n", Name, message.c_str());
	std::exit(1);
}

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
	Require(result.result == vk::Result::eSuccess,
	        std::string(what) + " failed: " + vk::to_string(result.result));
	return std::move(result.value);
}

void Checked(vk::Result result, const char* what) {
	Require(result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result));
}

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	Require(required == nullptr || std::strcmp(required, "1") != 0,
	        std::string("ANYPS5_REQUIRE_VULKAN=1 and ") + why);
	std::printf("%s_draw: skipped, %s\n", Name, why);
	return 77;
}

int DrawGplGeneratedModuleLifetime() {
	const vk::detail::DynamicLoader loader;
	if (!loader.success()) return NoDevice("the Vulkan loader is missing");
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo application;
	application.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info;
	instance_info.pApplicationInfo = &application;
	const auto created             = vk::createInstance(instance_info);
	if (created.result != vk::Result::eSuccess) return NoDevice("no Vulkan 1.3 instance");
	const auto instance = created.value;
	dispatch.init(instance);

	vk::PhysicalDevice                         physical;
	vk::PhysicalDeviceProperties               physical_props {};
	vk::PhysicalDeviceMeshShaderPropertiesEXT  selected_mesh_props {};
	uint32_t                                   family                 = 0;
	bool                                       has_barycentric        = false;
	bool                                       lacking_device_features = false;
	for (const auto candidate: Checked(instance.enumeratePhysicalDevices(), "device enumeration")) {
		const auto extensions = Checked(candidate.enumerateDeviceExtensionProperties(nullptr),
		                                "device extension enumeration");
		const auto has_ext    = [&](const char* ext) {
			return std::ranges::any_of(extensions, [ext](const auto& props) {
				return std::string_view(props.extensionName.data()) == ext;
			});
		};
		if (!has_ext(VK_EXT_MESH_SHADER_EXTENSION_NAME) ||
		    !has_ext(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) ||
		    !has_ext(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) ||
		    !has_ext(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME) ||
		    !has_ext(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME)) {
			lacking_device_features = true;
			continue;
		}
		const bool candidate_bary = has_ext(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);

		vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary_features;
		vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT   gpl_features;
		vk::PhysicalDeviceMeshShaderFeaturesEXT                mesh_features;
		vk::PhysicalDeviceDepthClipControlFeaturesEXT          depth_clip_control;
		vk::PhysicalDeviceDepthClipEnableFeaturesEXT           depth_clip_enable;
		vk::PhysicalDeviceVulkan12Features                     v12_features;
		vk::PhysicalDeviceVulkan13Features                     v13_features;
		vk::PhysicalDeviceFeatures2                            features;
		v13_features.pNext       = &v12_features;
		v12_features.pNext       = &mesh_features;
		mesh_features.pNext      = &gpl_features;
		gpl_features.pNext       = &depth_clip_control;
		depth_clip_control.pNext = &depth_clip_enable;
		depth_clip_enable.pNext  = candidate_bary ? &bary_features : nullptr;
		features.pNext           = &v13_features;
		candidate.getFeatures2(&features);

		const auto families = candidate.getQueueFamilyProperties();
		const auto graphics = std::ranges::find_if(families, [](const auto& queue) {
			return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eGraphics);
		});
		if (!mesh_features.meshShader || !mesh_features.taskShader ||
		    !gpl_features.graphicsPipelineLibrary || !depth_clip_control.depthClipControl ||
		    !depth_clip_enable.depthClipEnable || !v12_features.bufferDeviceAddress ||
		    !v12_features.uniformBufferStandardLayout || !v13_features.dynamicRendering ||
		    !features.features.shaderInt64 || !features.features.tessellationShader ||
		    !features.features.depthClamp || !features.features.depthBounds ||
		    graphics == families.end()) {
			lacking_device_features = true;
			continue;
		}
		vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_props;
		vk::PhysicalDeviceProperties2             props;
		props.pNext = &mesh_props;
		candidate.getProperties2(&props);
		if (mesh_props.maxMeshWorkGroupInvocations < 32u ||
		    mesh_props.maxMeshWorkGroupSize[0] < 32u || mesh_props.maxMeshOutputVertices < 3u ||
		    mesh_props.maxMeshOutputPrimitives < 1u) {
			lacking_device_features = true;
			continue;
		}
		const bool software = props.properties.deviceType == vk::PhysicalDeviceType::eCpu;
		if (physical && software) continue;
		physical            = candidate;
		physical_props      = props.properties;
		selected_mesh_props = mesh_props;
		family              = static_cast<uint32_t>(graphics - families.begin());
		has_barycentric     = candidate_bary && bary_features.fragmentShaderBarycentric;
		if (!software) break;
	}
	if (!physical) {
		instance.destroy();
		if (lacking_device_features) {
			std::printf("%s_draw: skipped, the device lacks GPL, task/mesh shader, tessellation, or "
			            "depth clip features\n",
			            Name);
			return 77;
		}
		return NoDevice("no Vulkan 1.3 device with GPL and task/mesh shader support");
	}

	std::vector<const char*> enabled_extensions {
	    VK_EXT_MESH_SHADER_EXTENSION_NAME,
	    VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
	    VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME,
	    VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
	    VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME,
	};
	if (has_barycentric) {
		enabled_extensions.push_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
	}
	const float               priority = 1.0f;
	vk::DeviceQueueCreateInfo queue_info;
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;

	vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary_features;
	bary_features.fragmentShaderBarycentric = VK_TRUE;
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl_features;
	gpl_features.graphicsPipelineLibrary = VK_TRUE;
	vk::PhysicalDeviceMeshShaderFeaturesEXT mesh_features;
	mesh_features.meshShader = VK_TRUE;
	mesh_features.taskShader = VK_TRUE;
	vk::PhysicalDeviceDepthClipControlFeaturesEXT depth_clip_control;
	depth_clip_control.depthClipControl = VK_TRUE;
	vk::PhysicalDeviceDepthClipEnableFeaturesEXT depth_clip_enable;
	depth_clip_enable.depthClipEnable = VK_TRUE;
	vk::PhysicalDeviceVulkan12Features v12_features;
	v12_features.bufferDeviceAddress         = VK_TRUE;
	v12_features.uniformBufferStandardLayout = VK_TRUE;
	vk::PhysicalDeviceVulkan13Features v13_features;
	v13_features.dynamicRendering = VK_TRUE;
	v13_features.pNext            = &v12_features;
	v12_features.pNext            = &mesh_features;
	mesh_features.pNext           = &gpl_features;
	gpl_features.pNext            = &depth_clip_control;
	depth_clip_control.pNext      = &depth_clip_enable;
	depth_clip_enable.pNext       = has_barycentric ? &bary_features : nullptr;
	vk::PhysicalDeviceFeatures2 features;
	features.features.shaderInt64        = VK_TRUE;
	features.features.tessellationShader = VK_TRUE;
	features.features.depthClamp         = VK_TRUE;
	features.features.depthBounds        = VK_TRUE;
	features.pNext                       = &v13_features;

	vk::DeviceCreateInfo device_info;
	device_info.pNext                   = &features;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(enabled_extensions.size());
	device_info.ppEnabledExtensionNames = enabled_extensions.data();
	const auto device                   = Checked(physical.createDevice(device_info), "device creation");
	dispatch.init(device);

	vk::PhysicalDeviceDriverProperties driver;
	vk::PhysicalDeviceProperties2      properties;
	properties.pNext = &driver;
	physical.getProperties2(&properties);
	const auto gpu_name = std::string(properties.properties.deviceName.data()) + " (" +
	                      driver.driverName.data() + ", " + driver.driverInfo.data() + ")";
	const auto queue = device.getQueue(family, 0);

	GraphicContext graphics;
	graphics.instance                   = instance;
	graphics.physical_device            = physical;
	graphics.physical_device_properties = physical_props;
	graphics.device                     = device;
	graphics.queue_family               = family;
	graphics.queue                      = queue;
	graphics.mesh_shader_enabled        = true;
	graphics.mesh_shader_properties     = selected_mesh_props;

	const auto make_module = [&](std::span<const uint32_t> spirv, ShaderType stage, uint64_t hash,
	                             bool mesh_task = false) {
		ShaderModule mod {};
		mod.info.stage       = stage;
		mod.info.shader_hash = hash;
		mod.mesh_task        = mesh_task;
		vk::ShaderModuleCreateInfo info;
		info.codeSize = spirv.size() * sizeof(uint32_t);
		info.pCode    = spirv.data();
		mod.module    = Checked(device.createShaderModule(info), "shader module");
		return mod;
	};

	const auto mesh_mod    = make_module(IndirectMeshSpirv, ShaderType::Mesh, 0x6d65736800000101ull, true);
	const auto pixel_mod   = has_barycentric
	                             ? make_module(PerVertexPixelSpirv, ShaderType::Pixel, 0x7069786500000102ull)
	                             : make_module(SolidPixelSpirv, ShaderType::Pixel, 0x7069786500000103ull);
	const auto rect_vs_mod = make_module(RectListVertexSpirv, ShaderType::Vertex, 0x7665727400000201ull);
	const auto solid_ps_mod = make_module(SolidPixelSpirv, ShaderType::Pixel, 0x7069786500000202ull);

	constexpr uint32_t Side   = 4u;
	constexpr uint32_t Pixels = Side * Side;
	constexpr uint32_t Bytes  = Pixels * 4u;
	constexpr auto     Format = vk::Format::eR8G8B8A8Unorm;

	GraphicsPipelineKey mesh_key {};
	mesh_key.stage_ids                     = {101u, 0u, 0u, 102u};
	mesh_key.rendering.color_count         = 1u;
	mesh_key.rendering.color_formats[0]    = Format;
	mesh_key.rendering.depth_format        = vk::Format::eUndefined;
	mesh_key.rendering.stencil_format      = vk::Format::eUndefined;
	mesh_key.static_params.samples         = 1u;
	mesh_key.static_params.polygon_mode    = vk::PolygonMode::eFill;
	mesh_key.static_params.depth_clip_enable = true;
	mesh_key.static_params.color_mask[0]   = 0x0fu;
	mesh_key.static_params.color_srcblend[0] = static_cast<uint32_t>(Prospero::BlendFactor::kOne);
	mesh_key.static_params.color_destblend[0] = static_cast<uint32_t>(Prospero::BlendFactor::kZero);
	mesh_key.static_params.color_comb_fcn[0] = static_cast<uint32_t>(Prospero::BlendOp::kAdd);

	GraphicsPipelineKey rect_key = mesh_key;
	rect_key.stage_ids                         = {201u, 0u, 0u, 202u};
	rect_key.static_params.topology            = vk::PrimitiveTopology::ePatchList;
	rect_key.static_params.patch_control_points = 3u;

	const PipelineLibraries::Stages mesh_stages {&mesh_mod, nullptr, nullptr, &pixel_mod};
	const PipelineLibraries::Stages rect_stages {&rect_vs_mod, nullptr, nullptr, &solid_ps_mod};

	PipelineCache::Pipeline mesh_pipeline {};
	mesh_pipeline.native_bindings = true;
	CreateGraphicsPipelineLayout(graphics, mesh_pipeline, mesh_key, mesh_stages);

	PipelineCache::Pipeline rect_pipeline {};
	rect_pipeline.native_bindings = true;
	CreateGraphicsPipelineLayout(graphics, rect_pipeline, rect_key, rect_stages);

	g_spy.real_create           = dispatch.vkCreateShaderModule;
	g_spy.real_destroy          = dispatch.vkDestroyShaderModule;
	g_spy.real_destroy_pipeline = dispatch.vkDestroyPipeline;
	{
		std::lock_guard lock(g_spy.mutex);
		g_spy.active                    = true;
		g_spy.created                   = 0;
		g_spy.destroyed                 = 0;
		g_spy.destroyed_before_pipeline = 0;
		g_spy.pipelines_destroyed       = 0;
		g_spy.live.clear();
	}
	dispatch.vkCreateShaderModule  = SpyCreateShaderModule;
	dispatch.vkDestroyShaderModule = SpyDestroyShaderModule;
	dispatch.vkDestroyPipeline     = SpyDestroyPipeline;

	auto libraries = std::make_unique<PipelineLibraries>(graphics, nullptr, nullptr);

	Require(libraries->Link(mesh_pipeline, mesh_key, mesh_stages),
	        "PipelineLibraries::Link failed for indirect mesh + pixel pipeline");
	{
		std::lock_guard lock(g_spy.mutex);
		Require(g_spy.created >= 1u, "indirect mesh GPL link did not create generated task shader module");
		Require(g_spy.destroyed == 0u,
		        "generated task shader module was destroyed before PipelineLibraries teardown");
	}

	Require(libraries->Link(rect_pipeline, rect_key, rect_stages),
	        "PipelineLibraries::Link failed for rect_list pipeline");
	uint32_t gpl_created = 0;
	{
		std::lock_guard lock(g_spy.mutex);
		gpl_created = g_spy.created;
		Require(gpl_created >= 3u, "rect_list GPL link did not create generated tessellation shader modules");
		Require(g_spy.destroyed == 0u,
		        "generated tessellation shader module was destroyed before PipelineLibraries teardown");
	}

	vk::Pipeline whole_mesh_pipe = nullptr;
	Checked(CreateGraphicsPipelineWhole(graphics, mesh_pipeline, mesh_key, mesh_stages, nullptr, {},
	                                    whole_mesh_pipe),
	        "CreateGraphicsPipelineWhole indirect mesh");
	{
		std::lock_guard lock(g_spy.mutex);
		Require(g_spy.created == gpl_created + 1u && g_spy.destroyed == 1u,
		        "CreateGraphicsPipelineWhole must still destroy its temporary task module immediately");
	}
	device.destroyPipeline(whole_mesh_pipe);

	const auto memory_types = physical.getMemoryProperties();
	const auto allocate = [&](const vk::MemoryRequirements& needs, vk::MemoryPropertyFlags flags,
	                          bool device_address = false) {
		for (uint32_t type = 0; type < memory_types.memoryTypeCount; type++) {
			if (((needs.memoryTypeBits >> type) & 1u) == 0u ||
			    (memory_types.memoryTypes[type].propertyFlags & flags) != flags) {
				continue;
			}
			vk::MemoryAllocateFlagsInfo alloc_flags;
			alloc_flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
			vk::MemoryAllocateInfo info;
			info.pNext           = device_address ? &alloc_flags : nullptr;
			info.allocationSize  = needs.size;
			info.memoryTypeIndex = type;
			return Checked(device.allocateMemory(info), "memory allocation");
		}
		Require(false, "no suitable memory type");
		return vk::DeviceMemory {};
	};

	constexpr vk::DeviceSize IndirectBytes = 128u;
	vk::BufferCreateInfo     indirect_buf_info;
	indirect_buf_info.size  = IndirectBytes;
	indirect_buf_info.usage = vk::BufferUsageFlagBits::eStorageBuffer |
	                          vk::BufferUsageFlagBits::eShaderDeviceAddress;
	const auto indirect_buf = Checked(device.createBuffer(indirect_buf_info), "indirect params buffer");
	const auto indirect_mem =
	    allocate(device.getBufferMemoryRequirements(indirect_buf),
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	             true);
	Checked(device.bindBufferMemory(indirect_buf, indirect_mem, 0), "indirect buffer bind");
	vk::BufferDeviceAddressInfo bda_info;
	bda_info.buffer       = indirect_buf;
	const uint64_t bda    = device.getBufferAddress(bda_info);
	{
		auto* mapped = static_cast<uint8_t*>(
		    Checked(device.mapMemory(indirect_mem, 0, IndirectBytes), "indirect buffer map"));
		std::memset(mapped, 0, IndirectBytes);
		IndirectMeshParams params {};
		params.records        = bda + 64u;
		params.stride         = 20u;
		params.primitive_size = 3u;
		params.group_step     = 3u;
		params.max_groups     = 1u;
		params.max_instances  = 1u;
		params.max_total      = 1u;
		std::memcpy(mapped, &params, sizeof(params));
		const std::array<uint32_t, 5> record {3u, 1u, 0u, 0u, 0u};
		std::memcpy(mapped + 64u, record.data(), sizeof(record));
		device.unmapMemory(indirect_mem);
	}

	vk::ImageCreateInfo image_info;
	image_info.imageType   = vk::ImageType::e2D;
	image_info.format      = Format;
	image_info.extent      = vk::Extent3D {Side, Side, 1u};
	image_info.mipLevels   = 1u;
	image_info.arrayLayers = 1u;
	image_info.samples     = vk::SampleCountFlagBits::e1;
	image_info.tiling      = vk::ImageTiling::eOptimal;
	image_info.usage       = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
	const auto image        = Checked(device.createImage(image_info), "image creation");
	const auto image_memory = allocate(device.getImageMemoryRequirements(image),
	                                   vk::MemoryPropertyFlagBits::eDeviceLocal);
	Checked(device.bindImageMemory(image, image_memory, 0), "image bind");

	const vk::ImageSubresourceRange whole_image {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, 1u};
	vk::ImageViewCreateInfo         view_info;
	view_info.image            = image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = Format;
	view_info.subresourceRange = whole_image;
	const auto view            = Checked(device.createImageView(view_info), "image view creation");

	vk::BufferCreateInfo readback_info;
	readback_info.size  = Bytes;
	readback_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	const auto readback_buf = Checked(device.createBuffer(readback_info), "readback buffer");
	const auto readback_mem =
	    allocate(device.getBufferMemoryRequirements(readback_buf),
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	Checked(device.bindBufferMemory(readback_buf, readback_mem, 0), "readback buffer bind");

	vk::CommandPoolCreateInfo pool_info;
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	pool_info.queueFamilyIndex = family;
	const auto pool            = Checked(device.createCommandPool(pool_info), "command pool");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool        = pool;
	command_info.commandBufferCount = 1;
	const auto command =
	    Checked(device.allocateCommandBuffers(command_info), "command buffer").front();
	const auto fence = Checked(device.createFence(vk::FenceCreateInfo {}), "fence");

	const auto image_barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags source,
	                               vk::AccessFlags target, vk::PipelineStageFlags before,
	                               vk::PipelineStageFlags after) {
		vk::ImageMemoryBarrier barrier;
		barrier.srcAccessMask       = source;
		barrier.dstAccessMask       = target;
		barrier.oldLayout           = from;
		barrier.newLayout           = to;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = image;
		barrier.subresourceRange    = whole_image;
		command.pipelineBarrier(before, after, {}, 0, nullptr, 0, nullptr, 1, &barrier);
	};

	Checked(command.begin(vk::CommandBufferBeginInfo {}), "command begin");
	image_barrier(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
	              vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eTopOfPipe,
	              vk::PipelineStageFlagBits::eColorAttachmentOutput);
	vk::RenderingAttachmentInfo target;
	target.imageView   = view;
	target.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	target.loadOp      = vk::AttachmentLoadOp::eClear;
	target.storeOp     = vk::AttachmentStoreOp::eStore;
	const vk::Rect2D scissor {{0, 0}, {Side, Side}};
	vk::RenderingInfo pass;
	pass.renderArea           = scissor;
	pass.layerCount           = 1u;
	pass.colorAttachmentCount = 1u;
	pass.pColorAttachments    = &target;
	command.beginRendering(&pass);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, mesh_pipeline.pipeline);
	const vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(Side), static_cast<float>(Side), 0.0f, 1.0f};
	command.setViewportWithCount(1u, &viewport);
	command.setScissorWithCount(1u, &scissor);
	command.setLineWidth(1.0f);
	command.setCullMode(vk::CullModeFlagBits::eNone);
	command.setFrontFace(vk::FrontFace::eCounterClockwise);
	command.setDepthTestEnable(VK_FALSE);
	command.setDepthWriteEnable(VK_FALSE);
	command.setDepthCompareOp(vk::CompareOp::eAlways);
	command.setDepthBiasEnable(VK_FALSE);
	command.setDepthBias(0.0f, 0.0f, 0.0f);
	command.setStencilTestEnable(VK_FALSE);
	command.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, vk::StencilOp::eKeep,
	                     vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways);
	command.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, 0xffu);
	command.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0u);
	command.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, 0xffu);
	const float blend_constants[4] {0.0f, 0.0f, 0.0f, 0.0f};
	command.setBlendConstants(blend_constants);
#if !defined(__APPLE__)
	command.setDepthBoundsTestEnable(VK_FALSE);
	command.setDepthBounds(0.0f, 1.0f);
#endif
	std::array<uint32_t, Recompiler::IR::NativePushConstantSize / 4u> push {};
	std::memcpy(push.data(), &bda, sizeof(bda));
	command.pushConstants(mesh_pipeline.pipeline_layout, mesh_pipeline.push_stages, 0,
	                      Recompiler::IR::NativePushConstantSize, push.data());
	command.drawMeshTasksEXT(1u, 1u, 1u);
	command.endRendering();
	image_barrier(vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
	              vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
	              vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer);
	vk::BufferImageCopy copy;
	copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0u, 0u, 1u};
	copy.imageExtent      = vk::Extent3D {Side, Side, 1u};
	command.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, readback_buf, 1, &copy);

	vk::BufferMemoryBarrier host_barrier;
	host_barrier.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
	host_barrier.dstAccessMask       = vk::AccessFlagBits::eHostRead;
	host_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	host_barrier.buffer              = readback_buf;
	host_barrier.offset              = 0;
	host_barrier.size                = Bytes;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &host_barrier, 0, nullptr);
	Checked(command.end(), "command end");

	vk::SubmitInfo submit;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Checked(queue.submit(1, &submit, fence), "queue submit");
	Checked(device.waitForFences(1, &fence, VK_TRUE, 5000000000ull), "waitForFences");

	const auto* bytes =
	    static_cast<const uint8_t*>(Checked(device.mapMemory(readback_mem, 0, Bytes), "readback map"));
	uint32_t lit = 0;
	for (uint32_t i = 0; i < Pixels; i++) {
		if (bytes[i * 4u + 0u] == 255u && bytes[i * 4u + 1u] == 255u &&
		    bytes[i * 4u + 2u] == 255u && bytes[i * 4u + 3u] == 255u) {
			lit++;
		}
	}
	device.unmapMemory(readback_mem);
	Require(lit == Pixels,
	        "linked indirect mesh draw lit " + std::to_string(lit) + " pixels, expected " +
	            std::to_string(Pixels));

	device.destroyFence(fence);
	device.destroyCommandPool(pool);
	device.destroyBuffer(readback_buf);
	device.freeMemory(readback_mem);
	device.destroyImageView(view);
	device.destroyImage(image);
	device.freeMemory(image_memory);
	device.destroyBuffer(indirect_buf);
	device.freeMemory(indirect_mem);
	device.destroyPipeline(rect_pipeline.pipeline);
	device.destroyPipelineLayout(rect_pipeline.pipeline_layout);
	device.destroyDescriptorSetLayout(rect_pipeline.descriptor_set_layout);
	device.destroyPipeline(mesh_pipeline.pipeline);
	device.destroyPipelineLayout(mesh_pipeline.pipeline_layout);
	device.destroyDescriptorSetLayout(mesh_pipeline.descriptor_set_layout);

	{
		std::lock_guard lock(g_spy.mutex);
		g_spy.pipelines_destroyed       = 0;
		g_spy.destroyed_before_pipeline = 0;
	}
	libraries.reset();

	uint32_t remaining_live = 0;
	uint32_t early_destroys = 0;
	{
		std::lock_guard lock(g_spy.mutex);
		g_spy.active   = false;
		remaining_live = static_cast<uint32_t>(g_spy.live.size());
		early_destroys = g_spy.destroyed_before_pipeline;
	}
	dispatch.vkCreateShaderModule  = g_spy.real_create;
	dispatch.vkDestroyShaderModule = g_spy.real_destroy;
	dispatch.vkDestroyPipeline     = g_spy.real_destroy_pipeline;

	Require(remaining_live == 0u,
	        "PipelineLibraries teardown leaked " + std::to_string(remaining_live) + " generated shader module(s)");
	Require(early_destroys == 0u,
	        "PipelineLibraries destroyed a generated shader module before destroying its library pipeline");

	device.destroyShaderModule(solid_ps_mod.module);
	device.destroyShaderModule(rect_vs_mod.module);
	device.destroyShaderModule(pixel_mod.module);
	device.destroyShaderModule(mesh_mod.module);
	device.destroy();
	instance.destroy();

	std::printf("%s_draw: ok, barycentric=%s retained=%u lit=%u/%u on %s\n", Name,
	            has_barycentric ? "on" : "off", gpl_created, lit, Pixels, gpu_name.c_str());
	return 0;
}

}

extern "C" int RunKytyGplGeneratedModuleLifetimeTest(int, char**) {
	return DrawGplGeneratedModuleLifetime();
}

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;

constexpr const char* Name = "kyty_shim_mesh_lds_atomic_load";

constexpr std::array<uint32_t, 33> MeshSubwordLdsCode {
    0xbefc03ffu, 0x00001003u, 0xbf900009u, 0x7e0202f3u, 0x7e0402f3u, 0x7e060280u, 0x7e0802f2u,
    0x7e0a02ffu, 0x40400000u, 0x7d840084u, 0x02020b01u, 0x7d840088u, 0x02040b02u, 0x2c0c0082u,
    0x4a0e0c91u, 0xd8780000u, 0x00000706u, 0xbf8a0000u, 0x7e120280u, 0xd8d80000u, 0x07000009u,
    0xbf8cc07fu, 0x7d840effu, 0x14131211u, 0x7e1002ffu, 0x80200400u, 0x7e1202ffu, 0x00200400u,
    0x02101308u, 0xf80000cfu, 0x04030201u, 0xf8000941u, 0x00000008u,
};

constexpr std::array<uint32_t, 4> ComputeSubwordLdsCode {
    0x7e020280u, 0xd8780000u, 0x00000100u, 0xbf810000u,
};

constexpr std::array<uint32_t, 72> SolidPixelSpirv {
    0x07230203u, 0x00010300u, 0x0008000bu, 0x0000000cu, 0x00000000u, 0x00020011u, 0x00000001u,
    0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u,
    0x00000009u, 0x00030010u, 0x00000004u, 0x00000007u, 0x00040047u, 0x00000009u, 0x0000001eu,
    0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
    0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u,
    0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
    0x0004002bu, 0x00000006u, 0x0000000au, 0x3f800000u, 0x0007002cu, 0x00000007u, 0x0000000bu,
    0x0000000au, 0x0000000au, 0x0000000au, 0x0000000au, 0x00050036u, 0x00000002u, 0x00000004u,
    0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0003003eu, 0x00000009u, 0x0000000bu,
    0x000100fdu, 0x00010038u,
};

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

void ValidateSpirv(const std::vector<uint32_t>& words) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer(
	    [&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		    messages += message;
		    messages += '\n';
	    });
	spvtools::ValidatorOptions validator;
	validator.SetUniformBufferStandardLayout(true);
	validator.SetScalarBlockLayout(true);
	Require(tools.Validate(words.data(), words.size(), validator),
	        std::string("SPIR-V validation failed:\n") + messages);
}

std::vector<uint32_t> CompileMesh(uint32_t host_subgroup_size) {
	std::vector<uint32_t> code(MeshSubwordLdsCode.begin(), MeshSubwordLdsCode.end());
	code.push_back(0xbf810000u);

	ShaderVertexInputInfo vertex {};
	vertex.logical_stage              = ShaderType::Mesh;
	vertex.wave_size                  = 64u;
	vertex.mesh.lds_size_dwords       = 64u;
	vertex.mesh.threads_num[0]        = 64u;
	vertex.mesh.threads_num[1]        = 1u;
	vertex.mesh.threads_num[2]        = 1u;
	vertex.mesh.host_subgroup_size    = host_subgroup_size;
	vertex.mesh.vertices_per_group    = 3u;
	vertex.mesh.primitives_per_group  = 1u;
	vertex.mesh.max_vertices          = 64u;
	vertex.mesh.max_primitives        = 1u;
	vertex.mesh.input_primitive       = static_cast<uint32_t>(Prospero::PrimitiveType::kPointList);
	vertex.mesh.output_primitive_size = 3u;
	vertex.mesh.provoking_vertex      = 0u;
	vertex.mesh.fast_launch           = false;

	std::array<uint32_t, 2>    user_data {};
	Recompiler::Spirv::Profile profile {};
	Recompiler::CompileOptions options {};
	options.stage             = ShaderType::Mesh;
	options.wave_size         = 64u;
	options.user_data_base    = 8u;
	options.user_data         = user_data;
	options.shader_hash       = 0x6d65736800000021ull;
	options.profile           = &profile;
	options.input_info.vertex = &vertex;

	const auto                       source     = Recompiler::PrepareShaderSource(code, options);
	auto                             translated = Recompiler::TranslateProgram(source.decoded, options);
	const auto                       plan       = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime runtime {};
	Recompiler::IR::ResourceSnapshot       snapshot;
	Recompiler::IR::ResourceSpecialization specialization;
	Require(Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization),
	        "failed to materialize mesh resources");
	auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization,
	                                           Recompiler::IR::PushData::MeshDrawDwordCount);
	ValidateSpirv(compiled.spirv);
	return compiled.spirv;
}

std::vector<uint32_t> CompileCompute(uint32_t host_subgroup_size) {
	std::vector<uint32_t> code(ComputeSubwordLdsCode.begin(), ComputeSubwordLdsCode.end());

	ShaderComputeInputInfo compute {};
	compute.lds_size_dwords    = 64u;
	compute.threads_num[0]     = 64u;
	compute.threads_num[1]     = 1u;
	compute.threads_num[2]     = 1u;
	compute.wave_size          = 64u;
	compute.host_subgroup_size = host_subgroup_size;

	std::array<uint32_t, 2>    user_data {};
	Recompiler::Spirv::Profile profile {};
	Recompiler::CompileOptions options {};
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64u;
	options.user_data_base     = 0u;
	options.user_data          = user_data;
	options.shader_hash        = 0x636f6d7000000021ull;
	options.profile            = &profile;
	options.input_info.compute = &compute;

	const auto                       source     = Recompiler::PrepareShaderSource(code, options);
	auto                             translated = Recompiler::TranslateProgram(source.decoded, options);
	const auto                       plan       = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime runtime {};
	Recompiler::IR::ResourceSnapshot       snapshot;
	Recompiler::IR::ResourceSpecialization specialization;
	Require(Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization),
	        "failed to materialize compute resources");
	auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0u);
	ValidateSpirv(compiled.spirv);
	return compiled.spirv;
}

struct WorkgroupAtomicLoadStats {
	uint32_t total = 0;
	uint32_t with_acquire_workgroup = 0;
	uint32_t with_other_semantics = 0;
};

WorkgroupAtomicLoadStats InspectWorkgroupAtomicLoads(const std::vector<uint32_t>& words) {
	std::unordered_map<uint32_t, uint32_t> pointer_storage;
	std::unordered_map<uint32_t, uint32_t> value_type;
	std::unordered_map<uint32_t, uint32_t> u32_constants;
	uint32_t                               u32_type = 0;
	WorkgroupAtomicLoadStats               stats {};

	const uint32_t ExpectedSemantics =
	    spv::MemorySemanticsAcquireMask | spv::MemorySemanticsWorkgroupMemoryMask;

	for (size_t at = 5; at < words.size();) {
		const uint32_t count  = words[at] >> 16u;
		const auto     opcode = static_cast<spv::Op>(words[at] & 0xffffu);
		if (count == 0 || at + count > words.size()) break;
		const auto word = [&](uint32_t index) { return words[at + index]; };
		if (opcode == spv::OpTypeInt && count == 4 && word(2) == 32u && word(3) == 0u) {
			u32_type = word(1);
		} else if (opcode == spv::OpConstant && count == 4 && word(1) == u32_type) {
			u32_constants[word(2)] = word(3);
		} else if (opcode == spv::OpConstantNull && count == 3 && word(1) == u32_type) {
			u32_constants[word(2)] = 0u;
		} else if (opcode == spv::OpTypePointer && count == 4) {
			pointer_storage[word(1)] = word(2);
		} else if ((opcode == spv::OpVariable || opcode == spv::OpAccessChain ||
		            opcode == spv::OpInBoundsAccessChain) &&
		           count >= 4) {
			value_type[word(2)] = word(1);
		} else if (opcode == spv::OpAtomicLoad && count == 6) {
			const auto type_it = value_type.find(word(3));
			const auto storage_it =
			    type_it != value_type.end() ? pointer_storage.find(type_it->second) : pointer_storage.end();
			if (storage_it != pointer_storage.end() && storage_it->second == spv::StorageClassWorkgroup) {
				stats.total++;
				const auto sem_it = u32_constants.find(word(5));
				if (sem_it != u32_constants.end() && sem_it->second == ExpectedSemantics) {
					stats.with_acquire_workgroup++;
				} else {
					stats.with_other_semantics++;
				}
			}
		}
		at += count;
	}
	return stats;
}

void VerifyCodegen() {
	for (const uint32_t subgroup: {32u, 64u}) {
		const auto mesh = InspectWorkgroupAtomicLoads(CompileMesh(subgroup));
		Require(mesh.total > 0u, "mesh shader with ds_write_b8 emitted no workgroup OpAtomicLoad");
		Require(mesh.with_other_semantics == 0u && mesh.with_acquire_workgroup == mesh.total,
		        "mesh workgroup OpAtomicLoad must use Acquire | WorkgroupMemory semantics");

		const auto compute = InspectWorkgroupAtomicLoads(CompileCompute(subgroup));
		Require(compute.total > 0u, "compute shader with ds_write_b8 emitted no workgroup OpAtomicLoad");
		Require(compute.with_other_semantics == 0u && compute.with_acquire_workgroup == compute.total,
		        "compute workgroup OpAtomicLoad must use Acquire | WorkgroupMemory semantics");
	}
}

int DrawMeshLdsAtomicLoad() {
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

	vk::PhysicalDevice physical;
	uint32_t           family              = 0;
	uint32_t           subgroup            = 0;
	bool               has_gpl             = false;
	bool               has_amd_shader_info = false;
	bool               skipped_llvmpipe    = false;
	std::string        llvmpipe_info;
	for (const auto candidate: Checked(instance.enumeratePhysicalDevices(), "device enumeration")) {
		const auto extensions = Checked(candidate.enumerateDeviceExtensionProperties(nullptr),
		                                "device extension enumeration");
		const auto has_ext    = [&](const char* ext) {
			return std::ranges::any_of(extensions, [ext](const auto& props) {
				return std::string_view(props.extensionName.data()) == ext;
			});
		};
		if (!has_ext(VK_EXT_MESH_SHADER_EXTENSION_NAME)) continue;
		const bool candidate_gpl = has_ext(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) &&
		                           has_ext(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);

		vk::PhysicalDeviceMeshShaderFeaturesEXT             mesh_features;
		vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl_features;
		vk::PhysicalDeviceVulkan12Features                  v12_features;
		vk::PhysicalDeviceVulkan13Features                  v13_features;
		vk::PhysicalDeviceFeatures2                         features;
		v13_features.pNext  = &v12_features;
		v12_features.pNext  = &mesh_features;
		mesh_features.pNext = candidate_gpl ? &gpl_features : nullptr;
		features.pNext      = &v13_features;
		candidate.getFeatures2(&features);

		const auto families = candidate.getQueueFamilyProperties();
		const auto graphics = std::ranges::find_if(families, [](const auto& queue) {
			return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eGraphics);
		});
		if (!mesh_features.meshShader || !v12_features.bufferDeviceAddress ||
		    !v12_features.uniformBufferStandardLayout || !v13_features.dynamicRendering ||
		    !features.features.shaderInt64 || graphics == families.end()) {
			continue;
		}
		vk::PhysicalDeviceDriverProperties      driver_props;
		vk::PhysicalDeviceVulkan11Properties    props11;
		vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_props;
		vk::PhysicalDeviceProperties2           props;
		driver_props.pNext = &mesh_props;
		props11.pNext      = &driver_props;
		props.pNext        = &props11;
		candidate.getProperties2(&props);
		if (mesh_props.maxMeshWorkGroupInvocations < 64u ||
		    mesh_props.maxMeshWorkGroupSize[0] < 64u || mesh_props.maxMeshOutputVertices < 64u ||
		    mesh_props.maxMeshOutputPrimitives < 1u || props11.subgroupSize == 0u) {
			continue;
		}
		if (driver_props.driverID == vk::DriverId::eMesaLlvmpipe &&
		    props.properties.driverVersion <= VK_MAKE_API_VERSION(0, 26, 2, 4)) {
			skipped_llvmpipe = true;
			llvmpipe_info    = driver_props.driverInfo.data();
			continue;
		}
		const bool software = props.properties.deviceType == vk::PhysicalDeviceType::eCpu;
		if (physical && software) continue;
		physical            = candidate;
		family              = static_cast<uint32_t>(graphics - families.begin());
		subgroup            = props11.subgroupSize;
		has_gpl             = candidate_gpl && gpl_features.graphicsPipelineLibrary;
		has_amd_shader_info = has_ext(VK_AMD_SHADER_INFO_EXTENSION_NAME);
		if (!software) break;
	}
	if (!physical) {
		instance.destroy();
		if (skipped_llvmpipe) {
			std::printf("%s_draw: skipped, the device has no mesh shader compiler that handles "
			            "these programs (llvmpipe %s)\n",
			            Name, llvmpipe_info.c_str());
			return 77;
		}
		return NoDevice("no device with VK_EXT_mesh_shader and dynamic rendering");
	}

	const auto mesh_spirv = CompileMesh(subgroup);

	std::vector<const char*> enabled_extensions {VK_EXT_MESH_SHADER_EXTENSION_NAME};
	if (has_gpl) {
		enabled_extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
		enabled_extensions.push_back(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
	}
	if (has_amd_shader_info) {
		enabled_extensions.push_back(VK_AMD_SHADER_INFO_EXTENSION_NAME);
	}
	const float               priority = 1.0f;
	vk::DeviceQueueCreateInfo queue_info;
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;

	vk::PhysicalDeviceMeshShaderFeaturesEXT              mesh_features;
	mesh_features.meshShader = VK_TRUE;
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl_features;
	gpl_features.graphicsPipelineLibrary = VK_TRUE;
	vk::PhysicalDeviceVulkan12Features v12_features;
	v12_features.bufferDeviceAddress         = VK_TRUE;
	v12_features.uniformBufferStandardLayout = VK_TRUE;
	vk::PhysicalDeviceVulkan13Features v13_features;
	v13_features.dynamicRendering = VK_TRUE;
	v13_features.pNext            = &v12_features;
	v12_features.pNext            = &mesh_features;
	mesh_features.pNext           = has_gpl ? &gpl_features : nullptr;
	vk::PhysicalDeviceFeatures2 features;
	features.features.shaderInt64 = VK_TRUE;
	features.pNext                = &v13_features;

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

	vk::ShaderModuleCreateInfo mesh_mod_info;
	mesh_mod_info.codeSize = mesh_spirv.size() * sizeof(uint32_t);
	mesh_mod_info.pCode    = mesh_spirv.data();
	const auto mesh_module = Checked(device.createShaderModule(mesh_mod_info), "mesh shader module");

	vk::ShaderModuleCreateInfo pixel_mod_info;
	pixel_mod_info.codeSize = SolidPixelSpirv.size() * sizeof(uint32_t);
	pixel_mod_info.pCode    = SolidPixelSpirv.data();
	const auto pixel_module = Checked(device.createShaderModule(pixel_mod_info), "pixel shader module");

	vk::PushConstantRange push_range;
	push_range.stageFlags = vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment;
	push_range.offset     = 0;
	push_range.size       = Recompiler::IR::NativePushConstantSize;
	vk::PipelineLayoutCreateInfo layout_info;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_range;
	const auto layout = Checked(device.createPipelineLayout(layout_info), "pipeline layout");

	constexpr uint32_t Side   = 4u;
	constexpr uint32_t Pixels = Side * Side;
	constexpr uint32_t Bytes  = Pixels * 4u;
	constexpr auto     Format = vk::Format::eR8G8B8A8Unorm;

	vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(Side), static_cast<float>(Side), 0.0f, 1.0f};
	vk::Rect2D   scissor {{0, 0}, {Side, Side}};
	vk::PipelineViewportStateCreateInfo viewports;
	viewports.viewportCount = 1;
	viewports.pViewports    = &viewport;
	viewports.scissorCount  = 1;
	viewports.pScissors     = &scissor;

	vk::PipelineRasterizationStateCreateInfo raster;
	raster.polygonMode = vk::PolygonMode::eFill;
	raster.cullMode    = vk::CullModeFlagBits::eNone;
	raster.lineWidth   = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisample;
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

	vk::PipelineColorBlendAttachmentState blend_target;
	blend_target.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo blend;
	blend.attachmentCount = 1;
	blend.pAttachments    = &blend_target;

	vk::PipelineShaderStageCreateInfo mesh_stage;
	mesh_stage.stage  = vk::ShaderStageFlagBits::eMeshEXT;
	mesh_stage.module = mesh_module;
	mesh_stage.pName  = "main";

	if (has_gpl) {
		vk::PipelineRenderingCreateInfo          view_only;
		vk::GraphicsPipelineLibraryCreateInfoEXT lib_info;
		lib_info.flags = vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders;
		lib_info.pNext = &view_only;
		vk::GraphicsPipelineCreateInfo lib_pipe_info;
		lib_pipe_info.pNext               = &lib_info;
		lib_pipe_info.flags               = vk::PipelineCreateFlagBits::eLibraryKHR;
		lib_pipe_info.stageCount          = 1;
		lib_pipe_info.pStages             = &mesh_stage;
		lib_pipe_info.pViewportState      = &viewports;
		lib_pipe_info.pRasterizationState = &raster;
		lib_pipe_info.layout              = layout;
		const auto library =
		    Checked(device.createGraphicsPipeline(nullptr, lib_pipe_info), "pre-rasterization library");
		device.destroyPipeline(library);
	}

	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0]        = mesh_stage;
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = pixel_module;
	stages[1].pName  = "main";

	vk::PipelineRenderingCreateInfo rendering;
	rendering.colorAttachmentCount    = 1;
	rendering.pColorAttachmentFormats = &Format;

	vk::GraphicsPipelineCreateInfo pipeline_info;
	pipeline_info.pNext               = &rendering;
	pipeline_info.stageCount          = static_cast<uint32_t>(stages.size());
	pipeline_info.pStages             = stages.data();
	pipeline_info.pViewportState      = &viewports;
	pipeline_info.pRasterizationState = &raster;
	pipeline_info.pMultisampleState   = &multisample;
	pipeline_info.pColorBlendState    = &blend;
	pipeline_info.layout              = layout;
	const auto pipeline =
	    Checked(device.createGraphicsPipeline(nullptr, pipeline_info), "whole mesh pipeline");

	const auto memory_types = physical.getMemoryProperties();
	const auto allocate     = [&](const vk::MemoryRequirements& needs, vk::MemoryPropertyFlags flags) {
		for (uint32_t type = 0; type < memory_types.memoryTypeCount; type++) {
			if (((needs.memoryTypeBits >> type) & 1u) == 0u ||
			    (memory_types.memoryTypes[type].propertyFlags & flags) != flags) {
				continue;
			}
			vk::MemoryAllocateInfo info;
			info.allocationSize  = needs.size;
			info.memoryTypeIndex = type;
			return Checked(device.allocateMemory(info), "memory allocation");
		}
		Require(false, "no suitable memory type");
		return vk::DeviceMemory {};
	};

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

	const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, 1u};
	vk::ImageViewCreateInfo         view_info;
	view_info.image            = image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = Format;
	view_info.subresourceRange = whole;
	const auto view            = Checked(device.createImageView(view_info), "image view creation");

	vk::BufferCreateInfo buffer_info;
	buffer_info.size  = Bytes;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	const auto buffer = Checked(device.createBuffer(buffer_info), "readback buffer creation");
	const auto buffer_memory =
	    allocate(device.getBufferMemoryRequirements(buffer),
	             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	Checked(device.bindBufferMemory(buffer, buffer_memory, 0), "buffer bind");

	vk::CommandPoolCreateInfo pool_info;
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	pool_info.queueFamilyIndex = family;
	const auto pool            = Checked(device.createCommandPool(pool_info), "command pool");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool        = pool;
	command_info.commandBufferCount = 1;
	const auto command =
	    Checked(device.allocateCommandBuffers(command_info), "command buffer").front();

	const auto move = [&](vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags source,
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
		barrier.subresourceRange    = whole;
		command.pipelineBarrier(before, after, {}, 0, nullptr, 0, nullptr, 1, &barrier);
	};

	Checked(command.begin(vk::CommandBufferBeginInfo {}), "command begin");
	move(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
	     vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eTopOfPipe,
	     vk::PipelineStageFlagBits::eColorAttachmentOutput);
	vk::RenderingAttachmentInfo target;
	target.imageView   = view;
	target.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	target.loadOp      = vk::AttachmentLoadOp::eClear;
	target.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo pass;
	pass.renderArea           = scissor;
	pass.layerCount           = 1u;
	pass.colorAttachmentCount = 1u;
	pass.pColorAttachments    = &target;
	command.beginRendering(&pass);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	std::array<uint32_t, Recompiler::IR::NativePushConstantSize / 4u> push {};
	push[0] = 3u;
	command.pushConstants(layout, push_range.stageFlags, 0, Recompiler::IR::NativePushConstantSize,
	                      push.data());
	command.drawMeshTasksEXT(1u, 1u, 1u);
	command.endRendering();
	move(vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
	     vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
	     vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer);
	vk::BufferImageCopy copy;
	copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0u, 0u, 1u};
	copy.imageExtent      = vk::Extent3D {Side, Side, 1u};
	command.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, 1, &copy);
	Checked(command.end(), "command end");

	vk::SubmitInfo submit;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Checked(queue.submit(1, &submit, nullptr), "queue submit");
	Checked(device.waitIdle(), "device waitIdle");

	const auto* bytes =
	    static_cast<const uint8_t*>(Checked(device.mapMemory(buffer_memory, 0, Bytes), "readback map"));
	uint32_t lit = 0;
	for (uint32_t i = 0; i < Pixels; i++) {
		if (bytes[i * 4u] != 0u) lit++;
	}
	device.unmapMemory(buffer_memory);
	Require(lit == Pixels,
	        "sub-dword LDS CAS mesh draw lit " + std::to_string(lit) + " pixels, expected " +
	            std::to_string(Pixels));

	device.destroyCommandPool(pool);
	device.destroyBuffer(buffer);
	device.freeMemory(buffer_memory);
	device.destroyImageView(view);
	device.destroyImage(image);
	device.freeMemory(image_memory);
	device.destroyPipeline(pipeline);
	device.destroyPipelineLayout(layout);
	device.destroyShaderModule(pixel_module);
	device.destroyShaderModule(mesh_module);
	device.destroy();
	instance.destroy();
	std::printf("%s_draw: ok, subgroup=%u gpl=%s amd_shader_info=%s lit=%u/%u on %s\n", Name, subgroup,
	            has_gpl ? "on" : "off", has_amd_shader_info ? "on" : "off", lit, Pixels, gpu_name.c_str());
	return 0;
}

}

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "draw") == 0) return DrawMeshLdsAtomicLoad();
	VerifyCodegen();
	std::printf("%s: ok\n", Name);
	return 0;
}

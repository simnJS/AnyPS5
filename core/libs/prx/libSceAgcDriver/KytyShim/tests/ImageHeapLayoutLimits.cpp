#include "TestVulkanDevice.h"
#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

constexpr uint32_t ImageKind  = ShaderRecompiler::IR::FirstImageBinding;
constexpr uint64_t HeapImages = ShaderRecompiler::IR::ImageHeapCapacity;

void Require(bool value, const std::string& what) {
	if (!value) {
		throw std::runtime_error(what);
	}
}

std::vector<uint32_t> EmptyComputeShader() {
	spvtools::SpirvTools  tools(SPV_ENV_VULKAN_1_1);
	std::vector<uint32_t> spirv;
	Require(tools.Assemble(R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %main = OpFunction %void None %fn
      %entry = OpLabel
               OpReturn
               OpFunctionEnd
)",
	                       &spirv),
	        "the compute shader does not assemble");
	return spirv;
}

struct Outcome {
	bool        created = false;
	bool        stopped = false;
	std::string message;
};

Outcome CreateHeapPipeline(GraphicContext& graphics, const std::vector<uint32_t>& code, uint64_t images) {
	ShaderModule module;
	module.info.stage                                 = ShaderType::Compute;
	module.info.info.uses_image_heap                  = true;
	module.info.bindings.descriptor_mask              = uint64_t {1} << ImageKind;
	module.info.bindings.descriptor_counts[ImageKind] = static_cast<uint32_t>(images);
	module.spirv_words                                = static_cast<uint32_t>(code.size());
	const vk::ShaderModuleCreateInfo shader_info {.codeSize = code.size() * sizeof(uint32_t), .pCode = code.data()};
	Require(graphics.device.createShaderModule(&shader_info, nullptr, &module.module) == vk::Result::eSuccess,
	        "could not create the shader module");

	PipelineCache::Pipeline pipeline;
	Outcome                 outcome;
	try {
		const Common::SoftExitScope soft_exit;
		const auto                  result = CreateComputePipeline(graphics, pipeline, module, nullptr, ComputePipelineCacheMode::Full);
		outcome.created                    = result == vk::Result::eSuccess;
		if (!outcome.created) {
			outcome.message = vk::to_string(result);
		}
	} catch (const Common::SoftExitError& error) {
		outcome.stopped = true;
		outcome.message = error.what();
	}
	graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	graphics.device.destroyShaderModule(module.module, nullptr);
	return outcome;
}

int Run(GraphicContext& graphics) {
	vk::PhysicalDeviceVulkan12Properties heap_limits {};
	vk::PhysicalDeviceProperties2        properties {};
	properties.pNext = &heap_limits;
	graphics.physical_device.getProperties2(&properties);
	const auto&    limits     = graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t heap_room  = std::min({heap_limits.maxDescriptorSetUpdateAfterBindSampledImages,
	                                      heap_limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
	                                      heap_limits.maxPerStageUpdateAfterBindResources});
	const uint64_t plain_room = std::min({limits.maxDescriptorSetSampledImages,
	                                      limits.maxPerStageDescriptorSampledImages, limits.maxPerStageResources});
	const uint64_t within     = uint64_t {DescriptorHeap::MaxSampledImages} + 1u;
	const uint64_t above      = heap_room - HeapImages + 1u;
	if (within + HeapImages > heap_room || within > plain_room || above > plain_room || above <= within) {
		std::printf("kyty_shim_image_heap_layout_limits: skipped, the device allows %" PRIu64
		            " sampled images or resources with update after bind and %" PRIu64 " without\n",
		            heap_room, plain_room);
		return 77;
	}

	graphics.image_heap_layout = ImageHeap::CreateLayout(graphics.device);
	const auto code            = EmptyComputeShader();
	const auto accepted        = CreateHeapPipeline(graphics, code, within);
	const auto refused         = CreateHeapPipeline(graphics, code, above);
	graphics.device.destroyDescriptorSetLayout(graphics.image_heap_layout, nullptr);
	graphics.image_heap_layout = nullptr;

	Require(accepted.created && !accepted.stopped,
	        "an image heap pipeline with " + std::to_string(within) +
	            " sampled images in set 0 was not created: " + accepted.message);
	Require(!refused.created, "an image heap pipeline with " + std::to_string(above) +
	                              " sampled images in set 0 was created above the update-after-bind limits");
	Require(refused.stopped && refused.message.find("with update after bind") != std::string::npos,
	        "an image heap pipeline with " + std::to_string(above) +
	            " sampled images in set 0 did not stop at the update-after-bind check: " + refused.message);
	std::printf("kyty_shim_image_heap_layout_limits: ok\n");
	return 0;
}

}

extern "C" int RunKytyImageHeapLayoutLimitsTest() {
	try {
		KytyShimTests::TestVulkanDevice gpu;
		if (const auto* unavailable = gpu.Initialize()) {
			const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
			std::fprintf(stderr, "kyty_shim_image_heap_layout_limits: %s, no usable Vulkan device (%s)\n",
			             required ? "FAILED" : "skipped", unavailable);
			return required ? 1 : 77;
		}
		return Run(gpu.graphics);
	} catch (const std::exception& error) {
		std::fprintf(stderr, "kyty_shim_image_heap_layout_limits: FAILED: %s\n", error.what());
		return 1;
	}
}

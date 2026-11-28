#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"
#include "TestVulkanDevice.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/window.h"

#include <spirv-tools/libspirv.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace fs = std::filesystem;

constexpr const char* Name       = "kyty_shim_graphics_pipeline_binaries";
constexpr uint32_t    BigSteps   = 10'500;
constexpr uint32_t    SmallSteps = 64;
constexpr uint32_t    Constants  = 61;
constexpr size_t      HeaderSize = 84;
constexpr size_t      KeySize    = 36;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

void Require(bool condition, const std::string& what) {
	if (!condition) throw std::runtime_error(what);
}

uint32_t Constant(uint32_t step) {
	return (0x9e3779b9u * (step % Constants + 1u)) | 1u;
}

uint32_t Expected(uint32_t seed, uint32_t steps) {
	uint32_t value = seed;
	for (uint32_t step = 0; step < steps; step++) {
		const auto k = Constant(step);
		switch (step % 3u) {
			case 0: value *= k; break;
			case 1: value += k; break;
			default: value ^= k; break;
		}
	}
	return value;
}

std::vector<uint32_t> Assemble(const std::string& text, const char* what) {
	spvtools::SpirvTools  tools(SPV_ENV_VULKAN_1_3);
	std::vector<uint32_t> spirv;
	Require(tools.Assemble(text, &spirv), std::string(what) + " does not assemble");
	Require(tools.Validate(spirv), std::string(what) + " is not valid SPIR-V");
	return spirv;
}

std::vector<uint32_t> VertexShader() {
	return Assemble(R"(
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %index %position
               OpDecorate %index BuiltIn VertexIndex
               OpDecorate %position BuiltIn Position
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
         %v4 = OpTypeVector %float 4
     %in_int = OpTypePointer Input %int
     %out_v4 = OpTypePointer Output %v4
      %index = OpVariable %in_int Input
   %position = OpVariable %out_v4 Output
       %int1 = OpConstant %int 1
     %float4 = OpConstant %float 4
     %float1 = OpConstant %float 1
    %float_1 = OpConstant %float -1
     %float0 = OpConstant %float 0
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %i = OpLoad %int %index
         %bx = OpBitwiseAnd %int %i %int1
         %hi = OpShiftRightArithmetic %int %i %int1
         %by = OpBitwiseAnd %int %hi %int1
         %fx = OpConvertSToF %float %bx
         %fy = OpConvertSToF %float %by
         %sx = OpFMul %float %fx %float4
         %sy = OpFMul %float %fy %float4
          %x = OpFAdd %float %sx %float_1
          %y = OpFAdd %float %sy %float_1
          %p = OpCompositeConstruct %v4 %x %y %float0 %float1
               OpStore %position %p
               OpReturn
               OpFunctionEnd
)",
	                "the vertex shader");
}

std::vector<uint32_t> ChainFragmentShader(uint32_t steps) {
	std::string text = R"(
               OpCapability Shader
               OpCapability PhysicalStorageBufferAddresses
               OpExtension "SPV_KHR_physical_storage_buffer"
               OpMemoryModel PhysicalStorageBuffer64 GLSL450
               OpEntryPoint Fragment %main "main" %push
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %Push Block
               OpMemberDecorate %Push 0 Offset 0
               OpMemberDecorate %Push 1 Offset 8
               OpDecorate %Out Block
               OpMemberDecorate %Out 0 Offset 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
        %Out = OpTypeStruct %uint
     %outptr = OpTypePointer PhysicalStorageBuffer %Out
       %Push = OpTypeStruct %outptr %uint
    %pushptr = OpTypePointer PushConstant %Push
       %push = OpVariable %pushptr PushConstant
   %push_out = OpTypePointer PushConstant %outptr
  %push_uint = OpTypePointer PushConstant %uint
   %out_uint = OpTypePointer PhysicalStorageBuffer %uint
       %int0 = OpConstant %int 0
       %int1 = OpConstant %int 1
)";
	for (uint32_t i = 0; i < Constants; i++) {
		text += "%k" + std::to_string(i) + " = OpConstant %uint " + std::to_string(Constant(i)) + "\n";
	}
	text += R"(
       %main = OpFunction %void None %fn
      %entry = OpLabel
       %pout = OpAccessChain %push_out %push %int0
        %out = OpLoad %outptr %pout
      %pseed = OpAccessChain %push_uint %push %int1
         %v0 = OpLoad %uint %pseed
)";
	static constexpr const char* Ops[] = {"OpIMul", "OpIAdd", "OpBitwiseXor"};
	for (uint32_t step = 0; step < steps; step++) {
		text += "%v" + std::to_string(step + 1) + " = " + Ops[step % 3u] + " %uint %v" + std::to_string(step) + " %k" +
		        std::to_string(step % Constants) + "\n";
	}
	text += "%dst = OpAccessChain %out_uint %out %int0\nOpStore %dst %v" + std::to_string(steps) +
	        " Aligned 4\nOpReturn\nOpFunctionEnd\n";
	return Assemble(text, "the fragment chain");
}

struct Program {
	GraphicContext&  graphics;
	vk::ShaderModule vertex;
	vk::ShaderModule fragment;
	vk::PipelineLayout layout;
	uint32_t         words = 0;

	Program(GraphicContext& context, const std::vector<uint32_t>& vertex_code, const std::vector<uint32_t>& fragment_code)
	    : graphics(context) {
		const auto device = graphics.device;
		const vk::ShaderModuleCreateInfo vertex_info {.codeSize = vertex_code.size() * sizeof(uint32_t),
		                                              .pCode    = vertex_code.data()};
		const vk::ShaderModuleCreateInfo fragment_info {.codeSize = fragment_code.size() * sizeof(uint32_t),
		                                                .pCode    = fragment_code.data()};
		Require(device.createShaderModule(&vertex_info, nullptr, &vertex) == vk::Result::eSuccess,
		        "could not create the vertex module");
		Require(device.createShaderModule(&fragment_info, nullptr, &fragment) == vk::Result::eSuccess,
		        "could not create the fragment module");
		const vk::PushConstantRange push {.stageFlags = vk::ShaderStageFlagBits::eFragment, .offset = 0, .size = 16};
		const vk::PipelineLayoutCreateInfo layout_info {.pushConstantRangeCount = 1, .pPushConstantRanges = &push};
		Require(device.createPipelineLayout(&layout_info, nullptr, &layout) == vk::Result::eSuccess,
		        "could not create the pipeline layout");
		words = static_cast<uint32_t>(vertex_code.size() + fragment_code.size());
	}

	~Program() {
		graphics.device.destroyPipelineLayout(layout, nullptr);
		graphics.device.destroyShaderModule(vertex, nullptr);
		graphics.device.destroyShaderModule(fragment, nullptr);
	}

	Program(const Program&)            = delete;
	Program& operator=(const Program&) = delete;
};

struct PipelineInfo {
	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	vk::PipelineVertexInputStateCreateInfo           vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo         input_assembly {.topology = vk::PrimitiveTopology::eTriangleList};
	vk::Viewport                                     viewport {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
	vk::Rect2D                                       scissor {{0, 0}, {1, 1}};
	vk::PipelineViewportStateCreateInfo              viewport_state {
	                 .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};
	vk::PipelineRasterizationStateCreateInfo rasterizer {.polygonMode = vk::PolygonMode::eFill,
	                                                     .cullMode    = vk::CullModeFlagBits::eNone,
	                                                     .frontFace   = vk::FrontFace::eCounterClockwise,
	                                                     .lineWidth   = 1.0f};
	vk::PipelineMultisampleStateCreateInfo multisample {.rasterizationSamples = vk::SampleCountFlagBits::e1};
	vk::PipelineColorBlendStateCreateInfo  blend {};
	vk::PipelineRenderingCreateInfo        rendering {};
	vk::GraphicsPipelineCreateInfo         info {};

	PipelineInfo(const Program& program, vk::CullModeFlags cull) {
		stages[0] = {.stage = vk::ShaderStageFlagBits::eVertex, .module = program.vertex, .pName = "main"};
		stages[1] = {.stage = vk::ShaderStageFlagBits::eFragment, .module = program.fragment, .pName = "main"};
		rasterizer.cullMode = cull;
		info.pNext               = &rendering;
		info.stageCount          = static_cast<uint32_t>(stages.size());
		info.pStages             = stages.data();
		info.pVertexInputState   = &vertex_input;
		info.pInputAssemblyState = &input_assembly;
		info.pViewportState      = &viewport_state;
		info.pRasterizationState = &rasterizer;
		info.pMultisampleState   = &multisample;
		info.pColorBlendState    = &blend;
		info.layout              = program.layout;
	}

	PipelineInfo(const PipelineInfo&)            = delete;
	PipelineInfo& operator=(const PipelineInfo&) = delete;
};

struct Renderer {
	GraphicContext&   graphics;
	vk::Buffer        buffer;
	vk::DeviceMemory  memory;
	void*             mapped  = nullptr;
	vk::DeviceAddress address = 0;
	vk::CommandPool   pool;
	vk::CommandBuffer command;
	vk::Fence         fence;
	vk::Queue         queue;

	explicit Renderer(GraphicContext& context): graphics(context) {
		const auto device = graphics.device;
		const vk::BufferCreateInfo buffer_info {
		    .size  = 64,
		    .usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress};
		Require(device.createBuffer(&buffer_info, nullptr, &buffer) == vk::Result::eSuccess, "createBuffer failed");
		const auto requirements = device.getBufferMemoryRequirements(buffer);
		const auto& types       = graphics.physical_device_memory_properties;
		const auto  wanted = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
		uint32_t    type   = UINT32_MAX;
		for (uint32_t i = 0; i < types.memoryTypeCount && type == UINT32_MAX; i++) {
			if ((requirements.memoryTypeBits & (1u << i)) && (types.memoryTypes[i].propertyFlags & wanted) == wanted) {
				type = i;
			}
		}
		Require(type != UINT32_MAX, "no host-visible coherent memory type");
		const vk::MemoryAllocateFlagsInfo flags {.flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
		const vk::MemoryAllocateInfo      allocate {
		         .pNext = &flags, .allocationSize = requirements.size, .memoryTypeIndex = type};
		Require(device.allocateMemory(&allocate, nullptr, &memory) == vk::Result::eSuccess, "allocateMemory failed");
		Require(device.bindBufferMemory(buffer, memory, 0) == vk::Result::eSuccess, "bindBufferMemory failed");
		Require(device.mapMemory(memory, 0, VK_WHOLE_SIZE, {}, &mapped) == vk::Result::eSuccess, "mapMemory failed");
		const vk::BufferDeviceAddressInfo address_info {.buffer = buffer};
		address = device.getBufferAddress(&address_info);
		const vk::CommandPoolCreateInfo pool_info {.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
		                                           .queueFamilyIndex = graphics.queue_family};
		Require(device.createCommandPool(&pool_info, nullptr, &pool) == vk::Result::eSuccess,
		        "createCommandPool failed");
		const vk::CommandBufferAllocateInfo command_info {
		    .commandPool = pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
		Require(device.allocateCommandBuffers(&command_info, &command) == vk::Result::eSuccess,
		        "allocateCommandBuffers failed");
		const vk::FenceCreateInfo fence_info {};
		Require(device.createFence(&fence_info, nullptr, &fence) == vk::Result::eSuccess, "createFence failed");
		device.getQueue(graphics.queue_family, 0, &queue);
	}

	~Renderer() {
		const auto device = graphics.device;
		device.destroyFence(fence, nullptr);
		device.destroyCommandPool(pool, nullptr);
		device.destroyBuffer(buffer, nullptr);
		device.freeMemory(memory, nullptr);
	}

	Renderer(const Renderer&)            = delete;
	Renderer& operator=(const Renderer&) = delete;

	uint32_t Run(vk::Pipeline pipeline, vk::PipelineLayout layout, uint32_t seed) {
		struct Push {
			uint64_t address;
			uint32_t seed;
			uint32_t pad;
		} push {address, seed, 0};
		*static_cast<volatile uint32_t*>(mapped) = 0;
		const auto device = graphics.device;
		Require(device.resetFences(1, &fence) == vk::Result::eSuccess, "resetFences failed");
		Require(command.reset({}) == vk::Result::eSuccess, "command buffer reset failed");
		const vk::CommandBufferBeginInfo begin {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
		Require(command.begin(&begin) == vk::Result::eSuccess, "command buffer begin failed");
		const vk::RenderingInfo rendering {.renderArea = {{0, 0}, {1, 1}}, .layerCount = 1};
		command.beginRendering(&rendering);
		command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		command.pushConstants(layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(push), &push);
		command.draw(3, 1, 0, 0);
		command.endRendering();
		const vk::MemoryBarrier barrier {.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
		                                 .dstAccessMask = vk::AccessFlagBits::eHostRead};
		command.pipelineBarrier(vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eHost, {}, 1,
		                        &barrier, 0, nullptr, 0, nullptr);
		Require(command.end() == vk::Result::eSuccess, "command buffer end failed");
		const vk::SubmitInfo submit {.commandBufferCount = 1, .pCommandBuffers = &command};
		Require(queue.submit(1, &submit, fence) == vk::Result::eSuccess, "queue submit failed");
		Require(device.waitForFences(1, &fence, VK_TRUE, 10'000'000'000ull) == vk::Result::eSuccess,
		        "the draw did not finish");
		return *static_cast<volatile uint32_t*>(mapped);
	}
};

struct Files {
	size_t   binaries  = 0;
	size_t   temporary = 0;
	fs::path last;
};

Files ListFiles(const fs::path& directory) {
	Files           files;
	std::error_code error;
	for (fs::directory_iterator entry(directory, error), end; !error && entry != end; entry.increment(error)) {
		const auto extension = entry->path().extension();
		if (extension == ".bin") {
			files.binaries++;
			files.last = entry->path();
		} else if (extension == ".tmp") {
			files.temporary++;
		}
	}
	return files;
}

std::vector<char> ReadFile(const fs::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void WriteFile(const fs::path& path, const std::vector<char>& bytes) {
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
	Require(static_cast<bool>(file), "could not rewrite the binary file");
}

double Milliseconds(std::chrono::steady_clock::duration elapsed) {
	return std::chrono::duration<double, std::milli>(elapsed).count();
}

struct Attempt {
	std::optional<vk::Result> result;
	std::string_view          mode;
	vk::Pipeline              pipeline;
};

Attempt Create(GraphicContext& graphics, const PipelineInfo& info, ComputePipelineCacheMode cache_mode) {
	Attempt     attempt;
	const char* mode = "";
	attempt.result   = CreateGraphicsPipelineWithBinary(graphics, info.info, nullptr, cache_mode, attempt.pipeline, mode);
	attempt.mode     = mode != nullptr ? mode : "";
	return attempt;
}

void Destroy(GraphicContext& graphics, Attempt& attempt) {
	graphics.device.destroyPipeline(attempt.pipeline, nullptr);
	attempt.pipeline = nullptr;
}

void RefusedFile(GraphicContext& graphics, Renderer& renderer, const Program& program, const PipelineInfo& info,
                 const fs::path& path, const std::vector<char>& bytes, const std::string& what, uint32_t seed) {
	WriteFile(path, bytes);
	auto restored = Create(graphics, info, ComputePipelineCacheMode::Lookup);
	Check(!restored.result.has_value(), "a " + what + " file was restored");
	Check(!restored.pipeline, "a refused " + what + " file left a pipeline behind");
	Destroy(graphics, restored);
	auto compiled = Create(graphics, info, ComputePipelineCacheMode::Full);
	Check(compiled.result == vk::Result::eSuccess && compiled.mode == "binary-capture",
	      "the pipeline was not compiled and captured again after a " + what + " file");
	Check(compiled.pipeline && renderer.Run(compiled.pipeline, program.layout, seed) == Expected(seed, BigSteps),
	      "the pipeline compiled after a " + what + " file computes a wrong value");
	Destroy(graphics, compiled);
	auto again = Create(graphics, info, ComputePipelineCacheMode::Lookup);
	Check(again.result == vk::Result::eSuccess && again.mode == "binary",
	      "the file written again after a " + what + " file is not restored");
	Destroy(graphics, again);
}

int Run(GraphicContext& graphics) {
	const auto directory = fs::temp_directory_path() / ("kyty-graphics-binaries-" + std::to_string(
	                                                        std::chrono::steady_clock::now().time_since_epoch().count()));
	std::error_code error;
	fs::remove_all(directory, error);
	graphics.compute_pipeline_binary_directory = directory;

	const auto vertex_code = VertexShader();
	const auto big_code    = ChainFragmentShader(BigSteps);
	const auto small_code  = ChainFragmentShader(SmallSteps);
	int        result      = 0;
	{
		Program  big(graphics, vertex_code, big_code);
		Program  small(graphics, vertex_code, small_code);
		Renderer renderer(graphics);
		Require(big.words >= 50'000 && small.words < 50'000, "the test programs have the wrong sizes");

		ShaderModule vertex_stage;
		ShaderModule big_stage;
		ShaderModule small_stage;
		vertex_stage.spirv_words = static_cast<uint32_t>(vertex_code.size());
		big_stage.spirv_words    = static_cast<uint32_t>(big_code.size());
		small_stage.spirv_words  = static_cast<uint32_t>(small_code.size());
		const std::array<const ShaderModule*, 4> big_stages {&vertex_stage, nullptr, nullptr, &big_stage};
		const std::array<const ShaderModule*, 4> small_stages {&vertex_stage, nullptr, nullptr, &small_stage};

		if (!graphics.compute_pipeline_binaries) {
			Check(!CanUseGraphicsBinaries(graphics, big_stages), "binaries are used while the extensions are off");
			if (g_failures == 0) {
				std::printf("%s: ok (pipeline binaries unavailable on %s: only the selection is checked)\n", Name,
				            graphics.physical_device_properties.deviceName.data());
			}
			result = g_failures == 0 ? 0 : 1;
		} else {
			Check(CanUseGraphicsBinaries(graphics, big_stages), "a pipeline of 50,000 words or more is not selected");
			Check(!CanUseGraphicsBinaries(graphics, small_stages), "a pipeline below 50,000 words is selected");

			const PipelineInfo drawn(big, vk::CullModeFlagBits::eNone);
			const PipelineInfo culled(big, vk::CullModeFlagBits::eFrontAndBack);

			auto missing = Create(graphics, drawn, ComputePipelineCacheMode::Lookup);
			Check(!missing.result.has_value() && !missing.pipeline, "a lookup without a binary file succeeded");
			Destroy(graphics, missing);

			const auto compile_started = std::chrono::steady_clock::now();
			auto       captured        = Create(graphics, drawn, ComputePipelineCacheMode::Full);
			const auto compile_ms      = Milliseconds(std::chrono::steady_clock::now() - compile_started);
			Require(captured.result == vk::Result::eSuccess && captured.pipeline,
			        "the large graphics pipeline was not created");
			Check(captured.mode == "binary-capture", "the large graphics pipeline was not captured");
			Check(renderer.Run(captured.pipeline, big.layout, 3u) == Expected(3u, BigSteps),
			      "the captured graphics pipeline computes a wrong value");
			Destroy(graphics, captured);
			const auto files = ListFiles(directory);
			Require(files.binaries == 1 && files.temporary == 0,
			        "the capture wrote " + std::to_string(files.binaries) + " binary files and left " +
			            std::to_string(files.temporary) + " temporary files");
			const auto path      = files.last;
			const auto file_size = fs::file_size(path);

			const auto restore_started = std::chrono::steady_clock::now();
			auto       restored        = Create(graphics, drawn, ComputePipelineCacheMode::Lookup);
			const auto restore_ms      = Milliseconds(std::chrono::steady_clock::now() - restore_started);
			Check(restored.result == vk::Result::eSuccess && restored.mode == "binary" && restored.pipeline,
			      "the large graphics pipeline was not restored from its binary");
			Check(restored.pipeline && renderer.Run(restored.pipeline, big.layout, 11u) == Expected(11u, BigSteps),
			      "the restored graphics pipeline computes a wrong value");
			Destroy(graphics, restored);

			const auto old_time = fs::file_time_type::clock::now() - std::chrono::hours(1);
			fs::last_write_time(path, old_time);
			auto full = Create(graphics, drawn, ComputePipelineCacheMode::Full);
			Check(full.result == vk::Result::eSuccess && full.mode == "binary",
			      "a full creation with a valid binary did not restore it");
			Destroy(graphics, full);
			Check(fs::last_write_time(path) == old_time, "a full creation compiled although a valid binary exists");

			auto other_state = Create(graphics, culled, ComputePipelineCacheMode::Lookup);
			const bool same_key = other_state.result.has_value();
			if (same_key) {
				Check(other_state.result == vk::Result::eSuccess && other_state.mode == "binary" && other_state.pipeline,
				      "the culled pipeline shares the drawn pipeline's key but was not restored");
				Check(other_state.pipeline && renderer.Run(other_state.pipeline, big.layout, 43u) == 0u,
				      "the culled pipeline restored from the shared binary drew: the binary overrode its rasterizer state");
				Check(ListFiles(directory).binaries == 1, "a restore wrote a binary file");
			}
			Destroy(graphics, other_state);
			if (!same_key) {
				auto culled_capture = Create(graphics, culled, ComputePipelineCacheMode::Full);
				Check(culled_capture.result == vk::Result::eSuccess && culled_capture.mode == "binary-capture",
				      "the culled pipeline was not compiled and captured");
				Destroy(graphics, culled_capture);
				Check(ListFiles(directory).binaries == 2, "two rasterizer states with two keys did not give two files");
			}
			auto culled_restore = Create(graphics, culled, ComputePipelineCacheMode::Lookup);
			Check(culled_restore.result == vk::Result::eSuccess && culled_restore.mode == "binary",
			      "the culled pipeline was not restored from a binary");
			Check(culled_restore.pipeline && renderer.Run(culled_restore.pipeline, big.layout, 45u) == 0u,
			      "the restored culled pipeline drew");
			Destroy(graphics, culled_restore);
			auto drawn_again = Create(graphics, drawn, ComputePipelineCacheMode::Lookup);
			Check(drawn_again.result == vk::Result::eSuccess && drawn_again.pipeline &&
			          renderer.Run(drawn_again.pipeline, big.layout, 47u) == Expected(47u, BigSteps),
			      "the drawn pipeline restored after the culled one computes a wrong value");
			Destroy(graphics, drawn_again);

			const auto files_before_other = ListFiles(directory).binaries;
			{
				Program            changed(graphics, vertex_code, ChainFragmentShader(BigSteps + 1));
				const PipelineInfo changed_info(changed, vk::CullModeFlagBits::eNone);
				auto stale = Create(graphics, changed_info, ComputePipelineCacheMode::Lookup);
				Check(!stale.result.has_value() && !stale.pipeline,
				      "a pipeline whose fragment SPIR-V changed was served an existing binary");
				Destroy(graphics, stale);
				auto fresh = Create(graphics, changed_info, ComputePipelineCacheMode::Full);
				Check(fresh.result == vk::Result::eSuccess && fresh.mode == "binary-capture",
				      "the pipeline with changed SPIR-V was not compiled and captured");
				Check(fresh.pipeline && renderer.Run(fresh.pipeline, changed.layout, 53u) == Expected(53u, BigSteps + 1),
				      "the pipeline with changed SPIR-V computes a wrong value");
				Destroy(graphics, fresh);
				auto changed_restore = Create(graphics, changed_info, ComputePipelineCacheMode::Lookup);
				Check(changed_restore.result == vk::Result::eSuccess && changed_restore.pipeline &&
				          renderer.Run(changed_restore.pipeline, changed.layout, 59u) == Expected(59u, BigSteps + 1),
				      "the pipeline with changed SPIR-V is not restored from its own binary");
				Destroy(graphics, changed_restore);
			}
			Check(ListFiles(directory).binaries == files_before_other + 1,
			      "changed SPIR-V did not give exactly one more binary file");

			const auto valid = ReadFile(path);
			Require(valid.size() == file_size, "the binary file changed size");
			auto flipped   = valid;
			flipped.back() = static_cast<char>(flipped.back() ^ 0x5a);
			RefusedFile(graphics, renderer, big, drawn, path, flipped, "corrupted", 13u);
			RefusedFile(graphics, renderer, big, drawn, path, {valid.begin(), valid.end() - 1}, "truncated", 17u);
			RefusedFile(graphics, renderer, big, drawn, path, {valid.begin(), valid.begin() + HeaderSize}, "header-only",
			            19u);
			RefusedFile(graphics, renderer, big, drawn, path, {}, "empty", 23u);
			auto other_key                     = valid;
			other_key[8 + KeySize + 4]         = static_cast<char>(other_key[8 + KeySize + 4] ^ 0x01);
			RefusedFile(graphics, renderer, big, drawn, path, other_key, "foreign-key", 29u);
			auto bad_magic = valid;
			bad_magic[0]   = 'X';
			RefusedFile(graphics, renderer, big, drawn, path, bad_magic, "foreign-magic", 31u);
			Check(ListFiles(directory).temporary == 0, "temporary files were left behind");

			if (g_failures == 0) {
				std::printf("%s: ok (%s, %u words, file %ju bytes, compile %.1f ms, restore %.1f ms, culling %s)\n", Name,
				            graphics.physical_device_properties.deviceName.data(), big.words,
				            static_cast<uintmax_t>(file_size), compile_ms, restore_ms,
				            same_key ? "outside the driver key" : "inside the driver key");
			}
			result = g_failures == 0 ? 0 : 1;
		}
	}
	fs::remove_all(directory, error);
	return result;
}

struct Device {
	GraphicContext graphics;

	~Device() {
		if (graphics.device) graphics.device.destroy(nullptr);
		if (graphics.instance) graphics.instance.destroy(nullptr);
	}
};

}

extern "C" int RunKytyGraphicsPipelineBinariesTest() {
	try {
		const Common::SoftExitScope soft_exit;
		KytyShim::StartRuntime();
		std::string error;
		{
			Device device;
			bool   created = false;
			try {
				created = CreateHeadlessVulkan(device.graphics, nullptr, error);
			} catch (const Common::SoftExitError& stop) {
				error = stop.what();
			}
			if (created) return Run(device.graphics);
		}
		KytyShimTests::TestVulkanDevice plain;
		if (const auto* unavailable = plain.Initialize()) {
			const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
			std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s; %s)\n", Name, required ? "FAILED" : "skipped",
			             error.c_str(), unavailable);
			return required ? 1 : 77;
		}
		std::fprintf(stderr, "%s: the emulator's device is unavailable (%s), testing the selection on a plain device\n",
		             Name, error.c_str());
		plain.graphics.compute_pipeline_binaries = false;
		return Run(plain.graphics);
	} catch (const std::exception& error) {
		std::fprintf(stderr, "FAILED: %s: %s\n", Name, error.what());
		return 1;
	}
}

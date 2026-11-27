#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"
#include "TestVulkanDevice.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/window.h"

#include <spirv-tools/libspirv.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace fs = std::filesystem;

constexpr const char* Name       = "kyty_shim_compute_pipeline_binaries";
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

std::vector<uint32_t> ChainShader(uint32_t steps) {
	std::string text = R"(
               OpCapability Shader
               OpCapability PhysicalStorageBufferAddresses
               OpExtension "SPV_KHR_physical_storage_buffer"
               OpMemoryModel PhysicalStorageBuffer64 GLSL450
               OpEntryPoint GLCompute %main "main" %push
               OpExecutionMode %main LocalSize 1 1 1
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
	spvtools::SpirvTools  tools(SPV_ENV_VULKAN_1_3);
	std::vector<uint32_t> spirv;
	Require(tools.Assemble(text, &spirv), "the compute chain does not assemble");
	Require(tools.Validate(spirv), "the compute chain is not valid SPIR-V");
	return spirv;
}

void MakeModule(GraphicContext& graphics, ShaderModule& module, const std::vector<uint32_t>& code, uint64_t id) {
	module.info.stage  = ShaderType::Compute;
	module.spirv_words = static_cast<uint32_t>(code.size());
	module.id          = id;
	const vk::ShaderModuleCreateInfo info {.codeSize = code.size() * sizeof(uint32_t), .pCode = code.data()};
	Require(graphics.device.createShaderModule(&info, nullptr, &module.module) == vk::Result::eSuccess,
	        "could not create the shader module");
}

void Destroy(GraphicContext& graphics, PipelineCache::Pipeline& pipeline) {
	graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	pipeline = {};
}

struct Dispatcher {
	GraphicContext&   graphics;
	vk::Buffer        buffer;
	vk::DeviceMemory  memory;
	void*             mapped  = nullptr;
	vk::DeviceAddress address = 0;
	vk::CommandPool   pool;
	vk::CommandBuffer command;
	vk::Fence         fence;
	vk::Queue         queue;

	explicit Dispatcher(GraphicContext& context): graphics(context) {
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

	~Dispatcher() {
		const auto device = graphics.device;
		device.destroyFence(fence, nullptr);
		device.destroyCommandPool(pool, nullptr);
		device.destroyBuffer(buffer, nullptr);
		device.freeMemory(memory, nullptr);
	}

	Dispatcher(const Dispatcher&)            = delete;
	Dispatcher& operator=(const Dispatcher&) = delete;

	uint32_t Run(const PipelineCache::Pipeline& pipeline, uint32_t seed) {
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
		command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
		command.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		command.dispatch(1, 1, 1);
		Require(command.end() == vk::Result::eSuccess, "command buffer end failed");
		const vk::SubmitInfo submit {.commandBufferCount = 1, .pCommandBuffers = &command};
		Require(queue.submit(1, &submit, fence) == vk::Result::eSuccess, "queue submit failed");
		Require(device.waitForFences(1, &fence, VK_TRUE, 10'000'000'000ull) == vk::Result::eSuccess,
		        "the dispatch did not finish");
		return *static_cast<volatile uint32_t*>(mapped);
	}
};

struct Files {
	size_t binaries  = 0;
	size_t temporary = 0;
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

bool Restore(GraphicContext& graphics, PipelineCache::Pipeline& pipeline, const ShaderModule& module) {
	return FindComputePipeline(graphics, pipeline, module, nullptr);
}

vk::Result Create(GraphicContext& graphics, PipelineCache::Pipeline& pipeline, const ShaderModule& module,
                  ComputePipelineCacheMode mode) {
	return CreateComputePipeline(graphics, pipeline, module, nullptr, mode);
}

void RefusedFile(GraphicContext& graphics, Dispatcher& dispatcher, const ShaderModule& module, const fs::path& path,
                 const std::vector<char>& bytes, const std::string& what, uint32_t seed) {
	WriteFile(path, bytes);
	PipelineCache::Pipeline pipeline;
	Check(!Restore(graphics, pipeline, module), "a " + what + " file was restored");
	Check(!pipeline.pipeline, "a refused " + what + " file left a pipeline behind");
	Destroy(graphics, pipeline);
	Check(Create(graphics, pipeline, module, ComputePipelineCacheMode::Full) == vk::Result::eSuccess,
	      "the pipeline was not compiled again after a " + what + " file");
	Check(dispatcher.Run(pipeline, seed) == Expected(seed, BigSteps), "the pipeline compiled after a " + what +
	                                                                        " file computes a wrong value");
	Destroy(graphics, pipeline);
	Check(Restore(graphics, pipeline, module), "the file written again after a " + what + " file is not restored");
	Destroy(graphics, pipeline);
}

int RunFallbackOnly(GraphicContext& graphics, Dispatcher& dispatcher, const ShaderModule& big, const fs::path& directory) {
	PipelineCache::Pipeline pipeline;
	Check(!Restore(graphics, pipeline, big), "a lookup without binaries or a Vulkan pipeline cache succeeded");
	Destroy(graphics, pipeline);
	Check(Create(graphics, pipeline, big, ComputePipelineCacheMode::Full) == vk::Result::eSuccess,
	      "the large pipeline was not created without binaries");
	Check(dispatcher.Run(pipeline, 7u) == Expected(7u, BigSteps), "the large pipeline computes a wrong value");
	Destroy(graphics, pipeline);
	Check(ListFiles(directory).binaries == 0, "a binary file was written without the extensions");
	if (g_failures != 0) return 1;
	std::printf("%s: ok (compute pipeline binaries unavailable on %s: fallback path only)\n", Name,
	            graphics.physical_device_properties.deviceName.data());
	return 0;
}

int Run(GraphicContext& graphics) {
	const auto directory = fs::temp_directory_path() / ("kyty-compute-binaries-" + std::to_string(
	                                                        std::chrono::steady_clock::now().time_since_epoch().count()));
	std::error_code error;
	fs::remove_all(directory, error);
	const bool binaries = graphics.compute_pipeline_binaries;
	graphics.compute_pipeline_binary_directory = directory;

	const auto  big_code   = ChainShader(BigSteps);
	const auto  small_code = ChainShader(SmallSteps);
	ShaderModule big;
	ShaderModule small;
	MakeModule(graphics, big, big_code, 0x5100);
	MakeModule(graphics, small, small_code, 0x5101);
	Require(big.spirv_words >= 50'000 && small.spirv_words < 50'000, "the test modules have the wrong sizes");
	int result = 0;
	{
		Dispatcher dispatcher(graphics);
		if (!binaries) {
			result = RunFallbackOnly(graphics, dispatcher, big, directory);
		} else {
			PipelineCache::Pipeline pipeline;
			const auto              compile_started = std::chrono::steady_clock::now();
			Check(Create(graphics, pipeline, big, ComputePipelineCacheMode::Full) == vk::Result::eSuccess,
			      "the large pipeline was not created");
			const auto compile_ms = Milliseconds(std::chrono::steady_clock::now() - compile_started);
			Check(dispatcher.Run(pipeline, 3u) == Expected(3u, BigSteps), "the captured pipeline computes a wrong value");
			Destroy(graphics, pipeline);
			const auto files = ListFiles(directory);
			Require(files.binaries == 1 && files.temporary == 0,
			        "the capture wrote " + std::to_string(files.binaries) + " binary files and left " +
			            std::to_string(files.temporary) + " temporary files");
			const auto path      = files.last;
			const auto file_size = fs::file_size(path);

			const auto restore_started = std::chrono::steady_clock::now();
			Check(Restore(graphics, pipeline, big), "the large pipeline was not restored from its binary");
			const auto restore_ms = Milliseconds(std::chrono::steady_clock::now() - restore_started);
			Check(dispatcher.Run(pipeline, 11u) == Expected(11u, BigSteps),
			      "the restored pipeline computes a wrong value");
			Destroy(graphics, pipeline);

			const auto old_time = fs::file_time_type::clock::now() - std::chrono::hours(1);
			fs::last_write_time(path, old_time);
			Check(Create(graphics, pipeline, big, ComputePipelineCacheMode::Full) == vk::Result::eSuccess,
			      "a full creation with a valid binary failed");
			Destroy(graphics, pipeline);
			Check(fs::last_write_time(path) == old_time, "a full creation compiled although a valid binary exists");
			Check(Create(graphics, pipeline, big, ComputePipelineCacheMode::KnownMiss) == vk::Result::eSuccess,
			      "a creation for a known miss failed");
			Destroy(graphics, pipeline);
			Check(fs::last_write_time(path) != old_time, "a creation for a known miss read the binary instead of compiling");

			const auto valid = ReadFile(path);
			Require(valid.size() == file_size, "the binary file changed size when written again");
			auto flipped = valid;
			flipped.back() = static_cast<char>(flipped.back() ^ 0x5a);
			RefusedFile(graphics, dispatcher, big, path, flipped, "corrupted", 13u);
			RefusedFile(graphics, dispatcher, big, path, {valid.begin(), valid.end() - 1}, "truncated", 17u);
			RefusedFile(graphics, dispatcher, big, path, {valid.begin(), valid.begin() + HeaderSize}, "header-only", 19u);
			RefusedFile(graphics, dispatcher, big, path, {}, "empty", 23u);
			auto other_key = valid;
			other_key[8 + KeySize + 4] = static_cast<char>(other_key[8 + KeySize + 4] ^ 0x01);
			RefusedFile(graphics, dispatcher, big, path, other_key, "foreign-key", 29u);
			auto bad_magic = valid;
			bad_magic[0] = 'X';
			RefusedFile(graphics, dispatcher, big, path, bad_magic, "foreign-magic", 31u);

			Check(Create(graphics, pipeline, small, ComputePipelineCacheMode::Full) == vk::Result::eSuccess,
			      "the small pipeline was not created");
			Check(dispatcher.Run(pipeline, 5u) == Expected(5u, SmallSteps), "the small pipeline computes a wrong value");
			Destroy(graphics, pipeline);
			Check(ListFiles(directory).binaries == 1, "a module below 50,000 words was written as a binary");

			graphics.compute_pipeline_binaries = false;
			Check(!Restore(graphics, pipeline, big), "a binary was restored while the extensions are off");
			Destroy(graphics, pipeline);
			Check(Create(graphics, pipeline, big, ComputePipelineCacheMode::Full) == vk::Result::eSuccess,
			      "the large pipeline was not created without binaries");
			Check(dispatcher.Run(pipeline, 37u) == Expected(37u, BigSteps),
			      "the pipeline created without binaries computes a wrong value");
			Destroy(graphics, pipeline);
			graphics.compute_pipeline_binaries = true;

			fs::remove(path);
			constexpr uint32_t                     Writers = 4;
			std::vector<PipelineCache::Pipeline>   created(Writers);
			std::vector<vk::Result>                results(Writers, vk::Result::eErrorUnknown);
			std::vector<std::thread>               threads;
			for (uint32_t i = 0; i < Writers; i++) {
				threads.emplace_back([&, i] {
					results[i] = Create(graphics, created[i], big, ComputePipelineCacheMode::KnownMiss);
				});
			}
			for (auto& thread: threads) thread.join();
			for (uint32_t i = 0; i < Writers; i++) {
				Check(results[i] == vk::Result::eSuccess, "a concurrent creation failed");
				Destroy(graphics, created[i]);
			}
			const auto concurrent = ListFiles(directory);
			Check(concurrent.binaries == 1 && concurrent.temporary == 0,
			      "concurrent captures left " + std::to_string(concurrent.binaries) + " binary files and " +
			          std::to_string(concurrent.temporary) + " temporary files");
			Check(Restore(graphics, pipeline, big), "the binary written by concurrent captures is not restored");
			Check(dispatcher.Run(pipeline, 41u) == Expected(41u, BigSteps),
			      "the pipeline restored after concurrent captures computes a wrong value");
			Destroy(graphics, pipeline);

			if (g_failures == 0) {
				std::printf("%s: ok (%s, %u words, file %ju bytes, compile %.1f ms, restore %.1f ms)\n", Name,
				            graphics.physical_device_properties.deviceName.data(), big.spirv_words,
				            static_cast<uintmax_t>(file_size), compile_ms, restore_ms);
			}
			result = g_failures == 0 ? 0 : 1;
		}
	}
	graphics.device.destroyShaderModule(big.module, nullptr);
	graphics.device.destroyShaderModule(small.module, nullptr);
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

extern "C" int RunKytyComputePipelineBinariesTest() {
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
		std::fprintf(stderr, "%s: the emulator's device is unavailable (%s), testing the fallback on a plain device\n",
		             Name, error.c_str());
		return Run(plain.graphics);
	} catch (const std::exception& error) {
		std::fprintf(stderr, "FAILED: %s: %s\n", Name, error.what());
		return 1;
	}
}

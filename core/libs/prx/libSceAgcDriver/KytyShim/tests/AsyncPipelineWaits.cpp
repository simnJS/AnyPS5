#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/stallSnapshot.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/window.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

namespace Libs::Graphics {

struct PipelineCacheTestAccess {
	using Clock = std::chrono::steady_clock;

	static auto HoldCompilationSlot(PipelineCache& cache) { return cache.AdmitWorker({}, 0, false); }
	static bool AsyncWorkersRunning(const PipelineCache& cache) { return !cache.m_async_threads.empty(); }
	static void QueueComputeAhead(PipelineCache& cache, const ShaderModule& module) {
		cache.CreateComputePipelineAhead(module);
	}
	static void MarkGraphicsPending(PipelineCache& cache, const GraphicsPipelineKey& key) {
		std::lock_guard lock(cache.m_async_mutex);
		cache.m_async_pending_graphics.insert(key);
	}
	static void FinishGraphics(PipelineCache& cache, const GraphicsPipelineKey& key) {
		{
			std::lock_guard lock(cache.m_async_mutex);
			cache.m_async_failed_graphics.insert(key);
			cache.m_async_pending_graphics.erase(key);
		}
		cache.m_async_finished.notify_all();
	}
	static bool WaitGraphics(PipelineCache& cache, const GraphicsPipelineKey& key,
	                         std::optional<Clock::time_point> until) {
		std::unique_lock lock(cache.m_async_mutex);
		return cache.WaitGraphicsPipeline(lock, key, until);
	}
	static void StopAsync(PipelineCache& cache) { cache.StopAsyncPipelines(); }
};

}

namespace {

using namespace Libs::Graphics;
using Access = PipelineCacheTestAccess;
using namespace std::chrono_literals;

constexpr const char* Name = "kyty_shim_async_pipeline_waits";

constexpr auto Asleep  = 300ms;
constexpr auto Woken   = 10s;
constexpr double MaxCpuMs = 60.0;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

void Require(bool condition, const std::string& what) {
	if (!condition) throw std::runtime_error(what);
}

double ThreadCpuMs() {
#if defined(_WIN32)
	FILETIME created {}, exited {}, kernel {}, user {};
	if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)) return 0.0;
	const auto ticks = [](const FILETIME& time) {
		return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
	};
	return static_cast<double>(ticks(kernel) + ticks(user)) / 1e4;
#else
	timespec now {};
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &now);
	return static_cast<double>(now.tv_sec) * 1e3 + static_cast<double>(now.tv_nsec) / 1e6;
#endif
}

struct Gate {
	PFN_vkCreateComputePipelines   real = nullptr;
	std::mutex                     mutex;
	std::condition_variable        changed;
	bool                           open    = true;
	uint32_t                       entered = 0;
	std::vector<std::thread::id>   creators;
};

Gate g_gate;

VKAPI_ATTR VkResult VKAPI_CALL GatedCreateComputePipelines(VkDevice device, VkPipelineCache cache, uint32_t count,
                                                           const VkComputePipelineCreateInfo* infos,
                                                           const VkAllocationCallbacks*       allocator,
                                                           VkPipeline*                        pipelines) {
	{
		std::unique_lock lock(g_gate.mutex);
		g_gate.entered++;
		g_gate.creators.push_back(std::this_thread::get_id());
		g_gate.changed.notify_all();
		g_gate.changed.wait(lock, [] { return g_gate.open; });
	}
	return g_gate.real(device, cache, count, infos, allocator, pipelines);
}

void SetGate(bool open) {
	{
		std::lock_guard lock(g_gate.mutex);
		g_gate.open = open;
	}
	g_gate.changed.notify_all();
}

bool Created(std::thread::id thread) {
	std::lock_guard lock(g_gate.mutex);
	return std::ranges::find(g_gate.creators, thread) != g_gate.creators.end();
}

std::vector<uint32_t> EmptyComputeShader() {
	spvtools::SpirvTools  tools(SPV_ENV_VULKAN_1_3);
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

struct Module {
	GraphicContext&        graphics;
	ShaderModule           module;
	ShaderComputeInputInfo input;

	Module(GraphicContext& context, const std::vector<uint32_t>& code, uint64_t id): graphics(context) {
		module.info.stage  = ShaderType::Compute;
		module.spirv_words = static_cast<uint32_t>(code.size());
		module.id          = id;
		const vk::ShaderModuleCreateInfo info {.codeSize = code.size() * sizeof(uint32_t), .pCode = code.data()};
		Require(graphics.device.createShaderModule(&info, nullptr, &module.module) == vk::Result::eSuccess,
		        "could not create the shader module");
		input.stage.module = &module;
	}
	~Module() { graphics.device.destroyShaderModule(module.module, nullptr); }
	Module(const Module&)            = delete;
	Module& operator=(const Module&) = delete;
};

struct ComputeWait {
	const PipelineCache::Pipeline* pipeline = nullptr;
	std::thread::id                thread;
	double                         cpu_ms = 0.0;
};

std::future<ComputeWait> WaitCompute(PipelineCache& cache, const Module& module) {
	return std::async(std::launch::async, [&cache, &module] {
		ComputeWait wait;
		wait.thread      = std::this_thread::get_id();
		const auto start = ThreadCpuMs();
		wait.pipeline    = &cache.GetComputePipeline(module.input);
		wait.cpu_ms      = ThreadCpuMs() - start;
		return wait;
	});
}

struct GraphicsWait {
	bool   finished = false;
	double cpu_ms   = 0.0;
};

std::future<GraphicsWait> WaitGraphics(PipelineCache& cache, const GraphicsPipelineKey& key) {
	return std::async(std::launch::async, [&cache, key] {
		GraphicsWait wait;
		const auto   start = ThreadCpuMs();
		wait.finished      = Access::WaitGraphics(cache, key, std::nullopt);
		wait.cpu_ms        = ThreadCpuMs() - start;
		return wait;
	});
}

void ComputeWaitWokenByWorker(PipelineCache& cache, GraphicContext& graphics, const std::vector<uint32_t>& code) {
	const Module module(graphics, code, 0xa5c0001u);
	SetGate(false);
	uint32_t entered = 0;
	{
		std::lock_guard lock(g_gate.mutex);
		entered = g_gate.entered;
	}
	auto held = Access::HoldCompilationSlot(cache);
	Require(static_cast<bool>(held), "could not reserve the compilation slot");
	Access::QueueComputeAhead(cache, module.module);
	{
		std::unique_lock lock(g_gate.mutex);
		Check(!g_gate.changed.wait_for(lock, Asleep, [&] { return g_gate.entered > entered; }),
		      "a queued compute pipeline bypassed the occupied compilation slot");
	}
	held.Reset();
	{
		std::unique_lock lock(g_gate.mutex);
		Require(g_gate.changed.wait_for(lock, Woken, [&] { return g_gate.entered > entered; }),
		        "the async worker did not start the queued compute pipeline");
	}
	auto waiter = WaitCompute(cache, module);
	Check(waiter.wait_for(Asleep) == std::future_status::timeout,
	      "the compute wait returned while the worker was still creating the pipeline");
	SetGate(true);
	Require(waiter.wait_for(Woken) == std::future_status::ready,
	        "the compute wait did not wake when the worker finished");
	const auto wait = waiter.get();
	Check(wait.pipeline != nullptr && wait.pipeline->pipeline, "the compute wait returned no pipeline");
	Check(!Created(wait.thread), "the waiting dispatch created the pipeline itself instead of taking the worker's");
	Check(wait.cpu_ms < MaxCpuMs, "the compute wait used " + std::to_string(wait.cpu_ms) + " ms of CPU: it spins");
}

PFN_vkCreateShaderModule g_create_shader_module = nullptr;
thread_local StallSnapshot::CompileWait* t_translation = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL ObserveShaderModule(VkDevice device, const VkShaderModuleCreateInfo* info,
                                                    const VkAllocationCallbacks* allocator, VkShaderModule* module) {
	if (t_translation != nullptr) *t_translation = StallSnapshot::LongestCompileWait();
	return g_create_shader_module(device, info, allocator, module);
}

void TranslateBeforeWorkerAdmission(PipelineCache& cache) {
	static constexpr uint32_t code[] = {0xbf810000u};
	HW::ComputeShaderInfo regs {};
	regs.cs_regs.data_addr = reinterpret_cast<uintptr_t>(code);
	regs.cs_regs.wave_size = 32;
	regs.cs_regs.num_thread_x = regs.cs_regs.num_thread_y = regs.cs_regs.num_thread_z = 1;
	ShaderMapUserData(regs.cs_regs.data_addr, {.type = Prospero::ShaderBinaryType::kCs, .code_size_bytes = sizeof(code)});
	auto held = Access::HoldCompilationSlot(cache);
	Require(static_cast<bool>(held), "could not reserve the translation slot");
	ShaderComputeInputInfo input {};
	ProgramResolution ahead;
	Require(cache.GetComputeProgramAhead(regs, input, ahead) == nullptr && ahead.reads.Refusal() == AheadRefusal::Compiling,
	        "the program was not queued for translation");
	StallSnapshot::CompileWait translation;
	auto waiter = std::async(std::launch::async, [&] {
		t_translation = &translation;
		const auto* module = cache.GetComputeProgram(regs, input);
		t_translation = nullptr;
		return module;
	});
	const bool ready = waiter.wait_for(2s) == std::future_status::ready;
	held.Reset();
	Check(ready, "foreground translation waited for an unstarted worker");
	const auto* module = waiter.get();
	Require(module != nullptr, "foreground translation returned no module");
	Check(translation.count == 1 && translation.hash == ShaderHashAt(regs.cs_regs.data_addr) &&
	          translation.what != nullptr && std::string_view(translation.what) == "shader-translation" &&
	          !StallSnapshot::Stuck(StallSnapshot::Holder::NotIssued, StallSnapshot::StuckPeriods, translation),
	      "the watchdog did not recognize the foreground translation");
	Check(StallSnapshot::LongestCompileWait().count == 0, "the completed translation remained in the watchdog");
	Check(cache.GetComputeProgram(regs, input) == module, "the translated module was not reused");
}

void GraphicsWaitWokenAtTheEnd(PipelineCache& cache) {
	GraphicsPipelineKey key {};
	key.stage_ids = {0x9a0001u, 0, 0, 0x9a0002u};
	Access::MarkGraphicsPending(cache, key);
	const auto timed_from = std::chrono::steady_clock::now();
	Check(!Access::WaitGraphics(cache, key, timed_from + 50ms), "a timed graphics wait reported a pipeline it never got");
	Check(std::chrono::steady_clock::now() - timed_from >= 45ms, "a timed graphics wait returned before its deadline");
	auto waiter = WaitGraphics(cache, key);
	Check(waiter.wait_for(Asleep) == std::future_status::timeout,
	      "the graphics wait returned while its pipeline was still pending");
	Access::FinishGraphics(cache, key);
	Require(waiter.wait_for(Woken) == std::future_status::ready,
	        "the graphics wait did not wake when its pipeline left the pending set");
	const auto wait = waiter.get();
	Check(wait.finished, "the graphics wait did not report its pipeline as finished");
	Check(wait.cpu_ms < MaxCpuMs, "the graphics wait used " + std::to_string(wait.cpu_ms) + " ms of CPU: it spins");
	const auto ready_from = std::chrono::steady_clock::now();
	Check(Access::WaitGraphics(cache, key, ready_from + Woken) && std::chrono::steady_clock::now() - ready_from < 1s,
	      "a timed graphics wait on a finished pipeline slept");
}

void StopWakesEveryWait(PipelineCache& cache, GraphicContext& graphics, const std::vector<uint32_t>& code) {
	const Module        module(graphics, code, 0xa5c0002u);
	GraphicsPipelineKey key {};
	key.stage_ids = {0x9b0001u, 0, 0, 0x9b0002u};
	Access::MarkGraphicsPending(cache, key);
	auto held = Access::HoldCompilationSlot(cache);
	Access::QueueComputeAhead(cache, module.module);
	auto graphics_waiter = WaitGraphics(cache, key);
	auto compute_waiter  = WaitCompute(cache, module);
	Check(graphics_waiter.wait_for(Asleep) == std::future_status::timeout,
	      "the graphics wait returned before the stop");
	Check(compute_waiter.wait_for(0ms) == std::future_status::timeout, "the compute wait returned before the stop");
	Access::StopAsync(cache);
	Require(graphics_waiter.wait_for(Woken) == std::future_status::ready, "the stop did not wake the graphics wait");
	Require(compute_waiter.wait_for(Woken) == std::future_status::ready, "the stop did not wake the compute wait");
	Check(graphics_waiter.get().finished, "the graphics wait woken by the stop reported a timeout");
	const auto compute = compute_waiter.get();
	Check(compute.pipeline != nullptr && compute.pipeline->pipeline,
	      "the compute wait woken by the stop did not create its pipeline");
	Check(compute.cpu_ms < MaxCpuMs + 200.0,
	      "the compute wait woken by the stop used " + std::to_string(compute.cpu_ms) + " ms of CPU");
}

struct NoImageHeap final: ShaderRecompiler::IR::ImageHeapSource {
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest&, ShaderRecompiler::IR::ImageHeapView&) override {
		return false;
	}
};

struct Device {
	GraphicContext graphics;

	~Device() {
		if (graphics.image_heap_layout) graphics.device.destroyDescriptorSetLayout(graphics.image_heap_layout, nullptr);
		if (graphics.device) graphics.device.destroy(nullptr);
		if (graphics.instance) graphics.instance.destroy(nullptr);
	}
};

int Run(GraphicContext& graphics) {
	Require(Config::AsyncPipelinesEnabled(), "the test needs KYTY_ASYNC_PIPELINES on");
	const auto code = EmptyComputeShader();
	graphics.image_heap_layout = ImageHeap::CreateLayout(graphics.device);
	g_create_shader_module = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateShaderModule;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateShaderModule = ObserveShaderModule;
	g_gate.real = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines = GatedCreateComputePipelines;
	{
		NoImageHeap   heap;
		PipelineCache cache(graphics, &heap, {}, true, 1, 1);
		Require(Access::AsyncWorkersRunning(cache), "the pipeline cache started no async workers");
		TranslateBeforeWorkerAdmission(cache);
		ComputeWaitWokenByWorker(cache, graphics, code);
		GraphicsWaitWokenAtTheEnd(cache);
		StopWakesEveryWait(cache, graphics, code);
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines = g_gate.real;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateShaderModule = g_create_shader_module;
	if (g_failures != 0) return 1;
	std::printf("%s: ok\n", Name);
	return 0;
}

}

extern "C" int RunKytyAsyncPipelineWaitsTest() {
	try {
		const Common::SoftExitScope soft_exit;
		Common::InitializeThreads();
		KytyShim::StartRuntime();
		Device      device;
		std::string error;
		if (!CreateHeadlessVulkan(device.graphics, nullptr, error)) {
			const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr &&
			                      !error.starts_with("missing required Vulkan device extension ");
			std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s)\n", Name, required ? "FAILED" : "skipped",
			             error.c_str());
			return required ? 1 : 77;
		}
		return Run(device.graphics);
	} catch (const std::exception& error) {
		std::fprintf(stderr, "FAILED: %s: %s\n", Name, error.what());
		return 1;
	}
}

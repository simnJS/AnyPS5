#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace Libs::Graphics;

namespace {

constexpr const char* Name         = "kyty_shim_submit_ahead";
constexpr uint64_t    Page         = 0x4000;
constexpr uint32_t    Programs     = 4;
constexpr uint32_t    DispatchMode = 0x41;
constexpr uint32_t    Format32UInt = 20;

void Require(bool passed, const std::string& description) {
	if (!passed) throw std::runtime_error(description);
}

std::array<uint32_t, 3> DistinctShader(uint32_t literal) {
	return {0x7e0002ffu, literal, 0xbf810000u};
}

void AppendDispatch(std::vector<uint32_t>& commands, uint64_t code, uint64_t output) {
	constexpr uint32_t Control = (0x01016facu & ~(0x7fu << 12u)) | (Format32UInt << 12u) | (3u << 28u);
	commands.insert(commands.end(), {
	    KYTY_PM4(4, Pm4::IT_SET_SH_REG, 0), 0x20c, static_cast<uint32_t>(code >> 8u), static_cast<uint32_t>(code >> 40u),
	    KYTY_PM4(3, Pm4::IT_SET_SH_REG, 0), 0x212, 0,
	    KYTY_PM4(3, Pm4::IT_SET_SH_REG, 0), 0x213, 4u << 1u,
	    KYTY_PM4(5, Pm4::IT_SET_SH_REG, 0), 0x207, 1, 1, 1,
	    KYTY_PM4(6, Pm4::IT_SET_SH_REG, 0), 0x240, static_cast<uint32_t>(output),
	    static_cast<uint32_t>((output >> 32u) & 0xffffu), sizeof(uint32_t), Control,
	    KYTY_PM4(5, Pm4::IT_DISPATCH_DIRECT, 0), 1, 1, 1, DispatchMode,
	});
}

void AppendIndirectBuffer(std::vector<uint32_t>& commands, uint64_t address, uint32_t dwords, bool chain) {
	commands.insert(commands.end(), {KYTY_PM4(4, Pm4::IT_INDIRECT_BUFFER, 0), static_cast<uint32_t>(address),
	                                 static_cast<uint32_t>(address >> 32u), dwords | (chain ? 1u << 20u : 0u)});
}

struct Found {
	uint64_t code       = 0;
	uint32_t user_sgprs = 0;
	uint32_t user_data0 = 0;
	uint8_t  wave_size  = 0;
	uint32_t groups[3]  = {};
};

struct RecordingSink final: SubmitScanSink {
	std::vector<Found> dispatches;
	std::vector<uint64_t> draws;
	void Draw(CommandProcessor& shadow, const IndirectDrawRegisters*) override {
		draws.push_back(shadow.GetShCtx().GetPs().ps_regs.data_addr);
	}
	void Dispatch(CommandProcessor& shadow, const SubmitScanDispatch& dispatch) override {
		const auto& cs = shadow.GetShCtx().GetCs();
		dispatches.push_back({.code       = cs.cs_regs.data_addr,
		                      .user_sgprs = cs.cs_regs.user_sgpr,
		                      .user_data0 = cs.cs_user_sgpr.value[0],
		                      .wave_size  = cs.cs_regs.wave_size,
		                      .groups     = {dispatch.thread_groups[0], dispatch.thread_groups[1],
		                                     dispatch.thread_groups[2]}});
	}
};

void ScanFollowsTheStream(RenderContext& renderer, uint64_t code, uint64_t buffers, uint64_t output) {
	auto* const called  = reinterpret_cast<uint32_t*>(buffers);
	auto* const chained = reinterpret_cast<uint32_t*>(buffers + 0x1000);
	std::vector<uint32_t> inner;
	AppendDispatch(inner, code + 0x100, output + 4);
	std::memcpy(called, inner.data(), inner.size() * sizeof(uint32_t));
	std::vector<uint32_t> tail;
	AppendDispatch(tail, code + 0x200, output + 8);
	tail.insert(tail.end(), {KYTY_PM4(2, Pm4::IT_REWIND, 0), 0});
	AppendDispatch(tail, code + 0x300, output + 12);
	std::memcpy(chained, tail.data(), tail.size() * sizeof(uint32_t));
	std::vector<uint32_t> commands;
	AppendDispatch(commands, code, output);
	AppendIndirectBuffer(commands, buffers, static_cast<uint32_t>(inner.size()), false);
	commands.insert(commands.end(), {KYTY_PM4(3, Pm4::IT_SET_SH_REG, 0), Pm4::SPI_SHADER_PGM_LO_PS,
	                                 static_cast<uint32_t>(0x123400ull >> 8u)});
	commands.insert(commands.end(), {KYTY_PM4(3, Pm4::IT_DRAW_INDEX_AUTO, 0), 3, 2});
	AppendIndirectBuffer(commands, buffers + 0x1000, static_cast<uint32_t>(tail.size()), true);
	AppendDispatch(commands, code + 0x300, output + 12);

	CommandProcessor shadow(renderer, 0);
	shadow.Reset();
	RecordingSink sink;
	const auto result = ScanSubmission(commands, shadow, renderer, sink);
	Require(result.end == SubmitScanEnd::Rewind, "the scan did not stop at the rewind of the chained buffer");
	Require(sink.dispatches.size() == 3 && result.dispatches == 3,
	        "the scan found " + std::to_string(sink.dispatches.size()) + " dispatches instead of 3");
	for (uint32_t i = 0; i < 3; i++) {
		const auto& found = sink.dispatches[i];
		Require(found.code == code + 0x100u * i, "dispatch " + std::to_string(i) + " has the wrong program");
		Require(found.user_sgprs == 4 && found.user_data0 == static_cast<uint32_t>(output + 4u * i),
		        "dispatch " + std::to_string(i) + " has the wrong user data");
		Require(found.wave_size == Pm4::ComputeWaveSize(DispatchMode), "the dispatch wave size was not applied");
		Require(found.groups[0] == 1 && found.groups[1] == 1 && found.groups[2] == 1, "wrong thread groups");
	}
	Require(sink.draws.size() == 1 && result.draws == 1 && sink.draws[0] == 0x123400ull,
	        "the draw did not see the pixel program set before it");

	const std::vector<uint32_t> unmapped {KYTY_PM4(4, Pm4::IT_INDIRECT_BUFFER, 0), 0x1000u, 0x7fu, 16u};
	RecordingSink nothing;
	Require(ScanSubmission(unmapped, shadow, renderer, nothing).end == SubmitScanEnd::Unreadable &&
	            nothing.dispatches.empty(),
	        "an indirect buffer outside mapped memory was read");
	const std::vector<uint32_t> unknown {KYTY_PM4(3, Pm4::IT_SET_SH_REG, 0), Pm4::SH_NUM, 1u};
	Require(ScanSubmission(unknown, shadow, renderer, nothing).end == SubmitScanEnd::Error,
	        "a register the command processor rejects did not end the scan");
	Require(shadow.GetShCtx().GetCs().cs_regs.data_addr == 0, "the shadow kept its registers after an error");
	const std::vector<uint32_t> first(commands.begin(), commands.begin() + 26);
	RecordingSink again;
	Require(ScanSubmission(first, shadow, renderer, again).end == SubmitScanEnd::Complete && again.dispatches.size() == 1 &&
	            again.dispatches[0].code == code,
	        "a scan after an error did not find its dispatch");
	std::printf("%s: scan found 3 dispatches and 1 draw through a call, a chain and a rewind\n", Name);
}

struct CountingSink final: SubmitScanSink {
	uint64_t found = 0;
	void     Draw(CommandProcessor&, const IndirectDrawRegisters*) override { found++; }
	void     Dispatch(CommandProcessor&, const SubmitScanDispatch&) override { found++; }
};

void ScanCost(RenderContext& renderer, uint64_t code, uint64_t output) {
	constexpr uint32_t    Dispatches = 4096;
	constexpr uint32_t    Rounds     = 20;
	std::vector<uint32_t> commands;
	for (uint32_t i = 0; i < Dispatches; i++) {
		AppendDispatch(commands, code + 0x100u * (i % Programs), output + 4u * (i % Programs));
	}
	CommandProcessor shadow(renderer, 0);
	shadow.Reset();
	CountingSink sink;
	const auto   begin = std::chrono::steady_clock::now();
	for (uint32_t round = 0; round < Rounds; round++) {
		Require(ScanSubmission(commands, shadow, renderer, sink).end == SubmitScanEnd::Complete,
		        "the cost scan did not complete");
	}
	const auto ns = static_cast<double>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
	Require(sink.found == uint64_t {Dispatches} * Rounds, "the cost scan missed dispatches");
	std::printf("%s: scan cost %.1f ns per dispatch, %.2f ns per dword (%u dispatches of %zu dwords, %u rounds)\n", Name,
	            ns / static_cast<double>(sink.found), ns / static_cast<double>(commands.size() * Rounds), Dispatches,
	            commands.size() / Dispatches, Rounds);
}

PipelineCache::SubmitAheadCounts WaitForCounts(PipelineCache& cache, bool (*done)(const PipelineCache::SubmitAheadCounts&)) {
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
	for (;;) {
		const auto counts = cache.GetSubmitAheadCounts();
		if (done(counts) || std::chrono::steady_clock::now() > deadline) return counts;
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
}

void PrintCounts(const char* when, const PipelineCache::SubmitAheadCounts& counts) {
	std::printf("%s: %s: requested %llu (compute %llu), finished %llu, used %llu (ready %llu, compiling %llu), known %llu, "
	            "refused %llu, dropped %llu\n",
	            Name, when, static_cast<unsigned long long>(counts.requested_graphics + counts.requested_compute),
	            static_cast<unsigned long long>(counts.requested_compute),
	            static_cast<unsigned long long>(counts.finished),
	            static_cast<unsigned long long>(counts.used_ready + counts.used_compiling),
	            static_cast<unsigned long long>(counts.used_ready),
	            static_cast<unsigned long long>(counts.used_compiling), static_cast<unsigned long long>(counts.known),
	            static_cast<unsigned long long>(counts.refused), static_cast<unsigned long long>(counts.dropped));
}

void TranslatesBeforeTheCommandProcessor(RenderContext& renderer, uint64_t code, uint64_t label, uint64_t output) {
	renderer.InitializeGpu(nullptr);
	auto& cache = renderer.GetPipelineCache();
	auto& gpu   = renderer.GetGpu();
	const uint32_t zero = 0;
	Libs::LibKernel::Memory::WriteBacking(label, &zero, sizeof(zero));
	std::vector<uint32_t> commands {
	    KYTY_PM4(7, Pm4::IT_WAIT_REG_MEM, 0), 0x13u, static_cast<uint32_t>(label), static_cast<uint32_t>(label >> 32u),
	    1u, 0xffffffffu, 4u,
	};
	for (uint32_t i = 0; i < Programs; i++) {
		AppendDispatch(commands, code + 0x100u * i, output + 4u * i);
	}
	gpu.Submit(commands);
	const auto translated = WaitForCounts(cache, [](const PipelineCache::SubmitAheadCounts& counts) {
		return counts.finished >= Programs;
	});
	PrintCounts("before the wait is released", translated);
	Require(translated.requested_compute == Programs && translated.requested_graphics == 0,
	        "the submission did not request one translation per program");
	Require(translated.finished == Programs, "the translations requested at submission did not finish");
	Require(translated.used_ready + translated.used_compiling == 0,
	        "a program was used while the command processor waited");
	const uint32_t one = 1;
	Libs::LibKernel::Memory::WriteBacking(label, &one, sizeof(one));
	gpu.WaitForIdle();
	const auto used = WaitForCounts(cache, [](const PipelineCache::SubmitAheadCounts& counts) {
		return counts.used_ready + counts.used_compiling >= Programs;
	});
	PrintCounts("after the dispatches", used);
	Require(used.used_ready == Programs && used.used_compiling == 0,
	        "the dispatches did not use the translations made at submission");
	Require(used.dropped == 0 && used.refused == 0, "a translation made at submission was dropped or refused");

	gpu.Submit(commands);
	gpu.WaitForIdle();
	const auto again = cache.GetSubmitAheadCounts();
	Require(again.requested_compute == Programs && again.known == used.known,
	        "a resubmission of the same programs reached the pipeline cache again");

	std::vector<uint32_t> another;
	AppendDispatch(another, code + 0x100u * Programs, output + 4u * Programs);
	gpu.Submit(another);
	gpu.WaitForIdle();
	const auto added = WaitForCounts(cache, [](const PipelineCache::SubmitAheadCounts& counts) {
		return counts.requested_compute + counts.known > Programs;
	});
	PrintCounts("after a new program", added);
	Require(added.requested_compute + added.known == Programs + 1, "a new program after known ones was not seen once");
	renderer.ShutdownGpu();
}

void Probe(GraphicContext& graphics) {
	void* guest = nullptr;
	Require(sceKernelMapFlexibleMemory(&guest, 4 * Page, 0x03, 0) == 0 && guest != nullptr,
	        "could not map synthetic guest memory");
	const auto code    = reinterpret_cast<uint64_t>(guest);
	const auto buffers = code + Page;
	const auto label   = code + 2 * Page;
	const auto output  = code + 3 * Page;
	for (uint32_t i = 0; i <= Programs; i++) {
		const auto program = DistinctShader(0x5aa50000u + i);
		std::memcpy(reinterpret_cast<void*>(code + 0x100u * i), program.data(), sizeof(program));
		ShaderMapUserData(code + 0x100u * i,
		                  {.type = Prospero::ShaderBinaryType::kCs, .code_size_bytes = static_cast<uint32_t>(sizeof(program))});
	}
	{
		RenderContext renderer(graphics);
		renderer.MapMemory(code, 4 * Page);
		ScanFollowsTheStream(renderer, code, buffers, output);
		ScanCost(renderer, code, output);
		TranslatesBeforeTheCommandProcessor(renderer, code, label, output);
	}
	Require(sceKernelMunmap(code, 4 * Page) == 0, "could not unmap synthetic guest memory");
}

}

extern "C" int RunKytySubmitAheadTest() {
	try {
		Common::InitializeThreads();
		Config::Initialize();
		Config::Load(Config::ConfigOptions {});
		KytyShimTests::TestVulkanDevice gpu;
		if (const auto* unavailable = gpu.Initialize(false, true)) {
			const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
			std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s)\n", Name, required ? "FAILED" : "skipped",
			             unavailable);
			return required ? 1 : 77;
		}
		Require(gpu.graphics.CreateAllocator(), "could not create the Vulkan allocator");
		Probe(gpu.graphics);
		std::printf("%s: ok\n", Name);
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "%s: FAILED: %s\n", Name, error.what());
		return 1;
	}
}

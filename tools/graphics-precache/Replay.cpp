#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/presentation/window.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/shader/shaderCompiler.h"
#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string_view>
#include <thread>

namespace Libs::Graphics {

namespace {

PFN_vkCreateGraphicsPipelines createGraphicsPipelines;
thread_local bool requireCacheHit = false;
thread_local VkResult lastResult = VK_NOT_READY;

VKAPI_ATTR VkResult VKAPI_CALL VerifyLink(VkDevice device, VkPipelineCache cache, uint32_t count,
                                          const VkGraphicsPipelineCreateInfo* infos,
                                          const VkAllocationCallbacks* allocator, VkPipeline* pipelines) {
	if (!requireCacheHit) return createGraphicsPipelines(device, cache, count, infos, allocator, pipelines);
	if (count != 1) return VK_ERROR_INITIALIZATION_FAILED;
	auto info = infos[0];
	info.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
	lastResult = createGraphicsPipelines(device, cache, 1, &info, allocator, pipelines);
	return lastResult;
}


}

struct PipelineCacheTestAccess {
	static bool Wait(PipelineCache& cache) {
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::hours(1);
		while (std::chrono::steady_clock::now() < deadline) {
			const auto backlog = cache.GetManifestBacklog();
			if (backlog.modules == 0 && backlog.running == 0 && !backlog.scanning &&
			    backlog.modules_done != 0) {
				cache.ManifestProgress(true);
				return true;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
		}
		return false;
	}

	static const ShaderModule* Find(PipelineCache& cache, ManifestModule record) {
		cache.SetHostInput(record);
		const auto key = PipelineCache::ManifestProgramKey(record, false);
		cache.LockPrograms();
		const auto* module = cache.FindCompiledModule(key, record.specialization, record.push_data_cursor, record.generic_graphics);
		cache.UnlockPrograms();
		return module;
	}

	static void Replay(GraphicContext& graphics, PipelineCache& cache, const PipelineManifest& manifest) {
		std::vector<const ShaderModule*> modules;
		for (const auto& record: manifest.modules) {
			modules.push_back(Find(cache, record));
			std::printf("module %zu %d\n", modules.size() - 1, modules.back() != nullptr);
		}
		for (size_t i = 0; i < manifest.pipelines.size(); ++i) {
			const auto& record = manifest.pipelines[i];
			auto key = record.key;
			PipelineLibraries::Stages stages {};
			bool complete = true;
			for (size_t j = 0; j < stages.size(); ++j) {
				if (record.stages[j] == ManifestPipeline::NoStage) continue;
				stages[j] = modules.at(record.stages[j]);
				if (j == 3 && record.plain_pixel && stages[j]) stages[j] = stages[j]->plain.get();
				if (stages[j]) key.stage_ids[j] = stages[j]->id;
				else complete = false;
			}
			if (!complete || !PipelineLibraries::Eligible(stages) || !cache.SampledDescriptorBudgetFits(stages)) {
				std::printf("pipeline %zu ineligible\n", i);
				continue;
			}
			PipelineCache::Pipeline pipeline {};
			CreateGraphicsPipelineLayout(graphics, pipeline, key, stages);
			if (!pipeline.pipeline_layout) {
				std::printf("pipeline %zu layout-failed\n", i);
				continue;
			}
			cache.m_libraries->PrepareNow(key, stages, pipeline.native_bindings);
			const bool linked = cache.m_libraries->Link(pipeline, key, stages);
			graphics.device.destroyPipeline(pipeline.pipeline);
			pipeline.pipeline = nullptr;
			lastResult = VK_NOT_READY;
			requireCacheHit = true;
			const bool ready = linked && cache.m_libraries->Link(pipeline, key, stages, true);
			requireCacheHit = false;
			std::printf("pipeline %zu %s %d %d\n", i, ready ? "ready" : "failed", linked, static_cast<int>(lastResult));
			std::fflush(stdout);
			graphics.device.destroyPipeline(pipeline.pipeline);
			graphics.device.destroyPipelineLayout(pipeline.pipeline_layout);
			graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout);
		}
		cache.m_libraries->LogStats();
	}
};

}

int main(int argc, char** argv) {
	std::setvbuf(stdout, nullptr, _IOLBF, 0);
	using namespace Libs::Graphics;
	const bool check = argc == 3 && std::string_view(argv[2]) == "--check-manifest";
	if (argc != 2 && !check) {
		std::fputs("usage: graphics-replay evaluation.kpm [--check-manifest]\n", stderr);
		return 2;
	}
	std::ifstream file(argv[1], std::ios::binary);
	const std::vector<uint8_t> bytes {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
	PipelineManifest manifest;
	uint32_t skipped = 0;
	if (!file || !LoadManifest(bytes, manifest, &skipped) || skipped || manifest.modules.empty()) return 2;
	if (check) {
		std::printf("modules %zu pipelines %zu\n", manifest.modules.size(), manifest.pipelines.size());
		return 0;
	}
	Common::InitializeThreads();
	KytyShim::StartRuntime();
	GraphicContext graphics;
	std::string error;
	if (!CreateHeadlessVulkan(graphics, nullptr, error) || !graphics.pipeline_library_enabled || !graphics.pipeline_cache_lookup_enabled) {
		std::fprintf(stderr, "graphics pipeline libraries unavailable: %s\n", error.c_str());
		return 3;
	}
	KytyShim::ReportNativeRayTracing(graphics);
	createGraphicsPipelines = VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateGraphicsPipelines;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateGraphicsPipelines = VerifyLink;
	graphics.image_heap_layout = ImageHeap::CreateLayout(graphics.device);
	struct OfflineHeap final: ShaderRecompiler::IR::ImageHeapSource {
		bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest&, ShaderRecompiler::IR::ImageHeapView&) override {
			return false;
		}
	} heap;
	{
		PipelineCache cache(graphics, &heap, ShaderStorePath());
		if (!PipelineCacheTestAccess::Wait(cache)) return 4;
		PipelineCacheTestAccess::Replay(graphics, cache, manifest);
	}
	graphics.device.destroyDescriptorSetLayout(graphics.image_heap_layout);
	graphics.device.destroy();
	graphics.instance.destroy();
}

#include "common/fallback.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"

#include <xxhash.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iterator>

using namespace Libs::Graphics;

namespace {

namespace IR = ShaderRecompiler::IR;
using IR::ResourceSpecialization;

struct DispatchMode {
	bool thread_dimensions;
	bool indirect_threads;
};

constexpr std::array DispatchModes {DispatchMode {false, false}, DispatchMode {true, false},
                                    DispatchMode {true, true}};

}

int PrepareComputeRecipe(const char* signatures) {
	struct Input {
		uint64_t hash;
		uint32_t threads[3];
		uint32_t rsrc1, rsrc2, modifier, scratch, words;
	} input {};
	static_assert(sizeof(Input) == 40);
	if (std::fread(&input, sizeof(input), 1, stdin) != 1 || !input.words || input.words > 16 * 1024 * 1024) return 64;
	std::vector<uint32_t> code(input.words);
	if (std::fread(code.data(), sizeof(uint32_t), code.size(), stdin) != code.size() ||
	    std::fgetc(stdin) != EOF || XXH3_64bits(code.data(), code.size() * sizeof(uint32_t)) != input.hash) return 64;

	ShaderComputeInputInfo info {};
	std::ranges::copy(input.threads, info.threads_num);
	info.wave_size = Pm4::ComputeWaveSize(input.modifier);
	info.float_mode = KYTY_PM4_GET(input.rsrc1, COMPUTE_PGM_RSRC1, FLOAT_MODE);
	info.workgroup_register = KYTY_PM4_GET(input.rsrc2, COMPUTE_PGM_RSRC2, USER_SGPR);
	info.lds_size_dwords = 128u * KYTY_PM4_GET(input.rsrc2, COMPUTE_PGM_RSRC2, LDS_SIZE);
	info.scratch_size_dwords = input.scratch;
	info.thread_ids_num = 1 + KYTY_PM4_GET(input.rsrc2, COMPUTE_PGM_RSRC2, TIDIG_COMP_CNT);
	info.group_id[0] = KYTY_PM4_GET(input.rsrc2, COMPUTE_PGM_RSRC2, TGID_X_EN);
	info.group_id[1] = KYTY_PM4_GET(input.rsrc2, COMPUTE_PGM_RSRC2, TGID_Y_EN);
	info.group_id[2] = KYTY_PM4_GET(input.rsrc2, COMPUTE_PGM_RSRC2, TGID_Z_EN);
	info.tg_size_en = KYTY_PM4_GET(input.rsrc2, COMPUTE_PGM_RSRC2, TG_SIZE_EN);
	info.host_subgroup_size = 32;
	const std::vector<uint32_t> user_data(info.workgroup_register);
	ShaderRecompiler::CompileOptions options;
	options.wave_size = info.wave_size;
	options.shader_hash = input.hash;
	options.dump_ir = false;
	options.user_data = user_data;
	options.input_info.compute = &info;
	auto source = ShaderRecompiler::PrepareShaderSource(code, options);
	if (source.HasCalls() || source.unsupported_call_pc != UINT32_MAX || source.decoded.has_swap_pc) {
		std::fprintf(stderr, "Compute recipe requires call targets\n");
		return 65;
	}
	for (const auto& inst: source.decoded.instructions) {
		if (inst.unsupported_reason.empty()) continue;
		std::fprintf(stderr, "Unsupported instruction at 0x%x: %.*s\n", inst.pc,
		             static_cast<int>(inst.unsupported_reason.size()), inst.unsupported_reason.data());
		return 65;
	}
	PipelineManifest library;
	if (signatures != nullptr) {
		std::ifstream file(signatures, std::ios::binary);
		const std::vector<uint8_t> bytes {std::istreambuf_iterator<char>(file), {}};
		uint32_t skipped = 0;
		if (!LoadManifest(bytes, library, &skipped) || library.modules.empty()) return 64;
		if (skipped != 0) std::fprintf(stderr, "Ignored %u incompatible signature records\n", skipped);
	}
	const bool queue_dependent = ShaderRecompiler::Decoder::ComputeShaderDependsOnQueue(code);
	PipelineManifest manifest;
	size_t index = 0;
	bool observed_buffers = false;
	bool observed_samplers = false;
	for (const auto& record: library.modules) {
		if (record.stage == ShaderType::Compute && record.code_hash == input.hash) {
			observed_buffers |= record.runtime_buffers;
			observed_samplers |= record.runtime_samplers;
		}
	}
	const auto prepare = [&](uint32_t push_data_cursor, bool& buffers, bool& samplers) {
		ShaderRecompiler::TranslateResult translated;
		std::vector<ResourceSpecialization> variants;
		if (!Common::Fallback::Run("incompatible compute dispatch mode", "PrepareComputeRecipe", input.hash,
		                           index, true, [&] {
			    translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
			    const auto plan = IR::ExtractResourcePlan(translated.program);
			    buffers |= IR::HasIndirectBuffers(plan);
			    samplers |= std::ranges::any_of(plan.info.samplers, [&](const auto& sampler) {
				    return sampler.source < plan.descriptor_sources.size() &&
				           plan.descriptor_sources[sampler.source].indirect_descriptor.has_value();
			    });
			    auto recipes = SynthesizeComputeVariants(info, queue_dependent, translated.program, plan,
			                                             library.modules);
			    for (auto& recipe: recipes) {
			        auto& shape = recipe.specialization;
			        if (std::ranges::find(variants, shape) == variants.end()) {
			            variants.push_back(std::move(shape));
			        }
			    }
			})) {
			samplers |= IR::TakeRuntimeSamplerStop();
			return true;
		}
		for (size_t variant = 0; variant < variants.size(); ++variant) {
			const auto variant_index = index++;
			const auto& shape = variants[variant];
			ShaderRecompiler::CompileResult result;
			if (!Common::Fallback::Run("incompatible compute view signature", "PrepareComputeRecipe", input.hash,
			                           variant_index, true, [&] {
				    if (variant != 0) translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
				    result = ShaderRecompiler::CompileProgram(std::move(translated), options, shape, push_data_cursor);
				}) || result.spirv.empty()) continue;
			std::ofstream spv("compute-" + std::to_string(variant_index) + ".spv", std::ios::binary);
			spv.write(reinterpret_cast<const char*>(result.spirv.data()), result.spirv.size() * sizeof(uint32_t));
			spv.close();
			if (!spv) return false;
			manifest.modules.push_back({.stage = ShaderType::Compute, .code_hash = input.hash,
			                            .user_data_count = static_cast<uint32_t>(user_data.size()),
			                            .push_data_cursor = push_data_cursor,
			                            .runtime_buffers = options.runtime_buffer_descriptors,
			                            .runtime_samplers = options.runtime_sampler_descriptors,
			                            .input = info, .specialization = shape});
		}
		return true;
	};
	for (const auto mode: DispatchModes) {
		info.dispatch_thread_dimensions = mode.thread_dimensions;
		info.dispatch_indirect_threads = mode.indirect_threads;
		const auto push_data_cursor = mode.indirect_threads ? IR::PushData::NoStart : 0u;
		for (const bool async: {false, true}) {
			if (async && !queue_dependent) continue;
			info.async_compute = async;
			bool buffers = observed_buffers;
			bool samplers = observed_samplers;
			for (const bool runtime_samplers: {false, true}) {
				if (runtime_samplers && !samplers) continue;
				options.runtime_sampler_descriptors = runtime_samplers;
				for (const bool runtime_buffers: {false, true}) {
					if (runtime_buffers && !buffers) continue;
					options.runtime_buffer_descriptors = runtime_buffers;
					if (!prepare(push_data_cursor, buffers, samplers)) return 74;
				}
			}
		}
	}
	if (manifest.modules.empty()) return 66;
	const auto bytes = SaveManifest(manifest);
	std::ofstream out("compute.kpm", std::ios::binary);
	out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
	out.close();
	return out ? 0 : 74;
}

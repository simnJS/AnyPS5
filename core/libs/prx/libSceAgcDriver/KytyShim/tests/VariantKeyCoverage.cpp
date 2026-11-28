#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr uint32_t ShiftLane  = 0x34020082u;
constexpr uint32_t WaitAll    = 0xbf8c0000u;
constexpr uint32_t EndProgram = 0xbf810000u;

constexpr std::array<uint32_t, 2> LoadRaw {0xe0301000u, 0x80000201u};
constexpr std::array<uint32_t, 2> StoreRaw {0xe0701000u, 0x80010201u};

constexpr uint32_t Format32UInt  = 20;
constexpr uint32_t SwizzleXyzw   = 0xfacu;
constexpr uint64_t SourceAddress = 0x200810000ull;
constexpr uint64_t TargetAddress = 0x200820000ull;
constexpr uint32_t Records       = 128u;
constexpr uint64_t CodeHash      = 0x51u;
constexpr uint32_t BoundsBits    = IR::StructuredBufferBounds | IR::ExactBufferBounds;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Sharp = std::array<uint32_t, 4>;

Sharp MakeSharp(uint64_t address, uint32_t records, uint32_t select, uint32_t stride = 0) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u),
	        records, SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (select << 28u)};
}

struct Plan {
	ShaderComputeInputInfo info {};
	IR::ResourcePlan       plan;
};

Plan Translate() {
	const std::vector<uint32_t> code {ShiftLane, LoadRaw[0], LoadRaw[1], WaitAll, StoreRaw[0], StoreRaw[1], EndProgram};
	Plan                        result;
	auto&                       info = result.info;
	info.wave_size                   = 64;
	info.host_subgroup_size          = 64;
	info.threads_num[0]              = 64;
	info.threads_num[1]              = 1;
	info.threads_num[2]              = 1;
	info.thread_ids_num              = 1;
	const std::vector<uint32_t> user_data(16, 0u);
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = CodeHash;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	result.plan                = IR::ExtractResourcePlan(translated.program);
	return result;
}

struct Request {
	bool                       ok = false;
	IR::ResourceSpecialization specialization;
};

Request Dispatch(const IR::ResourcePlan& plan, const Sharp& source, bool runtime_stride) {
	const auto               target = MakeSharp(TargetAddress, Records * 4u, 3);
	std::array<uint32_t, 16> user_data {};
	std::copy(source.begin(), source.end(), user_data.begin());
	std::copy(target.begin(), target.end(), user_data.begin() + 4);
	const IR::SrtRuntime runtime {.user_data = user_data, .runtime_stride = runtime_stride};
	IR::ResourceSnapshot snapshot;
	Request              result;
	result.ok = IR::MaterializeResources(plan, runtime, snapshot, result.specialization);
	return result;
}

uint32_t SourceSlot(const IR::ResourcePlan& plan) {
	for (uint32_t i = 0; i < plan.info.buffers.size(); i++) {
		if (plan.info.buffers[i].read && !plan.info.buffers[i].written) return i;
	}
	return UINT32_MAX;
}

ManifestModule Recipe(const Plan& program, const IR::ResourceSpecialization& specialization) {
	ManifestModule module {.stage = ShaderType::Compute, .code_hash = CodeHash, .specialization = specialization};
	module.input = program.info;
	NormalizeManifestInput(module);
	return module;
}

uint64_t Identity(const ManifestModule& module) {
	return ManifestModuleIdentity(module, {});
}

uint32_t LiveBounds(const Sharp& sharp) {
	ShaderBufferResource descriptor;
	std::copy_n(sharp.begin(), std::size(descriptor.fields), descriptor.fields);
	return IR::RuntimeStrideData(descriptor) & BoundsBits;
}

std::vector<ManifestModule> Reload(const std::vector<ManifestModule>& recipes, std::string* text = nullptr) {
	PipelineManifest written;
	written.modules  = recipes;
	const auto bytes = SaveManifest(written);
	if (text != nullptr) text->assign(bytes.begin(), bytes.end());
	PipelineManifest loaded;
	uint32_t         skipped = 0;
	Check(LoadManifest(bytes, loaded, &skipped) && skipped == 0, "the manifest reads back");
	return loaded.modules;
}

}

int main() {
	const auto program = Translate();
	const auto slot    = SourceSlot(program.plan);
	Check(slot != UINT32_MAX, "the source buffer is planned");
	if (slot == UINT32_MAX) return 1;

	struct Bound {
		const char* name;
		Sharp       sharp;
		uint32_t    live;
	};
	const std::array bound {
	    Bound {"structured records", MakeSharp(SourceAddress, Records, 0, 16), IR::StructuredBufferBounds},
	    Bound {"records by count", MakeSharp(SourceAddress, Records * 4u, 1, 16), 0u},
	    Bound {"any record", MakeSharp(SourceAddress, Records * 4u, 2), 0u},
	    Bound {"exact bytes", MakeSharp(SourceAddress, Records * 4u, 3), IR::ExactBufferBounds},
	    Bound {"no records", MakeSharp(SourceAddress, 0u, 3), IR::ExactBufferBounds},
	};

	const auto first = Dispatch(program.plan, bound.front().sharp, true);
	Check(first.ok, "the first dispatch materializes");
	if (!first.ok) return 1;
	Check(IR::HasRuntimeStride(first.specialization.buffers[slot].packed_stride),
	      "the source slot reads its stride at run time");

	auto before_bounds = first.specialization;
	for (auto& buffer: before_bounds.buffers) {
		if (IR::HasRuntimeStride(buffer.packed_stride)) buffer.packed_stride = IR::RuntimeBufferStride;
	}
	auto structured                       = before_bounds;
	structured.buffers[slot].packed_stride = IR::RuntimeBufferStride | IR::StructuredBufferBounds;
	auto exact                            = before_bounds;
	exact.buffers[slot].packed_stride     = IR::RuntimeBufferStride | IR::ExactBufferBounds;
	auto fixed                            = before_bounds;
	fixed.buffers[slot].packed_stride     = 16u | IR::StructuredBufferBounds;

	std::string text;
	const auto  loaded = Reload({Recipe(program, before_bounds), Recipe(program, structured), Recipe(program, exact),
	                             Recipe(program, fixed)},
	                            &text);
	Check(text.find("st:16383") != std::string::npos, "the manifest holds a recipe recorded before the bounds");
	Check(text.find("st:2113535") != std::string::npos, "the manifest holds a recipe recorded with structured bounds");
	Check(text.find("st:4210687") != std::string::npos, "the manifest holds a recipe recorded with exact bounds");
	Check(loaded.size() == 4, "every recipe reads back");
	if (loaded.size() != 4) return 1;

	const auto runtime_recipe = std::ranges::find_if(loaded, [&](const ManifestModule& module) {
		return IR::HasRuntimeStride(module.specialization.buffers[slot].packed_stride);
	});
	Check(runtime_recipe != loaded.end(), "a recipe that reads its stride at run time reads back");
	if (runtime_recipe == loaded.end()) return 1;
	const auto precooked = Identity(*runtime_recipe);
	uint32_t   distinct  = 0;
	uint32_t   kept      = 0;
	for (const auto& module: loaded) {
		const auto stride = module.specialization.buffers[slot].packed_stride;
		if (IR::HasRuntimeStride(stride)) {
			Check((stride & BoundsBits) == 0u, "a recipe that reads its stride at run time reads back without bounds");
			distinct += Identity(module) != precooked ? 1u : 0u;
		} else {
			Check(stride == (16u | IR::StructuredBufferBounds), "a recipe with a fixed stride reads back with its bounds");
			Check(Identity(module) != precooked, "a recipe with a fixed stride stays its own variant");
			kept++;
		}
	}
	Check(distinct == 0, "recipes that differ by run-time bounds are one variant to prepare");
	Check(kept == 1, "the fixed-stride recipe is still listed");
	const auto rewritten = Reload(loaded);
	Check(rewritten.size() == 2, "the manifest written back lists the two variants left");
	Check(std::ranges::any_of(rewritten, [&](const ManifestModule& module) { return Identity(module) == precooked; }),
	      "a recipe written back keeps its identity");

	uint32_t covered = 0;
	for (const auto& entry: bound) {
		const auto request = Dispatch(program.plan, entry.sharp, true);
		Check(request.ok, std::string(entry.name) + ": the dispatch materializes");
		if (!request.ok) continue;
		const auto stride = request.specialization.buffers[slot].packed_stride;
		Check(IR::HasRuntimeStride(stride) && (stride & BoundsBits) == 0u,
		      std::string(entry.name) + ": the bounds stay out of the key");
		Check(LiveBounds(entry.sharp) == entry.live, std::string(entry.name) + ": the stride word carries the bounds");
		const bool hit = Identity(Recipe(program, request.specialization)) == precooked;
		Check(hit, std::string(entry.name) + ": the recipe recorded before the bounds is the variant requested");
		covered += hit ? 1u : 0u;
	}

	const auto baked_structured = Dispatch(program.plan, bound[0].sharp, false);
	const auto baked_exact      = Dispatch(program.plan, bound[3].sharp, false);
	Check(baked_structured.ok && baked_exact.ok, "the fixed-stride dispatches materialize");
	if (baked_structured.ok && baked_exact.ok) {
		Check((baked_structured.specialization.buffers[slot].packed_stride & BoundsBits) == IR::StructuredBufferBounds,
		      "a fixed stride keeps structured bounds in the key");
		Check((baked_exact.specialization.buffers[slot].packed_stride & BoundsBits) == IR::ExactBufferBounds,
		      "a fixed stride keeps exact bounds in the key");
	}

	std::printf("variant key coverage: %u of %zu requests are the prepared variant\n", covered, bound.size());
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_variant_key_coverage: ok\n");
	return 0;
}

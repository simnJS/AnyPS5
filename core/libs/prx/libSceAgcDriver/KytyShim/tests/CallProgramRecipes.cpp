#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <xxhash.h>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;
namespace Decoder    = Recompiler::Decoder;

constexpr uint32_t CallThroughS2   = 0xbe822102u;
constexpr uint32_t CallThroughS4   = 0xbe842104u;
constexpr uint32_t ReturnThroughS2 = 0xbe802002u;
constexpr uint32_t ReturnThroughS4 = 0xbe802004u;
constexpr uint32_t MoveOneToS10    = 0xbe8a0381u;
constexpr uint32_t MoveTwoToS10    = 0xbe8a0382u;
constexpr uint32_t MoveTwoToS11    = 0xbe8b0382u;
constexpr uint32_t MoveS6ToS2      = 0xbe820306u;
constexpr uint32_t MoveS7ToS3      = 0xbe830307u;
constexpr uint32_t EndProgram      = 0xbf810000u;

constexpr uint64_t FirstLeaf  = 0x200900000ull;
constexpr uint64_t SecondLeaf = 0x200900100ull;

const std::vector<uint32_t> Caller {CallThroughS2, CallThroughS4, EndProgram};
const std::vector<uint32_t> FirstCode {MoveOneToS10, ReturnThroughS2};
const std::vector<uint32_t> SecondCode {MoveTwoToS11, MoveOneToS10, ReturnThroughS4};

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Memory = std::map<uint64_t, std::vector<uint32_t>>;

bool ReadMemory(void* userdata, uint64_t address, std::span<uint32_t> words) {
	const auto& memory = *static_cast<const Memory*>(userdata);
	for (const auto& [base, code]: memory) {
		const uint64_t end = base + code.size() * sizeof(uint32_t);
		if (address < base || address >= end || (address - base) % sizeof(uint32_t) != 0) continue;
		const auto first = (address - base) / sizeof(uint32_t);
		for (size_t i = 0; i < words.size(); i++) {
			words[i] = first + i < code.size() ? code[first + i] : 0u;
		}
		return true;
	}
	return false;
}

struct Stage {
	ShaderComputeInputInfo     info {};
	std::vector<uint32_t>      user_data;
	Recompiler::CompileOptions options;

	Stage(): user_data(8, 0u) {
		info.wave_size          = 64;
		info.host_subgroup_size = 64;
		info.threads_num[0]     = 64;
		info.threads_num[1]     = 1;
		info.threads_num[2]     = 1;
		info.thread_ids_num     = 1;
		options.stage              = ShaderType::Compute;
		options.wave_size          = 64;
		options.shader_hash        = XXH3_64bits(Caller.data(), Caller.size() * sizeof(uint32_t));
		options.dump_ir            = false;
		options.user_data          = user_data;
		options.input_info.compute = &info;
	}
};

std::array<uint32_t, 8> Targets(uint64_t first, uint64_t second) {
	return {0u, 0u, static_cast<uint32_t>(first), static_cast<uint32_t>(first >> 32u),
	        static_cast<uint32_t>(second), static_cast<uint32_t>(second >> 32u), 0u, 0u};
}

std::vector<uint32_t> FunctionCode(const Recompiler::ShaderSource& source) {
	const auto linked = std::span<const uint32_t>(source.linked->code).subspan(source.code.size());
	return {linked.begin(), linked.end()};
}

bool SameProgram(const Recompiler::ShaderSource& left, const Recompiler::ShaderSource& right) {
	if (!left.linked || !right.linked || left.linked->code != right.linked->code ||
	    left.linked->functions.size() != right.linked->functions.size() ||
	    left.linked->observed_function.size() != right.linked->observed_function.size() ||
	    left.linked->decoded.instructions.size() != right.linked->decoded.instructions.size()) return false;
	for (size_t i = 0; i < left.linked->functions.size(); i++) {
		if (left.linked->functions[i].offset != right.linked->functions[i].offset ||
		    left.linked->functions[i].size != right.linked->functions[i].size) return false;
	}
	for (size_t i = 0; i < left.linked->decoded.instructions.size(); i++) {
		const auto& a = left.linked->decoded.instructions[i];
		const auto& b = right.linked->decoded.instructions[i];
		if (a.opcode != b.opcode || a.pc != b.pc || a.branch_target != b.branch_target ||
		    a.word_count != b.word_count || !std::equal(a.raw, a.raw + a.word_count, b.raw)) return false;
	}
	return true;
}

std::vector<uint32_t> Compile(const Stage& stage, const Recompiler::ShaderSource& source,
                              const std::array<uint32_t, 8>& user_data) {
	auto                       translated = Recompiler::TranslateProgram(source.linked->decoded, stage.options);
	const auto                 plan       = IR::ExtractResourcePlan(translated.program);
	const IR::SrtRuntime       runtime {.user_data = user_data};
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	if (!IR::MaterializeResources(plan, runtime, snapshot, specialization)) return {};
	return Recompiler::CompileProgram(std::move(translated), stage.options, specialization, 0u).spirv;
}

std::vector<ManifestModule> Reload(const std::vector<ManifestModule>& recipes, std::string& text) {
	PipelineManifest written;
	written.modules  = recipes;
	const auto bytes = SaveManifest(written);
	text.assign(bytes.begin(), bytes.end());
	PipelineManifest loaded;
	uint32_t         skipped = 0;
	Check(LoadManifest(bytes, loaded, &skipped) && skipped == 0, "the manifest reads back");
	return loaded.modules;
}

uint32_t Unreadable(std::string text) {
	PipelineManifest loaded;
	uint32_t         skipped = 0;
	const std::vector<uint8_t> bytes(text.begin(), text.end());
	LoadManifest(bytes, loaded, &skipped);
	return skipped + (loaded.modules.empty() ? 0u : 1000u);
}

void LinkedFromTheGame() {
	Stage  stage;
	Memory memory {{FirstLeaf, FirstCode}, {SecondLeaf, SecondCode}};
	const auto user_data = Targets(FirstLeaf, SecondLeaf);
	const IR::SrtRuntime runtime {.user_data = user_data, .userdata = &memory, .read_specialization_memory = ReadMemory};

	auto live = Recompiler::PrepareShaderSource(Caller, stage.options);
	Check(live.calls.size() == 2 && !live.expand_calls && live.unsupported_call_pc == UINT32_MAX,
	      "the caller has two plain call sites");
	Recompiler::RefreshShaderSource(live, runtime);
	Check(live.linked.has_value() && live.revision == 1, "the game links the two leaves");
	const auto functions = FunctionCode(live);
	std::vector<uint32_t> expected = FirstCode;
	expected.insert(expected.end(), SecondCode.begin(), SecondCode.end());
	Check(functions == expected, "the function code is the leaves in call order");

	ManifestModule recipe {.stage = ShaderType::Compute, .code_hash = stage.options.shader_hash, .user_data_count = 8,
	                       .function_code = functions};
	recipe.input = stage.info;
	NormalizeManifestInput(recipe);
	auto plain = recipe;
	plain.function_code.clear();
	auto other = recipe;
	other.function_code.front() = MoveTwoToS10;
	Check(ManifestModuleIdentity(recipe, {}) != ManifestModuleIdentity(plain, {}) &&
	      ManifestModuleIdentity(recipe, {}) != ManifestModuleIdentity(other, {}),
	      "the function code is part of the recipe identity");

	std::string text;
	const auto  loaded = Reload({recipe, other}, text);
	Check(text.find(" fn=be8a0381be802002be8b0382be8a0381be802004") != std::string::npos,
	      "the manifest line carries the function code");
	Check(loaded.size() == 2 && std::ranges::any_of(loaded, [&](const ManifestModule& module) {
		      return module.function_code == functions;
	      }) && std::ranges::any_of(loaded, [&](const ManifestModule& module) {
		      return module.function_code == other.function_code;
	      }), "both recipes read back with their function code");

	const auto line_end = text.find('\n', text.find("\nM ") + 1);
	const auto line     = text.substr(0, line_end + 1);
	const auto fn       = line.find(" fn=");
	Check(fn != std::string::npos && Unreadable(line) == 1000u, "one recipe line reads alone");
	auto odd = line;
	odd.insert(fn + 4, "1");
	Check(Unreadable(odd) == 1u, "a function code that is not whole words is refused");
	auto pixel = line;
	pixel.replace(pixel.find(" cs "), 4, " ps ");
	Check(Unreadable(pixel) == 1u, "a function code on a graphics stage is refused");

	auto prepared = Recompiler::PrepareShaderSource(Caller, stage.options);
	Check(Recompiler::LinkShaderSource(prepared, loaded.front().function_code == functions
	                                                 ? loaded.front().function_code : loaded.back().function_code),
	      "the recipe links at startup");
	Check(SameProgram(prepared, live), "the program linked from the recipe is the one the game linked");
	const auto from_recipe = Compile(stage, prepared, user_data);
	Check(!from_recipe.empty() && from_recipe == Compile(stage, live, user_data),
	      "both compile to the same SPIR-V");

	const auto revision = prepared.revision;
	Recompiler::RefreshShaderSource(prepared, runtime);
	Check(prepared.revision == revision && SameProgram(prepared, live),
	      "the game keeps the prepared program when its leaves are unchanged");
	Check(prepared.reads.size() == 2 && prepared.reads[0].first == FirstLeaf && prepared.reads[1].first == SecondLeaf,
	      "the game still watches both leaves");

	memory[FirstLeaf].front() = MoveTwoToS10;
	Recompiler::RefreshShaderSource(prepared, runtime);
	Check(prepared.revision == revision + 1 && FunctionCode(prepared) == other.function_code,
	      "the game relinks when a leaf changed");
}

void RefusedRecipes() {
	Stage stage;
	std::vector<uint32_t> functions = FirstCode;
	functions.insert(functions.end(), SecondCode.begin(), SecondCode.end());
	const auto refused = [&](std::vector<uint32_t> code, const char* what) {
		auto source = Recompiler::PrepareShaderSource(Caller, stage.options);
		Check(!Recompiler::LinkShaderSource(source, code) && !source.linked.has_value() && source.revision == 0, what);
	};
	auto shorter = functions;
	shorter.pop_back();
	refused(shorter, "a function code that ends inside a leaf is refused");
	auto longer = functions;
	longer.push_back(MoveOneToS10);
	refused(longer, "a function code with words left over is refused");
	refused(FirstCode, "a function code with one leaf for two call sites is refused");
	auto swapped = SecondCode;
	swapped.insert(swapped.end(), FirstCode.begin(), FirstCode.end());
	refused(swapped, "a leaf that returns through another register is refused");
	refused({EndProgram, ReturnThroughS2, MoveOneToS10, ReturnThroughS4}, "a leaf that ends the program is refused");
	refused({}, "an empty function code is refused");

	const std::vector<uint32_t> plain {MoveOneToS10, EndProgram};
	auto source = Recompiler::PrepareShaderSource(plain, stage.options);
	Check(source.calls.empty() && !Recompiler::LinkShaderSource(source, functions),
	      "a program without calls takes no function code");
}

void SharedLeaf() {
	Stage  stage;
	Memory memory {{FirstLeaf, FirstCode}};
	const std::vector<uint32_t> twice {CallThroughS2, MoveS6ToS2, MoveS7ToS3, CallThroughS2, EndProgram};
	stage.options.shader_hash = XXH3_64bits(twice.data(), twice.size() * sizeof(uint32_t));
	const std::array<uint32_t, 8> user_data {0u, 0u, static_cast<uint32_t>(FirstLeaf), static_cast<uint32_t>(FirstLeaf >> 32u),
	                                         0u, 0u, static_cast<uint32_t>(FirstLeaf), static_cast<uint32_t>(FirstLeaf >> 32u)};
	const IR::SrtRuntime runtime {.user_data = user_data, .userdata = &memory, .read_specialization_memory = ReadMemory};
	auto live = Recompiler::PrepareShaderSource(twice, stage.options);
	Check(live.calls.size() == 2 && !live.expand_calls, "the caller calls twice through one register pair");
	Recompiler::RefreshShaderSource(live, runtime);
	const auto functions = FunctionCode(live);
	Check(functions == std::vector<uint32_t> {MoveOneToS10, ReturnThroughS2, MoveOneToS10, ReturnThroughS2},
	      "two calls of one leaf record it twice");
	auto prepared = Recompiler::PrepareShaderSource(twice, stage.options);
	Check(Recompiler::LinkShaderSource(prepared, functions) && SameProgram(prepared, live),
	      "the recipe of a leaf called twice links like the game");
	const auto revision = prepared.revision;
	Recompiler::RefreshShaderSource(prepared, runtime);
	Check(prepared.revision == revision && prepared.reads.size() == 1,
	      "the game keeps the prepared program and reads the shared leaf once");
}

}

int main() {
	LinkedFromTheGame();
	RefusedRecipes();
	SharedLeaf();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_call_program_recipes: %d failed\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_call_program_recipes: ok\n");
	return 0;
}

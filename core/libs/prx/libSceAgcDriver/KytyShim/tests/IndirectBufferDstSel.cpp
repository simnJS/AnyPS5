#include "common/assert.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr std::array<uint32_t, 25> RowLoopWords {
    0x7e0e0300u, 0x7e100280u, 0x7e120280u, 0x7e140280u, 0x340c0084u, 0xbe8d0380u, 0xbf0a0c0du,
    0xbf84000eu, 0x930eff0du, 0x00000078u, 0x800e880eu, 0xf4280200u, 0x1c000000u, 0xbf8cc07fu,
    0xe0082000u, 0x80020007u, 0xbf8c3f70u, 0x3a100108u, 0x3a120309u, 0x3a14050au, 0x800d810du,
    0xbf82fff0u, 0xe07c1000u, 0x80010806u, 0xbf810000u,
};

constexpr size_t GuestBytes = 0x40000u;

alignas(4096) std::array<uint32_t, GuestBytes / 4u> g_guest {};

const uint64_t GuestRegion   = reinterpret_cast<uint64_t>(g_guest.data());
const uint64_t TableAddress  = GuestRegion + 0x10000u;
const uint64_t VertexAddress = GuestRegion + 0x20000u;
const uint64_t OutputAddress = GuestRegion + 0x30000u;

constexpr uint32_t RowStride       = 120u;
constexpr uint32_t RowOffset       = 8u;
constexpr uint32_t Format32UInt    = 20u;
constexpr uint32_t Format32x3Float = 74u;
constexpr uint32_t SwizzleXyzw     = 0xfacu;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

void Poke(uint64_t address, uint32_t value) {
	g_guest[(address - GuestRegion) / 4u] = value;
}

bool ReadMemory(void*, uint64_t address, std::span<uint32_t> values) {
	for (size_t index = 0; index < values.size(); ++index) {
		const auto at = address + index * 4u;
		values[index] = at >= GuestRegion && at + 4u <= GuestRegion + GuestBytes ? g_guest[(at - GuestRegion) / 4u] : 0u;
	}
	return true;
}

constexpr uint32_t Select(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
	return x | (y << 3u) | (z << 6u) | (w << 9u);
}

using Sharp = std::array<uint32_t, 4>;

Sharp MakeSharp(uint64_t address, uint32_t records, uint32_t format, uint32_t swizzle, uint32_t select) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu), records,
	        swizzle | (format << 12u) | (1u << 24u) | (select << 28u)};
}

Sharp Vertices(uint64_t address, uint32_t swizzle) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu) | (12u << 16u), 64u,
	        swizzle | (Format32x3Float << 12u)};
}

struct Outcome {
	bool table_plan       = false;
	bool materialized     = false;
	bool indirect_failure = false;
	bool compiled         = false;
};

Outcome Run(const std::vector<Sharp>& rows, bool runtime_buffers) {
	std::fill(g_guest.begin(), g_guest.end(), 0u);
	for (uint32_t row = 0; row < rows.size(); ++row) {
		for (uint32_t word = 0; word < 4u; ++word) {
			Poke(TableAddress + RowOffset + row * RowStride + word * 4u, rows[row][word]);
		}
	}
	std::array<uint32_t, 16> user_data {};
	const auto table  = MakeSharp(TableAddress, 0x1000u, Format32UInt, SwizzleXyzw, 3);
	const auto output = MakeSharp(OutputAddress, 64u * 16u, Format32UInt, SwizzleXyzw, 3);
	std::copy(table.begin(), table.end(), user_data.begin());
	std::copy(output.begin(), output.end(), user_data.begin() + 4);
	user_data[12] = static_cast<uint32_t>(rows.size());

	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	const std::vector<uint32_t> plan_user_data(16, 0u);
	const std::vector<uint32_t> code(RowLoopWords.begin(), RowLoopWords.end());
	Recompiler::CompileOptions  options;
	options.stage                      = ShaderType::Compute;
	options.wave_size                  = 64;
	options.shader_hash                = 0xd57u;
	options.dump_ir                    = false;
	options.user_data                  = plan_user_data;
	options.input_info.compute         = &info;
	options.runtime_buffer_descriptors = runtime_buffers;
	Outcome outcome;
	try {
		const auto source     = Recompiler::PrepareShaderSource(code, options);
		auto       translated = Recompiler::TranslateProgram(source.decoded, options);
		const auto plan       = IR::ExtractResourcePlan(translated.program);
		outcome.table_plan    = std::ranges::any_of(plan.descriptor_sources, [](const IR::DescriptorSource& source) {
            return source.indirect_descriptor.has_value() && source.indirect_descriptor->indexed &&
                   source.indirect_descriptor->table_stride == RowStride;
        });
		const IR::SrtRuntime       runtime {.user_data = user_data, .read_specialization_memory = ReadMemory};
		IR::ResourceSnapshot       snapshot;
		IR::ResourceSpecialization specialization;
		outcome.materialized     = IR::MaterializeResources(plan, runtime, snapshot, specialization);
		outcome.indirect_failure = IR::IndirectBufferMaterializationFailed();
		if (outcome.materialized) {
			auto compiled    = Recompiler::CompileProgram(std::move(translated), options, specialization, 0);
			outcome.compiled = !compiled.spirv.empty();
		}
	} catch (const std::exception& error) {
		std::fprintf(stderr, "stopped: %s\n", error.what());
	}
	return outcome;
}

void BoundRows() {
	const auto valid = Run({Vertices(VertexAddress, SwizzleXyzw), Vertices(VertexAddress + 0x400u, SwizzleXyzw)}, false);
	Check(valid.table_plan, "valid rows: the indexed vertex descriptor is chosen in a table of 120-byte rows");
	Check(valid.materialized && valid.compiled, "valid rows: the bound plan is materialized and compiled");

	const auto unread = Run({Vertices(VertexAddress, SwizzleXyzw), Vertices(VertexAddress, Select(4, 5, 6, 3))}, false);
	Check(unread.materialized && unread.compiled,
	      "reserved select on a component the shader does not read: the bound plan is still compiled");

	const auto read = Run({Vertices(VertexAddress, SwizzleXyzw), Vertices(VertexAddress, Select(4, 5, 3, 7))}, false);
	Check(!read.materialized, "reserved select on a read component: the candidate is refused before emit");
	Check(read.indirect_failure, "reserved select on a read component: the failure is on an indirect buffer");

	auto empty = Vertices(0u, Select(4, 2, 6, 7));
	empty[2]   = 0u;
	const auto cleared = Run({Vertices(VertexAddress, SwizzleXyzw), empty}, false);
	Check(!cleared.materialized && cleared.indirect_failure,
	      "empty row with a reserved select: the candidate is refused before emit");
}

void RuntimeRows() {
	const auto served = Run({Vertices(VertexAddress, SwizzleXyzw), Vertices(VertexAddress, Select(4, 5, 3, 7))}, true);
	Check(served.materialized && served.compiled,
	      "run-time descriptors: the dispatch refused by the bound plan is materialized and compiled");
}

void SelfSizedStores() {
	constexpr std::array<uint32_t, 20> code {
	    0xf4000080u, 0xfa000000u, 0xbf8cc07fu, 0xbe8303ffu, 0x31014facu, 0x4a020081u, 0x34060081u,
	    0xe0601100u, 0x80000103u, 0xe06017c0u, 0x80000103u, 0x340a0082u, 0x380400ffu, 0x00007700u,
	    0xe0681200u, 0x80000205u, 0x7e080d00u, 0xe86c1402u, 0x80000405u, 0xbf810000u,
	};
	std::fill(g_guest.begin(), g_guest.end(), 0u);
	Poke(OutputAddress, 0x800u);
	const std::array<uint32_t, 2> user_data {static_cast<uint32_t>(OutputAddress), static_cast<uint32_t>(OutputAddress >> 32u)};
	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	Recompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x55e1u;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	try {
		const std::vector<uint32_t> words(code.begin(), code.end());
		const auto source     = Recompiler::PrepareShaderSource(words, options);
		auto       translated = Recompiler::TranslateProgram(source.decoded, options);
		const auto& sized     = translated.program.info.self_sized_buffers;
		Check(sized.size() == 1u && sized.front().written,
		      "byte, short and typed stores through a self-sized V#: one written self-sized buffer");
		uint32_t stores = 0;
		for (const auto& memory: translated.program.memory_info) {
			if (memory.planning_only ||
			    (memory.kind != IR::ResourceKind::Buffer && memory.kind != IR::ResourceKind::IndirectBuffer)) continue;
			Check(memory.kind == IR::ResourceKind::IndirectBuffer,
			      "byte, short and typed stores through a self-sized V#: the stores read their V# at run time");
			stores++;
		}
		Check(stores == 4u, "byte, short and typed stores through a self-sized V#: every store is planned");
		const auto                 plan = IR::ExtractResourcePlan(translated.program);
		const IR::SrtRuntime       runtime {.user_data = user_data, .read_specialization_memory = ReadMemory};
		IR::ResourceSnapshot       snapshot;
		IR::ResourceSpecialization specialization;
		const bool materialized = IR::MaterializeResources(plan, runtime, snapshot, specialization);
		Check(materialized, "byte, short and typed stores through a self-sized V#: the plan is materialized");
		if (materialized) {
			auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0);
			Check(!compiled.spirv.empty(), "byte, short and typed stores through a self-sized V#: the program is compiled");
		}
	} catch (const std::exception& error) {
		Check(false, std::string("byte, short and typed stores through a self-sized V#: ") + error.what());
	}
}

}

int main() {
	Common::SoftExitScope soft;
	IR::SetComputeIndirectFloors(256, 8);
	BoundRows();
	RuntimeRows();
	SelfSizedStores();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_indirect_buffer_dst_sel: ok\n");
	return 0;
}

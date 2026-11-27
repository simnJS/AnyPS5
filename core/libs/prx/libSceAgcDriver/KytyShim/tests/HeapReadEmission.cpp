#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <span>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr std::array<uint32_t, 31> LoopWords {
    0xf4080600u, 0xfa000100u, 0x7e080d00u, 0x060808f0u, 0x100808ffu, 0x3c800000u, 0x7e0a02f0u, 0x340c0084u,
    0x7e100280u, 0x7e120280u, 0x7e140280u, 0x7e160280u, 0xbe830380u, 0xbf0a0203u, 0xbf84000du, 0x8f248503u,
    0xf40c0200u, 0x48000000u, 0xbf8cc07fu, 0xf09c0f08u, 0x00c20004u, 0xbf8c3f70u, 0x3a100108u, 0x3a120309u,
    0x3a14050au, 0x3a16070bu, 0x80038103u, 0xbf82fff1u, 0xe0781000u, 0x80010806u, 0xbf810000u,
};
constexpr uint64_t GuestRegion   = 0x200900000ull;
constexpr size_t   GuestBytes    = 0x40000u;
constexpr uint64_t TableAddress  = 0x200910000ull;
constexpr uint64_t TextureA      = 0x200920000ull;
constexpr uint64_t TextureB      = 0x200920100ull;
constexpr uint64_t OutputAddress = 0x200930000ull;
constexpr uint32_t Format8888    = 56;
constexpr uint32_t Format32UInt  = 20;
const char*        g_name        = "kyty_shim_heap_read_emission";

int       g_failures = 0;
uint32_t* g_guest    = nullptr;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "%s: FAILED %s\n", g_name, what.c_str());
	g_failures++;
}

bool MapGuest() {
#if defined(_WIN32)
	g_guest = static_cast<uint32_t*>(VirtualAlloc(reinterpret_cast<void*>(GuestRegion), GuestBytes,
	                                              MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
	void* mapped = mmap(reinterpret_cast<void*>(GuestRegion), GuestBytes, PROT_READ | PROT_WRITE,
	                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
	g_guest      = mapped == MAP_FAILED ? nullptr : static_cast<uint32_t*>(mapped);
#endif
	return g_guest == reinterpret_cast<uint32_t*>(GuestRegion);
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

using Texture = std::array<uint32_t, 8>;

Texture MakeTexture(uint64_t address, uint32_t format) {
	const auto base = address >> 8u;
	return {static_cast<uint32_t>(base), static_cast<uint32_t>((base >> 32u) & 0xffu) | (format << 20u) | (3u << 30u),
	        15u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
}

constexpr uint64_t HeapSlots = 0x200938000ull;

struct Heap final: IR::ImageHeapSource {
	std::array<IR::DescriptorValue, 1> kinds;

	bool Acquire(const IR::ImageHeapRequest& request, IR::ImageHeapView& result) override {
		if (!request.table || request.count != 3u) return false;
		result = {.slots = HeapSlots, .slot_count = 17, .kinds = kinds};
		return true;
	}
};

struct Compiled {
	std::vector<uint32_t> spirv;
	bool                  heap = false;
};

Compiled Compile(uint32_t host_subgroup) {
	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = host_subgroup;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	const std::vector<uint32_t> plan_user_data(16, 0u);
	const std::vector<uint32_t> code(LoopWords.begin(), LoopWords.end());
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x4ea0u + host_subgroup;
	options.dump_ir            = false;
	options.user_data          = plan_user_data;
	options.input_info.compute = &info;
	Compiled result;
	const auto source     = Recompiler::PrepareShaderSource(code, options);
	auto       translated = Recompiler::TranslateProgram(source.decoded, options);
	auto       plan       = IR::ExtractResourcePlan(translated.program);
	if (plan.info.images.size() != 1 || plan.info.buffers.empty()) {
		Check(false, "the fixture has one image and a buffer");
		return result;
	}
	auto& indirect = plan.descriptor_sources[plan.info.images.front().source].indirect_descriptor;
	if (!indirect) {
		Check(false, "the fixture image is read through a descriptor table");
		return result;
	}
	indirect->table_source = plan.info.buffers.front().source;
	indirect->table_array  = true;
	indirect->table_mask   = 0x7fu;
	indirect->table_offset = 0;
	translated.program.descriptor_sources = plan.descriptor_sources;

	const auto a = MakeTexture(TextureA, Format8888);
	const auto b = MakeTexture(TextureB, Format8888);
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	for (uint32_t i = 0; i < a.size(); ++i) {
		Poke(TableAddress + i * 4u, a[i]);
		Poke(TableAddress + 32u + i * 4u, b[i]);
	}
	std::array<uint32_t, 16> user {};
	user[0] = user[4] = static_cast<uint32_t>(TableAddress);
	user[1] = user[5] = static_cast<uint32_t>(TableAddress >> 32u);
	user[2]           = 3;
	user[6]           = 96;
	user[7]           = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u);
	(void)OutputAddress;
	Heap heap;
	heap.kinds[0].dword_count = 8;
	std::copy(a.begin(), a.end(), heap.kinds[0].dwords.begin());
	const IR::SrtRuntime       runtime {.user_data = user, .read_specialization_memory = ReadMemory, .image_heap = &heap};
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	if (!IR::MaterializeResources(plan, runtime, snapshot, specialization)) {
		Check(false, "the fixture materializes on the image heap");
		return result;
	}
	result.heap  = std::ranges::any_of(specialization.images, [](const auto& image) { return image.heap; });
	result.spirv = Recompiler::CompileProgram(std::move(translated), options, specialization, 0).spirv;
	return result;
}

struct Instruction {
	uint32_t                  op = 0;
	std::span<const uint32_t> words;
};

std::vector<Instruction> Parse(const std::vector<uint32_t>& spirv) {
	std::vector<Instruction> instructions;
	for (size_t at = 5; at < spirv.size();) {
		const auto count = spirv[at] >> 16u;
		if (count == 0u || at + count > spirv.size()) break;
		instructions.push_back({spirv[at] & 0xffffu, std::span(spirv).subspan(at, count)});
		at += count;
	}
	return instructions;
}

void CheckModule(const Compiled& compiled, const char* what) {
	const std::string label = what;
	Check(compiled.heap, label + ": the image is read through the heap");
	Check(!compiled.spirv.empty(), label + ": the program compiles");
	if (compiled.spirv.empty()) return;
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		messages += message;
		messages += '\n';
	});
	Check(tools.Validate(compiled.spirv), label + ": the module validates: " + messages);
	const auto                   instructions = Parse(compiled.spirv);
	std::map<uint32_t, uint32_t> sets, bindings;
	for (const auto& inst: instructions) {
		if (inst.op != spv::OpDecorate || inst.words.size() < 4u) continue;
		if (inst.words[2] == spv::DecorationDescriptorSet) sets[inst.words[1]] = inst.words[3];
		if (inst.words[2] == spv::DecorationBinding) bindings[inst.words[1]] = inst.words[3];
	}
	uint32_t reads = 0;
	for (const auto& [id, set]: sets) {
		if (set == IR::ImageHeapSet && bindings[id] == IR::ImageHeapReadsBinding) reads = id;
	}
	Check(reads != 0u, label + ": the heap read bitmap is declared at the heap set, binding 1");
	std::vector<uint32_t> chains;
	for (const auto& inst: instructions) {
		if (inst.op == spv::OpAccessChain && inst.words.size() > 3u && inst.words[3] == reads) chains.push_back(inst.words[2]);
	}
	size_t atomics = 0, size = 0, elects = 0, broadcasts = 0, heap_loads = 0;
	for (size_t at = 0; at < instructions.size(); ++at) {
		const auto& inst = instructions[at];
		elects += inst.op == spv::OpGroupNonUniformElect;
		broadcasts += inst.op == spv::OpGroupNonUniformBroadcastFirst;
		if (inst.op != spv::OpAtomicOr || std::ranges::find(chains, inst.words[3]) == chains.end()) continue;
		++atomics;
		auto first = at;
		while (first > 0 && !(instructions[first].op == spv::OpAccessChain && instructions[first].words[3] == reads)) --first;
		size = at - first + 7u;
	}
	for (const auto& inst: instructions) {
		heap_loads += inst.op == spv::OpLoad && inst.words.size() > 4u &&
		              (inst.words[4] & spv::MemoryAccessAlignedMask) != 0u && inst.words.size() == 6u;
	}
	Check(atomics != 0u, label + ": every heap access ORs its slot bit into the bitmap");
	Check(elects >= atomics && broadcasts >= atomics, label + ": the bit is set once per subgroup after a vote");
	std::printf("%s: %s: %zu heap read marks, %zu SPIR-V instructions each (element mask, address, load, test, vote, OR, merges), %zu words\n",
	            g_name, what, atomics, size, compiled.spirv.size());
	(void)heap_loads;
}

}

int main() {
	if (!MapGuest()) {
		std::fprintf(stderr, "%s: could not map the guest region\n", g_name);
		return 1;
	}
	try {
		ShaderInit();
		CheckModule(Compile(64u), "wave64 on 64 lanes");
		CheckModule(Compile(32u), "wave64 on 32 lanes");
	} catch (const std::exception& error) {
		Check(false, std::string("stopped: ") + error.what());
	}
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %d failure(s)\n", g_name, g_failures);
		return 1;
	}
	std::printf("%s: ok\n", g_name);
	return 0;
}

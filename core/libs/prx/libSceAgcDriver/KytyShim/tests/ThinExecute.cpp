#include "graphics/host_gpu/renderer/thinExecute.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;
using Thin::Refusal;
using BindingKind = IR::DescriptorBindingKind;

constexpr const char* Name = "kyty_shim_thin_execute";

uint32_t g_failures = 0;
uint32_t g_checks   = 0;

void Check(bool value, const std::string& what) {
	g_checks++;
	if (value) return;
	if (g_failures++ < 32) std::fprintf(stderr, "%s: FAILED: %s\n", Name, what.c_str());
}

HW::CsStageRegisters Registers() {
	HW::CsStageRegisters regs;
	regs.data_addr    = 0x0000'0012'3456'7000ull;
	regs.num_thread_x = 64;
	regs.num_thread_y = 1;
	regs.num_thread_z = 1;
	regs.vgprs        = 12;
	regs.float_mode   = 0xc0;
	regs.wave_size    = 64;
	regs.user_sgpr    = 8;
	regs.lds_size     = 16;
	return regs;
}

void Keys() {
	const auto regs = Registers();
	const auto base = Thin::MakeDispatchKey(regs, 7, 0x41, false, Thin::OperationKind::Direct);
	Check(base == Thin::MakeDispatchKey(regs, 7, 0x41, false, Thin::OperationKind::Direct), "key is deterministic");
	Check(Thin::DispatchKeyHash {}(base) ==
	          Thin::DispatchKeyHash {}(Thin::MakeDispatchKey(regs, 7, 0x41, false, Thin::OperationKind::Direct)),
	      "hash is deterministic");
	std::vector<Thin::DispatchKey> variants;
	const auto with = [&](auto change) {
		auto copy = regs;
		change(copy);
		variants.push_back(Thin::MakeDispatchKey(copy, 7, 0x41, false, Thin::OperationKind::Direct));
	};
	with([](auto& r) { r.data_addr += 0x100; });
	with([](auto& r) { r.num_thread_x = 32; });
	with([](auto& r) { r.num_thread_y = 2; });
	with([](auto& r) { r.num_thread_z = 2; });
	with([](auto& r) { r.vgprs = 13; });
	with([](auto& r) { r.priority = 1; });
	with([](auto& r) { r.float_mode = 0xf0; });
	with([](auto& r) { r.dx10_clamp = true; });
	with([](auto& r) { r.debug_mode = true; });
	with([](auto& r) { r.ieee_mode = true; });
	with([](auto& r) { r.require_forward_progress = true; });
	with([](auto& r) { r.fp16_overflow = true; });
	with([](auto& r) { r.threadgroup_configuration = true; });
	with([](auto& r) { r.wave_size = 32; });
	with([](auto& r) { r.scratch_en = true; });
	with([](auto& r) { r.user_sgpr = 9; });
	with([](auto& r) { r.tgid_x_en = true; });
	with([](auto& r) { r.tgid_y_en = true; });
	with([](auto& r) { r.tgid_z_en = true; });
	with([](auto& r) { r.tg_size_en = true; });
	with([](auto& r) { r.tidig_comp_cnt = 2; });
	with([](auto& r) { r.lds_size = 17; });
	with([](auto& r) { r.shared_vgprs = 3; });
	variants.push_back(Thin::MakeDispatchKey(regs, 8, 0x41, false, Thin::OperationKind::Direct));
	variants.push_back(Thin::MakeDispatchKey(regs, 7, 0x61, false, Thin::OperationKind::Direct));
	variants.push_back(Thin::MakeDispatchKey(regs, 7, 0x41, true, Thin::OperationKind::Direct));
	variants.push_back(Thin::MakeDispatchKey(regs, 7, 0x41, false, Thin::OperationKind::Indirect));
	for (size_t i = 0; i < variants.size(); ++i) {
		Check(!(variants[i] == base), "register change " + std::to_string(i) + " changes the key");
		for (size_t j = i + 1; j < variants.size(); ++j) {
			Check(!(variants[i] == variants[j]), "keys " + std::to_string(i) + " and " + std::to_string(j) + " differ");
		}
	}
}

Thin::ProgramFacts Ready() {
	Thin::ProgramFacts facts;
	facts.pipeline_ready = true;
	facts.push_layout    = true;
	return facts;
}

void Classification() {
	Check(Thin::Classify(Ready()) == 0, "a ready program with push data only is thin");
	Check(Thin::Classify({}) == (Thin::Bit(Refusal::PipelineWait) | Thin::Bit(Refusal::PushLayout)),
	      "unknown pipeline and push layout are refused");
	struct Case {
		const char* what;
		void (*change)(Thin::ProgramFacts&);
		Thin::RefusalMask expected;
		Thin::Owner       owner;
	};
	const Case cases[] = {
	    {"specialization reads", [](auto& f) { f.specialization_reads = true; }, Thin::Bit(Refusal::Specialized), Thin::Owner::B},
	    {"descriptor specialization", [](auto& f) { f.descriptor_specialization = true; }, Thin::Bit(Refusal::Specialized), Thin::Owner::B},
	    {"pipeline wait", [](auto& f) { f.pipeline_ready = false; }, Thin::Bit(Refusal::PipelineWait), Thin::Owner::B},
	    {"buffers", [](auto& f) { f.buffers = true; }, Thin::Bit(Refusal::CpuBuffers), Thin::Owner::A},
	    {"flattened SRT", [](auto& f) { f.flattened_srt = true; }, Thin::Bit(Refusal::CpuSrt), Thin::Owner::A},
	    {"memory access", [](auto& f) { f.memory_access = true; }, Thin::Bit(Refusal::MemorySync), Thin::Owner::A},
	    {"GPU flat loads", [](auto& f) { f.gpu_flat_loads = true; }, Thin::Bit(Refusal::GpuFlatLoads), Thin::Owner::A},
	    {"images", [](auto& f) { f.images = true; }, Thin::Bit(Refusal::CpuImages), Thin::Owner::E},
	    {"image tables", [](auto& f) { f.image_tables = true; }, Thin::Bit(Refusal::ImageTables), Thin::Owner::E},
	    {"image heap", [](auto& f) { f.image_heap = true; }, Thin::Bit(Refusal::ImageHeap), Thin::Owner::E},
	    {"samplers", [](auto& f) { f.samplers = true; }, Thin::Bit(Refusal::CpuSamplers), Thin::Owner::E},
	    {"indirect args", [](auto& f) { f.indirect_args_on_cpu = true; }, Thin::Bit(Refusal::IndirectArgs), Thin::Owner::D},
	    {"LDS storage", [](auto& f) { f.lds_storage = true; }, Thin::Bit(Refusal::SharedMemory), Thin::Owner::C},
	    {"GDS", [](auto& f) { f.gds = true; }, Thin::Bit(Refusal::Gds), Thin::Owner::C},
	    {"ray tracing", [](auto& f) { f.ray_tracing = true; }, Thin::Bit(Refusal::RayTracing), Thin::Owner::C},
	    {"write targets", [](auto& f) { f.write_targets = true; }, Thin::Bit(Refusal::WriteTargets), Thin::Owner::C},
	    {"uniform fill", [](auto& f) { f.uniform_fill = true; }, Thin::Bit(Refusal::CpuConsumed), Thin::Owner::C},
	    {"consumed", [](auto& f) { f.consumed_on_cpu = true; }, Thin::Bit(Refusal::CpuConsumed), Thin::Owner::C},
	    {"ordered append", [](auto& f) { f.ordered_append = true; }, Thin::Bit(Refusal::OrderedAppend), Thin::Owner::C},
	    {"checks", [](auto& f) { f.checked = true; }, Thin::Bit(Refusal::Checks), Thin::Owner::C},
	    {"push layout", [](auto& f) { f.push_layout = false; }, Thin::Bit(Refusal::PushLayout), Thin::Owner::C},
	};
	for (const auto& test: cases) {
		auto facts = Ready();
		test.change(facts);
		const auto refusals = Thin::Classify(facts);
		Check(refusals == test.expected, std::string(test.what) + " gives exactly its refusal");
		Check(Thin::OwnerMask(refusals) == 1u << static_cast<uint32_t>(test.owner),
		      std::string(test.what) + " belongs to its owner");
	}
	const auto with_kind = [](BindingKind kind, bool memory_access = false, bool memory_ready = false) {
		auto facts            = Ready();
		facts.descriptor_mask = uint64_t {1} << static_cast<uint32_t>(kind);
		facts.memory_access   = memory_access;
		facts.memory_ready    = memory_ready;
		return Thin::Classify(facts);
	};
	Check(with_kind(BindingKind::Buffers) == Thin::Bit(Refusal::CpuBuffers), "buffer binding");
	Check(with_kind(BindingKind::UniformBuffers) == Thin::Bit(Refusal::CpuBuffers), "uniform binding");
	Check(with_kind(BindingKind::FlattenedSrt) == Thin::Bit(Refusal::CpuSrt), "flattened SRT binding");
	Check(with_kind(BindingKind::Samplers) == Thin::Bit(Refusal::CpuSamplers), "sampler binding");
	for (uint32_t kind = IR::FirstImageBinding; kind < static_cast<uint32_t>(BindingKind::Samplers); ++kind) {
		Check(with_kind(static_cast<BindingKind>(kind)) == Thin::Bit(Refusal::CpuImages),
		      "image binding " + std::to_string(kind));
	}
	Check(with_kind(BindingKind::Gds) == Thin::Bit(Refusal::Gds), "GDS binding");
	Check(with_kind(BindingKind::SharedMemory) == Thin::Bit(Refusal::SharedMemory), "shared memory binding");
	Check(with_kind(BindingKind::ShaderData) == Thin::Bit(Refusal::PushLayout), "shader data binding");
	Check(with_kind(BindingKind::MipStats) == Thin::Bit(Refusal::Instrumentation), "mip statistics binding");
	const BindingKind named[] = {BindingKind::Buffers,      BindingKind::UniformBuffers, BindingKind::FlattenedSrt,
	                             BindingKind::Samplers,     BindingKind::Gds,            BindingKind::SharedMemory,
	                             BindingKind::ShaderData,   BindingKind::MipStats,       BindingKind::FaultBuffer,
#if defined(ANYPS5_KYTY)
	                             BindingKind::PackedTargets,
#endif
	};
	uint32_t other = 0;
	for (uint32_t kind = 0; kind < static_cast<uint32_t>(BindingKind::Count); ++kind) {
		const bool image = kind >= IR::FirstImageBinding && kind < static_cast<uint32_t>(BindingKind::Samplers);
		if (image || std::ranges::find(named, static_cast<BindingKind>(kind)) != std::end(named)) continue;
		other++;
		const auto binding = static_cast<BindingKind>(kind);
		Check(with_kind(binding) == Thin::Bit(Refusal::MemorySync), "other binding " + std::to_string(kind));
		Check(with_kind(binding, true) == Thin::Bit(Refusal::MemorySync), "other binding with memory");
		Check(with_kind(binding, true, true) == 0, "other binding once memory is ready");
	}
	std::printf("%s: %u other binding kinds wait for memory readiness\n", Name, other);
	Check(with_kind(BindingKind::FaultBuffer) == Thin::Bit(Refusal::Instrumentation), "loop guard fault buffer");
	Check(with_kind(BindingKind::FaultBuffer, true) == Thin::Bit(Refusal::MemorySync), "fault buffer with memory");
	Check(with_kind(BindingKind::FaultBuffer, true, true) == 0, "fault buffer once memory is ready");
	auto heap_pipeline = Ready();
	Thin::NotePipeline(heap_pipeline, true, true);
	Check(Thin::Classify(heap_pipeline) == Thin::Bit(Refusal::ImageHeap),
	      "a ready pipeline that reads the image heap waits for E, not for a pipeline");
	auto missing = Ready();
	Thin::NotePipeline(missing, false, false);
	Check(Thin::Classify(missing) == Thin::Bit(Refusal::PipelineWait), "a missing pipeline is a pipeline wait");
	auto bound_heap       = Ready();
	bound_heap.image_heap = true;
	Thin::NotePipeline(bound_heap, true, false);
	Check(bound_heap.image_heap && Thin::Classify(bound_heap) == Thin::Bit(Refusal::ImageHeap),
	      "a heap image seen in the program stays a heap access");
	struct CpuDescriptor {
		const char* what;
		void (*change)(Thin::ProgramFacts&);
		Refusal     refusal;
	};
	const CpuDescriptor cpu_descriptors[] = {
	    {"buffers", [](auto& f) { f.buffers = true; }, Refusal::CpuBuffers},
	    {"flattened SRT", [](auto& f) { f.flattened_srt = true; }, Refusal::CpuSrt},
	    {"images", [](auto& f) { f.images = true; }, Refusal::CpuImages},
	    {"image tables", [](auto& f) { f.image_tables = true; }, Refusal::ImageTables},
	    {"image heap", [](auto& f) { f.image_heap = true; }, Refusal::ImageHeap},
	    {"samplers", [](auto& f) { f.samplers = true; }, Refusal::CpuSamplers},
	};
	for (const auto& cpu: cpu_descriptors) {
		for (const bool reads: {false, true}) {
			auto facts                      = Ready();
			facts.specialization_reads      = reads;
			facts.descriptor_specialization = !reads;
			cpu.change(facts);
			Check(Thin::Classify(facts) == (Thin::Bit(cpu.refusal) | Thin::Bit(Refusal::SpecializedBehind)),
			      std::string("a specialization on ") + cpu.what + " the CPU binds is behind their owner");
		}
	}
	struct Visible {
		const char* what;
		void (*change)(Thin::ProgramFacts&);
		Refusal     refusal;
	};
	const Visible visible[] = {
	    {"memory access", [](auto& f) { f.memory_access = true; }, Refusal::MemorySync},
	    {"GPU flat loads", [](auto& f) { f.gpu_flat_loads = true; }, Refusal::GpuFlatLoads},
	    {"indirect args", [](auto& f) { f.indirect_args_on_cpu = true; }, Refusal::IndirectArgs},
	    {"push layout", [](auto& f) { f.push_layout = false; }, Refusal::PushLayout},
	    {"pipeline wait", [](auto& f) { f.pipeline_ready = false; }, Refusal::PipelineWait},
	};
	for (const auto& other: visible) {
		auto facts                 = Ready();
		facts.specialization_reads = true;
		other.change(facts);
		Check(Thin::Classify(facts) == (Thin::Bit(other.refusal) | Thin::Bit(Refusal::Specialized)),
		      std::string("a specialization next to ") + other.what + " stays a refusal of B");
	}
	uint32_t specialized_shapes = 0;
	for (uint32_t shape = 0; shape < (1u << 8u); ++shape) {
		auto facts                      = Ready();
		facts.specialization_reads      = (shape & 1u) != 0;
		facts.descriptor_specialization = (shape & 2u) != 0;
		for (uint32_t index = 0; index < std::size(cpu_descriptors); ++index) {
			if ((shape >> (index + 2u)) & 1u) cpu_descriptors[index].change(facts);
		}
		if (!facts.specialization_reads && !facts.descriptor_specialization) continue;
		specialized_shapes++;
		const auto refusals = Thin::Classify(facts);
		const auto b        = refusals & (Thin::Bit(Refusal::Specialized) | Thin::Bit(Refusal::SpecializedBehind));
		Check(refusals != 0 && (b == Thin::Bit(Refusal::Specialized) || b == Thin::Bit(Refusal::SpecializedBehind)),
		      "a specialized program is never thin and is counted once by B (shape " + std::to_string(shape) + ")");
	}
	Check(specialized_shapes == 192, "every specialized shape was classified");
	auto many            = Ready();
	many.buffers         = true;
	many.images          = true;
	many.indirect_args_on_cpu = true;
	many.gds             = true;
	const auto refusals = Thin::Classify(many);
	Check(Thin::OwnerMask(refusals) == ((1u << static_cast<uint32_t>(Thin::Owner::A)) |
	                                    (1u << static_cast<uint32_t>(Thin::Owner::C)) |
	                                    (1u << static_cast<uint32_t>(Thin::Owner::D)) |
	                                    (1u << static_cast<uint32_t>(Thin::Owner::E))),
	      "owners of several refusals");
	for (uint32_t index = 0; index < Thin::RefusalCount; ++index) {
		Check(Thin::RefusalName(static_cast<Refusal>(index)) != "?", "refusal " + std::to_string(index) + " is named");
	}
	Check(Thin::RefusalOwner(Refusal::SpecializedBehind) == Thin::Owner::B &&
	          Thin::RefusalName(Refusal::SpecializedBehind) == "specialized behind A or E",
	      "a specialization behind A or E is still B's");
	Check(Thin::RefusalOwner(Refusal::VertexBuffers) == Thin::Owner::A &&
	          Thin::RefusalOwner(Refusal::IndexBuffer) == Thin::Owner::A &&
	          Thin::RefusalOwner(Refusal::Targets) == Thin::Owner::C &&
	          Thin::RefusalOwner(Refusal::MeshDraw) == Thin::Owner::C,
	      "draw refusals belong to their owners");
	Check(Thin::RefusalName(Refusal::MeshDraw) == "mesh draw" && Thin::RefusalName(Refusal::Targets) == "targets" &&
	          Thin::RefusalName(Refusal::Count) == "?",
	      "names follow the refusal order");
}

void Description() {
	IR::CompiledShaderInfo       program;
	IR::ResourceSnapshot         snapshot;
	IR::ResourceSpecialization   specialization;
	auto facts = Thin::DescribeProgram(program, snapshot, specialization);
	Check(!facts.buffers && !facts.images && !facts.samplers && !facts.memory_access && !facts.image_heap &&
	          !facts.flattened_srt && !facts.specialization_reads && !facts.gpu_flat_loads && !facts.uniform_fill &&
	          !facts.gds && !facts.ray_tracing && !facts.write_targets && !facts.image_tables &&
	          !facts.descriptor_specialization,
	      "an empty program has no resources");
	specialization.buffers.emplace_back();
	Check(Thin::DescribeProgram(program, snapshot, specialization).descriptor_specialization,
	      "a module specialized on a buffer descriptor is seen");
	specialization = {};
	specialization.images.emplace_back();
	Check(Thin::DescribeProgram(program, snapshot, specialization).descriptor_specialization,
	      "a module specialized on an image descriptor is seen");
	specialization = {};
	specialization.samplers.emplace_back();
	Check(Thin::DescribeProgram(program, snapshot, specialization).descriptor_specialization,
	      "a module specialized on a sampler descriptor is seen");
	specialization = {};
	program.info.buffers.emplace_back();
	Check(Thin::DescribeProgram(program, snapshot, specialization).buffers, "a buffer is seen");
	program.info.buffers.back().indirect_write_table = true;
	Check(Thin::DescribeProgram(program, snapshot, specialization).write_targets, "an indirect write table is a write target");
	program.info.buffers.clear();
	program.info.images.emplace_back();
	program.info.images.back().heap = true;
	facts = Thin::DescribeProgram(program, snapshot, specialization);
	Check(facts.image_heap && !facts.images, "a heap image is a heap access, not a bound image");
	program.info.images.back().heap           = false;
	program.info.images.back().table_capacity = 4;
	facts = Thin::DescribeProgram(program, snapshot, specialization);
	Check(facts.images && facts.image_tables, "a table image is a bound image in a table");
	program.info.images.clear();
	program.info.uses_dma = true;
	Check(Thin::DescribeProgram(program, snapshot, specialization).memory_access, "DMA is memory access");
	program.info.uses_dma = false;
	program.has_address_writes = true;
	Check(Thin::DescribeProgram(program, snapshot, specialization).memory_access, "address writes are memory access");
	program.has_address_writes = false;
	program.info.self_sized_buffers.emplace_back();
	Check(Thin::DescribeProgram(program, snapshot, specialization).write_targets, "a self-sized buffer is a write target");
	program.info.self_sized_buffers.clear();
	program.info.ray_query = true;
	Check(Thin::DescribeProgram(program, snapshot, specialization).ray_tracing, "ray queries are ray tracing");
	program.info.ray_query = false;
	snapshot.flattened_srt.push_back(1);
	snapshot.specialization_reads.emplace_back(0x1000, 4);
	snapshot.gpu_flat_loads.push_back({});
	snapshot.uniform_fill.kind = IR::UniformFillKind::Buffer;
	facts = Thin::DescribeProgram(program, snapshot, specialization);
	Check(facts.flattened_srt && facts.specialization_reads && facts.gpu_flat_loads && facts.uniform_fill,
	      "snapshot contents are seen");
}

struct Layout {
	uint32_t              base       = 0;
	std::vector<uint32_t> registers;
	bool                  threads    = false;
	bool                  ray_query  = false;
	uint32_t              buffers    = 0;
	uint32_t              extra      = 0;
	uint32_t              start      = 0;
};

IR::CompiledShaderInfo Program(const Layout& layout) {
	IR::CompiledShaderInfo program;
	program.user_data_base           = layout.base;
	program.info.user_data_registers = layout.registers;
	auto& bindings                   = program.bindings;
	bindings.memory_offset_dword     = static_cast<uint32_t>(layout.registers.size());
	if (layout.threads) {
		bindings.dispatch_thread_dword = bindings.memory_offset_dword;
		bindings.memory_offset_dword += 3u;
	}
	if (layout.ray_query) {
		bindings.ray_query_dword = bindings.memory_offset_dword;
		bindings.memory_offset_dword += 2u;
	}
	bindings.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] = layout.buffers;
	bindings.clip_plane_dwords                                            = layout.extra;
	bindings.push_data_start_dword = IR::PushData::StartFor(layout.start, bindings.ShaderDataDwords());
	return program;
}

std::vector<uint32_t> ReferenceShaderData(const IR::CompiledShaderInfo& program, const std::vector<uint32_t>& user_data,
                                          const uint32_t (&threads)[3]) {
	std::vector<uint32_t> shader_data;
	for (const auto reg: program.info.user_data_registers) {
		shader_data.push_back(user_data[reg - program.user_data_base]);
	}
	shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.dispatch_thread_dword != IR::PushData::NoStart) {
		std::copy(std::begin(threads), std::end(threads), shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	return shader_data;
}

IR::PushData ReferencePush(const IR::CompiledShaderInfo& program, const std::vector<uint32_t>& shader_data) {
	IR::PushData push;
	if (program.bindings.UsesPushData()) {
		std::ranges::copy(shader_data, push.dwords.begin() + program.bindings.push_data_start_dword);
	}
	return push;
}

void PushPacking() {
	std::mt19937 random(0x7417u);
	const auto   next = [&] { return static_cast<uint32_t>(random()); };
	uint32_t     planned = 0;
	uint32_t     refused = 0;
	for (uint32_t round = 0; round < 20000; ++round) {
		Layout layout;
		layout.base           = next() % 3u;
		const uint32_t sgprs  = 1u + next() % 16u;
		const uint32_t count  = next() % (sgprs + 1u);
		for (uint32_t i = 0; i < count; ++i) {
			layout.registers.push_back(layout.base + next() % (sgprs + 2u));
		}
		layout.threads   = next() % 2u == 0;
		layout.ray_query = next() % 8u == 0;
		layout.buffers   = next() % 6u == 0 ? 1u + next() % 3u : 0u;
		layout.extra     = next() % 8u == 0 ? 1u : 0u;
		layout.start     = next() % 4u == 0 ? 6u : 0u;
		const auto program = Program(layout);
		std::vector<uint32_t> user_data(layout.base + sgprs + 2u);
		for (auto& word: user_data) word = next();
		const uint32_t threads[3] {next() % 4096u, next() % 64u, next() % 8u};
		Thin::PushPlan plan;
		const bool     fits_registers = std::ranges::all_of(layout.registers, [&](uint32_t reg) {
            return reg - layout.base < sgprs;
        });
		const bool expected = fits_registers && !layout.ray_query && layout.buffers == 0 && layout.extra == 0;
		const bool made     = Thin::MakePushPlan(program, sgprs, plan);
		Check(made == expected, "push plan accepted exactly the plain layouts (round " + std::to_string(round) + ")");
		if (!made) {
			refused++;
			continue;
		}
		planned++;
		const auto reference_data = ReferenceShaderData(program, user_data, threads);
		const auto reference      = ReferencePush(program, reference_data);
		IR::PushData packed;
		Thin::PackPushData(plan, std::span<const uint32_t> {user_data.data(), sgprs}, threads, packed);
		Check(packed.dwords == reference.dwords, "packed push data equals the full path (round " + std::to_string(round) + ")");
		Check(Thin::SamePushData(plan, reference_data, packed), "the learning check accepts the full path data");
		if (!reference_data.empty()) {
			auto changed = reference_data;
			changed[next() % changed.size()] ^= 0x10u;
			Check(!Thin::SamePushData(plan, changed, packed), "the learning check sees a changed word");
			changed = reference_data;
			changed.push_back(0);
			Check(!Thin::SamePushData(plan, changed, packed), "the learning check sees an extra word");
		}
	}
	Check(planned > 1000 && refused > 1000, "both plain and refused layouts were drawn");
	std::printf("%s: push layouts: %u planned, %u refused\n", Name, planned, refused);
}

void Table() {
	Thin::DispatchTable table;
	const auto          key = Thin::MakeDispatchKey(Registers(), 1, 0x41, false, Thin::OperationKind::Direct);
	Check(table.Find(key, 10) == nullptr, "an unknown key is not served");
	Thin::DispatchEntry entry;
	entry.learned          = 10;
	entry.plan.shader_hash = 0x1234;
	table.Learn(key, entry);
	const auto* found = table.Find(key, 10);
	Check(found != nullptr && found->plan.shader_hash == 0x1234, "a learned key is found");
	Check(table.Find(key, 10 + Thin::DispatchTable::RelearnFrames - 1) != nullptr, "a recent entry is kept");
	Check(table.Find(key, 10 + Thin::DispatchTable::RelearnFrames) == nullptr, "an old entry is learned again");
	Check(table.Find(key, 9) == nullptr, "a frame counter that went back learns again");
	entry.refusals = Thin::Bit(Refusal::CpuImages);
	table.Learn(key, entry);
	found = table.Find(key, 11);
	Check(found != nullptr && found->refusals == Thin::Bit(Refusal::CpuImages), "learning again replaces the entry");
	for (uint64_t i = 0; i < Thin::DispatchTable::MaxEntries; ++i) {
		auto regs      = Registers();
		regs.data_addr = 0x1000 + i * 0x100;
		table.Learn(Thin::MakeDispatchKey(regs, 1, 0x41, false, Thin::OperationKind::Direct), entry);
	}
	Check(table.Size() <= Thin::DispatchTable::MaxEntries, "the table stays bounded");
}

void Counting() {
	Thin::Counters counters;
	counters.NoteServed(Thin::OperationKind::Direct);
	counters.NoteServed(Thin::OperationKind::Direct);
	counters.NoteRefused(Thin::OperationKind::Direct, Thin::Bit(Refusal::CpuImages) | Thin::Bit(Refusal::CpuBuffers));
	counters.NoteRefused(Thin::OperationKind::Direct, Thin::Bit(Refusal::Gds));
	counters.NoteRefused(Thin::OperationKind::Direct,
	                     Thin::Bit(Refusal::CpuBuffers) | Thin::Bit(Refusal::SpecializedBehind));
	counters.NoteRefused(Thin::OperationKind::Indirect, Thin::Bit(Refusal::IndirectArgs));
	counters.NoteRefused(Thin::OperationKind::Draw, Thin::Bit(Refusal::VertexBuffers) | Thin::Bit(Refusal::IndexBuffer));
	counters.NoteRefused(Thin::OperationKind::Draw, Thin::Bit(Refusal::VertexBuffers) | Thin::Bit(Refusal::Targets));
	counters.NoteTime(Thin::OperationKind::Direct, true, 2000);
	counters.NoteTime(Thin::OperationKind::Direct, false, 600000);
	const auto& direct = counters.kinds[0];
	Check(direct.operations == 5 && direct.served == 2, "direct operations and thin ones are counted");
	Check(direct.without_c == 4 && direct.without_cd == 4, "a GDS refusal is C's");
	Check(counters.kinds[1].without_c == 1 && counters.kinds[1].without_cd == 0, "indirect arguments are D's");
	Check(direct.first[static_cast<size_t>(Refusal::CpuBuffers)] == 2 &&
	          direct.first[static_cast<size_t>(Refusal::SpecializedBehind)] == 0 &&
	          direct.refused[static_cast<size_t>(Refusal::SpecializedBehind)] == 1 &&
	          direct.first[static_cast<size_t>(Refusal::CpuImages)] == 0 &&
	          direct.refused[static_cast<size_t>(Refusal::CpuImages)] == 1,
	      "the first refusal follows the dependency order");
	const auto& draws = counters.kinds[static_cast<size_t>(Thin::OperationKind::Draw)];
	Check(draws.operations == 2 && draws.without_c == 1 && draws.without_cd == 1, "draw targets are C's");
	const auto line = counters.Summary(10.0, 2);
	std::printf("%s", line.c_str());
	Check(line.find("[kyty-thin]") == 0, "summary prefix");
	Check(line.find("direct dispatches: 5 (2.5 per frame), thin 2 (40.0 %)") != std::string::npos, "direct summary");
	Check(line.find("specialized behind A or E (B) 1 (first 0)") != std::string::npos,
	      "a specialization behind A or E is named in the summary");
	Check(line.find("us per operation thin 2.00 full 600.00") != std::string::npos, "timing summary");
	Check(line.find("CPU images (E) 1 (first 0)") != std::string::npos, "refusal owner in the summary");
	Check(line.find("indirect args (D) 1 (first 1)") != std::string::npos, "indirect refusal in the summary");
	Check(line.find("draws: 2 (1.0 per frame), thin 0 (0.0 %), without C refusals 50.0 %") != std::string::npos,
	      "draw summary");
	Check(line.find("vertex buffers (A) 2 (first 2), index buffer (A) 1 (first 0), targets (C) 1 (first 0)") !=
	          std::string::npos,
	      "draw refusals in the summary");
}

}

int main() {
	Keys();
	Classification();
	Description();
	PushPacking();
	Table();
	Counting();
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %u of %u checks FAILED\n", Name, g_failures, g_checks);
		return 1;
	}
	std::printf("%s: ok (%u checks)\n", Name, g_checks);
	return 0;
}

#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace {
using namespace Libs::Graphics;
namespace RC = ShaderRecompiler;
namespace IR = RC::IR;
namespace Memory = Libs::LibKernel::Memory;
using Sharp = std::array<uint32_t, 4>;

constexpr uint32_t Field = 60, Mask = 0xffffff, MaterialStride = 80, TableRows = 16;
constexpr std::array<uint32_t, 4> Keys {0xaa000005, 0xbb000002, 0xcc000010, 0xdd000000};
constexpr uint32_t Sentinel = 0xbaadf00d;

void Require(bool condition, const char* message) {
	if (condition) return;
	std::fprintf(stderr, "masked_material_store: FAILED: %s\n", message);
	std::exit(EXIT_FAILURE);
}

constexpr auto MakeConsumer(uint32_t materialIndexSource, uint32_t activeLanes) {
	return std::array {
    0x7e040200u | materialIndexSource,
    0x7e0a0280u,
    0x4a060491u,
    0xbefe03ffu, activeLanes,
    0xe03c2038u, 0x80000402u,
    0xbf8c0000u,
    0x360a0affu, Mask,
    0x7e180505u,
    0x8f0c840cu,
    0xf4280402u, 0x18000000u,
    0xbf8c0000u,
    0xe0102000u, 0x80040300u,
    0xbf810000u,
	};
}

constexpr auto Consumer = MakeConsumer(8u, 0x55555555u);
constexpr auto MixedLaneConsumer = MakeConsumer(256u, (1u << 1) | (1u << 2));

Sharp Buffer(uint64_t base, uint32_t stride, uint32_t records) {
	return {static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32u) | (stride << 16u), records,
	        (static_cast<uint32_t>(Prospero::BufferFormat::k32UInt) << 12u) | DstSel(4, 0, 0, 1)};
}

void Register(std::span<const uint32_t> code) {
	const auto address = reinterpret_cast<uint64_t>(code.data());
	ShaderMapUserData(address, {.type = Prospero::ShaderBinaryType::kCs,
	                            .code_size_bytes = static_cast<uint32_t>(code.size_bytes())});
}

IR::ResourcePlan CheckPlan(std::span<const uint32_t> code, unsigned sgprs, std::span<const uint32_t> store_pcs) {
	ShaderComputeInputInfo info {};
	info.wave_size = info.host_subgroup_size = info.threads_num[0] = 32;
	info.float_mode = 0xc0;
	info.threads_num[1] = info.threads_num[2] = 1;
	info.thread_ids_num = 1;
	info.group_id[0] = true;
	info.workgroup_register = sgprs;
	std::array<uint32_t, 10> inputs {};
	RC::CompileOptions options;
	options.wave_size = 32;
	options.dump_ir = false;
	options.user_data = std::span(inputs).first(sgprs);
	options.input_info.compute = &info;
	auto compiled = RC::TranslateProgram(code, options);
	const auto& ir = compiled.program;
	std::vector<uint32_t> found;
	for (auto* block : ir.blocks) {
		for (const auto& inst : *block) {
			if (inst.GetOpcode() != IR::ValueOpcode::StoreBufferU32) continue;
			const auto flags = inst.Flags<IR::MemoryFlags>();
			const auto& access = ir.memory_info.at(flags.index);
			if (!access.formatted || !access.idxen) continue;
			const auto& buffer = ir.info.buffers.at(access.resource);
			const auto& table = ir.descriptor_sources.at(buffer.source).indirect_descriptor;
			Require(buffer.written && table && table->indexed && table->table_stride == 16 &&
			        table->table_offset == 0 && table->table_immediate == 0 && !table->table_maximum &&
			        table->selector && table->selector->stride == 0 && table->selector->offset == Field &&
			        table->selector->bit_offset == 0 && table->selector->bit_mask == Mask,
			        "formatted STORE did not resolve to the masked DWORD at byte 60");
			found.push_back(flags.pc);
		}
	}
	std::ranges::sort(found);
	Require(std::ranges::equal(found, store_pcs), "missing or unexpected formatted STORE PC");
	auto plan = IR::ExtractResourcePlan(ir);
	Require(plan.resource_tracking_complete, "incomplete resource plan");
	return plan;
}

struct Data {
	std::array<uint32_t, Keys.size() * MaterialStride / 4> materials {};
	std::array<Sharp, TableRows + 1> table {};
	std::array<uint32_t, 32> output {};

	explicit Data(uint64_t destination) {
		for (size_t i = 0; i < Keys.size(); ++i) materials[(i * MaterialStride + Field) / 4] = Keys[i];
		output.fill(Sentinel);
		table[2] = Buffer(destination + 4, 4, 3);
		table[5] = Buffer(destination + 32, 4, 3);
		table[3] = Buffer(destination + 64, 4, 3);
		table[1] = table[3];
		table[1][3] |= 1u << 30u;
		table[TableRows] = Buffer(destination + 96, 4, 3);
	}
};

struct PlanReads {
	static constexpr uint64_t Materials = 0x100000, Table = 0x200000, Output = 0x300000;
	Data data {Output};
	uint32_t rows = 0, denied = UINT32_MAX;
	bool unexpected = false;

	static bool Read(void* ptr, uint64_t address, std::span<uint32_t> words) {
		auto& self = *static_cast<PlanReads*>(ptr);
		if (address >= Materials && address - Materials < sizeof(self.data.materials) &&
		    words.size_bytes() <= sizeof(self.data.materials) - (address - Materials)) {
			std::memcpy(words.data(), reinterpret_cast<const char*>(self.data.materials.data()) +
			                          (address - Materials), words.size_bytes());
			return true;
		}
		if (address >= Table && address - Table < sizeof(self.data.table)) {
			const auto row = (address - Table) / sizeof(Sharp);
			if ((row == 0 || row == 2 || row == 5) &&
			    (address - Table) % sizeof(Sharp) + words.size_bytes() <= sizeof(Sharp)) {
				self.rows |= 1u << row;
				if (self.denied == row) return false;
				std::memcpy(words.data(), reinterpret_cast<const char*>(self.data.table.data()) +
				                          (address - Table), words.size_bytes());
				return true;
			}
		}
		self.unexpected = true;
		return false;
	}
};

void SyntheticPlan() {
	const std::array<uint32_t, 1> pcs {0x3c};
	auto plan = CheckPlan(Consumer, 8, pcs);
	PlanReads reads;
	std::array<uint32_t, 8> sgprs;
	std::ranges::copy(Buffer(reads.Materials, MaterialStride, Keys.size()), sgprs.begin());
	std::ranges::copy(Buffer(reads.Table, sizeof(Sharp), TableRows), sgprs.begin() + 4);
	const IR::SrtRuntime runtime {.user_data = sgprs, .userdata = &reads,
	                              .read_specialization_memory = PlanReads::Read};
	IR::ResourceSnapshot snapshot;
	IR::ResourceSpecialization spec;
	const auto materialize = [&] { return IR::MaterializeResources(plan, runtime, snapshot, spec); };
	Require(materialize() && !reads.unexpected && reads.rows == ((1u << 5) | (1u << 2) | 1u),
	        "did not read exactly the named, in-range rows");
	for (const auto row : {2u, 5u}) {
		Require(std::ranges::any_of(snapshot.buffers, [&](const auto& v) {
			return v.dword_count == 4 && std::ranges::equal(std::span(v.dwords).first(4), reads.data.table[row]);
		}), "selected descriptor missing from snapshot");
	}
	constexpr uint32_t LargeTableRows = 196608;
	sgprs[6] = LargeTableRows;
	reads.data.materials[(2 * MaterialStride + Field) / 4] = 0xcc000000u | LargeTableRows;
	reads.rows = 0;
	Require(materialize() && !reads.unexpected && reads.rows == ((1u << 5) | (1u << 2) | 1u),
	        "large STORE table was enumerated or its OOB row was read");
	reads.data.materials[Field / 4] = 2;
	reads.denied = 5;
	reads.rows = 0;
	Require(materialize() && !reads.unexpected && reads.rows == ((1u << 2) | 1u),
	        "a changed material kept reading a stale selection");
	reads.denied = UINT32_MAX;
	reads.data.materials[Field / 4] = Keys[0];
	reads.data.materials[(2 * MaterialStride + Field) / 4] = Keys[2];
	sgprs[6] = TableRows;
	auto& selected = reads.data.table[5];
	const auto good = selected;
	selected = reads.data.table[1];
	Require(!materialize(), "accepted a non-null wrong-kind selected descriptor");
	selected = good;
	selected[3] &= ~(0x7fu << 12u);
	Require(!materialize(), "accepted a nonempty selected descriptor with invalid format");
	selected = Buffer(reads.Materials + Field, 4, 1);
	Require(!materialize(), "accepted a write that aliases captured selector data");
	selected = Buffer(reads.Table + 2 * sizeof(Sharp), 4, 1);
	Require(!materialize(), "accepted a write that aliases another selected descriptor");
	selected = good;
	reads.rows = 0;
	reads.denied = 2;
	Require(!materialize() && (reads.rows & (1u << 2)), "ignored failure to read a selected descriptor");
	Require(!reads.unexpected, "a rejection inspected an unrelated table row");
	std::puts("synthetic masked STORE plan: ok");
}

int Gpu() {
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* reason = gpu.Initialize(false, true)) {
		std::fprintf(stderr, "masked material STORE GPU unavailable: %s\n", reason);
		return std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr ? 1 : 77;
	}
	Require(gpu.graphics.CreateAllocator(), "Vulkan allocator");
	constexpr size_t Page = 0x4000, Bytes = 4 * Page;
	void* guest = nullptr;
	Require(sceKernelMapFlexibleMemory(&guest, Bytes, 3, 0) == 0 && guest != nullptr, "map guest memory");
	const auto base = reinterpret_cast<uint64_t>(guest);
	const auto materials = base + Page, table = base + 2 * Page, output = base + 3 * Page;
	Data data(output);
	data.materials.fill(3);
	Memory::WriteBacking(materials, data.materials.data(), sizeof(data.materials));
	Memory::WriteBacking(table, data.table.data(), sizeof(data.table));
	Memory::WriteBacking(output, data.output.data(), sizeof(data.output));
	std::vector<uint32_t> producer;
	for (size_t i = 0; i < Keys.size(); ++i) {
		producer.insert(producer.end(), {0x7e0002ffu, Keys[i],
		    0xe0700000u | static_cast<uint32_t>(i * MaterialStride + Field), 0x80000000u});
	}
	producer.push_back(0xbf810000u);
	const auto install = [&](uint64_t address, std::span<const uint32_t> code) {
		Memory::WriteBacking(address, code.data(), code.size_bytes());
		Register({reinterpret_cast<const uint32_t*>(address), code.size()});
	};
	install(base, producer);
	install(base + 0x100, Consumer);
	install(base + 0x200, MixedLaneConsumer);
	const auto mixedMaterials = materials + 0x400, mixedTable = table + 0x400, mixedOutput = output + 0x400;
	Data mixed(mixedOutput);
	mixed.materials[Field / 4] = 3;
	mixed.materials[(MaterialStride + Field) / 4] = 0xaa000002u;
	mixed.materials[(2 * MaterialStride + Field) / 4] = 0xbb000005u;
	Memory::WriteBacking(mixedMaterials, mixed.materials.data(), sizeof(mixed.materials));
	Memory::WriteBacking(mixedTable, mixed.table.data(), sizeof(mixed.table));
	Memory::WriteBacking(mixedOutput, mixed.output.data(), sizeof(mixed.output));

	RenderContext context(gpu.graphics);
	context.InitializeGpu(nullptr);
	Memory::InstallGpuResources(&context);
	context.GetGpu().SendCommandSync([&] {
		context.MapMemory(base, Bytes);
		CommandProcessor cp(context, 0);
		cp.BufferInit();
		auto& shaders = cp.GetShCtx();
		auto& cache = context.GetBufferCache();
		constexpr auto Mode = 0x41u | (1u << Pm4::COMPUTE_DISPATCH_INITIATOR_CS_W32_EN_SHIFT);
		const auto set = [&](uint32_t sgpr, const Sharp& desc) {
			for (unsigned i = 0; i < desc.size(); ++i) shaders.SetCsUserSgpr(sgpr + i, desc[i]);
		};
		shaders.SetCsShader({.data_addr = base, .num_thread_x = 1, .num_thread_y = 1,
		                     .num_thread_z = 1, .wave_size = 32, .user_sgpr = 4});
		auto raw = Buffer(materials, 0, sizeof(data.materials));
		raw[3] |= 3u << 28u;
		set(0, raw);
		cp.DispatchDirect(1, 1, 1, Mode);
		for (size_t i = 0; i < Keys.size(); ++i) {
			const auto address = materials + i * MaterialStride + Field;
			uint32_t word = 0;
			Require(cache.HasGpuDirtyBytes(address, 4) && Memory::TryReadBacking(address, &word, 4) && word == 3,
			        "material selector was not exclusively GPU-produced");
		}
		shaders.SetCsShader({.data_addr = base + 0x100, .num_thread_x = 32, .num_thread_y = 1,
		                     .num_thread_z = 1, .wave_size = 32, .user_sgpr = 8, .tgid_x_en = true});
		set(0, Buffer(materials, MaterialStride, Keys.size()));
		set(4, Buffer(table, sizeof(Sharp), TableRows));
		cp.DispatchDirect(Keys.size(), 1, 1, Mode);
		cache.ReadMemory(output, sizeof(data.output));
		auto expected = data.output;
		expected[8] = expected[10] = 17;
		expected[1] = expected[3] = 18;
		Require(Memory::TryReadBacking(output, data.output.data(), sizeof(data.output)) && data.output == expected,
		        "selected output, inactive lane, OOB key/index, stale target, or neighboring canary differs");
		shaders.SetCsShader({.data_addr = base + 0x200, .num_thread_x = 32, .num_thread_y = 1,
		                     .num_thread_z = 1, .wave_size = 32, .user_sgpr = 8});
		set(0, Buffer(mixedMaterials, MaterialStride, Keys.size()));
		set(4, Buffer(mixedTable, sizeof(Sharp), TableRows));
		cp.DispatchDirect(1, 1, 1, Mode);
		cache.ReadMemory(mixedOutput, sizeof(mixed.output));
		auto mixedExpected = mixed.output;
		mixedExpected[2] = 18;
		mixedExpected[3] = 19;
		Require(Memory::TryReadBacking(mixedOutput, mixed.output.data(), sizeof(mixed.output)) &&
		            mixed.output == mixedExpected,
		        "mixed active lanes did not both store through the first active lane's descriptor");
		context.UnmapMemory(base, Bytes);
		context.GetCommandScheduler().Finish();
	});
	Memory::InstallGpuResources(nullptr);
	context.ShutdownGpu();
	Require(sceKernelMunmap(base, Bytes) == 0, "unmap guest memory");
	std::puts("synthetic masked STORE GPU destinations and canaries: ok");
	return 0;
}
} // namespace

extern "C" int RunKytyMaskedMaterialStoreTest(int argc, char** argv) {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	if (argc == 2 && std::string_view(argv[1]) == "plan") {
		SyntheticPlan();
		return 0;
	}
	if (argc == 2 && std::string_view(argv[1]) == "gpu") return Gpu();
	std::fprintf(stderr, "usage: %s plan | gpu\n", argv[0]);
	return 2;
}

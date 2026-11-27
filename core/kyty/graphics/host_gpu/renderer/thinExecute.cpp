#include "graphics/host_gpu/renderer/thinExecute.h"

#include <algorithm>
#include <fmt/format.h>

namespace Libs::Graphics::Thin {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;
using PushData    = ShaderRecompiler::IR::PushData;

constexpr uint64_t KindBit(BindingKind kind) {
	return uint64_t {1} << static_cast<uint32_t>(kind);
}

constexpr uint64_t ImageKinds() {
	uint64_t mask = 0;
	for (uint32_t index = ShaderRecompiler::IR::FirstImageBinding; index < static_cast<uint32_t>(BindingKind::Samplers);
	     ++index) {
		mask |= uint64_t {1} << index;
	}
	return mask;
}

constexpr uint64_t NamedKinds() {
	uint64_t mask = ImageKinds() | KindBit(BindingKind::Buffers) | KindBit(BindingKind::UniformBuffers) |
	                KindBit(BindingKind::FlattenedSrt) | KindBit(BindingKind::Samplers) | KindBit(BindingKind::Gds) |
	                KindBit(BindingKind::SharedMemory) | KindBit(BindingKind::ShaderData) |
	                KindBit(BindingKind::MipStats) | KindBit(BindingKind::FaultBuffer);
#if defined(ANYPS5_KYTY)
	mask |= KindBit(BindingKind::PackedTargets);
#endif
	return mask;
}

constexpr uint64_t Mix(uint64_t hash, uint64_t value) {
	hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
	hash ^= hash >> 31u;
	hash *= 0xbf58476d1ce4e5b9ull;
	return hash ^ (hash >> 29u);
}

constexpr std::array<std::string_view, RefusalCount> RefusalNames {
    "unresolved",     "specialized",   "pipeline wait", "CPU buffers",   "CPU SRT",      "memory sync",
    "GPU flat loads", "CPU images",    "image tables",  "image heap",    "CPU samplers", "indirect args",
    "shared memory",  "GDS",           "ray tracing",   "write targets", "CPU consumed", "ordered append",
    "checks",         "push layout",   "instrumentation",
};

constexpr std::array<std::string_view, static_cast<size_t>(Owner::Count)> OwnerNames {"A", "B", "C", "D", "E"};

constexpr std::array<std::string_view, static_cast<size_t>(DispatchKind::Count)> KindNames {"direct dispatches",
                                                                                         "indirect dispatches"};

}

std::string_view RefusalName(Refusal refusal) {
	const auto index = static_cast<size_t>(refusal);
	return index < RefusalNames.size() ? RefusalNames[index] : std::string_view {"?"};
}

DispatchKey MakeDispatchKey(const HW::CsStageRegisters& regs, uint64_t map_generation, uint32_t mode,
                            bool async_compute, DispatchKind kind) {
	DispatchKey key;
	key.data_addr      = regs.data_addr;
	key.map_generation = map_generation;
	key.threads        = {regs.num_thread_x, regs.num_thread_y, regs.num_thread_z};
	key.mode           = mode;
	key.words0 = uint32_t {regs.vgprs} | (uint32_t {regs.priority} << 8u) | (uint32_t {regs.float_mode} << 16u) |
	             (uint32_t {regs.wave_size} << 24u);
	key.words1 = uint32_t {regs.user_sgpr} | (uint32_t {regs.tidig_comp_cnt} << 8u) | (uint32_t {regs.lds_size} << 16u);
	const bool flags[] = {regs.dx10_clamp,    regs.debug_mode,  regs.ieee_mode,
	                      regs.require_forward_progress,       regs.fp16_overflow,
	                      regs.threadgroup_configuration,      regs.scratch_en,
	                      regs.tgid_x_en,     regs.tgid_y_en,   regs.tgid_z_en,
	                      regs.tg_size_en,    async_compute};
	uint32_t packed = 0;
	for (uint32_t index = 0; index < std::size(flags); ++index) {
		packed |= uint32_t {flags[index]} << index;
	}
	key.words2 = uint32_t {regs.shared_vgprs} | (packed << 8u) | (static_cast<uint32_t>(kind) << 24u);
	return key;
}

size_t DispatchKeyHash::operator()(const DispatchKey& key) const noexcept {
	uint64_t hash = Mix(0, key.data_addr);
	hash          = Mix(hash, key.map_generation);
	hash          = Mix(hash, (uint64_t {key.threads[0]} << 32u) | key.threads[1]);
	hash          = Mix(hash, (uint64_t {key.threads[2]} << 32u) | key.mode);
	hash          = Mix(hash, (uint64_t {key.words0} << 32u) | key.words1);
	hash          = Mix(hash, key.words2);
	return static_cast<size_t>(hash);
}

ProgramFacts DescribeProgram(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                             const ShaderRecompiler::IR::ResourceSnapshot&  snapshot) {
	const auto&  info = program.info;
	ProgramFacts facts;
	facts.descriptor_mask = program.bindings.descriptor_mask;
	facts.buffers         = !info.buffers.empty() || !snapshot.buffers.empty();
	facts.images          = std::ranges::any_of(info.images, [](const auto& image) { return !image.heap; });
	facts.image_tables    = !snapshot.image_tables.empty() ||
	                     std::ranges::any_of(info.images, [](const auto& image) { return image.table_capacity != 0u; });
	facts.image_heap =
	    info.uses_image_heap || std::ranges::any_of(info.images, [](const auto& image) { return image.heap; });
	facts.samplers             = !info.samplers.empty() || !snapshot.samplers.empty();
	facts.flattened_srt        = info.uses_flattened_srt || !snapshot.flattened_srt.empty();
	facts.specialization_reads = !snapshot.specialization_reads.empty();
	facts.gpu_flat_loads       = !snapshot.gpu_flat_loads.empty();
	facts.uniform_fill         = snapshot.uniform_fill.kind != ShaderRecompiler::IR::UniformFillKind::None;
	facts.memory_access = info.uses_dma || info.uses_device_address || program.has_address_writes ||
	                      !info.dma_base_registers.empty() || !info.vsharp_base_registers.empty() ||
	                      !info.indexed_buffer_reads.empty();
	facts.gds           = info.uses_gds;
	facts.ray_tracing   = info.bvh || info.ray_query;
	facts.write_targets = !info.self_sized_buffers.empty() ||
	                      std::ranges::any_of(info.buffers, [](const auto& buffer) { return buffer.indirect_write_table; });
	return facts;
}

RefusalMask Classify(const ProgramFacts& facts) {
	const auto  has      = [&](BindingKind kind) { return (facts.descriptor_mask & KindBit(kind)) != 0; };
	RefusalMask refusals = 0;
	const auto  refuse   = [&](bool condition, Refusal refusal) {
		if (condition) refusals |= Bit(refusal);
	};
	const bool memory_bindings = (facts.descriptor_mask & ~NamedKinds()) != 0 ||
	                             (has(BindingKind::FaultBuffer) && facts.memory_access);
	refuse(facts.specialization_reads, Refusal::Specialized);
	refuse(!facts.pipeline_ready, Refusal::PipelineWait);
	refuse(facts.buffers || has(BindingKind::Buffers) || has(BindingKind::UniformBuffers), Refusal::CpuBuffers);
	refuse(facts.flattened_srt || has(BindingKind::FlattenedSrt), Refusal::CpuSrt);
	refuse(!facts.memory_ready && (facts.memory_access || memory_bindings), Refusal::MemorySync);
	refuse(facts.gpu_flat_loads, Refusal::GpuFlatLoads);
	refuse(facts.images || (facts.descriptor_mask & ImageKinds()) != 0, Refusal::CpuImages);
	refuse(facts.image_tables, Refusal::ImageTables);
	refuse(facts.image_heap, Refusal::ImageHeap);
	refuse(facts.samplers || has(BindingKind::Samplers), Refusal::CpuSamplers);
	refuse(facts.indirect_args_on_cpu, Refusal::IndirectArgs);
	refuse(facts.lds_storage || has(BindingKind::SharedMemory), Refusal::SharedMemory);
	refuse(facts.gds || has(BindingKind::Gds), Refusal::Gds);
	refuse(facts.ray_tracing, Refusal::RayTracing);
	refuse(facts.write_targets, Refusal::WriteTargets);
	refuse(facts.uniform_fill || facts.consumed_on_cpu, Refusal::CpuConsumed);
	refuse(facts.ordered_append, Refusal::OrderedAppend);
	refuse(facts.checked, Refusal::Checks);
	refuse(!facts.push_layout || has(BindingKind::ShaderData), Refusal::PushLayout);
	refuse(has(BindingKind::MipStats) || (has(BindingKind::FaultBuffer) && !facts.memory_access), Refusal::Instrumentation);
#if defined(ANYPS5_KYTY)
	refuse(has(BindingKind::PackedTargets), Refusal::Instrumentation);
#endif
	return refusals;
}

bool MakePushPlan(const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t user_sgpr_count, PushPlan& plan) {
	plan                  = {};
	const auto& bindings  = program.bindings;
	const auto& registers = program.info.user_data_registers;
	plan.start            = bindings.push_data_start_dword;
	plan.dwords           = bindings.ShaderDataDwords();
	plan.dispatch_thread  = bindings.dispatch_thread_dword;
	plan.count            = static_cast<uint32_t>(registers.size());
	const uint32_t threads = plan.dispatch_thread != PushData::NoStart ? 3u : 0u;
	if (bindings.ray_query_dword != PushData::NoStart || registers.size() > plan.sgprs.size()) {
		return false;
	}
	if (plan.start == PushData::NoStart) {
		return plan.dwords == 0;
	}
	if (!PushData::CanFit(plan.start, plan.dwords) || plan.dwords != plan.count + threads ||
	    (threads != 0 && plan.dispatch_thread != plan.count)) {
		return false;
	}
	for (uint32_t index = 0; index < plan.count; ++index) {
		const auto reg = registers[index];
		if (reg < program.user_data_base || reg - program.user_data_base >= user_sgpr_count) {
			return false;
		}
		plan.sgprs[index] = static_cast<uint8_t>(reg - program.user_data_base);
	}
	return true;
}

void PackPushData(const PushPlan& plan, std::span<const uint32_t> user_sgpr, const uint32_t (&dispatch_threads)[3],
                  PushData& out) {
	out = {};
	if (plan.start == PushData::NoStart) {
		return;
	}
	auto* words = out.dwords.data() + plan.start;
	for (uint32_t index = 0; index < plan.count; ++index) {
		const auto sgpr = plan.sgprs[index];
		words[index]    = sgpr < user_sgpr.size() ? user_sgpr[sgpr] : 0u;
	}
	if (plan.dispatch_thread != PushData::NoStart) {
		std::copy(std::begin(dispatch_threads), std::end(dispatch_threads), words + plan.dispatch_thread);
	}
}

bool SamePushData(const PushPlan& plan, std::span<const uint32_t> shader_data, const PushData& packed) {
	if (shader_data.size() != plan.dwords) {
		return false;
	}
	if (plan.start == PushData::NoStart) {
		return plan.dwords == 0;
	}
	return std::equal(shader_data.begin(), shader_data.end(), packed.dwords.begin() + plan.start);
}

const DispatchEntry* DispatchTable::Find(const DispatchKey& key, uint64_t frame) const {
	const auto found = m_entries.find(key);
	if (found == m_entries.end() || frame < found->second.learned || frame - found->second.learned >= RelearnFrames) {
		return nullptr;
	}
	return &found->second;
}

void DispatchTable::Learn(const DispatchKey& key, const DispatchEntry& entry) {
	if (m_entries.size() >= MaxEntries && !m_entries.contains(key)) {
		m_entries.clear();
	}
	m_entries.insert_or_assign(key, entry);
}

void Counters::NoteServed(DispatchKind kind) {
	auto& counts = kinds[static_cast<size_t>(kind)];
	counts.operations++;
	counts.served++;
	counts.without_c++;
	counts.without_cd++;
}

void Counters::NoteRefused(DispatchKind kind, RefusalMask refusals) {
	auto& counts = kinds[static_cast<size_t>(kind)];
	counts.operations++;
	bool first = true;
	for (uint32_t index = 0; index < RefusalCount; ++index) {
		if (((refusals >> index) & 1u) == 0) continue;
		counts.refused[index]++;
		if (first) counts.first[index]++;
		first = false;
	}
	const auto owners = OwnerMask(refusals);
	const auto c      = 1u << static_cast<uint32_t>(Owner::C);
	const auto d      = 1u << static_cast<uint32_t>(Owner::D);
	if ((owners & c) == 0) counts.without_c++;
	if ((owners & (c | d)) == 0) counts.without_cd++;
}

void Counters::NoteTime(DispatchKind kind, bool thin, uint64_t ns) {
	auto& counts = kinds[static_cast<size_t>(kind)];
	(thin ? counts.thin_ns : counts.full_ns) += ns;
	(thin ? counts.thin_timed : counts.full_timed)++;
}

std::string Counters::Summary(double seconds, uint64_t frames) const {
	const auto percent = [](uint64_t part, uint64_t whole) {
		return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
	};
	const auto micro = [](uint64_t ns, uint64_t count) {
		return count != 0 ? static_cast<double>(ns) / 1e3 / static_cast<double>(count) : 0.0;
	};
	std::string line = fmt::format("[kyty-thin] summary of the last {:.1f} s, {} frames", seconds, frames);
	for (size_t kind = 0; kind < kinds.size(); ++kind) {
		const auto& counts = kinds[kind];
		line += fmt::format("; {}: {} ({:.1f} per frame), thin {} ({:.1f} %), without C refusals {:.1f} %, "
		                    "without C or D refusals {:.1f} %, us per operation thin {:.2f} full {:.2f}",
		                    KindNames[kind], counts.operations,
		                    frames != 0 ? static_cast<double>(counts.operations) / static_cast<double>(frames) : 0.0,
		                    counts.served, percent(counts.served, counts.operations),
		                    percent(counts.without_c, counts.operations), percent(counts.without_cd, counts.operations),
		                    micro(counts.thin_ns, counts.thin_timed), micro(counts.full_ns, counts.full_timed));
		std::string refused;
		for (size_t index = 0; index < RefusalCount; ++index) {
			if (counts.refused[index] == 0) continue;
			const auto owner = RefusalOwner(static_cast<Refusal>(index));
			refused += fmt::format("{}{} ({}) {} (first {})", refused.empty() ? "" : ", ", RefusalNames[index],
			                       OwnerNames[static_cast<size_t>(owner)], counts.refused[index], counts.first[index]);
		}
		if (!refused.empty()) line += fmt::format(", refused: {}", refused);
	}
	return line + "\n";
}

}

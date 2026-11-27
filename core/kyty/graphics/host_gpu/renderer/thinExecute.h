#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_THINEXECUTE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_THINEXECUTE_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

namespace Libs::Graphics::Thin {

enum class Refusal : uint8_t {
	Unresolved,
	Specialized,
	PipelineWait,
	CpuBuffers,
	CpuSrt,
	MemorySync,
	GpuFlatLoads,
	CpuImages,
	ImageTables,
	ImageHeap,
	CpuSamplers,
	IndirectArgs,
	SharedMemory,
	Gds,
	RayTracing,
	WriteTargets,
	CpuConsumed,
	OrderedAppend,
	Checks,
	PushLayout,
	Instrumentation,
	Count,
};

inline constexpr size_t RefusalCount = static_cast<size_t>(Refusal::Count);
using RefusalMask                    = uint32_t;
static_assert(RefusalCount <= 32);

[[nodiscard]] constexpr RefusalMask Bit(Refusal refusal) {
	return RefusalMask {1} << static_cast<uint32_t>(refusal);
}

enum class Owner : uint8_t { A, B, C, D, E, Count };

[[nodiscard]] constexpr Owner RefusalOwner(Refusal refusal) {
	switch (refusal) {
		case Refusal::Specialized:
		case Refusal::PipelineWait: return Owner::B;
		case Refusal::CpuBuffers:
		case Refusal::CpuSrt:
		case Refusal::MemorySync:
		case Refusal::GpuFlatLoads: return Owner::A;
		case Refusal::CpuImages:
		case Refusal::ImageTables:
		case Refusal::ImageHeap:
		case Refusal::CpuSamplers: return Owner::E;
		case Refusal::IndirectArgs: return Owner::D;
		default: return Owner::C;
	}
}

[[nodiscard]] constexpr uint32_t OwnerMask(RefusalMask refusals) {
	uint32_t owners = 0;
	for (uint32_t index = 0; index < RefusalCount; ++index) {
		if ((refusals >> index) & 1u) owners |= 1u << static_cast<uint32_t>(RefusalOwner(static_cast<Refusal>(index)));
	}
	return owners;
}

[[nodiscard]] std::string_view RefusalName(Refusal refusal);

enum class DispatchKind : uint8_t { Direct, Indirect, Count };

struct DispatchKey {
	uint64_t                data_addr      = 0;
	uint64_t                map_generation = 0;
	std::array<uint32_t, 3> threads {};
	uint32_t                mode   = 0;
	uint32_t                words0 = 0;
	uint32_t                words1 = 0;
	uint32_t                words2 = 0;

	bool operator==(const DispatchKey& other) const = default;
};

[[nodiscard]] DispatchKey MakeDispatchKey(const HW::CsStageRegisters& regs, uint64_t map_generation, uint32_t mode,
                                          bool async_compute, DispatchKind kind);

struct DispatchKeyHash {
	size_t operator()(const DispatchKey& key) const noexcept;
};

struct ProgramFacts {
	uint64_t descriptor_mask      = 0;
	bool     buffers              = false;
	bool     images               = false;
	bool     image_tables         = false;
	bool     image_heap           = false;
	bool     samplers             = false;
	bool     flattened_srt        = false;
	bool     specialization_reads = false;
	bool     gpu_flat_loads       = false;
	bool     uniform_fill         = false;
	bool     memory_access        = false;
	bool     memory_ready         = false;
	bool     gds                  = false;
	bool     ray_tracing          = false;
	bool     lds_storage          = false;
	bool     write_targets        = false;
	bool     ordered_append       = false;
	bool     pipeline_ready       = false;
	bool     push_layout          = false;
	bool     indirect_args_on_cpu = false;
	bool     consumed_on_cpu      = false;
	bool     checked              = false;
};

[[nodiscard]] ProgramFacts DescribeProgram(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                           const ShaderRecompiler::IR::ResourceSnapshot&  snapshot);

[[nodiscard]] RefusalMask Classify(const ProgramFacts& facts);

struct PushPlan {
	uint32_t                                                    start           = ShaderRecompiler::IR::PushData::NoStart;
	uint32_t                                                    dwords          = 0;
	uint32_t                                                    dispatch_thread = ShaderRecompiler::IR::PushData::NoStart;
	uint32_t                                                    count           = 0;
	std::array<uint8_t, ShaderRecompiler::IR::PushData::DwordCount> sgprs {};
};

[[nodiscard]] bool MakePushPlan(const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t user_sgpr_count,
                                PushPlan& plan);

void PackPushData(const PushPlan& plan, std::span<const uint32_t> user_sgpr, const uint32_t (&dispatch_threads)[3],
                  ShaderRecompiler::IR::PushData& out);

[[nodiscard]] bool SamePushData(const PushPlan& plan, std::span<const uint32_t> shader_data,
                                const ShaderRecompiler::IR::PushData& packed);

struct DispatchPlan {
	vk::Pipeline            pipeline    = nullptr;
	vk::PipelineLayout      layout      = nullptr;
	vk::ShaderStageFlags    push_stages = {};
	PushPlan                push;
	bool                    thread_dimensions = false;
	bool                    writes            = false;
	uint64_t                shader_hash       = 0;
};

struct DispatchEntry {
	RefusalMask  refusals = 0;
	uint64_t     learned  = 0;
	DispatchPlan plan;
};

class DispatchTable {
public:
	static constexpr size_t   MaxEntries     = 1u << 16u;
	static constexpr uint64_t RelearnFrames  = 64;

	[[nodiscard]] const DispatchEntry* Find(const DispatchKey& key, uint64_t frame) const;
	void                               Learn(const DispatchKey& key, const DispatchEntry& entry);
	void                               Clear() { m_entries.clear(); }
	[[nodiscard]] size_t               Size() const { return m_entries.size(); }

private:
	std::unordered_map<DispatchKey, DispatchEntry, DispatchKeyHash> m_entries;
};

struct Counters {
	struct Kind {
		uint64_t                           operations = 0;
		uint64_t                           served     = 0;
		std::array<uint64_t, RefusalCount> refused {};
		std::array<uint64_t, RefusalCount> first {};
		uint64_t                           without_c  = 0;
		uint64_t                           without_cd = 0;
		uint64_t                           thin_ns    = 0;
		uint64_t                           thin_timed = 0;
		uint64_t                           full_ns    = 0;
		uint64_t                           full_timed = 0;
	};
	std::array<Kind, static_cast<size_t>(DispatchKind::Count)> kinds {};

	void NoteServed(DispatchKind kind);
	void NoteRefused(DispatchKind kind, RefusalMask refusals);
	void NoteTime(DispatchKind kind, bool thin, uint64_t ns);
	[[nodiscard]] std::string Summary(double seconds, uint64_t frames) const;
};

inline constexpr uint32_t TimingPeriod = 64;

}

#endif

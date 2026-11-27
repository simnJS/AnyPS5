#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/timer.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/thinExecute.h"
#include "graphics/host_gpu/renderer/waveSplitCheck.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <span>
#include <utility>

namespace Libs::Graphics {

struct RenderExecutor::ThinDispatches {
	Thin::DispatchTable table;
	Thin::Counters      counters;
	Thin::DispatchKind  kind = Thin::DispatchKind::Direct;
	Thin::DispatchKey   key;
	Thin::RefusalMask   refusals  = Thin::Bit(Thin::Refusal::Unresolved);
	bool                active    = false;
	bool                learning  = false;
	bool                served    = false;
	uint32_t            ticks     = 0;
	uint64_t            log_ns    = 0;
	uint64_t            log_frame = 0;
};

static constexpr uint64_t ThinSummaryNs     = 10'000'000'000ull;
static constexpr uint32_t ThinSummaryPeriod = 1024;

RenderExecutor::ThinDispatches& RenderExecutor::OwnThin() {
	if (!m_thin) {
		m_thin = {new ThinDispatches, [](ThinDispatches* thin) { delete thin; }};
	}
	return *m_thin;
}

RenderExecutor::ThinOperation::ThinOperation(RenderExecutor& executor, bool indirect): m_executor(executor) {
	auto& thin    = executor.OwnThin();
	thin.kind     = indirect ? Thin::DispatchKind::Indirect : Thin::DispatchKind::Direct;
	thin.active   = true;
	thin.learning = false;
	thin.served   = false;
	thin.refusals = Thin::Bit(Thin::Refusal::Unresolved);
	if (++thin.ticks % Thin::TimingPeriod == 0) {
		m_begin = Common::SteadyNs();
	}
}

RenderExecutor::ThinOperation::~ThinOperation() {
	auto& thin = *m_executor.m_thin;
	if (!thin.active) {
		return;
	}
	thin.active = false;
	if (thin.served) {
		thin.counters.NoteServed(thin.kind);
	} else {
		thin.counters.NoteRefused(thin.kind, thin.refusals);
	}
	if (m_begin != 0) {
		thin.counters.NoteTime(thin.kind, thin.served, Common::SteadyNs() - m_begin);
	}
	if (thin.ticks % ThinSummaryPeriod != 0) {
		return;
	}
	const auto now   = Common::SteadyNs();
	const auto frame = m_executor.m_context.FrameNumber();
	if (thin.log_ns == 0) {
		thin.log_ns    = now;
		thin.log_frame = frame;
		thin.counters  = {};
		return;
	}
	if (now - thin.log_ns < ThinSummaryNs) {
		return;
	}
	Log::WriteToConsoleAndLog(
	    thin.counters.Summary(static_cast<double>(now - thin.log_ns) / 1e9, frame - thin.log_frame));
	thin.log_ns    = now;
	thin.log_frame = frame;
	thin.counters  = {};
}

bool RenderExecutor::TryThinDispatch(CommandBuffer& buffer, const HW::ComputeShaderInfo& cs_regs,
                                     const uint32_t (&groups)[3], uint32_t mode, bool async_compute) {
	auto& thin = OwnThin();
	thin.key   = Thin::MakeDispatchKey(cs_regs.cs_regs, ShaderMapGeneration(), mode, async_compute, thin.kind);
	const auto* entry = thin.table.Find(thin.key, m_context.FrameNumber());
	if (entry == nullptr) {
		thin.learning = true;
		return false;
	}
	if (entry->refusals != 0 || thin.kind != Thin::DispatchKind::Direct) {
		thin.refusals = entry->refusals != 0 ? entry->refusals : Thin::Bit(Thin::Refusal::IndirectArgs);
		return false;
	}
	const auto& plan = entry->plan;
	buffer.EndRendering();
	uint32_t       group_counts[3] {groups[0], groups[1], groups[2]};
	const uint32_t no_threads[3] {};
	if (plan.thread_dimensions) {
		const uint32_t sizes[] {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y, cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size    = std::max(sizes[axis], 1u);
			group_counts[axis] = groups[axis] / size + (groups[axis] % size != 0u);
		}
	}
	ShaderRecompiler::IR::PushData push;
	Thin::PackPushData(plan.push,
	                   std::span<const uint32_t> {cs_regs.cs_user_sgpr.value,
	                                              std::min<size_t>(cs_regs.cs_regs.user_sgpr,
	                                                               HW::UserSgprInfo::SGPRS_MAX)},
	                   plan.thread_dimensions ? groups : no_threads, push);
	m_context.GetOcclusionQueries().WriteForReaders();
	const auto vk_buffer = buffer.Handle();
	if (plan.push.start != ShaderRecompiler::IR::PushData::NoStart) {
		vk_buffer.pushConstants(plan.layout, plan.push_stages, 0, sizeof(push), push.dwords.data());
	}
	auto& scheduler = m_context.GetCommandScheduler();
#if defined(ANYPS5_KYTY)
	scheduler.GetMasterSemaphore().NoteWork(scheduler.CurrentTick(), plan.shader_hash);
#endif
	const bool chained = std::exchange(m_dispatch_chained, false) && m_deferred_barrier;
	if (plan.writes && !chained) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, plan.pipeline);
	m_context.NoteFirstGuestCommand(true);
	vk_buffer.dispatch(group_counts[0], group_counts[1], group_counts[2]);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	m_deferred_barrier       = true;
	m_deferred_barrier_queue = m_guest_queue;
	thin.served              = true;
	return true;
}

void RenderExecutor::LearnThinDispatch(const ShaderComputeInputInfo& input_info, const PipelineCache::Pipeline& pipeline,
                                       const PreparedBindings& bindings, const HW::ComputeShaderInfo& cs_regs,
                                       uint32_t mode, bool writes, bool indirect_writes) {
	auto& thin = OwnThin();
	if (!thin.active || !thin.learning) {
		return;
	}
	thin.learning      = false;
	const auto& module = *input_info.stage.module;
	const auto& program = module.info;
	constexpr uint32_t OrderedAppend = 1u << 3u;
	auto facts = Thin::DescribeProgram(program, *input_info.stage.resources);
	facts.pipeline_ready       = pipeline.pipeline != nullptr && !pipeline.uses_image_heap;
	facts.lds_storage          = input_info.lds_storage || bindings.shared_memory.buffer != nullptr;
	facts.write_targets        = facts.write_targets || indirect_writes;
	facts.ordered_append       = (mode & OrderedAppend) != 0 && program.info.uses_gds;
	facts.indirect_args_on_cpu = thin.kind == Thin::DispatchKind::Indirect;
	facts.checked              = Config::WaveSplitCheckPeriod() != 0 && WaveSplitCheck::Candidate(input_info);
	Thin::DispatchEntry entry;
	auto&               plan = entry.plan;
	const auto user_sgprs = std::min<uint32_t>(cs_regs.cs_regs.user_sgpr, HW::UserSgprInfo::SGPRS_MAX);
	if (Thin::MakePushPlan(program, user_sgprs, plan.push)) {
		ShaderRecompiler::IR::PushData packed;
		Thin::PackPushData(plan.push, std::span<const uint32_t> {cs_regs.cs_user_sgpr.value, user_sgprs},
		                   input_info.dispatch_threads_num, packed);
		facts.push_layout = Thin::SamePushData(plan.push, bindings.shader_data, packed);
	}
	entry.refusals = Thin::Classify(facts);
	entry.learned  = m_context.FrameNumber();
	plan.pipeline          = pipeline.pipeline;
	plan.layout            = pipeline.pipeline_layout;
	plan.push_stages       = pipeline.push_stages;
	plan.thread_dimensions = input_info.dispatch_thread_dimensions;
	plan.writes            = writes;
	plan.shader_hash       = program.shader_hash;
	thin.table.Learn(thin.key, entry);
	thin.refusals = entry.refusals;
}

void RenderExecutor::NoteThinRefused(Thin::Refusal refusal, bool remember) {
	auto& thin = OwnThin();
	if (!thin.active) {
		return;
	}
	thin.refusals = Thin::Bit(refusal);
	if (remember && thin.learning) {
		thin.table.Learn(thin.key, {.refusals = thin.refusals, .learned = m_context.FrameNumber()});
	}
	thin.learning = false;
}

}

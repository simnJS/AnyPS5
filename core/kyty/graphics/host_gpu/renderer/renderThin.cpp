#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

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
	ThinOperation*      current   = nullptr;
	uint32_t            ticks     = 0;
	uint64_t            log_ns     = 0;
	uint64_t            log_frame  = 0;
	uint64_t            open_ns    = 0;
	uint64_t            open_frame = 0;
};

static constexpr uint64_t ThinSummaryNs     = 10'000'000'000ull;
static constexpr uint32_t ThinSummaryPeriod = 1024;

RenderExecutor::ThinDispatches& RenderExecutor::OwnThin() {
	if (!m_thin) {
		m_thin             = {new ThinDispatches, [](ThinDispatches* thin) { delete thin; }};
		m_thin->open_ns    = Common::SteadyNs();
		m_thin->open_frame = m_context.FrameNumber();
	}
	return *m_thin;
}

void RenderExecutor::ReportThin() {
	if (!m_thin || m_thin->current != nullptr) {
		return;
	}
	auto& thin = *m_thin;
	if (std::ranges::none_of(thin.counters.kinds, [](const auto& kind) { return kind.operations != 0; })) {
		return;
	}
	const auto now   = Common::SteadyNs();
	const auto frame = m_context.FrameNumber();
	const auto since = thin.log_ns != 0 ? thin.log_ns : thin.open_ns;
	const auto first = thin.log_ns != 0 ? thin.log_frame : thin.open_frame;
	Log::WriteToConsoleAndLog(thin.counters.Summary(static_cast<double>(now - since) / 1e9, frame - first));
	thin.log_ns    = now;
	thin.log_frame = frame;
	thin.counters  = {};
}

RenderExecutor::ThinOperation::ThinOperation(RenderExecutor& executor, Thin::OperationKind kind)
    : m_executor(executor), m_kind(kind) {
	auto& thin   = executor.OwnThin();
	m_outer      = std::exchange(thin.current, this);
	if (++thin.ticks % Thin::TimingPeriod == 0) {
		m_begin = Common::SteadyNs();
	}
}

RenderExecutor::ThinOperation::~ThinOperation() {
	auto& thin   = *m_executor.m_thin;
	thin.current = m_outer;
	if (m_served) {
		thin.counters.NoteServed(m_kind);
	} else {
		thin.counters.NoteRefused(m_kind, m_refusals);
	}
	if (m_begin != 0) {
		thin.counters.NoteTime(m_kind, m_served, Common::SteadyNs() - m_begin);
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
	auto* op   = thin.current;
	if (op == nullptr) {
		return false;
	}
	op->m_key         = Thin::MakeDispatchKey(cs_regs.cs_regs, ShaderMapGeneration(), mode, async_compute, op->m_kind);
	const auto* entry = thin.table.Find(op->m_key, m_context.FrameNumber());
	if (entry == nullptr) {
		op->m_learning = true;
		return false;
	}
	if (entry->refusals != 0 || op->m_kind != Thin::OperationKind::Direct) {
		op->m_refusals = entry->refusals != 0 ? entry->refusals : Thin::Bit(Thin::Refusal::IndirectArgs);
		return false;
	}
	const auto& plan = entry->plan;
	buffer.EndRendering();
	uint32_t       group_counts[3] {groups[0], groups[1], groups[2]};
	const uint32_t no_threads[3] {};
	if (plan.thread_dimensions) {
		const uint32_t sizes[] {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                        cs_regs.cs_regs.num_thread_z};
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
	if (GpuBreadcrumbs::PerOperation()) {
		GpuBreadcrumbs::NoteProgram(*plan.module);
		scheduler.GetMasterSemaphore().NoteWork(scheduler.CurrentTick(), plan.module->info.shader_hash);
	} else {
		scheduler.GetMasterSemaphore().NoteWork(scheduler.CurrentTick(), plan.shader_hash);
	}
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
	op->m_served             = true;
	return true;
}

static bool ThinPushLayout(const ShaderRecompiler::IR::CompiledShaderInfo& program, std::span<const uint32_t> user_data,
                           const uint32_t (&dispatch_threads)[3], std::span<const uint32_t> shader_data,
                           Thin::PushPlan& plan) {
	if (!Thin::MakePushPlan(program, static_cast<uint32_t>(user_data.size()), plan)) {
		return false;
	}
	ShaderRecompiler::IR::PushData packed;
	Thin::PackPushData(plan, user_data, dispatch_threads, packed);
	return Thin::SamePushData(plan, shader_data, packed);
}

void RenderExecutor::LearnThinDispatch(const ShaderComputeInputInfo& input_info, const PipelineCache::Pipeline& pipeline,
                                       const PreparedBindings& bindings, const HW::ComputeShaderInfo& cs_regs,
                                       uint32_t mode, bool writes, bool indirect_writes) {
	auto* op = OwnThin().current;
	if (op == nullptr || !op->m_learning) {
		return;
	}
	op->m_learning      = false;
	const auto& program = input_info.stage.module->info;
	constexpr uint32_t OrderedAppend = 1u << 3u;
	auto facts = Thin::DescribeProgram(program, *input_info.stage.resources, input_info.stage.module->specialization);
	Thin::NotePipeline(facts, pipeline.pipeline != nullptr, pipeline.uses_image_heap);
	facts.lds_storage          = input_info.lds_storage || bindings.shared_memory.buffer != nullptr;
	facts.write_targets        = facts.write_targets || indirect_writes;
	facts.ordered_append       = (mode & OrderedAppend) != 0 && program.info.uses_gds;
	facts.indirect_args_on_cpu = op->m_kind != Thin::OperationKind::Direct;
	facts.checked              = Config::WaveSplitCheckPeriod() != 0 && WaveSplitCheck::Candidate(input_info);
	Thin::DispatchEntry entry;
	auto&               plan = entry.plan;
	const auto user_sgprs = std::min<uint32_t>(cs_regs.cs_regs.user_sgpr, HW::UserSgprInfo::SGPRS_MAX);
	facts.push_layout     = ThinPushLayout(program, std::span<const uint32_t> {cs_regs.cs_user_sgpr.value, user_sgprs},
	                                       input_info.dispatch_threads_num, bindings.shader_data, plan.push);
	entry.refusals         = Thin::Classify(facts);
	entry.learned          = m_context.FrameNumber();
	plan.pipeline          = pipeline.pipeline;
	plan.layout            = pipeline.pipeline_layout;
	plan.push_stages       = pipeline.push_stages;
	plan.thread_dimensions = input_info.dispatch_thread_dimensions;
	plan.writes            = writes;
	if (GpuBreadcrumbs::PerOperation()) plan.module = input_info.stage.module;
	else plan.shader_hash = program.shader_hash;
	OwnThin().table.Learn(op->m_key, entry);
	op->m_refusals = entry.refusals;
}

void RenderExecutor::NoteThinRefused(Thin::Refusal refusal, bool remember) {
	auto& thin = OwnThin();
	auto* op   = thin.current;
	if (op == nullptr) {
		return;
	}
	op->m_refusals = Thin::Bit(refusal);
	if (remember && op->m_learning) {
		thin.table.Learn(op->m_key, {.refusals = op->m_refusals, .learned = m_context.FrameNumber()});
	}
	op->m_learning = false;
}

void RenderExecutor::ClassifyThinDraw(std::span<PreparedBindings* const> stages, const PipelineCache::Pipeline& pipeline,
                                      Thin::RefusalMask draw_refusals) {
	auto* op = OwnThin().current;
	if (op == nullptr) {
		return;
	}
	const uint32_t    no_threads[3] {};
	Thin::RefusalMask refusals = draw_refusals;
	for (const auto* prepared: stages) {
		const auto& runtime  = *prepared->runtime;
		const auto& program  = runtime.module->info;
		auto        facts    = Thin::DescribeProgram(program, *runtime.resources, runtime.module->specialization);
		Thin::NotePipeline(facts, pipeline.pipeline != nullptr, pipeline.uses_image_heap);
		facts.lds_storage    = prepared->shared_memory.buffer != nullptr;
		facts.indirect_args_on_cpu = op->m_kind == Thin::OperationKind::IndirectDraw;
		Thin::PushPlan plan;
		facts.push_layout = ThinPushLayout(program, runtime.resources->user_data, no_threads, prepared->shader_data, plan);
		refusals |= Thin::Classify(facts);
	}
	op->m_refusals = refusals;
}

}

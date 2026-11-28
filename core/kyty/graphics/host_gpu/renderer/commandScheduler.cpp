#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"
#include "graphics/host_gpu/renderer/stallSnapshot.h"
#if defined(ANYPS5_KYTY)
#include "graphics/guest_gpu/graphicsRun.h"
#endif

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <span>
#include <tuple>

#if defined(ANYPS5_KYTY)
#include <cstdlib>
#include <string_view>
#endif

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = MemoryTelemetry::Create(graphics.device, create, m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	MemoryTelemetry::Destroy(m_graphics.device, m_pool);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	return m_buffers[Acquire(m_master.CurrentTick())];
}

std::pair<vk::CommandBuffer, size_t> CommandScheduler::CommandPool::CommitDeferred() {
	const auto index = Acquire(std::numeric_limits<uint64_t>::max());
	return {m_buffers[index], index};
}

void CommandScheduler::CommandPool::Retire(size_t index, uint64_t tick) {
	EXIT_IF(index >= m_ticks.size() || m_ticks[index] != std::numeric_limits<uint64_t>::max());
	m_ticks[index] = tick;
}

void CommandScheduler::CommandPool::TrimCompleted() {
	if (m_ticks.empty()) return;
	m_master.Refresh();
	const auto done = m_master.KnownGpuTick();
	size_t keep = m_ticks.size();
	while (keep > GrowStep && m_ticks[keep - 1] <= done) {
		--keep;
	}
	if (keep < m_ticks.size()) {
		m_graphics.device.freeCommandBuffers(m_pool, static_cast<uint32_t>(m_ticks.size() - keep), m_buffers.data() + keep);
		m_ticks.resize(keep);
		m_buffers.resize(keep);
		m_hint %= keep;
	}
	if (std::ranges::all_of(m_ticks, [done](uint64_t tick) { return tick <= done; })) {
		EXIT_IF(m_graphics.device.resetCommandPool(m_pool, vk::CommandPoolResetFlagBits::eReleaseResources) != vk::Result::eSuccess);
	} else {
		for (size_t i = 0; i < keep; ++i) {
			if (m_ticks[i] <= done) {
				EXIT_IF(m_buffers[i].reset(vk::CommandBufferResetFlagBits::eReleaseResources) != vk::Result::eSuccess);
			}
		}
		m_graphics.device.trimCommandPool(m_pool, {});
	}
}

size_t CommandScheduler::CommandPool::Acquire(uint64_t busy_until) {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick, busy_until](size_t begin,
	                                                  size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = busy_until;
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = busy_until;
	}

	m_hint = (*found + 1) % m_ticks.size();
	return *found;
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics,
                                   bool deferred_recording, const char* name)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_deferred(deferred_recording),
      m_stream(&m_chunk_queue), m_command(*this, deferred_recording ? &m_stream : nullptr),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }),
      m_snapshot_name(std::string("scheduler of the ") + name + " ticks") {
	StallSnapshot::Name(m_master.Handle(), std::string("the ") + name + " ticks", true);
	StallSnapshot::Register(this, m_snapshot_name.c_str(), &CommandScheduler::WriteSnapshot);
	if (GpuBreadcrumbs::Requested()) {
		m_marker_slot = GpuBreadcrumbs::SubmissionSlot(name);
		m_master.SetMarkerSlot(m_marker_slot, name);
	}
	if (m_deferred) {
		m_host_copies = std::make_unique<HostCopyQueue>(
		    [] { Config::ConfigureGpuStageThread(Config::GpuStageThread::HostCopy); });
		m_recording_thread = std::jthread([this] { RecordingThread(); });
	}
}

CommandScheduler::~CommandScheduler() {
	StallSnapshot::Unregister(this);
	StallSnapshot::Forget(m_master.Handle());
	Shutdown();
}

void CommandScheduler::EnterRecord(RecordStage stage, uint64_t tick) noexcept {
	m_record_tick.store(tick, std::memory_order_relaxed);
	m_record_since.store(Common::SteadyNs(), std::memory_order_relaxed);
	m_record_stage.store(static_cast<uint32_t>(stage), std::memory_order_release);
}

void CommandScheduler::EnterSubmit(SubmitStage stage, uint64_t tick) noexcept {
	m_submit_tick.store(tick, std::memory_order_relaxed);
	m_submit_since.store(Common::SteadyNs(), std::memory_order_relaxed);
	m_submit_stage.store(static_cast<uint32_t>(stage), std::memory_order_release);
}

void CommandScheduler::WriteSnapshot(void* self, std::FILE* file) {
	auto&      scheduler = *static_cast<CommandScheduler*>(self);
	const auto now       = Common::SteadyNs();
	const auto ago       = [now](uint64_t then) { return then == 0 || then > now ? 0.0 : static_cast<double>(now - then) / 1e9; };
	static constexpr const char* RecordStages[] {"waiting for recorded commands", "replaying recorded commands",
	                                             "waiting for the host copies of a submission", "submitting"};
	static constexpr const char* SubmitStages[] {"none", "waiting for the queue lock", "in vkQueueSubmit"};
	if (scheduler.m_deferred) {
		const auto stage = std::min<uint32_t>(scheduler.m_record_stage.load(std::memory_order_acquire), std::size(RecordStages) - 1);
		std::fprintf(file, "recording thread: %s (tick %" PRIu64 ") for %.3f s\n", RecordStages[stage],
		             scheduler.m_record_tick.load(std::memory_order_relaxed),
		             ago(scheduler.m_record_since.load(std::memory_order_relaxed)));
		if (scheduler.m_host_copies != nullptr) {
			std::fprintf(file, "host copies: %" PRIu64 " handed to their thread, %" PRIu64 " done\n",
			             scheduler.m_host_copies->Published(), scheduler.m_host_copies->Completed());
		}
	} else {
		std::fprintf(file, "recorded on the submitting thread\n");
	}
	const auto submit = std::min<uint32_t>(scheduler.m_submit_stage.load(std::memory_order_acquire), std::size(SubmitStages) - 1);
	if (submit == 0) {
		std::fprintf(file, "queue submission: none in progress (the last was tick %" PRIu64 ", %.3f s ago)\n",
		             scheduler.m_submit_tick.load(std::memory_order_relaxed),
		             ago(scheduler.m_submit_since.load(std::memory_order_relaxed)));
	} else {
		std::fprintf(file, "queue submission: tick %" PRIu64 " %s for %.3f s\n", scheduler.m_submit_tick.load(std::memory_order_relaxed),
		             SubmitStages[submit], ago(scheduler.m_submit_since.load(std::memory_order_relaxed)));
	}
	const auto front = scheduler.m_pending_front_tick.load(std::memory_order_acquire);
	if (front != UINT64_MAX) std::fprintf(file, "deferred operations wait for tick %" PRIu64 "\n", front);
	scheduler.m_master.WriteSubmissions(file, 24);
}

void CommandScheduler::Shutdown() {
	{
		Common::UniqueLock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock.Native(), [this]() KYTY_REQUIRES(m_operation_mutex) { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	if (m_recording_thread.joinable()) {
		m_chunk_queue.Stop();
		m_recording_thread.join();
	}
	PopPendingOperations();
	DrainPriorityOperations();
#if defined(ANYPS5_KYTY)
	DrainFlipCompletions();
	if (m_flip_thread.joinable()) {
		m_flip_thread.request_stop();
		m_flip_thread.join();
	}
#endif
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		Common::ScopedLock lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active_tick != 0);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(const HW::Context& registers, const HW::UserConfig& user_config,
                             const HW::Shader& shaders) {
	if (!m_command.IsInvalid()) {
		m_command.Bind(registers, user_config, shaders);
		return;
	}
	{
		Common::ScopedLock lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);
	BeginNext();
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::TrimCommandPool() {
	if (m_deferred) {
		(void)m_stream.RecordSpecial<uint8_t>(CommandStream::RecordKind::TrimPool);
		m_stream.Publish();
	} else {
		m_command_pool.TrimCompleted();
	}
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	bool refreshed = false;
	if (const auto front = m_pending_front_tick.load(std::memory_order_acquire);
	    front == UINT64_MAX) {
		return;
	} else if (!m_master.IsFree(front)) {
		constexpr uint32_t RefreshInterval = 8;
		if (const auto skips = m_pending_skips.load(std::memory_order_relaxed) + 1;
		    skips < RefreshInterval) {
			m_pending_skips.store(skips, std::memory_order_relaxed);
			return;
		}
		m_pending_skips.store(0, std::memory_order_relaxed);
		m_master.Refresh();
		refreshed = true;
		if (!m_master.IsFree(front)) {
			return;
		}
	}
	for (;;) {
		PendingOperation operation;
		{
			Common::ScopedLock lock(m_operation_mutex);
			if (m_pending_operations.empty()) {
				return;
			}
			if (!m_master.IsFree(m_pending_operations.front().tick)) {
				if (refreshed) {
					return;
				}
				m_master.Refresh();
				refreshed = true;
				if (!m_master.IsFree(m_pending_operations.front().tick)) {
					return;
				}
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
			m_pending_front_tick.store(m_pending_operations.empty()
			                               ? UINT64_MAX
			                               : m_pending_operations.front().tick,
			                           std::memory_order_release);
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation, uint64_t address,
                                      uint64_t size) {
	QueueOperation(TrackGuestOperation(std::move(operation), address, size));
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation,
                                              uint64_t address, uint64_t size) {
	QueuePriorityOperation(TrackGuestOperation(std::move(operation), address, size));
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&&                     operation,
                                              const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	QueuePriorityOperation(TrackGuestOperation(std::move(operation), ranges));
}

void CommandScheduler::DeferHostOperation(Common::UniqueFunction<void>&& operation,
                                          bool                           priority) {
	if (priority) {
		QueuePriorityOperation(std::move(operation));
	} else {
		QueueOperation(std::move(operation));
	}
}

#if defined(ANYPS5_KYTY)
static bool EarlyFlipEnabled() {
	static const bool enabled = [] {
		const char*            value = std::getenv("KYTY_EARLY_FLIP");
		const std::string_view text {value == nullptr ? "" : value};
		if (!text.empty() && text != "0" && text != "1") {
			EXIT("invalid KYTY_EARLY_FLIP: %s (0 or 1)\n", value);
		}
		return text != "0";
	}();
	return enabled;
}

void CommandScheduler::DeferFlipCompletion(Common::UniqueFunction<void>&& complete) {
	CheckActive();
	EXIT_IF(!complete);
	const uint64_t previous = std::exchange(m_last_flip_tick, CurrentTick());
	if (!EarlyFlipEnabled()) {
		QueuePriorityOperation(std::move(complete));
		return;
	}
	std::call_once(m_flip_thread_once, [this] {
		m_flip_thread =
		    std::jthread([this](std::stop_token stop) { FlipCompletionThread(stop); });
	});
	Common::ScopedLock lock(m_flip_mutex);
	m_flip_completions.push_back({std::move(complete), m_last_flip_tick, previous});
	m_flips_pending.store(true, std::memory_order_release);
}

void CommandScheduler::FlipCompletionThread(std::stop_token stop) {
	for (;;) {
		FlipCompletion flip;
		{
			Common::UniqueLock lock(m_flip_mutex);
			if (!m_flip_changed.wait(lock, stop, [this]() KYTY_REQUIRES(m_flip_mutex) {
				    return !m_flip_completions.empty() &&
				           SubmittedTick() >= m_flip_completions.front().submitted_tick;
			    })) {
				return;
			}
			flip = std::move(m_flip_completions.front());
			m_flip_completions.pop_front();
			m_flips_pending.store(!m_flip_completions.empty(), std::memory_order_release);
			m_flip_running = true;
		}
		if (flip.previous_tick != 0) {
			m_master.Wait(flip.previous_tick);
			WaitPriorityOperations(flip.previous_tick);
		}
		RunOperation(std::move(flip.complete));
		{
			Common::ScopedLock lock(m_flip_mutex);
			m_flip_running = false;
		}
		m_flip_changed.notify_all();
	}
}

void CommandScheduler::DrainFlipCompletions() {
	Common::UniqueLock lock(m_flip_mutex);
	m_flip_changed.wait(lock, [this]() KYTY_REQUIRES(m_flip_mutex) { return m_flip_completions.empty() && !m_flip_running; });
}
#endif

Common::UniqueFunction<void>
CommandScheduler::TrackGuestOperation(Common::UniqueFunction<void>&& operation, uint64_t address,
                                      uint64_t size) {
	EXIT_IF(!operation);
	uint64_t id = 0;
	{
		Common::ScopedLock lock(m_guest_writes_mutex);
		id = ++m_next_guest_write;
		m_guest_writes.push_back(
		    {CurrentTick(), id, address, size > UINT64_MAX - address ? UINT64_MAX : address + size});
	}
	return [this, id, operation = std::move(operation)]() mutable {
		operation();
		Common::ScopedLock lock(m_guest_writes_mutex);
		const auto it = std::find_if(m_guest_writes.begin(), m_guest_writes.end(),
		                             [id](const GuestWrite& write) { return write.id == id; });
		EXIT_IF(it == m_guest_writes.end());
		m_guest_writes.erase(it);
	};
}

Common::UniqueFunction<void>
CommandScheduler::TrackGuestOperation(Common::UniqueFunction<void>&&                     operation,
                                      const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	EXIT_IF(!operation);
	uint64_t id = 0;
	{
		Common::ScopedLock lock(m_guest_writes_mutex);
		id = ++m_next_guest_write;
		for (const auto& [address, size]: ranges) {
			m_guest_writes.push_back(
			    {CurrentTick(), id, address, size > UINT64_MAX - address ? UINT64_MAX : address + size});
		}
	}
	return [this, id, operation = std::move(operation)]() mutable {
		operation();
		Common::ScopedLock lock(m_guest_writes_mutex);
		std::erase_if(m_guest_writes, [id](const GuestWrite& write) { return write.id == id; });
	};
}

#if defined(ANYPS5_KYTY)
uint64_t CommandScheduler::NewestPendingGuestOperation(uint64_t address, uint64_t size,
                                                        uint64_t after) {
	const auto      end = size > UINT64_MAX - address ? UINT64_MAX : address + size;
	Common::ScopedLock lock(m_guest_writes_mutex);
	for (auto it = m_guest_writes.rbegin(); it != m_guest_writes.rend() && it->id > after; ++it) {
		if (it->begin < end && address < it->end) {
			return it->id;
		}
	}
	return 0;
}
#endif

uint64_t CommandScheduler::PendingGuestTick(uint64_t address, uint64_t size) {
	const auto end = size > UINT64_MAX - address ? UINT64_MAX : address + size;
	Common::ScopedLock lock(m_guest_writes_mutex);
	for (auto it = m_guest_writes.rbegin(); it != m_guest_writes.rend(); ++it) {
		if (it->begin < end && address < it->end) return it->tick;
	}
	return 0;
}

bool CommandScheduler::HasPendingGuestOperations(uint64_t address, uint64_t size) {
	return PendingGuestTick(address, size) != 0;
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	Common::UniqueLock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		if (m_pending_operations.size() == 1) {
			m_pending_front_tick.store(m_pending_operations.front().tick,
			                           std::memory_order_release);
		}
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock.Native(),
	                           [this]() KYTY_REQUIRES(m_operation_mutex) { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::QueuePriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	Common::UniqueLock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock.Native(),
	                           [this]() KYTY_REQUIRES(m_operation_mutex) { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	Common::ScopedLock lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active_tick != 0;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			Common::UniqueLock lock(m_operation_mutex);
			m_operation_available.wait(lock.Native(), [this, &stop]() KYTY_REQUIRES(m_operation_mutex) {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			Common::ScopedLock lock(m_operation_mutex);
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	Common::UniqueLock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock.Native(), [this]() KYTY_REQUIRES(m_operation_mutex) { return m_priority_operations.empty() && m_priority_active_tick == 0; });
}


void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
#if defined(ANYPS5_KYTY)
	if (tick >= CurrentTick() && Active() && !m_command.IsInvalid() && GuestGpu::IsGpuThread() &&
	    HasPendingPriorityOperations()) {
		m_command.EndRendering();
		Flush();
	}
#endif
	Common::UniqueLock lock(m_operation_mutex);
	m_operation_available.wait(lock.Native(), [this, tick]() KYTY_REQUIRES(m_operation_mutex) {
		const bool active_before_or_at =
		    m_priority_active_tick != 0 && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	if (m_deferred) {
		m_command.m_open = true;
	} else {
		m_command.m_buffer = m_command_pool.Commit();
	}
	m_command.Begin();
	GpuBreadcrumbs::MarkSubmission(m_graphics, m_command.Handle(), m_marker_slot, m_master.CurrentTick());
	m_dynamic_state.Invalidate();
	if (GpuBreadcrumbs::PerOperation()) GpuBreadcrumbs::AfterBegin(*this);
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	if (m_submit_hook != nullptr) {
		m_submit_hook(m_submit_hook_data, m_command, submit);
	}

	if (GpuBreadcrumbs::PerOperation()) GpuBreadcrumbs::BeforeSubmit(*this);
	m_command.End();
	const auto     tick = m_master.NextTick();
	m_master.NoteIssued(tick);
	if (m_deferred) {
		auto* recorded = m_stream.RecordSpecial<RecordedSubmit>(CommandStream::RecordKind::Submit);
		recorded->info = submit;
		recorded->tick = tick;
		recorded->host_copies = m_host_copies != nullptr ? m_host_copies->Flush() : 0;
		m_stream.Publish();
		m_command.m_open = false;
		return tick;
	}

	QueueSubmit(m_command.m_buffer, submit, tick);
	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick) {
	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);
	submit.AddSignal(m_master.Handle(), tick);

	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = submit.wait_stages.data();
	submit_info.commandBufferCount   = buffer != nullptr ? 1 : 0;
	submit_info.pCommandBuffers      = buffer != nullptr ? &buffer : nullptr;
	submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

	vk::Result result;
	EnterSubmit(SubmitStage::QueueLock, tick);
	{
		Common::LockGuard lock(graphics.queue_mutex);
		EnterSubmit(SubmitStage::Submit, tick);
		m_master.NoteSubmitting(tick);
		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	EnterSubmit(SubmitStage::None, tick);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		GpuBreadcrumbs::ReportDeviceLost(graphics, result, "vkQueueSubmit", m_snapshot_name.c_str(), tick);
		EXIT("vkQueueSubmit failed: %s, tick=%" PRIu64 "\n", vk::to_string(result).c_str(), tick);
	}
	m_master.NoteSubmitted(tick, std::span(submit.wait_semaphores.data(), submit.num_wait_semaphores),
	                       std::span(submit.wait_ticks.data(), submit.num_wait_semaphores));
	m_submitted_tick.store(tick, std::memory_order_release);
#if defined(ANYPS5_KYTY)
	if (m_flips_pending.load(std::memory_order_acquire)) {
		{
			Common::ScopedLock lock(m_flip_mutex);
		}
		m_flip_changed.notify_all();
	}
#endif
}

void CommandScheduler::RecordingThread() {
	KYTY_PROFILER_THREAD("Thread_GpuRecord");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Recording);
	vk::CommandBuffer buffer       = nullptr;
	size_t            buffer_index = 0;
	const auto        acquire      = [this, &buffer, &buffer_index] {
		if (buffer == nullptr) {
			std::tie(buffer, buffer_index) = m_command_pool.CommitDeferred();
			vk::CommandBufferBeginInfo begin_info {};
			begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
			EXIT_NOT_IMPLEMENTED(buffer.begin(&begin_info) != vk::Result::eSuccess);
		}
		return buffer;
	};
	for (;;) {
		EnterRecord(RecordStage::Commands, m_submitted_tick.load(std::memory_order_relaxed));
		auto chunk = m_chunk_queue.Pop();
		if (chunk == nullptr) {
			break;
		}
		EnterRecord(RecordStage::Replaying, m_submitted_tick.load(std::memory_order_relaxed) + 1);
		CommandStream::Replay(*chunk, acquire, [&](const CommandStream::Entry& entry) {
			if (entry.kind == CommandStream::RecordKind::TrimPool) {
				m_command_pool.TrimCompleted();
				return;
			}
			EXIT_IF(entry.kind != CommandStream::RecordKind::Submit);
			SubmitRecorded(buffer, buffer_index,
			               *static_cast<const RecordedSubmit*>(entry.Payload()));
		});
		m_chunk_queue.Recycle(std::move(chunk));
	}
	EXIT_IF(buffer != nullptr);
}

void CommandScheduler::SubmitRecorded(vk::CommandBuffer& buffer, size_t buffer_index,
                                      const RecordedSubmit& recorded) {
	if (buffer != nullptr) {
		EXIT_NOT_IMPLEMENTED(buffer.end() != vk::Result::eSuccess);
	}
	if (m_host_copies != nullptr) {
		EnterRecord(RecordStage::HostCopies, recorded.tick);
		m_host_copies->WaitFor(recorded.host_copies);
	}
	EnterRecord(RecordStage::Submitting, recorded.tick);
	auto submit = recorded.info;
	QueueSubmit(buffer, submit, recorded.tick);
	EnterRecord(RecordStage::Replaying, recorded.tick + 1);
	if (buffer != nullptr) {
		m_command_pool.Retire(buffer_index, recorded.tick);
		buffer = nullptr;
	}
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

}

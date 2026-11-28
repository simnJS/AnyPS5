#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"

#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#endif

namespace Libs::Graphics {

#if defined(ANYPS5_KYTY)
namespace {

uint64_t SteadyNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

double Ago(uint64_t now, uint64_t then) {
	return then == 0 || then > now ? 0.0 : static_cast<double>(now - then) / 1e9;
}

}

void MasterSemaphore::NoteIssued(uint64_t tick) noexcept {
	auto& slot = m_submissions[tick % m_submissions.size()];
	slot.waits.store(0, std::memory_order_relaxed);
	slot.submitted_ns.store(0, std::memory_order_relaxed);
	slot.issued_ns.store(SteadyNs(), std::memory_order_relaxed);
	slot.tick.store(tick, std::memory_order_release);
}

void MasterSemaphore::NoteSubmitted(uint64_t tick, std::span<const vk::Semaphore> waits,
                                    std::span<const uint64_t> values) noexcept {
	auto& slot = m_submissions[tick % m_submissions.size()];
	if (slot.tick.load(std::memory_order_acquire) != tick) {
		slot.issued_ns.store(0, std::memory_order_relaxed);
		slot.tick.store(tick, std::memory_order_release);
	}
	const auto count = std::min(waits.size(), slot.semaphores.size());
	for (size_t i = 0; i < count; i++) {
		slot.semaphores[i].store(reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(waits[i])), std::memory_order_relaxed);
		slot.values[i].store(i < values.size() ? values[i] : 0, std::memory_order_relaxed);
	}
	slot.waits.store(static_cast<uint32_t>(count), std::memory_order_relaxed);
	slot.submitted_ns.store(SteadyNs(), std::memory_order_release);
	m_submitting_tick.store(0, std::memory_order_release);
	auto known = m_submitted.load(std::memory_order_relaxed);
	while (known < tick && !m_submitted.compare_exchange_weak(known, tick, std::memory_order_release, std::memory_order_relaxed)) {
	}
}

std::string MasterSemaphore::PendingWait(uint64_t tick) {
	const auto& slot = m_submissions[tick % m_submissions.size()];
	if (slot.tick.load(std::memory_order_acquire) != tick || slot.submitted_ns.load(std::memory_order_acquire) == 0) {
		return {};
	}
	std::string pending;
	std::string binary;
	const auto  count = std::min<size_t>(slot.waits.load(std::memory_order_relaxed), slot.semaphores.size());
	for (size_t i = 0; i < count; i++) {
		const auto semaphore = vk::Semaphore(reinterpret_cast<VkSemaphore>(slot.semaphores[i].load(std::memory_order_relaxed)));
		const auto value     = slot.values[i].load(std::memory_order_relaxed);
		if (!StallSnapshot::Timeline(semaphore)) {
			binary += (binary.empty() ? "" : ", ") + StallSnapshot::Describe(semaphore);
			continue;
		}
		uint64_t counter = 0;
		if (m_graphics.device.getSemaphoreCounterValue(semaphore, &counter) != vk::Result::eSuccess || counter >= value) continue;
		pending += fmt::format("{}{} to reach {}, which is at {}", pending.empty() ? "" : " and for ",
		                       StallSnapshot::Describe(semaphore), value, counter);
	}
	if (!pending.empty()) return "waits for " + pending;
	if (!binary.empty()) return "waits for nothing it can read; it also waits for " + binary + ", whose state cannot be read";
	return {};
}

void MasterSemaphore::WriteSubmissions(std::FILE* file, size_t count) {
	const auto now     = SteadyNs();
	uint64_t   reached = 0;
	(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &reached);
	const auto next = CurrentTick();
	std::fprintf(file, "ticks issued up to %" PRIu64 ", submitted to Vulkan up to %" PRIu64 ", the GPU is at %" PRIu64 "\n",
	             next - 1, SubmittedTick(), reached);
	for (uint64_t tick = next - 1; tick > 0 && next - tick <= count; tick--) {
		const auto& slot = m_submissions[tick % m_submissions.size()];
		if (slot.tick.load(std::memory_order_acquire) != tick) continue;
		const auto submitted = slot.submitted_ns.load(std::memory_order_acquire);
		std::fprintf(file, "tick %" PRIu64 "%s: issued %.3f s ago, ", tick, tick <= reached ? " finished" : " unfinished",
		             Ago(now, slot.issued_ns.load(std::memory_order_relaxed)));
		if (submitted == 0) {
			std::fprintf(file, "not submitted to Vulkan\n");
			continue;
		}
		std::fprintf(file, "submitted %.3f s ago", Ago(now, submitted));
		const auto waits = std::min<size_t>(slot.waits.load(std::memory_order_relaxed), slot.semaphores.size());
		if (waits == 0) std::fprintf(file, ", waits for no semaphore");
		for (size_t i = 0; i < waits; i++) {
			const auto semaphore = vk::Semaphore(reinterpret_cast<VkSemaphore>(slot.semaphores[i].load(std::memory_order_relaxed)));
			const auto value     = slot.values[i].load(std::memory_order_relaxed);
			uint64_t   counter   = 0;
			if (StallSnapshot::Timeline(semaphore) &&
			    m_graphics.device.getSemaphoreCounterValue(semaphore, &counter) == vk::Result::eSuccess) {
				std::fprintf(file, ", waits for %s to reach %" PRIu64 " (now %" PRIu64 ")", StallSnapshot::Describe(semaphore).c_str(),
				             value, counter);
			} else {
				std::fprintf(file, ", waits for %s (binary)", StallSnapshot::Describe(semaphore).c_str());
			}
		}
		std::fprintf(file, "\n");
	}
	std::fprintf(file, "compute shaders noted by tick, last first:");
	const auto work_next = m_work_next.load(std::memory_order_relaxed);
	uint32_t   shown     = 0;
	for (uint32_t i = 1; i <= m_work.size() && shown < 64; i++) {
		const auto& work = m_work[(work_next - i) % m_work.size()];
		const auto  tick = work.tick.load(std::memory_order_relaxed);
		if (tick == 0 || tick + 2 <= reached) continue;
		std::fprintf(file, " %" PRIu64 ":0x%016" PRIx64, tick, work.shader.load(std::memory_order_relaxed));
		shown++;
	}
	std::fprintf(file, shown == 0 ? " none\n" : "\n");
}

void MasterSemaphore::NoteSubmitting(uint64_t tick) noexcept {
	m_submitting_since.store(SteadyNs(), std::memory_order_relaxed);
	m_submitting_tick.store(tick, std::memory_order_release);
}

void MasterSemaphore::ReportStall(uint64_t tick, StallSnapshot::StallWatch& watch) {
	const auto periods = ++watch.periods;
	uint64_t   reached = 0;
	if (m_graphics.device.getSemaphoreCounterValue(m_semaphore, &reached) != vk::Result::eSuccess || reached >= tick) return;
	const auto submitted       = SubmittedTick();
	const auto next            = CurrentTick();
	const auto submitting_tick = m_submitting_tick.load(std::memory_order_acquire);
	const auto submitting_for  = Ago(SteadyNs(), m_submitting_since.load(std::memory_order_relaxed));
	const auto holder          = StallSnapshot::Classify(tick, next, submitted, submitting_tick, submitting_for);
	const auto compiling       = StallSnapshot::LongestCompileWait();
	const auto waited          = uint64_t {periods} * 10u;
	std::string verdict;
	bool        snapshot = false;
	if (!StallSnapshot::GpuSide(holder)) {
		const bool stuck = StallSnapshot::Stuck(holder, periods, compiling);
		if (periods != 1 && !(stuck && !m_reported_stuck.test_and_set())) {
			return;
		}
		const auto compile = compiling.count == 0
		                         ? std::string()
		                         : fmt::format(" while {} pipeline compile wait{} run{}, the longest for {:.1f} s ({} 0x{:016x})",
		                                       compiling.count, compiling.count == 1 ? "" : "s", compiling.count == 1 ? "s" : "",
		                                       compiling.seconds, compiling.what, compiling.hash);
		verdict  = stuck ? fmt::format("[kyty] tick {} has been awaited for {} s and is not submitted yet (next tick {}, submitted up "
		                               "to {}, the GPU is at {}), and no pipeline compile runs: the CPU side is stuck",
		                               tick, waited, next, submitted, reached)
		                 : fmt::format("[kyty] tick {} has been awaited for {} s and is not submitted yet (next tick {}, submitted up "
		                               "to {}, the GPU is at {}): the CPU side holds it{}",
		                               tick, waited, next, submitted, reached, compile);
		snapshot = stuck || m_cpu_snapshots.fetch_add(1, std::memory_order_relaxed) < 3;
	} else {
		double submitted_for = 0.0;
		if (holder == StallSnapshot::Holder::Gpu) {
			const auto& slot = m_submissions[tick % m_submissions.size()];
			const auto submitted_ns = slot.tick.load(std::memory_order_acquire) == tick
			                              ? slot.submitted_ns.load(std::memory_order_acquire)
			                              : 0;
			const auto seconds = watch.GpuStallSeconds(SteadyNs(), submitted_ns);
			if (!seconds) return;
			submitted_for = *seconds;
		}
		if (m_reported_gpu.test_and_set()) {
			return;
		}
		std::string began;
		if (const auto started = GpuBreadcrumbs::StartedTick(m_marker_slot, reached + 1); started.has_value() && submitted > reached) {
			began = *started > reached ? fmt::format(" (the GPU began tick {}: it runs or waits inside it)", reached + 1)
			                           : fmt::format(" (the GPU has not begun tick {}: it waits before it)", reached + 1);
		}
		const auto wait = holder == StallSnapshot::Holder::Gpu ? PendingWait(reached + 1) : std::string();
		if (holder == StallSnapshot::Holder::Driver) {
			verdict = fmt::format("[kyty] the GPU has not reached tick {} in {} s (it is at {}, {} ticks issued after it): tick {} has "
			                      "been in vkQueueSubmit for {:.1f} s, the driver holds it",
			                      tick, waited, reached, next - 1 - reached, submitting_tick, submitting_for);
		} else if (wait.starts_with("waits for ") && !wait.starts_with("waits for nothing")) {
			verdict = fmt::format("[kyty] the GPU has not reached tick {} (submitted at least {:.1f} s ago, GPU at {}, {} ticks issued "
			                      "after it): the submission of tick {} {}, the GPU waits for that signal{}",
			                      tick, submitted_for, reached, next - 1 - reached, reached + 1, wait, began);
		} else {
			verdict = fmt::format("[kyty] the GPU has not reached tick {} (submitted at least {:.1f} s ago, GPU at {}, {} ticks issued "
			                      "after it): the GPU stopped, look for an NVIDIA Xid in dmesg{}{}{}",
			                      tick, submitted_for, reached, next - 1 - reached, began, wait.empty() ? "" : "; its submission ", wait);
		}
		snapshot = true;
	}
	if (const auto path = snapshot ? StallSnapshot::Write(m_graphics, tick, reached, verdict) : std::string(); !path.empty()) {
		std::fprintf(stderr, "[kyty-hang-snapshot] written to %s\n", path.c_str());
	}
	if (StallSnapshot::GpuSide(holder)) {
		uint64_t current = 0;
		if (m_graphics.device.getSemaphoreCounterValue(m_semaphore, &current) != vk::Result::eSuccess || current != reached ||
		    (holder == StallSnapshot::Holder::Driver && m_submitting_tick.load(std::memory_order_acquire) != submitting_tick)) {
			m_reported_gpu.clear();
			return;
		}
	}
	std::fprintf(stderr, "%s\n", verdict.c_str());
	if (!StallSnapshot::GpuSide(holder)) {
		return;
	}
	std::array<uint64_t, 24> shaders {};
	size_t                   count = 0;
	const auto               work_next = m_work_next.load(std::memory_order_relaxed);
	for (uint32_t i = 1; i <= m_work.size() && count < shaders.size(); i++) {
		const auto& work = m_work[(work_next - i) % m_work.size()];
		if (work.tick.load(std::memory_order_relaxed) != reached + 1) {
			continue;
		}
		const auto shader = work.shader.load(std::memory_order_relaxed);
		if (std::find(shaders.begin(), shaders.begin() + count, shader) == shaders.begin() + count) {
			shaders[count++] = shader;
		}
	}
	std::fprintf(stderr, "[kyty] compute shaders of the unfinished tick %" PRIu64 ", last first:", reached + 1);
	for (size_t i = 0; i < count; i++) {
		std::fprintf(stderr, " 0x%016" PRIx64, shaders[i]);
	}
	std::fprintf(stderr, count == 0 ? " none recorded\n" : "\n");
}
#endif

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	m_graphics.device.destroySemaphore(m_semaphore, nullptr);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
#if defined(ANYPS5_KYTY)
	constexpr uint64_t StallNs = 10'000'000'000ull;
	auto result = m_graphics.device.waitSemaphores(&wait_info, StallNs);
	StallSnapshot::StallWatch watch;
	while (result == vk::Result::eTimeout) {
		ReportStall(tick, watch);
		result = m_graphics.device.waitSemaphores(&wait_info, StallNs);
	}
#else
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
#endif
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

}

#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#if defined(ANYPS5_KYTY)
#include <array>
#include <cstdio>
#include <span>
#include <string>
#endif

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }

	void Refresh();
	void Wait(uint64_t tick);
#if defined(ANYPS5_KYTY)
	void ReportStall(uint64_t tick);
	void NoteIssued(uint64_t tick) noexcept;
	void NoteSubmitted(uint64_t tick, std::span<const vk::Semaphore> waits, std::span<const uint64_t> values) noexcept;
	[[nodiscard]] uint64_t SubmittedTick() const noexcept { return m_submitted.load(std::memory_order_acquire); }
	void                   WriteSubmissions(std::FILE* file, size_t count);
	[[nodiscard]] std::string PendingWait(uint64_t tick);
	void SetMarkerSlot(uint32_t slot) noexcept { m_marker_slot = slot; }
	void NoteWork(uint64_t tick, uint64_t shader) noexcept {
		const auto next = m_work_next.load(std::memory_order_relaxed);
		const auto at   = next % m_work.size();
		m_work_next.store(next + 1, std::memory_order_relaxed);
		m_work[at].tick.store(tick, std::memory_order_relaxed);
		m_work[at].shader.store(shader, std::memory_order_relaxed);
	}
#endif

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
#if defined(ANYPS5_KYTY)
	struct Work {
		std::atomic<uint64_t> tick {0};
		std::atomic<uint64_t> shader {0};
	};
	std::array<Work, 512> m_work {};
	std::atomic<uint32_t> m_work_next {0};
	struct Submission {
		std::atomic<uint64_t>                 tick {0};
		std::atomic<uint64_t>                 issued_ns {0};
		std::atomic<uint64_t>                 submitted_ns {0};
		std::atomic<uint32_t>                 waits {0};
		std::array<std::atomic<uint64_t>, 4> semaphores {};
		std::array<std::atomic<uint64_t>, 4> values {};
	};
	std::array<Submission, 128> m_submissions {};
	std::atomic<uint64_t>       m_submitted {0};
	uint32_t                    m_marker_slot      = UINT32_MAX;
	std::atomic_flag            m_reported_gpu     = ATOMIC_FLAG_INIT;
	std::atomic_flag            m_reported_pending = ATOMIC_FLAG_INIT;
#endif
};

}

#endif

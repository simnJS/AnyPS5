#include "TestVulkanDevice.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace Libs::Graphics;

namespace {

void Require(bool passed, const char* description) {
	if (!passed) {
		throw std::runtime_error(description);
	}
}

class TimelineGate {
public:
	explicit TimelineGate(vk::Device device) : m_device(device) {
		const vk::SemaphoreTypeCreateInfo type {.semaphoreType = vk::SemaphoreType::eTimeline};
		const vk::SemaphoreCreateInfo     info {.pNext = &type};
		Require(m_device.createSemaphore(&info, nullptr, &m_semaphore) == vk::Result::eSuccess,
		        "could not create timeline semaphore");
	}

	~TimelineGate() {
		if (!m_released) {
			const vk::SemaphoreSignalInfo info {.semaphore = m_semaphore, .value = 1};
			(void)m_device.signalSemaphore(info);
		}
		(void)m_device.waitIdle();
		m_device.destroySemaphore(m_semaphore);
	}

	TimelineGate(const TimelineGate&)            = delete;
	TimelineGate& operator=(const TimelineGate&) = delete;

	void Signal(uint64_t value) {
		const vk::SemaphoreSignalInfo info {.semaphore = m_semaphore, .value = value};
		Require(m_device.signalSemaphore(info) == vk::Result::eSuccess, "could not signal timeline semaphore");
		m_released = true;
	}

	[[nodiscard]] vk::Semaphore Handle() const { return m_semaphore; }

private:
	vk::Device    m_device;
	vk::Semaphore m_semaphore;
	bool          m_released = false;
};

const char* ProbeQueueSubmitBatching(GraphicContext& graphics) {
	RenderContext    renderer(graphics);
	CommandProcessor processor(renderer, 0);
	processor.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(processor.GetCtx(), processor.GetUcfg(), processor.GetShCtx());

	constexpr uint32_t BatchCount = 16;
	Buffer             output(graphics, scheduler, MemoryUsage::Download, 0,
	                          vk::BufferUsageFlagBits::eTransferDst, BatchCount * sizeof(uint32_t));

	scheduler.Flush();
	scheduler.Finish();
	Common::DebugCounters::SetEnabled(true);

	const auto submits_before = Common::DebugCounters::Get(Common::DebugCounters::Counter::QueueSubmits);
	std::vector<uint64_t> ticks;
	ticks.reserve(BatchCount);
	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		scheduler.Flush();
		for (uint32_t i = 0; i < BatchCount; ++i) {
			output.Fill(i * sizeof(uint32_t), sizeof(uint32_t), 0x1000u + i);
			ticks.push_back(scheduler.CurrentTick());
			scheduler.Flush();
		}
	}

	scheduler.Wait(ticks.back());
	const auto submits_after = Common::DebugCounters::Get(Common::DebugCounters::Counter::QueueSubmits);
	Require(submits_after > submits_before, "at least one submission must occur");
	Require(submits_after - submits_before < BatchCount,
	        "back-to-back unforced submits must coalesce into fewer vkQueueSubmit calls");

	auto& master = scheduler.GetMasterSemaphore();
	master.Refresh();
	for (const auto tick: ticks) {
		Require(master.IsFree(tick), "every coalesced intermediate tick must be marked complete");
	}

	output.Invalidate(0, BatchCount * sizeof(uint32_t));
	std::array<uint32_t, BatchCount> words {};
	std::memcpy(words.data(), output.Mapped().data(), sizeof(words));
	for (uint32_t i = 0; i < BatchCount; ++i) {
		Require(words[i] == 0x1000u + i, "coalesced buffer fill wrote wrong value");
	}

	TimelineGate boundary_gate(graphics.device);
	output.Fill(0, sizeof(uint32_t), 0xdeadbeefu);
	const auto pre_wait_tick = scheduler.CurrentTick();
	scheduler.Flush();

	output.Fill(sizeof(uint32_t), sizeof(uint32_t), 0xfeedfaceu);
	SubmitInfo wait_boundary;
	wait_boundary.AddWait(boundary_gate.Handle(), 1);
	const auto wait_tick = scheduler.CurrentTick();
	scheduler.Flush(wait_boundary);

	const auto semaphore = master.Handle();
	const vk::SemaphoreWaitInfo wait_pre {.semaphoreCount = 1, .pSemaphores = &semaphore, .pValues = &pre_wait_tick};
	constexpr uint64_t FiveSecondsNs = 5'000'000'000ull;
	Require(graphics.device.waitSemaphores(wait_pre, FiveSecondsNs) == vk::Result::eSuccess,
	        "submit before a wait boundary must flush before the wait boundary blocks");
	master.Refresh();
	Require(!master.IsFree(wait_tick), "wait boundary must not complete before its semaphore is signaled");

	output.Invalidate(0, 2 * sizeof(uint32_t));
	std::array<uint32_t, 2> boundary_words {};
	std::memcpy(boundary_words.data(), output.Mapped().data(), sizeof(boundary_words));
	Require(boundary_words[0] == 0xdeadbeefu, "pre-wait fill must land before wait boundary unblocks");
	Require(boundary_words[1] == 0x1001u,
	        "wait-boundary commands must not execute while the wait semaphore is unsignaled");

	boundary_gate.Signal(1);
	scheduler.Wait(wait_tick);
	Require(master.IsFree(wait_tick), "wait boundary must complete after its semaphore is signaled");
	output.Invalidate(sizeof(uint32_t), sizeof(uint32_t));
	std::memcpy(boundary_words.data(), output.Mapped().data(), sizeof(boundary_words));
	Require(boundary_words[1] == 0xfeedfaceu,
	        "wait-boundary fill must land after the wait semaphore is signaled");

	renderer.ShutdownGpu();
	return nullptr;
}

}

extern "C" int RunKytyQueueSubmitBatchingTest() {
	try {
		Common::InitializeThreads();
		Config::Initialize();
		Config::Load(Config::ConfigOptions {});
		KytyShimTests::TestVulkanDevice gpu;
		if (const auto* unavailable = gpu.Initialize()) {
			const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
			std::fprintf(stderr, "kyty_shim_queue_submit_batching: %s, no usable Vulkan device (%s)\n",
			             required ? "FAILED" : "skipped", unavailable);
			return required ? 1 : 77;
		}
		Require(gpu.graphics.CreateAllocator(), "could not create the Vulkan allocator");
		if (const auto* failure = ProbeQueueSubmitBatching(gpu.graphics)) {
			std::fprintf(stderr, "kyty_shim_queue_submit_batching: FAILED: %s\n", failure);
			return 1;
		}
		std::printf("kyty_shim_queue_submit_batching: ok\n");
		return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "kyty_shim_queue_submit_batching: FAILED: %s\n", error.what());
		return 1;
	}
}

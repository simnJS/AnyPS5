#include "graphics/host_gpu/graphicContext.h"

#include "common/hostStats.h"
#include "common/logging/log.h"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <thread>

namespace Libs::Graphics {

namespace {

using Clock = std::chrono::steady_clock;

class MemoryMinutes {
public:
	static MemoryMinutes& Instance() {
		static auto* minutes = [] {
			auto* instance = new MemoryMinutes;
			std::atexit([] { Instance().Stop(nullptr); });
			return instance;
		}();
		return *minutes;
	}

	void Start(GraphicContext& graphics) {
		std::lock_guard lifecycle(m_lifecycle);
		Finish();
		m_owner = &graphics;
		m_started = Clock::now();
		m_thread = std::jthread([this](std::stop_token stop) {
			constexpr auto interval = std::chrono::seconds(2);
			auto next = m_started + interval;
			auto minute = m_started + std::chrono::minutes(1);
			std::unique_lock lock(m_wait_mutex);
			while (!stop.stop_requested()) {
				m_wake.wait_until(lock, stop, next, [] { return false; });
				if (stop.stop_requested()) return;
				if (Clock::now() >= minute) {
					m_owner->LogMemoryMinute("minute", Elapsed());
					do {
						minute += std::chrono::minutes(1);
					} while (minute <= Clock::now());
				} else {
					m_owner->LogMemoryMinute("pressure", Elapsed(), true);
				}
				do {
					next += interval;
				} while (next <= Clock::now());
			}
		});
	}

	void Stop(const GraphicContext* owner) {
		std::lock_guard lifecycle(m_lifecycle);
		if (!owner || owner == m_owner) Finish();
	}

private:
	void Finish() {
		if (!m_owner) return;
		m_thread.request_stop();
		m_thread.join();
		m_owner->LogMemoryMinute("exit", Elapsed());
		m_owner = nullptr;
	}

	uint64_t Elapsed() const {
		return std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - m_started).count();
	}

	GraphicContext* m_owner = nullptr;
	Clock::time_point m_started;
	std::mutex m_lifecycle;
	std::mutex m_wait_mutex;
	std::condition_variable_any m_wake;
	std::jthread m_thread;
};

std::string Value(std::optional<uint64_t> bytes) {
	return bytes ? std::to_string(*bytes) : "na";
}

}

void GraphicContext::StartMemoryMinutes() {
	MemoryMinutes::Instance().Start(*this);
}

void GraphicContext::StopMemoryMinutes() {
	MemoryMinutes::Instance().Stop(this);
}

void GraphicContext::LogMemoryMinute(const char* reason, uint64_t seconds, bool pressure_only) const {
	using namespace MemoryTelemetry;
	const auto start = Clock::now();
	const auto host = pressure_only ? Common::ProcessMemory {} : Common::ReadProcessMemory();
	vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget {};
	if (memory_budget_ext_enabled) {
		vk::PhysicalDeviceMemoryProperties2 properties {.pNext = &budget};
		physical_device.getMemoryProperties2(&properties);
	}
	uint64_t local_usage = 0, local_budget = 0;
	for (uint32_t i = 0; i < physical_device_memory_properties.memoryHeapCount; ++i) {
		if (physical_device_memory_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
			local_usage += budget.heapUsage[i];
			local_budget += budget.heapBudget[i];
		}
	}
	if (pressure_only && (local_budget == 0 || local_usage < HeapImageThreshold(local_budget))) return;
	auto line = fmt::format("[kyty-mem] reason={} seconds={} rss={} private={} malloc={} malloc_free={} cpu_copies={} pipeline_ram={}",
	                        reason, seconds, Value(host.rss), Value(host.private_bytes), Value(host.malloc_bytes), Value(host.malloc_free),
	                        cpu_copies.load(std::memory_order_relaxed), pipeline_ram.load(std::memory_order_relaxed));
	std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> primary {}, spill {};
	if (allocator) vmaGetHeapBudgets(allocator, primary.data());
	if (spill_allocator) vmaGetHeapBudgets(spill_allocator, spill.data());
	uint64_t local_blocks = 0;
	for (uint32_t i = 0; i < physical_device_memory_properties.memoryHeapCount; ++i) {
		const auto& a = primary[i].statistics;
		const auto& b = spill[i].statistics;
		const auto blocks = a.blockBytes + b.blockBytes;
		const auto allocated = a.allocationBytes + b.allocationBytes;
		const bool local = bool(physical_device_memory_properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		line += fmt::format(" heap{}={},{},{},{},{},{},{}", i, local ? "local" : "system",
		                    memory_budget_ext_enabled ? std::to_string(budget.heapUsage[i]) : "na",
		                    memory_budget_ext_enabled ? std::to_string(budget.heapBudget[i]) : "na",
		                    blocks, allocated, blocks - allocated, uint64_t(a.allocationCount) + b.allocationCount);
		if (local) local_blocks += blocks;
	}
	line += fmt::format(" local_usage={} local_budget={} local_vma_blocks={} outside_vma={}",
	                    memory_budget_ext_enabled ? std::to_string(local_usage) : "na",
	                    memory_budget_ext_enabled ? std::to_string(local_budget) : "na", local_blocks,
	                    memory_budget_ext_enabled ? std::to_string(int64_t(local_usage) - int64_t(local_blocks)) : "na");
	for (size_t i = 0; i < Names.size(); ++i) {
		const auto& local = totals[i][1];
		const auto& system = totals[i][0];
		line += fmt::format(" {}={},{},{},{}", Names[i], local.bytes.load(std::memory_order_relaxed),
		                    local.count.load(std::memory_order_relaxed), system.bytes.load(std::memory_order_relaxed),
		                    system.count.load(std::memory_order_relaxed));
	}
	const auto driver = [&](DriverMemoryReport::Category kind) {
		return driver_memory_report ? Value(driver_memory_report->Bytes(kind)) : "na";
	};
	line += fmt::format(" parked_local={} mirror_bound={} imported_host={} driver_pipeline={}"
	                    " driver_descriptor_pool={} driver_command_pool={}",
	                    parked_local.load(std::memory_order_relaxed), resident.load(std::memory_order_relaxed),
	                    imported.load(std::memory_order_relaxed), driver(DriverMemoryReport::Pipeline), driver(DriverMemoryReport::DescriptorPool),
	                    driver(DriverMemoryReport::CommandPool));
	line += fmt::format(" pipelines={} pipeline_caches={} descriptor_pools={} command_pools={}",
	                    live_objects<vk::Pipeline>.load(std::memory_order_relaxed),
	                    live_objects<vk::PipelineCache>.load(std::memory_order_relaxed),
	                    live_objects<vk::DescriptorPool>.load(std::memory_order_relaxed),
	                    live_objects<vk::CommandPool>.load(std::memory_order_relaxed));
	line += fmt::format(" sample_us={}\n", std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count());
	::Log::WriteToConsoleAndLog(line);
}

}

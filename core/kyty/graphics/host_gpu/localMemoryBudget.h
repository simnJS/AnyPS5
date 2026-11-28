#pragma once

#include "graphics/host_gpu/vulkanCommon.h"

#include <chrono>
#include <functional>
#include <mutex>
#include <optional>

namespace Libs::Graphics {

struct BoardMemory {
	uint64_t total;
	uint64_t used;
};

struct LocalMemorySample {
	uint64_t usage = 0;
	uint64_t budget = 0;
	uint64_t target = 0;
	std::optional<BoardMemory> board;
};

class LocalMemoryBudget {
public:
	void Initialize(vk::PhysicalDevice device, const vk::PhysicalDeviceProperties& properties, uint64_t cap);
	LocalMemorySample Read(vk::PhysicalDevice device, vk::PhysicalDeviceMemoryBudgetPropertiesEXT* heaps = nullptr) const;
	static uint64_t Target(uint64_t budget, uint64_t usage, uint64_t heap_size,
	                       std::optional<BoardMemory> board);

private:
	uint64_t m_cap = UINT64_MAX;
	bool m_discrete = false;
	std::function<std::optional<BoardMemory>()> m_read_board;
	mutable std::mutex m_mutex;
	mutable std::chrono::steady_clock::time_point m_next_sample {};
	mutable LocalMemorySample m_sample;
};

}

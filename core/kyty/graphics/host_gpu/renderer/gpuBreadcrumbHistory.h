#pragma once

#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

#include <array>
#include <cstdio>
#include <deque>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics { enum class ShaderType; }

namespace Libs::Graphics::GpuBreadcrumbs {

inline constexpr uint32_t Slots = 8;

struct Markers {
	volatile uint32_t tick, started, completed;
};

struct Program {
	ShaderType stage;
	uint64_t hash, module;
	uint8_t bvh_paths;

	bool operator==(const Program&) const = default;
};

class History {
public:
	uint32_t Record(uint32_t slot, uint64_t tick, const char* kind, uint64_t submit_id, const char* callback);
	void AddProgram(uint32_t id, Program program);
	void Retire(uint32_t slot, uint64_t reached);
	void Sample(std::span<const Markers> markers);
	void Describe(std::FILE* file, std::span<const Markers> markers);
	void Report(GraphicContext& graphics, vk::Result result, const char* call, const char* scheduler, uint64_t tick,
	            std::span<const Markers> markers);

private:
	struct Operation {
		uint32_t slot;
		uint64_t tick;
		const char* kind;
		uint64_t submit_id;
		const char* callback;
		std::vector<Program> programs;
	};
	std::string describeOperation(uint32_t id, bool include_finished = false) const;
	std::string describeMarkers(std::span<const Markers> markers) const;

	std::mutex mutex;
	std::once_flag lost;
	uint32_t next = 0;
	std::unordered_map<uint32_t, Operation> operations;
	std::array<std::deque<uint32_t>, Slots> pending;
	std::array<std::optional<std::pair<uint32_t, Operation>>, Slots> last_retired;
	std::deque<std::string> samples;
};

}

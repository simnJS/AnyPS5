#include "graphics/host_gpu/driverMemoryReport.h"
#include "graphics/host_gpu/memoryTelemetry.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

using namespace Libs::Graphics;

namespace {

void Require(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "memory telemetry: %s\n", what);
		std::exit(1);
	}
}

void CheckResources() {
	using namespace MemoryTelemetry;
	for (size_t index = 0; index < Names.size(); ++index) {
		for (bool local: {false, true}) {
			Allocation charge;
			const auto& value = totals[index][local];
			charge.Add(static_cast<Kind>(index), local, 4096);
			Require(value.bytes == 4096 && value.count == 1, Names[index]);
			charge.Reclassify(Kind::Other);
			charge.Release();
			Require(value.bytes == 0 && value.count == 0, Names[index]);
			const auto& other = totals[static_cast<size_t>(Kind::Other)][local];
			Require(other.bytes == 0 && other.count == 0, "reclassification retained bytes");
		}
	}
	{
		PipelineBytes first(42);
		auto second = std::move(first);
		second.Set(7);
		std::vector<uint32_t, PipelineAllocator<uint32_t>> words(100);
		Require(pipeline_ram == 7 + words.capacity() * sizeof(uint32_t), "cache allocation and move");
	}
	Require(pipeline_ram == 0, "cache release");
}

void CheckDriver() {
	auto report = std::make_unique<DriverMemoryReport>(2);
	constexpr std::array types {VK_OBJECT_TYPE_PIPELINE, VK_OBJECT_TYPE_PIPELINE_CACHE,
	                            VK_OBJECT_TYPE_DESCRIPTOR_POOL, VK_OBJECT_TYPE_COMMAND_POOL};
	for (const auto type: types) {
		const auto category = type == VK_OBJECT_TYPE_DESCRIPTOR_POOL ? DriverMemoryReport::DescriptorPool
		                    : type == VK_OBJECT_TYPE_COMMAND_POOL ? DriverMemoryReport::CommandPool
		                                                          : DriverMemoryReport::Pipeline;
		VkDeviceMemoryReportCallbackDataEXT event {};
		event.objectType = type;
		event.type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT;
		event.memoryObjectId = 0;
		event.size = 8192;
		event.heapIndex = 0;
		report->Observe(event);
		Require(report->Bytes(category) == 0, "system heap counted as local");
		event.memoryObjectId = 1;
		event.heapIndex = 1;
		report->Observe(event);
		Require(report->Bytes(category) == 8192, "driver allocation");
		event.type = VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT;
		event.size = UINT64_MAX;
		event.heapIndex = UINT32_MAX;
		report->Observe(event);
		Require(report->Bytes(category) == 0, "driver free");
	}
}

void MeasureCounters() {
	using namespace MemoryTelemetry;
	using Clock = std::chrono::steady_clock;
	constexpr size_t pairs = 1000000;
	const auto start = Clock::now();
	for (size_t i = 0; i < pairs; ++i) {
		Allocation charge;
		charge.Add(Kind::Buffer, true, 4096);
		charge.Release();
	}
	const auto elapsed = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
	std::printf("allocation/free pairs=%zu ns/pair=%.2f\n", pairs, elapsed / pairs);
}

}

int main(int argc, char** argv) {
	CheckResources();
	CheckDriver();
	if (argc == 2 && std::string_view(argv[1]) == "--cost") MeasureCounters();
	return 0;
}

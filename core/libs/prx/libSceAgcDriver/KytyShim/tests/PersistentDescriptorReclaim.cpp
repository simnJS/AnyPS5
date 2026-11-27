#include "TestVulkanDevice.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include <algorithm>
#include <array>
#include <cstdio>

namespace {

int Check(bool condition, const char* what) {
	if (condition) {
		return 0;
	}
	std::fprintf(stderr, "kyty_shim_persistent_descriptor_reclaim: FAILED %s\n", what);
	return 1;
}

}

int main() {
	KytyShimTests::TestVulkanDevice vulkan;
	if (const char* skip = vulkan.Initialize(); skip != nullptr) {
		std::printf("kyty_shim_persistent_descriptor_reclaim: skipped (%s)\n", skip);
		return 77;
	}

	const std::array bindings_a {
	    vk::DescriptorSetLayoutBinding {.binding         = 0,
	                                    .descriptorType  = vk::DescriptorType::eSampledImage,
	                                    .descriptorCount = 1,
	                                    .stageFlags      = vk::ShaderStageFlagBits::eCompute},
	};
	const std::array bindings_b {
	    vk::DescriptorSetLayoutBinding {.binding         = 0,
	                                    .descriptorType  = vk::DescriptorType::eStorageBuffer,
	                                    .descriptorCount = 1,
	                                    .stageFlags      = vk::ShaderStageFlagBits::eCompute},
	};

	vk::DescriptorSetLayout layout_a {};
	vk::DescriptorSetLayout layout_b {};
	const vk::DescriptorSetLayoutCreateInfo info_a {
	    .bindingCount = static_cast<uint32_t>(bindings_a.size()),
	    .pBindings    = bindings_a.data(),
	};
	const vk::DescriptorSetLayoutCreateInfo info_b {
	    .bindingCount = static_cast<uint32_t>(bindings_b.size()),
	    .pBindings    = bindings_b.data(),
	};
	if (vulkan.graphics.device.createDescriptorSetLayout(&info_a, nullptr, &layout_a) !=
	        vk::Result::eSuccess ||
	    vulkan.graphics.device.createDescriptorSetLayout(&info_b, nullptr, &layout_b) !=
	        vk::Result::eSuccess) {
		std::fprintf(stderr, "kyty_shim_persistent_descriptor_reclaim: FAILED create layout\n");
		return 1;
	}

	Libs::Graphics::DescriptorHeap::RegisterLayout(layout_a, bindings_a);
	Libs::Graphics::DescriptorHeap::RegisterLayout(layout_b, bindings_b);

	int failures = 0;
	{
		Libs::Graphics::MasterSemaphore semaphore(vulkan.graphics);
		Libs::Graphics::DescriptorHeap  heap(vulkan.graphics, semaphore);

		const auto set_a1 = heap.CommitPersistent(layout_a);
		failures += Check(static_cast<bool>(set_a1), "CommitPersistent layout_a first");
		failures += Check(heap.ActivePersistentPoolCount() == 1, "active count after first");
		failures += Check(heap.RetiredPersistentPoolCount() == 0, "retired count after first");

		const uint64_t in_flight_tick = semaphore.NextTick();
		heap.RetirePersistent(set_a1, in_flight_tick);
		failures += Check(heap.ActivePersistentPoolCount() == 0, "active count after retire");
		failures += Check(heap.RetiredPersistentPoolCount() == 1, "retired count after retire");

		const auto set_a2 = heap.CommitPersistent(layout_a);
		failures += Check(static_cast<bool>(set_a2) && set_a2 != set_a1,
		                  "in-flight retired set must not be reused");
		failures += Check(heap.ActivePersistentPoolCount() == 1, "active count while in-flight");
		failures += Check(heap.RetiredPersistentPoolCount() == 1, "in-flight pool retained");

		if (vulkan.graphics.device.signalSemaphore(
		        vk::SemaphoreSignalInfo {.semaphore = semaphore.Handle(), .value = in_flight_tick}) !=
		    vk::Result::eSuccess) {
			failures += Check(false, "signalSemaphore in_flight_tick");
		}

		const auto set_a3 = heap.CommitPersistent(layout_a);
		failures += Check(set_a3 == set_a1, "completed retired set with same layout reused");
		failures += Check(heap.ActivePersistentPoolCount() == 2, "active count after reuse");
		failures += Check(heap.RetiredPersistentPoolCount() == 0, "retired count after reuse");

		const uint64_t tick_2 = std::max(semaphore.NextTick(), in_flight_tick + 1);
		heap.RetirePersistent(set_a2, tick_2);
		if (vulkan.graphics.device.signalSemaphore(
		        vk::SemaphoreSignalInfo {.semaphore = semaphore.Handle(), .value = tick_2}) !=
		    vk::Result::eSuccess) {
			failures += Check(false, "signalSemaphore tick_2");
		}

		const auto set_b1 = heap.CommitPersistent(layout_b);
		failures += Check(static_cast<bool>(set_b1), "CommitPersistent layout_b");
		failures += Check(heap.ActivePersistentPoolCount() == 2, "active count after layout_b");
		failures += Check(heap.RetiredPersistentPoolCount() == 0,
		                  "completed mismatched retired pool destroyed");
	}

	vulkan.graphics.device.destroyDescriptorSetLayout(layout_a, nullptr);
	vulkan.graphics.device.destroyDescriptorSetLayout(layout_b, nullptr);

	if (failures != 0) {
		return 1;
	}
	std::printf("kyty_shim_persistent_descriptor_reclaim: ok\n");
	return 0;
}

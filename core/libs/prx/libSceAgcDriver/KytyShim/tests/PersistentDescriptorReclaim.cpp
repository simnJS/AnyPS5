#include "TestVulkanDevice.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

int Check(bool condition, const char* what) {
	if (condition) {
		return 0;
	}
	std::fprintf(stderr, "kyty_shim_persistent_descriptor_reclaim: FAILED %s\n", what);
	return 1;
}

void CompleteTick(vk::Queue queue, vk::Semaphore semaphore, uint64_t tick) {
	const vk::SemaphoreSubmitInfo signal {
	    .semaphore = semaphore, .value = tick, .stageMask = vk::PipelineStageFlagBits2::eAllCommands};
	const vk::SubmitInfo2 submit {.signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal};
	if (queue.submit2(1, &submit, nullptr) != vk::Result::eSuccess ||
	    queue.waitIdle() != vk::Result::eSuccess) {
		std::fprintf(stderr, "kyty_shim_persistent_descriptor_reclaim: FAILED queue completion\n");
		std::exit(1);
	}
}

}

int main() {
	KytyShimTests::TestVulkanDevice vulkan;
	if (const char* skip = vulkan.Initialize(); skip != nullptr) {
		const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
		std::fprintf(stderr, "kyty_shim_persistent_descriptor_reclaim: %s\n", skip);
		return required != nullptr && required[0] != '\0' && required[0] != '0' ? 1 : 77;
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

		CompleteTick(vulkan.graphics.queue, semaphore.Handle(), in_flight_tick);

		const auto set_a3 = heap.CommitPersistent(layout_a);
		failures += Check(set_a3 == set_a1, "completed retired set with same layout reused");
		failures += Check(heap.ActivePersistentPoolCount() == 2, "active count after reuse");
		failures += Check(heap.RetiredPersistentPoolCount() == 0, "retired count after reuse");

		const auto set_b1 = heap.CommitPersistent(layout_b);
		const auto tick_2 = semaphore.NextTick();
		heap.RetirePersistent(set_b1, in_flight_tick);
		heap.RetirePersistent(set_a2, tick_2);
		CompleteTick(vulkan.graphics.queue, semaphore.Handle(), tick_2);
		const auto set_a4 = heap.CommitPersistent(layout_a);
		failures += Check(set_a4 == set_a2, "reuse behind a completed mismatched pool");
		failures += Check(heap.ActivePersistentPoolCount() == 2, "active count after mixed layouts");
		failures += Check(heap.RetiredPersistentPoolCount() == 0, "completed mismatch destroyed");

		const auto tick_3 = semaphore.NextTick();
		heap.RetirePersistent(set_a3, tick_3);
		heap.RetirePersistent(set_a4, tick_2);
		heap.TrimCompletedPools();
		failures += Check(heap.RetiredPersistentPoolCount() == 1, "trim removes the completed pool");
		const auto set_a5 = heap.CommitPersistent(layout_a);
		failures += Check(set_a5 != set_a3 && heap.RetiredPersistentPoolCount() == 1,
		                  "pending set stays retired after trim");
		CompleteTick(vulkan.graphics.queue, semaphore.Handle(), tick_3);
		heap.TrimCompletedPools();
		failures += Check(heap.RetiredPersistentPoolCount() == 0, "trim after the last allocation");
	}

	vulkan.graphics.device.destroyDescriptorSetLayout(layout_a, nullptr);
	vulkan.graphics.device.destroyDescriptorSetLayout(layout_b, nullptr);

	if (failures != 0) {
		return 1;
	}
	std::printf("kyty_shim_persistent_descriptor_reclaim: ok\n");
	return 0;
}

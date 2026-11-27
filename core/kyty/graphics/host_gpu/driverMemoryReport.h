#pragma once

#include <vulkan/vulkan_core.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>

namespace Libs::Graphics {

class DriverMemoryReport {
public:
	enum Category { Pipeline, DescriptorPool, CommandPool, Count };

	explicit DriverMemoryReport(uint32_t local_heaps): m_local_heaps(local_heaps) {}

	void Observe(const VkDeviceMemoryReportCallbackDataEXT& event) noexcept {
		if (event.type != VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT &&
		    event.type != VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT) return;
		const auto id = event.memoryObjectId;
		auto& bucket = m_buckets[((id ^ (id >> 32)) * 11400714819323198485ull) >> 52];
		if (event.type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT) {
			Category kind;
			switch (event.objectType) {
				case VK_OBJECT_TYPE_PIPELINE:
				case VK_OBJECT_TYPE_PIPELINE_CACHE:
				case VK_OBJECT_TYPE_PIPELINE_LAYOUT:
				case VK_OBJECT_TYPE_SHADER_MODULE: kind = Pipeline; break;
				case VK_OBJECT_TYPE_DESCRIPTOR_SET:
				case VK_OBJECT_TYPE_DESCRIPTOR_POOL: kind = DescriptorPool; break;
				case VK_OBJECT_TYPE_COMMAND_BUFFER:
				case VK_OBJECT_TYPE_COMMAND_POOL: kind = CommandPool; break;
				default: return;
			}
			if ((m_local_heaps & (1u << event.heapIndex)) == 0) return;
			for (auto& slot: bucket) {
				State empty = Empty;
				if (slot.state.compare_exchange_strong(empty, Filling, std::memory_order_acq_rel)) {
					slot.id.store(id, std::memory_order_relaxed);
					slot.bytes = event.size;
					slot.kind = kind;
					m_bytes[kind].fetch_add(event.size, std::memory_order_relaxed);
					slot.state.store(Live, std::memory_order_release);
					return;
				}
			}
			m_incomplete[kind].store(true, std::memory_order_relaxed);
		} else {
			for (auto& slot: bucket) {
				if (slot.state.load(std::memory_order_acquire) == Live && slot.id.load(std::memory_order_relaxed) == id) {
					m_bytes[slot.kind].fetch_sub(slot.bytes, std::memory_order_relaxed);
					slot.state.store(Empty, std::memory_order_release);
					return;
				}
			}
		}
	}

	[[nodiscard]] std::optional<uint64_t> Bytes(Category kind) const {
		if (m_incomplete[kind].load(std::memory_order_relaxed)) return {};
		return m_bytes[kind].load(std::memory_order_relaxed);
	}

private:
	enum State : uint8_t { Empty, Filling, Live };
	struct Slot {
		std::atomic<State> state {};
		std::atomic<uint64_t> id {};
		uint64_t bytes = 0;
		Category kind = Pipeline;
	};

	uint32_t m_local_heaps;
	std::array<std::array<Slot, 8>, 4096> m_buckets {};
	std::array<std::atomic<uint64_t>, Count> m_bytes {};
	std::array<std::atomic<bool>, Count> m_incomplete {};
};

}

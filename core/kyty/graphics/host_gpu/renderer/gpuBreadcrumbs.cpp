#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/stallSnapshot.h"

#include <fmt/format.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace Libs::Graphics::GpuBreadcrumbs {
namespace {

constexpr uint32_t RingSize        = 1u << 14u;
constexpr uint32_t StartedOffset   = 0;
constexpr uint32_t CompletedOffset = 4;
constexpr uint32_t FirstSlot       = 2;
constexpr uint32_t Slots           = 8;

struct Operation {
	std::atomic<uint32_t>    id {0};
	std::atomic<uint64_t>    tick {0};
	std::atomic<uint64_t>    submit_id {0};
	std::atomic<const char*> kind {nullptr};
};

std::mutex                           g_mutex;
bool                                 g_initialized = false;
VkBuffer                             g_buffer      = VK_NULL_HANDLE;
VmaAllocation                        g_allocation  = nullptr;
volatile uint32_t*                   g_markers     = nullptr;
std::atomic<uint32_t>                g_slots {0};
std::array<const char*, Slots>       g_slot_names {};
std::atomic<uint32_t>                g_next {0};
std::array<Operation, RingSize>      g_ring {};
int                                  g_owner = 0;

void Section(void*, std::FILE* file) {
	if (g_markers == nullptr) {
		std::fprintf(file, "no marker buffer\n");
		return;
	}
	const auto slots = std::min(g_slots.load(std::memory_order_acquire), Slots);
	for (uint32_t slot = 0; slot < slots; slot++) {
		std::fprintf(file, "the GPU began the submission of %s tick %u (low 32 bits)\n", g_slot_names[slot],
		             g_markers[FirstSlot + slot]);
	}
	if (!PerOperation()) {
		std::fprintf(file, "guest operations are not marked (KYTY_GPU_BREADCRUMBS=1 marks them)\n");
		return;
	}
	const uint32_t recorded  = g_next.load(std::memory_order_acquire);
	const uint32_t started   = g_markers[StartedOffset / 4];
	const uint32_t completed = g_markers[CompletedOffset / 4];
	std::fprintf(file, "guest operations: %u marked, the GPU began %u and finished %u\n", recorded, started, completed);
	const auto describe = [file](const char* what, uint32_t id) {
		const auto& entry = g_ring[id % RingSize];
		if (entry.id.load(std::memory_order_acquire) != id) return;
		const auto* kind = entry.kind.load(std::memory_order_relaxed);
		std::fprintf(file, "  %s: operation %u, %s of submission %" PRIu64 " in tick %" PRIu64 "\n", what, id,
		             kind != nullptr ? kind : "?", entry.submit_id.load(std::memory_order_relaxed),
		             entry.tick.load(std::memory_order_relaxed));
	};
	if (completed != 0) describe("last finished", completed);
	const uint32_t first = completed + 1u;
	for (uint32_t id = first; id <= std::max(started, first) && id - first < 32u && id <= recorded; id++) {
		describe(id <= started ? "begun, not finished" : "next", id);
	}
}

bool Initialize(GraphicContext& graphics) {
	std::scoped_lock lock {g_mutex};
	if (g_initialized) return g_markers != nullptr;
	g_initialized = true;
	if (!graphics.buffer_marker_enabled || graphics.allocator == nullptr) return false;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size  = 256;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags         = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.usage         = VMA_MEMORY_USAGE_AUTO;
	allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VmaAllocationInfo result {};
	if (vmaCreateBuffer(graphics.allocator, &buffer_info, &allocation_info, &g_buffer, &g_allocation, &result) != VK_SUCCESS ||
	    result.pMappedData == nullptr) {
		Log::WriteToConsoleAndLog("[kyty] GPU breadcrumbs: no host-visible marker buffer\n");
		return false;
	}
	g_markers = static_cast<volatile uint32_t*>(result.pMappedData);
	for (uint32_t i = 0; i < 64; i++) g_markers[i] = 0;
	StallSnapshot::Register(&g_owner, "GPU markers (VK_AMD_buffer_marker)", &Section);
	return true;
}

}

bool Requested() {
	return StallSnapshot::Enabled() || PerOperation();
}

bool PerOperation() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_BREADCRUMBS");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool Available() {
	return g_markers != nullptr;
}

uint32_t SubmissionSlot(const char* name) {
	const auto slot = g_slots.fetch_add(1, std::memory_order_acq_rel);
	if (slot >= Slots) return UINT32_MAX;
	g_slot_names[slot] = name;
	return slot;
}

void MarkSubmission(GraphicContext& graphics, const CommandRecorder& command, uint32_t slot, uint64_t tick) {
	if (slot >= Slots || !Requested() || !Initialize(graphics)) return;
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_buffer, (FirstSlot + slot) * 4u,
	                             static_cast<uint32_t>(tick));
}

std::optional<uint64_t> StartedTick(uint32_t slot, uint64_t near) {
	if (slot >= Slots || g_markers == nullptr) return std::nullopt;
	const uint64_t low = g_markers[FirstSlot + slot];
	if (low == 0) return std::nullopt;
	uint64_t tick = (near & ~uint64_t {0xffffffff}) | low;
	if (tick > near + (uint64_t {1} << 31u) && tick >= (uint64_t {1} << 32u)) tick -= uint64_t {1} << 32u;
	if (tick + (uint64_t {1} << 31u) < near) tick += uint64_t {1} << 32u;
	return tick;
}

uint32_t BeginOperation(GraphicContext& graphics, const CommandRecorder& command, uint64_t tick, const char* kind,
                        uint64_t submit_id) {
	if (!PerOperation() || !Initialize(graphics)) return 0;
	auto id = g_next.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (id == 0) id = g_next.fetch_add(1, std::memory_order_relaxed) + 1u;
	auto& entry = g_ring[id % RingSize];
	entry.tick.store(tick, std::memory_order_relaxed);
	entry.submit_id.store(submit_id, std::memory_order_relaxed);
	entry.kind.store(kind, std::memory_order_relaxed);
	entry.id.store(id, std::memory_order_release);
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_buffer, StartedOffset, id);
	return id;
}

void EndOperation(const CommandRecorder& command, uint32_t id) {
	if (id == 0 || g_markers == nullptr) return;
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, g_buffer, CompletedOffset, id);
}

}

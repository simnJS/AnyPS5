#include "graphics/host_gpu/renderer/gpuBreadcrumbHistory.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreTypes.h"
#include "graphics/host_gpu/renderer/stallSnapshot.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <fmt/format.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <thread>

namespace Libs::Graphics::GpuBreadcrumbs {
namespace {

std::mutex g_init_mutex;
GraphicContext* g_graphics = nullptr;
VkBuffer g_buffer = VK_NULL_HANDLE;
VmaAllocation g_allocation = nullptr;
std::atomic<Markers*> g_markers {nullptr};
std::array<std::atomic<const char*>, Slots> g_names {};
std::unique_ptr<History> g_history;
std::jthread g_sampler;
thread_local OperationScope* g_current = nullptr;

const char* Name(uint32_t slot) {
	const char* name = slot < Slots ? g_names[slot].load() : nullptr;
	return name ? name : "unknown";
}

std::span<const Markers> Mapped() {
	const auto* markers = g_markers.load();
	return {markers, markers ? Slots : 0u};
}

void Section(void*, std::FILE* file) {
	const auto markers = Mapped();
	for (uint32_t slot = 0; slot < markers.size(); ++slot) {
		if (g_names[slot].load()) {
			std::fprintf(file, "the GPU began the submission of %s tick %u (low 32 bits)\n", Name(slot), markers[slot].tick);
		}
	}
	if (g_history) g_history->Describe(file, markers);
	else std::fputs("guest operations are not marked (KYTY_GPU_BREADCRUMBS=1 marks them)\n", file);
}

void Initialize(GraphicContext& graphics) {
	std::scoped_lock lock {g_init_mutex};
	if (g_graphics) return;
	g_graphics = &graphics;
	if (PerOperation() && !g_history) g_history = std::make_unique<History>();
	if (!graphics.buffer_marker_enabled || !graphics.allocator) return;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size  = Slots * sizeof(Markers);
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
	allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VmaAllocationInfo result {};
	if (vmaCreateBuffer(graphics.allocator, &buffer_info, &allocation_info, &g_buffer, &g_allocation, &result) != VK_SUCCESS) {
		Log::WriteToConsoleAndLog("[kyty] GPU breadcrumbs: no host-visible marker buffer\n");
		return;
	}
	std::memset(result.pMappedData, 0, buffer_info.size);
	g_markers.store(static_cast<Markers*>(result.pMappedData));
	StallSnapshot::Register(&g_buffer, "GPU markers (VK_AMD_buffer_marker)", &Section);
	if (g_history) {
		g_sampler = std::jthread([](std::stop_token stop) {
			while (!stop.stop_requested()) {
				g_history->Sample(Mapped());
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
			}
		});
	}
}

void Mark(CommandScheduler& scheduler, uint32_t id, bool end) {
	auto command = scheduler.Current().Handle();
	if (g_markers.load()) {
		const auto offset = scheduler.MarkerSlot() * sizeof(Markers) + (end ? offsetof(Markers, completed) : offsetof(Markers, started));
		command.writeBufferMarkerAMD(end ? vk::PipelineStageFlagBits::eBottomOfPipe : vk::PipelineStageFlagBits::eTopOfPipe,
		                             g_buffer, offset, id);
	}
	if (scheduler.Graphics().checkpoints_enabled) command.setCheckpointNV((uintptr_t{id} << 1) | uintptr_t{end});
}

std::string Paths(uint8_t mask) {
	std::string text;
	for (const auto& [bit, name]: {std::pair{BvhPath::Native, "native"}, {BvhPath::Oracle, "oracle"}, {BvhPath::Empty, "empty"},
	                               {BvhPath::Emulated, "emulated"}, {BvhPath::Triangle, "triangle"}}) {
		if (mask & static_cast<uint8_t>(bit)) {
			if (!text.empty()) text += '+';
			text += name;
		}
	}
	return text.empty() ? "none" : text;
}

}

uint32_t History::Record(uint32_t slot, uint64_t tick, const char* kind, uint64_t submit_id, const char* callback) {
	std::scoped_lock lock(mutex);
	do { ++next; } while (next == 0 || operations.contains(next));
	operations.emplace(next, Operation{slot, tick, kind, submit_id, callback, {}});
	pending.at(slot).push_back(next);
	return next;
}

void History::AddProgram(uint32_t id, Program program) {
	std::scoped_lock lock(mutex);
	auto& programs = operations.at(id).programs;
	if (std::find(programs.begin(), programs.end(), program) == programs.end()) programs.push_back(program);
}

void History::Retire(uint32_t slot, uint64_t reached) {
	std::scoped_lock lock(mutex);
	auto& queue = pending.at(slot);
	while (!queue.empty() && operations.at(queue.front()).tick <= reached) {
		auto entry = operations.extract(queue.front());
		if (queue.size() == 1 || operations.at(queue[1]).tick > reached)
			last_retired[slot].emplace(entry.key(), std::move(entry.mapped()));
		queue.pop_front();
	}
}

std::string History::describeOperation(uint32_t id, bool include_finished) const {
	const auto found = operations.find(id);
	const Operation* entry = found == operations.end() ? nullptr : &found->second;
	if (!entry && include_finished) {
		for (const auto& last: last_retired)
			if (last && last->first == id) entry = &last->second;
	}
	if (!entry) return fmt::format("operation {} (descriptor retired or unavailable)", id);
	const auto& op = *entry;
	auto text = fmt::format("operation {} scheduler={} tick={} submission={} type={}", id, Name(op.slot), op.tick, op.submit_id, op.kind);
	if (op.callback) text += fmt::format(" command={}", op.callback);
	if (op.programs.empty()) text += " program=none recorded";
	for (const auto& p: op.programs)
		text += fmt::format(" {}=0x{:016x} module=0x{:016x} bvh={}", ShaderRecompiler::StageName(p.stage), p.hash, p.module, Paths(p.bvh_paths));
	return text;
}

std::string History::describeMarkers(std::span<const Markers> markers) const {
	std::string text;
	for (size_t i = 0; i < markers.size(); ++i) {
		const auto started = markers[i].started;
		const auto completed = markers[i].completed;
		if (markers[i].started != started) {
			text += fmt::format("{} marker changed while sampling\n", Name(i));
			continue;
		}
		if (started == 0 && completed == 0) continue;
		text += fmt::format("{} started={} completed={}\n", Name(i), started, completed);
		if (started && started != completed) {
			const auto found = operations.find(started);
			if (found != operations.end() && found->second.slot == i)
				text += fmt::format("last begun, not finished: {}\n", describeOperation(started));
			else
				text += fmt::format("unresolved start: operation {} (descriptor retired or unavailable)\n", started);
		} else if (completed)
			text += fmt::format("last finished: {}\n", describeOperation(completed, true));
	}
	return text.empty() ? "no operation markers observed\n" : text;
}

void History::Sample(std::span<const Markers> markers) {
	std::scoped_lock lock(mutex);
	const auto now = std::chrono::steady_clock::now().time_since_epoch();
	samples.push_back(fmt::format("sample steady_ms={}\n{}", std::chrono::duration_cast<std::chrono::milliseconds>(now).count(),
	                              describeMarkers(markers)));
	if (samples.size() > 8) samples.pop_front();
}

void History::Describe(std::FILE* file, std::span<const Markers> markers) {
	std::scoped_lock lock(mutex);
	std::fputs(describeMarkers(markers).c_str(), file);
}

void History::Report(GraphicContext& graphics, vk::Result result, const char* call, const char* scheduler, uint64_t tick,
                     std::span<const Markers> markers) {
	if (result != vk::Result::eErrorDeviceLost || !PerOperation() || !StallSnapshot::Enabled()) return;
	std::call_once(lost, [&] {
		std::error_code error;
		const std::filesystem::path dir(StallSnapshot::Directory());
		std::filesystem::create_directories(dir, error);
		const auto path = dir / fmt::format("device-lost-{}-tick{}.txt", std::chrono::system_clock::now().time_since_epoch().count(), tick);
		auto* file = std::fopen(path.string().c_str(), "w");
		if (!file) {
			std::perror("[kyty] opening device-loss report");
			return;
		}
		std::scoped_lock lock(mutex);
		std::fprintf(file, "VK_ERROR_DEVICE_LOST call=%s scheduler=%s tick=%llu\n", call, scheduler, static_cast<unsigned long long>(tick));
		for (const auto& sample: samples) std::fputs(sample.c_str(), file);
		std::fflush(file);
		std::fputs("mapped markers at device loss (best effort)\n", file);
		std::fputs(describeMarkers(markers).c_str(), file);
		std::fflush(file);
		if (graphics.checkpoints_enabled && graphics.queue) {
			uint32_t count = 0;
			graphics.queue.getCheckpointDataNV(&count, nullptr);
			std::vector<vk::CheckpointDataNV> data(count);
			if (count) graphics.queue.getCheckpointDataNV(&count, data.data());
			std::fprintf(file, "VK_NV_device_diagnostic_checkpoints: %u\n", count);
			uintptr_t top = 0;
			uintptr_t bottom = 0;
			for (uint32_t i = 0; i < count; ++i) {
				const auto token = reinterpret_cast<uintptr_t>(data[i].pCheckpointMarker);
				if (data[i].stage == vk::PipelineStageFlagBits::eTopOfPipe) top = token;
				if (data[i].stage == vk::PipelineStageFlagBits::eBottomOfPipe) bottom = token;
				std::fprintf(file, "stage=%s checkpoint=%s %s\n", vk::to_string(data[i].stage).c_str(), token & 1 ? "end" : "begin",
				             describeOperation(static_cast<uint32_t>(token >> 1), true).c_str());
			}
			if (top && bottom && ((top >> 1) != (bottom >> 1) || !(bottom & 1))) {
				std::fprintf(file, "NV last begun, not finished: %s\n", describeOperation(static_cast<uint32_t>(top >> 1)).c_str());
			}
		}
		if (std::fclose(file) != 0) std::perror("[kyty] closing device-loss report");
		else std::fprintf(stderr, "[kyty] device-loss report: %s\n", path.string().c_str());
	});
}

bool Requested() {
	return StallSnapshot::Enabled() || PerOperation();
}

bool PerOperation() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GPU_BREADCRUMBS");
		return value && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

bool NeedsBvhPaths(const ShaderModule& module) {
	return module.bvh != ShaderBvh::None && module.bvh_paths == 0 && PerOperation();
}

bool Available() {
	return g_markers.load() != nullptr;
}

uint32_t SubmissionSlot(const char* name) {
	for (uint32_t slot = 0; slot < Slots; ++slot) {
		const char* empty = nullptr;
		if (g_names[slot].compare_exchange_strong(empty, name)) return slot;
	}
	return UINT32_MAX;
}

void MarkSubmission(GraphicContext& graphics, const CommandRecorder& command, uint32_t slot, uint64_t tick) {
	if (slot >= Slots || !Requested()) return;
	Initialize(graphics);
	if (g_markers.load()) command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_buffer, slot * sizeof(Markers),
	                                                  static_cast<uint32_t>(tick));
}

std::optional<uint64_t> StartedTick(uint32_t slot, uint64_t reference) {
	const auto markers = Mapped();
	if (slot >= markers.size()) return std::nullopt;
	const uint64_t low = markers[slot].tick;
	if (low == 0) return std::nullopt;
	uint64_t tick = (reference & ~uint64_t {0xffffffff}) | low;
	if (tick > reference + (uint64_t {1} << 31u) && tick >= (uint64_t {1} << 32u)) tick -= uint64_t {1} << 32u;
	if (tick + (uint64_t {1} << 31u) < reference) tick += uint64_t {1} << 32u;
	return tick;
}

OperationScope::OperationScope(CommandScheduler& scheduler, const char* kind, uint64_t submit_id, const char* callback)
    : scheduler(scheduler), kind(kind), submit_id(submit_id), callback(callback), outer(g_current) {
	if (outer) outer->EndSegment();
	g_current = this;
	BeginSegment();
}

OperationScope::~OperationScope() {
	EndSegment();
	g_current = outer;
	if (outer) outer->BeginSegment();
}

void OperationScope::BeginSegment() {
	if (scheduler.MarkerSlot() >= Slots || !scheduler.Recording() || (!Available() && !scheduler.Graphics().checkpoints_enabled)) return;
	id = g_history->Record(scheduler.MarkerSlot(), scheduler.CurrentTick(), kind, submit_id, callback);
	Mark(scheduler, id, false);
}

void OperationScope::EndSegment() {
	if (!id) return;
	Mark(scheduler, id, true);
	id = 0;
}

void BeforeSubmit(CommandScheduler& scheduler) {
	if (g_current && &g_current->scheduler == &scheduler) g_current->EndSegment();
}

void AfterBegin(CommandScheduler& scheduler) {
	if (g_history && scheduler.MarkerSlot() < Slots)
		g_history->Retire(scheduler.MarkerSlot(), scheduler.GetMasterSemaphore().KnownGpuTick());
	if (g_current && &g_current->scheduler == &scheduler) g_current->BeginSegment();
}

void NoteProgram(const ShaderModule& module) {
	if (g_current && g_current->id)
		g_history->AddProgram(g_current->id, {module.info.stage, module.info.shader_hash, module.id, module.bvh_paths});
}

void ReportDeviceLost(GraphicContext& graphics, vk::Result result, const char* call, const char* scheduler, uint64_t tick) {
	if (result != vk::Result::eErrorDeviceLost || !PerOperation()) return;
	std::scoped_lock lock(g_init_mutex);
	if (!g_history) g_history = std::make_unique<History>();
	g_history->Report(graphics, result, call, scheduler, tick, Mapped());
}

void Shutdown(GraphicContext& graphics) {
	std::scoped_lock lock(g_init_mutex);
	if (g_graphics != &graphics) return;
	g_sampler = {};
	StallSnapshot::Unregister(&g_buffer);
	g_markers.store(nullptr);
	if (g_buffer) vmaDestroyBuffer(graphics.allocator, g_buffer, g_allocation);
	g_buffer = VK_NULL_HANDLE;
	g_allocation = nullptr;
	g_history.reset();
	g_graphics = nullptr;
	for (auto& name: g_names) name.store(nullptr);
}

}

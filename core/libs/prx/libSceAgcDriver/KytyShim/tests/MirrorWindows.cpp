#include "TestVulkanDevice.h"
#include "common/alignment.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "gpu_tiler_shaders/buffer_scatter_spv.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/guestHostView.h"
#include "graphics/host_gpu/renderer/cache/guestMirror.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/frontend/decode/RecordReach.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"
#include "prx/libkernel/KernelErrors.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <utility>
#include <stdexcept>
#include <vector>

namespace Libs::Graphics {

struct MirrorWindowsTestAccess {
	static bool ReadBack(BufferCache& cache, uint64_t address, uint64_t size) {
		std::vector<vk::BufferCopy> copies;
		uint64_t                    total = 0;
		cache.CollectDownloadCopies(address, size, copies, total);
		return !copies.empty() && cache.DownloadOnReadbackQueue(copies, total);
	}
	static void CheckRange(BufferCache& cache, uint64_t address, uint64_t size) { cache.CheckMirrorRange(address, size); }
	static void DiscardGpuWrites(BufferCache& cache, uint64_t address, uint64_t size) {
		cache.DiscardGpuWrites(address, size);
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr const char* Name         = "kyty_shim_mirror_windows";
constexpr uint64_t    GuestBytes   = 0x80000;
constexpr int         CpuReadWrite = 0x03;
constexpr uint64_t    FirstBytes   = 0x8000;
constexpr uint64_t    JoinedBytes  = 0x20000;
constexpr uint64_t    WrittenAt    = 0x20000;
constexpr uint64_t    NextAt   = 0x24000;
constexpr uint64_t    StreamedAt   = 0x30000;
constexpr uint64_t    MirroredAt   = 0x34000;
constexpr uint64_t    WrittenBytes = 0x1000;
constexpr uint64_t    FarAt        = 0x70000;
constexpr uint64_t    GrowAt       = 0x40000;
constexpr uint64_t    GrowBlock    = 0x10000;
constexpr uint64_t    LeapBytes    = 0x200000;
constexpr uint32_t    GpuValue     = 0x5eed1234u;
constexpr uint64_t    TableAt      = 0x200;
constexpr uint64_t    TableBytes   = 0x400;

std::string g_skipped;
std::string g_unchecked;

uint32_t Pattern(uint64_t index) {
	return static_cast<uint32_t>(index * 2654435761u) ^ 0xa5a5a5a5u;
}

class AddressReader {
public:
	AddressReader(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t bytes)
	    : m_graphics(graphics),
	      m_jobs(graphics, scheduler, MemoryUsage::Upload, 0,
	             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, Slots * SlotBytes),
	      m_data(graphics, scheduler, MemoryUsage::Upload, 0,
	             vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, Slots * SlotBytes),
	      m_out(graphics, scheduler, MemoryUsage::Download, 0,
	            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
	                vk::BufferUsageFlagBits::eTransferDst,
	            bytes) {
		const vk::PushConstantRange        push {vk::ShaderStageFlagBits::eCompute, 0, 16};
		const vk::PipelineLayoutCreateInfo layout_info {.pushConstantRangeCount = 1, .pPushConstantRanges = &push};
		RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout_info, nullptr, &m_layout), "reader layout");
		const auto                              module = CompileSPV(BUFFER_SCATTER_SPV, graphics.device);
		const vk::PipelineShaderStageCreateInfo stage {
		    .stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"};
		const vk::ComputePipelineCreateInfo pipeline_info {.stage = stage, .layout = m_layout};
		RequireVulkanSuccess(graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline),
		                     "reader pipeline");
		graphics.device.destroyShaderModule(module);
	}
	~AddressReader() {
		m_graphics.device.destroyPipeline(m_pipeline);
		m_graphics.device.destroyPipelineLayout(m_layout);
	}

	void Record(CommandScheduler& scheduler, vk::DeviceAddress source, uint64_t bytes) {
		Copy(scheduler, source, m_out.BufferDeviceAddress(), bytes);
	}

	void Store(CommandScheduler& scheduler, vk::DeviceAddress destination, const uint32_t* words, uint64_t bytes) {
		const auto slot = m_slot % Slots * SlotBytes;
		std::memcpy(m_data.Mapped().data() + slot, words, bytes);
		m_data.Flush(slot, bytes);
		Copy(scheduler, m_data.BufferDeviceAddress() + slot, destination, bytes);
	}

	void Copy(CommandScheduler& scheduler, vk::DeviceAddress source, vk::DeviceAddress destination, uint64_t bytes) {
		struct Job {
			uint64_t source;
			uint64_t destination;
			uint32_t size;
			uint32_t pad0;
			uint64_t pad1;
		} job {source, destination, static_cast<uint32_t>(bytes), 0, 0};
		const auto slot = m_slot++ % Slots * SlotBytes;
		std::memcpy(m_jobs.Mapped().data() + slot, &job, sizeof(job));
		m_jobs.Flush(slot, sizeof(job));
		auto& command = scheduler.Current();
		command.EndRendering();
		const auto        native = command.Handle();
		vk::MemoryBarrier before {.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
		                          .dstAccessMask = vk::AccessFlagBits::eShaderRead};
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {}, 1,
		                       &before, 0, nullptr, 0, nullptr);
		const struct {
			uint64_t jobs;
			uint32_t count;
			uint32_t pad;
		} push {m_jobs.BufferDeviceAddress() + slot, 1, 0};
		native.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
		native.pushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		native.dispatch(1, 1, 1);
		vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
		                         .dstAccessMask = vk::AccessFlagBits::eHostRead};
		native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1, &after,
		                       0, nullptr, 0, nullptr);
	}

	[[nodiscard]] std::vector<uint32_t> Words(uint64_t bytes) {
		m_out.Invalidate(0, bytes);
		std::vector<uint32_t> words(bytes / sizeof(uint32_t));
		std::memcpy(words.data(), m_out.Mapped().data(), bytes);
		return words;
	}

private:
	static constexpr uint64_t Slots     = 8;
	static constexpr uint64_t SlotBytes = 32;

	GraphicContext&    m_graphics;
	Buffer             m_jobs;
	Buffer             m_data;
	Buffer             m_out;
	uint64_t           m_slot = 0;
	vk::PipelineLayout m_layout   = nullptr;
	vk::Pipeline       m_pipeline = nullptr;
};

void Finish(CommandScheduler& scheduler) {
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
}

bool IsResident(CommandScheduler& scheduler, BufferCache& buffers, Buffer& bitmap, uint64_t guest) {
	const auto block = BufferCache::GuestOffset(guest) / BufferCache::MIRROR_BLOCK_SIZE;
	bitmap.CopyFrom(scheduler.Current(), *buffers.GetFaultBuffer(),
	                BufferCache::RESIDENCY_OFFSET + (block / BufferCache::RESIDENCY_BLOCKS) * sizeof(uint32_t), 0, sizeof(uint32_t));
	Finish(scheduler);
	bitmap.Invalidate(0, sizeof(uint32_t));
	uint32_t bits;
	std::memcpy(&bits, bitmap.Mapped().data(), sizeof(bits));
	return (bits & (uint32_t {1} << (block % BufferCache::RESIDENCY_BLOCKS))) != 0;
}

bool ReadsPattern(const std::vector<uint32_t>& words, uint64_t first) {
	for (size_t index = 0; index < words.size(); index++) {
		if (words[index] != Pattern(first + index)) return false;
	}
	return true;
}

bool AllWords(const std::vector<uint32_t>& words, uint32_t value) {
	for (const auto word: words) {
		if (word != value) return false;
	}
	return true;
}

bool BackingHolds(uint64_t address, uint64_t bytes, uint32_t value) {
	std::vector<uint32_t> words(bytes / sizeof(uint32_t));
	return Libs::LibKernel::Memory::TryReadBacking(address, words.data(), bytes) && AllWords(words, value);
}

const char* CheckHeapConsumer(RenderContext& renderer, uint64_t address, AddressReader& reader) {
	namespace IR = ShaderRecompiler::IR;
	auto& heap      = renderer.GetImageHeap();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& buffers   = renderer.GetBufferCache();
	auto& graphics  = renderer.GetGraphics();
	constexpr uint32_t first = 0x10000, second = 0x20000, third = 0x30000, fourth = 0x40000;
	std::array<uint32_t, 16> words {};
	words[0] = first;
	words[1] = static_cast<uint32_t>(Prospero::BufferFormat::k8_8_8_8UNorm) << 20u;
	words[3] = (static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28u) | IR::ShaderImageIdentitySwizzle;
	std::copy_n(words.begin(), 8, words.begin() + 8);
	words[8] = second;
	renderer.MapMemory(address, sizeof(words));
	const std::array<uint32_t, 16> empty_table {};
	const auto cpu_write = [&](uint64_t at, const std::array<uint32_t, 16>& table) {
		if (!Libs::LibKernel::Memory::TryWriteBacking(at, table.data(), sizeof(table))) return false;
		buffers.InvalidateMemory(at, sizeof(table));
		return true;
	};
	if (!cpu_write(address, empty_table)) return "could not clear the descriptor table in guest backing";

	vk::DescriptorSetLayout empty;
	const vk::DescriptorSetLayoutCreateInfo empty_info {};
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&empty_info, nullptr, &empty), "empty set layout");
	const std::array sets {empty, graphics.image_heap_layout};
	const vk::PipelineLayoutCreateInfo layout_info {.setLayoutCount = 2, .pSetLayouts = sets.data()};
	vk::PipelineLayout layout;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout_info, nullptr, &layout), "consumer layout");
	const IR::ImageResource root {.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D};
	const IR::ImageHeapRequest request {.base = address, .size = sizeof(words), .step = 32, .count = 2, .root = &root};
	bool waited = false;
	const auto consume_at = [&](const IR::ImageHeapRequest& wanted, uint64_t bytes, std::initializer_list<uint32_t> writes,
	                            size_t kinds) {
		IR::ImageHeapView view;
		const auto        tick = scheduler.CurrentTick();
		heap.BeginDraw();
		if (!heap.Acquire(wanted, view) || view.kinds.size() != kinds) return std::vector<uint32_t> {};
		heap.Rebind(renderer.GetRenderExecutor());
		for (const auto value: writes) {
			const auto [buffer, offset] = buffers.ObtainRange(wanted.base, sizeof(uint32_t), true);
			buffer->Fill(offset, sizeof(uint32_t), value);
		}
		heap.Commit(scheduler.Current(), vk::PipelineBindPoint::eCompute, layout);
		waited |= scheduler.CurrentTick() != tick;
		reader.Record(scheduler, view.slots, bytes);
		Finish(scheduler);
		return reader.Words(bytes);
	};
	const auto consume = [&](std::initializer_list<uint32_t> writes = {}) {
		return consume_at(request, sizeof(words), writes, 1u);
	};

	heap.BeginDraw();
	IR::ImageHeapView empty_view;
	auto outside = request;
	outside.base = BufferCache::GUEST_END - 4u;
	if (heap.Acquire(outside, empty_view)) return "a table crossing the mirror limit was accepted";
	auto crossing = request;
	crossing.base = Common::AlignDown(address, GuestMirror::WindowSize) - 4u;
	if (!renderer.IsMapped(crossing.base, crossing.size)) return "cross-window table fixture is not mapped";
	if (heap.Acquire(crossing, empty_view)) return "an image table crossing two windows was admitted";
	auto cpu_request = request;
	cpu_request.step = sizeof(words);
	if (!heap.Acquire(cpu_request, empty_view) || !empty_view.kinds.empty()) return "unexpected initial CPU kind";
	if (!cpu_write(address, words)) return "CPU table write failed";
	const auto armed = consume_at(cpu_request, sizeof(uint32_t), {}, 1u);
	if (armed.empty() || armed.front() == 0u) return "the first CPU descriptor used slot zero at its first draw";
	auto unfiltered = root;
	unfiltered.unfiltered_access_only = true;
	auto other_request = cpu_request;
	other_request.root = &unfiltered;
	const auto shared = consume_at(other_request, sizeof(uint32_t), {}, 1u);
	if (shared.empty() || shared.front() != armed.front()) return "equivalent image bindings occupied different slots";

	const auto initial = consume();
	if (initial.empty() || initial.front() == 0 || initial[8] == 0 || initial.front() == initial[8]) {
		return "the two seeded descriptors did not produce distinct slots at their first draw";
	}
	if (heap.LateMisses() != 0u) return "a CPU-written table reported a missing key";

	auto groups = heap.FillGroups();
	if (!cpu_write(address, words)) return "identical CPU table write failed";
	if (consume() != initial || heap.FillGroups() != groups) return "a page rewritten with the same bytes refilled its slots";

	auto swapped = words;
	swapped[0]   = second;
	if (!cpu_write(address, swapped)) return "CPU key change failed";
	const auto changed = consume();
	if (changed.empty() || changed[0] != initial[8] || changed[8] != initial[8]) return "a changed CPU key missed its first draw";
	if (!cpu_write(address, words)) return "CPU key restore failed";
	if (consume() != initial) return "a restored CPU key missed its first draw";

	const auto produced = consume({second});
	if (produced.empty() || produced[0] != initial[8] || produced[8] != initial[8]) {
		return "a resident key written by the GPU during the preparation missed its own draw";
	}
	if (heap.LateMisses() != 0u) return "a resident key was reported missing";

	const auto novel = consume({third});
	if (novel.empty() || novel[0] != 0u || novel[8] != initial[8]) return "an unknown key did not fall back to slot zero";
	const auto served = consume();
	if (served.empty() || served[0] == 0u || served[0] == initial[0] || served[0] == initial[8] || served[8] != initial[8]) {
		return "a key missed by the in-stream fill was not served at its next draw";
	}
	if (heap.LateMisses() != 1u) return "the missing key was not counted once";

	MirrorWindowsTestAccess::DiscardGpuWrites(buffers, address, sizeof(words));
	if (!cpu_write(address, words)) return "CPU table rewrite failed";
	if (consume() != initial) return "CPU bytes equal to the last CPU copy did not replace a GPU-written key";

	const auto [mirror, offset] = buffers.ObtainRange(address, sizeof(words), false, false, ObtainFrom::Mirror);
	renderer.PrepareBda(true, false, BdaWriteTracking::PageBitmap);
	mirror->Fill(offset, sizeof(uint32_t), second);
	const auto indirect = consume();
	if (indirect.empty() || indirect[0] != initial[8]) return "a consumer missed a write by address of unknown reach";
	if (!cpu_write(address, words) || consume() != initial) return "CPU bytes did not replace a key written by address";

	const auto prepare_writer = [&](uint64_t base, bool bounded) {
		const std::array<uint32_t, 4> code {0x7e020280u, 0xdc708000u, bounded ? 1u : 0x007f0000u, 0xbf810000u};
		IR::ResourceSnapshot resources;
		resources.user_data = {static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32u)};
		ShaderPixelInputInfo pixel {};
		ShaderRecompiler::CompileOptions options;
		options.stage = ShaderType::Pixel;
		options.wave_size = 64u;
		options.dump_ir = false;
		options.user_data = resources.user_data;
		options.input_info.pixel = &pixel;
		auto translated = ShaderRecompiler::TranslateProgram(code, options);
		ShaderModule module;
		module.info = std::move(translated.program).TakeCompiledInfo();
		if (!module.info.has_address_writes || module.info.info.dma_windows.empty() == bounded) return false;
		const ShaderStageRuntime stage {&module, &resources};
		const ShaderStageRuntime* stages[] {&stage};
		renderer.PrepareShaderBda(stages, true);
		return true;
	};
	{
		const std::array<uint32_t, 8> code {0xf4000080u, 0xfa000000u, 0xbf8cc07fu, 0xbe8303ffu,
		                                    0x31016facu, 0xe0700000u, 0x80000000u, 0xbf810000u};
		const auto base = address + 0x6000u;
		renderer.MapMemory(base, 16u);
		const uint32_t count = 16u;
		if (!Libs::LibKernel::Memory::TryWriteBacking(base, &count, sizeof(count))) return "could not seed the writer size";
		buffers.InvalidateMemory(base, sizeof(count));
		const std::array<uint32_t, 4> user {static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32u), 0u, 0u};
		ShaderComputeInputInfo compute {};
		compute.wave_size = compute.host_subgroup_size = compute.threads_num[0] = 64;
		compute.threads_num[1] = compute.threads_num[2] = compute.thread_ids_num = 1;
		ShaderRecompiler::CompileOptions options;
		options.dump_ir = false;
		options.user_data = user;
		options.input_info.compute = &compute;
		auto translated = ShaderRecompiler::TranslateProgram(code, options);
		const auto plan = IR::ExtractResourcePlan(translated.program);
		IR::ResourceSnapshot resources;
		IR::ResourceSpecialization specialization;
		const IR::SrtRuntime runtime {.user_data = user, .read_specialization_memory = [](void*, uint64_t src, std::span<uint32_t> dst) {
			return Libs::LibKernel::Memory::TryReadBufferBacking(src, dst.data(), dst.size_bytes());
		}};
		if (!IR::MaterializeResources(plan, runtime, resources, specialization)) return "could not materialize the self-sized writer";
		auto compiled = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization, 0u);
		ShaderModule module;
		module.info = std::move(compiled.program).TakeCompiledInfo();
		const ShaderStageRuntime stage {&module, &resources};
		const ShaderStageRuntime* stages[] {&stage};
		if (compiled.spirv.empty() || !renderer.CacheSelfSizedBuffers(stage)) return "missing compiled write reach";
		renderer.PrepareShaderBda(stages, true);
		groups = heap.FillGroups();
		if (consume() != initial || heap.FillGroups() != groups) return "a compiled writer refilled a disjoint table";
	}
	if (!prepare_writer(address + 4u * BufferCache::CACHING_PAGESIZE, true)) return "no bound for a base-relative pixel store";
	groups = heap.FillGroups();
	if (consume() != initial || heap.FillGroups() != groups) return "a writer whose reach misses the table refilled it";
	if (!prepare_writer(address, true)) return "missing store window";
	if (consume() != initial || heap.FillGroups() == groups) return "a bounded store did not refill its watched table";
	groups = heap.FillGroups();
	if (!prepare_writer(address, false)) return "a full vector address incorrectly acquired a store window";
	if (consume() != initial || heap.FillGroups() == groups) return "an unknown store did not refill the watched tables";
	const auto collect_tick = scheduler.CurrentTick();
	buffers.ProcessWriteBuffer([](std::span<const uint64_t>) {});
	if (scheduler.CurrentTick() != collect_tick) return "the frame collection of the write bitmap waited for the GPU queue";
	{
		const auto other = address + 0x1000u;
		renderer.MapMemory(other, sizeof(words));
		if (!cpu_write(other, words)) return "second CPU table write failed";
		auto other_request = request;
		other_request.base = other;
		if (consume_at(other_request, sizeof(words), {}, 1u) != initial) return "a new CPU table missed its first draw";
		heap.RunGarbageCollector();
		heap.RunGarbageCollector();
		if (consume_at(other_request, sizeof(words), {}, 1u) != initial) return "a CPU table taken up again missed its draw";
	}
	{
		const auto other = address + 0x3000u;
		renderer.MapMemory(other, sizeof(words));
		if (!cpu_write(other, empty_table)) return "could not clear the unwatched backing";
		const auto [device, at] = buffers.ObtainRange(other, sizeof(words), false, false, ObtainFrom::Mirror);
		renderer.PrepareBda(true, false, BdaWriteTracking::PageBitmap);
		const auto upload_table = [&](uint64_t start, size_t bytes) {
			const auto [source, source_offset] = buffers.UploadHeapData(bytes, [&](uint8_t* dest, uint64_t) {
				std::memcpy(dest, words.data(), bytes);
			});
			device->CopyFrom(scheduler.Current(), *source, source_offset, at + start, bytes);
		};
		upload_table(0, sizeof(words));
		auto unseen = request;
		unseen.base = other;
		unseen.step = 48u;
		unseen.count = 1u;
		const auto resident = consume_at(unseen, sizeof(uint32_t), {}, 0u);
		if (resident.empty() || resident.front() != 0u) return "a table the CPU cannot read armed a kind before its first draw";
		const auto known = consume_at(unseen, sizeof(uint32_t), {}, 1u);
		if (known.empty() || known.front() != initial[0]) return "a key reported by the in-stream fill did not arm its shape";
		auto sized = request;
		sized.base = other;
		if (!cpu_write(other, empty_table)) return "could not reset the resize table";
		heap.BeginDraw();
		IR::ImageHeapView view;
		if (!heap.Acquire(sized, view)) return "could not read the resize table";
		heap.BeginDraw();
		sized.size = 32u;
		if (!heap.Acquire(sized, view)) return "could not shrink the resize table";
		renderer.PrepareBda(true, false, BdaWriteTracking::PageBitmap);
		words[0] = 0x60000u;
		upload_table(32u, 32u);
		words[0] = first;
		sized.size = sizeof(words);
		const auto late = heap.LateMisses();
		const auto grown = consume_at(sized, sizeof(words), {}, 1u);
		if (grown.empty() || grown[0] != 0u || grown[8] != 0u) return "an unknown key in a regrown table did not fall back to slot zero";
		const auto resized = consume_at(sized, sizeof(words), {}, 1u);
		if (resized.empty() || resized[0] != 0u || resized[8] == 0u || resized[8] == initial[0] || resized[8] == initial[8] ||
		    heap.LateMisses() != late + 1u) {
			return "the key of a regrown table was not served at its next draw";
		}
	}
	const auto bytes = buffers.GetUtilityBuffer(MemoryUsage::Stream).Size() + sizeof(uint32_t);
	const auto tick  = scheduler.CurrentTick();
	const auto [large, start] = buffers.UploadHeapData(bytes, [&](uint8_t* mapped, uint64_t) {
		std::memcpy(mapped + bytes - sizeof(second), &second, sizeof(second));
	});
	reader.Record(scheduler, large->BufferDeviceAddress() + start + bytes - sizeof(second), sizeof(second));
	const bool same_tick = scheduler.CurrentTick() == tick;
	Finish(scheduler);
	const bool large_ok = same_tick && reader.Words(sizeof(second)).front() == second;
	graphics.device.destroyPipelineLayout(layout);
	graphics.device.destroyDescriptorSetLayout(empty);
	if (!large_ok) return "a heap upload larger than the stream lost its tick or its last word";
	if (waited) return "an image heap draw advanced the GPU queue tick";
	(void)fourth;
	return nullptr;
}

const char* CheckWindows(GraphicContext& graphics, uint64_t address) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& buffers   = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	auto* mirror = buffers.GetGuestMirror();
	if (mirror == nullptr) return "the cache has no mirror";
	if (!mirror->Holds(address, GuestBytes)) {
		return "the high guest mapping was refused";
	}
	if (!renderer.MapMemory(address, GuestBytes)) return "could not register high windows";

	auto* guest = reinterpret_cast<uint32_t*>(address);
	for (uint64_t index = 0; index < GuestBytes / sizeof(uint32_t); index++) guest[index] = Pattern(index);

	if (buffers.IsRegionRegistered(address, GuestBytes)) return "the mirror holds blocks before any range was obtained";
	const auto [range, offset] = buffers.ObtainRange(address, FirstBytes, false);
	if (range->Handle() != mirror->Handle()) return "a range is not served from the mirror";
	if (offset >= mirror->Size() || mirror->Address(offset) != address) return "a mirror offset does not resolve to its high guest address";
	if (range->BufferDeviceAddress() + offset != mirror->DeviceAddress() + mirror->Offset(address)) {
		return "a range's device address is not the mirror's plus its guest address";
	}
	if (!mirror->Requested(address, FirstBytes)) return "a range's blocks are not requested when it is obtained";
	if (mirror->Ready(address, FirstBytes)) return "a range's blocks are bound before its submission";
	if (!buffers.IsRegionRegistered(address, 1)) return "the cache does not report the pages of an obtained range";
	if (!mirror->Owned(address, FirstBytes)) return "a range does not own its pages";
	const auto page  = uint64_t {1} << GuestMirror::PageBits;
	const auto aside = Common::AlignUp(address + FirstBytes, page) + page;
	if (mirror->Requested(aside, WrittenBytes)) {
		if (mirror->AnyOwned(aside, WrittenBytes)) return "a range owns pages of its block that it does not reach";
		if (buffers.IsRegionRegistered(aside, WrittenBytes)) return "the cache reports a page that is only bound with its block";
	} else {
		std::printf("%s: the first range ends its block, the pages bound beside a range are not checked\n", Name);
	}

	AddressReader reader(graphics, scheduler, JoinedBytes);
	Buffer bitmap(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, sizeof(uint32_t));
	const auto resident = [&](uint64_t guest) { return IsResident(scheduler, buffers, bitmap, guest); };
	reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(address), FirstBytes);
	Finish(scheduler);
	if (!mirror->Ready(address, FirstBytes)) return "a submission went without binding its blocks";
	if (!mirror->SharedQueue() && mirror->Totals().waits == 0) return "the submission did not wait for its bind";
	if (!ReadsPattern(reader.Words(FirstBytes), 0)) {
		return "a read by device address did not see the bytes uploaded into the mirror";
	}
	if (!resident(address) || resident(address + FarAt)) return "the shader residency map disagrees with the sparse binds";
	if (mirror->Requested(aside, WrittenBytes)) {
		buffers.SynchronizeBuffersInRange(aside, WrittenBytes);
		reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(aside), WrittenBytes);
		Finish(scheduler);
		if (!ReadsPattern(reader.Words(WrittenBytes), (aside - address) / sizeof(uint32_t))) {
			return "a page bound beside a range is not brought up to date for a read by device address";
		}
		if (mirror->AnyOwned(aside, WrittenBytes)) return "a page brought up to date beside a range is owned by it";
		buffers.NoteGpuWrites(aside, WrittenBytes);
		if (!buffers.IsRegionGpuModified(aside, WrittenBytes)) {
			return "a GPU write noted on a page a shader can reach by device address does not mark it";
		}
	}

	const auto [joined, joined_offset] = buffers.ObtainRange(address, JoinedBytes, false);
	if (joined->Handle() != mirror->Handle() || joined_offset != mirror->Offset(address)) {
		return "a wider range over an obtained one is not the same place in the mirror";
	}
	reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(address), JoinedBytes);
	Finish(scheduler);
	if (!ReadsPattern(reader.Words(JoinedBytes), 0)) return "a wider range lost or moved bytes";

	const auto [streamed, streamed_offset] = buffers.ObtainRange(address + StreamedAt, 0x40, false);
	const auto [mirrored, mirrored_offset] =
	    buffers.ObtainRange(address + MirroredAt, 0x40, false, false, ObtainFrom::Mirror);
	if (streamed->Handle() == mirror->Handle()) return "a small read is no longer served by a stream copy";
	if (mirrored->Handle() != mirror->Handle() || mirrored_offset != mirror->Offset(address + MirroredAt)) {
		return "ObtainFrom::Mirror is served by a stream copy instead of the mirror";
	}
	(void)streamed_offset;
	reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(address + MirroredAt), 0x40);
	Finish(scheduler);
	if (!ReadsPattern(reader.Words(0x40), MirroredAt / sizeof(uint32_t))) {
		return "a small range asked from the mirror does not read the guest bytes";
	}

	const std::pair<uint64_t, uint64_t>        table {address + WrittenAt + TableAt, TableBytes};
	const std::pair<uint64_t, uint64_t>        table_range {table.first, table.first + table.second};
	std::vector<std::pair<uint64_t, uint64_t>> taken;
	const auto                                 take = [&] {
		taken.clear();
		buffers.TakeHeapTableWrites([&](uint64_t begin, uint64_t end) { taken.emplace_back(begin, end); });
	};
	buffers.WatchHeapTables(std::span(&table, 1));
	take();
	if (!taken.empty()) return "a heap table reports a GPU write before any was made";

	buffers.NoteDeviceAddressUse();
	take();
	if (!taken.empty()) return "an indirect address write reaches the heap before its draw is recorded";
	buffers.BeginHeapDraw();
	take();
	if (taken.size() != 1 || taken.front() != table_range) {
		return "an indirect address write does not reach the heap at the next draw";
	}

	const auto [written, written_offset] = buffers.ObtainRange(address + WrittenAt, WrittenBytes, true);
	if (written->Handle() != mirror->Handle()) return "a written range is not served from the mirror";
	written->Fill(written_offset, WrittenBytes, GpuValue);
	(void)buffers.CommitWriteTicks();
	Finish(scheduler);
	if (BackingHolds(address + WrittenAt, WrittenBytes, GpuValue)) return "the GPU write reached guest memory by itself";
	if (graphics.readback_queue) {
		if (!MirrorWindowsTestAccess::ReadBack(buffers, address + WrittenAt, WrittenBytes)) {
			return "nothing was read back from the mirror on the readback queue";
		}
		if (!BackingHolds(address + WrittenAt, WrittenBytes, GpuValue)) {
			return "the readback from the mirror lost the GPU write";
		}
	} else {
		std::printf("%s: no readback queue on this device, its readback is not checked\n", Name);
	}
	MirrorWindowsTestAccess::DiscardGpuWrites(buffers, address + WrittenAt, WrittenBytes);
	take();
	if (taken.size() != 1 || taken.front() != table_range) {
		return "a GPU write into a heap table is lost once its bytes are read back or discarded";
	}
	take();
	if (!taken.empty()) return "a GPU write into a heap table is reported twice";

	const auto [back, back_offset] = buffers.ObtainRange(address + NextAt, FirstBytes, false);
	if (back->Handle() != mirror->Handle() || back_offset != mirror->Offset(address + NextAt)) {
		return "a range next to a written one is not served from the mirror";
	}
	reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(address + NextAt), FirstBytes);
	Finish(scheduler);
	if (!ReadsPattern(reader.Words(FirstBytes), NextAt / sizeof(uint32_t))) {
		return "a range next to a GPU-written one does not read the bytes of the guest memory";
	}
	if (mirror->Totals().unbound_reads != 0) return "a read of a block that was not bound was counted";

	(void)MemoryTracker::TakeWrittenUploadStats();
	(void)buffers.ObtainRange(address + WrittenAt, WrittenBytes, true);
	(void)buffers.ObtainRange(address + WrittenAt, WrittenBytes, true);
	if (MemoryTracker::TakeWrittenUploadStats().uploads != 1) {
		return "a written range is passed again although no page of it changed";
	}
	take();
	if (taken.size() != 1 || taken.front() != table_range) {
		return "a heap table misses a GPU write of a range written before";
	}
	buffers.WatchHeapTables({});
	MirrorWindowsTestAccess::DiscardGpuWrites(buffers, address + WrittenAt, WrittenBytes);
	(void)buffers.ObtainRange(address + WrittenAt, WrittenBytes, true);
	if (MemoryTracker::TakeWrittenUploadStats().uploads != 1) {
		return "a written range is not passed again after its pages changed";
	}
	take();
	if (!taken.empty()) return "a GPU write is reported for a table that is no longer watched";
	MirrorWindowsTestAccess::DiscardGpuWrites(buffers, address + WrittenAt, WrittenBytes);

	if (mirror->AnyRequested(address + FarAt, WrittenBytes)) return "a block no range reached is requested";
	buffers.NoteGpuWrites(address + FarAt, WrittenBytes);
	if (buffers.IsRegionGpuModified(address + FarAt, WrittenBytes)) {
		return "a GPU write noted outside the requested blocks marks pages the mirror does not hold";
	}
	buffers.RequestDeviceRange(address + FarAt, WrittenBytes);
	if (!resident(address + FarAt)) return "a range read by device address is missing from the shader residency map";
	if (!mirror->Requested(address + FarAt, WrittenBytes)) return "a range read by device address is not requested";
	if (!mirror->Owned(address + FarAt, WrittenBytes)) return "a range read by device address is not owned";

	const char* check = std::getenv("KYTY_MIRROR_CHECK");
	if (check == nullptr || std::strcmp(check, "1") != 0) {
		Finish(scheduler);
		mirror->WaitIdle();
		GuestMirror::Runs released;
		mirror->ForEachRequested(address + NextAt, WrittenBytes,
		                         [&](uint64_t begin, uint64_t end) { released.emplace_back(begin, end); });
		mirror->Release(released);
		mirror->WaitIdle();
		MirrorWindowsTestAccess::CheckRange(buffers, address + NextAt, WrittenBytes);
		if (mirror->Totals().unbound_reads != 1) return "a range whose blocks are not bound is not counted";
	}

	if (mirror->BlockSize() == GrowBlock) {
		const auto grow = address + GrowAt;
		(void)buffers.ObtainRange(grow, GrowBlock / 2, false);
		(void)buffers.ObtainRange(grow, GrowBlock + GrowBlock / 2, false);
		if (mirror->AnyRequested(grow + 2 * GrowBlock, GrowBlock)) return "a range that grew once binds ahead of itself";
		(void)buffers.ObtainRange(grow, 2 * GrowBlock + GrowBlock / 2, false);
		if (!mirror->Requested(grow + 3 * GrowBlock, LeapBytes)) {
			return "a range that grew again at the same edge does not bind ahead of itself";
		}
		if (mirror->AnyOwned(grow + 4 * GrowBlock, LeapBytes - GrowBlock)) {
			return "blocks bound ahead of a range are owned before a range reaches them";
		}
	} else {
		std::printf("%s: blocks of %llu KiB, the leap ahead of a growing range is not checked\n", Name,
		            static_cast<unsigned long long>(mirror->BlockSize() >> 10u));
	}

	const auto* failure = CheckHeapConsumer(renderer, address + MirroredAt, reader);
	Finish(scheduler);
	MirrorWindowsTestAccess::DiscardGpuWrites(buffers, address, GuestBytes);
	(void)buffers.DropFreedRange(address, GuestBytes);
	return failure;
}

const char* CheckCapacity(GraphicContext& graphics, uint64_t address) {
	const auto saved_size = graphics.mirror_size;
	graphics.mirror_size = 32 * GuestMirror::WindowSize;
	RenderContext renderer(graphics);
	graphics.mirror_size = saved_size;
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	auto& cache = renderer.GetBufferCache();
	auto& mirror = *cache.GetGuestMirror();
	const auto page = BufferCache::CACHING_PAGESIZE;
	const auto extended = Libs::LibKernel::Memory::kExtendedMemoryBase;
	Libs::LibKernel::Memory::InstallGpuResources(&renderer);
	struct Mappings {
		std::vector<uint64_t> live;
		~Mappings() {
			for (const auto base: live) sceKernelMunmap(base, BufferCache::CACHING_PAGESIZE);
			Libs::LibKernel::Memory::InstallGpuResources(nullptr);
		}
	} mappings;
	if (!renderer.MapMemory(extended, page)) return "could not translate an extended guest address";
	const auto keep = mirror.Offset(extended);
	if (!renderer.AdmitMemory(extended + page, page)) return "could not admit a page beside a live mapping";
	renderer.CancelMemoryAdmission(extended + page, page);
	if (mirror.Offset(extended) != keep) return "cancelling a neighbouring page lost a live mapping";
	const auto boundary = 880 * GuestMirror::GiB;
	void* misaligned = reinterpret_cast<void*>(boundary - 1);
	try {
		sceKernelMapFlexibleMemory(&misaligned, page, CpuReadWrite, 0x90);
		return "the kernel accepted an unaligned fixed placement";
	} catch (const std::invalid_argument&) {
		if (mirror.Offset(boundary - 1) != UINT64_MAX || mirror.Offset(boundary) != UINT64_MAX)
			return "failed host placement retained a provisional window";
	}
	uint64_t refused = 0;
	for (uint64_t n = 0; n <= 32; ++n) {
		const auto base = 900 * GuestMirror::GiB + n * GuestMirror::WindowSize;
		void* at = reinterpret_cast<void*>(base);
		const int result = sceKernelMapFlexibleMemory(&at, page, CpuReadWrite, 0x90);
		if (result == SCE_KERNEL_ERROR_ENOMEM) {
			refused = base;
			break;
		}
		if (result != 0 || reinterpret_cast<uint64_t>(at) != base) return "could not allocate capacity fixture";
		mappings.live.push_back(base);
	}
	if (refused == 0 || mappings.live.empty() || renderer.IsMapped(refused, page) ||
	    mirror.Offset(refused) != UINT64_MAX || mirror.Offset(extended) != keep) return "capacity refusal changed live mappings";
	const auto base = mappings.live.back();
	const auto slot = mirror.Offset(base);
	cache.RequestDeviceRange(base, page);
	Finish(scheduler);
	if (sceKernelMunmap(base, page) != 0) return "could not free the bound test window";
	mappings.live.pop_back();
	void* retry = reinterpret_cast<void*>(refused);
	if (sceKernelMapFlexibleMemory(&retry, page, CpuReadWrite, 0x90) != 0) return "could not retry a refused mapping";
	mappings.live.push_back(refused);
	if (mirror.Offset(base) != UINT64_MAX || mirror.Offset(refused) != slot || mirror.Offset(extended) != keep ||
	    mirror.Offset(address) == UINT64_MAX) return "reusing a mirror slot changed another mapping";
	renderer.UnmapMemory(extended, page);
	Finish(scheduler);
	return nullptr;
}

const char* CheckDispatch(GraphicContext& graphics, uint64_t address) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	auto* mirror = renderer.GetBufferCache().GetGuestMirror();
	const auto block = mirror->BlockSize();
	const auto window = Common::AlignUp(address + block, GuestMirror::WindowSize);
	const auto boundary = window + block;
	const auto end = address + GuestBytes;
	if (window - block < address || boundary + 3 * block > end) {
		g_skipped = "the dispatch fixture needs four sparse blocks after a mirror window boundary";
		return nullptr;
	}

	auto* guest = reinterpret_cast<uint32_t*>(address);
	for (uint64_t i = 0; i < GuestBytes / sizeof(uint32_t); ++i) guest[i] = Pattern(i);
	renderer.MapMemory(address, boundary + 2 * block - address);
	ShaderModule module {};
	module.info.stage = ShaderType::Compute;
	module.info.info.uses_dma = true;
	module.info.info.dma_bounded = true;
	module.info.info.dma_windows.push_back({0, 0, 8});
	const auto first = boundary - sizeof(uint32_t);
	ShaderRecompiler::IR::ResourceSnapshot resources;
	resources.user_data = {static_cast<uint32_t>(first), static_cast<uint32_t>(first >> 32)};
	const ShaderStageRuntime stage {&module, &resources};
	const ShaderStageRuntime* dispatch = &stage;
	AddressReader reader(graphics, scheduler, 8);
	Buffer bitmap(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, sizeof(uint32_t));
	auto& buffers = renderer.GetBufferCache();
	const auto prepare = [&]() {
		const auto synced = renderer.SynchronizeDmaFootprint(std::span {&dispatch, 1});
		renderer.PrepareBda(false, !synced);
		return synced;
	};
	const auto reads = [&](uint64_t source) {
		if (!prepare() || !mirror->Requested(source, 8)) return false;
		reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(source), 8);
		Finish(scheduler);
		return ReadsPattern(reader.Words(8), (source - address) / sizeof(uint32_t)) &&
		       mirror->Ready(source, 8) && IsResident(scheduler, buffers, bitmap, source) &&
		       IsResident(scheduler, buffers, bitmap, source + 7);
	};
	const auto leaves = [&](uint64_t source) {
		mirror->WaitIdle();
		const auto bound = mirror->Totals().bound_blocks;
		if (prepare()) return false;
		Finish(scheduler);
		mirror->WaitIdle();
		return !mirror->Requested(source, 8) && mirror->Totals().bound_blocks == bound &&
		       !IsResident(scheduler, buffers, bitmap, source);
	};
	const auto* view     = buffers.GetGuestHostView();
	const auto  imported = [&](uint64_t source) { return view != nullptr && view->Lookup(source) != 0; };
	const auto  binds    = [&](uint64_t source) {
		if (prepare() || !mirror->Requested(source, 8)) return false;
		reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(source), 8);
		Finish(scheduler);
		return ReadsPattern(reader.Words(8), (source - address) / sizeof(uint32_t)) && IsResident(scheduler, buffers, bitmap, source);
	};
	const auto verify = [&]() -> const char* {
		if (!reads(first)) return "the first dispatch did not read both sides of its DMA window";
		const auto unknown = boundary + block;
		if (mirror->Requested(unknown, 8)) return "a bounded dispatch requested an unrelated block";
		module.info.info.dma_bounded = false;
		module.info.info.dma_windows.clear();
		const auto added = boundary + 2 * block;
		if (!imported(unknown)) {
			if (!binds(unknown)) return "a dispatch with no known reach left mapped memory without an import out of the mirror";
			if (mirror->Requested(added, 8)) return "the dispatch requested an unmapped block";
			renderer.MapMemory(added, block);
			if (!binds(added)) return "a dispatch with no known reach left a newly mapped block without an import out of the mirror";
			return nullptr;
		}
		if (!leaves(unknown)) return "a dispatch with no known reach bound imported memory it cannot reach";
		if (mirror->Requested(added, 8)) return "the dispatch requested an unmapped block";
		renderer.MapMemory(added, block);
		if (!leaves(added)) return "a dispatch with no known reach bound a newly mapped block that is imported";
		buffers.ForgetGuestMemory(unknown, block);
		if (imported(unknown) || !imported(added)) return "dropping the import of one block touched its neighbours";
		if (!binds(unknown)) return "a dispatch with no known reach did not bind a mapped block that lost its import";
		if (mirror->Requested(added, 8)) return "binding a block without an import bound an imported one";
		return nullptr;
	};
	const auto* failure = verify();
	renderer.UnmapMemory(address, GuestBytes);
	return failure;
}

const char* CheckTableReach(GraphicContext& graphics, uint64_t address, bool at_end) {
	using Reach = ShaderRecompiler::Decoder::RecordReach;
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	auto&       buffers = renderer.GetBufferCache();
	auto*       mirror  = buffers.GetGuestMirror();
	const auto* view    = buffers.GetGuestHostView();
	const auto  block   = mirror->BlockSize();
	const auto  tables  = Common::AlignUp(address, block);
	const auto  bare    = tables + 2 * block;
	const auto  kept    = at_end ? tables : tables + 4 * block;
	const auto  first   = at_end ? bare + block - sizeof(uint32_t) : bare;
	if (tables + 5 * block > address + GuestBytes) {
		g_unchecked = "the fixture is too small for the blocks of a reach read from a table";
		return nullptr;
	}
	if (at_end && (first + sizeof(uint32_t) - mirror->Base()) % GuestMirror::WindowSize != 0) return "fixture misses the window boundary";
	auto* guest = reinterpret_cast<uint32_t*>(address);
	for (uint64_t i = 0; i < GuestBytes / sizeof(uint32_t); ++i) guest[i] = Pattern(i);
	const auto vsharp = [](uint64_t at, uint64_t base, uint32_t stride, uint32_t records) {
		const std::array<uint32_t, 4> words {static_cast<uint32_t>(base), static_cast<uint32_t>((base >> 32u) & 0xffffu) | (stride << 16u),
		                                     records, 0u};
		std::memcpy(reinterpret_cast<void*>(at), words.data(), sizeof(words));
	};
	constexpr uint32_t RecordBytes = 16;
	const auto         header      = tables + 0x100;
	const auto         records     = tables + 0x200;
	const auto         pointers    = tables + 0x300;
	vsharp(tables, header, 4, 2);
	vsharp(tables + 0x10, records, RecordBytes, 2);
	vsharp(tables + 0x20, pointers, 16, 2);
	const std::array<uint32_t, 2> range {0u, 2u};
	std::memcpy(reinterpret_cast<void*>(header), range.data(), sizeof(range));
	const std::array<uint32_t, 8> rows {1u, 0u, 4u, 0u, 1u, 0u, 4u, 1u};
	std::memcpy(reinterpret_cast<void*>(records), rows.data(), sizeof(rows));
	vsharp(pointers, first, 4, 2);
	vsharp(pointers + 16, at_end ? first : kept, 4, 2);
	Reach reach;
	reach.header        = {0, 0};
	reach.records       = {2, 0};
	reach.pointers      = {4, 0};
	reach.record_size   = RecordBytes;
	reach.threads_field = 0;
	reach.first_field   = 4;
	reach.stride_field  = 8;
	reach.pointer_field = 12;
	reach.attributes.push_back({Reach::None, 4, 8});
	ShaderModule module {};
	module.info.stage                 = ShaderType::Compute;
	module.info.info.uses_dma         = true;
	module.info.info.dma_bounded      = false;
	module.info.info.dma_record_reach = reach.Pack();
	ShaderRecompiler::IR::ResourceSnapshot resources;
	resources.user_data = {static_cast<uint32_t>(tables),        static_cast<uint32_t>(tables >> 32u),
	                       static_cast<uint32_t>(tables + 0x10), static_cast<uint32_t>((tables + 0x10) >> 32u),
	                       static_cast<uint32_t>(tables + 0x20), static_cast<uint32_t>((tables + 0x20) >> 32u)};
	const ShaderStageRuntime  stage {&module, &resources};
	const ShaderStageRuntime* dispatch = &stage;
	renderer.MapMemory(address, GuestBytes);
	renderer.PrepareBda(false, false);
	AddressReader reader(graphics, scheduler, 8);
	Buffer        bitmap(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, sizeof(uint32_t));
	const auto    verify = [&]() -> const char* {
		buffers.ForgetGuestMemory(bare, at_end ? 2 * block : block);
		const bool imports = view != nullptr && view->Lookup(kept) != 0;
		if (view != nullptr && view->Lookup(bare) != 0) return "a block kept its import after it was dropped";
		if (mirror->AnyRequested(bare, block) || (!at_end && mirror->AnyRequested(kept, block))) {
			return "the blocks of the table reach are bound before the dispatch";
		}
		if (!renderer.SynchronizeDmaFootprint(std::span {&dispatch, 1})) return "the reach read from a table was not found";
		renderer.PrepareBda(false, false);
		const uint64_t bytes = 8;
		if (!mirror->Requested(first, bytes)) {
			return at_end ? "a reach that crosses a window boundary lost a word"
			              : "a reach read from a table left a mapped block without an import out of the mirror";
		}
		if (!at_end && imports && mirror->AnyRequested(kept, block)) return "a reach read from a table bound an imported block";
		reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(first), bytes);
		Finish(scheduler);
		if (!ReadsPattern(reader.Words(bytes), (first - address) / sizeof(uint32_t)) || !IsResident(scheduler, buffers, bitmap, first)) {
			return "a block bound for a reach read from a table does not hold its guest memory";
		}
		return nullptr;
	};
	const auto* failure = verify();
	Finish(scheduler);
	renderer.UnmapMemory(address, GuestBytes);
	return failure;
}

const char* CheckCut(GraphicContext& graphics, uint64_t address) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	auto&       buffers = renderer.GetBufferCache();
	auto*       mirror  = buffers.GetGuestMirror();
	const auto* view    = buffers.GetGuestHostView();
	if (view == nullptr || !view->Enabled()) {
		g_unchecked = "the device does not import host memory";
		return nullptr;
	}
	const auto block  = mirror->BlockSize();
	const auto stored = Common::AlignUp(address + block, block);
	const auto read   = stored + 2 * block;
	if (read + block > address + GuestBytes) {
		g_unchecked = "the fixture is too small for the blocks of the cut";
		return nullptr;
	}
	auto* guest = reinterpret_cast<uint32_t*>(address);
	for (uint64_t i = 0; i < GuestBytes / sizeof(uint32_t); ++i) guest[i] = Pattern(i);
	renderer.MapMemory(address, GuestBytes);
	AddressReader reader(graphics, scheduler, 16);
	const auto    verify = [&]() -> const char* {
		renderer.PrepareBda(true, false);
		if (view->Lookup(stored) == 0 || view->Lookup(read) == 0) {
			g_unchecked = "the driver did not import the guest memory of the fixture";
			return nullptr;
		}
		Finish(scheduler);
		const std::array<uint32_t, 4> fresh {GpuValue, GpuValue + 1, GpuValue + 2, GpuValue + 3};
		reader.Store(scheduler, view->Lookup(stored), fresh.data(), sizeof(fresh));
		(void)buffers.ObtainRange(stored, sizeof(fresh), false, false, ObtainFrom::Mirror);
		reader.Record(scheduler, mirror->DeviceAddress() + mirror->Offset(stored), sizeof(fresh));
		Finish(scheduler);
		const auto words = reader.Words(sizeof(fresh));
		if (std::memcmp(words.data(), fresh.data(), sizeof(fresh)) != 0) {
			return "a block bound after a cut submission missed the store recorded behind the cut";
		}
		if (std::memcmp(reinterpret_cast<const void*>(stored), fresh.data(), sizeof(fresh)) != 0) {
			return "a store through an import did not reach guest memory";
		}
		renderer.PrepareBda(false, false);
		Finish(scheduler);
		reader.Record(scheduler, view->Lookup(read), 16);
		const auto waits = view->Totals().waits;
		renderer.UnmapMemory(read, block);
		if (view->Totals().waits != waits + 1) return "an unmap behind a cut did not wait for the submission that reads its import";
		if (view->Lookup(read) != 0) return "an unmap behind a cut left its import";
		if (!ReadsPattern(reader.Words(16), (read - address) / sizeof(uint32_t))) return "a read recorded before an unmap did not finish first";
		return nullptr;
	};
	const auto* failure = verify();
	Finish(scheduler);
	MirrorWindowsTestAccess::DiscardGpuWrites(buffers, address, GuestBytes);
	renderer.UnmapMemory(address, GuestBytes);
	return failure;
}

const char* CheckRemap(GraphicContext& graphics, uint64_t address) {
	constexpr uint64_t Page  = GuestHostView::PageSize;
	constexpr uint64_t Bytes = 4 * Page;
	constexpr uint64_t Half  = 2 * Page;
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	const auto* view = renderer.GetBufferCache().GetGuestHostView();
	if (view == nullptr || !view->Enabled()) {
		g_unchecked = "the device does not import host memory";
		return nullptr;
	}
	auto* guest = reinterpret_cast<uint32_t*>(address);
	for (uint64_t i = 0; i < GuestBytes / sizeof(uint32_t); ++i) guest[i] = Pattern(i);
	renderer.MapMemory(address, Bytes);
	renderer.PrepareBda(false, false);
	AddressReader reader(graphics, scheduler, 16);
	const auto    qword = [&](uint64_t device) {
		reader.Record(scheduler, device, 8);
		Finish(scheduler);
		const auto words = reader.Words(8);
		return words[0] | (uint64_t {words[1]} << 32u);
	};
	const auto entry = [&](uint64_t guest_address) -> uint64_t {
		const auto offset = guest_address - renderer.GetBufferCache().GetGuestMirror()->Base();
		const auto chunk  = qword(view->TableAddress() + (offset >> GuestHostView::ChunkBits) * sizeof(uint64_t));
		if (chunk == 0) return 0;
		return qword(chunk + ((offset >> GuestHostView::PageBits) & (GuestHostView::EntriesPerChunk - 1)) * sizeof(uint64_t));
	};
	const auto reads = [&](uint64_t guest_address) {
		const auto page = entry(guest_address & ~(Page - 1));
		if (page == 0 || page != view->Lookup(guest_address & ~(Page - 1))) return false;
		reader.Record(scheduler, page + (guest_address & (Page - 1)), 8);
		Finish(scheduler);
		const auto words = reader.Words(8);
		return std::memcmp(words.data(), reinterpret_cast<const void*>(guest_address), 8) == 0;
	};
	const auto verify = [&]() -> const char* {
		if (view->Lookup(address) == 0 || view->Lookup(address + Half) == 0) {
			g_unchecked = "the driver did not import the guest memory of the fixture";
			return nullptr;
		}
		if (!reads(address) || !reads(address + Half) || !reads(address + Bytes - 8)) {
			return "the table of guest pages does not lead to the guest memory mapped first";
		}
		if (entry(address + Bytes) != 0) return "the table holds a page beyond the mapping";
		const auto mapped = view->Totals();
		renderer.PrepareBda(false, false);
		renderer.UnmapMemory(address, Half);
		const auto parted = view->Totals();
		if (parted.waits != mapped.waits + 1) return "an unmap did not wait for the submission that could reach its import";
		if (parted.bytes != Half) return "an unmap of half an import did not leave the other half imported";
		if (entry(address) != 0 || entry(address + Page) != 0 || view->Lookup(address) != 0) {
			return "the table still leads to unmapped guest pages";
		}
		if (!reads(address + Half) || !reads(address + Bytes - 8)) return "the pages beside an unmapped range lost their guest memory";
		for (uint64_t i = 0; i < Half / sizeof(uint32_t); ++i) guest[i] = ~Pattern(i);
		renderer.MapMemory(address, Half);
		renderer.PrepareBda(false, false);
		const auto again = view->Totals();
		if (again.bytes != Bytes || again.imports != parted.imports + 1 || again.refused != mapped.refused) {
			return "a range mapped again beside its old neighbour is not imported once";
		}
		if (!reads(address) || !reads(address + Half - 8)) return "a range mapped again does not read its new guest memory";
		if (!reads(address + Half)) return "mapping a range again lost the guest memory beside it";
		renderer.UnmapMemory(address, Half);
		if (view->Totals().imports != parted.imports || view->Totals().bytes != Half) return "an unmap released more than its own import";
		if (entry(address) != 0 || !reads(address + Half) || !reads(address + Bytes - 8)) {
			return "the second unmap of a range touched the pages beside it";
		}
		renderer.MapMemory(address, Half);
		renderer.MapMemory(address, Half);
		renderer.PrepareBda(false, false);
		if (view->Totals().bytes != Bytes || !reads(address) || !reads(address + Half)) return "a range mapped twice is not imported once";
		return nullptr;
	};
	const auto* failure = verify();
	renderer.UnmapMemory(address, GuestBytes);
	if (failure == nullptr && (view->Totals().imports != 0 || view->Totals().bytes != 0)) return "an unmap of everything left an import";
	return failure;
}

}

extern "C" int RunKytyMirrorWindowsTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	auto&                           graphics = gpu.graphics;
	if (const auto* unavailable = gpu.Initialize(false, true)) {
		std::printf("%s: skipped, %s\n", Name, unavailable);
		return 77;
	}
	if (!graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", Name);
		return 1;
	}
	constexpr uint64_t Fixture = 960 * GuestMirror::GiB - 3 * BufferCache::MIRROR_BLOCK_SIZE;
	void* guest = reinterpret_cast<void*>(Fixture);
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0x90) != 0 || reinterpret_cast<uint64_t>(guest) != Fixture) {
		std::fprintf(stderr, "%s: FAILED: could not map guest backing\n", Name);
		return 1;
	}
	const auto  address = reinterpret_cast<uint64_t>(guest);
	const auto* failure = CheckWindows(graphics, address);
	if (failure == nullptr && g_skipped.empty()) failure = CheckCapacity(graphics, address);
	if (failure == nullptr && g_skipped.empty()) failure = CheckDispatch(graphics, address);
	if (failure == nullptr && g_skipped.empty()) failure = CheckRemap(graphics, address);
	if (failure == nullptr && g_skipped.empty()) failure = CheckCut(graphics, address);
	if (failure == nullptr && g_skipped.empty()) failure = CheckTableReach(graphics, address, false);
	if (failure == nullptr && g_skipped.empty()) failure = CheckTableReach(graphics, address, true);
	if (sceKernelMunmap(address, GuestBytes) != 0) {
		std::fprintf(stderr, "%s: FAILED: could not unmap guest backing\n", Name);
		return 1;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "%s: FAILED: %s\n", Name, failure);
		return 1;
	}
	if (!g_skipped.empty()) {
		std::printf("%s: skipped, %s\n", Name, g_skipped.c_str());
		return 77;
	}
	if (!g_unchecked.empty()) std::printf("%s: %s, imports of guest memory not checked\n", Name, g_unchecked.c_str());
	std::printf("%s: ok\n", Name);
	return 0;
}

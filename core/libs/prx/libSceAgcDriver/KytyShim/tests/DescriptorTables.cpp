#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "RenderExecutorTestAccess.h"
#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct ImageHeapTestAccess {
	static constexpr size_t TableLimit = ShaderRecompiler::IR::ImageHeapCapacity * 7u / 8u;

	static void Fill(ImageHeap& heap) {
		const auto occupied = heap.m_elements.size();
		heap.m_elements.resize(TableLimit);
		heap.m_uses.resize(TableLimit);
		for (size_t i = occupied; i < TableLimit; ++i) heap.m_elements[i].refs = 1;
	}

	static void Release(ImageHeap& heap, uint32_t index) {
		heap.Release(index);
		heap.Retire();
	}

	static size_t Pending(const ImageHeap& heap) { return heap.m_pending.size(); }
	static size_t Occupied(const ImageHeap& heap) { return heap.m_elements.size() - heap.m_free.size(); }
	static size_t TableSlotsFree(const ImageHeap& heap) { return TableLimit - std::min(TableLimit, Occupied(heap)); }
	static size_t ViewBytes() { return sizeof(ImageHeap::View); }

	static uint32_t LiveSlot(const ImageHeap& heap, const ShaderRecompiler::IR::DescriptorValue& value) {
		for (const auto& [key, slot]: heap.m_index) {
			if (key.value == value) return slot;
		}
		return 0u;
	}

	static size_t Buffers(const ImageHeap& heap) {
		size_t count = 0;
		for (const auto& memory: heap.m_memories) {
			for (const auto& view: memory->views) count += view->buffer != nullptr;
		}
		return count;
	}

	static const Buffer* ViewBuffer(const ImageHeap& heap, const ShaderRecompiler::IR::ImageHeapRequest& request) {
		const auto memory = std::ranges::find_if(heap.m_memories, [&](const auto& item) { return item->base == request.base; });
		if (memory == heap.m_memories.end()) return nullptr;
		const auto view = std::ranges::find_if((*memory)->views, [&](const auto& item) {
			return item->shape.first == request.first && item->shape.step == request.step &&
			       heap.m_profiles[item->shape.profile] == *request.root;
		});
		return view == (*memory)->views.end() ? nullptr : (*view)->buffer.get();
	}

	static bool Matches(const ImageHeap& heap, const ShaderRecompiler::IR::ImageHeapRequest& request,
	                    const ShaderRecompiler::IR::ImageTableSnapshot& table, const std::array<ImageId, 4>& images,
	                    std::span<const uint32_t> slots) {
		for (size_t i = 1; i < table.mapping.size(); ++i) {
			const auto offset = (request.first + (i - 1u) * request.step) / 4u;
			if (offset >= slots.size()) return false;
			const auto slot = slots[offset] & ((1u << ShaderRecompiler::IR::ImageHeapKindShift) - 1u);
			const auto ordinal = table.mapping[i];
			if (ordinal == 0u) {
				if (slot != 0u) return false;
				continue;
			}
			if (slot == 0u || slot >= heap.m_elements.size() || heap.m_elements[slot].key.value != table.slots[ordinal] ||
			    heap.m_elements[slot].image != images[ordinal]) return false;
		}
		return true;
	}
};

}

namespace {

using namespace Libs::Graphics;

int         g_failures = 0;
const char* g_name     = "kyty_shim_descriptor_tables";

void Check(bool condition, const std::string& what, uint32_t element = UINT32_MAX) {
	if (condition) return;
	if (element == UINT32_MAX) {
		std::fprintf(stderr, "%s: FAILED %s\n", g_name, what.c_str());
	} else {
		std::fprintf(stderr, "%s: FAILED %s at element %u\n", g_name, what.c_str(), element);
	}
	g_failures++;
}

int Finish() {
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %d failure(s)\n", g_name, g_failures);
		return 1;
	}
	std::printf("%s: ok\n", g_name);
	return 0;
}

namespace IR = ShaderRecompiler::IR;

IR::DescriptorValue HeapTexture(uint64_t address) {
	IR::DescriptorValue value {.dword_count = 8u};
	value.dwords[0] = static_cast<uint32_t>(address >> 8u);
	value.dwords[1] = static_cast<uint32_t>((address >> 40u) & 0xffu) |
	                 (static_cast<uint32_t>(Prospero::BufferFormat::k8UNorm) << 20u) | (3u << 30u);
	value.dwords[2] = 15u;
	value.dwords[3] = 0xfacu | (9u << 28u);
	return value;
}

void CheckHeapExhaustion(GraphicContext& graphics, uint64_t address, uint64_t bytes) {
	RenderContext renderer(graphics);
	renderer.MapMemory(address, bytes);
	CommandProcessor processor(renderer, 0);
	processor.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(processor.GetCtx(), processor.GetUcfg(), processor.GetShCtx());
	auto& heap = renderer.GetImageHeap();
	auto& buffers = renderer.GetBufferCache();
	IR::ImageResource root;
	root.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
	const auto profile = IR::ImageTableHeapProfile(root, false);
	const auto texture_a = HeapTexture(address + 0x4000u);
	const auto texture_b = HeapTexture(address + 0x8000u);
	const auto texture_c = HeapTexture(address + 0xc000u);
	const auto write = [&](uint64_t target, const IR::DescriptorValue& value) {
		Libs::LibKernel::Memory::WriteBacking(target, value.dwords.data(), sizeof(value.dwords));
		buffers.InvalidateMemory(target, sizeof(value.dwords));
	};
	write(address, texture_a);
	const IR::ImageHeapRequest resident {.base = address, .size = 32u, .step = 32u, .count = 1u, .root = &profile, .table = true};
	IR::ImageHeapView result;
	if (!heap.Acquire(resident, result)) {
		Check(false, "direct binding before exhaustion");
		return;
	}
	ImageHeapTestAccess::Fill(heap);

	constexpr uint32_t aliases = 4096;
	IR::ImageTableSnapshot table {.slots = {{}, texture_a, texture_b, texture_c},
	                             .mapping = std::vector<uint32_t>(aliases + 3u, 1u), .slots_id = 2};
	table.mapping.front() = aliases + 2u;
	table.mapping[aliases + 1u] = 2;
	table.mapping[aliases + 2u] = 3;
	const IR::ImageHeapRequest request {.base = address + 0x10000u, .size = (aliases + 2u) * 32u,
	                                    .step = 32u, .count = aliases + 2u, .root = &profile, .table = true};
	const auto write_table = [&](const IR::ImageHeapRequest& target, const IR::ImageTableSnapshot& source) {
		for (size_t i = 1; i < source.mapping.size(); ++i) {
			write(target.base + target.first + (i - 1u) * target.step, source.slots[source.mapping[i]]);
		}
		heap.BeginDraw();
	};
	write_table(request, table);
	const auto pending = ImageHeapTestAccess::Pending(heap);
	Check(!heap.Acquire(request, result), "full heap falls back for a mixed table with two missing slots");
	std::printf("%s: first failure, occupied including null=%zu, free for tables=0, required=2, heap_bytes=%zu, view_bytes=%zu\n",
	            g_name, ImageHeapTestAccess::Occupied(heap), sizeof(ImageHeap), ImageHeapTestAccess::ViewBytes());
	size_t failed_buffers = 0;
	for (uint32_t window = 0; window < 8; ++window) {
		for (uint32_t gc = 0; gc < 33; ++gc) {
			heap.BeginDraw();
			Check(heap.Acquire(resident, result), "keep the resident alias across reclamation windows");
			heap.RunGarbageCollector();
		}
		heap.BeginDraw();
		Check(!heap.Acquire(request, result), "a recreated exhausted view uses fallback");
		failed_buffers += ImageHeapTestAccess::Buffers(heap) - 1u;
		scheduler.Wait(scheduler.CurrentTick());
	}
	std::printf("%s: 8 recreated exhausted views, unnecessary_slot_buffers=%zu\n", g_name, failed_buffers);
	Check(failed_buffers == 0u, "recreated exhausted views must not allocate slot buffers");
	const auto device = graphics.device;
	const auto extensions = graphics.physical_device.enumerateDeviceExtensionProperties(nullptr);
	const bool memory_budget = extensions.result == vk::Result::eSuccess &&
	    std::ranges::any_of(extensions.value, [](const auto& extension) {
		    return std::strcmp(extension.extensionName.data(), VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
	    });
	vk::DeviceSize peak_device_bytes = 0;
	const auto sample_memory = [&] {
		if (!memory_budget) return;
		vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget {};
		vk::PhysicalDeviceMemoryProperties2 properties {.pNext = &budget};
		graphics.physical_device.getMemoryProperties2(&properties);
		vk::DeviceSize usage = 0;
		for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
			if (properties.memoryProperties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				usage += budget.heapUsage[i];
			}
		}
		peak_device_bytes = std::max(peak_device_bytes, usage);
	};
	auto fallback_root = profile;
	fallback_root.heap = false;
	fallback_root.table = 0;
	fallback_root.table_capacity = 256;
	fallback_root.table_descriptor_base = 1;
	fallback_root.descriptor_index = 0;
	const auto fallback_descriptors = fallback_root.table_descriptor_base + fallback_root.table_capacity;
	const auto kind = *IR::DescriptorBindingForImage(fallback_root);
	ShaderModule module;
	module.info.stage = ShaderType::Compute;
	module.info.info.images.push_back(fallback_root);
	module.info.bindings.descriptor_counts[static_cast<size_t>(kind)] = fallback_descriptors;
	module.info.bindings.descriptor_mask = uint64_t {1} << static_cast<size_t>(kind);
	const IR::ResourceSnapshot fallback {.images = {{.dword_count = 8u}},
	                                     .image_tables = {std::make_shared<const IR::ImageTableSnapshot>(table)}};
	const ShaderStageRuntime runtime {&module, &fallback};
	const vk::DescriptorSetLayoutBinding binding {
	    IR::NativeBinding(ShaderType::Compute, kind), vk::DescriptorType::eSampledImage,
	    fallback_descriptors, vk::ShaderStageFlagBits::eCompute};
	const auto set_layout = device.createDescriptorSetLayout({.bindingCount = 1u, .pBindings = &binding});
	if (set_layout.result != vk::Result::eSuccess) {
		Check(false, "fallback descriptor set layout");
		return;
	}
	DescriptorHeap::RegisterLayout(set_layout.value, std::span(&binding, 1u));
	const auto layout = device.createPipelineLayout({.setLayoutCount = 1u, .pSetLayouts = &set_layout.value});
	if (layout.result != vk::Result::eSuccess) {
		Check(false, "fallback pipeline layout");
		device.destroyDescriptorSetLayout(set_layout.value);
		return;
	}
	const std::array layouts {set_layout.value, graphics.image_heap_layout};
	const auto heap_layout = device.createPipelineLayout({.setLayoutCount = static_cast<uint32_t>(layouts.size()),
	                                                       .pSetLayouts = layouts.data()});
	if (heap_layout.result != vk::Result::eSuccess) {
		Check(false, "direct pipeline layout");
		device.destroyPipelineLayout(layout.value);
		device.destroyDescriptorSetLayout(set_layout.value);
		return;
	}
	const PipelineCache::Pipeline pipeline {.pipeline_layout = layout.value, .descriptor_set_layout = set_layout.value,
	                                       .push_stages = vk::ShaderStageFlagBits::eCompute};
	PreparedBindings prepared;
	auto& executor = renderer.GetRenderExecutor();
	const auto matches = [&](const IR::ImageHeapRequest& target, const IR::ImageTableSnapshot& expected,
	                         const std::array<ImageId, 4>& images) {
		const auto* slots_buffer = ImageHeapTestAccess::ViewBuffer(heap, target);
		if (slots_buffer == nullptr) return false;
		heap.Commit(scheduler.Current(), vk::PipelineBindPoint::eCompute, heap_layout.value);
		const auto slot_bytes = std::min<uint64_t>(target.size & ~uint64_t {3}, slots_buffer->Size());
		Buffer readback(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, slot_bytes);
		readback.CopyFrom(scheduler.Current(), *slots_buffer, 0, 0, slot_bytes);
		scheduler.Finish();
		std::vector<uint32_t> slots(slot_bytes / sizeof(uint32_t));
		std::memcpy(slots.data(), readback.Mapped().data(), slot_bytes);
		return ImageHeapTestAccess::Matches(heap, target, expected, images, slots);
	};
	const auto bind_fallback = [&] {
		RenderExecutorTestAccess::Reset(executor);
		executor.PrepareBindings(runtime, prepared);
		executor.RebindImages(prepared);
		std::array stages {&prepared};
		executor.CommitBindings(scheduler.Current(), vk::PipelineBindPoint::eCompute, pipeline, stages);
	};
	bind_fallback();
	sample_memory();
	std::array<ImageId, 4> fallback_images {};
	for (size_t i = 1; i < fallback_images.size() && i < prepared.table_slots.size(); ++i) {
		fallback_images[i] = prepared.table_images[prepared.table_slots[i]]->image_id;
	}
	Check(std::ranges::all_of(std::span(fallback_images).subspan(1), [](ImageId id) { return static_cast<bool>(id); }),
	      "the fallback resolved all three non-null descriptors");
	const auto retry = [&](uint32_t repeats) {
		std::chrono::nanoseconds direct {}, binding_time {};
		uint32_t failures = 0;
		for (uint32_t i = 0; i < repeats; ++i) {
			heap.BeginDraw();
			const auto start = std::chrono::steady_clock::now();
			const bool acquired = heap.Acquire(request, result);
			const auto after_direct = std::chrono::steady_clock::now();
			direct += after_direct - start;
			if (!acquired) {
				++failures;
				bind_fallback();
				binding_time += std::chrono::steady_clock::now() - after_direct;
			}
			if ((i + 1u) % 64u == 0u) {
				sample_memory();
				scheduler.Wait(scheduler.CurrentTick());
			}
		}
		std::printf("%s: %u failed requests, direct_ns=%lld fallback_ns=%lld pending_allocations=%zu\n",
		            g_name, failures, static_cast<long long>(direct.count()), static_cast<long long>(binding_time.count()),
		            ImageHeapTestAccess::Pending(heap) - pending);
		Check(failures == repeats, "exhausted requests consistently use fallback");
	};
	retry(4096);
	ImageHeapTestAccess::Release(heap, 2);
	scheduler.Wait(scheduler.CurrentTick());
	heap.BeginDraw();
	Check(heap.Acquire(resident, result), "keep the resident alias while reclaiming one slot");
	heap.RunGarbageCollector();
	auto replacement = request;
	replacement.base = address + 0x40000u;
	write_table(replacement, table);
	Check(!heap.Acquire(replacement, result), "a fresh alias-heavy view needs two slots when only one is free");
	Check(ImageHeapTestAccess::Pending(heap) == pending, "insufficient reclamation must not allocate a partial view");
	retry(4096);
	std::printf("%s: after insufficient reclamation, occupied including null=%zu, free for tables=%zu, required=2\n",
	            g_name, ImageHeapTestAccess::Occupied(heap), ImageHeapTestAccess::TableSlotsFree(heap));
	Check(ImageHeapTestAccess::Pending(heap) == pending && ImageHeapTestAccess::TableSlotsFree(heap) == 1,
	      "insufficient reclaim must not intern then roll back a slot");
	std::printf("%s: fallback persistent ring sets=%zu\n", g_name, RenderExecutorTestAccess::RingSets(executor, module));
	ImageHeapTestAccess::Release(heap, 3);
	Check(!heap.Acquire(request, result), "a slot retired on the current tick cannot count as free");
	scheduler.Wait(scheduler.CurrentTick());
	const vk::SemaphoreTypeCreateInfo timeline {.semaphoreType = vk::SemaphoreType::eTimeline};
	const auto gate = device.createSemaphore({.pNext = &timeline});
	if (gate.result != vk::Result::eSuccess) {
		Check(false, "fallback submission gate");
		scheduler.Finish();
		device.destroyPipelineLayout(heap_layout.value);
		device.destroyPipelineLayout(layout.value);
		device.destroyDescriptorSetLayout(set_layout.value);
		return;
	}
	bind_fallback();
	SubmitInfo submit;
	submit.AddWait(gate.value);
	const auto fallback_tick = scheduler.CurrentTick();
	scheduler.Flush(submit);
	Check(!scheduler.IsFree(fallback_tick), "the previous fallback descriptor set is in flight");
	const bool recovered = heap.Acquire(request, result);
	Check(recovered, "two completed slots recover the alias-heavy direct binding");
	heap.Rebind(executor);
	sample_memory();
	Check(device.signalSemaphore({.semaphore = gate.value, .value = 1u}) == vk::Result::eSuccess, "release the fallback submission");
	scheduler.Finish();
	device.destroySemaphore(gate.value);
	if (recovered) {
		Check(matches(request, table, fallback_images), "recovered heap preserves the fallback descriptor at every ordinal");
	}

	const auto full = ImageHeapTestAccess::Pending(heap);
	IR::ImageTableSnapshot unique {.slots = {{}, texture_a, texture_b, texture_c}, .mapping = {260, 1, 2, 3},
	                               .slots_id = 3};
	for (uint32_t i = 4; i <= 260; ++i) {
		unique.slots.push_back(HeapTexture(address + uint64_t {i} * 0x10000u));
		unique.mapping.push_back(i);
	}
	auto changed = replacement;
	changed.size = 260u * 32u;
	changed.count = 260u;
	write_table(changed, unique);
	Check(!heap.Acquire(changed, result) && ImageHeapTestAccess::Pending(heap) == full,
	      "mostly unique replacement at the same owner stays on fallback without allocating");
	const IR::ImageTableSnapshot remapped {.slots = {{}, texture_c}, .mapping = {4, 1, 1, 1, 1}, .slots_id = 4};
	changed.size = 4u * 32u;
	changed.count = 4u;
	write_table(changed, remapped);
	Check(heap.Acquire(changed, result), "a changed snapshot can use resident aliases at zero free slots");

	write(address, texture_a);
	write(address + 32u, texture_b);
	write(address + 64u, unique.slots[4]);
	const IR::ImageHeapRequest mapped {.base = address, .size = 96, .step = 32, .count = 3, .root = &profile, .table = true};
	heap.BeginDraw();
	Check(!heap.Acquire(mapped, result), "memory-backed view starts on fallback");
	std::thread invalidator([&] {
		for (uint32_t i = 0; i < 128; ++i) buffers.InvalidateMemory(address, 96u);
	});
	bool exhausted = true;
	for (uint32_t i = 0; i < 128; ++i) {
		heap.BeginDraw();
		exhausted &= !heap.Acquire(mapped, result);
	}
	invalidator.join();
	Check(exhausted, "concurrent invalidation preserves the memory-backed failure");
	write(address + 64u, texture_c);
	heap.BeginDraw();
	Check(heap.Acquire(mapped, result), "memory-backed descriptor mutation invalidates a failed request");
	IR::ImageTableSnapshot expected {.slots = {{}, texture_a, texture_b, texture_c}, .mapping = {3, 1, 2, 3}};
	heap.Rebind(executor);
	Check(matches(mapped, expected, fallback_images), "changed descriptors select the current images");
	auto truncated = mapped;
	truncated.size = 76u;
	Check(!heap.Acquire(truncated, result) && matches(mapped, expected, fallback_images),
	      "a different extent in the same draw preserves the active view and uses fallback");
	expected.mapping.back() = 0u;
	heap.BeginDraw();
	Check(heap.Acquire(truncated, result) && matches(truncated, expected, fallback_images),
	      "truncating a descriptor clears its previous image");
	renderer.UnmapMemory(address, 0x4000u);
	renderer.MapMemory(address, 0x4000u);
	write(address + 64u, texture_a);
	heap.BeginDraw();
	Check(heap.Acquire(mapped, result), "a view remapped at the same guest address uses the new table");
	expected.mapping.back() = 1u;
	heap.Rebind(executor);
	Check(matches(mapped, expected, fallback_images), "remapped slots resolve to the current images");

	const IR::ImageTableSnapshot keep {.slots = {{}, texture_a, texture_c}, .mapping = {2, 1, 2}};
	changed.size = 64u;
	changed.count = 2u;
	write_table(changed, keep);
	Check(heap.Acquire(changed, result), "keep the two unchanged images resident");
	renderer.UnmapMemory(request.base, request.size);
	const auto old_slot = ImageHeapTestAccess::LiveSlot(heap, texture_b);
	const auto next = HeapTexture(address + 0x38000u);
	write(address + 32u, next);
	heap.BeginDraw();
	Check(!heap.Acquire(mapped, result), "the retired image slot cannot be overwritten on the recording tick");
	bool recycled = false;
	for (uint32_t frame = 0; frame < 256u && !recycled; ++frame) {
		heap.RunGarbageCollector();
		scheduler.Finish();
		heap.BeginDraw();
		Check(heap.Acquire(changed, result), "keep the two unchanged images resident while the replaced slot ages");
		recycled = heap.Acquire(mapped, result);
		heap.Commit(scheduler.Current(), vk::PipelineBindPoint::eCompute, heap_layout.value);
	}
	Check(recycled, "a completed image slot accepts a different descriptor");
	heap.Rebind(executor);
	const auto next_image = executor.ResolveTexture(IR::ImageTableHeapResource(profile, next), next, true).image_id;
	expected.slots[2] = next;
	auto next_images = fallback_images;
	next_images[2] = next_image;
	Check(old_slot != 0u && old_slot == ImageHeapTestAccess::LiveSlot(heap, next) &&
	      next_image && next_image != fallback_images[2] && matches(mapped, expected, next_images),
	      "a recycled slot resolves the replacement image at every ordinal");
	scheduler.Finish();
	device.destroyPipelineLayout(heap_layout.value);
	device.destroyPipelineLayout(layout.value);
	device.destroyDescriptorSetLayout(set_layout.value);
	renderer.UnmapMemory(address, bytes);
	if (memory_budget) {
		std::printf("%s: sampled peak device-local bytes=%llu\n", g_name,
		            static_cast<unsigned long long>(peak_device_bytes));
	}
}

int HeapExhaustion() {
	g_name = "kyty_shim_image_heap_exhaustion";
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize(true, true)) {
		std::fprintf(stderr, "%s: %s\n", g_name, unavailable);
		return std::getenv("ANYPS5_REQUIRE_VULKAN") ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) return 1;
	void* guest = nullptr;
	constexpr uint64_t bytes = 0x100000u;
	if (sceKernelMapFlexibleMemory(&guest, bytes, 3, 0) != 0 || guest == nullptr) {
		Check(false, "map backing for the descriptor table and textures");
		return Finish();
	}
	const auto address = reinterpret_cast<uint64_t>(guest);
	CheckHeapExhaustion(gpu.graphics, address, bytes);
	Check(sceKernelMunmap(address, bytes) == 0, "unmap test backing");
	return Finish();
}

constexpr uint32_t TableCapacity = 256;
constexpr uint32_t SlotCount     = 2 * TableCapacity;
constexpr uint32_t HeadGap       = 16;
constexpr uint32_t MiddleGap     = 8;
constexpr uint32_t Tail          = 4;
constexpr uint32_t ElementCount  = HeadGap + TableCapacity + MiddleGap + TableCapacity + Tail;

std::vector<ImageTableFill> Tables(uint32_t head, uint32_t middle) {
	return {{.base = head, .capacity = TableCapacity, .first_slot = TableCapacity},
	        {.base = head + TableCapacity + middle, .capacity = TableCapacity, .first_slot = 0}};
}

const ImageTableFill* TableOf(std::span<const ImageTableFill> tables, uint32_t element) {
	for (const auto& table: tables) {
		if (element >= table.base && element < table.base + table.capacity) return &table;
	}
	return nullptr;
}

vk::ImageView FakeView(uint64_t value) {
	return vk::ImageView(reinterpret_cast<VkImageView>(static_cast<uintptr_t>(value)));
}

vk::DescriptorSet FakeSet(uint64_t value) {
	return vk::DescriptorSet(reinterpret_cast<VkDescriptorSet>(static_cast<uintptr_t>(value)));
}

vk::DescriptorImageInfo FakeInfo(uint64_t view) {
	return {nullptr, view == 0 ? nullptr : FakeView(view), vk::ImageLayout::eShaderReadOnlyOptimal};
}

struct Fills {
	std::vector<vk::WriteDescriptorSet> gaps;
	std::vector<vk::WriteDescriptorSet> writes;
	std::vector<vk::CopyDescriptorSet>  copies;

	[[nodiscard]] uint64_t Copied() const {
		uint64_t count = 0;
		for (const auto& copy: copies) count += copy.descriptorCount;
		return count;
	}
};

struct HostCase {
	uint32_t                             binding = 3;
	uint32_t                             count   = 0;
	std::vector<ImageTableFill>          tables;
	std::vector<vk::DescriptorImageInfo> images;
	std::vector<vk::DescriptorImageInfo> slot_infos;
	vk::DescriptorSet                    persistent = FakeSet(0xabc000u);

	[[nodiscard]] vk::WriteDescriptorSet Whole() const {
		vk::WriteDescriptorSet whole {};
		whole.dstBinding      = binding;
		whole.descriptorType  = vk::DescriptorType::eSampledImage;
		whole.descriptorCount = count;
		whole.pImageInfo      = images.data();
		return whole;
	}

	[[nodiscard]] Fills Emit(bool writes, std::span<const uint8_t> live) const {
		Fills fills;
		AppendImageTableGaps(Whole(), tables, fills.gaps);
		AppendImageTableFills(Whole(), tables, writes, persistent, slot_infos, live, fills.writes, fills.copies);
		return fills;
	}
};

HostCase MakeHostCase(uint32_t head, uint32_t middle, uint32_t tail) {
	HostCase result;
	result.tables = Tables(head, middle);
	result.count  = head + TableCapacity + middle + TableCapacity + tail;
	result.images.resize(result.count);
	for (uint32_t element = 0; element < result.count; element++) {
		result.images[element] = {nullptr, FakeView(0x100000u + element), vk::ImageLayout::eGeneral};
	}
	result.slot_infos.resize(SlotCount);
	for (uint32_t slot = 0; slot < SlotCount; slot++) {
		result.slot_infos[slot] = FakeInfo(slot % 7 == 0 ? 0 : 0x200000u + slot);
	}
	return result;
}

void VerifyFills(const std::string& name, const HostCase& c, const Fills& fills, bool writes,
                 std::span<const uint8_t> live) {
	struct Source {
		int                            hits        = 0;
		bool                           copy        = false;
		const vk::DescriptorImageInfo* info        = nullptr;
		uint32_t                       src_element = 0;
	};
	std::vector<Source> sources(c.count);
	const auto put_write = [&](const vk::WriteDescriptorSet& write) {
		Check(write.dstBinding == c.binding, name + ": write binding");
		Check(write.descriptorType == vk::DescriptorType::eSampledImage, name + ": write type");
		Check(!write.dstSet, name + ": write set left to the caller");
		Check(write.descriptorCount != 0, name + ": empty write");
		for (uint32_t k = 0; k < write.descriptorCount; k++) {
			const auto element = write.dstArrayElement + k;
			if (element >= c.count) {
				Check(false, name + ": write past the binding", element);
				return;
			}
			sources[element].hits++;
			sources[element].copy = false;
			sources[element].info = write.pImageInfo + k;
		}
	};
	for (const auto& write: fills.gaps) put_write(write);
	for (const auto& write: fills.writes) put_write(write);
	for (const auto& copy: fills.copies) {
		Check(copy.srcSet == c.persistent, name + ": copy source set");
		Check(!copy.dstSet, name + ": copy destination left to the caller");
		Check(copy.srcBinding == c.binding && copy.dstBinding == c.binding, name + ": copy binding");
		Check(copy.srcArrayElement == copy.dstArrayElement, name + ": copy keeps the element");
		Check(copy.descriptorCount != 0, name + ": empty copy");
		for (uint32_t k = 0; k < copy.descriptorCount; k++) {
			const auto element = copy.dstArrayElement + k;
			if (element >= c.count) {
				Check(false, name + ": copy past the binding", element);
				return;
			}
			sources[element].hits++;
			sources[element].copy        = true;
			sources[element].src_element = copy.srcArrayElement + k;
		}
	}
	uint64_t not_alive = 0;
	for (uint32_t element = 0; element < c.count; element++) {
		const auto& source = sources[element];
		Check(source.hits == 1, name + ": element filled exactly once", element);
		const auto* table = TableOf(c.tables, element);
		if (table == nullptr) {
			Check(!source.copy && source.info == &c.images[element], name + ": gap from the binding's infos",
			      element);
			continue;
		}
		const auto slot  = table->first_slot + (element - table->base);
		const bool alive = live.empty() || live[slot] != 0;
		if (!writes || !alive) {
			Check(source.copy && source.src_element == element,
			      name + ": table element copied from the persistent set", element);
			not_alive += writes ? 1 : 0;
		} else {
			Check(!source.copy && source.info == &c.slot_infos[slot], name + ": table element written from its slot",
			      element);
		}
	}
	if (writes) {
		Check(fills.Copied() == not_alive, name + ": only slots not proven alive are copied");
	} else {
		Check(fills.writes.empty(), name + ": copy mode writes no table");
		Check(fills.copies.size() == c.tables.size(), name + ": copy mode has one copy per table");
	}
}

void CheckCopyMode(bool writes) {
	const auto c     = MakeHostCase(HeadGap, MiddleGap, Tail);
	const auto fills = c.Emit(writes, {});
	VerifyFills("copies", c, fills, false, {});
	for (size_t t = 0; t < fills.copies.size() && t < c.tables.size(); t++) {
		Check(fills.copies[t].srcArrayElement == c.tables[t].base &&
		          fills.copies[t].descriptorCount == c.tables[t].capacity,
		      "copies: one whole-capacity copy per table");
	}
}

void CheckGaps() {
	const auto c = MakeHostCase(HeadGap, MiddleGap, Tail);
	Fills      fills;
	AppendImageTableGaps(c.Whole(), c.tables, fills.gaps);
	const std::array<std::pair<uint32_t, uint32_t>, 3> expected {
	    {{0u, HeadGap}, {HeadGap + TableCapacity, MiddleGap}, {ElementCount - Tail, Tail}}};
	Check(fills.gaps.size() == expected.size(), "gaps: a head, a middle and a tail gap");
	for (size_t index = 0; index < fills.gaps.size() && index < expected.size(); index++) {
		const auto& gap = fills.gaps[index];
		Check(gap.dstArrayElement == expected[index].first && gap.descriptorCount == expected[index].second &&
		          gap.pImageInfo == c.images.data() + expected[index].first,
		      "gaps: gap " + std::to_string(index) + " at its element, from the binding's infos");
	}
	const auto full = MakeHostCase(0, 0, 0);
	Fills      none;
	AppendImageTableGaps(full.Whole(), full.tables, none.gaps);
	Check(none.gaps.empty(), "gaps: a binding made only of tables has no gap");
}

void CheckFills() {
	CheckCopyMode(false);
	{
		const auto c     = MakeHostCase(HeadGap, MiddleGap, Tail);
		const auto fills = c.Emit(true, {});
		VerifyFills("writes", c, fills, true, {});
		Check(fills.copies.empty(), "writes: no copies");
		Check(fills.writes.size() == c.tables.size(), "writes: one write per table");
		for (size_t t = 0; t < fills.writes.size() && t < c.tables.size(); t++) {
			Check(fills.writes[t].dstArrayElement == c.tables[t].base &&
			          fills.writes[t].descriptorCount == c.tables[t].capacity &&
			          fills.writes[t].pImageInfo == c.slot_infos.data() + c.tables[t].first_slot,
			      "writes: whole table written from its first slot");
		}
	}
	{
		const auto           c = MakeHostCase(HeadGap, MiddleGap, Tail);
		std::vector<uint8_t> live(SlotCount, 1);
		for (const uint32_t slot: {0u, 3u, 4u, 5u, 255u, 256u, 511u}) live[slot] = 0;
		const auto fills = c.Emit(true, live);
		VerifyFills("fallback", c, fills, true, live);
		Check(fills.writes.size() == 3, "fallback: the runs proven alive are written");
		Check(fills.copies.size() == 5, "fallback: the runs not proven alive are copied");
		Check(fills.Copied() == 7, "fallback: seven slots copied");
	}
	{
		const auto           c = MakeHostCase(0, 0, 0);
		std::vector<uint8_t> live(SlotCount, 0);
		const auto           fills = c.Emit(true, live);
		VerifyFills("no slot alive", c, fills, true, live);
		Check(fills.writes.empty(), "no slot alive: no table writes");
		Check(fills.copies.size() == 2, "no slot alive: one copy per table");
	}
}

void CheckSeeds() {
	const auto  c     = MakeHostCase(HeadGap, MiddleGap, Tail);
	const auto  set   = FakeSet(0xdef000u);
	const auto& table = c.tables[0];
	std::vector<vk::WriteDescriptorSet> writes;
	std::vector<vk::CopyDescriptorSet>  copies;
	AppendImageTableSeed(c.binding, vk::DescriptorType::eSampledImage, table, true, set, c.persistent, c.slot_infos,
	                     writes, copies);
	Check(copies.empty() && writes.size() == 1, "seed writes: one write");
	if (writes.size() == 1) {
		const auto& seed = writes[0];
		Check(seed.dstSet == c.persistent && seed.dstBinding == c.binding && seed.dstArrayElement == table.base &&
		          seed.descriptorCount == table.capacity && seed.descriptorType == vk::DescriptorType::eSampledImage,
		      "seed writes: the table's elements of the persistent set");
		Check(seed.pImageInfo == c.slot_infos.data() + table.first_slot, "seed writes: from the table's first slot");
	}
	writes.clear();
	AppendImageTableSeed(c.binding, vk::DescriptorType::eSampledImage, table, false, set, c.persistent, c.slot_infos,
	                     writes, copies);
	Check(writes.empty() && copies.size() == 1, "seed copies: one copy");
	if (copies.size() == 1) {
		const auto& seed = copies[0];
		Check(seed.srcSet == set && seed.dstSet == c.persistent && seed.srcBinding == c.binding &&
		          seed.dstBinding == c.binding && seed.srcArrayElement == table.base &&
		          seed.dstArrayElement == table.base && seed.descriptorCount == table.capacity,
		      "seed copies: the table's elements from the draw's set into the persistent set");
	}
}

void CheckSetUpdate() {
	const auto                          set        = FakeSet(0x111000u);
	const auto                          persistent = FakeSet(0x222000u);
	std::vector<vk::WriteDescriptorSet> writes(2);
	writes[0].dstBinding = 1;
	writes[1].dstBinding = 2;
	std::vector<vk::WriteDescriptorSet> table_writes(1);
	table_writes[0].dstBinding = 3;
	std::vector<vk::CopyDescriptorSet> copies(1);
	copies[0].srcSet = persistent;
	std::vector<vk::WriteDescriptorSet> slot_set_writes(1);
	slot_set_writes[0].dstSet     = persistent;
	slot_set_writes[0].dstBinding = 4;
	PatchTableSetUpdate(set, writes, table_writes, copies, slot_set_writes);
	Check(writes.size() == 4, "set update: binding, table and slot set writes in one list");
	for (size_t index = 0; index < writes.size(); index++) {
		Check(writes[index].dstBinding == index + 1, "set update: writes in order " + std::to_string(index));
		Check(writes[index].dstSet == (index < 3 ? set : persistent),
		      "set update: write " + std::to_string(index) + " goes to its set");
	}
	Check(copies[0].dstSet == set && copies[0].srcSet == persistent, "set update: copies go to the set");
}

void CheckBindingFills() {
	const auto c           = MakeHostCase(HeadGap, MiddleGap, Tail);
	const int  state_a     = 0;
	const int  state_b     = 0;
	const auto other_set   = FakeSet(0xbbb000u);
	const auto binding_for = [&](const int* state, vk::DescriptorSet persistent) {
		ImageTableBinding binding {
		    .state = state, .persistent = persistent, .slot_infos = c.slot_infos, .whole = c.Whole()};
		for (const auto& table: c.tables) binding.tables[binding.count++] = table;
		return binding;
	};
	const std::vector<ImageTableBinding> bindings {binding_for(&state_a, c.persistent),
	                                               binding_for(&state_a, c.persistent),
	                                               binding_for(&state_b, other_set),
	                                               binding_for(&state_a, c.persistent)};
	std::vector<uint8_t> first_proof(SlotCount, 1);
	first_proof[5] = 0;
	std::vector<const void*> proved;
	const auto               prove = [&](const void* state) {
		proved.push_back(state);
		return proved.size() == 1 ? std::span<const uint8_t> {first_proof} : std::span<const uint8_t> {};
	};
	Fills fills;
	AppendTableBindingFills(bindings, true, prove, fills.writes, fills.copies);
	Check(proved == std::vector<const void*> {&state_a, &state_b, &state_a},
	      "binding fills: a state is proven once per run of its bindings and again after another state");
	Check(fills.copies.size() == 2 && fills.Copied() == 2,
	      "binding fills: the slot the first proof failed is copied in both bindings of its run, and nothing after "
	      "the passing proofs");
	Check(fills.writes.size() == 10, "binding fills: the other runs and tables are written");
	for (const auto& copy: fills.copies) {
		Check(copy.srcSet == c.persistent && copy.dstArrayElement == c.tables[1].base + 5,
		      "binding fills: the failed slot is copied from its binding's persistent set");
	}
	proved.clear();
	Fills copies;
	AppendTableBindingFills(bindings, false, prove, copies.writes, copies.copies);
	Check(proved.empty(), "binding fills: copy mode proves nothing");
	Check(copies.writes.empty() && copies.copies.size() == 8, "binding fills: copy mode copies every table");
	for (size_t index = 0; index < copies.copies.size(); index++) {
		Check(copies.copies[index].srcSet == (index / 2 == 2 ? other_set : c.persistent),
		      "binding fills: each binding copies from its own persistent set");
	}
}

void CheckLiveElements() {
	const std::vector<vk::DescriptorImageInfo> infos {FakeInfo(0x11), FakeInfo(0), FakeInfo(0x22), FakeInfo(0x33),
	                                                  FakeInfo(0x66)};
	const auto views = [](std::initializer_list<uint64_t> handles) {
		std::vector<CachedImageView> result;
		for (const auto handle: handles) result.push_back({.info = {}, .view = FakeView(handle)});
		return result;
	};
	const std::vector<CachedImageView> pinned_image   = views({0x10, 0x11});
	const std::vector<CachedImageView> other_image    = views({0x44, 0x55});
	const std::vector<CachedImageView> reused_handles = views({0x66});
	const std::array<const std::vector<CachedImageView>*, 5> images {&pinned_image, &pinned_image, nullptr,
	                                                                 &other_image, &reused_handles};
	std::vector<uint32_t> asked;
	std::vector<uint8_t>  live;
	MarkLiveTableElements(
	    infos,
	    [&](size_t index) {
		    asked.push_back(static_cast<uint32_t>(index));
		    return images[index];
	    },
	    live);
	const std::array<uint8_t, 5> expected {1, 1, 0, 0, 1};
	const std::array<const char*, 5> reasons {"a view its image lists", "a null view", "an image no longer in its slot",
	                                          "a slot now holding an image that does not list the view",
	                                          "a slot now holding an image that lists the same handle"};
	Check(live.size() == expected.size(), "live elements: one mark per element");
	for (uint32_t index = 0; index < expected.size() && index < live.size(); index++) {
		Check(live[index] == expected[index], std::string("live elements: an element with ") + reasons[index],
		      index);
	}
	Check(std::ranges::find(asked, 1u) == asked.end(), "live elements: a null view needs no image");
}

void CheckLiveSlots() {
	const std::vector<vk::DescriptorImageInfo> infos {FakeInfo(0x11), FakeInfo(0x22), FakeInfo(0), FakeInfo(0x44)};
	const std::vector<uint8_t>                 element_live {1, 0, 1, 1};
	const std::vector<uint32_t>                slots {0, 0, 1, 2, 3, 3, 7, 1};
	const std::vector<vk::DescriptorImageInfo> slot_infos {FakeInfo(0x11), FakeInfo(0x99), FakeInfo(0x22),
	                                                       FakeInfo(0),    FakeInfo(0x44), FakeInfo(0x55),
	                                                       FakeInfo(0x66), FakeInfo(0)};
	const std::vector<uint8_t>                 expected {1, 0, 0, 1, 1, 0, 0, 1};
	const std::array<const char*, 8>           reasons {"the view of its element",
	                                                    "a view that is not its element's",
	                                                    "the view of an element not proven alive",
	                                                    "a null view",
	                                                    "the view of its element",
	                                                    "a view that is not its element's",
	                                                    "an element past the table's elements",
	                                                    "a null view of an element not proven alive"};
	std::vector<uint8_t> live;
	Check(!MarkLiveTableSlots(slots, slot_infos, infos, element_live, live),
	      "live slots: a slot not proven alive is reported");
	Check(live.size() == expected.size(), "live slots: one mark per slot");
	for (uint32_t slot = 0; slot < expected.size() && slot < live.size(); slot++) {
		Check(live[slot] == expected[slot], std::string("live slots: a slot holding ") + reasons[slot], slot);
	}
	const std::vector<uint32_t>                live_slots {0, 3, 2, 3};
	const std::vector<vk::DescriptorImageInfo> live_infos {FakeInfo(0x11), FakeInfo(0x44), FakeInfo(0),
	                                                       FakeInfo(0x44)};
	Check(MarkLiveTableSlots(live_slots, live_infos, infos, element_live, live) &&
	          std::ranges::all_of(live, [](uint8_t mark) { return mark == 1; }),
	      "live slots: every slot proven alive");
}

constexpr std::array DefaultDrivers {vk::DriverId::eMesaRadv, vk::DriverId::eNvidiaProprietary,
                                     vk::DriverId::eAmdProprietary, vk::DriverId::eMesaLlvmpipe,
                                     vk::DriverId::eMoltenvk};

void CheckDriverDefault(const std::string& what) {
	for (const auto driver: DefaultDrivers) {
		const bool expected = driver == vk::DriverId::eMesaRadv;
		Check(TableDescriptorWrites(driver) == expected, what + ": tables are " + (expected ? "written" : "copied") +
		                                                     " on " + vk::to_string(driver));
	}
}

int HostTests() {
	CheckGaps();
	CheckFills();
	CheckSeeds();
	CheckSetUpdate();
	CheckBindingFills();
	CheckLiveElements();
	CheckLiveSlots();
	if (std::getenv("KYTY_TABLE_DESCRIPTOR_WRITES") == nullptr) {
		CheckDriverDefault("KYTY_TABLE_DESCRIPTOR_WRITES unset");
	}
	return Finish();
}

int DefaultSwitch(const char* expected) {
	g_name            = "kyty_shim_descriptor_tables_switch";
	const char* value = std::getenv("KYTY_TABLE_DESCRIPTOR_WRITES");
	Check(value != nullptr && std::strcmp(value, expected) == 0,
	      std::string("run with KYTY_TABLE_DESCRIPTOR_WRITES='") + expected + "'");
	CheckDriverDefault(std::string("KYTY_TABLE_DESCRIPTOR_WRITES='") + expected + "'");
	return Finish();
}

int WritesSwitch(const char* expected) {
	g_name            = "kyty_shim_descriptor_tables_switch";
	const char* value = std::getenv("KYTY_TABLE_DESCRIPTOR_WRITES");
	Check(value != nullptr && std::strcmp(value, expected) == 0,
	      std::string("run with KYTY_TABLE_DESCRIPTOR_WRITES='") + expected + "'");
	for (const auto driver: DefaultDrivers) {
		Check(TableDescriptorWrites(driver), std::string("KYTY_TABLE_DESCRIPTOR_WRITES='") + expected +
		                                         "': tables are written on " + vk::to_string(driver));
	}
	return Finish();
}

int CopiesSwitch() {
	g_name            = "kyty_shim_descriptor_tables_copies";
	const char* value = std::getenv("KYTY_TABLE_DESCRIPTOR_WRITES");
	Check(value != nullptr && std::strcmp(value, "0") == 0, "run with KYTY_TABLE_DESCRIPTOR_WRITES=0");
	for (const auto driver: DefaultDrivers) {
		Check(!TableDescriptorWrites(driver),
		      "KYTY_TABLE_DESCRIPTOR_WRITES=0: tables are copied on " + vk::to_string(driver));
	}
	const bool writes = TableDescriptorWrites(vk::DriverId::eMesaRadv);
	CheckCopyMode(writes);
	return Finish();
}

int BadSwitch() {
	g_name            = "kyty_shim_descriptor_tables_bad";
	const char* value  = std::getenv("KYTY_TABLE_DESCRIPTOR_WRITES");
	const bool  writes = TableDescriptorWrites(vk::DriverId::eMesaRadv);
	std::fprintf(stderr, "%s: KYTY_TABLE_DESCRIPTOR_WRITES=%s did not stop (read as %d)\n", g_name,
	             value == nullptr ? "(unset)" : value, writes ? 1 : 0);
	return 1;
}

constexpr uint32_t ImageCount = 4;

constexpr std::array<std::array<uint32_t, 4>, ImageCount> Texels {
    {{1u, 100u, 200u, 250u}, {17u, 101u, 201u, 250u}, {33u, 102u, 202u, 250u}, {49u, 103u, 203u, 250u}}};

std::string FetchShader() {
	return R"(
               OpCapability Shader
               OpCapability SampledImageArrayDynamicIndexing
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %group
               OpExecutionMode %main LocalSize 1 1 1
               OpDecorate %group BuiltIn WorkgroupId
               OpDecorate %images DescriptorSet 0
               OpDecorate %images Binding 0
               OpDecorate %out DescriptorSet 0
               OpDecorate %out Binding 1
               OpDecorate %texels ArrayStride 16
               OpMemberDecorate %block 0 Offset 0
               OpDecorate %block Block
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %uint3 = OpTypeVector %uint 3
      %uint4 = OpTypeVector %uint 4
       %int2 = OpTypeVector %int 2
      %image = OpTypeImage %uint 2D 0 0 0 1 Unknown
      %count = OpConstant %uint )" +
	       std::to_string(ElementCount) + R"(
      %array = OpTypeArray %image %count
  %array_ptr = OpTypePointer UniformConstant %array
  %image_ptr = OpTypePointer UniformConstant %image
     %images = OpVariable %array_ptr UniformConstant
     %texels = OpTypeRuntimeArray %uint4
      %block = OpTypeStruct %texels
  %block_ptr = OpTypePointer StorageBuffer %block
        %out = OpVariable %block_ptr StorageBuffer
  %texel_ptr = OpTypePointer StorageBuffer %uint4
  %input_ptr = OpTypePointer Input %uint3
      %group = OpVariable %input_ptr Input
       %zero = OpConstant %int 0
     %origin = OpConstantComposite %int2 %zero %zero
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %id = OpLoad %uint3 %group
          %x = OpCompositeExtract %uint %id 0
     %source = OpAccessChain %image_ptr %images %x
     %loaded = OpLoad %image %source
      %texel = OpImageFetch %uint4 %loaded %origin Lod %zero
     %target = OpAccessChain %texel_ptr %out %zero %x
               OpStore %target %texel
               OpReturn
               OpFunctionEnd
)";
}

void Require(bool value, const std::string& what) {
	if (value) return;
	std::fprintf(stderr, "%s: %s\n", g_name, what.c_str());
	std::abort();
}

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
	Require(result.result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result.result));
	return std::move(result.value);
}

void Checked(vk::Result result, const char* what) {
	Require(result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result));
}

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	Require(required == nullptr || std::strcmp(required, "1") != 0, std::string("ANYPS5_REQUIRE_VULKAN=1 and ") + why);
	std::printf("%s: skipped, %s\n", g_name, why);
	return 77;
}

std::vector<uint32_t> AssembleFetchShader() {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		messages += message;
		messages += '\n';
	});
	std::vector<uint32_t> spirv;
	Require(tools.Assemble(FetchShader(), &spirv), "the fetch shader does not assemble: " + messages);
	Require(tools.Validate(spirv), "the fetch shader is not valid SPIR-V: " + messages);
	return spirv;
}

struct DrawDevice {
	vk::Instance       instance;
	vk::PhysicalDevice physical;
	vk::Device         device;
	vk::Queue          queue;
	uint32_t           family = 0;
	std::string        name;
};

int OpenDevice(DrawDevice& out) {
	static const vk::detail::DynamicLoader loader;
	if (!loader.success()) return NoDevice("the Vulkan loader is missing");
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo application;
	application.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info;
	instance_info.pApplicationInfo = &application;
	const auto created             = vk::createInstance(instance_info);
	if (created.result != vk::Result::eSuccess) return NoDevice("no Vulkan 1.3 instance");
	out.instance = created.value;
	dispatch.init(out.instance);

	for (const auto candidate: Checked(out.instance.enumeratePhysicalDevices(), "device enumeration")) {
		const auto extensions = Checked(candidate.enumerateDeviceExtensionProperties(), "extension enumeration");
		const bool has_robustness2 = std::ranges::any_of(extensions, [](const vk::ExtensionProperties& extension) {
			return std::strcmp(extension.extensionName.data(), VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) == 0;
		});
		if (!has_robustness2) continue;
		vk::PhysicalDeviceRobustness2FeaturesEXT robustness2;
		vk::PhysicalDeviceFeatures2              features;
		features.pNext = &robustness2;
		candidate.getFeatures2(&features);
		const auto properties = candidate.getProperties();
		const auto families   = candidate.getQueueFamilyProperties();
		const auto compute    = std::ranges::find_if(families, [](const auto& queue) {
			return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eCompute);
		});
		if (!robustness2.nullDescriptor || !features.features.shaderSampledImageArrayDynamicIndexing ||
		    compute == families.end() || properties.apiVersion < VK_API_VERSION_1_1 ||
		    properties.limits.maxPerStageDescriptorSampledImages < ElementCount ||
		    properties.limits.maxDescriptorSetSampledImages < ElementCount)
			continue;
		const bool software = properties.deviceType == vk::PhysicalDeviceType::eCpu;
		if (out.physical && !software) continue;
		out.physical = candidate;
		out.family   = static_cast<uint32_t>(compute - families.begin());
		if (software) break;
	}
	if (!out.physical) return NoDevice("no device with nullDescriptor and 540 sampled images per stage");

	const float               priority = 1.0f;
	vk::DeviceQueueCreateInfo queue_info;
	queue_info.queueFamilyIndex = out.family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	vk::PhysicalDeviceRobustness2FeaturesEXT robustness2;
	robustness2.nullDescriptor = VK_TRUE;
	vk::PhysicalDeviceFeatures2 features;
	features.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
	features.pNext                                           = &robustness2;
	const char* const    extension                           = VK_EXT_ROBUSTNESS_2_EXTENSION_NAME;
	vk::DeviceCreateInfo device_info;
	device_info.pNext                   = &features;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = 1;
	device_info.ppEnabledExtensionNames = &extension;
	out.device                          = Checked(out.physical.createDevice(device_info), "device creation");
	vk::PhysicalDeviceDriverProperties driver;
	vk::PhysicalDeviceProperties2      properties;
	properties.pNext = &driver;
	out.physical.getProperties2(&properties);
	out.name = std::string(properties.properties.deviceName.data()) + " (" + driver.driverName.data() + ", " +
	           driver.driverInfo.data() + ")";
	dispatch.init(out.device);
	out.queue = out.device.getQueue(out.family, 0);
	return 0;
}

struct Expected {
	bool     null  = false;
	uint32_t image = 0;
};

int DrawTables() {
	g_name           = "kyty_shim_descriptor_tables_draw";
	const auto spirv = AssembleFetchShader();
	DrawDevice gpu;
	if (const auto skipped = OpenDevice(gpu); skipped != 0) return skipped;
	const auto device = gpu.device;

	const auto memory_types = gpu.physical.getMemoryProperties();
	const auto allocate     = [&](const vk::MemoryRequirements& needs, vk::MemoryPropertyFlags flags) {
		for (uint32_t type = 0; type < memory_types.memoryTypeCount; type++) {
			if ((needs.memoryTypeBits >> type & 1u) == 0u ||
			    (memory_types.memoryTypes[type].propertyFlags & flags) != flags)
				continue;
			vk::MemoryAllocateInfo info;
			info.allocationSize  = needs.size;
			info.memoryTypeIndex = type;
			return Checked(device.allocateMemory(info), "memory allocation");
		}
		Require(false, "the device has no memory type for the test");
		return vk::DeviceMemory {};
	};

	vk::CommandPoolCreateInfo pool_info;
	pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	pool_info.queueFamilyIndex = gpu.family;
	const auto command_pool    = Checked(device.createCommandPool(pool_info), "command pool creation");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool        = command_pool;
	command_info.commandBufferCount = 1;
	const auto command = Checked(device.allocateCommandBuffers(command_info), "command allocation").front();
	const auto submit  = [&](const auto& record, const char* what) {
		Checked(command.reset(), "command reset");
		vk::CommandBufferBeginInfo begin;
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		Checked(command.begin(begin), "command begin");
		record();
		Checked(command.end(), "command end");
		vk::SubmitInfo info;
		info.commandBufferCount = 1;
		info.pCommandBuffers    = &command;
		Checked(gpu.queue.submit(1, &info, nullptr), what);
		Checked(device.waitIdle(), what);
	};

	constexpr auto                           Format = vk::Format::eR8G8B8A8Uint;
	const vk::ImageSubresourceRange          whole {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, 1u};
	std::array<vk::Image, ImageCount>        images;
	std::array<vk::DeviceMemory, ImageCount> image_memory;
	std::array<vk::ImageView, ImageCount>    views;
	for (uint32_t index = 0; index < ImageCount; index++) {
		vk::ImageCreateInfo image_info;
		image_info.imageType   = vk::ImageType::e2D;
		image_info.format      = Format;
		image_info.extent      = vk::Extent3D {1u, 1u, 1u};
		image_info.mipLevels   = 1;
		image_info.arrayLayers = 1;
		image_info.samples     = vk::SampleCountFlagBits::e1;
		image_info.usage       = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		images[index]          = Checked(device.createImage(image_info), "image creation");
		image_memory[index]    = allocate(device.getImageMemoryRequirements(images[index]), {});
		Checked(device.bindImageMemory(images[index], image_memory[index], 0), "image binding");
		vk::ImageViewCreateInfo view_info;
		view_info.image            = images[index];
		view_info.viewType         = vk::ImageViewType::e2D;
		view_info.format           = Format;
		view_info.subresourceRange = whole;
		views[index]               = Checked(device.createImageView(view_info), "view creation");
	}
	const auto clear_images = [&] {
		for (uint32_t index = 0; index < ImageCount; index++) {
			vk::ImageMemoryBarrier barrier;
			barrier.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
			barrier.oldLayout           = vk::ImageLayout::eUndefined;
			barrier.newLayout           = vk::ImageLayout::eTransferDstOptimal;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.image               = images[index];
			barrier.subresourceRange    = whole;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {},
			                        0, nullptr, 0, nullptr, 1, &barrier);
			vk::ClearColorValue color {};
			color.setUint32(Texels[index]);
			command.clearColorImage(images[index], vk::ImageLayout::eTransferDstOptimal, &color, 1, &whole);
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
			barrier.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
			barrier.newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
			                        {}, 0, nullptr, 0, nullptr, 1, &barrier);
		}
	};
	submit(clear_images, "the image clears");

	constexpr vk::DeviceSize OutputBytes = vk::DeviceSize {ElementCount} * 16u;
	vk::BufferCreateInfo     buffer_info;
	buffer_info.size         = OutputBytes;
	buffer_info.usage        = vk::BufferUsageFlagBits::eStorageBuffer;
	const auto output        = Checked(device.createBuffer(buffer_info), "buffer creation");
	const auto output_memory = allocate(device.getBufferMemoryRequirements(output),
	                                    vk::MemoryPropertyFlagBits::eHostVisible |
	                                        vk::MemoryPropertyFlagBits::eHostCoherent);
	Checked(device.bindBufferMemory(output, output_memory, 0), "buffer binding");
	auto* texels =
	    static_cast<uint32_t*>(Checked(device.mapMemory(output_memory, 0, VK_WHOLE_SIZE), "buffer mapping"));

	std::array<vk::DescriptorSetLayoutBinding, 2> bindings;
	bindings[0].binding         = 0;
	bindings[0].descriptorType  = vk::DescriptorType::eSampledImage;
	bindings[0].descriptorCount = ElementCount;
	bindings[0].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	bindings[1].binding         = 1;
	bindings[1].descriptorType  = vk::DescriptorType::eStorageBuffer;
	bindings[1].descriptorCount = 1;
	bindings[1].stageFlags      = vk::ShaderStageFlagBits::eCompute;
	vk::DescriptorSetLayoutCreateInfo set_layout_info;
	set_layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	set_layout_info.pBindings    = bindings.data();
	const auto set_layout = Checked(device.createDescriptorSetLayout(set_layout_info), "set layout creation");
	constexpr uint32_t                          SetCount = 12;
	const std::array<vk::DescriptorPoolSize, 2> sizes {
	    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, ElementCount * SetCount},
	    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, SetCount}};
	vk::DescriptorPoolCreateInfo descriptor_pool_info;
	descriptor_pool_info.maxSets       = SetCount;
	descriptor_pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
	descriptor_pool_info.pPoolSizes    = sizes.data();
	const auto descriptor_pool = Checked(device.createDescriptorPool(descriptor_pool_info), "descriptor pool creation");
	const auto new_set = [&] {
		vk::DescriptorSetAllocateInfo info;
		info.descriptorPool     = descriptor_pool;
		info.descriptorSetCount = 1;
		info.pSetLayouts        = &set_layout;
		return Checked(device.allocateDescriptorSets(info), "set allocation").front();
	};

	vk::PipelineLayoutCreateInfo layout_info;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts    = &set_layout;
	const auto pipeline_layout = Checked(device.createPipelineLayout(layout_info), "pipeline layout creation");
	vk::ShaderModuleCreateInfo module_info;
	module_info.codeSize = spirv.size() * sizeof(uint32_t);
	module_info.pCode    = spirv.data();
	const auto module    = Checked(device.createShaderModule(module_info), "module creation");
	vk::ComputePipelineCreateInfo pipeline_info;
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = pipeline_layout;
	const auto pipeline = Checked(device.createComputePipeline(nullptr, pipeline_info), "pipeline creation");

	const auto tables = Tables(HeadGap, MiddleGap);
	const auto info   = [&](const Expected& value) {
		return vk::DescriptorImageInfo {nullptr, value.null ? nullptr : views[value.image],
		                                vk::ImageLayout::eShaderReadOnlyOptimal};
	};
	std::vector<Expected> slot_values(SlotCount);
	for (uint32_t slot = 0; slot < SlotCount; slot++) {
		slot_values[slot] = {.null = slot % 9 == 0, .image = (slot * 7u + slot / 13u) % ImageCount};
	}
	std::vector<Expected> element_values(ElementCount);
	for (uint32_t element = 0; element < ElementCount; element++) {
		element_values[element] = {.null = false, .image = (element * 3u + 1u) % ImageCount};
	}
	std::vector<vk::DescriptorImageInfo> slot_infos(SlotCount);
	std::vector<vk::DescriptorImageInfo> element_infos(ElementCount);
	std::vector<vk::DescriptorImageInfo> rebuilt_infos(ElementCount);
	const auto refresh_infos = [&] {
		for (uint32_t slot = 0; slot < SlotCount; slot++) slot_infos[slot] = info(slot_values[slot]);
		for (uint32_t element = 0; element < ElementCount; element++) {
			element_infos[element] = info(element_values[element]);
			const auto* table      = TableOf(tables, element);
			rebuilt_infos[element] =
			    table == nullptr ? element_infos[element] : slot_infos[table->first_slot + element - table->base];
		}
	};
	refresh_infos();
	const vk::DescriptorBufferInfo output_info {output, 0, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet         binding_write {};
	binding_write.dstBinding      = 0;
	binding_write.descriptorType  = vk::DescriptorType::eSampledImage;
	binding_write.descriptorCount = ElementCount;
	binding_write.pImageInfo      = element_infos.data();
	vk::WriteDescriptorSet output_write {};
	output_write.dstBinding      = 1;
	output_write.descriptorCount = 1;
	output_write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	output_write.pBufferInfo     = &output_info;

	const auto update = [&](vk::DescriptorSet set, std::vector<vk::WriteDescriptorSet> writes,
	                        std::span<const vk::WriteDescriptorSet> table_writes,
	                        std::vector<vk::CopyDescriptorSet>      copies,
	                        std::span<const vk::WriteDescriptorSet> slot_set_writes) {
		writes.push_back(output_write);
		PatchTableSetUpdate(set, writes, table_writes, copies, slot_set_writes);
		device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(),
		                            static_cast<uint32_t>(copies.size()), copies.data());
	};
	const auto fill = [&](vk::DescriptorSet set, bool writes, vk::DescriptorSet persistent,
	                      std::span<const uint8_t> live, std::span<const vk::WriteDescriptorSet> slot_set_writes,
	                      const std::string& what) {
		Fills fills;
		AppendImageTableGaps(binding_write, tables, fills.gaps);
		AppendImageTableFills(binding_write, tables, writes, persistent, slot_infos, live, fills.writes, fills.copies);
		uint64_t not_alive = 0;
		for (const auto mark: live) not_alive += mark == 0 ? 1u : 0u;
		if (writes) {
			Check(fills.Copied() == not_alive, what + ": only slots not proven alive are copied");
			Check(!fills.writes.empty(), what + ": tables are written");
		} else {
			Check(fills.writes.empty() && fills.copies.size() == tables.size(), what + ": each table is one copy");
		}
		update(set, fills.gaps, fills.writes, fills.copies, slot_set_writes);
	};

	uint32_t   runs    = 0;
	uint32_t   checked = 0;
	const auto run     = [&](vk::DescriptorSet set, const std::string& what) {
		std::memset(texels, 0xff, OutputBytes);
		const auto dispatch = [&] {
			command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
			command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, 1, &set, 0, nullptr);
			command.dispatch(ElementCount, 1, 1);
		};
		submit(dispatch, what.c_str());
		runs++;
		uint32_t wrong = 0;
		for (uint32_t element = 0; element < ElementCount; element++) {
			const auto* table = TableOf(tables, element);
			const auto  expected =
			    table == nullptr ? element_values[element] : slot_values[table->first_slot + element - table->base];
			const auto* got = texels + size_t {element} * 4u;
			const bool  ok  = expected.null ? got[0] == 0u && got[1] == 0u && got[2] == 0u &&
			                                    (got[3] == 0u || got[3] == 1u)
			                                  : std::equal(got, got + 4, Texels[expected.image].begin());
			if (!ok && wrong++ < 4) {
				std::fprintf(stderr, "%s: %s: element %u holds (%u %u %u %u), expected %s %u\n", g_name,
				             what.c_str(), element, got[0], got[1], got[2], got[3],
				             expected.null ? "null" : "image", expected.image);
			}
			checked++;
		}
		Check(wrong == 0, what + ": " + std::to_string(wrong) + " of " + std::to_string(ElementCount) +
		                      " elements differ from the host copy");
	};

	const auto rebuilt = new_set();
	{
		auto rebuilt_write       = binding_write;
		rebuilt_write.pImageInfo = rebuilt_infos.data();
		update(rebuilt, {rebuilt_write}, {}, {}, {});
	}
	run(rebuilt, "the draw that rebuilt its tables");
	const auto copy_persistent  = new_set();
	const auto write_persistent = new_set();
	{
		std::vector<vk::WriteDescriptorSet> seed_writes;
		std::vector<vk::CopyDescriptorSet>  seed_copies;
		for (const auto& table: tables) {
			AppendImageTableSeed(0, vk::DescriptorType::eSampledImage, table, false, rebuilt, copy_persistent,
			                     slot_infos, seed_writes, seed_copies);
			AppendImageTableSeed(0, vk::DescriptorType::eSampledImage, table, true, rebuilt, write_persistent,
			                     slot_infos, seed_writes, seed_copies);
		}
		device.updateDescriptorSets(static_cast<uint32_t>(seed_writes.size()), seed_writes.data(),
		                            static_cast<uint32_t>(seed_copies.size()), seed_copies.data());
	}

	const auto copied  = new_set();
	const auto written = new_set();
	const auto mixed   = new_set();
	fill(copied, false, copy_persistent, {}, {}, "copy mode");
	run(copied, "copy mode");
	fill(written, true, write_persistent, {}, {}, "write mode");
	run(written, "write mode");
	std::vector<uint8_t> live(SlotCount, 1);
	for (const uint32_t slot: {0u, 1u, 2u, 100u, 255u, 256u, 400u, 511u}) live[slot] = 0;
	for (uint32_t slot = 300; slot < 310; slot++) live[slot] = 0;
	fill(mixed, true, write_persistent, live, {}, "write mode with slots not proven alive");
	run(mixed, "write mode with slots not proven alive");

	const std::array<uint32_t, 3> changed_slots {9u, 300u, 511u};
	Require(slot_values[9].null && !slot_values[300].null && !slot_values[511].null,
	        "the changed slots are a null slot and two image slots");
	slot_values[9]   = {.null = false, .image = 1u};
	slot_values[300] = {.null = true, .image = 0u};
	slot_values[511] = {.null = false, .image = (slot_values[511].image + 2u) % ImageCount};
	refresh_infos();
	const auto slot_writes = [&](vk::DescriptorSet set) {
		std::vector<vk::WriteDescriptorSet> result;
		for (const auto slot: changed_slots) {
			const auto& table     = slot >= TableCapacity ? tables[0] : tables[1];
			auto        write     = binding_write;
			write.dstSet          = set;
			write.dstArrayElement = table.base + slot - table.first_slot;
			write.descriptorCount = 1;
			write.pImageInfo      = slot_infos.data() + slot;
			result.push_back(write);
		}
		return result;
	};
	const auto copied_again  = new_set();
	const auto written_again = new_set();
	const auto mixed_again   = new_set();
	fill(copied_again, false, copy_persistent, {}, slot_writes(copy_persistent),
	     "copy mode after a null, an image and a view-to-null change");
	run(copied_again, "copy mode after three slots changed");
	std::vector<uint8_t> live_again(SlotCount, 1);
	for (const auto slot: changed_slots) live_again[slot] = 0;
	live_again[10] = 0;
	fill(mixed_again, true, write_persistent, live_again, slot_writes(write_persistent),
	     "write mode after three slots changed, the changed slots not proven alive");
	run(mixed_again, "write mode after three slots changed, the changed slots not proven alive");
	fill(written_again, true, write_persistent, {}, {}, "write mode after three slots changed");
	run(written_again, "write mode after three slots changed");
	for (const auto set: {copied, written}) {
		const auto catch_up = slot_writes(set);
		device.updateDescriptorSets(static_cast<uint32_t>(catch_up.size()), catch_up.data(), 0, nullptr);
	}
	run(copied, "the copy mode set caught up with the three changed slots");
	run(written, "the write mode set caught up with the three changed slots");

	device.destroyPipeline(pipeline);
	device.destroyShaderModule(module);
	device.destroyPipelineLayout(pipeline_layout);
	device.destroyDescriptorPool(descriptor_pool);
	device.destroyDescriptorSetLayout(set_layout);
	device.destroyCommandPool(command_pool);
	device.unmapMemory(output_memory);
	device.destroyBuffer(output);
	device.freeMemory(output_memory);
	for (uint32_t index = 0; index < ImageCount; index++) {
		device.destroyImageView(views[index]);
		device.destroyImage(images[index]);
		device.freeMemory(image_memory[index]);
	}
	device.destroy();
	gpu.instance.destroy();
	if (g_failures != 0) return Finish();
	std::printf("%s: ok, %u dispatches of %u elements, %u checked, on %s\n", g_name, runs, ElementCount, checked,
	            gpu.name.c_str());
	return 0;
}

}

extern "C" int RunKytyDescriptorTablesTest(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "heap-exhaustion") == 0) return HeapExhaustion();
	if (argc > 1 && std::strcmp(argv[1], "draw") == 0) return DrawTables();
	if (argc > 1 && std::strcmp(argv[1], "copies") == 0) return CopiesSwitch();
	if (argc > 1 && std::strcmp(argv[1], "bad") == 0) return BadSwitch();
	if (argc > 1 && std::strcmp(argv[1], "empty") == 0) return DefaultSwitch("");
	if (argc > 1 && std::strcmp(argv[1], "one") == 0) return WritesSwitch("1");
	return HostTests();
}

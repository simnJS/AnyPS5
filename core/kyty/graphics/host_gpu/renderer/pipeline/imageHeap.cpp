#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "gpu_tiler_shaders/image_heap_fill_spv.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/guestMirror.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <xxhash.h>

namespace Libs::Graphics {
namespace {

namespace IR    = ShaderRecompiler::IR;
namespace Slots = ImageHeapSlots;

constexpr uint64_t MinTableBytes        = 64ull * 1024ull;
constexpr uint32_t LargeTableIdleFrames = 32;
constexpr uint32_t ElementMask          = (1u << IR::ImageHeapKindShift) - 1u;
constexpr uint32_t SlotsKeptFromTables  = IR::ImageHeapCapacity / 8u;
constexpr uint32_t RefillFrames         = 16;
constexpr uint32_t EvictFrames          = 64;
constexpr uint32_t SweepFrames          = 32;
constexpr uint32_t CheckReportFrames    = 300;
constexpr size_t   MaxChecks            = 256;
constexpr size_t   MaxRangesPerView     = 32;
constexpr uint32_t DirectoryGap         = 16;
constexpr size_t   MaxGpuRuns           = 64;
constexpr uint64_t StampBytes           = uint64_t {IR::ImageHeapCapacity} * sizeof(uint32_t);
static_assert(Slots::ArmShift == IR::ImageHeapKindShift);
static_assert(IR::ImageHeapCapacity - 1u <= Slots::IndexMask);

void Widen(size_t& begin, size_t& end, size_t first, size_t last) {
	const bool empty = begin == end;
	begin            = empty ? first : std::min(begin, first);
	end              = empty ? last : std::max(end, last);
}

uint64_t Probes(uint64_t size, uint64_t first, uint64_t step) {
	return size >= first + 4u ? (size - first - 4u) / step + 1u : 0u;
}

std::unique_ptr<Buffer> SlotBuffer(RenderContext& context, uint64_t size) {
	return std::make_unique<Buffer>(
	    context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc |
	        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
	    std::bit_ceil(size + 4u));
}

IR::ImageResource ResolutionProfile(const IR::ImageResource& root) {
	auto resolved                       = root;
	resolved.source                     = 0;
	resolved.descriptor_index           = UINT32_MAX;
	resolved.first_use_pc               = 0;
	resolved.indirect_root              = IR::ImageResource::NoIndirectImage;
	resolved.indirect_mapping_offset    = 0;
	resolved.indirect_search_iterations = 0;
	resolved.indirect_resources.clear();
	resolved.table                      = IR::ImageResource::NoImageTable;
	resolved.table_capacity             = 0;
	resolved.table_mapping_offset       = 0;
	resolved.table_entry_mask           = 0;
	resolved.table_descriptor_base      = UINT32_MAX;
	resolved.unfiltered_access_only     = false;
	resolved.sample_access_only         = false;
	return resolved;
}

std::unique_ptr<Buffer> HostBuffer(RenderContext& context, uint64_t size) {
	return std::make_unique<Buffer>(
	    context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::Download, 0,
	    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer |
	        vk::BufferUsageFlagBits::eShaderDeviceAddress,
	    size);
}

bool SlotCheckRequested() {
	const char* value = std::getenv("KYTY_IMAGE_HEAP_SLOT_CHECK");
	return value != nullptr && value[0] == '1';
}

template <typename Native>
void GlobalBarrier(const Native& native, vk::PipelineStageFlags source, vk::AccessFlags source_access,
                   vk::PipelineStageFlags destination, vk::AccessFlags destination_access) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = source_access;
	barrier.dstAccessMask = destination_access;
	native.pipelineBarrier(source, destination, {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

constexpr auto AnyAccess = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;

}

size_t ImageHeap::KeyHash::operator()(const Key& key) const noexcept {
	return XXH3_64bits_withSeed(key.value.dwords.data(), sizeof(key.value.dwords), key.profile);
}

ImageHeap::ImageHeap(RenderContext& context): m_context(context), m_check(SlotCheckRequested()) {
	auto& graphics = context.GetGraphics();
	vk::PhysicalDeviceVulkan12Properties limits {};
	vk::PhysicalDeviceProperties2        properties {};
	properties.pNext = &limits;
	graphics.physical_device.getProperties2(&properties);
	EXIT_NOT_IMPLEMENTED(std::min({limits.maxDescriptorSetUpdateAfterBindSampledImages,
	                               limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
	                               limits.maxPerStageUpdateAfterBindResources}) <
	                     IR::ImageHeapCapacity);

	m_layout = CreateLayout(graphics.device);
	const vk::DescriptorPoolSize pool_size {vk::DescriptorType::eSampledImage,
	                                        IR::ImageHeapCapacity};
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	EXIT_IF(graphics.device.createDescriptorPool(&pool_info, nullptr, &m_pool) !=
	        vk::Result::eSuccess);
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = m_pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &m_layout;
	EXIT_IF(graphics.device.allocateDescriptorSets(&allocate, &m_set) != vk::Result::eSuccess);
	graphics.image_heap_layout = m_layout;
	m_elements.emplace_back();
	m_elements.front().freed = true;
	m_uses.emplace_back();
	m_views.push_back(nullptr);
	m_writes.push_back({0u, {nullptr, nullptr, vk::ImageLayout::eShaderReadOnlyOptimal}});
	WriteDescriptors();
}

vk::DescriptorSetLayout ImageHeap::CreateLayout(vk::Device device) {
	const vk::DescriptorBindingFlags binding_flags =
	    vk::DescriptorBindingFlagBits::ePartiallyBound |
	    vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	    vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = 1;
	flags_info.pBindingFlags = &binding_flags;
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eSampledImage,
	                                              IR::ImageHeapCapacity,
	                                              vk::ShaderStageFlagBits::eAll, nullptr};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	vk::DescriptorSetLayout layout = nullptr;
	EXIT_IF(device.createDescriptorSetLayout(&layout_info, nullptr, &layout) != vk::Result::eSuccess);
	return layout;
}

ImageHeap::~ImageHeap() {
	auto& graphics             = m_context.GetGraphics();
	graphics.image_heap_layout = nullptr;
	if (m_fill_pipeline) graphics.device.destroyPipeline(m_fill_pipeline, nullptr);
	if (m_fill_layout) graphics.device.destroyPipelineLayout(m_fill_layout, nullptr);
	graphics.device.destroyDescriptorPool(m_pool, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
}

bool ImageHeap::Acquire(const IR::ImageHeapRequest& request, IR::ImageHeapView& result) {
	KYTY_PROFILER_FUNCTION();
	const auto size = request.size & ~uint64_t {3};
	if (request.count == 0u || request.step == 0u ||
	    ((request.base | request.first | request.step) & 3u) != 0u) {
		return false;
	}
	auto memory = std::ranges::find_if(
	    m_memories, [&](const auto& candidate) { return candidate->base == request.base; });
	if (memory == m_memories.end() || ((*memory)->size != size && (*memory)->draw != m_draw)) {
		if (!m_context.IsMapped(request.base, size)) return false;
		if (memory == m_memories.end()) {
			memory = std::ranges::find_if(m_memories, [&](const auto& candidate) {
				return candidate->size == size && candidate->idle != 0u &&
				       candidate->draw <= m_suspend_draw;
			});
		}
		if (memory == m_memories.end()) {
			std::scoped_lock lock {m_mutex};
			memory = m_memories.insert(m_memories.end(), std::make_unique<Memory>());
		}
		Resize(**memory, request.base, size);
		WatchTables();
	}
	(*memory)->draw    = m_draw;
	const auto profile = Profile(request.table ? *request.root : IR::ImageHeapResource(*request.root, 0u),
	                             request.table);
	const auto bytes   = (*memory)->size;
	const auto probes  = Probes(size, request.first, request.step);
	const auto limit   = request.count < probes ? request.count : UINT64_MAX;
	const auto first   = limit == UINT64_MAX ? request.first % request.step : request.first;
	auto&      shape   = FindShape(request.step, first, profile);
	auto&      views   = (*memory)->views;
	auto       view    = std::ranges::find_if(views, [&](const auto& candidate) {
		return &candidate->shape == &shape && candidate->limit == limit;
	});
	if (view == views.end()) {
		auto created    = std::make_unique<View>(View {.memory = **memory,
		                                               .shape  = shape,
		                                               .limit  = limit,
		                                               .count  = std::min(limit, Probes(bytes, first, request.step))});
		created->id     = NewViewId(created.get());
		created->buffer = SlotBuffer(m_context, bytes);
		view            = views.insert(views.end(), std::move(created));
		if (shape.kind_keys.empty()) Seed(**view);
	}
	auto& selected = **view;
	if (selected.overflowed) return false;
	selected.arms = static_cast<uint32_t>(shape.kind_keys.size());
	if (std::ranges::find(m_active, &selected) == m_active.end()) m_active.push_back(&selected);
	result = {.slots      = selected.buffer->BufferDeviceAddress(),
	          .slot_count = static_cast<uint32_t>(bytes / 4u),
	          .kinds      = shape.kinds};
	return true;
}

uint32_t ImageHeap::Profile(const IR::ImageResource& root, bool table) {
	for (uint32_t profile = 0; profile < m_profiles.size(); ++profile) {
		if (m_table_profiles[profile] == table && m_profiles[profile] == root) return profile;
	}
	m_profiles.push_back(root);
	m_table_profiles.push_back(table);
	const auto profile = static_cast<uint32_t>(m_profiles.size() - 1u);
	m_key_profiles.push_back(profile);
	if (auto resolved = ResolutionProfile(root); resolved != root) {
		const auto key_profile  = Profile(resolved, table);
		m_key_profiles[profile] = key_profile;
	}
	return profile;
}

ImageHeap::Shape& ImageHeap::FindShape(uint64_t step, uint64_t first, uint32_t profile) {
	auto& shape = m_shapes[{.step = step, .first = first, .profile = profile}];
	if (shape == nullptr) {
		shape = std::make_unique<Shape>(Shape {.step = step, .first = first, .profile = profile});
	}
	return *shape;
}

ImageHeap::Shape& ImageHeap::ListShape(uint32_t profile) {
	auto& shape = m_list_shapes[profile];
	if (shape == nullptr) shape = std::make_unique<Shape>(Shape {.profile = profile});
	return *shape;
}

void ImageHeap::Seed(View& view) {
	const auto& memory = view.memory;
	m_scratch.resize(memory.size / 4u);
	if (m_scratch.empty() ||
	    !LibKernel::Memory::TryReadBufferBacking(memory.base, m_scratch.data(), memory.size)) {
		return;
	}
	const auto& shape     = view.shape;
	const auto& profile   = m_profiles[shape.profile];
	const bool  table     = m_table_profiles[shape.profile];
	const auto  key_words = profile.r128 ? 4u : 8u;
	for (uint64_t probe = 0; probe < view.count && !view.overflowed; ++probe) {
		Key key {.value = {.dword_count = 8u}, .profile = shape.profile};
		key.value.dwords = Slots::ReadKey(m_scratch, (shape.first + probe * shape.step) / 4u, key_words);
		const auto kind  = table ? IR::ImageTableHeapKind(profile, key.value)
		                         : IR::ImageHeapKind(profile, key.value);
		if (kind != 0u) Intern(view, kind, key);
	}
}

void ImageHeap::Intern(View& view, uint32_t kind, const Key& key) {
	uint32_t arm = 0;
	if (Entry(key, kind, true) == 0u ||
	    !Arm(view.shape, kind, key.value, Slots::MaxArms, arm)) {
		view.overflowed = true;
	}
}

uint32_t ImageHeap::InternList(Shape& shape, uint32_t kind, const Key& key) {
	const auto index = Entry(key, kind, false);
	uint32_t   arm   = 0;
	(void)Arm(shape, kind, key.value, UINT32_MAX, arm);
	++m_elements[index].refs;
	return arm << IR::ImageHeapKindShift | index;
}

uint32_t ImageHeap::Entry(const Key& key, uint32_t kind, bool table) {
	auto [found, inserted] = m_index.try_emplace(key, 0u);
	if (inserted) {
		const auto index = Allocate(table);
		if (index == 0u) {
			m_index.erase(found);
			m_sweep_frame = std::min(m_sweep_frame, m_frame);
			return 0u;
		}
		found->second = index;
		auto& element = m_elements[index];
		element.key   = key;
		element.kind  = kind;
		element.born  = m_frame;
		m_pending.push_back(index);
	}
	const auto index   = found->second;
	auto&      element = m_elements[index];
	if (table && !element.table) {
		element.table = true;
		if (element.resolved) Hash(index);
	}
	return index;
}

bool ImageHeap::Arm(Shape& shape, uint32_t kind, const DescriptorValue& value, uint32_t limit,
                    uint32_t& arm) {
	arm = static_cast<uint32_t>(std::ranges::find(shape.kind_keys, kind) - shape.kind_keys.begin());
	if (arm == shape.kind_keys.size()) {
		if (arm >= limit) return false;
		shape.kind_keys.push_back(kind);
		shape.kinds.push_back(value);
	} else if (m_table_profiles[shape.profile] &&
	           !IR::ImageTableHeapExemplar(m_profiles[shape.profile], shape.kinds[arm]) &&
	           IR::ImageTableHeapExemplar(m_profiles[shape.profile], value)) {
		shape.kinds[arm] = value;
	}
	return true;
}

void ImageHeap::Hash(uint32_t index) {
	auto& element = m_elements[index];
	if (element.hashed) return;
	if (m_directory.Insert(element.key.value.dwords, element.key.profile,
	                       index | element.kind << Slots::KindShift, 0u)) {
		element.hashed = true;
		return;
	}
	element.table = false;
	Invalidate(element.key.profile);
	Settle(index);
}

void ImageHeap::Settle(uint32_t index) {
	const auto& element = m_elements[index];
	if (element.freed || element.table || element.refs != 0u) return;
	Forget(index);
	Free(index);
}

void ImageHeap::Forget(uint32_t index) {
	auto& element = m_elements[index];
	if (element.state == State::Live) element.state = State::Stale;
	if (const auto found = m_index.find(element.key);
	    found != m_index.end() && found->second == index) {
		m_index.erase(found);
	}
}

void ImageHeap::Free(uint32_t index) {
	auto& element = m_elements[index];
	element.freed = true;
	auto& textures = m_context.GetTextureCache();
	if (auto* image = textures.m_slot_images.try_get(element.image);
	    image != nullptr && image->serial == element.serial) {
		image->heap_refs.fetch_sub(1u, std::memory_order_relaxed);
		image->tick_accessed_last = textures.m_gc_tick;
	}
	m_released.push_back(index);
}

void ImageHeap::Invalidate(uint32_t profile) {
	for (const auto& memory: m_memories) {
		for (const auto& view: memory->views) {
			if (view->shape.profile == profile) view->full = true;
		}
	}
}

bool ImageHeap::AcquireList(const IR::ImageHeapList& list, IR::ImageHeapView& result) {
	KYTY_PROFILER_FUNCTION();
	if (list.root == nullptr) {
		return false;
	}
	const auto profile = Profile(IR::ImageHeapResource(*list.root, 0u), false);
	auto&      shape   = ListShape(profile);
	auto&      held    = m_lists[{list.owner, list.resource}];
	held.shape         = &shape;
	m_scratch.clear();
	m_scratch.reserve(list.values.size());
	for (auto value: list.values) {
		const auto kind = IR::ImageHeapKind(m_profiles[profile], value);
		m_scratch.push_back(kind == 0u ? 0u : InternList(shape, kind, {.value = value, .profile = m_key_profiles[profile]}));
	}
	const auto slots = list.values.size() + 2u;
	for (auto slot = slots - 1u; slot < held.slots.size(); ++slot) Assign(held, slot, 0u);
	const auto previous = held.slots.size();
	held.slots.resize(slots, 0u);
	held.uses.resize(slots);
	if (previous < slots) Widen(held.dirty_begin, held.dirty_end, previous, slots);
	held.dirty_end   = std::min(held.dirty_end, slots);
	held.dirty_begin = std::min(held.dirty_begin, held.dirty_end);
	for (size_t value = 0; value < m_scratch.size(); ++value) Assign(held, value + 1u, m_scratch[value]);
	if (held.buffer == nullptr || slots * sizeof(uint32_t) > held.buffer->Size()) {
		if (held.buffer != nullptr && m_context.GetCommandScheduler().Active()) {
			m_context.GetCommandScheduler().DeferOperation([buffer = std::move(held.buffer)] {});
		}
		held.buffer = SlotBuffer(m_context, (slots - 1u) * sizeof(uint32_t));
		Widen(held.dirty_begin, held.dirty_end, 0u, slots);
	}
	if (std::ranges::find(m_active_lists, &held) == m_active_lists.end()) {
		m_active_lists.push_back(&held);
	}
	result = {.slots      = held.buffer->BufferDeviceAddress(),
	          .slot_count = static_cast<uint32_t>(slots - 1u),
	          .kinds      = shape.kinds};
	return true;
}

void ImageHeap::Assign(List& list, size_t slot, uint32_t entry) {
	const auto previous = std::exchange(list.slots[slot], entry);
	if (previous != entry) {
		Unlink(list, slot, previous);
		Link(list, slot, entry);
	}
	Release(previous);
	if (previous != entry) Widen(list.dirty_begin, list.dirty_end, slot, slot + 1u);
}

void ImageHeap::Link(List& list, size_t slot, uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u) return;
	auto& element_uses = m_uses[index];
	list.uses[slot]    = static_cast<uint32_t>(element_uses.size());
	element_uses.push_back({.list = &list, .slot = static_cast<uint32_t>(slot)});
}

void ImageHeap::Unlink(List& list, size_t slot, uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u) return;
	auto&      element_uses = m_uses[index];
	const auto position     = list.uses[slot];
	const auto moved        = element_uses.back();
	element_uses[position]  = moved;
	moved.list->uses[moved.slot] = position;
	element_uses.pop_back();
}

void ImageHeap::Upload(CommandBuffer& buffer, const std::vector<uint32_t>& slots, Buffer& device,
                       size_t& dirty_begin, size_t& dirty_end) {
	if (dirty_begin == dirty_end) return;
	const auto words  = std::span(slots).subspan(dirty_begin, dirty_end - dirty_begin);
	auto&      stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = stream.Copy(words.data(), words.size_bytes(), sizeof(uint32_t));
	device.CopyFrom(buffer, stream, offset, dirty_begin * sizeof(uint32_t), words.size_bytes());
	dirty_begin = dirty_end = 0;
}

void ImageHeap::Release(uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u || --m_elements[index].refs != 0u) return;
	Settle(index);
}

void ImageHeap::Retire() {
	if (m_released.empty()) return;
	const auto released = [&](uint32_t index) { return m_elements[index].freed; };
	std::erase_if(m_held, released);
	std::erase_if(m_dcc, released);
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	for (const auto index: m_released) m_retired.emplace_back(tick, index);
	m_released.clear();
}

void ImageHeap::Mark(uint32_t index, State state) {
	auto& element = m_elements[index];
	element.state = state;
	if (!m_uses[index].empty()) m_marked.push_back(index);
	if (const auto found = m_index.find(element.key);
	    found != m_index.end() && found->second == index) {
		m_index.erase(found);
	}
	if (!element.table) return;
	element.table = false;
	if (element.hashed) {
		element.hashed = false;
		(void)m_directory.Erase(element.key.value.dwords, element.key.profile);
	}
	if (state == State::Stale) {
		m_requeue.push_back(index);
	} else if (m_directory.Insert(element.key.value.dwords, element.key.profile, 0u,
	                              Slots::NegativeFlag | Slots::DeadFlag)) {
		m_dead.emplace_back(element.key.value.dwords, element.key.profile);
	}
	Invalidate(element.key.profile);
	Settle(index);
}

template <typename Fn>
bool ImageHeap::ForEachChanged(Fn&& fn) {
	bool found = false;
	for (uint32_t index = 1; !m_changes.empty() && index < m_elements.size(); ++index) {
		const auto& element = m_elements[index];
		if (!element.freed && (element.table || element.refs != 0u) && element.image &&
		    std::ranges::binary_search(m_changes, element.serial, {}, &Image::HeapChange::serial)) {
			fn(index);
			found = true;
		}
	}
	return found;
}

void ImageHeap::Repoint() {
	std::ranges::sort(m_marked);
	m_marked.erase(std::ranges::unique(m_marked).begin(), m_marked.end());
	m_visits.clear();
	for (const auto index: m_marked) {
		if (m_elements[index].state == State::Live) continue;
		m_visits.insert(m_visits.end(), m_uses[index].begin(), m_uses[index].end());
	}
	m_marked.clear();
	if (m_visits.empty()) return;
	size_t order = 0;
	for (auto& [owner, list]: m_lists) list.order = order++;
	std::ranges::sort(m_visits, {}, [](const Use& use) { return std::pair {use.list->order, use.slot}; });
	for (const auto& use: m_visits) {
		auto&       list    = *use.list;
		const auto& element = m_elements[list.slots[use.slot] & ElementMask];
		if (element.state == State::Buried) {
			Assign(list, use.slot, 0u);
		} else if (element.state == State::Stale) {
			const auto key  = element.key;
			const auto kind = element.kind;
			Assign(list, use.slot, InternList(*list.shape, kind, key));
		}
	}
}

void ImageHeap::Requeue() {
	for (const auto index: m_requeue) {
		const auto key  = m_elements[index].key;
		const auto kind = m_elements[index].kind;
		(void)Entry(key, kind, true);
	}
	m_requeue.clear();
}

void ImageHeap::TakeChanges() {
	m_changes.clear();
	{
		std::scoped_lock lock {Image::heap_mutex};
		std::swap(m_changes, Image::heap_changes);
	}
	std::ranges::sort(m_changes, {}, &Image::HeapChange::serial);
	const auto [first, last] = std::ranges::unique(m_changes, {}, &Image::HeapChange::serial);
	m_changes.erase(first, last);
}

ImageId ImageHeap::CurrentImage(const Image::HeapChange& change) {
	auto&            textures = m_context.GetTextureCache();
	Common::ScopedLock lock {textures.m_lock};
	ImageId          current;
	textures.ForEachImageWithAddress(change.data.address, [&](ImageId id, const Image& image) {
		if (image.serial == change.serial && !image.depth_id) current = id;
	});
	return current;
}

uint32_t ImageHeap::Allocate(bool may_fail) {
	auto& scheduler = m_context.GetCommandScheduler();
	const auto reclaim = [&] {
		while (!m_retired.empty() && scheduler.IsFree(m_retired.front().first)) {
			m_free.push_back(m_retired.front().second);
			m_retired.pop_front();
		}
	};
	reclaim();
	if (m_free.empty() && m_elements.size() + SlotsKeptFromTables >= IR::ImageHeapCapacity) {
		Retire();
		reclaim();
		if (m_free.empty() && m_elements.size() == IR::ImageHeapCapacity && !m_retired.empty() &&
		    m_retired.front().first < scheduler.CurrentTick()) {
			scheduler.Wait(m_retired.front().first);
			reclaim();
		}
	}
	if (may_fail && IR::ImageHeapCapacity - m_elements.size() + m_free.size() <= SlotsKeptFromTables) return 0u;
	if (m_free.empty()) {
		if (m_elements.size() == IR::ImageHeapCapacity) {
			if (!may_fail) {
				static std::atomic_bool reported {false};
				if (!reported.exchange(true, std::memory_order_relaxed)) {
					LOGF("Image heap: all %u slots are in use; a heap view that needs another one is released and its "
					     "draws bind their textures without the heap\n",
					     IR::ImageHeapCapacity);
				}
			}
			return 0u;
		}
		m_elements.emplace_back();
		m_uses.emplace_back();
		return static_cast<uint32_t>(m_elements.size() - 1u);
	}
	const auto index = m_free.back();
	m_free.pop_back();
	m_elements[index] = {};
	return index;
}

void ImageHeap::Resize(Memory& memory, uint64_t base, uint64_t size) {
	{
		std::scoped_lock lock {m_mutex};
		memory.base = base;
		memory.size = size;
	}
	memory.fresh   = true;
	memory.gpu_all = 0;
	memory.written.clear();
	memory.gpu_written.clear();
	memory.gpu_carried.clear();
	auto& scheduler = m_context.GetCommandScheduler();
	for (auto& view: memory.views) {
		view->count = std::min(view->limit, Probes(size, view->shape.first, view->shape.step));
		view->full  = true;
		view->pending.clear();
		if ((size / 4u + 1u) * sizeof(uint32_t) > view->buffer->Size()) {
			if (scheduler.Active()) {
				scheduler.DeferOperation([buffer = std::move(view->buffer)] {});
			}
			view->buffer  = SlotBuffer(m_context, size);
			view->clear   = true;
			view->written = 0;
		}
	}
}

uint32_t ImageHeap::NewViewId(View* view) {
	if (m_free_views.empty()) {
		m_views.push_back(view);
		return static_cast<uint32_t>(m_views.size() - 1u);
	}
	const auto id = m_free_views.back();
	m_free_views.pop_back();
	m_views[id] = view;
	return id;
}

ImageHeap::View* ImageHeap::ViewById(uint32_t id) const {
	return id < m_views.size() ? m_views[id] : nullptr;
}

void ImageHeap::ServiceMisses() {
	auto& scheduler = m_context.GetCommandScheduler();
	while (!m_rings.empty() && m_rings.front().tick != scheduler.CurrentTick() &&
	       scheduler.IsFree(m_rings.front().tick)) {
		auto ring = std::move(m_rings.front());
		m_rings.pop_front();
		Service(ring);
		m_free_rings.push_back(std::move(ring.buffer));
	}
}

void ImageHeap::Service(Ring& ring) {
	auto& buffer = *ring.buffer;
	buffer.Invalidate(0, Slots::RingBytes);
	const auto* bytes = buffer.Mapped().data();
	uint32_t    count = 0;
	std::memcpy(&count, bytes, sizeof(count));
	m_check_stats.misses += count;
	for (uint32_t at = 0; at < std::min(count, Slots::RingCapacity); ++at) {
		Slots::Miss miss;
		std::memcpy(&miss, bytes + Slots::RingHeaderBytes + uint64_t {at} * sizeof(miss), sizeof(miss));
		if (miss.profile >= m_profiles.size()) continue;
		auto* view = ViewById(miss.view);
		if (view != nullptr && (view->shape.profile != miss.profile || view->overflowed)) view = nullptr;
		const Key key {.value = {.dwords = miss.words, .dword_count = 8u}, .profile = miss.profile};
		uint32_t  kind = 0;
		if (const auto found = m_index.find(key); found != m_index.end()) {
			kind = m_elements[found->second].kind;
			(void)Entry(key, kind, true);
		} else {
			auto        value   = key.value;
			const auto& profile = m_profiles[miss.profile];
			kind = m_table_profiles[miss.profile] ? IR::ImageTableHeapKind(profile, value)
			                                      : IR::ImageHeapKind(profile, value);
			if (kind == 0u) {
				(void)m_directory.Insert(miss.words, miss.profile, 0u, Slots::NegativeFlag);
				continue;
			}
			if (Entry(key, kind, true) == 0u) {
				if (view != nullptr) view->overflowed = true;
				continue;
			}
		}
		if (view == nullptr) continue;
		uint32_t arm = 0;
		if (!Arm(view->shape, kind, key.value, Slots::MaxArms, arm)) {
			view->overflowed = true;
			continue;
		}
		AddProbe(*view, miss.slot);
	}
	if (count > Slots::RingCapacity) {
		++m_check_stats.overflows;
		for (const auto id: ring.views) {
			if (auto* view = ViewById(id)) view->full = true;
		}
	}
}

void ImageHeap::AddProbe(View& view, uint32_t slot) {
	const auto offset = uint64_t {slot} * 4u;
	if (offset < view.shape.first || (offset - view.shape.first) % view.shape.step != 0u) return;
	const auto probe = (offset - view.shape.first) / view.shape.step;
	if (probe >= view.count) return;
	view.pending.push_back({static_cast<uint32_t>(probe), static_cast<uint32_t>(probe + 1u)});
}

void ImageHeap::AddBytes(View& view, uint64_t begin, uint64_t end) {
	const auto&    shape = view.shape;
	const uint64_t bytes = m_profiles[shape.profile].r128 ? 16u : 32u;
	const uint64_t first = begin < shape.first + bytes ? 0u : (begin - shape.first - bytes) / shape.step + 1u;
	const uint64_t last  = end <= shape.first ? 0u
	                                          : std::min<uint64_t>((end - shape.first - 1u) / shape.step + 1u,
	                                                               view.count);
	if (first < last) view.pending.push_back({static_cast<uint32_t>(first), static_cast<uint32_t>(last)});
}

void ImageHeap::SyncMemory(Memory& memory) {
	if (memory.prepared == m_draw) return;
	memory.prepared = m_draw;
	auto& cache = m_context.GetBufferCache();
	m_context.CollectHotPageWrites(memory.base, memory.size);
	cache.ConsumeCpuWrites(CpuWriteConsumer::ImageHeapTables, memory.base, memory.size,
	                       [&](uint64_t run, uint64_t bytes) {
		                       memory.written.emplace_back(run, run + bytes);
		                       const auto first = Common::AlignDown(run, TRACKER_PAGE_SIZE);
		                       const auto last  = Common::AlignUp(run + bytes, TRACKER_PAGE_SIZE);
		                       for (const auto& other: m_memories) {
			                       if (other.get() != &memory && other->base < last &&
			                           first < other->base + other->size) {
				                       other->written.emplace_back(first, last);
			                       }
		                       }
	                       });
	cache.EnsureRangeContents(memory.base, memory.size);
}

void ImageHeap::UpdateProbes(Memory& memory) {
	if (memory.synced == m_draw) return;
	memory.synced = m_draw;
	const bool whole = memory.fresh || memory.gpu_all != 0u;
	memory.fresh = false;
	if (memory.gpu_all != 0u) --memory.gpu_all;
	m_runs.clear();
	m_runs.insert(m_runs.end(), memory.written.begin(), memory.written.end());
	memory.written.clear();
	m_runs.insert(m_runs.end(), memory.gpu_carried.begin(), memory.gpu_carried.end());
	m_runs.insert(m_runs.end(), memory.gpu_written.begin(), memory.gpu_written.end());
	std::swap(memory.gpu_carried, memory.gpu_written);
	memory.gpu_written.clear();
	for (const auto& view: memory.views) {
		if (view->overflowed) continue;
		if (whole || m_frame - view->filled >= RefillFrames) view->full = true;
		if (!view->full) {
			for (const auto& [begin, end]: m_runs) {
				const auto first = std::max(begin, memory.base);
				const auto last  = std::min(end, memory.base + memory.size);
				if (first < last) AddBytes(*view, first - memory.base, last - memory.base);
			}
		}
	}
}

void ImageHeap::RecordFills(CommandBuffer& buffer) {
	TakeGpuWrites();
	m_jobs.clear();
	m_groups.clear();
	m_cleared.clear();
	auto&       cache  = m_context.GetBufferCache();
	const auto* mirror = cache.GetGuestMirror();
	for (auto* view: m_active) {
		if (view->overflowed) continue;
		auto& memory = view->memory;
		UpdateProbes(memory);
		if (!view->full && view->pending.empty()) continue;
		Slots::Job job {.table       = mirror->DeviceAddress() + (memory.base - mirror->Base()),
		                .slots       = view->buffer->BufferDeviceAddress(),
		                .table_words = static_cast<uint32_t>(memory.size / 4u),
		                .first_word  = static_cast<uint32_t>(view->shape.first / 4u),
		                .step_words  = static_cast<uint32_t>(view->shape.step / 4u),
		                .count       = static_cast<uint32_t>(view->count),
		                .profile     = view->shape.profile,
		                .key_words   = m_profiles[view->shape.profile].r128 ? 4u : 8u,
		                .view        = view->id,
		                .arm_count   = std::min(view->arms, Slots::MaxArms)};
		std::copy_n(view->shape.kind_keys.begin(), job.arm_count, job.kinds.begin());
		if (view->clear) {
			m_cleared.push_back(view);
			view->clear = false;
		}
		if (view->full) {
			job.begin     = 0u;
			job.end       = std::max(job.count, view->written);
			view->written = job.count;
			view->filled  = m_frame;
			Slots::AddJob(m_jobs, m_groups, job);
		} else {
			auto& ranges = view->pending;
			std::ranges::sort(ranges, {}, &Range::begin);
			m_ranges.clear();
			for (const auto& range: ranges) {
				if (!m_ranges.empty() && range.begin <= m_ranges.back().end) {
					m_ranges.back().end = std::max(m_ranges.back().end, range.end);
				} else {
					m_ranges.push_back(range);
				}
			}
			if (m_ranges.size() > MaxRangesPerView) {
				m_ranges.front().end = m_ranges.back().end;
				m_ranges.resize(1u);
			}
			for (const auto& range: m_ranges) {
				job.begin = range.begin;
				job.end   = std::min(range.end, job.count);
				Slots::AddJob(m_jobs, m_groups, job);
			}
		}
		view->full = false;
		view->pending.clear();
	}
	const bool snapshot = StepSweep();
	if (m_jobs.empty() && m_cleared.empty() && !m_directory.Dirty() && !snapshot &&
	    !(m_check && !m_active.empty())) {
		return;
	}
	buffer.EndRendering();
	PrepareGpu(buffer);
	for (auto* view: m_cleared) view->buffer->Fill(0u, view->buffer->Size(), 0u);
	FlushDirectory(buffer);
	if (snapshot) {
		const bool first = m_sweep == Sweep::Snapshot;
		Snapshot(buffer, first ? m_sweep_first : m_sweep_second);
		if (!first) m_sweep = Sweep::Confirm;
	}
	Dispatch(buffer);
	if (m_check) RecordChecks(buffer);
}

void ImageHeap::PrepareGpu(CommandBuffer& buffer) {
	(void)buffer;
	if (m_fill_pipeline) return;
	auto&      graphics  = m_context.GetGraphics();
	auto&      scheduler = m_context.GetCommandScheduler();
	const auto flags     = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc |
	                   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	m_buckets = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, flags,
	                                     uint64_t {Slots::BucketCount} * sizeof(Slots::Bucket));
	m_used    = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, flags, StampBytes);
	m_used->Fill(0u, StampBytes, 0u);
	const vk::PushConstantRange  push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Slots::Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout_info, nullptr, &m_fill_layout),
	                     "create image heap fill pipeline layout");
	const auto                        module = CompileSPV(IMAGE_HEAP_FILL_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_fill_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_fill_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create image heap fill pipeline");
	SetVulkanObjectNameF(graphics.device, m_fill_pipeline, "Kyty.ImageHeapFill");
}

void ImageHeap::FlushDirectory(CommandBuffer& buffer) {
	if (!m_directory.Dirty()) return;
	auto&                                      stream  = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto                                 buckets = m_directory.Buckets();
	std::vector<std::pair<uint32_t, uint32_t>> runs;
	m_directory.TakeDirty([&](uint32_t first, uint32_t count) {
		if (!runs.empty() && first <= runs.back().second + DirectoryGap) {
			runs.back().second = first + count;
			return;
		}
		runs.emplace_back(first, first + count);
	});
	uint64_t total = 0;
	for (const auto& [first, end]: runs) total += uint64_t {end - first} * sizeof(Slots::Bucket);
	const auto [mapped, reserved] = stream.Map(total, 16);
	EXIT_IF(mapped == nullptr);
	std::vector<vk::BufferCopy> copies;
	copies.reserve(runs.size());
	uint64_t at = 0;
	for (const auto& [first, end]: runs) {
		const auto bytes = uint64_t {end - first} * sizeof(Slots::Bucket);
		std::memcpy(mapped + at, buckets.data() + first, bytes);
		copies.push_back({reserved + at, uint64_t {first} * sizeof(Slots::Bucket), bytes});
		at += bytes;
	}
	stream.Commit();
	const auto native = buffer.Handle();
	GlobalBarrier(native, vk::PipelineStageFlagBits::eAllCommands, AnyAccess, vk::PipelineStageFlagBits::eTransfer,
	              vk::AccessFlagBits::eTransferWrite);
	native.copyBuffer(stream.Handle(), m_buckets->Handle(), static_cast<uint32_t>(copies.size()), copies.data());
	GlobalBarrier(native, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite,
	              vk::PipelineStageFlagBits::eAllCommands, AnyAccess);
}

ImageHeap::Ring& ImageHeap::CurrentRing() {
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	if (m_rings.empty() || m_rings.back().tick != tick) {
		Ring ring {.tick = tick};
		if (m_free_rings.empty()) {
			ring.buffer = HostBuffer(m_context, Slots::RingBytes);
		} else {
			ring.buffer = std::move(m_free_rings.back());
			m_free_rings.pop_back();
		}
		ring.buffer->Fill(0u, Slots::RingHeaderBytes, 0u);
		m_rings.push_back(std::move(ring));
	}
	return m_rings.back();
}

void ImageHeap::Dispatch(CommandBuffer& buffer) {
	if (m_jobs.empty()) return;
	auto&      stream             = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto job_bytes          = m_jobs.size() * sizeof(Slots::Job);
	const auto group_bytes        = m_groups.size() * sizeof(uint32_t);
	const auto [mapped, reserved] = stream.Map(job_bytes + group_bytes, 16);
	EXIT_IF(mapped == nullptr);
	std::memcpy(mapped, m_jobs.data(), job_bytes);
	std::memcpy(mapped + job_bytes, m_groups.data(), group_bytes);
	stream.Commit();
	auto&       ring = CurrentRing();
	Slots::Push push {.jobs    = stream.BufferDeviceAddress() + reserved,
	                  .groups  = stream.BufferDeviceAddress() + reserved + job_bytes,
	                  .buckets = m_buckets->BufferDeviceAddress(),
	                  .ring    = ring.buffer->BufferDeviceAddress(),
	                  .used    = m_used->BufferDeviceAddress(),
	                  .frame   = m_frame};
	const auto native = buffer.Handle();
	GlobalBarrier(native, vk::PipelineStageFlagBits::eAllCommands, AnyAccess,
	              vk::PipelineStageFlagBits::eComputeShader,
	              vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
	native.bindPipeline(vk::PipelineBindPoint::eCompute, m_fill_pipeline);
	const auto total = static_cast<uint32_t>(m_groups.size());
	for (uint32_t offset = 0; offset < total; offset += Slots::MaxGroups) {
		push.group_offset = offset;
		native.pushConstants(m_fill_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
		native.dispatch(std::min(Slots::MaxGroups, total - offset), 1, 1);
	}
	GlobalBarrier(native, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite,
	              vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	              AnyAccess | vk::AccessFlagBits::eHostRead);
	for (const auto& job: m_jobs) {
		if (ring.views.empty() || ring.views.back() != job.view) ring.views.push_back(job.view);
	}
}

void ImageHeap::Snapshot(CommandBuffer& buffer, std::unique_ptr<Buffer>& target) {
	if (target == nullptr) target = HostBuffer(m_context, StampBytes);
	const auto native = buffer.Handle();
	GlobalBarrier(native, vk::PipelineStageFlagBits::eAllCommands, AnyAccess, vk::PipelineStageFlagBits::eTransfer,
	              vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
	const vk::BufferCopy copy {0, 0, StampBytes};
	native.copyBuffer(m_used->Handle(), target->Handle(), 1, &copy);
	GlobalBarrier(native, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite,
	              vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	              AnyAccess | vk::AccessFlagBits::eHostRead);
	m_sweep_tick = m_context.GetCommandScheduler().CurrentTick();
}

bool ImageHeap::StepSweep() {
	auto& scheduler = m_context.GetCommandScheduler();
	switch (m_sweep) {
		case Sweep::Idle:
			if (m_frame < m_sweep_frame) return false;
			PurgeDirectory();
			m_sweep    = Sweep::Snapshot;
			m_sweep_at = m_frame;
			return true;
		case Sweep::Snapshot: {
			if (m_sweep_tick == scheduler.CurrentTick() || !scheduler.IsFree(m_sweep_tick)) return false;
			m_sweep_first->Invalidate(0, StampBytes);
			m_stamps.resize(IR::ImageHeapCapacity);
			std::memcpy(m_stamps.data(), m_sweep_first->Mapped().data(), StampBytes);
			std::vector<uint32_t> watched(m_held);
			watched.insert(watched.end(), m_dcc.begin(), m_dcc.end());
			std::ranges::sort(watched);
			m_candidates.clear();
			for (uint32_t index = 1; index < m_elements.size(); ++index) {
				auto& element = m_elements[index];
				if (element.freed || !element.table || !element.hashed || element.refs != 0u ||
				    element.evicting || element.born + EvictFrames >= m_sweep_at ||
				    m_stamps[index] + EvictFrames >= m_sweep_at ||
				    std::ranges::binary_search(watched, index)) {
					continue;
				}
				element.evicting = true;
				element.hashed   = false;
				(void)m_directory.Erase(element.key.value.dwords, element.key.profile);
				if (const auto found = m_index.find(element.key);
				    found != m_index.end() && found->second == index) {
					m_index.erase(found);
				}
				m_candidates.push_back(index);
			}
			if (m_candidates.empty()) {
				m_sweep       = Sweep::Idle;
				m_sweep_frame = m_frame + SweepFrames;
				return false;
			}
			m_sweep = Sweep::Removed;
			return true;
		}
		case Sweep::Removed: return true;
		case Sweep::Confirm: {
			if (m_sweep_tick == scheduler.CurrentTick() || !scheduler.IsFree(m_sweep_tick)) return false;
			m_sweep_second->Invalidate(0, StampBytes);
			const auto* second = reinterpret_cast<const uint32_t*>(m_sweep_second->Mapped().data());
			for (const auto index: m_candidates) {
				auto& element = m_elements[index];
				if (!element.evicting) continue;
				element.evicting = false;
				if (element.freed || !element.table) continue;
				uint32_t stamp = 0;
				std::memcpy(&stamp, second + index, sizeof(stamp));
				if (stamp == m_stamps[index]) {
					element.table = false;
					Settle(index);
					continue;
				}
				if (m_index.try_emplace(element.key, index).second) {
					Hash(index);
					continue;
				}
				element.table = false;
				Invalidate(element.key.profile);
				Settle(index);
			}
			m_candidates.clear();
			m_sweep       = Sweep::Idle;
			m_sweep_frame = m_frame + SweepFrames;
			return false;
		}
	}
	return false;
}

void ImageHeap::PurgeDirectory() {
	for (const auto& [words, profile]: m_dead) {
		if (const auto* bucket = m_directory.Find(words, profile);
		    bucket != nullptr && (bucket->flags & Slots::DeadFlag) != 0u) {
			(void)m_directory.Erase(words, profile);
		}
	}
	m_dead.clear();
	if (m_directory.Negative() <= Slots::BucketCount / 8u) return;
	m_directory.Rebuild([](const Slots::Bucket& bucket) { return (bucket.flags & Slots::NegativeFlag) == 0u; },
	                    [&](const Slots::Bucket& bucket) {
		                    const auto index   = bucket.value & Slots::IndexMask;
		                    auto&      element = m_elements[index];
		                    element.hashed     = false;
		                    element.table      = false;
		                    Invalidate(element.key.profile);
		                    Settle(index);
	                    });
}

void ImageHeap::RecordChecks(CommandBuffer& buffer) {
	auto& cache = m_context.GetBufferCache();
	for (auto* view: m_active) {
		if (view->overflowed || view->checked == m_frame || m_checks.size() >= MaxChecks) continue;
		const auto& memory = view->memory;
		if (cache.HasGpuDirtyBytes(memory.base, memory.size)) continue;
		m_scratch.resize(memory.size / 4u);
		if (m_scratch.empty() ||
		    !LibKernel::Memory::TryReadBufferBacking(memory.base, m_scratch.data(), memory.size)) {
			continue;
		}
		view->checked = m_frame;
		Slots::Job job {.table_words = static_cast<uint32_t>(memory.size / 4u),
		                .first_word  = static_cast<uint32_t>(view->shape.first / 4u),
		                .step_words  = static_cast<uint32_t>(view->shape.step / 4u),
		                .count       = static_cast<uint32_t>(view->count),
		                .profile     = view->shape.profile,
		                .key_words   = m_profiles[view->shape.profile].r128 ? 4u : 8u,
		                .arm_count   = std::min(view->arms, Slots::MaxArms)};
		std::copy_n(view->shape.kind_keys.begin(), job.arm_count, job.kinds.begin());
		Check check {.first = job.first_word, .step = job.step_words};
		check.expected.resize(job.count);
		for (uint32_t probe = 0; probe < job.count; ++probe) {
			check.expected[probe] = Slots::ExpectedSlot(m_directory, m_scratch, job, probe).slot;
		}
		const auto bytes = (memory.size / 4u + 1u) * sizeof(uint32_t);
		check.buffer     = HostBuffer(m_context, bytes);
		const auto native = buffer.Handle();
		GlobalBarrier(native, vk::PipelineStageFlagBits::eAllCommands, AnyAccess, vk::PipelineStageFlagBits::eTransfer,
		              vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite);
		const vk::BufferCopy copy {0, 0, bytes};
		native.copyBuffer(view->buffer->Handle(), check.buffer->Handle(), 1, &copy);
		GlobalBarrier(native, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite,
		              vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
		              AnyAccess | vk::AccessFlagBits::eHostRead);
		check.tick = m_context.GetCommandScheduler().CurrentTick();
		m_checks.push_back(std::move(check));
	}
}

void ImageHeap::FinishChecks() {
	auto& scheduler = m_context.GetCommandScheduler();
	while (!m_checks.empty() && m_checks.front().tick != scheduler.CurrentTick() &&
	       scheduler.IsFree(m_checks.front().tick)) {
		auto check = std::move(m_checks.front());
		m_checks.pop_front();
		const auto words = check.buffer->Size() / sizeof(uint32_t);
		check.buffer->Invalidate(0, check.buffer->Size());
		const auto* slots = reinterpret_cast<const uint32_t*>(check.buffer->Mapped().data());
		std::vector<uint8_t> probed(words, uint8_t {0});
		for (size_t probe = 0; probe < check.expected.size(); ++probe) {
			const auto slot = uint64_t {check.first} + probe * check.step;
			if (slot >= words) break;
			probed[slot] = 1u;
			uint32_t value = 0;
			std::memcpy(&value, slots + slot, sizeof(value));
			if (value == check.expected[probe]) continue;
			++(value == 0u ? m_check_stats.late : m_check_stats.mismatches);
		}
		for (size_t slot = 0; slot < words; ++slot) {
			uint32_t value = 0;
			std::memcpy(&value, slots + slot, sizeof(value));
			m_check_stats.extras += probed[slot] == 0u && value != 0u ? 1u : 0u;
		}
		++m_check_stats.views;
		m_check_stats.slots += check.expected.size();
	}
	if (m_frame >= m_check_stats.reported + CheckReportFrames) {
		m_check_stats.reported = m_frame;
		std::fprintf(stderr,
		             "[kyty-image-heap] slot check: %" PRIu64 " views, %" PRIu64 " slots, %" PRIu64
		             " mismatched, %" PRIu64 " late, %" PRIu64 " extra, %" PRIu64 " misses, %" PRIu64
		             " ring overflows, %zu directory entries\n",
		             m_check_stats.views, m_check_stats.slots, m_check_stats.mismatches, m_check_stats.late,
		             m_check_stats.extras, m_check_stats.misses, m_check_stats.overflows, m_directory.Size());
	}
}

void ImageHeap::Rebind(RenderExecutor& executor) {
	KYTY_PROFILER_FUNCTION();
	auto& textures = m_context.GetTextureCache();
	ServiceMisses();
	if (m_check) FinishChecks();
	TakeChanges();
	std::erase_if(m_changes, [&](const Image::HeapChange& change) {
		const auto id = CurrentImage(change);
		if (id) {
			textures.UpdateImage(id);
			const auto& image = textures.m_slot_images[id];
			if (const auto stencil = textures.FindStencilPlane(image, image.info.stencil)) {
				textures.UpdateImage(stencil);
			}
			m_away.emplace_back(change.serial, id);
		}
		return static_cast<bool>(id);
	});
	bool stale = ForEachChanged([&](uint32_t index) { Mark(index, State::Stale); });
	const auto resolve = [&](const Element& element) {
		const auto& profile = m_profiles[element.key.profile];
		if (m_table_profiles[element.key.profile]) {
			return executor.ResolveTexture(IR::ImageTableHeapResource(profile, element.key.value),
			                               element.key.value, true);
		}
		return executor.ResolveTexture(IR::ImageHeapResource(profile, element.kind), element.key.value);
	};
	const auto retry = [&](uint32_t index) {
		const auto& element = m_elements[index];
		if (element.state != State::Live || element.freed) return true;
		const auto binding = resolve(element);
		if (binding.image_id == element.image &&
		    (!element.image || textures.m_slot_images[element.image].serial == element.serial)) {
			return !element.image && binding.desc.info.data.Empty();
		}
		Mark(index, State::Stale);
		stale = true;
		return true;
	};
	std::erase_if(m_dcc, retry);
	if (const auto serial = Image::serials.load(std::memory_order_relaxed);
	    m_held_serial != serial) {
		m_held_serial = serial;
		std::erase_if(m_held, retry);
	}
	if (stale) Repoint();
	Requeue();
	for (auto* view: m_active) SyncMemory(view->memory);
	for (const auto index: m_pending) {
		if (m_elements[index].freed) continue;
		auto binding = resolve(m_elements[index]);
		auto& element = m_elements[index];
		if (binding.image_id) {
			auto& image        = textures.GetImage(binding.image_id);
			image.heap_refs.fetch_add(1u, std::memory_order_relaxed);
			element.image      = binding.image_id;
			element.serial     = image.serial;
			binding.layout     = image.ReadLayout();
			binding.image_view = textures.FindTexture(binding.image_id, binding.desc);
			if (binding.desc.info.metadata.kind == ImageMetadataKind::Dcc) m_dcc.push_back(index);
			m_away.emplace_back(image.serial, binding.image_id);
		} else if (binding.declined) {
			m_declined.push_back(index);
		} else if (!binding.desc.info.data.Empty()) {
			m_held.push_back(index);
		}
		m_writes.push_back({index, {nullptr, binding.image_view,
		                             binding.image_view ? binding.layout : vk::ImageLayout::eGeneral}});
		element.resolved = true;
		if (element.table) Hash(index);
	}
	m_pending.clear();
	if (!m_declined.empty()) {
		for (const auto index: m_declined) Mark(index, State::Buried);
		m_declined.clear();
		Repoint();
	}
	WriteDescriptors();
}

void ImageHeap::WriteDescriptors() {
	if (m_writes.empty()) return;
	std::vector<vk::WriteDescriptorSet> writes(m_writes.size());
	for (size_t write = 0; write < m_writes.size(); ++write) {
		writes[write].dstSet          = m_set;
		writes[write].dstArrayElement = m_writes[write].first;
		writes[write].descriptorCount = 1;
		writes[write].descriptorType  = vk::DescriptorType::eSampledImage;
		writes[write].pImageInfo      = &m_writes[write].second;
	}
	m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()),
	                                                    writes.data(), 0, nullptr);
	m_writes.clear();
}

void ImageHeap::Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point,
                       vk::PipelineLayout layout) {
	KYTY_PROFILER_FUNCTION();
	RecordFills(buffer);
	for (auto* list: m_active_lists) {
		Upload(buffer, list->slots, *list->buffer, list->dirty_begin, list->dirty_end);
	}
	m_active_lists.clear();
	m_active.clear();
	Retire();
	{
		std::scoped_lock lock {Image::heap_mutex};
		m_changes.assign(Image::heap_changes.begin(), Image::heap_changes.end());
	}
	std::ranges::sort(m_changes, {}, &Image::HeapChange::serial);
	std::erase_if(m_changes, [&](const Image::HeapChange& change) {
		const auto id = CurrentImage(change);
		if (id) m_away.emplace_back(change.serial, id);
		return static_cast<bool>(id);
	});
	ForEachChanged([&](uint32_t index) {
		m_away.emplace_back(m_elements[index].serial, m_elements[index].image);
	});
	std::ranges::sort(m_away);
	m_away.erase(std::ranges::unique(m_away).begin(), m_away.end());
	auto& textures = m_context.GetTextureCache();
	std::erase_if(m_away, [&](const std::pair<uint64_t, ImageId>& away) {
		auto* image = textures.m_slot_images.try_get(away.second);
		if (image == nullptr || image->serial != away.first ||
		    image->heap_refs.load(std::memory_order_relaxed) == 0u) {
			return true;
		}
		if (image->binding.is_target || image->binding.shader_write ||
		    image->binding.force_general) {
			return false;
		}
		image->Transit(image->ReadLayout(), vk::AccessFlagBits2::eShaderRead, std::nullopt,
		               buffer.Handle());
		return true;
	});
	buffer.Handle().bindDescriptorSets(bind_point, layout, IR::ImageHeapSet, 1, &m_set, 0,
	                                   nullptr);
}

void ImageHeap::UnmapMemory(uint64_t address, uint64_t size) {
	TakeChanges();
	const auto other = [&](const Image::HeapChange& change) {
		return change.data.Empty() || !ImageRangeOverlaps(change.data, {address, size});
	};
	{
		std::scoped_lock lock {Image::heap_mutex};
		std::ranges::copy_if(m_changes, std::back_inserter(Image::heap_changes), other);
	}
	std::erase_if(m_changes, other);
	std::scoped_lock lock {m_mutex};
	if (std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		    if (address >= memory->base + memory->size || memory->base >= address + size) {
			    return false;
		    }
		    Drop(*memory);
		    return true;
	    }) != 0u) {
		WatchTables();
	}
	if (ForEachChanged([&](uint32_t index) { Mark(index, State::Buried); })) Repoint();
}

void ImageHeap::RunGarbageCollector() {
	++m_frame;
	std::scoped_lock lock {m_mutex};
	if (std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		    memory->idle = memory->draw > m_suspend_draw ? 0u : memory->idle + 1u;
		    if (memory->idle < (memory->size < MinTableBytes ? 1u : LargeTableIdleFrames)) {
			    return false;
		    }
		    Drop(*memory);
		    return true;
	    }) != 0u) {
		WatchTables();
	}
	m_suspend_draw = m_draw;
}

void ImageHeap::WatchTables() {
	m_watched.clear();
	for (const auto& memory: m_memories) m_watched.emplace_back(memory->base, memory->size);
	m_context.GetBufferCache().WatchHeapTables(m_watched);
}

void ImageHeap::TakeGpuWrites() {
	m_context.GetBufferCache().TakeHeapTableWrites([&](uint64_t begin, uint64_t end) {
		for (const auto& memory: m_memories) {
			const auto first = std::max(begin, memory->base);
			const auto last  = std::min(end, memory->base + memory->size);
			if (first >= last) continue;
			if (memory->gpu_written.size() >= MaxGpuRuns) {
				memory->gpu_all = 2u;
				memory->gpu_written.clear();
				continue;
			}
			memory->gpu_written.emplace_back(first, last);
		}
	});
}

void ImageHeap::Drop(Memory& memory) {
	auto& scheduler = m_context.GetCommandScheduler();
	for (auto& view: memory.views) {
		std::erase(m_active, view.get());
		m_views[view->id] = nullptr;
		m_free_views.push_back(view->id);
		if (scheduler.Active()) {
			scheduler.DeferOperation([buffer = std::move(view->buffer)] {});
		}
	}
}

}

#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <iterator>
#include <type_traits>
#include <xxhash.h>

namespace Libs::Graphics {
namespace {

namespace IR = ShaderRecompiler::IR;

constexpr uint64_t MinTableBytes        = 64ull * 1024ull;
constexpr uint32_t LargeTableIdleFrames = 32;
constexpr uint32_t ElementMask          = (1u << IR::ImageHeapKindShift) - 1u;
constexpr uint32_t SlotsKeptFromTables  = IR::ImageHeapCapacity / 8u;

template <typename Pred, typename Fn>
void ForEachRun(size_t begin, size_t end, Pred&& pred, Fn&& fn) {
	while (begin < end) {
		if (!pred(begin)) {
			++begin;
			continue;
		}
		auto run = begin + 1u;
		while (run < end && pred(run)) ++run;
		fn(begin, run);
		begin = run;
	}
}

void Widen(size_t& begin, size_t& end, size_t first, size_t last) {
	const bool empty = begin == end;
	begin            = empty ? first : std::min(begin, first);
	end              = empty ? last : std::max(end, last);
}

uint64_t Probes(uint64_t size, uint64_t first, uint64_t step) {
	return size >= first + 4u ? (size - first - 4u) / step + 1u : 0u;
}

uint64_t Hash(const IR::DescriptorValue& value) {
	return XXH3_64bits(value.dwords.data(), sizeof(value.dwords));
}

size_t CountUnique(std::vector<IR::DescriptorValue>& values) {
	std::ranges::sort(values, {}, [](const IR::DescriptorValue& value) {
		return std::tie(value.dword_count, value.dwords);
	});
	return static_cast<size_t>(std::ranges::unique(values).begin() - values.begin());
}

std::unique_ptr<Buffer> SlotBuffer(RenderContext& context, uint64_t size) {
	return std::make_unique<Buffer>(
	    context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer |
	        vk::BufferUsageFlagBits::eShaderDeviceAddress,
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

std::pair<uint64_t, uint64_t> PageRange(uint64_t base, uint64_t size, size_t begin, size_t end) {
	const auto origin = Common::AlignDown(base, TRACKER_PAGE_SIZE);
	return {std::max(base, origin + begin * TRACKER_PAGE_SIZE),
	        std::min(base + size, origin + end * TRACKER_PAGE_SIZE)};
}

}

size_t ImageHeap::KeyHash::operator()(const Key& key) const noexcept {
	return XXH3_64bits_withSeed(key.value.dwords.data(), sizeof(key.value.dwords), key.profile);
}

ImageHeap::ImageHeap(RenderContext& context, PageManager& page_manager)
    : m_context(context), m_page_manager(page_manager) {
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
	m_uses.emplace_back();
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
	for (const auto& memory: m_memories) Unwatch(*memory);
	auto& graphics             = m_context.GetGraphics();
	graphics.image_heap_layout = nullptr;
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
	if (memory != m_memories.end() && (*memory)->size != size && (*memory)->draw == m_draw) return false;
	if (memory == m_memories.end() || (*memory)->size != size) {
		if (!m_context.IsMapped(request.base, size)) return false;
		if (memory == m_memories.end()) {
			memory = std::ranges::find_if(m_memories, [&](const auto& candidate) {
				return candidate->size == size && candidate->idle != 0u &&
				       candidate->draw <= m_suspend_draw && candidate->dead.empty();
			});
		}
		if (memory == m_memories.end()) {
			std::scoped_lock lock {m_mutex};
			memory = m_memories.insert(m_memories.end(), std::make_unique<Memory>());
		}
		Resize(**memory, request.base, size);
	}
	if ((*memory)->draw != m_draw) {
		if (!Refresh(**memory)) return false;
		(*memory)->draw = m_draw;
	}
	const auto profile = Profile(request.table ? *request.root : IR::ImageHeapResource(*request.root, 0u),
	                             request.table);
	auto&      views  = (*memory)->views;
	const auto bytes  = (*memory)->size;
	const auto probes = Probes(size, request.first, request.step);
	const auto limit  = request.count < probes ? request.count : UINT64_MAX;
	const auto first = limit == UINT64_MAX ? request.first % request.step : request.first;
	auto       view  = std::ranges::find_if(views, [&](const auto& candidate) {
		return candidate->step == request.step && candidate->first == first &&
		       candidate->limit == limit && candidate->profile == profile;
	});
	if (view == views.end()) {
		const auto count   = std::min(limit, Probes(bytes, first, request.step));
		auto       created = std::make_unique<View>(View {.memory  = **memory,
		                                                  .step    = request.step,
		                                                  .first   = first,
		                                                  .count   = count,
		                                                  .limit   = limit,
		                                                  .profile = profile});
		view = views.insert(views.end(), std::move(created));
	}
	auto& selected = **view;
	const bool rebuild = selected.overflowed || selected.buffer == nullptr;
	if (selected.changed_begin != selected.changed_end) selected.demand = {};
	if (selected.overflowed && !selected.released) ReleaseView(selected);
	if (rebuild || selected.changed_begin != selected.changed_end) {
		const auto begin = rebuild ? 0u : selected.changed_begin;
		const auto end   = rebuild ? bytes : selected.changed_end;
		selected.changed_begin = selected.changed_end = 0;
		if (!UpdateProbes(selected, begin, end)) {
			if (!rebuild) selected.demand = {};
			ReleaseView(selected);
			selected.overflowed = true;
			return false;
		}
		selected.overflowed = selected.released = false;
	}
	if (std::ranges::find(m_active, &selected) == m_active.end()) m_active.push_back(&selected);
	result = {.slots      = selected.buffer->BufferDeviceAddress(),
	          .slot_count = static_cast<uint32_t>(selected.slots.size() - 1u),
	          .kinds      = selected.kinds};
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

bool ImageHeap::AcquireList(const IR::ImageHeapList& list, IR::ImageHeapView& result) {
	KYTY_PROFILER_FUNCTION();
	if (list.root == nullptr) {
		return false;
	}
	const auto* table = list.table;
	size_t first = 1, step = 1, count = list.values.size();
	if (table != nullptr) {
		if ((table->offset & 31u) != 0 || table->mapping.empty() ||
		    table->mapping[0] != table->mapping.size() - 1u) return false;
		first = table->offset / 4u;
		step = 8;
		count = table->mapping[0];
		if (count == 0 || first + count * uint64_t {8} > uint64_t {IR::MaxIndirectDescriptorProbes} * 8u ||
		    std::ranges::any_of(std::span(table->mapping).subspan(1), [&](uint32_t index) {
			    return index >= table->slots.size();
		    })) return false;
	}
	const auto profile = Profile(table != nullptr ? *list.root : IR::ImageHeapResource(*list.root, 0u), table != nullptr);
	const ListKey key {list.owner, list.resource};
	auto memory = std::ranges::find_if(m_list_memories, [&](const auto& candidate) {
		return candidate->list_owner == key && candidate->views.front()->profile == profile;
	});
	if (memory == m_list_memories.end()) {
		auto created = std::make_unique<Memory>();
		created->list_owner = key;
		created->views.push_back(
		    std::make_unique<View>(View {.memory = *created, .profile = profile}));
		memory = m_list_memories.insert(m_list_memories.end(), std::move(created));
	}
	auto& view = *(*memory)->views.front();
	const auto slots = first + (count == 0 ? 0 : (count - 1u) * step + 1u) + 1u;
	if (table != nullptr) {
		size_t total = slots;
		for (const auto& [other_key, item]: m_lists) {
			if (item.view != nullptr && m_table_profiles[item.view->profile] && other_key != key) {
				total += item.slots.capacity();
			}
		}
		if (total > 4u * 1024u * 1024u) return false;
	}
	m_scratch.assign(slots, 0u);
	for (size_t index = 0; index < count; ++index) {
		auto value = table != nullptr ? table->slots[table->mapping[index + 1u]] : list.values[index];
		const auto kind = table != nullptr ? IR::ImageTableHeapKind(m_profiles[profile], value)
		                                   : IR::ImageHeapKind(m_profiles[profile], value);
		if (kind == 0u) continue;
		const auto element = Intern(view, kind, {.value = value, .profile = m_key_profiles[profile]}, m_scratch);
		if (table != nullptr && element == 0u) {
			for (const auto pending: m_scratch) Release(pending);
			return false;
		}
		m_scratch[first + index * step] = element;
	}
	auto& held = m_lists[key];
	held.view = &view;
	for (auto slot = slots - 1u; slot < held.slots.size(); ++slot) Assign(held, slot, 0u);
	const auto previous = held.slots.size();
	held.slots.resize(slots, 0u);
	held.uses.resize(slots);
	if (previous < slots) Widen(held.dirty_begin, held.dirty_end, previous, slots);
	held.dirty_end   = std::min(held.dirty_end, slots);
	held.dirty_begin = std::min(held.dirty_begin, held.dirty_end);
	for (size_t slot = 0; slot < m_scratch.size(); ++slot) Assign(held, slot, m_scratch[slot]);
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
	          .kinds      = view.kinds};
	return true;
}

bool ImageHeap::Refresh(Memory& memory) {
	const bool gpu_dirty = m_context.GetBufferCache().HasGpuDirtyBytes(memory.base, memory.size);
	const auto pages     = memory.dirty_pages.size();
	m_runs.clear();
	{
		std::scoped_lock lock {m_mutex};
		ForEachRun(
		    0, pages, [&](size_t page) { return memory.dirty_pages[page] != 0u; },
		    [&](size_t begin, size_t end) {
			    std::fill(memory.dirty_pages.begin() + begin, memory.dirty_pages.begin() + end,
			              uint8_t {0});
			    Watch<true>(memory, begin, end);
			    m_runs.emplace_back(begin, end);
		    });
	}
	if (gpu_dirty) m_runs.assign(1u, {0u, pages});
	for (size_t run = 0; run < m_runs.size(); ++run) {
		const auto [address, end] = PageRange(memory.base, memory.size, m_runs[run].first,
		                                      m_runs[run].second);
		m_scratch.resize((end - address) / 4u);
		if (!LibKernel::Memory::TryReadBufferBacking(address, m_scratch.data(), end - address)) {
			std::scoped_lock lock {m_mutex};
			for (; run < m_runs.size(); ++run) {
				MarkDirty(memory, m_runs[run].first, m_runs[run].second);
			}
			return false;
		}
		const auto words = std::span(memory.words).subspan((address - memory.base) / 4u,
		                                                   m_scratch.size());
		size_t     first = 0;
		size_t     last  = words.size();
		while (first < last && words[first] == m_scratch[first]) ++first;
		while (first < last && words[last - 1u] == m_scratch[last - 1u]) --last;
		if (first == last) continue;
		std::copy(m_scratch.begin() + first, m_scratch.begin() + last, words.begin() + first);
		const auto offset = address - memory.base;
		for (auto& view: memory.views) {
			Widen(view->changed_begin, view->changed_end, offset + first * 4u, offset + last * 4u);
		}
	}
	return true;
}

void ImageHeap::MarkDirty(Memory& memory, size_t begin, size_t end) {
	ForEachRun(
	    begin, end, [&](size_t page) { return memory.dirty_pages[page] == 0u; },
	    [&](size_t first, size_t last) {
		    std::fill(memory.dirty_pages.begin() + first, memory.dirty_pages.begin() + last,
		              uint8_t {1});
		    Watch<false>(memory, first, last);
	    });
}

template <bool track>
void ImageHeap::Watch(const Memory& memory, size_t begin, size_t end) {
	const auto [address, limit] = PageRange(memory.base, memory.size, begin, end);
	m_page_manager.UpdatePageWatchers<track>(address, limit - address);
}

void ImageHeap::Unwatch(const Memory& memory) {
	ForEachRun(
	    0, memory.dirty_pages.size(), [&](size_t page) { return memory.dirty_pages[page] == 0u; },
	    [&](size_t begin, size_t end) { Watch<false>(memory, begin, end); });
}

bool ImageHeap::InvalidateMemory(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_mutex};
	bool             watched = false;
	for (auto& memory: m_memories) {
		const auto origin = Common::AlignDown(memory->base, TRACKER_PAGE_SIZE);
		const auto begin  = std::max(address, origin);
		const auto end    = std::min(address + size,
		                             origin + memory->dirty_pages.size() * TRACKER_PAGE_SIZE);
		if (begin >= end) continue;
		watched = true;
		MarkDirty(*memory, (begin - origin) / TRACKER_PAGE_SIZE,
		          (end - 1u - origin) / TRACKER_PAGE_SIZE + 1u);
	}
	return watched;
}

bool ImageHeap::UpdateProbes(View& view, uint64_t begin, uint64_t end) {
	const uint64_t bytes = m_profiles[view.profile].r128 ? 16u : 32u;
	const auto first = begin < view.first + bytes ? 0u : (begin - view.first - bytes) / view.step + 1u;
	const auto last = std::min(view.count, end > view.first ? (end - view.first - 1u) / view.step + 1u : 0u);
	const auto count = last - std::min(first, last);
	const bool table = m_table_profiles[view.profile];
	const bool check = count > FreeSlots(table);
	if (check) {
		const auto available = ReclaimSlots(table);
		if (!CouldFit(view.demand, available)) return false;
		if (count > available) {
			std::vector<DescriptorValue> absent;
			for (auto probe = first; probe < last; ++probe) {
				const auto [key, kind] = ReadProbe(view, probe, false);
				if (kind != 0u && !m_index.contains(key)) absent.push_back(key.value);
			}
			const auto required = CountUnique(absent);
			if (required > available) {
				view.demand = {m_insertions, required};
				return false;
			}
		}
	}
	view.demand = {};
	if (view.buffer == nullptr) {
		view.slots.resize(view.memory.words.size() + 1u);
		view.uses.resize(view.slots.size());
		view.buffer = SlotBuffer(m_context, view.memory.size);
		view.dirty_end = view.slots.size();
	}
	if (check) m_scratch.resize(count);
	for (auto probe = first; probe < last; ++probe) {
		const auto [key, kind] = ReadProbe(view, probe, true);
		const auto pending     = check ? std::span(m_scratch).first(probe - first) : std::span<uint32_t> {};
		const auto entry       = kind == 0u ? 0u : Intern(view, kind, key, pending);
		if (check) {
			m_scratch[probe - first] = entry;
		} else {
			Assign(view, (view.first + probe * view.step) / 4u, entry);
		}
	}
	if (check) {
		for (auto probe = first; probe < last; ++probe) {
			Assign(view, (view.first + probe * view.step) / 4u, m_scratch[probe - first]);
		}
	}
	return true;
}

std::pair<ImageHeap::Key, uint32_t> ImageHeap::ReadProbe(View& view, uint64_t probe, bool update_dead) {
	const auto& profile = m_profiles[view.profile];
	const auto& words   = view.memory.words;
	const auto  offset  = static_cast<uint32_t>((view.first + probe * view.step) / 4u);
	Key         key {.value = {.dword_count = 8u}, .profile = m_key_profiles[view.profile]};
	std::copy_n(words.begin() + offset,
	            std::min<size_t>(profile.r128 ? 4u : 8u, words.size() - offset),
	            key.value.dwords.begin());
	auto& dead = view.memory.dead;
	if (const auto found = dead.find(offset); found != dead.end()) {
		if (found->second == Hash(key.value)) return {key, 0u};
		if (update_dead) dead.erase(found);
	}
	const auto kind = m_table_profiles[view.profile] ? IR::ImageTableHeapKind(profile, key.value)
	                                                 : IR::ImageHeapKind(profile, key.value);
	return {key, kind};
}

void ImageHeap::ReleaseView(View& view) {
	if (view.released) return;
	view.released = true;
	for (uint64_t probe = 0; !view.slots.empty() && probe < view.count; ++probe) {
		Assign(view, (view.first + probe * view.step) / 4u, 0u);
	}
}

void ImageHeap::Assign(View& view, size_t slot, uint32_t entry) {
	const auto previous = std::exchange(view.slots[slot], entry);
	if (previous != entry) {
		Unlink(view, slot, previous);
		Link(view, slot, entry);
	}
	Release(previous);
	if (previous != entry) Widen(view.dirty_begin, view.dirty_end, slot, slot + 1u);
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

template <typename Owner>
void ImageHeap::Link(Owner& owner, size_t slot, uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u) return;
	auto& element_uses = m_uses[index];
	owner.uses[slot]   = static_cast<uint32_t>(element_uses.size());
	Use use {.slot = static_cast<uint32_t>(slot)};
	if constexpr (std::is_same_v<Owner, View>) {
		use.view = &owner;
	} else {
		use.list = &owner;
	}
	element_uses.push_back(use);
}

template <typename Owner>
void ImageHeap::Unlink(Owner& owner, size_t slot, uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u) return;
	auto&      element_uses = m_uses[index];
	const auto position     = owner.uses[slot];
	const auto moved        = element_uses.back();
	element_uses[position]  = moved;
	(moved.view != nullptr ? moved.view->uses : moved.list->uses)[moved.slot] = position;
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

void ImageHeap::ShiftArms(View& view, uint32_t arm, std::span<uint32_t> pending) {
	const auto shift = [&](std::vector<uint32_t>& slots, size_t& dirty_begin, size_t& dirty_end) {
		bool updated = false;
		for (size_t slot = 0; slot < slots.size(); ++slot) {
			const auto entry = slots[slot];
			if (entry != 0u && (entry >> IR::ImageHeapKindShift) >= arm) {
				slots[slot] = entry + (1u << IR::ImageHeapKindShift);
				Widen(dirty_begin, dirty_end, slot, slot + 1u);
				updated = true;
			}
		}
		return updated;
	};
	if (shift(view.slots, view.dirty_begin, view.dirty_end) && view.buffer != nullptr &&
	    std::ranges::find(m_active, &view) == m_active.end()) {
		m_active.push_back(&view);
	}
	for (auto& [list_key, list]: m_lists) {
		if (list.view == &view && shift(list.slots, list.dirty_begin, list.dirty_end) &&
		    list.buffer != nullptr && std::ranges::find(m_active_lists, &list) == m_active_lists.end()) {
			m_active_lists.push_back(&list);
		}
	}
	for (auto& entry: pending) {
		if (entry != 0u && (entry >> IR::ImageHeapKindShift) >= arm) {
			entry += 1u << IR::ImageHeapKindShift;
		}
	}
}

uint32_t ImageHeap::Intern(View& view, uint32_t kind, const Key& key, std::span<uint32_t> pending) {
	const bool table           = m_table_profiles[view.profile];
	auto [found, inserted] = m_index.try_emplace(key, 0u);
	if (inserted) {
		const auto index = Allocate(table);
		if (index == 0u) {
			m_index.erase(found);
			return 0u;
		}
		++m_insertions;
		found->second = index;
		auto& element = m_elements[found->second];
		element.key   = key;
		element.kind  = kind;
		m_pending.push_back(found->second);
	}
	const auto it  = std::ranges::lower_bound(view.kind_keys, kind);
	const auto arm = static_cast<uint32_t>(it - view.kind_keys.begin());
	if (it == view.kind_keys.end() || *it != kind) {
		const bool shifted = arm < view.kind_keys.size();
		view.kind_keys.insert(it, kind);
		view.kinds.insert(view.kinds.begin() + arm, key.value);
		if (shifted) ShiftArms(view, arm, pending);
	} else if (table && !IR::ImageTableHeapExemplar(m_profiles[view.profile], view.kinds[arm]) &&
	           IR::ImageTableHeapExemplar(m_profiles[view.profile], key.value)) {
		view.kinds[arm] = key.value;
	}
	++m_elements[found->second].refs;
	return arm << IR::ImageHeapKindShift | found->second;
}

void ImageHeap::Release(uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u || --m_elements[index].refs != 0u) return;
	auto& element = m_elements[index];
	Mark(index, State::Stale);
	auto& textures = m_context.GetTextureCache();
	if (auto* image = textures.m_slot_images.try_get(element.image);
	    image != nullptr && image->serial == element.serial) {
		image->heap_refs.fetch_sub(1u, std::memory_order_relaxed);
		image->tick_accessed_last = textures.m_gc_tick;
	}
	m_released.push_back(index);
}

void ImageHeap::Retire() {
	if (m_released.empty()) return;
	const auto released = [&](uint32_t index) { return m_elements[index].refs == 0u; };
	std::erase_if(m_pending, released);
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
}

template <typename Fn>
bool ImageHeap::ForEachChanged(Fn&& fn) {
	bool found = false;
	for (uint32_t index = 1; !m_changes.empty() && index < m_elements.size(); ++index) {
		const auto& element = m_elements[index];
		if (element.refs != 0u && element.image &&
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
	for (const auto& memory: m_memories) {
		for (const auto& view: memory->views) view->order = order++;
	}
	for (auto& [owner, list]: m_lists) list.order = order++;
	std::ranges::sort(m_visits, {}, [](const Use& use) {
		return std::pair {use.view != nullptr ? use.view->order : use.list->order, use.slot};
	});
	for (const auto& use: m_visits) {
		if (use.list != nullptr) {
			auto&       list    = *use.list;
			const auto& element = m_elements[list.slots[use.slot] & ElementMask];
			if (element.state == State::Buried) {
				Assign(list, use.slot, 0u);
			} else if (element.state == State::Stale) {
				const auto key  = element.key;
				const auto kind = element.kind;
				Assign(list, use.slot, Intern(*list.view, kind, key));
			}
			continue;
		}
		auto&       view    = *use.view;
		const auto& element = m_elements[view.slots[use.slot] & ElementMask];
		if (element.state == State::Buried) {
			view.memory.dead[use.slot] = Hash(element.key.value);
			for (auto& affected: view.memory.views) affected->demand = {};
			Assign(view, use.slot, 0u);
		} else if (element.state == State::Stale) {
			const auto key = element.key;
			const auto entry = Intern(view, element.kind, key);
			if (entry == 0u) view.overflowed = true;
			Assign(view, use.slot, entry);
		}
	}
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

size_t ImageHeap::FreeSlots(bool table) const {
	const auto available = IR::ImageHeapCapacity - m_elements.size() + m_free.size();
	return table ? available - std::min<size_t>(available, SlotsKeptFromTables) : available;
}

void ImageHeap::ReclaimCompleted(uint64_t tick) {
	while (!m_retired.empty() && m_retired.front().first <= tick) {
		m_free.push_back(m_retired.front().second);
		m_retired.pop_front();
	}
}

size_t ImageHeap::ReclaimSlots(bool table) {
	Retire();
	auto& master = m_context.GetCommandScheduler().GetMasterSemaphore();
	const auto tick = master.CurrentTick();
	ReclaimCompleted(master.KnownGpuTick());
	if (!m_retired.empty() && m_retired.front().first < tick && m_reclaim_tick != tick) {
		m_reclaim_tick = tick;
		master.Refresh();
		ReclaimCompleted(master.KnownGpuTick());
	}
	return FreeSlots(table);
}

bool ImageHeap::CouldFit(const SlotDemand& demand, size_t available) const {
	return demand.required <= available || demand.required - available <= m_insertions - demand.insertions;
}

uint32_t ImageHeap::Allocate(bool may_fail) {
	auto& scheduler = m_context.GetCommandScheduler();
	const auto reclaim = [&] {
		if (!m_retired.empty() && m_retired.front().first < scheduler.CurrentTick() &&
		    scheduler.IsFree(m_retired.front().first)) {
			ReclaimCompleted(scheduler.GetMasterSemaphore().KnownGpuTick());
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
	if (may_fail && FreeSlots(true) == 0u) return 0u;
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
	const auto previous_size = memory.size;
	{
		std::scoped_lock lock {m_mutex};
		Unwatch(memory);
		memory.base = base;
		memory.size = size;
		memory.dirty_pages.assign((Common::AlignUp(base + size, TRACKER_PAGE_SIZE) -
		                           Common::AlignDown(base, TRACKER_PAGE_SIZE)) /
		                              TRACKER_PAGE_SIZE,
		                          uint8_t {1});
	}
	memory.words.resize(size / 4u);
	std::erase_if(memory.dead, [&](const auto& word) { return word.first >= size / 4u; });
	auto& scheduler = m_context.GetCommandScheduler();
	for (auto& view: memory.views) {
		const auto count = std::min(view->limit, Probes(size, view->first, view->step));
		view->demand = {};
		if (size < previous_size) Widen(view->changed_begin, view->changed_end, size, previous_size);
		for (auto probe = count; view->buffer != nullptr && probe < view->count; ++probe) {
			Assign(*view, (view->first + probe * view->step) / 4u, 0u);
		}
		view->count       = count;
		if (view->buffer == nullptr) continue;
		const auto slots  = view->slots.size();
		view->slots.resize(size / 4u + 1u);
		view->uses.resize(view->slots.size());
		view->dirty_end   = std::min(view->dirty_end, view->slots.size());
		view->dirty_begin = std::min(view->dirty_begin, view->dirty_end);
		if (view->slots.size() * sizeof(uint32_t) > view->buffer->Size()) {
			if (scheduler.Active()) {
				scheduler.DeferOperation([buffer = std::move(view->buffer)] {});
			}
			view->buffer = SlotBuffer(m_context, size);
			Widen(view->dirty_begin, view->dirty_end, 0u, view->slots.size());
		} else if (slots < view->slots.size()) {
			Widen(view->dirty_begin, view->dirty_end, slots, view->slots.size());
		}
	}
}

void ImageHeap::Rebind(RenderExecutor& executor) {
	KYTY_PROFILER_FUNCTION();
	auto& textures = m_context.GetTextureCache();
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
		if (element.state != State::Live) return true;
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
	for (const auto index: m_pending) {
		if (m_elements[index].refs == 0u) continue;
		auto binding = resolve(m_elements[index]);
		if (binding.image_id) {
			auto& image   = textures.GetImage(binding.image_id);
			auto& element = m_elements[index];
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
	for (auto* list: m_active_lists) {
		Upload(buffer, list->slots, *list->buffer, list->dirty_begin, list->dirty_end);
	}
	m_active_lists.clear();
	for (auto* view: m_active) {
		if (view->dirty_begin == view->dirty_end) continue;
		const auto slots  = std::span(view->slots).subspan(view->dirty_begin,
		                                                   view->dirty_end - view->dirty_begin);
		auto&      stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		const auto offset = stream.Copy(slots.data(), slots.size_bytes(), sizeof(uint32_t));
		view->buffer->CopyFrom(buffer, stream, offset, view->dirty_begin * sizeof(uint32_t),
		                       slots.size_bytes());
		view->dirty_begin = view->dirty_end = 0;
	}
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
	std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		if (address >= memory->base + memory->size || memory->base >= address + size) {
			return false;
		}
		Drop(*memory);
		return true;
	});
	if (ForEachChanged([&](uint32_t index) { Mark(index, State::Buried); })) Repoint();
}

void ImageHeap::RunGarbageCollector() {
	m_reclaim_tick = 0;
	(void)ReclaimSlots(true);
	std::scoped_lock lock {m_mutex};
	std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		memory->idle = memory->draw > m_suspend_draw ? 0u : memory->idle + 1u;
		if (memory->idle < (memory->size < MinTableBytes ? 1u : LargeTableIdleFrames)) {
			return false;
		}
		Drop(*memory);
		return true;
	});
	m_suspend_draw = m_draw;
}

void ImageHeap::Drop(Memory& memory) {
	auto& scheduler = m_context.GetCommandScheduler();
	Unwatch(memory);
	for (auto& view: memory.views) {
		std::erase(m_active, view.get());
		for (uint64_t probe = 0; !view->slots.empty() && probe < view->count; ++probe) {
			const auto slot = (view->first + probe * view->step) / 4u;
			Unlink(*view, slot, view->slots[slot]);
			Release(view->slots[slot]);
		}
		if (scheduler.Active()) {
			scheduler.DeferOperation([buffer = std::move(view->buffer)] {});
		}
	}
}

}

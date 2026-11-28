#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr std::array<uint32_t, 31> LoopWords {
    0xf4080600u, 0xfa000100u, 0x7e080d00u, 0x060808f0u, 0x100808ffu, 0x3c800000u, 0x7e0a02f0u, 0x340c0084u,
    0x7e100280u, 0x7e120280u, 0x7e140280u, 0x7e160280u, 0xbe830380u, 0xbf0a0203u, 0xbf84000du, 0x8f248503u,
    0xf40c0200u, 0x48000000u, 0xbf8cc07fu, 0xf09c0f08u, 0x00c20004u, 0xbf8c3f70u, 0x3a100108u, 0x3a120309u,
    0x3a14050au, 0x3a16070bu, 0x80038103u, 0xbf82fff1u, 0xe0781000u, 0x80010806u, 0xbf810000u,
};

constexpr uint64_t GuestRegion   = 0x200800000ull;
constexpr size_t   GuestBytes    = 0x40000u;
constexpr uint64_t TableAddress  = 0x200810000ull;
constexpr uint64_t TextureA      = 0x200820000ull;
constexpr uint64_t TextureB      = 0x200820100ull;
constexpr uint64_t TextureC      = 0x200820200ull;
constexpr uint64_t OutputAddress = 0x200830000ull;
constexpr uint32_t Format8888    = 56;
constexpr uint32_t Format32UInt  = 20;

int       g_failures = 0;
uint32_t* g_guest    = nullptr;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

void CheckComputeQueueKeys() {
	ShaderInit();
	std::array<uint32_t, 2> code {0u, 0xbf810000u};
	HW::ComputeShaderInfo regs {};
	regs.cs_regs.data_addr = reinterpret_cast<uintptr_t>(code.data());
	const auto key = [&](bool async, bool indirect) {
		ShaderComputeInputInfo input {};
		input.async_compute = async;
		input.dispatch_indirect_threads = indirect;
		PrepareProgram(regs, input);
		std::vector<uint32_t> words;
		BuildStageStaticKey(input, words);
		return words;
	};
	for (const auto instruction: {0xbf800000u, 0xb9031818u, 0xb9070a18u, 0xbe802102u}) {
		code[0] = instruction;
		ShaderMapUserData(regs.cs_regs.data_addr, {.type = Prospero::ShaderBinaryType::kCs,
		                                          .code_size_bytes = sizeof(code)});
		const bool shared = instruction == 0xbf800000u;
		Check((key(false, false) == key(true, false)) == shared, "compute queue dependence of the program key");
		Check(key(false, false) != key(false, true), "indirect dispatch keeps its own program key");
	}
}

using Texture = std::array<uint32_t, 8>;

Texture MakeTexture(uint64_t address, uint32_t format) {
	const auto base = address >> 8u;
	return {static_cast<uint32_t>(base), static_cast<uint32_t>((base >> 32u) & 0xffu) | (format << 20u) | (3u << 30u),
	        15u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
}

bool MapGuest() {
#if defined(_WIN32)
	g_guest = static_cast<uint32_t*>(VirtualAlloc(reinterpret_cast<void*>(GuestRegion), GuestBytes,
	                                              MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
	void* mapped = mmap(reinterpret_cast<void*>(GuestRegion), GuestBytes, PROT_READ | PROT_WRITE,
	                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
	g_guest      = mapped == MAP_FAILED ? nullptr : static_cast<uint32_t*>(mapped);
#endif
	return g_guest == reinterpret_cast<uint32_t*>(GuestRegion);
}

void Poke(uint64_t address, uint32_t value) {
	g_guest[(address - GuestRegion) / 4u] = value;
}

bool ReadMemory(void*, uint64_t address, std::span<uint32_t> values) {
	for (size_t index = 0; index < values.size(); ++index) {
		const auto at = address + index * 4u;
		values[index] = at >= GuestRegion && at + 4u <= GuestRegion + GuestBytes ? g_guest[(at - GuestRegion) / 4u] : 0u;
	}
	return true;
}

struct Plan {
	ShaderComputeInputInfo info {};
	IR::ResourcePlan       plan;
};

Plan Translate() {
	Plan result;
	auto& info              = result.info;
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	const std::vector<uint32_t> user_data(16, 0u);
	const std::vector<uint32_t> code(LoopWords.begin(), LoopWords.end());
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x9e1u;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	result.plan                = IR::ExtractResourcePlan(translated.program);
	return result;
}

struct Materialized {
	bool                       ok = false;
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
};

Materialized Materialize(const IR::ResourcePlan& plan, const std::vector<Texture>& table,
                         uint32_t stride = 0, uint32_t bounds = 3) {
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	for (uint32_t entry = 0; entry < table.size(); ++entry) {
		for (uint32_t word = 0; word < 8u; ++word) Poke(TableAddress + entry * 32u + word * 4u, table[entry][word]);
	}
	const std::array<uint32_t, 4> sampler {0x12u, 0u, 0u, 0u};
	for (uint32_t word = 0; word < 4u; ++word) Poke(TableAddress + 0x100u + word * 4u, sampler[word]);
	std::array<uint32_t, 16> user_data {};
	user_data[0] = static_cast<uint32_t>(TableAddress);
	user_data[1] = static_cast<uint32_t>(TableAddress >> 32u);
	user_data[2] = static_cast<uint32_t>(table.size());
	user_data[4] = static_cast<uint32_t>(OutputAddress);
	user_data[5] = static_cast<uint32_t>(OutputAddress >> 32u) | (stride << 16u);
	user_data[6] = stride ? 64u : 64u * 16u;
	user_data[7] = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (bounds << 28u);
	const IR::SrtRuntime runtime {.user_data = user_data, .read_specialization_memory = ReadMemory};
	Materialized         result;
	result.ok = IR::MaterializeResources(plan, runtime, result.snapshot, result.specialization);
	return result;
}

uint32_t ChildFor(const IR::ResourceSpecialization& specialization, const IR::ResourceSnapshot& snapshot,
                  uint32_t key) {
	const auto& root    = specialization.images[0];
	const auto  offset  = root.indirect_mapping_offset;
	const auto  count   = snapshot.flattened_srt[offset];
	for (uint32_t entry = 0; entry < count; ++entry) {
		if (snapshot.flattened_srt[offset + 1u + entry * 2u] != key) continue;
		const auto ordinal = snapshot.flattened_srt[offset + 2u + entry * 2u];
		if (ordinal == 0u) return 0u;
		uint32_t seen = 0;
		for (uint32_t index = 1; index < specialization.images.size(); ++index) {
			if (specialization.images[index].indirect_root == 0u && ++seen == ordinal) return index;
		}
		return UINT32_MAX;
	}
	return 0u;
}

bool SameTexture(const IR::DescriptorValue& value, const Texture& texture) {
	return std::equal(texture.begin(), texture.end(), value.dwords.begin());
}

struct TestImageHeap final: IR::ImageHeapSource {
	std::array<IR::DescriptorValue, 1> kinds;
	std::vector<uint32_t> rebound_mapping;
	bool allow_direct = false;
	bool allow_rebind = true;

	bool Acquire(const IR::ImageHeapRequest& request, IR::ImageHeapView& result) override {
		if (!allow_direct || !request.table || request.count != 3u) return false;
		result = {.slots = TableAddress, .slot_count = 17, .kinds = kinds};
		return true;
	}

	bool AcquireList(const IR::ImageHeapList& request, IR::ImageHeapView& result) override {
		if (!allow_rebind || request.table == nullptr) return false;
		rebound_mapping = request.table->mapping;
		result = {.slots = TableAddress, .slot_count = 17, .kinds = kinds};
		return true;
	}
};

void CheckImageTableLifetime(IR::ResourcePlan& plan, IR::SrtRuntime runtime) {
	std::vector<size_t> reads;
	runtime.userdata = &reads;
	runtime.read_specialization_memory = [](void* userdata, uint64_t address, std::span<uint32_t> words) {
		if (address == TableAddress) static_cast<std::vector<size_t>*>(userdata)->push_back(words.size());
		return ReadMemory(nullptr, address, words);
	};

	IR::ResourceSpecialization specialization;
	IR::ResourceSnapshot snapshot;
	const auto capture = [&] {
		reads.clear();
		const bool ok = IR::MaterializeResources(plan, runtime, snapshot, specialization);
		Check(ok && snapshot.image_tables.size() == 1, "capture materializes one image table");
		Check(reads == std::vector<size_t>{24}, "capture reads all three descriptors including their upper words");
		return ok && snapshot.image_tables.size() == 1;
	};

	plan.info.images[0].r128 = true;
	if (!capture()) return;
	auto retained = snapshot;
	const std::weak_ptr<const IR::ImageTableSnapshot> observed = retained.image_tables.front();
	const auto& first = *retained.image_tables.front();
	const auto initial_id = first.slots_id;
	const auto original = first.raw[7];
	const auto old_mapping = first.mapping;
	const auto old_slots = first.slots;
	if (!capture()) return;
	Check(snapshot.image_tables.front() == retained.image_tables.front(), "unchanged table reuses the previous snapshot");
	snapshot = {};
	if (!capture()) return;
	Check(snapshot.image_tables.front() == retained.image_tables.front(), "cache hit shares the retained image table");
	{
		auto unavailable = runtime;
		unavailable.read_specialization_memory = [](void*, uint64_t address, std::span<uint32_t> words) {
			return address != TableAddress && ReadMemory(nullptr, address, words);
		};
		auto previous = snapshot;
		IR::ResourceSpecialization unused;
		Check(!IR::MaterializeResources(plan, unavailable, previous, unused),
		      "a cached table cannot hide a rejected guest read");
	}
	for (uint32_t change = 1; change <= 8; ++change) {
		Poke(TableAddress + 7u * 4u, original ^ change);
		if (!capture()) return;
		const auto& current = *snapshot.image_tables.front();
		Check(current.slots_id != initial_id && current.raw[7] == (original ^ change),
		      "a mutation of an unused R128 word produces a fresh capture");
		Check(current.mapping == old_mapping && current.slots == old_slots,
		      "an unused R128 word preserves slots and entry order");
	}
	Check(first.raw[7] == original && first.mapping == old_mapping && first.slots == old_slots,
	      "the retained snapshot survives mutation and cache eviction");
	retained = {};
	Check(observed.expired(), "evicted table is released with its last snapshot");
	Poke(TableAddress + 7u * 4u, original);
}

void CheckImageTableWidths(IR::ResourcePlan& plan, const IR::SrtRuntime& runtime, TestImageHeap& heap, bool on) {
	const auto original = plan.info.images.front();
	const auto base = MakeTexture(TextureA, Format8888);
	auto short_only = base;
	short_only[4] = 0xe0000000u;
	auto both = base;
	both[7] = 0x1234u;
	auto long_only = MakeTexture(TextureB, Format8888);
	long_only[3] = (long_only[3] & 0x0fffffffu) | (static_cast<uint32_t>(Prospero::ImageType::kColor2DArray) << 28u);
	const std::array entries {short_only, both, long_only};
	for (size_t i = 0; i < entries.size(); ++i)
		for (size_t j = 0; j < entries[i].size(); ++j)
			Poke(TableAddress + i * 32u + j * 4u, entries[i][j]);

	const auto check = [&](const Materialized& output, uint32_t root, bool short_view) {
		if (!output.ok || root >= output.specialization.images.size()) return false;
		const auto& image = output.specialization.images[root];
		if (image.table >= output.snapshot.image_tables.size()) return false;
		const auto& table = *output.snapshot.image_tables[image.table];
		const std::vector<uint32_t> expected = short_view ? std::vector<uint32_t>{3,1,1,0} : std::vector<uint32_t>{3,0,1,2};
		return table.mapping == expected && table.slots.size() == (short_view ? 2u : 3u) &&
		       SameTexture(table.slots[1], short_view ? base : both) &&
		       (short_view || SameTexture(table.slots[2], long_only)) &&
		       table.raw.size() == 24 && table.raw[4] == short_only[4] && table.raw[15] == both[7] &&
		       image.table_mapping_offset + expected.size() <= output.snapshot.flattened_srt.size() &&
		       std::equal(expected.begin(), expected.end(), output.snapshot.flattened_srt.begin() + image.table_mapping_offset);
	};
	heap.allow_direct = false;
	heap.allow_rebind = true;
	for (const bool first_short : {true,false}) {
		Materialized first;
		auto& root = plan.info.images[0];
		root.r128 = first_short;
		first.ok = IR::MaterializeResources(plan, runtime, first.snapshot, first.specialization);
		Check(check(first, 0, first_short), "first width selects the proper image slots and words");
		root.r128 = !first_short;
		Materialized other;
		other.ok = IR::MaterializeResources(plan, runtime, other.snapshot, other.specialization);
		Check(check(other, 0, !first_short), "cross-call cache keeps the two widths distinct");
		heap.allow_direct = true;
		Materialized target;
		target.ok = IR::MaterializeResources(plan, runtime, target.snapshot, target.specialization);
		heap.allow_direct = false;
		IR::ResourceSnapshot aligned;
		heap.rebound_mapping.clear();
		if (on && first.ok && other.ok && target.ok) {
			Check(IR::RebindComputeImages(plan, runtime, other.snapshot, other.specialization, target.specialization, aligned) &&
			      !heap.rebound_mapping.empty(), "matching captured image table can bind to the heap");
			heap.rebound_mapping.clear();
			IR::RebindComputeImages(plan, runtime, first.snapshot, first.specialization, target.specialization, aligned);
			Check(heap.rebound_mapping.empty(), "captured image table with another width never reaches the heap");
		}
		first.ok = IR::MaterializeResources(plan, runtime, first.snapshot, first.specialization);
		Check(check(first, 0, !first_short), "previous snapshot cannot provide slots of another width");
	}

	CheckImageTableLifetime(plan, runtime);

	plan.info.images.push_back(original);
	for (const bool first_short : {true,false}) {
		plan.info.images[0].r128 = first_short;
		plan.info.images[1].r128 = !first_short;
		plan.info.images[1].skip_degamma = true;
		plan.info.images[1].paired_view = 0u;
		IR::CompileSrtWalker(plan);
		Materialized mixed;
		mixed.ok = IR::MaterializeResources(plan, runtime, mixed.snapshot, mixed.specialization);
		Check(check(mixed, 0, first_short) && check(mixed, 1, !first_short) && mixed.snapshot.image_tables.size() == 2,
		      "coexisting widths get distinct table mappings in either order");
	}
}

void RebindTableAndHeap(bool on) {
	auto fixture = Translate();
	auto& plan = fixture.plan;
	Check(plan.info.images.size() == 1 && !plan.info.buffers.empty(), "fixture has a source image and buffer");
	if (plan.info.images.size() != 1 || plan.info.buffers.empty()) return;
	auto& indirect = plan.descriptor_sources[plan.info.images.front().source].indirect_descriptor;
	Check(indirect.has_value(), "fixture has an indirect image source");
	if (!indirect) return;
	indirect->table_source = plan.info.buffers.front().source;
	indirect->table_array = true;
	indirect->table_mask = 0x7fu;
	indirect->table_offset = 0;

	const auto a = MakeTexture(TextureA, Format8888);
	const auto b = MakeTexture(TextureB, Format8888);
	std::fill_n(g_guest, GuestBytes / 4u, 0u);
	for (uint32_t i = 0; i < a.size(); ++i) {
		Poke(TableAddress + i * 4u, a[i]);
		Poke(TableAddress + 32u + i * 4u, b[i]);
	}
	std::array<uint32_t, 16> user {};
	user[0] = user[4] = static_cast<uint32_t>(TableAddress);
	user[1] = user[5] = static_cast<uint32_t>(TableAddress >> 32u);
	user[2] = 3;
	user[6] = 96;
	user[7] = 0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u);
	TestImageHeap heap;
	heap.kinds[0].dword_count = 8;
	std::copy(a.begin(), a.end(), heap.kinds[0].dwords.begin());
	const IR::SrtRuntime runtime {.user_data = user, .read_specialization_memory = ReadMemory, .image_heap = &heap};
	Materialized table;
	Materialized native_heap;
	table.ok = IR::MaterializeResources(plan, runtime, table.snapshot, table.specialization);
	heap.allow_direct = true;
	native_heap.ok = IR::MaterializeResources(plan, runtime, native_heap.snapshot, native_heap.specialization);
	Check(table.ok && native_heap.ok, "the fixture materializes both table and heap addressing");
	if (!table.ok || !native_heap.ok) return;
	Materialized fallback;
	fallback.ok = IR::MaterializeFallbackResources(plan, runtime, fallback.snapshot, fallback.specialization);
	Check(fallback.ok && fallback.specialization == table.specialization &&
	          fallback.snapshot.flattened_srt == table.snapshot.flattened_srt && fallback.snapshot.image_tables.size() == 1,
	      "a materialization that avoids the heap takes the table form while the heap accepts the view");
	Check(table.snapshot.image_tables.size() == 1 && native_heap.snapshot.image_tables.empty(),
	      "table and heap use distinct address representations");
	if (table.snapshot.image_tables.size() != 1 || !native_heap.snapshot.image_tables.empty()) return;
	Check(IR::CanRebindComputeImages(plan, table.specialization, native_heap.specialization) == on,
	      "ready heap shape can represent the same typed table");
	IR::ResourceSnapshot rebound;
	const bool to_heap = IR::RebindComputeImages(plan, runtime, table.snapshot, table.specialization,
	                                            native_heap.specialization, rebound);
	Check(to_heap == on, "table dispatch can rematerialize compatible heap bindings");
	if (to_heap) {
		Check(heap.rebound_mapping == std::vector<uint32_t> {3, 1, 2, 0},
		      "heap bridge preserves entry ordinals and the null slot");
		Check(rebound.flattened_srt == native_heap.snapshot.flattened_srt && rebound.images == native_heap.snapshot.images,
		      "heap bridge produces the same resource shape as normal heap materialization");
	}
	const bool to_table = IR::RebindComputeImages(plan, runtime, native_heap.snapshot, native_heap.specialization,
	                                             table.specialization, rebound);
	Check(to_table == on, "heap dispatch can rematerialize compatible table bindings");
	if (to_table) {
		Check(rebound.image_tables.size() == 1 && rebound.image_tables[0]->mapping == table.snapshot.image_tables[0]->mapping &&
		          rebound.flattened_srt == table.snapshot.flattened_srt,
		      "table bridge preserves original descriptor lookup");
	}
	auto dynamic_heap = native_heap.specialization;
	IR::ReadStridesAtRunTime(plan, dynamic_heap);
	auto& output = dynamic_heap.buffers.front();
	Check(IR::HasRuntimeStride(output.packed_stride), "the output buffer can read its live bounds");
	output.packed_stride ^= IR::ExactBufferBounds;
	Check(IR::RebindComputeImages(plan, runtime, table.snapshot, table.specialization, dynamic_heap, rebound) == on,
	      "a table dispatch can bind a ready heap with a different dynamic output bound");
	using Step = IR::ServeRefusal::Step;
	const auto refused = [&](Step step, uint32_t index = UINT32_MAX, bool after_rebind = false) {
		const auto refusal = IR::ClosestServeRefusal();
		IR::ClearServeRefusal();
		return !on || (refusal.step == step && refusal.index == index && refusal.rebound == after_rebind);
	};
	IR::ClearServeRefusal();
	auto invalid = native_heap.specialization;
	invalid.images.back().numeric_class = Prospero::TextureNumericClass::Uint;
	Check(!IR::CanRebindComputeImages(plan, table.specialization, invalid), "heap bridge rejects different child types");
	Check(refused(Step::HeapKinds, 0), "a heap with other child types is named as the refusal");
	invalid = table.specialization;
	invalid.images[0].table_entry_mask ^= 1u;
	Check(!IR::CanRebindComputeImages(plan, native_heap.specialization, invalid), "table bridge rejects different masks");
	Check(refused(Step::TableConversion, 0), "a table with another mask is named as the refusal");
	heap.allow_rebind = false;
	Check(!IR::RebindComputeImages(plan, runtime, table.snapshot, table.specialization, native_heap.specialization, rebound),
	      "heap allocation rejection never installs an incompatible snapshot");
	Check(refused(Step::TableViewCount, UINT32_MAX, true),
	      "a heap that refuses the rebound views leaves a table of another view count, named as the refusal after the rebind");
	Check(!IR::AlignResources(plan, table.snapshot, table.specialization, native_heap.specialization, nullptr),
	      "a table dispatch does not align with a heap variant of another view count");
	Check(refused(Step::TableViewCount), "a table dispatch with another view count is named as the refusal");

	auto widened = table.specialization;
	widened.images[0].table_capacity *= 2;
	Check(IR::AlignResources(plan, table.snapshot, table.specialization, widened, nullptr) == on,
	      "an otherwise identical wider table accepts the existing slots");
	Check(IR::ClosestServeRefusal().step == (on ? Step::None : Step::ResourceCount), "a variant that serves names no refusal");
	IR::ClearServeRefusal();
	widened.images[0].table_capacity = 1;
	Check(!IR::AlignResources(plan, table.snapshot, table.specialization, widened, nullptr),
	      "an undersized table rejects the existing slots");
	Check(refused(Step::TableView, 0), "an undersized table is named as the refusal");
	CheckImageTableWidths(plan, runtime, heap, on);
}

void CheckBufferBounds(const IR::ResourcePlan& plan, const Texture& texture, bool on) {
	for (const auto mode: {0u, 1u}) {
		const auto live = Materialize(plan, {texture}, 16, mode);
		const auto previous = Materialize(plan, {texture}, 16, 1u - mode);
		Check(live.ok && previous.ok && live.specialization != previous.specialization,
		      "the two live descriptors require different buffer bounds");
		if (!live.ok || !previous.ok) return;
		auto ready = previous.specialization;
		IR::ReadStridesAtRunTime(plan, ready);
		IR::ResourceSnapshot aligned;
		const bool fits = IR::AlignResources(plan, live.snapshot, live.specialization, ready, &aligned);
		Check(fits == on && (!fits || aligned.buffers == live.snapshot.buffers),
		      "a runtime stride pipeline binds the current descriptor in either bounds mode");
		Check(!IR::AlignResources(plan, live.snapshot, live.specialization, previous.specialization, nullptr),
		      "a baked stride pipeline keeps its own bounds mode");
		const auto exact = Materialize(plan, {texture}, 16, 3);
		Check(exact.ok && IR::AlignResources(plan, exact.snapshot, exact.specialization, ready, nullptr) == on,
		      "runtime stride pipelines accept exact byte range checks");
		auto swizzled = ready;
		swizzled.buffers.front().packed_stride ^= 1u << 14u;
		Check(!IR::AlignResources(plan, live.snapshot, live.specialization, swizzled, nullptr),
		      "swizzled addressing keeps its own pipeline");
		const auto zero = Materialize(plan, {texture}, 0, 0);
		Check(zero.ok && !IR::AlignResources(plan, zero.snapshot, zero.specialization, ready, nullptr),
		      "zero stride out-of-bounds accesses keep their own pipeline");
	}
}

void Generic(bool on) {
	const auto program = Translate();
	const auto& plan   = program.plan;
	Check(plan.info.images.size() == 1u, "the shader plans one image root");
	if (plan.info.images.size() != 1u) return;
	const auto* source = &plan.descriptor_sources[plan.info.images[0].source];
	Check(source->indirect_descriptor && source->indirect_descriptor->sources.empty() &&
	          !source->indirect_descriptor->key_count.IsEmpty(),
	      "the image root reads a table of descriptors with a counted key");
	if (!source->indirect_descriptor) return;
	const auto a = MakeTexture(TextureA, Format8888);
	const auto b = MakeTexture(TextureB, Format8888);
	const auto c = MakeTexture(TextureC, Format32UInt);

	CheckBufferBounds(plan, a, on);
	const auto floats = Materialize(plan, {a, b});
	const auto plain  = Materialize(plan, {a, a});
	const auto mixed  = Materialize(plan, {a, c});
	Check(floats.ok && plain.ok && mixed.ok, "the three tables materialize");
	if (!floats.ok || !plain.ok || !mixed.ok) return;
	Check(floats.specialization.images[0].indirect_root == 0u &&
	          floats.specialization.images[0].indirect_search_iterations != 0u &&
	          floats.specialization.images.size() == 8u,
	      "two Float descriptors make a searched root with seven candidates");
	Check(plain.specialization.images.size() == 1u &&
	          plain.specialization.images[0].indirect_root == IR::ImageResource::NoIndirectImage,
	      "one descriptor makes a plain root");
	Check(mixed.specialization.images.size() == 8u &&
	          mixed.specialization.images[1].numeric_class == Prospero::TextureNumericClass::Uint,
	      "a Uint descriptor makes Uint candidates");
	Check(!(plain.specialization == floats.specialization) && !(mixed.specialization == floats.specialization),
	      "the three tables have three keys");

	IR::ResourceSnapshot aligned;
	const bool same = IR::AlignResources(plan, floats.snapshot, floats.specialization, floats.specialization, &aligned);
	Check(same == on, "a module serves its own shape");
	if (same) {
		Check(aligned.flattened_srt == floats.snapshot.flattened_srt && aligned.images[0] == floats.snapshot.images[0] &&
		          aligned.images[1] == floats.snapshot.images[1] && SameTexture(aligned.images[1], b),
		      "serving its own shape keeps its tables and the descriptors its keys reach");
	}

	const bool served = IR::AlignResources(plan, plain.snapshot, plain.specialization, floats.specialization, &aligned);
	Check(served == on, "a plain root runs on the searched module");
	if (served) {
		const auto offset = floats.specialization.images[0].indirect_mapping_offset;
		Check(aligned.images.size() == floats.specialization.images.size(), "the snapshot takes the module's layout");
		Check(SameTexture(aligned.images[0], a), "the root keeps the dispatch's descriptor");
		Check(aligned.flattened_srt.size() == floats.snapshot.flattened_srt.size() && aligned.flattened_srt[offset] == 0u,
		      "the plain root searches an empty table, so every key reaches the root");
		Check(std::all_of(aligned.images.begin() + 1, aligned.images.end(),
		                  [](const IR::DescriptorValue& value) { return value.dwords[0] == 0u; }),
		      "candidates the dispatch never reaches are null");
	}
	Check(!IR::AlignResources(plan, floats.snapshot, floats.specialization, plain.specialization, nullptr),
	      "a searched root never runs on the plain module");
	Check(!IR::AlignResources(plan, mixed.snapshot, mixed.specialization, floats.specialization, nullptr),
	      "a Uint candidate never runs on Float candidates");
	Check(!IR::AlignResources(plan, floats.snapshot, floats.specialization, mixed.specialization, nullptr),
	      "a Float candidate never runs on Uint candidates");

	IR::ResourceSpecialization                         widened;
	const std::array<const IR::ResourceSpecialization*, 1> known {&floats.specialization};
	const bool widens = IR::WidenSpecialization(plan, mixed.snapshot, mixed.specialization, known, widened);
	Check(widens == on, "a new shape widens over the known one");
	if (!widens) return;
	const auto count = [&](Prospero::TextureNumericClass type) {
		return std::count_if(widened.images.begin() + 1, widened.images.end(),
		                     [&](const auto& image) { return image.numeric_class == type; });
	};
	Check(count(Prospero::TextureNumericClass::Uint) == 15 && count(Prospero::TextureNumericClass::Float) == 7,
	      "the widened shape holds the known Float candidates and room for twice the new Uint ones");
	Check(widened.images[0].indirect_mapping_offset == floats.specialization.images[0].indirect_mapping_offset,
	      "the widened table starts where the SRT ends");

	const bool takes_mixed = IR::AlignResources(plan, mixed.snapshot, mixed.specialization, widened, &aligned);
	Check(takes_mixed, "the widened module serves the dispatch that made it");
	if (takes_mixed) {
		const auto child = ChildFor(widened, aligned, 1u);
		Check(child != 0u && child != UINT32_MAX && SameTexture(aligned.images[child], c) &&
		          widened.images[child].numeric_class == Prospero::TextureNumericClass::Uint,
		      "key 1 reaches the Uint descriptor in a Uint slot");
		Check(ChildFor(widened, aligned, 0u) == 0u, "key 0 reaches the root");
	}
	const bool takes_floats = IR::AlignResources(plan, floats.snapshot, floats.specialization, widened, &aligned);
	Check(takes_floats, "the widened module serves the Float shape");
	if (takes_floats) {
		const auto child = ChildFor(widened, aligned, 1u);
		Check(child != 0u && child != UINT32_MAX && SameTexture(aligned.images[child], b) &&
		          widened.images[child].numeric_class == Prospero::TextureNumericClass::Float,
		      "key 1 reaches the second Float descriptor in a Float slot");
	}
	Check(IR::AlignResources(plan, plain.snapshot, plain.specialization, widened, nullptr),
	      "the widened module serves the plain shape");
}

}

int main() {
	const char* setting = std::getenv("KYTY_GENERIC_COMPUTE_VARIANT");
	const bool  on      = setting == nullptr || std::strcmp(setting, "0") != 0;
	IR::SetComputeIndirectFloors(256, 8);
	if (!MapGuest()) {
		std::fprintf(stderr, "FAILED: the guest region could not be mapped\n");
		return 1;
	}
	CheckComputeQueueKeys();
	Generic(on);
	RebindTableAndHeap(on);
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_generic_compute_variants%s: ok\n", on ? "" : "_off");
	return 0;
}

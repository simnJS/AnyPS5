#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr uint32_t ShiftLane  = 0x34020082u;
constexpr uint32_t WaitAll    = 0xbf8c0000u;
constexpr uint32_t EndProgram = 0xbf810000u;

constexpr std::array<uint32_t, 2> LoadRaw {0xe0301000u, 0x80000201u};
constexpr std::array<uint32_t, 2> StoreRaw {0xe0701000u, 0x80010201u};

constexpr uint32_t Format32UInt  = 20;
constexpr uint32_t SwizzleXyzw   = 0xfacu;
constexpr uint64_t SourceAddress = 0x200810000ull;
constexpr uint64_t TargetAddress = 0x200820000ull;
constexpr uint32_t Records       = 128u;
constexpr uint64_t CodeHash      = 0x51u;
constexpr uint32_t BoundsBits    = IR::StructuredBufferBounds | IR::ExactBufferBounds;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Sharp = std::array<uint32_t, 4>;

Sharp MakeSharp(uint64_t address, uint32_t records, uint32_t select, uint32_t stride = 0) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u),
	        records, SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (select << 28u)};
}

struct Plan {
	ShaderComputeInputInfo info {};
	IR::ResourcePlan       plan;
};

Plan Translate() {
	const std::vector<uint32_t> code {ShiftLane, LoadRaw[0], LoadRaw[1], WaitAll, StoreRaw[0], StoreRaw[1], EndProgram};
	Plan                        result;
	auto&                       info = result.info;
	info.wave_size                   = 64;
	info.host_subgroup_size          = 64;
	info.threads_num[0]              = 64;
	info.threads_num[1]              = 1;
	info.threads_num[2]              = 1;
	info.thread_ids_num              = 1;
	const std::vector<uint32_t> user_data(16, 0u);
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = CodeHash;
	options.dump_ir            = false;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	result.plan                = IR::ExtractResourcePlan(translated.program);
	return result;
}

struct Request {
	bool                       ok = false;
	IR::ResourceSpecialization specialization;
};

Request Dispatch(const IR::ResourcePlan& plan, const Sharp& source, bool runtime_stride) {
	const auto               target = MakeSharp(TargetAddress, Records * 4u, 3);
	std::array<uint32_t, 16> user_data {};
	std::copy(source.begin(), source.end(), user_data.begin());
	std::copy(target.begin(), target.end(), user_data.begin() + 4);
	const IR::SrtRuntime runtime {.user_data = user_data, .runtime_stride = runtime_stride};
	IR::ResourceSnapshot snapshot;
	Request              result;
	result.ok = IR::MaterializeResources(plan, runtime, snapshot, result.specialization);
	return result;
}

uint32_t SourceSlot(const IR::ResourcePlan& plan) {
	for (uint32_t i = 0; i < plan.info.buffers.size(); i++) {
		if (plan.info.buffers[i].read && !plan.info.buffers[i].written) return i;
	}
	return UINT32_MAX;
}

ManifestModule Recipe(const Plan& program, const IR::ResourceSpecialization& specialization) {
	ManifestModule module {.stage = ShaderType::Compute, .code_hash = CodeHash, .specialization = specialization};
	module.input = program.info;
	NormalizeManifestInput(module);
	return module;
}

uint64_t Identity(const ManifestModule& module) {
	return ManifestModuleIdentity(module, {});
}

uint32_t LiveBounds(const Sharp& sharp) {
	ShaderBufferResource descriptor;
	std::copy_n(sharp.begin(), std::size(descriptor.fields), descriptor.fields);
	return IR::RuntimeStrideData(descriptor) & BoundsBits;
}

std::vector<ManifestModule> Reload(const std::vector<ManifestModule>& recipes, std::string* text = nullptr) {
	PipelineManifest written;
	written.modules  = recipes;
	const auto bytes = SaveManifest(written);
	if (text != nullptr) text->assign(bytes.begin(), bytes.end());
	PipelineManifest loaded;
	uint32_t         skipped = 0;
	Check(LoadManifest(bytes, loaded, &skipped) && skipped == 0, "the manifest reads back");
	return loaded.modules;
}

constexpr uint64_t TableAddress = 0x200000000ull;
constexpr uint32_t TableEntries = 6u;

std::array<uint32_t, TableEntries * 8u> g_table {};

bool ReadTable(void*, uint64_t address, std::span<uint32_t> values) {
	for (size_t index = 0; index < values.size(); ++index) {
		const auto at = address + index * 4u;
		values[index] = at >= TableAddress && at + 4u <= TableAddress + g_table.size() * 4u ? g_table[(at - TableAddress) / 4u] : 0u;
	}
	return true;
}

IR::ResourcePlan CandidatePlan(ShaderType stage) {
	IR::ResourcePlan plan;
	plan.stage                      = stage;
	plan.resource_tracking_complete = true;
	plan.srt_plan_complete          = true;
	const auto immediate = [](std::span<const uint32_t> words) {
		IR::DescriptorSource source;
		source.dword_count = static_cast<uint32_t>(words.size());
		for (size_t index = 0; index < words.size(); ++index) source.dwords[index] = IR::Value(words[index]);
		return source;
	};
	const std::array<uint32_t, 2> table {static_cast<uint32_t>(TableAddress), static_cast<uint32_t>(TableAddress >> 32u)};
	const std::array<uint32_t, 4> sampler_words {0x80000012u, 0u, 0u, 0u};
	IR::DescriptorSource          chosen;
	chosen.dword_count         = 8u;
	chosen.indirect_descriptor = IR::DescriptorSource::IndirectDescriptor {
	    .table_maximum = TableEntries - 1u,
	    .table_source  = 0u,
	    .table_offset  = 0u,
	    .table_stride  = 32u,
	    .table_scalar  = true,
	};
	plan.descriptor_sources.push_back(immediate(table));
	plan.descriptor_sources.push_back(std::move(chosen));
	plan.descriptor_sources.push_back(immediate(sampler_words));
	IR::ImageResource image;
	image.resource_class = IR::ImageResourceClass::Sampled;
	image.read           = true;
	image.dimension      = Recompiler::Decoder::ImageDimension::Dim2D;
	image.source         = 1u;
	plan.info.images.push_back(image);
	IR::SamplerResource sampler;
	sampler.source = 2u;
	plan.info.samplers.push_back(sampler);
	plan.info.sampled_pairs.push_back({.image = 0u, .sampler = 0u});
	IR::CompileSrtWalker(plan);
	return plan;
}

struct Candidates {
	bool                       ok = false;
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
};

uint32_t TextureAddress(uint32_t entry) {
	return 0x1000u * (entry + 1u);
}

IR::DescriptorValue TextureAt(uint32_t address, bool integer) {
	const auto          format = integer ? Prospero::BufferFormat::k8UInt : Prospero::BufferFormat::k8UNorm;
	IR::DescriptorValue value {.dword_count = 8u};
	value.dwords[0] = address;
	value.dwords[1] = (static_cast<uint32_t>(format) << 20u) | (3u << 30u);
	value.dwords[2] = 15u;
	value.dwords[3] = SwizzleXyzw | (9u << 28u);
	return value;
}

struct TableHeap final: IR::ImageHeapSource {
	std::vector<IR::DescriptorValue> kinds;

	bool Acquire(const IR::ImageHeapRequest&, IR::ImageHeapView& view) override {
		view.slots      = 0x300000000ull;
		view.slot_count = TableEntries;
		view.kinds      = kinds;
		return !kinds.empty();
	}
};

void FillTable(const std::array<uint32_t, TableEntries>& textures, const std::array<bool, TableEntries>& integer) {
	for (uint32_t entry = 0; entry < TableEntries; ++entry) {
		const auto value = TextureAt(textures[entry], integer[entry]);
		std::copy(value.dwords.begin(), value.dwords.end(), g_table.begin() + size_t {entry} * 8u);
	}
}

Candidates Choose(const IR::ResourcePlan& plan, const std::array<bool, TableEntries>& integer, TableHeap* heap = nullptr) {
	std::array<uint32_t, TableEntries> textures {};
	for (uint32_t entry = 0; entry < TableEntries; ++entry) textures[entry] = TextureAddress(entry);
	FillTable(textures, integer);
	const IR::SrtRuntime runtime {.read_specialization_memory = ReadTable, .image_heap = heap};
	Candidates           result;
	result.ok = IR::MaterializeResources(plan, runtime, result.snapshot, result.specialization);
	return result;
}

bool Follows(const Candidates& chosen, const std::array<bool, TableEntries>& integer) {
	const auto& images  = chosen.specialization.images;
	const auto& flat    = chosen.snapshot.flattened_srt;
	const auto  mapping = images[0].indirect_mapping_offset;
	if (mapping >= flat.size() || flat[mapping] < (TableEntries - 1u) * 8u + 1u || mapping + 1u + flat[mapping] > flat.size()) return false;
	for (uint32_t entry = 0; entry < TableEntries; ++entry) {
		const auto ordinal = flat[mapping + 1u + entry * 8u];
		if (ordinal == 0u || ordinal >= images.size() || ordinal >= chosen.snapshot.images.size()) return false;
		const bool is_integer = images[ordinal].numeric_class == Prospero::TextureNumericClass::Uint;
		if (chosen.snapshot.images[ordinal].dwords[0] != TextureAddress(entry) || is_integer != integer[entry]) return false;
	}
	return true;
}

void CandidateOrder(const Plan& program) {
	IR::SetComputeIndirectFloors(1u, 1u);
	const auto plan = CandidatePlan(ShaderType::Compute);
	const std::array<bool, TableEntries> first_order {false, true, false, false, true, false};
	const std::array<bool, TableEntries> second_order {true, false, false, true, false, false};
	const std::array<bool, TableEntries> integer_last {false, false, true, false, false, true};
	const auto first  = Choose(plan, first_order);
	const auto second = Choose(plan, second_order);
	const auto last   = Choose(plan, integer_last);
	Check(first.ok && second.ok && last.ok, "the three tables of candidates materialize");
	if (!first.ok || !second.ok || !last.ok) return;
	const auto& images = first.specialization.images;
	Check(images.size() == 8u && images[0].indirect_root == 0u && !images[0].heap && images[0].indirect_search_iterations == 0u,
	      "a table without an image heap takes one slot per candidate, padded to a power of two");
	if (images.size() != 8u) return;
	const auto integers = std::ranges::count(images | std::views::drop(1), Prospero::TextureNumericClass::Uint,
	                                         &IR::ResourceSpecialization::Image::numeric_class);
	Check(integers == 2, "the unused slot takes the kind most candidates have");
	uint32_t changes = 0;
	for (size_t index = 2; index < images.size(); ++index) changes += images[index].numeric_class != images[index - 1u].numeric_class;
	Check(changes == 1u, "the candidates of one kind are next to each other");
	Check(first.specialization == second.specialization, "the order of the candidates in memory does not change the variant");
	Check(first.specialization == last.specialization, "the kind of the last candidate in memory does not change the variant");
	Check(Follows(first, first_order) && Follows(second, second_order) && Follows(last, integer_last),
	      "each table entry still selects its own texture");

	auto recorded = first.specialization;
	std::swap(recorded.images[1], recorded.images[6]);
	auto other = first.specialization;
	std::ranges::reverse(other.images | std::views::drop(1));
	Check(!(recorded == first.specialization) && !(other == first.specialization) && !(recorded == other),
	      "the recipes of earlier sessions list the candidates in other orders");
	const auto loaded = Reload({Recipe(program, recorded), Recipe(program, other), Recipe(program, first.specialization)});
	Check(loaded.size() == 3u, "the three recipes read back");
	if (loaded.size() == 3u) {
		Check(loaded[0].specialization == first.specialization && loaded[1].specialization == first.specialization &&
		          loaded[2].specialization == first.specialization,
		      "a recipe recorded in another candidate order loads as the variant a dispatch requests");
		Check(Identity(loaded[0]) == Identity(loaded[2]) && Identity(loaded[1]) == Identity(loaded[2]),
		      "the recipes of several sessions are one prepared variant");
		Check(Identity(Recipe(program, second.specialization)) == Identity(loaded[0]) &&
		          Identity(Recipe(program, last.specialization)) == Identity(loaded[1]),
		      "a dispatch whose candidates have their kinds at other positions requests the variant the manifest prepared");
	}

	TableHeap heap;
	heap.kinds = {TextureAt(TextureAddress(0), false), TextureAt(TextureAddress(1), true)};
	const auto on_heap = Choose(plan, first_order, &heap);
	Check(on_heap.ok && on_heap.specialization.images.size() == 3u && on_heap.specialization.images[0].heap,
	      "the same table read through the image heap takes one slot per kind");
	if (on_heap.ok && on_heap.specialization.images.size() == 3u) {
		const IR::SrtRuntime runtime {.read_specialization_memory = ReadTable, .image_heap = &heap};
		IR::ResourceSnapshot rebound;
		Check(!(on_heap.specialization == first.specialization) &&
		          IR::CanRebindComputeImages(plan, on_heap.specialization, first.specialization),
		      "a variant prepared with one slot per candidate can stand in for the heap form");
		Check(IR::RebindComputeImages(plan, runtime, on_heap.snapshot, on_heap.specialization, first.specialization, rebound) &&
		          rebound.images == first.snapshot.images && rebound.flattened_srt == first.snapshot.flattened_srt,
		      "a dispatch that reads the heap binds the variant prepared with one slot per candidate");

		const std::array<uint32_t, TableEntries> few {TextureAddress(0), TextureAddress(0), TextureAddress(1),
		                                             TextureAddress(1), TextureAddress(0), TextureAddress(1)};
		const std::array<bool, TableEntries>     plain {};
		FillTable(few, plain);
		Candidates narrow;
		const IR::SrtRuntime no_heap {.read_specialization_memory = ReadTable};
		narrow.ok = IR::MaterializeResources(plan, no_heap, narrow.snapshot, narrow.specialization);
		std::array<uint32_t, TableEntries> every {};
		for (uint32_t entry = 0; entry < TableEntries; ++entry) every[entry] = TextureAddress(entry);
		FillTable(every, plain);
		Candidates wide;
		wide.ok = IR::MaterializeResources(plan, no_heap, wide.snapshot, wide.specialization);
		Check(narrow.ok && wide.ok && narrow.specialization.images.size() == 4u && wide.specialization.images.size() == 8u,
		      "two and six textures of one kind take three and seven slots");
		heap.kinds = {TextureAt(TextureAddress(0), false)};
		FillTable(few, plain);
		Candidates few_on_heap;
		few_on_heap.ok = IR::MaterializeResources(plan, runtime, few_on_heap.snapshot, few_on_heap.specialization);
		if (narrow.ok && wide.ok && few_on_heap.ok) {
			Check(IR::RebindComputeImages(plan, runtime, few_on_heap.snapshot, few_on_heap.specialization, wide.specialization,
			                              rebound) &&
			          rebound.images.size() == 8u && rebound.flattened_srt == narrow.snapshot.flattened_srt,
			      "a table with fewer textures fills the slots of the prepared variant");
			FillTable(every, plain);
			Candidates all_on_heap;
			all_on_heap.ok = IR::MaterializeResources(plan, runtime, all_on_heap.snapshot, all_on_heap.specialization);
			Check(all_on_heap.ok &&
			          !IR::RebindComputeImages(plan, runtime, all_on_heap.snapshot, all_on_heap.specialization,
			                                   narrow.specialization, rebound),
			      "a table with more textures than the prepared variant has slots is not served by it");
			Check(!IR::CanRebindComputeImages(plan, on_heap.specialization, wide.specialization),
			      "a prepared variant without a slot of one of the kinds on the heap is refused");
		}
	}

	const auto pixel_plan   = CandidatePlan(ShaderType::Pixel);
	const auto pixel_first  = Choose(pixel_plan, first_order);
	const auto pixel_second = Choose(pixel_plan, second_order);
	Check(pixel_first.ok && pixel_second.ok && !(pixel_first.specialization == pixel_second.specialization) &&
	          Follows(pixel_first, first_order) && Follows(pixel_second, second_order),
	      "a graphics stage keeps the candidates in their order in memory");
}

void GenericGraphics() {
	IR::ResourcePlan plan;
	plan.stage                      = ShaderType::Pixel;
	plan.resource_tracking_complete = true;
	plan.srt_plan_complete          = true;
	plan.descriptor_sources.emplace_back();
	plan.info.buffers.emplace_back().source = 0;
	plan.buffer_vector_access.push_back(1);
	IR::ResourceSpecialization shape;
	Check(IR::GenericGraphicsShape(plan, shape) && shape.buffers.size() == 1 &&
	          shape.buffers.front().packed_stride == IR::RuntimeBufferStride,
	      "generic graphics: the buffer reads its stride at run time, without bounds in the key");
	if (shape.buffers.size() != 1) return;
	struct Draw {
		const char* name;
		Sharp       sharp;
		uint32_t    stride;
	};
	const std::array draws {
	    Draw {"structured records", MakeSharp(SourceAddress, Records, 0, 16), 16u | IR::StructuredBufferBounds},
	    Draw {"records by count", MakeSharp(SourceAddress, Records * 4u, 1, 16), 16u},
	    Draw {"exact bytes", MakeSharp(SourceAddress, Records * 4u, 3, 1), 1u | IR::ExactBufferBounds},
	};
	for (const auto& draw: draws) {
		IR::ResourceSpecialization live;
		live.buffers.emplace_back().packed_stride = draw.stride;
		IR::ResourceSnapshot snapshot;
		auto&                value = snapshot.buffers.emplace_back();
		value.dword_count          = 4;
		std::copy(draw.sharp.begin(), draw.sharp.end(), value.dwords.begin());
		Check(IR::BindGenericGraphics(plan, snapshot, live, shape),
		      std::string("generic graphics: ") + draw.name + " bind the module prepared from the code alone");
		Check(LiveBounds(draw.sharp) == (draw.stride & BoundsBits),
		      std::string("generic graphics: ") + draw.name + " reach the module in the stride word");
	}
	IR::ResourceSpecialization swizzled;
	swizzled.buffers.emplace_back().packed_stride = 16u | (1u << 14u) | IR::StructuredBufferBounds;
	IR::ResourceSnapshot snapshot;
	snapshot.buffers.emplace_back().dword_count = 4;
	Check(!IR::BindGenericGraphics(plan, snapshot, swizzled, shape),
	      "generic graphics: a buffer whose stride stays fixed keeps its own variant");
}

}

int main() {
	GenericGraphics();
	const auto program = Translate();
	const auto slot    = SourceSlot(program.plan);
	Check(slot != UINT32_MAX, "the source buffer is planned");
	if (slot == UINT32_MAX) return 1;

	struct Bound {
		const char* name;
		Sharp       sharp;
		uint32_t    live;
	};
	const std::array bound {
	    Bound {"structured records", MakeSharp(SourceAddress, Records, 0, 16), IR::StructuredBufferBounds},
	    Bound {"records by count", MakeSharp(SourceAddress, Records * 4u, 1, 16), 0u},
	    Bound {"any record", MakeSharp(SourceAddress, Records * 4u, 2), 0u},
	    Bound {"exact bytes", MakeSharp(SourceAddress, Records * 4u, 3), IR::ExactBufferBounds},
	    Bound {"no records", MakeSharp(SourceAddress, 0u, 3), IR::ExactBufferBounds},
	};

	const auto first = Dispatch(program.plan, bound.front().sharp, true);
	Check(first.ok, "the first dispatch materializes");
	if (!first.ok) return 1;
	Check(IR::HasRuntimeStride(first.specialization.buffers[slot].packed_stride),
	      "the source slot reads its stride at run time");

	auto before_bounds = first.specialization;
	for (auto& buffer: before_bounds.buffers) {
		if (IR::HasRuntimeStride(buffer.packed_stride)) buffer.packed_stride = IR::RuntimeBufferStride;
	}
	auto structured                       = before_bounds;
	structured.buffers[slot].packed_stride = IR::RuntimeBufferStride | IR::StructuredBufferBounds;
	auto exact                            = before_bounds;
	exact.buffers[slot].packed_stride     = IR::RuntimeBufferStride | IR::ExactBufferBounds;
	auto fixed                            = before_bounds;
	fixed.buffers[slot].packed_stride     = 16u | IR::StructuredBufferBounds;

	std::string text;
	const auto  loaded = Reload({Recipe(program, before_bounds), Recipe(program, structured), Recipe(program, exact),
	                             Recipe(program, fixed)},
	                            &text);
	Check(text.find("st:16383") != std::string::npos, "the manifest holds a recipe recorded before the bounds");
	Check(text.find("st:2113535") != std::string::npos, "the manifest holds a recipe recorded with structured bounds");
	Check(text.find("st:4210687") != std::string::npos, "the manifest holds a recipe recorded with exact bounds");
	Check(loaded.size() == 4, "every recipe reads back");
	if (loaded.size() != 4) return 1;

	const auto runtime_recipe = std::ranges::find_if(loaded, [&](const ManifestModule& module) {
		return IR::HasRuntimeStride(module.specialization.buffers[slot].packed_stride);
	});
	Check(runtime_recipe != loaded.end(), "a recipe that reads its stride at run time reads back");
	if (runtime_recipe == loaded.end()) return 1;
	const auto precooked = Identity(*runtime_recipe);
	uint32_t   distinct  = 0;
	uint32_t   kept      = 0;
	for (const auto& module: loaded) {
		const auto stride = module.specialization.buffers[slot].packed_stride;
		if (IR::HasRuntimeStride(stride)) {
			Check((stride & BoundsBits) == 0u, "a recipe that reads its stride at run time reads back without bounds");
			distinct += Identity(module) != precooked ? 1u : 0u;
		} else {
			Check(stride == (16u | IR::StructuredBufferBounds), "a recipe with a fixed stride reads back with its bounds");
			Check(Identity(module) != precooked, "a recipe with a fixed stride stays its own variant");
			kept++;
		}
	}
	Check(distinct == 0, "recipes that differ by run-time bounds are one variant to prepare");
	Check(kept == 1, "the fixed-stride recipe is still listed");
	const auto rewritten = Reload(loaded);
	Check(rewritten.size() == 2, "the manifest written back lists the two variants left");
	Check(std::ranges::any_of(rewritten, [&](const ManifestModule& module) { return Identity(module) == precooked; }),
	      "a recipe written back keeps its identity");

	uint32_t covered = 0;
	for (const auto& entry: bound) {
		const auto request = Dispatch(program.plan, entry.sharp, true);
		Check(request.ok, std::string(entry.name) + ": the dispatch materializes");
		if (!request.ok) continue;
		const auto stride = request.specialization.buffers[slot].packed_stride;
		Check(IR::HasRuntimeStride(stride) && (stride & BoundsBits) == 0u,
		      std::string(entry.name) + ": the bounds stay out of the key");
		Check(LiveBounds(entry.sharp) == entry.live, std::string(entry.name) + ": the stride word carries the bounds");
		const bool hit = Identity(Recipe(program, request.specialization)) == precooked;
		Check(hit, std::string(entry.name) + ": the recipe recorded before the bounds is the variant requested");
		covered += hit ? 1u : 0u;
	}

	const auto baked_structured = Dispatch(program.plan, bound[0].sharp, false);
	const auto baked_exact      = Dispatch(program.plan, bound[3].sharp, false);
	Check(baked_structured.ok && baked_exact.ok, "the fixed-stride dispatches materialize");
	if (baked_structured.ok && baked_exact.ok) {
		Check((baked_structured.specialization.buffers[slot].packed_stride & BoundsBits) == IR::StructuredBufferBounds,
		      "a fixed stride keeps structured bounds in the key");
		Check((baked_exact.specialization.buffers[slot].packed_stride & BoundsBits) == IR::ExactBufferBounds,
		      "a fixed stride keeps exact bounds in the key");
	}

	CandidateOrder(program);

	std::printf("variant key coverage: %u of %zu requests are the prepared variant\n", covered, bound.size());
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_variant_key_coverage: ok\n");
	return 0;
}

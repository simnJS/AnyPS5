#include "graphics/host_gpu/renderer/pipeline/pipelineWaitCause.h"

#if defined(ANYPS5_KYTY)
#include <algorithm>
#include <array>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <vector>

namespace Libs::Graphics {

std::string SpecializationDifference(const ShaderRecompiler::IR::ResourceSpecialization& wanted,
                                     const ShaderRecompiler::IR::ResourceSpecialization& prepared, size_t& distance) {
	static constexpr std::array<const char*, 12> Names {
	    "buffer-bounds-bits", "buffer-empty", "buffer-format", "buffer-stride", "buffer-table", "image-class",
	    "image-dimension",    "image-format", "image-heap",    "image-search",  "image-table",  "sampler-table"};
	enum : uint32_t {
		BufferBounds, BufferEmpty, BufferFormat, BufferStride, BufferTable, ImageClass,
		ImageDimension, ImageFormat, ImageHeap, ImageSearch, ImageTable, SamplerTable
	};
	std::vector<std::string> reasons;
	uint32_t                 fields = 0;
	distance                        = 0;
	const auto same_count = [&](const char* name, size_t now, size_t before) {
		if (now == before) return true;
		reasons.push_back(fmt::format("{}-count({}->{})", name, before, now));
		distance += 1000u + (now > before ? now - before : before - now);
		return false;
	};
	if (same_count("buffer", wanted.buffers.size(), prepared.buffers.size())) {
		constexpr uint32_t bounds = ShaderRecompiler::IR::StructuredBufferBounds | ShaderRecompiler::IR::ExactBufferBounds;
		for (size_t i = 0; i < wanted.buffers.size(); i++) {
			const auto& now    = wanted.buffers[i];
			const auto& before = prepared.buffers[i];
			if (now == before) continue;
			distance++;
			if (now.packed_stride != before.packed_stride) {
				fields |= 1u << (((now.packed_stride ^ before.packed_stride) & ~bounds) == 0u ? BufferBounds : BufferStride);
			}
			if (now.descriptor_format != before.descriptor_format || now.descriptor_swizzle != before.descriptor_swizzle) {
				fields |= 1u << BufferFormat;
			}
			if (now.zero_stride_oob != before.zero_stride_oob || now.scalar_empty != before.scalar_empty) {
				fields |= 1u << BufferEmpty;
			}
			if (now.indirect_root != before.indirect_root || now.indirect_mapping_offset != before.indirect_mapping_offset ||
			    now.indirect_search_iterations != before.indirect_search_iterations) {
				fields |= 1u << BufferTable;
			}
		}
	}
	if (same_count("image", wanted.images.size(), prepared.images.size())) {
		for (size_t i = 0; i < wanted.images.size(); i++) {
			const auto& now    = wanted.images[i];
			const auto& before = prepared.images[i];
			if (now == before) continue;
			distance++;
			if (now.numeric_class != before.numeric_class) fields |= 1u << ImageClass;
			if (now.dimension != before.dimension || now.cube != before.cube) fields |= 1u << ImageDimension;
			if (now.table != before.table || now.table_capacity != before.table_capacity ||
			    now.table_mapping_offset != before.table_mapping_offset || now.table_entry_mask != before.table_entry_mask) {
				fields |= 1u << ImageTable;
			}
			if (now.heap != before.heap) fields |= 1u << ImageHeap;
			if (now.indirect_root != before.indirect_root || now.indirect_mapping_offset != before.indirect_mapping_offset ||
			    now.indirect_search_iterations != before.indirect_search_iterations) {
				fields |= 1u << ImageSearch;
			}
			if (now.mip_count != before.mip_count || now.conversion_format != before.conversion_format ||
			    now.shader_swizzle != before.shader_swizzle || now.srgb_decode != before.srgb_decode ||
			    now.skip_degamma != before.skip_degamma || now.fmask != before.fmask) {
				fields |= 1u << ImageFormat;
			}
		}
	}
	if (same_count("sampler", wanted.samplers.size(), prepared.samplers.size())) {
		for (size_t i = 0; i < wanted.samplers.size(); i++) {
			if (wanted.samplers[i] == prepared.samplers[i]) continue;
			distance++;
			fields |= 1u << SamplerTable;
		}
	}
	for (uint32_t field = 0; field < Names.size(); field++) {
		if ((fields & (1u << field)) != 0u) reasons.emplace_back(Names[field]);
	}
	std::ranges::sort(reasons);
	return fmt::format("{}", fmt::join(reasons, "+"));
}

std::string DescribeProgramWait(const ProgramWaitFacts& facts) {
	const auto unbound = [&] {
		if (facts.compute && !facts.serve_refusal.empty()) return "generic-unbound:" + facts.serve_refusal;
		return "generic-unbound:" + (facts.difference.empty() ? std::string("push-data-start") : facts.difference);
	};
	if (!facts.compute) {
		if (facts.draw == GenericServe::PipelineMissing) return "library-missing:generic-pipeline";
		if (facts.generic_prepared) return facts.other_variant ? unbound() : "generic-unbound:no-exact-recipe";
		if (!facts.generic_refusal.empty()) return "generic-refused:" + facts.generic_refusal;
		if (facts.other_variant) return "generic-not-prepared:no-generic-module";
	} else if (facts.other_variant) {
		return unbound();
	}
	if (!facts.recipe_refusal.empty()) return "generic-refused:" + facts.recipe_refusal;
	if (!facts.other_keys.empty()) return "other-key:" + facts.other_keys;
	if (facts.known == 0) return "unknown-program";
	if (facts.calls) return "generic-not-prepared:calls";
	return facts.known < 0 ? "generic-not-prepared:code-store-unread" : "generic-not-prepared:no-recipe";
}

std::string DescribeStallWait(const StallWaitFacts& facts) {
	if (!facts.module_cause.empty()) return facts.module_cause;
	if (facts.compute && !facts.serve_refusal.empty()) return "generic-unbound:" + facts.serve_refusal;
	if (facts.compute) return "other:prepared-module-without-driver-pipeline";
	const auto  route = facts.route;
	std::string parts;
	if (route.find("pr=queued") != std::string_view::npos || route.find("pr=building") != std::string_view::npos) parts = "pr";
	if (route.find("fs=queued") != std::string_view::npos || route.find("fs=building") != std::string_view::npos) {
		parts += parts.empty() ? "fs" : "+fs";
	}
	if (parts.empty()) parts = route.starts_with("linked") ? "vi+fo" : "pipeline";
	if (facts.draw == GenericServe::Unbound) return "generic-unbound:exact-stages-prepared+" + parts;
	if (facts.draw == GenericServe::NoModule && !facts.generic_refusal.empty()) return "generic-refused:" + facts.generic_refusal;
	return (facts.draw == GenericServe::PipelineMissing ? "library-missing:generic-pipeline+" : "library-missing:") + parts;
}

}
#endif

#include "graphics/host_gpu/renderer/pipeline/pipelineWaitCause.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using namespace Libs::Graphics;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

void Same(const std::string& found, const std::string& expected, const std::string& what) {
	Check(found == expected, what + ": " + found + " instead of " + expected);
}

std::string Difference(const IR::ResourceSpecialization& wanted, const IR::ResourceSpecialization& prepared,
                       size_t* distance = nullptr) {
	size_t     steps = 0;
	const auto text  = SpecializationDifference(wanted, prepared, steps);
	if (distance != nullptr) *distance = steps;
	return text;
}

void Differences() {
	IR::ResourceSpecialization prepared;
	prepared.buffers.resize(3);
	prepared.buffers[0].packed_stride = 48u;
	prepared.buffers[1].packed_stride = 1u;
	prepared.images.resize(2);
	prepared.images[1].table          = 0;
	prepared.images[1].table_capacity = 65536u;
	prepared.samplers.resize(1);

	size_t distance = 0;
	Same(Difference(prepared, prepared, &distance), "", "an identical variant has no difference");
	Check(distance == 0, "an identical variant is at distance zero");

	auto bounds = prepared;
	bounds.buffers[0].packed_stride |= IR::StructuredBufferBounds;
	bounds.buffers[1].packed_stride |= IR::ExactBufferBounds;
	Same(Difference(bounds, prepared, &distance), "buffer-bounds-bits", "bounds bits alone are named");
	Check(distance == 2, "two buffers differ");

	auto stride = bounds;
	stride.buffers[2].packed_stride = 16u;
	stride.buffers[2].scalar_empty  = true;
	Same(Difference(stride, prepared), "buffer-bounds-bits+buffer-empty+buffer-stride", "buffer fields are listed in order");

	auto heap = prepared;
	heap.images[1].table          = IR::ImageResource::NoImageTable;
	heap.images[1].table_capacity = 0;
	heap.images[1].heap           = true;
	heap.images[0].numeric_class  = Prospero::TextureNumericClass::Uint;
	Same(Difference(heap, prepared), "image-class+image-heap+image-table", "image fields are listed in order");

	auto search = prepared;
	search.images.resize(4098);
	Same(Difference(search, prepared, &distance), "image-count(2->4098)", "a search form names its number of views");
	Check(distance >= 1000, "another number of views is far");
	Same(Difference(prepared, search), "image-count(4098->2)", "the count reads from the prepared variant to the wanted one");

	auto sampler = prepared;
	sampler.samplers[0].indirect_root = 1;
	Same(Difference(sampler, prepared), "sampler-table", "a sampler chosen in a table is named");
}

void ProgramWaits() {
	Same(DescribeProgramWait({.compute = true, .known = 0}), "unknown-program", "a compute program out of the code store");
	Same(DescribeProgramWait({.compute = true, .known = 1, .calls = true}), "generic-not-prepared:calls",
	     "a known program with calls and no recipe");
	Same(DescribeProgramWait({.compute = true, .known = 1}), "generic-not-prepared:no-recipe",
	     "a known program without any recipe");
	Same(DescribeProgramWait({.compute = true, .known = -1}), "generic-not-prepared:code-store-unread",
	     "a program met before the code store is read");
	Same(DescribeProgramWait({.compute = true, .other_keys = "rs", .known = 1}), "other-key:rs",
	     "recipes under the run-time sampler key only");
	Same(DescribeProgramWait({.compute = true, .recipe_refusal = "specialization_does_not_fit", .other_keys = "rs", .known = 1}),
	     "generic-refused:specialization_does_not_fit", "recipes not rebuilt at startup come before another key");
	Same(DescribeProgramWait({.compute = true, .recipe_refusal = "specialization_does_not_fit", .other_variant = true,
	                          .difference = "image-heap+image-table", .known = 1}),
	     "generic-unbound:image-heap+image-table", "a prepared variant that does not bind is named by its difference");
	Same(DescribeProgramWait({.compute = true, .other_variant = true, .known = 1}), "generic-unbound:push-data-start",
	     "the same specialization at another push data start");

	Same(DescribeProgramWait({.draw = GenericServe::PipelineMissing, .generic_prepared = true, .other_variant = true,
	                          .difference = "buffer-bounds-bits", .known = 1}),
	     "library-missing:generic-pipeline", "a bound generic form without its pipeline");
	Same(DescribeProgramWait({.draw = GenericServe::Unbound, .generic_prepared = true, .other_variant = true,
	                          .difference = "buffer-bounds-bits", .known = 1}),
	     "generic-unbound:buffer-bounds-bits", "a generic form that does not bind");
	Same(DescribeProgramWait({.generic_prepared = true, .known = 1}), "generic-unbound:no-exact-recipe",
	     "a generic form that does not bind and no exact recipe");
	Same(DescribeProgramWait({.draw = GenericServe::NoModule, .generic_refusal = "several_indirect_tables",
	                          .other_variant = true, .difference = "buffer-bounds-bits", .known = 1}),
	     "generic-refused:several_indirect_tables", "a generic form refused at startup");
	Same(DescribeProgramWait({.draw = GenericServe::NoModule, .other_variant = true, .known = 1}),
	     "generic-not-prepared:no-generic-module", "exact recipes without a generic module");
	Same(DescribeProgramWait({.draw = GenericServe::NoModule, .known = 0}), "unknown-program",
	     "a graphics program out of the code store");
}

void StallWaits() {
	Same(DescribeStallWait({.compute = true, .module_cause = "generic-unbound:image-heap"}), "generic-unbound:image-heap",
	     "the cause of a module compiled in the game comes first");
	Same(DescribeStallWait({.compute = true}), "other:prepared-module-without-driver-pipeline",
	     "a prepared compute module without its pipeline");
	Same(DescribeStallWait({.route = "linked fs=queued pr=ready"}), "library-missing:fs", "the fragment library alone");
	Same(DescribeStallWait({.route = "linked fs=building pr=queued"}), "library-missing:pr+fs", "both shader libraries");
	Same(DescribeStallWait({.route = "linked fs=ready pr=ready"}), "library-missing:vi+fo", "the state libraries");
	Same(DescribeStallWait({.route = ""}), "library-missing:pipeline", "a pipeline waited without a route");
	Same(DescribeStallWait({.draw = GenericServe::PipelineMissing, .route = "linked fs=queued pr=ready"}),
	     "library-missing:generic-pipeline+fs", "the generic pipeline missing for the render state");
	Same(DescribeStallWait({.draw = GenericServe::Unbound, .route = "linked fs=queued pr=queued"}),
	     "generic-unbound:exact-stages-prepared+pr+fs", "prepared exact stages after an unbound generic form");
	Same(DescribeStallWait({.draw = GenericServe::NoModule, .generic_refusal = "formatted_buffer_written",
	                        .route = "linked fs=queued pr=ready"}),
	     "generic-refused:formatted_buffer_written", "prepared exact stages of a refused generic form");
	Same(DescribeStallWait({.draw = GenericServe::NoModule, .route = "linked fs=queued pr=ready"}), "library-missing:fs",
	     "no generic module and no refusal");
}

}

int main() {
	Differences();
	ProgramWaits();
	StallWaits();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_wait_cause: %d failed\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_wait_cause: ok\n");
	return 0;
}

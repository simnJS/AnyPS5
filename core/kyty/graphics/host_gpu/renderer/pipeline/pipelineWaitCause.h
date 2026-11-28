#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEWAITCAUSE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEWAITCAUSE_H_

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace Libs::Graphics {

enum class GenericServe : uint8_t { Served, NoModule, Unbound, PipelineMissing };

struct ProgramWaitFacts {
	bool         compute          = false;
	GenericServe draw             = GenericServe::Served;
	bool         generic_prepared = false;
	std::string  generic_refusal;
	std::string  recipe_refusal;
	bool         other_variant = false;
	std::string  difference;
	std::string  other_keys;
	int          known = -1;
	bool         calls = false;
};

struct StallWaitFacts {
	bool             compute = false;
	GenericServe     draw    = GenericServe::Served;
	std::string      module_cause;
	std::string      generic_refusal;
	std::string_view route;
};

[[nodiscard]] std::string SpecializationDifference(const ShaderRecompiler::IR::ResourceSpecialization& wanted,
                                                   const ShaderRecompiler::IR::ResourceSpecialization& prepared,
                                                   size_t&                                             distance);
[[nodiscard]] std::string DescribeProgramWait(const ProgramWaitFacts& facts);
[[nodiscard]] std::string DescribeStallWait(const StallWaitFacts& facts);

}

#endif

#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FLOATMINMAXPROBE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FLOATMINMAXPROBE_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <cstdint>

namespace Libs::Graphics {

[[nodiscard]] ShaderRecompiler::Spirv::FloatMinMax FloatMinMaxLevel(const uint32_t* pairs, const uint32_t* results,
                                                                    uint32_t pair_count);
ShaderRecompiler::Spirv::FloatMinMax ProbeFloatMinMax(const GraphicContext& graphics);

}

#endif

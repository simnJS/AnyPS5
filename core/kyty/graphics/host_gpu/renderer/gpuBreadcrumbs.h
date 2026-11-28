#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <optional>

namespace Libs::Graphics {

struct GraphicContext;
class CommandRecorder;

namespace GpuBreadcrumbs {

[[nodiscard]] bool Requested();
[[nodiscard]] bool PerOperation();
[[nodiscard]] bool Available();

[[nodiscard]] uint32_t SubmissionSlot(const char* name);
void                   MarkSubmission(GraphicContext& graphics, const CommandRecorder& command, uint32_t slot, uint64_t tick);
[[nodiscard]] std::optional<uint64_t> StartedTick(uint32_t slot, uint64_t near);

[[nodiscard]] uint32_t BeginOperation(GraphicContext& graphics, const CommandRecorder& command, uint64_t tick, const char* kind,
                                      uint64_t submit_id);
void                   EndOperation(const CommandRecorder& command, uint32_t id);

}

}

#endif

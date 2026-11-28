#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <optional>

namespace Libs::Graphics {

struct GraphicContext;
class CommandRecorder;
class CommandScheduler;
struct ShaderModule;

namespace GpuBreadcrumbs {

[[nodiscard]] bool Requested();
[[nodiscard]] bool PerOperation();
[[nodiscard]] bool Available();

[[nodiscard]] uint32_t SubmissionSlot(const char* name);
void                   MarkSubmission(GraphicContext& graphics, const CommandRecorder& command, uint32_t slot, uint64_t tick);
[[nodiscard]] std::optional<uint64_t> StartedTick(uint32_t slot, uint64_t near);

class OperationScope {
public:
	OperationScope(CommandScheduler& scheduler, const char* kind, uint64_t submit_id, const char* callback);
	~OperationScope();
	OperationScope(const OperationScope&) = delete;
	OperationScope& operator=(const OperationScope&) = delete;

	void BeginSegment();
	void EndSegment();

	CommandScheduler& scheduler;
	uint32_t id = 0;

private:
	const char* kind;
	uint64_t submit_id;
	const char* callback;
	OperationScope* outer;
};

void BeforeSubmit(CommandScheduler& scheduler);
void AfterBegin(CommandScheduler& scheduler);
void NoteProgram(const ShaderModule& module);
[[nodiscard]] bool NeedsBvhPaths(const ShaderModule& module);
void ReportDeviceLost(GraphicContext& graphics, vk::Result result, const char* call, const char* scheduler, uint64_t tick);
void Shutdown(GraphicContext& graphics);

}

}

#endif

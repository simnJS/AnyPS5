#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SUBMIT_SCAN_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SUBMIT_SCAN_H

#include "graphics/guest_gpu/command_processor/gpuOperation.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {

class CommandProcessor;
class RenderContext;

struct SubmitScanDispatch {
	uint32_t thread_groups[3] = {};
	uint32_t mode             = 0;
	uint64_t args_addr        = 0;
	bool     indirect         = false;
};

class SubmitScanSink {
public:
	virtual ~SubmitScanSink() = default;
	virtual void Draw(CommandProcessor& shadow, const IndirectDrawRegisters* indirect) = 0;
	virtual void Dispatch(CommandProcessor& shadow, const SubmitScanDispatch& dispatch) = 0;
};

enum class SubmitScanEnd : uint8_t { Complete, Rewind, Unreadable, Malformed, Budget, Error };

struct SubmitScanResult {
	SubmitScanEnd end        = SubmitScanEnd::Complete;
	uint32_t      draws      = 0;
	uint32_t      dispatches = 0;
};

[[nodiscard]] SubmitScanResult ScanSubmission(std::span<const uint32_t> commands, CommandProcessor& shadow,
                                              const RenderContext& renderer, SubmitScanSink& sink);

}

#endif

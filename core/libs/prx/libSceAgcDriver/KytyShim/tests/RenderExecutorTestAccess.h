#pragma once

#include "graphics/host_gpu/renderer/pipeline/pipelineStoreTypes.h"
#include "graphics/host_gpu/renderer/render.h"

namespace Libs::Graphics {

struct RenderExecutorTestAccess {
	static void Reset(RenderExecutor& executor) { executor.ResetBindings(); }
	static size_t RingSets(const RenderExecutor& executor, const ShaderModule& module) {
		return executor.m_table_states.at(&module.info).ring.size();
	}
};

}

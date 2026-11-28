#pragma once

#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <memory>

namespace Libs::Graphics {

inline constexpr uint32_t TableShaderKeyMarker = 0x54424c45u;
inline constexpr uint32_t LanePairsKey         = 0x574c5052u;
inline constexpr uint32_t RuntimeBuffersKey    = 0x52425546u;
inline constexpr uint32_t GenericGraphicsKey   = 0x47465847u;
inline constexpr uint32_t RuntimeSamplersKey   = 0x53414d50u;

struct ShaderProgramKey {
	ShaderType            stage           = ShaderType::Unknown;
	uint64_t              hash            = 0;
	uint32_t              user_data_count = 0;
	std::vector<uint32_t> static_state;
	bool guest_traversal = false;

	bool operator==(const ShaderProgramKey&) const = default;
};

struct ShaderModule {
	uint64_t                                     id     = 0;
	vk::ShaderModule                             module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	std::vector<vk::Format>                      vertex_formats;
	std::vector<RectListParameter>               rect_parameters;
	bool                                         mesh_task         = false;
	uint32_t                                     mesh_shared_bytes = 0;
	uint32_t                                     spirv_words       = 0;
	std::shared_ptr<const ShaderModule>          plain;
	bool                                         guest_traversal = false;
	ShaderBvh                                    bvh = ShaderBvh::None;
	uint32_t                                     rt_mode = 0;
	const ShaderRecompiler::IR::ResourcePlan*    plan = nullptr;
#if defined(ANYPS5_KYTY)
	mutable bool                                 pipeline_requested = false;
	bool                                         shared_variant     = false;
	bool                                         linked_prepared    = false;
#endif
};

inline constexpr uint64_t PlainModuleIdXor = 0x9e3779b97f4a7c15ull;

}

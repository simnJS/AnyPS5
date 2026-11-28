#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

inline constexpr uint32_t MeshAllocationLengthSpecId = 0;
inline constexpr uint32_t MeshAllocationBaseDwords   = 2;

enum class FloatMinMax : uint8_t {
	Cascade,
	OtherNanSelect,
	Exact,
};

struct Profile {
	bool maintenance8         = false;
	FloatMinMax float_min_max = FloatMinMax::Cascade;
	uint32_t device_clock_period_ps = 0;
	bool robust_buffer_access = false;
	bool r8_srgb_views   = false;
	bool r8g8_srgb_views = false;
	bool relaxed_float = false;
	bool ray_query = false;
	bool compute_derivative_quads = true;
	bool compute_derivative_linear = true;
	bool lds_lane_sync = true;
	bool native_float16 = true;
	bool f32_product_no_contraction = true;
	bool subgroup_arithmetic = true;
	bool subgroup_ballot = true;
	bool subgroup_shuffle = true;
	bool subgroup_quad = true;
	uint32_t subgroup_stages = ~0u;
	bool shader_resource_min_lod = true;
	bool mesh_lds_padding = false;
};

inline constexpr Profile DefaultProfile {};

struct SplitHints;

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t       push_data_start_dword = 0,
                                  const Profile& profile               = DefaultProfile,
                                  bool           emit_debug_info       = false,
                                  bool           mip_stats_records     = true,
                                  SplitHints*    split_hints           = nullptr, uint8_t* bvh_paths = nullptr);

[[nodiscard]] bool     UsesMipStats(const IR::Program& program);
[[nodiscard]] uint32_t MeshWorkgroupBytes(std::span<const uint32_t> spirv);
}

#endif

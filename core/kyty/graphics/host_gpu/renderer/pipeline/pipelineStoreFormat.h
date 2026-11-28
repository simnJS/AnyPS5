#pragma once

#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shader.h"

#include <xxhash.h>
#include <zstd.h>

namespace Libs::Graphics {
struct ShaderProgramKey;
}

namespace Libs::Graphics::ShaderStoreFormat {

inline constexpr uint32_t ShaderStoreFormatVersion = 1;

enum class Record : uint8_t { Module, GraphicsPipeline, TessellationLayout, TableModule, NativeGraphicsPipeline };

struct Frame {
	uint32_t compressed_size = 0;
	uint32_t raw_size        = 0;
	uint64_t checksum        = 0;
};

struct Key {
	ShaderType stage = ShaderType::Unknown;
	uint64_t hash = 0;
	uint32_t user_data_count = 0;
	std::vector<uint32_t> static_state;
};

[[nodiscard]] bool EncodeTableModuleRecord(const ShaderProgramKey& key, uint32_t cursor,
                                           const ShaderModule& module, const std::vector<uint32_t>& spirv,
                                           const std::vector<uint32_t>& plain_spirv, std::vector<uint8_t>& record);
[[nodiscard]] bool DecodeTableModuleRecord(ShaderRecompiler::IR::Reader& reader, ShaderProgramKey& key,
                                           uint32_t& cursor, ShaderModule& module, std::vector<uint32_t>& spirv,
                                           std::vector<uint32_t>& plain_spirv);
template <class Stream, class Stamp, class Profile, class Limit>
void IoStoreHeader(Stream& io, Stamp& stamp, Profile& profile, Limit& max_shared_memory) {
	ShaderRecompiler::IR::IoAll(io, stamp, profile, max_shared_memory);
}

template <class Stream, class Module>
void IoStoreModuleInfo(Stream& io, Module& module) {
	ShaderRecompiler::IR::IoAll(io, module.id, module.info, module.specialization, module.vertex_formats,
	                            module.rect_parameters, module.mesh_task, module.guest_traversal);
}

template <class Stream, class Program, class Cursor, class Module, class Spirv>
void IoStoreTableModule(Stream& io, Program& key, Cursor& cursor, Module& module, Spirv& spirv, Spirv& plain_spirv) {
	using ShaderRecompiler::IR::IoAll;
	IoAll(io, key.stage, key.hash, key.user_data_count, key.static_state, cursor);
	IoStoreModuleInfo(io, module);
	IoAll(io, spirv, plain_spirv);
}

template <class Stream, class Module>
void IoRayTracingMode(Stream& io, Module& module) {
	if constexpr (ShaderRecompiler::IR::Reading<Stream>) {
		if (io.bytes.empty()) return;
		ShaderRecompiler::IR::IoAll(io, module.bvh, module.rt_mode);
		io.ok = io.ok && (module.bvh == ShaderBvh::Bvh4 || module.bvh == ShaderBvh::Bvh8Only);
	} else if (module.bvh != ShaderBvh::None) {
		ShaderRecompiler::IR::IoAll(io, module.bvh, module.rt_mode);
	}
}

template <class Stream, class Program, class Module, class Plan, class Spirv>
void IoStoreModule(Stream& io, Program& key, Module& module, Plan& plan, Spirv& spirv, Spirv& plain_spirv) {
	using ShaderRecompiler::IR::IoAll;
	IoAll(io, key.stage, key.hash, key.user_data_count, key.static_state, plan);
	IoStoreModuleInfo(io, module);
	IoAll(io, spirv, plain_spirv);
	IoRayTracingMode(io, module);
}

inline bool DecodeShaderStoreFrame(const Frame& frame, std::span<const uint8_t> compressed,
                                   std::vector<uint8_t>& raw) {
	if (compressed.size() != frame.compressed_size ||
	    XXH3_64bits(compressed.data(), compressed.size()) != frame.checksum ||
	    ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != frame.raw_size) {
		return false;
	}
	raw.resize(frame.raw_size);
	return ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size()) == raw.size();
}

}

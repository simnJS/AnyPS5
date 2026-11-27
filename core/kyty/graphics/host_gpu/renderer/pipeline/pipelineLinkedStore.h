#pragma once

#include "graphics/host_gpu/memoryTelemetry.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"

#include <istream>

namespace Libs::Graphics {

class LinkedComputeStore {
public:
	struct Entry {
		ShaderStoreFormat::Key key;
		std::vector<uint32_t> caller;
		ShaderModule module;
		std::vector<uint32_t> spirv;
	};

	bool Read(std::istream& input, std::span<const uint8_t> header);
	const Entry* Find(uint64_t hash, uint32_t user_data_count, std::span<const uint32_t> static_state,
	                  std::span<const uint32_t> caller, std::span<const uint32_t> functions,
	                  const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint32_t cursor) const;

private:
	MemoryTelemetry::PipelineBytes m_bytes;
	std::unordered_multimap<uint64_t, Entry> m_entries;
};

}

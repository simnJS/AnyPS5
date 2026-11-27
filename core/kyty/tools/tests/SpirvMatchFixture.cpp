#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"

#include <fstream>

namespace {

using namespace Libs::Graphics;
using namespace ShaderStoreFormat;
using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Writer;

struct NoIndexedReadsWriter: Writer {
	static constexpr bool shader_info_has_indexed_buffer_reads      = false;
	static constexpr bool shader_info_has_ray_query_header_programs = false;
};

struct NoHeaderProgramsWriter: Writer {
	static constexpr bool shader_info_has_ray_query_header_programs = false;
};

struct NoExtraSourcesWriter: Writer {
	static constexpr bool shader_info_has_ray_query_extra_sources   = false;
	static constexpr bool shader_info_has_extended_resources        = false;
	static constexpr bool shader_info_has_ray_query_header_programs = false;
};

struct CommonWriter: NoExtraSourcesWriter {
	static constexpr bool shader_info_has_indexed_buffer_reads = false;
};

struct LegacyWriter: CommonWriter {
	static constexpr bool shader_info_has_relative_vgprs = false;
};

void Write(std::ostream& output, const auto& bytes) {
	output.write(reinterpret_cast<const char*>(bytes.data()), std::span(bytes).size_bytes());
}

void WriteDump(const char* path, const std::vector<uint32_t>& words) {
	std::ofstream output(path, std::ios::binary);
	Write(output, words);
}

void WriteFrame(std::ostream& output, const Writer& record) {
	std::vector<uint8_t> compressed(ZSTD_compressBound(record.bytes.size()));
	compressed.resize(ZSTD_compress(compressed.data(), compressed.size(), record.bytes.data(), record.bytes.size(), 3));
	const Frame frame{static_cast<uint32_t>(compressed.size()), static_cast<uint32_t>(record.bytes.size()),
	                  XXH3_64bits(compressed.data(), compressed.size())};
	Write(output, std::span(&frame, 1));
	Write(output, compressed);
}

template <class ModuleWriter = Writer>
void WriteModule(std::ostream& output, uint64_t hash, uint64_t id, ShaderType stage,
                 const std::vector<uint32_t>& spirv, const std::vector<uint32_t>& plain, Record type = Record::Module) {
	ModuleWriter record;
	Key key{.stage = stage, .hash = hash};
	ShaderModule module;
	module.id = id;
	module.info.stage = stage;
	module.info.info.images.emplace_back().paired_view = 3;
	module.info.info.samplers.emplace_back().runtime_key = 5;
	module.info.bindings.clip_plane_dwords = 8;
	module.info.info.ray_query_extra_sources.push_back({1, 4, 8, 12, {1, 2, 3, 4}});
	module.info.info.indexed_buffer_reads.push_back({2, 3, 12, {4, 16, 0}});
	IoAll(record, type);
	if (type == Record::TableModule) {
		const uint32_t cursor = 0;
		IoStoreTableModule(record, key, cursor, module, spirv, plain);
	} else {
		const std::vector<uint8_t> plan;
		IoStoreModule(record, key, module, plan, spirv, plain);
	}
	WriteFrame(output, record);
}

}

int main() {
	const std::vector<uint32_t> vertex{0x07230203, 0x00010000, 1, 5, 0};
	const std::vector<uint32_t> pixel{0x07230203, 0x00010000, 2, 5, 0};
	const std::vector<uint32_t> plain{0x07230203, 0x00010000, 3, 5, 0};
	const std::vector<uint32_t> missing{0x07230203, 0x00010000, 4, 5, 0};
	const uint64_t stamp = 1;
	const uint32_t shared_memory = 32768;
	Writer header;
	IoStoreHeader(header, stamp, ShaderRecompiler::Spirv::DefaultProfile, shared_memory);
	std::ofstream store("fixture.shaders", std::ios::binary);
	Write(store, header.bytes);
	WriteModule(store, 0x102, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule(store, 0x101, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule(store, 0x101, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule(store, 0x103, 0x300, ShaderType::Pixel, pixel, plain, Record::TableModule);
	Writer pipeline;
	IoAll(pipeline, Record::GraphicsPipeline);
	const GraphicsPipelineKey key{};
	pipeline.Raw(&key, sizeof(key));
	WriteFrame(store, pipeline);
	Writer native;
	IoAll(native, Record::NativeGraphicsPipeline);
	native.Raw(&key, sizeof(key));
	WriteFrame(store, native);
	store.close();
	std::ofstream legacy("legacy.shaders", std::ios::binary);
	Writer old_header;
	const bool off = false;
	const bool on = true;
	const uint32_t clock_period = 0;
	IoAll(old_header, stamp, off, off, clock_period, off, off, off, off, off, on, shared_memory);
	Write(legacy, old_header.bytes);
	WriteModule<LegacyWriter>(legacy, 0x102, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule<LegacyWriter>(legacy, 0x101, 0x201, ShaderType::Vertex, vertex, {});
	std::ofstream previous("previous.shaders", std::ios::binary);
	Writer previous_header;
	const uint32_t all_stages = ~0u;
	IoAll(previous_header, stamp, off, off, clock_period, off, off, off, off, off,
	      on, on, on, on, on, on, on, all_stages, shared_memory);
	Write(previous, previous_header.bytes);
	WriteModule<CommonWriter>(previous, 0x102, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule<CommonWriter>(previous, 0x101, 0x201, ShaderType::Vertex, vertex, {});
	std::ofstream intermediate("intermediate.shaders", std::ios::binary);
	Write(intermediate, header.bytes);
	WriteModule<CommonWriter>(intermediate, 0x102, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule<CommonWriter>(intermediate, 0x101, 0x201, ShaderType::Vertex, vertex, {});
	intermediate.close();
	std::ofstream no_extra("no-extra-sources.shaders", std::ios::binary);
	Writer min_lod_header;
	std::array<bool, 14> subgroup_flags {};
	IoAll(min_lod_header, stamp, off, off, clock_period, subgroup_flags, all_stages, on, shared_memory);
	Write(no_extra, min_lod_header.bytes);
	WriteModule<NoExtraSourcesWriter>(no_extra, 0x102, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule<NoExtraSourcesWriter>(no_extra, 0x101, 0x201, ShaderType::Vertex, vertex, {});
	no_extra.close();
	std::ofstream no_programs("no-header-programs.shaders", std::ios::binary);
	Write(no_programs, header.bytes);
	WriteModule<NoHeaderProgramsWriter>(no_programs, 0x102, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule<NoHeaderProgramsWriter>(no_programs, 0x101, 0x201, ShaderType::Vertex, vertex, {});
	no_programs.close();

	std::ofstream no_indexed("no-indexed-reads.shaders", std::ios::binary);
	Writer subgroup_header;
	IoAll(subgroup_header, stamp, off, off, clock_period, subgroup_flags, all_stages, shared_memory);
	Write(no_indexed, subgroup_header.bytes);
	WriteModule<NoIndexedReadsWriter>(no_indexed, 0x102, 0x201, ShaderType::Vertex, vertex, {});
	WriteModule<NoIndexedReadsWriter>(no_indexed, 0x101, 0x201, ShaderType::Vertex, vertex, {});

	std::ofstream malformed("malformed.shaders", std::ios::binary);
	Write(malformed, header.bytes);
	WriteModule(malformed, 0x101, 0x201, ShaderType::Vertex, vertex, {});
	const Writer invalid{{0xff}};
	WriteFrame(malformed, invalid);
	Writer invalid_header = std::move(header);
	invalid_header.bytes[sizeof(stamp)] = 2;
	std::ofstream bad_header("bad-header.shaders", std::ios::binary);
	Write(bad_header, invalid_header.bytes);
	WriteDump("vs.spv", vertex);
	WriteDump("ps.spv", pixel);
	WriteDump("plain.spv", plain);
	WriteDump("missing.spv", missing);
	WriteDump("invalid.spv", {0, 0, 0, 0, 0});
	return !store || !legacy || !previous || !intermediate || !no_extra || !no_indexed || !malformed || !bad_header;
}

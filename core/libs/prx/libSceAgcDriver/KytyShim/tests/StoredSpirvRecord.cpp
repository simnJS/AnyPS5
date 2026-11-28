#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"
#include "graphics/host_gpu/renderer/pipeline/storedSpirv.h"

#include <xxhash.h>
#include <zstd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

bool g_failed = false;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", message);
		g_failed = true;
	}
}

std::vector<uint8_t> Framed(const std::vector<uint8_t>& record) {
	std::vector<uint8_t> bytes(sizeof(ShaderStoreFormat::Frame) + ZSTD_compressBound(record.size()));
	const auto size = ZSTD_compress(bytes.data() + sizeof(ShaderStoreFormat::Frame),
	                                bytes.size() - sizeof(ShaderStoreFormat::Frame), record.data(), record.size(), 3);
	if (ZSTD_isError(size)) return {};
	const ShaderStoreFormat::Frame frame {.compressed_size = static_cast<uint32_t>(size),
	                                      .raw_size        = static_cast<uint32_t>(record.size()),
	                                      .checksum = XXH3_64bits(bytes.data() + sizeof(ShaderStoreFormat::Frame), size)};
	std::memcpy(bytes.data(), &frame, sizeof(frame));
	bytes.resize(sizeof(frame) + size);
	return bytes;
}

uint64_t Checksum(const std::vector<uint8_t>& frame) {
	ShaderStoreFormat::Frame header;
	std::memcpy(&header, frame.data(), sizeof(header));
	return header.checksum;
}

}

extern "C" int RunKytyStoredSpirvRecordTest() {
	ShaderModule module;
	module.id               = 0x1234;
	module.info.stage       = ShaderType::Compute;
	module.info.shader_hash = 0x5678;
	const std::vector<uint32_t> spirv {0x07230203u, 0x00010600u, 0, 17, 0, 0x00020011u, 1};
	const std::vector<uint32_t> plain {0x07230203u, 0x00010600u, 0, 9, 0};
	ShaderRecompiler::IR::ResourcePlan plan;
	std::vector<uint8_t>               record;
	Check(EncodeModuleRecord(ShaderType::Compute, 0x5678, 4, {1, 2}, plan, module, spirv, plain, record),
	      "a module record is encoded");
	const auto frame = Framed(record);
	Check(!frame.empty(), "a module record is framed");
	const auto checksum = Checksum(frame);

	std::vector<uint32_t> decoded;
	Check(DecodeStoredSpirv(frame, checksum, false, decoded) && decoded == spirv,
	      "the module's SPIR-V is decoded from its frame");
	Check(DecodeStoredSpirv(frame, checksum, true, decoded) && decoded == plain,
	      "the plain SPIR-V is decoded from its frame");
	Check(!DecodeStoredSpirv(frame, checksum ^ 1u, false, decoded),
	      "a frame whose checksum differs from the one seen at load is refused");

	std::vector<uint8_t> no_plain_record;
	Check(EncodeModuleRecord(ShaderType::Compute, 0x5678, 4, {}, plan, module, spirv, {}, no_plain_record),
	      "a module record without plain SPIR-V is encoded");
	const auto no_plain = Framed(no_plain_record);
	Check(!DecodeStoredSpirv(no_plain, Checksum(no_plain), true, decoded), "a module without plain SPIR-V gives none");

	auto damaged = frame;
	damaged.back() ^= 0x5au;
	Check(!DecodeStoredSpirv(damaged, checksum, false, decoded), "a damaged frame is refused");
	Check(!DecodeStoredSpirv(std::span(frame).first(frame.size() - 1), checksum, false, decoded),
	      "a truncated frame is refused");
	Check(!DecodeStoredSpirv(std::span(frame).first(4), checksum, false, decoded),
	      "a frame shorter than its header is refused");

	std::vector<uint8_t> pipeline_record {static_cast<uint8_t>(ShaderStoreFormat::Record::GraphicsPipeline), 0, 0, 0};
	const auto pipeline = Framed(pipeline_record);
	Check(!DecodeStoredSpirv(pipeline, Checksum(pipeline), false, decoded), "a pipeline record has no SPIR-V");

	const auto path = std::filesystem::temp_directory_path() /
	                  ("kyty-stored-spirv-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	const std::vector<uint8_t> header(38, 0x5a);
	{
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
		file.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
		file.write(reinterpret_cast<const char*>(pipeline.data()), static_cast<std::streamsize>(pipeline.size()));
		file.write(reinterpret_cast<const char*>(frame.data()), static_cast<std::streamsize>(frame.size()));
	}
	const uint64_t offset = header.size() + pipeline.size();
	if (const auto store = StoreFile::Open(path); store == nullptr) {
		Check(false, "the store file opens for reading");
	} else {
		std::vector<uint8_t> read(frame.size());
		Check(store->Read(offset, read) && read == frame, "a frame is read again at its offset in the store");
		Check(DecodeStoredSpirv(read, checksum, false, decoded) && decoded == spirv,
		      "the SPIR-V read again from the store is the one that was stored");
		std::filesystem::resize_file(path, offset + frame.size() / 2);
		Check(!store->Read(offset, read), "a frame cut by a truncated store is not read");
		std::vector<uint8_t> beyond(16);
		Check(!store->Read(offset + frame.size() * 4, beyond), "a read past the end of the store fails");
	}
	std::error_code error;
	std::filesystem::remove(path, error);
	Check(StoreFile::Open(path) == nullptr, "a missing store file does not open");

	if (g_failed) return 1;
	std::puts("kyty_shim_stored_spirv_record: ok");
	return 0;
}

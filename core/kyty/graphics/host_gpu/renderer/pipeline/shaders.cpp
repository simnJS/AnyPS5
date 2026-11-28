#include "common/assert.h"
#include "common/common.h"
#if defined(ANYPS5_KYTY)
#include "common/logging/log.h"
#endif
#include "common/profiler.h"
#include "gpu_blit_shaders/gpu_mesh_indirect_spv.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#endif
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <array>
#include <bit>
#include <cinttypes>
#if defined(ANYPS5_KYTY)
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <xxhash.h>
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <process.h>
#else
#include <unistd.h>
#endif
#endif
#include <fmt/format.h>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {

namespace {

struct DriverCreation {
	bool        active = false;
	const char* names[4] {};
	uint64_t    hashes[4] {};
	uint64_t    words = 0;
};

thread_local DriverCreation t_driver_creation;

constexpr size_t                                   MaxDriverCreations = 64;
std::array<std::atomic<int64_t>, MaxDriverCreations> g_driver_creations {};

int64_t SteadyNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

class ScopedDriverCreation {
public:
	ScopedDriverCreation(const std::array<const char*, 4>& names,
	                     const std::array<uint64_t, 4>&    hashes) {
		t_driver_creation.active = true;
		t_driver_creation.words  = 0;
		for (size_t i = 0; i < 4; i++) {
			t_driver_creation.names[i]  = names[i];
			t_driver_creation.hashes[i] = hashes[i];
		}
		const int64_t now = SteadyNs();
		for (size_t i = 0; i < MaxDriverCreations; i++) {
			int64_t free = 0;
			if (g_driver_creations[i].compare_exchange_strong(free, now, std::memory_order_acq_rel)) {
				m_slot = static_cast<int>(i);
				break;
			}
		}
	}
	~ScopedDriverCreation() {
		t_driver_creation = {};
		if (m_slot >= 0) {
			g_driver_creations[static_cast<size_t>(m_slot)].store(0, std::memory_order_release);
		}
	}

	ScopedDriverCreation(const ScopedDriverCreation&)            = delete;
	ScopedDriverCreation& operator=(const ScopedDriverCreation&) = delete;

private:
	int m_slot = -1;
};

#if defined(ANYPS5_KYTY)
bool PipelineCompileAuditEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PIPELINE_COMPILE_AUDIT");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return enabled;
}

vk::Result CreateAuditedPipeline(const GraphicContext& graphics, const char* kind, uint64_t variant,
                                 vk::PipelineCreateFlags flags, vk::Pipeline& out, auto&& create) {
	if (!PipelineCompileAuditEnabled()) {
		return create(flags);
	}

	constexpr auto no_compile = vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
	const bool     lookup     = static_cast<bool>(flags & no_compile);
	const auto     log        = [&](const char* event, vk::Result result) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-driver-pipeline] kind={} variant={:016x} mode={} event={} result={} monotonic_ns={}\n",
		    kind, variant, lookup ? "lookup" : "create", event, vk::to_string(result), SteadyNs()));
	};
	if (!graphics.pipeline_cache_lookup_enabled) {
		log("unverified", vk::Result::eErrorFeatureNotPresent);
		return create(flags);
	}

	const auto probe = create(flags | no_compile);
	if (probe == vk::Result::eSuccess && out) {
		log("no_compile", probe);
		return probe;
	}
	const bool compile_required = probe == vk::Result::ePipelineCompileRequired;
	log(compile_required ? "compile_required" : "unverified", probe);
	if (lookup) {
		return probe;
	}
	if (out) {
		MemoryTelemetry::Destroy(graphics.device, out);
		out = nullptr;
	}
	return create(flags);
}

constexpr uint32_t                MaxComputeBinaryCount = 16;
constexpr uint32_t                MaxComputeBinaryBytes = 32u << 20;
constexpr uint32_t                MinComputeBinaryWords = 50'000;
constexpr std::array<char, 8>     ComputeBinaryMagic {'K', 'Y', 'T', 'Y', 'C', 'B', '1', '\0'};
std::atomic<uint64_t>             g_compute_binary_files {0};

struct BinaryLookup {
	const char* reason = "absent";
	vk::Result  result = vk::Result::eSuccess;
};

struct LastBinaryLookup {
	uint64_t     pipeline = 0;
	BinaryLookup lookup;
};

thread_local LastBinaryLookup t_last_binary_lookup;

std::string DriverCreationName() {
	std::string text;
	for (size_t i = 0; i < 4; i++) {
		if (t_driver_creation.names[i] == nullptr) {
			continue;
		}
		if (!text.empty()) {
			text += ' ';
		}
		text += fmt::format("{}={:016x}", t_driver_creation.names[i], t_driver_creation.hashes[i]);
	}
	return text;
}

std::string BinaryLookupText(const BinaryLookup& lookup) {
	if (lookup.result == vk::Result::eSuccess) {
		return lookup.reason;
	}
	return fmt::format("{}:{}", lookup.reason, vk::to_string(lookup.result));
}

void LogDriverCompile(const char* kind, std::string_view file, std::string_view binary, std::string_view cache,
                      std::chrono::steady_clock::duration elapsed, vk::Result result, bool saved) {
	Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-pipeline-compile] kind={} {} words={} file={} binary={} cache={} ms={:.1f} result={} saved={}\n", kind,
	    DriverCreationName(), t_driver_creation.words, file, binary, cache,
	    std::chrono::duration<double, std::milli>(elapsed).count(), vk::to_string(result), saved ? 1 : 0));
}

void LogBinaryRefused(const char* kind, std::string_view file, uint64_t pipeline, const BinaryLookup& lookup) {
	const auto& last = t_last_binary_lookup;
	if (last.pipeline == pipeline && last.lookup.reason == lookup.reason && last.lookup.result == lookup.result) {
		return;
	}
	Log::WriteToConsoleAndLog(fmt::format("[kyty-pipeline-binary] refused kind={} {} words={} file={} reason={}\n", kind,
	                                      DriverCreationName(), t_driver_creation.words, file,
	                                      BinaryLookupText(lookup)));
}

struct ComputeBinaryKey {
	uint32_t                                                  size = 0;
	std::array<uint8_t, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR> bytes {};

	[[nodiscard]] bool     Valid() const { return size != 0 && size <= bytes.size(); }
	[[nodiscard]] uint64_t Hash() const { return XXH3_64bits(bytes.data(), size); }
	bool                   operator==(const ComputeBinaryKey&) const = default;
};

ComputeBinaryKey ComputeBinaryDeviceKey(const GraphicContext& graphics) {
	const auto&    properties = graphics.physical_device_properties;
	const uint32_t ids[] {properties.vendorID, properties.deviceID, properties.driverVersion,
	                      static_cast<uint32_t>(graphics.driver_id)};
	ComputeBinaryKey key;
	static_assert(sizeof(ids) + VK_UUID_SIZE == VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR);
	key.size = VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR;
	std::memcpy(key.bytes.data(), ids, sizeof(ids));
	std::memcpy(key.bytes.data() + sizeof(ids), properties.pipelineCacheUUID.data(), VK_UUID_SIZE);
	return key;
}

ComputeBinaryKey ComputeBinaryKeyOf(const vk::PipelineBinaryKeyKHR& key) {
	ComputeBinaryKey out;
	if (key.keySize == 0 || key.keySize > out.bytes.size()) {
		return out;
	}
	out.size = key.keySize;
	std::memcpy(out.bytes.data(), key.key.data(), key.keySize);
	return out;
}

struct ComputeBinaryHeader {
	std::array<char, 8> magic = ComputeBinaryMagic;
	ComputeBinaryKey    device;
	ComputeBinaryKey    pipeline;
	uint32_t            count = 0;
};
static_assert(sizeof(ComputeBinaryHeader) == 84);
static_assert(std::is_trivially_copyable_v<ComputeBinaryHeader>);

struct ComputeBinaryRecord {
	ComputeBinaryKey key;
	uint32_t         size     = 0;
	uint64_t         checksum = 0;
};
static_assert(sizeof(ComputeBinaryRecord) == 48);
static_assert(std::is_trivially_copyable_v<ComputeBinaryRecord>);

constexpr size_t MaxComputeBinaryFile =
    sizeof(ComputeBinaryHeader) + MaxComputeBinaryCount * sizeof(ComputeBinaryRecord) + MaxComputeBinaryBytes;

struct ComputeBinaryHandles {
	vk::Device                         device;
	std::vector<vk::PipelineBinaryKHR> handles;

	ComputeBinaryHandles(vk::Device owner, uint32_t count): device(owner), handles(count) {}
	~ComputeBinaryHandles() {
		for (const auto handle: handles) {
			if (handle) {
				device.destroyPipelineBinaryKHR(handle, nullptr);
			}
		}
	}
	ComputeBinaryHandles(const ComputeBinaryHandles&)            = delete;
	ComputeBinaryHandles& operator=(const ComputeBinaryHandles&) = delete;
};

template <class CreateInfo>
std::optional<vk::Result> RestorePipelineBinary(vk::Device device, const std::filesystem::path& path,
                                                const ComputeBinaryHeader& expected, CreateInfo info,
                                                vk::Pipeline& pipeline, BinaryLookup& lookup) {
	lookup = {};
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		return std::nullopt;
	}
	const auto end = file.tellg();
	lookup.reason  = "size";
	if (end < static_cast<std::streamoff>(sizeof(ComputeBinaryHeader)) ||
	    end > static_cast<std::streamoff>(MaxComputeBinaryFile)) {
		return std::nullopt;
	}
	std::vector<uint8_t> bytes(static_cast<size_t>(end));
	file.seekg(0);
	lookup.reason = "read";
	if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
		return std::nullopt;
	}
	ComputeBinaryHeader header;
	std::memcpy(&header, bytes.data(), sizeof(header));
	lookup.reason = "header";
	if (header.magic != ComputeBinaryMagic || !header.device.Valid() || !header.pipeline.Valid() ||
	    header.device != expected.device || header.pipeline != expected.pipeline || header.count == 0 ||
	    header.count > MaxComputeBinaryCount) {
		return std::nullopt;
	}
	size_t offset = sizeof(header);
	lookup.reason = "records";
	if (bytes.size() - offset < header.count * sizeof(ComputeBinaryRecord)) {
		return std::nullopt;
	}
	std::vector<ComputeBinaryRecord> records(header.count);
	std::memcpy(records.data(), bytes.data() + offset, header.count * sizeof(ComputeBinaryRecord));
	offset += header.count * sizeof(ComputeBinaryRecord);
	std::vector<vk::PipelineBinaryKeyKHR>  keys(header.count);
	std::vector<vk::PipelineBinaryDataKHR> data(header.count);
	uint64_t                               total = 0;
	for (uint32_t i = 0; i < header.count; i++) {
		const auto& record = records[i];
		total += record.size;
		if (!record.key.Valid() || record.size == 0 || total > MaxComputeBinaryBytes ||
		    bytes.size() - offset < record.size) {
			return std::nullopt;
		}
		if (XXH3_64bits(bytes.data() + offset, record.size) != record.checksum) {
			lookup.reason = "checksum";
			return std::nullopt;
		}
		keys[i].keySize = record.key.size;
		std::memcpy(keys[i].key.data(), record.key.bytes.data(), record.key.size);
		data[i].dataSize = record.size;
		data[i].pData    = bytes.data() + offset;
		offset += record.size;
	}
	if (offset != bytes.size()) {
		lookup.reason = "size";
		return std::nullopt;
	}
	const vk::PipelineBinaryKeysAndDataKHR keys_and_data {.binaryCount         = header.count,
	                                                      .pPipelineBinaryKeys = keys.data(),
	                                                      .pPipelineBinaryData = data.data()};
	const vk::PipelineBinaryCreateInfoKHR  create {.pKeysAndDataInfo = &keys_and_data};
	ComputeBinaryHandles                   handles(device, header.count);
	vk::PipelineBinaryHandlesInfoKHR       created {.pipelineBinaryCount = header.count,
	                                                .pPipelineBinaries   = handles.handles.data()};
	auto result = device.createPipelineBinariesKHR(&create, nullptr, &created);
	if (result == vk::Result::eErrorDeviceLost) {
		return result;
	}
	if (result != vk::Result::eSuccess) {
		lookup = {.reason = "driver-binaries", .result = result};
		return std::nullopt;
	}
	vk::PipelineBinaryInfoKHR binaries {.pNext             = info.pNext,
	                                    .binaryCount       = header.count,
	                                    .pPipelineBinaries = handles.handles.data()};
	info.pNext = &binaries;
	result     = MemoryTelemetry::Create(device, nullptr, info, pipeline);
	if (result == vk::Result::eSuccess && pipeline) {
		lookup = {.reason = "restored"};
		return result;
	}
	if (pipeline) {
		MemoryTelemetry::Destroy(device, pipeline);
		pipeline = nullptr;
	}
	if (result == vk::Result::eErrorDeviceLost) {
		return result;
	}
	lookup = {.reason = "driver-pipeline", .result = result};
	return std::nullopt;
}

uint64_t ProcessId() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return static_cast<uint64_t>(_getpid());
#else
	return static_cast<uint64_t>(getpid());
#endif
}

bool WriteComputeBinaryFile(const std::filesystem::path& path, const ComputeBinaryHeader& header,
                            std::span<const ComputeBinaryRecord> records,
                            std::span<const std::vector<uint8_t>> blobs) {
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	auto temporary = path;
	temporary += fmt::format(".{}.{}.{}.tmp", ProcessId(), SteadyNs(),
	                         g_compute_binary_files.fetch_add(1, std::memory_order_relaxed));
	bool written = false;
	{
		std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
		if (file) {
			file.write(reinterpret_cast<const char*>(&header), sizeof(header));
			file.write(reinterpret_cast<const char*>(records.data()),
			           static_cast<std::streamsize>(records.size_bytes()));
			for (const auto& blob: blobs) {
				file.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(blob.size()));
			}
			file.close();
			written = static_cast<bool>(file);
		}
	}
	if (written) {
		std::filesystem::rename(temporary, path, error);
		written = !error;
	}
	if (!written) {
		std::filesystem::remove(temporary, error);
	}
	return written;
}

bool SaveComputeBinary(vk::Device device, const std::filesystem::path& path, ComputeBinaryHeader header,
                       vk::Pipeline pipeline) {
	const vk::PipelineBinaryCreateInfoKHR create {.pipeline = pipeline};
	vk::PipelineBinaryHandlesInfoKHR      counted {};
	if (device.createPipelineBinariesKHR(&create, nullptr, &counted) != vk::Result::eSuccess ||
	    counted.pipelineBinaryCount == 0 || counted.pipelineBinaryCount > MaxComputeBinaryCount) {
		return false;
	}
	ComputeBinaryHandles             handles(device, counted.pipelineBinaryCount);
	vk::PipelineBinaryHandlesInfoKHR created {.pipelineBinaryCount = counted.pipelineBinaryCount,
	                                          .pPipelineBinaries   = handles.handles.data()};
	if (device.createPipelineBinariesKHR(&create, nullptr, &created) != vk::Result::eSuccess ||
	    created.pipelineBinaryCount != counted.pipelineBinaryCount) {
		return false;
	}
	std::vector<ComputeBinaryRecord>  records(created.pipelineBinaryCount);
	std::vector<std::vector<uint8_t>> blobs(created.pipelineBinaryCount);
	uint64_t                          total = 0;
	for (uint32_t i = 0; i < created.pipelineBinaryCount; i++) {
		const vk::PipelineBinaryDataInfoKHR info {.pipelineBinary = handles.handles[i]};
		vk::PipelineBinaryKeyKHR            key {};
		size_t                              size = 0;
		if (device.getPipelineBinaryDataKHR(&info, &key, &size, nullptr) != vk::Result::eSuccess || size == 0 ||
		    size > MaxComputeBinaryBytes - total) {
			return false;
		}
		blobs[i].resize(size);
		if (device.getPipelineBinaryDataKHR(&info, &key, &size, blobs[i].data()) != vk::Result::eSuccess ||
		    size != blobs[i].size()) {
			return false;
		}
		records[i].key      = ComputeBinaryKeyOf(key);
		records[i].size     = static_cast<uint32_t>(size);
		records[i].checksum = XXH3_64bits(blobs[i].data(), size);
		if (!records[i].key.Valid()) {
			return false;
		}
		total += size;
	}
	header.count = created.pipelineBinaryCount;
	return WriteComputeBinaryFile(path, header, records, blobs);
}

template <class CreateInfo>
std::optional<vk::Result> CreatePipelineWithBinary(const GraphicContext& graphics, CreateInfo info,
                                                   vk::PipelineCache driver_cache, ComputePipelineCacheMode cache_mode,
                                                   vk::Pipeline& pipeline, const char*& mode) {
	const auto device = graphics.device;
	info.flags &= ~vk::PipelineCreateFlags(vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired);
	vk::PipelineBinaryKeyKHR  exact {};
	vk::PipelineCreateInfoKHR keyed {.pNext = &info};
	if (device.getPipelineKeyKHR(&keyed, &exact) != vk::Result::eSuccess) {
		return std::nullopt;
	}
	const ComputeBinaryHeader header {.device = ComputeBinaryDeviceKey(graphics), .pipeline = ComputeBinaryKeyOf(exact)};
	if (!header.device.Valid() || !header.pipeline.Valid()) {
		return std::nullopt;
	}
	const auto path = graphics.compute_pipeline_binary_directory /
	                  fmt::format("{:016x}-{:016x}.bin", header.device.Hash(), header.pipeline.Hash());
	constexpr const char* kind = std::is_same_v<CreateInfo, vk::ComputePipelineCreateInfo> ? "compute" : "graphics";
	const auto            name = path.filename().string();
	const auto            hash = header.pipeline.Hash();
	BinaryLookup          lookup {.reason = "skipped"};
	if (cache_mode != ComputePipelineCacheMode::KnownMiss) {
		const auto restored = RestorePipelineBinary(device, path, header, info, pipeline, lookup);
		if (restored) {
			mode = "binary";
			return restored;
		}
		if (std::string_view(lookup.reason) != "absent") {
			LogBinaryRefused(kind, name, hash, lookup);
		}
		t_last_binary_lookup = {.pipeline = hash, .lookup = lookup};
	} else if (t_last_binary_lookup.pipeline == hash) {
		lookup = t_last_binary_lookup.lookup;
	}
	if (cache_mode == ComputePipelineCacheMode::Lookup) {
		return std::nullopt;
	}
	if (driver_cache && !graphics.pipeline_cache_lookup_enabled) {
		return std::nullopt;
	}
	const char* cache = cache_mode == ComputePipelineCacheMode::KnownMiss ? "miss" : "none";
	if (cache_mode == ComputePipelineCacheMode::Full && driver_cache) {
		auto probe = info;
		probe.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
		const auto found = MemoryTelemetry::Create(device, driver_cache, probe, pipeline);
		if (found == vk::Result::eSuccess && pipeline) {
			mode = "lookup";
			return found;
		}
		if (pipeline) {
			MemoryTelemetry::Destroy(device, pipeline);
			pipeline = nullptr;
		}
		if (found == vk::Result::eErrorDeviceLost) {
			return found;
		}
		cache = "miss";
	}
	const vk::PipelineCreateFlags2CreateInfoKHR flags {
	    .pNext = info.pNext,
	    .flags = vk::PipelineCreateFlags2KHR(static_cast<VkPipelineCreateFlags2KHR>(
	                 static_cast<VkPipelineCreateFlags>(info.flags))) |
	             vk::PipelineCreateFlagBits2KHR::eCaptureDataKHR};
	auto capture  = info;
	capture.pNext = &flags;
	const auto started = std::chrono::steady_clock::now();
	const auto result  = MemoryTelemetry::Create(device, nullptr, capture, pipeline);
	const auto elapsed = std::chrono::steady_clock::now() - started;
	if (result != vk::Result::eSuccess || !pipeline) {
		if (pipeline) {
			MemoryTelemetry::Destroy(device, pipeline);
			pipeline = nullptr;
		}
		LogDriverCompile(kind, name, BinaryLookupText(lookup), cache, elapsed, result, false);
		if (result == vk::Result::eErrorDeviceLost) {
			return result;
		}
		return std::nullopt;
	}
	const bool saved = SaveComputeBinary(device, path, header, pipeline);
	LogDriverCompile(kind, name, BinaryLookupText(lookup), cache, elapsed, result, saved);
	const vk::ReleaseCapturedPipelineDataInfoKHR release {.pipeline = pipeline};
	static_cast<void>(device.releaseCapturedPipelineDataKHR(&release, nullptr));
	mode = "binary-capture";
	return result;
}
#endif

}

#if defined(ANYPS5_KYTY)
static bool PipelineBinariesUsable(const GraphicContext& graphics) {
	const auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
	return graphics.compute_pipeline_binaries && !graphics.compute_pipeline_binary_directory.empty() &&
	       !PipelineCompileAuditEnabled() && dispatcher.vkGetPipelineKeyKHR != nullptr &&
	       dispatcher.vkCreatePipelineBinariesKHR != nullptr && dispatcher.vkDestroyPipelineBinaryKHR != nullptr &&
	       dispatcher.vkGetPipelineBinaryDataKHR != nullptr && dispatcher.vkReleaseCapturedPipelineDataKHR != nullptr;
}

bool CanUseComputeBinaries(const GraphicContext& graphics, const ShaderModule& module) {
	return module.spirv_words >= MinComputeBinaryWords && PipelineBinariesUsable(graphics);
}

bool CanUseGraphicsBinaries(const GraphicContext& graphics, const std::array<const ShaderModule*, 4>& stages) {
	uint64_t words = 0;
	for (const auto* stage: stages) {
		if (stage != nullptr) {
			words += stage->spirv_words;
		}
	}
	return words >= MinComputeBinaryWords && PipelineBinariesUsable(graphics);
}

std::optional<vk::Result> CreateGraphicsPipelineWithBinary(const GraphicContext&                 graphics,
                                                           const vk::GraphicsPipelineCreateInfo& info,
                                                           vk::PipelineCache                     driver_cache,
                                                           ComputePipelineCacheMode              cache_mode,
                                                           vk::Pipeline& pipeline, const char*& mode) {
	return CreatePipelineWithBinary(graphics, info, driver_cache, cache_mode, pipeline, mode);
}
#endif

DriverCreations GetDriverCreations(double slow_seconds) {
	DriverCreations result;
	const int64_t   now = SteadyNs();
	for (const auto& slot: g_driver_creations) {
		const int64_t began = slot.load(std::memory_order_acquire);
		if (began == 0) {
			continue;
		}
		const double seconds = static_cast<double>(now - began) / 1e9;
		result.running++;
		if (seconds >= slow_seconds) {
			result.slow++;
		}
		result.longest = std::max(result.longest, seconds);
	}
	return result;
}

void DescribeDriverPipelineCreation(std::string& report) {
	const auto& creation = t_driver_creation;
	if (!creation.active) {
		return;
	}
	report += "pipeline being created by the driver:";
	for (size_t i = 0; i < 4; i++) {
		if (creation.names[i] != nullptr) {
			report += fmt::format(" {} 0x{:016x}", creation.names[i], creation.hashes[i]);
		}
	}
	report += "\n";
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits stage, bool native_bindings) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto index = std::countr_zero(mask);
		const auto kind = static_cast<Kind>(index);
		const auto native = NativeDescriptorType(program.bindings, kind);
		const auto type   = !native_bindings || (kind != Kind::FlattenedSrt && kind != Kind::ShaderData)
		                        ? native
		                    : native == vk::DescriptorType::eUniformBuffer ? vk::DescriptorType::eUniformBufferDynamic
		                                                                   : vk::DescriptorType::eStorageBufferDynamic;
		descriptor_bindings.push_back({ShaderRecompiler::IR::NativeBinding(program.stage, kind), type,
		                               program.bindings.descriptor_counts[index], stage, nullptr});
	}
}

static void CreateLayouts(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                          std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
	uint64_t descriptor_count = 0;
	uint64_t storage_buffers = 0;
	uint64_t samplers = 0;
	std::array<uint64_t, 32> stage_buffers {}, stage_resources {}, stage_samplers {};
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
		const bool storage = binding.descriptorType == vk::DescriptorType::eStorageBuffer ||
		                     binding.descriptorType == vk::DescriptorType::eStorageBufferDynamic;
		const bool sampler = binding.descriptorType == vk::DescriptorType::eSampler;
		if (storage) storage_buffers += binding.descriptorCount;
		if (sampler) samplers += binding.descriptorCount;
		for (auto stages = static_cast<uint32_t>(binding.stageFlags); stages != 0; stages &= stages - 1u) {
			const auto stage = std::countr_zero(stages);
			if (storage) stage_buffers[stage] += binding.descriptorCount;
			if (sampler) stage_samplers[stage] += binding.descriptorCount;
			else
				stage_resources[stage] += binding.descriptorCount;
			if (stage_buffers[stage] > limits.maxPerStageDescriptorStorageBuffers ||
			    stage_resources[stage] > limits.maxPerStageResources ||
			    stage_samplers[stage] > limits.maxPerStageDescriptorSamplers) {
				EXIT("native descriptor stage %u exceeds device limits: buffers=%" PRIu64
				     "/%u resources=%" PRIu64 "/%u samplers=%" PRIu64 "/%u\n", stage, stage_buffers[stage],
				     limits.maxPerStageDescriptorStorageBuffers, stage_resources[stage], limits.maxPerStageResources,
				     stage_samplers[stage], limits.maxPerStageDescriptorSamplers);
			}
		}
	}
	if (storage_buffers > limits.maxDescriptorSetStorageBuffers) {
		EXIT("native storage-buffer descriptors exceed device limit: %" PRIu64 "/%u\n",
		     storage_buffers, limits.maxDescriptorSetStorageBuffers);
	}
	if (samplers > limits.maxDescriptorSetSamplers) {
		EXIT("native sampler descriptors exceed device limit: %" PRIu64 "/%u\n",
		     samplers, limits.maxDescriptorSetSamplers);
	}
	uint64_t                 sampled_images = 0;
	std::array<uint64_t, 32> stage_sampled {};
	for (const auto& binding: bindings) {
		if (binding.descriptorType != vk::DescriptorType::eSampledImage) continue;
		sampled_images += binding.descriptorCount;
		for (auto stages = static_cast<uint32_t>(binding.stageFlags); stages != 0; stages &= stages - 1u)
			stage_sampled[std::countr_zero(stages)] += binding.descriptorCount;
	}
	const auto max_stage_sampled = *std::ranges::max_element(stage_sampled);
	if (sampled_images > limits.maxDescriptorSetSampledImages ||
	    max_stage_sampled > limits.maxPerStageDescriptorSampledImages) {
		EXIT("pipeline needs %" PRIu64 " sampled images (%" PRIu64 " in one stage); the device allows "
		     "%u per set and %u per stage\n",
		     sampled_images, max_stage_sampled, limits.maxDescriptorSetSampledImages,
		     limits.maxPerStageDescriptorSampledImages);
	}
	if (pipeline.uses_image_heap) {
		vk::PhysicalDeviceVulkan12Properties heap_limits {};
		vk::PhysicalDeviceProperties2        properties {};
		properties.pNext = &heap_limits;
		graphics.physical_device.getProperties2(&properties);
		const uint64_t heap                = ShaderRecompiler::IR::ImageHeapCapacity;
		const auto     max_stage_resources = *std::ranges::max_element(stage_resources);
		if (sampled_images + heap > heap_limits.maxDescriptorSetUpdateAfterBindSampledImages ||
		    max_stage_sampled + heap > heap_limits.maxPerStageDescriptorUpdateAfterBindSampledImages ||
		    max_stage_resources + heap > heap_limits.maxPerStageUpdateAfterBindResources) {
			EXIT("pipeline needs %" PRIu64 " sampled images (%" PRIu64 " in one stage) and %" PRIu64
			     " resources in one stage next to the image heap's %" PRIu64 "; with update after bind the device "
			     "allows %u sampled images per pipeline layout, %u per stage and %u resources per stage\n",
			     sampled_images, max_stage_sampled, max_stage_resources, heap,
			     heap_limits.maxDescriptorSetUpdateAfterBindSampledImages,
			     heap_limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
			     heap_limits.maxPerStageUpdateAfterBindResources);
		}
	}
	pipeline.uses_push_descriptors =
	    !pipeline.native_bindings && descriptor_count <= graphics.max_push_descriptors;

	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = pipeline.uses_push_descriptors
	                          ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                          : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	EXIT_IF(graphics.device.createDescriptorSetLayout(
	            &create, nullptr, &pipeline.descriptor_set_layout) != vk::Result::eSuccess);
	if (!pipeline.uses_push_descriptors && !pipeline.native_bindings) {
		DescriptorHeap::RegisterLayout(pipeline.descriptor_set_layout, bindings);
	}
	const vk::PushConstantRange push_constants {pipeline.push_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	const std::array set_layouts {pipeline.descriptor_set_layout, graphics.image_heap_layout};

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = pipeline.uses_image_heap ? 2u : 1u;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_NOT_IMPLEMENTED(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &pipeline.pipeline_layout) !=
	                     vk::Result::eSuccess);
}

std::optional<uint32_t> MeshLdsWgpPadDwords(size_t usage, size_t workgroup_limit, size_t max_pad_bytes) {
	constexpr size_t KiB = 1024;
	constexpr size_t CuBytes = 64 * KiB;
	if (usage == 0 || usage > CuBytes || workgroup_limit != CuBytes) return 0;
	const auto has_reserve = [](size_t allocation, size_t cu_min, size_t wgp_min) {
		const size_t cu_groups  = std::min(CuBytes / allocation, size_t {4});
		const size_t wgp_groups = std::min(2 * CuBytes / allocation, size_t {8});
		return CuBytes - cu_groups * allocation >= cu_min &&
		       2 * CuBytes - wgp_groups * allocation >= wgp_min;
	};
	const size_t base = (usage + KiB - 1) & ~(KiB - 1);
	for (size_t pad = 0; pad <= CuBytes - base; pad += KiB) {
		if (has_reserve(base + pad, 4 * KiB, 8 * KiB)) {
			if (pad > max_pad_bytes) return std::nullopt;
			return static_cast<uint32_t>(pad / sizeof(uint32_t));
		}
	}
	if (has_reserve(base, 3 * KiB, 3 * KiB)) return 0;
	return std::nullopt;
}

std::optional<uint32_t> QueryMeshLdsWgpPadDwords(const GraphicContext& graphics, vk::Pipeline pipeline,
                                                 const ShaderModule& module, uint32_t existing_pad_dwords,
                                                 bool allow_missing_statistics) {
	VkShaderStatisticsInfoAMD stats {};
	size_t size = sizeof(stats);
	if (!graphics.fn_get_shader_info || !pipeline) return std::nullopt;
	const auto result = graphics.fn_get_shader_info(graphics.device, pipeline, VK_SHADER_STAGE_MESH_BIT_EXT,
	                                                 VK_SHADER_INFO_TYPE_STATISTICS_AMD, &size, &stats);
	if (result == VK_ERROR_FEATURE_NOT_PRESENT && allow_missing_statistics) return 0;
	if (result != VK_SUCCESS || size != sizeof(stats)) return std::nullopt;
	const auto& limits = graphics.mesh_shader_properties;
	const uint32_t payload = module.mesh_task ? ShaderRecompiler::IR::PushData::MeshTaskPayloadDwordCount * 4u : 0u;
	const uint32_t combined = limits.maxMeshPayloadAndSharedMemorySize;
	const uint32_t limit = std::min(limits.maxMeshSharedMemorySize, combined > payload ? combined - payload : 0u);
	const uint64_t shared = module.mesh_shared_bytes + uint64_t {existing_pad_dwords} * sizeof(uint32_t);
	const size_t budget = module.mesh_shared_bytes != 0 && shared < limit ? limit - shared : 0;
	const auto& usage = stats.resourceUsage;
	return MeshLdsWgpPadDwords(usage.ldsUsageSizeInBytes, usage.ldsSizePerLocalWorkGroup, budget);
}

static vk::Result PadMeshPipeline(const GraphicContext& graphics, const ShaderModule& module,
                                  vk::PipelineShaderStageCreateInfo& mesh_stage, vk::PipelineCache cache,
                                  const vk::GraphicsPipelineCreateInfo& info, vk::Pipeline& pipeline) {
	uint32_t total_pad = 0;
	for (unsigned attempt = 0;; ++attempt) {
		const auto pad = QueryMeshLdsWgpPadDwords(graphics, pipeline, module, total_pad, total_pad == 0);
		if (pad == 0) return vk::Result::eSuccess;
		MemoryTelemetry::Destroy(graphics.device, pipeline);
		pipeline = nullptr;
		if (!pad || attempt == 2) return vk::Result::eErrorInitializationFailed;
		total_pad += *pad;
		const uint32_t length = ShaderRecompiler::Spirv::MeshAllocationBaseDwords + total_pad;
		const vk::SpecializationMapEntry entry {.constantID = ShaderRecompiler::Spirv::MeshAllocationLengthSpecId,
		                                        .offset = 0, .size = sizeof(length)};
		const vk::SpecializationInfo spec {.mapEntryCount = 1, .pMapEntries = &entry,
		                                   .dataSize = sizeof(length), .pData = &length};
		mesh_stage.pSpecializationInfo = &spec;
		const auto result = MemoryTelemetry::Create(graphics.device, cache, info, pipeline);
		mesh_stage.pSpecializationInfo = nullptr;
		if (result != vk::Result::eSuccess) return result;
	}
}

#if defined(ANYPS5_KYTY)
namespace {

struct GraphicsLayoutBindings {
	std::vector<vk::DescriptorSetLayoutBinding> descriptors;
	vk::ShaderStageFlags push_stages = vk::ShaderStageFlagBits::eFragment;
	bool image_heap = false;

	GraphicsLayoutBindings(const PipelineLibraries::Stages& stages, bool native) {
		if (stages[0]->mesh_task) push_stages |= vk::ShaderStageFlagBits::eTaskEXT;
		for (const auto* stage: stages) {
			if (stage == nullptr) continue;
			const auto flag = NativeShaderStage(stage->info.stage);
			AddLayoutBindings(descriptors, stage->info, flag, native);
			push_stages |= flag;
			image_heap |= stage->info.info.uses_image_heap;
		}
	}

	void Apply(GraphicContext& graphics, PipelineCache::Pipeline& pipeline) const {
		pipeline.push_stages = push_stages;
		pipeline.uses_image_heap |= image_heap;
		CreateLayouts(graphics, pipeline, descriptors);
	}
};

class GraphicsPipelineState {
public:
	GraphicsPipelineState(GraphicContext& graphics, const GraphicsPipelineKey& key,
	                      const std::array<const ShaderModule*, 4>& stages, bool generated_shaders)
	    : m_graphics(graphics), m_key(key), m_stages(stages) {
		const auto& rendering     = key.rendering;
		const auto& vertex_input  = key.vertex_input;
		const auto& static_params = key.static_params;
		const auto& vertex        = *stages[0];
		const auto* pixel         = stages[3];
		with_depth = rendering.depth_format != vk::Format::eUndefined ||
		             rendering.stencil_format != vk::Format::eUndefined;
		mesh = vertex.info.stage == ShaderType::Mesh;
		EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
		const bool rect_list = static_params.patch_control_points != 0 && stages[1] == nullptr;

		if (vertex.mesh_task) {
			task_module = generated_shaders ? CompileSPV(GPU_MESH_INDIRECT_SPV, graphics.device)
			                                : vk::ShaderModule {};
		}
		if (rect_list && generated_shaders) {
			std::span<const RectListParameter> parameters;
			if (pixel != nullptr) parameters = pixel->rect_parameters;
			const auto shaders = BuildRectListShaders(parameters, vertex.info.param_export_mask);
			tess_control_shader_module = CompileSPV(shaders.control, graphics.device);
			tess_eval_shader_module    = CompileSPV(shaders.evaluation, graphics.device);
		}

		for (const auto* stage: stages) {
			if (stage != nullptr) {
				shader_stages[shader_stage_count++] = {.stage  = NativeShaderStage(stage->info.stage),
				                                       .module = stage->module,
				                                       .pName  = "main"};
			}
		}
		if (task_module) {
			shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eTaskEXT,
			                                       .module = task_module,
			                                       .pName  = "main"};
		}
		if (tess_control_shader_module) {
			shader_stages[shader_stage_count++] = {.stage =
			                                           vk::ShaderStageFlagBits::eTessellationControl,
			                                       .module = tess_control_shader_module,
			                                       .pName  = "main"};
			shader_stages[shader_stage_count++] = {.stage =
			                                           vk::ShaderStageFlagBits::eTessellationEvaluation,
			                                       .module = tess_eval_shader_module,
			                                       .pName  = "main"};
		}

		for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
			input_desc[binding].binding   = binding;
			input_desc[binding].stride    = vertex_input.bindings[binding].stride;
			input_desc[binding].inputRate = vertex_input.bindings[binding].instance
			                                    ? vk::VertexInputRate::eInstance
			                                    : vk::VertexInputRate::eVertex;
		}
		for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
			input_attr[index].binding  = vertex_input.attributes[index].binding;
			input_attr[index].location = index;
			input_attr[index].offset   = vertex_input.attributes[index].offset;
			input_attr[index].format   = vertex.vertex_formats[index];
		}

		vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
		vertex_input_info.pVertexBindingDescriptions      = input_desc;
		vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
		vertex_input_info.pVertexAttributeDescriptions    = input_attr;

		input_assembly.topology = static_params.topology;
		input_assembly.primitiveRestartEnable =
		    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

		depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);
		viewport_state.pNext                = &depth_clip_control;

		clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;
#if !defined(__APPLE__)
		rasterizer.depthClampEnable = VK_TRUE;
		rasterizer.pNext            = &clip_ext;
#endif
		EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
		                     !graphics.provoking_vertex_last_enabled);
		if (graphics.provoking_vertex_last_enabled) {
			provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
			    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
			provoking_vertex.pNext = rasterizer.pNext;
			rasterizer.pNext       = &provoking_vertex;
		}
		rasterizer.polygonMode = static_params.polygon_mode;

		multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
		multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
		multisampling.minSampleShading     = 1.0f;
		multisampling.alphaToCoverageEnable =
		    static_params.alpha_to_coverage_enable ? VK_TRUE : VK_FALSE;

		for (uint32_t i = 0; i < rendering.color_count; i++) {
			EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
			color_blend_attachment[i].colorWriteMask =
			    vk::ColorComponentFlags {static_params.color_mask[i]};
			color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
			color_blend_attachment[i].srcColorBlendFactor =
			    GetBlendFactor(static_params.color_srcblend[i]);
			color_blend_attachment[i].dstColorBlendFactor =
			    GetBlendFactor(static_params.color_destblend[i]);
			color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
			color_blend_attachment[i].srcAlphaBlendFactor =
			    (static_params.separate_alpha_blend[i]
			         ? GetBlendFactor(static_params.alpha_srcblend[i])
			         : color_blend_attachment[i].srcColorBlendFactor);
			color_blend_attachment[i].dstAlphaBlendFactor =
			    (static_params.separate_alpha_blend[i]
			         ? GetBlendFactor(static_params.alpha_destblend[i])
			         : color_blend_attachment[i].dstColorBlendFactor);
			color_blend_attachment[i].alphaBlendOp =
			    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
			                                           : color_blend_attachment[i].colorBlendOp);
		}

		color_blending.attachmentCount = rendering.color_count;
		color_blending.pAttachments    = color_blend_attachment;

		dynamic_states = {
		    vk::DynamicState::eViewportWithCount,
		    vk::DynamicState::eScissorWithCount,
		    vk::DynamicState::eLineWidth,
		    vk::DynamicState::eCullMode,
		    vk::DynamicState::eFrontFace,
		    vk::DynamicState::eDepthTestEnable,
		    vk::DynamicState::eDepthWriteEnable,
		    vk::DynamicState::eDepthCompareOp,
		    vk::DynamicState::eDepthBiasEnable,
		    vk::DynamicState::eDepthBias,
		    vk::DynamicState::eStencilTestEnable,
		    vk::DynamicState::eStencilOp,
		    vk::DynamicState::eStencilCompareMask,
		    vk::DynamicState::eStencilReference,
		    vk::DynamicState::eStencilWriteMask,
		    vk::DynamicState::eBlendConstants,
		};
#if !defined(__APPLE__)
		dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
		dynamic_states.push_back(vk::DynamicState::eDepthBounds);
#endif
		if (graphics.attachment_feedback_loop_enabled) {
			dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
		}
		dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic_state.pDynamicStates    = dynamic_states.data();

		rendering_info.colorAttachmentCount    = rendering.color_count;
		rendering_info.pColorAttachmentFormats = rendering.color_formats.data();
		rendering_info.depthAttachmentFormat   = rendering.depth_format;
		rendering_info.stencilAttachmentFormat = rendering.stencil_format;
		tessellation_state.patchControlPoints  = static_params.patch_control_points;

		pipeline_info.pNext               = &rendering_info;
		pipeline_info.stageCount          = shader_stage_count;
		pipeline_info.pStages             = shader_stages.data();
		pipeline_info.pVertexInputState   = mesh ? nullptr : &vertex_input_info;
		pipeline_info.pInputAssemblyState = mesh ? nullptr : &input_assembly;
		pipeline_info.pTessellationState =
		    static_params.patch_control_points != 0 ? &tessellation_state : nullptr;
		pipeline_info.pViewportState      = &viewport_state;
		pipeline_info.pRasterizationState = &rasterizer;
		pipeline_info.pMultisampleState   = &multisampling;
		pipeline_info.pDepthStencilState  = (with_depth ? &depth_stencil_info : nullptr);
		pipeline_info.pColorBlendState    = &color_blending;
		pipeline_info.pDynamicState       = &dynamic_state;
	}
	~GraphicsPipelineState() {
		m_graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
		m_graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
		m_graphics.device.destroyShaderModule(task_module, nullptr);
	}
	KYTY_CLASS_NO_COPY(GraphicsPipelineState);

	void ApplyLayout(PipelineCache::Pipeline& pipeline) {
		GraphicsLayoutBindings(m_stages, pipeline.native_bindings).Apply(m_graphics, pipeline);
	}

	vk::Result Create(vk::PipelineCache cache, vk::PipelineLayout layout, vk::PipelineCreateFlags flags,
	                  vk::Pipeline& out) {
		pipeline_info.layout = layout;
		const ScopedDriverCreation creation(
		    {Name(0, mesh ? "ms" : "vs"), Name(1, "hs"), Name(2, "ds"), Name(3, "ps")},
		    {Hash(0), Hash(1), Hash(2), Hash(3)});
		for (const auto* stage: m_stages) {
			t_driver_creation.words += stage != nullptr ? stage->spirv_words : 0;
		}
		const bool lookup     = static_cast<bool>(flags & vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired);
		const bool mesh_pad   = m_graphics.fn_get_shader_info && mesh;
		const bool use_binary = !mesh_pad && CanUseGraphicsBinaries(m_graphics, m_stages);
		if (use_binary) {
			const char* mode   = nullptr;
			pipeline_info.flags = flags;
			if (const auto binary = CreateGraphicsPipelineWithBinary(
			        m_graphics, pipeline_info, cache,
			        lookup ? ComputePipelineCacheMode::Lookup : ComputePipelineCacheMode::Full, out, mode)) {
				return *binary;
			}
		}
		const auto create = [&](vk::PipelineCreateFlags create_flags) {
			pipeline_info.flags = create_flags;
			const auto result = MemoryTelemetry::Create(m_graphics.device, cache, pipeline_info, out);
			if (result == vk::Result::eSuccess && m_graphics.fn_get_shader_info && mesh && m_stages[3]) {
				return PadMeshPipeline(m_graphics, *m_stages[0], shader_stages[0], cache, pipeline_info, out);
			}
			return result;
		};
		const auto started = std::chrono::steady_clock::now();
		const auto result  = CreateAuditedPipeline(m_graphics, "graphics", PipelineCache::GraphicsPipelineKeyHash {}(m_key),
		                                           flags, out, create);
		if (!lookup) {
			const char* binary_state = mesh_pad                              ? "mesh-lds"
			                           : !PipelineBinariesUsable(m_graphics) ? "off"
			                           : use_binary                          ? "failed"
			                                                                 : "small";
			LogDriverCompile("graphics", "-", binary_state, cache ? "direct" : "none",
			                 std::chrono::steady_clock::now() - started, result, false);
		}
		return result;
	}

	vk::Pipeline CreateLibrary(vk::PipelineLayout layout, PipelineLibraryPart part) {
		using Flag = vk::GraphicsPipelineLibraryFlagBitsEXT;
		vk::GraphicsPipelineLibraryCreateInfoEXT library {};
		vk::PipelineRenderingCreateInfo          view_only {};
		vk::GraphicsPipelineCreateInfo           create {};
		std::array<vk::PipelineShaderStageCreateInfo, 6> picked {};
		uint32_t                                         picked_count = 0;
		const auto pick = [&](bool fragment) {
			for (uint32_t i = 0; i < shader_stage_count; i++) {
				if ((shader_stages[i].stage == vk::ShaderStageFlagBits::eFragment) == fragment) {
					picked[picked_count++] = shader_stages[i];
				}
			}
			create.stageCount = picked_count;
			create.pStages    = picked.data();
		};
		create.flags         = vk::PipelineCreateFlagBits::eLibraryKHR;
		create.pDynamicState = &dynamic_state;
		switch (part) {
			case PipelineLibraryPart::VertexInput:
				library.flags              = Flag::eVertexInputInterface;
				create.pVertexInputState   = &vertex_input_info;
				create.pInputAssemblyState = &input_assembly;
				break;
			case PipelineLibraryPart::PreRasterization:
				library.flags = Flag::ePreRasterizationShaders;
				library.pNext = &view_only;
				pick(false);
				create.pViewportState      = &viewport_state;
				create.pRasterizationState = &rasterizer;
				create.pTessellationState  = pipeline_info.pTessellationState;
				create.layout              = layout;
				break;
			case PipelineLibraryPart::FragmentShader:
				library.flags = Flag::eFragmentShader;
				library.pNext = &view_only;
				pick(true);
				create.pMultisampleState  = &multisampling;
				create.pDepthStencilState = &depth_stencil_info;
				create.layout             = layout;
				break;
			case PipelineLibraryPart::FragmentOutput:
				library.flags             = Flag::eFragmentOutputInterface;
				library.pNext             = &rendering_info;
				create.pMultisampleState  = &multisampling;
				create.pColorBlendState   = &color_blending;
				break;
		}
		create.pNext       = &library;
		vk::Pipeline handle = nullptr;
		vk::Result   result = vk::Result::eSuccess;
		{
			const ScopedDriverCreation creation(
			    {Name(0, "vs-lib"), Name(1, "hs-lib"), Name(2, "ds-lib"), Name(3, "ps-lib")},
			    {Hash(0), Hash(1), Hash(2), Hash(3)});
			const auto build = [&](vk::PipelineCreateFlags flags) {
				create.flags = flags;
				return MemoryTelemetry::Create(m_graphics.device, nullptr, create, handle);
			};
			result = CreateAuditedPipeline(m_graphics, "graphics_library", PipelineCache::GraphicsPipelineKeyHash {}(m_key),
			                               create.flags, handle, build);
		}
		return result == vk::Result::eSuccess ? handle : vk::Pipeline {};
	}

	[[nodiscard]] uint64_t Hash(size_t stage) const {
		return m_stages[stage] != nullptr ? m_stages[stage]->info.shader_hash : uint64_t {0};
	}
	[[nodiscard]] const char* Name(size_t stage, const char* stage_name) const {
		return m_stages[stage] != nullptr ? stage_name : nullptr;
	}

	GraphicContext&                            m_graphics;
	const GraphicsPipelineKey&                 m_key;
	const std::array<const ShaderModule*, 4>& m_stages;
	bool                                       mesh       = false;
	bool                                       with_depth = false;
	vk::ShaderModule                           tess_control_shader_module = nullptr;
	vk::ShaderModule                           tess_eval_shader_module    = nullptr;
	vk::ShaderModule                           task_module                = nullptr;
	std::array<vk::PipelineShaderStageCreateInfo, 6> shader_stages {};
	uint32_t                                         shader_stage_count = 0;
	vk::VertexInputAttributeDescription input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription   input_desc[ShaderVertexInputInfo::RES_MAX] {};
	vk::PipelineVertexInputStateCreateInfo                     vertex_input_info {};
	vk::PipelineInputAssemblyStateCreateInfo                   input_assembly {};
	vk::PipelineViewportDepthClipControlCreateInfoEXT          depth_clip_control {};
	vk::PipelineViewportStateCreateInfo                        viewport_state {};
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT       clip_ext {};
	vk::PipelineRasterizationStateCreateInfo                   rasterizer {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	vk::PipelineMultisampleStateCreateInfo                     multisampling {};
	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	vk::PipelineColorBlendStateCreateInfo       color_blending {};
	vk::PipelineDepthStencilStateCreateInfo     depth_stencil_info {};
	std::vector<vk::DynamicState>               dynamic_states;
	vk::PipelineDynamicStateCreateInfo          dynamic_state {};
	vk::PipelineTessellationStateCreateInfo     tessellation_state {};
	vk::PipelineRenderingCreateInfo             rendering_info {};
	vk::GraphicsPipelineCreateInfo              pipeline_info {};
};

}

void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache) {
	GraphicsPipelineState state(graphics, key, stages, true);
	state.ApplyLayout(pipeline);
	const auto result = state.Create(driver_cache, pipeline.pipeline_layout, {}, pipeline.pipeline);
	if (result != vk::Result::eSuccess) {
		EXIT("vkCreateGraphicsPipelines failed: %s, %s 0x%016" PRIx64 ", hs 0x%016" PRIx64
		     ", ds 0x%016" PRIx64 ", ps 0x%016" PRIx64 "\n",
		     vk::to_string(result).c_str(), state.mesh ? "ms" : "vs", state.Hash(0), state.Hash(1),
		     state.Hash(2), state.Hash(3));
	}
}

std::string GraphicsLayoutKey(const PipelineLibraries::Stages& stages, bool native) {
	const GraphicsLayoutBindings layout(stages, native);
	std::string key;
	const auto append = [&](uint32_t word) { key.append(reinterpret_cast<const char*>(&word), sizeof(word)); };
	append(native);
	append(layout.image_heap);
	append(static_cast<uint32_t>(layout.push_stages));
	for (const auto& binding: layout.descriptors) {
		append(binding.binding);
		append(static_cast<uint32_t>(binding.descriptorType));
		append(binding.descriptorCount);
		append(static_cast<uint32_t>(binding.stageFlags));
	}
	return key;
}

void CreateGraphicsPipelineLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                  const GraphicsPipelineKey&,
                                  const std::array<const ShaderModule*, 4>& stages) {
	GraphicsLayoutBindings(stages, pipeline.native_bindings).Apply(graphics, pipeline);
}

vk::Result CreateGraphicsPipelineWhole(GraphicContext& graphics, const PipelineCache::Pipeline& layout,
                                       const GraphicsPipelineKey&                 key,
                                       const std::array<const ShaderModule*, 4>& stages,
                                       vk::PipelineCache cache, vk::PipelineCreateFlags flags,
                                       vk::Pipeline& out) {
	GraphicsPipelineState state(graphics, key, stages, true);
	return state.Create(cache, layout.pipeline_layout, flags, out);
}

vk::Pipeline CreateGraphicsPipelineLibrary(GraphicContext& graphics, vk::PipelineLayout layout,
                                           bool, const GraphicsPipelineKey& key,
                                           const std::array<const ShaderModule*, 4>& stages,
                                           PipelineLibraryPart                         part) {
	GraphicsPipelineState state(graphics, key, stages,
	                            part == PipelineLibraryPart::PreRasterization);
	return state.CreateLibrary(layout, part);
}

vk::Pipeline LinkGraphicsPipeline(GraphicContext& graphics, vk::PipelineLayout layout,
                                  const GraphicsPipelineKey& key, std::span<const vk::Pipeline> libraries) {
	vk::PipelineLibraryCreateInfoKHR linked {};
	linked.libraryCount = static_cast<uint32_t>(libraries.size());
	linked.pLibraries   = libraries.data();
	vk::GraphicsPipelineCreateInfo create {};
	create.pNext        = &linked;
	create.layout       = layout;
	vk::Pipeline handle = nullptr;
	const auto link = [&](vk::PipelineCreateFlags flags) {
		create.flags = flags;
		return MemoryTelemetry::Create(graphics.device, nullptr, create, handle);
	};
	const auto result = CreateAuditedPipeline(graphics, "graphics_link", PipelineCache::GraphicsPipelineKeyHash {}(key),
	                                         {}, handle, link);
	if (result != vk::Result::eSuccess) {
		return {};
	}
	return handle;
}
#else
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache) {
	const auto& rendering     = key.rendering;
	const auto& vertex_input  = key.vertex_input;
	const auto& static_params = key.static_params;
	const auto& vertex        = *stages[0];
	const auto* pixel         = stages[3];
	const bool  with_depth    = rendering.depth_format != vk::Format::eUndefined ||
	                            rendering.stencil_format != vk::Format::eUndefined;
	const bool  mesh          = vertex.info.stage == ShaderType::Mesh;
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list = static_params.patch_control_points != 0 && stages[1] == nullptr;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;
	const auto       task_module =
	    vertex.mesh_task ? CompileSPV(GPU_MESH_INDIRECT_SPV, graphics.device) : vk::ShaderModule {};

	if (rect_list) {
		std::span<const RectListParameter> parameters;
		if (pixel != nullptr) parameters = pixel->rect_parameters;
		const auto shaders = BuildRectListShaders(parameters, vertex.info.param_export_mask);
		tess_control_shader_module = CompileSPV(shaders.control, graphics.device);
		tess_eval_shader_module    = CompileSPV(shaders.evaluation, graphics.device);
	}

	vk::PipelineShaderStageCreateInfo shader_stages[4] {};
	uint32_t                          shader_stage_count = 0;
	for (const auto* stage: stages) {
		if (stage != nullptr) {
			shader_stages[shader_stage_count++] = {.stage  = NativeShaderStage(stage->info.stage),
			                                       .module = stage->module,
			                                       .pName  = "main"};
		}
	}
	if (task_module) {
		shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eTaskEXT,
		                                       .module = task_module,
		                                       .pName  = "main"};
	}
	if (rect_list) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationControl,
		                                       .module = tess_control_shader_module,
		                                       .pName  = "main"};
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                       .module = tess_eval_shader_module,
		                                       .pName  = "main"};
	}

	vk::VertexInputAttributeDescription input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription   input_desc[ShaderVertexInputInfo::RES_MAX] {};

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		input_attr[index].binding  = vertex_input.attributes[index].binding;
		input_attr[index].location = index;
		input_attr[index].offset   = vertex_input.attributes[index].offset;
		input_attr[index].format   = vertex.vertex_formats[index];
	}

	vk::PipelineVertexInputStateCreateInfo vertex_input_info {};
	vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	vertex_input_info.pVertexBindingDescriptions      = input_desc;
	vertex_input_info.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	vertex_input_info.pVertexAttributeDescriptions    = input_attr;

	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = static_params.topology;
	input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);

	vk::PipelineViewportStateCreateInfo viewport_state {};
	viewport_state.pNext = &depth_clip_control;

	vk::PipelineRasterizationDepthClipStateCreateInfoEXT clip_ext {};
	clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineRasterizationStateCreateInfo rasterizer {};
#if !defined(__APPLE__)
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext = &clip_ext;
#endif
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &provoking_vertex;
	}
	rasterizer.polygonMode = static_params.polygon_mode;

	vk::PipelineMultisampleStateCreateInfo multisampling {};
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;
	multisampling.alphaToCoverageEnable =
	    static_params.alpha_to_coverage_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i])
		         : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	vk::PipelineColorBlendStateCreateInfo color_blending {};
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment;

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	pipeline.push_stages = vk::ShaderStageFlagBits::eFragment;
	if (task_module) {
		pipeline.push_stages |= vk::ShaderStageFlagBits::eTaskEXT;
	}
	for (const auto* stage: stages) {
		if (stage != nullptr) {
			const auto native_stage = NativeShaderStage(stage->info.stage);
			AddLayoutBindings(descriptor_bindings, stage->info, native_stage, pipeline.native_bindings);
			pipeline.uses_image_heap |= stage->info.info.uses_image_heap;
			pipeline.push_stages |= native_stage;
		}
	}
	CreateLayouts(graphics, pipeline, descriptor_bindings);

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_info {};

	std::vector<vk::DynamicState> dynamic_states {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eCullMode,
	    vk::DynamicState::eFrontFace,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilTestEnable,
	    vk::DynamicState::eStencilOp,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
#if !defined(__APPLE__)
	dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
	dynamic_states.push_back(vk::DynamicState::eDepthBounds);
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}

	vk::PipelineDynamicStateCreateInfo dynamic_state {};
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates    = dynamic_states.data();

	vk::GraphicsPipelineCreateInfo  pipeline_info {};
	vk::PipelineRenderingCreateInfo rendering_info {};
	rendering_info.colorAttachmentCount    = rendering.color_count;
	rendering_info.pColorAttachmentFormats = rendering.color_formats.data();
	rendering_info.depthAttachmentFormat   = rendering.depth_format;
	rendering_info.stencilAttachmentFormat = rendering.stencil_format;
	pipeline_info.pNext                    = &rendering_info;
	pipeline_info.stageCount               = shader_stage_count;
	pipeline_info.pStages                  = shader_stages;
	pipeline_info.pVertexInputState        = mesh ? nullptr : &vertex_input_info;
	pipeline_info.pInputAssemblyState      = mesh ? nullptr : &input_assembly;
	vk::PipelineTessellationStateCreateInfo tessellation_state {};
	tessellation_state.patchControlPoints = static_params.patch_control_points;
	pipeline_info.pTessellationState =
	    static_params.patch_control_points != 0 ? &tessellation_state : nullptr;
	pipeline_info.pViewportState      = &viewport_state;
	pipeline_info.pRasterizationState = &rasterizer;
	pipeline_info.pMultisampleState   = &multisampling;
	pipeline_info.pDepthStencilState  = (with_depth ? &depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState    = &color_blending;
	pipeline_info.pDynamicState       = &dynamic_state;
	pipeline_info.layout              = pipeline.pipeline_layout;

	const auto hash = [&](size_t stage) {
		return stages[stage] != nullptr ? stages[stage]->info.shader_hash : uint64_t {0};
	};
	const auto name = [&](size_t stage, const char* stage_name) {
		return stages[stage] != nullptr ? stage_name : nullptr;
	};
	vk::Result result = vk::Result::eSuccess;
	{
		const ScopedDriverCreation creation(
		    {name(0, mesh ? "ms" : "vs"), name(1, "hs"), name(2, "ds"), name(3, "ps")},
		    {hash(0), hash(1), hash(2), hash(3)});
		result = MemoryTelemetry::Create(graphics.device, driver_cache, pipeline_info, pipeline.pipeline);
	}
	if (result == vk::Result::eSuccess && graphics.fn_get_shader_info && mesh && pixel) {
		result = PadMeshPipeline(graphics, vertex, shader_stages[0], driver_cache, pipeline_info, pipeline.pipeline);
	}
	if (result != vk::Result::eSuccess) {
		EXIT("vkCreateGraphicsPipelines failed: %s, %s 0x%016" PRIx64 ", hs 0x%016" PRIx64
		     ", ds 0x%016" PRIx64 ", ps 0x%016" PRIx64 "\n",
		     vk::to_string(result).c_str(), mesh ? "ms" : "vs", hash(0), hash(1), hash(2), hash(3));
	}

	graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
	graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
	graphics.device.destroyShaderModule(task_module, nullptr);
}

#endif

vk::Result CreateComputePipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                 const ShaderModule& module, vk::PipelineCache driver_cache,
                                 ComputePipelineCacheMode cache_mode, std::chrono::nanoseconds* driver_time) {
	const bool lookup = cache_mode == ComputePipelineCacheMode::Lookup;
	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = module.module;
	comp_shader_stage_info.pName  = "main";
	const auto wave_size = module.info.wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	pipeline.native_bindings = module.info.table_mode;
	AddLayoutBindings(descriptor_bindings, module.info, vk::ShaderStageFlagBits::eCompute, module.info.table_mode);
	pipeline.uses_image_heap = module.info.info.uses_image_heap;
	pipeline.push_stages     = vk::ShaderStageFlagBits::eCompute;
	CreateLayouts(graphics, pipeline, descriptor_bindings);

	vk::ComputePipelineCreateInfo info {};
	if (module.info.info.uses_gds) {
		info.flags = vk::PipelineCreateFlagBits::eDispatchBase;
	}
	if (lookup) {
		info.flags |= vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
	}
	info.stage  = comp_shader_stage_info;
	info.layout = pipeline.pipeline_layout;

	using Clock = std::chrono::steady_clock;
	bool timing = driver_time != nullptr;
#if defined(ANYPS5_KYTY)
	static const auto log_ms = []() -> std::optional<int64_t> {
		const char* value = std::getenv("KYTY_PIPELINE_STALL_LOG_MS");
		if (value == nullptr) {
			return std::nullopt;
		}
		return std::strtoll(value, nullptr, 10);
	}();
	timing = timing || log_ms.has_value();
#endif
	Clock::time_point finished {};
	Clock::duration elapsed {};
	vk::Result result;
#if defined(ANYPS5_KYTY)
	const char* mode = lookup ? "lookup" : "create";
#endif
	{
		const ScopedDriverCreation creation({"cs", nullptr, nullptr, nullptr},
		                                    {module.info.shader_hash, 0, 0, 0});
		const auto started = timing ? Clock::now() : Clock::time_point {};
#if defined(ANYPS5_KYTY)
		t_driver_creation.words = module.spirv_words;
		const bool driver_lookup = lookup && graphics.pipeline_cache_lookup_enabled && driver_cache;
		std::optional<vk::Result> binary;
		if (CanUseComputeBinaries(graphics, module)) {
			binary = CreatePipelineWithBinary(graphics, info, driver_cache, cache_mode, pipeline.pipeline, mode);
		}
		if (binary) {
			result = *binary;
		} else if (lookup && !driver_lookup) {
			result = vk::Result::ePipelineCompileRequired;
		} else {
			const auto create = [&](vk::PipelineCreateFlags flags) {
				info.flags = flags;
				return MemoryTelemetry::Create(graphics.device, driver_cache, info, pipeline.pipeline);
			};
			const auto created = Clock::now();
			result = CreateAuditedPipeline(graphics, "compute", module.id, info.flags, pipeline.pipeline, create);
			if (!lookup) {
				const char* binary_state = !PipelineBinariesUsable(graphics)                  ? "off"
				                           : module.spirv_words < MinComputeBinaryWords ? "small"
				                                                                          : "failed";
				const char* cache = cache_mode == ComputePipelineCacheMode::KnownMiss ? "miss"
				                    : driver_cache                                     ? "direct"
				                                                                       : "none";
				LogDriverCompile("compute", "-", binary_state, cache, Clock::now() - created, result, false);
			}
		}
#else
		result = MemoryTelemetry::Create(graphics.device, driver_cache, info, pipeline.pipeline);
#endif
		if (timing) {
			finished = Clock::now();
			elapsed  = finished - started;
		}
	}
	if (driver_time != nullptr) {
		*driver_time = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
	}
#if defined(ANYPS5_KYTY)
	if (log_ms && std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= *log_ms) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-gpl] compute-driver {:.6f} s 0x{:016x} module={} spirv_words={} mode={} result={} @{}\n",
		    std::chrono::duration<double>(elapsed).count(), module.info.shader_hash, module.id,
		    module.spirv_words, mode, vk::to_string(result),
		    finished.time_since_epoch().count()));
	}
#endif
	return result;
}

void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache,
                            ComputePipelineCacheMode cache_mode) {
	const auto result = CreateComputePipeline(graphics, pipeline, module, driver_cache, cache_mode);
	if (result != vk::Result::eSuccess) {
		EXIT("vkCreateComputePipelines failed: %s, cs 0x%016" PRIx64 "\n", vk::to_string(result).c_str(),
		     module.info.shader_hash);
	}
}

bool FindComputePipeline(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                         const ShaderModule& module, vk::PipelineCache cache) {
	const bool driver_lookup = graphics.pipeline_cache_lookup_enabled && cache != nullptr;
#if defined(ANYPS5_KYTY)
	if (!driver_lookup && !CanUseComputeBinaries(graphics, module)) {
		return false;
	}
#else
	if (!driver_lookup) {
		return false;
	}
#endif
	if (CreateComputePipeline(graphics, pipeline, module, cache, ComputePipelineCacheMode::Lookup) ==
	        vk::Result::eSuccess &&
	    pipeline.pipeline != nullptr) {
		return true;
	}
	MemoryTelemetry::Destroy(graphics.device, pipeline.pipeline);
	graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
	pipeline = {};
	return false;
}

}

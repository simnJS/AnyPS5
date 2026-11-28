#pragma once

#include "common/file.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace Libs::Graphics {

struct ShaderModule;

class StoreFile {
public:
	[[nodiscard]] static std::shared_ptr<const StoreFile> Open(const std::filesystem::path& path);
	StoreFile() = default;
	StoreFile(const StoreFile&)            = delete;
	StoreFile& operator=(const StoreFile&) = delete;

	[[nodiscard]] bool Read(uint64_t offset, std::span<uint8_t> bytes) const;

private:
	mutable std::mutex   m_mutex;
	mutable Common::File m_file;
};

class StoredSpirv {
public:
	using Decode = bool (*)(std::span<const uint8_t> frame, uint64_t checksum, bool plain, std::vector<uint32_t>& spirv);

	StoredSpirv(std::shared_ptr<const StoreFile> file, uint64_t offset, uint32_t size, uint64_t checksum, bool plain,
	            Decode decode)
	    : m_file(std::move(file)), m_offset(offset), m_size(size), m_checksum(checksum), m_plain(plain), m_decode(decode) {}
	StoredSpirv(const StoredSpirv&)            = delete;
	StoredSpirv& operator=(const StoredSpirv&) = delete;

	vk::ShaderModule Acquire(vk::Device device);
	void             Release(vk::Device device);
	[[nodiscard]] uint32_t Users() const;

private:
	mutable std::mutex               m_mutex;
	std::shared_ptr<const StoreFile> m_file;
	uint64_t                         m_offset   = 0;
	uint32_t                         m_size     = 0;
	uint64_t                         m_checksum = 0;
	bool                             m_plain    = false;
	Decode                           m_decode   = nullptr;
	vk::ShaderModule                 m_module   = nullptr;
	uint32_t                         m_users    = 0;
};

[[nodiscard]] bool HasShaderModule(const ShaderModule& module);
vk::ShaderModule   AcquireShaderModule(vk::Device device, const ShaderModule& module);
void               ReleaseShaderModule(vk::Device device, const ShaderModule& module);

class ShaderModuleUse {
public:
	ShaderModuleUse(vk::Device device, const ShaderModule& module)
	    : m_device(device), m_module(&module), m_handle(AcquireShaderModule(device, module)) {}
	ShaderModuleUse(const ShaderModuleUse&)            = delete;
	ShaderModuleUse& operator=(const ShaderModuleUse&) = delete;
	~ShaderModuleUse() {
		if (m_handle) ReleaseShaderModule(m_device, *m_module);
	}
	[[nodiscard]] vk::ShaderModule Get() const { return m_handle; }

private:
	vk::Device          m_device;
	const ShaderModule* m_module;
	vk::ShaderModule    m_handle;
};

}

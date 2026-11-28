#include "graphics/host_gpu/renderer/pipeline/storedSpirv.h"

#include "graphics/host_gpu/renderer/pipeline/pipelineStoreTypes.h"

#include "common/logging/log.h"

#include <fmt/format.h>

namespace Libs::Graphics {

std::shared_ptr<const StoreFile> StoreFile::Open(const std::filesystem::path& path) {
	auto file = std::make_shared<StoreFile>();
	if (!file->m_file.Open(path, Common::File::Mode::Read) || file->m_file.IsInvalid()) return nullptr;
	return file;
}

bool StoreFile::Read(uint64_t offset, std::span<uint8_t> bytes) const {
	std::scoped_lock lock(m_mutex);
	uint32_t read = 0;
	if (bytes.size() > UINT32_MAX || !m_file.Seek(offset)) return false;
	m_file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
	return read == bytes.size();
}

vk::ShaderModule StoredSpirv::Acquire(vk::Device device) {
	std::scoped_lock lock(m_mutex);
	if (m_users != 0) {
		m_users++;
		return m_module;
	}
	std::vector<uint8_t>  frame(m_size);
	std::vector<uint32_t> spirv;
	if (m_file == nullptr || m_decode == nullptr || !m_file->Read(m_offset, frame) ||
	    !m_decode(frame, m_checksum, m_plain, spirv)) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader cache: the stored module at offset {} could not be read again from the shader store\n", m_offset));
		return nullptr;
	}
	m_module = CompileSPV(spirv, device);
	if (m_module) m_users = 1;
	return m_module;
}

void StoredSpirv::Release(vk::Device device) {
	std::scoped_lock lock(m_mutex);
	if (m_users == 0 || --m_users != 0) return;
	device.destroyShaderModule(m_module, nullptr);
	m_module = nullptr;
}

uint32_t StoredSpirv::Users() const {
	std::scoped_lock lock(m_mutex);
	return m_users;
}

bool HasShaderModule(const ShaderModule& module) {
	return module.module || module.stored != nullptr;
}

vk::ShaderModule AcquireShaderModule(vk::Device device, const ShaderModule& module) {
	if (module.module || module.stored == nullptr) return module.module;
	return module.stored->Acquire(device);
}

void ReleaseShaderModule(vk::Device device, const ShaderModule& module) {
	if (!module.module && module.stored != nullptr) module.stored->Release(device);
}

}

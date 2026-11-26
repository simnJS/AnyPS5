#include "TestVulkanDevice.h"
#include "TextureCacheTestAccess.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using namespace Libs::Graphics;

constexpr int      CpuReadWrite = 0x03;
constexpr uint32_t Size         = 2048;
constexpr uint32_t Levels       = 12;
constexpr uint64_t Slot         = 0x400000;
constexpr uint64_t GuestBytes   = 4 * Slot;

ImageInfo Texture(uint64_t address) {
	ImageInfo info;
	info.data.address     = address;
	info.pixel_format     = vk::Format::eBc1RgbaUnormBlock;
	info.guest_format     = Prospero::BufferFormat::kBc1UNorm;
	info.extent           = vk::Extent3D {Size, Size, 1};
	info.resources.levels = Levels;
	info.UpdateSize();
	return info;
}

void FillGuest(uint64_t address, uint64_t size) {
	auto* bytes = reinterpret_cast<uint8_t*>(address);
	for (uint64_t i = 0; i < size; i++) {
		bytes[i] = static_cast<uint8_t>((i * 2654435761u) >> 13u) | 1u;
	}
}

struct Readback {
	const char*          failure = nullptr;
	std::vector<uint8_t> bytes;
};

Readback Download(GraphicContext& graphics, CommandScheduler& scheduler, Image& image) {
	Readback   result;
	const auto size   = image.info.LinearSize();
	const auto copies = image.info.BufferCopies(0);
	Buffer     readback(graphics, scheduler, MemoryUsage::Download, 0,
	                    vk::BufferUsageFlagBits::eTransferDst, size);
	image.Download(copies, readback.Handle(), 0, size);
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
	readback.Invalidate(0, size);
	result.bytes.resize(size);
	std::memcpy(result.bytes.data(), readback.Mapped().data(), size);
	return result;
}

TextureCache::ImageDesc Sampled(const ImageInfo& info, uint32_t base_level, uint32_t level_count) {
	TextureCache::ImageDesc desc {.info = info};
	desc.view_info.format      = info.pixel_format;
	desc.view_info.base_level  = base_level;
	desc.view_info.level_count = level_count;
	return desc;
}

const char* CheckOmittedLevelView(GraphicContext& graphics, uint64_t address) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const auto partial_info = Texture(address);
	const auto control_info = Texture(address + Slot);
	FillGuest(address, partial_info.data.size);
	std::memcpy(reinterpret_cast<void*>(address + Slot), reinterpret_cast<const void*>(address),
	            partial_info.data.size);

	graphics.partial_textures_enabled = true;
	const auto partial_id = TextureCacheTestAccess::Create(textures, partial_info);
	graphics.partial_textures_enabled = false;
	const auto control_id = TextureCacheTestAccess::Create(textures, control_info);
	graphics.partial_textures_enabled = true;
	auto& partial = textures.GetImage(partial_id);
	auto& control = textures.GetImage(control_id);
	if (!partial.IsPartiallyResident()) return "the texture was not created partially resident";
	if (control.IsPartiallyResident()) return "the control texture was created partially resident";

	if (textures.FindTexture(partial_id, Sampled(partial_info, 0, Levels)) == nullptr) {
		return "no view of every level";
	}
	if (!partial.IsPartiallyResident()) return "a view of every level backed the omitted levels";
	if (textures.FindTexture(partial_id, Sampled(partial_info, 0, 1)) == nullptr) {
		return "no view of the omitted top level";
	}
	if (partial.IsPartiallyResident()) return "a view of the omitted top level left it unbacked";
	(void)textures.FindTexture(control_id, Sampled(control_info, 0, 1));

	const auto partial_bytes = Download(graphics, scheduler, partial);
	const auto control_bytes = Download(graphics, scheduler, control);
	if (partial_bytes.bytes.empty() || partial_bytes.bytes != control_bytes.bytes) {
		return "the backed top level differs from the fully backed control";
	}
	return nullptr;
}

const char* CheckCopyFromOmittedLevels(GraphicContext& graphics, uint64_t address) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const auto source_info  = Texture(address);
	const auto control_info = Texture(address + Slot);
	FillGuest(address, source_info.data.size);
	std::memcpy(reinterpret_cast<void*>(address + Slot), reinterpret_cast<const void*>(address),
	            source_info.data.size);

	graphics.partial_textures_enabled = true;
	const auto source_id = TextureCacheTestAccess::Create(textures, source_info);
	graphics.partial_textures_enabled = false;
	const auto control_id      = TextureCacheTestAccess::Create(textures, control_info);
	const auto copy_id         = TextureCacheTestAccess::Create(textures, Texture(address + 2 * Slot));
	const auto control_copy_id = TextureCacheTestAccess::Create(textures, Texture(address + 3 * Slot));
	graphics.partial_textures_enabled = true;
	if (!textures.GetImage(source_id).IsPartiallyResident()) {
		return "the copy source was not created partially resident";
	}
	(void)textures.FindTexture(source_id, Sampled(source_info, 0, Levels));
	(void)textures.FindTexture(control_id, Sampled(control_info, 0, Levels));

	TextureCacheTestAccess::Copy(textures, copy_id, source_id);
	TextureCacheTestAccess::Copy(textures, control_copy_id, control_id);
	if (textures.GetImage(source_id).IsPartiallyResident()) {
		return "a copy read the omitted levels of its source";
	}
	const auto copied  = Download(graphics, scheduler, textures.GetImage(copy_id));
	const auto control = Download(graphics, scheduler, textures.GetImage(control_copy_id));
	if (copied.bytes.empty() || copied.bytes != control.bytes) {
		return "the copy of a partially resident texture differs from the fully backed control";
	}
	return nullptr;
}

}

extern "C" int RunKytyPartialTexturesTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	auto& graphics = gpu.graphics;
	if (const auto* unavailable = gpu.Initialize(false, true)) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_partial_textures: skipped, no Vulkan device with sparse residency (%s)%s\n",
		             unavailable, required ? "; ANYPS5_REQUIRE_VULKAN does not cover optional sparse support" : "");
		return 77;
	}
	if (!graphics.CreateAllocator()) {
		std::fputs("kyty_shim_partial_textures: FAILED: could not create the allocator\n", stderr);
		return 1;
	}
	void* guest = nullptr;
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fputs("kyty_shim_partial_textures: FAILED: could not map guest backing\n", stderr);
		return 1;
	}
	const auto address = reinterpret_cast<uint64_t>(guest);
	if (Texture(address).data.size > Slot) {
		std::fputs("kyty_shim_partial_textures: FAILED: the texture does not fit its slot\n", stderr);
		return 1;
	}
	auto failure = CheckOmittedLevelView(graphics, address);
	if (failure == nullptr) {
		failure = CheckCopyFromOmittedLevels(graphics, address);
	}
	if (sceKernelMunmap(address, GuestBytes) != 0) {
		std::fputs("kyty_shim_partial_textures: FAILED: could not unmap guest backing\n", stderr);
		return 1;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "kyty_shim_partial_textures: FAILED: %s\n", failure);
		return 1;
	}
	std::puts("kyty_shim_partial_textures: ok");
	return 0;
}

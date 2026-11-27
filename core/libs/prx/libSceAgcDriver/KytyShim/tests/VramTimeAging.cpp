#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

struct VramTimeAgingTestAccess {
	static ImageId CreateImage(TextureCache& cache, const ImageInfo& info) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.InsertImage(info);
	}

	static bool IsImageRegistered(TextureCache& cache, ImageId id) {
		Common::ScopedLock lock {cache.m_lock};
		const auto*        image = cache.m_slot_images.try_get(id);
		return image != nullptr && image->registered;
	}

	static void RewindTextureClock(TextureCache& cache, std::chrono::milliseconds delta) {
		Common::ScopedLock lock {cache.m_lock};
		cache.m_trigger_gc_memory  = 0;
		cache.m_pressure_gc_memory = UINT64_MAX;
		cache.m_critical_gc_memory = UINT64_MAX;
		cache.m_last_gc_time       = std::chrono::steady_clock::now() - delta;
	}

	static uint64_t TextureTick(TextureCache& cache) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.m_gc_tick;
	}

	static void RewindBufferClock(BufferCache& cache, std::chrono::milliseconds delta,
	                              bool trigger_gc) {
		cache.m_trigger_gc_memory  = trigger_gc ? 0 : UINT64_MAX;
		cache.m_critical_gc_memory = UINT64_MAX;
		cache.m_last_gc_time       = std::chrono::steady_clock::now() - delta;
	}

	static uint64_t BufferTick(const BufferCache& cache) { return cache.m_gc_tick; }
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t Page         = TRACKER_PAGE_SIZE;
constexpr uint64_t GuestBytes   = 0x40000;
constexpr int      CpuReadWrite = 0x03;

const char* CheckTextureWallTimeAging(TextureCache& textures, uint64_t address) {
	ImageId id {};
	if (address != 0) {
		ImageInfo info;
		info.data.address = address;
		info.pixel_format = vk::Format::eR8G8B8A8Unorm;
		info.guest_format = Prospero::BufferFormat::k8_8_8_8UNorm;
		info.extent       = vk::Extent3D {64, 64, 1};
		info.pitch        = 64;
		info.UpdateSize();

		id = VramTimeAgingTestAccess::CreateImage(textures, info);
		if (!VramTimeAgingTestAccess::IsImageRegistered(textures, id)) {
			return "test image was not registered";
		}
	}
	VramTimeAgingTestAccess::RewindTextureClock(textures, std::chrono::milliseconds(500));
	textures.RunGarbageCollector();
	if (VramTimeAgingTestAccess::TextureTick(textures) < 30) {
		return "texture cache GC tick did not advance with elapsed wall time";
	}
	if (address != 0 && VramTimeAgingTestAccess::IsImageRegistered(textures, id)) {
		return "texture cache did not evict an unused image after 500 ms wall time";
	}
	return nullptr;
}

const char* CheckBufferWallTimeAging(BufferCache& buffers, uint64_t address) {
	const auto buf_addr = address != 0 ? address + 0x10000 : 0;
	if (buf_addr != 0) {
		(void)buffers.FindBuffer(buf_addr, Page);
		if (!buffers.RegisteredBuffer(buf_addr, Page)) {
			return "test buffer was not registered";
		}
	}
	VramTimeAgingTestAccess::RewindBufferClock(buffers, std::chrono::milliseconds(3000), true);
	buffers.RunGarbageCollector();
	if (VramTimeAgingTestAccess::BufferTick(buffers) < 180) {
		return "buffer cache GC tick did not advance with elapsed wall time";
	}
	if (buf_addr != 0 && buffers.RegisteredBuffer(buf_addr, Page)) {
		return "buffer cache did not evict an unused buffer after 3000 ms wall time";
	}

	const auto stale_addr = address != 0 ? address + 0x20000 : 0;
	if (stale_addr != 0) {
		(void)buffers.FindBuffer(stale_addr, Page);
		if (!buffers.RegisteredBuffer(stale_addr, Page)) {
			return "stale test buffer was not registered";
		}
	}
	const auto before_stale_tick = VramTimeAgingTestAccess::BufferTick(buffers);
	VramTimeAgingTestAccess::RewindBufferClock(buffers, std::chrono::milliseconds(35005), false);
	buffers.RunGarbageCollector();
	if (VramTimeAgingTestAccess::BufferTick(buffers) < before_stale_tick + 2100) {
		return "buffer cache GC tick did not advance across stale wall-time jump";
	}
	if (stale_addr != 0 && buffers.RegisteredBuffer(stale_addr, Page)) {
		return "buffer cache stale collection skipped an unaligned wall-time tick jump";
	}
	return nullptr;
}

const char* CheckTimeAging(GraphicContext& graphics, uint64_t address) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	auto& buffers   = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const char* failure = CheckTextureWallTimeAging(textures, address);
	if (failure == nullptr) {
		failure = CheckBufferWallTimeAging(buffers, address);
	}
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	if (address != 0) {
		(void)buffers.DropFreedRange(address, GuestBytes, nullptr, nullptr);
	}
	return failure;
}

}

extern "C" int RunKytyVramTimeAgingTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize()) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_vram_time_aging: %s, no usable Vulkan device (%s)\n",
		             required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fputs("kyty_shim_vram_time_aging: FAILED: could not create the allocator\n", stderr);
		return 1;
	}
	void* guest = nullptr;
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fputs("kyty_shim_vram_time_aging: FAILED: could not map guest backing\n", stderr);
		return 1;
	}
	const auto address     = reinterpret_cast<uint64_t>(guest);
	const bool valid_guest = GuestRange {address, GuestBytes}.Valid();
	if (!valid_guest && std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) {
		(void)sceKernelMunmap(address, GuestBytes);
		std::fprintf(stderr, "kyty_shim_vram_time_aging: FAILED: guest range 0x%llx is outside Kyty bounds\n",
		             static_cast<unsigned long long>(address));
		return 1;
	}
	const char* failure = CheckTimeAging(gpu.graphics, valid_guest ? address : 0);
	if (sceKernelMunmap(address, GuestBytes) != 0) {
		std::fputs("kyty_shim_vram_time_aging: FAILED: could not unmap guest backing\n", stderr);
		return 1;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "kyty_shim_vram_time_aging: FAILED: %s\n", failure);
		return 1;
	}
	std::puts("kyty_shim_vram_time_aging: ok");
	return 0;
}

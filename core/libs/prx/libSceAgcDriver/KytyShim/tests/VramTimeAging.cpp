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

	using Clock = std::chrono::steady_clock;

	static void SetMemoryUsage(GraphicContext& graphics, uint64_t usage) {
		graphics.m_memory_override_enabled = true;
		graphics.m_memory_budget_override = 6ull << 30;
		graphics.m_memory_usage_override = usage;
	}

	static void Collect(TextureCache& textures, BufferCache& buffers, Clock::time_point now) {
		textures.CollectAt(now);
		buffers.CollectAt(now);
	}

	static void Touch(TextureCache& textures, ImageId id) {
		Common::ScopedLock lock {textures.m_lock};
		(void)textures.GetImage(id);
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t Page         = TRACKER_PAGE_SIZE;
constexpr uint64_t GuestBytes   = 0x50000;
constexpr int      CpuReadWrite = 0x03;

const char* CheckEviction(GraphicContext& graphics, TextureCache& textures, BufferCache& buffers, uint64_t address) {
	using namespace std::chrono_literals;
	using Access = VramTimeAgingTestAccess;
	ImageInfo info;
	info.data.address = address;
	info.pixel_format = vk::Format::eR8G8B8A8Unorm;
	info.guest_format = Prospero::BufferFormat::k8_8_8_8UNorm;
	info.extent       = vk::Extent3D {64, 64, 1};
	info.pitch        = 64;
	info.UpdateSize();
	const auto cold_image = Access::CreateImage(textures, info);
	info.data.address += 0x4000;
	const auto warm_image = Access::CreateImage(textures, info);
	const auto cold = address + 0x10000;
	const auto warm = address + 0x20000;
	const auto pinned = address + 0x30000;
	const auto dirty = address + 0x40000;
	(void)buffers.FindBuffer(cold, Page);
	const auto warm_id = buffers.FindBuffer(warm, Page);
	buffers.CacheFaultedMemory(pinned, Page);
	(void)buffers.ObtainBuffer(dirty, Page, true);
	if (!Access::IsImageRegistered(textures, cold_image) || !buffers.RegisteredBuffer(cold, Page) ||
	    !Access::IsImageRegistered(textures, warm_image) || !warm_id) {
		return "test resources were not registered";
	}
	auto now = Access::Clock::time_point {};
	for (unsigned pass = 0; pass < 12; ++pass) {
		if (pass % 3 == 0) {
			Access::Touch(textures, warm_image);
			if (buffers.FindBuffer(warm, Page) != warm_id) {
				return "buffer lookup replaced a reused buffer";
			}
		}
		now += 2857ms;
		Access::Collect(textures, buffers, now);
		if (!Access::IsImageRegistered(textures, warm_image) || buffers.RegisteredBuffer(warm, Page) != warm_id) {
			return "GC evicted a resource used by alternating slow frames";
		}
	}
	if (Access::IsImageRegistered(textures, cold_image) || buffers.RegisteredBuffer(cold, Page)) {
		return "slow frames did not retire the unused image and buffer under pressure";
	}
	if (!buffers.RegisteredBuffer(pinned, Page)) {
		return "GC evicted a device-address buffer before its use scan";
	}

	Access::SetMemoryUsage(graphics, 0);
	(void)buffers.FindBuffer(cold, Page);
	for (unsigned pass = 0; pass < 4; ++pass) {
		now += 35005ms;
		(void)buffers.FindBuffer(warm, Page);
		Access::Collect(textures, buffers, now);
	}
	if (buffers.RegisteredBuffer(cold, Page) || buffers.RegisteredBuffer(warm, Page) != warm_id ||
	    !buffers.RegisteredBuffer(pinned, Page)) {
		return "stale collection failed to retire only the unused unpinned buffer";
	}
	if (!buffers.RegisteredBuffer(dirty, Page) || !buffers.IsRegionGpuModified(dirty, Page)) {
		return "GC lost a GPU-written buffer below critical pressure";
	}
	return nullptr;
}

const char* CheckTimeAging(GraphicContext& graphics, uint64_t address) {
	VramTimeAgingTestAccess::SetMemoryUsage(graphics, 3ull << 30);
	RenderContext renderer(graphics);
	renderer.InitializeGpu(nullptr);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	auto& buffers   = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const char* failure = CheckEviction(graphics, textures, buffers, address);
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	(void)buffers.DropFreedRange(address, GuestBytes, nullptr, nullptr);
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
	if (!valid_guest) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		(void)sceKernelMunmap(address, GuestBytes);
		std::fprintf(stderr, "kyty_shim_vram_time_aging: %s: guest range 0x%llx is outside Kyty bounds\n",
		             required ? "FAILED" : "skipped", static_cast<unsigned long long>(address));
		return required ? 1 : 77;
	}
	const char* failure = CheckTimeAging(gpu.graphics, address);
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

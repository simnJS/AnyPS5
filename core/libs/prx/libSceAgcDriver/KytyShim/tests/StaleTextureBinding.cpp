#include "RenderExecutorTestAccess.h"
#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreTypes.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shader.h"
#include "prx/libkernel/include/Api.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

using namespace Libs::Graphics;

constexpr uint64_t GuestBytes      = 0x40000;
constexpr uint64_t CaseBytes       = GuestBytes / 2;
constexpr int      CpuReadWrite    = 0x03;
constexpr uint32_t Width           = 64;
constexpr uint32_t Height          = 64;
constexpr uint32_t IdentitySwizzle = 4u | 5u << 3u | 6u << 6u | 7u << 9u;

void Require(bool value, const std::string& what) {
	if (!value) {
		throw std::runtime_error(what);
	}
}

ShaderRecompiler::IR::DescriptorValue LinearTexture(uint64_t address, uint32_t last_level) {
	const auto                            format = static_cast<uint32_t>(Prospero::BufferFormat::k8_8_8_8UNorm);
	ShaderRecompiler::IR::DescriptorValue value {.dword_count = 8};
	value.dwords[0] = static_cast<uint32_t>(address >> 8u);
	value.dwords[1] = static_cast<uint32_t>(address >> 40u) | format << 20u | ((Width - 1u) & 0x3u) << 30u;
	value.dwords[2] = (Width - 1u) >> 2u | (Height - 1u) << 14u;
	value.dwords[3] = IdentitySwizzle | last_level << 16u | static_cast<uint32_t>(Prospero::TileMode::kLinear) << 20u |
	                  static_cast<uint32_t>(Prospero::ImageType::kColor2D) << 28u;
	value.dwords[5] = last_level << 4u;
	return value;
}

ShaderRecompiler::IR::ImageResource SampledImage() {
	ShaderRecompiler::IR::ImageResource image;
	image.resource_class = ShaderRecompiler::IR::ImageResourceClass::Sampled;
	image.numeric_class  = Prospero::TextureNumericClass::Float;
	image.dimension      = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
	image.read           = true;
	return image;
}

void CheckRebind(GraphicContext& graphics, uint64_t base, bool table) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	auto& executor  = renderer.GetRenderExecutor();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	ShaderModule plain;
	plain.info.stage = ShaderType::Compute;
	plain.info.info.images.push_back(SampledImage());
	auto root           = SampledImage();
	root.table          = 0;
	root.table_capacity = 2;
	ShaderModule tabled;
	tabled.info.stage = ShaderType::Compute;
	tabled.info.info.images.push_back(root);

	const auto                                   address = base;
	const ShaderRecompiler::IR::DescriptorValue  null_value {.dword_count = 8};
	const auto                                   texture_value = LinearTexture(address, 0);
	const ShaderRecompiler::IR::ResourceSnapshot texture {.images = {texture_value}};
	const ShaderRecompiler::IR::ResourceSnapshot texture_table {
	    .images       = {null_value},
	    .image_tables = {std::make_shared<ShaderRecompiler::IR::ImageTableSnapshot>(
	        ShaderRecompiler::IR::ImageTableSnapshot {.slots = {null_value, texture_value}, .slots_id = 1})}};
	const ShaderRecompiler::IR::ResourceSnapshot mipmapped {.images = {LinearTexture(address, 1)}};
	const ShaderRecompiler::IR::ResourceSnapshot unrelated {.images = {LinearTexture(base + CaseBytes / 2, 0)}};
	const ShaderStageRuntime texture_stage = table ? ShaderStageRuntime {&tabled, &texture_table}
	                                               : ShaderStageRuntime {&plain, &texture};
	const ShaderStageRuntime mipmapped_stage {&plain, &mipmapped};
	const ShaderStageRuntime unrelated_stage {&plain, &unrelated};
	PreparedBindings         first;
	PreparedBindings         second;
	PreparedBindings         third;
	const char*              path = table ? "table texture" : "texture";

	executor.PrepareBindings(texture_stage, first);
	const auto binding = [&]() -> TextureBinding& {
		if (!table) {
			return first.images.at(0);
		}
		Require(first.table_slots.size() == 2, "the table texture was not prepared");
		return *first.table_images.at(first.table_slots[1]);
	};
	const auto handed = binding().image_id;
	Require(static_cast<bool>(handed), std::string(path) + ": the texture was not resolved");
	executor.PrepareBindings(mipmapped_stage, second);
	Require(second.images.at(0).image_id && second.images.at(0).image_id != handed,
	        std::string(path) + ": the mipmapped texture at the same address did not replace the texture's image");
	scheduler.Wait(scheduler.CurrentTick());
	scheduler.PopPendingOperations();
	auto other = std::make_unique<RenderExecutor>(renderer);
	other->PrepareBindings(unrelated_stage, third);
	Require(third.images.at(0).image_id == handed,
	        std::string(path) + ": the unrelated texture did not take the freed image slot");
	const auto unrelated_binding = textures.GetImage(handed).binding;
	const auto unrelated_pins    = textures.GetImage(handed).table_pins;
	Require(unrelated_binding.is_bound, std::string(path) + ": the unrelated texture was not bound");
	Require(ImageViewOps::FormatsCompatible(textures.GetImage(handed).backing.format,
	                                        binding().desc.view_info.format),
	        std::string(path) + ": the unrelated texture cannot take a view of the texture's format");

	executor.RebindImages(first);
	const auto& rebound = binding();
	Require(static_cast<bool>(rebound.image_id), std::string(path) + ": RebindImages left the texture unbound");
	auto& image = textures.GetImage(rebound.image_id);
	Require(image.info.data.address == address,
	        std::string(path) + ": RebindImages bound the texture to the unrelated image that took its slot");
	Require(rebound.TableImage() == TableImageRef {rebound.image_id, image.serial},
	        std::string(path) + ": the rebound texture does not carry the serial of its image");
	Require(std::ranges::find(image.views, rebound.image_view, &CachedImageView::view) != image.views.end(),
	        std::string(path) + ": the rebound view is not a view of the texture's image");
	const auto& kept = textures.GetImage(handed);
	Require(kept.binding.is_bound == unrelated_binding.is_bound &&
	            kept.binding.is_target == unrelated_binding.is_target &&
	            kept.binding.needs_rebind == unrelated_binding.needs_rebind &&
	            kept.binding.force_general == unrelated_binding.force_general &&
	            kept.binding.shader_write == unrelated_binding.shader_write && kept.table_pins == unrelated_pins,
	        std::string(path) + ": RebindImages changed the binding state of the unrelated image in the slot");

	RenderExecutorTestAccess::Reset(executor);
	Require(!image.binding.is_bound && kept.binding.is_bound,
	        std::string(path) + ": resetting bindings did not respect the image owners");
	RenderExecutorTestAccess::Reset(*other);
	Require(!kept.binding.is_bound, std::string(path) + ": the replacement binding was not reset");
}

}

extern "C" int RunKytyStaleTextureBindingTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize(true, true)) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_stale_texture_binding: %s, no usable Vulkan device (%s)\n",
		             required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fputs("kyty_shim_stale_texture_binding: FAILED: could not create the allocator\n", stderr);
		return 1;
	}
	void* guest = nullptr;
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fputs("kyty_shim_stale_texture_binding: FAILED: could not map guest memory\n", stderr);
		return 1;
	}
	const auto  address = reinterpret_cast<uint64_t>(guest);
	std::string failure;
	for (const bool table: {false, true}) {
		try {
			CheckRebind(gpu.graphics, address + (table ? CaseBytes : 0), table);
		} catch (const std::exception& error) {
			failure += failure.empty() ? error.what() : std::string("; ") + error.what();
		}
	}
	if (sceKernelMunmap(address, GuestBytes) != 0 && failure.empty()) {
		failure = "could not unmap guest memory";
	}
	if (!failure.empty()) {
		std::fprintf(stderr, "kyty_shim_stale_texture_binding: FAILED: %s\n", failure.c_str());
		return 1;
	}
	std::puts("kyty_shim_stale_texture_binding: ok");
	return 0;
}

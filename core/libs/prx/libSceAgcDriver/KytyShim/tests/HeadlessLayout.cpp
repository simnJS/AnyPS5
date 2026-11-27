#include "graphics/host_gpu/graphicContext.h"
#include "graphics/presentation/window.h"

#include <cstdio>
#include <string>

extern "C" int RunKytyHeadlessLayoutTest() {
	using Libs::Graphics::GraphicContext;
	constexpr size_t Foreign = sizeof(GraphicContext) + 8u;
	GraphicContext   graphics;
	std::string      error;
	if (Libs::Graphics::CreateHeadlessVulkanChecked(graphics, nullptr, error, Foreign)) {
		std::printf("kyty_shim_headless_layout: FAILED: a caller built with another GraphicContext was accepted\n");
		return 1;
	}
	if (graphics.instance || graphics.device) {
		std::printf("kyty_shim_headless_layout: FAILED: Vulkan was touched before the layout was checked\n");
		return 1;
	}
	if (error.find(std::to_string(Foreign)) == std::string::npos ||
	    error.find(std::to_string(sizeof(GraphicContext))) == std::string::npos) {
		std::printf("kyty_shim_headless_layout: FAILED: the refusal does not name both sizes: %s\n", error.c_str());
		return 1;
	}
	std::printf("kyty_shim_headless_layout: ok\n");
	return 0;
}

#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STALLSNAPSHOT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STALLSNAPSHOT_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace Libs::Graphics {

struct GraphicContext;

namespace StallSnapshot {

using Section = void (*)(void* owner, std::FILE* file);

[[nodiscard]] bool               Enabled();
[[nodiscard]] const std::string& Directory();

void Register(void* owner, const char* name, Section section);
void Unregister(void* owner);

void Name(vk::Semaphore semaphore, std::string name, bool timeline);
void Forget(vk::Semaphore semaphore);
[[nodiscard]] std::string Describe(vk::Semaphore semaphore);
[[nodiscard]] bool        Timeline(vk::Semaphore semaphore);

void NoteOperation(uint64_t tick, const char* kind, uint32_t queue, uint64_t submit_id) noexcept;

[[nodiscard]] std::string Write(GraphicContext& graphics, uint64_t awaited, uint64_t reached, const std::string& verdict);

}

}

#endif

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

struct CompileWait {
	const char* what    = nullptr;
	uint64_t    hash    = 0;
	double      seconds = 0;
	uint32_t    count   = 0;
};
void                      EnterCompileWait(const void* key, const char* what, uint64_t hash);
void                      LeaveCompileWait(const void* key);
[[nodiscard]] CompileWait LongestCompileWait();

enum class Holder : uint8_t { NotIssued, Recording, Driver, Gpu };
inline constexpr double DriverHoldSeconds = 10.0;
inline constexpr uint32_t StuckPeriods    = 6;
[[nodiscard]] Holder Classify(uint64_t tick, uint64_t next_tick, uint64_t submitted, uint64_t submitting_tick,
                              double submitting_seconds) noexcept;
[[nodiscard]] bool   GpuSide(Holder holder) noexcept;
[[nodiscard]] bool   Stuck(Holder holder, uint32_t periods, const CompileWait& compiling) noexcept;

[[nodiscard]] std::string Write(GraphicContext& graphics, uint64_t awaited, uint64_t reached, const std::string& verdict);

}

}

#endif

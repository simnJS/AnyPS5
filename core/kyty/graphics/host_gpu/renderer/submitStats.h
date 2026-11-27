#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SUBMITSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SUBMITSTATS_H_

#include <cstdint>
#include <map>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics::SubmitStats {

enum class Queue : uint8_t { Graphics, Readback, Sparse, Count };

struct Call {
	Queue                queue           = Queue::Graphics;
	uint32_t             batches         = 1;
	uint32_t             command_buffers = 0;
	uint32_t             signal_only     = 0;
	uint32_t             waits           = 0;
	uint64_t             lock_ns         = 0;
	uint64_t             call_ns         = 0;
	uint64_t             tick            = 0;
	std::source_location where {};
};

struct QueueWindow {
	uint64_t              calls           = 0;
	uint64_t              batches         = 0;
	uint64_t              command_buffers = 0;
	uint64_t              signal_only     = 0;
	uint64_t              call_ns         = 0;
	uint64_t              lock_ns         = 0;
	Call                  slowest {};
	std::vector<uint64_t> durations;
};

struct FrameQueue {
	uint64_t calls           = 0;
	uint64_t batches         = 0;
	uint64_t command_buffers = 0;
	uint64_t call_ns         = 0;
	Call     slowest {};
};

struct Frame {
	FrameQueue queues[static_cast<int>(Queue::Count)] {};

	[[nodiscard]] uint64_t CallNs() const;
};

struct Caller {
	uint64_t calls   = 0;
	uint64_t call_ns = 0;
};

class Window {
public:
	void                                   Record(const Call& call, bool ends_frame = false);
	[[nodiscard]] bool                     Empty() const;
	[[nodiscard]] std::vector<std::string> Summary(double seconds);
	[[nodiscard]] Window                   TakeCompleted();

private:
	QueueWindow                                             m_queues[static_cast<int>(Queue::Count)] {};
	Frame                                                   m_frame {};
	std::vector<Frame>                                      m_frames;
	std::map<std::pair<std::string_view, uint32_t>, Caller> m_callers;
};

[[nodiscard]] bool        Enabled() noexcept;
[[nodiscard]] const char* QueueName(Queue queue) noexcept;
[[nodiscard]] std::string CallerName(const std::source_location& where);
void                      Record(const Call& call, bool ends_frame = false);

}

#endif

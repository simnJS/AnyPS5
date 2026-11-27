#include "graphics/host_gpu/renderer/submitStats.h"

#include <cstdio>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace Stats = Libs::Graphics::SubmitStats;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		++g_failures;
	}
}

bool AnyContains(const std::vector<std::string>& lines, std::string_view part) {
	for (const auto& line: lines) {
		if (line.find(part) != std::string::npos) {
			return true;
		}
	}
	return false;
}

Stats::Call Graphics(uint64_t ns, uint32_t buffers, uint64_t tick,
                     std::source_location where = std::source_location::current()) {
	return {.queue           = Stats::Queue::Graphics,
	        .batches         = 1,
	        .command_buffers = buffers,
	        .signal_only     = buffers == 0 ? 1u : 0u,
	        .waits           = buffers == 0 ? 0u : 1u,
	        .lock_ns         = 1000,
	        .call_ns         = ns,
	        .tick            = tick,
	        .where           = where};
}

void CountsAndSlowestCall() {
	Stats::Window window;
	Check(window.Empty(), "a new window is empty");
	const auto flush = std::source_location::current();
	const auto wait  = std::source_location::current();
	window.Record(Graphics(10'000, 1, 1, flush));
	window.Record(Graphics(20'000, 0, 2, flush));
	window.Record({.queue = Stats::Queue::Readback, .command_buffers = 1, .waits = 1, .call_ns = 40'000, .tick = 2});
	window.Record(Graphics(90'000, 1, 3, wait), true);
	Check(!window.Empty(), "a window with calls is not empty");
	const auto lines = window.Summary(10.0);
	Check(lines.size() == 5, "a window with a frame prints the summary, caller, per-frame, slowest and median lines");
	Check(AnyContains(lines, "graphics: 3 calls, 3 batches, 2 command buffers, 1 signal-only"),
	      "the graphics queue counts its calls, batches, command buffers and signal-only batches");
	Check(AnyContains(lines, "median 20.0 us"), "the median call is the middle of three");
	Check(AnyContains(lines, "slowest 90.0 us at tick 3 (1 batches, 1 command buffers, 1 waits, from SubmitStats.cpp:" +
	                             std::to_string(wait.line()) + ")"),
	      "the slowest call keeps its tick, batches, command buffers and waits");
	Check(AnyContains(lines, "0.003 ms waiting for the queue lock"), "the lock wait of the three calls is summed");
	Check(AnyContains(lines, "readback: 1 calls"), "the readback queue is reported apart");
	Check(AnyContains(lines, "40.0 us waiting on tick 2"), "readback reports the timeline tick it waited for");
	Check(AnyContains(lines, "per frame: graphics calls median 3 p99 3 max 3"), "a frame counts the graphics calls in it");
	Check(AnyContains(lines, "by caller: SubmitStats.cpp:" + std::to_string(flush.line()) + " 2 (0.030 ms), SubmitStats.cpp:" +
	                             std::to_string(wait.line()) + " 1 (0.090 ms)"),
	      "graphics submissions are counted by caller, most frequent first");
}

void SlowestAndMedianFrames() {
	Stats::Window window;
	for (uint64_t frame = 0; frame < 5; frame++) {
		window.Record(Graphics(frame == 3 ? 900'000 : 10'000 + frame, 1, frame * 2 + 1));
		window.Record(Graphics(5'000, 0, frame * 2 + 2), true);
	}
	const auto lines = window.Summary(10.0);
	Check(AnyContains(lines, "slowest frame: 905.0 us in vkQueueSubmit"), "the slowest frame is the one with the long call");
	Check(AnyContains(lines, "slowest 900.0 us at tick 7"), "the slowest frame names its slowest call");
	Check(AnyContains(lines, "median frame: 15.0 us in vkQueueSubmit"), "the median frame is the middle of five");
	const auto completed = window.TakeCompleted();
	Check(!completed.Empty(), "the completed window retains its samples");
	Check(window.Empty(), "a reset window is empty");
	Check(window.Summary(10.0).size() == 1, "a reset window has no frames");
}

void CrossWindowFrame() {
	Stats::Window window;
	window.Record({.queue = Stats::Queue::Readback, .call_ns = 12'000});
	auto previous = window.TakeCompleted();
	Check(AnyContains(previous.Summary(10.0), "0 frames; readback: 1 calls"), "a partial frame stays open at rollover");
	window.Record(Graphics(3'000, 1, 1), true);
	Check(AnyContains(window.Summary(10.0), "slowest frame: 15.0 us"), "a frame keeps work from both sides of rollover");
	window.Record({.queue = Stats::Queue::Sparse, .call_ns = 60'000});
	window.Record(Graphics(4'000, 1, 2), true);
	const auto lines = window.Summary(10.0);
	Check(AnyContains(lines, "2 frames; graphics: 2 calls"), "only new submissions count in the next window");
	Check(AnyContains(lines, "median 64.0 us, p99 64.0 us, max 64.0 us"), "the sparse call belongs after the first flip");
	Check(AnyContains(lines, "slowest frame: 64.0 us"), "the flip submit closes its own frame");
}

void TailPercentiles() {
	Stats::Window window;
	for (uint64_t count = 1; count <= 101; ++count) {
		for (uint64_t i = 0; i < count; ++i) {
			window.Record(Graphics(1'000, 1, i), i + 1 == count);
		}
	}
	const auto lines = window.Summary(10.0);
	Check(AnyContains(lines, "graphics calls median 51 p99 100 max 101"), "the p99 count excludes the slowest of 101 frames");
	Check(AnyContains(lines, "median 51.0 us, p99 100.0 us, max 101.0 us"), "the p99 duration excludes the slowest of 101 frames");
}

void NoFramesNoFrameLines() {
	Stats::Window window;
	window.Record({.queue = Stats::Queue::Sparse, .command_buffers = 1, .call_ns = 1'000});
	const auto lines = window.Summary(10.0);
	Check(lines.size() == 1, "a window without frames prints the summary line only");
	Check(AnyContains(lines, "0 frames; sparse: 1 calls"), "the sparse path is reported");
}

}

int main() {
	CountsAndSlowestCall();
	SlowestAndMedianFrames();
	CrossWindowFrame();
	TailPercentiles();
	NoFramesNoFrameLines();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_submit_stats: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_submit_stats: ok\n");
	return 0;
}

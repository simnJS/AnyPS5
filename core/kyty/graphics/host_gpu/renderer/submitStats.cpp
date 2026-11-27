#include "graphics/host_gpu/renderer/submitStats.h"

#include "common/logging/log.h"
#include "common/threadSafety.h"
#include "common/timer.h"

#include <algorithm>
#include <cstdlib>
#include <fmt/format.h>
#include <iterator>
#include <optional>

namespace Libs::Graphics::SubmitStats {

namespace {

constexpr uint64_t SummaryNs    = 10'000'000'000ull;
constexpr size_t   ShownCallers = 8;

struct State {
	Common::AnnotatedMutex mutex;
	Window window          KYTY_GUARDED_BY(mutex);
	uint64_t               window_start KYTY_GUARDED_BY(mutex) = 0;
};

State& GetState() {
	static State state;
	return state;
}

double Us(uint64_t ns) {
	return static_cast<double>(ns) / 1e3;
}

double Ms(uint64_t ns) {
	return static_cast<double>(ns) / 1e6;
}

uint64_t Percentile(std::vector<uint64_t>& values, double fraction) {
	if (values.empty()) {
		return 0;
	}
	const auto index = std::min(values.size() - 1, static_cast<size_t>(fraction * static_cast<double>(values.size())));
	std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
	return values[index];
}

std::string SlowestCall(const Call& call) {
	auto text = fmt::format("{:.1f} us", Us(call.call_ns));
	if (call.queue == Queue::Graphics) {
		text += fmt::format(" at tick {}", call.tick);
	} else if (call.queue == Queue::Readback) {
		text += fmt::format(" waiting on tick {}", call.tick);
	}
	text += fmt::format(" ({} batches, {} command buffers, {} waits", call.batches, call.command_buffers, call.waits);
	if (call.where.line() != 0) {
		text += fmt::format(", from {}", CallerName(call.where));
	}
	return text + ")";
}

std::string FrameText(const Frame& frame) {
	std::string text = fmt::format("{:.1f} us in vkQueueSubmit", Us(frame.CallNs()));
	for (int index = 0; index < static_cast<int>(Queue::Count); index++) {
		const auto& queue = frame.queues[index];
		if (queue.calls == 0) {
			continue;
		}
		text += fmt::format("; {} {} calls, {} batches, {} command buffers, {:.1f} us, slowest {}",
		                    QueueName(static_cast<Queue>(index)), queue.calls, queue.batches, queue.command_buffers,
		                    Us(queue.call_ns), SlowestCall(queue.slowest));
	}
	return text;
}

}

uint64_t Frame::CallNs() const {
	uint64_t total = 0;
	for (const auto& queue: queues) {
		total += queue.call_ns;
	}
	return total;
}

void Window::Record(const Call& call, bool ends_frame) {
	auto& queue = m_queues[static_cast<int>(call.queue)];
	queue.calls++;
	queue.batches += call.batches;
	queue.command_buffers += call.command_buffers;
	queue.signal_only += call.signal_only;
	queue.call_ns += call.call_ns;
	queue.lock_ns += call.lock_ns;
	queue.durations.push_back(call.call_ns);
	if (call.call_ns >= queue.slowest.call_ns) {
		queue.slowest = call;
	}
	auto& frame = m_frame.queues[static_cast<int>(call.queue)];
	frame.calls++;
	frame.batches += call.batches;
	frame.command_buffers += call.command_buffers;
	frame.call_ns += call.call_ns;
	if (call.call_ns >= frame.slowest.call_ns) {
		frame.slowest = call;
	}
	if (call.queue == Queue::Graphics && call.where.line() != 0) {
		auto& caller = m_callers[{call.where.file_name(), call.where.line()}];
		caller.calls++;
		caller.call_ns += call.call_ns;
	}
	if (ends_frame) {
		m_frames.push_back(m_frame);
		m_frame = {};
	}
}

bool Window::Empty() const {
	return std::all_of(std::begin(m_queues), std::end(m_queues), [](const QueueWindow& queue) { return queue.calls == 0; });
}

std::vector<std::string> Window::Summary(double seconds) {
	std::vector<std::string> lines;
	std::string text = fmt::format("[kyty-submit] summary of the last {:.0f} s, {} frames", seconds, m_frames.size());
	for (int index = 0; index < static_cast<int>(Queue::Count); index++) {
		auto& queue = m_queues[index];
		if (queue.calls == 0) {
			continue;
		}
		const auto median = Percentile(queue.durations, 0.5);
		const auto p99    = Percentile(queue.durations, 0.99);
		text += fmt::format(
		    "; {}: {} calls, {} batches, {} command buffers, {} signal-only, {:.3f} ms in vkQueueSubmit (median {:.1f} us, "
		    "p99 {:.1f} us, slowest {}), {:.3f} ms waiting for the queue lock",
		    QueueName(static_cast<Queue>(index)), queue.calls, queue.batches, queue.command_buffers, queue.signal_only,
		    Ms(queue.call_ns), Us(median), Us(p99), SlowestCall(queue.slowest), Ms(queue.lock_ns));
	}
	lines.push_back(text + "\n");
	if (!m_callers.empty()) {
		std::vector<std::pair<std::pair<std::string_view, uint32_t>, Caller>> callers(m_callers.begin(), m_callers.end());
		std::sort(callers.begin(), callers.end(), [](const auto& a, const auto& b) {
			return a.second.calls != b.second.calls ? a.second.calls > b.second.calls : a.first < b.first;
		});
		std::string by_caller = "[kyty-submit] graphics submissions by caller:";
		for (size_t index = 0; index < std::min(callers.size(), ShownCallers); index++) {
			const auto& [where, caller] = callers[index];
			const auto file             = where.first.substr(where.first.find_last_of("/\\") + 1);
			by_caller += fmt::format("{} {}:{} {} ({:.3f} ms)", index == 0 ? "" : ",", file, where.second, caller.calls,
			                         Ms(caller.call_ns));
		}
		if (callers.size() > ShownCallers) {
			uint64_t calls = 0;
			for (size_t index = ShownCallers; index < callers.size(); index++) {
				calls += callers[index].second.calls;
			}
			by_caller += fmt::format(", {} more from {} other callers", calls, callers.size() - ShownCallers);
		}
		lines.push_back(by_caller + "\n");
	}
	if (!m_frames.empty()) {
		std::vector<uint64_t> calls;
		std::vector<uint64_t> batches;
		calls.reserve(m_frames.size());
		batches.reserve(m_frames.size());
		for (const auto& frame: m_frames) {
			const auto& graphics = frame.queues[static_cast<int>(Queue::Graphics)];
			calls.push_back(graphics.calls);
			batches.push_back(graphics.batches);
		}
		auto by_time = m_frames;
		std::sort(by_time.begin(), by_time.end(),
		          [](const Frame& a, const Frame& b) { return a.CallNs() < b.CallNs(); });
		const auto max_calls    = *std::max_element(calls.begin(), calls.end());
		const auto median_calls = Percentile(calls, 0.5);
		const auto p99_calls    = Percentile(calls, 0.99);
		lines.push_back(fmt::format(
		    "[kyty-submit] per frame: graphics calls median {} p99 {} max {}, graphics batches median {}; time in vkQueueSubmit "
		    "median {:.1f} us, p99 {:.1f} us, max {:.1f} us\n",
		    median_calls, p99_calls, max_calls, Percentile(batches, 0.5), Us(by_time[by_time.size() / 2].CallNs()),
		    Us(by_time[std::min(by_time.size() - 1, by_time.size() * 99 / 100)].CallNs()), Us(by_time.back().CallNs())));
		lines.push_back(fmt::format("[kyty-submit] slowest frame: {}\n", FrameText(by_time.back())));
		lines.push_back(fmt::format("[kyty-submit] median frame: {}\n", FrameText(by_time[by_time.size() / 2])));
	}
	return lines;
}

Window Window::TakeCompleted() {
	Window next;
	next.m_frame = m_frame;
	return std::exchange(*this, std::move(next));
}

bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SUBMIT_STATS");
		return value != nullptr && std::string_view(value) == "1";
	}();
	return enabled;
}

const char* QueueName(Queue queue) noexcept {
	switch (queue) {
		case Queue::Graphics: return "graphics";
		case Queue::Readback: return "readback";
		case Queue::Sparse: return "sparse";
		default: return "?";
	}
}

std::string CallerName(const std::source_location& where) {
	const std::string_view file = where.file_name();
	return fmt::format("{}:{}", file.substr(file.find_last_of("/\\") + 1), where.line());
}

void Record(const Call& call, bool ends_frame) {
	auto&                 state = GetState();
	std::optional<Window> completed;
	uint64_t              elapsed = 0;
	{
		Common::ScopedLock lock(state.mutex);
		const auto now = Common::SteadyNs();
		if (state.window_start == 0) {
			state.window_start = now;
		}
		elapsed = now - state.window_start;
		if (elapsed >= SummaryNs) {
			completed.emplace(state.window.TakeCompleted());
			state.window_start = now;
		}
		state.window.Record(call, ends_frame);
	}
	if (completed) {
		for (const auto& line: completed->Summary(static_cast<double>(elapsed) / 1e9)) {
			::Log::WriteToConsoleAndLog(line);
		}
	}
}

}

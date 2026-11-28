#include "common/skipLog.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Common::SkipLog {
namespace {

constexpr int64_t WindowNs   = 10'000'000'000;
constexpr size_t  ShownLines = 12;

struct Key {
	uint64_t hash;
	Kind     kind;
	Cause    cause;
	bool     operator==(const Key& other) const = default;
};

struct KeyHash {
	size_t operator()(const Key& key) const noexcept {
		return static_cast<size_t>(key.hash ^ (uint64_t {static_cast<uint8_t>(key.kind)} << 56u) ^
		                           (uint64_t {static_cast<uint8_t>(key.cause)} << 48u));
	}
};

struct Count {
	uint64_t window = 0;
	uint64_t total  = 0;
};

struct State {
	std::mutex                                 mutex;
	std::unordered_map<Key, Count, KeyHash>    counts;
	std::array<uint64_t, static_cast<size_t>(Kind::Count)> totals {};
	int64_t                                    window_start_ns = 0;
	uint64_t                                   window_frame    = 0;
	bool                                       frame_known     = false;
};

State& Get() {
	static State state;
	return state;
}

std::atomic<uint64_t> g_pending {0};

thread_local uint64_t t_hash  = 0;
thread_local Cause    t_cause = Cause::Unknown;

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

std::string_view KindName(Kind kind) {
	return kind == Kind::Draw ? "draw" : "dispatch";
}

Window Flush(State& state, int64_t now, uint64_t frame) {
	Window window;
	window.frames = state.frame_known && frame >= state.window_frame ? frame - state.window_frame : 0;
	const double seconds = 1e-9 * static_cast<double>(std::max<int64_t>(now - state.window_start_ns, 0));
	std::vector<std::pair<Key, Count>> lines;
	std::array<std::array<uint64_t, static_cast<size_t>(Cause::Count)>, static_cast<size_t>(Kind::Count)> by_cause {};
	for (auto& [key, count]: state.counts) {
		if (count.window == 0) continue;
		lines.emplace_back(key, count);
		by_cause[static_cast<size_t>(key.kind)][static_cast<size_t>(key.cause)] += count.window;
		(key.kind == Kind::Draw ? window.draws : window.dispatches) += count.window;
		count.window = 0;
	}
	state.window_start_ns = now;
	state.window_frame    = frame;
	state.frame_known     = true;
	if (lines.empty()) return window;
	std::ranges::sort(lines, [](const auto& a, const auto& b) {
		return a.second.window != b.second.window ? a.second.window > b.second.window : a.first.hash < b.first.hash;
	});
	const double per_frame =
	    window.frames != 0 ? static_cast<double>(window.draws + window.dispatches) / static_cast<double>(window.frames) : 0.0;
	std::string causes;
	for (size_t kind = 0; kind < by_cause.size(); kind++) {
		for (size_t cause = 0; cause < by_cause[kind].size(); cause++) {
			if (by_cause[kind][cause] == 0) continue;
			causes += fmt::format("{}{} {} {}", causes.empty() ? "" : ", ", KindName(static_cast<Kind>(kind)),
			                      CauseName(static_cast<Cause>(cause)), by_cause[kind][cause]);
		}
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-skip] summary of the last {:.1f} s, {} frames: {} draws and {} dispatches not executed "
	    "({:.2f} per frame; since start {} draws, {} dispatches); by cause: {}\n",
	    seconds, window.frames, window.draws, window.dispatches, per_frame,
	    state.totals[static_cast<size_t>(Kind::Draw)], state.totals[static_cast<size_t>(Kind::Dispatch)], causes));
	const auto shown = std::min(lines.size(), ShownLines);
	for (size_t i = 0; i < shown; i++) {
		const auto& [key, count] = lines[i];
		Log::WriteToConsoleAndLog(fmt::format("[kyty-skip]   {} 0x{:016x} {}: +{} (total {})\n", KindName(key.kind),
		                                      key.hash, CauseName(key.cause), count.window, count.total));
	}
	if (lines.size() > shown) {
		Log::WriteToConsoleAndLog(fmt::format("[kyty-skip]   (+{} more programs)\n", lines.size() - shown));
	}
	window.lines = static_cast<uint32_t>(lines.size());
	return window;
}

}

std::string_view CauseName(Cause cause) {
	switch (cause) {
		case Cause::Unknown: return "unknown";
		case Cause::UnresolvableResources: return "unresolvable-resources";
		case Cause::CallTables: return "call-tables";
		case Cause::UnsupportedCall: return "unsupported-call";
		case Cause::RecompilerStopped: return "recompiler-stopped";
		case Cause::StageNotPrepared: return "stage-not-prepared";
		case Cause::FailedVariant: return "failed-variant";
		case Cause::ModuleNotBuilt: return "module-not-built";
		case Cause::NoPipeline: return "no-pipeline";
		case Cause::TessellationOff: return "tessellation-off";
		case Cause::TessellationWithGeometry: return "tessellation-with-geometry";
		case Cause::TessellationLayout: return "tessellation-layout";
		case Cause::MeshLimits: return "mesh-limits";
		case Cause::Count: break;
	}
	return "unknown";
}

void Request(uint64_t shader_hash) {
	t_hash  = shader_hash;
	t_cause = Cause::Unknown;
}

void Note(std::string_view hack, uint64_t shader_hash) {
	struct Known {
		std::string_view prefix;
		Cause            cause;
	};
	static constexpr std::array<Known, 4> Causes {{
	    {"draw or dispatch with unresolvable resources", Cause::UnresolvableResources},
	    {"dispatch whose call tables do not expand", Cause::CallTables},
	    {"draw or dispatch skipped: the shader recompiler stopped", Cause::RecompilerStopped},
	    {"draw or dispatch skipped: its shader stage could not be prepared", Cause::StageNotPrepared},
	}};
	for (const auto& known: Causes) {
		if (hack.starts_with(known.prefix)) {
			Fail(known.cause, shader_hash);
			return;
		}
	}
}

void Fail(Cause cause, uint64_t shader_hash) {
	t_cause = cause;
	if (shader_hash != 0) t_hash = shader_hash;
}

void Lost(Kind kind) {
	Lost(kind, t_cause, t_hash);
}

void Lost(Kind kind, Cause cause, uint64_t shader_hash) {
	t_hash  = 0;
	t_cause = Cause::Unknown;
	auto& state = Get();
	{
		std::scoped_lock lock(state.mutex);
		auto& count = state.counts[Key {shader_hash, kind, cause}];
		count.window++;
		count.total++;
		state.totals[static_cast<size_t>(kind)]++;
	}
	g_pending.fetch_add(1, std::memory_order_release);
}

void Summarize(uint64_t frame) {
	auto& state = Get();
	const auto now = NowNs();
	std::scoped_lock lock(state.mutex);
	if (!state.frame_known) {
		state.window_start_ns = now;
		state.window_frame    = frame;
		state.frame_known     = true;
		return;
	}
	if (now - state.window_start_ns < WindowNs) return;
	if (g_pending.exchange(0, std::memory_order_acq_rel) == 0) {
		state.window_start_ns = now;
		state.window_frame    = frame;
		return;
	}
	Flush(state, now, frame);
}

Window SummarizeNow(uint64_t frame) {
	auto& state = Get();
	std::scoped_lock lock(state.mutex);
	g_pending.store(0, std::memory_order_relaxed);
	return Flush(state, NowNs(), frame);
}

}

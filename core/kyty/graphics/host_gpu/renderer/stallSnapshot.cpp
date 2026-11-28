#include "graphics/host_gpu/renderer/stallSnapshot.h"

#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <span>
#include <sstream>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace Libs::Graphics::StallSnapshot {
namespace {

constexpr size_t Operations = 2048;

struct Operation {
	std::atomic<uint64_t>    tick {0};
	std::atomic<const char*> kind {nullptr};
	std::atomic<uint32_t>    queue {0};
	std::atomic<uint64_t>    submit_id {0};
	std::atomic<uint64_t>    ns {0};
};

struct Entry {
	void*       owner   = nullptr;
	const char* name    = nullptr;
	Section     section = nullptr;
};

struct Named {
	VkSemaphore handle   = VK_NULL_HANDLE;
	std::string name;
	bool        timeline = false;
};

std::mutex                       g_sections_mutex;
std::vector<Entry>               g_sections;
std::mutex                       g_names_mutex;
std::vector<Named>               g_names;
std::array<Operation, Operations> g_operations;
std::atomic<uint64_t>            g_operation_next {0};

struct Compile {
	const void* key   = nullptr;
	const char* what  = nullptr;
	uint64_t    hash  = 0;
	uint64_t    since = 0;
};
std::mutex           g_compiles_mutex;
std::vector<Compile> g_compiles;

uint64_t SteadyNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

double Ago(uint64_t now, uint64_t then) {
	return then == 0 || then > now ? 0.0 : static_cast<double>(now - then) / 1e9;
}

void WriteOperations(std::FILE* file, uint64_t reached, uint64_t now) {
	const auto next  = g_operation_next.load(std::memory_order_acquire);
	const auto first = next > Operations ? next - Operations : 0;
	struct Tick {
		std::map<std::string_view, uint64_t> kinds;
		uint64_t                             first_id = UINT64_MAX;
		uint64_t                             last_id  = 0;
		uint64_t                             first_ns = UINT64_MAX;
		uint64_t                             last_ns  = 0;
	};
	std::map<uint64_t, Tick> ticks;
	for (auto index = first; index < next; index++) {
		const auto& operation = g_operations[index % Operations];
		const auto  tick      = operation.tick.load(std::memory_order_relaxed);
		if (tick + 2 <= reached) continue;
		const auto* kind = operation.kind.load(std::memory_order_relaxed);
		auto&       slot = ticks[tick];
		slot.kinds[kind != nullptr ? std::string_view(kind) : std::string_view("?")]++;
		const auto id = operation.submit_id.load(std::memory_order_relaxed);
		const auto ns = operation.ns.load(std::memory_order_relaxed);
		slot.first_id = std::min(slot.first_id, id);
		slot.last_id  = std::max(slot.last_id, id);
		slot.first_ns = std::min(slot.first_ns, ns);
		slot.last_ns  = std::max(slot.last_ns, ns);
	}
	std::fprintf(file, "\n== guest operations by the tick they were recorded in (the GPU is at %" PRIu64 ", %" PRIu64
	                   " operations noted in all)\n",
	             reached, next);
	if (ticks.empty()) std::fprintf(file, "none noted from tick %" PRIu64 " on\n", reached > 0 ? reached - 1 : 0);
	for (const auto& [tick, slot]: ticks) {
		std::fprintf(file, "tick %" PRIu64 "%s: submissions %" PRIu64 " to %" PRIu64 ", first %.3f s ago, last %.3f s ago:", tick,
		             tick <= reached ? " (finished)" : " (unfinished)", slot.first_id, slot.last_id, Ago(now, slot.first_ns),
		             Ago(now, slot.last_ns));
		for (const auto& [kind, count]: slot.kinds) {
			std::fprintf(file, " %.*s %" PRIu64, static_cast<int>(kind.size()), kind.data(), count);
		}
		std::fprintf(file, "\n");
	}
	std::fprintf(file, "\n== the last guest operations, last first\n");
	for (uint64_t i = 1; i <= std::min<uint64_t>(next, 48); i++) {
		const auto& operation = g_operations[(next - i) % Operations];
		const auto* kind      = operation.kind.load(std::memory_order_relaxed);
		std::fprintf(file, "%.3f s ago tick %" PRIu64 " %s queue %u submission %" PRIu64 "\n",
		             Ago(now, operation.ns.load(std::memory_order_relaxed)), operation.tick.load(std::memory_order_relaxed),
		             kind != nullptr ? kind : "?", operation.queue.load(std::memory_order_relaxed),
		             operation.submit_id.load(std::memory_order_relaxed));
	}
}

void WriteCompiles(std::FILE* file, uint64_t now) {
	std::scoped_lock lock(g_compiles_mutex);
	std::fprintf(file, "\n== pipeline compile waits in progress: %zu\n", g_compiles.size());
	for (const auto& compile: g_compiles) {
		std::fprintf(file, "%s 0x%016" PRIx64 " for %.3f s\n", compile.what != nullptr ? compile.what : "?", compile.hash,
		             Ago(now, compile.since));
	}
}

void WriteHeaps(std::FILE* file, GraphicContext& graphics) {
	std::fprintf(file, "\n== device memory heaps\n");
	if (!graphics.CanReportMemoryUsage()) {
		std::fprintf(file, "the driver does not report the heap budgets\n");
		return;
	}
	std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
	const auto                                                 count = graphics.GetHeapUsage(heaps);
	for (uint32_t heap = 0; heap < count; heap++) {
		std::fprintf(file, "heap %u %s: %" PRIu64 " MiB used of a %" PRIu64 " MiB budget, %u allocations\n", heap,
		             heaps[heap].device_local ? "device local" : "system", heaps[heap].usage >> 20u, heaps[heap].budget >> 20u,
		             heaps[heap].allocations);
	}
}

#if defined(__linux__)
std::string ReadSmall(const std::string& path) {
	std::ifstream in(path);
	std::string   text;
	std::getline(in, text, '\0');
	while (!text.empty() && (text.back() == '\n' || text.back() == ' ')) text.pop_back();
	return text;
}

std::string FdPath(uint64_t fd) {
	std::error_code error;
	const auto      target = std::filesystem::read_symlink("/proc/self/fd/" + std::to_string(fd), error);
	return error ? std::string("?") : target.string();
}

const char* SyscallName(int64_t number) {
#if defined(__x86_64__)
	switch (number) {
		case 0: return "read";
		case 1: return "write";
		case 7: return "poll";
		case 16: return "ioctl";
		case 17: return "pread64";
		case 18: return "pwrite64";
		case 23: return "select";
		case 24: return "sched_yield";
		case 35: return "nanosleep";
		case 61: return "wait4";
		case 202: return "futex";
		case 230: return "clock_nanosleep";
		case 232: return "epoll_wait";
		case 270: return "pselect6";
		case 271: return "ppoll";
		case 281: return "epoll_pwait";
		case 441: return "epoll_pwait2";
		case 449: return "futex_waitv";
		default: return nullptr;
	}
#else
	(void)number;
	return nullptr;
#endif
}

bool TakesFd(int64_t number) {
#if defined(__x86_64__)
	return number == 0 || number == 1 || number == 16 || number == 17 || number == 18;
#else
	(void)number;
	return false;
#endif
}

struct ThreadSample {
	int         tid   = 0;
	std::string comm;
	char        state = '?';
	uint64_t    cpu   = 0;
};

std::vector<ThreadSample> SampleThreads() {
	std::vector<ThreadSample> threads;
	std::error_code           error;
	for (const auto& entry: std::filesystem::directory_iterator("/proc/self/task", error)) {
		ThreadSample sample;
		sample.tid       = std::atoi(entry.path().filename().c_str());
		const auto stat  = ReadSmall(entry.path().string() + "/stat");
		const auto open  = stat.find('(');
		const auto close = stat.rfind(')');
		if (sample.tid == 0 || open == std::string::npos || close == std::string::npos || close < open) continue;
		sample.comm = stat.substr(open + 1, close - open - 1);
		std::istringstream rest(stat.substr(close + 1));
		std::string        field;
		std::vector<std::string> fields;
		while (rest >> field) fields.push_back(field);
		if (fields.size() > 12) {
			sample.state = fields[0].empty() ? '?' : fields[0][0];
			sample.cpu   = std::strtoull(fields[11].c_str(), nullptr, 10) + std::strtoull(fields[12].c_str(), nullptr, 10);
		}
		threads.push_back(std::move(sample));
	}
	return threads;
}

void WriteThreads(std::FILE* file) {
	constexpr auto Interval = std::chrono::milliseconds(250);
	const auto     before   = SampleThreads();
	std::this_thread::sleep_for(Interval);
	auto       after = SampleThreads();
	const auto hz    = static_cast<double>(std::max<long>(sysconf(_SC_CLK_TCK), 1));
	struct Line {
		ThreadSample sample;
		double       cpu = 0;
		std::string  call;
		std::string  wchan;
		bool         driver = false;
	};
	std::vector<Line> lines;
	const auto        self = static_cast<int>(gettid());
	for (auto& sample: after) {
		Line line;
		const auto earlier = std::ranges::find_if(before, [&](const ThreadSample& other) { return other.tid == sample.tid; });
		if (earlier != before.end() && sample.cpu >= earlier->cpu) {
			line.cpu = static_cast<double>(sample.cpu - earlier->cpu) / hz /
			           std::chrono::duration<double>(Interval).count() * 100.0;
		}
		const auto base = "/proc/self/task/" + std::to_string(sample.tid);
		line.wchan      = ReadSmall(base + "/wchan");
		const auto call = sample.tid == self ? std::string("writing this report") : ReadSmall(base + "/syscall");
		std::istringstream words(call);
		std::string        first;
		words >> first;
		if (sample.tid == self) {
			line.call = call;
		} else if (first.empty() || first == "running") {
			line.call = first.empty() ? "?" : "running";
		} else {
			const auto number = std::strtoll(first.c_str(), nullptr, 10);
			std::vector<uint64_t> args;
			std::string           word;
			while (words >> word && args.size() < 3) args.push_back(std::strtoull(word.c_str(), nullptr, 16));
			const auto* name = SyscallName(number);
			if (number < 0) {
				line.call = "not in a system call";
			} else {
				line.call = name != nullptr ? std::string(name) : "system call " + std::to_string(number);
				if (TakesFd(number) && !args.empty()) {
					const auto path = FdPath(args[0]);
					line.call += " on " + path;
					line.driver = path.starts_with("/dev/nvidia") || path.starts_with("/dev/dri");
					if (number == 16 && args.size() > 1) {
						char request[32];
						std::snprintf(request, sizeof(request), " request 0x%" PRIx64, args[1]);
						line.call += request;
					}
				}
			}
		}
		line.sample = std::move(sample);
		lines.push_back(std::move(line));
	}
	std::ranges::sort(lines, [](const Line& a, const Line& b) {
		if (a.driver != b.driver) return a.driver;
		if (a.cpu != b.cpu) return a.cpu > b.cpu;
		return a.sample.tid < b.sample.tid;
	});
	uint64_t    in_driver = 0;
	std::string names;
	for (const auto& line: lines) {
		if (!line.driver) continue;
		in_driver++;
		names += (names.empty() ? "" : ", ") + line.sample.comm + " (" + std::to_string(line.sample.tid) + ")";
	}
	std::fprintf(file, "\n== threads (%zu), CPU over %lld ms; %" PRIu64 " in a call to the GPU driver%s%s\n", lines.size(),
	             static_cast<long long>(Interval.count()), in_driver, in_driver != 0 ? ": " : "", names.c_str());
	for (const auto& line: lines) {
		std::fprintf(file, "%d %s state %c cpu %.0f %% wchan %s, %s\n", line.sample.tid, line.sample.comm.c_str(),
		             line.sample.state, line.cpu, line.wchan.empty() ? "?" : line.wchan.c_str(), line.call.c_str());
	}
}
#endif

}

bool Enabled() {
	static const bool enabled = !Directory().empty();
	return enabled;
}

const std::string& Directory() {
	static const std::string directory = [] {
		const char* value = std::getenv("KYTY_HANG_SNAPSHOT_DIR");
		return value != nullptr ? std::string(value) : std::string();
	}();
	return directory;
}

void Register(void* owner, const char* name, Section section) {
	std::scoped_lock lock(g_sections_mutex);
	g_sections.push_back({owner, name, section});
}

void Unregister(void* owner) {
	std::scoped_lock lock(g_sections_mutex);
	std::erase_if(g_sections, [owner](const Entry& entry) { return entry.owner == owner; });
}

void Name(vk::Semaphore semaphore, std::string name, bool timeline) {
	if (!semaphore) return;
	std::scoped_lock lock(g_names_mutex);
	const auto       handle = static_cast<VkSemaphore>(semaphore);
	std::erase_if(g_names, [handle](const Named& named) { return named.handle == handle; });
	g_names.push_back({handle, std::move(name), timeline});
}

void Forget(vk::Semaphore semaphore) {
	std::scoped_lock lock(g_names_mutex);
	const auto       handle = static_cast<VkSemaphore>(semaphore);
	std::erase_if(g_names, [handle](const Named& named) { return named.handle == handle; });
}

std::string Describe(vk::Semaphore semaphore) {
	std::scoped_lock lock(g_names_mutex);
	const auto       handle = static_cast<VkSemaphore>(semaphore);
	for (const auto& named: g_names) {
		if (named.handle == handle) return named.name;
	}
	char text[48];
	std::snprintf(text, sizeof(text), "semaphore 0x%" PRIx64, reinterpret_cast<uint64_t>(handle));
	return text;
}

bool Timeline(vk::Semaphore semaphore) {
	std::scoped_lock lock(g_names_mutex);
	const auto       handle = static_cast<VkSemaphore>(semaphore);
	return std::ranges::any_of(g_names, [handle](const Named& named) { return named.handle == handle && named.timeline; });
}

void NoteOperation(uint64_t tick, const char* kind, uint32_t queue, uint64_t submit_id) noexcept {
	if (!Enabled()) return;
	auto& operation = g_operations[g_operation_next.load(std::memory_order_relaxed) % Operations];
	operation.tick.store(tick, std::memory_order_relaxed);
	operation.kind.store(kind, std::memory_order_relaxed);
	operation.queue.store(queue, std::memory_order_relaxed);
	operation.submit_id.store(submit_id, std::memory_order_relaxed);
	operation.ns.store(SteadyNs(), std::memory_order_relaxed);
	g_operation_next.fetch_add(1, std::memory_order_release);
}

void EnterCompileWait(const void* key, const char* what, uint64_t hash) {
	std::scoped_lock lock(g_compiles_mutex);
	g_compiles.push_back({key, what, hash, SteadyNs()});
}

void LeaveCompileWait(const void* key) {
	std::scoped_lock lock(g_compiles_mutex);
	std::erase_if(g_compiles, [key](const Compile& compile) { return compile.key == key; });
}

CompileWait LongestCompileWait() {
	const auto       now = SteadyNs();
	std::scoped_lock lock(g_compiles_mutex);
	CompileWait      longest;
	longest.count = static_cast<uint32_t>(g_compiles.size());
	for (const auto& compile: g_compiles) {
		const auto seconds = Ago(now, compile.since);
		if (longest.what == nullptr || seconds > longest.seconds) {
			longest.what    = compile.what != nullptr ? compile.what : "?";
			longest.hash    = compile.hash;
			longest.seconds = seconds;
		}
	}
	return longest;
}

Holder Classify(uint64_t tick, uint64_t next_tick, uint64_t submitted, uint64_t submitting_tick,
                double submitting_seconds) noexcept {
	if (tick <= submitted) return Holder::Gpu;
	if (submitting_tick != 0 && submitting_tick == submitted + 1 && submitting_seconds >= DriverHoldSeconds) return Holder::Driver;
	if (tick >= next_tick) return Holder::NotIssued;
	return Holder::Recording;
}

bool GpuSide(Holder holder) noexcept {
	return holder == Holder::Gpu || holder == Holder::Driver;
}

bool Stuck(Holder holder, uint32_t periods, const CompileWait& compiling) noexcept {
	return !GpuSide(holder) && periods >= StuckPeriods && compiling.count == 0;
}

std::string Write(GraphicContext& graphics, uint64_t awaited, uint64_t reached, const std::string& verdict) {
	if (!Enabled()) return {};
	std::error_code error;
	std::filesystem::create_directories(Directory(), error);
	const auto now  = SteadyNs();
	const auto wall = std::time(nullptr);
	char       stamp[32] {};
	std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", std::localtime(&wall));
#if defined(__linux__)
	const auto pid = static_cast<long long>(getpid());
#else
	const long long pid = 0;
#endif
	const auto path = (std::filesystem::path(Directory()) /
	                   ("stall-" + std::string(stamp) + "-tick" + std::to_string(awaited) + ".txt"))
	                      .string();
	auto* file = std::fopen(path.c_str(), "w");
	if (file == nullptr) return {};
	std::fprintf(file, "%s\nprocess %lld, %s, tick %" PRIu64 " awaited, the GPU is at %" PRIu64 "\n", verdict.c_str(), pid, stamp,
	             awaited, reached);
	{
		std::scoped_lock lock(g_sections_mutex);
		for (const auto& entry: g_sections) {
			std::fprintf(file, "\n== %s\n", entry.name);
			entry.section(entry.owner, file);
			std::fflush(file);
		}
	}
	WriteCompiles(file, now);
	WriteOperations(file, reached, now);
	WriteHeaps(file, graphics);
	std::fflush(file);
#if defined(__linux__)
	WriteThreads(file);
#endif
	std::fclose(file);
	return path;
}

}

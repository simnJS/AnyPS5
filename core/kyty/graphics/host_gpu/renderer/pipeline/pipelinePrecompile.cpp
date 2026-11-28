#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#endif
#include "graphics/presentation/precompileNotice.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" void KernelPauseGuestClock_nid_no_patch();
extern "C" void KernelResumeGuestClock_nid_no_patch();

namespace Libs::Graphics {

namespace {

constexpr double SlowSeconds   = 0.5;
constexpr auto   NoticeLinger  = std::chrono::seconds(3);
constexpr auto     WatchPeriod = std::chrono::milliseconds(100);
constexpr auto     LogPeriod   = std::chrono::seconds(10);
constexpr auto     StallShown  = std::chrono::seconds(3);
constexpr auto     StallGap    = std::chrono::seconds(1);
constexpr auto     ManifestBarDelay  = std::chrono::seconds(1);
constexpr auto     ManifestBarLinger = std::chrono::seconds(2);

template <typename... Args>
void PrecompileLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

#if defined(ANYPS5_KYTY)
struct DrawGeneric {
	GenericServe state   = GenericServe::Served;
	uint64_t     program = 0;
};
thread_local DrawGeneric t_draw_generic;
thread_local std::string t_added_cause;

#endif

struct Estimate {
	uint32_t pending  = 0;
	uint64_t finished = 0;
	uint32_t shaders  = 0;
	uint64_t shaders_finished = 0;
	uint32_t running  = 0;
	uint64_t modules_found     = 0;
	uint64_t modules_built     = 0;
	uint64_t pipelines_created = 0;
};

}

#if defined(ANYPS5_KYTY)
std::string SpecializationDifference(const ShaderRecompiler::IR::ResourceSpecialization& wanted,
                                     const ShaderRecompiler::IR::ResourceSpecialization& prepared, size_t& distance) {
	static constexpr std::array<const char*, 12> Names {
	    "buffer-bounds-bits", "buffer-empty", "buffer-format", "buffer-stride", "buffer-table", "image-class",
	    "image-dimension",    "image-format", "image-heap",    "image-search",  "image-table",  "sampler-table"};
	enum : uint32_t {
		BufferBounds, BufferEmpty, BufferFormat, BufferStride, BufferTable, ImageClass,
		ImageDimension, ImageFormat, ImageHeap, ImageSearch, ImageTable, SamplerTable
	};
	std::vector<std::string> reasons;
	uint32_t                 fields = 0;
	distance                        = 0;
	const auto same_count = [&](const char* name, size_t now, size_t before) {
		if (now == before) return true;
		reasons.push_back(fmt::format("{}-count({}->{})", name, before, now));
		distance += 1000u + (now > before ? now - before : before - now);
		return false;
	};
	if (same_count("buffer", wanted.buffers.size(), prepared.buffers.size())) {
		constexpr uint32_t bounds = ShaderRecompiler::IR::StructuredBufferBounds | ShaderRecompiler::IR::ExactBufferBounds;
		for (size_t i = 0; i < wanted.buffers.size(); i++) {
			const auto& now    = wanted.buffers[i];
			const auto& before = prepared.buffers[i];
			if (now == before) continue;
			distance++;
			if (now.packed_stride != before.packed_stride) {
				fields |= 1u << (((now.packed_stride ^ before.packed_stride) & ~bounds) == 0u ? BufferBounds : BufferStride);
			}
			if (now.descriptor_format != before.descriptor_format || now.descriptor_swizzle != before.descriptor_swizzle) {
				fields |= 1u << BufferFormat;
			}
			if (now.zero_stride_oob != before.zero_stride_oob || now.scalar_empty != before.scalar_empty) {
				fields |= 1u << BufferEmpty;
			}
			if (now.indirect_root != before.indirect_root || now.indirect_mapping_offset != before.indirect_mapping_offset ||
			    now.indirect_search_iterations != before.indirect_search_iterations) {
				fields |= 1u << BufferTable;
			}
		}
	}
	if (same_count("image", wanted.images.size(), prepared.images.size())) {
		for (size_t i = 0; i < wanted.images.size(); i++) {
			const auto& now    = wanted.images[i];
			const auto& before = prepared.images[i];
			if (now == before) continue;
			distance++;
			if (now.numeric_class != before.numeric_class) fields |= 1u << ImageClass;
			if (now.dimension != before.dimension || now.cube != before.cube) fields |= 1u << ImageDimension;
			if (now.table != before.table || now.table_capacity != before.table_capacity ||
			    now.table_mapping_offset != before.table_mapping_offset || now.table_entry_mask != before.table_entry_mask) {
				fields |= 1u << ImageTable;
			}
			if (now.heap != before.heap) fields |= 1u << ImageHeap;
			if (now.indirect_root != before.indirect_root || now.indirect_mapping_offset != before.indirect_mapping_offset ||
			    now.indirect_search_iterations != before.indirect_search_iterations) {
				fields |= 1u << ImageSearch;
			}
			if (now.mip_count != before.mip_count || now.conversion_format != before.conversion_format ||
			    now.shader_swizzle != before.shader_swizzle || now.srgb_decode != before.srgb_decode ||
			    now.skip_degamma != before.skip_degamma || now.fmask != before.fmask) {
				fields |= 1u << ImageFormat;
			}
		}
	}
	if (same_count("sampler", wanted.samplers.size(), prepared.samplers.size())) {
		for (size_t i = 0; i < wanted.samplers.size(); i++) {
			if (wanted.samplers[i] == prepared.samplers[i]) continue;
			distance++;
			fields |= 1u << SamplerTable;
		}
	}
	for (uint32_t field = 0; field < Names.size(); field++) {
		if ((fields & (1u << field)) != 0u) reasons.emplace_back(Names[field]);
	}
	std::ranges::sort(reasons);
	return fmt::format("{}", fmt::join(reasons, "+"));
}

std::string DescribeProgramWait(const ProgramWaitFacts& facts) {
	const auto unbound = [&] {
		if (facts.compute && !facts.serve_refusal.empty()) return "generic-unbound:" + facts.serve_refusal;
		return "generic-unbound:" + (facts.difference.empty() ? std::string("push-data-start") : facts.difference);
	};
	if (!facts.compute) {
		if (facts.draw == GenericServe::PipelineMissing) return "library-missing:generic-pipeline";
		if (facts.generic_prepared) return facts.other_variant ? unbound() : "generic-unbound:no-exact-recipe";
		if (!facts.generic_refusal.empty()) return "generic-refused:" + facts.generic_refusal;
		if (facts.other_variant) return "generic-not-prepared:no-generic-module";
	} else if (facts.other_variant) {
		return unbound();
	}
	if (!facts.recipe_refusal.empty()) return "generic-refused:" + facts.recipe_refusal;
	if (!facts.other_keys.empty()) return "other-key:" + facts.other_keys;
	if (facts.known == 0) return "unknown-program";
	if (facts.calls) return "generic-not-prepared:calls";
	return facts.known < 0 ? "generic-not-prepared:code-store-unread" : "generic-not-prepared:no-recipe";
}

std::string DescribeStallWait(const StallWaitFacts& facts) {
	if (!facts.module_cause.empty()) return facts.module_cause;
	if (facts.compute && !facts.serve_refusal.empty()) return "generic-unbound:" + facts.serve_refusal;
	if (facts.compute) return "other:prepared-module-without-driver-pipeline";
	const auto  route = facts.route;
	std::string parts;
	if (route.find("pr=queued") != std::string_view::npos || route.find("pr=building") != std::string_view::npos) parts = "pr";
	if (route.find("fs=queued") != std::string_view::npos || route.find("fs=building") != std::string_view::npos) {
		parts += parts.empty() ? "fs" : "+fs";
	}
	if (parts.empty()) parts = route.starts_with("linked") ? "vi+fo" : "pipeline";
	if (facts.draw == GenericServe::Unbound) return "generic-unbound:exact-stages-prepared+" + parts;
	if (facts.draw == GenericServe::NoModule && !facts.generic_refusal.empty()) return "generic-refused:" + facts.generic_refusal;
	return (facts.draw == GenericServe::PipelineMissing ? "library-missing:generic-pipeline+" : "library-missing:") + parts;
}
#endif

PipelineCache::GuestClockHold::GuestClockHold() {
	KernelPauseGuestClock_nid_no_patch();
}

PipelineCache::GuestClockHold::~GuestClockHold() {
	KernelResumeGuestClock_nid_no_patch();
}

PipelineCache::StallScope::StallScope(PipelineCache& cache): StallScope(cache, true) {}

PipelineCache::StallScope::StallScope(PipelineCache& cache, bool hold_clock): m_cache(cache) {
	if (hold_clock) {
		m_clock_hold.emplace();
	}
	std::scoped_lock lock(cache.m_stall_mutex);
	if (cache.m_stall.threads++ == 0) {
		cache.m_stall.since = std::chrono::steady_clock::now();
	}
}

#if defined(ANYPS5_KYTY)
PipelineCache::StallScope::StallScope(PipelineCache& cache, const char* what, uint64_t hash, const char* why,
                                      bool hold_clock)
    : StallScope(cache, hold_clock) {
	PipelineRouteNote() = nullptr;
	m_what    = what;
	m_hash    = hash;
	m_why     = why;
	std::scoped_lock lock(cache.m_stall_mutex);
	m_started = std::chrono::steady_clock::now();
	if (hash != 0) cache.m_shader_waits.push_back({this, hash, m_started});
}

static const char* ComputeWrites(const ShaderModule& module) {
	const auto& program = module.info;
	if (program.has_address_writes ||
	    std::ranges::any_of(program.info.buffers, [](const auto& buffer) { return buffer.written; })) {
		return "writes-buffers";
	}
	if (std::ranges::any_of(program.info.images, [](const auto& image) { return image.written; })) {
		return "writes-images";
	}
	return "writes-nothing";
}

PipelineCache::StallScope::StallScope(PipelineCache& cache, const char* what, const ShaderModule& module)
    : StallScope(cache, what, module.info.shader_hash, ComputeWrites(module)) {
	m_variants[0] = module.id;
	m_programs[0] = module.info.shader_hash;
	m_compute     = true;
}

void PipelineCache::StallScope::Stages(const std::array<const ShaderModule*, 4>& stages) {
	for (size_t i = 0; i < stages.size(); i++) {
		m_variants[i] = stages[i] != nullptr ? stages[i]->id : 0u;
		m_programs[i] = stages[i] != nullptr ? stages[i]->info.shader_hash : 0u;
	}
}

void PipelineCache::NoteGenericServe(GenericServe state, uint64_t program) {
	if (state != GenericServe::Served && state != GenericServe::PipelineMissing &&
	    t_draw_generic.state != GenericServe::Served) {
		return;
	}
	t_draw_generic = {state, program};
}

std::string PipelineCache::TakeAddedCause() {
	return std::exchange(t_added_cause, {});
}

void PipelineCache::NoteWaitCause(const ProgramKey& key, const Program& program, const ShaderModule& module) {
	auto cause = ProgramWaitCause(key, program, module);
	t_added_cause = fmt::format(" cause={} variant={:016x}", cause, module.id);
	std::scoped_lock lock(m_stall_mutex);
	if (module.plain != nullptr) m_wait_causes.insert_or_assign(module.plain->id, cause);
	m_wait_causes.insert_or_assign(module.id, std::move(cause));
}

std::string PipelineCache::ProgramWaitCause(const ProgramKey& key, const Program& program,
                                            const ShaderModule& module) const {
	ProgramWaitFacts facts {.compute = key.stage == ShaderType::Compute, .draw = t_draw_generic.state,
	                        .calls = program.call_source != nullptr};
	size_t closest = SIZE_MAX;
	for (const auto& other: program.modules) {
		if (&other == &module) continue;
		size_t distance = 0;
		auto   reasons  = SpecializationDifference(module.specialization, other.specialization, distance);
		if (!std::ranges::equal(other.function_code, module.function_code)) {
			reasons += reasons.empty() ? "function-code" : "+function-code";
			distance++;
		}
		if (distance < closest) {
			closest          = distance;
			facts.difference = std::move(reasons);
		}
	}
	facts.other_variant = closest != SIZE_MAX;
	if (!facts.compute) {
		auto generic_key = key;
		generic_key.static_state.push_back(GenericGraphicsKey);
		const auto generic     = m_program_cache.find(generic_key);
		facts.generic_prepared = generic != m_program_cache.end() && !generic->second.modules.empty();
		facts.generic_refusal  = StartupRefusal(key.hash, true);
	}
	facts.recipe_refusal = StartupRefusal(key.hash, false);
	std::set<std::string> keys;
	const auto has = [](const ProgramKey& candidate, uint32_t marker) {
		return std::ranges::find(candidate.static_state, marker) != candidate.static_state.end();
	};
	for (const auto& [other, candidate]: m_program_cache) {
		if (other.stage != key.stage || other.hash != key.hash || &candidate == &program || candidate.modules.empty() ||
		    has(other, GenericGraphicsKey)) {
			continue;
		}
		bool named = false;
		for (const auto [marker, name]: {std::pair {RuntimeBuffersKey, "rb"}, std::pair {RuntimeSamplersKey, "rs"}}) {
			if (has(other, marker) != has(key, marker)) {
				keys.insert(name);
				named = true;
			}
		}
		if (other.guest_traversal != key.guest_traversal) {
			keys.insert("guest-traversal");
		} else if (other.user_data_count != key.user_data_count) {
			keys.insert("user-data");
		} else if (!named) {
			keys.insert("static-interface");
		}
	}
	facts.other_keys = fmt::format("{}", fmt::join(keys, "+"));
	facts.known      = KnownProgram(key.hash);
	return DescribeProgramWait(facts);
}

std::string PipelineCache::StallCause(std::span<const uint64_t> variants, std::span<const uint64_t> programs,
                                      bool compute, const char* route) const {
	StallWaitFacts facts {.compute = compute, .draw = t_draw_generic.state, .route = route != nullptr ? route : ""};
	uint64_t       variant = compute || variants[3] == 0 ? variants[0] : variants[3];
	{
		std::scoped_lock lock(m_stall_mutex);
		for (const auto candidate: variants) {
			if (candidate == 0) continue;
			if (const auto found = m_wait_causes.find(candidate); found != m_wait_causes.end()) {
				facts.module_cause = found->second;
				variant            = candidate;
				break;
			}
		}
	}
	if (facts.module_cause.empty() && facts.draw == GenericServe::NoModule) {
		facts.generic_refusal = StartupRefusal(t_draw_generic.program != 0 ? t_draw_generic.program : programs[0], true);
	}
	return fmt::format(" cause={} variant={:016x}", DescribeStallWait(facts), variant);
}
#endif

PipelineCache::StallScope::~StallScope() {
#if defined(ANYPS5_KYTY)
	const auto now = std::chrono::steady_clock::now();
	uint64_t   ns  = 0;
	{
		std::scoped_lock lock(m_cache.m_stall_mutex);
		if (--m_cache.m_stall.threads == 0) {
			m_cache.m_stall.ended = now;
		}
		if (m_hash != 0) std::erase_if(m_cache.m_shader_waits, [this](const auto& wait) { return wait.scope == this; });
		if (m_what != nullptr) {
			ns           = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_started).count());
			auto& stats = m_cache.m_stall_stats;
			stats.count++;
			stats.total_ns += ns;
			stats.max_ns = std::max(stats.max_ns, ns);
			stats.over_100ms += ns >= 100'000'000u ? 1u : 0u;
			stats.over_500ms += ns >= 500'000'000u ? 1u : 0u;
		}
	}
	static const uint64_t log_ns = [] {
		const char* value = std::getenv("KYTY_PIPELINE_STALL_LOG_MS");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 100u) * 1'000'000u;
	}();
	if (m_what != nullptr && ns >= log_ns) {
		const char* route = PipelineRouteNote();
		const auto  cause = m_variants[0] != 0 ? m_cache.StallCause(m_variants, m_programs, m_compute, route)
		                                       : std::string {};
		PrecompileLog("[kyty-gpl] stall {:.3f} s {} 0x{:016x} {}{}{}{} @{}", static_cast<double>(ns) / 1e9, m_what,
		              m_hash, route != nullptr ? route : "", m_why != nullptr ? " why=" : "",
		              m_why != nullptr ? m_why : "", cause, now.time_since_epoch().count());
	}
#else
	std::scoped_lock lock(m_cache.m_stall_mutex);
	if (--m_cache.m_stall.threads == 0) {
		m_cache.m_stall.ended = std::chrono::steady_clock::now();
	}
#endif
}

#if defined(ANYPS5_KYTY)
void PipelineCache::LogStallStats(bool force) {
	const std::array<uint64_t, 3> speculative {m_speculative_failed.load(std::memory_order_relaxed),
	                                           m_speculative_refused.load(std::memory_order_relaxed),
	                                           m_speculative_skipped.load(std::memory_order_relaxed)};
	if (force || speculative != m_speculative_logged) {
		m_speculative_logged = speculative;
		PrecompileLog("[kyty-spec-compile] compiles ahead failed {} and refused before compiling {}, each kept for its "
		              "variant; {} later compiles ahead skipped as kept",
		              speculative[0], speculative[1], speculative[2]);
	}
	StallStats stats;
	{
		std::scoped_lock lock(m_stall_mutex);
		m_stall_stats.pipelines = m_pipelines_created.load(std::memory_order_relaxed);
		stats                   = m_stall_stats;
		if (!force && stats.count == m_stall_logged.count && stats.pipelines == m_stall_logged.pipelines) {
			return;
		}
		m_stall_logged = stats;
	}
	PrecompileLog("[kyty-gpl] stalls {} total {:.3f} s max {:.3f} s >=0.1s {} >=0.5s {}; driver pipelines {}",
	              stats.count, static_cast<double>(stats.total_ns) / 1e9, static_cast<double>(stats.max_ns) / 1e9,
	              stats.over_100ms, stats.over_500ms, stats.pipelines);
	if (!m_async_threads.empty()) {
		const auto waits = m_async_first_waits.load(std::memory_order_relaxed);
		PrecompileLog("[kyty-async] mode {}: first draws waited {} times for {:.3f} s, {} served by the worker; "
		              "probes {} for {:.3f} s, {} created at once; flips {}",
		              Config::AsyncPipelinesProbe() ? 2 : 1, waits,
		              static_cast<double>(m_async_first_wait_ns.load(std::memory_order_relaxed)) / 1e9,
		              m_async_first_ready.load(std::memory_order_relaxed),
		              m_async_probes.load(std::memory_order_relaxed),
		              static_cast<double>(m_async_probe_ns.load(std::memory_order_relaxed)) / 1e9,
		              m_async_created_now.load(std::memory_order_relaxed),
		              m_render != nullptr ? m_render->FlipNumber() : 0u);
	}
	if (m_libraries != nullptr) {
		m_libraries->LogStats();
	}
	if (!m_chain_threads.empty()) {
		const auto prepared = m_chain_prepared.load(std::memory_order_relaxed);
		const auto failed   = m_chain_failed.load(std::memory_order_relaxed);
		const auto runs     = prepared + failed;
		PrecompileLog("[kyty-gpl] chains queued {} dropped {} prepared {} failed {} modules compiled {} ({:.1f} ms avg)",
		              m_chain_queued.load(std::memory_order_relaxed), m_chain_dropped.load(std::memory_order_relaxed),
		              prepared, failed, m_chain_compiled.load(std::memory_order_relaxed),
		              runs != 0 ? static_cast<double>(m_chain_ns.load(std::memory_order_relaxed)) / 1e6 /
		                              static_cast<double>(runs)
		                        : 0.0);
	}
}
#endif

PipelineCache::PipelineStall PipelineCache::GetPipelineStall() const {
	std::scoped_lock lock(m_stall_mutex);
	auto stall = m_stall;
#if defined(ANYPS5_KYTY)
	if (!m_shader_waits.empty()) {
		stall.hash = m_shader_waits.front().hash;
		stall.hash_since = m_shader_waits.front().since;
	}
#endif
	return stall;
}

void PipelineCache::StartPrecompile() {
	m_precompile = std::jthread([this](const std::stop_token& stop) { Precompile(stop); });
}

void PipelineCache::Precompile(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("Thread_PipelinePrecompile");
	using Clock = std::chrono::steady_clock;

	const auto estimate = [this] {
		const auto backlog = GetManifestBacklog();
		Estimate   e;
		e.pending          = backlog.pipelines;
		e.finished         = backlog.pipelines_done;
		e.shaders          = backlog.modules;
		e.shaders_finished = backlog.modules_done;
		e.running          = backlog.running;
		e.modules_found     = backlog.modules_found;
		e.modules_built     = backlog.modules_built;
		e.pipelines_created = backlog.pipelines_created;
		return e;
	};

	Clock::time_point                      slow_until;
	bool                                   stalled    = false;
	Clock::time_point                      stall_since;
	Clock::time_point                      stall_logged;
#if defined(ANYPS5_KYTY)
	uint64_t                               stall_created = 0;
	uint64_t                               stall_modules = 0;
	uint64_t                               stall_waits   = 0;
#else
	uint64_t                               stall_base = 0;
#endif
	bool                                   shown       = false;
#if defined(ANYPS5_KYTY)
	uint64_t                               notice_created = 0;
#else
	uint64_t                               notice_base = 0;
#endif
	Clock::time_point                      notice_shown_at;
	bool                                   bar_shown = false;
	std::optional<Clock::time_point>       bar_work_since;
	Clock::time_point                      bar_last_work;
	uint64_t                               bar_base = 0;
	const auto seconds = [](Clock::duration duration) { return std::chrono::duration<double>(duration).count(); };

	std::mutex                  sleep_mutex;
	std::condition_variable_any sleeper;
	const auto                  sleep = [&](Clock::duration period) {
		std::unique_lock lock(sleep_mutex);
		sleeper.wait_for(lock, stop, period, [] { return false; });
	};
#if defined(ANYPS5_KYTY)
	Clock::time_point stats_logged = Clock::now();
	Clock::time_point shader_wait_logged;
#endif
	while (!stop.stop_requested()) {
		sleep(WatchPeriod);
#if defined(ANYPS5_KYTY)
		if (m_libraries != nullptr) {
			m_libraries->Tick();
		}
		if (Clock::now() - stats_logged >= LogPeriod) {
			stats_logged = Clock::now();
			LogStallStats(false);
		}
#endif
		const auto e     = estimate();
		const auto stall = GetPipelineStall();
		const auto now   = Clock::now();
		const auto left  = std::max(e.pending, stall.threads);
#if defined(ANYPS5_KYTY)
		if (stall.hash != 0 && stall.hash_since != shader_wait_logged && now - stall.hash_since >= StallShown) {
			shader_wait_logged = stall.hash_since;
			PrecompileLog("[kyty-gpl] waiting_shader shader={:016x} elapsed_s={:.1f} monotonic_ns={}",
			              stall.hash, seconds(now - stall.hash_since),
			              std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
		}
#endif
		const auto creations = GetDriverCreations(SlowSeconds);
		const bool waits_slow = stall.threads != 0 && now - stall.since >= std::chrono::duration<double>(SlowSeconds);
		const uint32_t slow = creations.slow;
		if (slow != 0 || waits_slow) {
			slow_until = now + NoticeLinger;
		}
		if (!stalled && stall.threads != 0 && now - stall.since >= StallShown) {
			stalled      = true;
			stall_since  = stall.since;
			stall_logged = now;
#if defined(ANYPS5_KYTY)
			stall_created = m_pipelines_created.load(std::memory_order_relaxed);
			stall_modules = m_chain_compiled.load(std::memory_order_relaxed);
			{
				std::scoped_lock lock(m_stall_mutex);
				stall_waits = m_stall_stats.count;
			}
#else
			stall_base = e.finished;
#endif
			PrecompileLog("Pipeline stall: the game has waited {:.1f} s for the pipelines the driver compiles "
			              "({} left)",
			              seconds(now - stall_since), left);
		} else if (stalled && stall.threads == 0 && now - stall.ended >= StallGap) {
			stalled = false;
#if defined(ANYPS5_KYTY)
			const uint64_t created = m_pipelines_created.load(std::memory_order_relaxed) - stall_created;
			const uint64_t modules = m_chain_compiled.load(std::memory_order_relaxed) - stall_modules;
			uint64_t       waits   = 0;
			{
				std::scoped_lock lock(m_stall_mutex);
				waits = m_stall_stats.count - stall_waits;
			}
			PrecompileLog("Pipeline stall: the game goes on after {:.1f} s ({} compiles finished meanwhile: {} "
			              "pipelines created by the driver, {} shader modules built by the chains; {} waits ended)",
			              seconds(stall.ended - stall_since), created + modules, created, modules, waits);
#else
			PrecompileLog("Pipeline stall: the game goes on after {:.1f} s ({} pipelines compiled meanwhile)",
			              seconds(stall.ended - stall_since), e.finished - stall_base);
#endif
		} else if (stalled && now - stall_logged >= LogPeriod) {
			stall_logged = now;
			PrecompileLog("Pipeline stall: {} pipelines left at {:.0f} s", left, seconds(now - stall_since));
		}
		if (m_manifest_warm) {
			const uint64_t finished = e.finished + e.shaders_finished;
			const uint32_t left     = e.pending + e.running + e.shaders;
			if (left != 0) {
				bar_last_work = now;
				if (!bar_shown && !bar_work_since) {
					bar_work_since = now;
					bar_base       = finished;
				}
			} else if (!bar_shown) {
				bar_work_since.reset();
			}
			if (!bar_shown && bar_work_since && now - *bar_work_since >= ManifestBarDelay) {
				bar_shown = true;
				PrecompileLog("Pipeline manifest bar: shown ({} pipelines left, {} shaders to translate)",
				              e.pending + e.running, e.shaders);
			} else if (bar_shown && now - bar_last_work >= ManifestBarLinger) {
				bar_shown = false;
				PrecompileLog("Pipeline manifest bar: hidden after {:.1f} s ({} shaders and pipelines done)",
				              seconds(now - *bar_work_since), finished - std::min(bar_base, finished));
				bar_work_since.reset();
			}
			if (bar_shown) {
				const auto done = static_cast<uint32_t>(finished - std::min(bar_base, finished));
				ShowManifestProgress({.done       = done,
				                      .total      = done + left,
				                      .pipelines  = e.pending + e.running,
				                      .shaders    = e.shaders,
				                      .cached     = e.modules_found,
				                      .translated = e.modules_built,
				                      .compiled   = e.pipelines_created});
			} else {
				HideManifestProgress();
			}
		}

		std::optional<PrecompileNoticeState> state;
		if (stalled || waits_slow) {
			state = PrecompileNoticeState::Waiting;
		} else if (now < slow_until) {
			state = PrecompileNoticeState::Compiling;
		}
		if (!state) {
			if (shown) {
				shown = false;
				HidePrecompileNotice();
				PrecompileLog("Pipeline notice: hidden after {:.1f} s, {:.1f} s after the last compilation over {:.1f} s",
				              seconds(now - notice_shown_at), seconds(now - (slow_until - NoticeLinger)), SlowSeconds);
			}
			continue;
		}
		if (!shown) {
			shown           = true;
			notice_shown_at = now;
#if defined(ANYPS5_KYTY)
			notice_created = m_pipelines_created.load(std::memory_order_relaxed);
#else
			notice_base = e.finished;
#endif
			PrecompileLog("Pipeline notice: shown, state {} ({} driver creations for {:.1f} s, {} pipelines pending, "
			              "{} game threads waiting)",
			              *state == PrecompileNoticeState::Waiting ? "waiting" : "compiling", creations.running,
			              creations.longest, e.pending, stall.threads);
		}
#if defined(ANYPS5_KYTY)
		const auto     done = static_cast<uint32_t>(m_pipelines_created.load(std::memory_order_relaxed) - notice_created);
		const uint32_t todo = std::max({left, e.pending + e.running, creations.running});
#else
		const auto     done = static_cast<uint32_t>(e.finished - notice_base);
		const uint32_t todo = left;
#endif
		PrecompileNotice notice {.visible          = true,
		                         .state            = *state,
		                         .done             = done,
		                         .total            = done + todo,
		                         .shaders          = *state == PrecompileNoticeState::Waiting ? 0u : e.shaders,
		                         .waited_pipelines = *state == PrecompileNoticeState::Compiling ? slow : stall.threads,
		                         .seconds_waited   = *state == PrecompileNoticeState::Compiling
		                                                 ? static_cast<uint32_t>(creations.longest)
		                                                 : static_cast<uint32_t>(seconds(now - (stall.threads != 0 ? stall.since : stall_since)))};
#if defined(ANYPS5_KYTY)
		if (stall.threads != 0 && stall.hash != 0) {
			notice.waiting_hash     = stall.hash;
			notice.waiting_seconds  = static_cast<uint32_t>(seconds(now - stall.hash_since));
		}
#endif
		ShowPrecompileNotice(notice);
	}
	HidePrecompileNotice();
	HideManifestProgress();
#if defined(ANYPS5_KYTY)
	LogStallStats(true);
#endif
}

}

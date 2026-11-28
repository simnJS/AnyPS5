#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <thread>
#include <unordered_map>

#if defined(__linux__)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#endif

namespace Libs::Graphics {

struct ManifestAdmissionSample {
	uint64_t total     = 0;
	uint64_t available = 0;
	uint64_t process   = 0;
	double   busy_cpus = 0.0;
	uint32_t cpus      = 0;
};

class ManifestAdmission {
public:
	using Clock = std::chrono::steady_clock;

	static constexpr uint64_t MiB               = uint64_t {1} << 20u;
	static constexpr uint64_t GiB               = uint64_t {1} << 30u;
	static constexpr uint64_t LightBytes        = 512 * MiB;
	static constexpr uint64_t MinLightBytes     = 64 * MiB;
	static constexpr uint64_t BaseBytes         = 256 * MiB;
	static constexpr uint64_t MinDriverBytes    = 256 * MiB;
	static constexpr uint64_t LoneMargin        = 2 * GiB;
	static constexpr double   FirstBytesPerWord = 6144.0;
	static constexpr double   MinBytesPerWord   = 768.0;
	static constexpr double   MaxBytesPerWord   = 65536.0;
	static constexpr double   SafetyFactor      = 1.25;
	static constexpr double   PriorWords        = 200'000.0;
	static constexpr double   PriorLights       = 20.0;
	static constexpr double   LearnDecay        = 0.95;
	static constexpr double   SpareCpus         = 1.0;
	static constexpr auto     LoneWait          = std::chrono::seconds(30);
	static constexpr auto     StarveWait        = std::chrono::seconds(60);
	static constexpr auto     ReservationLife   = std::chrono::seconds(2);

	enum class Refusal { None, Limit, Cpu, Memory, Closed };

	using ThreadBytes = std::optional<uint64_t> (*)(int thread);

	explicit ManifestAdmission(uint32_t limit = 1, ThreadBytes thread_bytes = FaultedThreadBytes)
	    : m_limit(std::max(1u, limit)), m_thread_bytes(thread_bytes) {}

	[[nodiscard]] static int CurrentThread() {
#if defined(__linux__)
		return static_cast<int>(::gettid());
#else
		return 0;
#endif
	}

	[[nodiscard]] static std::optional<uint64_t> FaultedThreadBytes(int thread) {
#if defined(__linux__)
		if (thread <= 0) return std::nullopt;
		char path[64];
		std::snprintf(path, sizeof(path), "/proc/self/task/%d/stat", thread);
		std::FILE* file = std::fopen(path, "re");
		if (file == nullptr) return std::nullopt;
		char       text[1024];
		const auto size = std::fread(text, 1, sizeof(text) - 1, file);
		std::fclose(file);
		text[size]       = '\0';
		const char* field = std::strrchr(text, ')');
		if (field == nullptr) return std::nullopt;
		for (int skipped = 0; skipped < 8; ++skipped) {
			field = std::strchr(field + 1, ' ');
			if (field == nullptr) return std::nullopt;
		}
		char*      end    = nullptr;
		const auto faults = std::strtoull(field + 1, &end, 10);
		if (end == field + 1) return std::nullopt;
		static const auto page = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
		return static_cast<uint64_t>(faults) * page;
#else
		static_cast<void>(thread);
		return std::nullopt;
#endif
	}

	void SetLimit(uint32_t limit) { m_limit = std::max(1u, limit); }
	void Close() { m_closed = true; }

	void Sample(Clock::time_point now, const ManifestAdmissionSample& sample) {
		m_sample = sample;
		if (!m_sampled || m_jobs.empty() || sample.process < m_baseline) m_baseline = sample.process;
		m_sampled = true;
		m_others  = std::max(0.0, sample.busy_cpus - static_cast<double>(m_jobs.size()));
		const auto usage     = Usage();
		const auto committed = Committed();
		if (committed != 0) {
			for (auto& job: m_jobs | std::views::values) {
				auto share = static_cast<uint64_t>(static_cast<double>(usage) * static_cast<double>(job.estimate) /
				                                   static_cast<double>(committed));
				if (const auto own = ThreadUse(job)) share = std::min(share, *own);
				job.observed = std::min(job.estimate, share);
				job.peak = std::max(job.peak, share);
				++job.samples;
			}
		}
		if (m_reservation && now - m_reservation->seen > ReservationLife) m_reservation.reset();
	}

	[[nodiscard]] std::optional<uint64_t> Admit(Clock::time_point now, size_t words, bool driver, std::thread::id caller,
	                                            Clock::time_point waiting_since, int thread = CurrentThread()) {
		const auto estimate = Estimate(words, driver);
		m_refusal           = Decide(now, estimate, caller, waiting_since);
		if (m_refusal != Refusal::None) return std::nullopt;
		if (m_reservation && m_reservation->caller == caller) m_reservation.reset();
		const auto ticket = ++m_next_ticket;
		Job        job {.estimate = estimate, .words = words, .driver = driver, .start = now, .thread = thread};
		if (m_thread_bytes != nullptr && thread > 0) job.thread_start = m_thread_bytes(thread);
		m_jobs.emplace(ticket, job);
		return ticket;
	}

	[[nodiscard]] std::optional<uint64_t> AdmitOptimization(Clock::time_point now, size_t words) {
		if (!m_closed && m_sampled && static_cast<int64_t>(Estimate(words, true)) > Room()) {
			m_refusal = Refusal::Memory;
			return std::nullopt;
		}
		return Admit(now, words, true, std::this_thread::get_id(), now);
	}

	void Finish(uint64_t ticket) {
		const auto found = m_jobs.find(ticket);
		if (found == m_jobs.end()) return;
		const auto job = found->second;
		m_jobs.erase(found);
		if (m_jobs.empty()) m_baseline = std::min(m_baseline, m_sample.process);
		(job.driver ? m_finished_driver : m_finished_light)++;
		if (job.samples == 0) return;
		if (!job.driver) {
			m_light_bytes  = m_light_bytes * LearnDecay + static_cast<double>(job.peak);
			m_light_weight = m_light_weight * LearnDecay + 1.0;
			return;
		}
		if (job.words == 0) return;
		const auto excess = static_cast<double>(job.peak > BaseBytes ? job.peak - BaseBytes : 0);
		m_learned_bytes   = m_learned_bytes * LearnDecay + excess;
		m_learned_words   = m_learned_words * LearnDecay + static_cast<double>(job.words);
	}

	[[nodiscard]] uint64_t Estimate(size_t words, bool driver) const {
		if (!driver) return LightEstimate();
		const auto bytes = static_cast<double>(BaseBytes) + BytesPerWord() * static_cast<double>(words);
		return std::clamp(static_cast<uint64_t>(bytes), MinDriverBytes, MaxDriverBytes());
	}

	[[nodiscard]] uint64_t LightEstimate() const {
		const auto bytes = SafetyFactor * m_light_bytes / m_light_weight;
		return std::clamp(static_cast<uint64_t>(bytes), MinLightBytes, LightBytes);
	}

	[[nodiscard]] double BytesPerWord() const {
		return std::clamp(SafetyFactor * m_learned_bytes / m_learned_words, MinBytesPerWord, MaxBytesPerWord);
	}

	[[nodiscard]] uint64_t MaxDriverBytes() const {
		const auto cap = m_sample.total != 0 ? m_sample.total / 4 : 16 * GiB;
		return std::max(MinDriverBytes, std::min(16 * GiB, cap));
	}

	[[nodiscard]] static uint64_t Reserve(uint64_t total) {
		return total != 0 ? std::clamp(total / 5, 4 * GiB, 12 * GiB) : 4 * GiB;
	}

	[[nodiscard]] uint64_t Reserve() const { return Reserve(m_sample.total); }

	[[nodiscard]] uint64_t Committed() const {
		uint64_t total = 0;
		for (const auto& job: m_jobs | std::views::values) total += job.estimate;
		return total;
	}

	[[nodiscard]] uint64_t Usage() const { return m_sample.process > m_baseline ? m_sample.process - m_baseline : 0; }

	[[nodiscard]] uint64_t Pending() const {
		uint64_t pending = 0;
		for (const auto& job: m_jobs | std::views::values) pending += job.estimate - job.observed;
		return pending;
	}

	[[nodiscard]] int64_t Room() const {
		return static_cast<int64_t>(m_sample.available) - static_cast<int64_t>(Reserve()) -
		       static_cast<int64_t>(Pending());
	}

	[[nodiscard]] uint32_t CpuCap() const {
		if (!m_sampled || m_sample.cpus == 0) return m_limit;
		const auto free = std::floor(static_cast<double>(m_sample.cpus) - m_others - SpareCpus);
		return std::clamp(static_cast<uint32_t>(std::max(1.0, free)), 1u, m_limit);
	}

	[[nodiscard]] double OtherBusyCpus() const { return m_others; }
	[[nodiscard]] uint32_t Running() const { return static_cast<uint32_t>(m_jobs.size()); }
	[[nodiscard]] uint32_t Limit() const { return m_limit; }
	[[nodiscard]] Refusal LastRefusal() const { return m_refusal; }
	[[nodiscard]] bool MemoryBlocked() const { return m_refusal == Refusal::Memory; }
	[[nodiscard]] const ManifestAdmissionSample& LastSample() const { return m_sample; }
	[[nodiscard]] std::optional<uint64_t> Reserved() const {
		return m_reservation ? std::optional<uint64_t> {m_reservation->bytes} : std::nullopt;
	}

	struct Finished {
		uint32_t driver = 0;
		uint32_t light  = 0;
	};
	Finished TakeFinished() {
		const Finished finished {.driver = m_finished_driver, .light = m_finished_light};
		m_finished_driver = 0;
		m_finished_light  = 0;
		return finished;
	}

private:
	struct Job {
		uint64_t                estimate = 0;
		size_t                  words    = 0;
		bool                    driver   = false;
		Clock::time_point       start;
		int                     thread = 0;
		std::optional<uint64_t> thread_start {};
		uint64_t                thread_taken = 0;
		uint64_t                observed     = 0;
		uint64_t                peak         = 0;
		uint32_t                samples      = 0;
	};

	std::optional<uint64_t> ThreadUse(Job& job) const {
		if (!job.thread_start || m_thread_bytes == nullptr) return std::nullopt;
		if (const auto now = m_thread_bytes(job.thread); now && *now > *job.thread_start) {
			job.thread_taken = std::max(job.thread_taken, *now - *job.thread_start);
		}
		return job.thread_taken;
	}

	struct Reservation {
		std::thread::id   caller;
		uint64_t          bytes = 0;
		Clock::time_point seen;
	};

	Refusal Decide(Clock::time_point now, uint64_t estimate, std::thread::id caller, Clock::time_point waiting_since) {
		if (m_closed) return Refusal::Closed;
		if (m_reservation && m_reservation->caller == caller) m_reservation->seen = now;
		if (m_jobs.size() >= m_limit) return Refusal::Limit;
		if (!m_sampled) return Refusal::None;
		if (m_jobs.size() >= CpuCap()) return Refusal::Cpu;
		const bool other_reserved = m_reservation && m_reservation->caller != caller;
		const auto reserved       = other_reserved ? static_cast<int64_t>(m_reservation->bytes) : 0;
		if (static_cast<int64_t>(estimate) + reserved <= Room()) return Refusal::None;
		const auto waited = now - waiting_since;
		if (m_jobs.empty() && !other_reserved && (estimate + LoneMargin <= m_sample.available ||
		                                       (waited >= LoneWait && estimate <= m_sample.available))) {
			return Refusal::None;
		}
		if (waited >= StarveWait && (!m_reservation || m_reservation->caller == caller)) {
			m_reservation = Reservation {.caller = caller, .bytes = estimate, .seen = now};
		}
		return Refusal::Memory;
	}

	uint32_t                               m_limit;
	ThreadBytes                            m_thread_bytes = nullptr;
	bool                                   m_closed  = false;
	bool                                   m_sampled = false;
	ManifestAdmissionSample                m_sample;
	uint64_t                               m_baseline = 0;
	double                                 m_others   = 0.0;
	std::unordered_map<uint64_t, Job>      m_jobs;
	uint64_t                               m_next_ticket = 0;
	std::optional<Reservation>             m_reservation;
	Refusal                                m_refusal       = Refusal::None;
	double                                 m_learned_bytes = FirstBytesPerWord * PriorWords;
	double                                 m_learned_words = PriorWords;
	double                                 m_light_bytes   = static_cast<double>(LightBytes) / SafetyFactor * PriorLights;
	double                                 m_light_weight  = PriorLights;
	uint32_t                               m_finished_driver = 0;
	uint32_t                               m_finished_light  = 0;
};

}

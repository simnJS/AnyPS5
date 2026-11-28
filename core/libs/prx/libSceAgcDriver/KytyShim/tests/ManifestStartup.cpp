#include "graphics/host_gpu/renderer/pipeline/manifestAdmission.h"
#include "graphics/host_gpu/renderer/pipeline/manifestStartup.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <unordered_map>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace std::chrono_literals;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "%s\n", message);
		std::exit(1);
	}
}

void ManifestDiscovery() {
	namespace fs = std::filesystem;
	const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto root = fs::temp_directory_path() / ("kyty_manifest_startup_" + std::to_string(stamp));
	Check(fs::create_directory(root), "new manifest fixture directory");
	const std::array folders {root / "packaged", root / "local"};
	for (const auto& folder: folders) Check(fs::create_directory(folder), "manifest fixture subdirectory");
	const auto write = [](const fs::path& path) {
		std::ofstream file(path);
		file << "# kyty pipeline manifest v1\n";
		file.close();
		Check(static_cast<bool>(file), "manifest fixture file");
	};

	const auto local_compute = folders[1] / "fixture.compute.kpm";
	Check(FindManifestFiles(folders, "fixture", "fixture").empty(), "no startup data with empty cache directories");
	write(local_compute);
	Check(FindManifestFiles(folders, "fixture", "fixture") == std::vector {local_compute},
	      "an unversioned compute-only cache counts as startup data");
	const auto packaged_compute = folders[0] / "fixture_1.compute.kpm";
	write(packaged_compute);
	Check(FindManifestFiles(folders, "fixture", "fixture_1") == std::vector {packaged_compute},
	      "a packaged compute-only cache is discovered once");
	const auto primary = folders[1] / "fixture_1.kpm";
	const auto shared = folders[1] / "fixture_1.shared.kpm";
	const auto versioned_compute = folders[1] / "fixture_1.compute.kpm";
	write(primary);
	write(shared);
	write(versioned_compute);
	const auto graphics = folders[0] / "fixture_1.graphics.kpm";
	write(graphics);
	Check(FindManifestFiles(folders, "fixture", "fixture_1") ==
	          std::vector {primary, shared, packaged_compute, graphics, versioned_compute},
	      "primary and shared manifests preserve priority and compute and graphics sidecars are loaded");
	fs::remove_all(root);
}

void ModuleScheduling() {
	const std::array<uint32_t, 4> ranks {0, 0, 1, 1};
	const std::array<size_t, 4> words {2048, 4000, 16, 8000};
	std::deque<uint32_t> jobs {0, 1, 2, 3};
	const auto next = [&](bool allow_large, bool prefer_large) {
		return NextManifestModule(jobs, ranks, words, allow_large, prefer_large);
	};
	Check(*next(true, true) == 1, "a large worker overlaps work within the first rank");
	Check(*next(true, false) == 0, "a small pool preserves normal priority");
	Check(*next(false, false) == 0, "a 2048-word input qualifies as small");
	jobs.pop_front();
	Check(*next(false, false) == 2, "a small worker can pass a large job");
	jobs.erase(next(false, false));
	Check(next(false, false) == jobs.end(), "only large workers can take the remaining jobs");
}

void VideoWait() {
	const auto start = ManifestStartupWait::Clock::time_point {};
	Check(ManifestStartupWait(start, 0s).ReleaseReason(start) == "timed out", "zero wait releases immediately");
	Check(ManifestStartupWait(start, 5s).ReleaseReason(start + 5s) == "timed out", "a short user limit wins over the stall timeout");
	ManifestStartupWait wait(start, 900s);
	Check(wait.ReleaseReason(start + 119s).empty(), "a running job has time to finish");
	Check(wait.ReleaseReason(start + 120s) == "stalled", "a stalled job releases video");
	for (auto time = start + 100s; time < start + 900s; time += 100s) {
		wait.last_progress = time;
		Check(wait.ReleaseReason(time + 99s).empty(), "ongoing progress extends the stall deadline");
	}
	Check(wait.ReleaseReason(start + 900s) == "timed out", "progress cannot extend the maximum wait");
}

void MemoryAdmission() {
	constexpr uint64_t GiB = ManifestAdmission::GiB;
	constexpr size_t   Words = 150'000;
	const auto start = ManifestAdmission::Clock::time_point {};
	const auto first = std::this_thread::get_id();
	std::thread second_thread([] {});
	const auto second = second_thread.get_id();
	second_thread.join();

	ManifestAdmission admission(28, nullptr);
	admission.Sample(start, {.total = 60 * GiB, .available = 40 * GiB, .process = 4 * GiB, .busy_cpus = 0.0, .cpus = 32});
	Check(admission.Reserve() == 12 * GiB, "a fifth of the memory stays free, at most 12 GiB");
	std::vector<uint64_t> running;
	while (const auto ticket = admission.Admit(start, Words, true, first, start)) running.push_back(*ticket);
	const auto fits = static_cast<size_t>((40 - 12) * GiB / admission.Estimate(Words, true));
	Check(running.size() == fits, "compilations are admitted while their estimates fit above the reserve");
	Check(admission.MemoryBlocked(), "the next compilation waits for memory");

	admission.Sample(start + 5s, {.total = 60 * GiB, .available = 30 * GiB, .process = 14 * GiB, .busy_cpus = 20.0, .cpus = 32});
	Check(!admission.Admit(start + 5s, Words, true, first, start), "memory taken by admitted compilations is not counted twice");

	admission.Sample(start + 10s, {.total = 60 * GiB, .available = 6 * GiB, .process = 20 * GiB, .busy_cpus = 20.0, .cpus = 32});
	for (auto time = start + 10s; time < start + 40s; time += 1s) {
		Check(!admission.Admit(time, 1, false, first, start + 10s), "no translation is admitted under the reserve");
		Check(!admission.Admit(time, Words, true, second, start + 10s), "no compilation is admitted under the reserve");
	}
	Check(admission.Running() == running.size(), "running compilations are never suspended");
	for (const auto ticket: running) admission.Finish(ticket);
	Check(admission.Running() == 0, "running compilations finish under memory pressure");
	running.clear();

	const auto low = start + 60s;
	admission.Sample(low, {.total = 60 * GiB, .available = 6 * GiB, .process = 4 * GiB, .busy_cpus = 0.0, .cpus = 32});
	const auto alone = admission.Admit(low, Words, true, first, low);
	Check(alone.has_value(), "under the reserve one compilation runs alone when it fits in what is available");
	Check(!admission.Admit(low, Words, true, second, low), "only one compilation runs under the reserve");
	admission.Finish(*alone);

	admission.Sample(low + 1s, {.total = 60 * GiB, .available = 2 * GiB, .process = 4 * GiB, .busy_cpus = 0.0, .cpus = 32});
	Check(!admission.Admit(low + 1s, Words, true, first, low + 1s), "nothing starts when even one compilation does not fit");
	Check(!admission.Admit(low + 30s, Words, true, first, low + 1s), "nothing starts for 30 s when even one does not fit");
	const auto lone = admission.Admit(low + 31s, Words, true, first, low + 1s);
	Check(lone.has_value(), "one compilation runs alone after 30 s so that work never stops");
	Check(!admission.Admit(low + 32s, Words, true, second, low + 1s), "and only one");
	admission.Finish(*lone);

	admission.Sample(low + 40s, {.total = 60 * GiB, .available = 30 * GiB, .process = 4 * GiB, .busy_cpus = 0.0, .cpus = 32});
	Check(admission.Admit(low + 40s, Words, true, second, low + 40s).has_value(), "work resumes when memory comes back");
	admission.Close();
	Check(!admission.Admit(low + 41s, 1, false, first, low + 41s) && admission.LastRefusal() == ManifestAdmission::Refusal::Closed,
	      "a closed admission takes no new work");
}

void CpuAdmission() {
	constexpr uint64_t GiB = ManifestAdmission::GiB;
	const auto now = ManifestAdmission::Clock::time_point {};
	const auto id = std::this_thread::get_id();
	ManifestAdmission admission(28, nullptr);
	admission.Sample(now, {.total = 60 * GiB, .available = 50 * GiB, .process = 1 * GiB, .busy_cpus = 0.0, .cpus = 32});
	Check(admission.CpuCap() == 28, "an idle machine runs as many compilations as the limit");
	Check(admission.Admit(now, 1, false, id, now) && admission.Admit(now, 1, false, id, now), "two translations start");
	admission.Sample(now + 1s, {.total = 60 * GiB, .available = 50 * GiB, .process = 1 * GiB, .busy_cpus = 30.0, .cpus = 32});
	Check(admission.CpuCap() == 3, "threads busy elsewhere lower the cap, one CPU stays spare");
	Check(admission.Admit(now + 1s, 1, false, id, now + 1s).has_value(), "the third compilation fits the free CPUs");
	Check(!admission.Admit(now + 1s, 1, false, id, now + 1s) && admission.LastRefusal() == ManifestAdmission::Refusal::Cpu,
	      "the fourth waits for a free CPU");
	admission.Sample(now + 2s, {.total = 60 * GiB, .available = 50 * GiB, .process = 1 * GiB, .busy_cpus = 34.0, .cpus = 32});
	Check(admission.CpuCap() == 1, "a saturated machine still keeps one compilation");
}

void LearnedEstimate() {
	constexpr uint64_t GiB = ManifestAdmission::GiB;
	const auto now = ManifestAdmission::Clock::time_point {};
	const auto id = std::this_thread::get_id();
	ManifestAdmission admission(4, nullptr);
	admission.Sample(now, {.total = 60 * GiB, .available = 50 * GiB, .process = 2 * GiB, .busy_cpus = 0.0, .cpus = 32});
	const auto before = admission.Estimate(100'000, true);
	Check(admission.Estimate(100'000, false) == ManifestAdmission::LightBytes, "a translation has a fixed estimate");
	Check(admission.Estimate(1, true) >= ManifestAdmission::MinDriverBytes && admission.Estimate(1, true) < ManifestAdmission::MinDriverBytes + 1 * ManifestAdmission::MiB, "a driver compilation needs at least its minimum estimate");
	Check(admission.Estimate(1'000'000'000, true) == 15 * GiB, "an estimate never exceeds a quarter of the memory");
	const auto ticket = admission.Admit(now, 100'000, true, id, now);
	Check(ticket.has_value(), "a compilation starts");
	admission.Sample(now + 10s, {.total = 60 * GiB, .available = 42 * GiB, .process = 10 * GiB, .busy_cpus = 1.0, .cpus = 32});
	admission.Finish(*ticket);
	Check(admission.Estimate(100'000, true) > before, "an observed peak raises the estimate");
	const auto quick = admission.Admit(now + 11s, 100'000, true, id, now + 11s);
	const auto learned = admission.Estimate(100'000, true);
	admission.Finish(*quick);
	Check(admission.Estimate(100'000, true) == learned, "a compilation shorter than a sample teaches nothing");
}

void LearnedEstimateFollowsMeasuredUse() {
	constexpr uint64_t GiB = ManifestAdmission::GiB;
	constexpr uint64_t MiB = ManifestAdmission::MiB;
	constexpr size_t   Words = 150'000;
	constexpr uint64_t Real = 512 * MiB;
	const auto start = ManifestAdmission::Clock::time_point {};
	const auto id = std::this_thread::get_id();
	ManifestAdmission admission(28, nullptr);
	const ManifestAdmissionSample idle {.total = 60 * GiB, .available = 48 * GiB, .process = 2 * GiB, .busy_cpus = 0.0, .cpus = 32};
	admission.Sample(start, idle);
	Check(admission.BytesPerWord() > 4096.0, "the first estimate starts cautious");

	auto now = start;
	for (int round = 0; round < 60; ++round) {
		std::vector<uint64_t> batch;
		for (int i = 0; i < 8; ++i) {
			const auto ticket = admission.Admit(now, Words, true, id, now);
			Check(ticket.has_value(), "a batch starts while the memory is free");
			batch.push_back(*ticket);
		}
		for (int tick = 1; tick <= 4; ++tick) {
			now += 250ms;
			admission.Sample(now, {.total = 60 * GiB, .available = 48 * GiB - 8 * Real, .process = 2 * GiB + 8 * Real * tick / 4,
			                       .busy_cpus = 8.0, .cpus = 32});
		}
		for (const auto ticket: batch) admission.Finish(ticket);
		now += 250ms;
		admission.Sample(now, idle);
	}

	const auto per_job = admission.Estimate(Words, true);
	Check(per_job < 2 * Real, "the estimate drops to about what a compilation really takes");
	Check(per_job >= Real, "the estimate does not go below what a compilation really takes");
	Check(admission.BytesPerWord() >= ManifestAdmission::MinBytesPerWord, "the learned size per word keeps its floor");
	Check(admission.BytesPerWord() < 4096.0, "the learned size per word is no longer held at 4 KiB");

	const ManifestAdmissionSample loaded {.total = 60 * GiB, .available = 34 * GiB, .process = 2 * GiB, .busy_cpus = 0.0, .cpus = 32};
	admission.Sample(now + 1s, loaded);
	size_t admitted = 0;
	while (admitted < 28 && admission.Admit(now + 1s, Words, true, id, now + 1s)) ++admitted;
	Check(admitted == 28, "the real memory lets all 28 compilations run");
}

void MemoryFloorStaysLow() {
	constexpr uint64_t GiB = ManifestAdmission::GiB;
	const auto start = ManifestAdmission::Clock::time_point {};
	const auto id = std::this_thread::get_id();
	ManifestAdmission admission(4, nullptr);
	auto now = start;
	admission.Sample(now, {.total = 60 * GiB, .available = 50 * GiB, .process = 2 * GiB, .busy_cpus = 0.0, .cpus = 32});
	for (int round = 0; round < 200; ++round) {
		const auto ticket = admission.Admit(now, 100'000, true, id, now);
		Check(ticket.has_value(), "a compilation starts");
		now += 500ms;
		admission.Sample(now, {.total = 60 * GiB, .available = 50 * GiB, .process = 2 * GiB, .busy_cpus = 1.0, .cpus = 32});
		admission.Finish(*ticket);
	}
	Check(admission.BytesPerWord() == ManifestAdmission::MinBytesPerWord, "a size per word never goes under its floor");
	Check(admission.Estimate(100'000, true) >= ManifestAdmission::BaseBytes + 100'000 * 768, "the estimate keeps the fixed base");
}

std::unordered_map<int, uint64_t> g_thread_bytes;

std::optional<uint64_t> SimulatedThreadBytes(int thread) {
	const auto found = g_thread_bytes.find(thread);
	if (found == g_thread_bytes.end()) return std::nullopt;
	return found->second;
}

struct GrowthRun {
	double   measured_per_word = 0.0;
	double   blind_per_word    = 0.0;
	uint64_t measured_estimate = 0;
};

GrowthRun RunWithGrowth(uint64_t game_growth_per_tick, uint64_t real_per_job, int first_thread) {
	constexpr uint64_t GiB   = ManifestAdmission::GiB;
	constexpr size_t   Words = 150'000;
	constexpr int      Jobs  = 4;
	const auto         id    = std::this_thread::get_id();
	ManifestAdmission  measured(16, SimulatedThreadBytes);
	ManifestAdmission  blind(16, nullptr);
	g_thread_bytes.clear();
	uint64_t process = 20 * GiB;
	auto     now     = ManifestAdmission::Clock::time_point {};
	const auto sample = [&](uint64_t taken) {
		const ManifestAdmissionSample values {.total = 44 * GiB, .available = 30 * GiB - taken, .process = process,
		                                      .busy_cpus = 4.0, .cpus = 32};
		measured.Sample(now, values);
		blind.Sample(now, values);
	};
	sample(0);
	for (int round = 0; round < 80; ++round) {
		std::vector<uint64_t> measured_jobs;
		std::vector<uint64_t> blind_jobs;
		for (int job = 0; job < Jobs; ++job) {
			const int thread = first_thread + job;
			if (first_thread > 0) g_thread_bytes.try_emplace(thread, 0);
			const auto a = measured.Admit(now, Words, true, id, now, thread);
			const auto b = blind.Admit(now, Words, true, id, now, thread);
			Check(a.has_value() && b.has_value(), "the compilations of a round start");
			measured_jobs.push_back(*a);
			blind_jobs.push_back(*b);
		}
		for (int tick = 1; tick <= 4; ++tick) {
			now += 250ms;
			process += game_growth_per_tick + Jobs * real_per_job / 4;
			for (auto& bytes: g_thread_bytes | std::views::values) bytes += real_per_job / 4;
			sample(Jobs * real_per_job * tick / 4);
		}
		for (const auto ticket: measured_jobs) measured.Finish(ticket);
		for (const auto ticket: blind_jobs) blind.Finish(ticket);
		now += 250ms;
		process -= Jobs * real_per_job;
		sample(0);
	}
	return {.measured_per_word = measured.BytesPerWord(), .blind_per_word = blind.BytesPerWord(),
	        .measured_estimate = measured.Estimate(Words, true)};
}

void GameGrowthIsNotLearned() {
	constexpr uint64_t GiB  = ManifestAdmission::GiB;
	constexpr uint64_t MiB  = ManifestAdmission::MiB;
	constexpr uint64_t Real = 512 * MiB;
	const auto run = RunWithGrowth(3 * GiB, Real, 101);
	Check(run.blind_per_word > 8192.0, "without a measure per thread the game's growth is learned as compilation memory");
	Check(run.measured_per_word < 4096.0, "the game's growth is not learned as compilation memory");
	Check(run.measured_estimate >= Real, "the estimate still covers what a compilation really takes");
	Check(run.measured_estimate < 2 * Real, "the estimate stays near what a compilation really takes");
}

void CompilationGrowthStillLearned() {
	constexpr uint64_t MiB  = ManifestAdmission::MiB;
	constexpr uint64_t Real = 768 * MiB;
	const auto run = RunWithGrowth(0, Real, 201);
	Check(std::abs(run.measured_per_word - run.blind_per_word) <= run.blind_per_word * 0.01,
	      "when only the compilations grow, the measure per thread learns what the process memory shows");
	Check(run.measured_estimate >= Real, "the learned estimate covers a compilation");
	const auto unmeasured = RunWithGrowth(0, Real, 0);
	Check(unmeasured.measured_per_word == unmeasured.blind_per_word,
	      "a job whose thread cannot be measured is learned from the process memory as before");
}

void LargeJobsDoNotStarve() {
	constexpr uint64_t GiB = ManifestAdmission::GiB;
	const auto start = ManifestAdmission::Clock::time_point {};
	const auto big = std::this_thread::get_id();
	std::thread other_thread([] {});
	const auto small = other_thread.get_id();
	other_thread.join();
	ManifestAdmission admission(28, nullptr);
	admission.Sample(start, {.total = 60 * GiB, .available = 20 * GiB, .process = 1 * GiB, .busy_cpus = 0.0, .cpus = 32});
	const auto running = admission.Admit(start, 1, true, small, start);
	Check(running.has_value(), "a small compilation runs");
	Check(!admission.Admit(start + 59s, 2'000'000, true, big, start), "a large compilation waits for room");
	Check(admission.Admit(start + 59s, 1, true, small, start + 59s).has_value(), "small work passes before the wait is long");
	Check(!admission.Admit(start + 60s, 2'000'000, true, big, start) && admission.Reserved().has_value(),
	      "after a minute the large compilation reserves its room");
	Check(!admission.Admit(start + 60s, 1, true, small, start + 60s), "small work no longer passes the reservation");
	admission.Sample(start + 63s, {.total = 60 * GiB, .available = 20 * GiB, .process = 1 * GiB, .busy_cpus = 0.0, .cpus = 32});
	Check(!admission.Reserved().has_value(), "a reservation its owner no longer renews expires");
}

void GenericRefusalSkipsPipelines() {
	const auto pending = [](int stages) {
		uint8_t missing = 0;
		for (int i = 0; i < stages; ++i) AddManifestStage(missing, ManifestStage::Pending);
		return missing;
	};

	auto both = pending(2);
	Check(!FinishManifestPipelineStages(both).queue, "a pipeline waits for its pending stages");
	auto change = SettleManifestStage(both, ManifestStage::Built);
	Check(!change.queue && change.skipped == 0 && ManifestPipelineUnresolved(both), "one built stage is not enough");
	change = SettleManifestStage(both, ManifestStage::Built);
	Check(change.queue && !ManifestPipelineUnresolved(both), "the last built stage queues the pipeline");

	auto generic = pending(2);
	change = SettleManifestStage(generic, ManifestStage::Refused);
	Check(!change.queue && change.skipped == 1 && !ManifestPipelineUnresolved(generic),
	      "a refused generic stage skips its pipeline without leaving it unresolved");
	change = SettleManifestStage(generic, ManifestStage::Built);
	Check(!change.queue && change.skipped == 0 && generic == ManifestPipelineSkipped,
	      "a skipped pipeline is not queued by its other stages");
	change = SettleManifestStage(generic, ManifestStage::Refused);
	Check(change.skipped == 0, "a skipped pipeline is counted once");
	change = SettleManifestStage(generic, ManifestStage::Failed);
	Check(change.skipped == -1 && ManifestPipelineUnresolved(generic),
	      "a real failure still leaves a skipped pipeline unresolved");

	auto failed = pending(1);
	change = SettleManifestStage(failed, ManifestStage::Failed);
	Check(!change.queue && change.skipped == 0 && ManifestPipelineUnresolved(failed),
	      "a failed stage leaves its pipeline unresolved");
	change = SettleManifestStage(failed, ManifestStage::Refused);
	Check(change.skipped == 0 && ManifestPipelineUnresolved(failed), "a refusal cannot hide an earlier failure");

	uint8_t late = 0;
	AddManifestStage(late, ManifestStage::Refused);
	AddManifestStage(late, ManifestStage::Pending);
	change = FinishManifestPipelineStages(late);
	Check(!change.queue && change.skipped == 1 && !ManifestPipelineUnresolved(late),
	      "a pipeline added after its generic stage was refused is skipped at once");
	Check(!SettleManifestStage(late, ManifestStage::Built).queue, "its pending stage does not queue it");

	uint8_t mixed = 0;
	AddManifestStage(mixed, ManifestStage::Failed);
	AddManifestStage(mixed, ManifestStage::Refused);
	change = FinishManifestPipelineStages(mixed);
	Check(change.skipped == 0 && ManifestPipelineUnresolved(mixed), "a failed stage wins over a refused one");

	uint8_t ready = 0;
	Check(FinishManifestPipelineStages(ready).queue, "a pipeline with every stage built is queued at once");
}

}

int main() {
	ManifestDiscovery();
	ModuleScheduling();
	VideoWait();
	MemoryAdmission();
	CpuAdmission();
	LearnedEstimate();
	LearnedEstimateFollowsMeasuredUse();
	MemoryFloorStaysLow();
	GameGrowthIsNotLearned();
	CompilationGrowthStillLearned();
	LargeJobsDoNotStarve();
	GenericRefusalSkipsPipelines();
	return 0;
}

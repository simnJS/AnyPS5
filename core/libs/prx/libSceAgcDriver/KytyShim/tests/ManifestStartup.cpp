#include "graphics/host_gpu/renderer/pipeline/manifestStartup.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>

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

void MemoryPause() {
	constexpr uint64_t GiB = uint64_t {1} << 30;
	const auto now = ManifestMemoryPause::Clock::time_point {};
	const auto first = std::this_thread::get_id();
	std::thread second_thread([] {});
	const auto second = second_thread.get_id();
	second_thread.join();
	for (const auto total: {16 * GiB, 32 * GiB}) {
		const auto low = std::min(12 * GiB, total / 2);
		const auto high = std::min(14 * GiB, total / 12 * 7);
		ManifestMemoryPause memory;
		memory.Sample(now, total, low);
		Check(memory.Admit(now, first), "work starts at the low threshold");
		memory.Sample(now, total, low - 1);
		Check(!memory.Admit(now + 29s, first), "work pauses below the low threshold");
		memory.Sample(now + 30s, total, high);
		Check(memory.Admit(now + 30s, first), "a persistent pause admits one worker");
		Check(!memory.Admit(now + 31s, second), "other workers wait under pressure");
		Check(memory.Admit(now + 31s, first), "a worker can finish its nested pipeline work");
		memory.Finished(first);
		Check(memory.Admit(now + 31s, second), "the next worker can make progress");
		memory.Sample(now + 32s, total, high + 1);
		Check(memory.Admit(now + 32s, first), "all workers resume above the high threshold");
	}
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
	MemoryPause();
	GenericRefusalSkipsPipelines();
	return 0;
}

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace Libs::Graphics {

inline std::vector<std::filesystem::path> FindManifestFiles(std::span<const std::filesystem::path> folders,
                                                            const std::string& title, const std::string& stem) {
	std::vector<std::filesystem::path> files;
	if (folders.empty()) return files;

	const auto append = [&](const std::filesystem::path& path) {
		std::error_code error;
		if (std::filesystem::is_regular_file(path, error) && std::ranges::find(files, path) == files.end()) {
			files.push_back(path);
		}
	};
	for (const auto& folder: folders) {
		append(folder / (stem + ".kpm"));
	}
	if (files.empty()) {
		const auto prefix = std::filesystem::path(title + "_");
		for (const auto& folder: folders) {
			std::error_code error;
			for (const auto& entry: std::filesystem::directory_iterator(folder, error)) {
				const auto path = entry.path();
				if (path.filename().native().starts_with(prefix.native()) && path.extension() == ".kpm") {
					append(path);
				}
			}
		}
	}
	append(folders.back() / (stem + ".shared.kpm"));
	for (const auto& folder: folders) {
		append(folder / (stem + ".compute.kpm"));
		append(folder / (stem + ".graphics.kpm"));
	}
	return files;
}

inline auto NextManifestModule(std::deque<uint32_t>& jobs, std::span<const uint32_t> ranks,
                               std::span<const std::size_t> guest_words, bool allow_large, bool prefer_large) {
	constexpr std::size_t SmallGuestWords = 2048;
	const auto small = [&](uint32_t job) { return guest_words[job] <= SmallGuestWords; };
	if (!allow_large) return std::ranges::find_if(jobs, small);
	if (prefer_large && !jobs.empty()) {
		const auto rank = [&](uint32_t job) { return ranks[job]; };
		const auto last = std::prev(std::ranges::upper_bound(jobs, rank(jobs.front()), {}, rank));
		if (!small(*last)) return last;
	}
	return jobs.begin();
}

struct ManifestStartupWait {
	using Clock = std::chrono::steady_clock;

	ManifestStartupWait(Clock::time_point now, std::chrono::seconds maximum)
	    : limit(now + maximum), last_progress(now) {}

	[[nodiscard]] Clock::time_point Deadline() const {
		return std::min(limit, last_progress + std::chrono::seconds(120));
	}

	[[nodiscard]] std::string_view ReleaseReason(Clock::time_point now) const {
		if (now >= limit) return "timed out";
		if (now >= Deadline()) return "stalled";
		return {};
	}

	Clock::time_point limit;
	Clock::time_point last_progress;
};

struct ManifestMemoryPause {
	using Clock = std::chrono::steady_clock;

	std::optional<Clock::time_point> since;
	std::thread::id worker;

	void Sample(Clock::time_point now, uint64_t total, uint64_t available) {
		constexpr uint64_t GiB = uint64_t {1} << 30;
		if (!since && available < std::min(12 * GiB, total / 2)) {
			since = now;
		} else if (since && available > std::min(14 * GiB, total / 12 * 7)) {
			since.reset();
			worker = {};
		}
	}

	bool Admit(Clock::time_point now, std::thread::id id) {
		if (!since) return true;
		if (now - *since >= std::chrono::seconds(30) && worker == std::thread::id {}) worker = id;
		return worker == id;
	}

	void Finished(std::thread::id id) {
		if (worker == id) worker = {};
	}
};

}

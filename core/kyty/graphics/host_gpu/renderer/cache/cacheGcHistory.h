#ifndef KYTY_GRAPHICS_CACHE_GC_HISTORY_H_
#define KYTY_GRAPHICS_CACHE_GC_HISTORY_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace Libs::Graphics {

template <size_t MaxAge>
class CacheGcHistory {
public:
	using Clock = std::chrono::steady_clock;

	void FinishPass(uint64_t pass, Clock::time_point now) {
		m_finished[pass % MaxAge] = now;
	}

	template <size_t Age>
	[[nodiscard]] bool OldEnough(uint64_t pass, uint64_t last_use) const {
		static_assert(Age > 0 && Age <= MaxAge);
		if (last_use > pass) {
			return false;
		}
		const auto idle = pass - last_use;
		if (idle >= Age) {
			return true;
		}
		constexpr uint64_t MinIdlePasses = 3;
		const auto elapsed = m_finished[pass % MaxAge] - m_finished[last_use % MaxAge];
		return idle >= MinIdlePasses && elapsed >= std::chrono::milliseconds(16 * Age);
	}

	template <size_t Age>
	[[nodiscard]] std::optional<uint64_t> Cutoff(uint64_t pass) const {
		uint64_t first = pass >= Age ? pass - Age + 1 : 0;
		uint64_t end = pass;
		while (first < end) {
			const auto middle = first + (end - first) / 2;
			if (OldEnough<Age>(pass, middle)) {
				first = middle + 1;
			} else {
				end = middle;
			}
		}
		return first == 0 ? std::nullopt : std::optional {first - 1};
	}

private:
	std::array<Clock::time_point, MaxAge> m_finished {};
};

}

#endif

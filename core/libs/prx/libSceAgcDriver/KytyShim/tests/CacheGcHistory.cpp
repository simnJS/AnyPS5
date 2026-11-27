#include "graphics/host_gpu/renderer/cache/cacheGcHistory.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <optional>

int main() {
	using namespace std::chrono_literals;
	using History = Libs::Graphics::CacheGcHistory<16>;
	History history;
	std::array<History::Clock::time_point, 96> ends {};
	auto now = History::Clock::time_point {};
	for (size_t pass = 0; pass < ends.size(); ++pass) {
		auto interval = pass % 32 < 16 ? 8ms : 16ms;
		if (pass % 32 == 0) interval = 2857ms;
		now += interval;
		ends[pass] = now;
		history.FinishPass(pass, now);
		std::optional<uint64_t> expected;
		for (size_t last = 0; last <= pass; ++last) {
			if (pass - last >= 16 || (pass - last >= 3 && now - ends[last] >= 256ms)) {
				expected = last;
			}
		}
		if (history.Cutoff<16>(pass) != expected) {
			std::fprintf(stderr, "incorrect idle cutoff on pass %zu\n", pass);
			return 1;
		}
	}
	return 0;
}

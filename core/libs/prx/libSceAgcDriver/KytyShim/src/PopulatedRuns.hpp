#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::LibKernel::Memory::PopulatedRuns {

constexpr uint64_t None = UINT64_MAX;

template <typename Seek>
bool Append(Seek&& seek, uint64_t page, uint64_t join, uint64_t offset, uint64_t end, uint64_t cursor, uint64_t stop,
            std::vector<std::pair<uint64_t, uint64_t>>& runs) {
	const auto mask = page - 1u;
	const auto step = std::max(page, (join / 2u) & ~mask);
	const auto add  = [&](uint64_t first, uint64_t last) {
		const auto begin  = std::max(offset, first & ~mask);
		const auto finish = std::min(end, (last + mask) & ~mask);
		if (!runs.empty() && offset + runs.back().first + runs.back().second + join >= begin) {
			runs.back().second = std::max(runs.back().second, finish - offset - runs.back().first);
		} else {
			runs.emplace_back(begin - offset, finish - begin);
		}
	};
	const auto empty_from = [&](uint64_t at, uint64_t bound, bool& empty) {
		uint64_t found = None;
		if (!seek(at, found)) return false;
		empty = found >= bound;
		return true;
	};
	uint64_t data = None;
	if (!seek(cursor, data)) return false;
	while (data < stop) {
		const auto first = data;
		auto       held  = data & ~mask;
		for (;;) {
			const auto probe = held + step;
			auto       next  = None;
			if (probe < stop) {
				if (!seek(probe, next)) return false;
				if (next < probe + page) {
					held = probe;
					continue;
				}
				if (next < stop && held + page + join >= (next & ~mask)) {
					held = next & ~mask;
					continue;
				}
			}
			const auto bound = std::min(probe, stop);
			auto       low   = held;
			auto       high  = (bound + mask) & ~mask;
			if (high - low > page) {
				const auto last  = high - page;
				bool       empty = false;
				if (!empty_from(last, bound, empty)) return false;
				if (empty) {
					high = last;
				} else {
					low = last;
				}
			}
			while (high - low > page) {
				const auto middle = low + ((high - low) / page / 2u) * page;
				bool       empty  = false;
				if (!empty_from(middle, bound, empty)) return false;
				if (empty) {
					high = middle;
				} else {
					low = middle;
				}
			}
			if (next < stop && high + join >= (next & ~mask)) {
				held = next & ~mask;
				continue;
			}
			add(first, high);
			data = next;
			break;
		}
	}
	return true;
}

}

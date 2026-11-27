#include "graphics/host_gpu/renderer/pipeline/tableWatch.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics;

int         g_failures = 0;
const char* g_name     = "kyty_shim_table_watch";

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "%s: FAILED %s\n", g_name, what.c_str());
	g_failures++;
}

void JournalOrder() {
	auto journal = std::make_unique<ImageChangeJournal>();
	const auto start = journal->Head();
	std::vector<uint32_t> seen;
	Check(journal->ForEach(start, start, [&](uint32_t index) { seen.push_back(index); }) && seen.empty(),
	      "journal: an empty range is known and empty");
	for (const uint32_t index: {7u, 3u, 7u, (1u << 22u) - 1u}) {
		journal->Push(index);
	}
	Check(journal->ForEach(start, journal->Head(), [&](uint32_t index) { seen.push_back(index); }),
	      "journal: published entries are known");
	Check(seen == std::vector<uint32_t> {7u, 3u, 7u, (1u << 22u) - 1u}, "journal: entries come back in order");
	const auto head = journal->Head();
	Check(!journal->ForEach(head, head + 1, [](uint32_t) {}), "journal: a reserved but unpublished place is unknown");
	Check(!journal->ForEach(head, start, [](uint32_t) {}), "journal: a cursor ahead of the range is unknown");
}

void JournalOverflow() {
	auto       journal = std::make_unique<ImageChangeJournal>();
	const auto start   = journal->Head();
	for (uint64_t i = 0; i < ImageChangeJournal::Capacity + 1; i++) {
		journal->Push(static_cast<uint32_t>(i & 0xffffu));
	}
	const auto head = journal->Head();
	Check(!journal->ForEach(start, head, [](uint32_t) {}), "journal: an overwritten range is unknown");
	uint64_t count = 0;
	Check(journal->ForEach(head - ImageChangeJournal::Capacity, head, [&](uint32_t) { count++; }),
	      "journal: the last capacity entries are known");
	Check(count == ImageChangeJournal::Capacity, "journal: the last capacity entries are all read");
}

void JournalThreads() {
	auto                     journal = std::make_unique<ImageChangeJournal>();
	constexpr uint32_t       Writers = 4;
	constexpr uint32_t       Pushes  = 12000;
	std::atomic<uint32_t>    done {0};
	std::vector<std::thread> writers;
	for (uint32_t writer = 0; writer < Writers; writer++) {
		writers.emplace_back([&, writer] {
			for (uint32_t i = 0; i < Pushes; i++) {
				journal->Push(writer);
			}
			done.fetch_add(1);
		});
	}
	std::vector<uint64_t> counts(Writers, 0);
	uint64_t              cursor  = journal->Head();
	const uint64_t        first   = cursor;
	uint64_t              retries = 0;
	for (;;) {
		const bool finished = done.load() == Writers;
		const auto head     = journal->Head();
		std::vector<uint32_t> batch;
		if (journal->ForEach(cursor, head, [&](uint32_t index) { batch.push_back(index); })) {
			for (const auto index: batch) {
				counts[index]++;
			}
			cursor = head;
			if (finished) break;
		} else {
			retries++;
		}
	}
	for (auto& writer: writers) {
		writer.join();
	}
	Check(cursor - first == uint64_t {Writers} * Pushes, "journal threads: every push is read once");
	for (uint32_t writer = 0; writer < Writers; writer++) {
		Check(counts[writer] == Pushes, "journal threads: writer " + std::to_string(writer) + " read " +
		                                    std::to_string(counts[writer]) + " times");
	}
	(void)retries;
}

struct Checked {
	bool     valid         = false;
	bool     null_image    = false;
	uint32_t image         = 0;
	uint32_t image_version = 0;
	uint32_t slot          = 0;
	uint32_t slot_version  = 0;
};

struct World {
	std::unique_ptr<ImageChangeJournal>  images   = std::make_unique<ImageChangeJournal>();
	std::unique_ptr<TableElementJournal> elements = std::make_unique<TableElementJournal>();
	std::vector<uint32_t>                image_versions;
	std::vector<uint32_t>                slot_versions;

	void BumpImage(uint32_t image) {
		images->Push(image);
		image_versions[image]++;
	}
	void BumpSlot(uint32_t slot) {
		elements->Push(slot);
		slot_versions[slot]++;
	}
};

struct Table {
	std::vector<Checked>  checked;
	std::vector<uint32_t> slots;
	std::vector<uint32_t> images;
	TableWatch            watch;
};

void Track(Table& table, uint32_t element) {
	const auto& checked = table.checked[element];
	table.watch.Track(element, checked.valid, !checked.null_image, checked.image, checked.slot,
	                  [&](uint32_t other, uint32_t slot) {
		                  return other < table.checked.size() && table.checked[other].valid &&
		                         table.checked[other].slot == slot;
	                  });
}

uint64_t g_rounds_fast = 0;
uint64_t g_rounds_full = 0;
uint64_t g_considered  = 0;
uint64_t g_elements    = 0;

void Simulate(uint32_t seed, uint32_t images, uint32_t elements, uint32_t rounds, bool overflows) {
	std::mt19937 random {seed};
	const auto   pick = [&](uint32_t count) { return static_cast<uint32_t>(random() % count); };
	World        world;
	world.image_versions.assign(images, 0);
	world.slot_versions.assign(elements * 2, 0);
	Table table;
	const auto add_element = [&] {
		table.checked.emplace_back();
		table.slots.push_back(static_cast<uint32_t>(table.slots.size()));
		table.images.push_back(pick(images));
	};
	for (uint32_t element = 0; element < elements; element++) {
		add_element();
	}
	const auto seen = [&](uint32_t element) {
		Checked checked {.valid = true, .null_image = table.images[element] == UINT32_MAX};
		checked.image         = checked.null_image ? 0 : table.images[element];
		checked.image_version = checked.null_image ? 0 : world.image_versions[checked.image];
		checked.slot          = table.slots[element];
		checked.slot_version  = world.slot_versions[checked.slot];
		return checked;
	};
	for (uint32_t round = 0; round < rounds; round++) {
		const auto changes = pick(6);
		for (uint32_t change = 0; change < changes; change++) {
			world.BumpImage(pick(images));
		}
		if (pick(4) == 0) {
			world.BumpSlot(table.slots[pick(static_cast<uint32_t>(table.slots.size()))]);
		}
		if (overflows && pick(64) == 0) {
			for (uint64_t i = 0; i <= ImageChangeJournal::Capacity; i++) {
				world.images->Push(pick(images));
			}
		}
		if (pick(50) == 0 && table.slots.size() < elements * 2) {
			add_element();
		}
		if (pick(80) == 0) {
			table.watch.Invalidate();
		}
		std::vector<uint32_t> bound;
		for (uint32_t i = pick(3); i > 0; i--) {
			bound.push_back(pick(images));
		}
		const auto count = static_cast<uint32_t>(table.checked.size());
		std::vector<uint8_t> marked(count, 0);
		for (uint32_t i = pick(3); i > 0; i--) {
			marked[pick(count)] = 1;
		}
		const auto unchanged = [&](uint32_t element) {
			const auto& checked = table.checked[element];
			if (!checked.valid || world.slot_versions[checked.slot] != checked.slot_version) return false;
			if (checked.null_image) return true;
			return std::ranges::find(bound, checked.image) == bound.end() &&
			       world.image_versions[checked.image] == checked.image_version;
		};
		std::vector<uint32_t> reference;
		for (uint32_t element = 0; element < count; element++) {
			if (marked[element] == 0 && !unchanged(element)) reference.push_back(element);
		}
		table.watch.Begin(count);
		const auto            image_head   = world.images->Head();
		const auto            element_head = world.elements->Head();
		std::vector<uint32_t> fast;
		uint64_t              considered = 0;
		const bool            known =
		    table.watch.Collect(*world.images, image_head, *world.elements, element_head, bound, [&](uint32_t element) {
			    if (!table.watch.Consider(element)) return;
			    considered++;
			    if (marked[element] == 0 && !unchanged(element)) fast.push_back(element);
		    });
		if (known) {
			std::ranges::sort(fast);
			Check(fast == reference, "simulation seed " + std::to_string(seed) + " round " + std::to_string(round) +
			                             ": " + std::to_string(fast.size()) + " checks, " +
			                             std::to_string(reference.size()) + " expected");
			g_rounds_fast++;
			g_considered += considered;
		} else {
			fast = reference;
			g_rounds_full++;
			g_considered += count;
		}
		g_elements += count;
		std::vector<uint32_t> visits;
		for (uint32_t element = 0; element < count; element++) {
			if (marked[element] != 0) visits.push_back(element);
		}
		for (const auto element: fast) {
			if (pick(5) == 0) {
				table.checked[element] = {};
				visits.push_back(element);
				continue;
			}
			if (pick(4) == 0) {
				table.images[element] = pick(8) == 0 ? UINT32_MAX : pick(images);
			}
			table.checked[element] = seen(element);
		}
		for (const auto element: visits) {
			table.checked[element] = {};
		}
		table.watch.BeginTracking(!known);
		if (!known) {
			for (uint32_t element = 0; element < count; element++) Track(table, element);
		} else {
			for (const auto element: fast) Track(table, element);
			for (const auto element: visits) Track(table, element);
		}
		table.watch.Finish(image_head, element_head);
	}
}

}

extern "C" int RunKytyTableWatchTest(int, char**) {
	JournalOrder();
	JournalOverflow();
	JournalThreads();
	for (uint32_t seed = 1; seed <= 12; seed++) {
		Simulate(seed, 4 + seed * 7, 40 + seed * 50, 3000, seed % 3 == 0);
	}
	Simulate(97, 3, 600, 2000, false);
	Simulate(98, 5000, 3000, 1500, true);
	Check(g_rounds_fast > g_rounds_full * 10, "simulation: the journal path serves most rounds (" +
	                                              std::to_string(g_rounds_fast) + " fast, " +
	                                              std::to_string(g_rounds_full) + " full)");
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %d failure(s)\n", g_name, g_failures);
		return 1;
	}
	std::printf("%s: %llu journal rounds, %llu full scans, %llu of %llu elements considered\n", g_name,
	            static_cast<unsigned long long>(g_rounds_fast), static_cast<unsigned long long>(g_rounds_full),
	            static_cast<unsigned long long>(g_considered), static_cast<unsigned long long>(g_elements));
	std::printf("%s: ok\n", g_name);
	return 0;
}

#include "graphics/host_gpu/renderer/cache/readbackPublication.h"

#include <array>
#include <atomic>
#include <barrier>
#include <thread>
#include <utility>

namespace {

using Libs::Graphics::ReadbackPublication;

int PublishedValue() {
	constexpr uint64_t page = 0x4000;
	ReadbackPublication publication;
	auto first = publication.Begin(page, page + 8192);
	if (publication.Find(page - 1) || publication.Find(page + 8192) || publication.Find(page + 4096) != first ||
	    publication.Find(page) != first) return 1;

	std::atomic<bool> acquired {false};
	int backing = 0;
	int observed = 0;
	std::thread reader([&] {
		auto request = publication.Find(page);
		acquired.store(true, std::memory_order_release);
		acquired.notify_one();
		ReadbackPublication::Wait(request);
		observed = backing;
	});
	acquired.wait(false, std::memory_order_acquire);
	auto second = publication.Begin(page, page + 8192);
	backing = 42;
	publication.Finish(first);
	reader.join();
	if (observed != 42 || publication.Find(page) != second || second->released.load()) return 2;

	publication.Finish(second);
	ReadbackPublication::Wait(second);
	return publication.Find(page) ? 3 : 0;
}

bool ConcurrentSlotReuse() {
	constexpr uint64_t page = 0x8000;
	constexpr unsigned rounds = 128;
	std::array<std::thread, 4> readers;
	std::barrier phase(static_cast<int>(readers.size()) + 1);
	ReadbackPublication publication;
	std::atomic<bool> valid {true};
	for (auto& reader: readers) {
		reader = std::thread([&] {
			for (unsigned i = 0; i < rounds; ++i) {
				phase.arrive_and_wait();
				if (auto request = publication.Find(page)) {
					ReadbackPublication::Wait(request);
					if (!request->copied.load(std::memory_order_relaxed)) valid.store(false);
				}
				phase.arrive_and_wait();
			}
		});
	}
	for (unsigned i = 0; i < rounds; ++i) {
		auto first = publication.Begin(page, page + 4096);
		phase.arrive_and_wait();
		std::array pending {first, publication.Begin(page, page + 4096)};
		if (i % 2) std::swap(pending[0], pending[1]);
		for (const auto& request: pending) {
			request->copied.store(true, std::memory_order_relaxed);
			publication.Finish(request);
		}
		phase.arrive_and_wait();
	}
	for (auto& reader: readers) reader.join();
	return valid.load() && !publication.Find(page);
}

}

int main() {
	if (const auto failure = PublishedValue()) return failure;
	return ConcurrentSlotReuse() ? 0 : 4;
}

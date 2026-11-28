#include "graphics/host_gpu/renderer/cache/readbackPublication.h"

#include <atomic>
#include <thread>

int main() {
	using Libs::Graphics::ReadbackPublication;
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

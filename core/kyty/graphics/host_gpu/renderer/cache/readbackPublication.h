#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_READBACKPUBLICATION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_READBACKPUBLICATION_H_

#include <atomic>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

class ReadbackPublication {
public:
	struct Request {
		Request(uint64_t begin, uint64_t end): begin(begin), end(end) {}

		const uint64_t    begin, end;
		std::atomic<bool> copied {false};
		std::atomic<bool> released {false};
	};

	using Ticket = std::shared_ptr<Request>;

	Ticket Begin(uint64_t begin, uint64_t end) {
		auto request = std::make_shared<Request>(begin, end);
		m_current.store(request, std::memory_order_release);
		return request;
	}

	[[nodiscard]] Ticket Find(uint64_t page) const {
		auto request = m_current.load(std::memory_order_acquire);
		return request && request->begin <= page && page < request->end ? request : nullptr;
	}

	static void Wait(const Ticket& request) {
		request->released.wait(false, std::memory_order_acquire);
	}

	void Finish(const Ticket& request) {
		auto expected = request;
		m_current.compare_exchange_strong(expected, nullptr);
		request->released.store(true, std::memory_order_release);
		request->released.notify_all();
	}

private:
	std::atomic<Ticket> m_current;
};

}

#endif

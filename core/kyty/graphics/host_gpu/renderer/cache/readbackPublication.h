#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_READBACKPUBLICATION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_READBACKPUBLICATION_H_

#include <atomic>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

class ReadbackPublication {
public:
	struct Request {
		explicit Request(uint64_t address): page(address) {}

		const uint64_t    page;
		std::atomic<bool> released {false};
	};

	using Ticket = std::shared_ptr<Request>;

	Ticket Begin(uint64_t page) {
		auto request = std::make_shared<Request>(page);
		m_current.store(request, std::memory_order_release);
		return request;
	}

	[[nodiscard]] Ticket Find(uint64_t page) const {
		auto request = m_current.load(std::memory_order_acquire);
		return request && request->page == page ? request : nullptr;
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

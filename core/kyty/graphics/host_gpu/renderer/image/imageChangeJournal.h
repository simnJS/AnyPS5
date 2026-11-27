#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGECHANGEJOURNAL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGECHANGEJOURNAL_H_

#include <array>
#include <atomic>
#include <cstdint>

namespace Libs::Graphics {

template <uint32_t CapacityBits>
class ChangeJournal final {
public:
	static constexpr uint64_t Capacity  = uint64_t {1} << CapacityBits;
	static constexpr uint32_t IndexBits = 24;
	static constexpr uint64_t IndexMask = (uint64_t {1} << IndexBits) - 1;
	static constexpr uint64_t SeqMask   = (uint64_t {1} << (64 - IndexBits)) - 1;

	void Push(uint32_t index) noexcept {
		const auto seq = m_head.fetch_add(1, std::memory_order_acq_rel);
		m_entries[seq & (Capacity - 1)].store(((seq & SeqMask) << IndexBits) | (index & IndexMask),
		                                      std::memory_order_release);
	}

	[[nodiscard]] uint64_t Head() const noexcept { return m_head.load(std::memory_order_acquire); }

	template <typename Fn>
	[[nodiscard]] bool ForEach(uint64_t from, uint64_t to, Fn&& fn) const {
		if (from > to || to - from > Capacity) {
			return false;
		}
		for (auto seq = from; seq < to; ++seq) {
			const auto entry = m_entries[seq & (Capacity - 1)].load(std::memory_order_acquire);
			if ((entry >> IndexBits) != (seq & SeqMask)) {
				return false;
			}
			fn(static_cast<uint32_t>(entry & IndexMask));
		}
		return true;
	}

private:
	std::atomic<uint64_t>                       m_head {1};
	std::array<std::atomic<uint64_t>, Capacity> m_entries {};
};

using ImageChangeJournal = ChangeJournal<16>;

}

#endif

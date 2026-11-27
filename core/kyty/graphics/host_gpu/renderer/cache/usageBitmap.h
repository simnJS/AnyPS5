#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_USAGEBITMAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_USAGEBITMAP_H_

#include <bit>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

class UsageBitmap {
public:
	static constexpr uint64_t ChunkItems = 16384;

	UsageBitmap() = default;
	explicit UsageBitmap(uint64_t items) { Resize(items); }

	void Resize(uint64_t items) {
		m_items = items;
		m_bits.assign((items + 63u) / 64u, 0u);
		m_dirty.clear();
		m_stamps.clear();
		m_stamps.resize((items + ChunkItems - 1u) / ChunkItems);
	}

	void Mark(uint64_t index) noexcept { MarkBits(index / 64u, uint64_t {1} << (index % 64u)); }

	void Mark(uint64_t first, uint64_t last) noexcept {
		for (auto word = first / 64u; word <= last / 64u; word++) {
			const auto low  = word == first / 64u ? first % 64u : 0u;
			const auto high = word == last / 64u ? last % 64u : 63u;
			MarkBits(word, (high == 63u ? ~uint64_t {0} : (uint64_t {2} << high) - 1u) & ~((uint64_t {1} << low) - 1u));
		}
	}

	void MarkBits(uint64_t word, uint64_t bits) noexcept {
		if (bits == 0u || word >= m_bits.size()) return;
		if (m_bits[word] == 0u) m_dirty.push_back(word);
		m_bits[word] |= bits;
	}

	void MarkWords(std::span<const uint32_t> words) noexcept {
		for (size_t word = 0; word < words.size(); word += 2u) {
			const uint64_t high = word + 1u < words.size() ? words[word + 1u] : 0u;
			MarkBits(word / 2u, uint64_t {words[word]} | high << 32u);
		}
	}

	template <typename Fn>
	void Fold(uint32_t stamp, Fn&& marked) {
		for (const auto word: m_dirty) {
			for (auto bits = m_bits[word]; bits != 0u; bits &= bits - 1u) {
				const auto index = word * 64u + static_cast<uint64_t>(std::countr_zero(bits));
				if (index >= m_items) break;
				auto& chunk = m_stamps[index / ChunkItems];
				if (!chunk) chunk = std::make_unique<uint32_t[]>(ChunkItems);
				chunk[index % ChunkItems] = stamp;
				marked(index);
			}
			m_bits[word] = 0u;
		}
		m_dirty.clear();
	}

	void Fold(uint32_t stamp) {
		Fold(stamp, [](uint64_t) {});
	}

	[[nodiscard]] uint64_t PendingWord(uint64_t word) const noexcept { return m_bits[word]; }
	[[nodiscard]] bool     Pending(uint64_t index) const noexcept { return (m_bits[index / 64u] >> (index % 64u) & 1u) != 0u; }

	[[nodiscard]] uint32_t Stamp(uint64_t index) const noexcept {
		const auto& chunk = m_stamps[index / ChunkItems];
		return chunk ? chunk[index % ChunkItems] : 0u;
	}

	[[nodiscard]] uint64_t Items() const noexcept { return m_items; }
	[[nodiscard]] uint64_t Words() const noexcept { return m_bits.size(); }

private:
	uint64_t                                 m_items = 0;
	std::vector<uint64_t>                    m_bits;
	std::vector<uint64_t>                    m_dirty;
	std::vector<std::unique_ptr<uint32_t[]>> m_stamps;
};

}

#endif

#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAPSLOTS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAPSLOTS_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics::ImageHeapSlots {

inline constexpr uint32_t KeyWords     = 8u;
inline constexpr uint32_t EmptyProfile = 0xffffffffu;
inline constexpr uint32_t BucketBits   = 18u;
inline constexpr uint32_t BucketCount  = 1u << BucketBits;
inline constexpr uint32_t BucketMask   = BucketCount - 1u;
inline constexpr uint32_t MaxDistance  = 64u;
inline constexpr uint32_t MaxArms      = 8u;
inline constexpr uint32_t KindShift    = 16u;
inline constexpr uint32_t IndexMask    = (1u << KindShift) - 1u;
inline constexpr uint32_t ArmShift     = 20u;
inline constexpr uint32_t GroupProbes  = 64u;
inline constexpr uint32_t MaxGroups    = 65535u;
inline constexpr uint32_t RingCapacity = 4096u;
inline constexpr uint32_t NegativeFlag = 1u;
inline constexpr uint32_t DeadFlag     = 2u;

using KeyWordArray = std::array<uint32_t, KeyWords>;

struct Bucket {
	KeyWordArray words {};
	uint32_t     profile = EmptyProfile;
	uint32_t     value   = 0;
	uint32_t     flags   = 0;
	uint32_t     pad     = 0;
};
static_assert(sizeof(Bucket) == 48u);

struct Job {
	uint64_t                      table       = 0;
	uint64_t                      slots       = 0;
	uint64_t                      guest       = 0;
	uint32_t                      table_words = 0;
	uint32_t                      first_word  = 0;
	uint32_t                      step_words  = 0;
	uint32_t                      count       = 0;
	uint32_t                      begin       = 0;
	uint32_t                      end         = 0;
	uint32_t                      first_group = 0;
	uint32_t                      profile     = 0;
	uint32_t                      key_words   = 0;
	uint32_t                      view        = 0;
	uint32_t                      arm_count   = 0;
	uint32_t                      shape       = 0;
	std::array<uint32_t, MaxArms> kinds {};
};
static_assert(sizeof(Job) == 104u);

struct Miss {
	KeyWordArray words {};
	uint32_t     profile = 0;
	uint32_t     view    = 0;
	uint32_t     slot    = 0;
	uint32_t     shape   = 0;
	uint64_t     guest   = 0;
	uint64_t     pad     = 0;
};
static_assert(sizeof(Miss) == 64u);

inline constexpr uint64_t RingHeaderBytes = 16u;
inline constexpr uint64_t RingBytes       = RingHeaderBytes + uint64_t {RingCapacity} * sizeof(Miss);

struct Push {
	uint64_t jobs          = 0;
	uint64_t groups        = 0;
	uint64_t buckets       = 0;
	uint64_t ring          = 0;
	uint64_t used          = 0;
	uint32_t group_offset  = 0;
	uint32_t frame         = 0;
	uint32_t ring_capacity = RingCapacity;
	uint32_t pad           = 0;
};
static_assert(sizeof(Push) == 56u);

constexpr uint32_t Mix(uint32_t value) {
	value ^= value >> 16u;
	value *= 0x7feb352du;
	value ^= value >> 15u;
	value *= 0x846ca68bu;
	value ^= value >> 16u;
	return value;
}

constexpr uint32_t KeyHash(const KeyWordArray& words, uint32_t profile) {
	uint32_t hash = Mix(profile ^ 0x9e3779b9u);
	for (const auto word: words) hash = Mix(hash ^ word);
	return hash;
}

inline KeyWordArray ReadKey(std::span<const uint32_t> table, uint64_t slot, uint32_t key_words) {
	KeyWordArray words {};
	for (uint32_t word = 0; word < key_words && slot + word < table.size(); ++word) {
		words[word] = table[slot + word];
	}
	return words;
}

class Directory {
public:
	Directory(): m_buckets(BucketCount), m_marked(BucketCount, uint8_t {0}) {}

	[[nodiscard]] const Bucket* Find(const KeyWordArray& words, uint32_t profile) const {
		const auto home = KeyHash(words, profile);
		for (uint32_t distance = 0; distance < MaxDistance; ++distance) {
			const auto& bucket = m_buckets[(home + distance) & BucketMask];
			if (bucket.profile == EmptyProfile) return nullptr;
			if (bucket.profile == profile && bucket.words == words) return &bucket;
		}
		return nullptr;
	}

	bool Insert(const KeyWordArray& words, uint32_t profile, uint32_t value, uint32_t flags) {
		const auto home = KeyHash(words, profile);
		for (uint32_t distance = 0; distance < MaxDistance; ++distance) {
			const auto at     = (home + distance) & BucketMask;
			auto&      bucket = m_buckets[at];
			const bool empty  = bucket.profile == EmptyProfile;
			if (!empty && (bucket.profile != profile || bucket.words != words)) continue;
			if (empty) {
				++m_size;
			} else if ((bucket.flags & NegativeFlag) != 0u) {
				--m_negative;
			}
			if ((flags & NegativeFlag) != 0u) ++m_negative;
			bucket = {.words = words, .profile = profile, .value = value, .flags = flags};
			Mark(at);
			return true;
		}
		return false;
	}

	bool Erase(const KeyWordArray& words, uint32_t profile) {
		const auto home = KeyHash(words, profile);
		for (uint32_t distance = 0; distance < MaxDistance; ++distance) {
			const auto  at     = (home + distance) & BucketMask;
			const auto& bucket = m_buckets[at];
			if (bucket.profile == EmptyProfile) return false;
			if (bucket.profile == profile && bucket.words == words) {
				Remove(at);
				return true;
			}
		}
		return false;
	}

	template <typename Keep, typename Lost>
	void Rebuild(Keep&& keep, Lost&& lost) {
		std::vector<Bucket> kept;
		kept.reserve(m_size);
		for (const auto& bucket: m_buckets) {
			if (bucket.profile != EmptyProfile && keep(bucket)) kept.push_back(bucket);
		}
		std::fill(m_buckets.begin(), m_buckets.end(), Bucket {});
		m_size     = 0;
		m_negative = 0;
		for (const auto& bucket: kept) {
			if (!Insert(bucket.words, bucket.profile, bucket.value, bucket.flags)) lost(bucket);
		}
		m_all = true;
	}

	template <typename Fn>
	void TakeDirty(Fn&& fn) {
		if (m_all) {
			for (const auto at: m_dirty) m_marked[at] = 0u;
			m_dirty.clear();
			m_all = false;
			fn(uint32_t {0}, BucketCount);
			return;
		}
		std::ranges::sort(m_dirty);
		for (size_t first = 0; first < m_dirty.size();) {
			auto last = first + 1u;
			while (last < m_dirty.size() && m_dirty[last] == m_dirty[last - 1u] + 1u) ++last;
			fn(m_dirty[first], static_cast<uint32_t>(last - first));
			first = last;
		}
		for (const auto at: m_dirty) m_marked[at] = 0u;
		m_dirty.clear();
	}

	[[nodiscard]] bool                    Dirty() const noexcept { return m_all || !m_dirty.empty(); }
	[[nodiscard]] size_t                  Size() const noexcept { return m_size; }
	[[nodiscard]] size_t                  Negative() const noexcept { return m_negative; }
	[[nodiscard]] std::span<const Bucket> Buckets() const noexcept { return m_buckets; }

private:
	void Mark(uint32_t at) {
		if (m_marked[at] != 0u) return;
		m_marked[at] = 1u;
		m_dirty.push_back(at);
	}

	void Remove(uint32_t hole) {
		--m_size;
		if ((m_buckets[hole].flags & NegativeFlag) != 0u) --m_negative;
		for (auto next = (hole + 1u) & BucketMask; m_buckets[next].profile != EmptyProfile;
		     next = (next + 1u) & BucketMask) {
			const auto home = KeyHash(m_buckets[next].words, m_buckets[next].profile) & BucketMask;
			if (((next - home) & BucketMask) >= ((next - hole) & BucketMask)) {
				m_buckets[hole] = m_buckets[next];
				Mark(hole);
				hole = next;
			}
		}
		m_buckets[hole] = {};
		Mark(hole);
	}

	std::vector<Bucket>   m_buckets;
	std::vector<uint8_t>  m_marked;
	std::vector<uint32_t> m_dirty;
	size_t                m_size     = 0;
	size_t                m_negative = 0;
	bool                  m_all      = true;
};

struct Lookup {
	uint32_t slot = 0;
	bool     miss = false;
};

inline Lookup Resolve(const Directory& directory, const KeyWordArray& words, uint32_t profile,
                      std::span<const uint32_t> kinds) {
	const auto* bucket = directory.Find(words, profile);
	if (bucket == nullptr) return {.slot = 0u, .miss = true};
	if (bucket->value == 0u) return {};
	const auto arm = static_cast<uint32_t>(std::ranges::find(kinds, bucket->value >> KindShift) - kinds.begin());
	if (arm == kinds.size()) return {.slot = 0u, .miss = true};
	return {.slot = arm << ArmShift | (bucket->value & IndexMask)};
}

inline Lookup ExpectedSlot(const Directory& directory, std::span<const uint32_t> table, const Job& job,
                           uint32_t probe) {
	if (probe >= job.count) return {};
	const auto slot = uint64_t {job.first_word} + uint64_t {probe} * job.step_words;
	return Resolve(directory, ReadKey(table, slot, job.key_words), job.profile,
	               std::span(job.kinds).first(std::min(job.arm_count, MaxArms)));
}

inline void AddJob(std::vector<Job>& jobs, std::vector<uint32_t>& groups, Job job) {
	if (job.end <= job.begin) return;
	job.first_group = static_cast<uint32_t>(groups.size());
	groups.insert(groups.end(), (uint64_t {job.end} - job.begin + GroupProbes - 1u) / GroupProbes,
	              static_cast<uint32_t>(jobs.size()));
	jobs.push_back(job);
}

}

#endif

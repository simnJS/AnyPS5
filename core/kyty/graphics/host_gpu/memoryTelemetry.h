#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <memory>

namespace Libs::Graphics::MemoryTelemetry {

enum class Kind : uint8_t {
	Mirror, Texture, HeapTexture, Color, Depth, Buffer, Stream, Staging, Upload, Download, Indirect, Tiling,
	Blas, Tlas, Other, Count
};

inline constexpr std::array Names {"mirror", "texture", "heap_texture", "color", "depth", "buffer", "stream",
                                   "staging", "upload", "download", "indirect", "tiling", "blas", "tlas", "other"};
static_assert(Names.size() == static_cast<size_t>(Kind::Count));
static_assert(std::atomic<uint64_t>::is_always_lock_free);

struct Totals {
	std::atomic<uint64_t> bytes {};
	std::atomic<uint64_t> count {};
};

inline constinit std::array<std::array<Totals, 2>, Names.size()> totals {};
inline constinit std::atomic<uint64_t> resident {}, imported {}, cpu_copies {}, pipeline_ram {}, parked_local {};
inline constinit std::atomic<uint64_t> released_pipelines {}, released_reused {}, released_missed {}, released_pressure {};

struct Allocation {
	uint64_t bytes = 0;
	Kind     kind  = Kind::Other;
	bool     local = false;

	void Add(Kind category, bool device_local, uint64_t size) {
		kind  = category;
		local = device_local;
		bytes = size;
		auto& total = totals[static_cast<size_t>(kind)][local];
		total.bytes.fetch_add(bytes, std::memory_order_relaxed);
		total.count.fetch_add(1, std::memory_order_relaxed);
	}
	void Release() {
		if (bytes == 0) return;
		auto& total = totals[static_cast<size_t>(kind)][local];
		total.bytes.fetch_sub(bytes, std::memory_order_relaxed);
		total.count.fetch_sub(1, std::memory_order_relaxed);
		bytes = 0;
	}
	void Reclassify(Kind category) {
		if (kind == category || bytes == 0) return;
		const auto size = bytes;
		Release();
		Add(category, local, size);
	}
};

template <class T>
struct PipelineAllocator {
	using value_type = T;

	PipelineAllocator() = default;
	template <class U> PipelineAllocator(const PipelineAllocator<U>&) noexcept {}

	[[nodiscard]] T* allocate(size_t count) {
		auto* data = std::allocator<T>{}.allocate(count);
		pipeline_ram.fetch_add(count * sizeof(T), std::memory_order_relaxed);
		return data;
	}

	void deallocate(T* data, size_t count) noexcept {
		std::allocator<T>{}.deallocate(data, count);
		pipeline_ram.fetch_sub(count * sizeof(T), std::memory_order_relaxed);
	}

	template <class U> bool operator==(const PipelineAllocator<U>&) const noexcept { return true; }
};

template <std::atomic<uint64_t>& total>
class ScopedBytes {
public:
	explicit ScopedBytes(uint64_t size = 0) { Set(size); }
	ScopedBytes(const ScopedBytes&) = delete;
	ScopedBytes& operator=(const ScopedBytes&) = delete;
	ScopedBytes(ScopedBytes&& other) noexcept : m_size(std::exchange(other.m_size, 0)) {}
	ScopedBytes& operator=(ScopedBytes&& other) noexcept {
		Set(0);
		m_size = std::exchange(other.m_size, 0);
		return *this;
	}
	~ScopedBytes() { Set(0); }

	void Set(uint64_t size) {
		if (size != m_size) total.fetch_add(size - m_size, std::memory_order_relaxed);
		m_size = size;
	}

private:
	uint64_t m_size = 0;
};

using PipelineBytes = ScopedBytes<pipeline_ram>;

}

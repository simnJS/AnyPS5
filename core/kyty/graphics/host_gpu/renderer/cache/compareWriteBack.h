#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COMPAREWRITEBACK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COMPAREWRITEBACK_H_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

namespace Libs::Graphics {

struct WriteBackCounts {
	uint64_t written = 0;
	uint64_t kept    = 0;
};

template <class Write>
WriteBackCounts CompareWriteBack(uint64_t address, const uint8_t* result, const uint8_t* snapshot,
                                 const uint8_t* current, uint64_t size, Write&& write) {
	constexpr uint64_t Chunk = 64;
	WriteBackCounts    counts;
	uint64_t           run_begin = 0;
	uint64_t           run_size  = 0;
	const auto         flush     = [&] {
		if (run_size != 0) {
			write(address + run_begin, result + run_begin, run_size);
			counts.written += run_size;
			run_size = 0;
		}
	};
	const auto append = [&](uint64_t offset, uint64_t bytes) {
		if (run_size != 0 && run_begin + run_size == offset) {
			run_size += bytes;
			return;
		}
		flush();
		run_begin = offset;
		run_size  = bytes;
	};
	for (uint64_t offset = 0; offset < size; offset += Chunk) {
		const uint64_t bytes    = std::min(Chunk, size - offset);
		const bool     produced = std::memcmp(result + offset, snapshot + offset, bytes) != 0;
		if (std::memcmp(current + offset, snapshot + offset, bytes) == 0) {
			if (produced) {
				append(offset, bytes);
			}
			continue;
		}
		for (uint64_t index = offset; index < offset + bytes; index++) {
			if (current[index] != snapshot[index]) {
				counts.kept += result[index] != current[index] ? 1u : 0u;
			} else if (result[index] != snapshot[index]) {
				append(index, 1);
			}
		}
	}
	flush();
	return counts;
}

inline constexpr uint64_t WriteBackPieceSize = 1024 * 1024;

template <class Read, class Write>
WriteBackCounts CompareWriteBackPieces(uint64_t address, const uint8_t* result, uint8_t* snapshot,
                                       uint64_t size, Read&& read, Write&& write) {
	const auto capacity = std::min(size, WriteBackPieceSize);
	const auto current = std::make_unique_for_overwrite<uint8_t[]>(capacity);
	if (size <= capacity) {
		return read(address, current.get(), size)
		           ? CompareWriteBack(address, result, snapshot, current.get(), size, write)
		           : WriteBackCounts {};
	}

	WriteBackCounts counts;
	for (uint64_t offset = 0; offset < size; offset += capacity) {
		const auto bytes = std::min(size - offset, capacity);
		if (!read(address + offset, current.get(), bytes)) {
			return {};
		}
		uint64_t cursor = offset;
		const auto part = CompareWriteBack(address + offset, result + offset, snapshot + offset, current.get(), bytes,
		    [&](uint64_t at, const uint8_t*, uint64_t length) {
			    const auto start = at - address;
			    std::fill(snapshot + cursor, snapshot + start, uint8_t {0});
			    std::fill_n(snapshot + start, length, uint8_t {1});
			    cursor = start + length;
		    });
		std::fill(snapshot + cursor, snapshot + offset + bytes, uint8_t {0});
		counts.written += part.written;
		counts.kept += part.kept;
	}
	for (uint64_t offset = 0; offset < size;) {
		const auto begin = static_cast<const uint8_t*>(std::memchr(snapshot + offset, 1, size - offset));
		if (!begin) {
			break;
		}
		const auto start = static_cast<uint64_t>(begin - snapshot);
		const auto end = static_cast<const uint8_t*>(std::memchr(begin, 0, size - start));
		offset = end ? static_cast<uint64_t>(end - snapshot) : size;
		write(address + start, result + start, offset - start);
	}
	return counts;
}

}

#endif

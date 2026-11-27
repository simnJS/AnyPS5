#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_LABELORDER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_LABELORDER_H_

#include <cstddef>
#include <cstdint>

namespace Libs::Graphics::LabelOrder {

struct Facts {
	bool deferable      = false;
	bool tracked_page   = false;
	bool tracked_bytes  = false;
	bool around_enabled = false;
	bool backing        = false;
};

enum class Route : uint8_t { Defer, DeferAround, Drain };

enum class Reason : uint8_t {
	NotDeferable,
	CacheBytes,
	AroundOff,
	NoBacking,
	Other,
	Count
};

inline constexpr size_t ReasonCount = static_cast<size_t>(Reason::Count);

[[nodiscard]] constexpr Route Current(const Facts& facts) noexcept {
	if (!facts.deferable) {
		return Route::Drain;
	}
	if (!facts.tracked_page) {
		return Route::Defer;
	}
	return facts.around_enabled && !facts.tracked_bytes && facts.backing ? Route::DeferAround : Route::Drain;
}

[[nodiscard]] constexpr Reason DrainReason(const Facts& facts) noexcept {
	if (!facts.deferable) {
		return Reason::NotDeferable;
	}
	if (!facts.tracked_page) {
		return Reason::Other;
	}
	if (facts.tracked_bytes) {
		return Reason::CacheBytes;
	}
	if (!facts.around_enabled) {
		return Reason::AroundOff;
	}
	return facts.backing ? Reason::Other : Reason::NoBacking;
}

}

#endif

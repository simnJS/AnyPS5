#include "graphics/guest_gpu/labelOrder.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace Libs::Graphics;
using LabelOrder::Facts;
using LabelOrder::Reason;
using LabelOrder::Route;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		++g_failures;
	}
}

Facts FromBits(unsigned bits) {
	return {.deferable      = (bits & 1u) != 0,
	        .tracked_page   = (bits & 2u) != 0,
	        .tracked_bytes  = (bits & 4u) != 0,
	        .around_enabled = (bits & 8u) != 0,
	        .backing        = (bits & 16u) != 0};
}

bool Reference(const Facts& facts, bool* around) {
	const bool gpu_page = facts.deferable && facts.tracked_page;
	*around = gpu_page && facts.around_enabled && !facts.tracked_bytes && facts.backing;
	return facts.deferable && (!gpu_page || *around);
}

void EveryDrainHasAReason() {
	for (unsigned bits = 0; bits < 32; bits++) {
		const auto facts    = FromBits(bits);
		bool       around   = false;
		const bool deferred = Reference(facts, &around);
		const auto route    = LabelOrder::Current(facts);
		Check((route != Route::Drain) == deferred, "the route differs from the rule of ExecuteReleaseMem");
		Check((route == Route::DeferAround) == (deferred && around), "wrong kind of deferral");
		Check((LabelOrder::DrainReason(facts) == Reason::Other) == deferred, "a drain without a reason, or a reason without a drain");
	}
	Check(LabelOrder::Current({.deferable = true, .tracked_page = true, .around_enabled = true, .backing = true}) ==
	          Route::DeferAround,
	      "a label on a GPU-written page with no other label pending is not deferred to its producer");
	Check(LabelOrder::DrainReason({.deferable = true, .tracked_page = true, .around_enabled = true}) == Reason::NoBacking,
	      "a label outside the backing view is not named so");
	Check(LabelOrder::DrainReason({.deferable = true, .tracked_page = true, .tracked_bytes = true}) == Reason::CacheBytes,
	      "a label on GPU-written bytes of the cache is not named so");
	Check(LabelOrder::DrainReason({.deferable = true, .tracked_page = true}) == Reason::AroundOff,
	      "a label on a GPU-written page with writes around off is not named so");
}

}

int main() {
	EveryDrainHasAReason();
	if (g_failures != 0) {
		return 1;
	}
	std::puts("kyty_shim_label_order: ok");
	return 0;
}

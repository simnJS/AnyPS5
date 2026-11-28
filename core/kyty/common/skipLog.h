#ifndef KYTY_COMMON_SKIP_LOG_H_
#define KYTY_COMMON_SKIP_LOG_H_

#include <cstdint>
#include <string_view>

namespace Common::SkipLog {

enum class Kind : uint8_t { Draw, Dispatch, Count };

enum class Cause : uint8_t {
	Unknown,
	UnresolvableResources,
	CallTables,
	UnsupportedCall,
	RecompilerStopped,
	StageNotPrepared,
	FailedVariant,
	ModuleNotBuilt,
	NoPipeline,
	TessellationOff,
	TessellationWithGeometry,
	TessellationLayout,
	MeshLimits,
	Count
};

[[nodiscard]] std::string_view CauseName(Cause cause);

void Request(uint64_t shader_hash);

void Note(std::string_view hack, uint64_t shader_hash);

void Fail(Cause cause, uint64_t shader_hash = 0);

void Lost(Kind kind);

void Lost(Kind kind, Cause cause, uint64_t shader_hash);

void Summarize(uint64_t frame);

struct Window {
	uint64_t draws      = 0;
	uint64_t dispatches = 0;
	uint64_t frames     = 0;
	uint32_t lines      = 0;
};

Window SummarizeNow(uint64_t frame);

}

#endif

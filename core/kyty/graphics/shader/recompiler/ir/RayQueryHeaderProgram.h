#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERYHEADERPROGRAM_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERYHEADERPROGRAM_H_

#include <array>
#include <cstdint>
#include <optional>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

constexpr uint32_t RayQueryHeaderProgram        = 0xfffe0000u;
constexpr uint32_t MaxRayQueryHeaderProgramSize = 256;
constexpr uint32_t MaxRayQueryHeaderProgramLoads = 16;
constexpr uint32_t MaxRayQueryHeaderProgramDepth = 16;

enum class HeaderOp : uint32_t {
	User   = 1,
	Const  = 2,
	Load   = 3,
	Add    = 4,
	Mul    = 5,
	And    = 6,
	Or     = 7,
	Xor    = 8,
	Shr    = 9,
	Shl    = 10,
	Bfe    = 11,
	Mask   = 12,
	Hi     = 13,
	Join   = 14,
	Record = 15,
	UMin   = 16,
	UMax   = 17,
	ScalarBuffer = 18,
};

[[nodiscard]] constexpr uint32_t HeaderOpArgs(HeaderOp op) noexcept {
	switch (op) {
		case HeaderOp::User:
		case HeaderOp::Load:
		case HeaderOp::ScalarBuffer:
		case HeaderOp::Shr:
		case HeaderOp::Shl:
		case HeaderOp::Mask: return 1;
		case HeaderOp::Const:
		case HeaderOp::Mul: return 2;
		default: return 0;
	}
}

[[nodiscard]] constexpr uint32_t HeaderOpPops(HeaderOp op) noexcept {
	switch (op) {
		case HeaderOp::User:
		case HeaderOp::Const: return 0;
		case HeaderOp::Load:
		case HeaderOp::Mul:
		case HeaderOp::Mask:
		case HeaderOp::Hi: return 1;
		case HeaderOp::Bfe:
		case HeaderOp::ScalarBuffer: return 3;
		case HeaderOp::Record: return 5;
		default: return 2;
	}
}

[[nodiscard]] constexpr bool ValidHeaderProgram(std::span<const uint32_t> program, uint32_t user_count) noexcept {
	if (program.empty() || program.size() > MaxRayQueryHeaderProgramSize) return false;
	uint32_t depth = 0;
	uint32_t loads = 0;
	for (size_t at = 0; at < program.size();) {
		const auto op = static_cast<HeaderOp>(program[at]);
		if (program[at] < static_cast<uint32_t>(HeaderOp::User) || program[at] > static_cast<uint32_t>(HeaderOp::ScalarBuffer)) return false;
		const auto args = HeaderOpArgs(op);
		if (program.size() - at - 1 < args) return false;
		if (op == HeaderOp::User && program[at + 1] >= user_count) return false;
		if ((op == HeaderOp::Load || op == HeaderOp::ScalarBuffer) && program[at + 1] != 4u && program[at + 1] != 8u) return false;
		if ((op == HeaderOp::Shr || op == HeaderOp::Shl) && program[at + 1] != 32u && program[at + 1] != 64u) return false;
		if (op == HeaderOp::Mask && (program[at + 1] == 0u || program[at + 1] > 64u)) return false;
		if (op == HeaderOp::Load && ++loads > MaxRayQueryHeaderProgramLoads) return false;
		const auto pops = HeaderOpPops(op);
		if (depth < pops) return false;
		depth = depth - pops + 1u;
		if (depth > MaxRayQueryHeaderProgramDepth) return false;
		at += 1u + args;
	}
	return depth == 1;
}

[[nodiscard]] constexpr uint64_t HeaderMask(uint32_t bits) noexcept {
	return bits >= 64u ? ~uint64_t {0} : (uint64_t {1} << bits) - 1u;
}

template <class Read>
[[nodiscard]] std::optional<uint64_t> EvaluateHeaderProgram(std::span<const uint32_t> program, std::span<const uint32_t> user,
                                                            const Read& read) {
	if (!ValidHeaderProgram(program, static_cast<uint32_t>(user.size()))) return std::nullopt;
	std::array<uint64_t, MaxRayQueryHeaderProgramDepth> stack {};
	uint32_t                                            depth = 0;
	const auto pop = [&]() { return stack[--depth]; };
	for (size_t at = 0; at < program.size();) {
		const auto op  = static_cast<HeaderOp>(program[at]);
		const auto arg = [&](uint32_t index) { return program[at + 1u + index]; };
		uint64_t   value = 0;
		switch (op) {
			case HeaderOp::User: value = user[arg(0)]; break;
			case HeaderOp::Const: value = uint64_t {arg(0)} | uint64_t {arg(1)} << 32u; break;
			case HeaderOp::Load: {
				const auto address = pop();
				const auto bytes   = arg(0);
				if (address == 0 || address % 4u != 0 || !read(address, bytes, value)) return std::nullopt;
				break;
			}
			case HeaderOp::Mul: value = pop() * (uint64_t {arg(0)} | uint64_t {arg(1)} << 32u); break;
			case HeaderOp::Mask: value = pop() & HeaderMask(arg(0)); break;
			case HeaderOp::Hi: value = pop() >> 32u; break;
			case HeaderOp::Add:
			case HeaderOp::And:
			case HeaderOp::Or:
			case HeaderOp::Xor:
			case HeaderOp::UMin:
			case HeaderOp::UMax:
			case HeaderOp::Join: {
				const auto b = pop();
				const auto a = pop();
				switch (op) {
					case HeaderOp::Add: value = a + b; break;
					case HeaderOp::And: value = a & b; break;
					case HeaderOp::Or: value = a | b; break;
					case HeaderOp::Xor: value = a ^ b; break;
					case HeaderOp::UMin: value = a < b ? a : b; break;
					case HeaderOp::UMax: value = a < b ? b : a; break;
					default: value = (a & 0xffffffffu) | b << 32u; break;
				}
				break;
			}
			case HeaderOp::Shr:
			case HeaderOp::Shl: {
				const auto shift = pop() & (arg(0) - 1u);
				const auto a     = pop() & HeaderMask(arg(0));
				value            = (op == HeaderOp::Shr ? a >> shift : a << shift) & HeaderMask(arg(0));
				break;
			}
			case HeaderOp::Bfe: {
				const auto count  = pop();
				const auto offset = pop();
				if (offset > 32u || count > 32u - offset) return std::nullopt;
				value             = (pop() & 0xffffffffu) >> offset & HeaderMask(static_cast<uint32_t>(count));
				break;
			}
			case HeaderOp::ScalarBuffer: {
				const auto offset = pop();
				const auto records = static_cast<uint32_t>(pop());
				const auto descriptor = pop();
				const auto stride = (descriptor >> 48u) & 0x3fffu;
				const auto size = records * (stride == 0 ? 1u : stride);
				if (offset > size || size - offset < arg(0)) return std::nullopt;
				value = (descriptor & (HeaderMask(48) & ~uint64_t {3})) + offset;
				break;
			}
			case HeaderOp::Record: {
				const auto offset = static_cast<uint32_t>(pop());
				const auto index  = static_cast<uint32_t>(pop());
				const auto word3  = static_cast<uint32_t>(pop());
				const auto word2  = static_cast<uint32_t>(pop());
				const auto base   = pop();
				if ((word3 & (1u << 23u)) != 0u || (base >> 62u) != 0u) return std::nullopt;
				const auto stride = static_cast<uint32_t>(base >> 48u) & 0x3fffu;
				if (stride == 0u ? offset > word2 || word2 - offset < 4u : index >= word2 || offset > stride || stride - offset < 4u) {
					return std::nullopt;
				}
				const auto mode = (word3 >> 28u) & 3u;
				const auto byte = uint64_t {index} * stride + offset;
				if ((word3 >> 30u) != 0 || ((word3 >> 12u) & 0x7fu) == 0u) return std::nullopt;
				if (mode == 0u && stride == 0u) return std::nullopt;
				if (mode == 1u && index >= word2) return std::nullopt;
				if (mode == 3u && byte + 4u > word2) return std::nullopt;
				value = (base & HeaderMask(48)) + static_cast<uint32_t>(byte);
				break;
			}
		}
		stack[depth++] = value;
		at += 1u + HeaderOpArgs(op);
	}
	return stack[0];
}

}

#endif

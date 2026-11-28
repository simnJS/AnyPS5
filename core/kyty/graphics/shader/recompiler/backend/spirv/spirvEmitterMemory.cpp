#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <algorithm>
#include <bit>
#include <tuple>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

void EmitLdsLaneSync(EmitterState& state, const IR::MemoryInfo& mem) {
	if (!state.profile.lds_lane_sync || state.program.stage != ShaderType::Compute || mem.kind != IR::ResourceKind::Lds ||
	    state.lane_count != 2 ||
	    state.lds_storage_class != spv::StorageClassWorkgroup || state.program.info.bvh_stack) {
		return;
	}
	state.builder.AddFunction(
	    spv::OpControlBarrier, ConstantU32(state, spv::ScopeSubgroup), ConstantU32(state, spv::ScopeWorkgroup),
	    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask));
}

uint32_t AndCondition(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, spv::OpLogicalAnd, TypeBool(state), lhs, rhs);
}

uint32_t EmitDsMaskedLaneRead(EmitterState& state, uint32_t source, uint32_t target,
                              uint32_t exec) {
	if (state.lane_count == 2) {
		target = Binary(state, spv::OpBitwiseAnd, TypeU32(state), target, ConstantU32(state, 31));
	}
	const auto shuffled = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), shuffled,
	                          ConstantU32(state, spv::ScopeSubgroup), source, target);
	const auto source_exec = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeBool(state), source_exec,
	                          ConstantU32(state, spv::ScopeSubgroup), exec, target);
	const auto source_active =
	    AndCondition(state, source_exec, EmitSubgroupLaneActiveBool(state, target));
	return Select(state, TypeU32(state), source_active, shuffled, ConstantU32(state, 0));
}

struct BufferAddress {
	uint32_t offset;
	uint32_t byte;
};

BufferAddress CalculateBufferAddress(EmitterState& state, uint32_t index, uint32_t offset,
                                     uint32_t soffset, uint32_t immediate, uint32_t stride,
                                     uint32_t swizzle, uint32_t index_stride) {
	const auto zero = ConstantU32(state, 0);
	const auto one  = ConstantU32(state, 1);
	const auto add = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero ? rhs : rhs == zero ? lhs
		                                      : Binary(state, spv::OpIAdd, TypeU32(state), lhs, rhs);
	};
	const auto mul = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero || rhs == zero ? zero
		       : lhs == one              ? rhs
		       : rhs == one              ? lhs
		                                 : Binary(state, spv::OpIMul, TypeU32(state), lhs, rhs);
	};
	if (immediate != 0u) {
		offset = add(offset, ConstantU32(state, immediate));
	}
	auto address = add(mul(index, stride), offset);
	if (swizzle != 0u) {
		const auto index_shift =
		    Binary(state, spv::OpIAdd, TypeU32(state), index_stride, ConstantU32(state, 3));
		const auto indices = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                            ConstantU32(state, 1), index_shift);
		const auto index_msb =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), index, index_shift);
		const auto index_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), index,
		           Binary(state, spv::OpISub, TypeU32(state), indices, ConstantU32(state, 1)));
		const auto offset_msb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u));
		const auto offset_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, 3u));
		const auto msb = mul(add(mul(index_msb, stride), offset_msb), indices);
		const auto lsb = add(Binary(state, spv::OpShiftLeftLogical, TypeU32(state), index_lsb,
		                            ConstantU32(state, 2u)), offset_lsb);
		address = Select(state, TypeU32(state), swizzle, add(msb, lsb), address);
	}
	return {offset, add(address, soffset)};
}

uint32_t BufferLane(EmitterState& state) {
	const auto id = state.lane_count == 1 && state.program.lane_local ? EmitLocalInvocationIndex(state)
	                                                                  : EmitSubgroupLocalInvocationId(state);
	return Binary(state, spv::OpBitwiseAnd, TypeU32(state), id, ConstantU32(state, 63));
}

uint32_t BufferStrideValue(EmitterState& state, const IR::MemoryInfo& mem, uint32_t packed) {
	const auto stride = packed & IR::BufferStrideMask;
	const bool swizzle = stride != 0u && (packed & (1u << 14u)) != 0u;
	if (state.program.table_mode && !swizzle && mem.resource < state.table_strides.size()) {
		const auto value = state.table_strides[mem.resource];
		return value != 0u ? value : ConstantU32(state, stride);
	}
	if (IR::HasRuntimeStride(packed)) {
		const auto binding = state.program.info.buffers[mem.resource].descriptor_index;
		const auto value = binding < state.memory_strides.size() ? state.memory_strides[binding] : 0u;
		if (value == 0u) {
			ExitDescriptorBindingFailure(state, IR::DescriptorBindingKind::Buffers, mem.resource,
			                             "the buffer's run-time stride was not prepared");
		}
		return value;
	}
	return ConstantU32(state, stride);
}

uint32_t BufferByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto&      state  = ctx.state;
	const auto packed = StorageBufferPackedStride(state, mem);
	const auto stride = packed & IR::BufferStrideMask;
	auto       index  = ctx.Arg(inst, 1);
	if ((packed & (1u << 20u)) != 0u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, BufferLane(state));
	}
	const bool swizzle = stride != 0u && (packed & (1u << 14u)) != 0u;
	return CalculateBufferAddress(state, index, ctx.Arg(inst, 2), ctx.Arg(inst, 3), mem.offset,
	                              BufferStrideValue(state, mem, packed),
	                              swizzle ? ConstantBool(state, true) : 0u,
	                              ConstantU32(state, (packed >> 16u) & 3u))
	    .byte;
}

uint32_t AddU64Low(EmitterState& state, uint32_t low, uint32_t high, uint32_t add_low,
                   uint32_t add_high, uint32_t& out_high) {
	const auto result = Binary(state, spv::OpIAdd, TypeU32(state), low, add_low);
	const auto carry  = Binary(state, spv::OpULessThan, TypeBool(state), result, low);
	out_high =
	    Binary(state, spv::OpIAdd, TypeU32(state),
	           Binary(state, spv::OpIAdd, TypeU32(state), high, add_high),
	           Select(state, TypeU32(state), carry, ConstantU32(state, 1), ConstantU32(state, 0)));
	return result;
}

uint32_t AddAddressOffset(EmitterState& state, const IR::MemoryInfo& mem, uint32_t low,
                          uint32_t& high) {
	const auto immediate = static_cast<int32_t>(mem.offset);
	if (immediate == 0) return low;
	const auto immediate_low  = ConstantU32(state, static_cast<uint32_t>(immediate));
	const auto immediate_high = ConstantU32(state, immediate < 0 ? UINT32_MAX : 0u);
	return AddU64Low(state, low, high, immediate_low, immediate_high, high);
}

uint32_t ScratchByteAddress(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t low,
                            uint32_t high) {
	auto& state = ctx.state;
	low = AddAddressOffset(state, mem, low, high);
	const auto valid = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, 0));
	return Select(state, TypeU32(state), valid, low, ConstantU32(state, UINT32_MAX));
}

uint32_t GuestAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	auto  low   = ctx.Arg(inst, 1);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, ~3u));
	}
	uint32_t address = 0;
	if (mem.address_is_full) {
		address = PackU64(state, low, ctx.Arg(inst, 2));
	} else {
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) {
			ctx.Fail(inst, "has no address base pair");
			return ConstantU64(state, 0);
		}
		auto base_low = ctx.Arg(*handle, 0);
		if (mem.kind == IR::ResourceKind::ScalarAddress) {
			base_low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), base_low,
			                  ConstantU32(state, ~3u));
		}
		const auto base = PackU64(state, base_low, ctx.Arg(*handle, 1));
		address         = Binary(state, spv::OpIAdd, TypeU64(state), base,
		                         Unary(state, spv::OpUConvert, TypeU64(state), low));
	}
	auto immediate = static_cast<int32_t>(mem.offset);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		immediate = static_cast<int32_t>(static_cast<uint32_t>(immediate) & ~3u);
	}
	return immediate == 0
	           ? address
	           : Binary(state, spv::OpIAdd, TypeU64(state), address,
	                    ConstantU64(state, static_cast<uint64_t>(static_cast<int64_t>(immediate))));
}

uint32_t FaultElementPointer(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.fault_buffer_variable, ConstantU32(state, 0), index);
	return pointer;
}

uint32_t BdaPageIndex(EmitterState& state, uint32_t address, uint32_t shift = BufferCache::CACHING_PAGEBITS) {
	const auto type     = TypeU64(state);
	const auto extended = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address,
	                             ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase));
	const auto packed   = Select(state, type, extended,
	                             Binary(state, spv::OpISub, type, address,
	                                    ConstantU64(state,
	                                        LibKernel::Memory::kExtendedMemoryBase - LOWER_ADDRESS_SIZE)),
	                             address);
	const auto page64   = Binary(state, spv::OpShiftRightLogical, type, packed,
	                             ConstantU64(state, shift));
	return Unary(state, spv::OpUConvert, TypeU32(state), page64);
}

uint32_t BdaAddressRepresentable(EmitterState& state, uint32_t address) {
	const auto b     = TypeBool(state);
	const auto lower = Binary(state, spv::OpULessThan, b, address, ConstantU64(state, LOWER_ADDRESS_SIZE));
	const auto extended = AndCondition(
	    state, Binary(state, spv::OpUGreaterThanEqual, b, address, ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase)),
	    Binary(state, spv::OpULessThan, b,
	           Binary(state, spv::OpISub, TypeU64(state), address, ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase)),
	           ConstantU64(state, LibKernel::Memory::kExtendedMemorySize)));
	return Binary(state, spv::OpLogicalOr, b, lower, extended);
}

void RecordBdaPageBit(EmitterState& state, uint32_t page, uint64_t bitmap_offset) {
	const auto word = Binary(
	    state, spv::OpIAdd, TypeU32(state),
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5)),
	    ConstantU32(state, static_cast<uint32_t>(bitmap_offset / sizeof(uint32_t))));
	const auto bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	const auto pointer = FaultElementPointer(state, word);
	const auto scope   = ConstantU32(state, spv::ScopeDevice);
	const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
	if (state.program.info.bvh) {
		state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(),
		                          pointer, scope, relaxed, bit);
		return;
	}
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), value, pointer, scope, relaxed);
	const auto missing =
	    Binary(state, spv::OpIEqual, TypeBool(state),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, bit), ConstantU32(state, 0));
	EmitIfCondition(state, missing, [&]() {
		state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(),
		                          pointer, scope, relaxed, bit);
	});
}

void RecordBdaWrite(EmitterState& state, uint32_t page) {
	static_assert(BufferCache::WRITE_BITMAP_OFFSET % sizeof(uint32_t) == 0 &&
	              BufferCache::WRITE_BITMAP_OFFSET / sizeof(uint32_t) <= UINT32_MAX);
	RecordBdaPageBit(state, page, BufferCache::WRITE_BITMAP_OFFSET);
}

uint32_t MissingBdaDword(EmitterState& state) {
	return ConstantU32(state, state.program.info.bvh ? ~0u : 0u);
}

uint32_t EmitBdaPointerLoad(EmitterState& state, uint32_t bda, bool coherent) {
	auto pointer = state.builder.AllocateId();
	if (coherent) {
		const auto block = state.builder.DecoratedType(
		    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
		                        {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
		                        {spv::OpDecorate, {spv::DecorationBlock}}}, TypeU32(state));
		state.builder.AddFunction(
		    spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block),
		    pointer, bda);
		const auto base = pointer;
		pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU32Pointer(state), pointer,
		                          base, ConstantU32(state, 0));
	} else {
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                          bda);
	}
	const auto         value     = state.builder.AllocateId();
	constexpr uint32_t alignment = sizeof(uint32_t);
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer,
	                          spv::MemoryAccessAlignedMask |
	                              (coherent ? spv::MemoryAccessVolatileMask : spv::MemoryAccessMaskNone),
	                          alignment);
	return value;
}

uint32_t LoadBdaDwordIf(ValueEmitContext& ctx, uint32_t condition, uint32_t address, bool coherent = false) {
	auto& state = ctx.state;
	if (condition == ConstantBool(state, true)) condition = 0;
	auto [entry, inserted] = state.bda_load_functions.try_emplace(
	    std::array {uint32_t(coherent), uint32_t(condition != 0)}, 0u);
	if (inserted) entry->second = state.builder.AllocateId();
	const auto function = entry->second;
	const auto result   = state.builder.AllocateId();
	if (condition != 0) {
		state.builder.AddFunction(spv::OpFunctionCall, TypeU32(state), result, function, condition, address);
	} else {
		state.builder.AddFunction(spv::OpFunctionCall, TypeU32(state), result, function, address);
	}
	return result;
}

uint32_t LoadBdaDword(ValueEmitContext& ctx, uint32_t address, bool coherent = false) {
	return LoadBdaDwordIf(ctx, 0, address, coherent);
}

uint32_t LoadBdaPointerDword(ValueEmitContext& ctx, uint32_t bda, bool coherent, uint32_t missing) {
	auto&      state   = ctx.state;
	const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
	return EmitValueOrDefaultIfCondition(state, present, TypeU32(state), missing,
	                                     [&]() { return EmitBdaPointerLoad(state, bda, coherent); });
}

std::pair<uint32_t, uint32_t> LoadBdaCrossingDwords(ValueEmitContext& ctx, uint32_t aligned, uint32_t low,
                                                    uint32_t crosses, bool coherent) {
	auto&      state = ctx.state;
	const auto u32 = TypeU32(state), u64 = TypeU64(state), boolean = TypeBool(state);
	static_assert(BufferCache::CACHING_PAGESIZE <= (uint64_t {1} << 32u));
	const auto last_dword =
	    ConstantU32(state, static_cast<uint32_t>(BufferCache::CACHING_PAGESIZE - 1u) & ~3u);
	const auto next_page = Binary(
	    state, spv::OpLogicalAnd, boolean, crosses,
	    Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, low, last_dword), last_dword));
	const auto zero      = ConstantU64(state, 0);
	const auto preheader = state.current_label;
	const auto header    = state.builder.AllocateId();
	const auto body      = state.builder.AllocateId();
	const auto latch     = state.builder.AllocateId();
	const auto merge     = state.builder.AllocateId();
	const auto pass      = state.builder.AllocateId();
	const auto kept      = state.builder.AllocateId();
	const auto pointer   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(state, header);
	state.builder.AddFunction(spv::OpPhi, boolean, pass, ConstantBool(state, false), preheader,
	                          ConstantBool(state, true), latch);
	state.builder.AddFunction(spv::OpPhi, u64, kept, zero, preheader, pointer, latch);
	state.builder.AddFunction(spv::OpLoopMerge, merge, latch, spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranch, body);
	EmitLabel(state, body);
	const auto step = Select(state, u64, pass, ConstantU64(state, sizeof(uint32_t)), zero);
	state.builder.AddFunction(spv::OpCopyObject, u64, pointer,
	                          GetBdaPointer(state, Binary(state, spv::OpIAdd, u64, aligned, step)));
	state.builder.AddFunction(spv::OpBranch, latch);
	EmitLabel(state, latch);
	const auto again =
	    Binary(state, spv::OpLogicalAnd, boolean, next_page, Unary(state, spv::OpLogicalNot, boolean, pass));
	state.builder.AddFunction(spv::OpBranchConditional, again, header, merge);
	EmitLabel(state, merge);
	const auto first_pointer = Select(state, u64, pass, kept, pointer);
	const auto same_page_next =
	    Select(state, u64,
	           Binary(state, spv::OpLogicalAnd, boolean, crosses,
	                  Binary(state, spv::OpINotEqual, boolean, first_pointer, zero)),
	           Binary(state, spv::OpIAdd, u64, first_pointer, ConstantU64(state, sizeof(uint32_t))), zero);
	const auto second_pointer = Select(state, u64, pass, pointer, same_page_next);
	const auto first  = LoadBdaPointerDword(ctx, first_pointer, coherent, MissingBdaDword(state));
	const auto second = EmitValueOrZeroIfCondition(state, crosses, [&]() {
		return LoadBdaPointerDword(ctx, second_pointer, coherent, MissingBdaDword(state));
	});
	return {first, second};
}

uint32_t LoadBda(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t bits,
                 bool coherent = false) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, active, [&]() {
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto low  = Unary(state, spv::OpUConvert, TypeU32(state), address);
		const auto byte = Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 3));
		uint32_t   first = 0, second = 0;
		if (bits == 8u) {
			first  = LoadBdaDword(ctx, aligned, coherent);
			second = ConstantU32(state, 0);
		} else {
			const auto crosses = Binary(state, bits == 16u ? spv::OpUGreaterThan : spv::OpINotEqual,
			                            TypeBool(state), byte, ConstantU32(state, bits == 16u ? 2u : 0u));
			std::tie(first, second) = LoadBdaCrossingDwords(ctx, aligned, low, crosses, coherent);
		}
		const auto shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3));
		const auto upper_shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		           Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                  Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 4), byte),
		                  ConstantU32(state, 3)),
		           ConstantU32(state, 3));
		const auto merged =
		    Binary(state, spv::OpBitwiseOr, TypeU32(state),
		           Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, shift),
		           Binary(state, spv::OpShiftLeftLogical, TypeU32(state), second, upper_shift));
		return bits == 32u ? merged
		                   : Binary(state, spv::OpBitwiseAnd, TypeU32(state), merged,
		                            ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
	});
}

constexpr uint64_t BdaGroupPage =
    std::min<uint64_t>({GuestHostView::PageSize, uint64_t {1} << BufferCache::MIRROR_BLOCK_BITS, GuestMirror::WindowSize});
constexpr int64_t BdaGroupSpan = 252;
static_assert(LOWER_ADDRESS_SIZE % BdaGroupPage == 0 && LibKernel::Memory::kExtendedMemoryBase % BdaGroupPage == 0 &&
              LibKernel::Memory::kExtendedMemorySize % BdaGroupPage == 0 && BdaGroupPage > uint64_t {BdaGroupSpan} + 8u);

bool BdaGroupScalar(const IR::Program& program, const IR::Inst& inst, bool& scalar) {
	if (inst.GetOpcode() != IR::ValueOpcode::ReadConstBuffer && inst.GetOpcode() != IR::ValueOpcode::LoadAddressU32) {
		return false;
	}
	const auto& mem = program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
	if (inst.GetOpcode() == IR::ValueOpcode::ReadConstBuffer) {
		scalar = true;
		return !mem.planning_only && mem.kind == IR::ResourceKind::IndirectBuffer &&
		       static_cast<int32_t>(mem.offset) >= 0;
	}
	if (mem.planning_only || mem.kind == IR::ResourceKind::FlatLocal || mem.kind == IR::ResourceKind::Scratch) return false;
	scalar = mem.kind == IR::ResourceKind::ScalarAddress;
	return scalar || IR::AddressOpcodeInfoOf(inst.GetOpcode()).access == IR::AddressAccess::Read;
}

int64_t BdaGroupOffset(const IR::MemoryInfo& mem, bool scalar) {
	const auto offset = scalar ? mem.offset & ~3u : mem.offset;
	return static_cast<int64_t>(static_cast<int32_t>(offset));
}

bool SameBdaGroup(const IR::Program& program, const IR::Inst& a, const IR::Inst& b) {
	const auto& x = program.memory_info.at(a.Flags<IR::MemoryFlags>().index);
	const auto& y = program.memory_info.at(b.Flags<IR::MemoryFlags>().index);
	if (a.GetOpcode() != b.GetOpcode() || x.kind != y.kind || x.coherent != y.coherent ||
	    x.address_is_full != y.address_is_full || a.NumArgs() != b.NumArgs()) {
		return false;
	}
	for (size_t index = 0; index < a.NumArgs(); index++) {
		if (!(a.Arg(index).Resolve() == b.Arg(index).Resolve())) return false;
	}
	const bool scalar = x.kind == IR::ResourceKind::ScalarAddress || x.kind == IR::ResourceKind::IndirectBuffer;
	return (BdaGroupOffset(x, scalar) - BdaGroupOffset(y, scalar)) % 4 == 0;
}

void PlanBdaGroups(EmitterState& state) {
	state.bda_groups_planned = true;
	const auto& program = state.program;
	for (const auto* block: program.blocks) {
		std::vector<const IR::Inst*> candidates;
		std::vector<bool>            scalars;
		for (const auto& inst: *block) {
			bool scalar = false;
			if (BdaGroupScalar(program, inst, scalar)) {
				candidates.push_back(&inst);
				scalars.push_back(scalar);
			}
		}
		std::vector<bool> taken(candidates.size(), false);
		for (size_t i = 0; i < candidates.size(); i++) {
			if (taken[i]) continue;
			const auto& lead   = *candidates[i];
			const auto  origin = BdaGroupOffset(program.memory_info.at(lead.Flags<IR::MemoryFlags>().index), scalars[i]);
			std::vector<std::pair<const IR::Inst*, int64_t>> members {{&lead, 0}};
			int64_t low = 0, high = 0;
			for (size_t j = i + 1; j < candidates.size(); j++) {
				if (taken[j] || scalars[j] != scalars[i] || !SameBdaGroup(program, lead, *candidates[j])) continue;
				const auto delta =
				    BdaGroupOffset(program.memory_info.at(candidates[j]->Flags<IR::MemoryFlags>().index), scalars[i]) - origin;
				if (std::max(high, delta) - std::min(low, delta) > BdaGroupSpan) continue;
				low  = std::min(low, delta);
				high = std::max(high, delta);
				taken[j] = true;
				members.emplace_back(candidates[j], delta);
			}
			if (members.size() < 2u) continue;
			for (const auto& [member, delta]: members) {
				state.bda_group_plan[member] = {.leader = &lead, .delta = delta, .low = low, .high = high, .scalar = scalars[i]};
			}
		}
	}
}

const BdaGroupPlan* FindBdaGroup(EmitterState& state, const IR::Inst& inst) {
	if (!state.bda_groups_planned) PlanBdaGroups(state);
	const auto found = state.bda_group_plan.find(&inst);
	return found == state.bda_group_plan.end() ? nullptr : &found->second;
}

std::pair<uint32_t, uint32_t> ResolveBdaGroupPages(EmitterState& state, uint32_t first, uint32_t last, uint32_t crosses) {
	const auto u64 = TypeU64(state), boolean = TypeBool(state);
	const auto zero      = ConstantU64(state, 0);
	const auto preheader = state.current_label;
	const auto header    = state.builder.AllocateId();
	const auto body      = state.builder.AllocateId();
	const auto latch     = state.builder.AllocateId();
	const auto merge     = state.builder.AllocateId();
	const auto pass      = state.builder.AllocateId();
	const auto kept      = state.builder.AllocateId();
	const auto pointer   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(state, header);
	state.builder.AddFunction(spv::OpPhi, boolean, pass, ConstantBool(state, false), preheader,
	                          ConstantBool(state, true), latch);
	state.builder.AddFunction(spv::OpPhi, u64, kept, zero, preheader, pointer, latch);
	state.builder.AddFunction(spv::OpLoopMerge, merge, latch, spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranch, body);
	EmitLabel(state, body);
	state.builder.AddFunction(spv::OpCopyObject, u64, pointer, GetBdaPointer(state, Select(state, u64, pass, last, first)));
	state.builder.AddFunction(spv::OpBranch, latch);
	EmitLabel(state, latch);
	state.builder.AddFunction(spv::OpBranchConditional,
	                          Binary(state, spv::OpLogicalAnd, boolean, crosses, Unary(state, spv::OpLogicalNot, boolean, pass)),
	                          header, merge);
	EmitLabel(state, merge);
	return {Select(state, u64, pass, kept, pointer), Select(state, u64, pass, pointer, zero)};
}

BdaGroupPointers EmitBdaGroupPointers(EmitterState& state, BdaGroupPointers group, uint32_t first, uint32_t active) {
	const auto u32 = TypeU32(state), u64 = TypeU64(state), boolean = TypeBool(state);
	group.page_low = Binary(state, spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, first),
	                        ConstantU32(state, static_cast<uint32_t>(BdaGroupPage - 1u)));
	const auto last = Binary(state, spv::OpIAdd, u64, first, Unary(state, spv::OpUConvert, u64, group.last_delta));
	const auto crosses =
	    Binary(state, spv::OpUGreaterThanEqual, boolean, Binary(state, spv::OpIAdd, u32, group.page_low, group.last_delta),
	           ConstantU32(state, static_cast<uint32_t>(BdaGroupPage)));
	if (active == ConstantBool(state, true)) {
		std::tie(group.first, group.last) = ResolveBdaGroupPages(state, first, last, crosses);
		return group;
	}
	const auto zero        = ConstantU64(state, 0);
	const auto header      = state.current_label;
	const auto then_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, active, then_label, merge_label);
	EmitLabel(state, then_label);
	const auto [first_pointer, last_pointer] = ResolveBdaGroupPages(state, first, last, crosses);
	const auto then_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	group.first = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, u64, group.first, first_pointer, then_exit, zero, header);
	group.last = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, u64, group.last, last_pointer, then_exit, zero, header);
	return group;
}

BdaGroupPointers EmitBdaGroup(ValueEmitContext& ctx, const BdaGroupPlan& plan, uint32_t address, uint32_t active) {
	auto&      state = ctx.state;
	const auto u32 = TypeU32(state), u64 = TypeU64(state), boolean = TypeBool(state);
	BdaGroupPointers group {.epoch = state.block_epoch, .delta = plan.delta};
	const auto start =
	    plan.low == plan.delta
	        ? address
	        : Binary(state, spv::OpIAdd, u64, address, ConstantU64(state, static_cast<uint64_t>(plan.low - plan.delta)));
	const auto span = static_cast<uint32_t>(plan.high - plan.low);
	uint32_t   first = start;
	if (plan.scalar) {
		group.byte       = ConstantU32(state, 0);
		group.unaligned  = ConstantBool(state, false);
		group.last_delta = ConstantU32(state, span);
	} else {
		first            = Binary(state, spv::OpBitwiseAnd, u64, start, ConstantU64(state, ~uint64_t {3}));
		group.byte       = Binary(state, spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, start), ConstantU32(state, 3));
		group.unaligned  = Binary(state, spv::OpINotEqual, boolean, group.byte, ConstantU32(state, 0));
		group.last_delta = Select(state, u32, group.unaligned, ConstantU32(state, span + 4u), ConstantU32(state, span));
	}
	return EmitBdaGroupPointers(state, group, first, active == 0 ? ConstantBool(state, true) : active);
}

uint32_t BdaGroupDwordPointer(EmitterState& state, const BdaGroupPointers& group, uint32_t rel) {
	const auto u64 = TypeU64(state), boolean = TypeBool(state);
	const auto zero     = ConstantU64(state, 0);
	const auto in_first = Binary(state, spv::OpULessThan, boolean, group.page_low,
	                             ConstantU32(state, static_cast<uint32_t>(BdaGroupPage) - rel));
	const auto base     = Select(state, u64, in_first, group.first, group.last);
	const auto offset   = Select(state, u64, in_first, ConstantU64(state, rel),
	                             Binary(state, spv::OpISub, u64, ConstantU64(state, rel),
	                                    Unary(state, spv::OpUConvert, u64, group.last_delta)));
	return Select(state, u64, Binary(state, spv::OpINotEqual, boolean, base, zero),
	              Binary(state, spv::OpIAdd, u64, base, offset), zero);
}

uint32_t LoadGroupedBda(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem, const BdaGroupPlan& plan) {
	auto&      state  = ctx.state;
	const auto active = plan.scalar ? ConstantBool(state, true) : ctx.Arg(inst, inst.NumArgs() - 1);
	auto       found  = ctx.bda_groups.find(plan.leader);
	if (found == ctx.bda_groups.end() || found->second.epoch != state.block_epoch) {
		found = ctx.bda_groups.insert_or_assign(plan.leader, EmitBdaGroup(ctx, plan, GuestAddress(ctx, inst, mem), active)).first;
	}
	const auto& group = found->second;
	const auto  rel   = static_cast<uint32_t>(plan.delta - plan.low);
	const auto  dword = [&](uint32_t at) {
		return LoadBdaPointerDword(ctx, BdaGroupDwordPointer(state, group, at), !plan.scalar && mem.coherent,
		                           MissingBdaDword(state));
	};
	if (plan.scalar) return dword(rel);
	return EmitValueOrZeroIfCondition(state, active, [&]() {
		const auto u32    = TypeU32(state);
		const auto first  = dword(rel);
		const auto second = EmitValueOrZeroIfCondition(state, group.unaligned, [&]() { return dword(rel + 4u); });
		const auto shift  = Binary(state, spv::OpShiftLeftLogical, u32, group.byte, ConstantU32(state, 3));
		const auto upper_shift =
		    Binary(state, spv::OpShiftLeftLogical, u32,
		           Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpISub, u32, ConstantU32(state, 4), group.byte),
		                  ConstantU32(state, 3)),
		           ConstantU32(state, 3));
		return Binary(state, spv::OpBitwiseOr, u32, Binary(state, spv::OpShiftRightLogical, u32, first, shift),
		              Binary(state, spv::OpShiftLeftLogical, u32, second, upper_shift));
	});
}

uint32_t ByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                     uint32_t prefix = 0) {
	if (mem.kind == IR::ResourceKind::Buffer) {
		const auto resource = ctx.state.program.info.buffers.at(mem.resource).descriptor_index;
		if (prefix == 0u) {
			prefix = ctx.state.program.table_mode ? ConstantU32(ctx.state, 0)
			                                      : ctx.state.memory_byte_offsets[resource];
		}
		const auto address = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state),
		                            BufferByteAddress(ctx, inst, mem), prefix);
		return Select(ctx.state, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), address, prefix),
		              address, ConstantU32(ctx.state, UINT32_MAX));
	}
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		if (mem.offset == 0u) {
			return ctx.Arg(inst, 0);
		}
		return Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), ctx.Arg(inst, 0),
		              ConstantU32(ctx.state, mem.offset));
	}
	if (mem.kind != IR::ResourceKind::Scratch) {
		EXIT("physical address memory must use the BDA emitter\n");
	}
	return ScratchByteAddress(ctx, mem, ctx.Arg(inst, 1), ctx.Arg(inst, 2));
}

uint32_t DwordIndex(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                    uint32_t prefix = 0) {
	return Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state),
	              ByteAddress(ctx, inst, mem, prefix), ConstantU32(ctx.state, 2));
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index);

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits);

template <typename Fn>
auto EmitBufferAccess(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem,
                      uint32_t components, Fn&& emit) {
	auto& state = ctx.state;
	constexpr bool has_result = !std::is_void_v<
	    std::invoke_result_t<Fn, const IR::MemoryInfo&, const MemoryResourceAccess&>>;
	const auto access = [&](uint32_t resource) {
		mem.resource = resource;
		if (mem.kind == IR::ResourceKind::Buffer) {
			const auto& buffer = state.program.info.buffers[resource];
			if (buffer.packed_stride == 0u &&
			    buffer.source < state.program.descriptor_sources.size() &&
			    state.program.descriptor_sources[buffer.source].indirect_descriptor) {
				return emit(mem, MemoryResourceAccess{});
			}
		}
		return emit(mem, PrepareMemoryResourceAccess(state, mem));
	};
	if (mem.kind != IR::ResourceKind::Buffer ||
	    state.program.info.buffers[mem.resource].indirect_root != mem.resource) {
		return access(mem.resource);
	}
	const auto root_resource = mem.resource;
	const auto& buffer = state.program.info.buffers[root_resource];
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto selected = ctx.Arg(handle, 0);
	const auto representative = SharedIndirectBufferResource(state, root_resource);
	const auto shared = representative != UINT32_MAX;
	const auto empty_root = shared && buffer.packed_stride == 0u;
	const auto selected_access = [&]() {
		if (shared) {
			const auto root_slot = buffer.descriptor_index;
			const auto child_slot = state.program.info.buffers[buffer.indirect_resources[1]].descriptor_index;
			auto slot = EmitAddU32(state, selected, ConstantU32(state, child_slot - 1u));
			if (!empty_root && child_slot != root_slot + 1u) {
				slot = Select(state, TypeU32(state),
				    Binary(state, spv::OpIEqual, TypeBool(state), selected, ConstantU32(state, 0u)),
				    ConstantU32(state, root_slot), slot);
			}
			mem.resource = representative;
			return emit(mem, PrepareMemoryResourceAccess(state, mem, slot));
		}
		return EmitIndexSwitch(state, selected, buffer.indirect_resources.size(),
		                       has_result ? TypeU32Composite(state, components) : 0u,
		                       [&](uint32_t item) { return access(buffer.indirect_resources[item]); });
	};
	if constexpr (!has_result) {
		if (!shared) return selected_access();
	}
	auto valid = Binary(state, spv::OpULessThan, TypeBool(state), selected,
	                          ConstantU32(state, buffer.indirect_resources.size()));
	if constexpr (has_result) {
		return EmitValueOrDefaultIfCondition(state, valid, TypeU32Composite(state, components),
		    ConstantU32CompositeZero(state, components), [&]() {
			    if (!empty_root) return selected_access();
			    return EmitValueOrDefaultIfCondition(state,
			        Binary(state, spv::OpINotEqual, TypeBool(state), selected, ConstantU32(state, 0u)),
			        TypeU32Composite(state, components), access(root_resource), selected_access);
		    });
	} else {
		if (empty_root) valid = AndCondition(state, valid,
		    Binary(state, spv::OpINotEqual, TypeBool(state), selected, ConstantU32(state, 0u)));
		EmitIfCondition(state, valid, selected_access);
	}
}

uint32_t BufferIndexInBounds(ValueEmitContext& ctx, const IR::Inst& inst,
                             const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                             uint32_t payload_in_bounds, uint32_t payload_bytes = 0u) {
	if (mem.kind != IR::ResourceKind::Buffer) return payload_in_bounds;
	auto& state = ctx.state;
	const auto& buffer = state.program.info.buffers[mem.resource];
	const auto& sources = state.program.descriptor_sources;
	const bool structured = (buffer.packed_stride & IR::StructuredBufferBounds) != 0u;
	const bool indexed = buffer.source < sources.size() &&
	    sources[buffer.source].indirect_descriptor &&
	    sources[buffer.source].indirect_descriptor->indexed;
	const auto binding = buffer.descriptor_index;
	const auto live_bounds = !indexed && binding < state.memory_bounds.size()
	    ? state.memory_bounds[binding].structured : 0u;
	if (!structured && !indexed && live_bounds == 0u) return payload_in_bounds;
	const auto stride = BufferStrideValue(state, mem, buffer.packed_stride);
	const auto bytes = Binary(state, spv::OpIMul, TypeU32(state), resource.length,
	                          ConstantU32(state, resource.element_bits / 8u));
	const auto available = Binary(state, spv::OpISub, TypeU32(state), bytes, resource.byte_offset);
	const auto last_record = Binary(state, spv::OpUDiv, TypeU32(state),
	    Binary(state, spv::OpISub, TypeU32(state), available, ConstantU32(state, 1u)),
	    EmitUMax32(state, stride, ConstantU32(state, 1u)));
	auto index = ctx.Arg(inst, 1);
	auto in_bounds = payload_in_bounds;
	if ((buffer.packed_stride & (1u << 20u)) != 0u) {
		const auto original = index;
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, BufferLane(state));
		in_bounds = AndCondition(state, in_bounds,
		                         Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), index, original));
	}
	const auto original_offset = ctx.Arg(inst, 2);
	const auto offset = mem.offset == 0u ? original_offset
	    : Binary(state, spv::OpIAdd, TypeU32(state), original_offset,
	             ConstantU32(state, mem.offset));
	if (mem.offset != 0u)
		in_bounds = AndCondition(state, in_bounds,
		                         Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), offset, original_offset));
	in_bounds = AndCondition(state, in_bounds,
	    AndCondition(state,
	        Binary(state, spv::OpUGreaterThan, TypeBool(state), bytes, resource.byte_offset),
	        Binary(state, spv::OpULessThanEqual, TypeBool(state), index, last_record)));
	if (!structured && live_bounds == 0u) return in_bounds;
	const auto offset_in_bounds = payload_bytes == 0u
	    ? Binary(state, spv::OpULessThan, TypeBool(state), offset, stride)
	    : AndCondition(state,
	        Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), stride, payload_bytes),
	        Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
	               Binary(state, spv::OpISub, TypeU32(state), stride, payload_bytes)));
	in_bounds = AndCondition(state, in_bounds, offset_in_bounds);
	if (live_bounds == 0u) return in_bounds;
	return Select(state, TypeBool(state), live_bounds, in_bounds,
	              payload_in_bounds != 0u ? payload_in_bounds : ConstantBool(state, true));
}

uint32_t StorageBufferElementInBounds(EmitterState& state, const IR::MemoryInfo& mem,
                                      const MemoryResourceAccess& resource, uint32_t index) {
	if (mem.kind == IR::ResourceKind::Buffer) {
		const auto& buffer = state.program.info.buffers[mem.resource];
		const auto binding = buffer.descriptor_index;
		if (binding < state.memory_bounds.size() && state.memory_bounds[binding].relaxed != 0u && !resource.physical) {
			return Binary(state, spv::OpLogicalOr, TypeBool(state), state.memory_bounds[binding].relaxed,
			              EmitMemoryElementInBounds(state, resource, index));
		}
		if ((buffer.packed_stride & IR::ExactBufferBounds) != 0u)
			return EmitMemoryElementInBounds(state, resource, index);
	}
	return EmitStorageElementInBounds(state, resource, index);
}

uint32_t LoadWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource) {
	const auto index = DwordIndex(ctx, inst, mem, resource.byte_offset);
	return EmitValueOrZeroIfCondition(
	    ctx.state, BufferIndexInBounds(ctx, inst, mem, resource,
	                                   StorageBufferElementInBounds(ctx.state, mem, resource, index)),
	    [&]() { return LoadWordInBounds(ctx, resource, index); });
}

inline bool HasPrivateArrays(const EmitterState& state) {
	return state.scratch_storage_class == spv::StorageClassPrivate || state.lds_storage_class == spv::StorageClassPrivate;
}
bool IsPrivateArray(const EmitterState& state, const MemoryResourceAccess& resource);
uint32_t LoadPrivateArray(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                          uint32_t valid);
void StorePrivateArray(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t valid, uint32_t data);

uint32_t LoadWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	if (HasPrivateArrays(ctx.state) && (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Scratch)) {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		if (IsPrivateArray(ctx.state, resource)) {
			const auto index = DwordIndex(ctx, inst, mem, resource.byte_offset);
			return LoadPrivateArray(ctx, resource, index,
			                        AndCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1),
			                                     EmitMemoryElementInBounds(ctx.state, resource, index)));
		}
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadWordPrepared(ctx, inst, mem, resource);
	});
}

uint32_t LoadSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const MemoryResourceAccess& resource, uint32_t bits) {
	const auto address = ByteAddress(ctx, inst, mem, resource.byte_offset);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	return EmitValueOrZeroIfCondition(
	    ctx.state, BufferIndexInBounds(ctx, inst, mem, resource,
	                                   StorageBufferElementInBounds(ctx.state, mem, resource, index)),
	    [&]() { return LoadSubwordInBounds(ctx, resource, address, index, bits); });
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index) {
	return EmitAccessLoad(ctx.state, TypeU32(ctx.state), EmitMemoryElementPointer(ctx.state, resource, index),
	                      resource);
}

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits) {
	uint32_t value;
	if (resource.element_bits < 32u) {
		const auto loaded = EmitAccessLoad(ctx.state, ctx.state.storage_buffers[bits == 8u ? 0 : 1].element_type,
		                                   EmitMemoryElementPointer(ctx.state, resource, index), resource);
		value = Unary(ctx.state, spv::OpUConvert, TypeU32(ctx.state), loaded);
	} else {
		const auto word = LoadWordInBounds(ctx, resource, index);
		const auto byte = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                         ConstantU32(ctx.state, 3));
		const auto shift = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), byte,
		                          ConstantU32(ctx.state, 3));
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
		               Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), word, shift),
		               ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu));
	}
	return value;
}

uint32_t LoadSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem,
                     uint32_t bits) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadSubwordPrepared(ctx, inst, mem, resource, bits);
	});
}

Prospero::BufferFormat BufferFormat(const ValueEmitContext& ctx, const IR::MemoryInfo& mem) {
	return mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
	                 : StorageBufferFormat(ctx.state, mem);
}

IR::MemoryInfo RebaseRawComponent(IR::MemoryInfo mem, uint32_t component) {
	mem.offset += component * 4u;
	mem.data_dwords     = 1u;
	mem.component_index = component;
	return mem;
}

using Format::FormattedSource;
using Format::FormattedSourceKind;

FormattedSource ResolveFormattedSource(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t                        output_component) {
	if (mem.typed) {
		return output_component < info.component_count
		           ? FormattedSource {FormattedSourceKind::Memory, output_component}
		           : FormattedSource {};
	}
	const auto selector = GetDstSel(ctx.state.program.info.buffers[mem.resource].descriptor_swizzle,
	                                output_component);
	const auto source = Format::ResolveFormattedSource(info, selector);
	if (source.kind == FormattedSourceKind::Invalid) {
		ExitDescriptorBindingFailure(ctx.state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer descriptor has reserved dst_sel");
	}
	return source;
}

uint32_t FormattedConstant(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                           FormattedSourceKind kind) {
	return ConstantU32(ctx.state, Format::FormattedConstantBits(info, kind));
}

template <typename LoadWordFn, typename LoadSubwordFn>
uint32_t LoadFormattedComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                                FormattedSource source, LoadWordFn&& load_word,
                                LoadSubwordFn&& load_subword) {
	if (source.kind != FormattedSourceKind::Memory) {
		return FormattedConstant(ctx, info, source.kind);
	}
	const auto component = source.component;
	const auto bits      = info.component_bits[component];
	auto raw = info.packed_bitfield || bits == 32u ? load_word(component)
	                                                : load_subword(component, bits);
	if (info.packed_bitfield || (bits < 32u && IsSignedFormatComponent(info.type))) {
		const auto type =
		    IsSignedFormatComponent(info.type) ? TypeI32(ctx.state) : TypeU32(ctx.state);
		const auto source_value =
		    type == TypeI32(ctx.state) ? Unary(ctx.state, spv::OpBitcast, type, raw) : raw;
		const auto extracted = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(IsSignedFormatComponent(info.type) ? spv::OpBitFieldSExtract
		                                                                 : spv::OpBitFieldUExtract,
		                              type, extracted, source_value,
		                              ConstantU32(ctx.state, info.packed_bitfield
		                                                         ? info.component_bit_offset[component]
		                                                         : 0u),
		                              ConstantU32(ctx.state, bits));
		raw = type == TypeI32(ctx.state)
		          ? Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), extracted)
		          : extracted;
	}
	return NormalizeFormatComponent(ctx.state, info, component, raw);
}

void StoreSubwordInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t address, uint32_t index,
                          uint32_t bits, uint32_t data) {
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	if (resource.element_bits < 32u) {
		EmitAccessStore(ctx.state, pointer,
		                Unary(ctx.state, spv::OpUConvert, ctx.state.storage_buffers[bits == 8u ? 0 : 1].element_type,
		                      data),
		                resource);
		return;
	}
	const auto shift   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
	                                   ConstantU32(ctx.state, 3)),
	                            ConstantU32(ctx.state, 3));
	const auto mask    = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu), shift);
	const auto value   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), data,
	                                   ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu)),
	                            shift);
	const auto merge   = [&](uint32_t old) {
		return Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), old,
		                     Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), mask)),
		              value);
	};
	if (mem.kind == IR::ResourceKind::Scratch) {
		const auto old = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), old, pointer);
		ctx.state.builder.AddFunction(spv::OpStore, pointer, merge(old));
	} else {
		AtomicUpdate(ctx.state, pointer, mem.kind, merge);
	}
}

void StoreSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t bits, uint32_t data) {
	if (resource.object_pointer == 0u) return;
	const auto address = ByteAddress(ctx, inst, mem, resource.byte_offset);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	const auto in_bounds = resource.element_bits < 32u
	                           ? StorageBufferElementInBounds(ctx.state, mem, resource, index)
	                           : EmitMemoryElementInBounds(ctx.state, resource, index);
	EmitIfCondition(ctx.state, BufferIndexInBounds(ctx, inst, mem, resource, in_bounds), [&]() {
		StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data);
	});
}

void StoreSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		EmitBufferAccess(ctx, inst, mem, 0u, [&](const auto& selected, const auto& resource) {
			StoreSubwordPrepared(ctx, inst, selected, resource, bits, ctx.Arg(inst, inst.NumArgs() - 2));
		});
	});
}

void StoreWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t data) {
	EmitAccessStore(ctx.state, EmitMemoryElementPointer(ctx.state, resource, index), data, resource);
}

bool IsPrivateArray(const EmitterState& state, const MemoryResourceAccess& resource) {
	return (resource.kind == IR::ResourceKind::Scratch && state.scratch_storage_class == spv::StorageClassPrivate) ||
	       (resource.kind == IR::ResourceKind::Lds && state.lds_storage_class == spv::StorageClassPrivate);
}

uint32_t PrivateArraySafeIndex(EmitterState& state, const MemoryResourceAccess& resource, uint32_t index) {
	return EmitUMin32(state, index,
	                  Binary(state, spv::OpISub, TypeU32(state), resource.length, ConstantU32(state, 1u)));
}

uint32_t LoadPrivateArray(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                          uint32_t valid) {
	auto&      state = ctx.state;
	const auto value = LoadWordInBounds(ctx, resource, PrivateArraySafeIndex(state, resource, index));
	return Select(state, TypeU32(state), valid, value, ConstantU32(state, 0));
}

void StorePrivateArray(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t valid, uint32_t data) {
	auto&      state = ctx.state;
	const auto safe  = PrivateArraySafeIndex(state, resource, index);
	const auto old   = LoadWordInBounds(ctx, resource, safe);
	StoreWordInBounds(ctx, resource, safe, Select(state, TypeU32(state), valid, data, old));
}

template <typename Fn>
void ForEachLocalFlatAccess(ValueEmitContext& ctx, const IR::Inst& inst, Fn&& emit) {
	auto& state = ctx.state;
	auto mem = ctx.Memory(inst);
	auto high = ctx.Arg(inst, 2);
	const auto low = AddAddressOffset(state, mem, ctx.Arg(inst, 1), high);
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), low, ConstantU32(state, 2));
	const auto aligned = Binary(state, spv::OpIEqual, TypeBool(state), ConstantU32(state, 0),
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 3)));
	for (const auto kind: {IR::ResourceKind::Scratch, IR::ResourceKind::Lds}) {
		mem.kind = kind;
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto aperture = kind == IR::ResourceKind::Scratch ? Decoder::PrivateApertureHigh
		                                                       : Decoder::SharedApertureHigh;
		const auto selected = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, aperture));
		const auto valid = AndCondition(state, aligned,
		    AndCondition(state, selected, EmitMemoryElementInBounds(state, resource, index)));
		emit(resource, index, valid);
	}
}

uint32_t LoadLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto private_array = [&](IR::ResourceKind kind) {
		auto mem = ctx.Memory(inst);
		mem.kind = kind;
		return IsPrivateArray(ctx.state, PrepareMemoryResourceAccess(ctx.state, mem));
	};
	if (HasPrivateArrays(ctx.state) && private_array(IR::ResourceKind::Scratch) && private_array(IR::ResourceKind::Lds)) {
		const auto exec   = ctx.Arg(inst, inst.NumArgs() - 1);
		uint32_t   result = 0;
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			const auto value = LoadPrivateArray(ctx, resource, index, AndCondition(ctx.state, exec, valid));
			result = result == 0 ? value : Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), result, value);
		});
		return result;
	}
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		uint32_t result = 0;
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			const auto value = EmitValueOrZeroIfCondition(ctx.state, valid, [&]() {
				return LoadWordInBounds(ctx, resource, index);
			});
			result = result == 0 ? value : Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), result, value);
		});
		return result;
	});
}

void StoreLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto private_array = [&](IR::ResourceKind kind) {
		auto mem = ctx.Memory(inst);
		mem.kind = kind;
		return IsPrivateArray(ctx.state, PrepareMemoryResourceAccess(ctx.state, mem));
	};
	if (HasPrivateArrays(ctx.state) && private_array(IR::ResourceKind::Scratch) && private_array(IR::ResourceKind::Lds)) {
		const auto exec = ctx.Arg(inst, inst.NumArgs() - 1);
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			StorePrivateArray(ctx, resource, index, AndCondition(ctx.state, exec, valid),
			                  ctx.Arg(inst, inst.NumArgs() - 2));
		});
		return;
	}
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			EmitIfCondition(ctx.state, valid, [&]() {
				StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, inst.NumArgs() - 2));
			});
		});
	});
}

uint32_t LastLaneValueOfSharedAddress(EmitterState& state, uint32_t index, uint32_t data) {
	const auto scope  = ConstantU32(state, spv::ScopeSubgroup);
	const auto shared = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformAllEqual, TypeBool(state), shared, scope, index);
	const auto ballot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot, scope,
	                          ConstantBool(state, true));
	const auto last = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallotFindMSB, TypeU32(state), last, scope, ballot);
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), value, scope, data, last);
	return Select(state, TypeU32(state), shared, value, data);
}

void StoreWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t data) {
	if (resource.object_pointer == 0u) return;
	const auto index = DwordIndex(ctx, inst, mem, resource.byte_offset);
	const auto in_bounds = BufferIndexInBounds(ctx, inst, mem, resource,
	    StorageBufferElementInBounds(ctx.state, mem, resource, index));
	EmitIfCondition(ctx.state, in_bounds, [&]() {
		StoreWordInBounds(ctx, resource, index,
		                  resource.kind == IR::ResourceKind::Lds && ctx.state.lds_last_lane_stores
		                      ? LastLaneValueOfSharedAddress(ctx.state, index, data)
		                      : data);
	});
}

void StoreWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	if (HasPrivateArrays(ctx.state) && (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Scratch)) {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		if (IsPrivateArray(ctx.state, resource)) {
			const auto index = DwordIndex(ctx, inst, mem, resource.byte_offset);
			StorePrivateArray(ctx, resource, index,
			                  AndCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1),
			                               EmitMemoryElementInBounds(ctx.state, resource, index)),
			                  ctx.Arg(inst, inst.NumArgs() - 2));
			return;
		}
	}
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		EmitBufferAccess(ctx, inst, mem, 0u, [&](const auto& selected, const auto& resource) {
			StoreWordPrepared(ctx, inst, selected, resource, ctx.Arg(inst, inst.NumArgs() - 2));
		});
	});
}

template <typename Fn>
uint32_t EmitAtomicAccess(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& operation) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		auto index = DwordIndex(ctx, inst, mem);
		if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
			index = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), index,
			               ConstantU32(ctx.state, 0x3fffu));
		}
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return EmitValueOrZeroIfCondition(
		    ctx.state, BufferIndexInBounds(ctx, inst, mem, resource,
		        EmitMemoryElementInBounds(ctx.state, resource, index),
		        mem.kind == IR::ResourceKind::Buffer ? ConstantU32(ctx.state, 4u) : 0u), [&]() {
			    const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
			    if (ctx.state.split_hints != nullptr && ctx.state.unordered_update && mem.kind == IR::ResourceKind::Lds) {
				    ctx.state.split_hints->unordered_updates.emplace_back(pointer, *ctx.state.unordered_update);
			    }
			    return operation(pointer);
		    });
	});
}

template <typename Fn>
uint32_t EmitAtomicUpdate(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& replacement) {
	const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		return AtomicUpdate(ctx.state, pointer, mem.kind, [&](uint32_t old) {
			return replacement(ctx.state, old, value);
		});
	});
}

uint32_t AtomicIncrement(EmitterState& state, uint32_t old, uint32_t limit) {
	const auto wrap = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), old, limit);
	const auto next = Binary(state, spv::OpIAdd, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, ConstantU32(state, 0), next);
}

uint32_t AtomicDecrement(EmitterState& state, uint32_t old, uint32_t limit) {
	const auto zero  = Binary(state, spv::OpIEqual, TypeBool(state), old, ConstantU32(state, 0));
	const auto above = Binary(state, spv::OpUGreaterThan, TypeBool(state), old, limit);
	const auto wrap  = Binary(state, spv::OpLogicalOr, TypeBool(state), zero, above);
	const auto next  = Binary(state, spv::OpISub, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, limit, next);
}

struct PreparedFormattedMemory {
	Format::BufferFormatInfo info;
	MemoryResourceAccess     resource;
	uint32_t                 base = 0;
	uint32_t                 in_bounds = 0;
};

PreparedFormattedMemory PrepareFormattedMemory(ValueEmitContext& ctx, const IR::Inst& inst,
                                               const IR::MemoryInfo& mem,
                                               const MemoryResourceAccess& resource,
                                               const Format::BufferFormatInfo& info) {
	PreparedFormattedMemory plan;
	plan.info      = info;
	plan.resource  = resource;
	const auto start = ByteAddress(ctx, inst, mem, resource.byte_offset);
	const auto end = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), start,
	                        ConstantU32(ctx.state, info.byte_size - 1u));
	const auto last_index = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), end,
	                               ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	plan.in_bounds = BufferIndexInBounds(ctx, inst, mem, resource, AndCondition(
	    ctx.state, Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), end, start),
	    EmitMemoryElementInBounds(ctx.state, plan.resource, last_index)),
	    ConstantU32(ctx.state, info.byte_size));
	plan.base = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), start,
	                   ConstantU32(ctx.state, ~(std::min(4u, info.byte_size) - 1u)));
	return plan;
}

std::pair<uint32_t, uint32_t> FormattedComponentAddress(EmitterState& state,
                                                       const PreparedFormattedMemory& plan,
                                                       uint32_t component) {
	const auto offset = Format::GetFormatComponentByteOffset(plan.info, component);
	const auto address = offset == 0u ? plan.base : Binary(
	    state, spv::OpIAdd, TypeU32(state), plan.base, ConstantU32(state, offset));
	return {address, Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                        ConstantU32(state, std::countr_zero(plan.resource.element_bits / 8u)))};
}

uint32_t LoadFormattedInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                               const PreparedFormattedMemory& plan, uint32_t output_component) {
	return LoadFormattedComponent(
	    ctx, plan.info, ResolveFormattedSource(ctx, mem, plan.info, output_component),
	    [&](uint32_t component) {
		    return LoadWordInBounds(ctx, plan.resource,
		                            FormattedComponentAddress(ctx.state, plan, component).second);
	    },
	    [&](uint32_t component, uint32_t bits) {
		    const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
		    return LoadSubwordInBounds(ctx, plan.resource, address, index, bits);
	    });
}

uint32_t ConstructU32Composite(EmitterState& state, uint32_t components,
                               const std::array<uint32_t, 4>& values) {
	if (components == 1u) return values[0];
	const auto            result = state.builder.AllocateId();
	std::vector<uint32_t> words {spv::OpCompositeConstruct, TypeU32Composite(state, components),
	                             result};
	words.insert(words.end(), values.begin(), values.begin() + components);
	state.builder.AddFunction(words);
	return result;
}

uint32_t FormattedOutOfBoundsValue(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                   const PreparedFormattedMemory& plan, uint32_t components) {
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
		values[component] = FormattedConstant(ctx, plan.info, source.kind);
	}
	return ConstructU32Composite(ctx.state, components, values);
}

uint32_t LoadFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                               const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                               const Format::BufferFormatInfo& info, uint32_t components) {
	auto&      state = ctx.state;
	const auto count = mem.data_bits == 16u ? mem.component_count : components;
	const auto plan = resource.object_pointer == 0u ? PreparedFormattedMemory{.info = info}
	                                              : PrepareFormattedMemory(ctx, inst, mem, resource, info);
	const auto out_of_bounds = FormattedOutOfBoundsValue(ctx, mem, plan, count);
	const auto value = resource.object_pointer == 0u ? out_of_bounds : EmitValueOrDefaultIfCondition(
	    state, plan.in_bounds, TypeU32Composite(state, count), out_of_bounds, [&]() {
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < count; component++) {
			    values[component] = LoadFormattedInBounds(ctx, mem, plan, component);
		    }
		    return ConstructU32Composite(state, count, values);
	    });
	if (mem.data_bits != 16u) return value;
	const bool integer = info.type == Format::ComponentType::Uint ||
	                     info.type == Format::ComponentType::Sint;
	const auto scalar = [&](uint32_t component) {
		if (component >= count) return integer ? ConstantU32(state, 0u) : ConstantF32(state, 0);
		auto result = value;
		if (count != 1u) {
			result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), result, value,
			                          component);
		}
		if (info.type == Format::ComponentType::Uint) {
			return EmitUMin32(state, result, ConstantU32(state, 0xffffu));
		}
		if (info.type == Format::ComponentType::Sint) {
			const auto lower = EmitSMax32(state, result, ConstantU32(state, 0xffff8000u));
			return EmitSMin32(state, lower, ConstantU32(state, 0x7fffu));
		}
		return EmitBitCastF32U32(state, result);
	};
	std::array<uint32_t, 4> packed {};
	for (uint32_t word = 0; word < components; word++) {
		const auto low  = scalar(word * 2u);
		const auto high = scalar(word * 2u + 1u);
		if (integer) {
			packed[word] = Binary(
			    state, spv::OpBitwiseOr, TypeU32(state),
			    Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 0xffffu)),
			    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), high, ConstantU32(state, 16u)));
		} else {
			const auto pair = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair,
			                          low, high);
			packed[word] = EmitPackHalf2x16(state, pair);
		}
	}
	return ConstructU32Composite(state, components, packed);
}

void StoreFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                             const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                             const Format::BufferFormatInfo& info, uint32_t data,
                             uint32_t components) {
	if (resource.object_pointer == 0u) return;
	const auto count = mem.data_bits == 16u ? mem.component_count : components;
	const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
	EmitIfCondition(ctx.state, plan.in_bounds, [&]() {
		auto packed = ConstantU32(ctx.state, 0);
		for (uint32_t component = 0; component < info.component_count; component++) {
			auto value = ConstantU32(ctx.state, 0);
			if (component < count) {
				const auto word = mem.data_bits == 16u ? component / 2u : component;
				value = data;
				if (components != 1u) {
					value = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
					                              value, data, word);
				}
				if (mem.data_bits == 16u && (component & 1u) != 0u)
					value = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), value,
					               ConstantU32(ctx.state, 16u));
			}
			value = EncodeFormattedStoreComponent(ctx.state, info, component, value, mem.data_bits);
			const auto bits = info.component_bits[component];
			if (info.packed_bitfield) {
				packed = EmitBitFieldInsert(
				    ctx.state, packed, value,
				    ConstantU32(ctx.state, info.component_bit_offset[component]),
				    ConstantU32(ctx.state, bits));
			} else {
				const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
				if (bits == 8u || bits == 16u) {
					StoreSubwordInBounds(ctx, mem, plan.resource, address, index, bits, value);
				} else {
					StoreWordInBounds(ctx, plan.resource, index, value);
				}
			}
		}
		if (info.packed_bitfield) {
			StoreWordInBounds(ctx, plan.resource,
			                  FormattedComponentAddress(ctx.state, plan, 0u).second, packed);
		}
	});
}

uint32_t LoadGroupedIndirectScalar(ValueEmitContext& ctx, const BdaGroupPlan& plan, uint32_t base, uint32_t offset,
                                   uint32_t size) {
	auto&      state = ctx.state;
	const auto u32 = TypeU32(state), u64 = TypeU64(state), boolean = TypeBool(state);
	const auto dword = ConstantU64(state, sizeof(uint32_t));
	auto       found = ctx.bda_groups.find(plan.leader);
	if (found == ctx.bda_groups.end() || found->second.epoch != state.block_epoch) {
		const auto first_offset =
		    plan.low == plan.delta
		        ? offset
		        : Binary(state, spv::OpIAdd, u64, offset, ConstantU64(state, static_cast<uint64_t>(plan.low - plan.delta)));
		const auto active =
		    Binary(state, spv::OpULessThanEqual, boolean, Binary(state, spv::OpIAdd, u64, first_offset, dword), size);
		const auto room = Binary(state, spv::OpBitwiseAnd, u64,
		                         Binary(state, spv::OpISub, u64, Binary(state, spv::OpISub, u64, size, dword), first_offset),
		                         ConstantU64(state, ~uint64_t {3}));
		const auto span = ConstantU64(state, static_cast<uint64_t>(plan.high - plan.low));
		BdaGroupPointers group {.epoch = state.block_epoch, .delta = plan.delta};
		group.byte       = ConstantU32(state, 0);
		group.unaligned  = ConstantBool(state, false);
		group.last_delta = Unary(state, spv::OpUConvert, u32,
		                         Select(state, u64,
		                                AndCondition(state, active, Binary(state, spv::OpULessThan, boolean, room, span)),
		                                room, Select(state, u64, active, span, ConstantU64(state, 0))));
		found = ctx.bda_groups
		            .insert_or_assign(plan.leader, EmitBdaGroupPointers(state, group, Binary(state, spv::OpIAdd, u64, base, first_offset),
		                                                                active))
		            .first;
	}
	const auto& group = found->second;
	const auto  rel   = static_cast<uint32_t>(plan.delta - plan.low);
	return EmitValueOrDefaultIfCondition(
	    state, Binary(state, spv::OpULessThanEqual, boolean, Binary(state, spv::OpIAdd, u64, offset, dword), size), u32,
	    ConstantU32(state, 0), [&]() {
		    return LoadBdaPointerDword(ctx, BdaGroupDwordPointer(state, group, rel), false, MissingBdaDword(state));
	    });
}

uint32_t LoadIndirectScalarBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  word1  = ctx.Arg(handle, 1);
	const auto  stride = EmitBitFieldUExtract(state, word1, ConstantU32(state, 16),
	                                         ConstantU32(state, 14));
	const auto  size   = Binary(state, spv::OpIMul, TypeU64(state),
	                            Unary(state, spv::OpUConvert, TypeU64(state),
	                                  EmitUMax32(state, stride, ConstantU32(state, 1))),
	                            Unary(state, spv::OpUConvert, TypeU64(state), ctx.Arg(handle, 2)));
	const auto  offset = Binary(state, spv::OpIAdd, TypeU64(state),
	                            Unary(state, spv::OpUConvert, TypeU64(state),
	                                  Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                                         ctx.Arg(inst, 1), ConstantU32(state, ~3u))),
	                            ConstantU64(state, ctx.Memory(inst).offset & ~3u));
	const auto  end =
	    Binary(state, spv::OpIAdd, TypeU64(state), offset, ConstantU64(state, sizeof(uint32_t)));
	const auto scalar_address = [&]() {
		const auto base = PackU64(state,
		                          Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                                 ctx.Arg(handle, 0), ConstantU32(state, ~3u)),
		                          Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1,
		                                 ConstantU32(state, 0xffffu)));
		return Binary(state, spv::OpIAdd, TypeU64(state), base, offset);
	};
	if (const auto* group = FindBdaGroup(state, inst); group != nullptr) {
		const auto base = PackU64(state,
		                          Binary(state, spv::OpBitwiseAnd, TypeU32(state), ctx.Arg(handle, 0), ConstantU32(state, ~3u)),
		                          Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1, ConstantU32(state, 0xffffu)));
		return LoadGroupedIndirectScalar(ctx, *group, base, offset, size);
	}
	return LoadBdaDwordIf(ctx, Binary(state, spv::OpULessThanEqual, TypeBool(state), end, size),
	                      scalar_address());
}

struct IndirectBufferAccess {
	uint32_t address;
	uint32_t offset;
	uint32_t stride;
	uint32_t swizzle;
	uint32_t format;
	uint32_t selector;
	uint32_t mode;
	uint32_t records;
	uint32_t index_in_bounds;
	uint32_t scalar_in_bounds;
	uint32_t raw_records;
	uint32_t raw_index_in_bounds;
	uint32_t index;
	uint32_t logical_offset;
	uint32_t soffset;
	uint32_t index_stride;
	uint32_t base;
	uint32_t linear;
};

void ApplyIndirectTypedFormat(EmitterState& state, const IR::MemoryInfo& mem, std::array<uint32_t, 8>& args) {
	if (!mem.typed) return;
	const auto format = Format::DecodeTBufferFormat(mem.data_format, mem.number_format);
	const auto info   = Format::GetFormatInfo(format);
	uint32_t   fields = static_cast<uint32_t>(format) << 12u;
	for (uint32_t component = 0; component < info.component_count; ++component) fields |= (4u + component) << (component * 3u);
	const auto descriptor = args[3];
	const auto control    = Binary(state, spv::OpBitwiseAnd, TypeU32(state), descriptor, ConstantU32(state, ~0x7ffffu));
	const auto valid      = Binary(state, spv::OpINotEqual, TypeBool(state),
	                               EmitBitFieldUExtract(state, descriptor, ConstantU32(state, 12u), ConstantU32(state, 7u)),
	                               ConstantU32(state, 0u));
	args[3] = Select(state, TypeU32(state), valid,
	                 Binary(state, spv::OpBitwiseOr, TypeU32(state), control, ConstantU32(state, fields)), control);
}

std::array<uint32_t, 8> IndirectBufferArgs(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	auto offset = ctx.Arg(inst, 2);
	if (ctx.Memory(inst).offset != 0u) {
		const auto immediate = ConstantU32(state, ctx.Memory(inst).offset);
		offset = offset == ConstantU32(state, 0) ? immediate
		    : Binary(state, spv::OpIAdd, TypeU32(state), offset, immediate);
	}
	return {ctx.Arg(handle, 0), ctx.Arg(handle, 1), ctx.Arg(handle, 2), ctx.Arg(handle, 3),
	        ctx.Arg(inst, 1), offset, ctx.Arg(inst, 3), BufferLane(state)};
}

IndirectBufferAccess PrepareIndirectBuffer(EmitterState& state,
                                           const std::array<uint32_t, 8>& args, bool formatted) {
	const auto& [word0, word1, records, word3, input_index, offset, soffset, lane] = args;
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto stride  = field(word1, 16, 14);
	const auto swizzle = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto index = Binary(
	    state, spv::OpIAdd, TypeU32(state), input_index,
	    Select(state, TypeU32(state), nonzero(field(word3, 23, 1)), lane,
	           ConstantU32(state, 0)));
	const auto index_stride = field(word3, 21, 2);
	const auto address = CalculateBufferAddress(state, index, offset, soffset, 0u, stride, swizzle,
	                                            index_stride);
	const auto base        = PackU64(state, word0, field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	const auto wide        = [&](uint32_t value) { return Unary(state, spv::OpUConvert, TypeU64(state), value); };
	const auto linear      = Binary(state, spv::OpIAdd, TypeU64(state),
	                                Binary(state, spv::OpIMul, TypeU64(state), wide(index), wide(stride)),
	                                wide(address.offset));
	const auto buffer_type = Binary(state, spv::OpIEqual, TypeBool(state), field(word3, 30, 2), ConstantU32(state, 0));
	return {
	    .address          = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                               Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .offset           = address.offset,
	    .stride           = stride,
	    .swizzle          = swizzle,
	    .format           = Select(state, TypeU32(state), buffer_type, field(word3, 12, 7), ConstantU32(state, 0)),
	    .selector         = formatted ? field(word3, 0, 3) : 0u,
	    .mode             = field(word3, 28, 2),
	    .records          = records,
	    .index_in_bounds  = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records      = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	    .index               = index,
	    .logical_offset      = address.offset,
	    .soffset             = soffset,
	    .index_stride        = index_stride,
	    .base                = base,
	    .linear              = linear,
	};
}

uint32_t IndirectBufferAddressAt(EmitterState& state, const IndirectBufferAccess& buffer, uint32_t displacement) {
	const auto offset  = Binary(state, spv::OpIAdd, TypeU32(state), buffer.logical_offset, displacement);
	const auto address = CalculateBufferAddress(state, buffer.index, offset, buffer.soffset, 0u, buffer.stride,
	                                            buffer.swizzle, buffer.index_stride);
	return Binary(state, spv::OpIAdd, TypeU64(state), buffer.base,
	              Unary(state, spv::OpUConvert, TypeU64(state), address.byte));
}

uint32_t IndirectComponentAddress(EmitterState& state, const IndirectBufferAccess& buffer, uint32_t element,
                                  uint32_t displacement, uint32_t alignment) {
	const auto linear   = Binary(state, spv::OpIAdd, TypeU64(state), element,
	                             Unary(state, spv::OpUConvert, TypeU64(state), displacement));
	const auto swizzled = Binary(state, spv::OpBitwiseAnd, TypeU64(state),
	                             IndirectBufferAddressAt(state, buffer, displacement),
	                             Unary(state, spv::OpNot, TypeU64(state), Unary(state, spv::OpUConvert, TypeU64(state), alignment)));
	return Select(state, TypeU64(state), buffer.swizzle, swizzled, linear);
}

uint32_t IndirectBufferInBounds(EmitterState& state, const IndirectBufferAccess& buffer,
                                uint32_t displacement, uint32_t bytes, bool formatted) {
	const auto offset = displacement == 0u ? buffer.offset : Binary(
	    state, spv::OpIAdd, TypeU32(state), buffer.offset, ConstantU32(state, displacement));
	const auto has_payload = [&](uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size, bytes),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, bytes)));
	};
	const auto structured = AndCondition(
	    state, buffer.index_in_bounds,
	    formatted ? has_payload(buffer.stride)
	              : Binary(state, spv::OpULessThan, TypeBool(state), offset, buffer.stride));
	const auto wide       = [&](uint32_t value) { return Unary(state, spv::OpUConvert, TypeU64(state), value); };
	const auto linear_end = Binary(state, spv::OpIAdd, TypeU64(state),
	                               Binary(state, spv::OpIAdd, TypeU64(state), buffer.linear,
	                                      ConstantU64(state, displacement)),
	                               wide(bytes));
	const auto raw = AndCondition(
	    state, buffer.scalar_in_bounds,
	    Select(state, TypeBool(state), buffer.swizzle,
	           AndCondition(state, buffer.raw_index_in_bounds, has_payload(buffer.stride)),
	           Binary(state, spv::OpULessThanEqual, TypeBool(state), linear_end, wide(buffer.raw_records))));
	auto in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 0)),
	    structured, buffer.index_in_bounds);
	in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	    in_bounds,
	    Select(state, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	           Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records, ConstantU32(state, 0)),
	           raw));
	return in_bounds;
}

uint32_t LoadIndirectBuffer(ValueEmitContext& ctx, const IndirectBufferAccess& buffer,
                            uint32_t bits, uint32_t components, bool coherent) {
	auto& state = ctx.state;
	const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
	                                  ConstantU32(state, 0));
	const auto base         = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                                 ConstantU64(state, ~uint64_t {3}));
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; ++component) {
		const auto address = component == 0u ? base
		                                     : IndirectComponentAddress(state, buffer, base, ConstantU32(state, component * 4u),
		                                                                ConstantU32(state, 3));
		const auto condition = AndCondition(state, valid_format,
		                                    IndirectBufferInBounds(state, buffer, component * 4u,
		                                                           ConstantU32(state, bits / 8u), false));
		const auto finish = [&](uint32_t value) {
			if (bits == 32u) return value;
			const auto byte = Binary(
			    state, spv::OpBitwiseAnd, TypeU32(state),
			    Unary(state, spv::OpUConvert, TypeU32(state), buffer.address),
			    ConstantU32(state, bits == 8u ? 3u : 2u));
			return EmitBitFieldUExtract(
			    state, value,
			    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3u)),
			    ConstantU32(state, bits));
		};
		values[component] = finish(LoadBdaDwordIf(ctx, condition, address, coherent));
	}
	return ConstructU32Composite(state, components, values);
}

template <typename Write>
void WriteThroughGuest(EmitterState& state, uint32_t aligned, uint32_t bda, Write&& write) {
	const auto u64        = TypeU64(state);
	const auto boolean    = TypeBool(state);
	const auto guest_base = LoadMirrorMetadata64(state, 8);
	const auto size       = LoadMirrorMetadata64(state, 16);
	const auto offset     = Binary(state, spv::OpISub, u64, aligned, guest_base);
	const auto inside     = AndCondition(state, BdaAddressRepresentable(state, aligned),
	                                     AndCondition(state, Binary(state, spv::OpUGreaterThanEqual, boolean, aligned, guest_base),
	                                                  Binary(state, spv::OpULessThan, boolean, offset, size)));
	const auto host = EmitValueOrDefaultIfCondition(state, inside, u64, ConstantU64(state, 0), [&] {
		return EmitGuestHostAddress(state, offset, size, ConstantU32(state, sizeof(uint32_t)), true);
	});
	const auto through = AndCondition(state, Binary(state, spv::OpINotEqual, boolean, host, ConstantU64(state, 0)),
	                                  Binary(state, spv::OpINotEqual, boolean, host, bda));
	EmitIfCondition(state, through, [&]() {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, host);
		write(pointer);
	});
}

void StoreBdaDword(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t value) {
	auto& state = ctx.state;
	if (state.converged_calls) {
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto bda     = GetBdaPointerMasked(state, aligned, active);
		const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
		EmitIfCondition(state, present, [&]() {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda);
			constexpr uint32_t alignment = sizeof(uint32_t);
			state.builder.AddFunction(spv::OpStore, pointer, value, spv::MemoryAccessAlignedMask, alignment);
			WriteThroughGuest(state, aligned, bda, [&](uint32_t host) {
				state.builder.AddFunction(spv::OpStore, host, value, spv::MemoryAccessAlignedMask, alignment);
			});
			RecordBdaWrite(state, BdaPageIndex(state, aligned));
		});
		return;
	}
	EmitIfCondition(state, active, [&]() {
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto bda     = GetBdaPointer(state, aligned);
		const auto present =
		    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
		EmitIfCondition(state, present, [&]() {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
			                          bda);
			constexpr uint32_t alignment = sizeof(uint32_t);
			state.builder.AddFunction(spv::OpStore, pointer, value, spv::MemoryAccessAlignedMask,
			                          alignment);
			WriteThroughGuest(state, aligned, bda, [&](uint32_t host) {
				state.builder.AddFunction(spv::OpStore, host, value, spv::MemoryAccessAlignedMask, alignment);
			});
			RecordBdaWrite(state, BdaPageIndex(state, aligned));
		});
	});
}

void StoreIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	const auto store = [&]() {
		const auto buffer = PrepareIndirectBuffer(state, IndirectBufferArgs(ctx, inst), false);
		auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
		                           ConstantU32(state, 0));
		if (state.converged_calls) valid_format = AndCondition(state, valid_format, ctx.Arg(inst, inst.NumArgs() - 1));
		const auto base = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
		                         ConstantU64(state, ~uint64_t {3}));
		const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
		for (uint32_t component = 0; component < components; component++) {
			auto value = data;
			if (components != 1u) {
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data,
				                          component);
			}
			const auto address = component == 0u ? base
			                                     : IndirectComponentAddress(state, buffer, base, ConstantU32(state, component * 4u),
			                                                                ConstantU32(state, 3));
			StoreBdaDword(ctx, address,
			              AndCondition(state, valid_format,
			                           IndirectBufferInBounds(state, buffer, component * 4u,
			                                                  ConstantU32(state, 4u), false)),
			              value);
		}
	};
	if (state.converged_calls) store();
	else EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), store);
}

static uint32_t ExtractU32(EmitterState& state, uint32_t composite, uint32_t component);

void MergeBdaBits(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t value, uint32_t bits);

void StoreIndirectSubword(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t bits) {
	auto& state = ctx.state;
	const auto store = [&]() {
		const auto buffer = PrepareIndirectBuffer(state, IndirectBufferArgs(ctx, inst), false);
		auto       valid  = AndCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format, ConstantU32(state, 0)),
		                                 IndirectBufferInBounds(state, buffer, 0u, ConstantU32(state, bits / 8u), false));
		if (state.converged_calls) valid = AndCondition(state, valid, ctx.Arg(inst, inst.NumArgs() - 1));
		const auto address = bits == 16u ? Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address, ConstantU64(state, ~uint64_t {1}))
		                                 : buffer.address;
		const auto value = Binary(state, spv::OpBitwiseAnd, TypeU32(state), ctx.Arg(inst, inst.NumArgs() - 2),
		                          ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
		MergeBdaBits(ctx, address, valid, value, ConstantU32(state, bits));
	};
	if (state.converged_calls) store();
	else EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), store);
}

enum class IndirectStoreKind : uint32_t { Drop, Raw, Uint, Sint, Unorm, Snorm, Half, Uscaled, Sscaled };

uint32_t IndirectStoreFormat(EmitterState& state, uint32_t format) {
	constexpr auto format_count = static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u;
	const auto     result_type  = TypeU32Composite(state, 4);
	const auto known = Select(state, TypeU32(state),
	                          Binary(state, spv::OpULessThan, TypeBool(state), format, ConstantU32(state, format_count)),
	                          format, ConstantU32(state, 0));
	return EmitIndexSwitch(state, known, format_count, result_type, [&](uint32_t index) {
		const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(index));
		const auto bits = info.component_bits[0];
		bool       same = info.component_count != 0u && !info.packed_bitfield && info.byte_size != 0u &&
		            (bits == 8u || bits == 16u || bits == 32u);
		for (uint32_t component = 1; same && component < info.component_count; ++component)
			same = info.component_bits[component] == bits;
		auto kind = IndirectStoreKind::Drop;
		if (same) {
			switch (info.type) {
				case Format::ComponentType::Uint: kind = bits == 32u ? IndirectStoreKind::Raw : IndirectStoreKind::Uint; break;
				case Format::ComponentType::Sint: kind = bits == 32u ? IndirectStoreKind::Raw : IndirectStoreKind::Sint; break;
				case Format::ComponentType::Float:
					kind = bits == 32u ? IndirectStoreKind::Raw : bits == 16u ? IndirectStoreKind::Half : IndirectStoreKind::Drop;
					break;
				case Format::ComponentType::Unorm: kind = bits == 32u ? IndirectStoreKind::Drop : IndirectStoreKind::Unorm; break;
				case Format::ComponentType::Snorm: kind = bits == 32u ? IndirectStoreKind::Drop : IndirectStoreKind::Snorm; break;
				case Format::ComponentType::Uscaled: kind = IndirectStoreKind::Uscaled; break;
				case Format::ComponentType::Sscaled: kind = IndirectStoreKind::Sscaled; break;
				default: break;
			}
		}
		return state.builder.Constant(spv::OpConstantComposite, result_type,
		                              ConstantU32(state, static_cast<uint32_t>(kind)), ConstantU32(state, bits),
		                              ConstantU32(state, info.component_count), ConstantU32(state, info.byte_size));
	});
}

uint32_t ConvertIndirectStoreValue(EmitterState& state, uint32_t value, uint32_t kind, uint32_t bits) {
	const auto u    = TypeU32(state);
	const auto f    = TypeF32(state);
	const auto b    = TypeBool(state);
	const auto is   = [&](IndirectStoreKind wanted) {
		return Binary(state, spv::OpIEqual, b, kind, ConstantU32(state, static_cast<uint32_t>(wanted)));
	};
	const auto low  = Select(state, u, Binary(state, spv::OpIEqual, b, bits, ConstantU32(state, 32)),
	                         ConstantU32(state, 0xffffffffu),
	                         Binary(state, spv::OpISub, u,
	                                Binary(state, spv::OpShiftLeftLogical, u, ConstantU32(state, 1), bits),
	                                ConstantU32(state, 1)));
	const auto real = Unary(state, spv::OpBitcast, f, value);
	const auto clamp = [&](uint32_t x, float lo, float hi) {
		const auto above = Select(state, f, Binary(state, spv::OpFOrdGreaterThan, b, x, ConstantF32Value(state, lo)), x,
		                          ConstantF32Value(state, lo));
		return Select(state, f, Binary(state, spv::OpFOrdLessThan, b, above, ConstantF32Value(state, hi)), above,
		              ConstantF32Value(state, hi));
	};
	const auto unorm_max = Unary(state, spv::OpConvertUToF, f, low);
	const auto unorm     = Unary(state, spv::OpConvertFToU, u,
	                             Binary(state, spv::OpFAdd, f, Binary(state, spv::OpFMul, f, clamp(real, 0.0f, 1.0f), unorm_max),
	                                    ConstantF32Value(state, 0.5f)));
	const auto snorm_max = Unary(state, spv::OpConvertUToF, f, Binary(state, spv::OpShiftRightLogical, u, low, ConstantU32(state, 1)));
	const auto snormed   = Binary(state, spv::OpFMul, f, clamp(real, -1.0f, 1.0f), snorm_max);
	const auto rounding  = Select(state, f, Binary(state, spv::OpFOrdLessThan, b, snormed, ConstantF32Value(state, 0.0f)),
	                              ConstantF32Value(state, -0.5f), ConstantF32Value(state, 0.5f));
	const auto snorm = Unary(state, spv::OpBitcast, u,
	                         Unary(state, spv::OpConvertFToS, TypeI32(state), Binary(state, spv::OpFAdd, f, snormed, rounding)));
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, real, ConstantF32Value(state, 0.0f));
	const auto half    = EmitGlsl<GLSLstd450PackHalf2x16, IR::Type::U32>(state, pair);
	const auto uscaled = Unary(state, spv::OpConvertFToU, u, clamp(real, 0.0f, 4294967040.0f));
	const auto sscaled = Unary(state, spv::OpBitcast, u, Unary(state, spv::OpConvertFToS, TypeI32(state),
	                                                           clamp(real, -2147483648.0f, 2147483520.0f)));
	auto converted = value;
	converted = Select(state, u, is(IndirectStoreKind::Unorm), unorm, converted);
	converted = Select(state, u, is(IndirectStoreKind::Snorm), snorm, converted);
	converted = Select(state, u, is(IndirectStoreKind::Half), half, converted);
	converted = Select(state, u, is(IndirectStoreKind::Uscaled), uscaled, converted);
	converted = Select(state, u, is(IndirectStoreKind::Sscaled), sscaled, converted);
	return Binary(state, spv::OpBitwiseAnd, u, converted, low);
}

void MergeBdaBits(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t value, uint32_t bits) {
	auto&      state   = ctx.state;
	const auto u       = TypeU32(state);
	const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address, ConstantU64(state, ~uint64_t {3}));
	const auto shift   = Binary(state, spv::OpShiftLeftLogical, u,
	                            Binary(state, spv::OpBitwiseAnd, u, Unary(state, spv::OpUConvert, u, address),
	                                   ConstantU32(state, 3)),
	                            ConstantU32(state, 3));
	const auto low  = Binary(state, spv::OpISub, u, Binary(state, spv::OpShiftLeftLogical, u, ConstantU32(state, 1), bits),
	                         ConstantU32(state, 1));
	const auto mask = Binary(state, spv::OpShiftLeftLogical, u, low, shift);
	const auto bits_in_place = Binary(state, spv::OpShiftLeftLogical, u, value, shift);
	const auto write = [&](uint32_t bda) {
		const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
		EmitIfCondition(state, present, [&]() {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda);
			const auto scope   = ConstantU32(state, spv::ScopeDevice);
			const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
			state.builder.AddFunction(spv::OpAtomicAnd, u, state.builder.AllocateId(), pointer, scope, relaxed,
			                          Unary(state, spv::OpNot, u, mask));
			state.builder.AddFunction(spv::OpAtomicOr, u, state.builder.AllocateId(), pointer, scope, relaxed,
			                          bits_in_place);
			WriteThroughGuest(state, aligned, bda, [&](uint32_t host) {
				state.builder.AddFunction(spv::OpAtomicAnd, u, state.builder.AllocateId(), host, scope, relaxed,
				                          Unary(state, spv::OpNot, u, mask));
				state.builder.AddFunction(spv::OpAtomicOr, u, state.builder.AllocateId(), host, scope, relaxed,
				                          bits_in_place);
			});
			RecordBdaWrite(state, BdaPageIndex(state, aligned));
		});
	};
	if (state.converged_calls) {
		write(GetBdaPointerMasked(state, aligned, active));
		return;
	}
	EmitIfCondition(state, active, [&]() { write(GetBdaPointer(state, aligned)); });
}

void StoreIndirectFormatted(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	const auto store = [&]() {
		const auto u      = TypeU32(state);
		const auto b      = TypeBool(state);
		auto args = IndirectBufferArgs(ctx, inst);
		ApplyIndirectTypedFormat(state, ctx.Memory(inst), args);
		const auto buffer = PrepareIndirectBuffer(state, args, true);
		auto       valid  = Binary(state, spv::OpINotEqual, b, buffer.format, ConstantU32(state, 0));
		if (state.converged_calls) valid = AndCondition(state, valid, ctx.Arg(inst, inst.NumArgs() - 1));
		const auto layout = IndirectStoreFormat(state, buffer.format);
		const auto kind   = ExtractU32(state, layout, 0);
		const auto bits   = ExtractU32(state, layout, 1);
		const auto count  = ExtractU32(state, layout, 2);
		const auto total  = ExtractU32(state, layout, 3);
		const auto bytes  = Binary(state, spv::OpShiftRightLogical, u, bits, ConstantU32(state, 3));
		valid = AndCondition(state, valid,
		                     Binary(state, spv::OpINotEqual, b, kind,
		                            ConstantU32(state, static_cast<uint32_t>(IndirectStoreKind::Drop))));
		valid = AndCondition(state, valid, IndirectBufferInBounds(state, buffer, 0u, total, true));
		const auto element_alignment = Binary(
		    state, spv::OpISub, u,
		    Select(state, u, Binary(state, spv::OpULessThan, b, total, ConstantU32(state, 4)), total, ConstantU32(state, 4)),
		    ConstantU32(state, 1));
		const auto element = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
		                            Unary(state, spv::OpNot, TypeU64(state),
		                                  Unary(state, spv::OpUConvert, TypeU64(state), element_alignment)));
		const auto component_alignment = Binary(state, spv::OpISub, u,
		                                        Select(state, u, Binary(state, spv::OpIEqual, b, bytes, ConstantU32(state, 0)),
		                                               ConstantU32(state, 1), bytes),
		                                        ConstantU32(state, 1));
		const auto data  = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto whole = Binary(state, spv::OpIEqual, b, bits, ConstantU32(state, 32));
		for (uint32_t component = 0; component < 4u; component++) {
			const auto value = component >= components ? ConstantU32(state, 0)
			                 : components == 1u        ? data
			                                           : ExtractU32(state, data, component);
			const auto present = AndCondition(state, valid,
			                                  Binary(state, spv::OpULessThan, b, ConstantU32(state, component), count));
			const auto displacement = Binary(state, spv::OpIMul, u, bytes, ConstantU32(state, component));
			const auto address      = IndirectComponentAddress(state, buffer, element, displacement, component_alignment);
			const auto converted    = ConvertIndirectStoreValue(state, value, kind, bits);
			StoreBdaDword(ctx, address, AndCondition(state, present, whole), converted);
			MergeBdaBits(ctx, address, AndCondition(state, present, Unary(state, spv::OpLogicalNot, b, whole)), converted,
			             Select(state, u, whole, ConstantU32(state, 16), bits));
		}
	};
	if (state.converged_calls) store();
	else EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), store);
}

uint32_t AtomicIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto exec   = ctx.Arg(inst, inst.NumArgs() - 1);
	const auto atomic = [&]() {
		const auto buffer = PrepareIndirectBuffer(state, IndirectBufferArgs(ctx, inst), false);
		const auto valid  = AndCondition(
		    state, Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format, ConstantU32(state, 0)),
		    IndirectBufferInBounds(state, buffer, 0u, ConstantU32(state, 4u), false));
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto run = [&](uint32_t bda) {
			const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
			return EmitValueOrZeroIfCondition(state, present, [&]() {
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda);
				const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
				WriteThroughGuest(state, aligned, bda,
				                  [&](uint32_t host) { (void)EmitAtomicOperation(ctx, inst, host, spv::ScopeDevice); });
				EmitAtomicMemoryBarrier(state, IR::ResourceKind::IndirectBuffer);
				RecordBdaWrite(state, BdaPageIndex(state, aligned));
				return old;
			});
		};
		if (state.converged_calls)
			return run(GetBdaPointerMasked(state, aligned, AndCondition(state, valid, exec)));
		return EmitValueOrZeroIfCondition(state, valid, [&]() { return run(GetBdaPointer(state, aligned)); });
	};
	if (state.converged_calls) return atomic();
	return EmitValueOrZeroIfCondition(state, exec, atomic);
}

uint32_t LoadIndirectFormattedComponent(ValueEmitContext& ctx, const IndirectBufferAccess& buffer,
                                        uint32_t geometry, uint32_t selector) {
	auto& state = ctx.state;
	const auto field = [&](uint32_t index) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, geometry, index);
		return value;
	};
	const auto bytes = field(0);
	const auto component = Binary(
	    state, spv::OpUMod, TypeU32(state),
	    Binary(state, spv::OpISub, TypeU32(state), selector, ConstantU32(state, 4)), field(1));
	const auto alignment = Unary(state, spv::OpUConvert, TypeU64(state), field(3));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                         Unary(state, spv::OpNot, TypeU64(state), alignment));
	const auto component_bytes = field(2);
	const auto component_alignment = Select(
	    state, TypeU32(state), Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), component_bytes, ConstantU32(state, 4)),
	    ConstantU32(state, 3),
	    Select(state, TypeU32(state), Binary(state, spv::OpIEqual, TypeBool(state), component_bytes, ConstantU32(state, 2)),
	           ConstantU32(state, 1), ConstantU32(state, 0)));
	const auto address = IndirectComponentAddress(state, buffer, base,
	                                              Binary(state, spv::OpIMul, TypeU32(state), component, component_bytes),
	                                              component_alignment);
	const auto memory = AndCondition(
	    state, Binary(state, spv::OpINotEqual, TypeBool(state), bytes, ConstantU32(state, 0)),
	    Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), selector, ConstantU32(state, 4)));
	const auto raw_condition = AndCondition(state, memory, IndirectBufferInBounds(state, buffer, 0u, bytes, true));
	const auto raw_address   = [&]() {
		return Binary(state, spv::OpBitwiseAnd, TypeU64(state), address, ConstantU64(state, ~uint64_t {3}));
	};
	const auto raw_shifted = [&](uint32_t word) {
		const auto byte = Binary(
		    state, spv::OpBitwiseAnd, TypeU32(state),
		    Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3));
		return Binary(state, spv::OpShiftRightLogical, TypeU32(state), word,
		              Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3)));
	};
	const auto raw = raw_shifted(LoadBdaDwordIf(ctx, raw_condition, raw_address()));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU32(state), result,
	                          state.indirect_format_function, buffer.format, selector,
	                          component, raw);
	return result;
}

uint32_t IndirectFormatGeometry(EmitterState& state, const IndirectBufferAccess& buffer) {
	const auto geometry = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU32Composite(state, 4), geometry,
	                          state.indirect_format_geometry_function, buffer.format);
	return geometry;
}

uint32_t LoadIndirectFormatted(ValueEmitContext& ctx, const IndirectBufferAccess& buffer,
                               const std::array<uint32_t, 8>& args, uint32_t components) {
	auto& state = ctx.state;
	const auto geometry = IndirectFormatGeometry(state, buffer);
	if (components == 1u) {
		return LoadIndirectFormattedComponent(ctx, buffer, geometry, buffer.selector);
	}
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		const auto selector =
		    component == 0u ? buffer.selector
		                    : EmitBitFieldUExtract(state, args[3], ConstantU32(state, component * 3u),
		                                           ConstantU32(state, 3));
		values[component] = LoadIndirectFormattedComponent(ctx, buffer, geometry, selector);
	}
	return ConstructU32Composite(state, components, values);
}

uint32_t LoadRawBuffer(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t components) {
	auto&      state  = ctx.state;
	const auto scalar = [&] {
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < components; component++) {
			values[component] =
			    LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
		}
		return ConstructU32Composite(state, components, values);
	};
	if (!mem.IsCoherentBufferPairLoad(inst.GetOpcode())) return scalar();

	const auto first  = ByteAddress(ctx, inst, mem, resource.byte_offset);
	const auto second = ByteAddress(ctx, inst, RebaseRawComponent(mem, 1), resource.byte_offset);
	const auto wide =
	    PrepareStorageBufferResourceAccess(state, mem, state.storage_buffers[3],
	                                       resource.non_uniform ? resource.descriptor_index : 0u);
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, ConstantU32(state, 3));
	auto atomic =
	    Binary(state, spv::OpIEqual, TypeBool(state),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 7)),
	           ConstantU32(state, 0));
	atomic = AndCondition(state, atomic,
	                      Binary(state, spv::OpIEqual, TypeBool(state), second,
	                             EmitAddU32(state, first, ConstantU32(state, 4))));
	atomic = AndCondition(state, atomic, EmitMemoryElementInBounds(state, wide, index));
	for (uint32_t component = 0; component < 2u; component++) {
		const auto word  = RebaseRawComponent(mem, component);
		const auto dword = DwordIndex(ctx, inst, word, resource.byte_offset);
		atomic = AndCondition(state, atomic,
		                      BufferIndexInBounds(ctx, inst, word, resource,
		                                          StorageBufferElementInBounds(state, word, resource, dword)));
	}
	const auto atomic_label = state.builder.AllocateId(), scalar_label = state.builder.AllocateId();
	const auto merge = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, atomic, atomic_label, scalar_label);
	EmitLabel(state, atomic_label);
	const auto pointer = EmitStorageBufferElementPointer(
	    state, wide, index, TypeStorageBufferElementPointer(state, 64));
	const auto snapshot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU64(state), snapshot, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsAcquireMask |
	                                                 spv::MemorySemanticsUniformMemoryMask));
	const std::array<uint32_t, 4> words {
	    EmitCompositeExtractU64(state, snapshot, IR::Value(0u)),
	    EmitCompositeExtractU64(state, snapshot, IR::Value(1u))};
	const auto pair = ConstructU32Composite(state, 2, words);
	state.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(state, scalar_label);
	const auto fallback    = scalar();
	const auto scalar_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(state, merge);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32Composite(state, 2), result, pair, atomic_label,
	                          fallback, scalar_exit);
	return result;
}

bool UsesRuntimeFormat(const EmitterState& state, const IR::MemoryInfo& mem) {
	if (state.program.generic_graphics)
		return mem.kind == IR::ResourceKind::Buffer && mem.formatted && !mem.typed;
	return state.program.table_mode && mem.kind == IR::ResourceKind::Buffer && mem.formatted && !mem.typed &&
	       mem.resource < 32u && ((state.program.table_plan.runtime_formats >> mem.resource) & 1u) != 0 &&
	       mem.resource < state.table_buffer_words.size() && state.table_buffer_words[mem.resource] != 0u;
}

uint32_t LoadRuntimeFormatted(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                              const MemoryResourceAccess& resource, uint32_t components) {
	auto&      state = ctx.state;
	const auto u32   = TypeU32(state);
	const auto word  = state.program.generic_graphics
	    ? EmitShaderDataDwordLoad(state, state.program.bindings.FormatDword() +
	                                   state.program.info.buffers[mem.resource].descriptor_index)
	    : state.table_buffer_words[mem.resource];
	const auto bits  = [&](uint32_t value, uint32_t offset, uint32_t count) {
		return EmitBitFieldUExtract(state, value, ConstantU32(state, offset), ConstantU32(state, count));
	};
	const auto format   = bits(word, 12, 7);
	const auto geometry = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU32Composite(state, 4), geometry,
	                          state.indirect_format_geometry_function, format);
	const auto field = [&](uint32_t index) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, u32, value, geometry, index);
		return value;
	};
	const auto bytes   = field(0);
	const auto unknown = Binary(state, spv::OpIEqual, TypeBool(state), bytes, ConstantU32(state, 0));
	const auto start = ByteAddress(ctx, inst, mem, resource.byte_offset);
	const auto end   = Binary(state, spv::OpIAdd, u32, start,
	                          Binary(state, spv::OpISub, u32,
	                                 Select(state, u32, unknown, ConstantU32(state, 1), bytes), ConstantU32(state, 1)));
	const auto in_bounds = BufferIndexInBounds(
	    ctx, inst, mem, resource,
	    AndCondition(state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), end, start),
	                 EmitMemoryElementInBounds(
	                     state, resource, Binary(state, spv::OpShiftRightLogical, u32, end, ConstantU32(state, 2)))),
	    bytes);
	const auto base = Binary(state, spv::OpBitwiseAnd, u32, start, Unary(state, spv::OpNot, u32, field(3)));
	std::array<uint32_t, 4> values {};
	for (uint32_t output = 0; output < components; output++) {
		const auto selector  = bits(word, output * 3u, 3);
		const auto component = Binary(state, spv::OpUMod, u32,
		                              Binary(state, spv::OpISub, u32, selector, ConstantU32(state, 4)), field(1));
		const auto address   = Binary(state, spv::OpIAdd, u32, base, Binary(state, spv::OpIMul, u32, component, field(2)));
		const auto memory    = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), selector, ConstantU32(state, 4));
		const auto dword     = EmitValueOrZeroIfCondition(state, AndCondition(state, memory, in_bounds), [&]() {
			return LoadWordInBounds(ctx, resource, Binary(state, spv::OpShiftRightLogical, u32, address, ConstantU32(state, 2)));
		});
		const auto raw = Binary(state, spv::OpShiftRightLogical, u32, dword,
		                        Binary(state, spv::OpShiftLeftLogical, u32,
		                               Binary(state, spv::OpBitwiseAnd, u32, address, ConstantU32(state, 3)),
		                               ConstantU32(state, 3)));
		const auto decoded = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, u32, decoded, state.indirect_format_function, format, selector,
		                          component, raw);
		const auto plain = LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, output), resource);
		values[output]   = Select(state, u32, unknown, plain, decoded);
	}
	return ConstructU32Composite(state, components, values);
}

uint32_t LoadBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	const auto mem = ctx.Memory(inst);
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		const auto function = state.indirect_buffer_load_functions.at(
		    {mem.data_bits, components, uint32_t(mem.coherent), uint32_t(mem.formatted)});
		auto args = IndirectBufferArgs(ctx, inst);
		ApplyIndirectTypedFormat(state, mem, args);
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, TypeU32Composite(state, components), result,
		                          function, args, ctx.Arg(inst, inst.NumArgs() - 1));
		return result;
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    return EmitBufferAccess(ctx, inst, mem, components, [&](const auto& selected, const auto& resource) {
			    const auto info = Format::GetFormatInfo(
			        selected.formatted ? BufferFormat(ctx, selected) : Prospero::BufferFormat::kInvalid);
			    if (info.type != Format::ComponentType::Unknown) {
				    return LoadFormattedPrepared(ctx, inst, selected, resource, info, components);
			    }
			    if (resource.object_pointer == 0u) return ConstantU32CompositeZero(state, components);
			    if (UsesRuntimeFormat(state, selected)) {
				    return LoadRuntimeFormatted(ctx, inst, selected, resource, components);
			    }
			    return LoadRawBuffer(ctx, inst, selected, resource, components);
		    });
	    });
}

static uint32_t ExtractU32(EmitterState& state, uint32_t composite, uint32_t component) {
	const auto data = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), data, composite, component);
	return data;
}

template <typename Scalar>
static void StoreRawBufferPair(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                               const MemoryResourceAccess& resource, uint32_t composite, Scalar&& scalar) {
	auto&      state  = ctx.state;
	const auto first  = ByteAddress(ctx, inst, mem, resource.byte_offset);
	const auto second = ByteAddress(ctx, inst, RebaseRawComponent(mem, 1), resource.byte_offset);
	const auto wide   = PrepareStorageBufferResourceAccess(state, mem, state.storage_buffers[3],
	                                                       resource.non_uniform ? resource.descriptor_index : 0u);
	const auto index  = Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, ConstantU32(state, 3));
	auto       atomic = Binary(state, spv::OpIEqual, TypeBool(state),
	                           Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 7)),
	                           ConstantU32(state, 0));
	atomic = AndCondition(state, atomic,
	                      Binary(state, spv::OpIEqual, TypeBool(state), second,
	                             EmitAddU32(state, first, ConstantU32(state, 4))));
	atomic = AndCondition(state, atomic, EmitMemoryElementInBounds(state, wide, index));
	for (uint32_t component = 0; component < 2u; component++) {
		const auto word  = RebaseRawComponent(mem, component);
		const auto dword = DwordIndex(ctx, inst, word, resource.byte_offset);
		atomic = AndCondition(state, atomic,
		                      BufferIndexInBounds(ctx, inst, word, resource,
		                                          StorageBufferElementInBounds(state, word, resource, dword)));
	}
	const auto atomic_label = state.builder.AllocateId(), scalar_label = state.builder.AllocateId();
	const auto merge        = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, atomic, atomic_label, scalar_label);
	EmitLabel(state, atomic_label);
	const auto pointer = EmitStorageBufferElementPointer(state, wide, index, TypeStorageBufferElementPointer(state, 64));
	const auto low     = Unary(state, spv::OpUConvert, TypeU64(state), ExtractU32(state, composite, 0u));
	const auto high    = Unary(state, spv::OpUConvert, TypeU64(state), ExtractU32(state, composite, 1u));
	const auto value   = Binary(state, spv::OpBitwiseOr, TypeU64(state), low,
	                            Binary(state, spv::OpShiftLeftLogical, TypeU64(state), high, ConstantU32(state, 32)));
	state.builder.AddFunction(spv::OpAtomicStore, pointer, ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsReleaseMask |
	                                                 spv::MemorySemanticsUniformMemoryMask),
	                          value);
	state.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(state, scalar_label);
	scalar();
	state.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(state, merge);
}

void StoreBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem = ctx.Memory(inst);
		const auto composite = ctx.Arg(inst, inst.NumArgs() - 2);
		EmitBufferAccess(ctx, inst, mem, 0u, [&](const auto& selected, const auto& resource) {
			const auto info = Format::GetFormatInfo(
			    selected.formatted ? BufferFormat(ctx, selected) : Prospero::BufferFormat::kInvalid);
			if (info.type != Format::ComponentType::Unknown) {
				StoreFormattedPrepared(ctx, inst, selected, resource, info, composite, components);
				return;
			}
			if (selected.IsCoherentBufferPairStore(inst.GetOpcode()) && resource.object_pointer != 0u) {
				StoreRawBufferPair(ctx, inst, selected, resource, composite, [&]() {
					for (uint32_t component = 0; component < components; component++) {
						const auto data = ExtractU32(state, composite, component);
						StoreWordPrepared(ctx, inst, RebaseRawComponent(selected, component), resource, data);
					}
				});
				return;
			}
			for (uint32_t component = 0; component < components; component++) {
				auto data = composite;
				if (components != 1u) {
					data = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), data, composite,
					                          component);
				}
				StoreWordPrepared(ctx, inst, RebaseRawComponent(selected, component), resource, data);
			}
		});
	});
}

uint32_t LoadWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto              mem      = ctx.Memory(inst);
		    const auto              resource = PrepareMemoryResourceAccess(state, mem);
		    const auto              base     = ByteAddress(ctx, inst, mem);
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    const auto address   = component == 0u
			                               ? base
			                               : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                        ConstantU32(state, component * 4u));
			    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
			                                  address, ConstantU32(state, 2));
			    values[component]    = EmitValueOrZeroIfCondition(
			        state, EmitMemoryElementInBounds(state, resource, index),
			        [&]() { return LoadWordInBounds(ctx, resource, index); });
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem      = ctx.Memory(inst);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto base     = ByteAddress(ctx, inst, mem);
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                              ConstantU32(state, 2));
			EmitIfCondition(state, EmitMemoryElementInBounds(state, resource, index),
			                [&]() {
				                StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, component + 1u));
			                });
		}
	});
}

struct IndirectFormatField {
	uint32_t offset = 0;
	uint32_t bits   = 0;
};

constexpr IndirectFormatField IndirectFormatWidth {0, 6};
constexpr IndirectFormatField IndirectFormatOffset {6, 5};
constexpr IndirectFormatField IndirectFormatSigned {11, 1};
constexpr IndirectFormatField IndirectFormatConversion {12, 6};
constexpr IndirectFormatField IndirectFormatIntegerOne {18, 1};
constexpr IndirectFormatField IndirectFormatUnknown {19, 1};

struct IndirectFormatTable {
	uint32_t                              variable = 0;
	std::vector<Format::BufferFormatInfo> conversions;
};

uint32_t IndirectFormatEntry(const Format::BufferFormatInfo& info, uint32_t component,
                             std::vector<Format::BufferFormatInfo>& conversions) {
	const auto field = [](IndirectFormatField position, uint32_t value) {
		EXIT_IF(value >= (1u << position.bits));
		return value << position.offset;
	};
	if (info.type == Format::ComponentType::Unknown) {
		return field(IndirectFormatUnknown, 1);
	}
	const auto source = info.packed_bitfield && component < info.component_count ? component : 0u;
	const auto bits   = info.component_bits[source];
	const bool sized  = info.type == Format::ComponentType::Unorm || info.type == Format::ComponentType::Snorm ||
	                   info.type == Format::ComponentType::Float;
	Format::BufferFormatInfo conversion;
	conversion.type              = info.type;
	conversion.component_bits[0] = sized ? bits : 0u;
	const auto found = std::find_if(conversions.begin(), conversions.end(), [&](const auto& known) {
		return known.type == conversion.type && known.component_bits[0] == conversion.component_bits[0];
	});
	const auto index = static_cast<uint32_t>(found - conversions.begin());
	if (found == conversions.end()) {
		conversions.push_back(conversion);
	}
	const auto width  = info.packed_bitfield ? bits : bits == 32u ? 32u : bits == 8u ? 8u : 16u;
	const auto offset = info.packed_bitfield ? info.component_bit_offset[source] : 0u;
	return field(IndirectFormatWidth, width) | field(IndirectFormatOffset, offset) |
	       field(IndirectFormatSigned, IsSignedFormatComponent(info.type) ? 1u : 0u) |
	       field(IndirectFormatConversion, index) |
	       field(IndirectFormatIntegerOne, Format::FormattedConstantBits(info, FormattedSourceKind::One) == 1u ? 1u : 0u);
}

IndirectFormatTable DefineIndirectFormatTable(EmitterState& state, uint32_t format_count) {
	IndirectFormatTable   table;
	std::vector<uint32_t> constants;
	for (uint32_t format = 0; format < format_count; format++) {
		const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
		for (uint32_t component = 0; component < 4u; component++) {
			constants.push_back(ConstantU32(state, IndirectFormatEntry(info, component, table.conversions)));
		}
	}
	const auto array_type = state.builder.Type(spv::OpTypeArray, TypeU32(state),
	                                           ConstantU32(state, static_cast<uint32_t>(constants.size())));
	table.variable        = state.builder.DefineGlobalVariable(
	    TypePointer(state, spv::StorageClassPrivate, array_type), spv::StorageClassPrivate,
	    state.builder.Constant(spv::OpConstantComposite, array_type, constants));
	return table;
}

void DefineIndirectBufferFormats(ValueEmitContext& ctx) {
	auto& state = ctx.state;
	const auto type = TypeU32(state);
	constexpr auto format_count =
	    static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u;
	{
		const auto result_type = TypeU32Composite(state, 4);
		const auto function_type = state.builder.Type(spv::OpTypeFunction, result_type, type);
		state.indirect_format_geometry_function = state.builder.AllocateId();
		const auto format = state.builder.AllocateId();
		state.builder.AddName(state.indirect_format_geometry_function,
		                      "indirect_buffer_format_geometry");
		state.builder.AddFunction(spv::OpFunction, result_type,
		                          state.indirect_format_geometry_function,
		                          spv::FunctionControlMaskNone, function_type);
		state.builder.AddFunction(spv::OpFunctionParameter, type, format);
		EmitLabel(state, state.builder.AllocateId());
		const auto result = EmitIndexSwitch(
		    state, format, format_count, result_type, [&](uint32_t format) {
			    const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
			    return state.builder.Constant(
			        spv::OpConstantComposite, result_type, ConstantU32(state, info.byte_size),
			        ConstantU32(state, std::max(1u, info.component_count)),
			        ConstantU32(state, info.packed_bitfield ? 0u : info.component_bits[0] / 8u),
			        ConstantU32(state, info.byte_size == 0u ? 0u : std::min(4u, info.byte_size) - 1u));
		    });
		state.builder.AddFunction(spv::OpReturnValue, result);
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
	const auto table = DefineIndirectFormatTable(state, format_count);
	const auto function_type = state.builder.Type(spv::OpTypeFunction, type, type, type, type, type);
	state.indirect_format_function = state.builder.AllocateId();
	const auto format = state.builder.AllocateId();
	const auto selector = state.builder.AllocateId();
	const auto component = state.builder.AllocateId();
	const auto raw = state.builder.AllocateId();
	state.builder.AddName(state.indirect_format_function, "convert_indirect_buffer_format");
	state.builder.AddFunction(spv::OpFunction, type, state.indirect_format_function,
	                          SharedHelperControl(state), function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, type, format);
	state.builder.AddFunction(spv::OpFunctionParameter, type, selector);
	state.builder.AddFunction(spv::OpFunctionParameter, type, component);
	state.builder.AddFunction(spv::OpFunctionParameter, type, raw);
	EmitLabel(state, state.builder.AllocateId());
	const auto b     = TypeBool(state);
	const auto below = [&](uint32_t value, uint32_t limit) {
		return Binary(state, spv::OpULessThan, b, value, ConstantU32(state, limit));
	};
	const auto known = Select(state, type, below(format, format_count), format, ConstantU32(state, 0));
	const auto slot  = Select(state, type, below(component, 4), component, ConstantU32(state, 0));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassPrivate, type), pointer,
	                          table.variable,
	                          Binary(state, spv::OpIAdd, type,
	                                 Binary(state, spv::OpShiftLeftLogical, type, known, ConstantU32(state, 2)), slot));
	const auto entry = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, entry, pointer);
	const auto field = [&](IndirectFormatField position) {
		return EmitBitFieldUExtract(state, entry, ConstantU32(state, position.offset), ConstantU32(state, position.bits));
	};
	const auto flag = [&](IndirectFormatField position) {
		return Binary(state, spv::OpINotEqual, b, field(position), ConstantU32(state, 0));
	};
	const auto one      = Select(state, type, flag(IndirectFormatIntegerOne), ConstantU32(state, 1),
	                             ConstantU32(state, 0x3f800000u));
	const auto constant = Select(state, type, Binary(state, spv::OpIEqual, b, selector, ConstantU32(state, 1)), one,
	                             ConstantU32(state, 0));
	const auto read     = Binary(state, spv::OpUGreaterThanEqual, b, selector, ConstantU32(state, 4));
	const auto value    = EmitValueOrDefaultIfCondition(state, read, type, constant, [&]() {
		const auto width          = field(IndirectFormatWidth);
		const auto offset         = field(IndirectFormatOffset);
		const auto unsigned_value = EmitBitFieldUExtract(state, raw, offset, width);
		const auto signed_value   = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), signed_value,
		                          Unary(state, spv::OpBitcast, TypeI32(state), raw), offset, width);
		const auto extracted = Select(state, type, flag(IndirectFormatSigned),
		                              Unary(state, spv::OpBitcast, type, signed_value), unsigned_value);
		return EmitIndexSwitch(state, field(IndirectFormatConversion),
		                       static_cast<uint32_t>(table.conversions.size()), type, [&](uint32_t index) {
			                       return NormalizeFormatComponent(state, table.conversions[index], 0u, extracted);
		                       });
	});
	const auto result = Select(state, type, flag(IndirectFormatUnknown), ConstantU32(state, 0), value);
	state.builder.AddFunction(spv::OpReturnValue, result);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

}

uint32_t GetBdaPointer(EmitterState& state, uint32_t address, uint32_t extent) {
	const auto result = state.builder.AllocateId();
	if (extent == sizeof(uint32_t)) {
		if (state.bda_dword_function == 0u) state.bda_dword_function = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, TypeU64(state), result, state.bda_dword_function, address);
		return result;
	}
	state.builder.AddFunction(spv::OpFunctionCall, TypeU64(state), result,
	                          state.bda_pointer_function, address, ConstantU32(state, extent));
	return result;
}

void DefineIndirectBufferLoads(ValueEmitContext& ctx) {
	auto& state = ctx.state;
	if (state.program.generic_graphics && std::ranges::any_of(state.program.info.buffers, &IR::BufferResource::formatted)) {
		DefineIndirectBufferFormats(ctx);
	}
	if (state.program.table_mode && state.indirect_format_function == 0u) {
		const auto& buffers = state.program.info.buffers;
		for (size_t i = 0; i < buffers.size() && i < 32; ++i) {
			if (((state.program.table_plan.runtime_formats >> i) & 1u) != 0 &&
			    buffers[i].descriptor_format == Prospero::BufferFormat::kInvalid) {
				DefineIndirectBufferFormats(ctx);
				break;
			}
		}
	}
	if (state.program.info.ray_query) {
		for (const auto* block: state.program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() == IR::ValueOpcode::RayQueryAlpha && state.indirect_format_function == 0u) {
					DefineIndirectBufferFormats(ctx);
				}
			}
		}
	}
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			const auto components = IR::BufferComponentCount(inst.GetOpcode());
			if (components == 0u || inst.GetOpcode() == IR::ValueOpcode::ReadConstBuffer) continue;
			const auto& memory = ctx.Memory(inst);
			if (memory.kind != IR::ResourceKind::IndirectBuffer ||
			    !memory.SupportsIndirectBufferLoad(inst.GetOpcode())) continue;
			auto [entry, inserted] = state.indirect_buffer_load_functions.try_emplace(
			    std::array {memory.data_bits, components, uint32_t(memory.coherent),
			                uint32_t(memory.formatted)}, 0u);
			if (!inserted) continue;
			if (memory.formatted && state.indirect_format_function == 0u) {
				DefineIndirectBufferFormats(ctx);
			}
			const auto function = entry->second = state.builder.AllocateId();
			const auto type = TypeU32(state);
			const auto result_type = TypeU32Composite(state, components);
			const auto signature = state.builder.Type(
			    spv::OpTypeFunction, result_type, type, type, type, type, type, type, type, type,
			    TypeBool(state));
			const auto name = "load_indirect_buffer_" + std::to_string(memory.data_bits) + "x" +
			                  std::to_string(components) + (memory.coherent ? "_coherent" : "") +
			                  (memory.formatted ? "_formatted" : "");
			state.builder.AddName(function, name.c_str());
			state.builder.AddFunction(spv::OpFunction, result_type, function,
			                          SharedHelperControl(state), signature);
			std::array<uint32_t, 8> args;
			for (auto& arg: args) {
				arg = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFunctionParameter, type, arg);
			}
			const auto active = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionParameter, TypeBool(state), active);
			EmitLabel(state, state.builder.AllocateId());
			args[3] = Select(state, type, active, args[3],
			                 Binary(state, spv::OpBitwiseAnd, type, args[3], ConstantU32(state, ~(0x7fu << 12))));
			const auto buffer = PrepareIndirectBuffer(state, args, memory.formatted);
			const auto result = memory.formatted ? LoadIndirectFormatted(ctx, buffer, args, components)
			    : LoadIndirectBuffer(ctx, buffer, memory.data_bits, components, memory.coherent);
			std::array<uint32_t, 4> values {};
			for (uint32_t component = 0; component < components; component++) {
				values[component] = Select(state, type, active,
				                           components == 1u ? result : ExtractU32(state, result, component),
				                           ConstantU32(state, 0));
			}
			state.builder.AddFunction(spv::OpReturnValue, ConstructU32Composite(state, components, values));
			state.builder.AddFunction(spv::OpFunctionEnd);
		}
	}
}

void EmitLoopGuardMark(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.fault_buffer_variable == 0 || (ctx.other_half != nullptr && ctx.half != 0)) return;
	const auto one   = ConstantU32(state, 1);
	const auto zero  = ConstantU32(state, 0);
	auto       left  = ctx.Arg(inst, 0);
	auto       lanes = Select(state, TypeU32(state), left, one, zero);
	if (ctx.other_half != nullptr) {
		const auto other = ctx.other_half->Arg(inst, 0);
		lanes = Binary(state, spv::OpIAdd, TypeU32(state), lanes, Select(state, TypeU32(state), other, one, zero));
		left  = Binary(state, spv::OpLogicalOr, TypeBool(state), left, other);
	}
	EmitIfCondition(state, left, [&] {
		EmitRayMarkCount(state, BufferCache::HACK_MARK_COUNTS + static_cast<uint32_t>(HackMark::LoopGuard), lanes);
	});
}

uint32_t RayStatsEnabled(EmitterState& state) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.fault_buffer_variable, ConstantU32(state, 0),
	                          ConstantU32(state, BufferCache::RT_STATS_OFFSET / sizeof(uint32_t)));
	const auto enabled = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), enabled, pointer);
	return Binary(state, spv::OpINotEqual, TypeBool(state), enabled, ConstantU32(state, 0));
}

void EmitGuestRayCount(EmitterState& state, uint32_t count) {
	EmitIfCondition(state, RayStatsEnabled(state), [&] {
		EmitRayMarkCount(state, BufferCache::HACK_MARK_GUEST_RAYS, count);
	});
}

void EmitRayMarkCount(EmitterState& state, uint32_t counter, uint32_t count) {
	EXIT_IF(state.fault_buffer_variable == 0 || counter < BufferCache::HACK_MARK_COUNTS ||
	        counter >= BufferCache::HACK_MARK_DWORDS);
	const auto hash  = state.program.shader_hash;
	const auto first = BufferCache::HACK_MARKS_OFFSET / sizeof(uint32_t) +
	                   uint64_t {BufferCache::HackMarkEntry(hash)} * BufferCache::HACK_MARK_DWORDS;
	const auto low   = static_cast<uint32_t>(hash);
	const auto high  = static_cast<uint32_t>(hash >> 32u);
	const std::array<std::pair<uint32_t, uint32_t>, 5> updates {{
	    {0, ConstantU32(state, low)},
	    {1, ConstantU32(state, high)},
	    {2, ConstantU32(state, ~low)},
	    {3, ConstantU32(state, ~high)},
	    {counter, count},
	}};
	const auto pointer_type = TypeStorageBufferElementPointer(state);
	const auto scope        = ConstantU32(state, spv::ScopeDevice);
	const auto relaxed      = ConstantU32(state, spv::MemorySemanticsMaskNone);
	for (const auto& [dword, value]: updates) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer, state.fault_buffer_variable,
		                          ConstantU32(state, 0), ConstantU32(state, static_cast<uint32_t>(first + dword)));
		state.builder.AddFunction(dword < BufferCache::HACK_MARK_COUNTS ? spv::OpAtomicOr : spv::OpAtomicIAdd,
		                          TypeU32(state), state.builder.AllocateId(), pointer, scope, relaxed, value);
	}
}

namespace {

constexpr std::array<uint32_t, 4> MirrorMetadataOffsets {0, 8, 16, static_cast<uint32_t>(BufferCache::HOST_TABLE_OFFSET)};

uint32_t LoadFaultMetadata64(EmitterState& state, uint32_t offset) {
	const auto load = [&](uint32_t index) {
		const auto pointer = FaultElementPointer(state, ConstantU32(state, index));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	return PackU64(state, load(offset / 4), load(offset / 4 + 1));
}

uint32_t LoadPhysicalU64(EmitterState& state, uint32_t address) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr,
	                          TypePointer(state, spv::StorageClassPhysicalStorageBuffer, TypeU64(state)), pointer, address);
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU64(state), value, pointer, spv::MemoryAccessAlignedMask,
	                          static_cast<uint32_t>(sizeof(uint64_t)));
	return value;
}

}

void DeclareMirrorMetadata(EmitterState& state) {
	if (!state.program.info.uses_dma || state.fault_buffer_variable == 0) return;
	for (size_t i = 0; i < MirrorMetadataOffsets.size(); i++) {
		state.mirror_metadata[i] = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassPrivate, TypeU64(state)), spv::StorageClassPrivate);
	}
}

void StoreMirrorMetadata(EmitterState& state) {
	for (size_t i = 0; i < MirrorMetadataOffsets.size(); i++) {
		if (state.mirror_metadata[i] == 0) continue;
		state.builder.AddFunction(spv::OpStore, state.mirror_metadata[i],
		                          LoadFaultMetadata64(state, MirrorMetadataOffsets[i]));
	}
}

uint32_t LoadMirrorMetadata64(EmitterState& state, uint32_t offset) {
	for (size_t i = 0; i < MirrorMetadataOffsets.size(); i++) {
		if (MirrorMetadataOffsets[i] != offset || state.mirror_metadata[i] == 0) continue;
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU64(state), value, state.mirror_metadata[i]);
		return value;
	}
	return LoadFaultMetadata64(state, offset);
}

uint32_t EmitGuestHostAddress(EmitterState& state, uint32_t offset, uint32_t size, uint32_t extent, bool in_page) {
	using View = GuestHostView;
	const auto u64 = TypeU64(state), boolean = TypeBool(state);
	const auto zero = ConstantU64(state, 0);
	const auto load = [&](uint32_t pointer) { return LoadPhysicalU64(state, pointer); };
	const auto nonzero = [&](uint32_t value) { return Binary(state, spv::OpINotEqual, boolean, value, zero); };
	const auto slot = [&](uint32_t table, uint32_t index) {
		return Binary(state, spv::OpIAdd, u64, table,
		              Binary(state, spv::OpShiftLeftLogical, u64, index, ConstantU32(state, 3)));
	};
	const auto table = LoadMirrorMetadata64(state, BufferCache::HOST_TABLE_OFFSET);
	const auto entry_of = [&](uint32_t at, uint32_t wanted) {
		const auto present = wanted == ConstantBool(state, true) ? nonzero(table) : AndCondition(state, wanted, nonzero(table));
		const auto chunk = EmitValueOrDefaultIfCondition(state, present, u64, zero, [&] {
			return load(slot(table, Binary(state, spv::OpShiftRightLogical, u64, at, ConstantU32(state, View::ChunkBits))));
		});
		const auto page = Binary(state, spv::OpBitwiseAnd, u64,
		                         Binary(state, spv::OpShiftRightLogical, u64, at, ConstantU32(state, View::PageBits)),
		                         ConstantU64(state, View::EntriesPerChunk - 1));
		return EmitValueOrDefaultIfCondition(state, nonzero(chunk), u64, zero, [&] { return load(slot(chunk, page)); });
	};
	const auto entry = entry_of(offset, ConstantBool(state, true));
	const auto inside = Binary(state, spv::OpBitwiseAnd, u64, offset, ConstantU64(state, View::PageSize - 1));
	if (in_page) {
		return Select(state, u64, nonzero(entry), Binary(state, spv::OpIAdd, u64, entry, inside), zero);
	}
	const auto crosses = Binary(state, spv::OpUGreaterThan, boolean,
	                            Binary(state, spv::OpIAdd, u64, inside, Unary(state, spv::OpUConvert, u64, extent)),
	                            ConstantU64(state, View::PageSize));
	const auto after = Binary(state, spv::OpIAdd, u64, Binary(state, spv::OpISub, u64, offset, inside), ConstantU64(state, View::PageSize));
	const auto next = entry_of(after, AndCondition(state, AndCondition(state, nonzero(entry), crosses),
	                                               Binary(state, spv::OpULessThan, boolean, after, size)));
	const auto follows = Binary(state, spv::OpIEqual, boolean, next,
	                            Binary(state, spv::OpIAdd, u64, entry, ConstantU64(state, View::PageSize)));
	const auto usable = AndCondition(state, nonzero(entry),
	                                 Binary(state, spv::OpLogicalOr, boolean, Unary(state, spv::OpLogicalNot, boolean, crosses), follows));
	return Select(state, u64, usable, Binary(state, spv::OpIAdd, u64, entry, inside), zero);
}

uint32_t EmitGuestPageAddress(EmitterState& state, uint32_t offset) {
	using View = GuestHostView;
	const auto u64 = TypeU64(state), boolean = TypeBool(state);
	const auto zero = ConstantU64(state, 0);
	const auto slot = [&](uint32_t table, uint32_t index) {
		return Binary(state, spv::OpIAdd, u64, table, Binary(state, spv::OpShiftLeftLogical, u64, index, ConstantU32(state, 3)));
	};
	const auto table = LoadMirrorMetadata64(state, BufferCache::HOST_TABLE_OFFSET);
	const auto chunk = LoadPhysicalU64(state,
	                                   slot(table, Binary(state, spv::OpShiftRightLogical, u64, offset, ConstantU32(state, View::ChunkBits))));
	const auto present = Binary(state, spv::OpINotEqual, boolean, chunk, zero);
	const auto page = Binary(state, spv::OpBitwiseAnd, u64,
	                         Binary(state, spv::OpShiftRightLogical, u64, offset, ConstantU32(state, View::PageBits)),
	                         ConstantU64(state, View::EntriesPerChunk - 1));
	const auto entry = LoadPhysicalU64(state, Select(state, u64, present, slot(chunk, page), table));
	return Select(state, u64, AndCondition(state, present, Binary(state, spv::OpINotEqual, boolean, entry, zero)),
	              Binary(state, spv::OpIAdd, u64, entry,
	                     Binary(state, spv::OpBitwiseAnd, u64, offset, ConstantU64(state, View::PageSize - 1))),
	              zero);
}

uint32_t EmitMirrorDwordAddress(EmitterState& state, uint32_t address, uint32_t active) {
	using LibKernel::Memory::kExtendedMemoryBase;
	using LibKernel::Memory::kExtendedMemorySize;
	const auto u32 = TypeU32(state), u64 = TypeU64(state), boolean = TypeBool(state);
	const auto zero = ConstantU64(state, 0);
	const auto extended = Binary(state, spv::OpUGreaterThanEqual, boolean, address, ConstantU64(state, kExtendedMemoryBase));
	const auto packed = Select(state, u64, extended,
	                           Binary(state, spv::OpISub, u64, address, ConstantU64(state, kExtendedMemoryBase - LOWER_ADDRESS_SIZE)),
	                           address);
	const auto limit = Select(state, u64, extended, ConstantU64(state, LOWER_ADDRESS_SIZE + kExtendedMemorySize),
	                          ConstantU64(state, LOWER_ADDRESS_SIZE));
	const auto representable = Binary(state, spv::OpULessThan, boolean, packed, limit);
	const auto valid = active == ConstantBool(state, true) ? representable : AndCondition(state, active, representable);
	const auto block = Unary(state, spv::OpUConvert, u32,
	                         Binary(state, spv::OpShiftRightLogical, u64, Select(state, u64, valid, packed, zero),
	                                ConstantU32(state, BufferCache::MIRROR_BLOCK_BITS)));
	const auto entry = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, u32, entry, FaultElementPointer(state, Binary(state, spv::OpIAdd, u32,
	    ConstantU32(state, BufferCache::RESIDENCY_OFFSET / 4), Binary(state, spv::OpShiftRightLogical, u32, block, ConstantU32(state, 4)))));
	const auto resident = Binary(state, spv::OpINotEqual, boolean,
	    Binary(state, spv::OpBitwiseAnd, u32, entry, Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1),
	        Binary(state, spv::OpBitwiseAnd, u32, block, ConstantU32(state, BufferCache::RESIDENCY_BLOCKS - 1)))),
	    ConstantU32(state, 0));
	const auto ready = AndCondition(state, valid, resident);
	const auto index = Binary(state, spv::OpShiftRightLogical, u32, block, ConstantU32(state, 5));
	const auto bit = Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1),
	                        Binary(state, spv::OpBitwiseAnd, u32, block, ConstantU32(state, 31)));
	const auto word = [&](uint64_t bitmap) {
		return FaultElementPointer(state, Binary(state, spv::OpIAdd, u32, ConstantU32(state, bitmap / 4), index));
	};
	const auto scope = ConstantU32(state, spv::ScopeDevice);
	const auto relaxed = ConstantU32(state, spv::MemorySemanticsMaskNone);
	const auto use = word(BufferCache::USE_BITMAP_OFFSET);
	const auto used = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, u32, used, use, scope, relaxed);
	const auto unused = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, used, bit), ConstantU32(state, 0));
	EmitIfCondition(state, AndCondition(state, ready, unused), [&] {
		state.builder.AddFunction(spv::OpAtomicOr, u32, state.builder.AllocateId(), use, scope, relaxed, bit);
	});
	const auto miss = [&](uint32_t offset) {
		state.builder.AddFunction(spv::OpAtomicOr, u32, state.builder.AllocateId(), word(BufferCache::FAULT_BITMAP_OFFSET),
		                          scope, relaxed, bit);
		const auto guest = EmitGuestPageAddress(state, offset);
		state.builder.AddFunction(spv::OpAtomicIAdd, u32, state.builder.AllocateId(),
		                          FaultElementPointer(state, ConstantU32(state, BufferCache::HOST_ACCESS_OFFSET / 4)),
		                          scope, relaxed,
		                          Select(state, u32, Binary(state, spv::OpINotEqual, boolean, guest, zero),
		                                 ConstantU32(state, 1), ConstantU32(state, 0)));
		return guest;
	};
	const auto missed = AndCondition(state, valid, Unary(state, spv::OpLogicalNot, boolean, resident));
	const auto offset = Binary(state, spv::OpISub, u64, address, LoadMirrorMetadata64(state, 8));
	const auto inside = Binary(state, spv::OpULessThan, boolean, offset, LoadMirrorMetadata64(state, 16));
	const auto guest_page = EmitValueOrDefaultIfCondition(state, AndCondition(state, missed, inside), u64, zero,
	                                                      [&] { return miss(offset); });
	const auto translated = Binary(state, spv::OpBitwiseOr, u64,
	    Binary(state, spv::OpShiftLeftLogical, u64,
	           Unary(state, spv::OpUConvert, u64, Binary(state, spv::OpShiftRightLogical, u32, entry, ConstantU32(state, BufferCache::RESIDENCY_BLOCKS))),
	           ConstantU32(state, GuestMirror::WindowBits)),
	    Binary(state, spv::OpBitwiseAnd, u64, address, ConstantU64(state, GuestMirror::WindowSize - 1)));
	return Select(state, u64, ready, Binary(state, spv::OpIAdd, u64, LoadMirrorMetadata64(state, 0), translated), guest_page);
}

uint32_t EmitMirrorDeviceAddress(EmitterState& state, uint32_t address, uint32_t active, uint32_t extent, bool in_page) {
	if (in_page && extent == ConstantU32(state, sizeof(uint32_t))) return EmitMirrorDwordAddress(state, address, active);
	const auto u32 = TypeU32(state), u64 = TypeU64(state), boolean = TypeBool(state);
	const auto mirror_base = LoadMirrorMetadata64(state, 0);
	const auto guest_base = LoadMirrorMetadata64(state, 8);
	const auto size = LoadMirrorMetadata64(state, 16);
	const auto offset = Binary(state, spv::OpISub, u64, address, guest_base);
	const auto valid = AndCondition(state, active, BdaAddressRepresentable(state, address));
	const auto zero = ConstantU64(state, 0);
	const auto span = Unary(state, spv::OpUConvert, u64, extent);
	const auto last_byte = Binary(state, spv::OpIAdd, u64, offset,
	                              Binary(state, spv::OpISub, u64, span, ConstantU64(state, 1)));
	const auto last_address = Binary(state, spv::OpIAdd, u64, guest_base, last_byte);
	const auto end_valid = in_page ? valid : AndCondition(state, valid, BdaAddressRepresentable(state, last_address));
	const auto inside = AndCondition(state, end_valid, AndCondition(state,
	    Binary(state, spv::OpULessThan, boolean, offset, size),
	    Binary(state, spv::OpULessThan, boolean, last_byte, size)));
	const auto block = BdaPageIndex(state, Select(state, u64, inside, address, zero), BufferCache::MIRROR_BLOCK_BITS);
	const auto record = [&](uint32_t page) {
		const auto index = Binary(state, spv::OpShiftRightLogical, u32, page, ConstantU32(state, 4));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, u32, value, FaultElementPointer(state, Binary(state, spv::OpIAdd, u32,
		    ConstantU32(state, BufferCache::RESIDENCY_OFFSET / 4), index)));
		return value;
	};
	const auto resident = [&](uint32_t entry, uint32_t page) {
		const auto bit = Binary(state, spv::OpBitwiseAnd, u32, page, ConstantU32(state, BufferCache::RESIDENCY_BLOCKS - 1));
		const auto mask = Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1), bit);
		return Binary(state, spv::OpINotEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, entry, mask), ConstantU32(state, 0));
	};
	const auto translate = [&](uint32_t entry, uint32_t at) {
		const auto slot = Binary(state, spv::OpShiftRightLogical, u32, entry, ConstantU32(state, BufferCache::RESIDENCY_BLOCKS));
		return Binary(state, spv::OpBitwiseOr, u64,
		    Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, slot), ConstantU32(state, GuestMirror::WindowBits)),
		    Binary(state, spv::OpBitwiseAnd, u64, at, ConstantU64(state, GuestMirror::WindowSize - 1)));
	};
	const auto entry = record(block);
	const auto translated = translate(entry, offset);
	const auto index = Binary(state, spv::OpShiftRightLogical, u32, block, ConstantU32(state, 5));
	const auto bit = Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1),
	                        Binary(state, spv::OpBitwiseAnd, u32, block, ConstantU32(state, 31)));
	const auto word = [&](uint64_t bitmap) {
		return FaultElementPointer(state, Binary(state, spv::OpIAdd, u32, ConstantU32(state, bitmap / 4), index));
	};
	const auto mark = [&](uint32_t pointer) {
		state.builder.AddFunction(spv::OpAtomicOr, u32, state.builder.AllocateId(), pointer,
		                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone),
		                          bit);
	};
	const auto first_ready = AndCondition(state, inside, resident(entry, block));
	auto ready = first_ready;
	auto has_resident = first_ready;
	if (!in_page) {
		const auto crosses = Binary(state, spv::OpINotEqual, boolean,
		    Binary(state, spv::OpShiftRightLogical, u64, offset, ConstantU32(state, BufferCache::MIRROR_BLOCK_BITS)),
		    Binary(state, spv::OpShiftRightLogical, u64, last_byte, ConstantU32(state, BufferCache::MIRROR_BLOCK_BITS)));
		const auto tail = EmitValueOrDefaultIfCondition(state, AndCondition(state, inside, crosses), u32, ConstantU32(state, 0), [&] {
			const auto last_block = BdaPageIndex(state, last_address, BufferCache::MIRROR_BLOCK_BITS);
			const auto last_entry = record(last_block);
			const auto last_translated = translate(last_entry, last_byte);
			const auto contiguous = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpISub, u64, last_translated, translated),
			                               Binary(state, spv::OpISub, u64, last_byte, offset));
			const auto last_ready = resident(last_entry, last_block);
			EmitIfCondition(state, AndCondition(state, last_ready, Unary(state, spv::OpLogicalNot, boolean, first_ready)),
			                [&] { mark(word(BufferCache::FAULT_BITMAP_OFFSET)); });
			EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, last_ready),
			                [&] { RecordBdaPageBit(state, last_block, BufferCache::FAULT_BITMAP_OFFSET); });
			EmitIfCondition(state, AndCondition(state, first_ready, AndCondition(state, contiguous, last_ready)),
			                [&] { RecordBdaPageBit(state, last_block, BufferCache::USE_BITMAP_OFFSET); });
			return Select(state, u32, last_ready, Select(state, u32, contiguous, ConstantU32(state, 2), ConstantU32(state, 1)), ConstantU32(state, 0));
		});
		ready = AndCondition(state, ready, Binary(state, spv::OpLogicalOr, boolean,
		    Unary(state, spv::OpLogicalNot, boolean, crosses), Binary(state, spv::OpIEqual, boolean, tail, ConstantU32(state, 2))));
		has_resident = Binary(state, spv::OpLogicalOr, boolean, first_ready, Binary(state, spv::OpINotEqual, boolean, tail, ConstantU32(state, 0)));
	}
	const auto use = word(BufferCache::USE_BITMAP_OFFSET);
	const auto used = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, u32, used, use, ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone));
	const auto unused = Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, used, bit), ConstantU32(state, 0));
	EmitIfCondition(state, AndCondition(state, ready, unused), [&] { mark(use); });
	const auto missed = AndCondition(state, inside, Unary(state, spv::OpLogicalNot, boolean, has_resident));
	const auto guest_page = EmitValueOrDefaultIfCondition(state, missed, u64, ConstantU64(state, 0), [&] {
		mark(word(BufferCache::FAULT_BITMAP_OFFSET));
		const auto guest = EmitGuestHostAddress(state, offset, size, extent, in_page);
		EmitIfCondition(state, Binary(state, spv::OpINotEqual, boolean, guest, ConstantU64(state, 0)), [&] {
			state.builder.AddFunction(spv::OpAtomicIAdd, u32, state.builder.AllocateId(),
			                          FaultElementPointer(state, ConstantU32(state, BufferCache::HOST_ACCESS_OFFSET / 4)),
			                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone),
			                          ConstantU32(state, 1));
		});
		return guest;
	});
	return Select(state, u64, ready, Binary(state, spv::OpIAdd, u64, mirror_base, translated), guest_page);
}

void DefineGetBdaPointerMasked(EmitterState& state) {
	const auto type          = TypeU64(state);
	const auto function_type = state.builder.Type(spv::OpTypeFunction, type, type, TypeBool(state), TypeU32(state));
	state.bda_masked_function = state.builder.AllocateId();
	const auto address        = state.builder.AllocateId();
	const auto active         = state.builder.AllocateId();
	const auto extent         = state.builder.AllocateId();
	state.builder.AddName(state.bda_masked_function, "get_bda_pointer_masked");
	state.builder.AddFunction(spv::OpFunction, type, state.bda_masked_function, SharedHelperControl(state),
	                          function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, type, address);
	state.builder.AddFunction(spv::OpFunctionParameter, TypeBool(state), active);
	state.builder.AddFunction(spv::OpFunctionParameter, TypeU32(state), extent);
	EmitLabel(state, state.builder.AllocateId());
	state.builder.AddFunction(spv::OpReturnValue, EmitMirrorDeviceAddress(state, address, active, extent));
	state.builder.AddFunction(spv::OpFunctionEnd);
}

void DefineBdaDwordPointers(EmitterState& state) {
	const auto u64 = TypeU64(state), boolean = TypeBool(state);
	const auto define = [&](uint32_t function, bool masked) {
		if (function == 0u) return;
		const auto function_type = masked ? state.builder.Type(spv::OpTypeFunction, u64, u64, boolean)
		                                  : state.builder.Type(spv::OpTypeFunction, u64, u64);
		state.builder.AddName(function, masked ? "get_bda_pointer_dword_masked" : "get_bda_pointer_dword");
		state.builder.AddFunction(spv::OpFunction, u64, function, SharedHelperControl(state), function_type);
		const auto address = state.builder.AllocateId();
		const auto active  = masked ? state.builder.AllocateId() : ConstantBool(state, true);
		state.builder.AddFunction(spv::OpFunctionParameter, u64, address);
		if (masked) state.builder.AddFunction(spv::OpFunctionParameter, boolean, active);
		EmitLabel(state, state.builder.AllocateId());
		state.builder.AddFunction(spv::OpReturnValue, EmitMirrorDeviceAddress(state, address, active,
		                                                                      ConstantU32(state, sizeof(uint32_t)), true));
		state.builder.AddFunction(spv::OpFunctionEnd);
	};
	define(state.bda_dword_function, false);
	define(state.bda_dword_masked_function, true);
}

void DefineBdaLoads(EmitterState& state) {
	state.current_block = nullptr;
	DefineBdaDwordPointers(state);
	if (state.bda_load_functions.empty()) return;
	const auto u32 = TypeU32(state), u64 = TypeU64(state), boolean = TypeBool(state);
	for (const auto& [key, function]: state.bda_load_functions) {
		const auto [coherent, guarded] = key;
		{
			const auto function_type = guarded != 0u
			    ? state.builder.Type(spv::OpTypeFunction, u32, boolean, u64)
			    : state.builder.Type(spv::OpTypeFunction, u32, u64);
			const auto name = std::string("bda_load_u32") + (coherent != 0u ? "_coherent" : "") +
			                  (guarded != 0u ? "_g" : "");
			state.builder.AddName(function, name.c_str());
			state.builder.AddFunction(spv::OpFunction, u32, function, SharedHelperControl(state),
			                          function_type);
			const auto condition = guarded != 0u ? state.builder.AllocateId() : 0u;
			const auto address   = state.builder.AllocateId();
			if (guarded != 0u) state.builder.AddFunction(spv::OpFunctionParameter, boolean, condition);
			state.builder.AddFunction(spv::OpFunctionParameter, u64, address);
			EmitLabel(state, state.builder.AllocateId());
			const auto body = [&]() {
				const auto host = EmitMirrorDeviceAddress(state, address, ConstantBool(state, true),
				                                          ConstantU32(state, sizeof(uint32_t)), true);
				const auto present = Binary(state, spv::OpINotEqual, boolean, host, ConstantU64(state, 0));
				return EmitValueOrDefaultIfCondition(state, present, u32, MissingBdaDword(state),
				    [&] { return EmitBdaPointerLoad(state, host, coherent != 0u); });
			};
			const auto result = guarded != 0u
			    ? EmitValueOrDefaultIfCondition(state, condition, u32, ConstantU32(state, 0), body)
			    : body();
			state.builder.AddFunction(spv::OpReturnValue, result);
			state.builder.AddFunction(spv::OpFunctionEnd);
		}
	}
}

void DefineGetBdaPointer(EmitterState& state) {
	if (!state.program.info.uses_dma) {
		return;
	}
	const auto type            = TypeU64(state);
	const auto function_type   = state.builder.Type(spv::OpTypeFunction, type, type, TypeU32(state));
	state.bda_pointer_function = state.builder.AllocateId();
	const auto address         = state.builder.AllocateId();
	const auto extent          = state.builder.AllocateId();
	const auto entry_label     = state.builder.AllocateId();
	state.builder.AddName(state.bda_pointer_function, "get_bda_pointer");
	state.builder.AddFunction(spv::OpFunction, type, state.bda_pointer_function,
	                          SharedHelperControl(state), function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, type, address);
	state.builder.AddFunction(spv::OpFunctionParameter, TypeU32(state), extent);
	EmitLabel(state, entry_label);

	state.builder.AddFunction(spv::OpReturnValue, EmitMirrorDeviceAddress(state, address, ConstantBool(state, true), extent));
	state.builder.AddFunction(spv::OpFunctionEnd);
	if (state.converged_calls) {
		DefineGetBdaPointerMasked(state);
	}
}

uint32_t GetBdaPointerMasked(EmitterState& state, uint32_t address, uint32_t active, uint32_t extent) {
	const auto result = state.builder.AllocateId();
	if (extent == sizeof(uint32_t)) {
		if (state.bda_dword_masked_function == 0u) state.bda_dword_masked_function = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, TypeU64(state), result, state.bda_dword_masked_function, address,
		                          active);
		return result;
	}
	state.builder.AddFunction(spv::OpFunctionCall, TypeU64(state), result, state.bda_masked_function, address,
	                          active, ConstantU32(state, extent));
	return result;
}

uint32_t EmitGuestDwordLoad(ValueEmitContext& ctx, uint32_t address) {
	return LoadBdaDword(ctx, Binary(ctx.state, spv::OpBitwiseAnd, TypeU64(ctx.state), address,
	                                ConstantU64(ctx.state, ~uint64_t {3})));
}

GuestObject EmitGuestObject(ValueEmitContext& ctx, uint32_t address, uint32_t size) {
	auto&      state  = ctx.state;
	const auto wide   = TypeU64(state);
	const auto inside = Binary(
	    state, spv::OpULessThanEqual, TypeBool(state),
	    Binary(state, spv::OpIAdd, wide,
	           Binary(state, spv::OpBitwiseAnd, wide, address, ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1)),
	           ConstantU64(state, size)),
	    ConstantU64(state, BufferCache::CACHING_PAGESIZE));
	const auto host = EmitValueOrDefaultIfCondition(state, inside, wide, ConstantU64(state, 0),
	                                                [&]() { return GetBdaPointer(state, address, size); });
	return {address, host, inside};
}

uint32_t EmitGuestObjectDword(ValueEmitContext& ctx, const GuestObject& object, uint32_t offset) {
	return EmitGuestObjectDwordAt(ctx, object, ConstantU32(ctx.state, offset & ~3u));
}

uint32_t EmitGuestObjectDwordAt(ValueEmitContext& ctx, const GuestObject& object, uint32_t offset_id) {
	auto&      state = ctx.state;
	const auto wide  = TypeU64(state);
	const auto offset64 = Unary(state, spv::OpUConvert, wide, offset_id);
	const auto fast  = EmitValueOrZeroIfCondition(
	    state,
	    AndCondition(state, object.inside,
	                 Binary(state, spv::OpINotEqual, TypeBool(state), object.host, ConstantU64(state, 0))),
	    [&]() {
		    const auto pointer = state.builder.AllocateId();
		    state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer,
		                              Binary(state, spv::OpIAdd, wide, object.host, offset64));
		    const auto value = state.builder.AllocateId();
		    state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, spv::MemoryAccessAlignedMask,
		                              uint32_t {sizeof(uint32_t)});
		    return value;
	    });
	const auto slow = EmitValueOrZeroIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), object.inside), [&]() {
		return EmitGuestDwordLoad(ctx, Binary(state, spv::OpIAdd, wide, object.address, offset64));
	});
	return Select(state, TypeU32(state), object.inside, fast, slow);
}

void EmitScalarBufferObject(ValueEmitContext& ctx, const std::array<uint32_t, 4>& vsharp, uint32_t offset,
                            uint32_t size, ScalarBufferObject& object) {
	auto&      state  = ctx.state;
	const auto wide   = TypeU64(state);
	const auto stride = EmitBitFieldUExtract(state, vsharp[1], ConstantU32(state, 16), ConstantU32(state, 14));
	object.size = Binary(state, spv::OpIMul, wide,
	                     Unary(state, spv::OpUConvert, wide, EmitUMax32(state, stride, ConstantU32(state, 1))),
	                     Unary(state, spv::OpUConvert, wide, vsharp[2]));
	object.offset = Unary(state, spv::OpUConvert, wide,
	                      Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u)));
	const auto base = PackU64(
	    state, Binary(state, spv::OpBitwiseAnd, TypeU32(state), vsharp[0], ConstantU32(state, ~3u)),
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), vsharp[1], ConstantU32(state, 0xffffu)));
	object.object = EmitGuestObject(ctx, Binary(state, spv::OpIAdd, wide, base, object.offset), size);
}

uint32_t EmitScalarBufferObjectDword(ValueEmitContext& ctx, const ScalarBufferObject& object, uint32_t offset) {
	auto&      state = ctx.state;
	const auto wide  = TypeU64(state);
	const auto end   = Binary(state, spv::OpIAdd, wide, object.offset, ConstantU64(state, offset + sizeof(uint32_t)));
	return EmitValueOrZeroIfCondition(state, Binary(state, spv::OpULessThanEqual, TypeBool(state), end, object.size),
	                                  [&]() { return EmitGuestObjectDword(ctx, object.object, offset); });
}

uint32_t EmitScalarBufferDwordLoad(ValueEmitContext& ctx, const std::array<uint32_t, 4>& vsharp,
                                   uint32_t offset) {
	auto&      state  = ctx.state;
	const auto wide   = TypeU64(state);
	const auto stride = EmitBitFieldUExtract(state, vsharp[1], ConstantU32(state, 16), ConstantU32(state, 14));
	const auto size   = Binary(state, spv::OpIMul, wide,
	                           Unary(state, spv::OpUConvert, wide, EmitUMax32(state, stride, ConstantU32(state, 1))),
	                           Unary(state, spv::OpUConvert, wide, vsharp[2]));
	const auto byte   = Unary(state, spv::OpUConvert, wide,
	                          Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u)));
	const auto end    = Binary(state, spv::OpIAdd, wide, byte, ConstantU64(state, sizeof(uint32_t)));
	const auto dword_address = [&]() {
		const auto base = PackU64(
		    state, Binary(state, spv::OpBitwiseAnd, TypeU32(state), vsharp[0], ConstantU32(state, ~3u)),
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), vsharp[1], ConstantU32(state, 0xffffu)));
		return Binary(state, spv::OpIAdd, wide, base, byte);
	};
	return LoadBdaDwordIf(ctx, Binary(state, spv::OpULessThanEqual, TypeBool(state), end, size),
	                      dword_address());
}

uint32_t EmitFormattedBufferLoadX(ValueEmitContext& ctx, const std::array<uint32_t, 4>& vsharp,
                                  uint32_t index) {
	auto&      state = ctx.state;
	const auto zero  = ConstantU32(state, 0);
	const std::array<uint32_t, 8> args {vsharp[0], vsharp[1], vsharp[2], vsharp[3], index, zero, zero,
	                                    BufferLane(state)};
	return LoadIndirectFormatted(ctx, PrepareIndirectBuffer(state, args, true), args, 1u);
}

std::array<uint32_t, 3> EmitTriangleIndices(ValueEmitContext& ctx, const std::array<uint32_t, 4>& vsharp,
                                            uint32_t first) {
	auto&      state = ctx.state;
	const auto u = TypeU32(state), b = TypeBool(state), wide = TypeU64(state);
	const auto vector = TypeU32Vector(state, 3);
	const auto field = [&](uint32_t word, uint32_t offset, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, offset), ConstantU32(state, count));
	};
	const auto equal = [&](uint32_t value, uint32_t constant) {
		return Binary(state, spv::OpIEqual, b, value, ConstantU32(state, constant));
	};
	const auto format = field(vsharp[3], 12, 7);
	const auto stride = field(vsharp[1], 16, 14);
	const auto narrow = AndCondition(state, equal(format, static_cast<uint32_t>(Prospero::BufferFormat::k16UInt)),
	                                 equal(stride, 2));
	const auto dwords = AndCondition(state, equal(format, static_cast<uint32_t>(Prospero::BufferFormat::k32UInt)),
	                                 equal(stride, 4));
	const auto plain = AndCondition(
	    state,
	    AndCondition(state, AndCondition(state, equal(field(vsharp[1], 31, 1), 0), equal(field(vsharp[3], 23, 1), 0)),
	                 equal(field(vsharp[3], 28, 2), 0)),
	    equal(field(vsharp[3], 0, 3), 4));
	const auto fast = AndCondition(state, plain, Binary(state, spv::OpLogicalOr, b, narrow, dwords));
	const auto zero = ConstantU32CompositeZero(state, 3);
	const auto compose = [&](const std::array<uint32_t, 3>& values) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, vector, result, values[0], values[1], values[2]);
		return result;
	};
	const auto fast_values = EmitValueOrDefaultIfCondition(state, fast, vector, zero, [&]() {
		const auto base    = PackU64(state, vsharp[0], field(vsharp[1], 0, 16));
		const auto start   = Binary(state, spv::OpIAdd, wide, base,
		                            Unary(state, spv::OpUConvert, wide, Binary(state, spv::OpIMul, u, first, stride)));
		const auto aligned = Binary(state, spv::OpBitwiseAnd, wide, start, ConstantU64(state, ~uint64_t {3}));
		const auto object  = EmitGuestObject(ctx, aligned, 16);
		const auto bias    = Unary(state, spv::OpUConvert, u, Binary(state, spv::OpBitwiseAnd, wide, start, ConstantU64(state, 3)));
		std::array<uint32_t, 3> values {};
		for (uint32_t k = 0; k < 3; k++) {
			const auto byte  = Binary(state, spv::OpIAdd, u, bias, Binary(state, spv::OpIMul, u, ConstantU32(state, k), stride));
			const auto value = EmitGuestObjectDwordAt(ctx, object, Binary(state, spv::OpBitwiseAnd, u, byte, ConstantU32(state, ~3u)));
			const auto shift = Binary(state, spv::OpShiftLeftLogical, u,
			                          Binary(state, spv::OpBitwiseAnd, u, byte, ConstantU32(state, 3)), ConstantU32(state, 3));
			const auto element =
			    Select(state, u, narrow,
			           Binary(state, spv::OpBitwiseAnd, u, Binary(state, spv::OpShiftRightLogical, u, value, shift),
			                  ConstantU32(state, 0xffffu)),
			           value);
			const auto in_bounds =
			    Binary(state, spv::OpULessThan, b, Binary(state, spv::OpIAdd, u, first, ConstantU32(state, k)), vsharp[2]);
			values[k] = Select(state, u, in_bounds, element, ConstantU32(state, 0));
		}
		return compose(values);
	});
	const auto slow_values = EmitValueOrDefaultIfCondition(state, Unary(state, spv::OpLogicalNot, b, fast), vector, zero, [&]() {
		const auto entry    = state.current_label;
		const auto header   = state.builder.AllocateId();
		const auto body     = state.builder.AllocateId();
		const auto cont     = state.builder.AllocateId();
		const auto merge    = state.builder.AllocateId();
		const auto index    = state.builder.AllocateId();
		const auto gathered = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBranch, header);
		EmitLabel(state, header);
		const auto index_phi    = state.builder.AddDeferredPhi(u, index, 2);
		const auto gathered_phi = state.builder.AddDeferredPhi(vector, gathered, 2);
		const auto more         = Binary(state, spv::OpULessThan, b, index, ConstantU32(state, 3));
		state.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlDontUnrollMask);
		state.builder.AddFunction(spv::OpBranchConditional, more, body, merge);
		EmitLabel(state, body);
		const auto value = EmitFormattedBufferLoadX(ctx, vsharp, Binary(state, spv::OpIAdd, u, first, index));
		const auto next  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpVectorInsertDynamic, vector, next, gathered, value, index);
		state.builder.AddFunction(spv::OpBranch, cont);
		EmitLabel(state, cont);
		const auto next_index = Binary(state, spv::OpIAdd, u, index, ConstantU32(state, 1));
		state.builder.AddFunction(spv::OpBranch, header);
		state.builder.PatchDeferredPhi(index_phi, 0, ConstantU32(state, 0), entry);
		state.builder.PatchDeferredPhi(index_phi, 1, next_index, cont);
		state.builder.PatchDeferredPhi(gathered_phi, 0, zero, entry);
		state.builder.PatchDeferredPhi(gathered_phi, 1, next, cont);
		EmitLabel(state, merge);
		return gathered;
	});
	const auto fast_components = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeBoolVector(state, 3), fast_components,
	                          fast, fast, fast);
	const auto selected = Select(state, vector, fast_components, fast_values, slow_values);
	std::array<uint32_t, 3> result {};
	for (uint32_t k = 0; k < 3; k++) {
		result[k] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, u, result[k], selected, k);
	}
	return result;
}

struct UnorderedUpdateScope {
	UnorderedUpdateScope(EmitterState& state_, std::optional<SplitHints::Update> update): state(state_) {
		state.unordered_update = update;
	}
	~UnorderedUpdateScope() { state.unordered_update.reset(); }
	UnorderedUpdateScope(const UnorderedUpdateScope&)            = delete;
	UnorderedUpdateScope& operator=(const UnorderedUpdateScope&) = delete;
	EmitterState& state;
};

static std::optional<SplitHints::Update> UnorderedSharedUpdate(const IR::Inst& inst) {
	if (inst.HasUses()) return std::nullopt;
	switch (inst.GetOpcode()) {
		case IR::ValueOpcode::SharedAtomicUMin32: return SplitHints::Update::UMin;
		case IR::ValueOpcode::SharedAtomicUMax32: return SplitHints::Update::UMax;
		case IR::ValueOpcode::SharedAtomicSMin32: return SplitHints::Update::SMin;
		case IR::ValueOpcode::SharedAtomicSMax32: return SplitHints::Update::SMax;
		case IR::ValueOpcode::SharedAtomicIAdd32: return SplitHints::Update::Add;
		case IR::ValueOpcode::SharedAtomicISub32: return SplitHints::Update::Sub;
		case IR::ValueOpcode::SharedAtomicAnd32: return SplitHints::Update::And;
		case IR::ValueOpcode::SharedAtomicOr32: return SplitHints::Update::Or;
		case IR::ValueOpcode::SharedAtomicXor32: return SplitHints::Update::Xor;
		default: return std::nullopt;
	}
}

bool PrivateLdsStorage(const EmitterState& state) {
	return state.lds_storage_class != spv::StorageClassWorkgroup &&
	       state.lds_storage_class != spv::StorageClassStorageBuffer;
}

uint32_t EmitPrivateAtomic32Value(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t old) {
	auto&      state = ctx.state;
	const auto type  = TypeU32(state);
	const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
	switch (inst.GetOpcode()) {
		case IR::ValueOpcode::SharedAtomicIAdd32:
			return Binary(state, spv::OpIAdd, type, old, value);
		case IR::ValueOpcode::SharedAtomicISub32:
			return Binary(state, spv::OpISub, type, old, value);
		case IR::ValueOpcode::SharedAtomicSMin32: return EmitSMin32(state, old, value);
		case IR::ValueOpcode::SharedAtomicSMax32: return EmitSMax32(state, old, value);
		case IR::ValueOpcode::SharedAtomicUMin32: return EmitUMin32(state, old, value);
		case IR::ValueOpcode::SharedAtomicUMax32: return EmitUMax32(state, old, value);
		case IR::ValueOpcode::SharedAtomicAnd32:
			return Binary(state, spv::OpBitwiseAnd, type, old, value);
		case IR::ValueOpcode::SharedAtomicOr32:
			return Binary(state, spv::OpBitwiseOr, type, old, value);
		case IR::ValueOpcode::SharedAtomicXor32:
			return Binary(state, spv::OpBitwiseXor, type, old, value);
		case IR::ValueOpcode::SharedAtomicSwap32: return value;
		case IR::ValueOpcode::SharedAtomicCmpSwap32: {
			auto desired    = ctx.Arg(inst, inst.NumArgs() - 3);
			auto comparator = ctx.Arg(inst, inst.NumArgs() - 2);
			std::swap(desired, comparator);
			return Select(state, type,
			              Binary(state, spv::OpIEqual, TypeBool(state), old, comparator), desired,
			              old);
		}
		default:
			EXIT("LDS atomic 0x%08x in a stage without workgroup storage\n",
			     static_cast<uint32_t>(inst.GetOpcode()));
	}
}

uint32_t EmitAtomic32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem = ctx.Memory(inst);
	if (mem.kind == IR::ResourceKind::IndirectBuffer) return AtomicIndirectBuffer(ctx, inst);
	const UnorderedUpdateScope unordered(ctx.state, UnorderedSharedUpdate(inst));
	return EmitAtomicAccess(ctx, inst, mem, [&](uint32_t pointer) {
		if (mem.kind == IR::ResourceKind::Lds && PrivateLdsStorage(ctx.state)) {
			const auto old = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), old, pointer);
			ctx.state.builder.AddFunction(
			    spv::OpStore, pointer, EmitPrivateAtomic32Value(ctx, inst, old));
			return old;
		}
		const auto scope =
		    mem.kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
		const auto old = EmitAtomicOperation(ctx, inst, pointer, scope);
		EmitAtomicMemoryBarrier(ctx.state, mem.kind);
		return old;
	});
}

uint32_t EmitBufferAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem   = ctx.Memory(inst);
	auto&       state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU64(state), ConstantU64(state, 0), [&]() {
		    auto resource =
		        PrepareStorageBufferResourceAccess(state, mem, state.storage_buffers[3]);
		    resource.element_bits = 64u;
		    const auto byte_address = ByteAddress(ctx, inst, mem);
		    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte_address,
		                              ConstantU32(state, 3u));
		    return EmitValueOrDefaultIfCondition(
		        state, BufferIndexInBounds(ctx, inst, mem, resource,
		            EmitMemoryElementInBounds(state, resource, index), ConstantU32(state, 8u)), TypeU64(state),
		        ConstantU64(state, 0), [&]() {
			        const auto pointer = EmitStorageBufferElementPointer(
			            state, resource, index, TypeStorageBufferElementPointer(state, 64));
			        const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
			        EmitAtomicMemoryBarrier(state, mem.kind);
			        return old;
		        });
	    });
}

void EmitSharedAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	EnsureLdsStorage(state);
	const auto address = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                            ByteAddress(ctx, inst, mem), ConstantU32(state, 0xfff8u));
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                          ConstantU32(state, 3u));
	const auto in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index,
	                              ConstantU32(state, LdsDwordCount(state) / 2u));
	const auto active = EmitLogicalAndBool(state, ctx.Arg(inst, inst.NumArgs() - 1), in_bounds);
	const auto emit = [&]() {
		auto native_index = index;
		auto semantics = spv::MemorySemanticsWorkgroupMemoryMask;
		if (state.lds_storage_class == spv::StorageClassStorageBuffer) {
			native_index = EmitAddU32(state, index, EmitBinaryU32(
			    state, spv::OpShiftRightLogical, state.lds_base_dwords, ConstantU32(state, 1)));
			semantics = spv::MemorySemanticsUniformMemoryMask;
		}
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, state.lds_storage_class, TypeU64(state)),
		    pointer, state.lds_u64_variable, ConstantU32(state, 0), native_index);
		const auto old    = state.builder.AllocateId();
		const auto opcode = SpirvAtomicOpcode(inst.GetOpcode());
		if (PrivateLdsStorage(state)) {
			const auto wide  = TypeU64(state);
			const auto value = ctx.Arg(inst, 1);
			state.builder.AddFunction(spv::OpLoad, wide, old, pointer);
			uint32_t next = 0;
			switch (opcode) {
				case spv::OpAtomicIAdd:
					next = Binary(state, spv::OpIAdd, wide, old, value);
					break;
				case spv::OpAtomicOr:
					next = Binary(state, spv::OpBitwiseOr, wide, old, value);
					break;
				case spv::OpAtomicUMax:
					next = EmitGlsl<GLSLstd450UMax, IR::Type::U64, uint32_t, uint32_t>(
					    state, old, value);
					break;
				case spv::OpAtomicCompareExchange: {
					const auto desired    = ctx.Arg(inst, 2);
					const auto comparator = ctx.Arg(inst, 1);
					next = Select(state, wide,
					              Binary(state, spv::OpIEqual, TypeBool(state), old, comparator),
					              desired, old);
					break;
				}
				default:
					EXIT("LDS 64-bit atomic 0x%08x in a stage without workgroup storage\n",
					     static_cast<uint32_t>(inst.GetOpcode()));
			}
			state.builder.AddFunction(spv::OpStore, pointer, next);
			return old;
		}
		const auto scope = ConstantU32(state, spv::ScopeWorkgroup);
		const auto acquire_release =
		    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | semantics);
		if (opcode == spv::OpAtomicCompareExchange) {
			state.builder.AddFunction(
			    opcode, TypeU64(state), old, pointer, scope, acquire_release,
			    ConstantU32(state, spv::MemorySemanticsAcquireMask | semantics),
			    ctx.Arg(inst, 2), ctx.Arg(inst, 1));
		} else {
			state.builder.AddFunction(opcode, TypeU64(state), old, pointer, scope,
			                          acquire_release, ctx.Arg(inst, 1));
		}
		return old;
	};
	if (inst.HasUses()) {
		ctx.Define(inst, EmitValueOrDefaultIfCondition(
		    state, active, TypeU64(state), ConstantU64(state, 0), emit));
	} else {
		EmitIfCondition(state, active, emit);
	}
}

uint32_t EmitBufferFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::BufferAtomicFMax32;
	return EmitAtomicUpdate(ctx, inst, mem,
	                        [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                        return EmitFloatAtomicReplacement(state, old, value, max_value);
	                        });
}

void EmitSharedFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::SharedAtomicFMax32;
	const UnorderedUpdateScope unordered(
	    ctx.state, inst.HasUses() ? std::nullopt
	                              : std::optional {max_value ? SplitHints::Update::FMax : SplitHints::Update::FMin});
	EmitAtomicUpdate(ctx, inst, mem,
	                 [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                 return EmitDsFloatAtomicReplacement(state, old, value, max_value);
	                 });
}

uint32_t EmitAppendConsume(ValueEmitContext& ctx, const IR::Inst& inst) {
	const bool append = inst.GetOpcode() == IR::ValueOpcode::DataAppend;
	auto&      state  = ctx.state;
	if (ctx.half == 1) {
		return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
	}
	const auto mem           = ctx.Memory(inst);
	const auto offset        = mem.offset & 0xfffcu;
	auto       address       = ConstantU32(state, offset);
	uint32_t   region_bounds = 0;
	if (mem.kind == IR::ResourceKind::Gds) {
		const auto m0 = ctx.Arg(inst, 0);
		const auto base =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), m0, ConstantU32(state, 16));
		const auto size =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), m0, ConstantU32(state, 0xffffu));
		address = Binary(state, spv::OpIAdd, TypeU32(state), base, address);
		region_bounds = AndCondition(
		    state, Binary(state, spv::OpULessThan, TypeBool(state),
		                  ConstantU32(state, offset + 3u), size),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state),
		           Binary(state, spv::OpIAdd, TypeU32(state), base, size),
		           ConstantU32(state, 0xc000u)));
	}
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
	const auto access = PrepareMemoryResourceAccess(state, mem);
	const auto low   = ctx.Arg(inst, 1);
	const auto high  = ctx.Arg(inst, 2);
	const auto count = Binary(state, spv::OpIAdd, TypeU32(state),
	                          Unary(state, spv::OpBitCount, TypeU32(state), low),
	                          Unary(state, spv::OpBitCount, TypeU32(state), high));
	const auto mask  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), mask, low, high,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	const auto first = ctx.FirstLane(mask);
	const auto source_lane =
	    state.lane_count == 2
	        ? Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 31))
	        : first;
	const auto is_first       = Binary(state, spv::OpIEqual, TypeBool(state),
	                                   EmitSubgroupLocalInvocationId(state), source_lane);
	auto bounds = EmitMemoryElementInBounds(state, access, index);
	if (mem.kind == IR::ResourceKind::Gds) {
		bounds = AndCondition(state, bounds, region_bounds);
	}
	const auto condition = AndCondition(
	    state, is_first,
	    AndCondition(state,
	                 Binary(state, spv::OpINotEqual, TypeBool(state), count, ConstantU32(state, 0)),
	                 bounds));
	const auto atomic = EmitValueOrZeroIfCondition(state, condition, [&]() {
		const auto pointer = EmitMemoryElementPointer(state, access, index);
		const auto value = state.builder.AllocateId();
		if (mem.kind == IR::ResourceKind::Lds && PrivateLdsStorage(state)) {
			state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
			state.builder.AddFunction(
			    spv::OpStore, pointer,
			    Binary(state, append ? spv::OpIAdd : spv::OpISub, TypeU32(state), value, count));
			return value;
		}
		state.builder.AddFunction(append ? spv::OpAtomicIAdd : spv::OpAtomicISub, TypeU32(state),
		                          value, pointer,
		                          ConstantU32(state, mem.kind == IR::ResourceKind::Gds
		                                                 ? spv::ScopeDevice
		                                                 : spv::ScopeWorkgroup),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), count);
		return value;
	});
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), atomic, source_lane);
	return result;
}

uint32_t EmitReadConst(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.program.table_mode) {
		const auto slot = inst.Arg(1).Resolve().U32();
		return slot < state.table_slot_values.size() && state.table_slot_values[slot] != 0
		           ? state.table_slot_values[slot]
		           : ConstantU32(state, 0);
	}
	if (state.flattened_srt_variable == 0) {
		ctx.Fail(inst, "requires the flattened SRT descriptor");
	}
	const auto slot = inst.Arg(1).Resolve();
	if (const auto uniform = state.program.bindings.flattened_srt_uniform_index;
	    uniform != UINT32_MAX && slot.IsImmediate() && slot.U32() < IR::FlattenedSrtUniformWords) {
		return LoadUniformBufferWord(state, uniform, ConstantU32(state, slot.U32()));
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, state.flattened_srt_pointer, pointer,
	                          state.flattened_srt_variable, ConstantU32(state, 0),
	                          ctx.Arg(inst, 1));
	return EmitNative<spv::OpLoad, IR::Type::U32>(state, pointer);
}

void EmitReadConstBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto mem = ctx.Memory(inst);
	if (mem.planning_only) return;
	if (static_cast<int32_t>(mem.offset) < 0) ctx.Fail(inst, "has a negative scalar buffer immediate offset");
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		ctx.Define(inst, LoadIndirectScalarBuffer(ctx, inst));
		return;
	}
	auto& state        = ctx.state;
	mem.kind           = IR::ResourceKind::ScalarBuffer;
	const auto offset = inst.Arg(1).Resolve();
	const auto static_index = offset.IsImmediate()
	                              ? (offset.U32() >> 2u) + (mem.offset >> 2u)
	                              : 0u;
	auto index = offset.IsImmediate()
	                 ? ConstantU32(state, static_index)
	                 : Binary(state, spv::OpShiftRightLogical, TypeU32(state), ctx.Arg(inst, 1),
	                          ConstantU32(state, 2));
	if (!offset.IsImmediate() && mem.offset >= 4u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index,
		               ConstantU32(state, mem.offset >> 2u));
	}
	const auto storage_load = [&]() {
		const auto access  = PrepareMemoryResourceAccess(state, mem);
		auto       element = Binary(state, spv::OpIAdd, TypeU32(state), index,
		                            Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.byte_offset,
		                                   ConstantU32(state, 2)));
		constexpr uint32_t last_element = UINT32_MAX >> 2u;
		if (state.profile.robust_buffer_access &&
		    (!offset.IsImmediate() || static_index > last_element - 63u)) {
			element = EmitGlsl<GLSLstd450UMin, IR::Type::U32>(state, element, ConstantU32(state, last_element));
		}
		const auto condition = EmitStorageElementInBounds(state, access, element);
		return EmitValueOrZeroIfCondition(state, condition, [&]() {
			return EmitAccessLoad(state, TypeU32(state), EmitMemoryElementPointer(state, access, element), access);
		});
	};
	const auto& resource = state.program.info.buffers.at(mem.resource);
	if (resource.uniform_index == UINT32_MAX || state.memory_byte_offsets[resource.descriptor_index] == 0) {
		ctx.Define(inst, storage_load());
		return;
	}
	const auto element = Binary(
	    state, spv::OpIAdd, TypeU32(state), index,
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), state.memory_byte_offsets[resource.descriptor_index],
	           ConstantU32(state, 2)));
	const auto uniform_load = [&]() { return LoadUniformBufferWord(state, resource.uniform_index, element); };
	if (state.profile.robust_buffer_access && resource.constant_only) {
		ctx.Define(inst, uniform_load());
		return;
	}
	const auto bounded_load = [&]() {
		const auto in_bounds =
		    Binary(state, spv::OpULessThan, TypeBool(state), element,
		           Binary(state, spv::OpShiftRightLogical, TypeU32(state),
		                  state.memory_byte_sizes[resource.descriptor_index], ConstantU32(state, 2)));
		return state.profile.robust_buffer_access
		           ? Select(state, TypeU32(state), in_bounds, uniform_load(), ConstantU32(state, 0))
		           : EmitValueOrZeroIfCondition(state, in_bounds, uniform_load);
	};
	constexpr uint32_t uniform_words = IR::UniformBufferBytes / sizeof(uint32_t);
	constexpr uint32_t base_words    = IR::UniformBufferBaseSlack / sizeof(uint32_t);
	if (offset.IsImmediate() && static_index < uniform_words - base_words) {
		ctx.Define(inst, bounded_load());
		return;
	}
	const auto fits = Binary(state, spv::OpULessThan, TypeBool(state), element, ConstantU32(state, uniform_words));
	ctx.Define(inst, EmitValueIfElse(state, fits, TypeU32(state), bounded_load, storage_load));
}

uint32_t EmitLoadDeviceAddressU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, ctx.Arg(inst, 3), [&]() {
		const auto address = Binary(
		    state, spv::OpIAdd, TypeU64(state), PackU64(state, ctx.Arg(inst, 0), ctx.Arg(inst, 1)),
		    Unary(state, spv::OpUConvert, TypeU64(state), ctx.Arg(inst, 2)));
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpLoad, TypeU32(state), value,
		    Unary(state, spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), address),
		    spv::MemoryAccessAlignedMask, 4u);
		return value;
	});
}

uint32_t EmitGetBufferResourceIndex(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& buffer = ctx.state.program.info.buffers.at(inst.Flags<uint32_t>());
	if (buffer.indirect_resources.empty()) return ConstantU32(ctx.state, 0u);
	return EmitIndirectResourceIndex(ctx.state, ctx.Arg(inst, 0), buffer.indirect_mapping_offset,
	                                 buffer.indirect_search_iterations, UINT32_MAX);
}

void EmitLoadMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op  = inst.GetOpcode();
	const auto& mem = ctx.Memory(inst);
	if (op == IR::ValueOpcode::LoadAddressU32 && mem.planning_only) return;
	EmitLdsLaneSync(ctx.state, mem);
	const auto buffer_components = IR::BufferComponentCount(op);
	const auto shared_components = IR::SharedComponentCount(op);
	const auto address_info      = IR::AddressOpcodeInfoOf(op);
	uint32_t   value;
	if (mem.kind == IR::ResourceKind::FlatLocal)
		value = LoadLocalFlat(ctx, inst);
	else if (buffer_components > 1u ||
	         (buffer_components == 1u && mem.kind == IR::ResourceKind::IndirectBuffer) ||
	         (op == IR::ValueOpcode::LoadBufferU32 && mem.formatted))
		value = LoadBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		value = LoadWideShared(ctx, inst, shared_components);
	else if (const auto* group = op == IR::ValueOpcode::LoadAddressU32 ? FindBdaGroup(ctx.state, inst) : nullptr;
	         group != nullptr)
		value = LoadGroupedBda(ctx, inst, mem, *group);
	else if (mem.kind == IR::ResourceKind::ScalarAddress)
		value = LoadBdaDword(ctx, GuestAddress(ctx, inst, mem));
	else if (address_info.access == IR::AddressAccess::Read &&
	         mem.kind != IR::ResourceKind::Scratch)
		value = LoadBda(ctx, GuestAddress(ctx, inst, mem), ctx.Arg(inst, inst.NumArgs() - 1),
		                address_info.data_bits, mem.coherent);
	else if (inst.GetType() == IR::Type::U8)
		value = LoadSubword(ctx, inst, mem, 8);
	else if (inst.GetType() == IR::Type::U16)
		value = LoadSubword(ctx, inst, mem, 16);
	else
		value = LoadWord(ctx, inst, mem);
	ctx.Define(inst, value);
}

void EmitStoreMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op                = inst.GetOpcode();
	const auto& mem               = ctx.Memory(inst);
	const auto  buffer_components = IR::BufferComponentCount(op);
	const auto  shared_components = IR::SharedComponentCount(op);
	const auto  type              = inst.Arg(inst.NumArgs() - 2).GetType();
	EmitLdsLaneSync(ctx.state, mem);
	if (mem.kind == IR::ResourceKind::FlatLocal)
		StoreLocalFlat(ctx, inst);
	else if (mem.kind == IR::ResourceKind::IndirectBuffer && mem.formatted)
		StoreIndirectFormatted(ctx, inst, std::max(buffer_components, 1u));
	else if (mem.kind == IR::ResourceKind::IndirectBuffer && (type == IR::Type::U8 || type == IR::Type::U16))
		StoreIndirectSubword(ctx, inst, type == IR::Type::U8 ? 8u : 16u);
	else if (mem.kind == IR::ResourceKind::IndirectBuffer)
		StoreIndirectBuffer(ctx, inst, std::max(buffer_components, 1u));
	else if (buffer_components > 1u)
		StoreBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		StoreWideShared(ctx, inst, shared_components);
	else if (op == IR::ValueOpcode::StoreBufferU32 && mem.formatted)
		StoreBuffer(ctx, inst, 1u);
	else if (type == IR::Type::U8)
		StoreSubword(ctx, inst, mem, 8);
	else if (type == IR::Type::U16)
		StoreSubword(ctx, inst, mem, 16);
	else
		StoreWord(ctx, inst, mem);
}

uint32_t EmitSharedIncDec(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto replacement =
	    inst.GetOpcode() == IR::ValueOpcode::SharedAtomicInc32 ? AtomicIncrement : AtomicDecrement;
	return EmitAtomicUpdate(ctx, inst, ctx.Memory(inst), replacement);
}

void EmitSharedAtomicMaskedOr32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto keep = Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), ctx.Arg(inst, 1));
	EmitAtomicUpdate(ctx, inst, ctx.Memory(inst),
	                 [keep](EmitterState& state, uint32_t old, uint32_t value) {
		                 return Binary(state, spv::OpBitwiseOr, TypeU32(state),
		                               Binary(state, spv::OpBitwiseAnd, TypeU32(state), old, keep), value);
	                 });
}

uint32_t EmitSwizzleU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	state.builder.AddFunction(spv::OpStore, ctx.scratch_u32_variable, ctx.Arg(inst, 0));
	const auto source = EmitNative<spv::OpLoad, IR::Type::U32>(state, ctx.scratch_u32_variable);
	const auto target = EmitDsSwizzleTargetLane(state, EmitSubgroupLocalInvocationId(state),
	                                            inst.Arg(1).IsImmediate() ? inst.Arg(1).U32() : 0);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

namespace {

uint32_t RelativeVgprPointer(ValueEmitContext& ctx, uint32_t index) {
	auto&      state   = ctx.state;
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, state.scratch_storage_class, TypeU32(state)),
	                          pointer, ctx.relative_vgpr_variable, index);
	return pointer;
}

}

uint32_t EmitLoadRelativeVgpr(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitNative<spv::OpLoad, IR::Type::U32>(ctx.state, RelativeVgprPointer(ctx, ctx.Arg(inst, 0)));
}

void EmitStoreRelativeVgpr(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.state.builder.AddFunction(spv::OpStore, RelativeVgprPointer(ctx, ctx.Arg(inst, 0)), ctx.Arg(inst, 1));
}

uint32_t EmitPermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state   = ctx.state;
	auto       lane    = EmitSubgroupLocalInvocationId(state);
	if (state.lane_count == 2) {
		lane = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane, ConstantU32(state, 31));
	}
	const auto address = ctx.Arg(inst, 1);
	const auto word    = Binary(state, spv::OpShiftRightLogical, TypeU32(state), lane,
	                            ConstantU32(state, 5));
	const auto ballot_word = [&](uint32_t predicate) {
		const auto ballot = state.builder.AllocateId();
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), predicate);
		state.builder.AddFunction(spv::OpVectorExtractDynamic, TypeU32(state), result, ballot, word);
		return result;
	};
	auto writers = ballot_word(ctx.Arg(inst, 2));
	for (uint32_t bit = 0; bit < 5; ++bit) {
		const auto address_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), address,
		                                ConstantU32(state, 1u << (bit + 2)));
		const auto mask = ballot_word(Binary(state, spv::OpINotEqual, TypeBool(state),
		                                      address_bit, ConstantU32(state, 0)));
		const auto lane_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
		                             ConstantU32(state, 1u << bit));
		const auto selected = Select(
		    state, TypeU32(state),
		    Binary(state, spv::OpINotEqual, TypeBool(state), lane_bit, ConstantU32(state, 0)),
		    mask, Unary(state, spv::OpNot, TypeU32(state), mask));
		writers = Binary(state, spv::OpBitwiseAnd, TypeU32(state), writers, selected);
	}
	const auto active = Binary(state, spv::OpINotEqual, TypeBool(state), writers,
	                           ConstantU32(state, 0));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
	                         ConstantU32(state, ~31u));
	const auto source = Select(state, TypeU32(state), active,
	                           Binary(state, spv::OpBitwiseOr, TypeU32(state), base,
	                                  EmitFindUMsb32(state, writers)), lane);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), ctx.Arg(inst, 0), source);
	return Select(state, TypeU32(state), active, result, ConstantU32(state, 0));
}

uint32_t EmitBpermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto source = ctx.Arg(inst, 0);
	const auto index  = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                                  ctx.Arg(inst, 1), ConstantU32(state, 2)),
	                           ConstantU32(state, 31));
	const auto base   = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           EmitSubgroupLocalInvocationId(state), ConstantU32(state, ~31u));
	const auto target = Binary(state, spv::OpBitwiseOr, TypeU32(state), base, index);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

}

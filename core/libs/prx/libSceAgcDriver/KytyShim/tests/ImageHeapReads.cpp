#include "RenderExecutorTestAccess.h"
#include "TestVulkanDevice.h"
#include "common/alignment.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/memoryTelemetry.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {

struct ImageHeapReadsTestAccess {
	using Clock = std::chrono::steady_clock;

	static void Collect(ImageHeap& heap, Clock::time_point now) { heap.CollectAt(now); }
	static vk::DescriptorSet Set(const ImageHeap& heap) { return heap.m_set; }
	static const Buffer& Reads(const ImageHeap& heap) { return *heap.m_reads; }
	static size_t Parking(const ImageHeap& heap) { return heap.m_parking.size(); }

	static uint32_t Slot(const ImageHeap& heap, const ShaderRecompiler::IR::DescriptorValue& value) {
		for (const auto& [key, slot]: heap.m_index) {
			if (key.value == value) return slot;
		}
		return 0u;
	}
	static bool     Parked(const ImageHeap& heap, uint32_t slot) { return heap.m_elements[slot].parked; }
	static ImageId  ImageOf(const ImageHeap& heap, uint32_t slot) { return heap.m_elements[slot].image; }
	static uint32_t ReadStamp(const ImageHeap& heap, uint32_t slot) { return heap.m_read_stamps.Stamp(slot); }

	static Image* Find(TextureCache& textures, ImageId id) { return textures.m_slot_images.try_get(id); }

	static void Pressure(GraphicContext& graphics, uint64_t budget, uint64_t usage) {
		graphics.m_memory_override_enabled = true;
		graphics.m_memory_budget_override  = budget;
		graphics.m_memory_usage_override   = usage;
	}
	static void Relax(GraphicContext& graphics) { graphics.m_memory_override_enabled = false; }
	static bool Crowded(const ImageHeap& heap) { return heap.m_crowded; }
	static void MarkRead(ImageHeap& heap, uint32_t slot) { heap.m_read_stamps.Mark(slot); }
	static bool Overridden(const GraphicContext& graphics) { return graphics.m_memory_override_enabled; }
	static bool CanSpill(const GraphicContext& graphics) { return graphics.spill_allocator != nullptr; }

	static std::string Views(const ImageHeap& heap, uint32_t slot) {
		std::string text = " profiles " + std::to_string(heap.m_elements[slot].table_profiles.size()) + ", views";
		for (const auto& memory: heap.m_memories) {
			for (const auto& view: memory->views) {
				text += " " + std::to_string(view->shape.profile) + (view->full ? " full" : " partial") + " " +
				        std::to_string(view->count);
			}
		}
		return text;
	}

	static std::vector<uint32_t> PendingProbes(const ImageHeap& heap) {
		std::vector<uint32_t> probes;
		for (const auto& memory: heap.m_memories) {
			for (const auto& view: memory->views) {
				for (const auto& range: view->pending) {
					for (auto probe = range.begin; probe < range.end; ++probe) probes.push_back(probe);
				}
			}
		}
		std::ranges::sort(probes);
		probes.erase(std::ranges::unique(probes).begin(), probes.end());
		return probes;
	}
};

}

namespace {

using namespace Libs::Graphics;
namespace IR = ShaderRecompiler::IR;
using Access = ImageHeapReadsTestAccess;
using Clock  = std::chrono::steady_clock;

int         g_failures = 0;
const char* g_name     = "kyty_shim_image_heap_reads";

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "%s: FAILED %s\n", g_name, what.c_str());
	g_failures++;
}

int Finish() {
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %d failure(s)\n", g_name, g_failures);
		return 1;
	}
	std::printf("%s: ok\n", g_name);
	return 0;
}

constexpr uint32_t TexelCount = 16;
constexpr uint32_t Textures   = 4;
constexpr uint32_t Lanes      = 64;

IR::DescriptorValue SquareTexture(uint64_t address, uint32_t side) {
	IR::DescriptorValue value {.dword_count = 8u};
	value.dwords[0] = static_cast<uint32_t>(address >> 8u);
	value.dwords[1] = static_cast<uint32_t>((address >> 40u) & 0xffu) |
	                  (static_cast<uint32_t>(Prospero::BufferFormat::k8UNorm) << 20u) | (((side - 1u) & 0x3u) << 30u);
	value.dwords[2] = (side - 1u) >> 2u | (side - 1u) << 14u;
	value.dwords[3] = 0xfacu | (9u << 28u);
	return value;
}

IR::DescriptorValue HeapTexture(uint64_t address) {
	IR::DescriptorValue value {.dword_count = 8u};
	value.dwords[0] = static_cast<uint32_t>(address >> 8u);
	value.dwords[1] = static_cast<uint32_t>((address >> 40u) & 0xffu) |
	                  (static_cast<uint32_t>(Prospero::BufferFormat::k8UNorm) << 20u) | (3u << 30u);
	value.dwords[2] = TexelCount - 1u;
	value.dwords[3] = 0xfacu | (9u << 28u);
	return value;
}

std::string ShaderText(bool mark, bool fetch) {
	std::string text = R"(
OpCapability Shader
OpCapability GroupNonUniform
OpCapability GroupNonUniformBallot
OpCapability ShaderNonUniform
OpCapability SampledImageArrayNonUniformIndexing
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid %lid %images %reads %out %push
OpExecutionMode %main LocalSize 64 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %lid BuiltIn LocalInvocationId
OpDecorate %images DescriptorSet 1
OpDecorate %images Binding 0
OpDecorate %reads DescriptorSet 1
OpDecorate %reads Binding 1
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
OpDecorate %runtime ArrayStride 4
OpMemberDecorate %Words 0 Offset 0
OpDecorate %Words Block
OpMemberDecorate %Push 0 Offset 0
OpMemberDecorate %Push 1 Offset 4
OpMemberDecorate %Push 2 Offset 8
OpDecorate %Push Block
OpDecorate %image_ptr NonUniform
OpDecorate %image NonUniform
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%u32 = OpTypeInt 32 0
%i32 = OpTypeInt 32 1
%f32 = OpTypeFloat 32
%v3u = OpTypeVector %u32 3
%v2i = OpTypeVector %i32 2
%v4f = OpTypeVector %f32 4
%c0 = OpConstant %u32 0
%c1 = OpConstant %u32 1
%c2 = OpConstant %u32 2
%c3 = OpConstant %u32 3
%c5 = OpConstant %u32 5
%c15 = OpConstant %u32 15
%c31 = OpConstant %u32 31
%cmask = OpConstant %u32 65535
%i0 = OpConstant %i32 0
%ptr_in = OpTypePointer Input %v3u
%ptr_in_u32 = OpTypePointer Input %u32
%gid = OpVariable %ptr_in Input
%lid = OpVariable %ptr_in Input
%img = OpTypeImage %f32 2D 0 0 0 1 Unknown
%cheap = OpConstant %u32 65536
%img_array = OpTypeArray %img %cheap
%ptr_images = OpTypePointer UniformConstant %img_array
%ptr_img = OpTypePointer UniformConstant %img
%images = OpVariable %ptr_images UniformConstant
%runtime = OpTypeRuntimeArray %u32
%Words = OpTypeStruct %runtime
%ptr_words = OpTypePointer StorageBuffer %Words
%ptr_word = OpTypePointer StorageBuffer %u32
%reads = OpVariable %ptr_words StorageBuffer
%out = OpVariable %ptr_words StorageBuffer
%Push = OpTypeStruct %u32 %u32 %u32
%ptr_push = OpTypePointer PushConstant %Push
%ptr_push_u32 = OpTypePointer PushConstant %u32
%push = OpVariable %ptr_push PushConstant
%ptr_fn_u32 = OpTypePointer Function %u32
%main = OpFunction %void None %fn
%entry = OpLabel
%acc = OpVariable %ptr_fn_u32 Function
%iter = OpVariable %ptr_fn_u32 Function
OpStore %acc %c0
OpStore %iter %c0
%gid_x_ptr = OpAccessChain %ptr_in_u32 %gid %c0
%gid_x = OpLoad %u32 %gid_x_ptr
%lid_x_ptr = OpAccessChain %ptr_in_u32 %lid %c0
%lid_x = OpLoad %u32 %lid_x_ptr
%p0 = OpAccessChain %ptr_push_u32 %push %c0
%index0 = OpLoad %u32 %p0
%p1 = OpAccessChain %ptr_push_u32 %push %c1
%index1 = OpLoad %u32 %p1
%p2 = OpAccessChain %ptr_push_u32 %push %c2
%count = OpLoad %u32 %p2
%odd_bit = OpBitwiseAnd %u32 %lid_x %c1
%odd = OpINotEqual %bool %odd_bit %c0
%selected = OpSelect %u32 %odd %index1 %index0
OpBranch %header
%header = OpLabel
OpLoopMerge %done %continue None
OpBranch %check
%check = OpLabel
%i = OpLoad %u32 %iter
%more = OpULessThan %bool %i %count
OpBranchConditional %more %body %done
%body = OpLabel
%element = OpBitwiseAnd %u32 %selected %cmask
)";
	if (fetch) {
		text += R"(
%x_sum = OpIAdd %u32 %gid_x %i
%x = OpBitwiseAnd %u32 %x_sum %c15
%x_i = OpBitcast %i32 %x
%coord = OpCompositeConstruct %v2i %x_i %i0
%image_ptr = OpAccessChain %ptr_img %images %element
%image = OpLoad %img %image_ptr
%texel = OpImageFetch %v4f %image %coord Lod %i0
%red = OpCompositeExtract %f32 %texel 0
%bits = OpBitcast %u32 %red
%old_acc = OpLoad %u32 %acc
%new_acc = OpBitwiseXor %u32 %old_acc %bits
OpStore %acc %new_acc
)";
	} else {
		text += R"(
%image_ptr = OpAccessChain %ptr_img %images %element
%image = OpLoad %img %image_ptr
)";
	}
	if (mark) {
		text += R"(
%word = OpShiftRightLogical %u32 %element %c5
%word_ptr = OpAccessChain %ptr_word %reads %c0 %word
%low = OpBitwiseAnd %u32 %element %c31
%bit = OpShiftLeftLogical %u32 %c1 %low
%seen_word = OpLoad %u32 %word_ptr
%seen_bits = OpBitwiseAnd %u32 %seen_word %bit
%unseen = OpIEqual %bool %seen_bits %c0
OpSelectionMerge %checked None
OpBranchConditional %unseen %vote %checked
%vote = OpLabel
%first = OpGroupNonUniformBroadcastFirst %u32 %c3 %element
%elected = OpGroupNonUniformElect %bool %c3
%other = OpINotEqual %bool %element %first
%store = OpLogicalOr %bool %elected %other
OpSelectionMerge %marked None
OpBranchConditional %store %mark %marked
%mark = OpLabel
%previous = OpAtomicOr %u32 %word_ptr %c1 %c0 %bit
OpBranch %marked
%marked = OpLabel
OpBranch %checked
%checked = OpLabel
)";
	}
	text += R"(
OpBranch %continue
%continue = OpLabel
%next = OpIAdd %u32 %i %c1
OpStore %iter %next
OpBranch %header
%done = OpLabel
%out_ptr = OpAccessChain %ptr_word %out %c0 %gid_x
%result = OpLoad %u32 %acc
OpStore %out_ptr %result
OpReturn
OpFunctionEnd
)";
	return text;
}

enum class Mark { None, Vote, VoteNoBranch, Every, Seen, SeenVote };

std::string CostAccess(Mark mark, uint32_t k, const char* index, const char* x_base) {
	const auto n = std::to_string(k);
	std::string text = "%element_" + n + " = OpBitwiseAnd %u32 " + index + " %cmask\n" +
	                   "%x_sum_" + n + " = OpIAdd %u32 " + x_base + " %k" + n + "\n" +
	                   "%x_" + n + " = OpBitwiseAnd %u32 %x_sum_" + n + " %c15\n" +
	                   "%x_i_" + n + " = OpBitcast %i32 %x_" + n + "\n" +
	                   "%coord_" + n + " = OpCompositeConstruct %v2i %x_i_" + n + " %i0\n" +
	                   "%image_ptr_" + n + " = OpAccessChain %ptr_img %images %element_" + n + "\n" +
	                   "%image_" + n + " = OpLoad %img %image_ptr_" + n + "\n" +
	                   "%texel_" + n + " = OpImageFetch %v4f %image_" + n + " %coord_" + n + " Lod %i0\n" +
	                   "%red_" + n + " = OpCompositeExtract %f32 %texel_" + n + " 0\n" +
	                   "%bits_" + n + " = OpBitcast %u32 %red_" + n + "\n" +
	                   "%old_acc_" + n + " = OpLoad %u32 %acc\n" +
	                   "%new_acc_" + n + " = OpBitwiseXor %u32 %old_acc_" + n + " %bits_" + n + "\n" +
	                   "OpStore %acc %new_acc_" + n + "\n";
	if (mark == Mark::None) return text;
	text += "%word_" + n + " = OpShiftRightLogical %u32 %element_" + n + " %c5\n" +
	        "%word_ptr_" + n + " = OpAccessChain %ptr_word %reads %c0 %word_" + n + "\n" +
	        "%low_" + n + " = OpBitwiseAnd %u32 %element_" + n + " %c31\n" +
	        "%bit_" + n + " = OpShiftLeftLogical %u32 %c1 %low_" + n + "\n";
	if (mark == Mark::SeenVote) {
		return text + "%seen_" + n + " = OpLoad %u32 %word_ptr_" + n + "\n" + "%seen_bits_" + n + " = OpBitwiseAnd %u32 %seen_" + n +
		       " %bit_" + n + "\n" + "%unseen_" + n + " = OpIEqual %bool %seen_bits_" + n + " %c0\n" +
		       "OpSelectionMerge %checked_" + n + " None\n" + "OpBranchConditional %unseen_" + n + " %vote_" + n + " %checked_" + n +
		       "\n" + "%vote_" + n + " = OpLabel\n" + "%first_" + n + " = OpGroupNonUniformBroadcastFirst %u32 %c3 %element_" + n +
		       "\n" + "%elected_" + n + " = OpGroupNonUniformElect %bool %c3\n" + "%other_" + n + " = OpINotEqual %bool %element_" + n +
		       " %first_" + n + "\n" + "%store_" + n + " = OpLogicalOr %bool %elected_" + n + " %other_" + n + "\n" +
		       "OpSelectionMerge %marked_" + n + " None\n" + "OpBranchConditional %store_" + n + " %mark_" + n + " %marked_" + n +
		       "\n" + "%mark_" + n + " = OpLabel\n" + "%previous_" + n + " = OpAtomicOr %u32 %word_ptr_" + n + " %c1 %c0 %bit_" + n +
		       "\n" + "OpBranch %marked_" + n + "\n" + "%marked_" + n + " = OpLabel\n" + "OpBranch %checked_" + n + "\n" +
		       "%checked_" + n + " = OpLabel\n";
	}
	if (mark == Mark::Seen) {
		return text + "%seen_" + n + " = OpLoad %u32 %word_ptr_" + n + "\n" + "%seen_bits_" + n + " = OpBitwiseAnd %u32 %seen_" + n +
		       " %bit_" + n + "\n" + "%unseen_" + n + " = OpIEqual %bool %seen_bits_" + n + " %c0\n" +
		       "OpSelectionMerge %marked_" + n + " None\n" + "OpBranchConditional %unseen_" + n + " %mark_" + n + " %marked_" + n +
		       "\n" + "%mark_" + n + " = OpLabel\n" + "%previous_" + n + " = OpAtomicOr %u32 %word_ptr_" + n + " %c1 %c0 %bit_" + n +
		       "\n" + "OpBranch %marked_" + n + "\n" + "%marked_" + n + " = OpLabel\n";
	}
	if (mark == Mark::Every) {
		return text + "%previous_" + n + " = OpAtomicOr %u32 %word_ptr_" + n + " %c1 %c0 %bit_" + n + "\n";
	}
	text += "%first_" + n + " = OpGroupNonUniformBroadcastFirst %u32 %c3 %element_" + n + "\n" +
	        "%elected_" + n + " = OpGroupNonUniformElect %bool %c3\n" +
	        "%other_" + n + " = OpINotEqual %bool %element_" + n + " %first_" + n + "\n" +
	        "%store_" + n + " = OpLogicalOr %bool %elected_" + n + " %other_" + n + "\n";
	if (mark == Mark::VoteNoBranch) {
		return text + "%masked_" + n + " = OpSelect %u32 %store_" + n + " %bit_" + n + " %c0\n" +
		       "%previous_" + n + " = OpAtomicOr %u32 %word_ptr_" + n + " %c1 %c0 %masked_" + n + "\n";
	}
	return text + "OpSelectionMerge %marked_" + n + " None\n" +
	       "OpBranchConditional %store_" + n + " %mark_" + n + " %marked_" + n + "\n" +
	       "%mark_" + n + " = OpLabel\n" +
	       "%previous_" + n + " = OpAtomicOr %u32 %word_ptr_" + n + " %c1 %c0 %bit_" + n + "\n" +
	       "OpBranch %marked_" + n + "\n" +
	       "%marked_" + n + " = OpLabel\n";
}

constexpr uint32_t StraightAccesses = 8;

std::string CostText(Mark mark, bool loop) {
	std::string text = R"(
OpCapability Shader
OpCapability GroupNonUniform
OpCapability GroupNonUniformBallot
OpCapability ShaderNonUniform
OpCapability SampledImageArrayNonUniformIndexing
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid %lid %images %reads %out %push
OpExecutionMode %main LocalSize 64 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %lid BuiltIn LocalInvocationId
OpDecorate %images DescriptorSet 1
OpDecorate %images Binding 0
OpDecorate %reads DescriptorSet 1
OpDecorate %reads Binding 1
OpDecorate %out DescriptorSet 0
OpDecorate %out Binding 0
OpDecorate %runtime ArrayStride 4
OpMemberDecorate %Words 0 Offset 0
OpDecorate %Words Block
OpMemberDecorate %Push 0 Offset 0
OpMemberDecorate %Push 1 Offset 4
OpMemberDecorate %Push 2 Offset 8
OpMemberDecorate %Push 3 Offset 12
OpDecorate %Push Block
@DECOR@
%void = OpTypeVoid
%fn = OpTypeFunction %void
%bool = OpTypeBool
%u32 = OpTypeInt 32 0
%i32 = OpTypeInt 32 1
%f32 = OpTypeFloat 32
%v3u = OpTypeVector %u32 3
%v2i = OpTypeVector %i32 2
%v4f = OpTypeVector %f32 4
%c0 = OpConstant %u32 0
%c1 = OpConstant %u32 1
%c2 = OpConstant %u32 2
%c3 = OpConstant %u32 3
%c5 = OpConstant %u32 5
%c15 = OpConstant %u32 15
%k0 = OpConstant %u32 0
%k1 = OpConstant %u32 1
%k2 = OpConstant %u32 2
%k3 = OpConstant %u32 3
%k4 = OpConstant %u32 4
%k5 = OpConstant %u32 5
%k6 = OpConstant %u32 6
%k7 = OpConstant %u32 7
%c31 = OpConstant %u32 31
%cmask = OpConstant %u32 65535
%i0 = OpConstant %i32 0
%ptr_in = OpTypePointer Input %v3u
%ptr_in_u32 = OpTypePointer Input %u32
%gid = OpVariable %ptr_in Input
%lid = OpVariable %ptr_in Input
%img = OpTypeImage %f32 2D 0 0 0 1 Unknown
%cheap = OpConstant %u32 65536
%img_array = OpTypeArray %img %cheap
%ptr_images = OpTypePointer UniformConstant %img_array
%ptr_img = OpTypePointer UniformConstant %img
%images = OpVariable %ptr_images UniformConstant
%runtime = OpTypeRuntimeArray %u32
%Words = OpTypeStruct %runtime
%ptr_words = OpTypePointer StorageBuffer %Words
%ptr_word = OpTypePointer StorageBuffer %u32
%reads = OpVariable %ptr_words StorageBuffer
%out = OpVariable %ptr_words StorageBuffer
%Push = OpTypeStruct %u32 %u32 %u32 %u32
%ptr_push = OpTypePointer PushConstant %Push
%ptr_push_u32 = OpTypePointer PushConstant %u32
%push = OpVariable %ptr_push PushConstant
%ptr_fn_u32 = OpTypePointer Function %u32
%main = OpFunction %void None %fn
%entry = OpLabel
%acc = OpVariable %ptr_fn_u32 Function
%iter = OpVariable %ptr_fn_u32 Function
OpStore %acc %c0
OpStore %iter %c0
%gid_x_ptr = OpAccessChain %ptr_in_u32 %gid %c0
%gid_x = OpLoad %u32 %gid_x_ptr
%lid_x_ptr = OpAccessChain %ptr_in_u32 %lid %c0
%lid_x = OpLoad %u32 %lid_x_ptr
%p0 = OpAccessChain %ptr_push_u32 %push %c0
%index0 = OpLoad %u32 %p0
%p1 = OpAccessChain %ptr_push_u32 %push %c1
%index1 = OpLoad %u32 %p1
%p2 = OpAccessChain %ptr_push_u32 %push %c2
%count = OpLoad %u32 %p2
%p3 = OpAccessChain %ptr_push_u32 %push %c3
%lane_mask = OpLoad %u32 %p3
%odd_bit = OpBitwiseAnd %u32 %lid_x %lane_mask
%odd = OpINotEqual %bool %odd_bit %c0
%selected = OpSelect %u32 %odd %index1 %index0
)";
	std::string decorations;
	for (uint32_t k = 0; k < (loop ? 1u : StraightAccesses); ++k) {
		decorations += "OpDecorate %image_ptr_" + std::to_string(k) + " NonUniform\nOpDecorate %image_" + std::to_string(k) +
		               " NonUniform\n";
	}
	text.replace(text.find("@DECOR@"), 7, decorations);
	if (loop) {
		text += R"(
OpBranch %header
%header = OpLabel
OpLoopMerge %done %continue None
OpBranch %check
%check = OpLabel
%i = OpLoad %u32 %iter
%more = OpULessThan %bool %i %count
OpBranchConditional %more %body %done
%body = OpLabel
%x_base = OpIAdd %u32 %gid_x %i
)";
		text += CostAccess(mark, 0, "%selected", "%x_base");
		text += R"(
OpBranch %continue
%continue = OpLabel
%next = OpIAdd %u32 %i %c1
OpStore %iter %next
OpBranch %header
%done = OpLabel
)";
	} else {
		text += "%alternate = OpSelect %u32 %odd %index0 %index1\n";
		for (uint32_t k = 0; k < StraightAccesses; ++k) {
			text += CostAccess(mark, k, (k & 1u) ? "%alternate" : "%selected", "%gid_x");
		}
	}
	text += R"(
%out_ptr = OpAccessChain %ptr_word %out %c0 %gid_x
%result = OpLoad %u32 %acc
OpStore %out_ptr %result
OpReturn
OpFunctionEnd
)";
	return text;
}

std::optional<std::vector<uint32_t>> Assemble(const std::string& text, std::string& messages) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		messages += message;
		messages += '\n';
	});
	std::vector<uint32_t> spirv;
	if (!tools.Assemble(text, &spirv) || !tools.Validate(spirv)) return std::nullopt;
	return spirv;
}

struct Kernel {
	vk::Device              device;
	vk::DescriptorSetLayout out_layout;
	vk::DescriptorPool      pool;
	vk::DescriptorSet       out_set;
	vk::PipelineLayout      layout;
	vk::Pipeline            marked;
	vk::Pipeline            plain;
	vk::Pipeline            mark_only;
	std::vector<vk::Pipeline> extra;
	vk::QueryPool           queries;
	std::unique_ptr<Buffer> out;
	std::unique_ptr<Buffer> host;
	uint32_t                lanes = 0;

	~Kernel() {
		if (!device) return;
		for (const auto pipeline: {marked, plain, mark_only}) {
			if (pipeline) device.destroyPipeline(pipeline);
		}
		for (const auto pipeline: extra) {
			if (pipeline) device.destroyPipeline(pipeline);
		}
		if (layout) device.destroyPipelineLayout(layout);
		if (pool) device.destroyDescriptorPool(pool);
		if (out_layout) device.destroyDescriptorSetLayout(out_layout);
		if (queries) device.destroyQueryPool(queries);
	}
};

vk::Pipeline Pipeline(Kernel& kernel, const std::string& text) {
	std::string messages;
	const auto  spirv = Assemble(text, messages);
	if (!spirv) {
		Check(false, "the marking shader assembles and validates: " + messages);
		return nullptr;
	}
	const auto module = kernel.device.createShaderModule({.codeSize = spirv->size() * sizeof(uint32_t), .pCode = spirv->data()});
	if (module.result != vk::Result::eSuccess) return nullptr;
	const vk::ComputePipelineCreateInfo info {
	    .stage  = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module.value, .pName = "main"},
	    .layout = kernel.layout};
	auto pipeline = kernel.device.createComputePipelines(nullptr, info);
	kernel.device.destroyShaderModule(module.value);
	return pipeline.result == vk::Result::eSuccess ? pipeline.value.front() : nullptr;
}

bool MakeKernel(GraphicContext& graphics, CommandScheduler& scheduler, Kernel& kernel, uint32_t lanes) {
	kernel.device = graphics.device;
	kernel.lanes  = lanes;
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute};
	kernel.out_layout = graphics.device.createDescriptorSetLayout({.bindingCount = 1, .pBindings = &binding}).value;
	const std::array set_layouts {kernel.out_layout, graphics.image_heap_layout};
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, 4 * sizeof(uint32_t)};
	kernel.layout = graphics.device
	                    .createPipelineLayout({.setLayoutCount         = static_cast<uint32_t>(set_layouts.size()),
	                                           .pSetLayouts            = set_layouts.data(),
	                                           .pushConstantRangeCount = 1,
	                                           .pPushConstantRanges    = &push})
	                    .value;
	const vk::DescriptorPoolSize size {vk::DescriptorType::eStorageBuffer, 1};
	kernel.pool    = graphics.device.createDescriptorPool({.maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &size}).value;
	kernel.out_set = graphics.device.allocateDescriptorSets({.descriptorPool = kernel.pool, .descriptorSetCount = 1,
	                                                          .pSetLayouts = &kernel.out_layout})
	                     .value.front();
	const auto bytes = uint64_t {lanes} * sizeof(uint32_t);
	kernel.out = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
	                                      vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc,
	                                      bytes);
	kernel.host = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
	                                       vk::BufferUsageFlagBits::eTransferDst, bytes);
	const vk::DescriptorBufferInfo out_info {kernel.out->Handle(), 0, bytes};
	const vk::WriteDescriptorSet   write {.dstSet          = kernel.out_set,
	                                      .descriptorCount = 1,
	                                      .descriptorType  = vk::DescriptorType::eStorageBuffer,
	                                      .pBufferInfo     = &out_info};
	graphics.device.updateDescriptorSets(write, {});
	kernel.queries = graphics.device.createQueryPool({.queryType = vk::QueryType::eTimestamp, .queryCount = 2}).value;
	kernel.marked    = Pipeline(kernel, ShaderText(true, true));
	kernel.plain     = Pipeline(kernel, ShaderText(false, true));
	kernel.mark_only = Pipeline(kernel, ShaderText(true, false));
	return kernel.marked && kernel.plain && kernel.mark_only;
}

void Run(ImageHeap& heap, CommandScheduler& scheduler, Kernel& kernel, vk::Pipeline pipeline, uint32_t index0,
         uint32_t index1, uint32_t iterations, uint32_t groups, uint32_t lane_mask = 1u, bool timed = false) {
	auto& command = scheduler.Current();
	command.EndRendering();
	heap.Commit(command, vk::PipelineBindPoint::eCompute, kernel.layout);
	const std::array<uint32_t, 4> push {index0, index1, iterations, lane_mask};
	const auto layout = kernel.layout;
	const auto set    = kernel.out_set;
	const auto pool   = kernel.queries;
	const auto out    = kernel.out->Handle();
	const auto host   = kernel.host->Handle();
	command.Handle().Custom([=](vk::CommandBuffer native) {
		native.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
		native.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, set, {});
		native.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push.data());
		if (timed) {
			native.resetQueryPool(pool, 0, 2);
			native.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, pool, 0);
		}
		native.dispatch(groups, 1, 1);
		if (timed) native.writeTimestamp(vk::PipelineStageFlagBits::eAllCommands, pool, 1);
		const vk::MemoryBarrier barrier {.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
		                                 .dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eShaderRead};
		native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
		                       vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader, {}, barrier,
		                       {}, {});
		const vk::BufferCopy copy {0, 0, uint64_t {Lanes} * sizeof(uint32_t)};
		native.copyBuffer(out, host, copy);
	});
}

void Wait(CommandScheduler& scheduler) {
	const auto tick = scheduler.CurrentTick();
	scheduler.Finish();
	scheduler.Wait(tick);
}

std::vector<uint32_t> Lanes0(Kernel& kernel) {
	std::vector<uint32_t> values(Lanes);
	kernel.host->Invalidate(0, values.size() * sizeof(uint32_t));
	std::memcpy(values.data(), kernel.host->Mapped().data(), values.size() * sizeof(uint32_t));
	return values;
}

std::vector<uint8_t> ReadImage(CommandScheduler& scheduler, GraphicContext& graphics, Image& image) {
	const auto bytes = uint64_t {TexelCount};
	Buffer     host(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, bytes);
	auto&      command = scheduler.Current();
	command.EndRendering();
	image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command.Handle());
	const auto source = image.backing.image;
	const auto target = host.Handle();
	command.Handle().Custom([=](vk::CommandBuffer native) {
		const vk::BufferImageCopy region {.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
		                                  .imageExtent      = {TexelCount, 1, 1}};
		native.copyImageToBuffer(source, vk::ImageLayout::eTransferSrcOptimal, target, region);
	});
	Wait(scheduler);
	std::vector<uint8_t> values(bytes);
	host.Invalidate(0, bytes);
	std::memcpy(values.data(), host.Mapped().data(), bytes);
	return values;
}

bool BitmapClear(CommandScheduler& scheduler, GraphicContext& graphics, const ImageHeap& heap) {
	const auto& reads = Access::Reads(heap);
	Buffer      host(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, reads.Size());
	auto&       command = scheduler.Current();
	command.EndRendering();
	const vk::MemoryBarrier barrier {.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
	                                 .dstAccessMask = vk::AccessFlagBits::eTransferRead};
	command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
	                                 &barrier, 0, nullptr, 0, nullptr);
	host.CopyFrom(command, reads, 0, 0, reads.Size());
	Wait(scheduler);
	host.Invalidate(0, reads.Size());
	std::vector<uint32_t> words(reads.Size() / sizeof(uint32_t));
	std::memcpy(words.data(), host.Mapped().data(), reads.Size());
	return std::ranges::all_of(words, [](uint32_t word) { return word == 0u; });
}

uint32_t TexelBits(uint8_t value) {
	const float unorm = static_cast<float>(value) / 255.0f;
	uint32_t    bits  = 0;
	std::memcpy(&bits, &unorm, sizeof(bits));
	return bits;
}

bool SameTexel(uint32_t bits, uint8_t expected) {
	float value = 0.0f;
	std::memcpy(&value, &bits, sizeof(value));
	return std::lround(value * 255.0f) == expected && (expected != 0u || bits == 0u);
}

void CheckReads(GraphicContext& graphics, uint64_t address, uint64_t bytes) {
	RenderContext renderer(graphics);
	if (!renderer.MapMemory(address, bytes)) {
		Check(false, "map the texture table into the GPU mirror");
		return;
	}
	CommandProcessor processor(renderer, 0);
	processor.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(processor.GetCtx(), processor.GetUcfg(), processor.GetShCtx());
	auto& heap     = renderer.GetImageHeap();
	auto& buffers  = renderer.GetBufferCache();
	auto& textures = renderer.GetTextureCache();
	auto& executor = renderer.GetRenderExecutor();
	RenderExecutorTestAccess::Reset(executor);

	std::array<IR::DescriptorValue, Textures>                  values {};
	std::array<std::array<uint8_t, TexelCount>, Textures>      texels {};
	for (uint32_t texture = 0; texture < Textures; ++texture) {
		const auto base = address + 0x10000u + uint64_t {texture} * 0x4000u;
		values[texture] = HeapTexture(base);
		for (uint32_t texel = 0; texel < TexelCount; ++texel) {
			texels[texture][texel] = static_cast<uint8_t>(1u + texture * 37u + texel * 11u);
		}
		Libs::LibKernel::Memory::WriteBacking(base, texels[texture].data(), TexelCount);
		buffers.InvalidateMemory(base, TexelCount);
		Libs::LibKernel::Memory::WriteBacking(address + uint64_t {texture} * 32u, values[texture].dwords.data(),
		                                      sizeof(values[texture].dwords));
		buffers.InvalidateMemory(address + uint64_t {texture} * 32u, sizeof(values[texture].dwords));
	}
	IR::ImageResource root;
	root.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
	const auto                 profile = IR::ImageTableHeapProfile(root, false);
	const IR::ImageHeapRequest request {.base  = address,
	                                    .size  = Textures * 32u,
	                                    .step  = 32u,
	                                    .count = Textures,
	                                    .root  = &profile,
	                                    .table = true};
	IR::ImageHeapView view;
	heap.BeginDraw();
	if (!heap.Acquire(request, view)) {
		Check(false, "the heap admits a table of four textures");
		return;
	}
	heap.Rebind(executor);
	constexpr uint32_t Read = 0, Unread = 1, Target = 2, Written = 3;
	std::array<uint32_t, Textures> slots {};
	std::array<ImageId, Textures>  images {};
	for (uint32_t texture = 0; texture < Textures; ++texture) {
		slots[texture]  = Access::Slot(heap, values[texture]);
		images[texture] = slots[texture] != 0u ? Access::ImageOf(heap, slots[texture]) : ImageId {};
		Check(slots[texture] != 0u && static_cast<bool>(images[texture]), "each texture resolves to a heap slot and an image");
	}
	if (!std::ranges::all_of(images, [](ImageId id) { return static_cast<bool>(id); })) return;
	Access::Find(textures, images[Target])->usage.render_target = true;
	Access::Find(textures, images[Written])->MarkGpuModified();
	const auto before = ReadImage(scheduler, graphics, *Access::Find(textures, images[Unread]));
	Check(std::ranges::equal(before, texels[Unread]), "the unread texture holds the guest bytes before eviction");

	Kernel kernel;
	if (!MakeKernel(graphics, scheduler, kernel, Lanes)) {
		Check(false, "the marking pipelines");
		return;
	}
	const auto t0 = Clock::now();
	Run(heap, scheduler, kernel, kernel.marked, slots[Read], slots[Read], 1, 1);
	Access::Collect(heap, t0 + std::chrono::seconds(1));
	Wait(scheduler);
	const auto first = Lanes0(kernel);
	Check(SameTexel(first[0], texels[Read][0]) && SameTexel(first[5], texels[Read][5]),
	      "a heap read through the marking shader returns the texture");
	Check(BitmapClear(scheduler, graphics, heap), "the device bitmap is zero after the readback copies it");
	Run(heap, scheduler, kernel, kernel.marked, slots[Read], slots[Read], 1, 1);
	Access::Collect(heap, t0 + std::chrono::seconds(62));
	Wait(scheduler);
	Check(Access::ReadStamp(heap, slots[Read]) != 0u, "a shader read stamps its slot");
	Check(Access::ReadStamp(heap, slots[Unread]) == 0u, "an unread slot has no read stamp");
	Access::Pressure(graphics, uint64_t {4} << 30u, uint64_t {38} << 27u);
	Access::Collect(heap, t0 + std::chrono::seconds(63));
	Check(Access::Parking(heap) == 1u, "only the texture unread for 60 s is parked");
	heap.BeginDraw();
	Check(heap.Acquire(request, view), "the table stays on the heap while a slot is parked");
	heap.Rebind(executor);
	const auto parked = Access::Slot(heap, values[Unread]);
	Check(parked != 0u && parked != slots[Unread] && Access::Parked(heap, parked) && !Access::ImageOf(heap, parked),
	      "the unread texture moves to a parked slot with a null descriptor");
	for (const auto texture: {Read, Target, Written}) {
		const auto slot = Access::Slot(heap, values[texture]);
		Check(slot == slots[texture] && !Access::Parked(heap, slot) && Access::ImageOf(heap, slot) == images[texture],
		      "a read texture, a render target and a GPU-written texture keep their slot and image");
	}
	Check(Access::Find(textures, images[Unread])->heap_refs.load() == 0u, "the parked texture is no longer pinned by the heap");
	Check(Access::Find(textures, images[Read])->heap_refs.load() != 0u, "the read texture stays pinned");

	Common::DebugCounters::SetEnabled(true);
	const auto evictions = Common::DebugCounters::Get(Common::DebugCounters::Counter::TextureEvictions);
	for (uint32_t pass = 0; pass < 8u; ++pass) {
		heap.BeginDraw();
		(void)heap.Acquire(request, view);
		heap.Rebind(executor);
		Run(heap, scheduler, kernel, kernel.marked, slots[Read], slots[Read], 1, 1);
		textures.RunGarbageCollector();
		Wait(scheduler);
	}
	const auto* evicted = Access::Find(textures, images[Unread]);
	Check(evicted == nullptr || !evicted->registered, "the texture cache frees the parked texture under pressure");
	Check(Common::DebugCounters::Get(Common::DebugCounters::Counter::TextureEvictions) > evictions,
	      "the eviction is counted");
	for (const auto texture: {Read, Target, Written}) {
		const auto* image = Access::Find(textures, images[texture]);
		Check(image != nullptr && image->registered, "the read texture, the target and the GPU-written texture stay");
	}

	Access::Collect(heap, t0 + std::chrono::seconds(64));
	Wait(scheduler);
	Access::Collect(heap, t0 + std::chrono::seconds(65));
	Wait(scheduler);
	const auto stamp = Access::ReadStamp(heap, slots[Read]);
	Access::Collect(heap, t0 + std::chrono::seconds(66));
	Wait(scheduler);
	Access::Collect(heap, t0 + std::chrono::seconds(67));
	Wait(scheduler);
	Check(Access::ReadStamp(heap, slots[Read]) == stamp, "a cleared bitmap reports no read when no shader ran");
	Check(BitmapClear(scheduler, graphics, heap), "the device bitmap stays zero without reads");

	Run(heap, scheduler, kernel, kernel.marked, parked, slots[Read], 1, 1);
	Wait(scheduler);
	const auto null_reads = Lanes0(kernel);
	Check(null_reads[0] == 0u && SameTexel(null_reads[1], texels[Read][1]),
	      "a parked slot reads zero while its neighbour lane reads the live texture");
	Access::Collect(heap, t0 + std::chrono::seconds(68));
	Wait(scheduler);
	Access::Collect(heap, t0 + std::chrono::seconds(69));
	heap.BeginDraw();
	Check(heap.Acquire(request, view), "the table is admitted after the read");
	heap.Rebind(executor);
	const auto back = Access::Slot(heap, values[Unread]);
	const auto back_image = back != 0u ? Access::ImageOf(heap, back) : ImageId {};
	Check(back != 0u && back != parked && !Access::Parked(heap, back) && static_cast<bool>(back_image),
	      "a parked slot read by a shader comes back with an image");
	Run(heap, scheduler, kernel, kernel.marked, back, back, 1, 1);
	Wait(scheduler);
	Check(SameTexel(Lanes0(kernel)[3], texels[Unread][3]), "the shader reads the texture again through its new slot");
	if (back_image) {
		auto* image = Access::Find(textures, back_image);
		Check(image != nullptr && image->heap_refs.load() != 0u, "the texture made again is pinned by the heap");
		if (image != nullptr) {
			const auto after = ReadImage(scheduler, graphics, *image);
			Check(after == before, "the texture made again from guest memory has the same bytes");
		}
	}
	Access::Relax(graphics);
	scheduler.Finish();
	renderer.UnmapMemory(address, bytes);
	std::printf("%s: parked=%llu read_again=%llu readbacks=%llu\n", g_name,
	            static_cast<unsigned long long>(heap.Reads().parked),
	            static_cast<unsigned long long>(heap.Reads().unparked),
	            static_cast<unsigned long long>(heap.Reads().readbacks));
}

void CheckCrowded(GraphicContext& graphics, uint64_t address, uint64_t bytes) {
	RenderContext renderer(graphics);
	renderer.MapMemory(address, bytes);
	CommandProcessor processor(renderer, 0);
	processor.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(processor.GetCtx(), processor.GetUcfg(), processor.GetShCtx());
	auto& heap     = renderer.GetImageHeap();
	auto& buffers  = renderer.GetBufferCache();
	auto& textures = renderer.GetTextureCache();
	auto& executor = renderer.GetRenderExecutor();
	RenderExecutorTestAccess::Reset(executor);

	std::array<IR::DescriptorValue, Textures>             values {};
	std::array<std::array<uint8_t, TexelCount>, Textures> texels {};
	for (uint32_t texture = 0; texture < Textures; ++texture) {
		const auto base = address + 0x10000u + uint64_t {texture} * 0x4000u;
		values[texture] = HeapTexture(base);
		for (uint32_t texel = 0; texel < TexelCount; ++texel) {
			texels[texture][texel] = static_cast<uint8_t>(3u + texture * 41u + texel * 13u);
		}
		Libs::LibKernel::Memory::WriteBacking(base, texels[texture].data(), TexelCount);
		buffers.InvalidateMemory(base, TexelCount);
		Libs::LibKernel::Memory::WriteBacking(address + uint64_t {texture} * 32u, values[texture].dwords.data(),
		                                      sizeof(values[texture].dwords));
		buffers.InvalidateMemory(address + uint64_t {texture} * 32u, sizeof(values[texture].dwords));
	}
	IR::ImageResource root;
	root.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
	const auto                 profile = IR::ImageTableHeapProfile(root, false);
	const IR::ImageHeapRequest request {.base  = address,
	                                    .size  = Textures * 32u,
	                                    .step  = 32u,
	                                    .count = Textures,
	                                    .root  = &profile,
	                                    .table = true};
	IR::ImageHeapView view;
	heap.BeginDraw();
	if (!heap.Acquire(request, view)) {
		Check(false, "the heap admits a table of four textures");
		return;
	}
	heap.Rebind(executor);
	constexpr uint32_t Hot = 0, Once = 1, ColdFirst = 2, ColdSecond = 3;
	std::array<uint32_t, Textures> slots {};
	std::array<ImageId, Textures>  images {};
	for (uint32_t texture = 0; texture < Textures; ++texture) {
		slots[texture]  = Access::Slot(heap, values[texture]);
		images[texture] = slots[texture] != 0u ? Access::ImageOf(heap, slots[texture]) : ImageId {};
		Check(slots[texture] != 0u && static_cast<bool>(images[texture]), "each texture resolves to a heap slot and an image");
	}
	if (!std::ranges::all_of(images, [](ImageId id) { return static_cast<bool>(id); })) return;
	std::array<uint64_t, Textures> charges {};
	for (uint32_t texture = 0; texture < Textures; ++texture) {
		const auto& charge = Access::Find(textures, images[texture])->backing.charge;
		charges[texture]   = charge.bytes;
		Check(charge.local && charge.bytes != 0u, "each texture is charged to device-local memory");
	}
	const auto cold_before = ReadImage(scheduler, graphics, *Access::Find(textures, images[ColdFirst]));
	Check(std::ranges::equal(cold_before, texels[ColdFirst]), "the cold texture holds the guest bytes before eviction");

	Kernel kernel;
	if (!MakeKernel(graphics, scheduler, kernel, Lanes)) {
		Check(false, "the marking pipelines");
		return;
	}
	const auto t0 = Clock::now();
	const auto at = [&](uint32_t ms) { return t0 + std::chrono::milliseconds(ms); };
	Run(heap, scheduler, kernel, kernel.marked, slots[Hot], slots[Once], 1, 1);
	Access::Collect(heap, at(100));
	Wait(scheduler);
	for (uint32_t pass = 1; pass <= 6u; ++pass) {
		Run(heap, scheduler, kernel, kernel.marked, slots[Hot], slots[Hot], 1, 1);
		Access::Collect(heap, at(100u + pass * 600u));
		Wait(scheduler);
	}
	Check(Access::ReadStamp(heap, slots[Hot]) > Access::ReadStamp(heap, slots[Once]) &&
	          Access::ReadStamp(heap, slots[Once]) != 0u && Access::ReadStamp(heap, slots[ColdFirst]) == 0u,
	      "the read stamps order the hot, the once read and the cold textures");
	Check(Access::Parking(heap) == 0u && !Access::Crowded(heap), "nothing is parked below the crowded mark");

	const uint64_t two_cold = charges[ColdFirst] + charges[ColdSecond];
	const uint64_t wanted   = two_cold - std::min(charges[ColdFirst], charges[ColdSecond]) / 2u;
	const uint64_t unit     = wanted / 5u;
	const uint64_t budget   = unit * 100u;
	const uint64_t usage    = GraphicContext::HeapImageThreshold(budget, true) + wanted;
	Check(usage >= GraphicContext::HeapImageThreshold(budget), "the test usage is past the crowded mark");
	heap.BeginDraw();
	Check(heap.Acquire(request, view), "the table is drawn again before the crowded check");
	heap.Rebind(executor);
	Run(heap, scheduler, kernel, kernel.marked, slots[Hot], slots[Hot], 1, 1);
	Wait(scheduler);
	Access::Pressure(graphics, budget, usage);
	Check(graphics.LocalMemoryCrowded(), "the context reports crowded device-local memory");
	Access::Collect(heap, at(4800));
	Check(Access::Crowded(heap), "the heap enters the crowded state at the high watermark");
	Check(Access::Parking(heap) == 2u, "the crowded heap parks the two least read textures and no more than the bytes it needs");
	const auto pending = Access::PendingProbes(heap);
	std::string probes = Access::Views(heap, slots[ColdFirst]) + ";";
	for (const auto probe: pending) probes += " " + std::to_string(probe);
	Check(std::ranges::binary_search(pending, ColdFirst) && std::ranges::binary_search(pending, ColdSecond) &&
	          !std::ranges::binary_search(pending, Hot) && !std::ranges::binary_search(pending, Once),
	      "the batched repend queues the probes of the parked textures only (pending:" + probes + ")");
	for (const auto texture: {ColdFirst, ColdSecond}) {
		Check(Access::Find(textures, images[texture])->heap_refs.load() == 0u, "a parked texture is no longer pinned");
	}
	heap.BeginDraw();
	Check(heap.Acquire(request, view), "the table stays on the heap while slots are parked");
	heap.Rebind(executor);
	for (const auto texture: {ColdFirst, ColdSecond}) {
		const auto slot = Access::Slot(heap, values[texture]);
		Check(slot != 0u && slot != slots[texture] && Access::Parked(heap, slot) && !Access::ImageOf(heap, slot),
		      "a cold texture moves to a parked slot with a null descriptor");
	}
	for (const auto texture: {Hot, Once}) {
		const auto slot = Access::Slot(heap, values[texture]);
		Check(slot == slots[texture] && !Access::Parked(heap, slot) && Access::ImageOf(heap, slot) == images[texture],
		      "the hot and the once read textures keep their slot and image");
	}
	Common::DebugCounters::SetEnabled(true);
	const auto evictions = Common::DebugCounters::Get(Common::DebugCounters::Counter::TextureEvictions);
	textures.RunGarbageCollector();
	for (const auto texture: {ColdFirst, ColdSecond}) {
		const auto* image = Access::Find(textures, images[texture]);
		Check(image == nullptr || !image->registered, "the texture cache frees a parked texture in the same pass");
	}
	Check(Common::DebugCounters::Get(Common::DebugCounters::Counter::TextureEvictions) >= evictions + 2u,
	      "the two evictions are counted");
	for (const auto texture: {Hot, Once}) {
		const auto* image = Access::Find(textures, images[texture]);
		Check(image != nullptr && image->registered, "the hot and the once read textures stay");
	}
	Wait(scheduler);

	Access::Pressure(graphics, budget, unit * 95u);
	Access::Collect(heap, at(5100));
	Wait(scheduler);
	Check(Access::Parking(heap) == 1u, "a higher usage parks the texture read once, the next least read");
	heap.BeginDraw();
	(void)heap.Acquire(request, view);
	heap.Rebind(executor);
	const auto hot = Access::Slot(heap, values[Hot]);
	Check(hot == slots[Hot] && !Access::Parked(heap, hot), "a texture read in each of the last readbacks is never parked");
	const auto once = Access::Slot(heap, values[Once]);
	Check(once != slots[Once] && Access::Parked(heap, once), "the texture read once moves to a parked slot");
	textures.RunGarbageCollector();
	Wait(scheduler);

	const auto parked = Access::Slot(heap, values[ColdFirst]);
	Run(heap, scheduler, kernel, kernel.marked, parked, slots[Hot], 1, 1);
	Wait(scheduler);
	const auto null_reads = Lanes0(kernel);
	Check(null_reads[0] == 0u && SameTexel(null_reads[1], texels[Hot][1]),
	      "a parked slot reads zero while its neighbour lane reads the hot texture");
	Access::Collect(heap, at(5400));
	Wait(scheduler);
	Access::Collect(heap, at(5700));
	heap.BeginDraw();
	Check(heap.Acquire(request, view), "the table is admitted after the read");
	heap.Rebind(executor);
	const auto back       = Access::Slot(heap, values[ColdFirst]);
	const auto back_image = back != 0u ? Access::ImageOf(heap, back) : ImageId {};
	Check(back != 0u && back != parked && !Access::Parked(heap, back) && static_cast<bool>(back_image),
	      "a parked slot read by a shader comes back with an image");
	if (back_image) {
		auto* image = Access::Find(textures, back_image);
		Check(image != nullptr && ReadImage(scheduler, graphics, *image) == cold_before,
		      "the texture made again has the same bytes");
	}
	graphics.TrimRecycledImages();
	vk::ImageCreateInfo heap_info {};
	heap_info.imageType     = vk::ImageType::e2D;
	heap_info.format        = vk::Format::eR8G8B8A8Unorm;
	heap_info.extent        = vk::Extent3D {48, 48, 1};
	heap_info.mipLevels     = 1;
	heap_info.arrayLayers   = 1;
	heap_info.samples       = vk::SampleCountFlagBits::e1;
	heap_info.tiling        = vk::ImageTiling::eOptimal;
	heap_info.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	heap_info.sharingMode   = vk::SharingMode::eExclusive;
	heap_info.initialLayout = vk::ImageLayout::eUndefined;
	Check(GraphicContext::HeapImageThreshold(100ull << 30u) == (83ull << 30u), "large heaps reserve at most two extra GiB");
	Access::Pressure(graphics, 100ull << 20u, 69ull << 20u);
	Check(!graphics.LocalMemoryCrowded(), "heap allocation still prefers local memory at 69 percent");
	Access::Pressure(graphics, 100ull << 20u, 70ull << 20u);
	VulkanImage crowded_image {};
	Check(graphics.CreateImage(heap_info, crowded_image, true) && crowded_image.spilled == Access::CanSpill(graphics) &&
	          crowded_image.charge.local != Access::CanSpill(graphics),
	      "a heap image made while memory is crowded is placed in system memory");
	if (crowded_image.image) graphics.DeleteImage(crowded_image);
	Run(heap, scheduler, kernel, kernel.marked, back, back, 1, 1);
	Wait(scheduler);
	Check(SameTexel(Lanes0(kernel)[3], texels[ColdFirst][3]), "the shader reads the texture again through its new slot");
	Access::Pressure(graphics, budget, GraphicContext::HeapImageThreshold(budget, true) - 1u);
	Access::Collect(heap, at(6100));
	Check(!Access::Crowded(heap) && !graphics.LocalMemoryCrowded(), "the crowded state ends below the low watermark");
	graphics.TrimRecycledImages();
	VulkanImage relaxed_image {};
	Check(graphics.CreateImage(heap_info, relaxed_image, true) && !relaxed_image.spilled && relaxed_image.charge.local,
	      "a heap image made below the crowded mark stays on device-local memory");
	if (relaxed_image.image) graphics.DeleteImage(relaxed_image);
	graphics.TrimRecycledImages();
	Access::Relax(graphics);
	scheduler.Finish();
	renderer.UnmapMemory(address, bytes);
	std::printf("%s: crowded parks=%llu (%llu bytes) read_again=%llu readbacks=%llu\n", g_name,
	            static_cast<unsigned long long>(heap.Reads().crowded),
	            static_cast<unsigned long long>(heap.Reads().crowded_bytes),
	            static_cast<unsigned long long>(heap.Reads().unparked),
	            static_cast<unsigned long long>(heap.Reads().readbacks));
}

constexpr uint32_t ScaleTextures = 8192;
constexpr uint32_t ScaleSide     = 64;
constexpr uint64_t ScaleBytes    = 0x10000u + uint64_t {ScaleTextures} * ScaleSide * ScaleSide;

void MeasureCrowded(GraphicContext& graphics, uint64_t address, uint64_t bytes) {
	using Clock      = std::chrono::steady_clock;
	const auto since = [](Clock::time_point start) {
		return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
	};
	RenderContext renderer(graphics);
	renderer.MapMemory(address, bytes);
	CommandProcessor processor(renderer, 0);
	processor.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(processor.GetCtx(), processor.GetUcfg(), processor.GetShCtx());
	auto& heap     = renderer.GetImageHeap();
	auto& buffers  = renderer.GetBufferCache();
	auto& textures = renderer.GetTextureCache();
	auto& executor = renderer.GetRenderExecutor();
	RenderExecutorTestAccess::Reset(executor);
	std::vector<IR::DescriptorValue> values(ScaleTextures);
	std::vector<uint8_t>             texels(uint64_t {ScaleSide} * ScaleSide);
	for (uint32_t texture = 0; texture < ScaleTextures; ++texture) {
		const auto base = address + 0x10000u + uint64_t {texture} * texels.size();
		values[texture] = SquareTexture(base, ScaleSide);
		for (size_t texel = 0; texel < texels.size(); ++texel) texels[texel] = static_cast<uint8_t>(texture * 7u + texel);
		Libs::LibKernel::Memory::WriteBacking(base, texels.data(), texels.size());
		Libs::LibKernel::Memory::WriteBacking(address + uint64_t {texture} * 32u, values[texture].dwords.data(),
		                                      sizeof(values[texture].dwords));
	}
	buffers.InvalidateMemory(address, bytes);
	IR::ImageResource root;
	root.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
	const auto                 profile = IR::ImageTableHeapProfile(root, false);
	const IR::ImageHeapRequest request {.base  = address,
	                                    .size  = ScaleTextures * 32u,
	                                    .step  = 32u,
	                                    .count = ScaleTextures,
	                                    .root  = &profile,
	                                    .table = true};
	IR::ImageHeapView view;
	const auto        resolve_start = Clock::now();
	for (uint32_t attempt = 0; attempt < 4u; ++attempt) {
		heap.BeginDraw();
		(void)heap.Acquire(request, view);
		heap.Rebind(executor);
	}
	const auto resolve_ms = since(resolve_start);
	std::vector<uint32_t> slots(ScaleTextures);
	uint32_t              resolved = 0;
	for (uint32_t texture = 0; texture < ScaleTextures; ++texture) {
		slots[texture] = Access::Slot(heap, values[texture]);
		if (slots[texture] != 0u && Access::ImageOf(heap, slots[texture])) ++resolved;
	}
	Check(resolved == ScaleTextures, "every texture of the large table resolves to a heap slot (" + std::to_string(resolved) + ")");
	if (resolved != ScaleTextures) return;
	Kernel kernel;
	if (!MakeKernel(graphics, scheduler, kernel, Lanes)) {
		Check(false, "the marking pipelines");
		return;
	}
	constexpr uint32_t Hot = ScaleTextures / 4u;
	const auto         t0  = Clock::now();
	const auto         at  = [&](uint32_t ms) { return t0 + std::chrono::milliseconds(ms); };
	for (uint32_t pass = 0; pass <= 6u; ++pass) {
		for (uint32_t texture = 0; texture < Hot; ++texture) Access::MarkRead(heap, slots[texture]);
		Run(heap, scheduler, kernel, kernel.marked, slots[0], slots[1], 1, 1);
		Access::Collect(heap, at(100u + pass * 600u));
		Wait(scheduler);
	}
	const auto& local_heap = MemoryTelemetry::totals[static_cast<size_t>(MemoryTelemetry::Kind::HeapTexture)][1];
	const auto  live       = local_heap.bytes.load();
	const auto  live_count = local_heap.count.load();
	const auto  real       = [&] {
        const bool overridden = Access::Overridden(graphics);
        if (overridden) Access::Relax(graphics);
        const auto local = graphics.GetLocalMemoryUsage();
        const auto space = graphics.LocalBlockSpace();
        return std::pair {local.usage, space};
	};
	const auto before = real();
	for (uint32_t texture = 0; texture < Hot; ++texture) Access::MarkRead(heap, slots[texture]);
	const uint64_t budget = live * 10u;
	const uint64_t usage  = GraphicContext::HeapImageThreshold(budget);
	const uint64_t wanted = usage - GraphicContext::HeapImageThreshold(budget, true);
	Access::Pressure(graphics, budget, usage);
	const auto check_start = Clock::now();
	Access::Collect(heap, at(4800));
	const auto check_ms     = since(check_start);
	const auto parked       = heap.Reads().crowded;
	const auto parked_bytes = heap.Reads().crowded_bytes;
	const auto rebind_start = Clock::now();
	heap.BeginDraw();
	(void)heap.Acquire(request, view);
	heap.Rebind(executor);
	const auto rebind_ms = since(rebind_start);
	const auto gc_start  = Clock::now();
	textures.RunGarbageCollector();
	const auto gc_ms = since(gc_start);
	Wait(scheduler);
	Run(heap, scheduler, kernel, kernel.marked, slots[0], slots[1], 1, 1);
	Wait(scheduler);
	graphics.TrimRecycledImages();
	const auto after_live  = local_heap.bytes.load();
	const auto after_count = local_heap.count.load();
	const auto after       = real();
	uint32_t   hot_parked  = 0;
	for (uint32_t texture = 0; texture < Hot; ++texture) {
		const auto slot = Access::Slot(heap, values[texture]);
		if (slot != slots[texture] || Access::Parked(heap, slot)) ++hot_parked;
	}
	Check(hot_parked == 0u, "no texture read in the last readbacks is parked (" + std::to_string(hot_parked) + ")");
	Check(parked_bytes >= wanted && parked_bytes < wanted + live / live_count * 2u,
	      "the crowded check parks the bytes it needs and stops there");
	Check(live - after_live >= parked_bytes, "the texture cache frees on device-local memory every byte the heap parked");
	std::printf("%s: %u textures of %ux%u (%llu heap images, %.1f MiB on device-local memory) resolved in %.1f ms; "
	            "%u read in each readback; asked %.1f MiB, crowded check %.2f ms parked %llu slots (%.1f MiB); rebind %.2f ms; "
	            "texture collection %.2f ms; heap images on device-local memory %.1f -> %.1f MiB (%llu -> %llu); "
	            "process device-local usage %.1f -> %.1f MiB, free space inside VMA blocks %.1f -> %.1f MiB\n",
	            g_name, ScaleTextures, ScaleSide, ScaleSide, static_cast<unsigned long long>(live_count),
	            static_cast<double>(live) / 1048576.0, resolve_ms, Hot, static_cast<double>(wanted) / 1048576.0, check_ms,
	            static_cast<unsigned long long>(parked), static_cast<double>(parked_bytes) / 1048576.0, rebind_ms, gc_ms,
	            static_cast<double>(live) / 1048576.0, static_cast<double>(after_live) / 1048576.0,
	            static_cast<unsigned long long>(live_count), static_cast<unsigned long long>(after_count),
	            static_cast<double>(before.first) / 1048576.0, static_cast<double>(after.first) / 1048576.0,
	            static_cast<double>(before.second) / 1048576.0, static_cast<double>(after.second) / 1048576.0);
	scheduler.Finish();
	renderer.UnmapMemory(address, bytes);
}

void MeasureCost(GraphicContext& graphics, uint64_t address, uint64_t bytes) {
	RenderContext renderer(graphics);
	if (!renderer.MapMemory(address, bytes)) {
		Check(false, "map the texture table into the GPU mirror");
		return;
	}
	CommandProcessor processor(renderer, 0);
	processor.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	scheduler.Begin(processor.GetCtx(), processor.GetUcfg(), processor.GetShCtx());
	auto& heap     = renderer.GetImageHeap();
	auto& buffers  = renderer.GetBufferCache();
	auto& executor = renderer.GetRenderExecutor();
	RenderExecutorTestAccess::Reset(executor);
	std::array<IR::DescriptorValue, 2> values {};
	for (uint32_t texture = 0; texture < values.size(); ++texture) {
		const auto base = address + 0x10000u + uint64_t {texture} * 0x4000u;
		values[texture] = HeapTexture(base);
		std::array<uint8_t, TexelCount> texels {};
		for (uint32_t texel = 0; texel < TexelCount; ++texel) texels[texel] = static_cast<uint8_t>(texel * 7u + texture);
		Libs::LibKernel::Memory::WriteBacking(base, texels.data(), TexelCount);
		buffers.InvalidateMemory(base, TexelCount);
		Libs::LibKernel::Memory::WriteBacking(address + uint64_t {texture} * 32u, values[texture].dwords.data(),
		                                      sizeof(values[texture].dwords));
		buffers.InvalidateMemory(address + uint64_t {texture} * 32u, sizeof(values[texture].dwords));
	}
	IR::ImageResource root;
	root.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
	const auto                 profile = IR::ImageTableHeapProfile(root, false);
	const IR::ImageHeapRequest request {.base = address, .size = 64u, .step = 32u, .count = 2u, .root = &profile, .table = true};
	IR::ImageHeapView view;
	heap.BeginDraw();
	if (!heap.Acquire(request, view)) {
		Check(false, "the cost table is admitted");
		return;
	}
	heap.Rebind(executor);
	const auto a = Access::Slot(heap, values[0]);
	const auto b = Access::Slot(heap, values[1]);
	if (a == 0u || b == 0u || std::max(a, b) - std::min(a, b) != 1u) {
		Check(false, "the cost table acquired two consecutive live heap slots");
		return;
	}
	constexpr uint32_t Groups     = 16384;
	constexpr uint32_t Iterations = 64;
	Kernel kernel;
	if (!MakeKernel(graphics, scheduler, kernel, Groups * Lanes)) {
		Check(false, "the cost pipelines");
		return;
	}
	const auto period = graphics.physical_device.getProperties().limits.timestampPeriod;
	const auto time   = [&](vk::Pipeline pipeline, bool divergent) {
		double best = 1e30;
		for (uint32_t repeat = 0; repeat < 7u; ++repeat) {
			Run(heap, scheduler, kernel, pipeline, a, b, Iterations, Groups, divergent ? 1u : 0u, true);
			Wait(scheduler);
			std::array<uint64_t, 2> stamps {};
			if (graphics.device.getQueryPoolResults(kernel.queries, 0, 2, sizeof(stamps), stamps.data(), sizeof(uint64_t),
			                                        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait) !=
			    vk::Result::eSuccess) {
				return -1.0;
			}
			if (repeat != 0u) best = std::min(best, static_cast<double>(stamps[1] - stamps[0]) * period);
			heap.RunGarbageCollector();
		}
		return best;
	};
	const std::array<std::pair<Mark, const char*>, 6> marks {std::pair {Mark::None, "no mark"},
	                                                          std::pair {Mark::Vote, "vote and branch"},
	                                                          std::pair {Mark::VoteNoBranch, "vote, no branch"},
	                                                          std::pair {Mark::Every, "every lane, no vote"},
	                                                          std::pair {Mark::Seen, "load, then OR if unset"},
	                                                          std::pair {Mark::SeenVote, "load, then vote and OR if unset (emitted)"}};
	for (const bool loop: {true, false}) {
		const double accesses = double {Groups} * Lanes * (loop ? Iterations : StraightAccesses);
		std::array<vk::Pipeline, marks.size()> pipelines {};
		for (size_t mark = 0; mark < marks.size(); ++mark) {
			pipelines[mark] = Pipeline(kernel, CostText(marks[mark].first, loop));
			kernel.extra.push_back(pipelines[mark]);
		}
		if (std::ranges::any_of(pipelines, [](vk::Pipeline pipeline) { return !pipeline; })) {
			Check(false, "the cost pipelines assemble");
			return;
		}
		for (const bool divergent: {false, true}) {
			std::array<double, marks.size()> times {};
			for (size_t mark = 0; mark < marks.size(); ++mark) times[mark] = time(pipelines[mark], divergent);
			std::printf("%s: %s, %s index, %.0f fetches:", g_name, loop ? "loop of 64 fetches" : "8 straight fetches",
			            divergent ? "two-way divergent" : "wave-uniform", accesses);
			for (size_t mark = 0; mark < marks.size(); ++mark) {
				std::printf(" %s %.3f ms (%+.4f ns per access);", marks[mark].second, times[mark] / 1e6,
				            (times[mark] - times[0]) / accesses);
			}
			std::printf("\n");
			Check(std::ranges::all_of(times, [](double value) { return value > 0.0; }), "the timestamps are read");
		}
	}
	scheduler.Finish();
	renderer.UnmapMemory(address, bytes);
}

}

extern "C" int RunKytyImageHeapReadsTest(int argc, char** argv) {
	const bool cost    = argc > 1 && std::strcmp(argv[1], "cost") == 0;
	const bool crowded = argc > 1 && std::strcmp(argv[1], "crowded") == 0;
	const bool scale   = argc > 1 && std::strcmp(argv[1], "crowded-scale") == 0;
	if (cost) g_name = "kyty_shim_image_heap_reads_cost";
	if (crowded) g_name = "kyty_shim_image_heap_reads_crowded";
	if (scale) g_name = "kyty_shim_image_heap_reads_crowded_scale";
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize(true, true)) {
		std::fprintf(stderr, "%s: %s\n", g_name, unavailable);
		return std::getenv("ANYPS5_REQUIRE_VULKAN") ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) return 1;
	void*              guest = nullptr;
	const uint64_t     bytes = scale ? Common::AlignUp(ScaleBytes, size_t {0x100000u}) : uint64_t {0x100000u};
	if (sceKernelMapFlexibleMemory(&guest, bytes, 3, 0) != 0 || guest == nullptr) {
		Check(false, "map backing for the table and textures");
		return Finish();
	}
	const auto address = reinterpret_cast<uint64_t>(guest);
	if (ClassifyGuestMapping(address, bytes) == GuestMappingSpace::CpuOnly) {
		if (sceKernelMunmap(address, bytes) != 0) {
			Check(false, "release the CPU-only guest allocation");
			return Finish();
		}
		std::fprintf(stderr, "%s: guest allocation at %p lies outside the GPU address ranges\n", g_name, guest);
		return 77;
	}
	if (cost) {
		MeasureCost(gpu.graphics, address, bytes);
	} else if (crowded) {
		CheckCrowded(gpu.graphics, address, bytes);
	} else if (scale) {
		MeasureCrowded(gpu.graphics, address, bytes);
	} else {
		CheckReads(gpu.graphics, address, bytes);
	}
	Check(sceKernelMunmap(address, bytes) == 0, "unmap test backing");
	return Finish();
}

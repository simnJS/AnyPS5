#include "graphics/host_gpu/renderer/pipeline/floatMinMaxProbe.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <bit>
#include <cstring>

namespace Libs::Graphics {

namespace {

constexpr uint32_t kProbeModule[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000040, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0005000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x00060010, 0x00000004, 0x00000011,
    0x00000001, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000069, 0x00030005, 0x00000011, 0x00006e49,
    0x00050006, 0x00000011, 0x00000000, 0x61746164, 0x00000000, 0x00030005, 0x00000013, 0x006e6976,
    0x00030005, 0x0000001c, 0x00000061, 0x00030005, 0x00000023, 0x00000062, 0x00030005, 0x0000002b,
    0x0074754f, 0x00050006, 0x0000002b, 0x00000000, 0x61746164, 0x00000000, 0x00040005, 0x0000002d,
    0x74756f76, 0x00000000, 0x00040047, 0x00000010, 0x00000006, 0x00000004, 0x00030047, 0x00000011,
    0x00000003, 0x00050048, 0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00040047, 0x00000013,
    0x00000021, 0x00000000, 0x00040047, 0x00000013, 0x00000022, 0x00000000, 0x00040047, 0x0000002a,
    0x00000006, 0x00000004, 0x00030047, 0x0000002b, 0x00000003, 0x00050048, 0x0000002b, 0x00000000,
    0x00000023, 0x00000000, 0x00040047, 0x0000002d, 0x00000021, 0x00000001, 0x00040047, 0x0000002d,
    0x00000022, 0x00000000, 0x00040047, 0x0000003f, 0x0000000b, 0x00000019, 0x00020013, 0x00000002,
    0x00030021, 0x00000003, 0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020,
    0x00000007, 0x00000007, 0x00000006, 0x0004002b, 0x00000006, 0x00000009, 0x00000000, 0x0003001d,
    0x00000010, 0x00000006, 0x0003001e, 0x00000011, 0x00000010, 0x00040020, 0x00000012, 0x00000002,
    0x00000011, 0x0004003b, 0x00000012, 0x00000013, 0x00000002, 0x00040015, 0x00000015, 0x00000020,
    0x00000001, 0x00020014, 0x00000018, 0x00030016, 0x0000001a, 0x00000020, 0x00040020, 0x0000001b,
    0x00000007, 0x0000001a, 0x0004002b, 0x00000015, 0x0000001d, 0x00000000, 0x00040020, 0x0000001f,
    0x00000002, 0x00000006, 0x0004002b, 0x00000006, 0x00000025, 0x00000001, 0x0003001d, 0x0000002a,
    0x00000006, 0x0003001e, 0x0000002b, 0x0000002a, 0x00040020, 0x0000002c, 0x00000002, 0x0000002b,
    0x0004003b, 0x0000002c, 0x0000002d, 0x00000002, 0x0004002b, 0x00000006, 0x0000003b, 0x00000002,
    0x00040017, 0x0000003e, 0x00000006, 0x00000003, 0x0006002c, 0x0000003e, 0x0000003f, 0x00000025,
    0x00000025, 0x00000025, 0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x0004003b, 0x0000001b, 0x0000001c,
    0x00000007, 0x0004003b, 0x0000001b, 0x00000023, 0x00000007, 0x0003003e, 0x00000008, 0x00000009,
    0x000200f9, 0x0000000a, 0x000200f8, 0x0000000a, 0x000400f6, 0x0000000c, 0x0000000d, 0x00000000,
    0x000200f9, 0x0000000e, 0x000200f8, 0x0000000e, 0x0004003d, 0x00000006, 0x0000000f, 0x00000008,
    0x00050044, 0x00000006, 0x00000014, 0x00000013, 0x00000000, 0x0004007c, 0x00000015, 0x00000016,
    0x00000014, 0x0004007c, 0x00000006, 0x00000017, 0x00000016, 0x000500b0, 0x00000018, 0x00000019,
    0x0000000f, 0x00000017, 0x000400fa, 0x00000019, 0x0000000b, 0x0000000c, 0x000200f8, 0x0000000b,
    0x0004003d, 0x00000006, 0x0000001e, 0x00000008, 0x00060041, 0x0000001f, 0x00000020, 0x00000013,
    0x0000001d, 0x0000001e, 0x0004003d, 0x00000006, 0x00000021, 0x00000020, 0x0004007c, 0x0000001a,
    0x00000022, 0x00000021, 0x0003003e, 0x0000001c, 0x00000022, 0x0004003d, 0x00000006, 0x00000024,
    0x00000008, 0x00050080, 0x00000006, 0x00000026, 0x00000024, 0x00000025, 0x00060041, 0x0000001f,
    0x00000027, 0x00000013, 0x0000001d, 0x00000026, 0x0004003d, 0x00000006, 0x00000028, 0x00000027,
    0x0004007c, 0x0000001a, 0x00000029, 0x00000028, 0x0003003e, 0x00000023, 0x00000029, 0x0004003d,
    0x00000006, 0x0000002e, 0x00000008, 0x0004003d, 0x0000001a, 0x0000002f, 0x0000001c, 0x0004003d,
    0x0000001a, 0x00000030, 0x00000023, 0x0007000c, 0x0000001a, 0x00000031, 0x00000001, 0x0000004f,
    0x0000002f, 0x00000030, 0x0004007c, 0x00000006, 0x00000032, 0x00000031, 0x00060041, 0x0000001f,
    0x00000033, 0x0000002d, 0x0000001d, 0x0000002e, 0x0003003e, 0x00000033, 0x00000032, 0x0004003d,
    0x00000006, 0x00000034, 0x00000008, 0x00050080, 0x00000006, 0x00000035, 0x00000034, 0x00000025,
    0x0004003d, 0x0000001a, 0x00000036, 0x0000001c, 0x0004003d, 0x0000001a, 0x00000037, 0x00000023,
    0x0007000c, 0x0000001a, 0x00000038, 0x00000001, 0x00000050, 0x00000036, 0x00000037, 0x0004007c,
    0x00000006, 0x00000039, 0x00000038, 0x00060041, 0x0000001f, 0x0000003a, 0x0000002d, 0x0000001d,
    0x00000035, 0x0003003e, 0x0000003a, 0x00000039, 0x000200f9, 0x0000000d, 0x000200f8, 0x0000000d,
    0x0004003d, 0x00000006, 0x0000003c, 0x00000008, 0x00050080, 0x00000006, 0x0000003d, 0x0000003c,
    0x0000003b, 0x0003003e, 0x00000008, 0x0000003d, 0x000200f9, 0x0000000a, 0x000200f8, 0x0000000c,
    0x000100fd, 0x00010038,
};

constexpr uint32_t kProbePairCount = 16;
constexpr std::array<uint32_t, 2 * kProbePairCount> kProbePairs = {{
    0x00000000u, 0x80000000u, 0x80000000u, 0x00000000u, 0x40000000u, 0xC0400000u, 0xC0400000u, 0x40000000u,
    0x7FC00000u, 0x40A00000u, 0x40A00000u, 0x7FC00000u, 0x7FC00000u, 0x7FC00000u, 0x7F800000u, 0xFF800000u,
    0xFF800000u, 0x7F800000u, 0x7FC00000u, 0xFF800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x40A00000u,
    0x7FC00000u, 0x7FC00001u, 0x7F800001u, 0x40A00000u, 0x40A00000u, 0xFF800001u, 0x7FA00000u, 0x7F800001u,
}};

constexpr bool IsNanBits(uint32_t bits) {
	return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0;
}

constexpr void Expected(uint32_t lhs, uint32_t rhs, uint32_t* min_value, uint32_t* max_value) {
	if (IsNanBits(lhs) || IsNanBits(rhs)) {
		const uint32_t source = IsNanBits(rhs) ? lhs : rhs;
		*min_value            = source;
		*max_value            = source;
		return;
	}
	const float lhs_float = std::bit_cast<float>(lhs);
	const float rhs_float = std::bit_cast<float>(rhs);
	if (lhs_float < rhs_float) {
		*min_value = lhs;
		*max_value = rhs;
	} else if (lhs_float > rhs_float) {
		*min_value = rhs;
		*max_value = lhs;
	} else {
		const bool lhs_negative = (lhs & 0x80000000u) != 0;
		*min_value              = lhs_negative ? lhs : rhs;
		*max_value              = lhs_negative ? rhs : lhs;
	}
}

ShaderRecompiler::Spirv::FloatMinMax RunProbe(const GraphicContext& graphics) {
	using ShaderRecompiler::Spirv::FloatMinMax;
	constexpr uint32_t byte_size = sizeof(uint32_t) * kProbePairs.size();

	const auto alignment = graphics.physical_device.getProperties().limits.minStorageBufferOffsetAlignment;
	const auto result_offset = Common::AlignUp<vk::DeviceSize>(byte_size, alignment);
	const vk::BufferCreateInfo buffer_info {.size = result_offset + byte_size, .usage = vk::BufferUsageFlagBits::eStorageBuffer};
	vk::Buffer buffer;
	RequireVulkanSuccess(graphics.device.createBuffer(&buffer_info, nullptr, &buffer), "create min/max probe buffer");
	const auto requirements = graphics.device.getBufferMemoryRequirements(buffer);
	const auto properties = graphics.physical_device.getMemoryProperties();
	constexpr auto flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	uint32_t type = 0;
	while (type < properties.memoryTypeCount &&
	       (!(requirements.memoryTypeBits & (1u << type)) || (properties.memoryTypes[type].propertyFlags & flags) != flags)) {
		++type;
	}
	if (type == properties.memoryTypeCount) {
		graphics.device.destroyBuffer(buffer, nullptr);
		return FloatMinMax::Cascade;
	}
	const vk::MemoryAllocateInfo alloc {.allocationSize = requirements.size, .memoryTypeIndex = type};
	vk::DeviceMemory memory;
	RequireVulkanSuccess(graphics.device.allocateMemory(&alloc, nullptr, &memory), "allocate min/max probe memory");
	RequireVulkanSuccess(graphics.device.bindBufferMemory(buffer, memory, 0), "bind min/max probe buffer");
	void* mapped = nullptr;
	RequireVulkanSuccess(graphics.device.mapMemory(memory, 0, VK_WHOLE_SIZE, {}, &mapped), "map min/max probe memory");
	std::memcpy(mapped, kProbePairs.data(), byte_size);
	const auto module = CompileSPV(kProbeModule, graphics.device);

	std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	}};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.bindingCount       = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings          = bindings.data();
	vk::DescriptorSetLayout layout = nullptr;
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &layout),
	                     "create min/max probe descriptor layout");

	vk::DescriptorPoolSize       pool_size {vk::DescriptorType::eStorageBuffer, 2};
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	vk::DescriptorPool pool = nullptr;
	RequireVulkanSuccess(graphics.device.createDescriptorPool(&pool_info, nullptr, &pool),
	                     "create min/max probe descriptor pool");

	vk::DescriptorSetAllocateInfo set_info {};
	set_info.descriptorPool     = pool;
	set_info.descriptorSetCount = 1;
	set_info.pSetLayouts        = &layout;
	vk::DescriptorSet set       = nullptr;
	RequireVulkanSuccess(graphics.device.allocateDescriptorSets(&set_info, &set), "allocate min/max probe descriptor set");

	std::array<vk::DescriptorBufferInfo, 2> buffer_descriptors = {{
	    {buffer, 0, byte_size},
	    {buffer, result_offset, byte_size},
	}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t binding = 0; binding < writes.size(); ++binding) {
		writes[binding].dstBinding      = binding;
		writes[binding].dstSet          = set;
		writes[binding].descriptorCount = 1;
		writes[binding].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[binding].pBufferInfo     = &buffer_descriptors[binding];
	}
	graphics.device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &layout;
	vk::PipelineLayout pipeline_layout  = nullptr;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout),
	                     "create min/max probe pipeline layout");

	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = pipeline_layout;
	vk::Pipeline pipeline      = nullptr;
	RequireVulkanSuccess(graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline),
	                     "create min/max probe pipeline");

	vk::CommandPoolCreateInfo command_pool_info {};
	command_pool_info.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	command_pool_info.queueFamilyIndex = graphics.queue_family;
	vk::CommandPool command_pool       = nullptr;
	RequireVulkanSuccess(graphics.device.createCommandPool(&command_pool_info, nullptr, &command_pool),
	                     "create min/max probe command pool");

	vk::CommandBufferAllocateInfo command_info {};
	command_info.commandPool        = command_pool;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command       = nullptr;
	RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&command_info, &command),
	                     "allocate min/max probe command buffer");

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin_info), "begin min/max probe commands");
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, 1, &set, 0, nullptr);
	command.dispatch(1, 1, 1);
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1,
	                        &barrier, 0, nullptr, 0, nullptr);
	RequireVulkanSuccess(command.end(), "end min/max probe commands");

	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence = nullptr;
	RequireVulkanSuccess(graphics.device.createFence(&fence_info, nullptr, &fence), "create min/max probe fence");

	vk::SubmitInfo submit_info {};
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &command;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		const auto queue = graphics.device.getQueue(graphics.queue_family, 0);
		RequireVulkanSuccess(queue.submit(1, &submit_info, fence), "submit min/max probe commands");
	}

	if (graphics.device.waitForFences(1, &fence, VK_TRUE, 10000000000ull) != vk::Result::eSuccess) {
		LOGF("[kyty-minmax] min/max probe timed out: compare and select lowering\n");
		return FloatMinMax::Cascade;
	}
	std::array<uint32_t, 2 * kProbePairCount> results {};
	std::memcpy(results.data(), static_cast<uint8_t*>(mapped) + result_offset, byte_size);
	const auto level = FloatMinMaxLevel(kProbePairs.data(), results.data(), kProbePairCount);

	graphics.device.destroyFence(fence, nullptr);
	graphics.device.freeCommandBuffers(command_pool, 1, &command);
	graphics.device.destroyCommandPool(command_pool, nullptr);
	graphics.device.destroyPipeline(pipeline, nullptr);
	graphics.device.destroyPipelineLayout(pipeline_layout, nullptr);
	graphics.device.destroyDescriptorPool(pool, nullptr);
	graphics.device.destroyDescriptorSetLayout(layout, nullptr);
	graphics.device.destroyShaderModule(module, nullptr);
	graphics.device.unmapMemory(memory);
	graphics.device.destroyBuffer(buffer, nullptr);
	graphics.device.freeMemory(memory, nullptr);

	static constexpr const char* names[] = {"compare and select", "NMin/NMax with a NaN-pair select", "NMin/NMax"};
	LOGF("[kyty-minmax] float min/max lowering: %s\n", names[static_cast<int>(level)]);
	return level;
}

}

ShaderRecompiler::Spirv::FloatMinMax FloatMinMaxLevel(const uint32_t* pairs, const uint32_t* results,
                                                      uint32_t pair_count) {
	using ShaderRecompiler::Spirv::FloatMinMax;
	auto level = FloatMinMax::Exact;
	for (uint32_t pair = 0; pair < pair_count; ++pair) {
		const uint32_t lhs       = pairs[2 * pair];
		const uint32_t rhs       = pairs[2 * pair + 1];
		uint32_t       min_value = 0;
		uint32_t       max_value = 0;
		Expected(lhs, rhs, &min_value, &max_value);
		if (results[2 * pair] == min_value && results[2 * pair + 1] == max_value) {
			continue;
		}
		if (!IsNanBits(lhs) || !IsNanBits(rhs)) {
			return FloatMinMax::Cascade;
		}
		level = FloatMinMax::OtherNanSelect;
	}
	return level;
}

ShaderRecompiler::Spirv::FloatMinMax ProbeFloatMinMax(const GraphicContext& graphics) {
	if (!graphics.device || !graphics.physical_device || graphics.queue_family == UINT32_MAX) {
		return ShaderRecompiler::Spirv::FloatMinMax::Cascade;
	}
	std::call_once(graphics.float_min_max_once, [&] { graphics.float_min_max = RunProbe(graphics); });
	return graphics.float_min_max;
}

}

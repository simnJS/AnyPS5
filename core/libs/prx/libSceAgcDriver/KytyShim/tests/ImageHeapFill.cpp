#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "gpu_tiler_shaders/image_heap_fill_spv.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/guestMirror.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeapSlots.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Slots = ImageHeapSlots;

constexpr const char* Name         = "kyty_shim_image_heap_fill";
constexpr uint64_t    GuestBytes   = 0x40000;
constexpr int         CpuReadWrite = 0x03;
constexpr uint32_t    Untouched    = 0xdeadbeefu;
constexpr uint32_t    Frame        = 77u;
constexpr uint32_t    PoolKeys     = 300u;
constexpr uint32_t    Unknown      = 40u;

std::string g_skipped;

class Random {
public:
	explicit Random(uint32_t seed): m_state(seed) {}
	uint32_t Next() {
		m_state = Slots::Mix(m_state + 0x9e3779b9u);
		return m_state;
	}

private:
	uint32_t m_state;
};

Slots::KeyWordArray KeyOf(uint32_t index, uint32_t salt) {
	Slots::KeyWordArray words {};
	for (uint32_t word = 0; word < Slots::KeyWords; ++word) words[word] = Slots::Mix(index * 8u + word + salt);
	return words;
}

const char* CheckDirectory() {
	Slots::Directory                                         directory;
	std::map<std::pair<Slots::KeyWordArray, uint32_t>, uint32_t> reference;
	Random                                                   random(12345u);
	for (uint32_t step = 0; step < 300000u; ++step) {
		const auto key     = KeyOf(random.Next() % 30000u, 7u);
		const auto profile = random.Next() % 3u;
		if (random.Next() % 3u != 0u) {
			const auto value = random.Next();
			if (!directory.Insert(key, profile, value, 0u)) return "an insertion found no bucket in a half empty directory";
			reference[{key, profile}] = value;
		} else {
			const bool erased = directory.Erase(key, profile);
			if (erased != (reference.erase({key, profile}) != 0u)) return "an erase disagrees with the reference map";
		}
	}
	if (directory.Size() != reference.size()) return "the directory size disagrees with the reference map";
	for (const auto& [key, value]: reference) {
		const auto* bucket = directory.Find(key.first, key.second);
		if (bucket == nullptr || bucket->value != value) return "a key is lost after backward shift deletions";
	}
	size_t occupied = 0;
	for (const auto& bucket: directory.Buckets()) occupied += bucket.profile != Slots::EmptyProfile ? 1u : 0u;
	if (occupied != reference.size()) return "occupied buckets disagree with the directory size";
	for (uint32_t index = 0; index < 1000u; ++index) {
		if (directory.Find(KeyOf(index, 99u), 0u) != nullptr) return "an absent key is found";
	}
	uint64_t dirty = 0;
	directory.TakeDirty([&](uint32_t, uint32_t count) { dirty += count; });
	if (dirty != Slots::BucketCount || directory.Dirty()) return "a new directory does not upload every bucket once";
	const auto first = reference.begin()->first;
	(void)directory.Insert(KeyOf(5u, 1234u), 1u, 0u, Slots::NegativeFlag);
	(void)directory.Erase(first.first, first.second);
	reference.erase(reference.begin());
	dirty = 0;
	directory.TakeDirty([&](uint32_t, uint32_t count) { dirty += count; });
	if (dirty == 0u || dirty > 2u * Slots::MaxDistance + 2u) return "a few changes upload too much or nothing";
	if (directory.Negative() != 1u) return "a negative entry is not counted";
	size_t lost = 0;
	directory.Rebuild([](const Slots::Bucket& bucket) { return (bucket.flags & Slots::NegativeFlag) == 0u; },
	                  [&](const Slots::Bucket&) { ++lost; });
	if (lost != 0u || directory.Negative() != 0u || directory.Size() != reference.size()) {
		return "a rebuild that drops negatives keeps them or loses positives";
	}
	for (const auto& [key, value]: reference) {
		const auto* bucket = directory.Find(key.first, key.second);
		if (bucket == nullptr || bucket->value != value) return "a key is lost by a rebuild";
	}
	return nullptr;
}

struct Table {
	uint64_t             offset = 0;
	uint64_t             bytes  = 0;
	std::vector<uint32_t> words;
};

struct Output {
	std::unique_ptr<Buffer> buffer;
	uint32_t                words = 0;
};

std::unique_ptr<Buffer> HostBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                                   uint64_t bytes) {
	return std::make_unique<Buffer>(graphics, scheduler, usage, 0,
	                                vk::BufferUsageFlagBits::eStorageBuffer |
	                                    vk::BufferUsageFlagBits::eShaderDeviceAddress |
	                                    vk::BufferUsageFlagBits::eTransferDst,
	                                bytes);
}

void Finish(CommandScheduler& scheduler) {
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
}

std::vector<uint32_t> Words(Buffer& buffer, uint64_t count) {
	buffer.Invalidate(0, count * sizeof(uint32_t));
	std::vector<uint32_t> words(count);
	std::memcpy(words.data(), buffer.Mapped().data(), count * sizeof(uint32_t));
	return words;
}

const char* CheckFill(GraphicContext& graphics, uint64_t address) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& cache     = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());
	auto* mirror = cache.GetGuestMirror();
	if (mirror == nullptr) return "the cache has no mirror";
	if (!renderer.MapMemory(address, GuestBytes)) {
		return "the high guest mapping was refused";
	}

	Random random(777u);
	std::vector<Slots::KeyWordArray> pool;
	for (uint32_t index = 0; index < PoolKeys + Unknown; ++index) pool.push_back(KeyOf(index, 31u));
	Slots::Directory directory;
	for (uint32_t index = 0; index < PoolKeys; ++index) {
		const bool negative = index % 17u == 0u;
		const auto value    = negative ? 0u : (index + 1u) | (1u + index % 3u) << Slots::KindShift;
		(void)directory.Insert(pool[index], 0u, value, negative ? Slots::NegativeFlag : 0u);
		auto short_key = pool[index];
		std::fill(short_key.begin() + 4, short_key.end(), 0u);
		(void)directory.Insert(short_key, 1u, value, 0u);
	}

	std::array<Table, 4> tables {{{.offset = 0x0, .bytes = 0x20000},
	                              {.offset = 0x20000, .bytes = 0x2000},
	                              {.offset = 0x24000, .bytes = 0x1000},
	                              {.offset = 0x28000, .bytes = 32u * 100u + 12u}}};
	auto* guest = reinterpret_cast<uint32_t*>(address);
	for (auto& table: tables) {
		table.words.resize(table.bytes / 4u);
		for (uint64_t word = 0; word < table.words.size(); word += 8u) {
			const auto& key  = pool[random.Next() % pool.size()];
			const bool  zero = random.Next() % 16u == 0u;
			for (uint32_t at = 0; at < 8u && word + at < table.words.size(); ++at) {
				table.words[word + at] = zero ? 0u : key[at];
			}
		}
		std::memcpy(guest + table.offset / 4u, table.words.data(), table.words.size() * sizeof(uint32_t));
	}
	for (const auto& table: tables) cache.EnsureRangeContents(address + table.offset, table.bytes);

	std::vector<Slots::Job> jobs;
	std::vector<uint32_t>   groups;
	std::array<Output, 4>   outputs;
	const auto              base_job = [&](uint32_t view, uint32_t first, uint32_t step, uint32_t count, uint32_t key_words,
	                          uint32_t profile) {
		const auto& table = tables[view];
		Slots::Job  job {.table       = mirror->DeviceAddress() + mirror->Offset(address + table.offset),
		                 .slots       = 0,
		                 .guest       = address + table.offset,
		                 .table_words = static_cast<uint32_t>(table.bytes / 4u),
		                 .first_word  = first / 4u,
		                 .step_words  = step / 4u,
		                 .count       = count,
		                 .profile     = profile,
		                 .key_words   = key_words,
		                 .view        = 100u + view,
		                 .arm_count   = 2u,
		                 .shape       = 7u + view};
		job.kinds[0] = 1u;
		job.kinds[1] = 2u;
		return job;
	};
	for (uint32_t view = 0; view < outputs.size(); ++view) {
		outputs[view].words  = static_cast<uint32_t>(tables[view].bytes / 4u + 1u);
		outputs[view].buffer = HostBuffer(graphics, scheduler, MemoryUsage::Download, outputs[view].words * sizeof(uint32_t));
		std::fill_n(reinterpret_cast<uint32_t*>(outputs[view].buffer->Mapped().data()), outputs[view].words, Untouched);
		outputs[view].buffer->Flush(0, outputs[view].words * sizeof(uint32_t));
	}
	{
		auto job  = base_job(0, 0, 32, 0x20000 / 32, 8, 0);
		job.begin = 0;
		job.end   = job.count;
		jobs.push_back(job);
	}
	{
		auto job  = base_job(1, 0, 32, 40, 4, 1);
		job.begin = 10;
		job.end   = 60;
		jobs.push_back(job);
	}
	{
		auto job  = base_job(2, 4, 4, (0x1000 - 4 - 4) / 4 + 1, 8, 0);
		job.begin = 0;
		job.end   = job.count;
		jobs.push_back(job);
	}
	{
		auto job  = base_job(3, 0, 32, 101, 8, 0);
		job.begin = 0;
		job.end   = job.count;
		jobs.push_back(job);
	}
	std::vector<Slots::Job> recorded;
	for (auto& job: jobs) {
		job.slots = outputs[job.view - 100u].buffer->BufferDeviceAddress();
		Slots::AddJob(recorded, groups, job);
	}

	const auto bucket_bytes = uint64_t {Slots::BucketCount} * sizeof(Slots::Bucket);
	auto       buckets      = HostBuffer(graphics, scheduler, MemoryUsage::Upload, bucket_bytes);
	std::memcpy(buckets->Mapped().data(), directory.Buckets().data(), bucket_bytes);
	buckets->Flush(0, bucket_bytes);
	auto job_buffer = HostBuffer(graphics, scheduler, MemoryUsage::Upload, recorded.size() * sizeof(Slots::Job));
	std::memcpy(job_buffer->Mapped().data(), recorded.data(), recorded.size() * sizeof(Slots::Job));
	job_buffer->Flush(0, recorded.size() * sizeof(Slots::Job));
	auto group_buffer = HostBuffer(graphics, scheduler, MemoryUsage::Upload, groups.size() * sizeof(uint32_t));
	std::memcpy(group_buffer->Mapped().data(), groups.data(), groups.size() * sizeof(uint32_t));
	group_buffer->Flush(0, groups.size() * sizeof(uint32_t));
	constexpr uint64_t UsedBytes = 65536u * sizeof(uint32_t);
	auto               used      = HostBuffer(graphics, scheduler, MemoryUsage::Download, UsedBytes);
	std::memset(used->Mapped().data(), 0, UsedBytes);
	used->Flush(0, UsedBytes);
	std::array<std::unique_ptr<Buffer>, 2> rings;
	for (auto& ring: rings) {
		ring = HostBuffer(graphics, scheduler, MemoryUsage::Download, Slots::RingBytes);
		std::memset(ring->Mapped().data(), 0, Slots::RingBytes);
		ring->Flush(0, Slots::RingBytes);
	}

	vk::PipelineLayout          layout = nullptr;
	vk::Pipeline                pipeline = nullptr;
	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Slots::Push)};
	const vk::PipelineLayoutCreateInfo layout_info {.pushConstantRangeCount = 1, .pPushConstantRanges = &push_range};
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout_info, nullptr, &layout), "fill layout");
	const auto                              module = CompileSPV(IMAGE_HEAP_FILL_SPV, graphics.device);
	const vk::PipelineShaderStageCreateInfo stage {
	    .stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"};
	const vk::ComputePipelineCreateInfo pipeline_info {.stage = stage, .layout = layout};
	RequireVulkanSuccess(graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline),
	                     "fill pipeline");
	graphics.device.destroyShaderModule(module);

	const auto total = static_cast<uint32_t>(groups.size());
	const auto run   = [&](Buffer& ring, uint32_t capacity, uint32_t split) {
		auto& command = scheduler.Current();
		command.EndRendering();
		const auto        native = command.Handle();
		vk::MemoryBarrier before {.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
		                          .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {}, 1,
		                       &before, 0, nullptr, 0, nullptr);
		Slots::Push push {.jobs          = job_buffer->BufferDeviceAddress(),
		                  .groups        = group_buffer->BufferDeviceAddress(),
		                  .buckets       = buckets->BufferDeviceAddress(),
		                  .ring          = ring.BufferDeviceAddress(),
		                  .used          = used->BufferDeviceAddress(),
		                  .frame         = Frame,
		                  .ring_capacity = capacity};
		native.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
		for (uint32_t offset = 0; offset < total; offset += split) {
			push.group_offset = offset;
			native.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
			native.dispatch(std::min(split, total - offset), 1, 1);
		}
		vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
		                         .dstAccessMask = vk::AccessFlagBits::eHostRead};
		native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1, &after,
		                       0, nullptr, 0, nullptr);
		Finish(scheduler);
	};
	run(*rings[0], Slots::RingCapacity, 7u);

	std::vector<std::tuple<Slots::KeyWordArray, uint32_t, uint32_t, uint32_t>> expected_misses;
	std::vector<uint32_t>                                                       expected_used(65536u, 0u);
	const char*                                                                 failure = nullptr;
	for (uint32_t view = 0; view < outputs.size() && failure == nullptr; ++view) {
		const auto slots = Words(*outputs[view].buffer, outputs[view].words);
		const auto& job  = jobs[view];
		std::vector<uint8_t> covered(outputs[view].words, uint8_t {0});
		for (uint32_t probe = job.begin; probe < job.end; ++probe) {
			const auto slot   = job.first_word + probe * job.step_words;
			const auto lookup = Slots::ExpectedSlot(directory, tables[view].words, job, probe);
			covered[slot]     = 1u;
			if (slots[slot] != lookup.slot) {
				std::fprintf(stderr, "%s: view %u probe %u slot %u holds 0x%08x, 0x%08x expected\n", Name, view, probe,
				             slot, slots[slot], lookup.slot);
				failure = "a slot written by the GPU differs from the CPU reference";
				break;
			}
			if (lookup.miss) {
				expected_misses.emplace_back(
				    Slots::ReadKey(tables[view].words, slot, job.key_words), job.profile, job.view, slot);
			}
			if ((lookup.slot & Slots::IndexMask) != 0u) expected_used[lookup.slot & Slots::IndexMask] = Frame;
		}
		for (uint32_t slot = 0; failure == nullptr && slot < outputs[view].words; ++slot) {
			if (covered[slot] == 0u && slots[slot] != Untouched) failure = "the GPU writes a slot outside its jobs";
		}
	}
	if (failure == nullptr) {
		const auto header = Words(*rings[0], Slots::RingBytes / sizeof(uint32_t));
		if (header[0] != expected_misses.size()) {
			std::fprintf(stderr, "%s: %u misses in the ring, %zu expected\n", Name, header[0], expected_misses.size());
			failure = "the miss ring does not count every miss";
		} else if (expected_misses.empty() || expected_misses.size() > Slots::RingCapacity) {
			failure = "the fixture does not exercise the miss ring";
		} else {
			std::vector<std::tuple<Slots::KeyWordArray, uint32_t, uint32_t, uint32_t>> found;
			for (uint32_t at = 0; at < header[0]; ++at) {
				Slots::Miss miss;
				std::memcpy(&miss, reinterpret_cast<const uint8_t*>(header.data()) + Slots::RingHeaderBytes + at * sizeof(miss),
				            sizeof(miss));
				found.emplace_back(miss.words, miss.profile, miss.view, miss.slot);
				const auto view = miss.view - 100u;
				if (view >= tables.size() || miss.shape != 7u + view ||
				    miss.guest != address + tables[view].offset + uint64_t {miss.slot} * 4u) {
					failure = "a miss does not name the shape and guest address of its key";
				}
			}
			std::ranges::sort(found);
			std::ranges::sort(expected_misses);
			if (failure == nullptr && found != expected_misses) failure = "the miss ring holds other misses than the CPU reference";
		}
	}
	if (failure == nullptr && Words(*used, 65536u) != expected_used) {
		failure = "the use stamps differ from the entries the GPU wrote";
	}
	if (failure == nullptr) {
		run(*rings[1], 5u, Slots::MaxGroups);
		const auto header = Words(*rings[1], Slots::RingBytes / sizeof(uint32_t));
		if (header[0] != expected_misses.size()) failure = "a full ring stops counting misses";
		for (uint32_t word = Slots::RingHeaderBytes / 4u + 5u * sizeof(Slots::Miss) / 4u;
		     failure == nullptr && word < header.size(); ++word) {
			if (header[word] != 0u) failure = "a full ring is written past its capacity";
		}
	}
	graphics.device.destroyPipeline(pipeline);
	graphics.device.destroyPipelineLayout(layout);
	if (failure == nullptr) {
		std::printf("%s: %zu jobs, %u groups, %zu misses checked\n", Name, jobs.size(), total, expected_misses.size());
	}
	Finish(scheduler);
	(void)cache.DropFreedRange(address, GuestBytes);
	return failure;
}

}

extern "C" int RunKytyImageHeapFillTest() {
	if (const auto* failure = CheckDirectory()) {
		std::fprintf(stderr, "%s: FAILED: %s\n", Name, failure);
		return 1;
	}
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	auto&                           graphics = gpu.graphics;
	if (const auto* unavailable = gpu.Initialize(false, true)) {
		std::printf("%s: directory ok, GPU skipped, %s\n", Name, unavailable);
		return 77;
	}
	if (!graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", Name);
		return 1;
	}
	constexpr uint64_t Fixture = 960 * GuestMirror::GiB - 3 * BufferCache::MIRROR_BLOCK_SIZE;
	void* guest = reinterpret_cast<void*>(Fixture);
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0x90) != 0 || reinterpret_cast<uint64_t>(guest) != Fixture) {
		std::fprintf(stderr, "%s: FAILED: could not map guest backing\n", Name);
		return 1;
	}
	const auto  address = reinterpret_cast<uint64_t>(guest);
	const auto* failure = CheckFill(graphics, address);
	if (sceKernelMunmap(address, GuestBytes) != 0) {
		std::fprintf(stderr, "%s: FAILED: could not unmap guest backing\n", Name);
		return 1;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "%s: FAILED: %s\n", Name, failure);
		return 1;
	}
	if (!g_skipped.empty()) {
		std::printf("%s: directory ok, GPU skipped, %s\n", Name, g_skipped.c_str());
		return 77;
	}
	std::printf("%s: ok\n", Name);
	return 0;
}

#include "kernel/memory.h"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/MemoryShim.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t Page = 0x4000;
constexpr int ProtNone = 0x00;
constexpr int CpuReadWrite = 0x03;
constexpr int GpuRead = 0x10;
constexpr int GpuReadWrite = 0x30;
constexpr int MapFixed = 0x10;
constexpr int ProtReadWriteHost = 0x03;

struct HookLog {
	std::vector<std::pair<std::uint64_t, std::uint64_t>> mapped;
	std::vector<std::pair<std::uint64_t, std::uint64_t>> unmapped;
	bool backed_when_unmapped = true;
};

int g_failures = 0;
HookLog g_hooks;
std::atomic<int> g_gpu_jobs {0};

void Check(bool condition, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "kyty_shim_cpu_only_map_hooks: FAILED %s\n", what);
	++g_failures;
}

std::uint64_t Address(const void* pointer) {
	return reinterpret_cast<std::uintptr_t>(pointer);
}

bool Covers(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, std::uint64_t vaddr, std::uint64_t size) {
	return std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) {
		return range.first <= vaddr && vaddr + size <= range.first + range.second;
	});
}

void OnMap(std::uint64_t vaddr, std::uint64_t size) {
	g_hooks.mapped.emplace_back(vaddr, size);
}

void OnUnmap(std::uint64_t vaddr, std::uint64_t size) {
	g_hooks.unmapped.emplace_back(vaddr, size);
	std::uint8_t byte = 0;
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, &byte, 1)) g_hooks.backed_when_unmapped = false;
}

class FakeGpuThread {
public:
	FakeGpuThread() : m_thread([this] { Run(); }) {}

	FakeGpuThread(const FakeGpuThread&) = delete;
	FakeGpuThread& operator=(const FakeGpuThread&) = delete;

	~FakeGpuThread() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_all();
		m_thread.join();
	}

	void RunSync(std::function<void()> job) {
		std::unique_lock lock(m_mutex);
		m_job = std::move(job);
		m_done = false;
		m_wake.notify_all();
		m_finished.wait(lock, [this] { return m_done; });
	}

private:
	void Run() {
		std::unique_lock lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [this] { return m_stop || m_job != nullptr; });
			if (m_job == nullptr) return;
			auto job = std::move(m_job);
			m_job = nullptr;
			lock.unlock();
			job();
			lock.lock();
			m_done = true;
			m_finished.notify_all();
		}
	}

	std::mutex m_mutex;
	std::condition_variable m_wake;
	std::condition_variable m_finished;
	std::function<void()> m_job;
	bool m_done = false;
	bool m_stop = false;
	std::thread m_thread;
};

FakeGpuThread* g_gpu = nullptr;

void OnGpuWaitHook(std::uint64_t vaddr, std::uint64_t size) {
	if (g_gpu == nullptr) return;
	g_gpu->RunSync([vaddr, size] {
		(void)Libs::LibKernel::Memory::IsFullyMapped(vaddr, size);
		g_gpu_jobs.fetch_add(1, std::memory_order_relaxed);
	});
}

void CheckCpuOnlyFlexibleAndDirectSkipped() {
	namespace Shim = Libs::LibKernel::Memory::Shim;
	g_hooks = HookLog {};
	Shim::SetGpuRangeHooks(&OnMap, &OnUnmap);

	void* flex = nullptr;
	Check(sceKernelMapFlexibleMemory(&flex, Page * 2, CpuReadWrite, 0) == 0 && flex != nullptr, "map flexible cpu");
	Check(g_hooks.mapped.empty(), "fresh cpu-only flexible map triggered gpu map hook");
	Check(sceKernelMunmap(Address(flex), Page * 2) == 0, "unmap flexible cpu");
	Check(g_hooks.unmapped.empty(), "cpu-only flexible unmap triggered gpu unmap hook");

	const auto directSize = static_cast<std::int64_t>(sceKernelGetDirectMemorySize());
	std::int64_t phys = -1;
	Check(sceKernelAllocateDirectMemory(0, directSize, Page * 4, 0, 0, &phys) == 0 && phys >= 0, "alloc direct");
	void* reserved = nullptr;
	Check(sceKernelReserveVirtualRange(&reserved, Page * 4, 0, 0) == 0 && reserved != nullptr, "reserve range");
	for (int round = 0; round < 16; ++round) {
		void* target = reserved;
		Check(sceKernelMapDirectMemory(&target, Page * 4, CpuReadWrite, MapFixed, phys, 0) == 0 && target == reserved,
		      "map direct cpu into reservation");
		void* reReserved = reserved;
		Check(sceKernelReserveVirtualRange(&reReserved, Page * 4, MapFixed, 0) == 0 && reReserved == reserved,
		      "reserve over cpu direct mapping");
	}
	Check(g_hooks.mapped.empty(), "repeated cpu-only direct mappings triggered gpu map hook");
	Check(g_hooks.unmapped.empty(), "repeated cpu-only reserve-over-mapping triggered gpu unmap hook");
	Check(sceKernelMunmap(Address(reserved), Page * 4) == 0, "unmap reservation");
	Check(sceKernelReleaseDirectMemory(phys, Page * 4) == 0, "release direct");
	Shim::SetGpuRangeHooks(nullptr, nullptr);
}

void CheckCpuThenGpuProtectAndStrip() {
	namespace Shim = Libs::LibKernel::Memory::Shim;
	g_hooks = HookLog {};
	Shim::SetGpuRangeHooks(&OnMap, &OnUnmap);

	const auto directSize = static_cast<std::int64_t>(sceKernelGetDirectMemorySize());
	std::int64_t phys = -1;
	Check(sceKernelAllocateDirectMemory(0, directSize, Page * 4, 0, 0, &phys) == 0 && phys >= 0, "alloc direct promote");
	void* mapped = nullptr;
	Check(sceKernelMapDirectMemory(&mapped, Page * 4, CpuReadWrite, 0, phys, 0) == 0 && mapped != nullptr,
	      "map direct cpu-only");
	Check(g_hooks.mapped.empty(), "cpu-only direct mapping was not skipped");

	Check(sceKernelMprotect(mapped, Page * 4, CpuReadWrite | GpuRead) == 0, "mprotect cpu->gpu");
	Check(Covers(g_hooks.mapped, Address(mapped), Page * 4), "mprotect to gpu-readable did not deliver pending map");
	Check(g_hooks.mapped.size() == 1, "mprotect delivered wrong map count");

	Check(sceKernelMprotect(mapped, Page * 4, CpuReadWrite) == 0, "mprotect strip gpu bits");
	Check(g_hooks.unmapped.empty(), "stripping gpu bits unexpectedly emitted unmap");

	Check(sceKernelMprotect(mapped, Page * 4, CpuReadWrite | GpuReadWrite) == 0, "mprotect re-add gpu bits");
	Check(g_hooks.mapped.size() == 1, "re-adding gpu bits emitted duplicate map");

	Check(sceKernelMunmap(Address(mapped), Page * 4) == 0, "unmap promoted mapping");
	Check(Covers(g_hooks.unmapped, Address(mapped), Page * 4), "unmap after stripping/re-adding gpu bits was missed");
	Check(g_hooks.backed_when_unmapped, "unmap hook saw dropped backing");
	Check(sceKernelReleaseDirectMemory(phys, Page * 4) == 0, "release direct promote");

	void* stripped = nullptr;
	g_hooks = HookLog {};
	Check(sceKernelMapFlexibleMemory(&stripped, Page * 2, CpuReadWrite | GpuReadWrite, 0) == 0 && stripped != nullptr,
	      "map flexible gpu");
	Check(Covers(g_hooks.mapped, Address(stripped), Page * 2), "gpu flexible map not delivered");
	Check(sceKernelMprotect(stripped, Page * 2, CpuReadWrite) == 0, "strip flexible gpu bits");
	Check(sceKernelMunmap(Address(stripped), Page * 2) == 0, "unmap stripped flexible");
	Check(Covers(g_hooks.unmapped, Address(stripped), Page * 2),
	      "unmap of mapping whose gpu bits were removed was missed");

	Shim::SetGpuRangeHooks(nullptr, nullptr);
}

void CheckFixedReplacementAndSubrange() {
	namespace Shim = Libs::LibKernel::Memory::Shim;
	g_hooks = HookLog {};
	Shim::SetGpuRangeHooks(&OnMap, &OnUnmap);

	const auto directSize = static_cast<std::int64_t>(sceKernelGetDirectMemorySize());
	std::int64_t physA = -1;
	std::int64_t physB = -1;
	Check(sceKernelAllocateDirectMemory(0, directSize, Page * 4, 0, 0, &physA) == 0 && physA >= 0, "alloc physA");
	Check(sceKernelAllocateDirectMemory(0, directSize, Page * 4, 0, 0, &physB) == 0 && physB >= 0, "alloc physB");

	void* addr = nullptr;
	Check(sceKernelMapDirectMemory(&addr, Page * 4, CpuReadWrite | GpuReadWrite, 0, physA, 0) == 0 && addr != nullptr,
	      "map gpu physA");
	Check(Covers(g_hooks.mapped, Address(addr), Page * 4), "physA map hook missing");

	g_hooks = HookLog {};
	void* replaced = addr;
	Check(sceKernelMapDirectMemory(&replaced, Page * 4, CpuReadWrite, MapFixed, physB, 0) == 0 && replaced == addr,
	      "replace gpu with cpu-only physB");
	Check(Covers(g_hooks.unmapped, Address(addr), Page * 4), "replacing gpu mapping missed unmap hook");
	Check(g_hooks.mapped.empty(), "replacing gpu mapping with cpu-only emitted false map hook");

	g_hooks = HookLog {};
	void* promoted = addr;
	Check(sceKernelMapDirectMemory(&promoted, Page * 4, CpuReadWrite | GpuReadWrite, MapFixed, physA, 0) == 0 &&
	          promoted == addr,
	      "replace cpu-only with gpu physA");
	Check(g_hooks.unmapped.empty(), "replacing cpu-only mapping emitted false unmap hook");
	Check(Covers(g_hooks.mapped, Address(addr), Page * 4), "replacing cpu-only with gpu missed map hook");

	Check(sceKernelMunmap(Address(addr), Page * 4) == 0, "unmap replaced");
	Check(Covers(g_hooks.unmapped, Address(addr), Page * 4), "unmap replaced gpu missed unmap hook");

	g_hooks = HookLog {};
	void* sub = nullptr;
	Check(sceKernelMapFlexibleMemory(&sub, Page * 4, CpuReadWrite, 0) == 0 && sub != nullptr, "map subrange cpu");
	Check(g_hooks.mapped.empty(), "subrange cpu map emitted hook");
	Check(sceKernelMprotect(static_cast<std::uint8_t*>(sub) + Page, Page * 2, CpuReadWrite | GpuReadWrite) == 0,
	      "promote middle subrange");
	Check(g_hooks.mapped.size() == 1 && g_hooks.mapped[0].first == Address(sub) + Page &&
	          g_hooks.mapped[0].second == Page * 2,
	      "middle subrange promotion reported wrong range");
	Check(sceKernelMunmap(Address(sub), Page) == 0, "unmap cpu-only prefix");
	Check(g_hooks.unmapped.empty(), "unmapping cpu-only prefix emitted unmap hook");
	Check(sceKernelMunmap(Address(sub) + Page, Page * 3) == 0, "unmap promoted+tail");
	Check(Covers(g_hooks.unmapped, Address(sub) + Page, Page * 2), "unmap promoted+tail missed unmap hook");

	Check(sceKernelReleaseDirectMemory(physA, Page * 4) == 0, "release physA");
	Check(sceKernelReleaseDirectMemory(physB, Page * 4) == 0, "release physB");
	Shim::SetGpuRangeHooks(nullptr, nullptr);
}

void CheckPreInstallAndProtNoneCommitAndHostView() {
	namespace Shim = Libs::LibKernel::Memory::Shim;
	void* preCpu = nullptr;
	void* preGpu = nullptr;
	Check(sceKernelMapFlexibleMemory(&preCpu, Page * 2, CpuReadWrite, 0) == 0 && preCpu != nullptr, "pre cpu map");
	Check(sceKernelMapFlexibleMemory(&preGpu, Page * 2, CpuReadWrite | GpuReadWrite, 0) == 0 && preGpu != nullptr,
	      "pre gpu map");

	g_hooks = HookLog {};
	Shim::SetGpuRangeHooks(&OnMap, &OnUnmap);
	Check(!Covers(g_hooks.mapped, Address(preCpu), Page * 2), "pre-install cpu map was replayed");
	Check(Covers(g_hooks.mapped, Address(preGpu), Page * 2), "pre-install gpu map was not replayed");
	Check(sceKernelMprotect(preCpu, Page * 2, CpuReadWrite | GpuReadWrite) == 0, "promote pre-install cpu map");
	Check(Covers(g_hooks.mapped, Address(preCpu), Page * 2), "promoted pre-install cpu map was not delivered");

	g_hooks = HookLog {};
	void* protNone = nullptr;
	Check(sceKernelMapFlexibleMemory(&protNone, Page * 2, ProtNone, 0) == 0 && protNone != nullptr, "map prot none");
	Check(g_hooks.mapped.empty(), "prot none map emitted hook");
	Check(sceKernelMprotect(protNone, Page * 2, CpuReadWrite) == 0, "mprotect prot none -> cpu");
	Check(g_hooks.mapped.empty(), "mprotect prot none -> cpu emitted hook");
	Check(sceKernelMprotect(protNone, Page * 2, CpuReadWrite | GpuReadWrite) == 0, "mprotect cpu -> gpu after reserve");
	Check(Covers(g_hooks.mapped, Address(protNone), Page * 2), "mprotect to gpu after prot none missed hook");

	constexpr std::uint64_t hostAddress = 0xfe0000000ull + GuestArena::GuestAddressOffset;
	void* hostView = DirectMemoryMapHostView_nid_no_patch(hostAddress, Page * 2, ProtReadWriteHost);
	Check(Address(hostView) == hostAddress, "map host view");
	Check(Covers(g_hooks.mapped, hostAddress, Page * 2), "untracked host view map was not delivered");

	Shim::SetGpuRangeHooks(nullptr, nullptr);
	Check(sceKernelMunmap(Address(protNone), Page * 2) == 0, "unmap protNone");
	Check(sceKernelMunmap(Address(preGpu), Page * 2) == 0, "unmap preGpu");
	Check(sceKernelMunmap(Address(preCpu), Page * 2) == 0, "unmap preCpu");
}

void CheckGpuThreadRoundTripsAvoided() {
	namespace Shim = Libs::LibKernel::Memory::Shim;
	FakeGpuThread gpu;
	g_gpu = &gpu;
	g_gpu_jobs.store(0, std::memory_order_relaxed);
	Shim::SetGpuRangeHooks(&OnGpuWaitHook, &OnGpuWaitHook);

	const auto directSize = static_cast<std::int64_t>(sceKernelGetDirectMemorySize());
	std::int64_t phys = -1;
	Check(sceKernelAllocateDirectMemory(0, directSize, Page * 2, 0, 0, &phys) == 0 && phys >= 0, "alloc direct roundtrip");
	void* reserved = nullptr;
	Check(sceKernelReserveVirtualRange(&reserved, Page * 2, 0, 0) == 0 && reserved != nullptr, "reserve roundtrip");

	const int beforeCpu = g_gpu_jobs.load(std::memory_order_relaxed);
	for (int i = 0; i < 64; ++i) {
		void* target = reserved;
		Check(sceKernelMapDirectMemory(&target, Page * 2, CpuReadWrite, MapFixed, phys, 0) == 0 && target == reserved,
		      "cpu direct map roundtrip");
		void* reReserved = reserved;
		Check(sceKernelReserveVirtualRange(&reReserved, Page * 2, MapFixed, 0) == 0 && reReserved == reserved,
		      "cpu reserve over direct roundtrip");
	}
	Check(g_gpu_jobs.load(std::memory_order_relaxed) == beforeCpu, "cpu-only loop woke gpu thread");

	void* gpuMap = nullptr;
	Check(sceKernelMapDirectMemory(&gpuMap, Page * 2, CpuReadWrite | GpuReadWrite, 0, phys, 0) == 0 && gpuMap != nullptr,
	      "gpu direct map roundtrip");
	Check(g_gpu_jobs.load(std::memory_order_relaxed) == beforeCpu + 1, "gpu map did not sync once");
	Check(sceKernelMunmap(Address(gpuMap), Page * 2) == 0, "gpu direct unmap roundtrip");
	Check(g_gpu_jobs.load(std::memory_order_relaxed) == beforeCpu + 2, "gpu unmap did not sync once");

	Check(sceKernelMunmap(Address(reserved), Page * 2) == 0, "unmap roundtrip reservation");
	Check(sceKernelReleaseDirectMemory(phys, Page * 2) == 0, "release direct roundtrip");
	Shim::SetGpuRangeHooks(nullptr, nullptr);
	g_gpu = nullptr;
}

}

extern "C" int RunKytyCpuOnlyMapHooksTest() {
	Check(Libs::LibKernel::Memory::Shim::ViewEnabled(), "second view is not enabled");
	CheckCpuOnlyFlexibleAndDirectSkipped();
	CheckCpuThenGpuProtectAndStrip();
	CheckFixedReplacementAndSubrange();
	CheckPreInstallAndProtNoneCommitAndHostView();
	CheckGpuThreadRoundTripsAvoided();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_cpu_only_map_hooks: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_cpu_only_map_hooks: ok\n");
	return 0;
}

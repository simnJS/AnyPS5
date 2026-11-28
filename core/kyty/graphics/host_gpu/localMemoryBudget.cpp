#include "graphics/host_gpu/localMemoryBudget.h"

#include <algorithm>

#if defined(__linux__)
#include <SDL3/SDL_loadso.h>
#include <fstream>
#endif

namespace Libs::Graphics {

#if defined(__linux__)
namespace {

struct NvmlMemory {
	unsigned int version;
	unsigned long long total, reserved, free, used;
};

class Nvml {
public:
	using Device = struct nvmlDevice_st*;

	Nvml() {
		const auto init = reinterpret_cast<int (*)()>(SDL_LoadFunction(m_library, "nvmlInit_v2"));
		m_ready = init && m_shutdown && m_lookup && m_memory && init() == 0;
	}
	~Nvml() {
		if (m_ready) m_shutdown();
		SDL_UnloadObject(m_library);
	}
	Device Find(const char* pci_id) const {
		Device device = nullptr;
		return m_ready && m_lookup(pci_id, &device) == 0 ? device : nullptr;
	}
	std::optional<BoardMemory> Read(Device device) const {
		NvmlMemory memory {.version = sizeof(NvmlMemory) | (2u << 24u)};
		if (m_memory(device, &memory) != 0 || !memory.total || memory.used > memory.total) return {};
		return BoardMemory {memory.total, memory.used};
	}

private:
	SDL_SharedObject* m_library = SDL_LoadObject("libnvidia-ml.so.1");
	int (*m_shutdown)() = reinterpret_cast<decltype(m_shutdown)>(SDL_LoadFunction(m_library, "nvmlShutdown"));
	int (*m_lookup)(const char*, Device*) =
	    reinterpret_cast<decltype(m_lookup)>(SDL_LoadFunction(m_library, "nvmlDeviceGetHandleByPciBusId_v2"));
	int (*m_memory)(Device, NvmlMemory*) =
	    reinterpret_cast<decltype(m_memory)>(SDL_LoadFunction(m_library, "nvmlDeviceGetMemoryInfo_v2"));
	bool m_ready = false;
};

std::function<std::optional<BoardMemory>()> BoardReader(vk::PhysicalDevice device, uint32_t vendor) {
	const auto extensions = device.enumerateDeviceExtensionProperties(nullptr);
	if (extensions.result != vk::Result::eSuccess || !std::ranges::any_of(extensions.value, [](const auto& ext) {
		    return std::string_view(ext.extensionName.data()) == VK_EXT_PCI_BUS_INFO_EXTENSION_NAME;
	    })) return {};
	vk::PhysicalDevicePCIBusInfoPropertiesEXT pci;
	vk::PhysicalDeviceProperties2 properties {.pNext = &pci};
	device.getProperties2(&properties);
	const auto id = fmt::format("{:04x}:{:02x}:{:02x}.{:x}", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
	if (vendor == 0x10de) {
		static const Nvml nvml;
		if (auto handle = nvml.Find(id.c_str())) return [handle] { return nvml.Read(handle); };
	}
	if (vendor == 0x1002) {
		return [path = "/sys/bus/pci/devices/" + id]() -> std::optional<BoardMemory> {
			BoardMemory memory {};
			if (!(std::ifstream(path + "/mem_info_vram_total") >> memory.total) ||
			    !(std::ifstream(path + "/mem_info_vram_used") >> memory.used) ||
			    !memory.total || memory.used > memory.total) return {};
			return memory;
		};
	}
	return {};
}

}
#endif

void LocalMemoryBudget::Initialize(vk::PhysicalDevice device, const vk::PhysicalDeviceProperties& properties, uint64_t cap) {
	m_cap = cap;
	m_discrete = properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
#if defined(__linux__)
	m_read_board = m_discrete ? BoardReader(device, properties.vendorID) : nullptr;
#endif
	m_next_sample = {};
}

uint64_t LocalMemoryBudget::Target(uint64_t budget, uint64_t usage, uint64_t heap_size, std::optional<BoardMemory> board) {
	const uint64_t total = board ? board->total : heap_size;
	const uint64_t reserve = std::min<uint64_t>(1536ull << 20u, total / 8);
	const uint64_t outside = board ? board->used - std::min(usage, board->used) : total - std::min(budget, total);
	const uint64_t available = total - outside;
	return std::min(budget, available - std::min(reserve, available));
}

LocalMemorySample LocalMemoryBudget::Read(vk::PhysicalDevice device, vk::PhysicalDeviceMemoryBudgetPropertiesEXT* heaps) const {
	std::lock_guard lock(m_mutex);
	const auto now = std::chrono::steady_clock::now();
	if (!heaps && now < m_next_sample) return m_sample;
	vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget {};
	vk::PhysicalDeviceMemoryProperties2 properties {.pNext = &budget};
	device.getMemoryProperties2(&properties);
	if (heaps) *heaps = budget;
	LocalMemorySample sample {};
	uint64_t size = 0;
	for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
		const auto& heap = properties.memoryProperties.memoryHeaps[i];
		if (heap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
			sample.usage += budget.heapUsage[i];
			sample.budget += budget.heapBudget[i];
			size += heap.size;
		}
	}
	if (m_read_board) sample.board = m_read_board();
	sample.target = std::min(m_cap, m_discrete ? Target(sample.budget, sample.usage, size, sample.board) : sample.budget);
	m_sample = sample;
	m_next_sample = now + std::chrono::milliseconds(100);
	return sample;
}

}

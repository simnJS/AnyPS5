#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/guestHostView.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

constexpr bool HostContextLayout() {
	return BufferCache::CACHING_PAGEBITS == 14 && BufferCache::CACHING_PAGESIZE == 0x4000 &&
	       BufferCache::MIRROR_BLOCK_BITS == 16 && BufferCache::RT_CONTEXT_OFFSET == 24 &&
	       BufferCache::HOST_TABLE_OFFSET == 64 && BufferCache::HOST_ACCESS_OFFSET == 72 &&
	       BufferCache::RESIDENCY_OFFSET == 256 && BufferCache::RESIDENCY_BLOCKS == 16 &&
	       BufferCache::FAULT_BITMAP_OFFSET == 0x600100 && BufferCache::USE_BITMAP_OFFSET == 0x900100 &&
	       BufferCache::WRITE_BITMAP_OFFSET == 0xc00100 && BufferCache::HACK_MARKS_OFFSET == 0x1800100 &&
	       BufferCache::RT_STATS_OFFSET == 0x1804100;
}

constexpr bool HostMarkLayout() {
	return BufferCache::HACK_MARK_DWORDS == 16 && BufferCache::HACK_MARK_COUNTS == 4 &&
	       BufferCache::HACK_MARK_GUEST_RAYS == 5 && BufferCache::HACK_MARK_ORACLE_RETURNED == 6 &&
	       BufferCache::HACK_MARK_ORACLE_RAYS == 7 && BufferCache::HACK_MARK_ORACLE_CANDIDATES == 8 &&
	       BufferCache::HACK_MARK_ORACLE_ROUNDS == 9 && BufferCache::HACK_MARK_ORACLE_WAVES == 10 &&
	       BufferCache::HackMarkEntry(0x0123456789abcdefull) == 136 &&
	       static_cast<uint32_t>(HackMark::LoopGuard) == 0;
}

constexpr bool GuestAddressLayout() {
	return GuestMirror::WindowBits == 24 && GuestMirror::WindowSize == 0x1000000 && GuestHostView::PageBits == 14 &&
	       GuestHostView::PageSize == 0x4000 && GuestHostView::ChunkBits == 30 &&
	       GuestHostView::EntriesPerChunk == 0x10000 &&
	       LibKernel::Memory::kExtendedMemoryBase == 0x80000000000ull &&
	       LibKernel::Memory::kExtendedMemorySize == 0x8000000000ull && LOWER_ADDRESS_SIZE == 0x10000000000ull;
}

static_assert(HostContextLayout(),
              "the buffer cache context layout that shaders address changed: update these values, which changes the "
              "shader store stamp");
static_assert(HostMarkLayout(),
              "the shader mark layout of the buffer cache changed: update these values, which changes the shader "
              "store stamp");
static_assert(GuestAddressLayout(),
              "the guest address layout that shaders translate changed: update these values, which changes the shader "
              "store stamp");

}
}

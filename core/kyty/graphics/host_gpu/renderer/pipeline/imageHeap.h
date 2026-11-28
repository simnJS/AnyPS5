#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/usageBitmap.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeapSlots.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandBuffer;
class RenderContext;
class RenderExecutor;
struct ImageHeapTestAccess;

class ImageHeap final: public ShaderRecompiler::IR::ImageHeapSource {
public:
	explicit ImageHeap(RenderContext& context);
	~ImageHeap();
	KYTY_CLASS_NO_COPY(ImageHeap);

	[[nodiscard]] static vk::DescriptorSetLayout CreateLayout(vk::Device device);

	void BeginDraw();
	uint64_t FillGroups() const { return m_stats.groups; }
	uint64_t LateMisses() const { return m_stats.late; }
	bool ReadFallback(uint64_t address, std::span<uint32_t> words) const override;
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
	             ShaderRecompiler::IR::ImageHeapView&          result) override;
	bool AcquireList(const ShaderRecompiler::IR::ImageHeapList& list,
	                 ShaderRecompiler::IR::ImageHeapView&       result) override;
	void Rebind(RenderExecutor& executor);
	void Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point, vk::PipelineLayout layout);
	void UnmapMemory(uint64_t address, uint64_t size);
	void RunGarbageCollector();

	struct ReadStats {
		uint64_t readbacks     = 0;
		uint64_t parked        = 0;
		uint64_t parked_bytes  = 0;
		uint64_t unparked      = 0;
		uint64_t crowded       = 0;
		uint64_t crowded_bytes = 0;
		uint64_t live          = 0;
		uint64_t live_bytes    = 0;
		uint64_t cold          = 0;
		uint64_t cold_bytes    = 0;
		uint32_t reported      = 0;
		uint32_t reported_ms   = 0;
	};
	[[nodiscard]] const ReadStats& Reads() const noexcept { return m_read_stats; }

private:
	using DescriptorValue = ShaderRecompiler::IR::DescriptorValue;

	struct Key {
		DescriptorValue value;
		uint32_t        profile = 0;

		bool operator==(const Key&) const = default;
	};

	struct KeyHash {
		size_t operator()(const Key& key) const noexcept;
	};

	struct ListKey {
		const void* owner    = nullptr;
		uint32_t    resource = 0;
		bool        operator==(const ListKey&) const = default;
	};
	struct ListKeyHash {
		size_t operator()(const ListKey& key) const noexcept {
			return std::hash<const void*> {}(key.owner) ^ (size_t {key.resource} * 0x9e3779b97f4a7c15ull);
		}
	};

	struct ShapeKey {
		uint64_t step    = 0;
		uint64_t first   = 0;
		uint32_t profile = 0;
		bool     operator==(const ShapeKey&) const = default;
	};
	struct ShapeKeyHash {
		size_t operator()(const ShapeKey& key) const noexcept {
			return (key.step * 0x9e3779b97f4a7c15ull) ^ (key.first * 0xc2b2ae3d27d4eb4full) ^ key.profile;
		}
	};

	struct SlotDemand {
		uint64_t insertions = 0;
		size_t   required   = 0;
	};
	enum class State : uint8_t { Live, Stale, Buried };
	enum class Refusal : uint8_t { None, Capacity, Arms };
	enum class Sweep : uint8_t { Idle, Snapshot, Removed, Confirm };

	struct List;

	struct Use {
		List*    list = nullptr;
		uint32_t slot = 0;
	};

	struct Element {
		std::vector<uint32_t> table_profiles;
		Key      key;
		ImageId  image;
		uint64_t serial   = 0;
		uint32_t kind     = 0;
		uint32_t refs     = 0;
		uint32_t born     = 0;
		uint32_t since    = 0;
		State    state    = State::Live;
		bool     table    = false;
		bool     hashed   = false;
		bool     freed    = false;
		bool     evicting = false;
		bool     parked   = false;
	};

	struct Readback {
		std::unique_ptr<Buffer> buffer;
		uint64_t                tick  = 0;
		uint32_t                stamp = 0;
	};

	struct Shape {
		uint64_t                     step    = 0;
		uint64_t                     first   = 0;
		uint32_t                     profile = 0;
		uint32_t                     id      = 0;
		std::vector<uint32_t>        kind_keys;
		std::vector<DescriptorValue> kinds;
	};

	struct Range {
		uint32_t begin = 0;
		uint32_t end   = 0;
	};

	struct Cold {
		uint32_t last  = 0;
		uint32_t index = 0;
		uint64_t bytes = 0;
	};

	struct Releasing {
		uint64_t tick  = 0;
		uint32_t frame = 0;
		uint64_t bytes = 0;
	};

	static constexpr size_t RecentReads = 4;

	struct Memory;

	struct View {
		Memory&                 memory;
		Shape&                  shape;
		uint64_t                limit   = 0;
		uint64_t                count   = 0;
		uint32_t                arms    = 0;
		uint32_t                written = 0;
		uint32_t                filled  = 0;
		uint32_t                checked = 0;
		uint32_t                overflow_epoch = 0;
		std::unique_ptr<Buffer> buffer;
		std::vector<Range>      pending;
		std::vector<Range>      found;
		SlotDemand              demand;
		bool                    full         = true;
		bool                    clear        = true;
		bool                    discover     = true;
		bool                    discover_all = true;
		bool                    overflowed   = false;
		bool                    capacity     = false;
	};

	struct Memory {
		uint64_t                                   base     = 0;
		uint64_t                                   size     = 0;
		uint64_t                                   draw     = 0;
		uint64_t                                   synced   = 0;
		uint64_t                                   prepared = 0;
		uint64_t                                   fallback_draw = 0;
		uint32_t                                   idle     = 0;
		bool                                       gpu_all  = false;
		bool                                       cpu_all  = false;
		bool                                       fresh    = true;
		std::vector<uint32_t>                      words;
		RangeSet                                   gpu_bytes;
		std::vector<std::pair<uint64_t, uint64_t>> unread;
		std::vector<std::pair<uint64_t, uint64_t>> written;
		std::vector<std::pair<uint64_t, uint64_t>> gpu_written;
		std::vector<std::unique_ptr<View>>         views;
	};

	struct List {
		Shape*                  shape = nullptr;
		std::vector<uint32_t>   slots;
		std::vector<uint32_t>   uses;
		std::unique_ptr<Buffer> buffer;
		size_t                  dirty_begin = 0;
		size_t                  dirty_end   = 0;
		size_t                  order       = 0;
	};

	struct Ring {
		std::unique_ptr<Buffer> buffer;
		uint64_t                tick = 0;
	};

	struct Check {
		std::unique_ptr<Buffer> buffer;
		uint64_t                tick  = 0;
		uint32_t                first = 0;
		uint32_t                step  = 0;
		std::vector<uint32_t>   expected;
	};

	struct Stats {
		uint64_t reads           = 0;
		uint64_t read_bytes      = 0;
		uint64_t changed_bytes   = 0;
		uint64_t unchanged       = 0;
		uint64_t unstable        = 0;
		uint64_t discoveries     = 0;
		uint64_t probes          = 0;
		uint64_t fills           = 0;
		uint64_t groups          = 0;
		uint64_t late            = 0;
		uint64_t null_misses     = 0;
		uint64_t ring_overflows  = 0;
		uint64_t overflows       = 0;
		uint64_t fallback_reads  = 0;
		uint64_t stale_fallbacks = 0;
		uint32_t reported        = 0;
	};

	struct CheckStats {
		uint64_t views      = 0;
		uint64_t slots      = 0;
		uint64_t mismatches = 0;
		uint64_t late       = 0;
		uint64_t extras     = 0;
		uint64_t misses     = 0;
		uint32_t reported   = 0;
	};

	uint32_t Profile(const ShaderRecompiler::IR::ImageResource& root, bool table);
	Shape&   FindShape(uint64_t step, uint64_t first, uint32_t profile);
	Shape&   ListShape(uint32_t profile);
	void     Discover(View& view);
	Refusal  AdmitKey(Shape& shape, const ImageHeapSlots::KeyWordArray& words, uint32_t& kind);
	void     Refuse(View& view, Refusal refusal);
	void     AdmitLate(const ImageHeapSlots::Miss& miss);
	void     ApplyGpuWrite(uint64_t begin, uint64_t end);
	void     ApplyBytes(Memory& memory, uint64_t address, std::span<const uint8_t> bytes, bool uploaded);
	void     ReadStable(Memory& memory, uint64_t begin, uint64_t end);
	bool     Unchanged(const Memory& memory, uint64_t begin, uint64_t end) const;
	void     Upload(uint64_t begin, uint64_t end);
	void     UseFallback(Memory& memory);
	void     Report();
	void     AppendJobs(View& view);
	void     AdvanceSweep();
	uint32_t InternList(Shape& shape, uint32_t kind, const Key& key);
	uint32_t Entry(const Key& key, uint32_t kind, bool table);
	bool     Arm(Shape& shape, uint32_t kind, const DescriptorValue& value, uint32_t limit, uint32_t& arm);
	void     Hash(uint32_t index);
	void     Settle(uint32_t index);
	void     Free(uint32_t index);
	void     Forget(uint32_t index);
	void     Invalidate(uint32_t profile);
	void     Assign(List& list, size_t slot, uint32_t entry);
	void     Link(List& list, size_t slot, uint32_t entry);
	void     Unlink(List& list, size_t slot, uint32_t entry);
	void     Upload(CommandBuffer& buffer, const std::vector<uint32_t>& slots, Buffer& device,
	                size_t& dirty_begin, size_t& dirty_end);
	void     Release(uint32_t entry);
	void     Retire();
	void     Mark(uint32_t index, State state);
	template <typename Fn>
	bool     ForEachChanged(Fn&& fn);
	void     Repoint();
	void     Requeue();
	void     TakeChanges();
	ImageId  CurrentImage(const Image::HeapChange& change);
	uint32_t Allocate(bool may_fail);
	size_t   FreeSlots(bool table) const;
	size_t   ReclaimSlots(bool table);
	void     ReclaimCompleted(uint64_t tick);
	bool     CouldFit(const SlotDemand& demand, size_t available) const;
	bool     FitsFreeSlots(View& view);
	void     Resize(Memory& memory, uint64_t base, uint64_t size);
	void     WriteDescriptors();
	void     Drop(Memory& memory);
	void     WatchTables();
	void     TakeGpuWrites();
	void     ServiceMisses();
	void     Service(Ring& ring);
	void     SyncMemory(Memory& memory);
	void     UpdateProbes(Memory& memory);
	void     AddBytes(View& view, uint64_t begin, uint64_t end, bool discover);
	void     Repend(const Key& key, uint32_t profile);
	void     RecordFills(CommandBuffer& buffer);
	void     PrepareGpu();
	void     FlushDirectory(CommandBuffer& buffer);
	void     Dispatch(CommandBuffer& buffer);
	Ring&    CurrentRing();
	void     Snapshot(CommandBuffer& buffer, std::unique_ptr<Buffer>& target);
	bool     StepSweep();
	void     PurgeDirectory();
	void     RecordChecks(CommandBuffer& buffer);
	void     FinishChecks();
	void     CollectAt(std::chrono::steady_clock::time_point now);
	uint32_t ReadStamp(std::chrono::steady_clock::time_point now) const;
	void     TakeReads();
	void     RecordReads(uint32_t stamp);
	void     ParkUnread(uint32_t stamp);
	void     ParkLeastRead(uint64_t wanted);
	void     FlushRepends();
	[[nodiscard]] uint32_t CrowdedCutoff() const;
	const Image* Parkable(const Element& element, uint32_t index, uint32_t cutoff) const;
	void     ReportReads(uint32_t stamp);

	RenderContext&                                                      m_context;
	std::mutex                                                          m_mutex;
	std::vector<std::unique_ptr<Memory>>                                m_memories;
	std::unordered_map<ShapeKey, std::unique_ptr<Shape>, ShapeKeyHash> m_shapes;
	std::vector<Shape*>                                                 m_shape_ids;
	std::unordered_map<uint32_t, std::unique_ptr<Shape>>               m_list_shapes;
	std::unordered_map<ListKey, List, ListKeyHash>                      m_lists;
	std::vector<View*>                                                  m_active;
	std::vector<List*>                                                  m_active_lists;
	uint64_t                                                            m_draw         = 1;
	uint64_t                                                            m_admission_draw = 0;
	uint64_t                                                            m_suspend_draw = 0;
	uint32_t                                                            m_frame        = 1;
	uint32_t                                                            m_capacity_epoch = 0;
	bool                                                                m_capacity_changed = false;
	std::vector<ShaderRecompiler::IR::ImageResource>                    m_profiles;
	std::vector<bool>                                                   m_table_profiles;
	std::vector<uint32_t>                                               m_key_profiles;
	std::vector<Element>                                                m_elements;
	std::vector<std::vector<Use>>                                       m_uses;
	std::vector<uint32_t>                                               m_marked;
	std::vector<Use>                                                    m_visits;
	std::unordered_map<Key, uint32_t, KeyHash>                          m_index;
	uint64_t                                                            m_insertions   = 0;
	uint64_t                                                            m_reclaim_tick = 0;
	std::vector<ImageHeapSlots::KeyWordArray>                           m_absent;
	std::vector<uint32_t>                                               m_free;
	std::vector<uint32_t>                                               m_pending;
	std::vector<uint32_t>                                               m_released;
	std::vector<uint32_t>                                               m_held;
	std::vector<uint32_t>                                               m_declined;
	std::vector<uint32_t>                                               m_dcc;
	std::vector<uint32_t>                                               m_requeue;
	std::vector<std::pair<uint64_t, ImageId>>                           m_away;
	uint64_t                                                            m_held_serial = 0;
	std::deque<std::pair<uint64_t, uint32_t>>                           m_retired;
	std::vector<std::pair<uint64_t, uint64_t>>                          m_runs;
	std::vector<std::pair<uint64_t, uint64_t>>                          m_watched;
	std::vector<uint32_t>                                               m_scratch;
	std::vector<Image::HeapChange>                                      m_changes;
	std::vector<std::pair<uint32_t, vk::DescriptorImageInfo>>           m_writes;
	ImageHeapSlots::Directory                                           m_directory;
	std::vector<std::pair<ImageHeapSlots::KeyWordArray, uint32_t>>      m_dead;
	std::vector<ImageHeapSlots::Job>                                    m_jobs;
	std::vector<uint32_t>                                               m_groups;
	std::vector<View*>                                                  m_cleared;
	std::vector<Range>                                                  m_ranges;
	RangeSet                                                            m_unread;
	std::vector<uint8_t>                                                m_bytes;
	std::deque<Ring>                                                    m_rings;
	std::vector<std::unique_ptr<Buffer>>                                m_free_rings;
	std::unique_ptr<Buffer>                                             m_buckets;
	std::unique_ptr<Buffer>                                             m_used;
	vk::PipelineLayout                                                  m_fill_layout   = nullptr;
	vk::Pipeline                                                        m_fill_pipeline = nullptr;
	Sweep                                                               m_sweep         = Sweep::Idle;
	uint32_t                                                            m_sweep_frame   = 0;
	uint32_t                                                            m_sweep_at      = 0;
	uint64_t                                                            m_sweep_tick    = 0;
	std::unique_ptr<Buffer>                                             m_sweep_first;
	std::unique_ptr<Buffer>                                             m_sweep_second;
	std::vector<uint32_t>                                               m_candidates;
	std::vector<uint32_t>                                               m_stamps;
	bool                                                                m_check = false;
	bool                                                                m_have_fallback = false;
	std::deque<Check>                                                   m_checks;
	CheckStats                                                          m_check_stats;
	Stats                                                               m_stats;
	vk::DescriptorSetLayout                                             m_layout = nullptr;
	vk::DescriptorPool                                                  m_pool   = nullptr;
	vk::DescriptorSet                                                   m_set    = nullptr;
	std::chrono::steady_clock::time_point                               m_epoch;
	std::unique_ptr<Buffer>                                             m_reads;
	std::deque<Readback>                                                m_readbacks;
	std::vector<std::unique_ptr<Buffer>>                                m_free_readbacks;
	UsageBitmap                                                         m_read_stamps;
	std::unordered_set<Key, KeyHash>                                    m_parking;
	std::vector<uint32_t>                                               m_unparked;
	std::vector<Cold>                                                   m_cold;
	std::deque<Releasing>                                               m_releasing;
	std::vector<std::pair<uint32_t, ImageHeapSlots::KeyWordArray>>      m_repends;
	std::array<uint32_t, RecentReads>                                   m_recent_reads {};
	size_t                                                              m_recent_count = 0;
	uint32_t                                                            m_read_seen   = 0;
	uint32_t                                                            m_park_check  = 0;
	bool                                                                m_pressure    = false;
	bool                                                                m_crowded     = false;
	bool                                                                m_batch_repends = false;
	ReadStats                                                           m_read_stats;

	friend struct ImageHeapTestAccess;
	friend struct ImageHeapReadsTestAccess;
};

}

#endif

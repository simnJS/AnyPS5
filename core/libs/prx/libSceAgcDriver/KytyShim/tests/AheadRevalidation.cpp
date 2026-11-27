#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/recompiler/ir/passes/LoopGuard.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace {

using Libs::Graphics::AheadRefusal;
using Libs::Graphics::GuestReadLog;
using Libs::Graphics::StaleRead;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		++g_failures;
	}
}

void EmptyLog() {
	GuestReadLog log;
	log.Clear();
	Check(!log.Failed() && log.Refusal() == AheadRefusal::None, "a cleared log has no refusal");
	Check(log.StillValid() && log.Stale() == StaleRead::None, "a log without reads stays valid");
}

void UnbackedRead() {
	GuestReadLog log;
	log.Clear();
	uint32_t value = 0;
	Check(!log.Read(0x1000, &value, sizeof(value), false), "a read outside the guest backing is refused");
	Check(log.Failed() && log.Refusal() == AheadRefusal::Read, "the refusal names the read");
	log.Fail(AheadRefusal::Stage);
	Check(log.Refusal() == AheadRefusal::Read, "a later refusal keeps the first cause");
	Check(!log.StillValid() && log.Stale() == StaleRead::Failed, "revalidation refuses a failed resolution as failed");
	log.Clear();
	Check(!log.Failed() && log.Refusal() == AheadRefusal::None, "clearing forgets the refusal");
	Check(log.Stale() == StaleRead::None, "clearing forgets the stale read");
	Check(log.StillValid() && log.Stale() == StaleRead::None, "a cleared log is valid again");
}

void EmptyRead() {
	GuestReadLog log;
	log.Clear();
	uint32_t value = 0;
	Check(!log.Read(0x200000000ull, &value, 0, false), "an empty read is refused");
	Check(log.Refusal() == AheadRefusal::Read, "an empty read is a read refusal");
}

void EachCause() {
	for (auto cause: {AheadRefusal::Stage, AheadRefusal::Compiling, AheadRefusal::Shape, AheadRefusal::Calls,
	                  AheadRefusal::ImageHeap, AheadRefusal::Materialize, AheadRefusal::Variant,
	                  AheadRefusal::GpuVariant}) {
		GuestReadLog log;
		log.Clear();
		log.Fail(cause);
		Check(log.Failed() && log.Refusal() == cause, "each refusal is kept as given");
		Check(!log.StillValid() && log.Stale() == StaleRead::Failed, "each refused resolution fails revalidation");
	}
}

struct alignas(64) GuestWords {
	std::array<uint32_t, 16> words {};
	[[nodiscard]] uint64_t Address(size_t index) const { return reinterpret_cast<uint64_t>(&words[index]); }
};

void RecordAhead(GuestReadLog& log, const GuestWords& guest, std::initializer_list<size_t> indices) {
	std::vector<uint64_t> loads;
	for (const auto index: indices) {
		loads.push_back(guest.Address(index));
		loads.push_back(uint64_t {IR::NoFlatSlot} << 32u | guest.words[index]);
	}
	log.RecordLoads(loads);
}

void AroundReadThenGpuWrite() {
	static GuestWords guest;
	guest.words = {0x11u, 0x22u, 0x33u, 0x44u};
	GuestReadLog log;
	log.Clear();
	RecordAhead(log, guest, {0, 1, 2});
	std::vector<uint64_t> around {guest.Address(1)};
	log.NoteAroundLoads(around);
	Check(log.ReadAroundGpuWrites(), "a log with a word read around GPU writes says so");
	Check(log.StillValid() && log.Stale() == StaleRead::None, "words read around GPU writes still hold when nothing wrote them");
	guest.words[1] = 0x99u;
	Check(!log.StillValid(), "a GPU write between Resolve and Execute on a word read around it fails revalidation");
	Check(log.Stale() == StaleRead::AroundGpuWrite, "the refusal names the GPU write since the read around it");
	GuestReadLog again;
	again.Clear();
	RecordAhead(again, guest, {0, 1, 2});
	Check(again.StillValid() && again.Stale() == StaleRead::None, "the resolution redone by Execute holds");
	Check(!again.ReadAroundGpuWrites(), "the resolution redone by Execute read nothing around GPU writes");
	log.Clear();
	Check(!log.ReadAroundGpuWrites(), "clearing forgets the reads around GPU writes");
}

void OtherWordWrite() {
	static GuestWords guest;
	guest.words = {0x11u, 0x22u, 0x33u, 0x44u, 0x55u};
	GuestReadLog log;
	log.Clear();
	RecordAhead(log, guest, {0, 1});
	RecordAhead(log, guest, {4});
	std::vector<uint64_t> around {guest.Address(4)};
	log.NoteAroundLoads(around);
	guest.words[0] = 0x77u;
	Check(!log.StillValid() && log.Stale() == StaleRead::CpuWrite,
	      "a write to a word Resolve read directly stays a CPU write");
}

void AroundAddressOutsideReads() {
	static GuestWords guest;
	guest.words = {0x11u, 0x22u, 0x33u, 0x44u};
	GuestReadLog log;
	log.Clear();
	RecordAhead(log, guest, {0});
	std::vector<uint64_t> around {guest.Address(3)};
	log.NoteAroundLoads(around);
	Check(!log.ReadAroundGpuWrites(), "an address read around that the log did not record marks nothing");
	std::vector<uint64_t> none;
	log.NoteAroundLoads(none);
	Check(!log.ReadAroundGpuWrites(), "no address read around marks nothing");
}

using Wave = std::array<uint32_t, 64>;

constexpr uint32_t WalkLimit      = 16;
constexpr uint32_t RecordedLeaves = 512;
constexpr uint32_t Terminator     = 0x80000000u;
constexpr uint32_t Waves          = 4;

struct WalkResult {
	uint32_t marks = 0;
	Wave     cursor {};
};

WalkResult WalkLeaves(const Wave& start, uint32_t records, std::span<const uint32_t> entries) {
	using namespace Libs::Graphics::ShaderRecompiler;
	using namespace IR;
	using Op = ValueOpcode;
	Program program;
	for (uint32_t id = 0; id < 5; ++id) {
		auto& block = *program.block_storage.emplace_back(std::make_unique<Block>());
		block.id    = id;
		program.blocks.push_back(&block);
	}
	auto& entry  = *program.blocks[0];
	auto& header = *program.blocks[1];
	auto& test   = *program.blocks[2];
	auto& latch  = *program.blocks[3];
	auto& exit   = *program.blocks[4];
	const auto link = [](Block& from, Block& to) {
		from.terminator.kind       = CFG::TerminatorKind::Branch;
		from.terminator.true_block = &to;
		from.AddBranch(&to);
	};
	link(entry, header);
	link(header, test);
	link(test, latch);
	link(latch, header);
	header.terminator.loop_header    = true;
	header.terminator.merge_block    = &exit;
	header.terminator.continue_block = &latch;
	test.terminator.kind             = CFG::TerminatorKind::ConditionalBranch;
	test.terminator.false_block      = &exit;
	test.AddBranch(&exit);
	exit.terminator.kind = CFG::TerminatorKind::Return;
	const Value first(&entry.AppendNewInst(Op::GetUserData, {Value(static_cast<ScalarReg>(0))}));
	const Value bound(&entry.AppendNewInst(Op::GetUserData, {Value(static_cast<ScalarReg>(1))}));
	auto&       cursor  = header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U32));
	auto&       payload = header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U32));
	cursor.AddPhiOperand(&entry, first);
	payload.AddPhiOperand(&entry, Value(0u));
	const Value end(&test.AppendNewInst(Op::BitwiseAnd32, {Value(&payload), Value(Terminator)}));
	const Value live(&test.AppendNewInst(Op::IEqual32, {end, Value(0u)}));
	test.condition = Value(&test.AppendNewInst(Op::ConditionRef, {live},
	                                           static_cast<uint64_t>(CFG::BranchCondition::ExecNonZero)));
	auto& memory = program.memory_info.emplace_back();
	memory.kind  = ResourceKind::IndirectBuffer;
	memory.idxen = true;
	const Value resource(&latch.AppendNewInst(Op::GetBufferResource, {Value(0u), Value(0u), bound, Value(1u << 12u)}));
	const Value loaded(&latch.AppendNewInst(Op::LoadBufferU32, {resource, Value(&cursor), Value(0u), Value(0u), live}));
	payload.AddPhiOperand(&latch, Value(&latch.AppendNewInst(Op::SelectU32, {live, loaded, Value(&payload)})));
	const Value next(&latch.AppendNewInst(Op::IAdd32, {Value(&cursor), Value(1u)}));
	cursor.AddPhiOperand(&latch, Value(&latch.AppendNewInst(Op::SelectU32, {live, next, Value(&cursor)})));
	exit.AppendNewInst(Op::ReferenceU32, {Value(&cursor)});
	const auto guards = GuardComputeLoops(program, WalkLimit, false, {});
	Check(guards.guarded == 1, "the leaf walk is guarded");

	WalkResult                  result;
	std::map<const Inst*, Wave> values;
	values[first.Instruction()] = start;
	values[bound.Instruction()].fill(records);
	const auto read = [&](Value value, size_t lane) -> uint32_t {
		if (const auto* inst = value.TryInstruction()) return values.at(inst)[lane];
		return value.GetType() == Type::U1 ? value.U1() : value.U32();
	};
	const auto run = [&](const Block& block, const Block* from) {
		std::map<const Inst*, Wave> phis;
		for (auto& inst: block) {
			if (inst.GetOpcode() != Op::Phi) break;
			const auto source = inst.Arg(inst.PhiBlock(0) == from ? 0 : 1);
			for (size_t lane = 0; lane < 64; ++lane) phis[&inst][lane] = read(source, lane);
		}
		for (const auto& [inst, wave]: phis) values[inst] = wave;
		for (auto& inst: block) {
			const auto op = inst.GetOpcode();
			if (op == Op::Phi || op == Op::GetBufferResource || op == Op::ReferenceU32) continue;
			auto& output = values[&inst];
			if (op == Op::ConditionRef) {
				const auto& predicate = values.at(inst.Arg(0).Instruction());
				output.fill(std::ranges::any_of(predicate, [](uint32_t v) { return v != 0; }));
				continue;
			}
			for (size_t lane = 0; lane < 64; ++lane) {
				const auto arg = [&](size_t n) { return read(inst.Arg(n), lane); };
				switch (op) {
					case Op::LoadBufferU32: {
						const auto index = arg(1);
						const auto limit = read(inst.Arg(0).Instruction()->Arg(2), lane);
						output[lane] = arg(4) != 0 && index < limit && index < entries.size() ? entries[index] : 0u;
						break;
					}
					case Op::SelectU32: output[lane] = arg(0) != 0 ? arg(1) : arg(2); break;
					case Op::IAdd32: output[lane] = arg(0) + arg(1); break;
					case Op::ISub32: output[lane] = arg(0) - arg(1); break;
					case Op::BitwiseAnd32: output[lane] = arg(0) & arg(1); break;
					case Op::IEqual32: output[lane] = arg(0) == arg(1); break;
					case Op::INotEqual32: output[lane] = arg(0) != arg(1); break;
					case Op::ULessThan32: output[lane] = arg(0) < arg(1); break;
					case Op::ULessThanEqual32: output[lane] = arg(0) <= arg(1); break;
					case Op::UGreaterThanEqual32: output[lane] = arg(0) >= arg(1); break;
					case Op::LogicalNot: output[lane] = arg(0) == 0; break;
					case Op::LogicalAnd: output[lane] = arg(0) != 0 && arg(1) != 0; break;
					case Op::LogicalOr: output[lane] = arg(0) != 0 || arg(1) != 0; break;
					case Op::LogicalXor: output[lane] = (arg(0) != 0) != (arg(1) != 0); break;
					case Op::LoopGuardMark: result.marks += arg(0) != 0; break;
					default: Check(false, "the leaf walk uses only the operations of the interpreter"); return;
				}
			}
		}
	};
	const Block* from = &entry;
	for (uint32_t turn = 0; turn <= WalkLimit + 1; ++turn) {
		run(header, from);
		run(test, &header);
		if (read(test.condition, 0) == 0) break;
		run(latch, &test);
		from = &latch;
	}
	result.cursor = values.at(&cursor);
	return result;
}

uint32_t WalkMarks(uint32_t records, std::span<const uint32_t> entries) {
	uint32_t marks = 0;
	for (uint32_t wave = 0; wave < Waves; ++wave) {
		Wave start {};
		for (uint32_t lane = 0; lane < 64; ++lane) start[lane] = RecordedLeaves - 1u - (wave * 64u + lane) % 4u;
		const auto result = WalkLeaves(start, records, entries);
		marks += result.marks;
		if (result.marks == 0) {
			Check(std::ranges::all_of(result.cursor, [](uint32_t cursor) { return cursor == RecordedLeaves + 1u; }),
			      "every lane leaves the walk after the terminating entry");
		}
	}
	return marks;
}

void SelfSizedHeaderWords() {
	struct alignas(64) Blob {
		std::array<uint32_t, 64> header {};
	};
	static Blob                                        blob;
	static std::array<uint32_t, RecordedLeaves + 2u> entries {};
	static GuestWords                                  other;
	std::fill(entries.begin(), entries.end(), 1u);
	entries[RecordedLeaves] = Terminator;
	blob.header[0x60 / 4]  = RecordedLeaves;
	other.words[0]          = 0x55u;
	const auto base         = reinterpret_cast<uint64_t>(&blob);
	GuestReadLog log;
	log.Clear();
	log.RecordLoad(base + 0x60, RecordedLeaves, 0);
	log.RecordLoad(other.Address(0), 0x55u, 1);
	IR::CompiledShaderInfo program;
	program.user_data_base = 0;
	program.info.self_sized_buffers.push_back({.base_low_register = 0, .base_high_register = 1, .written = false});
	std::array<IR::ResourceSnapshot, 1> resources;
	resources[0].user_data     = {static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32u)};
	resources[0].flattened_srt = {0u, 0u};
	log.RefreshFlatLoads(resources, true, &program);
	const auto& copies = resources[0].gpu_flat_loads;
	Check(std::ranges::any_of(copies, [&](const auto& load) { return load.address == base + 0x60 && load.slot == 0; }),
	      "a size word in the header of a buffer sized by its own memory is copied on the GPU when the dispatch runs");
	Check(std::ranges::none_of(copies, [&](const auto& load) { return load.address == other.Address(0); }),
	      "a word outside such a header stays a CPU read");
	Check(resources[0].flattened_srt[1] == 0x55u, "the CPU read keeps the value of the word");
	Check(WalkMarks(RecordedLeaves, entries) == Waves * 64u,
	      "the count read when the dispatch was recorded leaves 256 lanes at the loop guard");
	blob.header[0x60 / 4] = RecordedLeaves + 1u;
	for (const auto& load: copies) {
		std::memcpy(&resources[0].flattened_srt[load.slot], reinterpret_cast<const void*>(load.address),
		            load.count * sizeof(uint32_t));
	}
	Check(resources[0].flattened_srt[0] == RecordedLeaves + 1u, "the GPU copy reads the count the earlier pass wrote");
	Check(WalkMarks(resources[0].flattened_srt[0], entries) == 0u,
	      "with the count of the dispatch time no lane reaches the loop guard");
}

}

extern "C" int RunKytyAheadRevalidationTest() {
	EmptyLog();
	UnbackedRead();
	EmptyRead();
	EachCause();
	AroundReadThenGpuWrite();
	OtherWordWrite();
	AroundAddressOutsideReads();
	SelfSizedHeaderWords();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_ahead_revalidation: %d failures\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_ahead_revalidation: ok\n");
	return 0;
}

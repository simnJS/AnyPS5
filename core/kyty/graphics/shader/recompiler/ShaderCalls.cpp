#include "graphics/shader/recompiler/ShaderCalls.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

void PrepareCallTargets(ShaderSource& source, const CompileOptions& options) {
	using namespace IR;
	EXIT_NOT_IMPLEMENTED(options.stage != ShaderType::Compute);
	source.wave_size = options.wave_size;
	for (uint32_t i = 0; i < source.decoded.instructions.size(); ++i) {
		if (source.decoded.instructions[i].opcode != Decoder::Opcode::S_SWAPPC_B64) continue;
		source.calls.push_back({.instruction = i});
	}
	const auto graph = CFG::BuildGraph(source.decoded);
	Program query;
	query.stage = options.stage;
	query.user_data_base = options.user_data_base;
	query.user_data_count = static_cast<uint32_t>(options.user_data.size());
	query.wave_size = options.wave_size;
	auto& block = *query.block_storage.emplace_back(std::make_unique<Block>());
	query.blocks.push_back(&block);
	struct Producer { bool ready = false; std::map<uint32_t, Value> values; };
	std::map<uint32_t, Producer> producers;
	bool ambiguous = false;
	std::function<Value(uint32_t, uint32_t)> Register;
	std::function<Value(uint32_t, Value)> Resolve = [&](uint32_t before, Value value) -> Value {
		value = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr) return value;
		uint32_t code = UINT32_MAX;
		switch (inst->GetOpcode()) {
			case ValueOpcode::GetScalarRegister: code = static_cast<uint32_t>(inst->Arg(0).ScalarRegister()); break;
			case ValueOpcode::GetVccLo: code = 106u; break;
			case ValueOpcode::GetVccHi: code = 107u; break;
			default: break;
		}
		if (code != UINT32_MAX) {
			const auto result = Register(before, code);
			inst->ReplaceUsesWith(result);
			return result;
		}
		for (size_t i = 0; i < inst->NumArgs(); ++i) inst->SetArg(i, Resolve(before, inst->Arg(i)));
		return value;
	};
	Register = [&](uint32_t before, uint32_t code) -> Value {
		std::vector<uint32_t> crossed_calls;
		const auto index = CFG::FindScalarDefinition(source.decoded, graph, before, code,
		                                             source.wave_size, crossed_calls);
		if (index == CFG::AmbiguousScalarDefinition) {
			ambiguous = true;
			return Value(uint32_t {0});
		}
		for (const auto crossed: crossed_calls) {
			std::ranges::find(source.calls, crossed, &ShaderSource::Call::instruction)->preserved_registers.set(code);
		}
		if (index == UINT32_MAX) {
			EXIT_IF(code < query.user_data_base || code - query.user_data_base >= query.user_data_count);
			return IREmitter(&block).GetUserData(static_cast<ScalarReg>(code));
		}
		auto [entry, inserted] = producers.try_emplace(index);
		auto& producer = entry->second;
		if (inserted) {
			Block translated;
			Frontend::Translator translator(query, &translated, 1u, Frontend::NoLaneSpills,
			    (options.input_info.compute->float_mode & 0x10u) == 0u,
			    !options.input_info.compute->async_compute);
			translator.TranslateInstruction(source.decoded.instructions[index]);
			for (const auto& inst: translated) {
				uint32_t destination = UINT32_MAX;
				Value value;
				switch (inst.GetOpcode()) {
					case ValueOpcode::SetScalarRegister:
						destination = static_cast<uint32_t>(inst.Arg(0).ScalarRegister()); value = inst.Arg(1); break;
					case ValueOpcode::SetVccLo: destination = 106u; value = inst.Arg(0); break;
					case ValueOpcode::SetVccHi: destination = 107u; value = inst.Arg(0); break;
					default: break;
				}
				if (destination != UINT32_MAX) producer.values.insert_or_assign(destination, value);
			}
			for (auto& [destination, value]: producer.values) value = Resolve(index, value);
			for (auto& inst: translated) inst.SetParent(&block);
			block.Instructions().splice(block.end(), translated.Instructions());
			producer.ready = true;
		}
		EXIT_IF(!producer.ready || !producer.values.contains(code));
		return producer.values.at(code).Resolve();
	};
	for (auto& site: source.calls) {
		const auto& call = source.decoded.instructions[site.instruction];
		EXIT_NOT_IMPLEMENTED(call.src0.kind != Decoder::OperandKind::Sgpr ||
		                     call.dst.kind != Decoder::OperandKind::Sgpr ||
		                     call.src0.reg != call.dst.reg || call.dst.reg >= 105u);
		for (uint32_t i = 0; i < 2u; ++i) {
			const auto value = Register(site.instruction, call.src0.reg + i);
			if (ambiguous) {
				source.unsupported_call_pc = call.pc;
				source.expand_calls        = true;
				source.wave_size           = options.wave_size;
				source.calls.clear();
				return;
			}
			if (!ValidateRuntimeValue(query, value, RuntimeValueType::Integer)) {
				EXIT_IF(value.Instruction()->GetOpcode() != ValueOpcode::ReadConstBuffer);
				source.unsupported_call_pc = call.pc;
				source.expand_calls        = true;
				source.wave_size           = options.wave_size;
				source.calls.clear();
				return;
			}
			auto& root = query.value_storage.emplace_back(ValueOpcode::ReferenceU32);
			root.SetArg(0, value);
			site.target[i] = Value(&root);
		}
	}
	RewriteToSsa(query.blocks);
	ConstantPropagationPass(query.blocks, query.wave_size);
	RemoveIdentities(query);
	EliminateDeadCode(query.blocks);
	for (auto& site: source.calls)
		for (auto& value: site.target) value = value.Instruction()->Arg(0).Resolve();
	query.value_storage.clear();
	for (auto& inst: block) {
		inst.SetParent(nullptr);
		if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 || inst.GetOpcode() == ValueOpcode::ReadConstBuffer)
			inst.SetFlags(MemoryFlags {.index = inst.Flags<MemoryFlags>().index});
	}
	query.value_storage.splice(query.value_storage.end(), block.Instructions());
	source.call_targets = std::move(static_cast<ResourcePlan&>(query));
	std::vector<Value> targets;
	for (const auto& site: source.calls) targets.insert(targets.end(), site.target.begin(), site.target.end());
	CompileSrtWalker(source.call_targets, targets);
}

namespace {

template <typename Read>
const char* DecodeLeaf(Read&& read, std::vector<Decoder::Instruction>& function) {
	function.clear();
	for (uint32_t index = 0;;) {
		std::array<uint32_t, Decoder::MaxInstructionRawWords> words {};
		if (index > UINT32_MAX / sizeof(uint32_t) - words.size()) return "leaf code is too long";
		uint32_t count = 1;
		Decoder::Instruction inst;
		for (;;) {
			if (!read(index, std::span(words).first(count))) return "cannot read leaf code";
			inst = {};
			Decoder::DecodeInstruction(words, 0u, inst);
			if (inst.word_count <= count) break;
			if (inst.word_count > words.size()) return "leaf instruction is too long";
			count = inst.word_count;
		}
		inst.pc = index * sizeof(uint32_t);
		if (Decoder::IsDirectBranch(inst.opcode)) inst.branch_target += inst.pc;
		index += inst.word_count;
		if (inst.opcode == Decoder::Opcode::S_SWAPPC_B64 || inst.opcode == Decoder::Opcode::S_GETPC_B64 ||
		    inst.opcode == Decoder::Opcode::S_ENDPGM) return "leaf code calls, reads its address or ends the program";
		function.push_back(inst);
		if (inst.opcode == Decoder::Opcode::S_SETPC_B64) return nullptr;
	}
}

template <typename Leaf>
const char* LinkCalls(ShaderSource& source, uint32_t& failed_pc, Leaf&& leaf) {
	ShaderSource::Linked linked {.code = source.code, .decoded = source.decoded};
	const auto& last = source.decoded.instructions.back();
	EXIT_NOT_IMPLEMENTED(last.opcode != Decoder::Opcode::S_ENDPGM);
	uint32_t function_pc = last.pc + last.word_count * sizeof(uint32_t);
	std::vector<Decoder::Instruction> function;
	for (size_t i = 0; i < source.calls.size(); ++i) {
		const auto& site = source.calls[i];
		const auto& call = source.decoded.instructions[site.instruction];
		failed_pc = call.pc;
		if (const char* unread = leaf(i, function)) return unread;
		const uint32_t function_size = function.back().pc + function.back().word_count * sizeof(uint32_t);
		if (function_pc > UINT32_MAX - function_size) return "linked code is too long";
		const auto function_offset = static_cast<uint32_t>(linked.code.size());
		for (auto inst: function) {
			if (CFG::MayWriteScalarRegister(inst, call.dst.reg, source.wave_size) ||
			    CFG::MayWriteScalarRegister(inst, call.dst.reg + 1u, source.wave_size))
				return "leaf code writes its return address";
			for (uint32_t code = 0; code < site.preserved_registers.size(); ++code) {
				if (site.preserved_registers[code] &&
				    CFG::MayWriteScalarRegister(inst, code, source.wave_size))
					return "leaf code clobbers a scalar target dependency";
			}
			linked.code.insert(linked.code.end(), inst.raw, inst.raw + inst.word_count);
			if (inst.opcode == Decoder::Opcode::S_SETPC_B64) {
				if (inst.src0.kind != Decoder::OperandKind::Sgpr || inst.src0.reg != call.dst.reg)
					return "leaf code returns through another register";
				inst.opcode = Decoder::Opcode::S_BRANCH;
				inst.branch_target = call.pc + call.word_count * sizeof(uint32_t);
			} else if (Decoder::IsDirectBranch(inst.opcode)) {
				if (inst.branch_target >= function_size) return "leaf code branches outside itself";
				inst.branch_target += function_pc;
			}
			inst.pc += function_pc;
			linked.decoded.instructions.push_back(std::move(inst));
		}
		linked.functions.push_back({function_offset, function_size / 4u});
		linked.decoded.instructions[site.instruction].branch_target = function_pc;
		function_pc += function_size;
	}
	linked.decoded.code = linked.code;
	linked.observed_function.resize(std::ranges::max_element(linked.functions, {},
	    &ShaderSource::Linked::Function::size)->size);
	source.linked = std::move(linked);
	++source.revision;
	return nullptr;
}

}

const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime) {
	EXIT_IF(source.unsupported_call_pc != UINT32_MAX && !source.expand_calls);
	auto& reads = source.reads;
	reads.clear();
	const auto read_code = [&](uint64_t address, std::span<uint32_t> words) {
		if (runtime.read_specialization_memory == nullptr ||
		    !runtime.read_specialization_memory(runtime.userdata, address, words)) return false;
		reads.emplace_back(address, words.size_bytes());
		return true;
	};
	if (source.expand_calls) {
		const Decoder::ShaderCodeReader reader = read_code;
		if (!source.expander) source.expander = std::make_shared<Decoder::ShaderFunctionExpander>();
		std::vector<uint32_t> expanded;
		source.refresh_failure.clear();
		source.refresh_failed = !source.expander->Expand(source.code, runtime.shader_base, runtime.user_data,
		                                                 reader, expanded, source.refresh_failure,
		                                                 source.wave_size) ||
		                        expanded.empty();
		if (source.refresh_failed) {
			if (source.refresh_failure.empty()) source.refresh_failure = "shader call expansion is empty";
			return source.decoded;
		}
		source.call_words = source.expander->RuntimeWords();
		if (source.linked &&
		    std::ranges::equal(std::span(source.linked->code).subspan(source.code.size()), expanded) &&
		    source.linked->decoded.long_branches == source.expander->LongBranches()) {
			return source.linked->decoded;
		}
		ShaderSource::Linked linked;
		linked.code = source.code;
		linked.code.insert(linked.code.end(), expanded.begin(), expanded.end());
		const auto expanded_code = std::span<const uint32_t>(linked.code).subspan(source.code.size());
		Decoder::DecodeProgram(expanded_code, linked.decoded, source.expander->LongBranches());
		linked.decoded.code = expanded_code;
		linked.decoded.call_bodies = source.expander->Bodies();
		linked.decoded.runtime_literals = source.expander->RuntimeLiterals();
		source.linked = std::move(linked);
		++source.revision;
		return source.linked->decoded;
	}
	std::vector<uint32_t> flat;
	const bool walked = IR::WalkSrt(source.call_targets, runtime, flat, &reads);
	for (auto& site: source.calls) {
		uint32_t low = 0, high = 0;
		if (!walked || !IR::WalkedValue(source.call_targets, site.target[0], low) ||
		    !IR::WalkedValue(source.call_targets, site.target[1], high))
			EXIT("shader call at pc 0x%08x has an unavailable scalar target",
			     source.decoded.instructions[site.instruction].pc);
		const auto address = ((uint64_t {high} << 32u) | low) & ~uint64_t {3};
		EXIT_IF(address == 0u);
		site.address = address;
	}
	const auto query_reads = reads.size();
	if (source.linked) {
		auto& linked = *source.linked;
		bool unchanged = true;
		for (size_t i = 0; i < source.calls.size(); ++i) {
			const auto& function = linked.functions[i];
			const auto expected = std::span(linked.code).subspan(function.offset, function.size);
			const auto preceding = std::span(source.calls).first(i);
			const auto previous = std::ranges::find(preceding, source.calls[i].address, &ShaderSource::Call::address);
			if (previous != preceding.end()) {
				const auto& observed = linked.functions[previous - preceding.begin()];
				if (std::ranges::equal(expected, std::span(linked.code).subspan(observed.offset, observed.size))) continue;
				unchanged = false;
				break;
			}
			linked.observed_function.resize(function.size);
			if (!read_code(source.calls[i].address, linked.observed_function) ||
			    !std::ranges::equal(expected, linked.observed_function)) {
				unchanged = false;
				break;
			}
		}
		if (unchanged) return linked.decoded;
		reads.resize(query_reads);
	}
	std::map<uint64_t, std::vector<Decoder::Instruction>> functions;
	uint32_t failed_pc = 0;
	const char* failure = LinkCalls(source, failed_pc, [&](size_t i, std::vector<Decoder::Instruction>& function) -> const char* {
		const auto address = source.calls[i].address;
		auto [entry, inserted] = functions.try_emplace(address);
		if (inserted) {
			const char* unread = DecodeLeaf([&](uint32_t index, std::span<uint32_t> words) {
				EXIT_IF(address > UINT64_MAX - (uint64_t {index} + Decoder::MaxInstructionRawWords) * sizeof(uint32_t));
				return read_code(address + uint64_t {index} * sizeof(uint32_t), words);
			}, entry->second);
			if (unread != nullptr) return unread;
		}
		function = entry->second;
		return nullptr;
	});
	if (failure != nullptr) EXIT("shader call at pc 0x%08x: %s", failed_pc, failure);
	reads.resize(query_reads);
	for (size_t i = 0; i < source.calls.size(); ++i) {
		const auto preceding = std::span(source.calls).first(i);
		if (std::ranges::find(preceding, source.calls[i].address, &ShaderSource::Call::address) == preceding.end())
			reads.emplace_back(source.calls[i].address, source.linked->functions[i].size * sizeof(uint32_t));
	}
	return source.linked->decoded;
}

bool LinkShaderSource(ShaderSource& source, std::span<const uint32_t> function_code) {
	if (source.calls.empty() || source.expand_calls || source.unsupported_call_pc != UINT32_MAX ||
	    source.decoded.instructions.empty() ||
	    source.decoded.instructions.back().opcode != Decoder::Opcode::S_ENDPGM) return false;
	size_t offset = 0;
	uint32_t failed_pc = 0;
	const auto revision = source.revision;
	const char* failure = LinkCalls(source, failed_pc, [&](size_t, std::vector<Decoder::Instruction>& function) -> const char* {
		const auto leaf = function_code.subspan(offset);
		const char* unread = DecodeLeaf([&](uint32_t index, std::span<uint32_t> words) {
			if (index > leaf.size() || words.size() > leaf.size() - index) return false;
			std::copy_n(leaf.begin() + index, words.size(), words.begin());
			return true;
		}, function);
		if (unread != nullptr) return unread;
		offset += function.back().pc / sizeof(uint32_t) + function.back().word_count;
		return nullptr;
	});
	if (failure != nullptr || offset != function_code.size()) {
		source.linked.reset();
		source.revision = revision;
		return false;
	}
	return true;
}

}

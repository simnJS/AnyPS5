#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#include "graphics/shader/recompiler/ir/RayQueryHeaderProgram.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryOracle.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryResults.h"
#include "graphics/shader/recompiler/ir/passes/RayQuerySource.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include <array>
#include <optional>
#include <bit>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <spirv-tools/libspirv.hpp>
#include <vector>

namespace {

namespace IR = Libs::Graphics::ShaderRecompiler::IR;
namespace CFG = Libs::Graphics::ShaderRecompiler::CFG;
namespace Spirv = Libs::Graphics::ShaderRecompiler::Spirv;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

enum class Node { Children, Uniform, Memory, Constant };
enum class Child { InLane, Ballot, ReadLane, Store };

struct Walk {
    IR::Block block;
    std::vector<IR::Inst*> sites;
};

IR::Inst& Intersection(IR::Block& block, IR::Value node, uint32_t node_words) {
    const IR::Value zero(uint32_t {0});
    const IR::Value one(uint32_t {0x3f800000u});
    auto& descriptor = block.AppendNewInst(IR::ValueOpcode::CompositeConstructU32x4, {zero, zero, zero, zero});
    IR::Value high = node_words == 2 ? IR::Value(uint32_t {0}) : one;
    auto& address = block.AppendNewInst(IR::ValueOpcode::MakeImageAddress,
                                        {node, high, one, zero, zero, zero, one, one, one, one, one, one, one});
    return block.AppendNewInst(IR::ValueOpcode::BvhIntersect, {IR::Value(&descriptor), IR::Value(&address), IR::Value(true)},
                               node_words);
}

void Build(Walk& walk, Node node, Child child, uint32_t node_words, uint32_t sites) {
    auto& block = walk.block;
    auto& phi = block.AppendNewInst(IR::ValueOpcode::Phi);
    phi.AddPhiOperand(&block, IR::Value(uint32_t {5}));
    IR::Value current(&phi);
    if (node == Node::Uniform) current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::ReadFirstLane, {current, IR::Value(true)}));
    if (node == Node::Memory) current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::LoadSharedU32, {current, IR::Value(true)}));
    if (node == Node::Constant) current = IR::Value(uint32_t {5});
    IR::Value last;
    for (uint32_t site = 0; site < sites; ++site) {
        auto& bvh = Intersection(block, current, node_words);
        walk.sites.push_back(&bvh);
        auto& first = block.AppendNewInst(IR::ValueOpcode::CompositeExtractU32x4, {IR::Value(&bvh), IR::Value(uint32_t {0})});
        auto& masked = block.AppendNewInst(IR::ValueOpcode::BitwiseAnd32, {IR::Value(&first), IR::Value(uint32_t {0xfffffff8u})});
        current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::BitwiseOr32, {IR::Value(&masked), IR::Value(uint32_t {5})}));
        last = IR::Value(&first);
    }
    auto& valid = block.AppendNewInst(IR::ValueOpcode::INotEqual32, {last, IR::Value(uint32_t {0xffffffffu})});
    switch (child) {
        case Child::InLane: break;
        case Child::Ballot: block.AppendNewInst(IR::ValueOpcode::Ballot, {IR::Value(&valid)}); break;
        case Child::ReadLane: block.AppendNewInst(IR::ValueOpcode::ReadLane, {last, IR::Value(uint32_t {0})}); break;
        case Child::Store: block.AppendNewInst(IR::ValueOpcode::WriteSharedU32, {IR::Value(uint32_t {0}), last, IR::Value(true)}); break;
    }
    if (node != Node::Constant) phi.AddPhiOperand(&block, current);
}

uint32_t Mark(Node node, Child child, uint32_t node_words = 1, uint32_t sites = 1) {
    Walk walk;
    Build(walk, node, child, node_words, sites);
    const auto flags = IR::MarkOracleWalk(walk.sites);
    for (const auto* bvh: walk.sites) {
        const auto set = bvh->Flags<uint32_t>();
        Expect((set & 0xffu) == node_words, "marking keeps the node word count");
        Expect((set & flags) == flags, "every intersection of the walk carries the walk flags");
    }
    return flags;
}

void EmptyWalks() {
    Walk served, left;
    Build(served, Node::Children, Child::InLane, 1, 1);
    Build(left, Node::Children, Child::InLane, 2, 2);
    IR::MarkOracleWalk(served.sites);
    std::vector<IR::Inst*> all(served.sites);
    all.insert(all.end(), left.sites.begin(), left.sites.end());
    IR::Program program;
    Expect(IR::BvhIntersectEmpty == 1u << 11, "the empty bit is bit 11");
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = false}) == 0, "the emulated variant runs no intersection empty");
    for (const auto* bvh: left.sites) {
        Expect((bvh->Flags<uint32_t>() & IR::BvhIntersectEmpty) == 0, "an emulated intersection keeps its box test");
    }
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = true}) == 2, "the native variant runs every unserved intersection empty");
    for (const auto* bvh: left.sites) {
        const auto set = bvh->Flags<uint32_t>();
        Expect((set & IR::BvhIntersectEmpty) != 0 && (set & IR::BvhIntersectNative) == 0, "an unserved intersection is empty, not native");
        Expect((set & 0xffu) == 2, "an empty intersection keeps its node word count");
    }
    for (const auto* bvh: served.sites) {
        Expect((bvh->Flags<uint32_t>() & IR::BvhIntersectEmpty) == 0, "an intersection the oracle serves is not empty");
    }
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = true}) == 0, "marking again adds nothing");
}

bool Lane(uint32_t flags) {
    return (flags & IR::BvhIntersectOracleLane) != 0;
}

void Marking() {
    const auto flags = Mark(Node::Children, Child::InLane);
    Expect((flags & IR::BvhIntersectNative) != 0 && (flags & IR::BvhIntersectOracle) != 0, "a marked walk is native and served by the oracle");
    Expect(IR::MarkOracleWalk({}) == 0u, "an empty walk marks nothing");
    Expect(IR::OracleRemaining == 1u && IR::OracleForms == 2u && IR::OracleOnly == 4u && IR::OraclePixel == 8u, "the oracle bits match KYTY_RT_ORACLE");
}

void Modes() {
    Expect(Lane(Mark(Node::Children, Child::InLane)), "a node fed only by the walk's own children walks per lane");
    Expect(Lane(Mark(Node::Children, Child::InLane, 2)), "a 64-bit node pointer fed by its children walks per lane");
    Expect(Lane(Mark(Node::Children, Child::InLane, 1, 2)), "two sites that feed each other walk per lane");
    Expect(!Lane(Mark(Node::Uniform, Child::InLane)), "a node read from the first lane walks the whole wave");
    Expect(!Lane(Mark(Node::Uniform, Child::InLane, 2)), "a 64-bit node read from the first lane walks the whole wave");
    Expect(!Lane(Mark(Node::Memory, Child::InLane)), "a node loaded from memory keeps the wave mode");
    Expect(!Lane(Mark(Node::Constant, Child::InLane)), "a constant node keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::Ballot)), "a child tested across lanes keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::ReadLane)), "a child read from another lane keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::Store)), "a child stored to memory keeps the wave mode");
}

void NodeCountFacts() {
    constexpr uint32_t Tlas = 1u << 29u;
    constexpr uint32_t Blas = 1u << 31u;
    IR::RayTracingLayout layout;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == 0, "both node counts at 88 by default");
    layout.node_count_80 = true;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == (Tlas | Blas), "an unset BLAS fact follows the TLAS one");
    layout.blas_node_count_80 = false;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == Tlas, "TLAS count at 80, BLAS count at 88");
    layout.node_count_80      = false;
    layout.blas_node_count_80 = true;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == Blas, "TLAS count at 88, BLAS count at 80");
    auto shaped = layout;
    shaped.blas_node_count_80.reset();
    Expect(shaped.Pack()[0] == layout.Pack()[0] && shaped.Pack()[1] == layout.Pack()[1] &&
               shaped.Pack()[2] == layout.Pack()[2] && ((shaped.Pack()[3] ^ layout.Pack()[3]) & ~Blas) == 0,
           "the BLAS fact changes bit 31 only");
}

void WaveScalarReadKindFacts() {
    IR::Program program;
    program.memory_info.push_back({.kind = IR::ResourceKind::ScalarBuffer, .offset = 0u});
    IR::Block block;
    const auto read = [&](IR::Value offset) -> IR::Inst& {
        return block.AppendNewInst(IR::ValueOpcode::ReadConstBuffer, {IR::Value(0u), offset});
    };
    auto& carried = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(3u), IR::Value(1u)});
    Expect(IR::WaveScalarReadKind(program, read(IR::Value(&carried))) == IR::RayQueryResultKind::Word,
           "a carry inside the register operand selects the second word");
    auto& dynamic = block.AppendNewInst(IR::ValueOpcode::GetUserData, {IR::Value(static_cast<IR::ScalarReg>(0u))});
    auto& unknown = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(&dynamic), IR::Value(1u)});
    Expect(!IR::WaveScalarReadKind(program, read(IR::Value(&unknown))),
           "an unknown register residue cannot identify a wave result word");
    auto& aligned = block.AppendNewInst(IR::ValueOpcode::ShiftLeftLogical32, {IR::Value(&dynamic), IR::Value(3u)});
    auto& first = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(&aligned), IR::Value(1u)});
    auto& second = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(&aligned), IR::Value(4u)});
    Expect(IR::WaveScalarReadKind(program, read(IR::Value(&first))) == IR::RayQueryResultKind::Node,
           "a proven aligned register plus one selects the first word");
    Expect(IR::WaveScalarReadKind(program, read(IR::Value(&second))) == IR::RayQueryResultKind::Word,
           "a proven aligned register plus four selects the second word");
    auto& masked = block.AppendNewInst(IR::ValueOpcode::BitwiseAnd32, {IR::Value(&dynamic), IR::Value(~7u)});
    auto& masked_second = block.AppendNewInst(IR::ValueOpcode::IAdd32, {IR::Value(&masked), IR::Value(4u)});
    Expect(IR::WaveScalarReadKind(program, read(IR::Value(&masked_second))) == IR::RayQueryResultKind::Word,
           "masking the register low bits proves the second word");
    auto& condition = block.AppendNewInst(IR::ValueOpcode::INotEqual32, {IR::Value(&dynamic), IR::Value(0u)});
    auto& selected = block.AppendNewInst(IR::ValueOpcode::SelectU32, {IR::Value(&condition), IR::Value(&aligned), IR::Value(5u)});
    Expect(!IR::WaveScalarReadKind(program, read(IR::Value(&selected))),
           "a selected aligned or unaligned register remains unclassified");
}

void TriangleIndexSpirv() {
    namespace Emit = Spirv::Emitter;
    const IR::Value unused(0u);
    IR::Block alpha;
    alpha.AppendNewInst(IR::ValueOpcode::RayQueryAlpha,
                        {unused, unused, unused, unused, unused, unused, unused,
                         unused, unused, unused, unused, unused, unused, unused});
    IR::Program program;
    program.blocks.push_back(&alpha);
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.lane_local = true;
    program.info.uses_dma = true;
    program.info.ray_query = true;
    const Libs::Graphics::ShaderComputeInputInfo compute {};
    const Spirv::Profile profile {.ray_query = true};
    Emit::EmitterState state(program, {.compute = &compute}, profile);
    Emit::ValueEmitContext ctx(state);
    Emit::DefineModule(state);
    Emit::DefineGetBdaPointer(state);
    Emit::DefineIndirectBufferLoads(ctx);
    auto& builder = state.builder;
    builder.AddFunction(spv::OpFunction, Emit::TypeVoid(state), state.main_func,
                        spv::FunctionControlMaskNone, Emit::TypeFunction(state));
    Emit::EmitLabel(state, state.entry_label);
    const auto zero = Emit::ConstantU32(state, 0);
    const auto indices = Emit::EmitTriangleIndices(ctx, {zero, zero, zero, zero}, zero);
    Expect(indices[0] != 0 && indices[1] != 0 && indices[2] != 0, "the triangle load emits three indices");
    builder.AddFunction(spv::OpReturn);
    builder.AddFunction(spv::OpFunctionEnd);
    Emit::DefineBdaLoads(state);
    builder.AddEntryPoint(spv::ExecutionModelGLCompute, state.main_func, "main", state.interface_variables);

    spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_3);
    validator.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        std::fprintf(stderr, "%s\n", message);
    });
    Expect(validator.Validate(builder.Build()), "the ray-query triangle index load emits valid SPIR-V");
}

enum class HighWord { Exact, StrideBit, SwizzleBit, Masked };

struct SourceWalk {
    IR::Block              entry, header, body, latch, merge, second_header, second_body, second_latch, second_merge;
    IR::Program            program;
    IR::Block*             block = &entry;
    IR::ResourceKind       address_kind = IR::ResourceKind::None;
    std::vector<IR::Inst*> sites;
    std::optional<IR::ResourceKind> user_kind;

    IR::Value Add(IR::ValueOpcode opcode, std::initializer_list<IR::Value> args, uint64_t flags = 0) {
        return IR::Value(&block->AppendNewInst(opcode, args, flags));
    }

    IR::Value Load(IR::ValueOpcode opcode, std::initializer_list<IR::Value> args, uint32_t offset, bool idxen) {
        IR::MemoryInfo memory;
        memory.offset = offset;
        memory.idxen  = idxen;
        program.memory_info.push_back(memory);
        const IR::MemoryFlags flags {static_cast<uint32_t>(program.memory_info.size() - 1), 0};
        return Add(opcode, args, std::bit_cast<uint64_t>(flags));
    }

    IR::Value LoadAddress(IR::Value resource, uint32_t offset, uint32_t displacement = 0) {
        const auto word = Load(IR::ValueOpcode::LoadAddressU32, {resource, IR::Value(displacement), IR::Value(uint32_t {0}), IR::Value(true)}, offset, false);
        program.memory_info.back().kind = address_kind;
        return word;
    }

    std::array<IR::Value, 2> Offset(IR::Value low, IR::Value high, IR::Value node) {
        using Op         = IR::ValueOpcode;
        const auto sum   = Add(Op::IAdd32, {low, node});
        const auto carry = Add(Op::SelectU32, {Add(Op::ULessThan32, {sum, low}), IR::Value(uint32_t {1}), IR::Value(uint32_t {0})});
        return {sum, Add(Op::IAdd32, {high, carry})};
    }

    IR::Value Intersect(std::array<IR::Value, 2> node, const std::array<IR::Value, 6>& ray) {
        const IR::Value zero(uint32_t {0});
        const IR::Value one(uint32_t {0x3f800000u});
        const auto descriptor = Add(IR::ValueOpcode::CompositeConstructU32x4,
                                    {zero, IR::Value(uint32_t {0x80000000u}), IR::Value(uint32_t {0xffffffffu}), IR::Value(uint32_t {0x810003ffu})});
        const auto address = Add(IR::ValueOpcode::MakeImageAddress,
                                 {node[0], node[1], IR::Value(uint32_t {0x47c35000u}), ray[0], ray[1], ray[2], ray[3], ray[4], ray[5], one, one, one, zero});
        auto& bvh = block->AppendNewInst(IR::ValueOpcode::BvhIntersect, {descriptor, address, IR::Value(true)}, 2);
        sites.push_back(&bvh);
        return IR::Value(&bvh);
    }
};

void Link(IR::Block& from, IR::Block& to) {
    from.terminator.kind       = CFG::TerminatorKind::Branch;
    from.terminator.true_block = &to;
    from.AddBranch(&to);
}

struct Loop {
    IR::Block& before;
    IR::Block& header;
    IR::Block& body;
    IR::Block& latch;
    IR::Block& merge;
};

void Number(SourceWalk& walk) {
    walk.program.stage = Libs::Graphics::ShaderType::Compute;
    uint32_t pc        = 0;
    for (auto* block: walk.program.blocks) {
        block->start_pc = pc;
        block->end_pc   = pc += 16;
    }
}

std::array<IR::Value, 2> UserHeader(SourceWalk& walk, uint32_t offset, uint32_t displacement = 0) {
    using Op           = IR::ValueOpcode;
    walk.block         = &walk.entry;
    const auto user    = [&](uint32_t reg) { return walk.Add(Op::GetUserData, {IR::Value(IR::ScalarReg {reg})}); };
    const auto pointer = walk.Add(Op::GetAddressResource, {user(4), user(5)});
    const auto kind    = walk.address_kind;
    walk.address_kind  = walk.user_kind.value_or(kind);
    const std::array<IR::Value, 2> header {walk.LoadAddress(pointer, offset, displacement), walk.LoadAddress(pointer, offset + 4, displacement)};
    walk.address_kind  = kind;
    return header;
}

std::array<IR::Value, 6> UserRay(SourceWalk& walk) {
    walk.block = &walk.entry;
    std::array<IR::Value, 6> ray;
    for (uint32_t k = 0; k < 6; ++k) ray[k] = walk.Add(IR::ValueOpcode::GetUserData, {IR::Value(IR::ScalarReg {8 + k})});
    return ray;
}

void BuildWalk(SourceWalk& walk, const Loop& loop, std::array<IR::Value, 2> header, const std::array<IR::Value, 6>& ray, HighWord kind, bool scalar_blas = false) {
    using Op               = IR::ValueOpcode;
    const auto u32         = [](uint32_t value) { return IR::Value(value); };
    const auto word        = static_cast<uint64_t>(IR::Type::U32);
    const auto header_low  = header[0];
    const auto header_high = header[1];
    walk.block             = &loop.before;
    const auto root        = walk.Add(Op::ShiftRightLogical64, {walk.Add(Op::CompositeConstructU64, {header_low, header_high}), u32(3)});
    const auto root_low    = walk.Add(Op::CompositeExtractU64, {root, u32(0)});
    const auto root_high   = walk.Add(Op::CompositeExtractU64, {root, u32(1)});
    Link(loop.before, loop.header);

    walk.block      = &loop.header;
    auto& base_low  = loop.header.AppendNewInst(Op::Phi, {}, word);
    auto& base_high = loop.header.AppendNewInst(Op::Phi, {}, word);
    auto& node      = loop.header.AppendNewInst(Op::Phi, {}, word);
    std::array<IR::Inst*, 6> rays {};
    std::array<IR::Value, 6> current;
    for (uint32_t k = 0; k < 6; ++k) {
        rays[k]    = &loop.header.AppendNewInst(Op::Phi, {}, word);
        current[k] = IR::Value(rays[k]);
    }
    Link(loop.header, loop.body);
    loop.header.terminator.merge_block    = &loop.merge;
    loop.header.terminator.continue_block = &loop.latch;
    loop.header.terminator.loop_header    = true;

    walk.block       = &loop.body;
    const auto bvh   = walk.Intersect(walk.Offset(IR::Value(&base_low), IR::Value(&base_high), IR::Value(&node)), current);
    const auto child = walk.Add(Op::CompositeExtractU32x4, {bvh, u32(0)});
    loop.body.condition              = walk.Add(Op::IEqual32, {child, u32(0xffffffffu)});
    loop.body.terminator.kind        = CFG::TerminatorKind::ConditionalBranch;
    loop.body.terminator.true_block  = &loop.merge;
    loop.body.terminator.false_block = &loop.latch;
    loop.body.AddBranch(&loop.merge);
    loop.body.AddBranch(&loop.latch);

    walk.block       = &loop.latch;
    const auto table = walk.Add(Op::GetAddressResource, {header_low, header_high});
    const auto low   = walk.Add(Op::IAdd32, {header_low, walk.LoadAddress(table, 0x30)});
    const auto carry = walk.Add(Op::SelectU32, {walk.Add(Op::ULessThan32, {low, header_low}), u32(1), u32(0)});
    auto       high  = walk.Add(Op::IAdd32, {walk.Add(Op::IAdd32, {header_high, walk.LoadAddress(table, 0x34)}), carry});
    if (kind == HighWord::StrideBit) high = walk.Add(Op::IAdd32, {high, u32(0x00010000u)});
    if (kind == HighWord::SwizzleBit) high = walk.Add(Op::BitwiseOr32, {high, u32(0x40000000u)});
    if (kind == HighWord::Masked) high = walk.Add(Op::BitwiseAnd32, {high, u32(0xffffu)});
    const auto buffer = walk.Add(Op::GetBufferResource, {low, walk.Add(Op::BitwiseOr32, {high, u32(0x00800000u)}), u32(0xffffffffu), u32(0x00027000u)});
    const auto index  = walk.Add(Op::ShiftRightLogical32, {child, u32(3)});
    IR::Value scalar_record;
    if (scalar_blas) {
        const auto array = walk.Add(Op::CompositeConstructU64, {low, high});
        const auto element = walk.Add(Op::ShiftLeftLogical64, {walk.Add(Op::CompositeConstructU64, {index, u32(0)}), u32(7)});
        const auto record = walk.Add(Op::IAdd64, {array, element});
        scalar_record = walk.Add(Op::GetAddressResource, {walk.Add(Op::CompositeExtractU64, {record, u32(0)}), walk.Add(Op::CompositeExtractU64, {record, u32(1)})});
    }
    const auto record = [&](uint32_t dword) {
        if (scalar_blas && dword >= 12) return walk.LoadAddress(scalar_record, dword * 4u);
        return walk.Load(Op::LoadBufferU32, {buffer, index, u32(0), u32(0), IR::Value(true)}, dword * 4u, true);
    };
    const auto number = [&](IR::Value value) { return walk.Add(Op::BitCastF32U32, {value}); };
    std::array<IR::Value, 6> local;
    for (uint32_t row = 0; row < 3; ++row) {
        std::array<IR::Value, 4> m;
        for (uint32_t c = 0; c < 4; ++c) m[c] = number(record(row * 4 + c));
        const auto origin =
            walk.Add(Op::FPMad32, {m[0], number(ray[0]), walk.Add(Op::FPMad32, {m[1], number(ray[1]), walk.Add(Op::FPMad32, {m[2], number(ray[2]), m[3]})})});
        const auto direction =
            walk.Add(Op::FPMad32, {m[0], number(ray[3]), walk.Add(Op::FPMad32, {m[1], number(ray[4]), walk.Add(Op::FPMul32, {m[2], number(ray[5])})})});
        local[row]     = walk.Add(Op::BitCastU32F32, {origin});
        local[row + 3] = walk.Add(Op::BitCastU32F32, {direction});
    }
    const auto leaf      = walk.Add(Op::IEqual32, {walk.Add(Op::BitwiseAnd32, {child, u32(7)}), u32(0)});
    const auto pick      = [&](IR::Value blas, IR::Value keep) { return walk.Add(Op::SelectU32, {leaf, blas, keep}); };
    const auto next_low  = pick(record(12), IR::Value(&base_low));
    const auto next_high = pick(walk.Add(Op::BitwiseAnd32, {record(13), u32(0x3fffffu)}), IR::Value(&base_high));
    const auto next_node = pick(record(14), child);
    Link(loop.latch, loop.header);
    base_low.AddPhiOperand(&loop.before, root_low);
    base_low.AddPhiOperand(&loop.latch, next_low);
    base_high.AddPhiOperand(&loop.before, root_high);
    base_high.AddPhiOperand(&loop.latch, next_high);
    node.AddPhiOperand(&loop.before, u32(5));
    node.AddPhiOperand(&loop.latch, next_node);
    for (uint32_t k = 0; k < 6; ++k) {
        rays[k]->AddPhiOperand(&loop.before, ray[k]);
        rays[k]->AddPhiOperand(&loop.latch, pick(local[k], current[k]));
    }
    loop.merge.AppendNewInst(Op::Phi, {}, word).AddPhiOperand(&loop.body, child);
}

IR::OracleWalks DeriveWithHighWord(HighWord kind, IR::ResourceKind address_kind = {}, uint32_t header_at = 0x20,
                                   std::optional<IR::ResourceKind> user_kind = std::nullopt) {
    SourceWalk walk;
    walk.address_kind   = address_kind;
    walk.user_kind      = user_kind;
    walk.program.blocks = {&walk.entry, &walk.header, &walk.body, &walk.latch, &walk.merge};
    Number(walk);
    const auto ray    = UserRay(walk);
    const auto header = UserHeader(walk, header_at);
    BuildWalk(walk, {walk.entry, walk.header, walk.body, walk.latch, walk.merge}, header, ray, kind);
    return IR::DeriveOracleWalks(walk.program, walk.sites);
}

void ScalarHeaderWalk() {
    SourceWalk walk;
    walk.program.blocks = {&walk.entry, &walk.header, &walk.body, &walk.latch, &walk.merge};
    Number(walk);
    walk.address_kind = IR::ResourceKind::ScalarAddress;
    const auto ray    = UserRay(walk);
    const auto header = UserHeader(walk, 0x23, 7);
    BuildWalk(walk, {walk.entry, walk.header, walk.body, walk.latch, walk.merge}, header, ray, HighWord::Exact, true);
    const auto traced = IR::TraceNativeTraversals(walk.program, {.wave_size = 32, .rewrite = true, .oracle = IR::OracleRemaining | IR::OracleOnly});
    const auto flags = walk.sites.front()->Flags<uint32_t>();
    Expect(traced.source_valid && traced.user_sgpr == 4 && traced.header_offset == 0x24,
           "a scalar header load rounds down its dynamic and immediate offsets independently");
    uint64_t pointer = 0x1003;
    const auto vsharp = IR::ResolveRayQueryUserTable(pointer, traced.header_vsharp);
    Expect(vsharp == IR::ShaderInfo::NoRayQuerySlot && pointer + traced.header_offset == 0x1024,
           "the host reads the same dword as the scalar load with an unaligned user pointer");
    pointer = 0x1003;
    const auto descriptor = IR::ResolveRayQueryUserTable(pointer, IR::ScalarRayQueryVsharpAt(0x10027));
    Expect(pointer == 0x1000 && descriptor == 0x10024, "a scalar descriptor load keeps the aligned table base and descriptor offset");
    pointer = 0x1003;
    Expect(IR::ResolveRayQueryUserTable(pointer, IR::ShaderInfo::UserDataRayQueryVsharp) == IR::ShaderInfo::UserDataRayQueryVsharp &&
               pointer == 0x1003,
           "a buffer descriptor supplied directly by the user retains its byte address");
    Expect((flags & IR::BvhIntersectOracle) != 0 && (flags & IR::BvhIntersectEmpty) == 0,
           "a walk reading its header and instance offset with scalar loads is served by a host query");
}

struct TwoWalks {
    SourceWalk             walk;
    IR::NativeTraversals   traced;
    std::array<uint32_t, 2> flags {};
};

void TraceTwoWalks(TwoWalks& two, uint32_t second_offset) {
    auto& walk          = two.walk;
    walk.program.blocks = {&walk.entry, &walk.header, &walk.body, &walk.latch, &walk.merge,
                           &walk.second_header, &walk.second_body, &walk.second_latch, &walk.second_merge};
    Number(walk);
    const auto ray    = UserRay(walk);
    const auto first  = UserHeader(walk, 0x20);
    const auto second = UserHeader(walk, second_offset);
    BuildWalk(walk, {walk.entry, walk.header, walk.body, walk.latch, walk.merge}, first, ray, HighWord::Exact);
    BuildWalk(walk, {walk.merge, walk.second_header, walk.second_body, walk.second_latch, walk.second_merge}, second, ray, HighWord::Exact);
    two.traced = IR::TraceNativeTraversals(walk.program, {.wave_size = 32, .rewrite = true, .oracle = IR::OracleRemaining | IR::OracleOnly});
    for (size_t at = 0; at < 2 && at < walk.sites.size(); ++at) two.flags[at] = walk.sites[at]->Flags<uint32_t>();
}

void TwoTlasWalks() {
    const auto multi = [](uint32_t flags) { return (flags & IR::BvhIntersectOracleMulti) != 0; };
    const auto host  = [](uint32_t flags) { return (flags & IR::BvhIntersectOracle) != 0; };
    TwoWalks   two;
    TraceTwoWalks(two, 0x40);
    Expect(two.traced.sites == 2 && host(two.flags[0]) && host(two.flags[1]), "both walks of a program over two TLASes are answered from the host query");
    Expect(two.traced.source_valid && two.traced.header_offset == 0x20u, "the first walk's TLAS is the primary source");
    Expect(two.traced.extra_sources.size() == 1 && two.traced.extra_sources.front().user_sgpr == 4u &&
               two.traced.extra_sources.front().header_offset == 0x40u,
           "the second walk's TLAS is bound as an extra source");
    Expect(!multi(two.flags[0]), "the walk over the primary TLAS keeps the single-TLAS query");
    Expect(multi(two.flags[1]), "the walk over the other TLAS selects its TLAS per lane");
    TwoWalks same;
    TraceTwoWalks(same, 0x20);
    Expect(same.traced.sites == 2 && same.traced.extra_sources.empty() && !multi(same.flags[0]) && !multi(same.flags[1]),
           "two walks over one TLAS bind no extra source and keep the single-TLAS query");
}

struct ChasedWalks {
    TwoWalks two;
    uint32_t first  = 0;
    uint32_t second = 0;
};

enum class ChasedSource { Address, Scalar, Lane, ScalarBuffer };

void TraceChasedWalks(ChasedWalks& chased, ChasedSource source) {
    auto& walk          = chased.two.walk;
    walk.program.blocks = {&walk.entry, &walk.header, &walk.body, &walk.latch, &walk.merge,
                           &walk.second_header, &walk.second_body, &walk.second_latch, &walk.second_merge};
    Number(walk);
    const auto ray    = UserRay(walk);
    const auto named  = UserHeader(walk, 0x20);
    const bool scalar = source == ChasedSource::Scalar;
    if (scalar) walk.address_kind = IR::ResourceKind::ScalarAddress;
    auto table = scalar ? UserHeader(walk, 0x23, 7) : UserHeader(walk, 0x1b4);
    if (source == ChasedSource::ScalarBuffer) {
        using Op = IR::ValueOpcode;
        const auto reg = [&](uint32_t n) { return walk.Add(Op::GetUserData, {IR::Value(IR::ScalarReg {n})}); };
        const auto buffer = walk.Add(Op::GetBufferResource, {reg(4), reg(5), reg(7), IR::Value(0x4000u)});
        table = {walk.Load(Op::ReadConstBuffer, {buffer, IR::Value(0u)}, 0x1b4, false),
                 walk.Load(Op::ReadConstBuffer, {buffer, IR::Value(0u)}, 0x1b8, false)};
    }
    auto low = table[0];
    if (source == ChasedSource::Lane) low = walk.Add(IR::ValueOpcode::IAdd32, {low, walk.Add(IR::ValueOpcode::LaneId, {})});
    const auto high     = walk.Add(IR::ValueOpcode::BitwiseAnd32, {table[1], IR::Value(uint32_t {0xffffu})});
    const auto resource = walk.Add(IR::ValueOpcode::GetAddressResource, {low, high});
    const uint32_t offset = scalar ? 0x43 : 0x40;
    const uint32_t displacement = scalar ? 7 : 0;
    const std::array<IR::Value, 2> header {walk.LoadAddress(resource, offset, displacement),
                                           walk.LoadAddress(resource, offset + 4, displacement)};
    walk.address_kind = IR::ResourceKind::None;
    BuildWalk(walk, {walk.entry, walk.header, walk.body, walk.latch, walk.merge}, named, ray, HighWord::Exact);
    BuildWalk(walk, {walk.merge, walk.second_header, walk.second_body, walk.second_latch, walk.second_merge}, header, ray, HighWord::Masked);
    chased.two.traced = IR::TraceNativeTraversals(walk.program, {.wave_size = 32, .rewrite = true, .oracle = IR::OracleRemaining | IR::OracleOnly});
    chased.first      = walk.sites[0]->Flags<uint32_t>();
    chased.second     = walk.sites[1]->Flags<uint32_t>();
}

void UntracedWalk() {
    ChasedWalks chased;
    TraceChasedWalks(chased, ChasedSource::Lane);
    const auto& traced = chased.two.traced;
    Expect(traced.sites == 1 && traced.source_valid && traced.header_offset == 0x20u && traced.extra_sources.empty(),
           "a walk over a named TLAS is served although another walk's TLAS is not traced");
    Expect((chased.first & IR::BvhIntersectOracle) != 0, "the walk over the named TLAS is answered from the host query");
    Expect((chased.second & IR::BvhIntersectOracle) == 0 && (chased.second & IR::BvhIntersectEmpty) != 0,
           "the walk stays empty when its header depends on a lane");
}

void ComputedHeader(ChasedSource source) {
    const bool scalar = source == ChasedSource::Scalar;
    ChasedWalks chased;
    TraceChasedWalks(chased, source);
    const auto& traced = chased.two.traced;
    Expect(traced.source_valid && traced.header_offset == 0x20u && traced.extra_sources.size() == 1,
           "a walk over a TLAS reached through a pointer chain is served as an extra tree");
    Expect((chased.first & IR::BvhIntersectOracle) != 0 && (chased.second & IR::BvhIntersectOracle) != 0 &&
               (chased.second & IR::BvhIntersectEmpty) == 0,
           "both walks are answered from the host query");
    if (traced.extra_sources.size() != 1) return;
    const auto& extra = traced.extra_sources[0];
    Expect(extra.header_vsharp == IR::RayQueryHeaderProgram && !extra.program.empty(), "the extra tree names its header with a program");
    std::array<uint32_t, 16> user {};
    user[4] = scalar ? 0x1003u : 0x10000u;
    user[5] = scalar ? 0u : 2u;
    user[7] = 0x1bcu;
    if (source == ChasedSource::ScalarBuffer) user[4] |= 3u;
    const uint64_t first = scalar ? 0x1024u : 0x2000101b4ull;
    const uint64_t table = scalar ? 0x1234500003ull : 0x1234500000ull;
    const auto     read    = [&](uint64_t address, uint32_t bytes, uint64_t& value) {
        if (address == first && bytes == 8) {
            value = table | 0xabcd000000000000ull;
            return true;
        }
        if (address == first && bytes == 4) {
            value = static_cast<uint32_t>(table);
            return true;
        }
        if (address == first + 4 && bytes == 4) {
            value = static_cast<uint32_t>((table | 0xabcd000000000000ull) >> 32u);
            return true;
        }
        return false;
    };
    const auto at = IR::EvaluateHeaderProgram(extra.program, user, read);
    Expect(at == (scalar ? 0x1234500044ull : table + 0x40), "each scalar load aligns its base, displacement and immediate independently");
    if (source == ChasedSource::ScalarBuffer) {
        user[7] -= 4;
        Expect(!IR::EvaluateHeaderProgram(extra.program, user, read), "a scalar buffer must contain both dwords of the pointer");
    }
    const auto unreadable = IR::EvaluateHeaderProgram(extra.program, user, [](uint64_t, uint32_t, uint64_t&) { return false; });
    Expect(!unreadable, "a program whose pointer chain is unreadable yields no header");
}

enum class WalkGuard { Blocked, Reachable, MixedVote, VoteAtJoin };

IR::OracleWalks GuardedWalks(WalkGuard guard) {
    using Op = IR::ValueOpcode;
    IR::Block   pre, join, second_pre, exit;
    SourceWalk  walk;
    walk.program.blocks = {&walk.entry, &pre, &walk.header, &walk.body, &walk.latch, &walk.merge, &join, &second_pre,
                           &walk.second_header, &walk.second_body, &walk.second_latch, &walk.second_merge, &exit};
    Number(walk);
    const auto u32    = [](uint32_t value) { return IR::Value(value); };
    const auto word   = static_cast<uint64_t>(IR::Type::U32);
    const auto ray    = UserRay(walk);
    const auto first  = UserHeader(walk, 0x40);
    const auto second = UserHeader(walk, 0x20);
    const auto flag   = walk.Add(Op::GetUserData, {IR::Value(IR::ScalarReg {14})});
    const auto ref = [&](CFG::BranchCondition kind, IR::Value predicate) {
        return walk.Add(Op::ConditionRef, {predicate}, static_cast<uint64_t>(kind));
    };
    IR::Value skip, later;
    switch (guard) {
        case WalkGuard::Blocked:
        case WalkGuard::Reachable:
            later = walk.Add(Op::INotEqual32, {flag, u32(0)});
            skip  = ref(CFG::BranchCondition::ScalarInstruction, later);
            break;
        case WalkGuard::MixedVote: {
            const auto lane_zero = walk.Add(Op::IEqual32, {walk.Add(Op::LaneId, {}), u32(0)});
            skip  = walk.Add(Op::LogicalNot, {ref(CFG::BranchCondition::VccNonZero, lane_zero)});
            later = walk.Add(Op::LogicalNot, {lane_zero});
            break;
        }
        case WalkGuard::VoteAtJoin:
            skip = walk.Add(Op::IEqual32, {walk.Add(Op::LaneId, {}), u32(0)});
            break;
    }
    walk.entry.condition              = skip;
    walk.entry.terminator.kind        = CFG::TerminatorKind::ConditionalBranch;
    walk.entry.terminator.true_block  = &join;
    walk.entry.terminator.false_block = &pre;
    walk.entry.terminator.merge_block = &join;
    walk.entry.AddBranch(&join);
    walk.entry.AddBranch(&pre);
    BuildWalk(walk, {pre, walk.header, walk.body, walk.latch, walk.merge}, first, ray, HighWord::Exact);

    walk.block        = &walk.merge;
    const auto table  = walk.Add(Op::GetAddressResource, {second[0], second[1]});
    const auto low    = walk.Add(Op::IAdd32, {second[0], walk.LoadAddress(table, 0x30)});
    const auto carry  = walk.Add(Op::SelectU32, {walk.Add(Op::ULessThan32, {low, second[0]}), u32(1), u32(0)});
    const auto high   = walk.Add(Op::IAdd32, {walk.Add(Op::IAdd32, {second[1], walk.LoadAddress(table, 0x34)}), carry});
    const auto buffer = walk.Add(Op::GetBufferResource, {low, walk.Add(Op::BitwiseOr32, {high, u32(0x00800000u)}), u32(0xffffffffu), u32(0x00027000u)});
    const auto index  = walk.Add(Op::ShiftRightLogical32, {flag, u32(3)});
    const auto record = [&](uint32_t dword) {
        return walk.Load(Op::LoadBufferU32, {buffer, index, u32(0), u32(0), IR::Value(true)}, dword * 4u, true);
    };
    const std::array<IR::Value, 2> raw {record(12), record(13)};
    Link(walk.merge, join);

    walk.block       = &join;
    auto& header_low  = join.AppendNewInst(Op::Phi, {}, word);
    auto& header_high = join.AppendNewInst(Op::Phi, {}, word);
    auto& gate        = join.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(IR::Type::U1));
    header_low.AddPhiOperand(&walk.entry, second[0]);
    header_low.AddPhiOperand(&walk.merge, raw[0]);
    header_high.AddPhiOperand(&walk.entry, second[1]);
    header_high.AddPhiOperand(&walk.merge, raw[1]);
    gate.AddPhiOperand(&walk.entry, IR::Value(false));
    gate.AddPhiOperand(&walk.merge, IR::Value(true));
    const auto open = guard == WalkGuard::Reachable ? IR::Value(&gate) : walk.Add(Op::LogicalNot, {IR::Value(&gate)});
    join.condition              = guard == WalkGuard::VoteAtJoin ? ref(CFG::BranchCondition::VccNonZero, open)
                                                                : walk.Add(Op::LogicalOr, {later, open});
    join.terminator.kind        = CFG::TerminatorKind::ConditionalBranch;
    join.terminator.true_block  = &second_pre;
    join.terminator.false_block = &exit;
    join.terminator.merge_block = &exit;
    join.AddBranch(&second_pre);
    join.AddBranch(&exit);
    BuildWalk(walk, {second_pre, walk.second_header, walk.second_body, walk.second_latch, walk.second_merge},
              {IR::Value(&header_low), IR::Value(&header_high)}, ray, HighWord::Masked);
    Link(walk.second_merge, exit);
    return IR::DeriveOracleWalks(walk.program, walk.sites);
}

void UnreachableEdgeWalks() {
    const auto bound = [](const IR::OracleWalks& derived, size_t loop) -> std::optional<uint32_t> {
        if (derived.loop_headers.size() <= loop || derived.loop_headers[loop].size() != 1) return std::nullopt;
        return derived.headers[derived.loop_headers[loop].front()].header_offset;
    };
    const auto blocked = GuardedWalks(WalkGuard::Blocked);
    if (blocked.refusal != nullptr) std::fprintf(stderr, "unreachable edge walks refused: %s\n", blocked.refusal);
    Expect(blocked.refusal == nullptr && bound(blocked, 0) == 0x40u && bound(blocked, 1) == 0x20u,
           "a header value whose edge cannot reach the walk does not reject its TLAS");
    const auto rejects_second = [&](WalkGuard guard) {
        const auto derived = GuardedWalks(guard);
        return derived.refusal == nullptr && derived.loop_headers.size() == 2 && bound(derived, 0) == 0x40u && derived.loop_headers[1].empty();
    };
    Expect(rejects_second(WalkGuard::Reachable), "a full-width record read that can reach the walk still rejects the TLAS");
    Expect(rejects_second(WalkGuard::MixedVote), "a positive wave vote leaves other lanes free to carry a full-width record to the walk");
    Expect(rejects_second(WalkGuard::VoteAtJoin), "a wave vote at the join includes lanes from both phi inputs");
}

void HeaderPrograms() {
    using Op = IR::HeaderOp;
    const auto op = [](Op value) { return static_cast<uint32_t>(value); };
    std::array<uint32_t, 16> user {};
    user[0] = 0x1000u;
    user[1] = 0x0040u << 16u;
    user[2] = 8u;
    user[3] = 1u << 12u;
    user[4] = 3u;
    const std::vector<uint32_t> record {op(Op::User), 0, op(Op::User), 1, op(Op::Join), op(Op::User), 2, op(Op::User), 3,
                                        op(Op::User), 4, op(Op::Const), 4, 0, op(Op::Record)};
    const auto none = [](uint64_t, uint32_t, uint64_t&) { return false; };
    Expect(IR::EvaluateHeaderProgram(record, user, none) == 0x1000u + 3u * 0x40u + 4u,
           "a buffer record address adds the index times the descriptor stride and the offset");
    user[3] |= 3u << 28u;
    Expect(!IR::EvaluateHeaderProgram(record, user, none), "a raw descriptor bounds the byte offset against its byte count");
    user[3] = 0u;
    Expect(!IR::EvaluateHeaderProgram(record, user, none), "a null buffer format yields no address");
    user[3] = 1u << 12u;
    user[4] = 8u;
    Expect(!IR::EvaluateHeaderProgram(record, user, none), "a buffer index past the record count yields no address");
    user[4] = 3u;
    user[3] |= 1u << 23u;
    Expect(!IR::EvaluateHeaderProgram(record, user, none), "a descriptor that adds the thread ID to its index yields no address");
    user[3] = 1u << 12u;
    user[1] |= 1u << 31u;
    Expect(!IR::EvaluateHeaderProgram(record, user, none), "a descriptor with swizzle enabled in its base yields no address");
    user[1] &= ~(1u << 31u);
    user[1] |= 1u << 30u;
    Expect(!IR::EvaluateHeaderProgram(record, user, none), "a descriptor with a cache swizzle in its base yields no address");
    user[1] &= ~(1u << 30u);
    const std::vector<uint32_t> past_stride {op(Op::User), 0, op(Op::User), 1, op(Op::Join), op(Op::User), 2, op(Op::User), 3,
                                             op(Op::User), 4, op(Op::Const), 0x3e, 0, op(Op::Record)};
    Expect(!IR::EvaluateHeaderProgram(past_stride, user, none), "a record read that crosses its stride yields no address");
    const std::vector<uint32_t> aligned_pair {op(Op::Const), 0x2004, 0, op(Op::Load), 8};
    Expect(IR::EvaluateHeaderProgram(aligned_pair, user, [](uint64_t, uint32_t, uint64_t& value) {
               value = 0x5000;
               return true;
           }) == 0x5000u,
           "a 64-bit pointer load needs only dword alignment");
    Expect(!IR::EvaluateHeaderProgram(record, std::span(user).first(4), none), "a program that reads a missing user register is invalid");
    const std::vector<uint32_t> unbalanced {op(Op::User), 0, op(Op::User), 1};
    Expect(!IR::ValidHeaderProgram(unbalanced, 16) && !IR::ValidHeaderProgram({}, 16), "a program must leave exactly one value");
    const std::vector<uint32_t> misaligned {op(Op::Const), 3, 0, op(Op::Load), 8};
    Expect(!IR::EvaluateHeaderProgram(misaligned, user, [](uint64_t, uint32_t, uint64_t& value) {
               value = 1;
               return true;
           }),
           "a misaligned load yields no address");
    const std::vector<uint32_t> fields {op(Op::Const), 0x12345678u, 0, op(Op::Const), 22, 0, op(Op::Const), 9, 0, op(Op::Bfe)};
    Expect(IR::EvaluateHeaderProgram(fields, user, none) == (0x12345678u >> 22u & 0x1ffu), "a bit-field extract reads the field");
    const std::vector<uint32_t> full {op(Op::User), 0, op(Op::Const), 0, 0, op(Op::Const), 32, 0, op(Op::Bfe)};
    Expect(IR::EvaluateHeaderProgram(full, user, none) == user[0], "a full-width bit-field extract preserves every bit");
}

void CodeWaveSize() {
    namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;
    const auto wave   = [](std::vector<uint32_t> code) {
        code.push_back(0xbf810000u);
        Decoder::Program program;
        Decoder::DecodeProgram(code, program);
        return Decoder::ExecWaveSize(program);
    };
    constexpr uint32_t mov_exec_lo  = 0xbefe0300u;
    constexpr uint32_t mov_exec     = 0xbefe0400u;
    constexpr uint32_t saveexec_b32 = 0xbe823c6au;
    constexpr uint32_t saveexec_b64 = 0xbe82246au;
    constexpr uint32_t mov_s2       = 0xbe820300u;
    constexpr uint32_t mov_exec_hi  = 0xbeff0300u;
    constexpr uint32_t and_exec_lo  = 0x877e007eu;
    constexpr uint32_t and_exec     = 0x87fe007eu;
    constexpr uint32_t count_bits   = 0xbefe1000u;
    constexpr uint32_t replicate    = 0xbefe3b00u;
    constexpr uint32_t shift_i64    = 0x91fe007eu;
    constexpr uint32_t cmp_exec_hi  = 0xd4c2007fu;
    constexpr uint32_t cmp_sources  = 0x00020501u;
    Expect(wave({cmp_exec_hi, cmp_sources}) == 0u, "a vector comparison destination does not establish a scalar mask width");
    Expect(wave({count_bits}) == 32u && wave({replicate}) == 64u, "mixed-type scalar operations use their destination width");
    Expect(wave({shift_i64}) == 64u && wave({count_bits, shift_i64}) == 0u, "signed scalar writes participate in width inference");
    Expect(wave({mov_exec_lo, saveexec_b32}) == 32u, "code that writes exec with 32-bit operations is wave32");
    Expect(wave({mov_exec, saveexec_b64}) == 64u, "code that writes exec with 64-bit operations is wave64");
    Expect(wave({saveexec_b32}) == 32u && wave({saveexec_b64}) == 64u, "a saveexec alone decides the width");
    Expect(wave({mov_exec_lo, saveexec_b64}) == 0u, "code that mixes both widths has no known wave size");
    Expect(wave({mov_s2}) == 0u && wave({}) == 0u, "code that never writes exec has no known wave size");
    Expect(wave({mov_exec_lo, mov_exec_hi}) == 64u, "code that writes exec_hi is wave64 even with 32-bit operations");
    Expect(wave({and_exec_lo}) == 32u && wave({and_exec}) == 64u, "a scalar ALU write to exec decides the width");
}

void RecordLayouts() {
    constexpr std::array<uint32_t, 4> masked {0x76543210u, 0x3dc6ba98u, 0x000801dfu, 0u};
    constexpr std::array<uint32_t, 4> plain {0x76543210u, 0x3dc6ba98u, 0u, 0u};
    constexpr std::array<uint32_t, 4> reordered {0x76543201u, 0x3dc6ba98u, 0u, 0u};
    Expect(IR::CompatibleRecordLayouts(masked, masked), "a record layout is compatible with itself");
    Expect(IR::CompatibleRecordLayouts(masked, plain) && IR::CompatibleRecordLayouts(plain, masked),
           "layouts that differ only in the mask fields read the same records");
    Expect(!IR::CompatibleRecordLayouts(plain, reordered), "layouts that place record fields differently are not compatible");
}

void DescriptorHighWord() {
    constexpr const char* unproven = "a buffer descriptor's high word is not proven to leave the stride bits clear";
    const auto            refused  = [&](HighWord kind) {
        const auto walks = DeriveWithHighWord(kind);
        return walks.refusal != nullptr && std::strcmp(walks.refusal, unproven) == 0;
    };
    const auto exact = DeriveWithHighWord(HighWord::Exact);
    Expect(exact.refusal == nullptr && !exact.sources.empty(), "a walk whose descriptor high word is the high dword of header plus instance offset is derived");
    Expect(!exact.sources.empty() && exact.sources.front().user_sgpr == 4u && exact.sources.front().header_offset == 0x20u,
           "the derived walk reads the TLAS header the user data names");
    const auto scalar = DeriveWithHighWord(HighWord::Exact, IR::ResourceKind::ScalarAddress);
    Expect(scalar.refusal == nullptr && !scalar.sources.empty() && scalar.sources.front().user_sgpr == 4u &&
               scalar.sources.front().header_offset == 0x20u,
           "a TLAS header read with scalar loads through the user data pointer is derived");
    const auto plain  = DeriveWithHighWord(HighWord::Exact);
    const auto loaded = DeriveWithHighWord(HighWord::Exact, IR::ResourceKind::ScalarAddress, 0x20, IR::ResourceKind {});
    Expect(loaded.refusal == nullptr && !loaded.sources.empty() && !plain.sources.empty() && loaded.sources.front().user_sgpr == plain.sources.front().user_sgpr &&
               loaded.sources.front().header_offset == plain.sources.front().header_offset &&
               loaded.sources.front().root == plain.sources.front().root && loaded.sources.front().layout == plain.sources.front().layout,
           "scalar reads through a header pointer loaded from memory name the same source as plain reads");
    const auto unaligned = DeriveWithHighWord(HighWord::Exact, IR::ResourceKind::ScalarAddress, 0x22);
    Expect(!unaligned.sources.empty() && unaligned.sources.front().header_offset == 0x20u,
           "a scalar header read at an unaligned offset names the dword the hardware reads");
    Expect(refused(HighWord::StrideBit), "a computed high word that sets a stride bit is refused");
    Expect(refused(HighWord::SwizzleBit), "a computed high word that sets a swizzle bit is refused");
}

void PrologueFlag() {
    using Op = IR::ValueOpcode;
    IR::Block   entry, header, body, merge;
    IR::Program program;
    program.blocks = {&entry, &header, &body, &merge};
    program.stage  = Libs::Graphics::ShaderType::Compute;
    const auto u32  = [](uint32_t value) { return IR::Value(value); };
    const auto flag = IR::Value(&entry.AppendNewInst(Op::INotEqual32, {IR::Value(&entry.AppendNewInst(Op::LaneId)), u32(2)}));
    entry.terminator.kind       = CFG::TerminatorKind::Branch;
    entry.terminator.true_block = &header;
    entry.AddBranch(&header);
    const auto word  = static_cast<uint64_t>(IR::Type::U32);
    auto& node       = header.AppendNewInst(Op::Phi, {}, word);
    auto& descriptor = header.AppendNewInst(Op::CompositeConstructU32x4, {u32(0), u32(0), u32(0), u32(0)});
    const IR::Value one(uint32_t {0x3f800000u});
    auto& address = header.AppendNewInst(Op::MakeImageAddress, {IR::Value(&node), one, one, u32(0), u32(0), u32(0), one, one, one, one, one, one, one});
    auto& bvh     = header.AppendNewInst(Op::BvhIntersect, {IR::Value(&descriptor), IR::Value(&address), flag}, 1);
    auto& first   = header.AppendNewInst(Op::CompositeExtractU32x4, {IR::Value(&bvh), u32(0)});
    header.condition              = IR::Value(&header.AppendNewInst(Op::IEqual32, {IR::Value(&first), u32(0xffffffffu)}));
    header.terminator.kind        = CFG::TerminatorKind::ConditionalBranch;
    header.terminator.true_block  = &merge;
    header.terminator.false_block = &body;
    header.terminator.merge_block = &merge;
    header.terminator.loop_header = true;
    header.AddBranch(&merge);
    header.AddBranch(&body);
    body.terminator.kind       = CFG::TerminatorKind::Branch;
    body.terminator.true_block = &header;
    body.AddBranch(&header);
    node.AddPhiOperand(&entry, u32(5));
    node.AddPhiOperand(&body, IR::Value(&first));
    merge.AppendNewInst(Op::Phi, {}, word).AddPhiOperand(&header, IR::Value(&first));
    std::vector<IR::Inst*> served {&bvh};
    const auto sample = IR::SampleWalkEntry(program, {.wave_size = 32}, &header, &body, served, 1);
    Expect(sample.failure == nullptr, "the prologue of the test walk runs");
    if (sample.failure != nullptr || sample.flags.size() != 1 || sample.active.size() != 32) {
        Expect(false, "the test walk samples one entry flag per lane");
        return;
    }
    const auto& lanes = sample.flags.front();
    Expect(sample.active[2] != 0 && lanes[2] == 0u, "an active lane whose prologue computes false enters the walk disabled");
    Expect(sample.active[0] != 0 && lanes[0] == 1u, "an active lane whose prologue computes true enters the walk enabled");
    for (uint32_t lane = 0; lane < 32; ++lane) {
        if (sample.active[lane] == 0) Expect(lanes[lane] == 0u, "an inactive lane enters the walk disabled");
    }
}

void EntryLanes() {
    Expect(IR::EntryFlagOf(0u, true) == 0u, "a lane the prologue disables stays disabled while active");
    Expect(IR::EntryFlagOf(1u, true) == 1u, "a lane the prologue enables stays enabled while active");
    Expect(IR::EntryFlagOf(1u, false) == 0u, "an inactive lane is disabled");
    Expect(IR::EntryMaskOf(0b0101u, 0b0111u) == 0b0101u, "a mask the prologue narrows is not widened to the active lanes");
    Expect(IR::EntryMaskOf(0b1111u, 0b0011u) == 0b0011u, "a mask is narrowed to the active lanes");
}

}

int main() {
    EntryLanes();
    PrologueFlag();
    ScalarHeaderWalk();
    TwoTlasWalks();
    RecordLayouts();
    UntracedWalk();
    ComputedHeader(ChasedSource::Address);
    ComputedHeader(ChasedSource::Scalar);
    ComputedHeader(ChasedSource::ScalarBuffer);
    UnreachableEdgeWalks();
    HeaderPrograms();
    CodeWaveSize();
    DescriptorHighWord();
    Marking();
    EmptyWalks();
    Modes();
    NodeCountFacts();
    WaveScalarReadKindFacts();
    TriangleIndexSpirv();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim ray tracing oracle: all checks passed\n");
    return 0;
}

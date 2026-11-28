#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::ShaderRecompiler::IR;

constexpr uint32_t GarbageType = 0x9008a150u;

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_shader_store_encode: %s\n", what);
    std::abort();
}

Value Corrupt(uint32_t type) {
    Value value;
    std::memcpy(&value, &type, sizeof(type));
    return value;
}

void Fill(ResourcePlan& plan, Value second) {
    auto& add = plan.value_storage.emplace_back(ValueOpcode::IAdd32);
    add.SetArg(0, Value(1u));
    add.SetArg(1, second);
    auto& multiply = plan.value_storage.emplace_back(ValueOpcode::IMul32);
    multiply.SetArg(0, Value(&add));
    multiply.SetArg(1, Value(7u));
}

void HealthyPlanRoundTrips() {
    ResourcePlan plan;
    Fill(plan, Value(2u));
    std::vector<uint8_t> bytes;
    EncodeFailure        failure;
    Require(SaveResourcePlan(plan, bytes, failure), "a healthy plan was refused");
    Require(!failure.Failed() && !bytes.empty(), "a healthy plan reported a failure or no bytes");
    ResourcePlan loaded;
    Require(DecodeResourcePlan(bytes, loaded), "a saved plan did not load back");
    Require(loaded.value_storage.size() == 2u, "the loaded plan lost instructions");
    const auto& add = loaded.value_storage.front();
    Require(add.GetOpcode() == ValueOpcode::IAdd32 && add.Arg(0) == Value(1u) && add.Arg(1) == Value(2u),
            "the loaded plan changed an operand");
    const auto& multiply = loaded.value_storage.back();
    Require(multiply.GetOpcode() == ValueOpcode::IMul32 && multiply.Arg(0) == Value(&loaded.value_storage.front()) &&
                multiply.Arg(1) == Value(7u),
            "the loaded plan changed an instruction reference");
    std::vector<uint8_t> roundTrip;
    Require(SaveResourcePlan(loaded, roundTrip, failure) && roundTrip == bytes, "the loaded plan encoded to different bytes");
}

void ImpossibleOperandRefusesThePlan() {
    ResourcePlan plan;
    Fill(plan, Corrupt(GarbageType));
    std::vector<uint8_t> bytes {1, 2, 3};
    EncodeFailure        failure;
    Require(!SaveResourcePlan(plan, bytes, failure), "a plan with an impossible operand was saved");
    Require(failure.reason == EncodeFailure::Reason::ValueType && failure.type == GarbageType,
            "the failure does not name the impossible value type");
    Require(failure.in_instruction && failure.opcode == ValueOpcode::IAdd32,
            "the failure does not name the opcode of the instruction holding the value");
    Require(bytes.empty(), "a refused plan left bytes behind");
}

void ImpossibleValueOutsideInstructionsRefusesThePlan() {
    ResourcePlan plan;
    Fill(plan, Value(2u));
    auto& source       = plan.descriptor_sources.emplace_back();
    source.dword_count = 1;
    source.dwords[0]   = Corrupt(GarbageType);
    std::vector<uint8_t> bytes;
    EncodeFailure        failure;
    Require(!SaveResourcePlan(plan, bytes, failure), "a plan with an impossible descriptor value was saved");
    Require(failure.reason == EncodeFailure::Reason::ValueType && failure.type == GarbageType &&
                !failure.in_instruction,
            "the failure does not say the value sits outside the instructions");
}

void InstructionOutsideThePlanRefusesThePlan() {
    Inst         outside(ValueOpcode::IAdd32);
    ResourcePlan plan;
    Fill(plan, Value(&outside));
    std::vector<uint8_t> bytes;
    EncodeFailure        failure;
    Require(!SaveResourcePlan(plan, bytes, failure), "a plan referring to a foreign instruction was saved");
    Require(failure.reason == EncodeFailure::Reason::UnknownInstruction && failure.in_instruction &&
                failure.opcode == ValueOpcode::IAdd32,
            "the failure does not describe the foreign instruction");
}

void RefusedPlanAbandonsTheStoreEntry() {
    const ShaderModule         module;
    const std::vector<uint32_t> state {1, 2, 3};
    const std::vector<uint32_t> spirv {0x07230203u, 0u};
    const std::vector<uint32_t> plain;

    ResourcePlan healthy;
    Fill(healthy, Value(2u));
    std::vector<uint8_t> record;
    Require(EncodeModuleRecord(ShaderType::Compute, 0x1111u, 4u, state, healthy, module, spirv, plain, record),
            "a healthy program was not encoded");
    Require(!record.empty(), "a healthy program produced an empty record");

    ResourcePlan broken;
    Fill(broken, Corrupt(GarbageType));
    std::vector<uint8_t> refused {9};
    Require(!EncodeModuleRecord(ShaderType::Compute, 0x2222u, 4u, state, broken, module, spirv, plain, refused),
            "a program with an impossible value was encoded");
    Require(refused.size() == 1u && refused.front() == 9u, "a refused program touched the record");
    Require(!EncodeModuleRecord(ShaderType::Compute, 0x2222u, 4u, state, broken, module, spirv, plain, refused),
            "a second encode of the same refused program was encoded");
    Require(!EncodeModuleRecord(ShaderType::Pixel, 0x3333u, 4u, state, broken, module, spirv, plain, refused),
            "a refused program of another hash was encoded");

    std::vector<uint8_t> after;
    Require(EncodeModuleRecord(ShaderType::Compute, 0x1111u, 4u, state, healthy, module, spirv, plain, after),
            "a healthy program was not encoded after a refusal");
    Require(after == record, "a refusal changed the encoding of the next program");
}

void GraphicsStageDataRoundTrips() {
    const auto vertex = PushData::GraphicsStageCursor(ShaderType::Vertex);
    const auto pixel = PushData::GraphicsStageCursor(ShaderType::Pixel);
    Require(PushData::GraphicsStageCursor(ShaderType::Mesh) == vertex &&
            PushData::GraphicsStageCursor(ShaderType::Local) == vertex, "pre-raster stage window changed");
    for (uint32_t size = 1; size <= PushData::DwordCount; ++size) {
        const auto v = PushData::StartFor(vertex, size);
        const auto p = PushData::StartFor(pixel, size);
        Require(v == PushData::NoStart || (v >= PushData::MeshDrawDwordCount && v + size <= pixel),
                "vertex data overlaps draw or pixel data");
        Require(p == PushData::NoStart || (p == pixel && p + size <= PushData::DwordCount),
                "pixel data exceeds push constants");
    }
    Bindings layout;
    layout.memory_offset_dword = 2;
    layout.clip_plane_dwords = 12;
    layout.push_data_start_dword = PushData::StartFor(vertex, layout.ShaderDataDwords());
    Require(layout.ClipPlaneDword() == 2 && layout.push_data_start_dword == PushData::NoStart,
            "clip planes did not spill with their stage");
    Writer writer;
    Io(writer, layout);
    Reader reader {.bytes = writer.bytes};
    Bindings restored;
    Io(reader, restored);
    Require(reader.ok && reader.bytes.empty() && restored == layout, "stored clip-plane layout changed");
}

void GenericStageDataRoundTrips() {
    Bindings layout;
    layout.stride_count = 3;
    layout.generic_graphics = true;
    layout.format_count = 2;
    layout.image_bank_count = 7;
    layout.clip_plane_dwords = 4;
    Writer writer;
    IoAll(writer, layout);
    Reader reader{writer.bytes};
    Bindings restored;
    IoAll(reader, restored);
    Require(reader.ok && reader.bytes.empty() && restored.generic_graphics && restored.stride_count == 3 &&
            restored.format_count == 2 && restored.image_bank_count == 7 &&
            restored.ClipPlaneDword() == layout.ClipPlaneDword() &&
            restored.ShaderDataDwords() == layout.ShaderDataDwords(), "generic stage ABI did not round trip");
    ResourcePlan plan;
    plan.resource_tracking_complete = true;
    plan.srt_plan_complete = true;
    plan.srt_user_data = 2;
    const std::array<uint32_t, 2> words{4, 7};
    SrtRuntime runtime{.user_data = words};
    ResourceSnapshot snapshot;
    snapshot.graphics_image_banks.push_back(1);
    snapshot.specialization_reads.emplace_back(0, 4);
    ResourceSpecialization live;
    live.images.emplace_back();
    Require(MaterializeGenericGraphics(plan, runtime, snapshot, live, {}) && live.images.empty() &&
            snapshot.graphics_image_banks.empty() && snapshot.specialization_reads.empty() &&
            std::ranges::equal(snapshot.user_data, words), "register-only stage retained an earlier draw");
    runtime.user_data = {};
    Require(!MaterializeGenericGraphics(plan, runtime, snapshot, live, {}), "missing stage registers were accepted");
}

void GenericSampledImageBanks() {
    ResourcePlan plan;
    plan.stage = ShaderType::Pixel;
    plan.resource_tracking_complete = true;
    plan.srt_plan_complete = true;
    plan.descriptor_sources.emplace_back();
    auto& image = plan.info.images.emplace_back();
    image.source = 0;
    image.resource_class = ImageResourceClass::Sampled;
    image.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
    ResourceSpecialization shape;
    Require(GenericGraphicsShape(plan, shape) && shape.images.size() == 3, "missing typed sampled image banks");
    for (size_t bank = 0; bank < shape.images.size(); ++bank) {
        ResourceSpecialization live;
        live.images.push_back(shape.images.front());
        live.images.front().indirect_root = ImageResource::NoIndirectImage;
        live.images.front().numeric_class = shape.images[bank].numeric_class;
        ResourceSnapshot snapshot;
        snapshot.images.resize(1);
        snapshot.images[0].dwords[0] = 9;
        Require(BindGenericGraphics(plan, snapshot, live, shape), "typed view refused");
        Require(snapshot.images.size() == 3 && snapshot.graphics_image_banks.size() == 1 &&
                snapshot.graphics_image_banks[0] == bank, "wrong per-draw bank");
        for (size_t i = 0; i < snapshot.images.size(); ++i)
            Require(snapshot.images[i].dwords[0] == (i == bank ? 9u : 0u), "descriptor bound to the wrong bank");
    }
}

void TableGraphicsModuleRoundTrips() {
    using namespace ShaderStoreFormat;

    const std::vector<uint32_t> spirv {0x07230203u, 0x00010300u, 0u, 1u, 0u};
    const std::vector<uint32_t> plain {0x07230203u, 0x00010300u, 0u, 2u, 0u};
    const std::vector<uint32_t> no_plain;
    for (const auto stage : {ShaderType::Vertex, ShaderType::Pixel}) {
        const ShaderProgramKey key {stage, 0x112233u, 4, {3, 9, TableShaderKeyMarker}};
        const auto cursor = PushData::GraphicsStageCursor(stage);
        const auto& plain_spirv = stage == ShaderType::Pixel ? plain : no_plain;
        ShaderModule module;
        module.info.stage = stage;
        module.info.shader_hash = key.hash;
        module.info.table_mode = true;
        auto& layout = module.info.bindings;
        layout.table_block_dwords = 2;
        layout.memory_offset_dword = 3;
        layout.push_data_start_dword = PushData::StartFor(cursor, layout.ShaderDataDwords());
        auto& table = module.info.table_plan;
        table.slots.emplace_back().low = {TablePlan::Operand::Kind::UserData, 2};
        table.cpu.push_back(1);
        table.gpu.push_back(1);
        module.specialization.samplers.emplace_back().indirect_root = 2;
        if (stage == ShaderType::Vertex) module.vertex_formats.push_back(vk::Format::eR32G32Sfloat);
        Writer identity;
        IoAll(identity, key.stage, key.hash, key.user_data_count, key.static_state,
              layout.push_data_start_dword, module.specialization);
        module.id = XXH3_64bits(identity.bytes.data(), identity.bytes.size());

        std::vector<uint8_t> bytes;
        Require(EncodeTableModuleRecord(key, cursor, module, spirv, plain_spirv, bytes),
                "could not encode a table graphics module");
        ShaderProgramKey restored_key;
        uint32_t restored_cursor = 0;
        ShaderModule restored;
        std::vector<uint32_t> restored_spirv, restored_plain;
        const auto decode = [&](std::span<const uint8_t> data) {
            Reader reader {.bytes = data};
            Record kind {};
            IoAll(reader, kind);
            return kind == Record::TableModule &&
                   DecodeTableModuleRecord(reader, restored_key, restored_cursor, restored, restored_spirv, restored_plain);
        };
        Require(decode(bytes), "could not decode a table graphics module");
        Require(restored_key == key && restored_cursor == cursor && restored.id == module.id &&
                    restored.specialization == module.specialization,
                "the table module lookup identity changed");
        Require(restored.info.table_mode && restored.info.bindings == layout && restored.info.table_plan == table &&
                    restored.vertex_formats == module.vertex_formats && restored_spirv == spirv && restored_plain == plain_spirv,
                "the table module binding layout or program changed");
        Require(!decode(std::span(bytes).first(bytes.size() - 1)), "accepted a truncated table module");
        bytes.push_back(0x21u);
        Require(!decode(bytes), "accepted trailing bytes in a table module");
        Require(!EncodeTableModuleRecord(key, cursor + 1, module, spirv, plain_spirv, bytes),
                "accepted the wrong requested data cursor");
        module.specialization.samplers.front().indirect_root++;
        Require(!EncodeTableModuleRecord(key, cursor, module, spirv, plain_spirv, bytes),
                "accepted a table module with the wrong specialization for its ID");
    }
}

void NativeGraphicsRequiresExactTableStages() {
    ShaderModule vertex;
    vertex.id = 123u;
    vertex.info.stage = ShaderType::Vertex;
    vertex.info.table_mode = true;
    vertex.info.bindings.push_data_start_dword = PushData::MeshDrawDwordCount;
    ShaderModule pixel = vertex;
    pixel.id = 456u;
    pixel.info.stage = ShaderType::Pixel;
    pixel.info.bindings.push_data_start_dword = PushData::PixelStart;
    ShaderModule plain = pixel;
    plain.id ^= PlainModuleIdXor;
    GraphicsPipelineKey key {.stage_ids = {vertex.id, 0, 0, plain.id}};
    Require(ValidNativeGraphicsStages(key, {&vertex, nullptr, nullptr, &plain}),
            "native pipeline could not find the recorded plain pixel stage");
    Require(!ValidNativeGraphicsStages(key, {&vertex, nullptr, nullptr, &pixel}),
            "native pipeline accepted a different pixel variant");
    Require(!ValidNativeGraphicsStages(key, {&vertex, nullptr, nullptr, nullptr}),
            "native pipeline accepted a missing pixel stage");
    plain.info.table_mode = false;
    Require(!ValidNativeGraphicsStages(key, {&vertex, nullptr, nullptr, &plain}),
            "native pipeline accepted a standard graphics layout");
    key.stage_ids[3] = 0;
    Require(ValidNativeGraphicsStages(key, {&vertex, nullptr, nullptr, nullptr}),
            "native pipeline rejected a vertex-only variant");
    key.stage_ids[1] = vertex.id;
    Require(!ValidNativeGraphicsStages(key, {&vertex, &vertex, nullptr, nullptr}),
            "native pipeline accepted an unsupported stage layout");
}

void StoredComputeModuleSelection() {
    using namespace ShaderStoreFormat;

    const auto frame = [](ShaderType stage, uint64_t id, ShaderBvh bvh = ShaderBvh::None, uint32_t mode = 0) {
        ShaderStoreFormat::Key key {.stage = stage, .hash = id};
        ShaderModule module;
        module.id = id;
        module.info.stage = stage;
        module.bvh = bvh;
        module.rt_mode = mode;
        const std::vector<uint8_t> plan;
        const std::vector<uint32_t> spirv {0x07230203u, 0x00010300u, 0u, 1u, 0u};
        const std::vector<uint32_t> plain;
        Writer record;
        IoAll(record, Record::Module);
        IoStoreModule(record, key, module, plan, spirv, plain);
        std::string compressed(ZSTD_compressBound(record.bytes.size()), '\0');
        const auto size = ZSTD_compress(compressed.data(), compressed.size(), record.bytes.data(), record.bytes.size(), 1);
        Require(!ZSTD_isError(size), "could not encode the synthetic store frame");
        compressed.resize(size);
        const Frame header {static_cast<uint32_t>(size), static_cast<uint32_t>(record.bytes.size()),
                            XXH3_64bits(compressed.data(), size)};
        return std::string(reinterpret_cast<const char*>(&header), sizeof(header)) + compressed;
    };
    const std::array<uint8_t, 3> expected {1, 2, 3};
    const std::string header(reinterpret_cast<const char*>(expected.data()), expected.size());
    const auto compute = frame(ShaderType::Compute, 42);
    const auto pixel = frame(ShaderType::Pixel, 17);
    ShaderModule module;
    std::vector<uint32_t> spirv;
    std::string error;
    const auto load = [&](std::string contents, uint64_t id) {
        std::istringstream stream(std::move(contents));
        return LoadStoredComputeModule(stream, expected, id, module, spirv, error);
    };
    Require(load(header + pixel + compute, 42) && module.id == 42 && module.spirv_words == spirv.size() &&
                spirv.size() == 5 && spirv.front() == 0x07230203u,
            "replay did not select the requested compute record");
    Require(!load(header + compute, 43) && error.find("not found") != std::string::npos,
            "replay accepted a missing ID");
    Require(!load(header + pixel, 17) && error.find("not a compute") != std::string::npos,
            "replay accepted a graphics module");
    Require(!load(header + compute + compute, 42) && error.find("duplicate") != std::string::npos,
            "replay accepted duplicate IDs");
    const auto native = frame(ShaderType::Compute, 42, ShaderBvh::Bvh4, 2);
    const auto guest = frame(ShaderType::Compute, 42, ShaderBvh::Bvh4, 0);
    Require(load(header + native + guest, 42) && module.bvh == ShaderBvh::Bvh4 && module.rt_mode == 2,
            "replay lost a shader shared by two traversal modes");
    Require(!load(header + native + guest + guest, 42) && error.find("duplicate") != std::string::npos,
            "replay accepted the same traversal mode twice");
    Require(!load("bad" + compute, 42) && error.find("incompatible") != std::string::npos,
            "replay accepted an incompatible header");
    Require(!load(header + compute + pixel.substr(0, pixel.size() - 1), 42),
            "replay accepted a truncated frame after the selected module");
}

void GenericShapeRefusalsNameTheirReason() {
    const auto refusal = [](const ResourcePlan& plan) -> std::string {
        ResourceSpecialization shape;
        const char* reason = nullptr;
        if (GenericGraphicsShape(plan, shape, &reason)) return "accepted";
        return reason == nullptr ? "unnamed" : reason;
    };
    const auto graphics = [] {
        ResourcePlan plan;
        plan.stage = ShaderType::Vertex;
        plan.resource_tracking_complete = true;
        plan.srt_plan_complete = true;
        return plan;
    };
    const auto table = [](bool array) {
        DescriptorSource source;
        source.dword_count = 8;
        source.indirect_descriptor.emplace();
        source.indirect_descriptor->table_array = array;
        return source;
    };

    auto loose = graphics();
    loose.descriptor_sources.push_back(table(false));
    Require(refusal(loose) == "indirect_descriptor_not_table_array", "a non-array indirect descriptor was not named");

    auto two = graphics();
    two.descriptor_sources.push_back(table(true));
    two.descriptor_sources.push_back(table(true));
    Require(refusal(two) == "several_indirect_tables", "two indirect tables were not named");

    auto written = graphics();
    written.stage = ShaderType::Pixel;
    auto& buffer = written.info.buffers.emplace_back();
    buffer.formatted = true;
    buffer.written = true;
    Require(refusal(written) == "formatted_buffer_written", "a written formatted buffer was not named");

    auto compute = graphics();
    compute.stage = ShaderType::Compute;
    Require(refusal(compute) == "stage", "a compute stage was not named");

    auto single = graphics();
    single.descriptor_sources.push_back(table(true));
    Require(refusal(single) == "accepted", "a single table array was refused");

    const auto table_image = [&](bool compare, bool written) {
        auto plan = graphics();
        plan.stage = ShaderType::Pixel;
        plan.descriptor_sources.push_back(table(true));
        auto& image = plan.info.images.emplace_back();
        image.source = 0;
        image.resource_class = ImageResourceClass::Sampled;
        image.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
        image.depth_compare = compare;
        image.written = written;
        return plan;
    };
    Require(refusal(table_image(true, false)) == "accepted", "a compared table image was refused");
    Require(refusal(table_image(false, true)) == "table_image_not_plain_sampled",
            "a written table image was not named");
}

}

int main() {
    HealthyPlanRoundTrips();
    ImpossibleOperandRefusesThePlan();
    ImpossibleValueOutsideInstructionsRefusesThePlan();
    InstructionOutsideThePlanRefusesThePlan();
    RefusedPlanAbandonsTheStoreEntry();
    GraphicsStageDataRoundTrips();
    GenericStageDataRoundTrips();
    GenericSampledImageBanks();
    GenericShapeRefusalsNameTheirReason();
    TableGraphicsModuleRoundTrips();
    NativeGraphicsRequiresExactTableStages();
    StoredComputeModuleSelection();
    std::puts("kyty_shim_shader_store_encode: ok");
    return 0;
}

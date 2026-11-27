#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::ShaderRecompiler;

void Require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stdout, "kyty_shim_image_format_diagnostics: %s\n", message);
    std::exit(1);
}

IR::DescriptorValue MakeStorageDescriptor(Prospero::BufferFormat format) {
    IR::DescriptorValue value {.dword_count = 8u};
    value.dwords[0] = 0x1000u;
    value.dwords[1] = (static_cast<uint32_t>(format) << 20u) | (3u << 30u);
    value.dwords[2] = 15u;
    value.dwords[3] = 0xfacu | (9u << 28u);
    return value;
}

IR::ResourcePlan MakeStorageImagePlan(Prospero::BufferFormat format, bool atomic) {
    IR::ResourcePlan plan;
    plan.stage                      = ShaderType::Compute;
    plan.resource_tracking_complete = true;
    plan.srt_plan_complete          = true;

    const auto descriptor = MakeStorageDescriptor(format);
    IR::DescriptorSource source;
    source.dword_count = 8u;
    for (size_t i = 0; i < 8u; ++i) {
        source.dwords[i] = IR::Value(descriptor.dwords[i]);
    }
    plan.descriptor_sources.push_back(source);

    IR::ImageResource image;
    image.resource_class = IR::ImageResourceClass::Storage;
    image.written        = true;
    image.atomic         = atomic;
    image.dimension      = Decoder::ImageDimension::Dim2D;
    image.source         = 0u;
    image.first_use_pc   = 0x1234u;
    plan.info.images.push_back(image);
    IR::CompileSrtWalker(plan);
    return plan;
}

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

}

int main() {
    const auto valid_plan = MakeStorageImagePlan(Prospero::BufferFormat::k32UInt, true);
    IR::SrtRuntime             runtime {};
    IR::ResourceSnapshot       valid_snap;
    IR::ResourceSpecialization valid_spec;
    Require(IR::MaterializeResources(valid_plan, runtime, valid_snap, valid_spec),
            "valid 32-bit atomic image failed materialization");

    const auto log_path = std::filesystem::temp_directory_path() /
                          ("kyty-image-format-diag-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                           ".log");
    FILE* redirected = std::freopen(log_path.string().c_str(), "w", stderr);
    Require(redirected != nullptr, "failed to redirect stderr");

    const auto atomic_plan = MakeStorageImagePlan(Prospero::BufferFormat::k8SInt, true);
    const auto storage_plan = MakeStorageImagePlan(Prospero::BufferFormat::k8SInt, false);
    for (const auto* plan : {&atomic_plan, &storage_plan, &atomic_plan, &storage_plan}) {
        IR::ResourceSnapshot       snapshot;
        IR::ResourceSpecialization spec;
        Require(!IR::MaterializeResources(*plan, runtime, snapshot, spec),
                "unsupported image format unexpectedly materialized");
    }
    Require(std::fclose(redirected) == 0, "failed to flush diagnostic log");
    const std::string log = ReadFile(log_path);
    std::filesystem::remove(log_path);

    for (const auto* kind : {"atomic", "storage"}) {
        const auto message = std::string(kind) +
            " image descriptor 0 uses unsupported format 6: root=0 source=0 "
            "first_use_pc=0x1234 r128=false words=00001000,c0600000,0000000f,90000fac,"
            "00000000,00000000,00000000,00000000";
        const auto first = log.find(message);
        Require(first != std::string::npos, "missing descriptor context");
        Require(log.find(message, first + message.size()) == std::string::npos,
                "repeated descriptor diagnostic");
    }

    std::puts("kyty_shim_image_format_diagnostics: ok");
    return 0;
}

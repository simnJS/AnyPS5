#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <cstdio>
#include <string_view>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::ShaderRecompiler;
using F = Prospero::BufferFormat;

constexpr std::array word_formats {F::k32UInt, F::k32Float, F::k32SInt};
constexpr std::array wide_formats {F::k32_32UInt, F::k32_32Float, F::k32_32SInt, F::k16_16_16_16Float};

ShaderTextureResource AtomicDescriptor(F format) {
    return {{0x1000u, (static_cast<uint32_t>(format) << 20u) | (3u << 30u), 15u, 0xfacu | (9u << 28u)}};
}

IR::ImageResource AtomicResource(bool wide) {
    IR::ImageResource image;
    image.resource_class = IR::ImageResourceClass::Storage;
    image.dimension = Decoder::ImageDimension::Dim2D;
    image.atomic = true;
    image.atomic64 = wide;
    image.read = true;
    image.written = true;
    image.source = 0u;
    return image;
}

void CheckAtomicStorageBinding(F format, bool wide, F narrow = F::kInvalid) {
    auto image = AtomicResource(wide);
    image.numeric_class = Prospero::TextureNumericClass::Uint;
    image.conversion_format = narrow;
    const auto descriptor = AtomicDescriptor(format);
    TileSizeAlign layout;
    TileGetTextureTotalSize(format, descriptor.Width5() + 1u, descriptor.Height5() + 1u,
                            1u, 1u, descriptor.TileMode(), false, layout);
    ValidateStorageTexture(image, descriptor, layout.size);
}

bool CheckAtomicImageMaterialization(Prospero::BufferFormat format, bool wide, bool expected,
                                     Prospero::BufferFormat narrow = Prospero::BufferFormat::kInvalid) {
    IR::ResourcePlan plan;
    plan.stage = ShaderType::Compute;
    plan.resource_tracking_complete = true;
    plan.srt_plan_complete = true;
    const auto descriptor = AtomicDescriptor(format);
    IR::DescriptorSource source;
    source.dword_count = std::size(descriptor.fields);
    for (size_t i = 0; i < source.dword_count; ++i) {
        source.dwords[i] = IR::Value(descriptor.fields[i]);
    }
    plan.descriptor_sources.push_back(std::move(source));
    plan.info.images.push_back(AtomicResource(wide));
    IR::CompileSrtWalker(plan);
    IR::ResourceSnapshot snapshot;
    IR::ResourceSpecialization specialization;
    const bool accepted = IR::MaterializeResources(plan, {}, snapshot, specialization);
    if (accepted == expected && (!accepted || (specialization.images.size() == 1u &&
            specialization.images.front().numeric_class == Prospero::TextureNumericClass::Uint &&
            specialization.images.front().conversion_format == narrow))) {
        return true;
    }
    std::fprintf(stderr, "atomic image format %u, width %u: expected %d, materialized %d\n",
                 static_cast<unsigned>(format), wide ? 64u : 32u, expected, accepted);
    return false;
}

}

extern "C" int RunKytyImageAtomicFormatsTests(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--bindings") {
        for (const auto format : word_formats) CheckAtomicStorageBinding(format, false);
        for (const auto format : wide_formats) CheckAtomicStorageBinding(format, true);
        CheckAtomicStorageBinding(F::k8UInt, false, F::k8UInt);
        return 0;
    }
    if (argc != 1) return 1;
    bool passed = true;
    for (const auto format : word_formats) {
        passed &= CheckAtomicImageMaterialization(format, false, true);
        passed &= CheckAtomicImageMaterialization(format, true, false);
    }
    for (const auto format : wide_formats) {
        passed &= CheckAtomicImageMaterialization(format, true, true);
        passed &= CheckAtomicImageMaterialization(format, false, false);
    }
    passed &= CheckAtomicImageMaterialization(F::k8UInt, false, true, F::k8UInt);
    passed &= CheckAtomicImageMaterialization(F::k8UInt, true, false);
    for (const auto format : {F::k8SInt, F::k8_8_8_8UNorm, F::k32_32_32_32UInt, F::kBc1UNorm,
                              F::kFmask32_S8_F4, F::kFmask64_S16_F8}) {
        passed &= CheckAtomicImageMaterialization(format, false, false);
        passed &= CheckAtomicImageMaterialization(format, true, false);
    }
    return passed ? 0 : 1;
}

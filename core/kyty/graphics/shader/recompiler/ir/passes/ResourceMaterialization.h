#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#if defined(ANYPS5_KYTY)
#include <optional>
#endif

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		Prospero::BufferFormat        srgb_decode                = Prospero::BufferFormat::kInvalid;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		uint32_t                      table                      = ImageResource::NoImageTable;
		uint32_t                      table_capacity             = 0;
		uint32_t                      table_mapping_offset       = 0;
		uint32_t                      table_entry_mask           = 0;
		bool                          cube                       = false;
		bool                          skip_degamma               = false;
		bool                          fmask                      = false;
		bool                          heap                       = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

ResourcePlan ExtractResourcePlan(const Program& program);

struct EncodeFailure {
	enum class Reason : uint8_t { None, ValueType, UnknownInstruction, PhiBlock };

	Reason      reason         = Reason::None;
	uint32_t    type           = 0;
	bool        in_instruction = false;
	ValueOpcode opcode         = ValueOpcode::Void;

	[[nodiscard]] bool Failed() const { return reason != Reason::None; }
};

bool SaveResourcePlan(const ResourcePlan& plan, std::vector<uint8_t>& bytes, EncodeFailure& failure);
bool DecodeResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan);
bool LoadResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan);

void SetComputeIndirectFloors(size_t keys, size_t images);

bool HasIndirectBuffers(const ResourcePlan& program);
bool IndirectBufferMaterializationFailed();
bool IndirectSamplerMaterializationFailed();
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

bool GenericGraphicsShape(const ResourcePlan& plan, ResourceSpecialization& shape);
bool MaterializeGenericGraphics(const ResourcePlan& plan, const SrtRuntime& runtime, ResourceSnapshot& snapshot,
                                ResourceSpecialization& live, const ResourceSpecialization& generic);
bool BindGenericGraphics(const ResourcePlan& plan, ResourceSnapshot& snapshot,
                         const ResourceSpecialization& live, const ResourceSpecialization& generic);

bool ReadsStrideAtRunTime(const ResourcePlan& program, uint32_t index,
                          const ResourceSpecialization::Buffer& buffer);
void ReadStridesAtRunTime(const ResourcePlan& program, ResourceSpecialization& specialization);
uint32_t FormattedReadComponents(const ResourcePlan& program, uint32_t index);
#if defined(ANYPS5_KYTY)
bool SharesEmptyBuffers();
void ShareNullBufferVariants(const ResourcePlan& program, ResourceSpecialization& specialization,
                             bool runtime_stride);
std::optional<uint32_t> EmptyBufferReadBits(const ResourceSpecialization::Buffer& buffer, uint32_t component);
bool SpecializedBuffersValid(const ResourceSpecialization& specialization, bool compute);
bool EmptyBufferServes(const ResourcePlan& program, const ResourceSnapshot& snapshot, uint32_t index,
                       const ResourceSpecialization::Buffer& requested,
                       const ResourceSpecialization::Buffer& candidate);
bool EmptyBuffersServe(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                       const ResourceSpecialization& requested, const ResourceSpecialization& candidate);
bool PopulatedBuffersCoverEmptySpecialization(const ResourcePlan& program, const ResourceSpecialization& empty,
                                              const ResourceSpecialization& populated);
bool SameModuleCode(const ResourcePlan& program, const ResourceSpecialization& requested,
                    const ResourceSpecialization& candidate);
bool GenericComputeVariants();
bool AlignResources(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                    const ResourceSpecialization& specialization, const ResourceSpecialization& target,
                    ResourceSnapshot* aligned);
bool CanRebindComputeImages(const ResourcePlan& program, const ResourceSpecialization& specialization,
                            const ResourceSpecialization& target);
bool RebindComputeImages(const ResourcePlan& program, const SrtRuntime& runtime, const ResourceSnapshot& snapshot,
                         const ResourceSpecialization& specialization, const ResourceSpecialization& target,
                         ResourceSnapshot& aligned);
bool WidenSpecialization(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                         const ResourceSpecialization&                       specialization,
                         std::span<const ResourceSpecialization* const>      known,
                         ResourceSpecialization&                             widened);
#endif
bool SpecializeDirectResources(const ResourcePlan& program, ResourceSnapshot& snapshot,
                               ResourceSpecialization& specialization);

bool ImageTableSlotCompatible(const ImageResource& root, const DescriptorValue& descriptor);

DescriptorValue RuntimeSamplerCandidate(const SamplerResource& sampler);
uint64_t ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization,
                                 uint32_t sampler_limit = UINT32_MAX);

bool ResourceSpecializationFits(const Program& program, const ResourceSpecialization& specialization);

uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
ImageResource ImageTableHeapProfile(const ImageResource& root, bool skip_degamma);
uint32_t ImageTableHeapKind(const ImageResource& profile, DescriptorValue& value);
bool ImageTableHeapExemplar(const ImageResource& profile, const DescriptorValue& value);
ImageResource ImageTableHeapResource(const ImageResource& profile, const DescriptorValue& value);
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);

}

#endif

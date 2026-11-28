#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x200800000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t VertexHeader = GuestBase + 0x100;
constexpr std::uint64_t PixelHeader = GuestBase + 0x300;
constexpr std::uint64_t VertexCode = GuestBase + 0x800;
constexpr std::uint64_t PixelCode = GuestBase + 0x900;
constexpr std::uint64_t Commands = GuestBase + 0x1000;
constexpr std::uint64_t Depth = GuestBase + Block;
constexpr std::uint64_t Color = GuestBase + 6 * Block;
constexpr std::uint64_t SamplerTable = GuestBase + 12 * Block;
constexpr std::uint64_t ImageTable = SamplerTable + 0x100;
constexpr std::uint64_t Records = SamplerTable + 0x200;
constexpr std::uint64_t Textures = GuestBase + 13 * Block;
constexpr std::uint64_t GuestBytes = 14 * Block;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t TextureTexels = 64;
constexpr std::uint32_t RecordBias = 5;
constexpr std::uint32_t SamplerField = 0x80;
constexpr std::uint32_t ImageField = 0x40;
constexpr std::uint32_t RecordBytes = 160;
constexpr std::uint32_t Format8888 = 56;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t SwizzleXyzw = 0xfac;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::string_view BuildLine = "[kyty-variant] build stage=";
constexpr std::string_view PipelineLine = "[kyty-startup] runtime_pipeline";
constexpr std::string_view FirstDraw = "[kyty-startup] first_draw";
constexpr std::string_view RefusedLine = "[kyty-startup] generic_refused";

constexpr std::array<std::uint32_t, 16> VertexWords{
    0x7e000d05, 0x080200f4, 0x10020300, 0x100202f4, 0x060202f3, 0x060400f3, 0x10040500, 0x060404f3,
    0x7e060280, 0x7e0802f2, 0xf80008cf, 0x04030201, EndProgram, EndProgram, EndProgram, EndProgram,
};

constexpr std::array<std::uint32_t, 31> SampledWords{
    0x7e080f02, 0x36080881, 0xbea0047e, 0xbe8e147e, 0xd760000c, 0x00001d04, 0xd4c2001c, 0x0002080c,
    0xbe9e241c, 0x98a30d0c, 0xf4200884, 0x46000080, 0xf42008c4, 0x46000040, 0xbf8cc07f, 0x8f228422,
    0x8f238523, 0xf4280600, 0x44000000, 0xf42c0402, 0x46000000, 0xbf8cc07f, 0xf0800f08, 0x00c40802,
    0xbf8c3f70, 0x89fe1e7e, 0xbf89ffe8, 0xbefe0420, 0xf800180f, 0x0b0a0908, EndProgram,
};

struct Keys {
    std::array<std::uint32_t, 2> samplers;
    std::array<std::uint32_t, 2> images;
};

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

std::uint32_t bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

std::array<std::uint32_t, 4> vsharp(std::uint64_t address, std::uint32_t bytes) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes,
            SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u)};
}

std::array<std::uint32_t, 8> tsharp(std::uint64_t address) {
    const auto base = address >> 8u;
    return {static_cast<std::uint32_t>(base),
            static_cast<std::uint32_t>((base >> 32u) & 0xffu) | (Format8888 << 20u) | (3u << 30u),
            15u, 0xfacu | (9u << 28u), 0u, 0u, 0u, 0u};
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> drawWords() {
    std::vector<std::uint32_t> words;
    const std::pair<std::uint32_t, std::uint32_t> context[]{
        {0x2d5, 0x2000},
        {0x1b3, 0x302}, {0x1b4, 0x302}, {0x1b6, 0}, {0x207, 0}, {0x200, 0x72}, {0x203, 0},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28628},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, ((Width - 1u) << 14u) | (Height - 1u)},
        {0x3b8, 0x906c000}, {0x1e0, 0},
        {0xc, 0}, {0xd, (Height << 16u) | Width},
        {0x81, 0x80000000}, {0x82, (Height << 16u) | Width},
        {0x90, 0x80000000}, {0x91, (Height << 16u) | Width},
        {0x94, 0x80000000}, {0x95, (Height << 16u) | Width},
        {0x10f, bits(Width / 2.0f)}, {0x110, bits(Width / 2.0f)},
        {0x111, bits(-(Height / 2.0f))}, {0x112, bits(Height / 2.0f)},
        {0x113, bits(0.5f)}, {0x114, bits(0.25f)},
        {0xb4, 0}, {0xb5, bits(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
        {0x318, low(Color)}, {0x390, high(Color)}, {0x292, 2},
        {0x000, 0}, {0x002, 0}, {0x007, ((Height - 1u) << 16u) | (Width - 1u)},
        {0x008, 0}, {0x009, bits(1.0f)}, {0x00a, 0}, {0x00b, 0},
        {0x010, 3}, {0x011, 0},
        {0x012, low(Depth)}, {0x014, low(Depth)}, {0x01a, high(Depth)}, {0x01c, high(Depth)},
        {0x013, 0}, {0x015, 0}, {0x01b, 0}, {0x01d, 0},
    };
    const std::pair<std::uint32_t, std::uint32_t> shader[]{
        {0xc8, low(VertexCode)}, {0xc9, high(VertexCode)}, {0x8b, 1u << 1u}, {0x8c, 0},
        {0x008, low(PixelCode)}, {0x009, high(PixelCode)}, {0x00a, 0}, {0x00b, 16u << 1u},
    };
    std::array<std::uint32_t, 16> user{};
    const auto samplers = vsharp(SamplerTable, 2u * 16u);
    const auto images = vsharp(ImageTable, 2u * 32u);
    const auto records = vsharp(Records, RecordBytes);
    std::copy(samplers.begin(), samplers.end(), user.begin());
    std::copy(images.begin(), images.end(), user.begin() + 4);
    std::copy(records.begin(), records.end(), user.begin() + 8);
    user[13] = RecordBias;
    append(words, 0x79, {0x242, 4});
    for (const auto& [offset, value] : context) append(words, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) append(words, 0x76, {offset, value});
    for (std::uint32_t i = 0; i < user.size(); ++i) append(words, 0x76, {0x00c + i, user[i]});
    append(words, 0x2d, {3, 2});
    return words;
}

template <typename T, std::size_t Extent>
void write(std::span<std::byte> memory, std::uint64_t address, std::span<const T, Extent> items) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || items.size_bytes() > memory.size() - offset) {
        throw std::runtime_error("fixture memory write past the guest mapping");
    }
    std::memcpy(memory.data() + offset, items.data(), items.size_bytes());
}

void writeWord(std::span<std::byte> memory, std::uint64_t address, std::uint32_t value) {
    write(memory, address, std::span<const std::uint32_t>(&value, 1));
}

void writeShader(std::span<std::byte> memory, std::uint64_t address, std::uint64_t code, std::uint8_t type,
                 std::span<const std::uint32_t> words) {
    struct Header {
        Shader shader{};
        ShaderUserData user{};
        std::array<ShaderRegister, 4> registers{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(words.size_bytes());
    header.shader.code = reinterpret_cast<const void*>(code);
    header.shader.type = type;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(address + offsetof(Header, user));
    if (type == 1) {
        header.registers = {{{0x008, low(code)}, {0x009, high(code)}, {0x00b, 16u << 1u}, {0x00a, 0}}};
        header.shader.sh_registers = reinterpret_cast<ShaderRegister*>(address + offsetof(Header, registers));
        header.shader.num_sh_registers = static_cast<std::uint8_t>(header.registers.size());
    }
    write(memory, address, std::span<const Header>(&header, 1));
    write(memory, code, words);
}

void makeCapture(const std::filesystem::path& path, const Keys& keys) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    const auto commands = drawWords();
    writeShader(memory, VertexHeader, VertexCode, 2, VertexWords);
    writeShader(memory, PixelHeader, PixelCode, 1, SampledWords);
    write(memory, Commands, std::span(commands));
    std::vector<std::uint32_t> depth((5 * Block) / sizeof(std::uint32_t), bits(0.75f));
    write(memory, Depth, std::span<const std::uint32_t>(depth));
    const std::array<std::array<std::uint32_t, 4>, 2> samplers{{{0x12u, 0x1000u, 0u, 0u}, {0x0u, 0x1000u, 0u, 0u}}};
    for (std::uint32_t i = 0; i < samplers.size(); ++i) {
        write(memory, SamplerTable + i * 16u, std::span<const std::uint32_t>(samplers[i]));
        const auto image = tsharp(Textures + i * 0x100u);
        write(memory, ImageTable + i * 32u, std::span<const std::uint32_t>(image));
        std::vector<std::uint32_t> texels(TextureTexels);
        for (std::uint32_t texel = 0; texel < TextureTexels; ++texel) {
            texels[texel] = 0xff000000u | ((0x9e3779b9u * (texel + 1u + i * 97u)) & 0x00ffffffu);
        }
        write(memory, Textures + i * 0x100u, std::span<const std::uint32_t>(texels));
    }
    for (std::uint32_t record = 0; record < 2u; ++record) {
        const auto base = Records + record * 16u + (RecordBias & ~3u);
        writeWord(memory, base + SamplerField, keys.samplers[record]);
        writeWord(memory, base + ImageField, keys.images[record]);
    }

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    std::uint64_t ordinal = 0;
    const auto emit = [&](AgcDriver::FrameCall call) {
        call.ordinal = ++ordinal;
        call.timeNs = ordinal * 1000u;
        call.thread = 1;
        writer.WriteCall(call);
    };
    emit({.kind = FrameCallKind::RegisterShader, .address = VertexHeader});
    emit({.kind = FrameCallKind::RegisterShader, .address = PixelHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Color,
          .values = {0, 0x8000000022000000ull, Width, Height, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":4,\"ranges\":1,"
                  "\"snapshot_bytes\":" + std::to_string(snapshot) +
                  ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

void setVariable(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

struct Replay {
    bool completed = false;
    std::string log;
    std::string frame;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const std::string& scenario,
              const char* phase, const Keys& keys, bool wait_manifest) {
    const auto directory = root / scenario;
    const auto run = directory / phase;
    std::filesystem::create_directories(directory / "app0" / "sce_sys");
    std::filesystem::create_directories(run);
    {
        std::ofstream param(directory / "app0" / "sce_sys" / "param.json", std::ios::binary | std::ios::trunc);
        param << "{\"titleId\":\"GSEL00001\",\"appVersion\":\"01.00\"}\n";
        if (!param) throw std::runtime_error("could not write synthetic title metadata");
    }
    makeCapture(run / "capture", keys);
    const auto log = run / "replay.log";
    const auto frames = run / "frames";
    const auto command = quote(tool) + " " + quote(run / "capture") + " --strict --no-synthesize --repeat 1 --cold-pipelines" +
                         (wait_manifest ? " --wait-manifest" : "") + " --frames " + quote(frames) + " > " + quote(log) +
                         " 2>&1";
    const auto previous = std::filesystem::current_path();
    std::filesystem::current_path(directory);
    std::fflush(nullptr);
#ifdef _WIN32
    const int status = std::system(('"' + command + '"').c_str());
#else
    const int status = std::system(command.c_str());
#endif
    std::filesystem::current_path(previous);
    Replay result;
    result.log = readFile(log);
    result.frame = readFile(frames / "replay-0.bmp");
    result.completed = status == 0 && !result.frame.empty();
    return result;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 4000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

int selectedSampler(const std::filesystem::path& tool, const std::filesystem::path& root) {
    const std::string name = "sampler";
    const auto require = [&](bool condition, const std::string& message, const Replay& result) {
        if (condition) return;
        std::cerr << "generic graphics selector replay files: " << root << '\n';
        printTail(result.log);
        throw std::runtime_error(name + ": " + message);
    };
    const Keys one{{0u, 0u}, {0u, 0u}};
    const Keys two{{0u, 1u}, {0u, 0u}};

    const auto seed = replay(tool, root, name, "seed", one, false);
    if (!seed.completed) {
        const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        std::cerr << name << ": the seed replay did not complete"
                  << (required ? "" : "; skipping (no usable Vulkan device?)") << '\n';
        printTail(seed.log);
        return required ? 1 : 77;
    }
    const auto manifest = root / name / "_PipelineCache" / "GSEL00001_01.00.kpm";
    require(std::filesystem::is_regular_file(manifest) && std::filesystem::file_size(manifest) > 0,
            "the seed did not save a manifest", seed);

    const auto control = replay(tool, root, name + "-control", "run", two, false);
    require(control.completed && control.frame != seed.frame,
            "the two selected candidates must draw different colors", control);

    const auto served = replay(tool, root, name, "first-use", two, true);
    require(served.completed, "the replay served from the manifest did not complete", served);
    require(served.log.find(RefusedLine) == std::string::npos,
            "the generic form of a stage with a selected descriptor was refused", served);
    require(served.frame == control.frame,
            "the generic module must draw what the exact variant of the new candidates draws", served);
    const auto first_draw = served.log.find(FirstDraw);
    require(first_draw != std::string::npos && served.log.find(BuildLine, first_draw) == std::string::npos &&
                served.log.find(PipelineLine) == std::string::npos,
            "the generic pipeline must draw a new selection without building a stage or a pipeline", served);
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_generic_graphics_selectors_tests <agc_frame_replay>");
        const auto tool = std::filesystem::absolute(argv[1]);
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-generic-graphics-selectors-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        setVariable("KYTY_PIPELINE_MANIFEST", "1");
        setVariable("KYTY_RECORD_PIPELINE_MANIFEST", "1");
        setVariable("KYTY_SHADER_VARIANT_LOG", "1");
        setVariable("KYTY_ASYNC_PIPELINES", "0");
        setVariable("KYTY_GPU_PIPELINE", "0");
        setVariable("KYTY_PIPELINE_STALL_LOG_MS", "0");
        setVariable("ANYPS5_NO_SHADER_CACHE", "1");
        const auto status = selectedSampler(tool, root);
        if (status != 0) return status;
        if (std::getenv("APS5_REPLAY_KEEP_ARTIFACTS") == nullptr) {
            std::filesystem::remove_all(root);
        } else {
            std::cout << "generic graphics selector replay files: " << root << '\n';
        }
        std::cout << "kyty_shim_generic_graphics_selectors: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

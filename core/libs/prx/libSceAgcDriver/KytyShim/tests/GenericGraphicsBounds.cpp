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
constexpr std::uint64_t Records = GuestBase + 12 * Block;
constexpr std::uint64_t GuestBytes = 14 * Block;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
constexpr std::uint32_t RecordBytes = 16;
constexpr std::uint32_t RecordCount = 4;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t SwizzleXyzw = 0xfac;
constexpr std::uint32_t EndProgram = 0xbf810000;
constexpr std::string_view BuildLine = "[kyty-variant] build stage=";
constexpr std::string_view PipelineLine = "[kyty-startup] runtime_pipeline";
constexpr std::string_view FirstDraw = "[kyty-startup] first_draw";

constexpr std::array<std::uint32_t, 16> VertexWords{
    0x7e000d05, 0x080200f4, 0x10020300, 0x100202f4, 0x060202f3, 0x060400f3, 0x10040500, 0x060404f3,
    0x7e060280, 0x7e0802f2, 0xf80008cf, 0x04030201, EndProgram, EndProgram, EndProgram, EndProgram,
};

constexpr std::array<std::uint32_t, 16> PixelWords{
    0x7e020290, 0xe0301000, 0x80000201, 0xbf8c0000, 0xf800180f, 0x02020202, EndProgram, EndProgram,
    EndProgram, EndProgram, EndProgram, EndProgram, EndProgram, EndProgram, EndProgram, EndProgram,
};

constexpr std::array<std::uint32_t, 16> MixedPixelWords{
    0x7e020290, 0xe0301000, 0x80000201, 0xe0301000, 0x80010301, 0xbf8c0000, 0xf800180f, 0x02020302,
    EndProgram, EndProgram, EndProgram, EndProgram, EndProgram, EndProgram, EndProgram, EndProgram,
};

enum class Bounds : std::uint32_t { Structured = 0, ByCount = 1 };

std::uint32_t low(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 8u);
}

std::uint32_t high(std::uint64_t address) {
    return static_cast<std::uint32_t>(address >> 40u);
}

std::uint32_t bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> drawWords(Bounds bounds, bool mixed) {
    std::vector<std::uint32_t> words;
    const std::pair<std::uint32_t, std::uint32_t> context[]{
        {0x2d5, 0x2000},
        {0x1b3, 0x2}, {0x1b4, 0x2}, {0x1b6, 0}, {0x207, 0}, {0x200, 0x72}, {0x203, 0},
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
    user[0] = static_cast<std::uint32_t>(Records);
    user[1] = static_cast<std::uint32_t>((Records >> 32u) & 0xffffu) | (RecordBytes << 16u);
    user[2] = RecordCount;
    user[3] = SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u) | (static_cast<std::uint32_t>(bounds) << 28u);
    if (mixed) {
        user[4] = user[0];
        user[5] = (Records >> 32u) & 0xffffu;
        user[6] = RecordCount;
        user[7] = SwizzleXyzw | (Format32UInt << 12u) | (1u << 24u);
    }
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

void makeCapture(const std::filesystem::path& path, Bounds bounds, bool mixed) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    const auto commands = drawWords(bounds, mixed);
    writeShader(memory, VertexHeader, VertexCode, 2, VertexWords);
    writeShader(memory, PixelHeader, PixelCode, 1, mixed ? MixedPixelWords : PixelWords);
    write(memory, Commands, std::span(commands));
    std::vector<std::uint32_t> depth((5 * Block) / sizeof(std::uint32_t), bits(0.75f));
    write(memory, Depth, std::span<const std::uint32_t>(depth));
    std::array<std::uint32_t, RecordCount * RecordBytes / sizeof(std::uint32_t)> records{};
    records[0] = bits(0.25f);
    records[RecordBytes / sizeof(std::uint32_t)] = bits(1.0f);
    write(memory, Records, std::span<const std::uint32_t>(records));

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

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& root, const char* scenario,
              const char* phase, Bounds bounds, bool wait_manifest, bool mixed = false) {
    const auto directory = root / scenario;
    const auto run = directory / phase;
    std::filesystem::create_directories(directory / "app0" / "sce_sys");
    std::filesystem::create_directories(run);
    {
        std::ofstream param(directory / "app0" / "sce_sys" / "param.json", std::ios::binary | std::ios::trunc);
        param << "{\"titleId\":\"GGFX00001\",\"appVersion\":\"01.00\"}\n";
        if (!param) throw std::runtime_error("could not write synthetic title metadata");
    }
    makeCapture(run / "capture", bounds, mixed);
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

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_generic_graphics_bounds_tests <agc_frame_replay>");
        const auto tool = std::filesystem::absolute(argv[1]);
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-generic-graphics-bounds-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        setVariable("KYTY_PIPELINE_MANIFEST", "1");
        setVariable("KYTY_RECORD_PIPELINE_MANIFEST", "1");
        setVariable("KYTY_SHADER_VARIANT_LOG", "1");
        setVariable("KYTY_ASYNC_PIPELINES", "0");
        setVariable("KYTY_GPU_PIPELINE", "0");
        setVariable("KYTY_PIPELINE_STALL_LOG_MS", "0");
        setVariable("ANYPS5_NO_SHADER_CACHE", "1");
        const auto require = [&](bool condition, const char* message, const Replay& result) {
            if (condition) return;
            std::cerr << "generic graphics replay files: " << root << '\n';
            printTail(result.log);
            throw std::runtime_error(message);
        };

        const auto seed = replay(tool, root, "bounds", "seed", Bounds::ByCount, false);
        if (!seed.completed) {
            const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
            std::cerr << "the seed replay did not complete" << (required ? "" : "; skipping (no usable Vulkan device?)") << '\n';
            printTail(seed.log);
            return required ? 1 : 77;
        }
        const auto manifest = root / "bounds" / "_PipelineCache" / "GGFX00001_01.00.kpm";
        require(std::filesystem::is_regular_file(manifest) && std::filesystem::file_size(manifest) > 0,
                "the seed did not save a manifest", seed);

        const auto control = replay(tool, root, "control", "run", Bounds::Structured, false);
        require(control.completed && control.frame != seed.frame,
                "the two bounds modes must draw different colors for a read past the first record", control);

        const auto served = replay(tool, root, "bounds", "first-use", Bounds::Structured, true);
        require(served.completed && served.frame == control.frame,
                "a generic module prepared from a recipe with other bounds must enforce the live record bounds", served);
        const auto first_draw = served.log.find(FirstDraw);
        require(first_draw != std::string::npos && served.log.find(BuildLine, first_draw) == std::string::npos &&
                    served.log.find(PipelineLine) == std::string::npos,
                "the generic pipeline must draw without building a stage or a pipeline at the first draw", served);

        const auto mixed_seed = replay(tool, root, "mixed", "seed", Bounds::ByCount, false, true);
        require(mixed_seed.completed, "the mixed-stage seed must draw", mixed_seed);
        const auto mixed_control = replay(tool, root, "mixed-control", "run", Bounds::Structured, false, true);
        require(mixed_control.completed && mixed_control.frame != mixed_seed.frame,
                "the mixed-stage control must observe the change in bounds", mixed_control);
        const auto mixed_manifest = root / "mixed" / "_PipelineCache" / manifest.filename();
        auto recipe = readFile(mixed_manifest);
        const auto stride = recipe.find("st:16383");
        require(stride != std::string::npos, "the exact pixel recipe must read the direct buffer bounds at run time", mixed_seed);
        recipe.replace(stride, std::string_view("st:16383").size(), "st:16");
        std::ofstream(mixed_manifest, std::ios::binary | std::ios::trunc) << recipe;
        const auto mixed = replay(tool, root, "mixed", "first-use", Bounds::Structured, true, true);
        const auto mixed_draw = mixed.log.find(FirstDraw);
        require(mixed.completed && mixed.frame == mixed_control.frame && mixed_draw != std::string::npos &&
                    mixed.log.find(BuildLine, mixed_draw) == std::string::npos && mixed.log.find(PipelineLine) == std::string::npos,
                "a legacy exact recipe must serve live bounds when the generic pixel module cannot bind", mixed);

        if (std::getenv("APS5_REPLAY_KEEP_ARTIFACTS") == nullptr) {
            std::filesystem::remove_all(root);
        } else {
            std::cout << "generic graphics replay files: " << root << '\n';
        }
        std::cout << "kyty_shim_generic_graphics_bounds: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

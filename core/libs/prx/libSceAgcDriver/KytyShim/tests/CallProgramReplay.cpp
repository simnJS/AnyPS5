#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x200800000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t Header = GuestBase + 0x100;
constexpr std::uint64_t Code = GuestBase + 0x800;
constexpr std::uint64_t Leaf = GuestBase + 0x1000;
constexpr std::uint64_t Commands = GuestBase + 0x8000;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t GuestBytes = 3 * Block;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Rows = 32;
constexpr std::uint32_t RowBytes = Lanes * 16u;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t One = 0x3f800000u;
constexpr std::uint32_t Two = 0x40000000u;
constexpr std::string_view Stem = "CALL00001_01.00";

constexpr std::uint32_t ShiftLane = 0x34020082u;
constexpr std::uint32_t CallThroughS0 = 0xbe802100u;
constexpr std::array<std::uint32_t, 2> StoreRaw{0xe0701000u, 0x80010201u};
constexpr std::uint32_t EndProgram = 0xbf810000u;
constexpr std::uint32_t MoveOneToV2 = 0x7e0402f2u;
constexpr std::uint32_t MoveTwoToV2 = 0x7e0402f4u;
constexpr std::uint32_t ReturnThroughS0 = 0xbe802000u;

constexpr std::array<std::uint32_t, 5> CallerWords{ShiftLane, CallThroughS0, StoreRaw[0], StoreRaw[1], EndProgram};

std::array<std::uint32_t, 4> output() {
    return {static_cast<std::uint32_t>(Output), static_cast<std::uint32_t>((Output >> 32u) & 0xffffu), RowBytes,
            0xfacu | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u)};
}

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    const auto target = output();
    append(words, 0x76, {0x20c, static_cast<std::uint32_t>(Code >> 8u), static_cast<std::uint32_t>(Code >> 40u)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, 8u << 1u});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    append(words, 0x76,
           {0x240, static_cast<std::uint32_t>(Leaf), static_cast<std::uint32_t>(Leaf >> 32u), 0u, 0u, target[0], target[1],
            target[2], target[3]});
    append(words, 0x15, {1u, 1u, 1u, 0x41u});
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

void makeCapture(const std::filesystem::path& path, std::uint32_t leaf_value) {
    using AgcDriver::FrameCallKind;
    struct ShaderHeader {
        Shader shader{};
        ShaderUserData user{};
    };
    auto memory = std::vector<std::byte>(GuestBytes);
    ShaderHeader header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(ShaderHeader);
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(CallerWords));
    header.shader.code = reinterpret_cast<const void*>(Code);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(Header + offsetof(ShaderHeader, user));
    write(memory, Header, std::span<const ShaderHeader>(&header, 1));
    write(memory, Code, std::span<const std::uint32_t>(CallerWords));
    const std::array<std::uint32_t, 2> leaf{leaf_value, ReturnThroughS0};
    write(memory, Leaf, std::span<const std::uint32_t>(leaf));
    const auto commands = dispatchWords();
    write(memory, Commands, std::span(commands));
    std::vector<std::uint32_t> filled(Width * Rows, Fill);
    write(memory, Output, std::span<const std::uint32_t>(filled));

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
    emit({.kind = FrameCallKind::RegisterShader, .address = Header});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Output,
          .values = {0, 0x8000000022000000ull, Width, Rows, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    emit({.kind = FrameCallKind::Submit, .address = Commands, .size = commands.size()});
    emit({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 2});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":2,\"prologue_calls\":3,\"ranges\":1,"
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
    int status = 0;
    std::string log;
    std::string dump;
};

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& directory, const char* phase,
              std::uint32_t leaf_value, bool wait_manifest) {
    const auto run = directory / phase;
    std::filesystem::create_directories(directory / "app0" / "sce_sys");
    std::filesystem::create_directories(run);
    {
        std::ofstream param(directory / "app0" / "sce_sys" / "param.json", std::ios::binary | std::ios::trunc);
        param << "{\"titleId\":\"CALL00001\",\"appVersion\":\"01.00\"}\n";
        if (!param) throw std::runtime_error("could not write synthetic title metadata");
    }
    makeCapture(run / "capture", leaf_value);
    const auto log = run / "replay.log";
    const auto dump = run / "output.bin";
    const auto command = quote(tool) + " " + quote(run / "capture") + " --strict --no-synthesize --repeat 1 --cold-pipelines" +
                         (wait_manifest ? " --wait-manifest" : "") + " --frames " + quote(run / "frames") + " --dump \"" +
                         std::to_string(Output) + ":" + std::to_string(Width * Rows * 4u) + ":" + dump.string() + "\"" +
                         " > " + quote(log) + " 2>&1";
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
    result.status = status;
    result.log = readFile(log);
    result.dump = readFile(dump);
    result.completed = status == 0 && result.dump.size() == Width * Rows * 4u;
    return result;
}

std::size_t count(const std::string& text, std::string_view line) {
    std::size_t found = 0;
    for (auto at = text.find(line); at != std::string::npos; at = text.find(line, at + line.size())) ++found;
    return found;
}

bool vulkanUnavailable(const Replay& result) {
    if (result.status == 0) return false;
    constexpr std::array<std::string_view, 5> diagnostics{
        "Could not find suitable device:",
        "kyty presentation: the Vulkan loader offers no window surface extension",
        "Unable to find a compatible Vulkan Driver",
        "Could not load Vulkan",
        " is required, but loader supports only "};
    return std::ranges::any_of(diagnostics, [&result](std::string_view diagnostic) {
        return result.log.find(diagnostic) != std::string::npos;
    });
}

bool wrote(const Replay& result, std::uint32_t value) {
    if (result.dump.size() != Width * Rows * 4u) return false;
    for (std::uint32_t word = 0; word < Width * Rows; ++word) {
        std::uint32_t found = 0;
        std::memcpy(&found, result.dump.data() + word * sizeof(found), sizeof(found));
        if (found != (word < Lanes ? value : Fill)) return false;
    }
    return true;
}

struct Startup {
    unsigned modules = 0;
    unsigned pipelines = 0;
    bool clean = false;
};

Startup startup(const std::string& log) {
    Startup result;
    unsigned built = 0, cached = 0, failed = 0;
    const auto modules = log.rfind("Pipeline manifest: startup modules ");
    const auto pipelines = log.rfind("; compute pipelines ");
    if (modules == std::string::npos || pipelines == std::string::npos ||
        std::sscanf(log.c_str() + modules, "Pipeline manifest: startup modules %u built, %u cached, %u failed", &built, &cached,
                    &failed) != 3 || failed != 0) return result;
    result.modules = built + cached;
    if (std::sscanf(log.c_str() + pipelines, "; compute pipelines %u built, %u cached, %u failed", &built, &cached, &failed) != 3 ||
        failed != 0) return result;
    result.pipelines = built + cached;
    result.clean = true;
    return result;
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 6000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_call_program_replay_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-call-program-replay-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        setVariable("KYTY_PIPELINE_MANIFEST", "1");
        setVariable("KYTY_RECORD_PIPELINE_MANIFEST", "1");
        setVariable("KYTY_ASYNC_PIPELINES", "0");
        setVariable("KYTY_GPU_PIPELINE", "0");
        setVariable("KYTY_PIPELINE_STALL_LOG_MS", "0");
        setVariable("ANYPS5_NO_SHADER_CACHE", "1");
        const auto require = [&](bool condition, const char* message, const Replay& result) {
            if (condition) return;
            std::cerr << "call program replay files: " << root << '\n';
            printTail(result.log);
            throw std::runtime_error(message);
        };
        const auto manifest = root / "_PipelineCache" / (std::string(Stem) + ".kpm");
        const auto recipes = [&] { return count(readFile(manifest), " fn="); };

        const auto seed = replay(tool, root, "seed", MoveOneToV2, false);
        if (!seed.completed) {
            std::cerr << "call program replay files: " << root << '\n';
            printTail(seed.log);
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr && vulkanUnavailable(seed)) return 77;
            throw std::runtime_error("the seed replay of the call program did not complete");
        }
        require(wrote(seed, One) && seed.log.find("TEMPORARY COMPUTE DISPATCH SKIP") == std::string::npos,
                "the leaf called by the seed dispatch did not write its value", seed);
        require(count(seed.log, "compute-sync") == 1 && count(seed.log, "[kyty-gpl] translate ") == 1 &&
                    count(seed.log, " cause=unknown-program variant=") +
                            count(seed.log, " cause=generic-not-prepared:code-store-unread variant=") == 2,
                "the first dispatch of a program never seen must wait once, named as an unknown program", seed);
        require(std::filesystem::is_regular_file(manifest) && recipes() == 1 &&
                    readFile(manifest).find(" fn=7e0402f2be802000") != std::string::npos,
                "the seed did not record the function code of the call program", seed);

        const auto warm = replay(tool, root, "warm", MoveOneToV2, true);
        const auto warm_startup = startup(warm.log);
        require(warm.completed && wrote(warm, One) && warm.dump == seed.dump, "the prepared call program wrote other bytes", warm);
        require(warm.log.find("Pipeline manifest: 1 modules, 0 pipelines in ") != std::string::npos && warm_startup.clean &&
                    warm_startup.modules == 1 && warm_startup.pipelines == 1,
                "the startup did not rebuild the call program and its pipeline from the manifest", warm);
        require(warm.log.find("compute-sync") == std::string::npos && warm.log.find("compute-wait") == std::string::npos &&
                    warm.log.find("[kyty-gpl] translate ") == std::string::npos &&
                    warm.log.find("\"dispatches\":1.0") != std::string::npos && recipes() == 1,
                "the prepared call program must be served on first use without translation or pipeline wait", warm);

        const auto changed = replay(tool, root, "changed", MoveTwoToV2, true);
        require(changed.completed && wrote(changed, Two), "a changed leaf must be linked again and run", changed);
        require(count(changed.log, "compute-sync") == 1 &&
                    count(changed.log, " cause=generic-unbound:function-code variant=") == 2 && recipes() == 2 &&
                    readFile(manifest).find(" fn=7e0402f4be802000") != std::string::npos,
                "the changed leaf must wait once, named by its function code, and be recorded", changed);

        const auto both = replay(tool, root, "both", MoveTwoToV2, true);
        const auto both_startup = startup(both.log);
        require(both.completed && wrote(both, Two) && both_startup.clean && both_startup.modules == 2 &&
                    both_startup.pipelines == 2 &&
                    both.log.find("compute-sync") == std::string::npos && both.log.find("compute-wait") == std::string::npos &&
                    recipes() == 2,
                "both recorded leaves must be prepared, and the second one served without a pipeline wait", both);
        const auto first = replay(tool, root, "first", MoveOneToV2, true);
        require(first.completed && wrote(first, One) && first.log.find("compute-sync") == std::string::npos &&
                    first.log.find("compute-wait") == std::string::npos,
                "with both leaves prepared, the first one must still be served without a pipeline wait", first);

        if (std::getenv("APS5_REPLAY_KEEP_ARTIFACTS") == nullptr) {
            std::filesystem::remove_all(root);
        } else {
            std::cout << "call program replay files: " << root << '\n';
        }
        std::cout << "kyty_shim_call_program_replay: ok (1 wait on the unknown program, 0 once prepared, 1 on a changed leaf)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

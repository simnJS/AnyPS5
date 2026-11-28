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
constexpr std::string_view Stem = "FIRST0001_01.00";

constexpr std::array<std::uint32_t, 5> ProgramWords{0x34020082u, 0x7e0402f2u, 0xe0701000u, 0x80010201u, 0xbf810000u};

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
    append(words, 0x76, {0x240, 0u, 0u, 0u, 0u, target[0], target[1], target[2], target[3]});
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

void makeCapture(const std::filesystem::path& path) {
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
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(ProgramWords));
    header.shader.code = reinterpret_cast<const void*>(Code);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(Header + offsetof(ShaderHeader, user));
    write(memory, Header, std::span<const ShaderHeader>(&header, 1));
    write(memory, Code, std::span<const std::uint32_t>(ProgramWords));
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

Replay replay(const std::filesystem::path& tool, const std::filesystem::path& directory, const char* phase, bool wait_manifest) {
    const auto run = directory / phase;
    std::filesystem::create_directories(directory / "app0" / "sce_sys");
    std::filesystem::create_directories(run);
    {
        std::ofstream param(directory / "app0" / "sce_sys" / "param.json", std::ios::binary | std::ios::trunc);
        param << "{\"titleId\":\"FIRST0001\",\"appVersion\":\"01.00\"}\n";
        if (!param) throw std::runtime_error("could not write synthetic title metadata");
    }
    makeCapture(run / "capture");
    const auto log = run / "replay.log";
    const auto dump = run / "output.bin";
    const auto command = quote(tool) + " " + quote(run / "capture") + " --strict --no-synthesize --repeat 1" +
                         (wait_manifest ? " --wait-manifest" : "") + " --dump \"" + std::to_string(Output) + ":" +
                         std::to_string(Width * Rows * 4u) + ":" + dump.string() + "\"" + " > " + quote(log) + " 2>&1";
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
    if (result.log.find("[Vulkan][E]") != std::string::npos || result.log.find("Validation Error:") != std::string::npos) return false;
    if (result.log.find("Could not find suitable device:") != std::string::npos) return true;
    if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) return false;
    constexpr std::array<std::string_view, 4> diagnostics{
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

struct Released {
    long checked = -1;
    long reused = -1;
    long missed = -1;
};

Released released(const std::string& log) {
    Released result;
    const auto line = log.rfind("[kyty-mem] reason=exit ");
    if (line == std::string::npos) return result;
    const auto at = log.find(" released_pipelines=", line);
    if (at == std::string::npos) return result;
    std::sscanf(log.c_str() + at, " released_pipelines=%ld,%ld,%ld", &result.checked, &result.reused, &result.missed);
    return result;
}

long releasedAtStartup(const std::string& log) {
    const auto at = log.rfind(" pipelines found in the cache and released until their first use");
    if (at == std::string::npos) return -1;
    const auto start = log.rfind(": ", at);
    return start == std::string::npos ? -1 : std::strtol(log.c_str() + start + 2, nullptr, 10);
}

void printTail(const std::string& log) {
    constexpr std::size_t TailBytes = 6000;
    std::cerr << (log.size() > TailBytes ? log.substr(log.size() - TailBytes) : log) << '\n';
}

std::vector<std::string> compiledPrograms(const std::string& log) {
    constexpr std::string_view Prefix = "[kyty-pipeline-compile] kind=compute cs=";
    std::vector<std::string> found;
    for (auto at = log.find(Prefix); at != std::string::npos; at = log.find(Prefix, at + Prefix.size())) {
        auto hash = log.substr(at + Prefix.size(), 16);
        if (std::ranges::find(found, hash) == found.end()) found.push_back(std::move(hash));
    }
    return found;
}

void appendSession(const std::filesystem::path& use, const std::vector<std::string>& hashes, bool in_first_minute) {
    std::ofstream stream(use, std::ios::binary | std::ios::app);
    stream << "S\nT\n";
    if (!in_first_minute) stream << "W\n";
    for (const auto& hash : hashes) stream << hash << '\n';
    if (in_first_minute) stream << "W\n";
    if (!stream) throw std::runtime_error("could not write the shader use order");
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_known_pipeline_first_use_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-known-pipeline-first-use-" +
                           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        setVariable("KYTY_PIPELINE_MANIFEST", "1");
        setVariable("KYTY_RECORD_PIPELINE_MANIFEST", "0");
        setVariable("KYTY_ASYNC_PIPELINES", "0");
        setVariable("KYTY_GPU_PIPELINE", "0");
        setVariable("KYTY_PIPELINE_STALL_LOG_MS", "0");
        setVariable("ANYPS5_NO_SHADER_CACHE", "0");
        const auto require = [&](bool condition, const char* message, const Replay& result) {
            if (condition) return;
            std::cerr << "known pipeline first use files: " << root << '\n';
            printTail(result.log);
            throw std::runtime_error(message);
        };
        const auto cache = root / "_PipelineCache";
        const auto use = cache / (std::string(Stem) + ".use");

        const auto seed = replay(tool, root, "seed", false);
        if (!seed.completed) {
            std::cerr << "known pipeline first use files: " << root << '\n';
            printTail(seed.log);
            if (vulkanUnavailable(seed)) {
                std::cerr << "kyty_shim_known_pipeline_first_use: skipped, no usable Vulkan device for replay\n";
                return 77;
            }
            throw std::runtime_error("the seed replay did not complete");
        }
        require(wrote(seed, One), "the seed dispatch did not write its value", seed);
        require(count(seed.log, "[kyty-pipeline-compile] kind=compute") >= 1,
                "the seed must compile the pipeline of the program it has never seen", seed);
        const auto seen = compiledPrograms(seed.log);
        require(std::filesystem::is_regular_file(cache / "FIRST0001.shaders") && seen.size() == 1,
                "the seed did not store the module of its one program", seed);

        appendSession(use, seen, false);
        const auto first = replay(tool, root, "released", true);
        require(first.completed && wrote(first, One) && first.dump == seed.dump,
                "the replay that creates its pipeline at first use wrote other bytes", first);
        require(first.log.find("Shader cache: restored ") != std::string::npos &&
                    first.log.find("Shader cache: restored 0 modules") == std::string::npos,
                "the stored module was not restored", first);
        require(releasedAtStartup(first.log) >= 1,
                "a known pipeline outside the first minute must be checked in the cache and released at startup",
                first);
        const auto first_use = released(first.log);
        require(first_use.checked >= 1 && first_use.reused >= 1 && first_use.missed == 0,
                "the released pipeline must be created again from the cache at its first use", first);
        require(count(first.log, "[kyty-pipeline-compile]") == 0 && first.log.find("compute-sync") == std::string::npos &&
                    first.log.find("released_pipeline_missed") == std::string::npos,
                "the first use of a known pipeline must not compile it", first);

        appendSession(use, seen, true);
        const auto kept = replay(tool, root, "kept", true);
        require(kept.completed && wrote(kept, One) && kept.dump == seed.dump,
                "the replay served by a pipeline created at startup wrote other bytes", kept);
        require(releasedAtStartup(kept.log) == 0,
                "a known pipeline of the first minute must stay created after its startup check", kept);
        const auto kept_use = released(kept.log);
        require(kept_use.checked == 0 && kept_use.reused == 0 && kept_use.missed == 0 &&
                    count(kept.log, "[kyty-pipeline-compile]") == 0 && kept.log.find("compute-sync") == std::string::npos,
                "a pipeline kept from startup must serve its first use without a creation", kept);

        if (std::getenv("APS5_REPLAY_KEEP_ARTIFACTS") == nullptr) {
            std::filesystem::remove_all(root);
        } else {
            std::cout << "known pipeline first use files: " << root << '\n';
        }
        std::cout << "kyty_shim_known_pipeline_first_use: ok (released at startup, created again from the cache at first "
                     "use with no compilation; kept when used in the first minute)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

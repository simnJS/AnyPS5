#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"

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
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x201400000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t ShaderHeader = GuestBase + 0x100;
constexpr std::uint64_t ShaderCode = GuestBase + 0x800;
constexpr std::uint64_t Commands = GuestBase + 0x7000;
constexpr std::uint64_t Table = GuestBase + Block;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t Display = GuestBase + 3 * Block;
constexpr std::uint64_t Target = GuestBase + 4 * Block;
constexpr std::uint64_t GuestBytes = 5 * Block;
constexpr std::uint64_t DumpBytes = 0x1000;
constexpr std::uint64_t Page = 0x4000;
constexpr std::uint64_t Tail = 0x100;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t LaneStride = 1020;
constexpr std::uint32_t LaneBytes = 32;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Wave64Dispatch = 0x41;

constexpr std::array<std::uint32_t, 16> PointerChaseCode{
    0x34020085u, 0x34040083u, 0x7e060200u, 0x7e080201u, 0xd70f6a05u, 0x00020503u, 0x500c0880u, 0xdc348000u,
    0x087d0005u, 0xbf8c0070u, 0xdc308000u, 0x0a7d0008u, 0xbf8c0070u, 0xe0701000u, 0x80010a01u, 0xbf810000u,
};

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    const std::array<std::uint32_t, 4> target{static_cast<std::uint32_t>(Output), static_cast<std::uint32_t>((Output >> 32u) & 0xffffu),
                                              static_cast<std::uint32_t>(Block),
                                              (0x01016facu & ~(0x7fu << 12u)) | (Format32UInt << 12u) | (3u << 28u)};
    append(words, 0x76, {0x20c, static_cast<std::uint32_t>(ShaderCode >> 8u), static_cast<std::uint32_t>(ShaderCode >> 40u)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, (8u << 1u) | (1u << 7u)});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    append(words, 0x76, {0x240, static_cast<std::uint32_t>(Table), static_cast<std::uint32_t>(Table >> 32u), 0u, 0u, target[0], target[1],
                         target[2], target[3]});
    append(words, 0x15, {1u, 1u, 1u, Wave64Dispatch});
    return words;
}

void write(std::span<std::byte> memory, std::uint64_t address, const void* data, std::size_t bytes) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || bytes > memory.size() - offset) throw std::runtime_error("fixture memory write past the guest mapping");
    std::memcpy(memory.data() + offset, data, bytes);
}

std::uint32_t targetWord(std::uint64_t address) {
    std::uint32_t state = static_cast<std::uint32_t>(address - Target) * 2654435761u + 0x51ed270bu;
    state ^= state >> 15u;
    return state == 0 ? 1u : state;
}

void makeCapture(const std::filesystem::path& path) {
    using AgcDriver::FrameCallKind;
    auto memory = std::vector<std::byte>(GuestBytes);
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(PointerChaseCode));
    header.shader.code = reinterpret_cast<const void*>(ShaderCode);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(ShaderHeader + offsetof(Header, user));
    write(memory, ShaderHeader, &header, sizeof(header));
    write(memory, ShaderCode, PointerChaseCode.data(), sizeof(PointerChaseCode));
    const auto words = dispatchWords();
    write(memory, Commands, words.data(), words.size() * sizeof(std::uint32_t));
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const std::uint64_t pointer = Target + std::uint64_t{lane} * LaneStride;
        write(memory, Table + lane * sizeof(pointer), &pointer, sizeof(pointer));
    }
    for (std::uint64_t address = Target; address < Target + Block; address += sizeof(std::uint32_t)) {
        const auto word = targetWord(address);
        write(memory, address, &word, sizeof(word));
    }
    const std::vector<std::uint32_t> fill(Block / sizeof(std::uint32_t), Fill);
    write(memory, Output, fill.data(), Block);

    AgcDriver::FrameCaptureWriter writer(path);
    writer.WriteRange(GuestBase, memory.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    const auto snapshot = writer.WriteSnapshot(GuestBase, memory);
    std::uint64_t ordinal = 0;
    std::uint64_t windowCalls = 0;
    const auto emit = [&](AgcDriver::FrameCall call) {
        call.ordinal = ++ordinal;
        call.timeNs = ordinal * 1000u;
        call.thread = 1;
        writer.WriteCall(call);
    };
    const auto windowed = [&](AgcDriver::FrameCall call) {
        emit(std::move(call));
        ++windowCalls;
    };
    emit({.kind = FrameCallKind::RegisterShader, .address = ShaderHeader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Display, .values = {0, 0x8000000022000000ull, 64, 4, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    windowed({.kind = FrameCallKind::Submit, .address = Commands, .size = words.size()});
    windowed({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 1});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":" + std::to_string(windowCalls) + ",\"prologue_calls\":3,\"ranges\":1,"
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

std::string hex(std::uint32_t value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", value);
    return text;
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_bda_pointer_chase_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-bda-pointer-chase-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        makeCapture(root / "capture");
        char address[32];
        std::snprintf(address, sizeof(address), "0x%llx", static_cast<unsigned long long>(Output));
        const auto log = root / "replay.log";
        const auto dump = root / "output.bin";
        const auto command = quote(tool) + " " + quote(root / "capture") + " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " +
                             quote(root / "frames") + " --dump " + address + ":" + std::to_string(DumpBytes) + ":" + quote(dump) + " > " +
                             quote(log) + " 2>&1";
        std::fflush(nullptr);
#ifdef _WIN32
        const int status = std::system(('"' + command + '"').c_str());
#else
        const int status = std::system(command.c_str());
#endif
        const auto raw = readFile(dump);
        if (status != 0 || raw.size() != DumpBytes) {
            const bool vulkanRequired = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
            std::cerr << "replay did not complete" << (vulkanRequired ? "" : "; skipping (no usable Vulkan device?)") << '\n';
            const auto text = readFile(log);
            std::cerr << text.substr(text.size() > 4000 ? text.size() - 4000 : 0) << '\n';
            return vulkanRequired ? 1 : 77;
        }
        std::vector<std::uint32_t> output(raw.size() / sizeof(std::uint32_t));
        std::memcpy(output.data(), raw.data(), output.size() * sizeof(std::uint32_t));
        int failures = 0;
        std::uint32_t tails = 0;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const std::uint64_t pointer = Target + std::uint64_t{lane} * LaneStride;
            if (pointer % Page > Page - Tail) ++tails;
            const auto want = targetWord(pointer);
            const auto got = output[lane * LaneBytes / sizeof(std::uint32_t)];
            if (got == want) continue;
            if (++failures <= 16) std::cerr << "lane " << lane << ": " << hex(got) << ", expected " << hex(want) << '\n';
        }
        if (tails == 0) throw std::runtime_error("no lane reads the tail of a guest page");
        if (failures != 0) {
            std::cerr << failures << " wrong values: a dynamic address read something other than guest memory on its first access\n";
            const auto text = readFile(log);
            std::cerr << text.substr(text.size() > 4000 ? text.size() - 4000 : 0) << '\n';
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_bda_pointer_chase: ok (" << tails << " lanes in the tail of a guest page)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

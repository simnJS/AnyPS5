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
constexpr std::uint64_t Output = GuestBase + Block;
constexpr std::uint64_t Display = GuestBase + 3 * Block;
constexpr std::uint64_t GuestBytes = 4 * Block;
constexpr std::uint64_t DumpBytes = 0x1000;
constexpr std::uint32_t Records = 0x800;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Wave64Dispatch = 0x41;

constexpr std::array<std::uint32_t, 12> StoreFormsCode{
    0xf4000080u, 0xfa000000u, 0xbf8cc07fu, 0xbe8303ffu, 0x31014facu, 0x4a020081u, 0x340a0082u,
    0xe0701300u, 0x80000105u, 0xe0c81600u, 0x80000105u, 0xbf810000u,
};

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::vector<std::uint32_t> dispatchWords() {
    std::vector<std::uint32_t> words;
    append(words, 0x76, {0x20c, static_cast<std::uint32_t>(ShaderCode >> 8u), static_cast<std::uint32_t>(ShaderCode >> 40u)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, (2u << 1u) | (1u << 7u)});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    append(words, 0x76, {0x240, static_cast<std::uint32_t>(Output), static_cast<std::uint32_t>((Output >> 32u) & 0xffffu)});
    append(words, 0x15, {1u, 1u, 1u, Wave64Dispatch});
    return words;
}

void write(std::span<std::byte> memory, std::uint64_t address, const void* data, std::size_t bytes) {
    const auto offset = address - GuestBase;
    if (offset > memory.size() || bytes > memory.size() - offset) throw std::runtime_error("fixture memory write past the guest mapping");
    std::memcpy(memory.data() + offset, data, bytes);
}

std::vector<std::uint8_t> expected() {
    std::vector<std::uint8_t> bytes(DumpBytes);
    for (std::size_t at = 0; at < bytes.size(); ++at) bytes[at] = static_cast<std::uint8_t>(Fill >> (8u * (at % 4u)));
    for (std::size_t k = 0; k < 4; ++k) bytes[k] = static_cast<std::uint8_t>(Records >> (8u * k));
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const std::uint32_t stored = lane + 1u;
        const std::uint32_t sum = Fill + lane + 1u;
        for (std::size_t k = 0; k < 4; ++k) {
            bytes[0x300 + 4 * lane + k] = static_cast<std::uint8_t>(stored >> (8u * k));
            bytes[0x600 + 4 * lane + k] = static_cast<std::uint8_t>(sum >> (8u * k));
        }
    }
    return bytes;
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
    header.shader.shader_size = static_cast<std::uint32_t>(sizeof(StoreFormsCode));
    header.shader.code = reinterpret_cast<const void*>(ShaderCode);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(ShaderHeader + offsetof(Header, user));
    write(memory, ShaderHeader, &header, sizeof(header));
    write(memory, ShaderCode, StoreFormsCode.data(), sizeof(StoreFormsCode));
    const auto words = dispatchWords();
    write(memory, Commands, words.data(), words.size() * sizeof(std::uint32_t));
    std::vector<std::uint32_t> fill(Block / sizeof(std::uint32_t), Fill);
    fill[0] = Records;
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

}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: kyty_shim_indirect_buffer_stores_tests <agc_frame_replay>");
        const std::filesystem::path tool = argv[1];
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-indirect-buffer-stores-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
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
        const auto want = expected();
        int failures = 0;
        for (std::size_t at = 0; at < want.size(); ++at) {
            const auto got = static_cast<std::uint8_t>(raw[at]);
            if (got == want[at]) continue;
            if (++failures <= 16) {
                char line[96];
                std::snprintf(line, sizeof(line), "byte 0x%03zx: 0x%02x, expected 0x%02x", at, got, want[at]);
                std::cerr << line << '\n';
            }
        }
        if (failures != 0) {
            std::cerr << failures << " wrong bytes\n";
            return 1;
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_indirect_buffer_stores: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

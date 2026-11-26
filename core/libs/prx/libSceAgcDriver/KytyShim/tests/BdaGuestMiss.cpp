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
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::uint64_t GuestBase = 0x201400000;
constexpr std::uint64_t Block = 0x10000;
constexpr std::uint64_t Page = 0x4000;
constexpr std::uint64_t LoadHeader = GuestBase + 0x100;
constexpr std::uint64_t LoadCode = GuestBase + 0x800;
constexpr std::uint64_t StoreHeader = GuestBase + 0x1100;
constexpr std::uint64_t StoreCode = GuestBase + 0x1800;
constexpr std::uint64_t CopyHeader = GuestBase + 0x2100;
constexpr std::uint64_t CopyCode = GuestBase + 0x2800;
constexpr std::uint64_t Commands = GuestBase + 0x7000;
constexpr std::uint64_t Table = GuestBase + Block;
constexpr std::uint64_t Output = GuestBase + 2 * Block;
constexpr std::uint64_t Display = GuestBase + 3 * Block;
constexpr std::uint64_t Target = GuestBase + 4 * Block;
constexpr std::uint64_t Descriptor = GuestBase + 5 * Block;
constexpr std::uint64_t Aside = GuestBase + 6 * Block;
constexpr std::uint64_t Scratch = GuestBase + 7 * Block;
constexpr std::uint64_t AsideCell = Table + 0x100;
constexpr std::uint64_t AsideDescriptor = Descriptor + 0x100;
constexpr std::uint32_t BusyGroups = 65535;
constexpr std::uint32_t BusyRows = 16;
constexpr std::uint64_t Lone = GuestBase + 8 * Block;
constexpr std::uint64_t SliceLast = 0x240000000 - Page;
constexpr std::uint64_t DumpBytes = 0x1000;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t LaneBytes = 32;
constexpr std::uint32_t StoreStride = 1024;
constexpr std::uint32_t StoreBias = 61;
constexpr std::uint32_t Fill = 0xdeadbeefu;
constexpr std::uint32_t Format32UInt = 20;
constexpr std::uint32_t Wave64Dispatch = 0x41;

constexpr std::array<std::uint32_t, 16> LoadThroughPointer{
    0x34020085u, 0x34040083u, 0x7e060200u, 0x7e080201u, 0xd70f6a05u, 0x00020503u, 0x500c0880u, 0xdc348000u,
    0x087d0005u, 0xbf8c0070u, 0xdc308000u, 0x0a7d0008u, 0xbf8c0070u, 0xe0701000u, 0x80010a01u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 16> StoreThroughDescriptor{
    0x34040083u, 0x7e060200u, 0x7e080201u, 0xdc348000u, 0x087d0003u, 0xbf8c0070u, 0x7e040508u, 0x7e060509u,
    0xf4080201u, 0xfa000000u, 0xbf8c0070u, 0x4a1404bdu, 0x3404008au, 0xe0701000u, 0x80020a02u, 0xbf810000u,
};

constexpr std::array<std::uint32_t, 8> CopyThroughBuffers{
    0x34020085u, 0x3404008au, 0xe0301000u, 0x80020a02u, 0xbf8c0070u, 0xe0701000u, 0x80010a01u, 0xbf810000u,
};

struct Range {
    std::uint64_t address = 0;
    std::vector<std::byte> bytes;
};

struct Memory {
    std::vector<Range> ranges;

    void write(std::uint64_t address, const void* data, std::size_t bytes) {
        for (auto& range : ranges) {
            if (address < range.address || address - range.address > range.bytes.size() || bytes > range.bytes.size() - (address - range.address)) continue;
            std::memcpy(range.bytes.data() + (address - range.address), data, bytes);
            return;
        }
        throw std::runtime_error("fixture memory write outside the guest mappings");
    }
};

void append(std::vector<std::uint32_t>& words, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    words.insert(words.end(), payload);
}

std::array<std::uint32_t, 4> bufferWords(std::uint64_t address) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), static_cast<std::uint32_t>(Block),
            (0x01016facu & ~(0x7fu << 12u)) | (Format32UInt << 12u) | (3u << 28u)};
}

void appendDispatch(std::vector<std::uint32_t>& words, std::uint64_t code, std::uint32_t registers, std::initializer_list<std::uint32_t> user,
                    std::uint32_t groups = 1, std::uint32_t rows = 1) {
    append(words, 0x76, {0x20c, static_cast<std::uint32_t>(code >> 8u), static_cast<std::uint32_t>(code >> 40u)});
    append(words, 0x76, {0x212, 0u});
    append(words, 0x76, {0x213, (registers << 1u) | (1u << 7u)});
    append(words, 0x76, {0x207, Lanes, 1u, 1u});
    std::vector<std::uint32_t> data{0x240};
    data.insert(data.end(), user);
    words.push_back(0xc0000000u | (static_cast<std::uint32_t>(data.size() - 1) << 16u) | (0x76u << 8u));
    words.insert(words.end(), data.begin(), data.end());
    append(words, 0x15, {groups, rows, 1u, Wave64Dispatch});
}

void writeShader(Memory& memory, std::uint64_t header_address, std::uint64_t code_address, std::span<const std::uint32_t> code) {
    struct Header {
        Shader shader{};
        ShaderUserData user{};
    };
    Header header{};
    header.shader.file_header = 0x34333231;
    header.shader.version = 0x18;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
    header.shader.code = reinterpret_cast<const void*>(code_address);
    header.shader.type = 0;
    header.shader.user_data = reinterpret_cast<ShaderUserData*>(header_address + offsetof(Header, user));
    memory.write(header_address, &header, sizeof(header));
    memory.write(code_address, code.data(), code.size_bytes());
}

std::uint32_t guestWord(std::uint64_t address) {
    std::uint32_t state = static_cast<std::uint32_t>(address) * 2654435761u + 0x51ed270bu;
    state ^= state >> 15u;
    return state < 0x10000u ? state + 0x10000u : state;
}

void fillWords(Memory& memory, std::uint64_t address, std::uint64_t bytes) {
    for (std::uint64_t at = address; at < address + bytes; at += sizeof(std::uint32_t)) {
        const auto word = guestWord(at);
        memory.write(at, &word, sizeof(word));
    }
}

std::uint64_t lonePointer(std::uint32_t lane) {
    switch (lane) {
        case 0: return Lone + Page - 4;
        case 1: return Lone;
        case 2: return Lone + Page - 8;
        case 3: return SliceLast + Page - 4;
        case 4: return SliceLast + Page - 0x100;
        case 5: return SliceLast;
        default: return Lone + std::uint64_t{lane} * 252;
    }
}

void writeCapture(const std::filesystem::path& path, const Memory& memory, std::span<const std::uint64_t> shaders, std::size_t words) {
    using AgcDriver::FrameCallKind;
    AgcDriver::FrameCaptureWriter writer(path);
    for (const auto& range : memory.ranges) writer.WriteRange(range.address, range.bytes.size(), AgcDriver::FrameRangeKind::Watched, 0, 7);
    std::uint64_t snapshot = 0;
    for (const auto& range : memory.ranges) snapshot += writer.WriteSnapshot(range.address, range.bytes);
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
    for (const auto shader : shaders) emit({.kind = FrameCallKind::RegisterShader, .address = shader});
    emit({.kind = FrameCallKind::RegisterVideoOutput, .values = {7}});
    emit({.kind = FrameCallKind::DisplayBuffer, .address = Display, .values = {0, 0x8000000022000000ull, 64, 4, 0, 0, 0, 0, 0}});
    emit({.kind = FrameCallKind::FrameStart, .values = {1}});
    windowed({.kind = FrameCallKind::Submit, .address = Commands, .size = words});
    windowed({.kind = FrameCallKind::Flip, .accepted = 1, .cpuFlip = true, .values = {7, UINT64_MAX, 1, 0}});
    emit({.kind = FrameCallKind::End, .accepted = 1});
    writer.Finish("{\"magic\":\"aps5-frame-capture 1\",\"arena_base\":65536,\"arena_bytes\":1082331693056,"
                  "\"start_boundary\":1,\"frames\":1,\"calls\":" + std::to_string(windowCalls) + ",\"prologue_calls\":" +
                  std::to_string(shaders.size() + 2) + ",\"ranges\":" + std::to_string(memory.ranges.size()) + ",\"snapshot_bytes\":" +
                  std::to_string(snapshot) + ",\"unreadable_bytes\":0,\"host_only_sample_bytes\":0,\"warnings\":[]}\n");
}

void makeLastWord(const std::filesystem::path& path) {
    Memory memory;
    memory.ranges.push_back({GuestBase, std::vector<std::byte>(4 * Block)});
    memory.ranges.push_back({Lone, std::vector<std::byte>(Page)});
    memory.ranges.push_back({SliceLast, std::vector<std::byte>(Page)});
    writeShader(memory, LoadHeader, LoadCode, LoadThroughPointer);
    std::vector<std::uint32_t> words;
    const auto output = bufferWords(Output);
    appendDispatch(words, LoadCode, 8, {static_cast<std::uint32_t>(Table), static_cast<std::uint32_t>(Table >> 32u), 0u, 0u, output[0], output[1], output[2], output[3]});
    memory.write(Commands, words.data(), words.size() * sizeof(std::uint32_t));
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        const auto pointer = lonePointer(lane);
        memory.write(Table + lane * sizeof(pointer), &pointer, sizeof(pointer));
    }
    fillWords(memory, Lone, Page);
    fillWords(memory, SliceLast, Page);
    const std::vector<std::uint32_t> fill(Block / sizeof(std::uint32_t), Fill);
    memory.write(Output, fill.data(), Block);
    const std::array shaders{LoadHeader};
    writeCapture(path, memory, shaders, words.size());
}

void makeStoreThenRead(const std::filesystem::path& path) {
    Memory memory;
    memory.ranges.push_back({GuestBase, std::vector<std::byte>(8 * Block)});
    writeShader(memory, StoreHeader, StoreCode, StoreThroughDescriptor);
    writeShader(memory, CopyHeader, CopyCode, CopyThroughBuffers);
    std::vector<std::uint32_t> words;
    const auto output = bufferWords(Output);
    const auto target = bufferWords(Target);
    const auto aside = bufferWords(Aside);
    const auto scratch = bufferWords(Scratch);
    const auto store = [&](std::uint64_t cell, std::uint32_t groups, std::uint32_t rows) {
        appendDispatch(words, StoreCode, 8, {static_cast<std::uint32_t>(cell), static_cast<std::uint32_t>(cell >> 32u), 0u, 0u, 0u, 0u, 0u, 0u}, groups, rows);
    };
    const auto copy = [&](const std::array<std::uint32_t, 4>& to, const std::array<std::uint32_t, 4>& from) {
        appendDispatch(words, CopyCode, 12, {0u, 0u, 0u, 0u, to[0], to[1], to[2], to[3], from[0], from[1], from[2], from[3]});
    };
    store(AsideCell, 1, 1);
    copy(scratch, aside);
    store(AsideCell, BusyGroups, BusyRows);
    store(Table, 1, 1);
    copy(output, target);
    memory.write(Commands, words.data(), words.size() * sizeof(std::uint32_t));
    memory.write(Table, &Descriptor, sizeof(Descriptor));
    memory.write(AsideCell, &AsideDescriptor, sizeof(AsideDescriptor));
    memory.write(Descriptor, target.data(), sizeof(target));
    memory.write(AsideDescriptor, aside.data(), sizeof(aside));
    fillWords(memory, Target, Block);
    fillWords(memory, Aside, Block);
    const std::vector<std::uint32_t> fill(Block / sizeof(std::uint32_t), Fill);
    memory.write(Output, fill.data(), Block);
    memory.write(Scratch, fill.data(), Block);
    const std::array shaders{StoreHeader, CopyHeader};
    writeCapture(path, memory, shaders, words.size());
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string quote(const std::filesystem::path& path) {
    return "\"" + path.string() + "\"";
}

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

std::vector<std::uint32_t> wordsOf(const std::string& raw) {
    std::vector<std::uint32_t> words(raw.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), raw.data(), words.size() * sizeof(std::uint32_t));
    return words;
}

}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: kyty_shim_bda_guest_miss_tests <agc_frame_replay> <last_word|store_then_read>");
        const std::filesystem::path tool = argv[1];
        const std::string_view scenario = argv[2];
        const bool stores = scenario == "store_then_read";
        if (!stores && scenario != "last_word") throw std::runtime_error("unknown scenario");
        const auto root = std::filesystem::temp_directory_path() /
                          ("kyty-bda-guest-miss-" + std::string(scenario) + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        if (stores) makeStoreThenRead(root / "capture");
        else makeLastWord(root / "capture");
        const auto log = root / "replay.log";
        const auto dump = root / "output.bin";
        const auto guest = root / "target.bin";
        auto command = quote(tool) + " " + quote(root / "capture") + " --strict --no-synthesize --repeat 1 --cold-pipelines --frames " +
                       quote(root / "frames") + " --dump " + hex(Output) + ":" + std::to_string(DumpBytes) + ":" + quote(dump);
        if (stores) command += " --dump " + hex(Target) + ":" + std::to_string(Block) + ":" + quote(guest);
        command += " > " + quote(log) + " 2>&1";
        std::fflush(nullptr);
#ifdef _WIN32
        const int status = std::system(('"' + command + '"').c_str());
#else
        const int status = std::system(command.c_str());
#endif
        const auto raw = readFile(dump);
        const auto text = readFile(log);
        const auto tail = text.substr(text.size() > 4000 ? text.size() - 4000 : 0);
        const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
        if (status != 0 || raw.size() != DumpBytes) {
            std::cerr << "replay did not complete" << (required ? "" : "; skipping (no usable Vulkan device?)") << '\n' << tail << '\n';
            return required ? 1 : 77;
        }
        if (text.find("[kyty-host-view] guest memory is not imported") != std::string::npos ||
            text.find("[kyty-host-view] guest memory 0x") != std::string::npos) {
            std::cerr << "the device does not import the guest memory of the fixture" << (required ? "" : "; skipping") << '\n' << tail << '\n';
            return required ? 1 : 77;
        }
        const auto output = wordsOf(raw);
        int failures = 0;
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto want = stores ? lane * 8u + StoreBias : guestWord(lonePointer(lane));
            const auto got = output[lane * LaneBytes / sizeof(std::uint32_t)];
            if (got == want) continue;
            if (++failures <= 16) std::cerr << "lane " << lane << ": " << hex(got) << ", expected " << hex(want) << '\n';
        }
        if (failures != 0) {
            std::cerr << failures << " wrong values: "
                      << (stores ? "a range bound behind a store by device address was read without that store\n"
                                 : "a word at the end of a lone guest page did not come from guest memory\n")
                      << tail << '\n';
            return 1;
        }
        if (stores) {
            const auto target = wordsOf(readFile(guest));
            if (target.size() != Block / sizeof(std::uint32_t)) throw std::runtime_error("the guest memory of the stores was not dumped");
            for (std::uint64_t index = 0; index < target.size(); ++index) {
                const auto offset = index * sizeof(std::uint32_t);
                const auto lane = static_cast<std::uint32_t>(offset / StoreStride);
                const auto want = offset % StoreStride == 0 ? lane * 8u + StoreBias : guestWord(Target + offset);
                if (target[index] == want) continue;
                if (++failures <= 16) std::cerr << "guest word " << hex(Target + offset) << ": " << hex(target[index]) << ", expected " << hex(want) << '\n';
            }
            if (failures != 0) {
                std::cerr << failures << " wrong guest words after a store by device address into a block that was not bound\n" << tail << '\n';
                return 1;
            }
        }
        std::filesystem::remove_all(root);
        std::cout << "kyty_shim_bda_guest_miss_" << scenario << ": ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

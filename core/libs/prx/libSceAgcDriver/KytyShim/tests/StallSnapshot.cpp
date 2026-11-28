#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/stallSnapshot.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

namespace Snapshot = Libs::Graphics::StallSnapshot;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "kyty_shim_stall_snapshot: FAILED %s\n", what);
	g_failures++;
}

std::string Read(const std::string& path) {
	std::ifstream      in(path);
	std::ostringstream text;
	text << in.rdbuf();
	return text.str();
}

void Section(void* owner, std::FILE* file) {
	std::fprintf(file, "section of %s\n", static_cast<const char*>(owner));
}

}

int main() {
	const auto directory = std::filesystem::temp_directory_path() / ("kyty-stall-snapshot-" + std::to_string(getpid()));
	std::filesystem::remove_all(directory);
	setenv("KYTY_HANG_SNAPSHOT_DIR", directory.c_str(), 1);
	Check(Snapshot::Enabled(), "the snapshot is on when its directory is set");

	const auto timeline = vk::Semaphore(reinterpret_cast<VkSemaphore>(uintptr_t {0x1000}));
	const auto binary   = vk::Semaphore(reinterpret_cast<VkSemaphore>(uintptr_t {0x2000}));
	const auto unknown  = vk::Semaphore(reinterpret_cast<VkSemaphore>(uintptr_t {0x3000}));
	Snapshot::Name(timeline, "the test binds", true);
	Snapshot::Name(binary, "the test acquisition", false);
	Check(Snapshot::Describe(timeline) == "the test binds", "a named semaphore is described by its name");
	Check(Snapshot::Timeline(timeline) && !Snapshot::Timeline(binary), "the kind of a named semaphore is kept");
	Check(Snapshot::Describe(unknown) == "semaphore 0x3000", "an unnamed semaphore is described by its handle");
	Check(!Snapshot::Timeline(unknown), "an unnamed semaphore is not read as a timeline");
	Snapshot::Name(timeline, "the renamed binds", true);
	Check(Snapshot::Describe(timeline) == "the renamed binds", "naming a semaphore again replaces its name");
	Snapshot::Forget(binary);
	Check(Snapshot::Describe(binary) == "semaphore 0x2000", "a forgotten semaphore loses its name");

	static const char owner[] = "the test scheduler";
	Snapshot::Register(const_cast<char*>(owner), "test section", &Section);

	Snapshot::NoteOperation(30, "draw", 0, 6);
	Snapshot::NoteOperation(40, "draw", 0, 7);
	Snapshot::NoteOperation(41, "dispatch", 0, 8);
	Snapshot::NoteOperation(42, "draw", 1, 9);
	Snapshot::NoteOperation(42, "draw", 1, 9);
	Snapshot::NoteOperation(42, "callback", 1, 10);

	int pipe_ends[2] = {-1, -1};
	Check(pipe(pipe_ends) == 0, "a pipe for a blocked thread");
	std::atomic<bool> reading {false};
	std::thread       reader([&] {
        reading.store(true);
        char byte = 0;
        (void)read(pipe_ends[0], &byte, 1);
	});
	while (!reading.load()) std::this_thread::yield();
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	Libs::Graphics::GraphicContext graphics;
	const auto path = Snapshot::Write(graphics, 42, 41, "[kyty] test verdict");
	Check(!path.empty() && std::filesystem::exists(path), "the snapshot is written into its directory");
	Check(std::filesystem::path(path).parent_path() == directory, "the snapshot lands in KYTY_HANG_SNAPSHOT_DIR");
	const auto text = Read(path);
	Check(text.starts_with("[kyty] test verdict\n"), "the snapshot starts with the verdict");
	Check(text.find("tick 42 awaited, the GPU is at 41") != std::string::npos, "the snapshot names the awaited tick");
	Check(text.find("== test section\nsection of the test scheduler\n") != std::string::npos, "a registered section is written");
	Check(text.find("tick 42 (unfinished): submissions 9 to 10") != std::string::npos, "the operations of the unfinished tick are grouped");
	Check(text.find("callback 1 draw 2") != std::string::npos, "the operations of a tick are counted by kind");
	Check(text.find("tick 41 (finished)") != std::string::npos, "the last finished tick is kept");
	Check(text.find("tick 30 (") == std::string::npos, "ticks long finished are left out of the grouping");
	Check(text.find("the driver does not report the heap budgets") != std::string::npos, "the heaps are reported");
	Check(text.find("== threads (") != std::string::npos, "the threads are listed");
	Check(text.find("read on pipe:") != std::string::npos, "a thread blocked in read names the pipe it reads");
	Check(text.find("writing this report") != std::string::npos, "the thread that writes the report is named as such");

	Check(write(pipe_ends[1], "x", 1) == 1, "the blocked thread is released");
	reader.join();
	close(pipe_ends[0]);
	close(pipe_ends[1]);

	Snapshot::Unregister(const_cast<char*>(owner));
	const auto again = Snapshot::Write(graphics, 43, 42, "[kyty] second verdict");
	Check(!again.empty() && again != path, "a second stall gets its own file");
	Check(Read(again).find("== test section") == std::string::npos, "an unregistered section is not written");

	std::filesystem::remove_all(directory);
	if (g_failures != 0) return 1;
	std::printf("kyty_shim_stall_snapshot: ok\n");
	return 0;
}

#include <string_view>

extern "C" int RunKytyVramBudgetTest(bool guest_eviction, bool wait_for_minute);
extern "C" int RunKytyTextureSpillContentTest();

int main(int argc, char** argv) {
	const std::string_view mode = argc == 2 ? argv[1] : "";
	if (mode == "spill-content") return RunKytyTextureSpillContentTest();
	return RunKytyVramBudgetTest(mode == "guest-eviction", mode == "--minute");
}

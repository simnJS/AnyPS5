#include <string_view>

extern "C" int RunKytyVramBudgetTest(bool guest_eviction, bool wait_for_minute);

int main(int argc, char** argv) {
	const std::string_view mode = argc == 2 ? argv[1] : "";
	return RunKytyVramBudgetTest(mode == "guest-eviction", mode == "--minute");
}

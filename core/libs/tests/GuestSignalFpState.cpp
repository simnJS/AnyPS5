#include "prx/libkernel/include/Api.hpp"
#include "prx/libc/include/Api.hpp"
#include "prx/libSceLibcInternal/include/Api.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static constexpr int GuestSigterm = 15;
static constexpr int GuestSigusr1 = 30;
static constexpr std::uint32_t ControlBits = 0xffc0u;
static constexpr std::uint32_t X87Bits = 0x0f3fu;
static constexpr std::uint32_t GuestMxcsr = 0x9fc0u;
static constexpr std::uint32_t GuestX87 = 0x037fu;

static void Require(bool value, const char* what) {
    if (!value) {
        std::fprintf(stderr, "guest_signal_fp_state: %s\n", what);
        std::abort();
    }
}

static std::uint32_t ReadMxcsr() {
    std::uint32_t value = 0;
    asm volatile("stmxcsr %0" : "=m"(value) : : "memory");
    return value;
}

static std::uint32_t ReadX87() {
    std::uint16_t value = 0;
    asm volatile("fnstcw %0" : "=m"(value) : : "memory");
    return value;
}

static void Write(std::uint32_t mxcsr, std::uint16_t x87) {
    asm volatile("ldmxcsr %0" : : "m"(mxcsr) : "memory");
    asm volatile("fldcw %0" : : "m"(x87) : "memory");
}

static std::atomic<int> calls{0};
static std::atomic<std::uint32_t> seenMxcsr{0};
static std::atomic<std::uint32_t> seenX87{0};

static void Record() {
    seenMxcsr.store(ReadMxcsr());
    seenX87.store(ReadX87());
    calls.fetch_add(1);
}

static void APS5_VABI SignalHandler(int) { Record(); }
static void APS5_VABI ExceptionHandler(int, void*) { Record(); }

static void CheckObserved(const char* what, std::uint32_t outerMxcsr, std::uint32_t outerX87) {
    Require(calls.load() == 1, what);
    Require((seenMxcsr.load() & ControlBits) == GuestMxcsr, "the handler did not start with MXCSR flush-to-zero and denormals-are-zero");
    Require((seenX87.load() & X87Bits) == (GuestX87 & X87Bits), "the handler did not start with the x87 control word 0x37f");
    Require((ReadMxcsr() & ControlBits) == (outerMxcsr & ControlBits), "MXCSR was not restored after the handler");
    Require((ReadX87() & X87Bits) == (outerX87 & X87Bits), "the x87 control word was not restored after the handler");
    calls.store(0);
}

int main() {
    constexpr std::uint32_t outerMxcsr = 0x1f80u;
    constexpr std::uint16_t outerX87 = 0x027fu;

    Require(signal_nid_postfix(GuestSigterm, SignalHandler) != reinterpret_cast<void (APS5_VABI*)(int)>(static_cast<std::uintptr_t>(-1)), "signal install");
    Write(outerMxcsr, outerX87);
    Require(raise_nid_postfix(GuestSigterm) == 0, "raise");
    CheckObserved("the signal handler was not called once", outerMxcsr, outerX87);

    Write(0x1f80u, 0x037f);
    Require(sceKernelInstallExceptionHandler(GuestSigusr1, reinterpret_cast<void*>(&ExceptionHandler)) == 0, "exception handler install");
    Write(outerMxcsr, outerX87);
    Require(sceKernelRaiseException(scePthreadSelf(), GuestSigusr1) == 0, "raise exception");
    CheckObserved("the exception handler was not called once", outerMxcsr, outerX87);

    Write(GuestMxcsr, static_cast<std::uint16_t>(GuestX87));
    std::puts("guest_signal_fp_state: ok");
    return 0;
}

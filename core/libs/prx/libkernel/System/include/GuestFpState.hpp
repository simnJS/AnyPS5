#ifndef CORE_LIBS_PRX_LIBKERNEL_SYSTEM_INCLUDE_GUESTFPSTATE_HPP
#define CORE_LIBS_PRX_LIBKERNEL_SYSTEM_INCLUDE_GUESTFPSTATE_HPP

#include <cstdint>

namespace GuestFpState {

inline constexpr std::uint32_t Mxcsr = 0x9fc0;
inline constexpr std::uint16_t X87Control = 0x037f;

class Scope {
public:
    Scope() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
        std::uint32_t guestMxcsr = Mxcsr;
        std::uint16_t guestControl = X87Control;
        asm volatile("stmxcsr %0" : "=m"(savedMxcsr) : : "memory");
        asm volatile("fnstcw %0" : "=m"(savedControl) : : "memory");
        asm volatile("ldmxcsr %0" : : "m"(guestMxcsr) : "memory");
        asm volatile("fldcw %0" : : "m"(guestControl) : "memory");
#endif
    }
    ~Scope() {
#if defined(__x86_64__) || defined(_M_X64)
        asm volatile("ldmxcsr %0" : : "m"(savedMxcsr) : "memory");
        asm volatile("fldcw %0" : : "m"(savedControl) : "memory");
#endif
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    std::uint32_t savedMxcsr = 0;
    std::uint16_t savedControl = 0;
};

}

#endif

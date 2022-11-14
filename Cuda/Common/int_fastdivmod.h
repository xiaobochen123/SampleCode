#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

template <typename T>
struct QuotientMod {
    T quotient;
    T mod;
    __host__ __device__ QuotientMod(T q, T m) : quotient(q), mod(m) {}
};

template <typename T>
struct FastIntDivider {
    FastIntDivider() {}
    FastIntDivider(T d) { divisor_ = d; };
    __forceinline__ __device__ __host__ T div(T n) { return n / divisor_; }
    __forceinline__ __device__ __host__ T mod(T n) { return n % divisor_; }
    __forceinline__ __device__ __host__ QuotientMod<T> divmod(T n) {
        return QuotientMod<T>(n / divisor_, n % divisor_);
    }
    T divisor_;
};

template <>
struct FastIntDivider<uint32_t> {
    FastIntDivider(){};

    FastIntDivider(uint32_t d) {
        assert(d >= 1);
        divisor_ = d;
        // if put 0 to __builtin_clz, the result undefined.
        if (d == 1) {
            rshift_ = 0;
        } else {
            rshift_ = 32 - __builtin_clz(d - 1);
        }
        uint64_t magic_t = ((1lu << (32 + rshift_)) + d - 1) / d;
        magic_ = uint32_t(magic_t);
    };

    __forceinline__ __device__ __host__ uint32_t div(uint32_t n) {
#if defined(__CUDA_ARCH__)
        uint32_t q = __umulhi(n, magic_);
#else
        uint32_t q = (uint64_t(n) * magic_) >> 32;
#endif
        // return (((n - q) >> 1) + q) >> (rshift_ - 1);
        return (n + q) >> rshift_;
    }

    __forceinline__ __device__ __host__ QuotientMod<uint32_t> divmod(
        uint32_t n) {
        uint32_t q = div(n);
        return QuotientMod<uint32_t>(q, n - divisor_ * q);
    }

    uint32_t magic_;
    uint32_t rshift_;
    uint32_t divisor_;
};

void test_fast_u32() {
    uint32_t d = 1;

    FastIntDivider<uint32_t> diver(d);
    std::cout << "7/3= " << uint32_t(7) / uint32_t(d) << " " << diver.div(7)
              << std::endl;
}
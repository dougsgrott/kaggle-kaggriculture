#include "pyrandom.hpp"

namespace kaggriculture::sim {

void PyRandom::init_genrand(uint32_t s) {
    mt_[0] = s;
    for (int mti = 1; mti < N; mti++) {
        mt_[mti] = 1812433253U * (mt_[mti - 1] ^ (mt_[mti - 1] >> 30)) + static_cast<uint32_t>(mti);
    }
    index_ = N;
}

void PyRandom::init_by_array(const uint32_t* key, int key_length) {
    init_genrand(19650218U);
    int i = 1, j = 0;
    int k = N > key_length ? N : key_length;
    for (; k; k--) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) + key[j] + static_cast<uint32_t>(j);
        i++;
        j++;
        if (i >= N) {
            mt_[0] = mt_[N - 1];
            i = 1;
        }
        if (j >= key_length) j = 0;
    }
    for (k = N - 1; k; k--) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) - static_cast<uint32_t>(i);
        i++;
        if (i >= N) {
            mt_[0] = mt_[N - 1];
            i = 1;
        }
    }
    mt_[0] = 0x80000000U;
}

void PyRandom::seed(uint64_t seed) {
    // CPython decomposes the seed's absolute value into little-endian 32-bit words, at least
    // one word even for zero. `seed` here is always non-negative (an episode seed, or
    // `(episode_seed * 1_000_003) ^ day` with both operands non-negative -- see sim.hpp), so
    // there is no sign to take the absolute value of.
    uint32_t lo = static_cast<uint32_t>(seed & 0xffffffffU);
    uint32_t hi = static_cast<uint32_t>(seed >> 32);
    if (hi == 0) {
        uint32_t key[1] = {lo};
        init_by_array(key, 1);
    } else {
        uint32_t key[2] = {lo, hi};
        init_by_array(key, 2);
    }
}

uint32_t PyRandom::next_u32() {
    static const uint32_t mag01[2] = {0x0U, MATRIX_A};

    if (index_ >= N) {
        int kk;
        for (kk = 0; kk < N - M; kk++) {
            uint32_t y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
            mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        for (; kk < N - 1; kk++) {
            uint32_t y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
            mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
        }
        uint32_t y = (mt_[N - 1] & UPPER_MASK) | (mt_[0] & LOWER_MASK);
        mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
        index_ = 0;
    }

    uint32_t y = mt_[index_++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

double PyRandom::random() {
    uint32_t a = next_u32() >> 5;
    uint32_t b = next_u32() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

uint32_t PyRandom::getrandbits(int k) {
    // Fast path only -- CPython's own fast path for k <= 32, which covers every draw vendor
    // makes (shop choice among 8 needs k=4; nothing here needs the >32-bit multi-word path).
    return next_u32() >> (32 - k);
}

uint32_t PyRandom::randbelow(uint32_t n) {
    if (n == 0) return 0;
    int k = 0;
    for (uint32_t t = n; t; t >>= 1) k++;  // n.bit_length()
    uint32_t r = getrandbits(k);
    while (r >= n) r = getrandbits(k);
    return r;
}

}  // namespace kaggriculture::sim

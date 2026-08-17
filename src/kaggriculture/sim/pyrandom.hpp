// A bit-exact port of CPython's `random.Random` (Modules/_randommodule.c): the classic
// MT19937 Mersenne Twister, seeded and sampled exactly as CPython does. Needed because
// vendor/kaggriculture.py reseeds a fresh `random.Random((seed * 1_000_003) ^ day)` every day
// and draws weed-spawn and shop-unlock randomness from it (issue 008's scope) -- an
// approximate RNG would make every downstream search result path-dependent on a fiction no
// human ever sees, since the real engine's replay is what actually gets scored.
//
// Three CPython behaviours are reproduced exactly, each verified against a live `python3 -c
// "import random; ..."` run before being trusted (see tests/sim/test_pyrandom_reference.py):
//   - seeding an int: absolute value, decomposed into little-endian 32-bit words, fed through
//     the reference `init_by_array` algorithm (not `init_genrand` directly).
//   - `random()`: the 53-bit float built from two consecutive 32-bit outputs
//     (`genrand_res53`).
//   - `choice(seq)` / `randrange(n)`: CPython's `_randbelow_with_getrandbits` -- rejection
//     sampling on `getrandbits(k)` where `k = n.bit_length()`, not `int(random() * n)`. This is
//     the one most ports get wrong, and it's exactly what vendor's shop-unlock draw uses.
#pragma once

#include <cstdint>

namespace kaggriculture::sim {

class PyRandom {
public:
    explicit PyRandom(uint64_t seed) { this->seed(seed); }

    // Reseeds exactly as `random.Random(seed)` would for a non-negative Python int seed (the
    // only case vendor ever constructs: `(seed * 1_000_003) ^ day` is always used as a plain
    // int, and CPython takes its absolute value before decomposing it -- so a "negative" 64-bit
    // XOR result must be abs()'d the same way Python would abs() the arbitrary-precision int).
    void seed(uint64_t seed);

    // Next raw 32-bit MT19937 output (`genrand_uint32`).
    uint32_t next_u32();

    // `random.random()`: uniform double in [0, 1), 53 bits of precision (`genrand_res53`).
    double random();

    // `getrandbits(k)` for k in [1, 32] -- the only range vendor's draws ever need (shop choice
    // among 8 items needs k=4; nothing here needs the >32-bit multi-word path).
    uint32_t getrandbits(int k);

    // `random.Random()._randbelow(n)`: uniform integer in [0, n), via rejection-sampled
    // getrandbits, not `int(random() * n)`. Returns 0 if n == 0 (matches CPython).
    uint32_t randbelow(uint32_t n);

    // `random.Random().choice(seq)`, where `seq` has `n` elements: returns the chosen index.
    uint32_t choice_index(uint32_t n) { return randbelow(n); }

private:
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfU;
    static constexpr uint32_t UPPER_MASK = 0x80000000U;
    static constexpr uint32_t LOWER_MASK = 0x7fffffffU;

    void init_genrand(uint32_t s);
    void init_by_array(const uint32_t* key, int key_length);

    uint32_t mt_[N];
    int index_ = N + 1;
};

}  // namespace kaggriculture::sim

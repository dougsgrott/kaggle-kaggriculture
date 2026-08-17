// Dumps PyRandom output in a simple, parseable format for tests/sim/test_pyrandom_reference.py
// to diff against a live `random.Random` run -- see pyrandom.hpp's header for why this needs to
// be bit-exact, not just statistically close.
#include <cstdint>
#include <cstdio>
#include <initializer_list>

#include "pyrandom.hpp"

using namespace kaggriculture::sim;

int main() {
    const uint64_t seeds[] = {0, 1, 42, 123456789, 4294967295ULL, 4294967296ULL, 9999999999ULL};
    for (uint64_t s : seeds) {
        PyRandom r(s);
        std::printf("random seed=%llu", static_cast<unsigned long long>(s));
        for (int i = 0; i < 8; i++) {
            double v = r.random();
            uint64_t bits;
            __builtin_memcpy(&bits, &v, 8);
            std::printf(" %016llx", static_cast<unsigned long long>(bits));
        }
        std::printf("\n");
    }
    for (uint64_t s : seeds) {
        PyRandom r(s);
        std::printf("choice8 seed=%llu", static_cast<unsigned long long>(s));
        for (int i = 0; i < 16; i++) {
            std::printf(" %u", r.choice_index(8));
        }
        std::printf("\n");
    }
    // The actual daily seeds vendor constructs: (episode_seed * 1_000_003) ^ day, for a spread
    // of episode seeds and every day in a 30-day season.
    for (uint64_t episode_seed : {0ULL, 1ULL, 70117ULL, 2147483647ULL}) {
        for (int day = 0; day < 30; day++) {
            uint64_t s = (episode_seed * 1000003ULL) ^ static_cast<uint64_t>(day);
            PyRandom r(s);
            std::printf("daily episode_seed=%llu day=%d", static_cast<unsigned long long>(episode_seed), day);
            for (int i = 0; i < 4; i++) {
                double v = r.random();
                uint64_t bits;
                __builtin_memcpy(&bits, &v, 8);
                std::printf(" %016llx", static_cast<unsigned long long>(bits));
            }
            std::printf(" choice=%u\n", r.choice_index(8));
        }
    }
    return 0;
}

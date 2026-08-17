// Whole-episode and whole-batch driving (issue 009). Deliberately pybind11-agnostic -- threading
// and GIL management live in bindings.cpp, which is the only file here that knows about Python.
// This keeps run_episode/run_batch testable and usable from a plain C++ benchmark or future
// search code without paying for a Python round-trip at all.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "market.hpp"
#include "policy.hpp"

namespace kaggriculture::sim {

// Episode-level configuration -- the fields of vendor's `configuration` this sim actually reads.
// Defaults are the generated config_defaults, never hand-typed (issues 007/008's discipline).
struct EpisodeConfig {
    int episode_steps = config_defaults::EPISODE_STEPS;
    int board_size = config_defaults::BOARD_SIZE;
    int turns_per_day = config_defaults::TURNS_PER_DAY;
    int shed_capacity = config_defaults::SHED_CAPACITY;
    double starting_money = config_defaults::STARTING_MONEY;
    MarketConfig market;
};

// Final money for both seats, per-day money/market-inventory trajectories, and the realized
// scenario (shop draws in order, weed-spawn counts per player) so a result is auditable without
// re-running the episode -- reproducing it exactly only ever needs `seed` + `config`, per
// issue 008's bit-exact RNG port, but this is cheap (~2KB/episode) and saves a re-run for
// analysis (issue 012's regime taxonomy, in particular, wants the shop-draw sequence directly).
struct EpisodeResult {
    uint64_t seed = 0;
    std::array<double, N_PLAYERS> final_money{};
    std::vector<std::array<double, N_PLAYERS>> daily_money;         // index = day
    std::vector<std::array<int, N_PRODUCTS>> daily_inventory;       // index = day
    std::vector<int> shop_draw_order;                                // ShopType indices, draw order
    std::array<int, N_PLAYERS> weed_spawn_count{};                   // total over the episode
};

// What happened on a day-boundary turn, for run_episode to fold into EpisodeResult -- separated
// out so advance_turns (below) can share the exact same turn logic and just discard this.
struct TurnOutcome {
    bool day_boundary = false;
    std::array<int, N_PLAYERS> weed_spawned{};
    int shop_drawn = -1;  // ShopType index, or -1 if none drawn this turn
};

// One turn, mutating `state`/`market` in place -- the shared core of run_episode and
// advance_turns below. Engine-exact order (see market.hpp's step_full_turn, which this mirrors
// but takes live policies instead of a pre-built TurnInput).
TurnOutcome run_turn(GameState& state, MarketTownState& market, Policy& policy_a, Policy& policy_b, const MarketConfig& cfg);

// Runs one episode to completion, calling `policy_a.act()`/`policy_b.act()` once per player per
// turn. Engine-exact turn order throughout.
EpisodeResult run_episode(Policy& policy_a, Policy& policy_b, const EpisodeConfig& cfg, uint64_t seed);

// Advances an existing (state, market) pair by `n_turns`, in place, with no EpisodeResult
// bookkeeping -- the branching-search primitive the issue's scope calls for: copy a GameState/
// MarketTownState snapshot (GameState::copy()/MarketTownState::copy() from Python -- both are
// trivially copyable, see sim.hpp/market.hpp), advance_turns() each candidate branch forward
// under a different policy, then compare final_money or a search-defined objective directly off
// `state.farms[*].money` without needing a full EpisodeResult per branch.
void advance_turns(GameState& state, MarketTownState& market, Policy& policy_a, Policy& policy_b, int n_turns,
                    const MarketConfig& cfg);

}  // namespace kaggriculture::sim

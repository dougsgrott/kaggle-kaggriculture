// The decision-making interface issue 009's episode runner drives. Two implementations ship
// here: TapePolicy (pure C++, the fast path search code in 013-015 actually uses -- an
// open-loop plan is exactly a fixed action tape) and a Python-callback policy (bindings.cpp,
// not here, since it needs pybind11 types) for interactive debugging, which is slow and is
// documented as such rather than optimized.
#pragma once

#include <utility>
#include <vector>

#include "sim.hpp"

namespace kaggriculture::sim {

// One player's actions for one turn -- everything TurnInput holds for a single player, split
// out because a Policy only ever controls one seat.
struct PlayerTurn {
    UnitAction unit_actions[MAX_UNITS_PER_PLAYER]{};
    int n_units_acting = 1;
    MarketOrder market_orders[MAX_MARKET_ORDERS]{};
    int n_market_orders = 0;
};

class Policy {
public:
    virtual ~Policy() = default;
    // `state`/`market` are the state BEFORE this turn's actions are applied. `player` is the
    // seat (0 or 1) this policy controls -- a Policy instance may be reused for either seat.
    virtual PlayerTurn act(const GameState& state, const MarketTownState& market, int player) = 0;
};

// Replays a precomputed sequence of PlayerTurns, one per turn, indexed by the ABSOLUTE
// `state.step` -- not a call counter local to this policy instance. Turns past the end of the
// tape PASS with no market orders (defined, not UB -- a short tape doesn't crash a longer
// episode; useful for "play N turns then let the episode run out"). This is the open-loop case
// issues 013-015 need: a plan, once found, is exactly a tape, and replaying one costs nothing
// beyond an array index -- no Python, no virtual-call-into-Python overhead.
//
// The absolute-step indexing matters for branching search (episode.hpp's advance_turns): a tape
// is normally a whole-episode plan, so a branch taken from a GameState snapshot at step 24
// should keep using indices from 24 onward, not restart at 0 -- reuse the SAME TapePolicy object
// (or one that agrees on absolute indices) across branches, rather than constructing a fresh
// short tape starting at index 0 for "just the next few turns" (it will never fire; see
// tests/sim/test_bindings.py's branching test for the fix, a CallbackPolicy, when that's what
// you actually want).
class TapePolicy : public Policy {
public:
    explicit TapePolicy(std::vector<PlayerTurn> tape) : tape_(std::move(tape)) {}

    PlayerTurn act(const GameState& state, const MarketTownState&, int) override {
        size_t t = static_cast<size_t>(state.step);
        if (t < tape_.size()) return tape_[t];
        return PlayerTurn{};
    }

    size_t size() const { return tape_.size(); }

private:
    std::vector<PlayerTurn> tape_;
};

}  // namespace kaggriculture::sim

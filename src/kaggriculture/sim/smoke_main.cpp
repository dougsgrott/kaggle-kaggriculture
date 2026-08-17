// Standalone smoke check for issue 007: proves the mechanical sim links, runs a full season
// without crashing, and produces sane-looking farm state at a few checkpoints. This is NOT a
// correctness claim -- issue 007's acceptance criterion defers correctness entirely to issue
// 010's parity gate against the real engine. It exists so a change here doesn't silently break
// in a way that only a full trace comparison would catch.

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "sim.hpp"

using namespace kaggriculture::sim;

namespace {

void assert_tile(TileKind expected, TileKind actual, const char* what) {
    if (expected != actual) {
        std::fprintf(stderr, "FAIL: %s -- expected tile kind %d, got %d\n", what, static_cast<int>(expected),
                     static_cast<int>(actual));
        std::exit(1);
    }
}

TurnInput blank_input() {
    TurnInput in;
    for (int p = 0; p < N_PLAYERS; p++) {
        in.n_units_acting[p] = 1;
        in.unit_actions[p][0] = {Op::PASS, Item::WHEAT, 1};
        in.n_market_orders[p] = 0;
    }
    return in;
}

}  // namespace

int main() {
    GameState state = init_game_state(/*board_size=*/10, /*turns_per_day=*/24, /*shed_capacity=*/100,
                                       /*starting_money=*/3000.0, /*seed=*/42);

    // A carrot planted at the farmer's spawn tile (4,4), watered every day through its bonus
    // window (ages 2-3), harvested on day 4. Expected final yield: 1 (base) + 1 + 1 = 3 -- the
    // number wiki/competition/pages/how-to-play.md and tests/model/test_yields.py both pin down
    // independently, so this cross-checks the C++ port against the Python model without needing
    // the real engine (that comparison is issue 010's job).
    state.privates[0].seeds[static_cast<int>(Crop::CARROT)] = 1;

    auto tile_at = [&]() -> Tile& { return state.farms[0].tiles[state.farms[0].unit_pos[0].y][state.farms[0].unit_pos[0].x]; };

    for (int step = 0; step < 120; step++) {
        TurnInput in = blank_input();
        int day = state.step / state.turns_per_day;
        if (step == 0) {
            in.unit_actions[0][0] = {Op::PLANT, Item::CARROT, 1};
        } else if (day == 0 && step == 1) {
            in.unit_actions[0][0] = {Op::WATER, Item::WHEAT, 1};
        } else if (day == 2 && state.step % state.turns_per_day == 1) {
            in.unit_actions[0][0] = {Op::WATER, Item::WHEAT, 1};
        } else if (day == 3 && state.step % state.turns_per_day == 1) {
            in.unit_actions[0][0] = {Op::WATER, Item::WHEAT, 1};
        } else if (day == 3 && state.step % state.turns_per_day == 2) {
            // Harvest the same day as the last watering, before decay starts at max_lifespan_step
            // (day4 hour0 for this crop: (planted_day 0 + max_yield_day 3 + 1) * 24 = 96).
            in.unit_actions[0][0] = {Op::HARVEST, Item::WHEAT, 1};
        }
        step_turn(state, in);
    }

    assert_tile(TileKind::EMPTY, tile_at().kind, "carrot tile after harvest");
    int carrot_item = static_cast<int>(Item::CARROT);
    int harvested = state.privates[0].inventory[0][carrot_item] + state.privates[0].shed[carrot_item];
    if (harvested != 3) {
        std::fprintf(stderr, "FAIL: expected 3 carrots harvested, got %d\n", harvested);
        return 1;
    }
    std::printf("OK: carrot lifecycle -- harvested %d (expected 3)\n", harvested);

    // A goose: build coop, place, feed+care daily, harvest. Proves BUILD_*, PLACE, FEED, CARE,
    // the animal-HARVEST branch, and the neglect/escape path all link and run.
    GameState g2 = init_game_state(10, 24, 100, 3000.0, 7);
    for (int step = 0; step < 240; step++) {
        TurnInput in = blank_input();
        Farm& farm = g2.farms[0];
        Tile& coop_tile = farm.tiles[farm.unit_pos[0].y][farm.unit_pos[0].x];
        if (step == 0) {
            in.unit_actions[0][0] = {Op::BUILD_COOP, Item::WHEAT, 1};
        } else if (coop_tile.kind == TileKind::COOP && !coop_tile.animal.occupied) {
            if (g2.privates[0].inventory[0][static_cast<int>(Item::GOOSE)] > 0) {
                in.unit_actions[0][0] = {Op::PLACE, Item::GOOSE, 1};
            } else {
                g2.privates[0].inventory[0][static_cast<int>(Item::GOOSE)] = 1;  // stand-in for BUY_ANIMAL (008)
            }
        } else if (coop_tile.animal.occupied) {
            if (!coop_tile.animal.fed_today) {
                if (g2.privates[0].inventory[0][static_cast<int>(Item::WHEAT)] <= 0) {
                    g2.privates[0].inventory[0][static_cast<int>(Item::WHEAT)] = 1;  // stand-in for BUY_PRODUCT (008)
                } else {
                    in.unit_actions[0][0] = {Op::FEED, Item::WHEAT, 1};
                }
            } else if (!coop_tile.animal.cared_today) {
                in.unit_actions[0][0] = {Op::CARE, Item::WHEAT, 1};
            } else if (coop_tile.animal.yield_units > 0) {
                in.unit_actions[0][0] = {Op::HARVEST, Item::WHEAT, 1};
            }
        }
        step_turn(g2, in);
    }
    int egg_item = static_cast<int>(Item::EGG);
    int eggs = g2.privates[0].inventory[0][egg_item] + g2.privates[0].shed[egg_item];
    if (eggs <= 0) {
        std::fprintf(stderr, "FAIL: expected at least one egg harvested over 10 days, got %d\n", eggs);
        return 1;
    }
    std::printf("OK: goose lifecycle -- %d eggs harvested\n", eggs);

    // Run a bare (all-PASS) full season to confirm nothing crashes across day/step boundaries.
    GameState g3 = init_game_state(10, 24, 100, 3000.0, 99);
    for (int step = 0; step < 720; step++) {
        step_turn(g3, blank_input());
    }
    if (g3.step != 720) {
        std::fprintf(stderr, "FAIL: expected step counter 720, got %d\n", g3.step);
        return 1;
    }
    std::printf("OK: 720-step all-PASS season completed, step=%d, money=[%.0f, %.0f]\n", g3.step, g3.farms[0].money,
                g3.farms[1].money);

    std::printf("ALL SMOKE CHECKS PASSED\n");
    return 0;
}

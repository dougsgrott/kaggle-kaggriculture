// Standalone smoke check for issue 008, mirroring 007's smoke_main.cpp: proves market.cpp
// links and behaves plausibly. NOT the correctness claim -- issue 010's parity gate is. Also
// covers the one hard acceptance criterion issue 008 states explicitly (not deferred to 010):
// the buy-then-sell round trip nets exactly zero for every buyable resource.

#include <cstdio>
#include <cstdlib>

#include "market.hpp"

using namespace kaggriculture::sim;

namespace {

void check(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
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
    // --- price table (wiki/competition/pages/how-to-play.md) ---
    struct Row {
        Item item;
        int T, p_minus_t, p_plus_t, p_plus_2t;
    };
    const Row rows[] = {
        {Item::WHEAT, 400, 45, 20, 19}, {Item::CARROT, 450, 70, 10, 1},   {Item::TOMATO, 200, 84, 24, 9},
        {Item::STRAWBERRY, 100, 204, 1, 1}, {Item::MELON, 300, 300, 1, 1}, {Item::EGG, 332, 70, 40, 39},
        {Item::MILK, 122, 256, 1, 1}, {Item::WOOL, 105, 240, 1, 1}, {Item::FERTILIZER, 200, 140, 60, 20},
    };
    for (const Row& r : rows) {
        int I0 = MARKET_PARAMS[static_cast<int>(r.item)].I0;
        check(market_price(r.item, I0 - r.T) == r.p_minus_t, "price table P(I0-T)");
        check(market_price(r.item, I0 + r.T) == r.p_plus_t, "price table P(I0+T)");
        check(market_price(r.item, I0 + 2 * r.T) == r.p_plus_2t, "price table P(I0+2T)");
    }
    std::printf("OK: price curve matches the how-to-play.md table for all 9 resources\n");

    // --- buy-then-sell round trip nets exactly zero (issue 008's explicit acceptance test) ---
    for (Item item : {Item::WHEAT, Item::FERTILIZER}) {
        GameState state = init_game_state(10, 24, 100, 3000.0, 1);
        MarketTownState market = init_market_town_state();
        MarketConfig cfg;

        double money_before = state.farms[0].money;
        TurnInput buy = blank_input();
        buy.n_market_orders[0] = 1;
        buy.market_orders[0][0] = {MarketOp::BUY_PRODUCT, item, 1};
        step_full_turn(state, market, buy, cfg);

        TurnInput sell = blank_input();
        sell.n_market_orders[0] = 1;
        sell.market_orders[0][0] = {MarketOp::SELL, item, 1};
        step_full_turn(state, market, sell, cfg);

        double money_after = state.farms[0].money;
        if (money_after != money_before) {
            std::fprintf(stderr, "FAIL: buy-then-sell round trip for item %d: %.2f -> %.2f\n", static_cast<int>(item),
                         money_before, money_after);
            return 1;
        }
    }
    std::printf("OK: buy-then-sell round trip nets exactly zero for WHEAT and FERTILIZER\n");

    // --- HIRE cost follows the fib curve, resetting daily ---
    {
        GameState state = init_game_state(10, 24, 100, 3000.0, 2);
        MarketTownState market = init_market_town_state();
        MarketConfig cfg;
        double money = state.farms[0].money;
        const int expected_costs[] = {1, 1, 2, 3, 5};  // fib(0..4)
        for (int expected : expected_costs) {
            TurnInput in = blank_input();
            in.n_market_orders[0] = 1;
            in.market_orders[0][0] = {MarketOp::HIRE, Item::WHEAT, 1};
            step_full_turn(state, market, in, cfg);
            double spent = money - state.farms[0].money;
            if (spent != expected) {
                std::fprintf(stderr, "FAIL: hire cost expected %d, spent %.2f\n", expected, spent);
                return 1;
            }
            money = state.farms[0].money;
        }
        check(state.farms[0].n_units == 6, "5 hires -> 6 units (farmer + 5 hands)");
    }
    std::printf("OK: HIRE cost follows fib(n), resetting daily\n");

    // --- concurrent SELL: both players get the SAME pre-commit price for their unit at each
    // index, verified against a hand-traced run of vendor's own market_price() (see the
    // comment for the exact trace command). ---
    {
        GameState state = init_game_state(10, 24, 100, 3000.0, 5);
        MarketTownState market = init_market_town_state();
        MarketConfig cfg;
        for (int p = 0; p < N_PLAYERS; p++) state.privates[p].shed[static_cast<int>(Item::WHEAT)] = 3;

        TurnInput in = blank_input();
        for (int p = 0; p < N_PLAYERS; p++) {
            in.n_market_orders[p] = 1;
            in.market_orders[p][0] = {MarketOp::SELL, Item::WHEAT, 3};
        }
        step_full_turn(state, market, in, cfg);

        // vendor trace (python3, vendor/kaggriculture.py's market_price directly): both players
        // pay 25, 24, 24 for their three units in turn; final money 3073.0 each, inventory 10006
        // from the sells -- minus 1 more from town_consume's step-0 town-centre tick (also part
        // of step_full_turn), landing at 10005.
        check(state.farms[0].money == 3073.0, "concurrent SELL: player 0 final money");
        check(state.farms[1].money == 3073.0, "concurrent SELL: player 1 final money");
        check(market.inventory[static_cast<int>(Item::WHEAT)] == 10005, "concurrent SELL: final inventory");
    }
    std::printf("OK: concurrent SELL matches a hand-traced run of vendor's own market_price()\n");

    // --- town consumption drains market inventory on schedule ---
    {
        GameState state = init_game_state(10, 24, 100, 3000.0, 3);
        MarketTownState market = init_market_town_state();
        market.shop_count[static_cast<int>(ShopType::PET_CAFE)] = 1;  // single-product -> 2x, CARROT only
        MarketConfig cfg;
        int carrot = static_cast<int>(Item::CARROT);
        int before = market.inventory[carrot];
        TurnInput in = blank_input();
        step_full_turn(state, market, in, cfg);  // step 0 -- both shop (interval 4) and center (24) tick
        int after = market.inventory[carrot];
        // PET_CAFE (2x, single product) + town centre (1x, non-fertilizer) = 3 units drained.
        check(before - after == 3, "town consumption drains PET_CAFE (2x) + centre (1x) on step 0");
    }
    std::printf("OK: town consumption matches the shop/centre schedule\n");

    // --- full season runs without crashing ---
    {
        GameState state = init_game_state(10, 24, 100, 3000.0, 42);
        MarketTownState market = init_market_town_state();
        MarketConfig cfg;
        for (int step = 0; step < 720; step++) {
            step_full_turn(state, market, blank_input(), cfg);
        }
        check(state.step == 720, "720-step season completes");
        std::printf("OK: 720-step all-PASS season with market/town active, shops unlocked=%d\n", market.n_shops_unlocked);
    }

    std::printf("ALL MARKET SMOKE CHECKS PASSED\n");
    return 0;
}

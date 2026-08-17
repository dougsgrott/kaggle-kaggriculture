// Kaggriculture C++ sim, economic half (issue 008): the price curve, the concurrent per-unit
// order lockstep (SELL/BUY_*/HIRE/BUY_LAND), town demand, and weed-spawn RNG. Ported directly
// from vendor/kaggriculture.py -- see sim.hpp's header for the scope boundary between the two
// issues and why it falls where it does.
//
// This file owns the RNG-carrying state that issue 007's plain-function-pointer hooks couldn't
// (see sim.hpp's comment above step_turn): MarketTownState holds a persistent PyRandom that's
// reseeded once per day and shared across that day's two weed-spawn calls and one shop-unlock
// call, exactly reproducing vendor's single `Random((seed * 1_000_003) ^ day)` stream. Because
// of that, integration goes through step_full_turn() here, not sim.hpp's step_turn().
#pragma once

#include "pyrandom.hpp"
#include "sim.hpp"

namespace kaggriculture::sim {

// Mirrors the subset of `get(cfg, "key", default)` calls vendor's market/town code reads.
// Defaults are the generated config_defaults (constants_generated.hpp), never hand-typed.
// `shedCapacity` and `boardSize` are deliberately NOT here even though vendor's market code
// reads both: GameState (sim.hpp) already carries `shed_capacity`/`board_size` for 007's own
// use, and duplicating them here would let a caller pass mismatched values -- market.cpp reads
// `state.shed_capacity`/`state.board_size` directly instead.
struct MarketConfig {
    int max_market_orders_per_turn = config_defaults::MAX_MARKET_ORDERS_PER_TURN;
    int farm_hand_cost_mult = config_defaults::FARM_HAND_COST_MULT;
    double weed_spawn_chance = config_defaults::WEED_SPAWN_CHANCE;
    int town_shop_unlock_interval = config_defaults::TOWN_SHOP_UNLOCK_INTERVAL;
    int town_shop_sell_interval = config_defaults::TOWN_SHOP_SELL_INTERVAL;
    int town_center_sell_interval = config_defaults::TOWN_CENTER_SELL_INTERVAL;
};

// The economic-half state GameState deliberately doesn't hold (see sim.hpp). `inventory` is
// indexed by Item (only [0, N_PRODUCTS) are ever written); `shop_count[t]` is how many
// instances of ShopType `t` have been drawn so far (duplicates consume independently, but
// consumption only ever depends on the count per type, never draw order -- see town_consume's
// comment). `rng`/`rng_day` implement the once-per-day reseed vendor's `_end_of_day` does.
struct MarketTownState {
    int inventory[N_PRODUCTS];
    int shop_count[static_cast<int>(ShopType::COUNT)] = {};
    int n_shops_unlocked = 0;
    PyRandom rng{0};
    int rng_day = -1;
};

// Market inventory starts at each product's I0 (vendor's `_new_market`).
MarketTownState init_market_town_state();

// One of the six curve shapes, evaluated at x >= 0 (negative x is clamped to 0) -- vendor's
// `_shape`. `T` only matters for HINGE.
double shape(ShapeFunc func, double x, double T);

// `price(inv)` for a market-tradeable item (Item index < N_PRODUCTS) -- vendor's `market_price`.
// Floored at $1, rounded to nearest (ties to even, matching Python's `round()`).
int market_price(Item item, int inventory);

// vendor's `_process_market`: up to `cfg.max_market_orders_per_turn` orders per player,
// processed index-by-index; HIRE/BUY_LAND resolve atomically per index (in player order), then
// SELL/BUY_SEED/BUY_PRODUCT/BUY_ANIMAL run the per-unit concurrent lockstep -- both players get
// the pre-commit price for their unit at this index before inventory moves.
void process_market_orders(GameState& state, MarketTownState& market,
                            const MarketOrder orders[N_PLAYERS][MAX_MARKET_ORDERS], const int n_orders[N_PLAYERS],
                            const MarketConfig& cfg);

// vendor's `_town_consume`: shop and town-centre draws on turn `step`.
void town_consume(MarketTownState& market, int step, const MarketConfig& cfg);

// vendor's `_spawn_weeds` for one player. Reseeds `market.rng` from `state.seed`/`day` on first
// use each day, so callers must invoke this for player 0 before player 1 (see sim.hpp's
// WeedSpawnHook comment) -- step_full_turn does this correctly; call directly only if you
// preserve that order yourself. Returns the number of tiles that turned into weeds this call
// (issue 009's episode runner reports this per player/day as part of the realized scenario).
int spawn_weeds(GameState& state, MarketTownState& market, int player, int day, const MarketConfig& cfg);

// vendor's with-replacement shop-unlock draw, run once after both players' spawn_weeds for the
// day. Continues `market.rng` rather than reseeding -- must run after both spawn_weeds calls.
// Returns the ShopType index drawn, or -1 if no draw happened this day (not a scheduled unlock
// day, or MAX_SHOP_INSTANCES already reached).
int unlock_shop(MarketTownState& market, int day, const MarketConfig& cfg);

// The full turn loop: unit actions (sim.hpp) -> market queue + town buys (this file) -> per-step
// decay (sim.hpp) -> day refresh, mechanical (sim.hpp) then stochastic (this file), on the day
// boundary -> advance step. Engine-exact order -- vendor's interpreter().
void step_full_turn(GameState& state, MarketTownState& market, const TurnInput& input, const MarketConfig& cfg);

}  // namespace kaggriculture::sim

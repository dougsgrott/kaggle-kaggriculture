// Implementation. Every function here is a direct, function-for-function port of the
// corresponding vendor/kaggriculture.py function (named in each comment) -- see market.hpp and
// sim.hpp's headers for the scope boundary between issues 007 and 008.

#include "market.hpp"

#include <algorithm>
#include <cmath>

namespace kaggriculture::sim {

namespace {

int fib(int n) {
    int a = 1, b = 1;
    for (int i = 0; i < n; i++) {
        int next = a + b;
        a = b;
        b = next;
    }
    return a;
}

int hire_cost(int n_already_today, int mult) { return mult * fib(n_already_today); }

// vendor's `_do_hire`.
void do_hire(Farm& farm, int board_size, int mult) {
    int cost = hire_cost(farm.hires_today, mult);
    if (farm.money < cost) return;
    if (farm.n_units >= MAX_UNITS_PER_PLAYER) return;  // sim-level safety net; see sim.hpp's header
    farm.money -= cost;
    farm.hires_today += 1;
    farm.unit_pos[farm.n_units] = spawn_hand_position(farm, board_size);
    farm.n_units += 1;
    // The new unit's inventory slot is already zero -- never written to before this.
}

// vendor's `_do_buy_land`.
void do_buy_land(Farm& farm, int board_size) {
    int n_extra = 0;
    for (int q = 1; q < 4; q++) {
        if (farm.unlocked_quadrant[q]) n_extra++;
    }
    if (n_extra >= 3) return;  // len(LAND_ORDER) == 3
    int cost = LAND_PRICES[n_extra];
    if (farm.money < cost) return;
    farm.money -= cost;
    int quadrant = n_extra + 1;  // LAND_ORDER = [NE, SW, SE] == quadrant_of() indices [1, 2, 3]
    farm.unlocked_quadrant[quadrant] = true;
    for (int y = 0; y < board_size; y++) {
        for (int x = 0; x < board_size; x++) {
            if (quadrant_of(x, y, board_size) == quadrant && farm.tiles[y][x].kind == TileKind::LOCKED) {
                farm.tiles[y][x] = Tile{};
            }
        }
    }
}

struct ParsedOrder {
    MarketOp op = MarketOp::NONE;
    Item item = Item::WHEAT;
    int remaining = 0;
    bool valid = false;
};

// vendor's `_parse_order`. HIRE/BUY_LAND carry no item/quantity (remaining is unused for them,
// left at 1 only so `valid` reads naturally); the rest need remaining > 0.
ParsedOrder parse_order(const MarketOrder& order) {
    ParsedOrder p;
    if (order.op == MarketOp::NONE) return p;
    if (order.op == MarketOp::HIRE || order.op == MarketOp::BUY_LAND) {
        p.op = order.op;
        p.valid = true;
        p.remaining = 1;
        return p;
    }
    if (order.n <= 0) return p;
    p.op = order.op;
    p.item = order.item;
    p.remaining = order.n;
    p.valid = true;
    return p;
}

// vendor's `_commit_unit`.
bool commit_unit(MarketOp op, Item item, int price, Farm& farm, Private& priv, MarketTownState& market, int shed_capacity) {
    int i = static_cast<int>(item);
    if (op == MarketOp::SELL) {
        if (priv.shed[i] <= 0) return false;
        priv.shed[i] -= 1;
        farm.money += price;
        // Sales at $1 do not increase market supply.
        if (price > 1) market.inventory[i] += 1;
        return true;
    }
    if (op == MarketOp::BUY_PRODUCT) {
        if (farm.money < price) return false;
        int total = 0;
        for (int j = 0; j < N_ITEMS; j++) total += priv.shed[j];
        if (total >= shed_capacity) return false;
        farm.money -= price;
        priv.shed[i] += 1;
        market.inventory[i] -= 1;
        return true;
    }
    if (op == MarketOp::BUY_SEED) {
        if (farm.money < price) return false;
        farm.money -= price;
        priv.seeds[crop_index(item)] += 1;
        return true;
    }
    if (op == MarketOp::BUY_ANIMAL) {
        if (farm.money < price) return false;
        int total = 0;
        for (int j = 0; j < N_ITEMS; j++) total += priv.shed[j];
        if (total >= shed_capacity) return false;
        farm.money -= price;
        priv.shed[i] += 1;
        return true;
    }
    return false;
}

}  // namespace

MarketTownState init_market_town_state() {
    MarketTownState m;
    for (int i = 0; i < N_PRODUCTS; i++) m.inventory[i] = MARKET_PARAMS[i].I0;
    return m;
}

double shape(ShapeFunc func, double x, double T) {
    x = std::max(0.0, x);
    switch (func) {
        case ShapeFunc::LINEAR: return x;
        case ShapeFunc::SQ: return x * x;
        case ShapeFunc::SQRT: return std::sqrt(x);
        case ShapeFunc::LOG: return std::log(1.0 + x);
        case ShapeFunc::LOG10: return std::log10(1.0 + x);
        case ShapeFunc::HINGE: {
            // Degenerates to linear if T is missing or non-positive.
            if (T <= 0) return x;
            double u = x / T;
            double over = std::max(0.0, u - 1.0);
            return u + 8.0 * over * over;
        }
    }
    return x;
}

int market_price(Item item, int inventory) {
    const MarketParamDef& p = MARKET_PARAMS[static_cast<int>(item)];
    double raw;
    if (inventory < p.I0) {
        double amp = p.below_target * p.base / shape(p.below_func, p.T, p.T);
        raw = p.base + amp * shape(p.below_func, p.I0 - inventory, p.T);
    } else {
        double amp = p.above_target * p.base / shape(p.above_func, p.T, p.T);
        raw = p.base - amp * shape(p.above_func, inventory - p.I0, p.T);
    }
    // std::nearbyint (not std::round) to match Python's round-half-to-even, not
    // round-half-away-from-zero. Only matters if `raw` lands exactly on a .5 boundary, which a
    // curve built from sqrt/log/hinge essentially never does -- included for exactness anyway.
    int rounded = static_cast<int>(std::nearbyint(raw));
    return std::max(1, rounded);
}

void process_market_orders(GameState& state, MarketTownState& market,
                            const MarketOrder orders[N_PLAYERS][MAX_MARKET_ORDERS], const int n_orders[N_PLAYERS],
                            const MarketConfig& cfg) {
    int max_orders = std::max(1, cfg.max_market_orders_per_turn);
    int n[N_PLAYERS];
    for (int p = 0; p < N_PLAYERS; p++) n[p] = std::min({n_orders[p], max_orders, MAX_MARKET_ORDERS});
    int max_len = std::max(n[0], n[1]);

    for (int i = 0; i < max_len; i++) {
        ParsedOrder ostate[N_PLAYERS];
        for (int p = 0; p < N_PLAYERS; p++) {
            ostate[p] = (i < n[p]) ? parse_order(orders[p][i]) : ParsedOrder{};
        }

        // Atomic orders (HIRE, BUY_LAND): handle once, in player order.
        for (int p = 0; p < N_PLAYERS; p++) {
            if (!ostate[p].valid) continue;
            if (ostate[p].op == MarketOp::HIRE) {
                do_hire(state.farms[p], state.board_size, cfg.farm_hand_cost_mult);
                ostate[p].valid = false;
            } else if (ostate[p].op == MarketOp::BUY_LAND) {
                do_buy_land(state.farms[p], state.board_size);
                ostate[p].valid = false;
            }
        }

        // Per-unit lockstep loop for SELL / BUY_*.
        for (int guard = 0; guard < 100000; guard++) {
            struct Quote {
                bool present = false;
                MarketOp op = MarketOp::NONE;
                Item item = Item::WHEAT;
                int price = 0;
            } quoted[N_PLAYERS];

            for (int p = 0; p < N_PLAYERS; p++) {
                if (!ostate[p].valid || ostate[p].remaining <= 0) continue;
                MarketOp op = ostate[p].op;
                Item item = ostate[p].item;
                if (op == MarketOp::SELL && !is_animal_item(item)) {
                    quoted[p] = {true, op, item, market_price(item, market.inventory[static_cast<int>(item)])};
                } else if (op == MarketOp::BUY_PRODUCT && (item == Item::WHEAT || item == Item::FERTILIZER)) {
                    // Quote at post-buy inventory so a buy/sell round-trip against an unchanged
                    // market nets zero.
                    quoted[p] = {true, op, item, market_price(item, market.inventory[static_cast<int>(item)] - 1)};
                } else if (op == MarketOp::BUY_SEED && is_crop_item(item)) {
                    quoted[p] = {true, op, item, CROPS[crop_index(item)].seed};
                } else if (op == MarketOp::BUY_ANIMAL && is_animal_item(item)) {
                    quoted[p] = {true, op, item, ANIMALS[animal_index(item)].cost};
                } else {
                    ostate[p].valid = false;  // malformed sub-op; abort this order
                }
            }

            if (!quoted[0].present && !quoted[1].present) break;

            bool committed_any = false;
            for (int p = 0; p < N_PLAYERS; p++) {
                if (!quoted[p].present) continue;
                bool ok = commit_unit(quoted[p].op, quoted[p].item, quoted[p].price, state.farms[p], state.privates[p],
                                       market, state.shed_capacity);
                if (ok) {
                    ostate[p].remaining -= 1;
                    committed_any = true;
                } else {
                    ostate[p].valid = false;
                }
            }
            if (!committed_any) break;
        }
        // vendor calls `_refresh_prices` here, after every order-index's lockstep settles; we
        // compute market_price() on demand rather than caching it, so there's nothing to refresh.
    }
}

void town_consume(MarketTownState& market, int step, const MarketConfig& cfg) {
    int shop_interval = std::max(1, cfg.town_shop_sell_interval);
    int center_interval = std::max(1, cfg.town_center_sell_interval);

    if (step % shop_interval == 0) {
        for (int s = 0; s < static_cast<int>(ShopType::COUNT); s++) {
            int count = market.shop_count[s];
            if (count <= 0) continue;
            const ShopDef& def = SHOPS[s];
            int multiplier = (def.n_products == 1) ? 2 : 1;
            for (int k = 0; k < def.n_products; k++) {
                market.inventory[def.products[k]] -= multiplier * count;
            }
        }
    }
    if (step % center_interval == 0) {
        for (int i = 0; i < N_PRODUCTS; i++) {
            if (static_cast<Item>(i) == Item::FERTILIZER) continue;
            market.inventory[i] -= 1;
        }
    }
}

int spawn_weeds(GameState& state, MarketTownState& market, int player, int day, const MarketConfig& cfg) {
    if (day != market.rng_day) {
        market.rng.seed((state.seed * 1000003ULL) ^ static_cast<uint64_t>(day));
        market.rng_day = day;
    }
    int spawned = 0;
    Farm& farm = state.farms[player];
    for (int y = 0; y < state.board_size; y++) {
        for (int x = 0; x < state.board_size; x++) {
            if (farm.tiles[y][x].kind == TileKind::EMPTY && market.rng.random() < cfg.weed_spawn_chance) {
                farm.tiles[y][x] = Tile{};
                farm.tiles[y][x].kind = TileKind::WEED;
                spawned++;
            }
        }
    }
    return spawned;
}

int unlock_shop(MarketTownState& market, int day, const MarketConfig& cfg) {
    int shop_interval = std::max(1, cfg.town_shop_unlock_interval);
    int next_day = day + 1;
    if (next_day % shop_interval != 0) return -1;
    if (market.n_shops_unlocked >= MAX_SHOP_INSTANCES) return -1;
    uint32_t idx = market.rng.choice_index(static_cast<uint32_t>(ShopType::COUNT));
    market.shop_count[idx] += 1;
    market.n_shops_unlocked += 1;
    return static_cast<int>(idx);
}

void step_full_turn(GameState& state, MarketTownState& market, const TurnInput& input, const MarketConfig& cfg) {
    int day = state.step / state.turns_per_day;

    apply_all_unit_actions(state, input, day);

    process_market_orders(state, market, input.market_orders, input.n_market_orders, cfg);
    town_consume(market, state.step, cfg);

    for (int p = 0; p < N_PLAYERS; p++) {
        decay_plants(state.farms[p], state.step);
    }

    if ((state.step + 1) % state.turns_per_day == 0) {
        for (int p = 0; p < N_PLAYERS; p++) {
            end_of_day_mechanical(state.farms[p], state.privates[p], day, state.board_size, state.turns_per_day,
                                   state.shed_capacity);
            spawn_weeds(state, market, p, day, cfg);
        }
        unlock_shop(market, day, cfg);
    }

    state.step += 1;
}

}  // namespace kaggriculture::sim

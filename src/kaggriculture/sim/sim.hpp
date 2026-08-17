// Kaggriculture C++ sim, mechanical half (issue 007). Ported directly from
// vendor/kaggriculture.py -- that file is the specification; this header/its .cpp reproduce its
// farm/unit/action/day-refresh behaviour exactly, function for function, comment for comment
// where a comment records a non-obvious rule. See vendor/kaggriculture.py's own header for the
// pinned engine version this was ported from.
//
// Scope boundary with issue 008 (market curve, order lockstep, town demand, weed-spawn RNG):
// GameState holds no market/town fields. step_turn() calls out to two hook points --
// MarketAndTownHook and StochasticDayRefreshHook -- that issue 008 implements; both default to
// no-ops here so this file compiles and runs standalone. See issues/007-sim-core-port.md's
// "Scope boundary" section for the precise split and why (HIRE/BUY_LAND are market orders, not
// farmer actions, despite being deterministically priced; weed-from-neglect is deterministic and
// therefore ours, weed-*spawning* is RNG-driven and therefore 008's).
//
// Design constraints (see issue 007's "Design notes"): GameState is value-semantics, trivially
// copyable, and does no heap allocation -- every container below is a fixed-size array sized to
// a documented sim limit, not a std::vector. That's what makes snapshot-and-branch search
// (issues 013+) cheap.

#pragma once

#include <cstdint>
#include <type_traits>

#include "constants_generated.hpp"

namespace kaggriculture::sim {

// ---------------------------------------------------------------------------------------------
// Sim-wide limits. boardSize is 10 in every configuration this competition actually uses (how-
// to-play.md: "Advanced uses 10 = four 5x5 quadrants"); MAX_BOARD gives headroom rather than
// hardcoding 10, so a differently-configured episode still fits. MAX_UNITS_PER_PLAYER caps hands
// at 32/day: fib(32) is already far beyond any money a rational agent would spend on a single
// day's crew, so this is a real ceiling on hire cost, not an arbitrary one.
// ---------------------------------------------------------------------------------------------
constexpr int MAX_BOARD = 16;
constexpr int N_CROPS = 5;
constexpr int N_ANIMALS = 3;
constexpr int N_ITEMS = 12;  // 9 PRODUCTS + 3 ANIMALS, see Item enum
constexpr int MAX_HANDS = 32;
constexpr int MAX_UNITS_PER_PLAYER = MAX_HANDS + 1;  // + the farmer
constexpr int N_PLAYERS = 2;
constexpr int MAX_MARKET_ORDERS = 10;  // maxMarketOrdersPerTurn default; step_turn takes the
                                        // configured value too, this only bounds the array

// ---------------------------------------------------------------------------------------------
// Enums. OPS/ITEMS/MOPS orderings are adopted verbatim from
// analysis/nb_clean/nikital7__4000x-environment-speedup-kaggriculture.py's export_trace.py -- a
// public, already-vetted wire format (reuse permitted under competition rules 2.6/3.6.b) -- so
// 010's parity validator and any Python-side trace exporter agree on the encoding without a
// second translation layer.
// ---------------------------------------------------------------------------------------------

// Farmer/hand unit actions ("farmer"/"hands" in the observation's action dict).
enum class Op : uint8_t {
    PASS,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    PICKUP,
    DROP,
    PLACE,
    PLANT,
    WATER,
    HARVEST,
    FERTILIZE,
    DIG,
    BUILD_COOP,
    BUILD_PASTURE,
    FEED,
    COLLECT_FERTILIZER,
    CARE,
    COUNT
};

// Tradeable/holdable item kinds. Indices 0-8 are PRODUCTS (market-tradeable); 9-11 are ANIMALS
// (shed/inventory-holdable, never market inventory). Order matches vendor's PRODUCTS list then
// its ANIMALS dict insertion order.
enum class Item : uint8_t {
    WHEAT,
    CARROT,
    TOMATO,
    STRAWBERRY,
    MELON,
    EGG,
    MILK,
    WOOL,
    FERTILIZER,
    GOOSE,
    COW,
    SHEEP,
    COUNT
};
constexpr int N_PRODUCTS = 9;  // Item indices [0, N_PRODUCTS) trade on the market

// Market order ops ("market" list in the action dict). Declared here (not in an 008 header)
// because sim.hpp is the shared wire-format contract between the two halves; 008 implements the
// processing, 007 just needs the type to declare step_turn()'s TurnInput.
enum class MarketOp : uint8_t { NONE, HIRE, BUY_LAND, BUY_SEED, BUY_PRODUCT, BUY_ANIMAL, SELL, COUNT };

enum class TileKind : uint8_t { EMPTY, LOCKED, PLANT, WEED, COOP, PASTURE };

// Crop/Animal are the sub-enums CROPS/ANIMALS (constants_generated.hpp) are indexed by; both are
// Item's first N_CROPS / last N_ANIMALS values re-namespaced for clarity at call sites.
enum class Crop : uint8_t { WHEAT, CARROT, TOMATO, STRAWBERRY, MELON, COUNT };
enum class Animal : uint8_t { GOOSE, COW, SHEEP, COUNT };
enum class Structure : uint8_t { COOP, PASTURE };  // matches AnimalDef::structure / TileKind::COOP,PASTURE

// (dx, dy); y grows downward, matching vendor's FARMER_MOVES.
struct Delta {
    int8_t dx;
    int8_t dy;
};
constexpr Delta move_delta(Op op) {
    switch (op) {
        case Op::NORTH: return {0, -1};
        case Op::SOUTH: return {0, 1};
        case Op::EAST: return {1, 0};
        case Op::WEST: return {-1, 0};
        default: return {0, 0};
    }
}
constexpr bool is_move(Op op) { return op == Op::NORTH || op == Op::SOUTH || op == Op::EAST || op == Op::WEST; }

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

struct UnitAction {
    Op op = Op::PASS;
    Item item = Item::WHEAT;  // meaningful only for PICKUP/DROP(implicit)/PLACE/PLANT
    int n = 1;                // PICKUP/PLACE quantity; ignored elsewhere
};

struct MarketOrder {
    MarketOp op = MarketOp::NONE;
    Item item = Item::WHEAT;
    int n = 1;
};

// Mirrors the plant-tile dict fields exactly (vendor's `_new_plant` / tile dict shape).
struct PlantTile {
    Crop crop = Crop::WHEAT;
    int planted_day = 0;
    bool watered_today = false;
    int consecutive_unwatered = 1;  // planting day counts as the first missed day
    int yield_units = 0;
    int max_lifespan_step = -1;  // -1 while an ongoing crop hasn't hit max_yield yet
    int fertilized_until_day = -1;
};

// Mirrors the animal structure dict (coop/pasture, optionally occupied).
struct AnimalTile {
    bool occupied = false;
    Animal animal = Animal::GOOSE;  // meaningful only if occupied
    int placed_day = 0;
    int yield_units = 0;
    int consecutive_unfed = 0;
    bool fed_today = false;
    bool cared_today = false;
    bool fertilizer_available = false;
    int pending_care_bonus = 0;
};

struct Tile {
    TileKind kind = TileKind::EMPTY;
    PlantTile plant{};   // valid iff kind == PLANT
    AnimalTile animal{}; // valid iff kind == COOP || kind == PASTURE
};

struct Position {
    int8_t x = 0;
    int8_t y = 0;
    bool operator==(const Position& o) const { return x == o.x && y == o.y; }
};

struct Farm {
    double money = 0;
    Tile tiles[MAX_BOARD][MAX_BOARD]{};
    Position unit_pos[MAX_UNITS_PER_PLAYER]{};  // [0] = farmer, [1..n_units) = hands
    int n_units = 1;                            // farmer always present; grows via HIRE (008)
    bool unlocked_quadrant[4] = {true, false, false, false};  // NW, NE, SW, SE (LAND_ORDER)
    int hires_today = 0;
};

struct Private {
    int shed[N_ITEMS]{};
    int seeds[N_CROPS]{};
    int inventory[MAX_UNITS_PER_PLAYER][N_ITEMS]{};
};

// The mechanical half's complete state. No market/town fields -- see the file header. `seed` is
// the one piece of RNG-adjacent state that lives here anyway: vendor reconstructs a fresh
// `Random((seed * 1_000_003) ^ day)` every day rather than threading a persistent generator, so
// nothing else needs to be carried between days -- but 008's hooks need the episode seed itself
// to reproduce that stream, and it has nowhere else to live.
struct GameState {
    int board_size = 10;
    int turns_per_day = 24;
    int shed_capacity = 100;
    int step = 0;
    uint64_t seed = 0;
    Farm farms[N_PLAYERS];
    Private privates[N_PLAYERS];
};
static_assert(std::is_trivially_copyable_v<GameState>, "GameState must stay snapshot-cheap");

// ---------------------------------------------------------------------------------------------
// Board geometry helpers (vendor's _quadrant_of / _shed_access_tiles / _is_shed_adjacent /
// _initial_tile / _default_spawn).
// ---------------------------------------------------------------------------------------------

int quadrant_of(int x, int y, int board_size);              // 0=NW,1=NE,2=SW,3=SE
void shed_access_tiles(int board_size, Position out[4]);    // NWSE order
bool is_shed_adjacent(Position p, int board_size);
Position default_spawn(int board_size);
Position spawn_hand_position(const Farm& farm, int board_size);  // least-occupied access tile, NWSE ties

// ---------------------------------------------------------------------------------------------
// Per-unit action application (vendor's _apply_unit_action). Invalid/illegal actions are silent
// no-ops, matching the engine exactly -- there is no error return.
// ---------------------------------------------------------------------------------------------
void apply_unit_action(Farm& farm, Private& priv, int unit_idx, UnitAction action, int board_size, int day,
                        int turns_per_day, int shed_capacity);

// The atomic-PLANT-oversubscription rule from interpreter(): if the sum of PLANT requests for a
// crop across a player's units this turn exceeds available seeds, ALL of that crop's PLANT
// requests are replaced with PASS before any unit acts. Mutates `actions` in place.
void apply_plant_oversubscription_guard(UnitAction actions[MAX_UNITS_PER_PLAYER], int n_units, const Private& priv);

// ---------------------------------------------------------------------------------------------
// Per-step and day-refresh mechanics (vendor's _decay_plants / _daily_refresh_plants /
// _daily_refresh_animals / _drop_inventories_to_shed). All deterministic -- no RNG.
// ---------------------------------------------------------------------------------------------
void decay_plants(Farm& farm, int step);
void daily_refresh_plants(Farm& farm, int current_day, int turns_per_day);
void daily_refresh_animals(Farm& farm, int day);
void drop_inventories_to_shed(Private& priv, int n_units, int shed_capacity);

// Bundles every deterministic end-of-day step vendor's _end_of_day performs, in order: the two
// daily_refresh_* calls, drop_inventories_to_shed, unit respawn (farmer + hands reset to a
// single farmer at the default spawn), and hires_today reset. Does NOT spawn weeds or unlock
// shops -- those are RNG-driven and go through StochasticDayRefreshHook instead.
void end_of_day_mechanical(Farm& farm, Private& priv, int day, int board_size, int turns_per_day, int shed_capacity);

// ---------------------------------------------------------------------------------------------
// The seam: issue 008 supplies these. Defaults are no-ops so this file compiles and step_turn()
// runs a full, legal (if economically inert) season on its own.
// ---------------------------------------------------------------------------------------------

// Runs the market-order queue (SELL/BUY_*/HIRE/BUY_LAND lockstep) and town consumption for this
// step -- vendor's _process_market + _town_consume, i.e. everything that needs a price curve,
// touches market inventory, or reads town/shop state. GameState carries no market/town fields,
// so this hook is expected to close over (or be passed, via a wider issue-008 struct not defined
// here) whatever state it needs; `orders`/`n_orders` are this step's already-truncated-to-
// maxMarketOrdersPerTurn per-player order lists.
using MarketAndTownHook = void (*)(GameState& state, const MarketOrder orders[N_PLAYERS][MAX_MARKET_ORDERS],
                                    const int n_orders[N_PLAYERS], int step);
void default_market_and_town_hook(GameState& state, const MarketOrder orders[N_PLAYERS][MAX_MARKET_ORDERS],
                                   const int n_orders[N_PLAYERS], int step);

// The RNG half of end-of-day, split into the two hooks vendor's _end_of_day calls at different
// points off the *same* `Random((seed * 1_000_003) ^ day)` stream -- weed spawning runs once per
// player inside the per-player loop, the shop draw runs once after both players are done, and
// the stream is shared across both, so 008's implementations must draw from it in that order
// (player 0's weeds, player 1's weeds, then the shop draw) to reproduce vendor exactly.
using WeedSpawnHook = void (*)(GameState& state, int player, int day);
void default_weed_spawn_hook(GameState& state, int player, int day);

using ShopUnlockHook = void (*)(GameState& state, int day);
void default_shop_unlock_hook(GameState& state, int day);

// ---------------------------------------------------------------------------------------------
// Turn loop (vendor's interpreter(), engine-exact order): validate (implicit in the no-op
// semantics of apply_unit_action/apply_plant_oversubscription_guard) -> player actions ->
// market queue -> town buys -> [market refresh, income update happen inside the market/town
// hook] -> per-step decay -> day refresh (mechanical, then stochastic) on the day boundary ->
// advance step.
// ---------------------------------------------------------------------------------------------
struct TurnInput {
    UnitAction unit_actions[N_PLAYERS][MAX_UNITS_PER_PLAYER]{};  // index 0 = farmer
    int n_units_acting[N_PLAYERS] = {1, 1};                      // how many of the above are live
    MarketOrder market_orders[N_PLAYERS][MAX_MARKET_ORDERS]{};
    int n_market_orders[N_PLAYERS] = {0, 0};
};

void step_turn(GameState& state, const TurnInput& input, MarketAndTownHook market_and_town = default_market_and_town_hook,
               WeedSpawnHook weed_spawn = default_weed_spawn_hook, ShopUnlockHook shop_unlock = default_shop_unlock_hook);

// ---------------------------------------------------------------------------------------------
// Construction (vendor's _new_farm / _new_private / _initialize's board-only half -- market/town
// init is 008's, called separately by whatever composes this with issue 008's state).
// ---------------------------------------------------------------------------------------------
Farm init_farm(int board_size, double starting_money);
Private init_private();
GameState init_game_state(int board_size, int turns_per_day, int shed_capacity, double starting_money, uint64_t seed);

}  // namespace kaggriculture::sim

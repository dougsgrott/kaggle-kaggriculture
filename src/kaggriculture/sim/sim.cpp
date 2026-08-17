// Implementation. Every function here is a direct, function-for-function port of the
// corresponding vendor/kaggriculture.py function (named in each comment) -- see sim.hpp's header
// for the scope boundary with issue 008.

#include "sim.hpp"

#include <algorithm>

namespace kaggriculture::sim {

namespace {

inline TileKind structure_kind(Structure s) { return s == Structure::COOP ? TileKind::COOP : TileKind::PASTURE; }

inline PlantTile new_plant(Crop crop, int day, int turns_per_day) {
    const CropDef& cd = CROPS[static_cast<int>(crop)];
    PlantTile t;
    t.crop = crop;
    t.planted_day = day;
    t.watered_today = false;
    t.consecutive_unwatered = 1;  // planting day counts as the first missed day
    t.yield_units = cd.ongoing ? 0 : 1;
    t.max_lifespan_step = cd.ongoing ? -1 : (day + cd.max_yield_day + 1) * turns_per_day;
    t.fertilized_until_day = -1;
    return t;
}

inline AnimalTile new_animal(Animal animal, int day) {
    AnimalTile t;
    t.occupied = true;
    t.animal = animal;
    t.placed_day = day;
    t.yield_units = 0;
    t.consecutive_unfed = 0;
    t.fed_today = false;
    t.cared_today = false;
    t.fertilizer_available = false;
    t.pending_care_bonus = 0;
    return t;
}

// True iff `tile` is a coop/pasture with an animal on it -- vendor's
// `isinstance(tile, dict) and "animal" in tile`.
inline bool is_occupied_structure(const Tile& tile) {
    return (tile.kind == TileKind::COOP || tile.kind == TileKind::PASTURE) && tile.animal.occupied;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Board geometry (_quadrant_of / _shed_access_tiles / _is_shed_adjacent / _default_spawn)
// ---------------------------------------------------------------------------------------------

int quadrant_of(int x, int y, int board_size) {
    int half = board_size / 2;
    bool north = y < half;
    bool west = x < half;
    if (north) return west ? 0 : 1;   // NW : NE
    return west ? 2 : 3;               // SW : SE
}

void shed_access_tiles(int board_size, Position out[4]) {
    int half = board_size / 2;
    out[0] = {static_cast<int8_t>(half - 1), static_cast<int8_t>(half - 1)};  // NW-adjacent
    out[1] = {static_cast<int8_t>(half), static_cast<int8_t>(half - 1)};      // NE-adjacent
    out[2] = {static_cast<int8_t>(half - 1), static_cast<int8_t>(half)};      // SW-adjacent
    out[3] = {static_cast<int8_t>(half), static_cast<int8_t>(half)};          // SE-adjacent
}

bool is_shed_adjacent(Position p, int board_size) {
    Position access[4];
    shed_access_tiles(board_size, access);
    for (const auto& a : access) {
        if (a == p) return true;
    }
    return false;
}

Position default_spawn(int board_size) {
    // shed_access_tiles()[0] is always (half-1, half-1), which is always in NW by construction
    // (x=half-1<half and y=half-1<half), matching vendor's NWSE-first-in-NW search exactly.
    Position access[4];
    shed_access_tiles(board_size, access);
    return access[0];
}

Position spawn_hand_position(const Farm& farm, int board_size) {
    Position access[4];
    shed_access_tiles(board_size, access);
    int occupancy[4] = {0, 0, 0, 0};
    for (int u = 0; u < farm.n_units; u++) {
        for (int i = 0; i < 4; i++) {
            if (access[i] == farm.unit_pos[u]) {
                occupancy[i]++;
                break;
            }
        }
    }
    int best = 0;
    for (int i = 1; i < 4; i++) {
        if (occupancy[i] < occupancy[best]) best = i;
    }
    return access[best];
}

// ---------------------------------------------------------------------------------------------
// Construction (_new_farm / _new_private)
// ---------------------------------------------------------------------------------------------

Farm init_farm(int board_size, double starting_money) {
    Farm farm;
    farm.money = starting_money;
    for (int y = 0; y < board_size; y++) {
        for (int x = 0; x < board_size; x++) {
            farm.tiles[y][x].kind = quadrant_of(x, y, board_size) == 0 ? TileKind::EMPTY : TileKind::LOCKED;
        }
    }
    farm.unit_pos[0] = default_spawn(board_size);
    farm.n_units = 1;
    farm.unlocked_quadrant[0] = true;
    farm.unlocked_quadrant[1] = farm.unlocked_quadrant[2] = farm.unlocked_quadrant[3] = false;
    farm.hires_today = 0;
    return farm;
}

Private init_private() { return Private{}; }  // shed/seeds/inventory all zero-initialized

GameState init_game_state(int board_size, int turns_per_day, int shed_capacity, double starting_money, uint64_t seed) {
    GameState state;
    state.board_size = board_size;
    state.turns_per_day = turns_per_day;
    state.shed_capacity = shed_capacity;
    state.step = 0;
    state.seed = seed;
    for (int p = 0; p < N_PLAYERS; p++) {
        state.farms[p] = init_farm(board_size, starting_money);
        state.privates[p] = init_private();
    }
    return state;
}

// ---------------------------------------------------------------------------------------------
// Per-unit action application (_apply_unit_action)
// ---------------------------------------------------------------------------------------------

void apply_unit_action(Farm& farm, Private& priv, int unit_idx, UnitAction action, int board_size, int day,
                        int turns_per_day, int shed_capacity) {
    if (unit_idx < 0 || unit_idx >= farm.n_units) return;  // vendor: _farmer_position returns None
    Position pos = farm.unit_pos[unit_idx];
    int fx = pos.x, fy = pos.y;
    int* inv = priv.inventory[unit_idx];

    if (is_move(action.op)) {
        Delta d = move_delta(action.op);
        int nx = fx + d.dx, ny = fy + d.dy;
        if (nx < 0 || nx >= board_size || ny < 0 || ny >= board_size) return;
        farm.unit_pos[unit_idx] = {static_cast<int8_t>(nx), static_cast<int8_t>(ny)};
        return;
    }

    if (action.op == Op::PASS) return;

    Tile& tile = farm.tiles[fy][fx];

    // Shed operations resolve before the LOCKED guard -- see vendor's comment at the same point.
    if (action.op == Op::DROP) {
        if (!is_shed_adjacent(pos, board_size)) return;
        // KNOWN PARITY RISK (flag for issue 010): vendor iterates `inv.items()` in Python dict
        // insertion order (the order items were first added since the last DROP/day-start), not
        // a fixed item order. That only changes behaviour when the shed is full enough for
        // partial-room allocation to matter, but when it does, this fixed WHEAT..SHEEP order can
        // award the last few slots to a different item than vendor would. Get this from the
        // parity trace (issue 010) before trusting shed-capacity-constrained trajectories.
        for (int i = 0; i < N_ITEMS; i++) {
            if (inv[i] <= 0) continue;
            int current = 0;
            for (int j = 0; j < N_ITEMS; j++) current += priv.shed[j];
            int room = std::max(0, shed_capacity - current);
            int take = std::min(inv[i], room);
            if (take > 0) priv.shed[i] += take;
            inv[i] = 0;  // vendor deletes the inventory entry unconditionally, discarding overflow
        }
        return;
    }

    if (action.op == Op::PICKUP) {
        if (!is_shed_adjacent(pos, board_size)) return;
        int i = static_cast<int>(action.item);
        int n = std::min(action.n, priv.shed[i]);
        if (n <= 0) return;
        priv.shed[i] -= n;
        inv[i] += n;
        return;
    }

    if (action.op == Op::PLACE) {
        Item item = action.item;
        // Animal placement: standing on a matching unoccupied structure.
        if (is_animal_item(item)) {
            Animal a = static_cast<Animal>(animal_index(item));
            Structure required = static_cast<Structure>(ANIMALS[static_cast<int>(a)].structure);
            if (tile.kind == structure_kind(required) && !tile.animal.occupied) {
                int i = static_cast<int>(item);
                if (inv[i] >= 1) {
                    inv[i] -= 1;
                    tile = Tile{};
                    tile.kind = structure_kind(required);
                    tile.animal = new_animal(a, day);
                }
                return;
            }
        }
        // Shed drop: orthogonally adjacent to the shed; obeys shedCapacity.
        if (is_shed_adjacent(pos, board_size)) {
            int i = static_cast<int>(item);
            int n = std::min(action.n, inv[i]);
            if (n <= 0) return;
            int current = 0;
            for (int j = 0; j < N_ITEMS; j++) current += priv.shed[j];
            int room = std::max(0, shed_capacity - current);
            n = std::min(n, room);
            if (n <= 0) return;
            inv[i] -= n;
            priv.shed[i] += n;
        }
        return;
    }

    // Everything below mutates the tile the unit stands on, so it requires that tile to be
    // owned (not LOCKED). An already-EMPTY tile is fine -- EMPTY and LOCKED are distinct kinds.
    if (tile.kind == TileKind::LOCKED) return;

    if (action.op == Op::PLANT) {
        if (!is_crop_item(action.item)) return;
        if (tile.kind != TileKind::EMPTY) return;
        int c = crop_index(action.item);
        if (priv.seeds[c] <= 0) return;
        priv.seeds[c] -= 1;
        tile = Tile{};
        tile.kind = TileKind::PLANT;
        tile.plant = new_plant(static_cast<Crop>(c), day, turns_per_day);
        return;
    }

    if (action.op == Op::WATER) {
        if (tile.kind != TileKind::PLANT) return;
        if (tile.plant.watered_today) return;
        tile.plant.watered_today = true;
        const CropDef& cd = CROPS[static_cast<int>(tile.plant.crop)];
        if (!cd.ongoing) {
            int age_days = day - tile.plant.planted_day;
            int window_start = (cd.max_yield_day + 1) / 2;
            if (age_days >= window_start && age_days <= cd.max_yield_day) {
                int bonus = tile.plant.fertilized_until_day >= day ? 2 : 1;
                tile.plant.yield_units = std::min(cd.max_yield, tile.plant.yield_units + bonus);
            }
        }
        return;
    }

    if (action.op == Op::HARVEST) {
        if (tile.kind == TileKind::PLANT) {
            if (tile.plant.yield_units <= 0) return;
            const CropDef& cd = CROPS[static_cast<int>(tile.plant.crop)];
            if (day - tile.plant.planted_day < cd.first_yield_day) return;  // immature; no-op
            int units = tile.plant.yield_units;
            int item = crop_index(item_of_crop(tile.plant.crop));
            tile.plant.yield_units = 0;
            inv[item] += units;
            if (!cd.ongoing) tile = Tile{};  // becomes EMPTY
        } else if (is_occupied_structure(tile)) {
            if (tile.animal.yield_units <= 0) return;
            const AnimalDef& ad = ANIMALS[static_cast<int>(tile.animal.animal)];
            int units = tile.animal.yield_units;
            tile.animal.yield_units = 0;
            inv[ad.product_item] += units;
        }
        return;
    }

    if (action.op == Op::FERTILIZE) {
        if (tile.kind != TileKind::PLANT) return;
        int fert = static_cast<int>(Item::FERTILIZER);
        if (inv[fert] < 1) return;
        inv[fert] -= 1;
        tile.plant.fertilized_until_day = std::max(tile.plant.fertilized_until_day, day + 2);
        return;
    }

    if (action.op == Op::DIG) {
        if (tile.kind == TileKind::EMPTY) return;
        if (is_occupied_structure(tile)) return;
        tile = Tile{};  // becomes EMPTY (clears PLANT, WEED, or an empty coop/pasture)
        return;
    }

    if (action.op == Op::BUILD_COOP) {
        if (tile.kind != TileKind::EMPTY) return;
        tile = Tile{};
        tile.kind = TileKind::COOP;
        return;
    }

    if (action.op == Op::BUILD_PASTURE) {
        if (tile.kind != TileKind::EMPTY) return;
        tile = Tile{};
        tile.kind = TileKind::PASTURE;
        return;
    }

    if (action.op == Op::FEED) {
        if (!is_occupied_structure(tile)) return;
        if (tile.animal.fed_today) return;
        int wheat = static_cast<int>(Item::WHEAT);
        if (inv[wheat] < 1) return;
        inv[wheat] -= 1;
        tile.animal.fed_today = true;
        return;
    }

    if (action.op == Op::COLLECT_FERTILIZER) {
        if (!is_occupied_structure(tile)) return;
        if (!tile.animal.fertilizer_available) return;
        tile.animal.fertilizer_available = false;
        inv[static_cast<int>(Item::FERTILIZER)] += 1;
        return;
    }

    if (action.op == Op::CARE) {
        if (!is_occupied_structure(tile)) return;
        if (tile.animal.cared_today) return;
        tile.animal.cared_today = true;
        return;
    }
}

void apply_plant_oversubscription_guard(UnitAction actions[MAX_UNITS_PER_PLAYER], int n_units, const Private& priv) {
    int demand[N_CROPS] = {0};
    for (int u = 0; u < n_units; u++) {
        if (actions[u].op == Op::PLANT && is_crop_item(actions[u].item)) {
            demand[crop_index(actions[u].item)]++;
        }
    }
    bool blocked[N_CROPS];
    for (int c = 0; c < N_CROPS; c++) blocked[c] = demand[c] > priv.seeds[c];
    for (int u = 0; u < n_units; u++) {
        if (actions[u].op == Op::PLANT && is_crop_item(actions[u].item) && blocked[crop_index(actions[u].item)]) {
            actions[u].op = Op::PASS;
        }
    }
}

void apply_all_unit_actions(GameState& state, const TurnInput& input, int day) {
    for (int p = 0; p < N_PLAYERS; p++) {
        Farm& farm = state.farms[p];
        Private& priv = state.privates[p];
        int n = input.n_units_acting[p];
        UnitAction actions[MAX_UNITS_PER_PLAYER];
        for (int u = 0; u < n; u++) actions[u] = input.unit_actions[p][u];
        apply_plant_oversubscription_guard(actions, n, priv);
        for (int u = 0; u < n; u++) {
            apply_unit_action(farm, priv, u, actions[u], state.board_size, day, state.turns_per_day, state.shed_capacity);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Per-step and day-refresh mechanics
// ---------------------------------------------------------------------------------------------

void decay_plants(Farm& farm, int step) {
    for (int y = 0; y < MAX_BOARD; y++) {
        for (int x = 0; x < MAX_BOARD; x++) {
            Tile& tile = farm.tiles[y][x];
            if (tile.kind != TileKind::PLANT) continue;
            int mls = tile.plant.max_lifespan_step;
            if (mls < 0 || step < mls) continue;
            if ((step - mls) % 2 != 0) continue;
            tile.plant.yield_units -= 1;
            if (tile.plant.yield_units <= 0) {
                tile = Tile{};
                tile.kind = TileKind::WEED;
            }
        }
    }
}

void daily_refresh_plants(Farm& farm, int current_day, int turns_per_day) {
    int next_day = current_day + 1;
    for (int y = 0; y < MAX_BOARD; y++) {
        for (int x = 0; x < MAX_BOARD; x++) {
            Tile& tile = farm.tiles[y][x];
            if (tile.kind != TileKind::PLANT) continue;
            bool was_watered = tile.plant.watered_today;
            tile.plant.consecutive_unwatered = was_watered ? 0 : tile.plant.consecutive_unwatered + 1;
            tile.plant.watered_today = false;
            if (tile.plant.consecutive_unwatered >= 2) {
                tile = Tile{};
                tile.kind = TileKind::WEED;
                continue;
            }
            const CropDef& cd = CROPS[static_cast<int>(tile.plant.crop)];
            if (!cd.ongoing) continue;
            int days_since_first = next_day - tile.plant.planted_day - cd.first_yield_day;
            if (days_since_first < 0) continue;
            if (days_since_first % cd.interval != 0) continue;
            int production_count = days_since_first / cd.interval + 1;
            if (production_count > cd.max_yield) continue;
            bool fertilized = was_watered && tile.plant.fertilized_until_day >= current_day;
            tile.plant.yield_units = std::min(cd.max_yield, tile.plant.yield_units + (fertilized ? 2 : 1));
            if (production_count == cd.max_yield) {
                tile.plant.max_lifespan_step = (next_day + 1) * turns_per_day;
            }
        }
    }
}

void daily_refresh_animals(Farm& farm, int day) {
    int next_day = day + 1;
    for (int y = 0; y < MAX_BOARD; y++) {
        for (int x = 0; x < MAX_BOARD; x++) {
            Tile& tile = farm.tiles[y][x];
            if (!is_occupied_structure(tile)) continue;
            AnimalTile& a = tile.animal;
            a.consecutive_unfed = a.fed_today ? 0 : a.consecutive_unfed + 1;
            if (a.consecutive_unfed >= 2) {
                TileKind structure = tile.kind;  // animal escapes; structure remains
                tile = Tile{};
                tile.kind = structure;
                continue;
            }
            const AnimalDef& ad = ANIMALS[static_cast<int>(a.animal)];
            int days_since_first = next_day - a.placed_day - ad.first_yield_day;
            if (days_since_first >= 0 && days_since_first % ad.interval == 0) {
                int bonus = a.fed_today ? a.pending_care_bonus : 0;
                a.yield_units = std::min(ad.max_held, a.yield_units + 1 + bonus);
                a.pending_care_bonus = 0;
            }
            if (a.cared_today && a.fed_today) a.pending_care_bonus += 1;
            a.fertilizer_available = true;
            a.fed_today = false;
            a.cared_today = false;
        }
    }
}

void drop_inventories_to_shed(Private& priv, int n_units, int shed_capacity) {
    // Same known parity risk as Op::DROP above: vendor processes each unit's items in dict
    // insertion order, this in fixed item order. Only matters when the shed is capacity-bound.
    for (int u = 0; u < n_units; u++) {
        for (int i = 0; i < N_ITEMS; i++) {
            int n = priv.inventory[u][i];
            if (n <= 0) continue;
            int current = 0;
            for (int j = 0; j < N_ITEMS; j++) current += priv.shed[j];
            int room = std::max(0, shed_capacity - current);
            int take = std::min(n, room);
            if (take > 0) priv.shed[i] += take;
            priv.inventory[u][i] = 0;  // overflow beyond shed capacity is discarded, unconditionally
        }
    }
}

void end_of_day_mechanical(Farm& farm, Private& priv, int day, int board_size, int turns_per_day, int shed_capacity) {
    daily_refresh_plants(farm, day, turns_per_day);
    daily_refresh_animals(farm, day);
    drop_inventories_to_shed(priv, farm.n_units, shed_capacity);
    farm.unit_pos[0] = default_spawn(board_size);
    farm.n_units = 1;
    farm.hires_today = 0;
}

// ---------------------------------------------------------------------------------------------
// The seam: no-op defaults so this file compiles and runs standalone (see sim.hpp).
// ---------------------------------------------------------------------------------------------

void default_market_and_town_hook(GameState&, const MarketOrder[][MAX_MARKET_ORDERS], const int[], int) {}
void default_weed_spawn_hook(GameState&, int, int) {}
void default_shop_unlock_hook(GameState&, int) {}

// ---------------------------------------------------------------------------------------------
// Turn loop (interpreter())
// ---------------------------------------------------------------------------------------------

void step_turn(GameState& state, const TurnInput& input, MarketAndTownHook market_and_town, WeedSpawnHook weed_spawn,
               ShopUnlockHook shop_unlock) {
    int day = state.step / state.turns_per_day;

    apply_all_unit_actions(state, input, day);

    market_and_town(state, input.market_orders, input.n_market_orders, state.step);

    for (int p = 0; p < N_PLAYERS; p++) {
        decay_plants(state.farms[p], state.step);
    }

    if ((state.step + 1) % state.turns_per_day == 0) {
        for (int p = 0; p < N_PLAYERS; p++) {
            end_of_day_mechanical(state.farms[p], state.privates[p], day, state.board_size, state.turns_per_day,
                                   state.shed_capacity);
            weed_spawn(state, p, day);
        }
        shop_unlock(state, day);
    }

    state.step += 1;
}

}  // namespace kaggriculture::sim

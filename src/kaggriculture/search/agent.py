"""Expands a search.schedule.Schedule into a live `agent(observation, configuration)` callable
(issue 013's "movement-aware expansion"), in the same real-engine action-dict shape as
kaggriculture.agents.baseline -- runnable directly through kaggriculture.eval.agents.wrap_agent,
or recorded once into a native TapePolicy (see resolve_policy's "greedy" branch) for the fast path.

Per-tile mechanics (when to WATER/HARVEST/DIG, PICKUP-before-FEED for animals, the sell floor) are
generalizations of issue 005 baseline.py's melon/goose-specific logic to an arbitrary
`Schedule.tile_role`, not a rewrite -- baseline's execution was already sound, issue 013's real
contribution is *what* to grow where (search.schedule's price-aware ranking) and *where a hand's
tiles live in the visit order* (the snake ordering + contiguous-chunk partition below), not a new
per-turn decision procedure.
"""

from __future__ import annotations

import math
from typing import Callable

from kaggriculture.model.constants import ANIMALS, CROPS, MARKET_PARAMS
from kaggriculture.model.economics import hand_cost, land_unlock_cost
from kaggriculture.model.price import units_sellable_before
from kaggriculture.model.yields import ongoing_crop_production_days
from kaggriculture.search.schedule import CASH_RESERVE, quadrant_tiles, shed_adjacent_tile

SELL_FLOOR_FRACTION = 0.5  # never sell a unit for less than half its base price -- baseline's rule
WHEAT_BUFFER_PER_ANIMAL = 3
MAX_SEED_QTY_PER_ORDER = 3  # throttles cash burn (see _market_orders' seed-buying step)
QUADRANT_NAMES = ("NW", "NE", "SW", "SE")


def _sell_floor(item: str) -> int:
    return max(1, round(MARKET_PARAMS[item]["base"] * SELL_FLOOR_FRACTION))


def _move_toward(pos: tuple[int, int], target: tuple[int, int]) -> list[str]:
    x, y = pos
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _crop_harvest_ready(crop: str, tile: dict, day: int) -> bool:
    cd = CROPS[crop]
    if cd["ongoing"]:
        return tile.get("yield_units", 0) > 0
    age = day - tile["planted_day"]
    return age >= cd["max_yield_day"] or tile.get("yield_units", 0) >= cd["max_yield"]


def _crop_exhausted(crop: str, tile: dict, day: int) -> bool:
    """True for an ongoing crop past its last scheduled production with nothing left to harvest
    -- HARVEST never frees an ongoing-crop tile (sim.cpp), so DIG is the only way to reclaim it
    for the schedule's next planting."""
    cd = CROPS[crop]
    if not cd["ongoing"]:
        return False
    last_day = ongoing_crop_production_days(crop, tile["planted_day"])[-1]
    return day > last_day and tile.get("yield_units", 0) == 0


def _crop_tile_target_action(role: str, tile, day: int, seeds_available: int, hour: int, turns_per_day: int) -> str | None:
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if tile["crop"] != role or _crop_exhausted(role, tile, day):
            return "DIG"  # wrong crop (shouldn't happen by construction) or done producing
        if _crop_harvest_ready(role, tile, day):
            return "HARVEST"
        if not tile["watered_today"]:
            return "WATER"
        return None
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return "DIG"
    if tile is None:
        return "PLANT" if seeds_available > 0 and hour < turns_per_day - 1 else None
    return None  # LOCKED, or an unexpected occupant -- don't fight it


def _animal_tile_target_action(role: str, tile, inv: dict, shed: dict) -> tuple[str, str | None] | None:
    structure = ANIMALS[role]["structure"]
    if tile is None:
        return ("BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE", None)
    if not isinstance(tile, dict) or tile.get("kind") not in ("COOP", "PASTURE"):
        return None
    if "animal" not in tile:
        if inv.get(role, 0) > 0:
            return ("PLACE", role)
        if shed.get(role, 0) > 0:
            return ("PICKUP", role)
        return None
    if tile["yield_units"] > 0:
        return ("HARVEST", None)
    if not tile["fed_today"]:
        if inv.get("WHEAT", 0) > 0:
            return ("FEED", None)
        if shed.get("WHEAT", 0) > 0:
            return ("PICKUP", "WHEAT")
        return None
    if not tile["cared_today"]:
        return ("CARE", None)
    return None


def _chunks(items: list, n_workers: int) -> list[list]:
    """Contiguous slices of `items` (already snake-ordered by search.schedule.quadrant_tiles) --
    unlike a round-robin/modulo partition, each worker's slice is a short run, not scattered
    across the whole board."""
    if n_workers <= 0:
        return []
    size = math.ceil(len(items) / n_workers)
    return [items[i * size : (i + 1) * size] for i in range(n_workers)]


def schedule_agent(schedule, config: dict) -> Callable[[dict, dict], dict]:
    """Builds the `agent(observation, configuration) -> action` callable that executes `schedule`.
    Tile roles are looked up from the schedule (static, decided once at plan time); crew/land
    timing come from its calendars; everything else (watering cadence, sell floor, wheat buffer)
    is generic per-role logic, not schedule-specific state."""
    crop_tiles = [pos for q in range(4) for pos in quadrant_tiles(q) if schedule.tile_kind.get(pos) == "crop"]
    animal_tiles = [shed_adjacent_tile(q) for q in range(4) if schedule.tile_kind.get(shed_adjacent_tile(q)) == "animal"]
    farmer_does_herd = len(animal_tiles) > 0
    hand_cost_mult = config.get("farmHandCostMult", 1)
    max_orders = config.get("maxMarketOrdersPerTurn", 10)

    def _scan_order(pos, my_tiles):
        """Indices into `my_tiles`, starting at the unit's own position (if assigned one) and
        wrapping forward -- not always from index 0. A hand whose front tiles need attention
        every single visit (dense enough maintenance load) would otherwise never advance past
        them and could go its whole assignment without ever reaching the tiles further down the
        list; scanning-from-here-forward instead sweeps the whole assignment over successive
        turns, since a tile just serviced this turn (e.g. watered_today) drops out of contention
        for the rest of the day and lets the scan move on."""
        n = len(my_tiles)
        start = my_tiles.index(tuple(pos)) if tuple(pos) in my_tiles else 0
        return (my_tiles[(start + offset) % n] for offset in range(n))

    def _unit_crop_action(pos, my_tiles, farm_tiles, seeds, day, hour, turns_per_day):
        if not my_tiles:
            return ["PASS"]
        for tx, ty in _scan_order(pos, my_tiles):
            role = schedule.tile_role.get((tx, ty))
            if role is None:
                continue
            action = _crop_tile_target_action(role, farm_tiles[ty][tx], day, seeds.get(role, 0), hour, turns_per_day)
            if action is None:
                continue
            if tuple(pos) == (tx, ty):
                return ["PLANT", role] if action == "PLANT" else [action]
            return _move_toward(pos, (tx, ty))
        return ["PASS"]

    def _unit_herd_action(pos, my_tiles, farm_tiles, inv, shed):
        if not my_tiles:
            return ["PASS"]
        for tx, ty in _scan_order(pos, my_tiles):
            role = schedule.tile_role.get((tx, ty))
            if role is None:
                continue
            found = _animal_tile_target_action(role, farm_tiles[ty][tx], inv, shed)
            if found is None:
                continue
            op, item = found
            if tuple(pos) == (tx, ty):
                return [op, item, 1] if op == "PICKUP" else ([op, item] if item else [op])
            return _move_toward(pos, (tx, ty))
        return ["PASS"]

    def _market_orders(farm, private, market, day):
        # Priority order mirrors search.schedule.build_schedule's own commit order (land, animal
        # structures, crew, tiles) -- on a money-constrained turn/day, execution should run out of
        # cash on the same things the plan considered lowest-priority, not on whatever this
        # function happened to list first.
        orders = []
        money = farm["money"]

        for extra_idx in sorted(schedule.land_days):
            if len(orders) >= max_orders:
                break
            if day < schedule.land_days[extra_idx]:
                continue
            quadrant_name = QUADRANT_NAMES[extra_idx + 1]
            if quadrant_name in farm["unlocked_quadrants"]:
                continue
            cost = land_unlock_cost(extra_idx)
            if money - cost >= CASH_RESERVE:
                orders.append(["BUY_LAND"])
                money -= cost

        # Animals: buy one for each structure that doesn't have one yet and isn't already owned.
        # Skip sites whose quadrant isn't unlocked yet (tile == "LOCKED") -- otherwise this buys
        # every planned animal on day 0 regardless of whether its pasture/coop can exist yet.
        for x, y in animal_tiles:
            if len(orders) >= max_orders:
                break
            tile = farm["tiles"][y][x]
            if isinstance(tile, str):  # "LOCKED"
                continue
            role = schedule.tile_role[(x, y)]
            has_animal = isinstance(tile, dict) and "animal" in tile
            owns_unplaced = private["shed"].get(role, 0) > 0 or any(inv.get(role, 0) > 0 for inv in private["inventories"])
            if not has_animal and not owns_unplaced and money - ANIMALS[role]["cost"] >= CASH_RESERVE:
                orders.append(["BUY_ANIMAL", role, 1])
                money -= ANIMALS[role]["cost"]

        target_hands = schedule.target_hands(day)
        hires_today = farm["hires_today"]
        n_hands = len(farm["hands"])
        while len(orders) < max_orders and n_hands < target_hands:
            cost = hand_cost(hires_today, hand_cost_mult)
            if cost > money:
                break
            orders.append(["HIRE"])
            money -= cost
            n_hands += 1
            hires_today += 1

        # Seeds: keep enough in stock for every currently-empty tile assigned to that crop -- but
        # not before the schedule's own first-assignment day for that tile, or this buys seeds for
        # the whole season's tiles on day 0 regardless of how small the day-0 crew actually is
        # (the pacing search.schedule's new_plant_budget was built to enforce).
        empty_by_role: dict[str, int] = {}
        for x, y in crop_tiles:
            role = schedule.tile_role.get((x, y))
            if role and farm["tiles"][y][x] is None and day >= schedule.tile_first_assigned_day.get((x, y), 0):
                empty_by_role[role] = empty_by_role.get(role, 0) + 1
        for role, n_empty in empty_by_role.items():
            if len(orders) >= max_orders:
                break
            shortfall = n_empty - private["seeds"].get(role, 0)
            if shortfall > 0 and money >= CROPS[role]["seed"]:
                affordable = int(money // CROPS[role]["seed"])
                # Capped per turn (not just per order): a single BUY_SEED order can name any
                # quantity, so an uncapped shortfall pays for the WHOLE season's tiles in one
                # order regardless of how many orders/turn are allowed -- unlike HIRE, which is
                # naturally throttled at 1/order. Spreading it over many turns instead matches how
                # fast the seeds actually get planted (a handful of tiles per turn, not all of them).
                qty = min(shortfall, affordable, MAX_SEED_QTY_PER_ORDER)
                if qty > 0:
                    orders.append(["BUY_SEED", role, qty])
                    money -= qty * CROPS[role]["seed"]

        if len(orders) < max_orders and animal_tiles and private["shed"].get("WHEAT", 0) < WHEAT_BUFFER_PER_ANIMAL * len(animal_tiles) and money > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", 1])

        for item, qty in private["shed"].items():
            if len(orders) >= max_orders:
                break
            if qty <= 0 or item not in MARKET_PARAMS or (item == "WHEAT" and animal_tiles):
                continue
            floor = _sell_floor(item)
            sellable = units_sellable_before(item, market["inventory"][item], floor)
            sell_qty = qty if sellable is None else min(qty, sellable)
            if sell_qty > 0:
                orders.append(["SELL", item, sell_qty])

        return orders[:max_orders]

    def agent(observation, configuration):
        player = observation["player"]
        farm = observation["farms"][player]
        private = observation["private"]
        market = observation["market"]
        day = observation.get("day", 0)
        hour = observation.get("hour", 0)
        turns_per_day = int(configuration.get("turnsPerDay", 24))

        n_hands = len(farm["hands"])
        n_crop_workers = n_hands if farmer_does_herd else n_hands + 1
        crop_chunks = _chunks(crop_tiles, max(1, n_crop_workers))

        hands_actions = []
        for i, pos in enumerate(farm["hands"]):
            my_tiles = crop_chunks[i] if i < len(crop_chunks) else []
            hands_actions.append(_unit_crop_action(tuple(pos), my_tiles, farm["tiles"], private["seeds"], day, hour, turns_per_day))

        if farmer_does_herd:
            farmer_inv = private["inventories"][0] if private["inventories"] else {}
            farmer_action = _unit_herd_action(tuple(farm["farmer"]), animal_tiles, farm["tiles"], farmer_inv, private["shed"])
        else:
            my_tiles = crop_chunks[n_hands] if n_hands < len(crop_chunks) else []
            farmer_action = _unit_crop_action(tuple(farm["farmer"]), my_tiles, farm["tiles"], private["seeds"], day, hour, turns_per_day)

        market_orders = _market_orders(farm, private, market, day)

        return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}

    return agent

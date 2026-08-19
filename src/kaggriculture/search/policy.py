"""A compact parametric policy (issue 015): unlike 013/014's precomputed `Schedule`, this agent
decides *live*, every turn, from the actual observed market price -- no projected/committed-
production price model needed (013's `_SEASON_ABSORPTION` credit was a workaround for not having
this). That also means it isn't limited to one-time crops the way 013 was forced to be (see 013's
Revision section on ongoing crops): since seed purchases are gated by real observed money, not an
optimistic planning ledger, TOMATO/STRAWBERRY are back on the table as ordinary candidates.

The ~20 tunable parameters below (see PARAM_SPECS) are weights and thresholds on top of the same
economics (`model.economics`/`model.price`) 013/014 already use -- CMA-ES (`search.cmaes`) tunes
them; this module only defines what they mean and how they turn into a live `agent(observation,
configuration)` callable, in the same shape as `kaggriculture.agents.baseline` and
`search.agent.schedule_agent` (so it plugs into `eval.agents.wrap_agent`/`CallbackPolicy`
unmodified).

Search space: CMA-ES searches an UNBOUNDED R^n (one coordinate per parameter); `decode_params`
squashes each coordinate through a sigmoid into that parameter's `[lo, hi]` range. This sidesteps
depending on a specific CMA-ES library's box-constraint semantics, and it's what makes "wide
initial sigma" (this issue's own scope bullet, and the public reference notebook's title) do the
right thing: a wide sigma in the raw sigmoid-domain explores close to *both* ends of every bounded
range, not just near the midpoint.
"""

from __future__ import annotations

import math
from typing import Callable

from kaggriculture.model.constants import ANIMALS, CROPS, MARKET_PARAMS
from kaggriculture.model.economics import hand_cost, land_unlock_cost, one_time_crop_profit_per_action, ongoing_crop_profit_per_action, animal_profit_per_action
from kaggriculture.model.price import units_sellable_before
from kaggriculture.model.yields import ongoing_crop_production_days
from kaggriculture.search.agent import _move_toward
from kaggriculture.search.schedule import quadrant_tiles, shed_adjacent_tile

CROP_NAMES = list(CROPS)  # all 5 -- WHEAT, CARROT, TOMATO, STRAWBERRY, MELON
ANIMAL_NAMES = list(ANIMALS)
QUADRANT_NAMES = ("NW", "NE", "SW", "SE")
MAX_SEED_QTY_PER_ORDER = 3  # same cash-burn throttle 013 found necessary (see its Revision)
TILES_PER_HAND_CAP = 5  # total (planted + seeded) tiles per hand, fixed (not CMA-ES-tuned) --
# a structural safety valve, not a behavioral choice; see the seed-buying step's comment

# (name, lo, hi, default). Defaults are informed by 013/014's own findings where relevant (e.g.
# max_hands=8 matching issue 005 baseline's own crew size), not arbitrary midpoints -- CMA-ES's
# x0 is these defaults' sigmoid-inverse, so the search starts from a known-reasonable point.
PARAM_SPECS: list[tuple[str, float, float, float]] = [
    ("crop_weight_WHEAT", 0.1, 3.0, 1.0),
    ("crop_weight_CARROT", 0.1, 3.0, 1.0),
    ("crop_weight_MELON", 0.1, 3.0, 1.0),
    ("crop_weight_TOMATO", 0.1, 3.0, 1.0),
    ("crop_weight_STRAWBERRY", 0.1, 3.0, 1.0),
    ("animal_weight_GOOSE", 0.1, 3.0, 1.0),
    ("animal_weight_COW", 0.1, 3.0, 1.0),
    ("animal_weight_SHEEP", 0.1, 3.0, 1.0),
    ("max_hands", 1.0, 20.0, 8.0),
    ("hand_value_scale", 1.0, 20.0, 8.0),
    ("cash_reserve", 0.0, 2800.0, 2200.0),
    ("land_aggressiveness", 0.0, 2.0, 1.0),
    ("sell_floor_fraction", 0.0, 1.0, 0.5),
    ("fertilizer_collect_priority", 0.0, 1.0, 0.3),
    ("liquidation_day", 15.0, 30.0, 28.0),
    ("wheat_buffer_per_animal", 0.0, 10.0, 3.0),
    ("seed_stock_target", 1.0, 10.0, 3.0),
    ("animal_build_day_threshold", 0.0, 20.0, 0.0),
    ("care_priority", 0.0, 1.0, 1.0),
    ("crop_saturation_rate", 0.0, 0.5, 0.08),
]
PARAM_NAMES = [spec[0] for spec in PARAM_SPECS]
N_PARAMS = len(PARAM_SPECS)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _inverse_sigmoid(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def default_x0() -> list[float]:
    """The raw (pre-sigmoid) coordinates whose decode equals PARAM_SPECS' own defaults -- CMA-ES's
    starting point."""
    return [_inverse_sigmoid((default - lo) / (hi - lo)) for _, lo, hi, default in PARAM_SPECS]


def decode_params(x) -> dict[str, float]:
    """Raw R^n vector -> {name: value in [lo, hi]}."""
    return {name: lo + (hi - lo) * _sigmoid(xi) for (name, lo, hi, _default), xi in zip(PARAM_SPECS, x)}


def _count_planted(farm_tiles) -> dict[str, int]:
    """How many of OUR OWN tiles are currently growing each crop -- an observable, live proxy for
    "how committed are we already", used to self-throttle further commitment to the same crop
    (see _rank_crops). This is the reactive-policy analogue of 013's `_SEASON_ABSORPTION` credit:
    that module had to *project* future market saturation from an offline model; this one can
    just count its own board, which is exactly the information a live policy has that an offline
    planner doesn't."""
    counts: dict[str, int] = {}
    for row in farm_tiles:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                counts[tile["crop"]] = counts.get(tile["crop"], 0) + 1
    return counts


def _rank_crops(market_prices: dict, weights: dict[str, float], planted_counts: dict[str, int]) -> list[tuple[float, str]]:
    saturation_rate = weights["crop_saturation_rate"]
    ranked = []
    for crop in CROP_NAMES:
        price = market_prices[crop]
        ppa = ongoing_crop_profit_per_action(crop, price) if CROPS[crop]["ongoing"] else one_time_crop_profit_per_action(crop, price)
        adjusted = ppa / (1.0 + planted_counts.get(crop, 0) * saturation_rate)
        ranked.append((adjusted * weights[f"crop_weight_{crop}"], crop))
    ranked.sort(reverse=True)
    return ranked


def _rank_animals(market_prices: dict, weights: dict[str, float], care: bool, structures: set[str] | None = None) -> list[tuple[float, str]]:
    """`structures`, if given, restricts to animals whose structure is in that set -- once a site
    is built as a COOP or PASTURE, only animals matching that structure are eligible there, so the
    live ranking used to decide FEED/PLACE for an already-built site must respect it (the
    unconstrained ranking is only for deciding what to build in the first place, on an empty
    site -- see _animal_tile_action)."""
    ranked = []
    for animal in ANIMAL_NAMES:
        if structures is not None and ANIMALS[animal]["structure"] not in structures:
            continue
        price = market_prices[ANIMALS[animal]["product"]]
        ppa = animal_profit_per_action(animal, price, care=care)
        ranked.append((ppa * weights[f"animal_weight_{animal}"], animal))
    ranked.sort(reverse=True)
    return ranked


def _crop_harvest_ready(crop: str, tile: dict, day: int) -> bool:
    cd = CROPS[crop]
    if cd["ongoing"]:
        return tile.get("yield_units", 0) > 0
    age = day - tile["planted_day"]
    return age >= cd["max_yield_day"] or tile.get("yield_units", 0) >= cd["max_yield"]


def _crop_exhausted(crop: str, tile: dict, day: int) -> bool:
    cd = CROPS[crop]
    if not cd["ongoing"]:
        return False
    last_day = ongoing_crop_production_days(crop, tile["planted_day"])[-1]
    return day > last_day and tile.get("yield_units", 0) == 0


def _sell_floor(item: str, fraction: float) -> int:
    return max(1, round(MARKET_PARAMS[item]["base"] * fraction))


def _all_tiles_snake_order() -> list[tuple[int, int]]:
    return [pos for q in range(4) for pos in quadrant_tiles(q)]


def policy_agent(params: dict[str, float], config: dict) -> Callable[[dict, dict], dict]:
    """Builds the live `agent(observation, configuration) -> action` callable for `params`
    (a decoded dict from `decode_params`). Every decision -- crop/animal ranking, crew size, land
    timing, selling -- is recomputed each turn from the ACTUAL observation, not a precomputed
    plan; `params` only weights/gates those live decisions."""
    max_orders = config.get("maxMarketOrdersPerTurn", 10)
    hand_cost_mult = config.get("farmHandCostMult", 1)
    max_hands = max(1, round(params["max_hands"]))
    animal_sites = [shed_adjacent_tile(q) for q in range(4)]
    all_tiles = _all_tiles_snake_order()
    care = params["care_priority"] > 0.5

    def _crop_tile_action(tile, day, hour, turns_per_day, seeds, market_prices, planted_counts):
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile["crop"]
            if _crop_exhausted(crop, tile, day):
                return ("DIG", None)
            if _crop_harvest_ready(crop, tile, day):
                return ("HARVEST", None)
            if not tile["watered_today"]:
                return ("WATER", None)
            return None
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return ("DIG", None)
        if tile is None and hour < turns_per_day - 1:
            ranked = _rank_crops(market_prices, params, planted_counts)
            for _ppa, crop in ranked:
                if seeds.get(crop, 0) > 0:
                    return ("PLANT", crop)
            return None
        return None

    def _unit_crop_action(pos, my_tiles, farm_tiles, day, hour, turns_per_day, seeds, market_prices, planted_counts):
        n = len(my_tiles)
        if n == 0:
            return ["PASS"]
        start = my_tiles.index(tuple(pos)) if tuple(pos) in my_tiles else 0
        for offset in range(n):
            tx, ty = my_tiles[(start + offset) % n]
            tile = farm_tiles[ty][tx]
            found = _crop_tile_action(tile, day, hour, turns_per_day, seeds, market_prices, planted_counts)
            if found is None:
                continue
            op, item = found
            if tuple(pos) == (tx, ty):
                return [op, item] if item else [op]
            return _move_toward(pos, (tx, ty))
        return ["PASS"]

    def _animal_tile_action(tile, inv, shed, market_prices):
        # Empty site: pick a structure by the unconstrained live ranking -- this is the only
        # point that decision gets made, and it's final the moment BUILD_* is issued (the next
        # observation shows a built COOP/PASTURE, never None again for this site).
        if tile is None:
            ranked = _rank_animals(market_prices, params, care)
            if not ranked:
                return None
            structure = ANIMALS[ranked[0][1]]["structure"]
            return ("BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE", None)
        if not isinstance(tile, dict) or tile.get("kind") not in ("COOP", "PASTURE"):
            return None
        if "animal" not in tile:
            # Built but unoccupied: only animals matching THIS structure are eligible now.
            compatible = _rank_animals(market_prices, params, care, structures={tile["kind"]})
            if not compatible:
                return None
            role = compatible[0][1]
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
        fertilizer_first = params["fertilizer_collect_priority"] > 0.5
        collect_fertilizer = ("COLLECT_FERTILIZER", None) if tile.get("fertilizer_available") else None
        if fertilizer_first and collect_fertilizer:
            return collect_fertilizer
        if care and not tile["cared_today"]:
            return ("CARE", None)
        if collect_fertilizer:
            # A free byproduct of a cared-for animal (see vendor's daily refresh) -- worth an
            # action on its own even with no crop use for it: FERTILIZER sells at $100 base and
            # this cost nothing to produce (ROADMAP's "fertilizer-as-cash" tactical primitive).
            return collect_fertilizer
        return None

    def _unit_herd_action(pos, my_sites, farm_tiles, inv, shed, market_prices):
        n = len(my_sites)
        if n == 0:
            return ["PASS"]
        start = my_sites.index(tuple(pos)) if tuple(pos) in my_sites else 0
        for offset in range(n):
            tx, ty = my_sites[(start + offset) % n]
            found = _animal_tile_action(farm_tiles[ty][tx], inv, shed, market_prices)
            if found is None:
                continue
            op, item = found
            if tuple(pos) == (tx, ty):
                return [op, item, 1] if op == "PICKUP" else ([op, item] if item else [op])
            return _move_toward(pos, (tx, ty))
        return ["PASS"]

    def _market_orders(farm, private, market, day, n_days, planted_counts):
        orders: list[list] = []
        money = farm["money"]
        n_hands = len(farm["hands"])
        unlocked = farm["unlocked_quadrants"]
        liquidating = day >= params["liquidation_day"]

        # Land: buy the next quadrant once a reserve-adjusted budget allows it -- skipped once
        # liquidating (no point buying land with no season left to work it).
        if not liquidating:
            for extra_idx in range(3):
                if len(orders) >= max_orders:
                    break
                quadrant_name = QUADRANT_NAMES[extra_idx + 1]
                if quadrant_name in unlocked:
                    continue
                cost = land_unlock_cost(extra_idx)
                required_reserve = params["cash_reserve"] * params["land_aggressiveness"]
                if money - cost >= required_reserve:
                    orders.append(["BUY_LAND"])
                    money -= cost
                break  # only ever the NEXT quadrant in order

        # Animals: one per unlocked, BUILT-but-unoccupied shed-adjacent site (a still-empty site
        # hasn't picked a structure type yet -- that happens when a herd unit actually visits it,
        # see _animal_tile_action -- so there's nothing safe to buy for it until then), ranked by
        # live ppa among animals compatible with that site's actual structure.
        if not liquidating and day >= params["animal_build_day_threshold"]:
            for q, pos in enumerate(animal_sites):
                if len(orders) >= max_orders:
                    break
                if QUADRANT_NAMES[q] not in unlocked:
                    continue
                x, y = pos
                tile = farm["tiles"][y][x]
                if not isinstance(tile, dict) or tile.get("kind") not in ("COOP", "PASTURE"):
                    continue
                if "animal" in tile:
                    continue
                compatible = _rank_animals(market["prices"], params, care, structures={tile["kind"]})
                compatible_animals = {a for _, a in compatible}
                owns_unplaced = any(
                    private["shed"].get(a, 0) > 0 or any(inv.get(a, 0) > 0 for inv in private["inventories"]) for a in compatible_animals
                )
                if owns_unplaced:
                    continue
                for _ppa, animal in compatible:
                    if money - ANIMALS[animal]["cost"] >= params["cash_reserve"]:
                        orders.append(["BUY_ANIMAL", animal, 1])
                        money -= ANIMALS[animal]["cost"]
                        break

        # Crew: hire up to max_hands while the marginal hand still clears its (recurring, see
        # 013's Revision on hands-as-day-labour) fib-priced daily cost.
        ranked_crops = _rank_crops(market["prices"], params, planted_counts)
        best_ppa = max((0.0, *(ppa for ppa, _ in ranked_crops)))
        target_hands = 0
        while target_hands < max_hands and hand_cost(target_hands, hand_cost_mult) < best_ppa * params["hand_value_scale"]:
            target_hands += 1
        if liquidating:
            target_hands = min(target_hands, n_hands)  # never grow the crew once liquidating
        hires_today = farm["hires_today"]
        while len(orders) < max_orders and n_hands < target_hands:
            cost = hand_cost(hires_today, hand_cost_mult)
            if cost > money:
                break
            orders.append(["HIRE"])
            money -= cost
            n_hands += 1
            hires_today += 1

        # Seeds: keep the best-ranked crop(s) stocked, throttled per turn (see 013's Revision on
        # why an uncapped BUY_SEED order alone can drain a whole crop's worth of cash at once) AND
        # capped in TOTAL relative to crew size -- the per-crop saturation adjustment in
        # _rank_crops only changes WHICH crop wins the ranking; with `n_hands` hands all seeking
        # seeds every turn, aggregate seed spend stays high regardless of which crop "wins", and
        # every crop still takes ~10+ days to pay anything back. Without this, 8 hands blow the
        # whole $3000 starting budget on seeds before day 2 (see the issue's Revision section for
        # the numbers -- the exact same class of bug 013 hit and fixed the same way).
        total_committed = sum(planted_counts.values()) + sum(private["seeds"].values())
        if not liquidating and total_committed < n_hands * TILES_PER_HAND_CAP:
            for _ppa, crop in ranked_crops:
                if len(orders) >= max_orders:
                    break
                shortfall = params["seed_stock_target"] - private["seeds"].get(crop, 0)
                if shortfall > 0 and money >= CROPS[crop]["seed"]:
                    qty = min(int(shortfall), int(money // CROPS[crop]["seed"]), MAX_SEED_QTY_PER_ORDER)
                    if qty > 0:
                        orders.append(["BUY_SEED", crop, qty])
                        money -= qty * CROPS[crop]["seed"]
                break  # only the top-ranked crop -- a live re-rank next turn covers the rest

        # Wheat buffer for animal feed.
        n_animal_sites_owned = sum(
            1 for q, pos in enumerate(animal_sites) if QUADRANT_NAMES[q] in unlocked and isinstance(farm["tiles"][pos[1]][pos[0]], dict)
        )
        if len(orders) < max_orders and n_animal_sites_owned and private["shed"].get("WHEAT", 0) < params["wheat_buffer_per_animal"] * n_animal_sites_owned and money > 0:
            orders.append(["BUY_PRODUCT", "WHEAT", 1])

        # Sell shed contents above the live floor. WHEAT is reserved for feed; FERTILIZER is NOT
        # excluded -- it's a free byproduct of a cared-for animal (see _animal_tile_action's
        # COLLECT_FERTILIZER), never bought, so selling it is pure found money ("fertilizer-as-
        # cash", one of ROADMAP's flagged tactical primitives -- discovered by this policy's own
        # generic sell logic, not a special case).
        for item, qty in private["shed"].items():
            if len(orders) >= max_orders:
                break
            if qty <= 0 or item not in MARKET_PARAMS or item == "WHEAT":
                continue
            floor = _sell_floor(item, params["sell_floor_fraction"])
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
        n_days = int(configuration.get("episodeSteps", 720)) // turns_per_day

        unlocked = set(farm["unlocked_quadrants"])
        my_crop_tiles = [pos for q_idx, pos in enumerate(all_tiles) if QUADRANT_NAMES[q_idx // 25] in unlocked and pos not in animal_sites]
        my_animal_sites = [pos for q, pos in enumerate(animal_sites) if QUADRANT_NAMES[q] in unlocked]
        planted_counts = _count_planted(farm["tiles"])

        n_hands = len(farm["hands"])
        n_crop_workers = max(1, n_hands)  # farmer always does herd duty when any site is unlocked
        chunk_size = max(1, -(-len(my_crop_tiles) // n_crop_workers)) if my_crop_tiles else 1
        crop_chunks = [my_crop_tiles[i * chunk_size : (i + 1) * chunk_size] for i in range(n_crop_workers)]

        hands_actions = []
        for i, pos in enumerate(farm["hands"]):
            my_tiles = crop_chunks[i] if i < len(crop_chunks) else []
            hands_actions.append(_unit_crop_action(tuple(pos), my_tiles, farm["tiles"], day, hour, turns_per_day, private["seeds"], market["prices"], planted_counts))

        farmer_inv = private["inventories"][0] if private["inventories"] else {}
        if my_animal_sites:
            farmer_action = _unit_herd_action(tuple(farm["farmer"]), my_animal_sites, farm["tiles"], farmer_inv, private["shed"], market["prices"])
        elif my_crop_tiles:
            # No animal site unlocked yet: the farmer pitches in on crop tiles instead of idling.
            farmer_action = _unit_crop_action(tuple(farm["farmer"]), my_crop_tiles, farm["tiles"], day, hour, turns_per_day, private["seeds"], market["prices"], planted_counts)
        else:
            farmer_action = ["PASS"]

        market_orders = _market_orders(farm, private, market, day, n_days, planted_counts)
        return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}

    return agent

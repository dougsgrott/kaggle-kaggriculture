# Self-contained entry point (issue 005): becomes submissions/baseline-v1/main.py verbatim,
# bundled alongside flattened copies of model/{constants,price,yields,economics}.py by
# kaggriculture.submit.build.build_bundle. Bare imports below (`import constants`, not
# `kaggriculture.model.constants`) are intentional -- see build_bundle's docstring. This file
# is never imported directly from the dev tree; it's tested through the built bundle
# (tests/test_baseline.py), the only form that actually runs.
#
# Strategy: melon-first field in the home (NW) quadrant, a single CARE'd goose on the shed
# corner tile, hands sized while marginal hire cost stays under melon's $/action, sells gated
# on the live market quote rather than a fixed price.

from constants import CROPS, MARKET_PARAMS
from economics import hand_cost, one_time_crop_profit_per_action
from price import units_sellable_before

MAX_HANDS = 8
SELL_FLOOR_FRACTION = 0.5  # never sell a unit for less than half its base price
WHEAT_BUFFER = 3  # keep this many WHEAT in the shed for the goose
GOOSE_MONEY_BUFFER = 300 + 200  # goose cost plus a small operating cushion


def _sell_floor(item: str) -> int:
    return max(1, round(MARKET_PARAMS[item]["base"] * SELL_FLOOR_FRACTION))


def _move_toward(pos, target):
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


def _melon_tile_positions(board_size: int) -> list[tuple[int, int]]:
    """Every NW-quadrant tile except the shed corner, which the coop occupies instead."""
    half = board_size // 2
    coop = (half - 1, half - 1)
    return [(x, y) for y in range(half) for x in range(half) if (x, y) != coop]


def _coop_position(board_size: int) -> tuple[int, int]:
    half = board_size // 2
    return (half - 1, half - 1)


def _melon_harvest_ready(tile: dict, day: int) -> bool:
    cd = CROPS["MELON"]
    age = day - tile["planted_day"]
    return age >= cd["max_yield_day"] or tile["yield_units"] >= cd["max_yield"]


def _best_melon_target(my_tiles, farm_tiles, seeds, day, hour, turns_per_day):
    harvest, water, weed, plant = [], [], [], []
    for x, y in my_tiles:
        tile = farm_tiles[y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "MELON":
            if _melon_harvest_ready(tile, day):
                harvest.append((x, y))
            elif not tile["watered_today"]:
                water.append((x, y))
        elif isinstance(tile, dict) and tile.get("kind") == "WEED":
            weed.append((x, y))
        elif tile is None:
            # A seed planted with no time left to water it today weeds out overnight, so don't.
            if seeds.get("MELON", 0) > 0 and hour < turns_per_day - 1:
                plant.append((x, y))
    for group, action in ((harvest, "HARVEST"), (water, "WATER"), (weed, "DIG"), (plant, "PLANT")):
        if group:
            return group[0], action
    return None, None


def _melon_hand_action(pos, my_tiles, farm_tiles, seeds, day, hour, turns_per_day):
    target, action = _best_melon_target(my_tiles, farm_tiles, seeds, day, hour, turns_per_day)
    if target is None:
        return ["PASS"]
    if tuple(pos) == target:
        return ["PLANT", "MELON"] if action == "PLANT" else [action]
    return _move_toward(pos, target)


def _herd_action(farm, private, board_size):
    """The farmer never leaves the coop's tile: shed access only depends on standing on one of
    the four corner tiles, not on what's built there, so PICKUP/FEED/CARE/HARVEST/PLACE all work
    from the same spot the coop occupies."""
    pos = tuple(farm["farmer"])
    coop_pos = _coop_position(board_size)
    if pos != coop_pos:
        return _move_toward(pos, coop_pos)

    tile = farm["tiles"][coop_pos[1]][coop_pos[0]]
    inv = private["inventories"][0] if private["inventories"] else {}
    shed = private["shed"]

    if tile is None:
        return ["BUILD_COOP"]

    if tile.get("kind") != "COOP":
        return ["PASS"]  # something unexpected occupies the coop tile; don't fight it

    if "animal" not in tile:
        if inv.get("GOOSE", 0) > 0:
            return ["PLACE", "GOOSE"]
        if shed.get("GOOSE", 0) > 0:
            return ["PICKUP", "GOOSE", 1]
        return ["PASS"]  # waiting on the market order to land a goose in the shed

    if tile["yield_units"] > 0:
        return ["HARVEST"]
    if not tile["fed_today"]:
        if inv.get("WHEAT", 0) > 0:
            return ["FEED"]
        if shed.get("WHEAT", 0) > 0:
            return ["PICKUP", "WHEAT", 1]
        return ["PASS"]  # waiting on the market order to land wheat in the shed
    if not tile["cared_today"]:
        return ["CARE"]
    return ["PASS"]


def _market_orders(farm, private, market, board_size, max_orders, hand_cost_mult):
    orders = []
    money = farm["money"]

    # Hire while the next hand still costs less than what an extra unit of melon-tending
    # capacity is worth, capped so hands don't outnumber the tiles they'd tend.
    melon_price = market["prices"]["MELON"]
    ppa = one_time_crop_profit_per_action("MELON", melon_price)
    n_hands = len(farm["hands"])
    hires_today = farm["hires_today"]
    while n_hands < MAX_HANDS and len(orders) < max_orders:
        cost = hand_cost(hires_today, hand_cost_mult)
        if cost >= ppa or cost > money:
            break
        orders.append(["HIRE"])
        money -= cost
        n_hands += 1
        hires_today += 1

    # One goose, bought once, kept fed from a small standing shed buffer.
    coop_pos = _coop_position(board_size)
    coop_tile = farm["tiles"][coop_pos[1]][coop_pos[0]]
    has_goose = isinstance(coop_tile, dict) and coop_tile.get("animal") == "GOOSE"
    owns_goose_unplaced = private["shed"].get("GOOSE", 0) > 0 or any(
        inv.get("GOOSE", 0) > 0 for inv in private["inventories"]
    )
    if len(orders) < max_orders and not has_goose and not owns_goose_unplaced and money >= GOOSE_MONEY_BUFFER:
        orders.append(["BUY_ANIMAL", "GOOSE", 1])
        money -= 300

    if len(orders) < max_orders and private["shed"].get("WHEAT", 0) < WHEAT_BUFFER and money > 0:
        orders.append(["BUY_PRODUCT", "WHEAT", 1])

    # Melon seeds: keep enough in stock to plant every currently-empty tile this hand set can
    # reach; buying too far ahead just ties up cash the market doesn't need yet.
    if len(orders) < max_orders:
        empty_tiles = sum(
            1 for x, y in _melon_tile_positions(board_size) if farm["tiles"][y][x] is None
        )
        seed_shortfall = empty_tiles - private["seeds"].get("MELON", 0)
        if seed_shortfall > 0 and money >= CROPS["MELON"]["seed"]:
            affordable = int(money // CROPS["MELON"]["seed"])
            qty = min(seed_shortfall, affordable)
            if qty > 0:
                orders.append(["BUY_SEED", "MELON", qty])

    # Sell shed contents, but never past the point where the price would fall below our floor.
    # WHEAT is excluded -- it's feed stock for the goose, not a crop, and selling the standing
    # buffer right after buying it would just churn money on the buy/sell spread.
    for item, qty in private["shed"].items():
        if len(orders) >= max_orders:
            break
        if qty <= 0 or item not in MARKET_PARAMS or item == "WHEAT":
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

    board_size = int(configuration.get("boardSize", 10))
    turns_per_day = int(configuration.get("turnsPerDay", 24))
    max_orders = int(configuration.get("maxMarketOrdersPerTurn", 10))
    hand_cost_mult = int(configuration.get("farmHandCostMult", 1))

    farmer_action = _herd_action(farm, private, board_size)

    all_tiles = _melon_tile_positions(board_size)
    n_hands = len(farm["hands"])
    hands_actions = []
    for i, pos in enumerate(farm["hands"]):
        my_tiles = [t for j, t in enumerate(all_tiles) if j % n_hands == i]
        hands_actions.append(
            _melon_hand_action(tuple(pos), my_tiles, farm["tiles"], private["seeds"], day, hour, turns_per_day)
        )

    market_orders = _market_orders(farm, private, market, board_size, max_orders, hand_cost_mult)

    return {"farmer": farmer_action, "hands": hands_actions, "market": market_orders}

"""Translates real-engine-shaped actions and configuration into this sim's native types. Shared
by issue 010's validate.py (replaying a recorded trace) and issue 011's eval arena (running a
live real-engine-style agent through this sim via build_observation() + CallbackPolicy) --
extracted here so both stay byte-for-byte the same decoder rather than drifting apart.
"""

from __future__ import annotations

from typing import Any

from kaggriculture.sim import _sim_native as native


def decode_unit_action(a: Any) -> native.UnitAction:
    if not isinstance(a, list) or not a:
        return native.UnitAction(native.Op.PASS)
    op = getattr(native.Op, a[0], native.Op.PASS) if isinstance(a[0], str) else native.Op.PASS
    item = native.Item.WHEAT
    if len(a) >= 2 and isinstance(a[1], str):
        item = getattr(native.Item, a[1], native.Item(255))
    n = 1
    if len(a) >= 3:
        try:
            n = int(a[2])
        except (TypeError, ValueError):
            n = 1
    return native.UnitAction(op, item, n)


def decode_market_order(o: Any) -> native.MarketOrder:
    if not isinstance(o, list) or not o or not isinstance(o[0], str):
        return native.MarketOrder(native.MarketOp.NONE)
    op = getattr(native.MarketOp, o[0], native.MarketOp.NONE)
    if op in (native.MarketOp.HIRE, native.MarketOp.BUY_LAND):
        return native.MarketOrder(op, native.Item.WHEAT, 1)
    if op == native.MarketOp.NONE or len(o) < 3:
        return native.MarketOrder(native.MarketOp.NONE)
    item = getattr(native.Item, o[1], native.Item(255)) if isinstance(o[1], str) else native.Item(255)
    try:
        n = int(o[2])
    except (TypeError, ValueError):
        return native.MarketOrder(native.MarketOp.NONE)
    if n <= 0:
        return native.MarketOrder(native.MarketOp.NONE)
    return native.MarketOrder(op, item, n)


def build_player_turn(act: dict, max_orders: int) -> native.PlayerTurn:
    """`act` is a raw action dict: `{"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}`."""
    turn = native.PlayerTurn()
    units = [act.get("farmer") or ["PASS"], *(act.get("hands") or [])]
    turn.unit_actions = [decode_unit_action(u) for u in units]
    # vendor truncates the raw order queue to maxMarketOrdersPerTurn *before* processing
    # (`q[:max_orders]` in _process_market) -- match that at decode time.
    turn.market_orders = [decode_market_order(o) for o in (act.get("market") or [])[:max_orders]]
    return turn


def episode_config(cfg: dict) -> native.EpisodeConfig:
    """`cfg` is a real `env.configuration`-shaped dict (or plain dict with the same keys)."""
    episode_cfg = native.EpisodeConfig()
    episode_cfg.episode_steps = cfg["episodeSteps"]
    episode_cfg.board_size = cfg["boardSize"]
    episode_cfg.turns_per_day = cfg["turnsPerDay"]
    episode_cfg.shed_capacity = cfg["shedCapacity"]
    episode_cfg.starting_money = float(cfg["startingMoney"])
    episode_cfg.market.max_market_orders_per_turn = cfg["maxMarketOrdersPerTurn"]
    episode_cfg.market.farm_hand_cost_mult = cfg["farmHandCostMult"]
    episode_cfg.market.weed_spawn_chance = cfg["weedSpawnChance"]
    episode_cfg.market.town_shop_unlock_interval = cfg["townShopUnlockInterval"]
    episode_cfg.market.town_shop_sell_interval = cfg["townShopSellInterval"]
    episode_cfg.market.town_center_sell_interval = cfg["townCenterSellInterval"]
    return episode_cfg


def default_config_dict() -> dict:
    """The engine's own defaults (kaggriculture.model.constants.CONFIG_DEFAULTS), for building an
    EpisodeConfig without a real env.configuration on hand."""
    from kaggriculture.model.constants import CONFIG_DEFAULTS

    return dict(CONFIG_DEFAULTS)

"""Town demand: per-product drain per turn from unlocked shops and the town center.

Mirrors `_town_consume` in vendor/kaggriculture.py. Shops are drawn with replacement (PR
#1394): `unlocked_shops` may repeat a name, and each instance consumes independently, so
duplicates are not deduplicated here.
"""

from kaggriculture.model.constants import CONFIG_DEFAULTS, SHOPS, TOWN_CENTER_PRODUCTS


def shop_drain_per_tick(unlocked_shops: list[str]) -> dict[str, int]:
    """Units removed from each product's market inventory on a turn where the shop-sell
    interval fires, given the current (possibly repeated) `unlocked_shops` list. Single-product
    shops pull 2x; every other shop pulls 1x per product."""
    drain: dict[str, int] = {}
    for shop_name in unlocked_shops:
        products = SHOPS[shop_name]
        multiplier = 2 if len(products) == 1 else 1
        for item in products:
            drain[item] = drain.get(item, 0) + multiplier
    return drain


def town_center_drain_per_tick() -> dict[str, int]:
    """Units removed from each non-fertilizer product's market inventory on a turn where the
    town-center-sell interval fires: flat 1 unit each, unconditional on shop unlocks."""
    return {item: 1 for item in TOWN_CENTER_PRODUCTS}


def drain_at_step(
    unlocked_shops: list[str],
    step: int,
    shop_interval: int = CONFIG_DEFAULTS["townShopSellInterval"],
    center_interval: int = CONFIG_DEFAULTS["townCenterSellInterval"],
) -> dict[str, int]:
    """Total per-product market-inventory drain on turn `step`, combining whichever of the
    shop tick and the town-center tick land on this step (`step % interval == 0`, matching
    `_town_consume`'s use of the *pre-increment* step)."""
    drain: dict[str, int] = {}
    if step % shop_interval == 0:
        for item, n in shop_drain_per_tick(unlocked_shops).items():
            drain[item] = drain.get(item, 0) + n
    if step % center_interval == 0:
        for item, n in town_center_drain_per_tick().items():
            drain[item] = drain.get(item, 0) + n
    return drain

"""price.py must reproduce the engine's market_price() exactly — not approximately.

See issues/004-domain-model.md: 56 public notebooks paste a stale MARKET_PARAMS table and
compute quotes from it. This is the guard against model/ becoming another one.
"""

import random

import pytest

from kaggriculture.model.constants import MARKET_I0, PRODUCTS, market_price
from kaggriculture.model.price import price, units_sellable_before

# (T, P(I0-T), P(I0+T), P(I0+2T)) from wiki/competition/pages/how-to-play.md's price table.
HOW_TO_PLAY_TABLE = {
    "WHEAT": (400, 45, 20, 19),
    "CARROT": (450, 70, 10, 1),
    "TOMATO": (200, 84, 24, 9),
    "STRAWBERRY": (100, 204, 1, 1),
    "MELON": (300, 300, 1, 1),
    "EGG": (332, 70, 40, 39),
    "MILK": (122, 256, 1, 1),
    "WOOL": (105, 240, 1, 1),
    "FERTILIZER": (200, 140, 60, 20),
}


@pytest.mark.parametrize("item", PRODUCTS)
def test_matches_how_to_play_table(item):
    T, p_minus_t, p_plus_t, p_plus_2t = HOW_TO_PLAY_TABLE[item]
    assert price(item, MARKET_I0 - T) == p_minus_t
    assert price(item, MARKET_I0 + T) == p_plus_t
    assert price(item, MARKET_I0 + 2 * T) == p_plus_2t


@pytest.mark.parametrize("item", PRODUCTS)
def test_matches_vendored_engine_exactly(item):
    rng = random.Random(0)
    for _ in range(3000):
        inv = rng.randint(0, 40_000)
        assert price(item, inv) == market_price(item, inv)


@pytest.mark.parametrize("item", ["WHEAT", "CARROT", "MELON", "STRAWBERRY", "FERTILIZER"])
def test_units_sellable_before_matches_brute_force(item):
    rng = random.Random(1)
    for _ in range(200):
        inv = rng.randint(9500, 10800)
        threshold = rng.randint(30, 250)
        n = units_sellable_before(item, inv, threshold)
        assert n is not None
        assert price(item, inv + n) <= threshold
        if n > 0:
            assert price(item, inv + n - 1) > threshold


def test_units_sellable_before_returns_zero_when_already_at_threshold():
    current = price("WHEAT", MARKET_I0)
    assert units_sellable_before("WHEAT", MARKET_I0, current) == 0


def test_units_sellable_before_returns_none_at_or_below_floor():
    assert units_sellable_before("WHEAT", MARKET_I0, 1) is None
    assert units_sellable_before("WHEAT", MARKET_I0, 0) is None

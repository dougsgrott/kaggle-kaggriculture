"""economics.py: marginal hand cost, marginal tile cost, and profit-per-action ranking.

Cross-checked against discussion 734033 ("profit per action for every crop and animal"),
which ranks melon highest and wheat lowest with a large (~9.5x) spread driven by action count,
not revenue. Exact dollar figures differ (that post uses a non-minimal watering schedule for
melon), so this only pins down the *ranking*, not the numbers.
"""

from kaggriculture.model.constants import MARKET_PARAMS
from kaggriculture.model.economics import (
    animal_profit_per_action,
    hand_cost,
    land_unlock_cost,
    marginal_tile_cost,
    ongoing_crop_profit_per_action,
    one_time_crop_profit_per_action,
)


def _base(item: str) -> float:
    return MARKET_PARAMS[item]["base"]


def test_hand_cost_is_fibonacci_resetting_daily():
    # fib(0)=1, 1, 2, 3, 5, 8, 13, 21 per how-to-play.md's worked example.
    assert [hand_cost(n) for n in range(8)] == [1, 1, 2, 3, 5, 8, 13, 21]


def test_land_costs_and_marginal_tile_cost():
    assert [land_unlock_cost(n) for n in range(3)] == [1000, 2000, 4000]
    assert marginal_tile_cost(0) == 1000 / 25
    assert marginal_tile_cost(2) == 4000 / 25


def test_profit_per_action_ranking_matches_discussion_734033():
    ranking = {
        "MELON": one_time_crop_profit_per_action("MELON", _base("MELON")),
        "SHEEP_CARE": animal_profit_per_action("SHEEP", _base("WOOL"), care=True),
        "COW_CARE": animal_profit_per_action("COW", _base("MILK"), care=True),
        "COW_FEED_ONLY": animal_profit_per_action("COW", _base("MILK"), care=False),
        "SHEEP_FEED_ONLY": animal_profit_per_action("SHEEP", _base("WOOL"), care=False),
        "GOOSE_CARE": animal_profit_per_action("GOOSE", _base("EGG"), care=True),
        "STRAWBERRY": ongoing_crop_profit_per_action("STRAWBERRY", _base("STRAWBERRY")),
        "GOOSE_FEED_ONLY": animal_profit_per_action("GOOSE", _base("EGG"), care=False),
        "TOMATO": ongoing_crop_profit_per_action("TOMATO", _base("TOMATO")),
        "CARROT": one_time_crop_profit_per_action("CARROT", _base("CARROT")),
        "WHEAT": one_time_crop_profit_per_action("WHEAT", _base("WHEAT")),
    }
    assert ranking["MELON"] == max(ranking.values())
    assert ranking["WHEAT"] == min(ranking.values())
    assert ranking["MELON"] > 5 * ranking["WHEAT"]  # "about 9.5x" in the discussion
    # CARE strictly beats feed-only for every animal — "one of the biggest levers in the game".
    assert ranking["SHEEP_CARE"] > ranking["SHEEP_FEED_ONLY"]
    assert ranking["COW_CARE"] > ranking["COW_FEED_ONLY"]
    assert ranking["GOOSE_CARE"] > ranking["GOOSE_FEED_ONLY"]

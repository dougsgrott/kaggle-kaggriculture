"""Profit per farmer action, and the two other marginal costs that gate action supply.

Cross-checked against discussion 734033 ("profit per action for every crop and animal") —
see tests/model/test_economics.py. Sale prices are supplied by the caller (base price, or a
dynamic quote from `price.py`); this module never assumes the market is static.
"""

from kaggriculture.model.constants import ANIMALS, CROPS, FARM_HAND_COST_MULT, LAND_PRICES
from kaggriculture.model.yields import ongoing_crop_production_days, one_time_crop_yield, watering_bonus_window

TILES_PER_QUADRANT = 25  # 5x5, per wiki/competition/pages/how-to-play.md


def fib(n: int) -> int:
    """fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5, ... Standalone (not delegating to
    constants._engine) so this module only needs *data* from constants, not vendor's live
    functions — the packaged submission freezes constants to a data-only snapshot (see
    kaggriculture.model.freeze) since vendor/ won't exist on Kaggle's grader."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def minimal_watering_schedule(crop: str) -> set[int]:
    """Fewest watering ages (earliest-first in the bonus window) that reach `max_yield`
    unfertilized. The actions-per-unit floor for a one-time crop."""
    cd = CROPS[crop]
    start, end = watering_bonus_window(crop)
    needed = cd["max_yield"] - 1  # `_new_plant` starts yield_units at 1
    ages = set()
    for age in range(start, end + 1):
        if needed <= 0:
            break
        ages.add(age)
        needed -= 1
    return ages


def one_time_crop_actions(watered_ages: set[int]) -> int:
    """PLANT + one WATER per watered age + HARVEST. Excludes movement/shed logistics, which
    are farm-layout dependent, not crop-dependent."""
    return 2 + len(watered_ages)


def one_time_crop_profit_per_action(
    crop: str,
    sale_price: float,
    watered_ages: set[int] | None = None,
    fertilized_ages: set[int] = frozenset(),
    fertilizer_price: float = 0.0,
) -> float:
    """$/action for a one-time crop (WHEAT, CARROT, MELON), defaulting to the minimal
    (unfertilized) watering schedule that reaches max_yield."""
    if watered_ages is None:
        watered_ages = minimal_watering_schedule(crop)
    yield_units = one_time_crop_yield(crop, watered_ages, fertilized_ages)
    revenue = yield_units * sale_price
    cost = CROPS[crop]["seed"] + len(fertilized_ages) * fertilizer_price
    return (revenue - cost) / one_time_crop_actions(watered_ages)


def ongoing_crop_profit_per_action(
    crop: str, sale_price: float, fertilized_event_indices: set[int] = frozenset(), fertilizer_price: float = 0.0
) -> float:
    """$/action for an ongoing crop (TOMATO, STRAWBERRY) across its full `max_yield`-event
    lifetime: PLANT + one WATER per production-eve day + one HARVEST per event."""
    cd = CROPS[crop]
    events = ongoing_crop_production_days(crop)
    yield_units = min(cd["max_yield"], sum(2 if k in fertilized_event_indices else 1 for k in range(len(events))))
    revenue = yield_units * sale_price
    cost = CROPS[crop]["seed"] + len(fertilized_event_indices) * fertilizer_price
    actions = 1 + len(events) + len(events)  # PLANT + one WATER/event-eve + one HARVEST/event
    return (revenue - cost) / actions


def animal_profit_per_action(animal: str, sale_price: float, care: bool, n_events: int = 12, feed_price: float = 0.0) -> float:
    """Steady-state $/action once production is underway: FEED every day (+ CARE every day if
    `care`), harvesting immediately after every scheduled production so the bank/tile never hit
    a cap mid-run. `n_events` sets the simulated window (in production events, not days) — large
    enough to average out ramp-up.
    """
    a = ANIMALS[animal]
    total_days = a["first_yield_day"] + n_events * a["interval"]
    total_harvested = 0
    pending_care_bonus = 0
    for day in range(total_days):
        next_day = day + 1
        days_since_first = next_day - a["first_yield_day"]
        if days_since_first >= 0 and days_since_first % a["interval"] == 0:
            bonus = pending_care_bonus  # animal is always fed in this steady-state schedule
            total_harvested += min(a["max_held"], 1 + bonus)
            pending_care_bonus = 0
        if care:
            pending_care_bonus += 1

    n_productions = sum(
        1
        for day in range(total_days)
        if (day + 1 - a["first_yield_day"]) >= 0 and (day + 1 - a["first_yield_day"]) % a["interval"] == 0
    )
    revenue = total_harvested * sale_price
    cost = total_days * feed_price
    actions = total_days * (2 if care else 1) + n_productions  # FEED (+CARE) daily, HARVEST per event
    return (revenue - cost) / actions


def hand_cost(n_already_hired_today: int, mult: int = FARM_HAND_COST_MULT) -> int:
    """Marginal cost of the next hire today: `farmHandCostMult * fib(n)`, resetting daily."""
    return mult * fib(n_already_hired_today)


def land_unlock_cost(n_extra_quadrants_already_unlocked: int) -> int:
    """Cost of the next quadrant beyond NW (0 = NE, 1 = SW, 2 = SE)."""
    return LAND_PRICES[n_extra_quadrants_already_unlocked]


def marginal_tile_cost(n_extra_quadrants_already_unlocked: int) -> float:
    """Average $/tile gained by unlocking the next quadrant."""
    return land_unlock_cost(n_extra_quadrants_already_unlocked) / TILES_PER_QUADRANT

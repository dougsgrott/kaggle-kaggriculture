"""Crop yield and animal production, standalone from full farm/tile state.

Pure functions mirroring the exact arithmetic of the engine's `_apply_unit_action` (WATER
branch), `_daily_refresh_plants`, `_daily_refresh_animals` and `_decay_plants` — see
vendor/kaggriculture.py. Each takes a caller-supplied watering/feeding/caring schedule rather
than simulating tile/inventory/movement state, which is out of scope here (see 007/008's C++
port for the full state machine).
"""

from kaggriculture.model.constants import ANIMALS, CROPS


def watering_bonus_window(crop: str) -> tuple[int, int]:
    """Inclusive (start, end) age range, in days since planting, during which watering a
    one-time crop adds yield. `start` is `max_yield_day` halved and rounded up."""
    cd = CROPS[crop]
    return ((cd["max_yield_day"] + 1) // 2, cd["max_yield_day"])


def one_time_crop_yield(crop: str, watered_ages: set[int], fertilized_ages: set[int] = frozenset()) -> int:
    """Final `yield_units` for a one-time crop (WHEAT, CARROT, MELON) given the ages (days
    since planting) it was watered, and the subset of those watering ages on which the
    fertilizer bonus was active. Starts at 1 (the engine's `_new_plant` default) and is
    capped at `max_yield`; only ages inside `watering_bonus_window` matter."""
    cd = CROPS[crop]
    start, end = watering_bonus_window(crop)
    yield_units = 1
    for age in watered_ages:
        if start <= age <= end:
            bonus = 2 if age in fertilized_ages else 1
            yield_units = min(cd["max_yield"], yield_units + bonus)
    return yield_units


def one_time_decay_start_step(crop: str, planted_day: int, turns_per_day: int = 24) -> int:
    """The turn at which an unharvested one-time crop begins decaying into a weed."""
    cd = CROPS[crop]
    return (planted_day + cd["max_yield_day"] + 1) * turns_per_day


def one_time_crop_watering_only_peak(crop: str) -> tuple[int, int]:
    """(age, yield_units) at which watering alone (no fertilizer) first reaches its own cap —
    not necessarily `max_yield` (fertilizer-only) and not necessarily `max_yield_day`: melon's
    watering-only cap lands at age 10, two days before its documented bonus window closes at
    age 12, so the last two days of that window are dead turns."""
    cd = CROPS[crop]
    start, end = watering_bonus_window(crop)
    extra_needed = cd["max_yield"] - 1
    watered_days = min(extra_needed, end - start + 1)
    peak_age = start + watered_days - 1
    peak_yield = 1 + watered_days
    return peak_age, peak_yield


def one_time_crop_yield_per_tile_per_day(crop: str) -> float:
    """Total units harvested divided by the days the tile is occupied, watering daily and
    harvesting the moment watering-only yield peaks — matches the "Yield / tile / day" column of
    how-to-play.md. Discrepancy: the table's own values are only self-consistent under this
    exact "occupied through peak day" denominator; see discussion 732450's "Yield-Per-Day
    Calculations" complaint about earlier, inconsistent versions of this figure."""
    peak_age, peak_yield = one_time_crop_watering_only_peak(crop)
    return peak_yield / (peak_age + 1)


def ongoing_crop_production_days(crop: str, planted_day: int = 0) -> list[int]:
    """The `max_yield` scheduled production days (offset from `planted_day`) for an ongoing
    crop (TOMATO, STRAWBERRY) — after the last one, the plant starts decaying."""
    cd = CROPS[crop]
    return [planted_day + cd["first_yield_day"] + k * cd["interval"] for k in range(cd["max_yield"])]


def ongoing_crop_yield(crop: str, fertilized_event_indices: set[int] = frozenset()) -> int:
    """Final `yield_units` for an ongoing crop that survives all `max_yield` scheduled
    productions (never neglected into a weed). `fertilized_event_indices` are the 0-indexed
    events (into `ongoing_crop_production_days`) that were watered and fertilizer-active on
    the eve of production, doubling that event's yield from 1 to 2."""
    cd = CROPS[crop]
    yield_units = 0
    for k in range(cd["max_yield"]):
        bonus = 2 if k in fertilized_event_indices else 1
        yield_units = min(cd["max_yield"], yield_units + bonus)
    return yield_units


def ongoing_decay_start_step(crop: str, planted_day: int, turns_per_day: int = 24) -> int:
    """The turn at which an ongoing crop begins decaying, one day after its last scheduled
    production (regardless of whether that production was ever harvested)."""
    last_production_day = ongoing_crop_production_days(crop, planted_day)[-1]
    return (last_production_day + 1) * turns_per_day


def ongoing_crop_yield_per_tile_per_day(crop: str) -> float:
    """Same metric as `one_time_crop_yield_per_tile_per_day`: unfertilized ongoing crops always
    reach exactly `max_yield` (1 unit x max_yield events, never capped), divided by the days
    from planting through the last scheduled production."""
    cd = CROPS[crop]
    last_production_day = ongoing_crop_production_days(crop)[-1]
    return cd["max_yield"] / (last_production_day + 1)


def decay_yield_units(yield_units_at_decay_start: int, step: int, decay_start_step: int) -> int:
    """`yield_units` remaining `step` turns into decay: -1 every other turn starting at
    `decay_start_step` itself, floored at 0 (the tile becomes a weed at 0)."""
    if decay_start_step < 0 or step < decay_start_step:
        return yield_units_at_decay_start
    offset = step - decay_start_step
    decrements = offset // 2 + 1
    return max(0, yield_units_at_decay_start - decrements)


def first_neglect_weed_age(watered: list[bool]) -> int | None:
    """The age (index into `watered`, days since planting) at the end of which a plant turns
    into a weed from two consecutive unwatered days. `consecutive_unwatered` starts at 1 at
    planting — the planting day itself counts as the first missed day — so a seed planted and
    left unwatered that same day weeds out that same night (age 0), with no grace period.
    Returns None if `watered` never produces two consecutive misses."""
    consecutive_unwatered = 1
    for age, was_watered in enumerate(watered):
        consecutive_unwatered = 0 if was_watered else consecutive_unwatered + 1
        if consecutive_unwatered >= 2:
            return age
    return None


def simulate_animal(animal: str, daily_schedule: list[tuple[bool, bool]]) -> dict:
    """Replays `_daily_refresh_animals` day by day. `daily_schedule[i] = (fed_today, cared_today)`
    for day i since placement (0-indexed). Raises ValueError if the schedule leaves the animal
    unfed two days running (it would escape in the real engine). Returns the resulting
    `{"yield_units": int, "pending_care_bonus": int}`, mirroring the tile's own fields."""
    a = ANIMALS[animal]
    yield_units = 0
    pending_care_bonus = 0
    consecutive_unfed = 0
    for day, (fed_today, cared_today) in enumerate(daily_schedule):
        next_day = day + 1
        consecutive_unfed = 0 if fed_today else consecutive_unfed + 1
        if consecutive_unfed >= 2:
            raise ValueError(f"{animal} escaped: unfed on days {day - 1} and {day}")

        days_since_first = next_day - a["first_yield_day"]
        if days_since_first >= 0 and days_since_first % a["interval"] == 0:
            bonus = pending_care_bonus if fed_today else 0
            yield_units = min(a["max_held"], yield_units + 1 + bonus)
            pending_care_bonus = 0

        if cared_today and fed_today:
            pending_care_bonus += 1

    return {"yield_units": yield_units, "pending_care_bonus": pending_care_bonus}

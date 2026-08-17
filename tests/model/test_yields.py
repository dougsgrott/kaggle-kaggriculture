"""yields.py against the numbers wiki/competition/pages/how-to-play.md states outright, plus
the engine-vs-docs discrepancies from discussion 732450 that motivated issue 002/004.
"""

import pytest

from kaggriculture.model.yields import (
    decay_yield_units,
    first_neglect_weed_age,
    ongoing_crop_production_days,
    ongoing_crop_yield,
    ongoing_crop_yield_per_tile_per_day,
    one_time_crop_yield,
    one_time_crop_yield_per_tile_per_day,
    simulate_animal,
    watering_bonus_window,
)


def test_wheat_peaks_at_4_watered_at_6_fertilized():
    start, end = watering_bonus_window("WHEAT")
    all_ages = set(range(start, end + 1))
    assert one_time_crop_yield("WHEAT", all_ages) == 4
    assert one_time_crop_yield("WHEAT", all_ages, fertilized_ages=all_ages) == 6


def test_carrot_peaks_at_3_watered_at_4_fertilized():
    start, end = watering_bonus_window("CARROT")
    all_ages = set(range(start, end + 1))
    assert one_time_crop_yield("CARROT", all_ages) == 3
    assert one_time_crop_yield("CARROT", all_ages, fertilized_ages=all_ages) == 4


def test_melon_bonus_window_has_two_dead_days():
    """Discrepancy: window is documented as ages 6-12, but the cap of 6 is reached at age 10
    (base 1 + 1/day for 5 days) — ages 11-12 add nothing."""
    start, end = watering_bonus_window("MELON")
    assert (start, end) == (6, 12)
    watering_through_cap = set(range(6, 11))  # ages 6-10
    watering_full_window = set(range(6, 13))  # ages 6-12
    assert one_time_crop_yield("MELON", watering_through_cap) == 6
    assert one_time_crop_yield("MELON", watering_full_window) == 6  # 11, 12 are dead days


def test_melon_fertilized_reaches_cap_at_age_8():
    ages = set(range(6, 9))  # ages 6, 7, 8
    assert one_time_crop_yield("MELON", ages, fertilized_ages=ages) == 6


def test_tomato_production_days_and_yield():
    assert ongoing_crop_production_days("TOMATO") == [8, 9, 10, 11]
    assert ongoing_crop_yield("TOMATO") == 4


def test_strawberry_capped_at_4_productions_not_indefinite():
    """Discrepancy: docs call strawberry an indefinite 'every other day' producer; the engine
    caps it at exactly 4 scheduled productions (ages 10, 12, 14, 16), then it decays."""
    assert ongoing_crop_production_days("STRAWBERRY") == [10, 12, 14, 16]
    assert ongoing_crop_yield("STRAWBERRY") == 4
    # A 5th scheduled event is never reached, so there's nothing past index 3 to fertilize.
    assert len(ongoing_crop_production_days("STRAWBERRY")) == 4


def test_fertilized_ongoing_event_doubles_that_events_yield_but_caps_at_max_yield():
    # Fertilizing every event would give 2*4=8, capped at max_yield=4.
    assert ongoing_crop_yield("STRAWBERRY", fertilized_event_indices={0, 1, 2, 3}) == 4
    # Fertilizing just the first event: 2 + 1 + 1 = 4 (still under cap by chance).
    assert ongoing_crop_yield("STRAWBERRY", fertilized_event_indices={0}) == 4


def test_decay_starts_immediately_at_decay_start_step_and_halves_rate():
    # -1 at step==decay_start, then every other step after.
    assert decay_yield_units(6, step=100, decay_start_step=100) == 5
    assert decay_yield_units(6, step=101, decay_start_step=100) == 5
    assert decay_yield_units(6, step=102, decay_start_step=100) == 4
    assert decay_yield_units(6, step=110, decay_start_step=100) == 0  # capped at 0, not negative
    assert decay_yield_units(6, step=99, decay_start_step=100) == 6  # not yet decaying


def test_planting_day_watering_discrepancy_weeds_same_night():
    """Discrepancy: a seed left unwatered on its own planting day dies that same night —
    there is no grace period, because consecutive_unwatered starts at 1, not 0."""
    assert first_neglect_weed_age([False]) == 0
    assert first_neglect_weed_age([True, False, False]) == 2
    assert first_neglect_weed_age([True, True, True]) is None
    assert first_neglect_weed_age([True, False, True, False, False]) == 4


def test_care_bonus_increments_by_1_not_2():
    """Discrepancy: the rulebook says the CARE bonus increments pending_care_bonus by 2; the
    engine increments it by 1 per fed+cared day. SHEEP (first_yield_day=6, interval=3, max_held=6)
    makes the two theories cleanly distinguishable: fed+cared every day for 3 days (days 0-2)
    banks a bonus that is *not yet* consumed (first production is day 6), so bank=3 under the
    +1 theory (vs. 6 under +2, which would already be visibly wrong against max_held=6)."""
    result = simulate_animal("SHEEP", [(True, True)] * 3)
    assert result == {"yield_units": 0, "pending_care_bonus": 3}


def test_animal_escapes_after_two_consecutive_unfed_days():
    with pytest.raises(ValueError):
        simulate_animal("GOOSE", [(True, False), (False, False), (False, False)])


def test_yield_per_tile_per_day_matches_how_to_play_table():
    """Discrepancy: discussion 732450 calls out the original 'Max yield / tile / DAY' table as
    using inconsistent formulas (e.g. Tomato listed as 4, Strawberry as 2). The current,
    corrected how-to-play.md table divides total watering-only yield by the days the tile is
    occupied through the peak/last-production day — these are those numbers."""
    assert round(one_time_crop_yield_per_tile_per_day("WHEAT"), 2) == 0.80
    assert round(one_time_crop_yield_per_tile_per_day("CARROT"), 2) == 0.75
    assert round(one_time_crop_yield_per_tile_per_day("MELON"), 2) == 0.55
    assert round(ongoing_crop_yield_per_tile_per_day("TOMATO"), 2) == 0.33
    assert round(ongoing_crop_yield_per_tile_per_day("STRAWBERRY"), 2) == 0.24

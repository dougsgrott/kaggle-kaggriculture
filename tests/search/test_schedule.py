"""Tests for the greedy scheduler's macro-planning half (issue 013)."""

import pytest

from kaggriculture.model.constants import ANIMALS
from kaggriculture.search.schedule import (
    BOARD_SIZE,
    CROP_NAMES,
    HALF,
    Schedule,
    build_schedule,
    quadrant_tiles,
    shed_adjacent_tile,
)
from kaggriculture.sim.decode import default_config_dict


def test_shed_adjacent_tiles_match_engine_geometry():
    """NWSE order, matching sim.cpp's shed_access_tiles (board_size=10, half=5)."""
    assert [shed_adjacent_tile(q) for q in range(4)] == [(4, 4), (5, 4), (4, 5), (5, 5)]


def test_quadrant_tiles_partitions_the_board_with_no_overlap():
    all_tiles = [t for q in range(4) for t in quadrant_tiles(q)]
    assert len(all_tiles) == BOARD_SIZE * BOARD_SIZE
    assert len(set(all_tiles)) == BOARD_SIZE * BOARD_SIZE  # no duplicates
    for x, y in all_tiles:
        assert 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def test_quadrant_tiles_is_a_snake_order():
    """Consecutive tiles within a quadrant are always orthogonally adjacent (a valid one-step
    move) -- the whole point of the boustrophedon ordering (see the module's docstring)."""
    for q in range(4):
        tiles = quadrant_tiles(q)
        assert len(tiles) == HALF * HALF
        for (x0, y0), (x1, y1) in zip(tiles, tiles[1:]):
            assert abs(x0 - x1) + abs(y0 - y1) == 1


def test_only_one_time_crops_are_candidates():
    from kaggriculture.model.constants import CROPS

    assert CROP_NAMES == [c for c in CROPS if not CROPS[c]["ongoing"]]
    assert "TOMATO" not in CROP_NAMES
    assert "STRAWBERRY" not in CROP_NAMES


def test_build_schedule_produces_a_populated_plan():
    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    assert isinstance(schedule, Schedule)
    assert schedule.n_days == cfg["episodeSteps"] // cfg["turnsPerDay"]
    assert schedule.tile_role  # at least one tile committed
    assert set(schedule.tile_role) == set(schedule.tile_kind)
    for pos, kind in schedule.tile_kind.items():
        assert kind in ("crop", "animal")
        if kind == "crop":
            assert schedule.tile_role[pos] in CROP_NAMES
        else:
            assert schedule.tile_role[pos] in ANIMALS


def test_animal_structures_are_restricted_to_shed_adjacent_tiles():
    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    animal_positions = {pos for pos, kind in schedule.tile_kind.items() if kind == "animal"}
    assert animal_positions
    assert animal_positions <= {shed_adjacent_tile(q) for q in range(4)}
    assert len(animal_positions) <= 4


def test_target_hands_uses_the_most_recent_day_at_or_before_the_query():
    schedule = Schedule(crew_size_by_day={0: 1, 5: 4, 10: 8})
    assert schedule.target_hands(0) == 1
    assert schedule.target_hands(3) == 1
    assert schedule.target_hands(5) == 4
    assert schedule.target_hands(7) == 4
    assert schedule.target_hands(20) == 8


def test_target_hands_is_zero_before_any_crew_day():
    schedule = Schedule(crew_size_by_day={5: 3})
    assert schedule.target_hands(0) == 0


def test_land_unlock_requires_a_cash_reserve():
    """With starting money just over the first quadrant's cost but under cost+CASH_RESERVE, land
    must NOT be bought on day 0 (see the issue's Revision section on why this matters) -- it may
    still unlock later once crop revenue rebuilds the reserve."""
    from kaggriculture.search.schedule import CASH_RESERVE

    cfg = dict(default_config_dict())
    cfg["startingMoney"] = 1000 + CASH_RESERVE - 1
    schedule = build_schedule(cfg)
    assert schedule.land_days.get(0) != 0


def test_land_unlock_happens_once_the_reserve_is_comfortably_affordable():
    from kaggriculture.search.schedule import CASH_RESERVE

    cfg = dict(default_config_dict())
    cfg["startingMoney"] = 1000 + CASH_RESERVE + 1
    schedule = build_schedule(cfg)
    assert schedule.land_days.get(0) == 0


def test_disallowing_the_4th_quadrant_never_unlocks_se():
    cfg = default_config_dict()
    schedule = build_schedule(cfg, allow_4th_quadrant=False)
    assert 2 not in schedule.land_days  # extra-quadrant index 2 == SE


def test_build_schedule_populates_arrivals_for_committed_products():
    """Issue 016's own production-schedule input -- see tests/search/test_sell_dp.py for the
    consumer side. A minimal structural check here; the tighter per-day bound against
    `committed_units` lives in test_sell_dp.py alongside the module that actually uses this
    field. `tile_role`'s final snapshot is NOT the right thing to compare against: a tile can be
    replanted to a different crop across cycles as prices shift, so a product harvested early in
    the season (and credited to `arrivals`/`committed_units`) may no longer own any tile by the
    season's final snapshot."""
    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    assert set(schedule.arrivals) <= set(schedule.diagnostics["committed_units"])
    assert schedule.arrivals  # at least one product actually landed something in the shed


@pytest.mark.slow
def test_build_schedule_is_deterministic():
    cfg = default_config_dict()
    a = build_schedule(cfg)
    b = build_schedule(cfg)
    assert a.tile_role == b.tile_role
    assert a.crew_size_by_day == b.crew_size_by_day
    assert a.arrivals == b.arrivals

"""Tests for issue 014's LNS search over issue 013's greedy plan."""

import random

import pytest

from kaggriculture.eval.agents import resolve_policy
from kaggriculture.search.lns import (
    _copy_schedule,
    _destroy_repair,
    _resize_crew,
    _resize_herd,
    _shift_land_day,
    _swap_crop,
    diverse_seed_schedules,
    evaluate_schedule,
    head_to_head,
    lns_search,
)
from kaggriculture.search.schedule import ANIMAL_NAMES, CROP_NAMES, build_schedule
from kaggriculture.sim.decode import default_config_dict


@pytest.fixture(scope="module")
def cfg():
    return default_config_dict()


@pytest.fixture(scope="module")
def base_schedule(cfg):
    return build_schedule(cfg)


def test_copy_schedule_is_independent(base_schedule):
    copy = _copy_schedule(base_schedule)
    assert copy.tile_role == base_schedule.tile_role
    assert copy is not base_schedule
    pos = next(iter(copy.tile_role))
    copy.tile_role[pos] = "__mutated__"
    assert base_schedule.tile_role[pos] != "__mutated__"


def test_swap_crop_changes_exactly_one_tile_to_a_different_one_time_crop(base_schedule):
    rng = random.Random(0)
    candidate = _swap_crop(base_schedule, rng)
    assert candidate is not None
    diffs = [pos for pos in base_schedule.tile_role if base_schedule.tile_role[pos] != candidate.tile_role[pos]]
    assert len(diffs) == 1
    pos = diffs[0]
    assert candidate.tile_role[pos] in CROP_NAMES
    assert candidate.tile_role[pos] != base_schedule.tile_role[pos]
    # everything else untouched
    assert candidate.tile_kind == base_schedule.tile_kind
    assert candidate.crew_size_by_day == base_schedule.crew_size_by_day


def test_swap_crop_returns_none_when_there_are_no_crop_tiles():
    from kaggriculture.search.schedule import Schedule

    empty = Schedule(tile_role={(0, 0): "SHEEP"}, tile_kind={(0, 0): "animal"})
    assert _swap_crop(empty, random.Random(0)) is None


def test_resize_herd_swaps_exactly_one_animal_site_to_a_different_animal(base_schedule):
    rng = random.Random(0)
    candidate = _resize_herd(base_schedule, rng)
    assert candidate is not None
    diffs = [pos for pos in base_schedule.tile_role if base_schedule.tile_role[pos] != candidate.tile_role[pos]]
    assert len(diffs) == 1
    pos = diffs[0]
    assert base_schedule.tile_kind[pos] == "animal"
    assert candidate.tile_role[pos] in ANIMAL_NAMES
    assert candidate.tile_role[pos] != base_schedule.tile_role[pos]


def test_resize_crew_changes_one_day_and_stays_non_negative(base_schedule):
    rng = random.Random(0)
    for _ in range(20):  # a few draws to exercise both +/- deltas
        candidate = _resize_crew(base_schedule, rng)
        assert candidate is not None
        diffs = {d for d in candidate.crew_size_by_day if candidate.crew_size_by_day[d] != base_schedule.crew_size_by_day.get(d)}
        assert len(diffs) <= 1
        assert all(v >= 0 for v in candidate.crew_size_by_day.values())


def test_shift_land_day_stays_within_the_season(base_schedule):
    rng = random.Random(0)
    for _ in range(20):
        candidate = _shift_land_day(base_schedule, rng)
        assert candidate is not None
        for day in candidate.land_days.values():
            assert 0 <= day < base_schedule.n_days


def test_destroy_repair_sometimes_explores_away_from_the_deterministic_plan(cfg):
    """`_destroy_repair` ignores the incumbent it's handed and rebuilds via build_schedule's own
    `rng_for_day` hook -- across enough seeds, at least one should land on a different tile
    assignment than the fully deterministic build (proving exploration actually fires), while
    every candidate stays a valid, populated plan."""
    deterministic = build_schedule(cfg)
    any_different = False
    for seed in range(10):
        candidate = _destroy_repair(deterministic, cfg, random.Random(seed), window_choices=(30,), allow_4th_quadrant=True)
        assert candidate.tile_role  # still a populated plan
        if candidate.tile_role != deterministic.tile_role:
            any_different = True
    assert any_different


def test_diverse_seed_schedules_includes_the_default_first(cfg, base_schedule):
    rng = random.Random(0)
    pool = diverse_seed_schedules(cfg, rng, 4)
    assert len(pool) == 4
    assert pool[0].tile_role == base_schedule.tile_role


def test_diverse_seed_schedules_restores_module_constants_after_perturbing(cfg):
    import kaggriculture.search.schedule as schedule_module

    before = schedule_module.TILES_PER_HAND
    diverse_seed_schedules(cfg, random.Random(0), 5)
    assert schedule_module.TILES_PER_HAND == before


@pytest.mark.slow
def test_evaluate_schedule_returns_a_finite_margin(cfg, base_schedule):
    opponents = [resolve_policy("pass", cfg)]
    score = evaluate_schedule(base_schedule, cfg, opponents, seed_set=[0, 1, 2])
    assert isinstance(score, float)
    assert score == score  # not NaN


@pytest.mark.slow
def test_evaluate_schedule_beats_pass_by_a_lot(cfg, base_schedule):
    """Sanity check tying this module to issue 013's own headline result: the seed plan should
    have a large positive margin against `pass` (013's exp-001: 40W-0L)."""
    opponents = [resolve_policy("pass", cfg)]
    score = evaluate_schedule(base_schedule, cfg, opponents, seed_set=list(range(5)))
    assert score > 1000


@pytest.mark.slow
def test_head_to_head_a_schedule_against_itself_is_undecided():
    from kaggriculture.search.schedule import Schedule

    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    result = head_to_head(schedule, schedule, cfg, seed_set=[0, 1, 2, 3])
    assert result["wins"] == 0
    assert result["losses"] == 0
    assert result["ties"] == result["n_games"]
    assert result["verdict"] == "undecided"


@pytest.mark.slow
def test_lns_search_is_reproducible_given_the_same_seed(cfg):
    opponents = [resolve_policy("pass", cfg), resolve_policy("starter", cfg)]

    def run():
        rng = random.Random(0)
        bases = diverse_seed_schedules(cfg, rng, 2)
        return lns_search(cfg, bases, opponents, seed_set=[0, 1, 2], n_iters=6, seed=0, n_threads=2)

    a, b = run(), run()
    assert a.incumbent.tile_role == b.incumbent.tile_role
    assert a.incumbent_score == b.incumbent_score


@pytest.mark.slow
def test_lns_search_never_ends_up_worse_than_its_own_seed(cfg, base_schedule):
    """The seed plan is always in base_schedules and always scored first, so incumbent_score
    (a running max) can never fall below it."""
    opponents = [resolve_policy("pass", cfg)]
    seed_score = evaluate_schedule(base_schedule, cfg, opponents, seed_set=[0, 1, 2, 3])
    result = lns_search(cfg, [base_schedule], opponents, seed_set=[0, 1, 2, 3], n_iters=8, seed=0, n_threads=2)
    assert result.incumbent_score >= seed_score


@pytest.mark.slow
def test_lns_cli_writes_experiment_files(tmp_path):
    from kaggriculture.search.lns import main

    out_dir = tmp_path / "exp"
    main(
        [
            "--opponents",
            "pass,starter",
            "--n-seeds",
            "3",
            "--holdout-n-seeds",
            "3",
            "--n-iters",
            "4",
            "--n-restarts",
            "2",
            "--episode-steps",
            "200",
            "--threads",
            "2",
            "--out",
            str(out_dir),
        ]
    )
    import json

    config = json.loads((out_dir / "config.json").read_text())
    assert config["n_iters"] == 4
    trace = json.loads((out_dir / "incumbent_trace.json").read_text())
    assert len(trace) == 5  # seed row + 4 iterations
    result = json.loads((out_dir / "result.json").read_text())
    assert "verdict_vs_seed_plan_holdout" in result

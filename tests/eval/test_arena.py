"""Tests for the eval arena (issue 011), including its two explicit acceptance criteria:
  - byte-identical policies -> CI contains 50%, verdict undecided
  - a policy known to be better by construction (baseline vs pass) -> `better` in well under
    100 games
"""

import json

import pytest

from kaggriculture.eval.arena import _next_experiment_dir, compare, evaluate_population, main


def test_identical_policies_are_undecided_pass_vs_pass():
    """The degenerate case: both players PASS forever, so every game ties exactly (not a 50/50
    split of wins and losses) -- the tie-gets-half-credit convention in stats.wilson_interval is
    what makes this correctly land at 50%, not a quirk of this specific matchup."""
    r = compare("pass", "pass", n_seeds=15, episode_steps=100)
    assert r.wins == 0
    assert r.losses == 0
    assert r.ties == r.n_games
    assert r.interval.lo <= 0.5 <= r.interval.hi
    assert r.verdict == "undecided"


def test_identical_policies_are_undecided_baseline_vs_baseline():
    """A less degenerate identical-policy case: two copies of a real, non-trivial deterministic
    agent. Farms mirror each other exactly (baseline's decisions depend only on "my farm", not
    on which seat it's in), so this also ties every game -- confirms the result above isn't
    specific to an agent that never acts."""
    r = compare("baseline", "baseline", n_seeds=8, episode_steps=200)
    assert r.ties == r.n_games
    assert r.interval.lo <= 0.5 <= r.interval.hi
    assert r.verdict == "undecided"


def test_baseline_beats_pass_decisively_under_100_games():
    """The acceptance criterion: baseline vs pass over the full season is called `better` well
    under 100 games. 40 seeds x 2 seat orders = 80 games; baseline should sweep it."""
    r = compare("baseline", "pass", n_seeds=40, episode_steps=720, max_n_for_sprt=80)
    assert r.n_games == 80
    assert r.n_games < 100
    assert r.verdict == "better"
    assert r.interval.lo > 0.5


def test_sprt_decides_within_100_games_sequentially():
    """Confirms the *sequential* stopping rule itself decides early, not just the fixed-n Wilson
    verdict -- checked incrementally, the way a real run would stop as soon as it can."""
    from kaggriculture.eval.agents import resolve_policy
    from kaggriculture.eval.stats import sprt
    from kaggriculture.sim import _sim_native as native
    from kaggriculture.sim.decode import default_config_dict, episode_config

    cfg_dict = default_config_dict()
    episode_cfg = episode_config(cfg_dict)
    baseline = resolve_policy("baseline", cfg_dict)
    pass_policy = resolve_policy("pass", cfg_dict)

    wins = losses = 0
    for seed in range(20):
        for a_seat in (0, 1):
            agents = (baseline, pass_policy) if a_seat == 0 else (pass_policy, baseline)
            result = native.run_episode(*agents, episode_cfg, seed)
            a_money, b_money = (
                (result.final_money[0], result.final_money[1]) if a_seat == 0 else (result.final_money[1], result.final_money[0])
            )
            if a_money > b_money:
                wins += 1
            elif a_money < b_money:
                losses += 1
            n = wins + losses
            if sprt(wins, losses).decision != "continue":
                assert n < 100
                return
    pytest.fail("SPRT never decided within the seed budget")


def test_paired_seeds_both_seats_uses_the_same_seed_set_twice():
    """Structural check: n_seeds seeds each produce exactly 2 games (both seat orders)."""
    r = compare("baseline", "pass", n_seeds=6, episode_steps=100)
    assert r.n_games == 12


def test_scenario_stratified_reporting_sums_to_the_total():
    r = compare("baseline", "pass", n_seeds=10, episode_steps=720)
    total_n = sum(iv.n for iv in r.by_regime.values())
    assert total_n == r.n_games


def test_population_evaluation_runs_challenger_against_every_opponent():
    results = evaluate_population("baseline", ["pass", "pass"], n_seeds=4, episode_steps=200)
    assert set(results) == {"pass"}  # dict keyed by opponent name; duplicate collapses, as expected
    assert results["pass"].n_games == 8


def test_next_experiment_dir_auto_numbers(tmp_path, monkeypatch):
    import kaggriculture.eval.arena as arena_module

    monkeypatch.setattr(arena_module, "EXPERIMENTS_DIR", tmp_path)
    first = _next_experiment_dir("foo")
    second = _next_experiment_dir("bar")
    assert first.name.startswith("exp-001-")
    assert second.name.startswith("exp-002-")


def test_cli_writes_result_json(tmp_path):
    out_dir = tmp_path / "exp"
    main(["pass", "pass", "--n", "3", "--episode-steps", "50", "--out", str(out_dir)])
    payload = json.loads((out_dir / "result.json").read_text())
    assert payload["policy_a"] == "pass"
    assert payload["result"]["verdict"] == "undecided"

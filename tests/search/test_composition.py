"""Tests for the LNS + sell-DP + terminal-liquidation composition (issue 028)."""

import pytest

from kaggriculture.search.composition import _track_shed_increases, build_composed_agent, measure_arrivals
from kaggriculture.search.schedule import build_schedule
from kaggriculture.sim.decode import default_config_dict


def _fake_agent_two_separate_arrivals_then_a_sale(obs: dict, config: dict) -> dict:
    """5 WHEAT on day 0, 3 more on day 1, then a sale on a LATER turn of day 1 -- buy and sell
    deliberately on different turns, so the correct gross-arrivals answer is unambiguous (see
    `_track_shed_increases`'s own docstring on the one edge case -- same-turn overlap -- this
    doesn't test)."""
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    if day == 0 and hour == 0:
        return {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 5]]}
    if day == 1 and hour == 0:
        return {"farmer": ["PASS"], "hands": [], "market": [["BUY_PRODUCT", "WHEAT", 3]]}
    if day == 1 and hour == 5:
        return {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 2]]}
    return {"farmer": ["PASS"], "hands": [], "market": []}


def test_track_shed_increases_measures_gross_arrivals_not_net_shed_level():
    config = dict(default_config_dict())
    config["episodeSteps"] = 3 * 24
    result = _track_shed_increases(_fake_agent_two_separate_arrivals_then_a_sale, config)
    assert result == {"WHEAT": {0: 5, 1: 3}}  # the day-1 sale of 2 must NOT reduce the day-1 arrival count


def test_track_shed_increases_ignores_a_product_that_never_arrives():
    config = dict(default_config_dict())
    config["episodeSteps"] = 2 * 24

    def agent(obs, config):
        return {"farmer": ["PASS"], "hands": [], "market": []}

    assert _track_shed_increases(agent, config) == {}


def test_measure_arrivals_runs_on_a_real_schedule_and_returns_only_positive_units():
    config = dict(default_config_dict())
    schedule = build_schedule(config)
    arrivals = measure_arrivals(schedule, config)
    assert arrivals  # the default schedule produces something over a full season
    for item, by_day in arrivals.items():
        for day, units in by_day.items():
            assert units > 0, (item, day, units)
            assert 0 <= day < config["episodeSteps"] // config["turnsPerDay"]


def test_measure_arrivals_diverges_from_schedule_arrivals_on_the_default_plan():
    """Documents the finding this issue's Revision names directly: `Schedule.arrivals` assumes a
    tile can grow a different crop cycle to cycle, but `schedule_agent`'s live execution only ever
    replants whatever `tile_role`'s FINAL snapshot says -- so the two disagree even for a fresh,
    never-LNS-touched default schedule, not just an LNS-mutated one. This test exists to catch a
    regression in that understanding (e.g. if `schedule_agent` is later changed to track
    multi-cycle crop choice), not to assert a specific numeric gap."""
    config = dict(default_config_dict())
    schedule = build_schedule(config)
    measured_totals = {k: sum(v.values()) for k, v in measure_arrivals(schedule, config).items()}
    analytical_totals = {k: sum(v.values()) for k, v in schedule.arrivals.items()}
    assert measured_totals != analytical_totals


def test_build_composed_agent_runs_a_short_episode_without_crashing():
    from kaggriculture.eval.agents import wrap_agent
    from kaggriculture.sim import _sim_native as native
    from kaggriculture.sim.decode import episode_config

    config = dict(default_config_dict())
    config["episodeSteps"] = 5 * 24  # short: this is a smoke test, not a full-season eval
    schedule = build_schedule(config)
    agent_fn = build_composed_agent(schedule, config, window_turns=24)
    policy = wrap_agent(agent_fn, config)
    ecfg = episode_config(config)
    result = native.run_episode(policy, native.TapePolicy([]), ecfg, 0)
    assert result.final_money[0] >= 0


@pytest.mark.slow
def test_composed_agent_beats_current017_and_lns014():
    """The issue's own acceptance criteria: the composition beats both of the pieces it's built
    from, individually, at a decisive sample size."""
    from kaggriculture.eval.agents import record_tape, wrap_agent
    from kaggriculture.search.agent import schedule_agent
    from kaggriculture.search.composition import _rebuild_lns_incumbent, head_to_head
    from kaggriculture.search.liquidation import terminal_liquidation_agent
    from kaggriculture.search.sell_dp import solve_all_products

    config = dict(default_config_dict())
    n_days = config["episodeSteps"] // config["turnsPerDay"]

    incumbent = _rebuild_lns_incumbent(config, threads=8)
    composed_policy = wrap_agent(build_composed_agent(incumbent, config), config)
    lns014_policy = record_tape(schedule_agent(incumbent, config), config)

    schedule016 = build_schedule(config)
    sell_results = solve_all_products(schedule016.arrivals, n_days, config["episodeSteps"], config["turnsPerDay"], discount=0.93)
    sell_plan016 = {item: r.plan for item, r in sell_results.items()}
    dp016_agent = schedule_agent(schedule016, config, sell_plan=sell_plan016)
    current017_policy = wrap_agent(terminal_liquidation_agent(dp016_agent, config), config)

    vs_current017 = head_to_head(composed_policy, current017_policy, config, n_seeds=20, n_threads=8, episode_steps=config["episodeSteps"])
    vs_lns014 = head_to_head(composed_policy, lns014_policy, config, n_seeds=20, n_threads=8, episode_steps=config["episodeSteps"])

    assert vs_current017["verdict"] == "better", vs_current017
    assert vs_lns014["verdict"] == "better", vs_lns014

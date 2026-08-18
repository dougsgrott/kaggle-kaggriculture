"""Tests for the greedy scheduler's expansion half (issue 013): schedule_agent() and the
"greedy" resolve_policy integration, including the issue's acceptance criteria against pass/
starter/random. NOTE: as documented in search/schedule.py's module docstring and issues/
013-greedy-scheduler.md's Revision section, this agent does NOT yet decisively beat issue 005's
baseline in head-to-head eval.arena -- that gap is left to issue 014's search, not asserted here.
"""

import pytest

from kaggriculture.eval.agents import resolve_policy
from kaggriculture.search.agent import schedule_agent
from kaggriculture.search.schedule import build_schedule
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import build_player_turn, default_config_dict, episode_config


def _run_greedy_smoke_episode(seed: int, episode_steps: int = 200):
    cfg = dict(default_config_dict())
    cfg["episodeSteps"] = episode_steps
    schedule = build_schedule(cfg)
    agent_fn = schedule_agent(schedule, cfg)
    max_orders = cfg["maxMarketOrdersPerTurn"]

    def callback(state, market, player):
        obs = native.build_observation(state, market, player)
        action = agent_fn(obs, cfg)
        assert isinstance(action, dict)
        assert set(action) == {"farmer", "hands", "market"}
        assert len(action["market"]) <= max_orders
        return build_player_turn(action, max_orders)

    policy = native.CallbackPolicy(callback)
    ecfg = episode_config(cfg)
    ecfg.episode_steps = episode_steps
    return native.run_episode(policy, native.TapePolicy([]), ecfg, seed)


def test_schedule_agent_runs_a_full_episode_without_crashing():
    result = _run_greedy_smoke_episode(seed=0)
    assert result.final_money[0] >= 0  # never goes negative


def test_schedule_agent_never_exceeds_the_market_order_budget_across_seeds():
    for seed in range(3):
        _run_greedy_smoke_episode(seed=seed)


def test_resolve_policy_greedy_returns_a_tape_policy():
    """The "greedy" branch records once into a native TapePolicy (see eval.agents.record_tape) --
    the fast path issues 014/015 need to seed thousands of restarts from this plan."""
    cfg = default_config_dict()
    policy = resolve_policy("greedy", cfg)
    assert isinstance(policy, native.TapePolicy)
    assert len(policy) == cfg["episodeSteps"]


def test_resolve_policy_greedy_is_cached_across_calls():
    from kaggriculture.eval.agents import _GREEDY_TAPE_CACHE

    _GREEDY_TAPE_CACHE.clear()
    cfg = default_config_dict()
    a = resolve_policy("greedy", cfg)
    b = resolve_policy("greedy", cfg)
    assert a is b


@pytest.mark.slow
def test_greedy_beats_pass_decisively():
    from kaggriculture.eval.arena import compare

    r = compare("greedy", "pass", n_seeds=20, episode_steps=720)
    assert r.verdict == "better"
    assert r.interval.lo > 0.5


@pytest.mark.slow
def test_greedy_beats_starter_decisively():
    from kaggriculture.eval.arena import compare

    r = compare("greedy", "starter", n_seeds=15, episode_steps=720)
    assert r.verdict == "better"


@pytest.mark.slow
def test_greedy_beats_random_decisively():
    from kaggriculture.eval.arena import compare

    r = compare("greedy", "random", n_seeds=15, episode_steps=720)
    assert r.verdict == "better"

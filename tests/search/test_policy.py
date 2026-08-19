"""Tests for the compact parametric policy (issue 015)."""

import math

import pytest

from kaggriculture.eval.agents import wrap_agent
from kaggriculture.search.policy import (
    N_PARAMS,
    PARAM_NAMES,
    PARAM_SPECS,
    _count_planted,
    _rank_crops,
    decode_params,
    default_x0,
    policy_agent,
)
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import default_config_dict, episode_config


def test_param_specs_bounds_are_well_formed():
    assert N_PARAMS == len(PARAM_SPECS) == len(PARAM_NAMES)
    seen = set()
    for name, lo, hi, default in PARAM_SPECS:
        assert name not in seen, f"duplicate param name {name!r}"
        seen.add(name)
        assert lo < hi
        assert lo <= default <= hi


def test_default_x0_decodes_back_to_the_specs_defaults():
    decoded = decode_params(default_x0())
    for name, _lo, _hi, default in PARAM_SPECS:
        assert decoded[name] == pytest.approx(default, abs=1e-6)


def test_decode_params_stays_within_bounds_at_extreme_raw_values():
    extreme_low = [-50.0] * N_PARAMS
    extreme_high = [50.0] * N_PARAMS
    low = decode_params(extreme_low)
    high = decode_params(extreme_high)
    for name, lo, hi, _default in PARAM_SPECS:
        assert lo <= low[name] <= hi
        assert lo <= high[name] <= hi
        assert low[name] == pytest.approx(lo, abs=1e-3)
        assert high[name] == pytest.approx(hi, abs=1e-3)


def test_decode_params_is_monotonic_in_each_raw_coordinate():
    base = default_x0()
    for i, (name, lo, hi, _default) in enumerate(PARAM_SPECS):
        lower_x = list(base)
        higher_x = list(base)
        lower_x[i] -= 1.0
        higher_x[i] += 1.0
        assert decode_params(lower_x)[name] < decode_params(base)[name] < decode_params(higher_x)[name]


def test_rank_crops_matches_known_base_price_economics_with_no_saturation():
    """discussion 734033 / model.economics: melon dominates at base prices with no commitment
    yet -- see 013's own economics-ranking test for the same fact."""
    from kaggriculture.model.constants import MARKET_PARAMS

    prices = {c: MARKET_PARAMS[c]["base"] for c in MARKET_PARAMS if c != "FERTILIZER"}
    weights = decode_params(default_x0())
    ranked = _rank_crops(prices, weights, planted_counts={})
    assert ranked[0][1] == "MELON"


def test_rank_crops_saturation_demotes_an_overplanted_crop():
    from kaggriculture.model.constants import MARKET_PARAMS

    prices = {c: MARKET_PARAMS[c]["base"] for c in MARKET_PARAMS if c != "FERTILIZER"}
    weights = decode_params(default_x0())
    unsaturated = _rank_crops(prices, weights, planted_counts={})
    saturated = _rank_crops(prices, weights, planted_counts={"MELON": 20})
    melon_ppa_before = next(ppa for ppa, crop in unsaturated if crop == "MELON")
    melon_ppa_after = next(ppa for ppa, crop in saturated if crop == "MELON")
    assert melon_ppa_after < melon_ppa_before


def test_count_planted_counts_only_plant_tiles_by_crop():
    farm_tiles = [
        [None, {"kind": "PLANT", "crop": "MELON"}, {"kind": "WEED"}],
        [{"kind": "PLANT", "crop": "MELON"}, {"kind": "PLANT", "crop": "CARROT"}, "LOCKED"],
        [{"kind": "COOP", "animal": "GOOSE"}, None, None],
    ]
    counts = _count_planted(farm_tiles)
    assert counts == {"MELON": 2, "CARROT": 1}


def _run_policy_episode(params, seed, opponent=None, episode_steps=720):
    cfg = dict(default_config_dict())
    cfg["episodeSteps"] = episode_steps
    agent_fn = policy_agent(params, cfg)
    policy = wrap_agent(agent_fn, cfg)
    ecfg = episode_config(cfg)
    ecfg.episode_steps = episode_steps
    return native.run_episode(policy, opponent or native.TapePolicy([]), ecfg, seed)


def test_policy_agent_runs_a_full_episode_without_crashing():
    params = decode_params(default_x0())
    result = _run_policy_episode(params, seed=0)
    assert result.final_money[0] >= 0


def test_policy_agent_action_shape_is_valid_every_turn():
    params = decode_params(default_x0())
    cfg = dict(default_config_dict())
    cfg["episodeSteps"] = 100
    agent_fn = policy_agent(params, cfg)
    max_orders = cfg["maxMarketOrdersPerTurn"]

    from kaggriculture.sim.decode import build_player_turn

    def callback(state, market, player):
        obs = native.build_observation(state, market, player)
        action = agent_fn(obs, cfg)
        assert set(action) == {"farmer", "hands", "market"}
        assert len(action["market"]) <= max_orders
        return build_player_turn(action, max_orders)

    policy = native.CallbackPolicy(callback)
    ecfg = episode_config(cfg)
    ecfg.episode_steps = 100
    native.run_episode(policy, native.TapePolicy([]), ecfg, 0)


def test_policy_agent_is_deterministic_given_the_same_seed():
    params = decode_params(default_x0())
    a = _run_policy_episode(params, seed=3)
    b = _run_policy_episode(params, seed=3)
    assert a.final_money == b.final_money


@pytest.mark.slow
def test_default_params_beat_pass_decisively_across_seeds():
    """The untuned x0 anchor point should already be solidly profitable (not just non-crashing) --
    see the issue's Revision section for the exact numbers this was tuned to clear."""
    params = decode_params(default_x0())
    moneys = [_run_policy_episode(params, seed=s).final_money[0] for s in range(6)]
    assert all(m > 5000 for m in moneys), moneys


@pytest.mark.slow
def test_zeroing_a_crop_weight_removes_it_from_the_plan():
    """A real behavioural check, not just a units test: driving crop_weight_MELON to its floor
    should mean the policy essentially never plants melon, verified by inspecting the final
    board state directly."""
    params = decode_params(default_x0())
    params["crop_weight_MELON"] = 0.1  # PARAM_SPECS' own floor
    cfg = dict(default_config_dict())
    agent_fn = policy_agent(params, cfg)
    max_orders = cfg["maxMarketOrdersPerTurn"]

    from kaggriculture.sim.decode import build_player_turn

    melon_tiles_ever = 0

    def callback(state, market, player):
        nonlocal melon_tiles_ever
        obs = native.build_observation(state, market, player)
        for row in obs["farms"][player]["tiles"]:
            for tile in row:
                if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "MELON":
                    melon_tiles_ever += 1
        action = agent_fn(obs, cfg)
        return build_player_turn(action, max_orders)

    policy = native.CallbackPolicy(callback)
    ecfg = episode_config(cfg)
    native.run_episode(policy, native.TapePolicy([]), ecfg, 0)
    assert melon_tiles_ever == 0

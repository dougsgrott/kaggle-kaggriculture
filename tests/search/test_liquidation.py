"""Tests for issue 017's terminal liquidation optimizer (search.liquidation)."""

import pytest

from kaggriculture.model.constants import MARKET_I0
from kaggriculture.search.agent import schedule_agent
from kaggriculture.search.liquidation import (
    _my_revenue_prefix_with_opponent,
    _opponent_trajectory,
    dump_everything_agent,
    dump_everything_first_turn,
    mirror_opponent_schedule,
    simulate_concurrent_sale,
    solve_endgame,
    terminal_liquidation_agent,
    wind_down_day,
)
from kaggriculture.search.schedule import build_schedule
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import build_player_turn, default_config_dict, episode_config


def test_simulate_concurrent_sale_matches_the_vendor_engine_exactly():
    """Ground truth: two SELL orders for the same item, same turn, replayed directly through
    `vendor.kaggriculture._process_market` with hand-seeded farm/private/market state -- not
    assumed, checked. See the issue's Revision section for how this was derived."""
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("_vendor_liquidation_test", repo_root / "vendor" / "kaggriculture.py")
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    def run_vendor_case(qty_a, qty_b, item="WOOL", start_shed=200):
        market = vendor._new_market()
        farms = [vendor._new_farm(10, 3000), vendor._new_farm(10, 3000)]
        privates = [vendor._new_private(), vendor._new_private()]
        privates[0]["shed"][item] = start_shed
        privates[1]["shed"][item] = start_shed

        class FakeState:
            def __init__(self, action):
                self.action = action
                self.observation = type("Obs", (), {})()

        class FakeEnv:
            def __init__(self, configuration, info):
                self.configuration = configuration
                self.info = info

        states = [FakeState({"market": [["SELL", item, qty_a]]}), FakeState({"market": [["SELL", item, qty_b]]})]
        for i, s in enumerate(states):
            s.observation.farms = farms
            s.observation.market = market
            s.observation.town = vendor._new_town()
            s.observation.private = privates[i]

        cfg = {"boardSize": 10, "maxMarketOrdersPerTurn": 10, "farmHandCostMult": 1, "shedCapacity": 1000}
        vendor._process_market(states, FakeEnv(cfg, {"seed": 0}))
        return farms[0]["money"] - 3000, farms[1]["money"] - 3000

    for qty_a, qty_b in [(40, 40), (50, 20), (50, 0), (0, 0), (17, 33)]:
        expected = run_vendor_case(qty_a, qty_b)
        got = simulate_concurrent_sale("WOOL", MARKET_I0, qty_a, qty_b)
        assert got == pytest.approx(expected), (qty_a, qty_b)


def test_simulate_concurrent_sale_is_symmetric_for_equal_orders():
    my_rev, opp_rev = simulate_concurrent_sale("MELON", MARKET_I0, 30, 30)
    assert my_rev == opp_rev


def test_simulate_concurrent_sale_a_smaller_order_gets_a_better_average_price():
    """The mechanic the issue's own "who gets the higher half" concern turns out to actually be
    about: whichever order is smaller exhausts first and keeps a better average price than the
    larger order's later units, which face a market growing at a slower combined rate alone."""
    big_rev, small_rev = simulate_concurrent_sale("WOOL", MARKET_I0, 50, 20)
    assert (small_rev / 20) > (big_rev / 50)


def test_concurrent_selling_costs_the_solo_seller_real_revenue():
    solo_rev, _ = simulate_concurrent_sale("WOOL", MARKET_I0, 50, 0)
    contested_rev, _ = simulate_concurrent_sale("WOOL", MARKET_I0, 50, 20)
    assert contested_rev < solo_rev


def test_wind_down_day_matches_first_yield_day_at_zero_committed():
    from kaggriculture.model.constants import CROPS

    n_days = 30
    for crop, cd in CROPS.items():
        assert wind_down_day(crop, n_days, {}) == n_days - 1 - cd["first_yield_day"]


def test_wind_down_day_is_none_once_saturated():
    assert wind_down_day("MELON", 30, {"MELON": 5000}) is None
    assert wind_down_day("CARROT", 30, {"CARROT": 5000}) is None


def test_wind_down_day_rejects_an_unknown_item():
    with pytest.raises(ValueError):
        wind_down_day("NOT_A_PRODUCT", 30, {})


def test_solve_endgame_matches_brute_force_on_a_tiny_scenario():
    item, holdings, window_turns, turn_start = "WOOL", 6, 3, 696
    base_inv = MARKET_I0 - 50
    unlocked_shops = ["YARN_STORE"]
    opponent_schedule = {0: 3}

    def revenue_of(plan):
        inv_at_turn = _opponent_trajectory(item, base_inv, opponent_schedule, window_turns, unlocked_shops, turn_start, 4, 24)
        total, r = 0.0, 0
        for t in range(window_turns):
            k = plan.get(t, 0)
            prefix = _my_revenue_prefix_with_opponent(item, inv_at_turn[t] + r, k, opponent_schedule.get(t, 0))
            total += prefix[-1]
            r += k
        return total

    best = -1.0
    for k0 in range(holdings + 1):
        for k1 in range(holdings - k0 + 1):
            k2 = holdings - k0 - k1
            best = max(best, revenue_of({0: k0, 1: k1, 2: k2}))

    plan = solve_endgame(item, holdings, window_turns, turn_start, base_inv, unlocked_shops, opponent_schedule)
    assert sum(plan.values()) == holdings
    assert revenue_of(plan) == pytest.approx(best)


def test_solve_endgame_fully_liquidates_by_the_last_turn():
    plan = solve_endgame("MELON", 20, 10, 0, MARKET_I0, [], opponent_schedule={})
    assert sum(plan.values()) == 20
    assert all(t < 10 for t in plan)


def test_solve_endgame_with_no_holdings_is_a_no_op():
    assert solve_endgame("MELON", 0, 10, 0, MARKET_I0, []) == {}


def test_mirror_opponent_schedule_dumps_everything_on_turn_zero():
    assert mirror_opponent_schedule(37) == {0: 37}
    assert mirror_opponent_schedule(0) == {}


def test_dump_everything_first_turn():
    assert dump_everything_first_turn(12) == {0: 12}
    assert dump_everything_first_turn(0) == {}


def _run_smoke_episode(agent_fn, cfg, seed=0):
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
    return native.run_episode(policy, native.TapePolicy([]), ecfg, seed)


def test_terminal_liquidation_agent_runs_a_full_episode_without_crashing():
    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    agent_fn = terminal_liquidation_agent(schedule_agent(schedule, cfg), cfg, window_turns=24)
    result = _run_smoke_episode(agent_fn, cfg)
    assert result.final_money[0] >= 0


def test_dump_everything_agent_runs_a_full_episode_without_crashing():
    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    agent_fn = dump_everything_agent(schedule_agent(schedule, cfg), cfg, window_turns=24)
    result = _run_smoke_episode(agent_fn, cfg)
    assert result.final_money[0] >= 0


@pytest.mark.slow
def test_terminal_liquidation_beats_dumping_everything_on_the_same_schedule():
    """The issue's own acceptance criterion, live (both agents share the identical production
    schedule -- the only difference is the final-day sell strategy): the endgame DP should beat
    the naive "sell everything on the last few turns" baseline decisively."""
    from kaggriculture.eval.agents import wrap_agent
    from kaggriculture.eval.stats import verdict, wilson_interval

    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    ecfg = episode_config(cfg)
    dp_policy = wrap_agent(terminal_liquidation_agent(schedule_agent(schedule, cfg), cfg, window_turns=24), cfg)
    naive_policy = wrap_agent(dump_everything_agent(schedule_agent(schedule, cfg), cfg, window_turns=24), cfg)

    wins = losses = ties = 0
    for seed in range(20):
        r1 = native.run_episode(dp_policy, naive_policy, ecfg, seed)
        r2 = native.run_episode(naive_policy, dp_policy, ecfg, seed)
        for dp_m, naive_m in [(r1.final_money[0], r1.final_money[1]), (r2.final_money[1], r2.final_money[0])]:
            if dp_m > naive_m:
                wins += 1
            elif dp_m < naive_m:
                losses += 1
            else:
                ties += 1
    n = wins + losses + ties
    interval = wilson_interval(wins + 0.5 * ties, n)
    assert verdict(interval) == "better"

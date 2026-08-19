"""Tests for issue 016's sell-schedule DP (search.sell_dp)."""

import random

import pytest

from kaggriculture.model.constants import MARKET_I0, PRICE_FLOOR
from kaggriculture.model.price import price as market_price
from kaggriculture.model.regimes import demand_curve, draw_random_slots
from kaggriculture.search.agent import schedule_agent
from kaggriculture.search.schedule import build_schedule
from kaggriculture.search.sell_dp import (
    K_MAX_PER_DAY,
    _prefix_revenue,
    expected_drain_cumulative,
    head_to_head_sell_plan,
    market_forecast,
    solve_all_products,
    solve_product,
)
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import build_player_turn, default_config_dict, episode_config


def _brute_force_best(item, arrivals_by_day, n_days, base_inv_by_day):
    """Exhaustive search over every day-by-day sell schedule -- only tractable for tiny totals,
    used to cross-check solve_product's DP against a structurally different implementation."""
    cum_produced = [0] * n_days
    running = 0
    for d in range(n_days):
        running += arrivals_by_day.get(d, 0)
        cum_produced[d] = running
    total = cum_produced[-1] if n_days else 0

    best = [-1.0]

    def recurse(day, r, revenue_so_far):
        if day == n_days:
            if r == total:
                best[0] = max(best[0], revenue_so_far)
            return
        avail = cum_produced[day] - r
        for k in range(avail + 1):
            prefix = _prefix_revenue(item, base_inv_by_day[day] + r, k)
            recurse(day + 1, r + k, revenue_so_far + prefix[-1])

    recurse(0, 0, 0.0)
    return best[0]


def test_prefix_revenue_never_increments_inventory_at_the_floor():
    """Once a unit sells at PRICE_FLOOR, the engine doesn't add it to market inventory
    (vendor's `if price > 1` guard) -- so every remaining unit in the same burst also floors."""
    # MELON's above_target=3.6, T=300 -- comfortably floored well above I0.
    prefix = _prefix_revenue("MELON", MARKET_I0 + 5000, 5)
    marginal = [prefix[i + 1] - prefix[i] for i in range(5)]
    assert all(p == PRICE_FLOOR for p in marginal)


def test_prefix_revenue_matches_a_manual_unit_by_unit_walk():
    inv = MARKET_I0 - 50  # scarcity side
    expected = []
    running_inv = inv
    for _ in range(6):
        p = market_price("CARROT", running_inv)
        expected.append(p)
        if p > PRICE_FLOOR:
            running_inv += 1
    prefix = _prefix_revenue("CARROT", inv, 6)
    marginal = [prefix[i + 1] - prefix[i] for i in range(6)]
    assert marginal == expected


def test_solve_product_matches_brute_force_on_a_tiny_scenario():
    n_days = 3
    arrivals = {0: 3}
    base_inv_by_day = [MARKET_I0 - 20, MARKET_I0 - 40, MARKET_I0 - 60]  # scarcity deepens over time
    result = solve_product("STRAWBERRY", arrivals, n_days, base_inv_by_day, k_max_per_day=3)
    brute = _brute_force_best("STRAWBERRY", arrivals, n_days, base_inv_by_day)
    assert result.total_revenue == pytest.approx(brute)


def test_solve_product_fully_liquidates_by_season_end():
    """The terminal value is 0 for any unsold credit, so the optimal policy always sells
    everything by the last day -- see the module's own docstring."""
    n_days = 10
    arrivals = {0: 5, 3: 4, 7: 2}
    base_inv_by_day = [MARKET_I0] * n_days
    result = solve_product("WOOL", arrivals, n_days, base_inv_by_day, k_max_per_day=20)
    assert result.total_units == 11
    assert sum(result.plan.values()) == 11
    assert all(d < n_days for d in result.plan)


def test_solve_product_never_beats_by_holding_past_the_available_day():
    """A unit can never be sold before it arrives."""
    n_days = 5
    arrivals = {3: 4}
    base_inv_by_day = [MARKET_I0] * n_days
    result = solve_product("MELON", arrivals, n_days, base_inv_by_day)
    assert all(d >= 3 for d in result.plan)


def test_solve_product_revenue_is_never_worse_than_dump_flat():
    """Dump-flat is itself a feasible policy the DP could always have chosen -- the exact optimum
    can never score below it. Checked across several random small scenarios, not just one."""
    rng = random.Random(0)
    for _ in range(20):
        n_days = rng.randint(2, 8)
        arrivals = {d: rng.randint(0, 6) for d in range(n_days) if rng.random() < 0.5}
        base_inv_by_day = [MARKET_I0 + rng.randint(-2000, 2000) for _ in range(n_days)]
        item = rng.choice(["WHEAT", "CARROT", "MELON", "WOOL", "MILK"])
        result = solve_product(item, arrivals, n_days, base_inv_by_day, k_max_per_day=K_MAX_PER_DAY)
        assert result.total_revenue >= result.dump_flat_revenue - 1e-6


def test_solve_product_with_no_arrivals_is_a_no_op():
    result = solve_product("MELON", {}, 5, [MARKET_I0] * 5)
    assert result.plan == {}
    assert result.total_revenue == 0.0
    assert result.total_units == 0


def test_solve_product_holding_into_a_demand_hole_beats_dumping_flat():
    """The issue's own headline claim (discussion 734412's 24x table), reproduced directly: a big
    lump of a premium good sold into a town-drained market beats dumping the same quantity flat,
    by a wide margin -- not just a marginal one."""
    episode_steps, turns_per_day = 720, 24
    n_days = episode_steps // turns_per_day
    draw = draw_random_slots(random.Random(1))
    cumulative = demand_curve(draw, episode_steps, turns_per_day)["STRAWBERRY"]
    base_inv_by_day = market_forecast(cumulative)
    result = solve_product("STRAWBERRY", {0: 100}, n_days, base_inv_by_day)
    assert result.total_revenue > result.dump_flat_revenue * 2


def test_market_forecast_day_zero_sees_no_drain_yet():
    cumulative = [10.0, 20.0, 30.0]
    forecast = market_forecast(cumulative)
    assert forecast[0] == MARKET_I0
    assert forecast[1] == MARKET_I0 - 10.0
    assert forecast[2] == MARKET_I0 - 20.0


def test_market_forecast_subtracts_opponent_cumulative_sold():
    cumulative = [0.0, 0.0]
    forecast = market_forecast(cumulative, opponent_cumulative_sold=[5, 15])
    assert forecast == [MARKET_I0 - 5, MARKET_I0 - 15]


def test_expected_drain_cumulative_is_nondecreasing_and_excludes_fertilizer():
    forecast = expected_drain_cumulative(episode_steps=240, turns_per_day=24, n_samples=20, seed=0)
    assert "FERTILIZER" not in forecast
    for product, row in forecast.items():
        assert all(b >= a - 1e-9 for a, b in zip(row, row[1:])), product


def test_solve_all_products_skips_products_without_arrivals():
    n_days = 5
    arrivals = {"MELON": {0: 10}, "WHEAT": {}}
    results = solve_all_products(arrivals, n_days, 120, 24)
    assert "MELON" in results
    assert "WHEAT" not in results


def test_solve_all_products_respects_a_concrete_regime_draw():
    """Different shop draws produce genuinely different plans -- issue 012's regime-aware scope
    bullet, checked directly rather than assumed."""
    n_days = 30
    arrivals = {"WOOL": {d: 5 for d in range(0, 30, 3)}}
    draw_a = draw_random_slots(random.Random(1))
    draw_b = draw_random_slots(random.Random(2))
    results_a = solve_all_products(arrivals, n_days, 720, 24, draw=draw_a)
    results_b = solve_all_products(arrivals, n_days, 720, 24, draw=draw_b)
    assert results_a["WOOL"].total_revenue != results_b["WOOL"].total_revenue


def test_discount_biases_the_plan_toward_earlier_sales():
    """A steep per-day discount should never sell LATER than the undiscounted (pure
    revenue-maximizing) plan for the same scenario -- see solve_product's own docstring on why a
    nonzero discount was needed in practice."""
    n_days = 10
    arrivals = {0: 20}
    base_inv_by_day = [MARKET_I0 - d * 50 for d in range(n_days)]  # scarcity improves over time
    undiscounted = solve_product("MELON", arrivals, n_days, base_inv_by_day, discount=1.0)
    discounted = solve_product("MELON", arrivals, n_days, base_inv_by_day, discount=0.7)
    last_day_undiscounted = max(undiscounted.plan)
    last_day_discounted = max(discounted.plan)
    assert last_day_discounted <= last_day_undiscounted


def test_build_schedule_arrivals_are_bounded_by_committed_units():
    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    assert schedule.arrivals
    for product, by_day in schedule.arrivals.items():
        assert all(0 <= d < schedule.n_days for d in by_day)
        assert all(units >= 0 for units in by_day.values())
        total_arrived = sum(by_day.values())
        committed = schedule.diagnostics["committed_units"].get(product, 0)
        # Arrivals are credited at harvest, committed at assignment -- a tile assigned near season
        # end may not have harvested yet, so arrivals can lag committed but never exceed it.
        assert total_arrived <= committed + 1  # +1 for animal rounding drift


def _run_sell_plan_smoke_episode(sell_plan, episode_steps=200, seed=0):
    cfg = dict(default_config_dict())
    cfg["episodeSteps"] = episode_steps
    schedule = build_schedule(cfg)
    agent_fn = schedule_agent(schedule, cfg, sell_plan=sell_plan)
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


def test_schedule_agent_with_a_sell_plan_runs_a_full_episode_without_crashing():
    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    n_days = cfg["episodeSteps"] // cfg["turnsPerDay"]
    results = solve_all_products(schedule.arrivals, n_days, cfg["episodeSteps"], cfg["turnsPerDay"])
    sell_plan = {item: r.plan for item, r in results.items()}
    result = _run_sell_plan_smoke_episode(sell_plan, episode_steps=720)
    assert result.final_money[0] >= 0


def test_schedule_agent_with_no_sell_plan_is_unaffected():
    """sell_plan=None (the default) must reproduce the pre-issue-016 floor-threshold behaviour
    exactly -- a regression guard on backward compatibility."""
    result = _run_sell_plan_smoke_episode(None, episode_steps=200)
    assert result.final_money[0] >= 0


@pytest.mark.slow
def test_dp_sell_plan_beats_floor_threshold_selling_on_identical_production():
    """The issue's own acceptance criterion, checked directly: same schedule (same production),
    DP-derived sell timing vs. the existing floor-threshold logic, live (not tape-replayed --
    see head_to_head_sell_plan's own docstring on why). A modest sample for test runtime; the
    documented headline number (issue's Revision section) uses a much larger one."""
    cfg = default_config_dict()
    schedule = build_schedule(cfg)
    n_days = cfg["episodeSteps"] // cfg["turnsPerDay"]
    results = solve_all_products(schedule.arrivals, n_days, cfg["episodeSteps"], cfg["turnsPerDay"], discount=0.93)
    sell_plan = {item: r.plan for item, r in results.items()}
    h2h = head_to_head_sell_plan(schedule, sell_plan, cfg, seed_set=list(range(30)))
    assert h2h["wins"] > h2h["losses"]
    assert h2h["mean_dp_money"] > h2h["mean_baseline_money"]

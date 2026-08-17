"""Tests for the exact-parity gate (issue 010). The fast tests here check the validator's own
diffing/decoding logic; the real acceptance criterion -- >=200 episodes against the real engine,
byte-exact -- is `test_full_parity_sweep`, marked slow (see pyproject.toml's addopts: skipped by
default, opt in with `pytest -m slow`).
"""

import copy

import pytest

from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.validate import (
    BUILTIN_AGENTS,
    _decode_market_order,
    _decode_unit_action,
    replay_and_diff,
    sweep_pairings,
    validate_seed,
)


def test_decode_unit_action_round_trips_known_ops():
    a = _decode_unit_action(["PLANT", "CARROT"])
    assert a.op == native.Op.PLANT
    assert a.item == native.Item.CARROT

    passed = _decode_unit_action(["PASS"])
    assert passed.op == native.Op.PASS

    empty = _decode_unit_action([])
    assert empty.op == native.Op.PASS

    malformed = _decode_unit_action("not a list")
    assert malformed.op == native.Op.PASS


def test_decode_unit_action_unrecognized_op_is_a_silent_pass():
    """Matches vendor: an op string _apply_unit_action doesn't recognize falls through every
    branch and does nothing -- Op.PASS is behaviourally identical, not just a placeholder."""
    a = _decode_unit_action(["SOME_FUTURE_OP", "WHEAT"])
    assert a.op == native.Op.PASS


def test_decode_market_order_hire_and_buy_land_carry_no_item():
    hire = _decode_market_order(["HIRE"])
    assert hire.op == native.MarketOp.HIRE

    land = _decode_market_order(["BUY_LAND"])
    assert land.op == native.MarketOp.BUY_LAND


def test_decode_market_order_rejects_non_positive_quantity():
    assert _decode_market_order(["SELL", "WHEAT", 0]).op == native.MarketOp.NONE
    assert _decode_market_order(["SELL", "WHEAT", -1]).op == native.MarketOp.NONE


def test_decode_market_order_sell_and_buy():
    sell = _decode_market_order(["SELL", "WHEAT", 5])
    assert (sell.op, sell.item, sell.n) == (native.MarketOp.SELL, native.Item.WHEAT, 5)


def test_sweep_pairings_covers_both_seat_orders_and_same_agent():
    pairings = sweep_pairings()
    assert set(pairings) == {
        ("starter", "random"),
        ("random", "starter"),
        ("starter", "pass"),
        ("pass", "starter"),
        ("random", "pass"),
        ("pass", "random"),
        ("starter", "starter"),
        ("random", "random"),
        ("pass", "pass"),
    }
    assert len(pairings) == 9


def test_validate_seed_passes_for_a_short_episode():
    result = validate_seed(1, ("starter", "random"), episode_steps=48)
    assert result.passed, result.divergence


def test_replay_and_diff_localizes_an_injected_money_divergence():
    """Meta-test of the validator itself: corrupt one recorded step's money and confirm
    replay_and_diff reports *exactly* that step as the first divergence -- the bisect-friendly
    failure report the issue's acceptance criterion asks for. This doesn't touch the C++ sim at
    all (see the module docstring for the separate real-bug-injection check that does)."""
    from kaggriculture.sim.export_trace import export_trace

    trace = export_trace(2, ("starter", "random"), episode_steps=48)
    assert replay_and_diff(trace).passed  # sanity: the real trace is clean before we corrupt it

    corrupted = copy.deepcopy(trace)
    corrupt_step = 10
    corrupted["money"][corrupt_step] = [corrupted["money"][corrupt_step][0] + 1.0, corrupted["money"][corrupt_step][1]]

    result = replay_and_diff(corrupted)
    assert not result.passed
    assert result.divergence.step == corrupt_step
    assert result.divergence.field == "money"


def test_replay_and_diff_localizes_an_injected_inventory_divergence():
    from kaggriculture.sim.export_trace import export_trace

    trace = export_trace(3, ("random", "random"), episode_steps=48)
    assert replay_and_diff(trace).passed

    corrupted = copy.deepcopy(trace)
    corrupt_step = 15
    corrupted["inventory"][corrupt_step][0] += 1

    result = replay_and_diff(corrupted)
    assert not result.passed
    assert result.divergence.step == corrupt_step
    assert result.divergence.field == "inventory"


def test_quick_parity_smoke():
    """A small, fast (few short episodes) parity check for the normal test run -- not the
    acceptance criterion itself (that's test_full_parity_sweep, slow-marked), just a tripwire
    that fires immediately if a change to sim/ broke something obvious."""
    for pairing in [("starter", "random"), ("random", "pass"), ("pass", "pass")]:
        for seed in (1, 2):
            result = validate_seed(seed, pairing, episode_steps=72)
            assert result.passed, f"{pairing} seed={seed}: {result.divergence}"


@pytest.mark.slow
def test_full_parity_sweep():
    """The acceptance criterion: >=200 episodes across seeds x {starter,random,pass} pairings in
    both seat orders, exact equality on money and market inventory at every step. Takes minutes
    (dominated by the real engine, not the C++ sim) -- opt in with `pytest -m slow`."""
    seeds = range(23)  # 23 seeds x 9 pairings = 207 >= 200
    failures = []
    total = 0
    for pairing in sweep_pairings(BUILTIN_AGENTS):
        for seed in seeds:
            total += 1
            result = validate_seed(seed, pairing, episode_steps=720)
            if not result.passed:
                failures.append(result)

    assert total >= 200
    if failures:
        first = failures[0]
        pytest.fail(
            f"{len(failures)}/{total} episodes diverged from the real engine. First failure: "
            f"seed={first.seed} agents={first.agents}\n{first.divergence}"
        )

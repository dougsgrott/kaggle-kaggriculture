"""Tests for kaggriculture.sim.decode -- the real-engine-action-shape -> native-type translator
shared by issue 010's validate.py and issue 011's eval arena.
"""

from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import decode_market_order, decode_unit_action


def test_decode_unit_action_round_trips_known_ops():
    a = decode_unit_action(["PLANT", "CARROT"])
    assert a.op == native.Op.PLANT
    assert a.item == native.Item.CARROT

    passed = decode_unit_action(["PASS"])
    assert passed.op == native.Op.PASS

    empty = decode_unit_action([])
    assert empty.op == native.Op.PASS

    malformed = decode_unit_action("not a list")
    assert malformed.op == native.Op.PASS


def test_decode_unit_action_unrecognized_op_is_a_silent_pass():
    """Matches vendor: an op string _apply_unit_action doesn't recognize falls through every
    branch and does nothing -- Op.PASS is behaviourally identical, not just a placeholder."""
    a = decode_unit_action(["SOME_FUTURE_OP", "WHEAT"])
    assert a.op == native.Op.PASS


def test_decode_market_order_hire_and_buy_land_carry_no_item():
    hire = decode_market_order(["HIRE"])
    assert hire.op == native.MarketOp.HIRE

    land = decode_market_order(["BUY_LAND"])
    assert land.op == native.MarketOp.BUY_LAND


def test_decode_market_order_rejects_non_positive_quantity():
    assert decode_market_order(["SELL", "WHEAT", 0]).op == native.MarketOp.NONE
    assert decode_market_order(["SELL", "WHEAT", -1]).op == native.MarketOp.NONE


def test_decode_market_order_sell_and_buy():
    sell = decode_market_order(["SELL", "WHEAT", 5])
    assert (sell.op, sell.item, sell.n) == (native.MarketOp.SELL, native.Item.WHEAT, 5)

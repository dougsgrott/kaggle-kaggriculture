"""Tests for kaggriculture.eval.stats -- Wilson intervals, the verdict rule, and SPRT."""

import pytest

from kaggriculture.eval.stats import sprt, verdict, wilson_interval


def test_wilson_interval_at_zero_n_is_maximally_uncertain():
    iv = wilson_interval(0, 0)
    assert iv.lo == 0.0
    assert iv.hi == 1.0


@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
def test_fifty_fifty_score_always_contains_half(n):
    iv = wilson_interval(n / 2, n)
    assert iv.lo <= 0.5 <= iv.hi
    assert verdict(iv) == "undecided"


def test_verdict_better_when_ci_clears_threshold():
    iv = wilson_interval(95, 100)
    assert verdict(iv) == "better"


def test_verdict_worse_when_ci_is_entirely_below_threshold():
    iv = wilson_interval(5, 100)
    assert verdict(iv) == "worse"


def test_verdict_undecided_at_small_n_even_with_lopsided_score():
    # 4/5 wins looks decisive but n is far too small to be sure.
    iv = wilson_interval(4, 5)
    assert verdict(iv) == "undecided"


def test_sprt_continues_with_no_data():
    r = sprt(0, 0)
    assert r.decision == "continue"
    assert r.llr == 0.0


def test_sprt_decides_better_for_a_strong_effect():
    # A true win rate near 1.0 crosses the p0=0.5/p1=0.55 upper bound in ~31 straight wins.
    r = sprt(35, 0)
    assert r.decision == "better"


def test_sprt_decides_worse_for_a_strong_negative_effect():
    r = sprt(0, 35)
    assert r.decision == "worse"


def test_sprt_stays_undecided_near_the_null():
    r = sprt(26, 24)  # ~52% observed, close to p0=0.5
    assert r.decision == "continue"

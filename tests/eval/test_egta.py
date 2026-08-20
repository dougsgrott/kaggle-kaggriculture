"""Tests for the empirical payoff-matrix analysis (issue 018): transitivity/cycle detection and
fictitious play, exercised against small hand-built matrices (not the real population -- that's
covered by test_population.py and this issue's own experiment run) so these stay fast and the
expected outcome is provable by construction rather than empirically observed."""

import pytest

from kaggriculture.eval.egta import (
    PayoffCell,
    cyclic_triples,
    fictitious_play,
    payoff,
    transitivity_report,
    verdict_of,
    what_beats_the_reference,
)


def _cell(p_hat: float, lo: float, hi: float) -> PayoffCell:
    v = "better" if lo > 0.5 else ("worse" if hi < 0.5 else "undecided")
    return PayoffCell(p_hat=p_hat, lo=lo, hi=hi, n_games=40, verdict=v)


# A strictly transitive chain: A > B > C, every pair decisive, no cycle.
_TRANSITIVE = {
    ("A", "B"): _cell(0.9, 0.8, 1.0),
    ("A", "C"): _cell(0.95, 0.85, 1.0),
    ("B", "C"): _cell(0.8, 0.65, 0.95),
}

# A rock-paper-scissors triple: A beats B, B beats C, C beats A, every pair decisive.
_CYCLIC = {
    ("A", "B"): _cell(0.9, 0.8, 1.0),
    ("B", "C"): _cell(0.9, 0.8, 1.0),
    ("A", "C"): _cell(0.1, 0.0, 0.2),  # stored (A, C); mirrors to C beats A at 0.9
}


def test_payoff_mirrors_the_stored_cell():
    assert payoff(_TRANSITIVE, "A", "B") == 0.9
    assert payoff(_TRANSITIVE, "B", "A") == pytest.approx(0.1)
    assert payoff(_TRANSITIVE, "A", "A") == 0.5


def test_verdict_of_mirrors_better_and_worse():
    assert verdict_of(_TRANSITIVE, "A", "B") == "better"
    assert verdict_of(_TRANSITIVE, "B", "A") == "worse"
    assert verdict_of(_TRANSITIVE, "A", "A") == "undecided"


def test_cyclic_triples_finds_the_rock_paper_scissors_pattern():
    cycles = cyclic_triples(_CYCLIC, ["A", "B", "C"])
    assert cycles == [("A", "B", "C")]


def test_cyclic_triples_finds_nothing_in_a_transitive_chain():
    assert cyclic_triples(_TRANSITIVE, ["A", "B", "C"]) == []


def test_transitivity_report_identifies_the_dominant_strategy():
    report = transitivity_report(_TRANSITIVE, ["A", "B", "C"])
    assert report.dominant_strategy == "A"
    assert report.n_cyclic_triples == 0


def test_transitivity_report_has_no_dominant_strategy_under_a_cycle():
    report = transitivity_report(_CYCLIC, ["A", "B", "C"])
    assert report.dominant_strategy is None
    assert report.n_cyclic_triples == 1


def test_fictitious_play_converges_to_the_dominant_pure_strategy():
    """Under strict dominance the best response is the same name every single round from
    iteration 1 on -- this must converge to a fixed point regardless of tie-breaking details,
    unlike a genuine cyclic population where the exact trajectory is delicate (Shapley's classic
    non-convergence example) and not something to hard-code an assumption about."""
    fp = fictitious_play(_TRANSITIVE, ["A", "B", "C"], iterations=100)
    assert fp.converged_to_fixed_point
    assert fp.average_strategy["A"] == 1.0
    assert fp.average_strategy["B"] == 0.0
    assert fp.average_strategy["C"] == 0.0
    assert set(fp.best_response_sequence) == {"A"}


def test_fictitious_play_never_settles_on_a_single_strategy_under_a_cycle():
    """No hard-coded claim about *which* pattern a cyclic population's best-response sequence
    settles into (that is an empirical question the module docstring is explicit about, and the
    repo's `main()`/notes writeup reports it for the real population) -- only that it can't
    converge to a single pure strategy the way the transitive case provably does, since every
    pure strategy in an RPS triple is beaten by another with the same margin."""
    fp = fictitious_play(_CYCLIC, ["A", "B", "C"], iterations=300)
    assert not fp.converged_to_fixed_point
    assert len(set(fp.best_response_sequence)) > 1


def test_what_beats_the_reference_reads_directly_off_the_matrix():
    report = what_beats_the_reference(_TRANSITIVE, ["A", "B", "C"], reference="B")
    assert report["beats_reference"] == ["A"]
    assert report["reference_beats"] == ["C"]
    assert report["undecided_against"] == []

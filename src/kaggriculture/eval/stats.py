"""Wilson score intervals, a verdict rule, and a sequential probability ratio test (SPRT) for
the eval arena (issue 011). Two questions, two tools: "what's the win rate, with uncertainty"
(Wilson) and "can we stop early" (SPRT) -- CLAUDE.md's non-negotiable #4: no claim without an
interval, and `undecided` is a first-class outcome, not a failure to compute one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975), hardcoded so this has no scipy dependency


@dataclass
class WilsonInterval:
    n: int
    score: float  # wins + 0.5*ties -- see wilson_interval's docstring on why ties get half credit
    p_hat: float
    lo: float
    hi: float

    def to_dict(self) -> dict:
        return {"n": self.n, "score": self.score, "p_hat": self.p_hat, "lo": self.lo, "hi": self.hi}


def wilson_interval(score: float, n: int, z: float = Z_95) -> WilsonInterval:
    """The Wilson score interval for a binomial proportion -- better-calibrated than the naive
    normal approximation at small n or p near 0/1, both common here (a challenger that's either
    obviously terrible or evaluated on few games).

    `score` is wins + 0.5*ties, the standard Elo/rating-system convention for games with a draw
    outcome (this engine's identical-policy self-play is not a corner case: two byte-for-byte
    identical deterministic policies -- e.g. baseline vs baseline -- mirror each other's farm
    exactly and tie exactly, every game). Treating a tie as a win for the numerator undercounts
    it; excluding it changes what `n` means turn to turn. Half credit keeps `n_games` fixed and
    correctly centers two-identical-policies at score/n = 0.5 -- exactly the acceptance criterion
    ("a win rate whose CI contains 50%, and a verdict of undecided"). Not a strict binomial CI
    once ties are involved (the underlying trial is now three-valued), but the standard practical
    approximation for this exact situation.
    """
    if n == 0:
        return WilsonInterval(0, 0.0, 0.5, 0.0, 1.0)
    p = score / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return WilsonInterval(n, score, p, max(0.0, center - margin), min(1.0, center + margin))


def verdict(interval: WilsonInterval, threshold: float = 0.5) -> str:
    """`better` / `worse` only when the CI clears 50% entirely; `undecided` otherwise -- this is
    the "refuses to answer when n is too small" behaviour the issue requires as first-class, not
    an edge case."""
    if interval.n == 0:
        return "undecided"
    if interval.lo > threshold:
        return "better"
    if interval.hi < threshold:
        return "worse"
    return "undecided"


@dataclass
class SPRTResult:
    llr: float
    upper: float
    lower: float
    p0: float
    p1: float
    decision: str  # "better", "worse", or "continue"

    def to_dict(self) -> dict:
        return {"llr": self.llr, "upper": self.upper, "lower": self.lower, "p0": self.p0, "p1": self.p1, "decision": self.decision}


def sprt(wins: int, losses: int, p0: float = 0.5, p1: float = 0.55, alpha: float = 0.05, beta: float = 0.05) -> SPRTResult:
    """Wald's SPRT for H0: win rate = p0 (no effect) vs H1: win rate = p1 (the smallest effect
    worth caring about). Ties count toward neither `wins` nor `losses` -- see arena.py's outcome
    classification. Log-likelihood ratio accumulates additively per game, so this is cheap to
    evaluate after every game rather than only at the end; that's what makes it "sequential" --
    an obviously-better challenger crosses the upper bound in a handful of games, a marginal one
    wanders between the bounds and keeps running (or hits a caller-imposed max_n and falls back
    to the Wilson verdict -- SPRT alone has no runtime bound when the true rate sits between p0
    and p1).
    """
    llr = 0.0
    if wins:
        llr += wins * math.log(p1 / p0)
    if losses:
        llr += losses * math.log((1 - p1) / (1 - p0))
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    if llr >= upper:
        decision = "better"
    elif llr <= lower:
        decision = "worse"
    else:
        decision = "continue"
    return SPRTResult(llr, upper, lower, p0, p1, decision)

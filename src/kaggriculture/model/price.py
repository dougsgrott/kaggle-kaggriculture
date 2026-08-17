"""The market price curve, standalone from the engine's `_process_market` loop.

    price(inv) = base + sign * amp * f(|inv - I0|)
    sign = +1 below I0 (scarcity), -1 above I0 (glut)
    amp  = target * base / f(T)      (derived, not stored)
    f in {linear, sq, sqrt, log, log10, hinge}

Floored at PRICE_FLOOR and rounded to the nearest dollar. See wiki/competition/pages/
how-to-play.md ("The Price Function") and vendor/kaggriculture.py's `market_price`, which
this module is a byte-for-byte reimplementation of (see tests/model/test_price.py).
"""

import math

from kaggriculture.model.constants import HINGE_GAIN, MARKET_PARAMS, PRICE_FLOOR


def shape(func: str, x: float, T: float | None = None) -> float:
    """One of the six curve shapes, evaluated at x >= 0 (negative x is clamped to 0)."""
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    if func == "log10":
        return math.log10(1.0 + x)
    if func == "hinge":
        # Linear in u = x/T below the knee; a quadratic term takes over above it.
        # f(T) == 1 by construction. Degenerates to linear if T is missing/non-positive.
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    raise ValueError(f"unknown shape function: {func!r}")


def price(item: str, inventory: float, params: dict | None = None) -> int:
    """The quoted price for `item` at the given market inventory level."""
    p = (params or MARKET_PARAMS)[item]
    base, I0, T = p["base"], p["I0"], p["T"]
    if inventory < I0:
        f, target = p["below_func"], p["below_target"]
        amp = target * base / shape(f, T, T)
        raw = base + amp * shape(f, I0 - inventory, T)
    else:
        f, target = p["above_func"], p["above_target"]
        amp = target * base / shape(f, T, T)
        raw = base - amp * shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, round(raw))


def units_sellable_before(item: str, inventory: int, threshold: int, params: dict | None = None) -> int | None:
    """How many consecutive SELL units, starting at `inventory`, until price <= threshold.

    Mirrors the engine's per-unit lockstep exactly for threshold > PRICE_FLOOR: each unit sold
    raises inventory by 1 (a sale at the $1 floor is the one exception, and does not raise
    inventory further — see `_commit_unit`), and price(inv) is monotonic non-increasing in
    inventory, so the count is a binary search rather than a per-unit replay.

    Returns 0 if already at or below threshold. Returns None if threshold <= PRICE_FLOOR: once
    the floor is hit, inventory stops climbing and every further unit still sells at $1 forever,
    so "before" never arrives.
    """
    if threshold <= PRICE_FLOOR:
        return None
    if price(item, inventory, params) <= threshold:
        return 0

    lo, hi = inventory, inventory + 1
    while price(item, hi, params) > threshold:
        hi = inventory + (hi - inventory) * 2

    while lo < hi:
        mid = (lo + hi) // 2
        if price(item, mid, params) <= threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo - inventory

"""Shop-draw scenario taxonomy and demand regimes (issue 012).

PR #1394 made shop unlocks a with-replacement draw: 8 instances, one per `townShopUnlockInterval`
(default 3) days, each drawn uniformly from the 8 shop types, landing on days 3, 6, ..., 24. Which
products end up scarce for the rest of the season is therefore a per-episode random variable, not
a fixed schedule -- this module characterizes that variable: the draw distribution, what it does
to per-product demand (and therefore price, via `model.price`) assuming zero production, and a
small named taxonomy of "regimes" a policy could condition on.

"Viable" (PR #1399's own framing, discussion 735311): a product's price starts increasing
significantly once town demand *alone* -- no player production -- drains more than `T` units
from it. That's exactly `hinge`'s knee (`u = x/T > 1`); see `demand_curve()` below.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from kaggriculture.model.constants import CONFIG_DEFAULTS, MARKET_I0, MARKET_PARAMS, SHOPS
from kaggriculture.model.price import price as market_price
from kaggriculture.model.town import drain_at_step

SHOP_NAMES: list[str] = sorted(SHOPS)  # the exact draw pool, vendor's own `sorted(SHOPS)`
N_SLOTS = 8
UNLOCK_INTERVAL = CONFIG_DEFAULTS["townShopUnlockInterval"]
SLOT_UNLOCK_DAYS = [UNLOCK_INTERVAL * k for k in range(1, N_SLOTS + 1)]  # [3, 6, ..., 24]
NON_FERTILIZER_PRODUCTS = [p for p in MARKET_PARAMS if p != "FERTILIZER"]


def draw_random_slots(rng: random.Random | None = None) -> list[str]:
    """One shop-draw sample: a shop type for each of the 8 unlock slots, i.i.d. uniform. This is
    a statistical characterization of the *distribution*, not a bit-exact replay of vendor's own
    RNG stream (issue 010 owns that) -- rejection sampling makes uniform-choice-of-8 exactly
    uniform regardless of implementation, and this is cross-checked against the native sim's own
    (bit-exact) RNG in tests, so there is no distributional gap to worry about here."""
    rng = rng or random
    return [rng.choice(SHOP_NAMES) for _ in range(N_SLOTS)]


def demand_curve(
    draw: list[str],
    episode_steps: int = CONFIG_DEFAULTS["episodeSteps"],
    turns_per_day: int = CONFIG_DEFAULTS["turnsPerDay"],
) -> dict[str, list[int]]:
    """Cumulative units drained from the market for each non-fertilizer product, per day,
    assuming zero production (the only thing touching market inventory is the town) -- reuses
    `model.town.drain_at_step` turn by turn, so this is exactly vendor's own drain formula, not a
    re-derivation. `draw[k]` is the shop unlocked at `SLOT_UNLOCK_DAYS[k]`.

    Returns `{product: [cumulative_at_day_0, cumulative_at_day_1, ...]}`, one entry per day.
    """
    n_days = episode_steps // turns_per_day
    cumulative = {p: [0] * n_days for p in NON_FERTILIZER_PRODUCTS}
    running = dict.fromkeys(NON_FERTILIZER_PRODUCTS, 0)
    unlocked_so_far: list[str] = []
    slot_idx = 0
    for day in range(n_days):
        while slot_idx < len(draw) and SLOT_UNLOCK_DAYS[slot_idx] <= day:
            unlocked_so_far.append(draw[slot_idx])
            slot_idx += 1
        for step in range(day * turns_per_day, (day + 1) * turns_per_day):
            for item, n in drain_at_step(unlocked_so_far, step).items():
                if item in running:
                    running[item] += n
        for p in NON_FERTILIZER_PRODUCTS:
            cumulative[p][day] = running[p]
    return cumulative


def is_viable(product: str, cumulative_drain_at_end: int) -> bool:
    """PR #1399's "price will increase significantly": cumulative town-only drain exceeds T,
    crossing the hinge shape's knee (see this module's docstring)."""
    return cumulative_drain_at_end > MARKET_PARAMS[product]["T"]


def price_trajectory(product: str, cumulative: list[int]) -> list[int]:
    """`model.price.price()` at each day, given zero-production cumulative drain -- "the revenue
    available at each point in the season given the drain" the issue's scope asks for."""
    return [market_price(product, MARKET_I0 - drained) for drained in cumulative]


@dataclass
class Scenario:
    """One sampled shop draw and its consequences."""

    seed: int
    draw: list[str]
    cumulative: dict[str, list[int]] = field(repr=False)  # product -> per-day cumulative drain

    def depletion_fraction(self, product: str, day: int = -1) -> float:
        """x/T at `day` (default: season end) -- the hinge shape's own coordinate, so 1.0 is
        exactly the viability knee."""
        return self.cumulative[product][day] / MARKET_PARAMS[product]["T"]

    def viable(self, product: str) -> bool:
        return is_viable(product, self.cumulative[product][-1])

    def shops_unlocked_by_day(self, day: int) -> list[str]:
        return [self.draw[k] for k in range(N_SLOTS) if SLOT_UNLOCK_DAYS[k] <= day]


def sample_scenarios(n: int, seed_start: int = 0, episode_steps: int = CONFIG_DEFAULTS["episodeSteps"]) -> list[Scenario]:
    """Samples `n` scenarios using this module's own draw distribution + demand_curve() (pure
    Python, no native sim needed). For a stronger cross-check against the actual validated engine
    mechanics, see sample_scenarios_from_native_sim() in this module's tests."""
    rng = random.Random()
    scenarios = []
    for i in range(n):
        draw = draw_random_slots(rng)
        scenarios.append(Scenario(seed=seed_start + i, draw=draw, cumulative=demand_curve(draw, episode_steps)))
    return scenarios


def viability_frequencies(scenarios: list[Scenario], products: tuple[str, ...] = ("TOMATO", "CARROT", "EGG")) -> dict[str, float]:
    return {p: sum(1 for s in scenarios if s.viable(p)) / len(scenarios) for p in products}


# The clustering basis: NON_FERTILIZER_PRODUCTS minus WHEAT/STRAWBERRY/MILK/MELON. Those four
# are structurally near-constant across draws -- WHEAT is demanded by 5/8 shop types, STRAWBERRY
# by 4/8, MILK by 3/8 (almost always demanded by *something*), and MELON by none at all (only the
# town centre ever touches it, so it has zero shop-driven variance). Clustering on raw depletion
# fraction across all 8 products is dominated by whichever of those four happens to have the
# largest absolute range, drowning out the genuinely per-episode-random signal: EGG/TOMATO/CARROT
# (2/8 shop types each) and WOOL (1/8) are the ones that actually differ game to game in a way a
# policy could condition on -- and not coincidentally, three of them are exactly PR #1399's
# "sometimes viable" set.
DIFFERENTIATING_PRODUCTS = ("WOOL", "CARROT", "TOMATO", "EGG")


@dataclass
class Regime:
    """One named cluster of shop draws."""

    label: int
    name: str
    frequency: float  # fraction of sampled scenarios assigned to this regime
    mean_depletion_fraction: dict[str, float]  # product -> mean x/T within this regime
    viable_products: tuple[str, ...]  # DIFFERENTIATING_PRODUCTS that clear x/T > 1 on average


@dataclass
class RegimeSet:
    """The taxonomy: fitted cluster centroids (for classifying new scenarios) plus the named
    regimes, exported as a reusable fixture for the eval harness (issue 011) and the search
    lines (issues 013+)."""

    regimes: list[Regime]
    centroids: list[list[float]]  # in DIFFERENTIATING_PRODUCTS' standardized-feature space
    scaler_mean: list[float]
    scaler_scale: list[float]

    def classify(self, scenario: Scenario) -> int:
        """Nearest-centroid label for a (possibly new) scenario, in the same standardized space
        the centroids were fit in."""
        x = [scenario.depletion_fraction(p) for p in DIFFERENTIATING_PRODUCTS]
        xs = [(x[i] - self.scaler_mean[i]) / self.scaler_scale[i] for i in range(len(x))]
        return min(range(len(self.centroids)), key=lambda k: _sq_dist(xs, self.centroids[k]))

    def classify_partial(self, shops_so_far: list[str]) -> int:
        """Nearest-centroid label using only shops unlocked *so far* (an early-game guess, for
        quantifying how soon a regime is identifiable -- see earliest_identifiable_day()).
        Approximates each not-yet-drawn slot at its marginal expectation (1/8 of a generic
        instance) rather than 0, so a partial draw isn't penalized just for being incomplete."""
        drawn_fraction = len(shops_so_far) / N_SLOTS
        partial = demand_curve_from_shop_counts(shops_so_far, remaining_slots=N_SLOTS - len(shops_so_far))
        x = [partial[p] / MARKET_PARAMS[p]["T"] for p in DIFFERENTIATING_PRODUCTS]
        xs = [(x[i] - self.scaler_mean[i]) / self.scaler_scale[i] for i in range(len(x))]
        return min(range(len(self.centroids)), key=lambda k: _sq_dist(xs, self.centroids[k]))


def _sq_dist(a: list[float], b: list[float]) -> float:
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def demand_curve_from_shop_counts(drawn_shops: list[str], remaining_slots: int) -> dict[str, int]:
    """Full-season expected cumulative drain given only the shops drawn *so far*, treating each
    of the `remaining_slots` not-yet-drawn slots as contributing its expectation over a uniform
    draw (this is what classify_partial() uses to guess a regime before all 8 slots are known)."""
    # Ticks a slot unlocked on day `d` contributes by season's end -- see demand_curve()'s
    # per-step loop; this closed form is the same computation, just without materializing days.
    episode_steps, turns_per_day, shop_interval = (
        CONFIG_DEFAULTS["episodeSteps"],
        CONFIG_DEFAULTS["turnsPerDay"],
        CONFIG_DEFAULTS["townShopSellInterval"],
    )
    last_step = episode_steps - 1

    def ticks_from(day: int) -> int:
        start_step = day * turns_per_day
        return 0 if start_step > last_step else (last_step - start_step) // shop_interval + 1

    totals = dict.fromkeys(NON_FERTILIZER_PRODUCTS, 0.0)
    for k, shop in enumerate(drawn_shops):
        products = SHOPS[shop]
        mult = 2 if len(products) == 1 else 1
        for item in products:
            if item in totals:
                totals[item] += ticks_from(SLOT_UNLOCK_DAYS[k]) * mult
    # Remaining (not-yet-drawn) slots: expected contribution is the average over the 8 shop
    # types, weighted by that slot's own tick count.
    n_drawn = len(drawn_shops)
    for k in range(n_drawn, n_drawn + remaining_slots):
        if k >= N_SLOTS:
            break
        expected_per_type = {p: 0.0 for p in NON_FERTILIZER_PRODUCTS}
        for shop in SHOP_NAMES:
            products = SHOPS[shop]
            mult = 2 if len(products) == 1 else 1
            for item in products:
                if item in expected_per_type:
                    expected_per_type[item] += mult / len(SHOP_NAMES)
        for item, v in expected_per_type.items():
            totals[item] += ticks_from(SLOT_UNLOCK_DAYS[k]) * v
    # Town centre: draw-independent, flat 1/day for every non-fertilizer product.
    n_days = episode_steps // turns_per_day
    for p in totals:
        totals[p] += n_days
    return totals


def _name_regime(mean_fraction: dict[str, float]) -> tuple[str, tuple[str, ...]]:
    viable = tuple(p for p in DIFFERENTIATING_PRODUCTS if mean_fraction[p] > 1.0)
    ranked = sorted(viable, key=lambda p: -mean_fraction[p])
    if not ranked:
        return "balanced", ranked
    return "+".join(p.lower() for p in ranked) + "-rich", ranked


def fit_regimes(scenarios: list[Scenario], k: int = 4, random_state: int = 0) -> RegimeSet:
    """Clusters `scenarios` by their standardized DIFFERENTIATING_PRODUCTS depletion fractions
    (k-means) and names each cluster by which of those products clear the viability knee
    (mean x/T > 1) on average, ranked by magnitude -- "wool-rich", "tomato+wool-rich", or
    "balanced" if none stand out. k=4 by default: the smallest k in the issue's "aim 4-6" range
    that still separates cleanly (inertia drops ~9% more at k=5, ~17% at k=6, but the k=4 clusters
    are already near-equal-sized (22-29%) with one clear dominant product each except the
    "balanced" one -- diminishing, not qualitative, returns past 4)."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = [[s.depletion_fraction(p) for p in DIFFERENTIATING_PRODUCTS] for s in scenarios]
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Xs)

    regimes = []
    for label in range(k):
        members = [s for s, lbl in zip(scenarios, km.labels_) if lbl == label]
        mean_fraction = {p: sum(s.depletion_fraction(p) for s in members) / len(members) for p in DIFFERENTIATING_PRODUCTS}
        name, viable = _name_regime(mean_fraction)
        regimes.append(
            Regime(label=label, name=name, frequency=len(members) / len(scenarios), mean_depletion_fraction=mean_fraction, viable_products=viable)
        )

    return RegimeSet(
        regimes=regimes,
        centroids=[list(c) for c in km.cluster_centers_],
        scaler_mean=list(scaler.mean_),
        scaler_scale=list(scaler.scale_),
    )


def earliest_identifiable_day(
    scenarios: list[Scenario], regime_set: RegimeSet, checkpoints: list[int] = SLOT_UNLOCK_DAYS, threshold: float = 0.8
) -> tuple[int | None, dict[int, float]]:
    """For each checkpoint day, the accuracy of classify_partial() (using only shops unlocked by
    that day) against the TRUE final-regime label -- and the first day that accuracy reaches
    `threshold`. Answers the issue's question directly: is the regime pinned down by day 9, or
    only by day 24?"""
    true_labels = [regime_set.classify(s) for s in scenarios]
    accuracy_by_day = {}
    earliest = None
    for day in checkpoints:
        correct = sum(
            1
            for s, true_label in zip(scenarios, true_labels)
            if regime_set.classify_partial(s.shops_unlocked_by_day(day)) == true_label
        )
        acc = correct / len(scenarios)
        accuracy_by_day[day] = acc
        if earliest is None and acc >= threshold:
            earliest = day
    return earliest, accuracy_by_day

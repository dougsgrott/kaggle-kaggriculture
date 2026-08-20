"""Empirical game-theoretic analysis over the population (issue 018).

`eval.arena.compare` already gives a paired-seed, both-seat, Wilson-CI win rate between any two
policies -- this module runs that over every pair in a population (`eval.population`) and treats
the result as a normal-form symmetric zero-sum game: `payoff(A, B) = P(A beats B)`, ties scored at
0.5 (`arena.compare`'s own convention), `payoff(B, A) = 1 - payoff(A, B)` by construction (the
same two games, seats swapped, so this is exact, not assumed).

Two questions issue 018 asks of that matrix:

  - **Is the meta transitive, cyclic, or converging?** `cyclic_triples` looks for A > B > C > A
    among pairs the arena's own CI called decisively (`verdict != "undecided"`) -- a coarser,
    game-theoretically meaningful notion of "beats" than a raw >50% point estimate.
  - **What does iterating best responses do?** `fictitious_play` runs the textbook algorithm
    directly on the empirical matrix: no new strategies are generated each round (that's what
    013-017 already are -- this session's own budget doesn't stretch to re-running a fresh search
    line per round), so this is "double oracle" without the oracle step. For a symmetric
    zero-sum game this still answers the question asked: Robinson's theorem guarantees fictitious
    play's *average* strategy converges to a maximin/Nash equilibrium, and watching the *pure*
    best-response sequence directly answers "does repeated best-responding converge to a fixed
    point or cycle" -- Kaito Fukami's own weaker hypothesis (734212).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from kaggriculture.eval.stats import wilson_interval
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import episode_config

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


@dataclass
class PayoffCell:
    p_hat: float  # P(row beats column), ties = 0.5
    lo: float
    hi: float
    n_games: int
    verdict: str  # "better" / "worse" / "undecided", from the row's perspective

    def to_dict(self) -> dict:
        return {"p_hat": self.p_hat, "lo": self.lo, "hi": self.hi, "n_games": self.n_games, "verdict": self.verdict}


PayoffMatrix = dict[tuple[str, str], PayoffCell]  # keyed (a, b) with a < b lexicographically; mirror for (b, a)


def compute_payoff_matrix(
    population: dict[str, native.Policy], config: dict, n_seeds: int = 15, n_threads: int = 8, episode_steps: int = 720
) -> PayoffMatrix:
    """One comparison per unordered pair (n*(n-1)/2 calls, not n^2 -- `payoff(b, a)` is read off
    `payoff(a, b)` by construction, not recomputed). Population values are already resolved
    `native.Policy` objects (see `eval.population`), not CLI spec strings, so this replays
    `arena.compare`'s own paired-seed/both-seat game loop directly rather than calling it."""
    matrix: PayoffMatrix = {}
    names = sorted(population)
    for a, b in combinations(names, 2):
        result = _compare_policies(population[a], population[b], config, n_seeds, n_threads, episode_steps)
        matrix[(a, b)] = result
    return matrix


def _compare_policies(policy_a: native.Policy, policy_b: native.Policy, config: dict, n_seeds: int, n_threads: int, episode_steps: int) -> PayoffCell:
    """Mirrors `arena.compare`'s own paired-seed/both-seat game loop, but takes resolved
    `native.Policy` objects directly (population members aren't CLI spec strings -- several are
    built in-process from a rerun search or loaded params, see `eval.population`)."""
    econfig = dict(config)
    econfig["episodeSteps"] = episode_steps
    ecfg = episode_config(econfig)
    seeds = range(n_seeds)
    pairs, configs, seed_list, orientations = [], [], [], []
    for seed in seeds:
        pairs.append((policy_a, policy_b))
        configs.append(ecfg)
        seed_list.append(seed)
        orientations.append(0)
        pairs.append((policy_b, policy_a))
        configs.append(ecfg)
        seed_list.append(seed)
        orientations.append(1)

    results = native.run_batch(pairs, configs, seed_list, n_threads)
    wins = losses = ties = 0
    for a_seat, result in zip(orientations, results):
        a_money, b_money = (result.final_money[0], result.final_money[1]) if a_seat == 0 else (result.final_money[1], result.final_money[0])
        if a_money > b_money:
            wins += 1
        elif a_money < b_money:
            losses += 1
        else:
            ties += 1
    n_games = wins + losses + ties
    interval = wilson_interval(wins + 0.5 * ties, n_games)
    if interval.lo > 0.5:
        v = "better"
    elif interval.hi < 0.5:
        v = "worse"
    else:
        v = "undecided"
    return PayoffCell(p_hat=interval.p_hat, lo=interval.lo, hi=interval.hi, n_games=n_games, verdict=v)


def payoff(matrix: PayoffMatrix, a: str, b: str) -> float:
    """P(a beats b), reading the mirrored cell when the matrix stores (b, a)."""
    if a == b:
        return 0.5
    if (a, b) in matrix:
        return matrix[(a, b)].p_hat
    return 1.0 - matrix[(b, a)].p_hat


def verdict_of(matrix: PayoffMatrix, a: str, b: str) -> str:
    """`a`'s verdict against `b` ("better"/"worse"/"undecided"), mirroring the stored cell's
    verdict when the matrix stores the pair the other way round."""
    if a == b:
        return "undecided"
    if (a, b) in matrix:
        return matrix[(a, b)].verdict
    mirror = {"better": "worse", "worse": "better", "undecided": "undecided"}
    return mirror[matrix[(b, a)].verdict]


# --------------------------------------------------------------------------- #
# Transitivity / cycle detection
# --------------------------------------------------------------------------- #


def cyclic_triples(matrix: PayoffMatrix, names: list[str]) -> list[tuple[str, str, str]]:
    """Every ordered triple (a, b, c) with a decisive a>b, b>c, c>a (a rock-paper-scissors
    pattern) -- decisive meaning the arena's own Wilson CI called it, not just a >50% point
    estimate. Returns each cycle once (canonicalized to start at its lexicographically smallest
    member) even though the same 3-cycle can be named starting from any of its 3 members."""
    cycles = set()
    for a, b, c in combinations(sorted(names), 3):
        for x, y, z in ((a, b, c), (a, c, b)):
            if verdict_of(matrix, x, y) == "better" and verdict_of(matrix, y, z) == "better" and verdict_of(matrix, z, x) == "better":
                triple = (x, y, z)
                start = triple.index(min(triple))
                cycles.add(triple[start:] + triple[:start])
    return sorted(cycles)


@dataclass
class TransitivityReport:
    n_names: int
    n_triples_checked: int
    n_decisive_triples: int  # all 3 pairwise verdicts were decisive (not "undecided")
    n_cyclic_triples: int
    cycles: list[tuple[str, str, str]] = field(default_factory=list)
    dominant_strategy: str | None = None  # a name that decisively beats every other population member

    def to_dict(self) -> dict:
        return {
            "n_names": self.n_names,
            "n_triples_checked": self.n_triples_checked,
            "n_decisive_triples": self.n_decisive_triples,
            "n_cyclic_triples": self.n_cyclic_triples,
            "cycles": self.cycles,
            "dominant_strategy": self.dominant_strategy,
        }


def transitivity_report(matrix: PayoffMatrix, names: list[str]) -> TransitivityReport:
    names = sorted(names)
    n_decisive = 0
    for a, b, c in combinations(names, 3):
        verdicts = [verdict_of(matrix, a, b), verdict_of(matrix, b, c), verdict_of(matrix, a, c)]
        if all(v != "undecided" for v in verdicts):
            n_decisive += 1
    cycles = cyclic_triples(matrix, names)

    dominant = None
    for candidate in names:
        if all(verdict_of(matrix, candidate, other) == "better" for other in names if other != candidate):
            dominant = candidate
            break

    return TransitivityReport(
        n_names=len(names),
        n_triples_checked=len(list(combinations(names, 3))),
        n_decisive_triples=n_decisive,
        n_cyclic_triples=len(cycles),
        cycles=cycles,
        dominant_strategy=dominant,
    )


# --------------------------------------------------------------------------- #
# Fictitious play: maximin strategy + best-response convergence/cycling
# --------------------------------------------------------------------------- #


@dataclass
class FictitiousPlayResult:
    iterations: int
    average_strategy: dict[str, float]  # converges to a maximin/Nash mix (Robinson's theorem, zero-sum)
    estimated_game_value: float  # ~0.5 for a balanced population, >0.5 if some strategies dominate
    best_response_sequence: list[str]  # the pure best response chosen each round
    converged_to_fixed_point: bool  # br_sequence[-1] == br_sequence[-2] and stayed there
    cycle: list[str] | None  # the repeating block, if the tail of the sequence entered one

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "average_strategy": self.average_strategy,
            "estimated_game_value": self.estimated_game_value,
            "best_response_sequence": self.best_response_sequence,
            "converged_to_fixed_point": self.converged_to_fixed_point,
            "cycle": self.cycle,
        }


def _best_response(matrix: PayoffMatrix, names: list[str], opponent_mix: dict[str, float]) -> str:
    def expected_payoff(a: str) -> float:
        return sum(opponent_mix[b] * payoff(matrix, a, b) for b in names)

    return max(names, key=expected_payoff)


def _detect_tail_cycle(sequence: list[str], max_period: int = 8) -> list[str] | None:
    """A repeating block at the very end of `sequence`, if one exists, checked from the shortest
    period up -- a period-1 cycle (a fixed point held for several rounds) is reported as a cycle
    of length 1, which `converged_to_fixed_point` also captures more plainly."""
    for period in range(1, max_period + 1):
        if len(sequence) < period * 3:
            continue
        tail = sequence[-period * 3 :]
        block = tail[:period]
        if all(tail[i : i + period] == block for i in range(0, len(tail), period)):
            return block
    return None


def fictitious_play(matrix: PayoffMatrix, names: list[str], iterations: int = 500, seed: int = 0) -> FictitiousPlayResult:
    """Standard fictitious play on the empirical matrix (Brown 1951): each round, both sides best-
    respond to the OTHER side's empirical play-frequency-so-far; since the game is symmetric, one
    running frequency count serves both sides. Starts from a uniform mix over the population --
    an arbitrary but standard and reproducible starting point (`seed` is unused by the
    deterministic core; kept for interface symmetry with the rest of the repo's search modules,
    in case a randomized tie-break is added later)."""
    names = sorted(names)
    counts = dict.fromkeys(names, 0)
    br_sequence: list[str] = []
    cumulative_payoff = 0.0

    for t in range(iterations):
        total = sum(counts.values())
        mix = {n: (counts[n] / total if total else 1.0 / len(names)) for n in names}
        br = _best_response(matrix, names, mix)
        br_sequence.append(br)
        cumulative_payoff += sum(mix[b] * payoff(matrix, br, b) for b in names)
        counts[br] += 1

    total = sum(counts.values())
    average_strategy = {n: counts[n] / total for n in names}
    estimated_game_value = cumulative_payoff / iterations

    tail = br_sequence[-min(50, len(br_sequence)) :]
    converged = len(set(tail)) == 1
    cycle = None if converged else _detect_tail_cycle(br_sequence)

    return FictitiousPlayResult(
        iterations=iterations,
        average_strategy=average_strategy,
        estimated_game_value=estimated_game_value,
        best_response_sequence=br_sequence,
        converged_to_fixed_point=converged,
        cycle=cycle,
    )


def what_beats_the_reference(matrix: PayoffMatrix, names: list[str], reference: str) -> dict:
    """Direct answer to the issue's own question: what does the reference lineage beat, and what
    beats it -- read straight off the matrix rows/columns rather than the aggregate analyses
    above."""
    beats_reference = [n for n in names if n != reference and verdict_of(matrix, n, reference) == "better"]
    reference_beats = [n for n in names if n != reference and verdict_of(matrix, reference, n) == "better"]
    undecided = [n for n in names if n != reference and verdict_of(matrix, reference, n) == "undecided"]
    return {"reference": reference, "beats_reference": beats_reference, "reference_beats": reference_beats, "undecided_against": undecided}


def _next_experiment_dir(slug: str) -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in EXPERIMENTS_DIR.glob("exp-*") if p.is_dir()]
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name.split("-")[1]))
        except (IndexError, ValueError):
            continue
    n = max(nums, default=0) + 1
    out = EXPERIMENTS_DIR / f"exp-{n:03d}-{slug}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    from kaggriculture.eval.population import assemble_population
    from kaggriculture.sim.decode import default_config_dict

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n-seeds", type=int, default=15, help="paired-seed, both-seat games per matrix cell (n_games = 2 x this)")
    parser.add_argument("--n-public", type=int, default=10, help="max public-pool cluster representatives")
    parser.add_argument("--cluster-threshold", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--fp-iterations", type=int, default=500, help="fictitious-play iterations")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = dict(default_config_dict())
    t0 = time.perf_counter()

    population, diagnostics = assemble_population(config, n_public_target=args.n_public, cluster_threshold=args.cluster_threshold, threads=args.threads)
    names = sorted(population)
    print(f"Population: {len(names)} members ({diagnostics.n_candidates_survived_smoke_test}/{diagnostics.n_candidates_discovered} public candidates survived, {diagnostics.n_clusters} opening clusters)")
    for n in names:
        print(f"  {n}")

    matrix = compute_payoff_matrix(population, config, n_seeds=args.n_seeds, n_threads=args.threads, episode_steps=args.episode_steps)
    matrix_seconds = time.perf_counter() - t0
    print(f"\nPayoff matrix: {len(matrix)} pairs, {matrix_seconds:.1f}s")

    report = transitivity_report(matrix, names)
    print(f"\nTransitivity: {report.n_decisive_triples}/{report.n_triples_checked} triples fully decisive, {report.n_cyclic_triples} cyclic")
    if report.dominant_strategy:
        print(f"  dominant strategy: {report.dominant_strategy} (beats every other population member decisively)")
    else:
        print("  no dominant strategy -- no population member decisively beats every other member")
    for cycle in report.cycles[:10]:
        print(f"  cycle: {' > '.join(cycle)} > {cycle[0]}")

    fp = fictitious_play(matrix, names, iterations=args.fp_iterations)
    print(f"\nFictitious play ({fp.iterations} iterations): estimated game value {fp.estimated_game_value:.3f}")
    print(f"  converged to a fixed point: {fp.converged_to_fixed_point}")
    if fp.cycle:
        print(f"  entered a cycle: {' -> '.join(fp.cycle)}")
    print("  average (maximin) strategy, top weights:")
    for name, weight in sorted(fp.average_strategy.items(), key=lambda kv: -kv[1])[:8]:
        if weight > 0.01:
            print(f"    {name:50s} {weight:.3f}")

    reference_candidates = [n for n in names if "boatlee" in n or "public-v12" in n or n == "baseline"]
    reference_reports = {ref: what_beats_the_reference(matrix, names, ref) for ref in reference_candidates}
    for ref, r in reference_reports.items():
        print(f"\nWhat beats {ref}: {r['beats_reference']}")
        print(f"What {ref} beats: {r['reference_beats']}")

    out_dir = args.out or _next_experiment_dir("egta")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "n_seeds": args.n_seeds,
                "n_public": args.n_public,
                "cluster_threshold": args.cluster_threshold,
                "threads": args.threads,
                "episode_steps": args.episode_steps,
                "fp_iterations": args.fp_iterations,
            },
            indent=2,
        )
    )
    (out_dir / "payoff_matrix.json").write_text(json.dumps({f"{a}|{b}": cell.to_dict() for (a, b), cell in matrix.items()}, indent=2))
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "population": names,
                "population_diagnostics": {
                    "n_candidates_discovered": diagnostics.n_candidates_discovered,
                    "n_candidates_survived_smoke_test": diagnostics.n_candidates_survived_smoke_test,
                    "n_clusters": diagnostics.n_clusters,
                    "clusters": diagnostics.clusters,
                    "representatives": diagnostics.representatives,
                },
                "transitivity": report.to_dict(),
                "fictitious_play": fp.to_dict(),
                "reference_reports": reference_reports,
                "seconds": time.perf_counter() - t0,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()

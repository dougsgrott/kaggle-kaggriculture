"""Large Neighbourhood Search over the issue 013 greedy plan (issue 014).

The seed is `search.schedule.build_schedule()`'s output: a `Schedule` (macro plan) that already
beats `pass`/`starter`/`random` decisively but not issue 005's baseline (see 013's Revision
section). This module searches the neighbourhood of that plan for something better, using the
same greedy machinery as the destroy/repair *engine* rather than a separate optimizer:

  - **Destroy/repair over day-blocks**: `build_schedule()` grew an optional `rng_for_day` hook
    (013's own function, not a new mechanism) that turns on epsilon-greedy exploration in the
    tile-assignment step for whichever days it's given an RNG for. "Destroy days `[d, d+k)` and
    re-solve them" is exactly calling `build_schedule` with `rng_for_day` returning an RNG only
    inside that window -- the days before and after replay deterministically (reconstructing the
    same state build_schedule would have reached on its own), and only the window explores.
  - **Direct neighbourhood moves** (swap a tile's crop, resize a day's crew target, shift a
    quadrant's unlock day) mutate a copy of the `Schedule` directly, no re-solve needed --
    `search.agent`'s execution layer already handles the resulting plan safely even if a move
    makes something momentarily unaffordable (it just doesn't spend money it doesn't have; see
    013's `CASH_RESERVE`/affordability-fallback logic), so these never need to re-verify
    feasibility here.
  - **Diverse greedy seeds**: `build_schedule()`'s own tunable constants (`TILES_PER_HAND`,
    `MAX_CREW`, `CASH_RESERVE`) materially change which plan the greedy construction converges to
    (013's own tuning swept them). Re-running `build_schedule()` under randomly perturbed values
    is a cheap (~0.01-0.05s each) restart pool.

**Objective**: mean `(candidate_money - opponent_money)` across a *portfolio* of opponents (not
just the baseline we ultimately want to beat -- issue 014's own "obvious trap" warning) and a
*fixed* seed set, both seats. A raw margin, not a win/loss count: hill-climbing over marginal plan
tweaks needs a continuous signal, not the coarse yes/no eval.arena's Wilson-CI machinery gives
(that machinery is used for the FINAL verdict instead -- see `main()`).

**Acceptance rule**: record-to-record travel (RRT), not simulated annealing with a temperature
schedule -- simpler to reason about and reproduce (`--seed` alone determines the whole run), and
well-suited to a bounded number of candidate evaluations. A move is accepted (becomes the new
current point to mutate from) if it's within `rrt_tolerance` of the best score seen so far; the
best-ever candidate is tracked separately and is what's actually returned.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from kaggriculture.eval.agents import record_tape, resolve_policy
from kaggriculture.eval.stats import wilson_interval, verdict as wilson_verdict
from kaggriculture.search import schedule as schedule_module
from kaggriculture.search.agent import schedule_agent
from kaggriculture.search.schedule import ANIMAL_NAMES, CROP_NAMES, Schedule, build_schedule
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import default_config_dict, episode_config

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

DEFAULT_OPPONENTS = ("pass", "starter", "baseline")
DEFAULT_WINDOW_CHOICES = (3, 5, 8, 12)
RRT_TOLERANCE = 0.02  # accept a candidate within 2% of the best-ever score (see module docstring)


def _copy_schedule(schedule: Schedule) -> Schedule:
    return Schedule(
        tile_role=dict(schedule.tile_role),
        tile_kind=dict(schedule.tile_kind),
        tile_first_assigned_day=dict(schedule.tile_first_assigned_day),
        crew_size_by_day=dict(schedule.crew_size_by_day),
        land_days=dict(schedule.land_days),
        n_days=schedule.n_days,
        diagnostics=dict(schedule.diagnostics),
    )


@contextlib.contextmanager
def _perturbed_constants(**overrides):
    """Temporarily overrides search.schedule's module-level tunables (TILES_PER_HAND, MAX_CREW,
    CASH_RESERVE, ...) -- build_schedule() reads them as globals at call time, so patching the
    module attribute before calling it changes its behaviour without a bigger parameterization
    refactor. Not thread-safe; only used for sequential diverse-seed generation."""
    originals = {k: getattr(schedule_module, k) for k in overrides}
    for k, v in overrides.items():
        setattr(schedule_module, k, v)
    try:
        yield
    finally:
        for k, v in originals.items():
            setattr(schedule_module, k, v)


def diverse_seed_schedules(config: dict, rng: random.Random, n_seeds: int, allow_4th_quadrant: bool = True) -> list[Schedule]:
    """A restart pool: `build_schedule()` under randomly perturbed tunables. Always includes the
    default (unperturbed) schedule first, so the pool is never worse than 013's own seed."""
    schedules = [build_schedule(config, allow_4th_quadrant=allow_4th_quadrant)]
    for _ in range(max(0, n_seeds - 1)):
        overrides = {
            "TILES_PER_HAND": rng.choice([2, 3, 4, 5, 6]),
            "MAX_CREW": rng.choice([4, 6, 8, 10]),
            "CASH_RESERVE": rng.choice([200.0, 500.0, 800.0]),
        }
        with _perturbed_constants(**overrides):
            schedules.append(build_schedule(config, allow_4th_quadrant=allow_4th_quadrant))
    return schedules


def _swap_crop(schedule: Schedule, rng: random.Random) -> Schedule | None:
    crop_tiles = [pos for pos, kind in schedule.tile_kind.items() if kind == "crop"]
    if not crop_tiles:
        return None
    pos = rng.choice(crop_tiles)
    alternatives = [c for c in CROP_NAMES if c != schedule.tile_role[pos]]
    if not alternatives:
        return None
    candidate = _copy_schedule(schedule)
    candidate.tile_role[pos] = rng.choice(alternatives)
    return candidate


def _resize_crew(schedule: Schedule, rng: random.Random) -> Schedule | None:
    if not schedule.crew_size_by_day:
        return None
    day = rng.choice(sorted(schedule.crew_size_by_day))
    delta = rng.choice([-2, -1, 1, 2])
    candidate = _copy_schedule(schedule)
    candidate.crew_size_by_day[day] = max(0, schedule.crew_size_by_day[day] + delta)
    return candidate


def _resize_herd(schedule: Schedule, rng: random.Random) -> Schedule | None:
    """Not literally "resize" (add/remove a structure) -- swapping which animal occupies an
    existing site is the cheap, always-safe move (removing a site outright would also need
    popping it from `tile_kind`/`tile_first_assigned_day`, and adding a new one only makes sense
    at a specific shed-adjacent position -- both doable, but swapping the type already explores
    the same "which animal" decision `build_schedule`'s own ranking makes, at much less risk of
    producing an inconsistent Schedule)."""
    animal_sites = [pos for pos, kind in schedule.tile_kind.items() if kind == "animal"]
    if not animal_sites:
        return None
    pos = rng.choice(animal_sites)
    alternatives = [a for a in ANIMAL_NAMES if a != schedule.tile_role[pos]]
    if not alternatives:
        return None
    candidate = _copy_schedule(schedule)
    candidate.tile_role[pos] = rng.choice(alternatives)
    return candidate


def _shift_land_day(schedule: Schedule, rng: random.Random) -> Schedule | None:
    if not schedule.land_days:
        return None
    extra_idx = rng.choice(sorted(schedule.land_days))
    delta = rng.choice([-3, -2, -1, 1, 2, 3])
    candidate = _copy_schedule(schedule)
    candidate.land_days[extra_idx] = max(0, min(schedule.n_days - 1, schedule.land_days[extra_idx] + delta))
    return candidate


def _destroy_repair(schedule: Schedule, config: dict, rng: random.Random, window_choices: tuple[int, ...], allow_4th_quadrant: bool) -> Schedule:
    """Ignores `schedule` (the incumbent) and rebuilds from scratch with a randomized window --
    see module docstring. Deterministic replay outside the window means this can (and often will)
    reproduce the exact incumbent; the exploration only bites inside `[start, start+window)`."""
    n_days = config["episodeSteps"] // config["turnsPerDay"]
    window = rng.choice(window_choices)
    start = rng.randint(0, max(0, n_days - 1))

    def rng_for_day(day: int) -> random.Random | None:
        return rng if start <= day < start + window else None

    return build_schedule(config, allow_4th_quadrant=allow_4th_quadrant, rng_for_day=rng_for_day)


_OPERATORS: dict[str, Callable] = {
    "swap_crop": lambda s, cfg, rng, wc, a4: _swap_crop(s, rng),
    "resize_herd": lambda s, cfg, rng, wc, a4: _resize_herd(s, rng),
    "resize_crew": lambda s, cfg, rng, wc, a4: _resize_crew(s, rng),
    "shift_land_day": lambda s, cfg, rng, wc, a4: _shift_land_day(s, rng),
    "destroy_repair": lambda s, cfg, rng, wc, a4: _destroy_repair(s, cfg, rng, wc, a4),
}


def evaluate_schedule(schedule: Schedule, config: dict, opponent_policies: list[native.Policy], seed_set: list[int], n_threads: int = 1) -> float:
    """Mean `(candidate_money - opponent_money)` across `opponent_policies x seed_set`, both
    seats -- common random numbers (the same `seed_set` for every candidate) is what makes this a
    fair comparison rather than noise from the weed stream / shop draw (issue's own warning)."""
    tape_policy = record_tape(schedule_agent(schedule, config), config)
    ecfg = episode_config(config)
    pairs, configs, seeds = [], [], []
    for opponent in opponent_policies:
        for seed in seed_set:
            pairs.append((tape_policy, opponent))
            configs.append(ecfg)
            seeds.append(seed)
            pairs.append((opponent, tape_policy))
            configs.append(ecfg)
            seeds.append(seed)
    results = native.run_batch(pairs, configs, seeds, n_threads)
    margins = [
        (r.final_money[0] - r.final_money[1]) if i % 2 == 0 else (r.final_money[1] - r.final_money[0]) for i, r in enumerate(results)
    ]
    return sum(margins) / len(margins)


@dataclass
class LNSResult:
    incumbent: Schedule
    incumbent_score: float
    trace: list[dict] = field(default_factory=list)
    n_evaluations: int = 0


def lns_search(
    config: dict,
    base_schedules: list[Schedule],
    opponent_policies: list[native.Policy],
    seed_set: list[int],
    n_iters: int,
    seed: int = 0,
    n_threads: int = 1,
    window_choices: tuple[int, ...] = DEFAULT_WINDOW_CHOICES,
    allow_4th_quadrant: bool = True,
) -> LNSResult:
    """Record-to-record travel over the neighbourhood operators (see module docstring). Starts
    from the best of `base_schedules` (013's seed plus any diverse-seed restarts), then applies
    one randomly chosen operator per iteration, evaluated against `opponent_policies x seed_set`
    with common random numbers."""
    rng = random.Random(seed)

    scored_bases = [(evaluate_schedule(s, config, opponent_policies, seed_set, n_threads), s) for s in base_schedules]
    current_score, current = max(scored_bases, key=lambda t: t[0])
    best, best_score = current, current_score
    trace = [{"iter": 0, "operator": "seed", "score": current_score, "accepted": True, "best_score": best_score}]
    n_evaluations = len(base_schedules)

    for it in range(1, n_iters + 1):
        operator_name = rng.choice(list(_OPERATORS))
        candidate = _OPERATORS[operator_name](current, config, rng, window_choices, allow_4th_quadrant)
        if candidate is None:
            trace.append({"iter": it, "operator": operator_name, "score": None, "accepted": False, "best_score": best_score})
            continue
        score = evaluate_schedule(candidate, config, opponent_policies, seed_set, n_threads)
        n_evaluations += 1

        accepted = score >= best_score - abs(best_score) * RRT_TOLERANCE
        if accepted:
            current, current_score = candidate, score
        if score > best_score:
            best, best_score = candidate, score

        trace.append({"iter": it, "operator": operator_name, "score": score, "accepted": accepted, "best_score": best_score})

    return LNSResult(incumbent=best, incumbent_score=best_score, trace=trace, n_evaluations=n_evaluations)


def head_to_head(schedule_a: Schedule, schedule_b: Schedule, config: dict, seed_set: list[int], n_threads: int = 1) -> dict:
    """Paired-seed, both-seat comparison of two schedules directly against each other (not
    against a portfolio) -- issue 014's own acceptance criterion ("improves final money over the
    issue 013 seed plan ... survives eval.arena"), computed the same way eval.arena.compare does
    (Wilson CI, ties scored 0.5)."""
    tape_a = record_tape(schedule_agent(schedule_a, config), config)
    tape_b = record_tape(schedule_agent(schedule_b, config), config)
    ecfg = episode_config(config)
    pairs, configs, seeds, orientations = [], [], [], []
    for seed in seed_set:
        pairs.append((tape_a, tape_b))
        orientations.append(0)
        pairs.append((tape_b, tape_a))
        orientations.append(1)
        configs.extend([ecfg, ecfg])
        seeds.extend([seed, seed])
    results = native.run_batch(pairs, configs, seeds, n_threads)
    wins = losses = ties = 0
    for orientation, r in zip(orientations, results):
        a_money, b_money = (r.final_money[0], r.final_money[1]) if orientation == 0 else (r.final_money[1], r.final_money[0])
        if a_money > b_money:
            wins += 1
        elif a_money < b_money:
            losses += 1
        else:
            ties += 1
    n_games = len(results)
    interval = wilson_interval(wins + 0.5 * ties, n_games)
    return {
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "interval": interval.to_dict(),
        "verdict": wilson_verdict(interval),
    }


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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS), help="comma-separated portfolio to optimize against")
    parser.add_argument("--n-seeds", type=int, default=20, help="search seed set size (common random numbers)")
    parser.add_argument("--holdout-n-seeds", type=int, default=20, help="disjoint seed set for the final reported verdict")
    parser.add_argument("--n-iters", type=int, default=60, help="LNS iterations")
    parser.add_argument("--n-restarts", type=int, default=4, help="diverse-seed restart pool size (includes 013's default seed)")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="master RNG seed -- determines the whole run (reproducibility)")
    parser.add_argument("--no-4th-quadrant", action="store_true", help="forbid the SE quadrant (see issue 013's ablation)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = dict(default_config_dict())
    config["episodeSteps"] = args.episode_steps
    allow_4th_quadrant = not args.no_4th_quadrant

    opponent_specs = args.opponents.split(",")
    opponent_policies = [resolve_policy(spec, config) for spec in opponent_specs]

    search_seeds = list(range(args.n_seeds))
    holdout_seeds = list(range(1_000_000, 1_000_000 + args.holdout_n_seeds))  # disjoint from search_seeds by construction

    rng = random.Random(args.seed)
    seed_schedule = build_schedule(config, allow_4th_quadrant=allow_4th_quadrant)  # issue 013's own default plan
    base_schedules = diverse_seed_schedules(config, rng, args.n_restarts, allow_4th_quadrant=allow_4th_quadrant)

    t0 = time.perf_counter()
    result = lns_search(
        config,
        base_schedules,
        opponent_policies,
        search_seeds,
        args.n_iters,
        seed=args.seed,
        n_threads=args.threads,
        allow_4th_quadrant=allow_4th_quadrant,
    )
    search_seconds = time.perf_counter() - t0

    seed_plan_search_score = evaluate_schedule(seed_schedule, config, opponent_policies, search_seeds, args.threads)
    incumbent_holdout_score = evaluate_schedule(result.incumbent, config, opponent_policies, holdout_seeds, args.threads)
    seed_plan_holdout_score = evaluate_schedule(seed_schedule, config, opponent_policies, holdout_seeds, args.threads)

    verdict_vs_seed = head_to_head(result.incumbent, seed_schedule, config, holdout_seeds, args.threads)

    print(f"LNS: {result.n_evaluations} evaluations, {search_seconds:.1f}s")
    print(f"  incumbent portfolio margin: search-set {result.incumbent_score:.1f}  holdout-set {incumbent_holdout_score:.1f}")
    print(f"  013 seed plan portfolio margin: search-set {seed_plan_search_score:.1f}  holdout-set {seed_plan_holdout_score:.1f}")
    overfit_gap = result.incumbent_score - incumbent_holdout_score
    print(f"  overfitting gap (search - holdout, incumbent): {overfit_gap:.1f}")
    print(
        f"  incumbent vs 013 seed plan (holdout, head-to-head): {verdict_vs_seed['wins']}W-{verdict_vs_seed['losses']}L-"
        f"{verdict_vs_seed['ties']}T  verdict={verdict_vs_seed['verdict']}  "
        f"CI=[{verdict_vs_seed['interval']['lo']:.3f},{verdict_vs_seed['interval']['hi']:.3f}]"
    )

    out_dir = args.out or _next_experiment_dir("lns")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "opponents": opponent_specs,
                "n_seeds": args.n_seeds,
                "holdout_n_seeds": args.holdout_n_seeds,
                "n_iters": args.n_iters,
                "n_restarts": args.n_restarts,
                "episode_steps": args.episode_steps,
                "threads": args.threads,
                "seed": args.seed,
                "allow_4th_quadrant": allow_4th_quadrant,
            },
            indent=2,
        )
    )
    (out_dir / "incumbent_trace.json").write_text(json.dumps(result.trace, indent=2))
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "n_evaluations": result.n_evaluations,
                "search_seconds": search_seconds,
                "incumbent_search_score": result.incumbent_score,
                "incumbent_holdout_score": incumbent_holdout_score,
                "seed_plan_search_score": seed_plan_search_score,
                "seed_plan_holdout_score": seed_plan_holdout_score,
                "overfitting_gap": overfit_gap,
                "verdict_vs_seed_plan_holdout": verdict_vs_seed,
                "incumbent_diagnostics": result.incumbent.diagnostics,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

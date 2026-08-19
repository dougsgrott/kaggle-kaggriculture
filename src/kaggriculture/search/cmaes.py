"""CMA-ES over `search.policy`'s parametric policy (issue 015).

Unlike 013/014 (one optimized plan for one world), this tunes a REACTIVE policy's ~20 weights so
the *same* parameters produce good behaviour across different shop draws and opponents -- the
input issue 021 (a scenario-conditional agent) needs. `cma` (pycma) does the actual optimization;
this module defines the objective (mean win rate, not money -- see the issue's own scope note
that money and wins can diverge), parallel fitness evaluation across processes, and the restart
strategy (IPOP, via pycma's own `restarts`/`incpopsize`).

**Why processes, not threads**: `search.policy`'s agent is inherently reactive -- every candidate
is evaluated through a `CallbackPolicy` (a Python function called once per turn), and
`sim.bindings`'s own docs are explicit that a `CallbackPolicy` re-acquires the GIL per call,
serializing any game that involves one regardless of how many `native.run_batch` threads are
asked for. True parallelism across a CMA-ES *population* therefore needs separate OS processes
(each with its own GIL), not `run_batch`'s thread pool -- the opposite tradeoff from 014, where
the LNS candidates were pre-recorded into fast native `TapePolicy` objects that thread-parallelize
fine.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cma

from kaggriculture.eval.agents import resolve_policy, wrap_agent
from kaggriculture.search.policy import N_PARAMS, PARAM_SPECS, decode_params, default_x0, policy_agent
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import default_config_dict, episode_config

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Process-local cache, populated once per worker by `_init_worker` (see module docstring on why
# processes, not threads/closures, are needed here) -- avoids re-resolving "baseline" (which
# rebuilds a submission bundle) on every single fitness call.
_worker_state: dict = {}


def _init_worker(config: dict, opponent_specs: list[str], seed_set: list[int]) -> None:
    _worker_state["config"] = config
    _worker_state["opponents"] = [resolve_policy(spec, config) for spec in opponent_specs]
    _worker_state["seed_set"] = seed_set


def _win_rate(policy: native.Policy, opponents: list[native.Policy], seed_set: list[int], config: dict) -> float:
    """Mean per-game score (win=1, tie=0.5, loss=0) across `opponents x seed_set`, both seats --
    the issue's own objective choice (win rate, not mean money): thread 734412 and the project's
    own eval harness both treat wins, not margin, as what actually matters for rating."""
    ecfg = episode_config(config)
    pairs, configs, seeds = [], [], []
    for opponent in opponents:
        for seed in seed_set:
            pairs.append((policy, opponent))
            configs.append(ecfg)
            seeds.append(seed)
            pairs.append((opponent, policy))
            configs.append(ecfg)
            seeds.append(seed)
    results = native.run_batch(pairs, configs, seeds, 1)  # n_threads=1: GIL-serialized anyway, see module docstring
    score = 0.0
    for i, r in enumerate(results):
        a_money, b_money = (r.final_money[0], r.final_money[1]) if i % 2 == 0 else (r.final_money[1], r.final_money[0])
        if a_money > b_money:
            score += 1.0
        elif a_money == b_money:
            score += 0.5
    return score / len(results)


def _worker_fitness(x) -> float:
    """CMA-ES minimizes; returns 1 - win_rate. Reads `_worker_state`, populated once per process
    by `_init_worker` via `ProcessPoolExecutor`'s `initializer`."""
    config = _worker_state["config"]
    params = decode_params(x)
    agent_fn = policy_agent(params, config)
    policy = wrap_agent(agent_fn, config)
    win_rate = _win_rate(policy, _worker_state["opponents"], _worker_state["seed_set"], config)
    return 1.0 - win_rate


@dataclass
class CMAESResult:
    best_x: list[float]
    best_params: dict[str, float]
    best_fitness: float  # 1 - win_rate
    generations: int
    n_evaluations: int
    trace: list[dict] = field(default_factory=list)


def run_cmaes(
    config: dict,
    opponent_specs: list[str],
    seed_set: list[int],
    sigma0: float = 3.0,
    popsize: int | None = None,
    maxiter: int = 20,
    restarts: int = 1,
    n_workers: int = 1,
    seed: int = 0,
) -> CMAESResult:
    """Wide-sigma (default `sigma0=3.0`, in the raw sigmoid-domain -- see `search.policy`'s module
    docstring on why that explores close to both ends of every bounded range) CMA-ES with IPOP
    restarts (`restarts`, `incpopsize=2` -- pycma's own built-in mechanism, not reimplemented
    here) over the win-rate objective, parallelized across `n_workers` processes."""
    _init_worker(config, opponent_specs, seed_set)  # also populate the main process's cache

    trace: list[dict] = []

    def sequential_objective(x):
        return _worker_fitness(x)

    if n_workers > 1:
        executor = ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(config, opponent_specs, seed_set))

        def parallel_objective(X):
            return list(executor.map(_worker_fitness, X))
    else:
        executor = None

        def parallel_objective(X):
            return [sequential_objective(x) for x in X]

    def callback(es):
        trace.append(
            {
                "generation": es.countiter,
                "best_fitness": float(es.result.fbest) if es.result.fbest is not None else None,
                "best_win_rate": 1.0 - float(es.result.fbest) if es.result.fbest is not None else None,
                "sigma": float(es.sigma),
                "popsize": es.popsize,
            }
        )

    # pycma treats seed=0 (and None) as "seed from the current time" -- see `cma.CMAOptions()`'s
    # own description of the "seed" option -- so `seed=0` (this function's own default, and the
    # CLI's) would silently make every run non-reproducible. Shift by one to keep 0 a valid,
    # meaningful input while never actually passing 0 through.
    options = {"maxiter": maxiter, "verbose": -9, "seed": seed + 1}
    if popsize is not None:
        options["popsize"] = popsize

    try:
        _x, es = cma.fmin2(
            sequential_objective,
            default_x0(),
            sigma0,
            options=options,
            restarts=max(0, restarts - 1),
            incpopsize=2,
            parallel_objective=parallel_objective,
            callback=callback,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    best_x = list(es.result.xbest)
    best_fitness = float(es.result.fbest)
    return CMAESResult(
        best_x=best_x,
        best_params=decode_params(best_x),
        best_fitness=best_fitness,
        generations=es.countiter,
        n_evaluations=es.countevals,
        trace=trace,
    )


def sensitivity_analysis(
    best_x: list[float], config: dict, opponent_specs: list[str], seed_set: list[int], step: float = 1.0, n_workers: int = 1
) -> list[dict]:
    """Perturbs each raw coordinate by +/-`step` around the best solution found and measures the
    win-rate change -- "which dimensions actually matter" (the issue's own acceptance criterion),
    reported so the *finding* (which parameters are load-bearing) survives even after the numbers
    themselves get re-fit post an engine change. `2*N_PARAMS` evaluations, parallelized the same
    way as the main search (see module docstring) -- sequential was the difference between this
    step taking seconds and minutes."""
    _init_worker(config, opponent_specs, seed_set)
    perturbed = []
    for i in range(len(PARAM_SPECS)):
        for direction in (-1, 1):
            x = list(best_x)
            x[i] += direction * step
            perturbed.append(x)
    all_x = [best_x] + perturbed
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(config, opponent_specs, seed_set)) as executor:
            fitnesses = list(executor.map(_worker_fitness, all_x))
    else:
        fitnesses = [_worker_fitness(x) for x in all_x]
    win_rates = [1.0 - f for f in fitnesses]
    base_win_rate, perturbed_win_rates = win_rates[0], win_rates[1:]

    report = []
    for i, (name, lo, hi, _default) in enumerate(PARAM_SPECS):
        delta_minus = perturbed_win_rates[2 * i] - base_win_rate
        delta_plus = perturbed_win_rates[2 * i + 1] - base_win_rate
        report.append(
            {
                "param": name,
                "value": decode_params(best_x)[name],
                "bounds": [lo, hi],
                "win_rate_delta_minus": delta_minus,
                "win_rate_delta_plus": delta_plus,
                "sensitivity": max(abs(delta_minus), abs(delta_plus)),
            }
        )
    report.sort(key=lambda r: -r["sensitivity"])
    return report


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
    parser.add_argument("--opponents", default="pass,starter,baseline", help="training portfolio")
    parser.add_argument("--holdout-opponent", default="random", help="opponent NOT in the training portfolio, for the final verdict")
    parser.add_argument("--n-seeds", type=int, default=8, help="training seed set size")
    parser.add_argument("--holdout-n-seeds", type=int, default=15, help="disjoint seed set for the final verdict")
    parser.add_argument("--sigma0", type=float, default=3.0)
    parser.add_argument("--popsize", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=1, help="1 = no IPOP restart, 2+ = that many total runs (IPOP-CMA-ES)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensitivity-step", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = dict(default_config_dict())
    config["episodeSteps"] = args.episode_steps
    opponent_specs = args.opponents.split(",")
    search_seeds = list(range(args.n_seeds))
    holdout_seeds = list(range(1_000_000, 1_000_000 + args.holdout_n_seeds))

    t0 = time.perf_counter()
    result = run_cmaes(
        config,
        opponent_specs,
        search_seeds,
        sigma0=args.sigma0,
        popsize=args.popsize,
        maxiter=args.maxiter,
        restarts=args.restarts,
        n_workers=args.workers,
        seed=args.seed,
    )
    search_seconds = time.perf_counter() - t0
    print(f"CMA-ES: {result.n_evaluations} evaluations over {result.generations} generations, {search_seconds:.1f}s")
    print(f"  best training win rate: {1.0 - result.best_fitness:.3f}")

    sensitivity = sensitivity_analysis(result.best_x, config, opponent_specs, holdout_seeds, step=args.sensitivity_step, n_workers=args.workers)
    print("  top 5 sensitive parameters:")
    for row in sensitivity[:5]:
        print(f"    {row['param']:<28} value={row['value']:.3f}  sensitivity={row['sensitivity']:.3f}")

    best_policy = wrap_agent(policy_agent(result.best_params, config), config)

    # Held-out verdict #1: against the training portfolio, on seeds never used during search.
    holdout_portfolio_policies = [resolve_policy(spec, config) for spec in opponent_specs]
    holdout_win_rate = _win_rate(best_policy, holdout_portfolio_policies, holdout_seeds, config)
    print(f"  holdout-seed win rate vs training portfolio: {holdout_win_rate:.3f}")

    # Held-out verdict #2: against an opponent NEVER seen during training, at all.
    unseen_policy = resolve_policy(args.holdout_opponent, config)
    unseen_win_rate = _win_rate(best_policy, [unseen_policy], holdout_seeds, config)
    print(f"  win rate vs unseen opponent ({args.holdout_opponent}): {unseen_win_rate:.3f}")

    out_dir = args.out or _next_experiment_dir("cmaes")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "opponents": opponent_specs,
                "holdout_opponent": args.holdout_opponent,
                "n_seeds": args.n_seeds,
                "holdout_n_seeds": args.holdout_n_seeds,
                "sigma0": args.sigma0,
                "popsize": args.popsize,
                "maxiter": args.maxiter,
                "restarts": args.restarts,
                "workers": args.workers,
                "episode_steps": args.episode_steps,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    (out_dir / "incumbent_trace.json").write_text(json.dumps(result.trace, indent=2))
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "n_evaluations": result.n_evaluations,
                "generations": result.generations,
                "search_seconds": search_seconds,
                "best_training_win_rate": 1.0 - result.best_fitness,
                "holdout_win_rate_vs_training_portfolio": holdout_win_rate,
                "win_rate_vs_unseen_opponent": unseen_win_rate,
                "unseen_opponent": args.holdout_opponent,
                "best_params": result.best_params,
                "sensitivity": sensitivity,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

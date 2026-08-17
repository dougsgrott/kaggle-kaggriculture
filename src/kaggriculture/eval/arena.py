"""The evaluation arena (issue 011): paired seeds, both seats, Wilson CI + SPRT, scenario
stratification, population evaluation. This is what makes every other result in the repo
believable -- CLAUDE.md's non-negotiable #4, and every promotion decision should cite the
`result.json` this writes.

    uv run python -m kaggriculture.eval.arena baseline pass --n 50
    uv run python -m kaggriculture.eval.arena baseline pass --population starter,random --n 30
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from kaggriculture.eval.agents import resolve_policy
from kaggriculture.eval.stats import SPRTResult, WilsonInterval, sprt, verdict, wilson_interval
from kaggriculture.model.regimes import N_SLOTS, RegimeSet, Scenario, demand_curve, fit_regimes, sample_scenarios
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import default_config_dict, episode_config

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


@dataclass
class GameOutcome:
    seed: int
    a_seat: int  # which seat policy_a occupied in this game (0 or 1)
    a_money: float
    b_money: float
    regime: str


def _outcome_class(o: GameOutcome) -> str:
    if o.a_money > o.b_money:
        return "win"
    if o.a_money < o.b_money:
        return "loss"
    return "tie"


_DEFAULT_REGIME_SET: RegimeSet | None = None


def default_regime_set() -> RegimeSet:
    """Fit once per process and cache -- issue 012's taxonomy, sampled at a size (20k scenarios)
    that gives stable cluster centroids; costs a few seconds the first time `--regimes` is used,
    nothing after."""
    global _DEFAULT_REGIME_SET
    if _DEFAULT_REGIME_SET is None:
        _DEFAULT_REGIME_SET = fit_regimes(sample_scenarios(20_000))
    return _DEFAULT_REGIME_SET


def classify_regime(result: native.EpisodeResult, regime_set: RegimeSet) -> str:
    """issue 012's real demand-regime taxonomy, applied to the shop draw a *played* episode
    actually experienced. Reconstructs the draw from `shop_draw_order` (the shop types unlocked,
    in slot order) and recomputes its zero-production demand curve rather than reading the
    episode's own `daily_inventory` -- a regime is a property of the *world* a game was drawn
    into, not of what the two policies did with it, and the actual inventory trajectory is
    confounded by whatever they bought and sold."""
    draw = [native.ShopType(idx).name for idx in result.shop_draw_order]
    if len(draw) < N_SLOTS:
        # A short (< ~24-day) episode won't have drawn all 8 slots; classify on what's known.
        label = regime_set.classify_partial(draw)
    else:
        scenario = Scenario(seed=result.seed, draw=draw, cumulative=demand_curve(draw))
        label = regime_set.classify(scenario)
    return next(r.name for r in regime_set.regimes if r.label == label)


@dataclass
class ArenaResult:
    policy_a: str
    policy_b: str
    n_games: int
    wins: int
    losses: int
    ties: int
    interval: WilsonInterval
    verdict: str
    sprt: SPRTResult
    sprt_budget_exhausted: bool
    by_regime: dict[str, WilsonInterval]
    seconds: float

    def to_dict(self) -> dict:
        return {
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "n_games": self.n_games,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "interval": self.interval.to_dict(),
            "verdict": self.verdict,
            "sprt": self.sprt.to_dict(),
            "sprt_budget_exhausted": self.sprt_budget_exhausted,
            "by_regime": {k: v.to_dict() for k, v in self.by_regime.items()},
            "seconds": self.seconds,
        }


def compare(
    policy_a_spec: str,
    policy_b_spec: str,
    n_seeds: int,
    seed_start: int = 0,
    episode_steps: int = 720,
    n_threads: int = 1,
    p1: float = 0.55,
    max_n_for_sprt: int | None = None,
    use_regimes: bool = False,
) -> ArenaResult:
    """Runs `n_seeds` seeds, each seed played twice (policy_a in both seats) -- paired seeds,
    both seats, so seat and per-seed world luck cancel between the two orientations rather than
    confounding the comparison. `n_games = 2 * n_seeds`.

    `use_regimes` turns on issue 012's scenario-stratified reporting (`by_regime`); off by
    default since fitting the taxonomy costs a few seconds the first time (cached after that,
    see default_regime_set()) that a quick comparison shouldn't have to pay for.
    """
    t0 = time.perf_counter()
    config_dict = default_config_dict()
    config_dict["episodeSteps"] = episode_steps
    episode_cfg = episode_config(config_dict)
    episode_cfg.episode_steps = episode_steps

    # Resolved once, reused across every game -- rebuilding a submission bundle (e.g. "baseline")
    # per game would dominate runtime for no reason; Policy.act() carries no cross-call state.
    policy_a = resolve_policy(policy_a_spec, config_dict)
    policy_b = resolve_policy(policy_b_spec, config_dict)

    seeds = range(seed_start, seed_start + n_seeds)
    pairs = []
    configs = []
    seed_list = []
    orientations = []  # a_seat for each entry, parallel to pairs/seeds
    for seed in seeds:
        pairs.append((policy_a, policy_b))
        configs.append(episode_cfg)
        seed_list.append(seed)
        orientations.append(0)
        pairs.append((policy_b, policy_a))
        configs.append(episode_cfg)
        seed_list.append(seed)
        orientations.append(1)

    results = native.run_batch(pairs, configs, seed_list, n_threads)

    regime_set = default_regime_set() if use_regimes else None
    outcomes = []
    for seed, a_seat, result in zip(seed_list, orientations, results):
        a_money, b_money = (result.final_money[0], result.final_money[1]) if a_seat == 0 else (result.final_money[1], result.final_money[0])
        regime = classify_regime(result, regime_set) if regime_set is not None else "unclassified"
        outcomes.append(GameOutcome(seed=seed, a_seat=a_seat, a_money=a_money, b_money=b_money, regime=regime))

    wins = sum(1 for o in outcomes if _outcome_class(o) == "win")
    losses = sum(1 for o in outcomes if _outcome_class(o) == "loss")
    ties = sum(1 for o in outcomes if _outcome_class(o) == "tie")
    n_games = len(outcomes)

    interval = wilson_interval(wins + 0.5 * ties, n_games)
    v = verdict(interval)
    sprt_result = sprt(wins, losses, p1=p1)  # ties excluded -- SPRT's LLR wants decisive trials only
    # SPRT alone has no runtime bound when the true rate sits between p0 and p1 (see stats.sprt's
    # docstring) -- `verdict` (Wilson-CI-based, always defined) is the fallback once max_n is
    # spent, reported via `sprt_budget_exhausted` rather than by silently overwriting sprt.decision
    # (which would make an un-crossed llr look like it crossed a bound it didn't).
    sprt_budget_exhausted = sprt_result.decision == "continue" and max_n_for_sprt is not None and n_games >= max_n_for_sprt

    by_regime: dict[str, WilsonInterval] = {}
    if use_regimes:
        regimes = sorted({o.regime for o in outcomes})
        for regime in regimes:
            regime_outcomes = [o for o in outcomes if o.regime == regime]
            regime_wins = sum(1 for o in regime_outcomes if _outcome_class(o) == "win")
            regime_ties = sum(1 for o in regime_outcomes if _outcome_class(o) == "tie")
            by_regime[regime] = wilson_interval(regime_wins + 0.5 * regime_ties, len(regime_outcomes))

    return ArenaResult(
        policy_a=policy_a_spec,
        policy_b=policy_b_spec,
        n_games=n_games,
        wins=wins,
        losses=losses,
        ties=ties,
        interval=interval,
        verdict=v,
        sprt=sprt_result,
        sprt_budget_exhausted=sprt_budget_exhausted,
        by_regime=by_regime,
        seconds=time.perf_counter() - t0,
    )


def evaluate_population(challenger_spec: str, opponents: list[str], **compare_kwargs) -> dict[str, ArenaResult]:
    """Population evaluation, not just head-to-head: score `challenger_spec` against every
    opponent in `opponents` (a portfolio, meant to include older generations -- Kaito Fukami's
    failure mode is optimizing only against the latest top-30 and losing to retired meta)."""
    return {opponent: compare(challenger_spec, opponent, **compare_kwargs) for opponent in opponents}


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
    parser.add_argument("policy_a")
    parser.add_argument("policy_b")
    parser.add_argument("--n", type=int, default=50, help="seeds (games = 2 x n, both seat orders)")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=720,
        help="full season is 720; shortening this can flip verdicts for investment-heavy agents "
        "(e.g. baseline spends heavily on hands/seeds/a goose before melons mature around day "
        "12-15, so it can trail `pass` badly through ~day 8 and still win the season decisively)",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--p1", type=float, default=0.55, help="SPRT alternative-hypothesis win rate")
    parser.add_argument("--population", default=None, help="comma-separated extra opponents to also test policy_a against")
    parser.add_argument("--regimes", action="store_true", help="stratify by issue 012's demand-regime taxonomy (fits it on first use, a few seconds)")
    parser.add_argument("--out", type=Path, default=None, help="experiment dir (default: auto-numbered under experiments/)")
    args = parser.parse_args(argv)

    report: dict = {
        "policy_a": args.policy_a,
        "policy_b": args.policy_b,
        "n_seeds": args.n,
        "episode_steps": args.episode_steps,
    }

    result = compare(
        args.policy_a, args.policy_b, args.n, args.seed_start, args.episode_steps, args.threads, args.p1, max_n_for_sprt=args.n * 2, use_regimes=args.regimes
    )
    print(f"{args.policy_a} vs {args.policy_b}: {result.wins}W-{result.losses}L-{result.ties}T over {result.n_games} games")
    print(f"  Wilson 95% CI: [{result.interval.lo:.3f}, {result.interval.hi:.3f}]  verdict: {result.verdict}")
    print(f"  SPRT: {result.sprt.decision} (llr={result.sprt.llr:.2f}, bounds=[{result.sprt.lower:.2f}, {result.sprt.upper:.2f}])")
    if result.sprt_budget_exhausted:
        print(f"    (SPRT never crossed a bound within the game budget -- use the Wilson verdict above: {result.verdict})")
    for regime, interval in result.by_regime.items():
        print(f"  regime={regime}: n={interval.n} p_hat={interval.p_hat:.3f} CI=[{interval.lo:.3f},{interval.hi:.3f}]")
    report["result"] = result.to_dict()

    if args.population:
        opponents = args.population.split(",")
        pop_results = evaluate_population(
            args.policy_a,
            opponents,
            n_seeds=args.n,
            seed_start=args.seed_start,
            episode_steps=args.episode_steps,
            n_threads=args.threads,
            p1=args.p1,
            use_regimes=args.regimes,
        )
        report["population"] = {k: v.to_dict() for k, v in pop_results.items()}
        print(f"\nPopulation evaluation ({args.policy_a} vs {opponents}):")
        for opponent, r in pop_results.items():
            print(f"  vs {opponent}: {r.wins}W-{r.losses}L-{r.ties}T  verdict={r.verdict}  CI=[{r.interval.lo:.3f},{r.interval.hi:.3f}]")

    out_dir = args.out or _next_experiment_dir(f"arena-{args.policy_a}-vs-{args.policy_b}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()

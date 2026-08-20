"""Compose 014's LNS-optimized production plan with 016's sell-timing DP and 017's terminal
liquidation layer (issue 028), closing the gap issue 018's payoff matrix found: `lns014` (LNS plan,
naive floor-threshold selling) beats `current017` (016+017's sell sophistication, but on top of
013's *default*, non-LNS-optimized plan) 87% of the time. Neither 016 nor 017 is wrong -- each was
validated against the same production plan on both sides -- the two pieces have simply never been
run on the same, better plan together.

**Why this needs its own `measure_arrivals`, not `Schedule.arrivals` directly.** `search.sell_dp.
solve_all_products` needs a faithful `{product: {day: units}}` production ledger.
`build_schedule()` computes `arrivals` as a byproduct of its own day-by-day construction loop; LNS's
direct-mutation operators (`_swap_crop`, `_resize_herd`, `_resize_crew`, `_shift_land_day` in
`search.lns`) deep-copy a candidate `Schedule` and mutate `tile_role`/`crew_size_by_day`/
`land_days` directly, without recomputing `arrivals` -- only `_destroy_repair` (which calls
`build_schedule()` fresh) keeps it current. Checked directly against `experiments/exp-005-lns/
incumbent_trace.json`: the accepted-move sequence that produced the reported LNS incumbent ENDS in
`resize_crew`, a direct-mutation operator, so `arrivals` on that final object is stale. Trusting it
would risk solving a sell plan against production numbers that don't match what the schedule
actually does -- `measure_arrivals` gets ground truth from the engine instead (CLAUDE.md's
non-negotiable #1), by running the schedule once and tracking real shed increases turn by turn.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

from kaggriculture.eval.agents import _call_agent, resolve_policy, wrap_agent
from kaggriculture.model.constants import MARKET_PARAMS
from kaggriculture.search.agent import schedule_agent
from kaggriculture.search.liquidation import terminal_liquidation_agent
from kaggriculture.search.schedule import Schedule
from kaggriculture.search.sell_dp import solve_all_products
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import build_player_turn, episode_config

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

_TRACKED_ITEMS = {name: native.Item.__members__[name].value for name in MARKET_PARAMS}


def _track_shed_increases(agent_fn: Callable[[dict, dict], dict], config: dict, opponent: native.Policy | None = None, seed: int = 0) -> dict[str, dict[int, int]]:
    """Runs `agent_fn` through one recording episode against `opponent` (default: `pass`),
    tracking `private_shed` turn by turn and attributing every *increase* to that turn's day -- a
    decrease is a sale, not a harvest, and is ignored. Turn granularity, not day granularity:
    harvesting and selling routinely land in different turns of the same day (a schedule's
    floor-threshold sell step doesn't have to fire the instant something arrives), and this also
    sidesteps shed-capacity overflow (a product hitting `shedCapacity=100` and getting silently
    destroyed, per `search.agent`'s own `SHED_SAFETY_MARGIN` doc) being mistaken for zero arrivals
    -- gross positive deltas are immune to whatever concurrent selling keeps the shed from
    actually overflowing. The one gap turn granularity doesn't close: an arrival and a sale of the
    SAME item landing in the exact same turn's own action set would net out and undercount that
    turn's true gross arrival -- accepted as a rare, low-impact edge case (most selling targets
    inventory that arrived on an earlier turn) rather than tracked at sub-turn order. Factored out
    from `measure_arrivals` so a test can drive it with a small, fully-predictable `agent_fn`
    instead of a real `Schedule`."""
    ecfg = episode_config(config)
    turns_per_day = config["turnsPerDay"]
    max_orders = config.get("maxMarketOrdersPerTurn", 10)

    arrivals: dict[str, dict[int, int]] = {}
    prev_shed: dict[str, int] = dict.fromkeys(_TRACKED_ITEMS, 0)

    def callback(state: native.GameState, market: native.MarketTownState, player: int) -> native.PlayerTurn:
        day = state.step // turns_per_day
        for item, item_idx in _TRACKED_ITEMS.items():
            cur = state.private_shed(0, item_idx)
            prev = prev_shed[item]
            if cur > prev:
                day_totals = arrivals.setdefault(item, {})
                day_totals[day] = day_totals.get(day, 0) + (cur - prev)
            prev_shed[item] = cur
        obs = native.build_observation(state, market, player)
        action = _call_agent(agent_fn, obs, config)
        return build_player_turn(action or {}, max_orders)

    native.run_episode(native.CallbackPolicy(callback), opponent or native.TapePolicy([]), ecfg, seed)
    return arrivals


def measure_arrivals(schedule: Schedule, config: dict, opponent: native.Policy | None = None, seed: int = 0) -> dict[str, dict[int, int]]:
    """`_track_shed_increases` driven by `schedule_agent(schedule, config)` (default
    floor-threshold selling, not a `sell_plan`) -- see the module docstring for why this, not
    `schedule.arrivals`, is what `solve_all_products` needs to be solved against."""
    return _track_shed_increases(schedule_agent(schedule, config), config, opponent, seed)


def build_composed_agent(schedule: Schedule, config: dict, discount: float = 0.93, window_turns: int = 24) -> Callable[[dict, dict], dict]:
    """014's plan + 016's sell-timing DP (solved against `measure_arrivals`, not `schedule.
    arrivals`) + 017's terminal-liquidation endgame layer."""
    n_days = config["episodeSteps"] // config["turnsPerDay"]
    arrivals = measure_arrivals(schedule, config)
    sell_results = solve_all_products(arrivals, n_days, config["episodeSteps"], config["turnsPerDay"], discount=discount)
    sell_plan = {item: r.plan for item, r in sell_results.items()}
    dp_agent = schedule_agent(schedule, config, sell_plan=sell_plan)
    return terminal_liquidation_agent(dp_agent, config, window_turns=window_turns)


def _rebuild_lns_incumbent(config: dict, threads: int = 8) -> Schedule:
    """Reproduces `exp-005-lns`'s own incumbent exactly, matching its recorded parameters --
    matching the precedent `eval.population.own_lineage_policies` already established for
    `lns014`."""
    import random

    from kaggriculture.search.lns import diverse_seed_schedules, lns_search

    rng = random.Random(0)
    base_schedules = diverse_seed_schedules(config, rng, n_seeds=6, allow_4th_quadrant=True)
    opponents = [resolve_policy(spec, config) for spec in ("pass", "starter", "baseline")]
    result = lns_search(config, base_schedules, opponents, list(range(15)), n_iters=60, seed=0, n_threads=threads)
    return result.incumbent


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


def head_to_head(policy_a: native.Policy, policy_b: native.Policy, config: dict, n_seeds: int, n_threads: int, episode_steps: int) -> dict:
    """Paired-seed, both-seat head-to-head with a real win/loss/tie count and a Wilson CI --
    mirrors `eval.arena.compare`'s own game loop, but takes resolved `Policy` objects directly
    (matching `eval.egta._compare_policies`'s own reasoning: population members here aren't CLI
    spec strings)."""
    from kaggriculture.eval.stats import verdict as wilson_verdict
    from kaggriculture.eval.stats import wilson_interval

    econfig = dict(config)
    econfig["episodeSteps"] = episode_steps
    ecfg = episode_config(econfig)
    pairs, configs, seed_list, orientations = [], [], [], []
    for seed in range(n_seeds):
        pairs += [(policy_a, policy_b), (policy_b, policy_a)]
        configs += [ecfg, ecfg]
        seed_list += [seed, seed]
        orientations += [0, 1]

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
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_games": n_games,
        "interval": interval.to_dict(),
        "verdict": wilson_verdict(interval),
    }


def main(argv: list[str] | None = None) -> None:
    from kaggriculture.eval.agents import record_tape
    from kaggriculture.eval.population import load_public_policy
    from kaggriculture.search.schedule import build_schedule
    from kaggriculture.sim.decode import default_config_dict

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = dict(default_config_dict())
    config["episodeSteps"] = args.episode_steps
    n_days = config["episodeSteps"] // config["turnsPerDay"]

    t0 = time.perf_counter()
    incumbent = _rebuild_lns_incumbent(config, threads=args.threads)
    composed_agent = build_composed_agent(incumbent, config)
    composed_policy = wrap_agent(composed_agent, config)
    build_seconds = time.perf_counter() - t0
    print(f"Built the composed agent in {build_seconds:.1f}s")

    # lns014 (naive-selling LNS plan) and current017 (016+017 on 013's default plan), rebuilt the
    # same way eval.population.own_lineage_policies does, so this comparison is apples to apples.
    lns014_policy = record_tape(schedule_agent(incumbent, config), config)

    schedule016 = build_schedule(config)
    sell_results = solve_all_products(schedule016.arrivals, n_days, config["episodeSteps"], config["turnsPerDay"], discount=0.93)
    sell_plan016 = {item: r.plan for item, r in sell_results.items()}
    dp016_agent = schedule_agent(schedule016, config, sell_plan=sell_plan016)
    current017_policy = wrap_agent(terminal_liquidation_agent(dp016_agent, config), config)

    opponents = {"current017": current017_policy, "lns014": lns014_policy, "baseline": resolve_policy("baseline", config)}
    try:
        opponents["beicicc__kaggriculture-c26-kaito-v23-dual-regime"] = load_public_policy("beicicc__kaggriculture-c26-kaito-v23-dual-regime", config)
    except Exception as exc:  # pragma: no cover -- corpus-dependent (analysis/nb_clean/ is gitignored), best-effort context only
        print(f"(skipped beicicc-c26 comparison: {exc})")

    results: dict[str, dict] = {}
    for label, opponent_policy in opponents.items():
        r = head_to_head(composed_policy, opponent_policy, config, args.n_seeds, args.threads, args.episode_steps)
        print(f"vs {label}: {r['wins']}W-{r['losses']}L-{r['ties']}T  CI=[{r['interval']['lo']:.3f},{r['interval']['hi']:.3f}]  verdict={r['verdict']}")
        results[label] = r

    out_dir = args.out or _next_experiment_dir("lns-sell-dp-composition")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"episode_steps": args.episode_steps, "n_seeds": args.n_seeds, "threads": args.threads}, indent=2))
    (out_dir / "result.json").write_text(json.dumps({"build_seconds": build_seconds, "results": results}, indent=2))
    print(f"\nwrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()

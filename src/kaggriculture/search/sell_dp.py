"""Sell-schedule DP against the price curve (issue 016).

Thread 734412 (see this issue's References) measured a strawberry season worth $100,445 sold into
the demand hole the town digs versus $4,173 dumped flat -- a 24x swing from *timing alone*, on
identical production. The mechanism: `model.price.price(item, inventory)` is monotonic
non-increasing in inventory, selling pushes inventory up the glut side, and the town (shops every
4 turns, the town centre once a day) drains it back down the scarcity side for free, all season.
Every product therefore has two values and which one a player gets depends only on *when* they
sell -- exactly the "sell at harvest" default every agent in this repo has used through issue 015
(`search.agent`'s `_market_orders`, `search.policy`'s `_sell_floor`: sell everything above a fixed
fraction of base price, the moment it's in the shed).

**The DP.** Given a production schedule (units of one product arriving in the shed on each day --
`search.schedule.Schedule.arrivals`, this module's own upstream dependency) and a forecast of the
market's town-driven drain (`model.regimes.demand_curve`, or this module's own `draw`-free
"expected" average over the shop-draw distribution), choose how many units to sell each day to
maximize total revenue. Exact, not approximate, per product:

  - **State**: `(day, R)` where `R` is cumulative units sold so far. Solved backward from the
    season's end, where the terminal value is 0 -- this alone guarantees the optimal policy fully
    liquidates by the final day (any credit left on the table at `day = n_days` is strictly worse
    than selling it on the last day at whatever price remains, however low), with no separate
    "liquidation" mechanism bolted on. A *further* end-of-season liquidation problem -- competing
    with the opponent's own last-minute dump, or squeezing out the literal final turn rather than
    this module's day-granularity -- is issue 017's territory, not re-solved here.
  - **Action**: sell `k` units today, `0 <= k <= min(remaining_available, k_max_per_day)`. Revenue
    for a `k`-unit burst replays the engine's own per-unit lockstep exactly (`_prefix_revenue`):
    each unit quotes at the *current* inventory, and inventory increments after each unit **except**
    a sale at the $1 floor, which does not increment it (`vendor.kaggriculture._commit_unit`'s own
    `if price > 1` guard) -- so a floor dump doesn't self-reinforce.
  - **Transition**: `R' = R + k`. The cross-day carry treats every sold unit as incrementing
    inventory (`base_inv(day) + R`), even though a floor-priced unit within a burst technically
    didn't. This is a deliberate, documented approximation, not an oversight: modeling the true
    floor-frozen state would need a second state dimension per product (`R` for shed/availability
    bookkeeping, a separate credited counter for pricing), roughly squaring this DP's cost for a
    correction that only matters once a day's burst is large enough to hit the floor -- something
    an optimizing DP has no incentive to do in the first place (marginal revenue at the floor is
    $1, worse than almost any alternative). The bias this introduces is conservative in direction:
    it slightly *overstates* future inventory after a big sell, so it slightly *understates* future
    prices, discouraging exactly the floor-chasing behaviour that would need the correction.
  - **Separability**: solved independently per product (the issue's own "products are coupled only
    through the order budget and the shed" framing). The order-budget reconciliation happens live
    at the agent level (`search.agent.schedule_agent`'s `sell_plan` argument): the plan's quantity
    for today is emitted as one order per product, same as the existing floor-based logic, so it
    competes for the 10-order budget exactly the way every other order type already does -- no
    separate Lagrangian pass turned out to be needed (see the issue's Revision section for why: with
    a season of daily granularity and 9 products, the order budget essentially never binds). Shed
    capacity (100, shared) is not modeled inside the DP itself; `schedule_agent`'s own runtime falls
    back to opportunistic (floor-threshold) selling whenever total shed holdings run high, which
    both liquidates real production drift away from this module's *analytical* arrivals forecast and
    keeps the shared shed from silently destroying overflow.

**Day granularity, not turn granularity.** Town drain happens on specific turns (every 4 for shops,
every 24 for the centre); this module decides once per day, using each day's *prior*-day cumulative
drain as that day's baseline inventory (a day's own drain, still to come, is not credited yet -- a
small conservative choice, not a turn-exact replay). The 24x example itself is about spreading sales
across the *season*, not particular turns within a day, so this doesn't cost the effect the issue is
chasing; a turn-granularity version is a legitimate follow-up, not attempted here.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from kaggriculture.eval.agents import wrap_agent
from kaggriculture.eval.stats import verdict as wilson_verdict
from kaggriculture.eval.stats import wilson_interval
from kaggriculture.model.constants import MARKET_I0, MARKET_PARAMS, PRICE_FLOOR
from kaggriculture.model.price import price as market_price
from kaggriculture.model.regimes import NON_FERTILIZER_PRODUCTS, demand_curve, draw_random_slots
from kaggriculture.search.agent import schedule_agent
from kaggriculture.search.schedule import Schedule, build_schedule
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import default_config_dict, episode_config

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

K_MAX_PER_DAY = 80  # nobody sanely dumps more than this of one product in one day -- shed cap is
# 100 units shared across every product, so this is already a generous per-product ceiling; it
# exists purely to bound DP cost (see module docstring's complexity note), not to reflect a real
# game rule.


def _cumulative_drain(item: str, draw: list[str], episode_steps: int, turns_per_day: int) -> list[int]:
    """Per-day cumulative town-only drain for `item`, zero for FERTILIZER (no shop or town-centre
    entry ever touches it -- `model.constants.TOWN_CENTER_PRODUCTS` and every `SHOPS` list agree)."""
    n_days = episode_steps // turns_per_day
    if item == "FERTILIZER":
        return [0] * n_days
    return demand_curve(draw, episode_steps, turns_per_day)[item]


def expected_drain_cumulative(
    episode_steps: int, turns_per_day: int, n_samples: int = 200, seed: int = 0
) -> dict[str, list[float]]:
    """The `draw`-free default forecast: per-day cumulative town drain averaged over `n_samples`
    i.i.d. shop draws (`model.regimes.draw_random_slots`'s own distribution) -- a full day-by-day
    generalization of `search.schedule`'s `_SEASON_ABSORPTION` (which only keeps the season-end
    total). Used when the caller doesn't have (or doesn't want to condition on) a specific regime's
    draw; pass a concrete `draw` to `solve_all_products` instead for issue 012's regime-aware
    solving, which this function intentionally does not do on its own."""
    rng = random.Random(seed)
    n_days = episode_steps // turns_per_day
    totals = {p: [0.0] * n_days for p in NON_FERTILIZER_PRODUCTS}
    for _ in range(n_samples):
        draw = draw_random_slots(rng)
        cum = demand_curve(draw, episode_steps, turns_per_day)
        for p in NON_FERTILIZER_PRODUCTS:
            row = totals[p]
            for d in range(n_days):
                row[d] += cum[p][d]
    for p in totals:
        totals[p] = [v / n_samples for v in totals[p]]
    return totals


def market_forecast(
    cumulative_drain: list[float],
    opponent_cumulative_sold: list[int] | None = None,
) -> list[float]:
    """`base_inv(day)` for `day in range(n_days)`: `MARKET_I0` minus town drain accumulated
    through the *previous* day (day 0 sees none yet -- see module docstring) minus the opponent's
    assumed cumulative units sold by that day (default: the opponent doesn't sell this product at
    all -- the same "exogenous, deliberately conservative" treatment issue 014 gave the opponent's
    plan)."""
    n_days = len(cumulative_drain)
    out = []
    for d in range(n_days):
        drained = cumulative_drain[d - 1] if d > 0 else 0.0
        opp = opponent_cumulative_sold[d] if opponent_cumulative_sold else 0
        out.append(MARKET_I0 - drained - opp)
    return out


def _prefix_revenue(item: str, base_inv: float, k_max: int, params: dict | None = None) -> list[float]:
    """`prefix[k]` = revenue from selling `k` units starting at `base_inv`, replaying the engine's
    per-unit lockstep exactly (see module docstring). `prefix[0] == 0`."""
    inv = base_inv
    revenue = 0.0
    prefix = [0.0]
    for _ in range(k_max):
        p = market_price(item, inv, params)
        revenue += p
        if p > PRICE_FLOOR:
            inv += 1
        prefix.append(revenue)
    return prefix


@dataclass
class SellDPResult:
    plan: dict[int, int]  # day -> units to sell
    total_revenue: float
    total_units: int
    dump_flat_revenue: float  # same total units, all sold the day they arrive -- the myopic baseline


def solve_product(
    item: str,
    arrivals_by_day: dict[int, int],
    n_days: int,
    base_inv_by_day: list[float],
    k_max_per_day: int = K_MAX_PER_DAY,
    params: dict | None = None,
    discount: float = 1.0,
) -> SellDPResult:
    """Exact DP over `(day, cumulative_units_sold)` -- see module docstring for the model.

    `discount` (per-day multiplicative decay applied to the OPTIMIZATION objective only --
    `total_revenue`/`plan` are reconstructed from the real, undiscounted per-day revenue along the
    chosen path) is a deliberate correction, not part of the original formulation: a pure
    revenue-maximizing DP has no reason to prefer day 5 over day 25 for two equally-priced units,
    but a dollar realized on day 5 can be reinvested into more crew/land/seeds for the following 25
    days, while a dollar realized on day 25 cannot buy anything the season still has time to use.
    `discount=1.0` (no time preference) is what the issue's own scope literally asks for --
    "maximize revenue" -- and is the default; see the issue's Revision section for why a nonzero
    discount was needed in practice once this was checked against final money, not sell revenue in
    isolation."""
    cum_produced = [0] * n_days
    running = 0
    for d in range(n_days):
        running += arrivals_by_day.get(d, 0)
        cum_produced[d] = running
    total = cum_produced[-1] if n_days else 0

    # "Dump flat": sell every arriving unit the day it lands -- the myopic baseline every agent in
    # this repo has used through issue 015. Its own cumulative-sold state (`r`) is carried forward
    # exactly like the DP's, just forced to always equal cumulative production to date.
    dump_flat_revenue = 0.0
    if total > 0:
        r = 0
        for d in range(n_days):
            k = arrivals_by_day.get(d, 0)
            if k <= 0:
                continue
            prefix = _prefix_revenue(item, base_inv_by_day[d] + r, k, params)
            dump_flat_revenue += prefix[-1]
            r += k

    if total == 0:
        return SellDPResult(plan={}, total_revenue=0.0, total_units=0, dump_flat_revenue=0.0)

    dp_next = [0.0] * (total + 1)  # dp after the last day: nothing left to decide, value 0
    decisions: list[list[int]] = []

    for day in reversed(range(n_days)):
        cap = cum_produced[day]
        base = base_inv_by_day[day]
        day_weight = discount**day
        dp_cur = [0.0] * (cap + 1)
        decision_cur = [0] * (cap + 1)
        for r in range(cap + 1):
            avail = min(cap - r, k_max_per_day)
            prefix = _prefix_revenue(item, base + r, avail, params)
            best_val, best_k = -1.0, 0
            for k in range(avail + 1):
                future = dp_next[r + k]
                val = day_weight * prefix[k] + future
                if val > best_val:
                    best_val, best_k = val, k
            dp_cur[r] = best_val
            decision_cur[r] = best_k
        decisions.append(decision_cur)
        dp_next = dp_cur

    decisions.reverse()
    plan: dict[int, int] = {}
    r = 0
    total_revenue = 0.0
    for day in range(n_days):
        k = decisions[day][r]
        if k > 0:
            plan[day] = k
            total_revenue += _prefix_revenue(item, base_inv_by_day[day] + r, k, params)[-1]
        r += k

    return SellDPResult(plan=plan, total_revenue=total_revenue, total_units=r, dump_flat_revenue=dump_flat_revenue)


def solve_all_products(
    arrivals: dict[str, dict[int, int]],
    n_days: int,
    episode_steps: int,
    turns_per_day: int,
    draw: list[str] | None = None,
    opponent_arrivals: dict[str, dict[int, int]] | None = None,
    k_max_per_day: int = K_MAX_PER_DAY,
    discount: float = 1.0,
) -> dict[str, SellDPResult]:
    """Solves every product with a nonzero `arrivals` entry. `draw=None` uses the draw-free
    `expected_drain_cumulative` forecast (issue's default, regime-agnostic); pass a concrete
    8-shop `draw` for issue 012's regime-aware solving instead."""
    if draw is not None:
        drain_by_product = {item: _cumulative_drain(item, draw, episode_steps, turns_per_day) for item in arrivals}
    else:
        expected = expected_drain_cumulative(episode_steps, turns_per_day)
        drain_by_product = {item: expected.get(item, [0.0] * n_days) for item in arrivals}

    results = {}
    for item, arrivals_by_day in arrivals.items():
        if item not in MARKET_PARAMS or not arrivals_by_day:
            continue
        opp_cum = None
        if opponent_arrivals and item in opponent_arrivals:
            opp_cum = [0] * n_days
            running = 0
            for d in range(n_days):
                running += opponent_arrivals[item].get(d, 0)
                opp_cum[d] = running
        base_inv_by_day = market_forecast(drain_by_product[item], opp_cum)
        results[item] = solve_product(item, arrivals_by_day, n_days, base_inv_by_day, k_max_per_day, discount=discount)
    return results


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


def head_to_head_sell_plan(
    schedule: Schedule, sell_plan: dict[str, dict[int, int]], config: dict, seed_set: list[int], n_threads: int = 1
) -> dict:
    """Paired-seed, both-seat comparison of the SAME schedule executed two ways: the existing
    floor-threshold sell logic (`schedule_agent(schedule, config)`) vs. this module's DP-derived
    `sell_plan` (`schedule_agent(schedule, config, sell_plan=...)`) -- isolates the gain
    attributable purely to sell *timing*, holding production identical, exactly as the issue's own
    acceptance text asks for.

    Deliberately live (`wrap_agent`/`CallbackPolicy`), not `record_tape`/`TapePolicy` -- see the
    issue's Revision section for why: a `sell_plan`'s runtime safety valves (`MIN_LIQUIDITY_RESERVE`,
    `SHED_SAFETY_MARGIN` in `search.agent`) only fire correctly against the *actual* live game
    they're running in. `record_tape` bakes a single playthrough's decisions (recorded against
    `pass`) into a fixed action sequence and replays it unchanged regardless of the real opponent --
    fine for the existing floor-threshold logic (which degrades gracefully to "sell most of the
    shed" either way) but silently defeats a sell plan's own reactive fallbacks the moment the real
    opponent's market activity diverges from the `pass`-recorded forecast, which a competing seller
    always will."""
    baseline_policy = wrap_agent(schedule_agent(schedule, config), config)
    dp_policy = wrap_agent(schedule_agent(schedule, config, sell_plan=sell_plan), config)
    ecfg = episode_config(config)
    pairs, configs, seeds, orientations = [], [], [], []
    for seed in seed_set:
        pairs.append((dp_policy, baseline_policy))
        orientations.append(0)
        pairs.append((baseline_policy, dp_policy))
        orientations.append(1)
        configs.extend([ecfg, ecfg])
        seeds.extend([seed, seed])
    results = native.run_batch(pairs, configs, seeds, n_threads)
    wins = losses = ties = 0
    dp_moneys, baseline_moneys = [], []
    for orientation, r in zip(orientations, results):
        dp_money, base_money = (r.final_money[0], r.final_money[1]) if orientation == 0 else (r.final_money[1], r.final_money[0])
        dp_moneys.append(dp_money)
        baseline_moneys.append(base_money)
        if dp_money > base_money:
            wins += 1
        elif dp_money < base_money:
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
        "mean_dp_money": sum(dp_moneys) / n_games,
        "mean_baseline_money": sum(baseline_moneys) / n_games,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--n-seeds", type=int, default=20, help="paired-seed, both-seat comparison seed count")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--draw-seed", type=int, default=None, help="solve against one concrete sampled draw instead of the expected/generic forecast")
    parser.add_argument(
        "--discount",
        type=float,
        default=0.93,
        help="per-day time-value-of-money discount on the DP's objective (see solve_product's docstring); "
        "1.0 is the issue's literal 'maximize revenue' formulation, but final money vs pass showed that "
        "undiscounted holding gives back its own gain in lost reinvestment cash -- 0.93 is the empirically "
        "tuned default (see the issue's Revision section for the sweep)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = dict(default_config_dict())
    config["episodeSteps"] = args.episode_steps
    n_days = config["episodeSteps"] // config["turnsPerDay"]

    schedule = build_schedule(config)
    draw = None
    if args.draw_seed is not None:
        draw = draw_random_slots(random.Random(args.draw_seed))

    t0 = time.perf_counter()
    results = solve_all_products(schedule.arrivals, n_days, config["episodeSteps"], config["turnsPerDay"], draw=draw, discount=args.discount)
    solve_seconds = time.perf_counter() - t0

    sell_plan = {item: r.plan for item, r in results.items()}
    total_dp_revenue = sum(r.total_revenue for r in results.values())
    total_dump_flat_revenue = sum(r.dump_flat_revenue for r in results.values())

    print(f"Sell DP: {len(results)} products, {solve_seconds:.2f}s")
    for item, r in sorted(results.items(), key=lambda kv: -kv[1].total_revenue):
        gain = (r.total_revenue / r.dump_flat_revenue - 1.0) * 100 if r.dump_flat_revenue > 0 else 0.0
        print(f"  {item:12s} units={r.total_units:4d}  DP revenue=${r.total_revenue:9.0f}  dump-flat=${r.dump_flat_revenue:9.0f}  ({gain:+.1f}%)")
    print(f"  TOTAL: DP=${total_dp_revenue:.0f}  dump-flat=${total_dump_flat_revenue:.0f}")

    seed_set = list(range(args.seed_start, args.seed_start + args.n_seeds))
    h2h = head_to_head_sell_plan(schedule, sell_plan, config, seed_set, args.threads)
    print(
        f"\nDP sell-plan vs floor-threshold, same schedule: {h2h['wins']}W-{h2h['losses']}L-{h2h['ties']}T "
        f"over {h2h['n_games']} games  verdict={h2h['verdict']}  CI=[{h2h['interval']['lo']:.3f},{h2h['interval']['hi']:.3f}]"
    )
    print(f"  mean money: DP=${h2h['mean_dp_money']:.0f}  floor-threshold=${h2h['mean_baseline_money']:.0f}")

    out_dir = args.out or _next_experiment_dir("sell-dp")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "episode_steps": args.episode_steps,
                "n_seeds": args.n_seeds,
                "seed_start": args.seed_start,
                "threads": args.threads,
                "draw_seed": args.draw_seed,
                "discount": args.discount,
            },
            indent=2,
        )
    )
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "solve_seconds": solve_seconds,
                "per_product": {
                    item: {"units": r.total_units, "dp_revenue": r.total_revenue, "dump_flat_revenue": r.dump_flat_revenue}
                    for item, r in results.items()
                },
                "total_dp_revenue": total_dp_revenue,
                "total_dump_flat_revenue": total_dump_flat_revenue,
                "head_to_head_vs_floor_threshold": h2h,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

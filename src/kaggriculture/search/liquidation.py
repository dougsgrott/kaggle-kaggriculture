"""Terminal liquidation optimizer (issue 017).

**Unsold shed inventory scores zero at turn 720** (`wiki/discussions/topics/733155.md`), so the
endgame is a forced sell-off with no next day to recover a bad price in. Issue 016's DP already
guarantees full liquidation by the season's last day (its terminal value is zero for unsold
credit), but it does so at *day* granularity with the opponent's own selling treated as exogenous
zero — appropriate mid-season (016's Revision section found the opponent-exogenous assumption
costly even there), but the wrong call for the specific case this issue is about: the final day,
where BOTH players are almost certainly liquidating something, and the within-day *turn* structure
(shop drain every 4 turns, `model.town.drain_at_step`) actually matters at the scale of a single
day in a way 016's day-averaged model can't see.

**The mechanic this module is built on, verified directly against `vendor.kaggriculture` rather
than assumed from the issue's own framing** (see the issue's Revision section for the numbers):
when two players SELL the same item in the same turn, `vendor._process_market`'s per-unit lockstep
quotes BOTH players at the *same* shared inventory each round and commits both before moving to the
next round — so simultaneous equal-sized orders get IDENTICAL per-unit prices (no "who goes first"
advantage from player order). What actually matters is order SIZE: whichever order is smaller
finishes first and gets to keep exactly the price trajectory both were seeing while both were
active; the larger order's *remaining* units, once alone, face a market whose inventory is growing
at half the rate their concurrent portion did — but growing for twice as long, i.e. *my* effective
inventory growth rate while the opponent's order is also active is roughly double a solo dump's,
which quantitatively is the whole reason "the opponent is liquidating too" is worth modeling
explicitly rather than assuming zero.

`simulate_concurrent_sale` / `_my_revenue_prefix_with_opponent` are exact replays of that lockstep,
not an approximation of it. `solve_endgame` is a DP built on top, at TURN (not day) granularity,
for a short final window.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

from kaggriculture.model.constants import ANIMALS, CROPS, MARKET_I0, MARKET_PARAMS, PRICE_FLOOR
from kaggriculture.model.economics import (
    animal_profit_per_action,
    one_time_crop_profit_per_action,
    ongoing_crop_profit_per_action,
)
from kaggriculture.model.price import price as market_price
from kaggriculture.model.regimes import demand_curve_from_shop_counts
from kaggriculture.model.town import drain_at_step

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Same formula as search.schedule's own (module-private) `_effective_price`/`_SEASON_ABSORPTION`:
# a marginal unit's price, given everything already committed to that product this season, netted
# against the season-long town-absorption credit. Recomputed here rather than imported (the
# original is a private symbol) -- see `wind_down_day`, which needs exactly this to decide whether
# a fresh planting this late is worth it at all.
_SEASON_ABSORPTION: dict[str, float] = demand_curve_from_shop_counts([], remaining_slots=8)


def _effective_price(item: str, committed: float) -> int:
    offset = max(0.0, committed - _SEASON_ABSORPTION.get(item, 0.0))
    return market_price(item, MARKET_I0 + offset)


def wind_down_day(item: str, n_days: int, committed_units: dict[str, float]) -> int | None:
    """The latest day a NEW planting (crop) or purchase (animal) of `item` can still start and be
    worth it: gated by (a) physically finishing before season end and (b) clearing a positive
    $/action at the price a fresh unit would fetch given what's already committed to that product
    this season (`_effective_price` -- the same glut a mid-season planting would face, just with no
    remaining season to wait it out). Returns `None` if never profitable at the current committed
    level, regardless of day. `item` is a crop name or an animal name."""
    if item in CROPS:
        cd = CROPS[item]
        price = _effective_price(item, committed_units.get(item, 0.0))
        ppa = ongoing_crop_profit_per_action(item, price) if cd["ongoing"] else one_time_crop_profit_per_action(item, price)
        return n_days - 1 - cd["first_yield_day"] if ppa > 0 else None
    if item in ANIMALS:
        a = ANIMALS[item]
        price = _effective_price(a["product"], committed_units.get(a["product"], 0.0))
        ppa = animal_profit_per_action(item, price, care=True)
        return n_days - 1 - a["first_yield_day"] if ppa > 0 else None
    raise ValueError(f"{item!r} is not a crop or animal name")


def _concurrent_walk(item: str, base_inv: float, my_units: int, opponent_units: int, params: dict | None = None) -> tuple[float, float, float]:
    """One round-by-round replay of `vendor._process_market`'s per-unit lockstep for two
    simultaneous SELL orders of the same item. Returns (my_revenue, opponent_revenue,
    ending_inventory)."""
    inv = base_inv
    my_remaining, opp_remaining = my_units, opponent_units
    my_revenue = opp_revenue = 0.0
    while my_remaining > 0 or opp_remaining > 0:
        p = market_price(item, inv, params)
        active = 0
        if my_remaining > 0:
            my_revenue += p
            my_remaining -= 1
            active += 1
        if opp_remaining > 0:
            opp_revenue += p
            opp_remaining -= 1
            active += 1
        if p > PRICE_FLOOR:
            inv += active
    return my_revenue, opp_revenue, inv


def simulate_concurrent_sale(item: str, base_inv: float, my_units: int, opponent_units: int, params: dict | None = None) -> tuple[float, float]:
    """Both players' exact revenue for simultaneous same-item SELL orders in one turn -- the
    direct tool for the issue's own "model the opponent liquidating concurrently" and
    denial-selling questions (see the issue's Revision section for a worked table)."""
    my_revenue, opp_revenue, _ = _concurrent_walk(item, base_inv, my_units, opponent_units, params)
    return my_revenue, opp_revenue


def _my_revenue_prefix_with_opponent(item: str, base_inv: float, k_max: int, opponent_units: int, params: dict | None = None) -> list[float]:
    """`prefix[j]` = my revenue from selling j units (j=0..k_max) this turn, while the opponent
    concurrently sells (up to) `opponent_units` of the same item. One forward walk yields every
    prefix: the opponent's own per-round decrement is independent of how many units I end up
    choosing (their order is a fixed sequence, not a reaction to mine), so `prefix[k]` for a
    smaller `k` is just a truncation of the same walk used for a larger one."""
    inv = base_inv
    opp_remaining = opponent_units
    revenue = 0.0
    prefix = [0.0]
    for _ in range(k_max):
        p = market_price(item, inv, params)
        revenue += p
        active = 1
        if opp_remaining > 0:
            opp_remaining -= 1
            active += 1
        if p > PRICE_FLOOR:
            inv += active
        prefix.append(revenue)
    return prefix


def mirror_opponent_schedule(holdings: int) -> dict[int, int]:
    """The default opponent liquidation model: assume the opponent holds about as much of this
    product as I do (the only observable proxy for their private shed -- see the issue's Revision
    section on why this is a documented assumption, not an inferred one) and dumps all of it on the
    FIRST turn of the window -- the same crude behaviour as the "sell everything on the last few
    turns" baseline this issue exists to beat, so assuming the opponent does exactly that is the
    realistic case to plan against, not a worst case invented for safety margin."""
    return {0: holdings} if holdings > 0 else {}


def _opponent_trajectory(
    item: str,
    base_inv: float,
    opponent_schedule: dict[int, int],
    window_turns: int,
    unlocked_shops: list[str],
    turn_start: int,
    shop_interval: int,
    center_interval: int,
    params: dict | None = None,
) -> list[float]:
    """Inventory level at the START of each turn in the window, from town drain plus the
    opponent's assumed schedule ALONE (not interacting with my own choices) -- an approximation
    (it ignores any feedback my own concurrent sales could have on whether the opponent's OWN
    sales hit the floor), documented the same way issue 016's cross-day `R`-as-inventory-proxy
    approximation is. The opponent's contribution DURING turn t is applied live inside that turn's
    `_my_revenue_prefix_with_opponent` call, not folded into `inv_at_turn[t]` here -- this function
    only carries PRIOR turns' opponent activity forward."""
    inv = base_inv
    trajectory = []
    for t in range(window_turns):
        trajectory.append(inv)
        for _ in range(opponent_schedule.get(t, 0)):
            p = market_price(item, inv, params)
            if p > PRICE_FLOOR:
                inv += 1
        drain = drain_at_step(unlocked_shops, turn_start + t, shop_interval, center_interval).get(item, 0)
        inv -= drain
    return trajectory


def solve_endgame(
    item: str,
    holdings: int,
    window_turns: int,
    turn_start: int,
    base_inv: float,
    unlocked_shops: list[str],
    opponent_schedule: dict[int, int] | None = None,
    shop_interval: int = 4,
    center_interval: int = 24,
    k_max_per_turn: int = 100,
    params: dict | None = None,
) -> dict[int, int]:
    """Exact DP over `(turn offset into the window, cumulative units sold)`, maximizing my own
    revenue for liquidating `holdings` units of `item` over `window_turns` turns starting at
    absolute step `turn_start`, against a concurrently-liquidating opponent (`opponent_schedule`,
    default `mirror_opponent_schedule(holdings)`). No new arrivals are modeled within the window
    (a fixed `holdings`, not a per-turn arrivals schedule) -- reasonable for a short (one-day)
    final window where the season's own bookkeeping already knows what's in the shed; see the
    issue's Revision section. Same terminal-value-zero property as issue 016's day-DP, so this
    always fully liquidates `holdings` by the window's last turn.
    """
    if holdings <= 0:
        return {}
    opponent_schedule = opponent_schedule if opponent_schedule is not None else mirror_opponent_schedule(holdings)
    inv_at_turn = _opponent_trajectory(
        item, base_inv, opponent_schedule, window_turns, unlocked_shops, turn_start, shop_interval, center_interval, params
    )

    dp_next = [0.0] * (holdings + 1)
    decisions: list[list[int]] = []
    for t in reversed(range(window_turns)):
        opp_units_t = opponent_schedule.get(t, 0)
        dp_cur = [0.0] * (holdings + 1)
        decision_cur = [0] * (holdings + 1)
        for r in range(holdings + 1):
            avail = min(holdings - r, k_max_per_turn)
            prefix = _my_revenue_prefix_with_opponent(item, inv_at_turn[t] + r, avail, opp_units_t, params)
            best_val, best_k = -1.0, 0
            for k in range(avail + 1):
                val = prefix[k] + dp_next[r + k]
                if val > best_val:
                    best_val, best_k = val, k
            dp_cur[r] = best_val
            decision_cur[r] = best_k
        decisions.append(decision_cur)
        dp_next = dp_cur

    decisions.reverse()
    plan: dict[int, int] = {}
    r = 0
    for t in range(window_turns):
        k = decisions[t][r]
        if k > 0:
            plan[t] = k
        r += k
    return plan


def dump_everything_first_turn(holdings: int) -> dict[int, int]:
    """The naive baseline this issue's acceptance criterion measures against: "sell everything on
    the last few turns," modeled here at its crudest -- one giant order the moment the window
    opens, ignoring price entirely."""
    return {0: holdings} if holdings > 0 else {}


def dump_everything_agent(base_agent_fn: Callable[[dict, dict], dict], config: dict, window_turns: int = 24) -> Callable[[dict, dict], dict]:
    """The naive baseline this issue's acceptance criterion measures against, made concrete and
    runnable: at the moment the final `window_turns`-turn window opens, sell everything held of
    every product in one giant order, ignoring price entirely (`dump_everything_first_turn`).
    Same wrapper shape as `terminal_liquidation_agent`, swapping only the plan source, so a
    head-to-head between the two isolates exactly the scheduling gain."""
    episode_steps = config["episodeSteps"]
    turns_per_day = config["turnsPerDay"]
    window_start_step = episode_steps - window_turns
    dumped: dict = {"done": False}

    def agent(observation: dict, configuration: dict) -> dict:
        action = base_agent_fn(observation, configuration)
        day = observation.get("day", 0)
        hour = observation.get("hour", 0)
        step = day * turns_per_day + hour
        if step < window_start_step or dumped["done"]:
            return action
        private = observation["private"]
        max_orders = configuration.get("maxMarketOrdersPerTurn", 10)
        market_orders = [o for o in (action.get("market") or []) if not (isinstance(o, list) and o and o[0] == "SELL")]
        for item, qty in private["shed"].items():
            if len(market_orders) >= max_orders:
                break
            if qty > 0 and item in MARKET_PARAMS:
                market_orders.append(["SELL", item, qty])
        dumped["done"] = True
        new_action = dict(action)
        new_action["market"] = market_orders[:max_orders]
        return new_action

    return agent


def terminal_liquidation_agent(
    base_agent_fn: Callable[[dict, dict], dict],
    config: dict,
    window_turns: int = 24,
    opponent_schedule_fn: Callable[[str, int], dict[int, int]] | None = None,
) -> Callable[[dict, dict], dict]:
    """Wraps ANY `agent(observation, configuration) -> action` callable, overriding only its
    market SELL orders during the final `window_turns` turns of the episode with this module's
    `solve_endgame` plan -- farmer/hand actions and every other market order type pass through
    unchanged, so this composes with `search.agent.schedule_agent` (with or without issue 016's own
    `sell_plan`) or any other agent without needing to touch it.

    The endgame plan is solved once, from the LIVE observation at the moment the window opens (real
    shed holdings, real market inventory, real unlocked shops for THIS game) -- issue 016's
    `sell_plan` is an offline, analytical forecast; this is deliberately not, since by the final day
    the season's actual trajectory is fully determined and there's no reason to keep guessing at it.
    `opponent_schedule_fn(item, my_holdings) -> {turn_offset: units}` defaults to
    `mirror_opponent_schedule`.
    """
    episode_steps = config["episodeSteps"]
    turns_per_day = config["turnsPerDay"]
    window_start_step = episode_steps - window_turns
    shop_interval = config.get("townShopSellInterval", 4)
    center_interval = config.get("townCenterSellInterval", 24)
    state: dict = {"plan": None, "window_start": None}

    def agent(observation: dict, configuration: dict) -> dict:
        action = base_agent_fn(observation, configuration)
        day = observation.get("day", 0)
        hour = observation.get("hour", 0)
        step = day * turns_per_day + hour
        if step < window_start_step:
            return action

        private = observation["private"]
        if state["plan"] is None:
            market = observation["market"]
            town = observation.get("town", {}) or {}
            unlocked_shops = town.get("unlocked_shops", [])
            plan = {}
            for item, qty in private["shed"].items():
                if qty <= 0 or item not in MARKET_PARAMS:
                    continue
                opp_schedule = opponent_schedule_fn(item, qty) if opponent_schedule_fn else mirror_opponent_schedule(qty)
                plan[item] = solve_endgame(
                    item,
                    qty,
                    window_turns,
                    step,
                    market["inventory"][item],
                    unlocked_shops,
                    opponent_schedule=opp_schedule,
                    shop_interval=shop_interval,
                    center_interval=center_interval,
                )
            state["plan"] = plan
            state["window_start"] = step

        turn_offset = step - state["window_start"]
        max_orders = configuration.get("maxMarketOrdersPerTurn", 10)
        market_orders = [o for o in (action.get("market") or []) if not (isinstance(o, list) and o and o[0] == "SELL")]
        for item, per_turn_plan in state["plan"].items():
            if len(market_orders) >= max_orders:
                break
            planned_qty = per_turn_plan.get(turn_offset, 0)
            sell_qty = min(planned_qty, private["shed"].get(item, 0))  # re-checked live: production
            if sell_qty > 0:  # can still land during the window (see the docstring's own caveat)
                market_orders.append(["SELL", item, sell_qty])

        new_action = dict(action)
        new_action["market"] = market_orders[:max_orders]
        return new_action

    return agent


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
    from kaggriculture.eval.agents import wrap_agent
    from kaggriculture.eval.stats import verdict as wilson_verdict
    from kaggriculture.eval.stats import wilson_interval
    from kaggriculture.search.agent import schedule_agent
    from kaggriculture.search.schedule import build_schedule
    from kaggriculture.sim import _sim_native as native
    from kaggriculture.sim.decode import default_config_dict, episode_config

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--window-turns", type=int, default=24, help="final-window size (24 = last day)")
    parser.add_argument("--n-seeds", type=int, default=30, help="paired-seed, both-seat comparison seed count")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    config = dict(default_config_dict())
    config["episodeSteps"] = args.episode_steps
    n_days = config["episodeSteps"] // config["turnsPerDay"]

    schedule = build_schedule(config)
    committed = schedule.diagnostics["committed_units"]
    wind_down = {}
    for item in list(CROPS) + list(ANIMALS):
        wind_down[item] = wind_down_day(item, n_days, committed)

    print("Wind-down day per product (given this schedule's own committed production):")
    for item, day in wind_down.items():
        print(f"  {item:12s} -> {day}")

    ecfg = episode_config(config)
    dp_policy = wrap_agent(terminal_liquidation_agent(schedule_agent(schedule, config), config, window_turns=args.window_turns), config)
    naive_policy = wrap_agent(dump_everything_agent(schedule_agent(schedule, config), config, window_turns=args.window_turns), config)

    t0 = time.perf_counter()
    wins = losses = ties = 0
    dp_moneys, naive_moneys = [], []
    for seed in range(args.n_seeds):
        r1 = native.run_episode(dp_policy, naive_policy, ecfg, seed)
        r2 = native.run_episode(naive_policy, dp_policy, ecfg, seed)
        for dp_m, naive_m in [(r1.final_money[0], r1.final_money[1]), (r2.final_money[1], r2.final_money[0])]:
            dp_moneys.append(dp_m)
            naive_moneys.append(naive_m)
            if dp_m > naive_m:
                wins += 1
            elif dp_m < naive_m:
                losses += 1
            else:
                ties += 1
    seconds = time.perf_counter() - t0
    n = wins + losses + ties
    interval = wilson_interval(wins + 0.5 * ties, n)
    v = wilson_verdict(interval)

    print(f"\nEndgame DP vs. dump-everything, same schedule, window_turns={args.window_turns}: {wins}W-{losses}L-{ties}T over {n} games ({seconds:.1f}s)")
    print(f"  Wilson 95% CI: [{interval.lo:.3f}, {interval.hi:.3f}]  verdict: {v}")
    print(f"  mean money: DP-endgame=${sum(dp_moneys) / n:.0f}  dump-everything=${sum(naive_moneys) / n:.0f}")

    out_dir = args.out or _next_experiment_dir("terminal-liquidation")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"episode_steps": args.episode_steps, "window_turns": args.window_turns, "n_seeds": args.n_seeds}, indent=2)
    )
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "wind_down_day": wind_down,
                "seconds": seconds,
                "n_games": n,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "interval": interval.to_dict(),
                "verdict": v,
                "mean_dp_money": sum(dp_moneys) / n,
                "mean_naive_money": sum(naive_moneys) / n,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()

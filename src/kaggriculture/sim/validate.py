"""The exact-parity gate (issue 010). **This is the most important module in the repo**: until
`sweep()` passes, nothing computed by the C++ sim may be used to make a decision. Any divergence
is a bug in our port, never in the engine -- never "fix" a mismatch by adjusting the reference.

    uv run python -m kaggriculture.sim.validate                        # default sweep, ~207 episodes
    uv run python -m kaggriculture.sim.validate --seeds 5 --agents starter,random

Note for anything downstream that assumes an episode `seed` fully determines the game (issues
011/012 in particular): vendor's built-in `random_agent` seeds its own decisions from
`random.Random()` with no argument -- OS entropy, not the episode seed -- so two exports of the
same seed against `random` record two *different* action sequences. That's harmless here (each
export's trace and its own replay always agree, since both start from the same recorded actions),
but it means a `random`-involving trace is not reproducible across separate export_trace() calls,
only within one. `starter` and `pass` have no internal RNG and are fully seed-reproducible.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass, field
from typing import Any

from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.export_trace import PRODUCT_ORDER, export_trace

BUILTIN_AGENTS = ("starter", "random", "pass")


def _decode_unit_action(a: Any) -> native.UnitAction:
    if not isinstance(a, list) or not a:
        return native.UnitAction(native.Op.PASS)
    op = getattr(native.Op, a[0], native.Op.PASS) if isinstance(a[0], str) else native.Op.PASS
    item = native.Item.WHEAT
    if len(a) >= 2 and isinstance(a[1], str):
        item = getattr(native.Item, a[1], native.Item(255))
    n = 1
    if len(a) >= 3:
        try:
            n = int(a[2])
        except (TypeError, ValueError):
            n = 1
    return native.UnitAction(op, item, n)


def _decode_market_order(o: Any) -> native.MarketOrder:
    if not isinstance(o, list) or not o or not isinstance(o[0], str):
        return native.MarketOrder(native.MarketOp.NONE)
    op = getattr(native.MarketOp, o[0], native.MarketOp.NONE)
    if op in (native.MarketOp.HIRE, native.MarketOp.BUY_LAND):
        return native.MarketOrder(op, native.Item.WHEAT, 1)
    if op == native.MarketOp.NONE or len(o) < 3:
        return native.MarketOrder(native.MarketOp.NONE)
    item = getattr(native.Item, o[1], native.Item(255)) if isinstance(o[1], str) else native.Item(255)
    try:
        n = int(o[2])
    except (TypeError, ValueError):
        return native.MarketOrder(native.MarketOp.NONE)
    if n <= 0:
        return native.MarketOrder(native.MarketOp.NONE)
    return native.MarketOrder(op, item, n)


def _build_player_turn(act: dict, max_orders: int) -> native.PlayerTurn:
    turn = native.PlayerTurn()
    units = [act["farmer"], *act["hands"]]
    turn.unit_actions = [_decode_unit_action(u) for u in units]
    # vendor truncates the raw order queue to maxMarketOrdersPerTurn *before* processing
    # (`q[:max_orders]` in _process_market) -- match that at decode time.
    turn.market_orders = [_decode_market_order(o) for o in act["market"][:max_orders]]
    return turn


def _episode_config(cfg: dict) -> native.EpisodeConfig:
    episode_cfg = native.EpisodeConfig()
    episode_cfg.episode_steps = cfg["episodeSteps"]
    episode_cfg.board_size = cfg["boardSize"]
    episode_cfg.turns_per_day = cfg["turnsPerDay"]
    episode_cfg.shed_capacity = cfg["shedCapacity"]
    episode_cfg.starting_money = float(cfg["startingMoney"])
    episode_cfg.market.max_market_orders_per_turn = cfg["maxMarketOrdersPerTurn"]
    episode_cfg.market.farm_hand_cost_mult = cfg["farmHandCostMult"]
    episode_cfg.market.weed_spawn_chance = cfg["weedSpawnChance"]
    episode_cfg.market.town_shop_unlock_interval = cfg["townShopUnlockInterval"]
    episode_cfg.market.town_shop_sell_interval = cfg["townShopSellInterval"]
    episode_cfg.market.town_center_sell_interval = cfg["townCenterSellInterval"]
    return episode_cfg


@dataclass
class Divergence:
    """A bisect-friendly failure report: the first step where the sim and the real engine
    disagree, what disagreed, and enough surrounding context to reproduce it standalone."""

    step: int
    field: str  # "money" or "inventory"
    expected: list
    actual: list
    actions: list | None  # the two players' raw action dicts applied to reach this step, if any
    recent_money: list  # trace["money"][max(0, step-3) : step+1] -- both sides agreed here
    recent_inventory: list

    def __str__(self) -> str:
        lines = [
            f"DIVERGENCE at step {self.step}, field={self.field}",
            f"  expected (real engine): {self.expected}",
            f"  actual   (C++ sim):     {self.actual}",
        ]
        if self.actions is not None:
            lines.append(f"  actions applied this turn: {self.actions}")
        lines.append(f"  recent money (real engine, steps {max(0, self.step - 3)}..{self.step}): {self.recent_money}")
        lines.append(f"  recent inventory (real engine, same steps): {self.recent_inventory}")
        return "\n".join(lines)


@dataclass
class ValidationResult:
    seed: int
    agents: tuple[str, str]
    passed: bool
    n_steps_checked: int
    divergence: Divergence | None = None


def _check(trace: dict, i: int, state, market, actions=None) -> Divergence | None:
    expected_money = trace["money"][i]
    actual_money = [state.farm_money(0), state.farm_money(1)]
    expected_inv = trace["inventory"][i]
    actual_inv = [market.inventory(j) for j in range(len(PRODUCT_ORDER))]

    field_name, expected, actual = None, None, None
    if actual_money != expected_money:
        field_name, expected, actual = "money", expected_money, actual_money
    elif actual_inv != expected_inv:
        field_name, expected, actual = "inventory", expected_inv, actual_inv
    if field_name is None:
        return None

    lo = max(0, i - 3)
    return Divergence(
        step=i,
        field=field_name,
        expected=expected,
        actual=actual,
        actions=actions,
        recent_money=trace["money"][lo : i + 1],
        recent_inventory=trace["inventory"][lo : i + 1],
    )


def replay_and_diff(trace: dict) -> ValidationResult:
    """Feeds a trace's recorded actions to the C++ sim and diffs per-step money (both seats) and
    per-step market inventory (all nine resources) exactly against the recording."""
    episode_cfg = _episode_config(trace["config"])
    state = native.GameState(
        episode_cfg.board_size, episode_cfg.turns_per_day, episode_cfg.shed_capacity, episode_cfg.starting_money, trace["seed"]
    )
    market = native.MarketTownState()

    divergence = _check(trace, 0, state, market)
    if divergence is not None:
        return ValidationResult(trace["seed"], tuple(trace["agents"]), False, 0, divergence)

    max_orders = episode_cfg.market.max_market_orders_per_turn
    tape_a = native.TapePolicy([_build_player_turn(turn[0], max_orders) for turn in trace["actions"]])
    tape_b = native.TapePolicy([_build_player_turn(turn[1], max_orders) for turn in trace["actions"]])

    for i, turn_actions in enumerate(trace["actions"]):
        native.advance_turns(state, market, tape_a, tape_b, 1, episode_cfg.market)
        divergence = _check(trace, i + 1, state, market, actions=turn_actions)
        if divergence is not None:
            return ValidationResult(trace["seed"], tuple(trace["agents"]), False, i + 1, divergence)

    return ValidationResult(trace["seed"], tuple(trace["agents"]), True, len(trace["actions"]))


def validate_seed(seed: int, agents: tuple[str, str], episode_steps: int = 720) -> ValidationResult:
    trace = export_trace(seed, agents, episode_steps)
    return replay_and_diff(trace)


def sweep_pairings(agent_pool: tuple[str, ...] = BUILTIN_AGENTS) -> list[tuple[str, str]]:
    """Every ordered pair from `agent_pool`, both seat orders, no same-agent duplicate (order
    doesn't matter when both seats run the same agent)."""
    pairs = []
    for a, b in itertools.permutations(agent_pool, 2):
        pairs.append((a, b))
    for a in agent_pool:
        pairs.append((a, a))
    return pairs


def sweep(seeds: range | list[int], agent_pool: tuple[str, ...] = BUILTIN_AGENTS, episode_steps: int = 720) -> list[ValidationResult]:
    results = []
    for pairing in sweep_pairings(agent_pool):
        for seed in seeds:
            results.append(validate_seed(seed, pairing, episode_steps))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=23, help="seeds per pairing (default 23 x 9 pairings = 207 >= 200)")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--agents", default=None, help="comma-separated single pair, e.g. starter,random (default: full sweep)")
    args = parser.parse_args(argv)

    if args.agents:
        pairings = [tuple(args.agents.split(","))]
        seeds = range(max(args.seeds, 200 // max(1, len(pairings))))
    else:
        pairings = sweep_pairings()
        seeds = range(args.seeds)

    total = 0
    failed = 0
    first_failure: ValidationResult | None = None
    for pairing in pairings:
        for seed in seeds:
            result = validate_seed(seed, pairing, args.episode_steps)
            total += 1
            status = "OK" if result.passed else "FAIL"
            print(f"[{status}] seed={seed} agents={pairing} steps_checked={result.n_steps_checked}")
            if not result.passed:
                failed += 1
                if first_failure is None:
                    first_failure = result

    print(f"\n{total - failed}/{total} passed")
    if first_failure is not None:
        print("\nFirst failure:")
        print(f"  seed={first_failure.seed} agents={first_failure.agents}")
        print(first_failure.divergence)
        sys.exit(1)


if __name__ == "__main__":
    main()

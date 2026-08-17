"""Records a real kaggle_environments episode's ACTUAL emitted actions, its exact configuration,
and its per-step money/market-inventory trajectory -- the ground truth issue 010's validate.py
diffs the C++ sim against.

Two things this is careful about, both learned from a public reference notebook's own mistakes
(analysis/nb_clean/nikital7__4000x-environment-speedup-kaggriculture.py):

  - Actions are read off `env.steps[t+1][seat].action` -- what the agent ACTUALLY emitted -- not
    replayed from the agent function. `starter` repairs around weeds and reorders sell slots, so
    calling it again would not reproduce the same sequence; only the recorded action does.
  - The configuration is read off `env.configuration` (the environment's actual resolved values)
    rather than assumed. A Kaggle notebook image and a local install once disagreed on
    `startingMoney` (2000 vs 3000), which silently miscalibrated an earlier port.

    uv run python -m kaggriculture.sim.export_trace 70117 1 2 3       # writes trace files
    uv run python -m kaggriculture.sim.export_trace --agents starter,random 1 2 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any

# Order matches Item (sim.hpp) indices [0, N_PRODUCTS) -- the only ones with market inventory.
PRODUCT_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

CONFIG_KEYS = [
    "episodeSteps",
    "boardSize",
    "startingMoney",
    "maxMarketOrdersPerTurn",
    "turnsPerDay",
    "shedCapacity",
    "weedSpawnChance",
    "townShopUnlockInterval",
    "townShopSellInterval",
    "townCenterSellInterval",
    "farmHandCostMult",
]


def export_trace(seed: int, agents: tuple[str, str] = ("starter", "random"), episode_steps: int = 720) -> dict[str, Any]:
    """Runs one real episode and returns the trace dict validate.py replays against."""
    from kaggle_environments import make

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = make("kaggriculture", configuration={"episodeSteps": episode_steps, "seed": seed}, debug=False)
        env.run(list(agents))

    n = len(env.steps) - 1  # number of acting turns
    cfg = env.configuration

    actions = []
    for t in range(n):
        turn = []
        for seat in (0, 1):
            act = env.steps[t + 1][seat].action or {}
            turn.append(
                {
                    "farmer": list(act.get("farmer") or ["PASS"]),
                    "hands": [list(h) for h in (act.get("hands") or [])],
                    "market": [list(o) for o in (act.get("market") or [])],
                }
            )
        actions.append(turn)

    money = []
    inventory = []
    for s in env.steps:
        obs0 = s[0].observation
        money.append([float(obs0.farms[0]["money"]), float(obs0.farms[1]["money"])])
        inv = obs0.market["inventory"]
        inventory.append([int(inv[name]) for name in PRODUCT_ORDER])

    return {
        "seed": seed,
        "agents": list(agents),
        "config": {k: cfg[k] for k in CONFIG_KEYS},
        "actions": actions,
        "money": money,
        "inventory": inventory,
    }


def _default_out_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "notes" / "sim-traces"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("seeds", type=int, nargs="+")
    parser.add_argument("--agents", default="starter,random", help="comma-separated agent pair, e.g. starter,random")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--out-dir", type=Path, default=_default_out_dir())
    args = parser.parse_args(argv)

    agents = tuple(args.agents.split(","))
    if len(agents) != 2:
        parser.error("--agents must name exactly two agents, e.g. starter,random")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        trace = export_trace(seed, agents, args.episode_steps)
        path = args.out_dir / f"trace_{agents[0]}_{agents[1]}_{seed}.json"
        path.write_text(json.dumps(trace))
        print(f"seed {seed}: final money {trace['money'][-1]} -> {path}")


if __name__ == "__main__":
    main()

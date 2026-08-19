"""Resolves an eval-arena CLI argument ("starter", "baseline", or a path to a main.py-style
agent file) into a `native.Policy`, and wraps a real-engine-style `agent(obs, config)` callable
to run through this sim via `build_observation()` + a `CallbackPolicy` -- explicitly the slow
integration path (one Python call per turn per player), but it lets every existing agent
(vendor's own starter_agent/random_agent, our submitted baseline.py) run unmodified at this sim's
speed instead of the real engine's ~1 episode/second.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import build_player_turn

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_vendor_module():
    vendor_path = REPO_ROOT / "vendor" / "kaggriculture.py"
    spec = importlib.util.spec_from_file_location("_vendor_kaggriculture_eval", vendor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_agent_from_file(path: Path) -> Callable:
    """Execs a main.py-style agent file and returns its `agent` function (or, absent that name,
    the last callable defined -- vendor's own `get_last_callable` convention), with the file's
    directory on sys.path so bare sibling imports (a bundle's flattened model/*.py) resolve."""
    source = path.read_text()
    namespace: dict = {"__file__": str(path)}
    exec_dir = str(path.parent)
    sys.path.insert(0, exec_dir)
    try:
        exec(compile(source, str(path), "exec"), namespace)
    finally:
        sys.path.remove(exec_dir)

    if "agent" in namespace and callable(namespace["agent"]):
        return namespace["agent"]
    callables = [v for v in namespace.values() if callable(v)]
    if not callables:
        raise ValueError(f"{path}: no callable found")
    return callables[-1]


def _call_agent(agent_fn: Callable, obs: dict, config: dict):
    """Mirrors vendor's own calling convention (kaggle_environments/agent.py's `callable_agent`):
    pass (observation, configuration), truncated to the function's actual arg count, so both
    one-arg agents (vendor's starter_agent/random_agent) and two-arg ones (issue 005's
    baseline.py) work unmodified."""
    args = [obs, config]
    if hasattr(agent_fn, "__code__"):
        args = args[: agent_fn.__code__.co_argcount]
    return agent_fn(*args)


def wrap_agent(agent_fn: Callable, config: dict) -> native.Policy:
    """Wraps a real-engine-style `agent(obs, config) -> action_dict` callable as a native Policy
    via build_observation() + decode.build_player_turn(). The slow, general-purpose path --
    prefer a native TapePolicy/CallbackPolicy((state,market,player)->PlayerTurn) directly for
    anything performance-sensitive (search-derived policies, issues 013+)."""
    max_orders = config.get("maxMarketOrdersPerTurn", 10)

    def callback(state: native.GameState, market: native.MarketTownState, player: int) -> native.PlayerTurn:
        obs = native.build_observation(state, market, player)
        action = _call_agent(agent_fn, obs, config)
        return build_player_turn(action or {}, max_orders)

    return native.CallbackPolicy(callback)


def record_tape(agent_fn: Callable, config: dict, opponent: native.Policy | None = None, seed: int = 0) -> native.TapePolicy:
    """Runs `agent_fn` through the native sim once (against `opponent`, default a bare
    TapePolicy([]) i.e. "pass") and freezes every PlayerTurn it returns into a TapePolicy, indexed
    by absolute step -- the open-loop-plan path issues 013+ actually want at evaluation time:
    build once (a Python callback, ~1 call/turn, "slow" but a single 720-turn episode is still a
    matter of seconds), replay for free afterwards. See sim/policy.hpp's TapePolicy docstring."""
    max_orders = config.get("maxMarketOrdersPerTurn", 10)
    episode_steps = config["episodeSteps"]
    tape: list[native.PlayerTurn | None] = [None] * episode_steps

    def callback(state: native.GameState, market: native.MarketTownState, player: int) -> native.PlayerTurn:
        obs = native.build_observation(state, market, player)
        action = _call_agent(agent_fn, obs, config)
        turn = build_player_turn(action or {}, max_orders)
        if state.step < episode_steps:
            tape[state.step] = turn
        return turn

    from kaggriculture.sim.decode import episode_config

    episode_cfg = episode_config(config)
    native.run_episode(native.CallbackPolicy(callback), opponent or native.TapePolicy([]), episode_cfg, seed)
    return native.TapePolicy([t if t is not None else native.PlayerTurn() for t in tape])


_GREEDY_TAPE_CACHE: dict[tuple, native.TapePolicy] = {}


def resolve_policy(spec: str, config: dict) -> native.Policy:
    """Resolves a CLI agent spec into a native.Policy:
      - "pass"            -> a bare TapePolicy([]) (every turn PASS/no orders) -- exactly
                              pass_agent's behaviour, and the fast native path, not a callback.
      - "starter"/"random" -> vendor's own agent function, wrapped (see wrap_agent).
      - "baseline"         -> a freshly built submission bundle (kaggriculture.submit.build),
                              loaded from its main.py and wrapped.
      - "greedy"           -> issue 013's greedy scheduler, built once and recorded into a
                              TapePolicy (native.TapePolicy, the fast path -- see record_tape).
      - anything else      -> treated as a path to a main.py-style agent file.
    """
    if spec == "pass":
        return native.TapePolicy([])
    if spec in ("starter", "random"):
        vendor = _load_vendor_module()
        agent_fn = vendor.starter_agent if spec == "starter" else vendor.random_agent
        return wrap_agent(agent_fn, config)
    if spec == "baseline":
        from kaggriculture.submit.build import SUBMISSIONS_DIR, build_bundle

        tag = "eval-baseline"
        main_path = SUBMISSIONS_DIR / tag / "main.py"
        if not main_path.exists():
            # Guards against a real race: issue 015's CMA-ES resolves "baseline" from several
            # worker PROCESSES at once (see search.cmaes's module docstring on why processes, not
            # threads), and build_bundle() always overwrites unconditionally -- concurrent writers
            # to the same tag corrupt each other's output (observed as a truncated main.py).
            # Skipping the rebuild once the bundle already exists on disk makes resolve_policy
            # safe to call from multiple processes for the same tag.
            build_bundle(tag=tag)  # bundles baseline.py with flattened constants/price/yields/economics.py
        return wrap_agent(load_agent_from_file(main_path), config)
    if spec == "greedy":
        cache_key = (config["episodeSteps"], config["turnsPerDay"], config["startingMoney"], config["farmHandCostMult"])
        cached = _GREEDY_TAPE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        from kaggriculture.search.agent import schedule_agent
        from kaggriculture.search.schedule import build_schedule

        schedule = build_schedule(config)
        tape_policy = record_tape(schedule_agent(schedule, config), config)
        _GREEDY_TAPE_CACHE[cache_key] = tape_policy
        return tape_policy

    path = Path(spec)
    if not path.exists():
        raise ValueError(f"unknown agent {spec!r}: not a built-in name (pass/starter/random/baseline/greedy) or an existing file path")
    return wrap_agent(load_agent_from_file(path), config)

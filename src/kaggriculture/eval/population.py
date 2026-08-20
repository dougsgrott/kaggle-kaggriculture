"""Population assembly for the empirical payoff matrix (issue 018).

Two pools:

  - **Our own lineage** (`own_lineage_policies`): every generation this repo has actually
    produced, from `pass` through the full 016+017 composition -- Kaito Fukami's stated failure
    mode (734212) is optimizing only against the *latest* generation and losing to an older one
    still active on the ladder, so the population needs the retired generations, not just the
    current best.
  - **The public pool** (`build_public_pool` / `assemble_population`): `analysis/nb_clean/`
    already holds 256 wikikit-cleaned public notebooks -- vendored, public, and citable per
    CLAUDE.md. Issue 018's own scope asks to mine `kaggle/kaggriculture-episodes-index` (a Kaggle
    dataset of daily top replays) to fingerprint opening signatures; this machine has no Kaggle
    credentials configured and downloading a new dataset for a private research repo shortly
    before a deadline is exactly the kind of casual external dependency worth avoiding when a
    strictly better substitute is sitting in the repo already. Since we *have every candidate's
    source*, not just a replay of its actions, running each one ourselves and reading its own
    engine-exact state is strictly more reliable fingerprinting than reverse-engineering an
    opening from a replay would be -- see `build_public_pool`/`_fingerprint_worker`.

Not every notebook that defines `agent(observation, configuration)` is safe to execute locally:
some import `requests`/`socket`/`subprocess` or read from `/kaggle/input` (either a network call
or a missing-file crash waiting to happen). `discover_public_agent_paths` filters those out
statically before anything is ever exec'd.
"""

from __future__ import annotations

import concurrent.futures
import importlib.abc
import importlib.machinery
import json
import os
import re
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from kaggriculture.eval.agents import _call_agent, load_agent_from_file, wrap_agent
from kaggriculture.sim import _sim_native as native
from kaggriculture.sim.decode import build_player_turn, episode_config

REPO_ROOT = Path(__file__).resolve().parents[3]
NB_CLEAN_DIR = REPO_ROOT / "analysis" / "nb_clean"

_AGENT_DEF = re.compile(r"^def agent\(", re.MULTILINE)
_RISKY_IMPORT = re.compile(
    r"^\s*(import\s+(requests|socket|subprocess)\b|from\s+(requests|socket|subprocess)\s+import)"
    r"|/kaggle/input|urllib\.request|os\.system",
    re.MULTILINE,
)

CROP_NAMES = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMAL_NAMES = ("GOOSE", "COW", "SHEEP")


# --------------------------------------------------------------------------- #
# Our own lineage
# --------------------------------------------------------------------------- #


def own_lineage_policies(config: dict, threads: int = 1) -> dict[str, native.Policy]:
    """Every generation of our own agent, from `pass` through the full 016+017 composition. This
    IS the "retain older generations" input Kaito Fukami's post (734212) argues for -- not a
    single "current best".

    `greedy`/`dp016`/`lns014` are tape-recorded against a fixed opponent (`pass`) before
    evaluation, matching the precedent `eval.agents.resolve_policy`'s own `"greedy"` branch and
    `search.lns.evaluate_schedule` already set for a `Schedule`-driven plan: its actions are a
    static macro plan, not reactive to the actual opponent, so recording once and replaying is a
    faithful (and much faster) stand-in for the live callback. `cmaes015` and `current017` are
    genuinely reactive (live market price / live shed holdings respectively) and are kept as live
    `wrap_agent` callbacks, matching how `search.cmaes.main` itself builds its own final policy.
    """
    from kaggriculture.eval.agents import record_tape, resolve_policy
    from kaggriculture.search.agent import schedule_agent
    from kaggriculture.search.liquidation import terminal_liquidation_agent
    from kaggriculture.search.lns import diverse_seed_schedules, lns_search
    from kaggriculture.search.policy import policy_agent
    from kaggriculture.search.schedule import build_schedule
    from kaggriculture.search.sell_dp import solve_all_products

    n_days = config["episodeSteps"] // config["turnsPerDay"]

    policies: dict[str, native.Policy] = {
        "pass": resolve_policy("pass", config),
        "starter": resolve_policy("starter", config),
        "random": resolve_policy("random", config),
        "baseline": resolve_policy("baseline", config),
        "greedy013": resolve_policy("greedy", config),
    }

    # 014: rerun LNS with exp-005-lns's own recorded parameters (config.json in that experiment
    # dir) so this reproduces the exact incumbent that experiment reported, rather than inventing
    # a new run's worth of RNG draws.
    import random as _random

    rng = _random.Random(0)
    base_schedules = diverse_seed_schedules(config, rng, n_seeds=6, allow_4th_quadrant=True)
    lns_opponents = [resolve_policy(spec, config) for spec in ("pass", "starter", "baseline")]
    lns_result = lns_search(config, base_schedules, lns_opponents, list(range(15)), n_iters=60, seed=0, n_threads=threads)
    policies["lns014"] = record_tape(schedule_agent(lns_result.incumbent, config), config)

    # 016: greedy schedule + the sell-timing DP (discount=0.93, matching sell_dp.main's own
    # empirically-tuned default -- see its Revision section).
    schedule016 = build_schedule(config)
    sell_results = solve_all_products(schedule016.arrivals, n_days, config["episodeSteps"], config["turnsPerDay"], discount=0.93)
    sell_plan = {item: r.plan for item, r in sell_results.items()}
    dp016_agent = schedule_agent(schedule016, config, sell_plan=sell_plan)
    policies["dp016"] = record_tape(dp016_agent, config)

    # 015: CMA-ES-tuned reactive policy -- load the trained parameters exp-006-cmaes already
    # found rather than re-running CMA-ES (which is itself the multi-minute search this issue
    # would otherwise have to redo just to get an opponent).
    cmaes_result_path = REPO_ROOT / "experiments" / "exp-006-cmaes" / "result.json"
    best_params = json.loads(cmaes_result_path.read_text())["best_params"]
    policies["cmaes015"] = wrap_agent(policy_agent(best_params, config), config)

    # 017: the full composition -- 016's schedule+sell-plan, wrapped by the terminal-liquidation
    # endgame layer. This is "our current best", but it's just one more population member here,
    # not privileged over the earlier generations.
    current_agent = terminal_liquidation_agent(dp016_agent, config)
    policies["current017"] = wrap_agent(current_agent, config)

    return policies


# --------------------------------------------------------------------------- #
# The public pool
# --------------------------------------------------------------------------- #


def discover_public_agent_paths() -> list[Path]:
    """Every `analysis/nb_clean/*.py` file that defines a top-level `agent(observation,
    configuration)` and does not import anything that could touch the network or a missing
    dataset path."""
    paths = []
    for path in sorted(NB_CLEAN_DIR.glob("*.py")):
        text = path.read_text(errors="replace")
        if _AGENT_DEF.search(text) and not _RISKY_IMPORT.search(text):
            paths.append(path)
    return paths


_STUB_BLOCKLIST_TOP = frozenset({"IPython", "matplotlib", "seaborn", "plotly"})


class _Stub(types.ModuleType):
    """Absorbs any attribute access or call and returns another absorbing stub -- stands in for
    a notebook's own plotting/display cells (`IPython.display.display(...)`, `plt.plot(...)`)
    that are never actually reached by a turn-by-turn `agent()` call, so a display library this
    sandbox doesn't have installed (or does, and would otherwise try to touch a real GUI backend)
    shouldn't be why a candidate gets excluded from the population."""

    def __getattr__(self, name: str):
        if name == "subplots":
            # `fig, ax = plt.subplots(...)` is common enough in these notebooks' own plotting
            # cells to special-case: a generic stub can't support unpacking to an arity it
            # doesn't know ahead of time, but this specific 2-tuple shape is the overwhelmingly
            # common call pattern.
            return lambda *_a, **_k: (_Stub("fig"), _Stub("ax"))
        return _Stub("stub")

    def __getitem__(self, _key):
        return _Stub("stub")

    def __call__(self, *_args, **_kwargs):
        return _Stub("stub")

    def __bool__(self):
        return True

    def __len__(self):
        return 0

    def __hash__(self):
        return id(self)

    def __eq__(self, _other):
        return False


# A notebook's plotting cell (`bar.get_height() / 2`, `-value`, `x < threshold`, ...) can run any
# arithmetic/comparison operator on a stub -- bind the whole closed set to "absorb and return
# another stub" rather than enumerating every method name a chart-annotation snippet might call.
for _op in (
    "add radd sub rsub mul rmul truediv rtruediv floordiv rfloordiv mod rmod pow rpow "
    "neg pos abs lt le gt ge"
).split():
    setattr(_Stub, f"__{_op}__", lambda self, *_a, **_k: _Stub("stub"))


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        stub = _Stub(spec.name)
        stub.__path__ = []
        return stub

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    """Intercepts an import of any module under `_STUB_BLOCKLIST_TOP`, at any depth
    (`matplotlib.patches`, `IPython.core.display`, ...) -- a `sys.modules` entry per exact
    dotted name would need every submodule a notebook might import enumerated up front; a
    meta-path finder handles arbitrary depth without that."""

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in _STUB_BLOCKLIST_TOP:
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


def _load_agent_isolated(path: Path) -> Callable:
    """`load_agent_from_file`, but with the process's cwd pointed at a throwaway temp directory
    for the duration of the load. Several candidates write files as a module-level side effect of
    their own self-packaging/validation cells (observed: `main.py`, `submission.py`,
    `submission.tar.gz`, a `kaggle_working/` directory) -- harmless to the candidate's own
    `agent()` logic, but without this they'd land in whatever the real cwd happens to be (this
    repo's root, when this runs from a dev shell), which is not this population's business to
    litter. The temp directory and anything written into it are discarded once the load
    finishes."""
    import shutil
    import tempfile

    original_cwd = Path.cwd()
    scratch = Path(tempfile.mkdtemp(prefix="kaggriculture-nb-load-"))
    try:
        os.chdir(scratch)
        return load_agent_from_file(path)
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(scratch, ignore_errors=True)


def _stub_notebook_only_modules() -> None:
    """Installs `_StubFinder` at the front of `sys.meta_path` (idempotent -- safe to call from
    every worker process AND the main process, repeatedly) and clears any real
    `_STUB_BLOCKLIST_TOP` package already imported, so every subsequent import of it (direct or
    nested) resolves to a stub instead. `load_public_policy` calls this too, not just
    `_fingerprint_worker`: a candidate that only survived the smoke test because plotting calls
    got stubbed out must see the exact same stubs when it's loaded again for real (in the main
    process, for the actual payoff-matrix games) -- otherwise it would smoke-test clean and then
    crash the moment it's actually used."""
    if any(isinstance(f, _StubFinder) for f in sys.meta_path):
        return
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_BLOCKLIST_TOP:
            del sys.modules[name]
    sys.meta_path.insert(0, _StubFinder())


def _fingerprint_worker(path_str: str, config: dict, snapshot_turn: int) -> dict | None:
    """Runs one candidate solo against `pass` for a full episode, snapshotting engine-exact state
    at `snapshot_turn` (default turn 47 == end of day 1, "first 48 turns" per the issue's own
    framing). Returns `None` on any failure (import error, runtime crash, illegal-action
    rejection) -- that candidate is simply excluded, the same way a smoke test would exclude it.
    Runs in its own process (see `build_public_pool`) so one candidate's crash, hang, or global
    state can't affect any other candidate's evaluation.
    """
    try:
        _stub_notebook_only_modules()
        agent_fn = _load_agent_isolated(Path(path_str))
        ecfg = episode_config(config)
        max_orders = config.get("maxMarketOrdersPerTurn", 10)
        captured: dict = {}

        def callback(state: native.GameState, market: native.MarketTownState, player: int) -> native.PlayerTurn:
            obs = native.build_observation(state, market, player)
            if state.step == snapshot_turn and player == 0:
                captured["obs"] = obs
            action = _call_agent(agent_fn, obs, config)
            return build_player_turn(action or {}, max_orders)

        native.run_episode(native.CallbackPolicy(callback), native.TapePolicy([]), ecfg, seed=0)
        if "obs" not in captured:
            return None
        return {"obs": captured["obs"], "starting_money": config["startingMoney"]}
    except Exception:
        return None


def build_public_pool(config: dict, snapshot_turn: int = 47, timeout: float = 20.0, max_workers: int = 8) -> dict[str, dict]:
    """Fingerprints every candidate `discover_public_agent_paths` finds, in parallel worker
    processes with a per-candidate timeout (guards against a notebook that hangs, e.g. one doing
    unbounded retraining work per turn). Returns `{name: {"obs": ..., "starting_money": ...}}`
    for survivors only.

    A handful of candidates abort their whole OS process (observed: a `std::runtime_error`
    `terminate()` from deep inside a candidate's own transitive import chain -- not a Python
    exception, so `_fingerprint_worker`'s own `try/except` can't catch it). `ProcessPoolExecutor`
    marks its *entire* pool broken the moment any one worker dies this way, which would silently
    fail every other still-pending candidate too, not just the one that actually crashed -- so
    this retries with a fresh pool for whatever candidates never got a result, up to
    `max_retries` times, rather than letting one bad candidate poison the whole run."""
    paths = discover_public_agent_paths()
    remaining = {str(p): p for p in paths}
    fingerprints: dict[str, dict] = {}
    max_retries = 5

    for _ in range(max_retries):
        if not remaining:
            break
        completed = set()
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_fingerprint_worker, path_str, config, snapshot_turn): path_str for path_str in remaining}
                for future, path_str in futures.items():
                    try:
                        fp = future.result(timeout=timeout)
                    except concurrent.futures.BrokenExecutor:
                        continue  # never got a fair run -- leave it in `remaining` for retry, not a real failure
                    except Exception:
                        fp = None
                    completed.add(path_str)
                    if fp is not None:
                        fingerprints[remaining[path_str].stem] = fp
        except concurrent.futures.BrokenExecutor:
            pass  # the pool itself broke before it could even hand back futures -- retry everything left
        remaining = {k: v for k, v in remaining.items() if k not in completed}

    return fingerprints


def _feature_vector(fingerprint: dict) -> tuple[float, ...]:
    """A normalized opening-signature vector from one engine-exact snapshot: crop mix (which
    tiles got planted with what), animal mix, hand count, quadrants unlocked, and how much of the
    starting money got spent by the snapshot turn -- the "first 48 turns" fingerprint the issue
    asks for, read directly off engine state instead of reverse-engineered from a replay."""
    obs = fingerprint["obs"]
    farm = obs["farms"][0]
    tiles = farm["tiles"]
    crop_counts = dict.fromkeys(CROP_NAMES, 0)
    animal_counts = dict.fromkeys(ANIMAL_NAMES, 0)
    for row in tiles:
        for cell in row:
            if not isinstance(cell, dict):
                continue
            if cell.get("kind") == "PLANT" and cell.get("crop") in crop_counts:
                crop_counts[cell["crop"]] += 1
            elif cell.get("kind") in ("COOP", "PASTURE") and cell.get("animal") in animal_counts:
                animal_counts[cell["animal"]] += 1
    n_planted = sum(crop_counts.values()) + sum(animal_counts.values())
    crop_frac = [crop_counts[c] / n_planted if n_planted else 0.0 for c in CROP_NAMES]
    animal_frac = [animal_counts[a] / n_planted if n_planted else 0.0 for a in ANIMAL_NAMES]
    n_hands = len(farm["hands"]) / 8.0  # MAX_HANDS across the repo's own agents
    n_quadrants = len(farm["unlocked_quadrants"]) / 4.0
    starting_money = fingerprint["starting_money"]
    money_spent = max(0.0, (starting_money - farm["money"])) / starting_money if starting_money else 0.0
    return (*crop_frac, *animal_frac, n_hands, n_quadrants, money_spent)


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def cluster_by_opening(fingerprints: dict[str, dict], threshold: float = 0.25) -> list[list[str]]:
    """Greedy single-link clustering on the opening feature vector: two candidates land in the
    same cluster iff some chain of pairwise distances under `threshold` connects them. The exact
    threshold doesn't matter much here -- the point (per the issue's own framing) is collapsing
    near-duplicate lineages ("33 notebooks share the same opening verbatim") down to one
    representative each, not a precise taxonomy."""
    names = list(fingerprints)
    vectors = {n: _feature_vector(fingerprints[n]) for n in names}
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if _distance(vectors[a], vectors[b]) < threshold:
                union(a, b)

    clusters: dict[str, list[str]] = {}
    for n in names:
        clusters.setdefault(find(n), []).append(n)
    return list(clusters.values())


def select_cluster_representatives(clusters: list[list[str]], fingerprints: dict[str, dict]) -> list[str]:
    """One representative per cluster: the medoid (minimum total distance to the rest of its own
    cluster) -- more central than an arbitrary first-alphabetical pick, and doesn't need any
    external popularity signal (vote counts aren't part of the fingerprint)."""
    reps = []
    for cluster in clusters:
        if len(cluster) == 1:
            reps.append(cluster[0])
            continue
        vectors = {n: _feature_vector(fingerprints[n]) for n in cluster}
        best, best_cost = None, float("inf")
        for n in cluster:
            cost = sum(_distance(vectors[n], vectors[m]) for m in cluster if m != n)
            if cost < best_cost:
                best, best_cost = n, cost
        reps.append(best)
    return reps


def load_public_policy(name: str, config: dict) -> native.Policy:
    _stub_notebook_only_modules()  # must match what the file saw when it survived the smoke test
    path = NB_CLEAN_DIR / f"{name}.py"
    return wrap_agent(_load_agent_isolated(path), config)


@dataclass
class PopulationDiagnostics:
    n_candidates_discovered: int
    n_candidates_survived_smoke_test: int
    n_clusters: int
    clusters: list[list[str]] = field(default_factory=list)
    representatives: list[str] = field(default_factory=list)


def assemble_population(
    config: dict, n_public_target: int = 10, cluster_threshold: float = 0.25, threads: int = 1
) -> tuple[dict[str, native.Policy], PopulationDiagnostics]:
    """Own lineage (unconditional) plus up to `n_public_target` public-pool representatives, one
    per opening-signature cluster, largest clusters first (a big cluster is a well-trodden
    lineage -- exactly what discussion 733924 says defines the current meta ceiling, so it earns
    a population slot before a singleton)."""
    own = own_lineage_policies(config, threads=threads)

    fingerprints = build_public_pool(config)
    clusters = cluster_by_opening(fingerprints, threshold=cluster_threshold)
    clusters.sort(key=len, reverse=True)
    reps = select_cluster_representatives(clusters, fingerprints)

    # vendor/public-soil's own reference (issue 002) always gets a slot if discovered, regardless
    # of which cluster medoid it lost to -- it's the named ceiling discussion 733924 measured
    # (ELO 3117-3131), the thing the rest of this issue's analysis is supposed to answer "what
    # beats it" for.
    reference_stem = "boatlee__v16-rc5-high-score-8c-4s-premium-market-lead"
    ordered_reps = list(reps)
    if reference_stem in fingerprints and reference_stem not in ordered_reps:
        ordered_reps.insert(0, reference_stem)
    elif reference_stem in ordered_reps:
        ordered_reps.remove(reference_stem)
        ordered_reps.insert(0, reference_stem)
    chosen = ordered_reps[:n_public_target]

    public_policies = {name: load_public_policy(name, config) for name in chosen}

    diagnostics = PopulationDiagnostics(
        n_candidates_discovered=len(discover_public_agent_paths()),
        n_candidates_survived_smoke_test=len(fingerprints),
        n_clusters=len(clusters),
        clusters=clusters,
        representatives=chosen,
    )
    return {**own, **public_policies}, diagnostics

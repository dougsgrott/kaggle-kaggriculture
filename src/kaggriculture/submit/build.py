"""Package an agent module into a Kaggle-submittable artifact.

Single-file agents produce submissions/<tag>/main.py. Passing extra_paths bundles
main.py with those files into submissions/<tag>/submission.tar.gz instead — Kaggle
unpacks a tar submission flat into /kaggle_simulations/agent/, so sibling modules
import by bare name (`import helper`), never through the `kaggriculture` package.
"""

import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AGENT = REPO_ROOT / "src" / "kaggriculture" / "agents" / "current.py"
MODEL_DIR = REPO_ROOT / "src" / "kaggriculture" / "model"
SEARCH_DIR = REPO_ROOT / "src" / "kaggriculture" / "search"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

_FORBIDDEN_IMPORT = re.compile(r"^\s*(from\s+kaggriculture|import\s+kaggriculture)\b", re.MULTILINE)
_MODEL_IMPORT = re.compile(r"from kaggriculture\.model(?:\.(\w+))? import")
_SEARCH_IMPORT = re.compile(r"from kaggriculture\.search\.(\w+) import")


class SelfContainmentError(RuntimeError):
    """Raised when a file meant to run outside the repo still imports the repo package."""


def default_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _check_self_contained(path: Path) -> None:
    text = path.read_text()
    match = _FORBIDDEN_IMPORT.search(text)
    if match:
        raise SelfContainmentError(
            f"{path} imports the kaggriculture package ({match.group(0)!r}); "
            "/kaggle_simulations/agent/ won't have it installed."
        )


def _flatten_model_imports(text: str) -> str:
    """`from kaggriculture.model.X import y` -> `from X import y`: what a bundle's sibling
    model modules need once they're unpacked flat into /kaggle_simulations/agent/."""
    return _MODEL_IMPORT.sub(lambda m: f"from {m.group(1)} import", text)


def _flatten_search_imports(text: str) -> str:
    """Same idea as `_flatten_model_imports`, for `from kaggriculture.search.X import y` ->
    `from X import y` -- what a bundled `search/*.py` module's own sibling search-module imports
    need once unpacked flat."""
    return _SEARCH_IMPORT.sub(lambda m: f"from {m.group(1)} import", _flatten_model_imports(text))


def build_bundle(
    tag: str | None = None,
    agent_path: Path = REPO_ROOT / "src" / "kaggriculture" / "agents" / "baseline.py",
    model_modules: tuple[str, ...] = ("constants", "price", "yields", "economics"),
) -> Path:
    """Package an agent that imports the *bundled, flattened* model modules by bare name
    (`import constants`, `from price import price`, ...) into submissions/<tag>/submission.tar.gz.

    `constants` is special-cased: the real model/constants.py loads vendor/kaggriculture.py from
    an absolute repo path that won't exist on Kaggle's grader, so it's replaced with a frozen,
    data-only snapshot (kaggriculture.model.freeze) instead of being flattened like the others.
    """
    from kaggriculture.model.freeze import frozen_constants_source

    tag = tag or default_tag()
    out_dir = SUBMISSIONS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    main_py = out_dir / "main.py"
    main_py.write_text(_flatten_model_imports(agent_path.read_text()))
    written.append(main_py)

    for name in model_modules:
        content = frozen_constants_source() if name == "constants" else _flatten_model_imports((MODEL_DIR / f"{name}.py").read_text())
        path = out_dir / f"{name}.py"
        path.write_text(content)
        written.append(path)

    for path in written:
        _check_self_contained(path)

    archive = out_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in written:
            tar.add(path, arcname=path.name)
    return archive


def build(tag: str | None = None, agent_path: Path = DEFAULT_AGENT, extra_paths: tuple[Path, ...] = ()) -> Path:
    """Write submissions/<tag>/main.py (and submission.tar.gz if extra_paths given).

    Returns the artifact path that `kaggle competitions submit -f` should point at.
    """
    tag = tag or default_tag()
    _check_self_contained(agent_path)
    for extra in extra_paths:
        _check_self_contained(extra)

    out_dir = SUBMISSIONS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    main_py = out_dir / "main.py"
    main_py.write_text(agent_path.read_text())

    if not extra_paths:
        return main_py

    for extra in extra_paths:
        (out_dir / extra.name).write_text(extra.read_text())

    archive = out_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(main_py, arcname="main.py")
        for extra in extra_paths:
            tar.add(out_dir / extra.name, arcname=extra.name)
    return archive


def _serialize_schedule(schedule) -> str:
    """A `Schedule` as Python source, not JSON -- tuple tile-position keys are directly valid
    dict-literal keys, so `repr()` of each field is exact, safe, round-trippable source, not a
    format needing a translation step."""
    return (
        "Schedule(\n"
        f"    tile_role={schedule.tile_role!r},\n"
        f"    tile_kind={schedule.tile_kind!r},\n"
        f"    tile_first_assigned_day={schedule.tile_first_assigned_day!r},\n"
        f"    crew_size_by_day={schedule.crew_size_by_day!r},\n"
        f"    land_days={schedule.land_days!r},\n"
        f"    n_days={schedule.n_days!r},\n"
        ")\n"
    )


def build_composed_bundle(tag: str | None = None, n_threads: int = 8, schedule=None) -> Path:
    """Freezes an LNS + sell-DP + terminal-liquidation composition (issue 028; the larger-budget
    version is issue 029) into a self-contained Kaggle submission bundle.

    The LNS-optimized `Schedule` and its measured sell plan are computed once, here, offline
    (`_rebuild_lns_incumbent`/`measure_arrivals`/`solve_all_products` -- the same ~5-minute search
    `search.composition`'s own CLI runs, unless `schedule` is passed in already-built) and baked
    into `frozen_plan.py` as literal data -- Kaggle's per-turn time budget has no room for
    re-running that search live. `schedule_agent`'s and `terminal_liquidation_agent`'s own
    per-turn DECISION code, by contrast, has to run live (the endgame DP genuinely needs each
    game's own actual final-day holdings, not a precomputed guess) -- bundled verbatim via the
    same import-rewriting `build_bundle` already uses for the model modules, not hand-transcribed,
    so there's no risk of a manual-port bug diverging from the already-tested dev-tree source.

    `schedule`, if given, skips `_rebuild_lns_incumbent`'s own exp-005-parameter search and uses
    this `Schedule` directly -- for a different (e.g. larger-budget) LNS run whose incumbent was
    already built elsewhere, so this doesn't have to re-run a potentially very expensive search
    just to freeze its result.
    """
    from kaggriculture.model.freeze import frozen_constants_source
    from kaggriculture.search.composition import _rebuild_lns_incumbent, measure_arrivals
    from kaggriculture.search.sell_dp import solve_all_products
    from kaggriculture.sim.decode import default_config_dict

    tag = tag or default_tag()
    out_dir = SUBMISSIONS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    config = dict(default_config_dict())
    n_days = config["episodeSteps"] // config["turnsPerDay"]
    incumbent = schedule if schedule is not None else _rebuild_lns_incumbent(config, threads=n_threads)
    arrivals = measure_arrivals(incumbent, config)
    sell_results = solve_all_products(arrivals, n_days, config["episodeSteps"], config["turnsPerDay"], discount=0.93)
    sell_plan = {item: r.plan for item, r in sell_results.items()}

    written = []
    for name in ("constants", "price", "yields", "economics", "town", "regimes"):
        content = frozen_constants_source() if name == "constants" else _flatten_model_imports((MODEL_DIR / f"{name}.py").read_text())
        path = out_dir / f"{name}.py"
        path.write_text(content)
        written.append(path)

    for name in ("schedule", "agent", "liquidation"):
        source = (SEARCH_DIR / f"{name}.py").read_text()
        # Drop the module's own CLI `main()` (and its `if __name__ == "__main__":` block, always
        # immediately after) -- never called by the frozen bundle, and its own local imports
        # (`kaggriculture.eval.*`, `kaggriculture.sim.*`) aren't part of this bundle's flattened
        # set. This must be a suffix cut, not applied to any earlier marker: everything
        # schedule_agent/terminal_liquidation_agent actually need is defined BEFORE `main(`.
        source = re.split(r"\ndef main\(", source, maxsplit=1)[0]
        # Separately, delete (not truncate at) the `REPO_ROOT = Path(__file__)...`/
        # `EXPERIMENTS_DIR = ...` pair some search modules define near their top for their own
        # CLI's experiment-dir bookkeeping -- a module-level statement that runs at IMPORT time,
        # not just when called. On Kaggle's grader `__file__` is `/kaggle_simulations/agent/
        # liquidation.py`, too shallow for `.parents[3]`, raising IndexError the instant the
        # bundle is imported (confirmed directly: this is exactly what broke the first submission
        # attempt). Local testing from this repo's own path never caught it -- every local path
        # this repo runs from has enough parents for `.parents[3]` to resolve harmlessly.
        source = re.sub(r"\nREPO_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[\d+\]\nEXPERIMENTS_DIR = REPO_ROOT / \"experiments\"\n", "\n", source)
        content = _flatten_search_imports(source)
        path = out_dir / f"{name}.py"
        path.write_text(content)
        written.append(path)

    frozen_plan_py = out_dir / "frozen_plan.py"
    frozen_plan_py.write_text(
        "# Frozen offline: issue 028's LNS-optimized Schedule and its measured sell plan.\n"
        "# Regenerate with kaggriculture.submit.build.build_composed_bundle, not by hand.\n"
        "from schedule import Schedule\n\n"
        f"SCHEDULE = {_serialize_schedule(incumbent)}\n"
        f"SELL_PLAN = {sell_plan!r}\n\n"
        f"CONFIG = {config!r}\n"
    )
    written.append(frozen_plan_py)

    main_py = out_dir / "main.py"
    main_py.write_text(
        "# Issue 028: 014's LNS-optimized production plan + 016's sell-timing DP + 017's\n"
        "# terminal-liquidation endgame layer. Built by kaggriculture.submit.build.build_composed_bundle.\n"
        "from agent import schedule_agent\n"
        "from liquidation import terminal_liquidation_agent\n"
        "from frozen_plan import CONFIG, SCHEDULE, SELL_PLAN\n\n"
        "agent = terminal_liquidation_agent(schedule_agent(SCHEDULE, CONFIG, sell_plan=SELL_PLAN), CONFIG)\n"
    )
    written.append(main_py)

    for path in written:
        _check_self_contained(path)

    archive = out_dir / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in written:
            tar.add(path, arcname=path.name)
    return archive

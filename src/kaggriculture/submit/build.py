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
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

_FORBIDDEN_IMPORT = re.compile(r"^\s*(from\s+kaggriculture|import\s+kaggriculture)\b", re.MULTILINE)


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

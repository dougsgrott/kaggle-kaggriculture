"""Thin wrappers around the `kaggle competitions` CLI for the submit-and-watch loop.

Each function shells out to the Kaggle CLI rather than re-implementing its API client;
it already handles auth (~/.kaggle/access_token) and matches whatever the CLI version
pinned in pyproject.toml does.
"""

import subprocess
from pathlib import Path

COMPETITION = "kaggriculture"


def _run(args: list[str]) -> str:
    result = subprocess.run(["kaggle", *args], capture_output=True, text=True, check=True)
    return result.stdout


def submissions() -> str:
    return _run(["competitions", "submissions", COMPETITION])


def episodes(submission_id: str) -> str:
    return _run(["competitions", "episodes", submission_id])


def download_replay(episode_id: str, out_dir: Path) -> str:
    """Downloads the replay into out_dir; returns the CLI's own status output.

    The CLI doesn't document a stable output filename, so read its stdout (or list
    out_dir) rather than assuming a name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return _run(["competitions", "replay", episode_id, "-p", str(out_dir)])


def download_logs(episode_id: str, agent_index: int, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    return _run(["competitions", "logs", episode_id, str(agent_index), "-p", str(out_dir)])

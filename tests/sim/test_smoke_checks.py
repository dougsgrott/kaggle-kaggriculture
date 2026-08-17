"""Runs the C++ sim's own standalone smoke checks (issues 007/008) through pytest, so a
regression here shows up in `uv run pytest` instead of only in `make -C src/kaggriculture/sim
all`. These are sanity checks, not the correctness claim -- issue 010's parity gate against the
real engine is.
"""

import subprocess
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[2] / "src" / "kaggriculture" / "sim"


def _compile_and_run(tmp_path, name: str, sources: list[str]) -> str:
    binary = tmp_path / name
    subprocess.run(
        ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-I", str(SIM_DIR), "-o", str(binary)]
        + [str(SIM_DIR / s) for s in sources],
        check=True,
    )
    return subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout


def test_mechanical_smoke_check_passes(tmp_path):
    out = _compile_and_run(tmp_path, "smoke", ["sim.cpp", "smoke_main.cpp"])
    assert "ALL SMOKE CHECKS PASSED" in out


def test_market_smoke_check_passes(tmp_path):
    out = _compile_and_run(tmp_path, "market_smoke", ["sim.cpp", "pyrandom.cpp", "market.cpp", "market_smoke_main.cpp"])
    assert "ALL MARKET SMOKE CHECKS PASSED" in out

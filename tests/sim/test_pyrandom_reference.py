"""Cross-checks kaggriculture.sim's PyRandom (a hand-ported CPython MT19937) against a live
`random.Random` run -- struct.pack hex of each `random()` double, and `choice()` index draws,
for the same seeds and (separately) the exact daily reseed formula vendor/kaggriculture.py uses:
`random.Random((episode_seed * 1_000_003) ^ day)`. Byte-for-byte agreement here is the whole
point of issue 008's RNG work: an approximate port would make every downstream search result
diverge from what the real engine (and therefore the scored replay) actually does.
"""

import random
import struct
import subprocess
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[2] / "src" / "kaggriculture" / "sim"
SEEDS = [0, 1, 42, 123456789, 4294967295, 4294967296, 9999999999]
EPISODE_SEEDS = [0, 1, 70117, 2147483647]


def _build(tmp_path) -> Path:
    binary = tmp_path / "pyrandom_dump"
    subprocess.run(
        ["g++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-I", str(SIM_DIR), "-o", str(binary),
         str(SIM_DIR / "pyrandom.cpp"), str(SIM_DIR / "pyrandom_dump.cpp")],
        check=True,
    )
    return binary


def _run(binary: Path) -> list[str]:
    return subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.splitlines()


def test_random_matches_cpython(tmp_path):
    lines = {line.split()[0:2][1]: line.split()[2:] for line in _run(_build(tmp_path)) if line.startswith("random ")}

    for seed in SEEDS:
        r = random.Random(seed)
        expected = [struct.pack(">d", r.random()).hex() for _ in range(8)]
        got = lines[f"seed={seed}"]
        assert got == expected, f"seed={seed}"


def test_choice_index_matches_cpython(tmp_path):
    lines = {line.split()[0:2][1]: line.split()[2:] for line in _run(_build(tmp_path)) if line.startswith("choice8 ")}

    for seed in SEEDS:
        r = random.Random(seed)
        expected = [str(r.choice(range(8))) for _ in range(16)]
        got = lines[f"seed={seed}"]
        assert got == expected, f"seed={seed}"


def test_vendor_daily_reseed_formula_matches_cpython(tmp_path):
    binary = _build(tmp_path)
    for line in _run(binary):
        if not line.startswith("daily "):
            continue
        parts = line.split()
        episode_seed = int(parts[1].split("=")[1])
        day = int(parts[2].split("=")[1])
        hexes = parts[3:7]
        choice = int(parts[7].split("=")[1])

        r = random.Random((episode_seed * 1_000_003) ^ day)
        expected_hexes = [struct.pack(">d", r.random()).hex() for _ in range(4)]
        expected_choice = r.choice(range(8))

        assert hexes == expected_hexes, (episode_seed, day)
        assert choice == expected_choice, (episode_seed, day)

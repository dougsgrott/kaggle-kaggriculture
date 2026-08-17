"""Builds src/kaggriculture/sim/_sim_native, the pybind11 extension (issue 009).

No cmake in this environment, so this uses pybind11's setup_helpers (plain setuptools + g++)
rather than the more common cmake-based pybind11 example layout. `uv sync` / `uv build` invoke
this automatically via the standard PEP 517 build_ext hook -- see pyproject.toml's [build-system].
"""

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

SIM_DIR = Path("src/kaggriculture/sim")
SOURCES = ["bindings.cpp", "sim.cpp", "market.cpp", "pyrandom.cpp", "episode.cpp"]

ext_modules = [
    Pybind11Extension(
        "kaggriculture.sim._sim_native",
        [str(SIM_DIR / f) for f in SOURCES],
        include_dirs=[str(SIM_DIR)],
        cxx_std=20,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})

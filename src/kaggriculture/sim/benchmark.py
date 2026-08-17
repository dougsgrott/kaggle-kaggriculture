"""Benchmark for the pybind11 sim bindings (issue 009): episodes/second single-thread and at
`--threads` (default: os.cpu_count()), plus the speedup ratio over the real engine's own
throughput (measured fresh each run, not hand-typed, so this stays honest as the machine or the
engine version changes). Also reports peak RSS for the parallel batch, since issue 009's
acceptance criterion caps it at 4 GB on this machine's 7.6 GB.

    uv run python -m kaggriculture.sim.benchmark
    uv run python -m kaggriculture.sim.benchmark --episodes 20000 --threads 20
"""

import argparse
import contextlib
import io
import os
import resource
import time

from kaggriculture.sim import _sim_native as native


def _bench_native(n_episodes: int, n_threads: int) -> float:
    tape = native.TapePolicy([])
    cfg = native.EpisodeConfig()
    seeds = list(range(n_episodes))
    pairs = [(tape, tape) for _ in seeds]
    configs = [cfg] * len(seeds)

    t0 = time.perf_counter()
    native.run_batch(pairs, configs, seeds, n_threads)
    elapsed = time.perf_counter() - t0
    return n_episodes / elapsed


def _bench_real_engine(n_episodes: int) -> float:
    from kaggle_environments import make

    with contextlib.redirect_stdout(io.StringIO()):
        make("kaggriculture", configuration={"episodeSteps": 720, "seed": 0}, debug=False).run(["pass", "pass"])  # warm up

    t0 = time.perf_counter()
    for i in range(n_episodes):
        with contextlib.redirect_stdout(io.StringIO()):
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": i}, debug=False)
            env.run(["pass", "pass"])
    elapsed = time.perf_counter() - t0
    return n_episodes / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000, help="episodes for the native single/multi-thread runs")
    parser.add_argument("--real-episodes", type=int, default=5, help="episodes for the real-engine baseline (slow: ~1/sec)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--skip-real-engine", action="store_true", help="skip the slow real-engine baseline")
    args = parser.parse_args()

    single = _bench_native(args.episodes, 1)
    print(f"native, 1 thread:   {single:9.1f} eps/sec  ({args.episodes} episodes)")

    multi = _bench_native(args.episodes, args.threads)
    print(f"native, {args.threads} threads: {multi:9.1f} eps/sec  ({args.episodes} episodes)")
    print(f"  scaling: {multi / single:.2f}x over 1 thread with {args.threads} threads")

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  peak RSS so far: {peak_rss_mb:.0f} MB (ceiling: 4096 MB)")

    if not args.skip_real_engine:
        real = _bench_real_engine(args.real_episodes)
        print(f"real engine, 1 thread: {real:9.3f} eps/sec  ({args.real_episodes} episodes)")
        print(f"  speedup: {single / real:,.0f}x (1 thread), {multi / real:,.0f}x ({args.threads} threads)")
        print(f"  floor is 1,000x; public reference claims ~4,000x")


if __name__ == "__main__":
    main()

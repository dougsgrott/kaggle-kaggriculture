"""Regression smoke test for the baseline-v1 agent (issue 005).

A small, fast paired-seed/both-seat sample — enough to catch a regression quickly. The
acceptance-level win-rate claim (>=90% vs starter, >=99% vs random) was measured separately
over a larger sample; see submissions/baseline-v1/README.md.
"""

from kaggle_environments import make

from kaggriculture.submit.build import build_bundle

N_SEEDS = 4


def _play(main_py: str, opponent: str, seed: int, baseline_seat: int) -> tuple[float, float]:
    agents = [main_py, opponent] if baseline_seat == 0 else [opponent, main_py]
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
    env.run(agents)
    farms = env.steps[-1][0].observation["farms"]
    return farms[baseline_seat]["money"], farms[1 - baseline_seat]["money"]


def _bundle(tmp_path, monkeypatch) -> str:
    import kaggriculture.submit.build as build_mod

    monkeypatch.setattr(build_mod, "SUBMISSIONS_DIR", tmp_path)
    build_bundle(tag="baseline-test")
    return str(tmp_path / "baseline-test" / "main.py")


def test_bundle_runs_cleanly_with_no_warnings_or_errors(tmp_path, monkeypatch, capsys):
    main_py = _bundle(tmp_path, monkeypatch)
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 0}, debug=True)
    env.run([main_py, "random"])

    assert env.steps[-1][0]["status"] == "DONE"
    assert env.steps[-1][1]["status"] == "DONE"
    out = capsys.readouterr()
    for stream in (out.out, out.err):
        assert "WARNING" not in stream
        assert "Traceback" not in stream


def test_beats_starter_over_paired_seeds_both_seats(tmp_path, monkeypatch):
    main_py = _bundle(tmp_path, monkeypatch)
    for seed in range(N_SEEDS):
        for seat in (0, 1):
            mine, theirs = _play(main_py, "starter", seed, seat)
            assert mine > theirs, (seed, seat, mine, theirs)


def test_beats_random_over_paired_seeds_both_seats(tmp_path, monkeypatch):
    main_py = _bundle(tmp_path, monkeypatch)
    for seed in range(N_SEEDS):
        for seat in (0, 1):
            mine, theirs = _play(main_py, "random", seed, seat)
            assert mine > theirs, (seed, seat, mine, theirs)

"""Tests for issue 015's CMA-ES driver over the parametric policy."""

import pytest

from kaggriculture.eval.agents import resolve_policy
from kaggriculture.search.cmaes import _init_worker, _win_rate, _worker_fitness, run_cmaes, sensitivity_analysis
from kaggriculture.search.policy import N_PARAMS, PARAM_SPECS, default_x0
from kaggriculture.sim.decode import default_config_dict


@pytest.fixture(scope="module")
def cfg():
    return default_config_dict()


def test_win_rate_of_a_policy_against_itself_is_undecided_at_one_half(cfg):
    """Two copies of the same deterministic policy mirror each other and tie every game -- the
    same "ties score 0.5" fact issue 011's eval arena relies on (see its own test of the same
    thing for baseline vs baseline)."""
    policy = resolve_policy("pass", cfg)
    rate = _win_rate(policy, [policy], seed_set=[0, 1, 2], config=cfg)
    assert rate == pytest.approx(0.5)


def test_worker_fitness_is_one_minus_win_rate(cfg):
    _init_worker(cfg, ["pass"], [0, 1])
    fitness = _worker_fitness(default_x0())
    assert 0.0 <= fitness <= 1.0
    # the default params should be solidly profitable against `pass` (see test_policy.py) --
    # fitness (a LOSS, cma minimizes) should be well under 0.5 (a coin flip).
    assert fitness < 0.5


@pytest.mark.slow
def test_run_cmaes_improves_or_matches_the_starting_point(cfg):
    """A tiny, fast run (small population, few generations) -- checks the search actually reduces
    (or at least never increases) the loss from generation to generation, and returns a
    well-formed result."""
    result = run_cmaes(cfg, ["pass"], [0, 1], sigma0=3.0, popsize=4, maxiter=3, restarts=1, n_workers=1, seed=0)
    assert len(result.best_x) == N_PARAMS
    assert 0.0 <= result.best_fitness <= 1.0
    assert 1 <= result.generations <= 3  # pycma can stop early once converged (e.g. fitness hits 0)
    assert result.n_evaluations > 0
    fitnesses = [row["best_fitness"] for row in result.trace]
    assert fitnesses == sorted(fitnesses, reverse=True) or all(f <= fitnesses[0] for f in fitnesses)


@pytest.mark.slow
def test_run_cmaes_is_reproducible_given_the_same_seed(cfg):
    def run():
        r = run_cmaes(cfg, ["pass"], [0, 1], sigma0=3.0, popsize=4, maxiter=2, restarts=1, n_workers=1, seed=0)
        return r.best_fitness, r.n_evaluations, r.generations

    a, b = run(), run()
    assert a == b


@pytest.mark.slow
def test_run_cmaes_with_multiple_workers_matches_single_worker_fitness_quality(cfg):
    """Not bit-for-bit reproducibility (process scheduling can affect which of several
    equally-fit candidates pycma reports as `xbest`) -- but the parallel path should find a
    solution at least as good as the sequential one against the same easy opponent."""
    sequential = run_cmaes(cfg, ["pass"], [0, 1], sigma0=3.0, popsize=4, maxiter=3, restarts=1, n_workers=1, seed=0)
    parallel = run_cmaes(cfg, ["pass"], [0, 1], sigma0=3.0, popsize=4, maxiter=3, restarts=1, n_workers=2, seed=0)
    assert parallel.best_fitness <= sequential.best_fitness + 1e-9


@pytest.mark.slow
def test_sensitivity_analysis_covers_every_parameter(cfg):
    result = run_cmaes(cfg, ["pass"], [0, 1], sigma0=3.0, popsize=4, maxiter=2, restarts=1, n_workers=1, seed=0)
    report = sensitivity_analysis(result.best_x, cfg, ["pass"], [2, 3], step=1.0, n_workers=1)
    assert len(report) == len(PARAM_SPECS)
    assert {row["param"] for row in report} == {name for name, *_ in PARAM_SPECS}
    for row in report:
        assert row["sensitivity"] >= 0.0
    # sorted most-sensitive first
    sensitivities = [row["sensitivity"] for row in report]
    assert sensitivities == sorted(sensitivities, reverse=True)


@pytest.mark.slow
def test_cmaes_cli_writes_experiment_files(tmp_path):
    from kaggriculture.search.cmaes import main

    out_dir = tmp_path / "exp"
    main(
        [
            "--opponents",
            "pass",
            "--holdout-opponent",
            "starter",
            "--n-seeds",
            "2",
            "--holdout-n-seeds",
            "2",
            "--sigma0",
            "3.0",
            "--popsize",
            "4",
            "--maxiter",
            "2",
            "--restarts",
            "1",
            "--workers",
            "1",
            "--episode-steps",
            "720",
            "--out",
            str(out_dir),
        ]
    )
    import json

    config = json.loads((out_dir / "config.json").read_text())
    assert config["maxiter"] == 2
    result = json.loads((out_dir / "result.json").read_text())
    assert "best_params" in result
    assert "sensitivity" in result
    assert len(result["sensitivity"]) == N_PARAMS
    assert "holdout_win_rate_vs_training_portfolio" in result
    assert "win_rate_vs_unseen_opponent" in result

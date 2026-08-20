"""Proves the packaging step produces something the engine can actually run.

See issues/003-kaggle-submit-path.md: a submission is only worth making once we know
`env.run(["main.py", ...])` completes locally, without the repo's own package imports.
"""

import pytest

from kaggriculture.submit.build import DEFAULT_AGENT, SelfContainmentError, build, build_composed_bundle


def test_build_produces_self_contained_main_py(tmp_path, monkeypatch):
    import kaggriculture.submit.build as build_mod

    monkeypatch.setattr(build_mod, "SUBMISSIONS_DIR", tmp_path)
    main_py = build(tag="test", agent_path=DEFAULT_AGENT)

    assert main_py == tmp_path / "test" / "main.py"
    assert "import kaggriculture" not in main_py.read_text()
    assert "from kaggriculture" not in main_py.read_text()


def test_build_rejects_agent_that_imports_the_repo_package(tmp_path):
    bad_agent = tmp_path / "bad_agent.py"
    bad_agent.write_text("from kaggriculture.model import stuff\n\ndef agent(obs, cfg):\n    return {}\n")

    with pytest.raises(SelfContainmentError):
        build(tag="test", agent_path=bad_agent)


def test_packaged_agent_runs_a_full_episode(tmp_path, monkeypatch):
    import kaggle_environments as kaggle_env

    import kaggriculture.submit.build as build_mod

    monkeypatch.setattr(build_mod, "SUBMISSIONS_DIR", tmp_path)
    main_py = build(tag="test", agent_path=DEFAULT_AGENT)

    env = kaggle_env.make("kaggriculture")
    env.run([str(main_py), "starter"])

    assert len(env.steps) == 720
    for agent_state in env.steps[-1]:
        assert agent_state["status"] == "DONE"


@pytest.mark.slow
def test_composed_bundle_matches_the_dev_tree_agent_exactly(tmp_path, monkeypatch):
    """Issue 028's submission (`submissions/lns-sell-dp-composition-v1/`): the bundle's own
    per-turn code is import-rewritten from the real source (not hand-transcribed), so it should
    behave byte-identically to the dev-tree composed agent, not just "close enough"."""
    import kaggle_environments as kaggle_env

    import kaggriculture.submit.build as build_mod
    from kaggriculture.eval.agents import load_agent_from_file, wrap_agent
    from kaggriculture.search.composition import _rebuild_lns_incumbent, build_composed_agent
    from kaggriculture.sim import _sim_native as native
    from kaggriculture.sim.decode import default_config_dict, episode_config

    monkeypatch.setattr(build_mod, "SUBMISSIONS_DIR", tmp_path)
    archive = build_composed_bundle(tag="test", n_threads=8)
    main_py = archive.parent / "main.py"

    config = default_config_dict()
    ecfg = episode_config(config)
    bundled_policy = wrap_agent(load_agent_from_file(main_py), config)

    incumbent = _rebuild_lns_incumbent(config, threads=8)
    dev_tree_policy = wrap_agent(build_composed_agent(incumbent, config), config)

    opponent = native.TapePolicy([])
    for seed in (0, 1, 2):
        bundled_result = native.run_episode(bundled_policy, opponent, ecfg, seed)
        dev_tree_result = native.run_episode(dev_tree_policy, opponent, ecfg, seed)
        assert bundled_result.final_money == dev_tree_result.final_money, seed

    # And through the real engine directly, not just this repo's own (proven-exact) sim.
    env = kaggle_env.make("kaggriculture", configuration={"episodeSteps": 720, "seed": 0}, debug=True)
    env.run([str(main_py), "starter"])
    assert env.steps[-1][0]["status"] == "DONE"
    assert env.steps[-1][1]["status"] == "DONE"


@pytest.mark.slow
def test_composed_bundle_survives_kaggles_actual_shallow_agent_path(tmp_path, monkeypatch):
    """Regression test for the bug that broke the first real submission attempt: `liquidation.py`
    (before the fix) defined `REPO_ROOT = Path(__file__).resolve().parents[3]` at MODULE level for
    its own CLI's experiment-dir bookkeeping -- harmless from this repo's own path (plenty of
    parents), but `IndexError` the instant it's imported from Kaggle's actual grader path,
    `/kaggle_simulations/agent/liquidation.py` (only 2 parents above root). Every other check in
    this file runs the bundle from this repo's own directory tree, which never would have caught
    this -- this test exists specifically because that blind spot let a broken bundle reach
    Kaggle undetected."""
    import shutil

    import kaggle_environments as kaggle_env

    import kaggriculture.submit.build as build_mod

    monkeypatch.setattr(build_mod, "SUBMISSIONS_DIR", tmp_path)
    archive = build_composed_bundle(tag="test", n_threads=8)

    shallow_agent_dir = tmp_path / "kaggle_simulations" / "agent"
    shallow_agent_dir.mkdir(parents=True)
    for path in archive.parent.glob("*.py"):
        shutil.copy(path, shallow_agent_dir / path.name)

    env = kaggle_env.make("kaggriculture", configuration={"episodeSteps": 720, "seed": 0}, debug=True)
    env.run([str(shallow_agent_dir / "main.py"), "random"])
    assert env.steps[-1][0]["status"] == "DONE"
    assert env.steps[-1][1]["status"] == "DONE"

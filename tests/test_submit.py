"""Proves the packaging step produces something the engine can actually run.

See issues/003-kaggle-submit-path.md: a submission is only worth making once we know
`env.run(["main.py", ...])` completes locally, without the repo's own package imports.
"""

import pytest

from kaggriculture.submit.build import DEFAULT_AGENT, SelfContainmentError, build


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

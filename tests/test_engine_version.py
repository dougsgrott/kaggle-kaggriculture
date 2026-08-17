"""Fails if the installed kaggle-environments drops below the pinned floor.

See issues/001-pin-engine.md: PR #1394 (2026-08-06) and PR #1399 (2026-08-15)
changed balance under us. Anything older than 1.32.7 is miscalibrated.
"""

from importlib.metadata import version

MIN_VERSION = (1, 32, 7)


def _parse(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split(".")[:3])


def test_kaggle_environments_version_floor():
    installed = version("kaggle-environments")
    assert _parse(installed) >= MIN_VERSION, (
        f"kaggle-environments {installed} is below the pinned floor "
        f"{'.'.join(map(str, MIN_VERSION))} (see issues/001-pin-engine.md)"
    )


def test_kaggriculture_episode_completes():
    import kaggle_environments as kaggle_env

    env = kaggle_env.make("kaggriculture")
    env.run(["starter", "random"])

    assert len(env.steps) == 720
    for agent_state in env.steps[-1]:
        assert agent_state["status"] == "DONE"

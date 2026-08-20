"""Tests for population assembly (issue 018).

`analysis/nb_clean/` is wikikit-generated and gitignored (see CLAUDE.md's Layout section) -- a
bare clone of this repo won't have it, so every test that depends on the real corpus is guarded
with `_HAVE_CORPUS` and skipped (not failed) when it's absent. The corpus-independent pieces
(clustering, feature vectors, the notebook-sandbox stub) are tested against small hand-built
fingerprints instead, so they run everywhere.
"""

import pytest

from kaggriculture.eval.population import (
    NB_CLEAN_DIR,
    _distance,
    _feature_vector,
    cluster_by_opening,
    discover_public_agent_paths,
    select_cluster_representatives,
)

_HAVE_CORPUS = NB_CLEAN_DIR.exists() and any(NB_CLEAN_DIR.glob("*.py"))
_skip_without_corpus = pytest.mark.skipif(not _HAVE_CORPUS, reason="analysis/nb_clean/ is wikikit-generated and gitignored")


def _obs(crop_counts: dict, animal_counts: dict, n_hands: int, n_quadrants: int, money_spent_frac: float) -> dict:
    """A minimal synthetic fingerprint in the same shape `_fingerprint_worker` produces: a
    10x10 board with just enough PLANT/COOP/PASTURE tiles to encode the requested crop/animal
    mix, nothing else populated (`_feature_vector` only reads `farms[0]`'s tiles/hands/
    unlocked_quadrants/money)."""
    tiles = [[None for _ in range(10)] for _ in range(10)]
    x = y = 0

    def next_tile():
        nonlocal x, y
        pos = (x, y)
        x += 1
        if x == 10:
            x, y = 0, y + 1
        return pos

    for crop, count in crop_counts.items():
        for _ in range(count):
            tx, ty = next_tile()
            tiles[ty][tx] = {"kind": "PLANT", "crop": crop}
    for animal, count in animal_counts.items():
        for _ in range(count):
            tx, ty = next_tile()
            tiles[ty][tx] = {"kind": "COOP", "animal": animal}

    starting_money = 3000.0
    farm = {
        "tiles": tiles,
        "hands": [(0, 0)] * n_hands,
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"][: round(n_quadrants * 4)],
        "money": starting_money * (1 - money_spent_frac),
    }
    return {"obs": {"farms": [farm]}, "starting_money": starting_money}


def test_discover_public_agent_paths_excludes_risky_imports():
    from kaggriculture.eval.population import _RISKY_IMPORT

    for path in discover_public_agent_paths():
        assert not _RISKY_IMPORT.search(path.read_text(errors="replace")), path


@_skip_without_corpus
def test_discover_public_agent_paths_only_returns_files_defining_agent():
    import re

    pattern = re.compile(r"^def agent\(", re.MULTILINE)
    for path in discover_public_agent_paths():
        assert pattern.search(path.read_text(errors="replace")), path


def test_feature_vector_reflects_crop_mix():
    all_melon = _obs({"MELON": 8}, {}, n_hands=1, n_quadrants=0.25, money_spent_frac=0.5)
    all_wheat = _obs({"WHEAT": 8}, {}, n_hands=1, n_quadrants=0.25, money_spent_frac=0.5)
    from kaggriculture.eval.population import CROP_NAMES

    melon_idx = CROP_NAMES.index("MELON")
    wheat_idx = CROP_NAMES.index("WHEAT")
    melon_vec = _feature_vector(all_melon)
    wheat_vec = _feature_vector(all_wheat)
    assert melon_vec[melon_idx] == pytest.approx(1.0)
    assert melon_vec[wheat_idx] == pytest.approx(0.0)
    assert wheat_vec[wheat_idx] == pytest.approx(1.0)
    assert _distance(melon_vec, wheat_vec) > 0.5


def test_feature_vector_reflects_money_spent():
    spent_little = _obs({"MELON": 4}, {}, n_hands=1, n_quadrants=0.25, money_spent_frac=0.1)
    spent_a_lot = _obs({"MELON": 4}, {}, n_hands=1, n_quadrants=0.25, money_spent_frac=0.9)
    vec_little = _feature_vector(spent_little)
    vec_a_lot = _feature_vector(spent_a_lot)
    assert vec_a_lot[-1] > vec_little[-1]


def test_cluster_by_opening_groups_identical_fingerprints_and_splits_distinct_ones():
    melon_a = _obs({"MELON": 8}, {}, n_hands=2, n_quadrants=0.25, money_spent_frac=0.5)
    melon_b = _obs({"MELON": 7}, {}, n_hands=2, n_quadrants=0.25, money_spent_frac=0.55)  # nearly identical
    wheat = _obs({"WHEAT": 8}, {}, n_hands=2, n_quadrants=0.25, money_spent_frac=0.5)  # far away
    fingerprints = {"melon_a": melon_a, "melon_b": melon_b, "wheat": wheat}

    clusters = cluster_by_opening(fingerprints, threshold=0.25)
    clusters_as_sets = {frozenset(c) for c in clusters}
    assert frozenset({"melon_a", "melon_b"}) in clusters_as_sets
    assert frozenset({"wheat"}) in clusters_as_sets
    assert len(clusters) == 2


def test_select_cluster_representatives_picks_the_medoid():
    a = _obs({"MELON": 8}, {}, n_hands=2, n_quadrants=0.25, money_spent_frac=0.50)
    b = _obs({"MELON": 8}, {}, n_hands=2, n_quadrants=0.25, money_spent_frac=0.51)  # near-center
    c = _obs({"MELON": 8}, {}, n_hands=2, n_quadrants=0.25, money_spent_frac=0.99)  # an outlier
    fingerprints = {"a": a, "b": b, "c": c}
    reps = select_cluster_representatives([["a", "b", "c"]], fingerprints)
    assert reps == ["b"]  # "b" sits closest to both others; "c" is the far outlier


def test_select_cluster_representatives_is_a_no_op_for_singleton_clusters():
    a = _obs({"MELON": 8}, {}, n_hands=2, n_quadrants=0.25, money_spent_frac=0.5)
    reps = select_cluster_representatives([["solo"]], {"solo": a})
    assert reps == ["solo"]


def test_stub_notebook_only_modules_makes_ipython_and_matplotlib_safe_to_use():
    """The exact failure mode `_fingerprint_worker` hit against the real corpus (a notebook's own
    plotting cell touching a real, possibly-absent-or-misbehaving IPython/matplotlib) -- run in
    this test's own process rather than a fork, since `_stub_notebook_only_modules` is documented
    as only ever safe to call inside an isolated subprocess; this process is torn down with the
    test runner regardless."""
    from kaggriculture.eval.population import _stub_notebook_only_modules

    _stub_notebook_only_modules()
    from IPython.display import display  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    display("anything")  # must not raise
    fig, ax = plt.subplots(figsize=(4, 2))  # the exact unpacking pattern that used to crash
    ax.plot([1, 2, 3], [1, 2, 3])  # arbitrary attribute-call chain on a stub must not raise
    fig.get_height() / 2  # arbitrary arithmetic on a stub must not raise
    assert fig is not None and ax is not None


@_skip_without_corpus
def test_build_public_pool_survivors_all_produce_a_valid_feature_vector():
    from kaggriculture.eval.population import build_public_pool
    from kaggriculture.sim.decode import default_config_dict

    fingerprints = build_public_pool(default_config_dict(), max_workers=4, timeout=25)
    assert fingerprints  # at least one public candidate should survive against the real corpus
    for name, fp in fingerprints.items():
        vec = _feature_vector(fp)
        assert len(vec) == 11  # 5 crops + 3 animals + hands + quadrants + money_spent
        assert all(0.0 <= x <= 1.0 + 1e-9 for x in vec), (name, vec)


@pytest.mark.slow
def test_own_lineage_policies_builds_every_generation():
    from kaggriculture.eval.population import own_lineage_policies
    from kaggriculture.sim import _sim_native as native
    from kaggriculture.sim.decode import default_config_dict

    config = default_config_dict()
    policies = own_lineage_policies(config, threads=8)
    expected = {"pass", "starter", "random", "baseline", "greedy013", "lns014", "cmaes015", "dp016", "current017"}
    assert set(policies) == expected
    for name, policy in policies.items():
        assert isinstance(policy, native.Policy), name

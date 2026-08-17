"""Tests for the pybind11 bindings (issue 009): run_episode/run_batch correctness, determinism,
TapePolicy, CallbackPolicy, and GameState/MarketTownState introspection. Not the parity gate --
issue 010 owns comparing against the real engine. These check the Python-facing glue (episode.cpp,
bindings.cpp) that 007/008's own C++ smoke checks don't exercise.
"""

import pytest

from kaggriculture.sim import _sim_native as native


def _all_pass_turn():
    t = native.PlayerTurn()
    t.unit_actions = [native.UnitAction(native.Op.PASS)]
    t.market_orders = []
    return t


def test_run_episode_all_pass_matches_starting_money():
    tape = native.TapePolicy([])
    cfg = native.EpisodeConfig()
    result = native.run_episode(tape, tape, cfg, 42)
    assert result.final_money == pytest.approx([3000.0, 3000.0])
    assert result.seed == 42
    assert len(result.daily_money) == cfg.episode_steps // cfg.turns_per_day
    assert len(result.daily_inventory) == len(result.daily_money)


def test_run_episode_is_deterministic():
    tape = native.TapePolicy([])
    cfg = native.EpisodeConfig()
    a = native.run_episode(tape, tape, cfg, 7)
    b = native.run_episode(tape, tape, cfg, 7)
    assert a.final_money == b.final_money
    assert a.shop_draw_order == b.shop_draw_order
    assert a.weed_spawn_count == b.weed_spawn_count
    assert a.daily_inventory == b.daily_inventory


def test_run_batch_matches_individual_run_episode_calls():
    tape = native.TapePolicy([])
    cfg = native.EpisodeConfig()
    seeds = [1, 2, 3, 4, 5]
    pairs = [(tape, tape) for _ in seeds]
    configs = [cfg] * len(seeds)

    batched = native.run_batch(pairs, configs, seeds, n_threads=4)
    individual = [native.run_episode(tape, tape, cfg, s) for s in seeds]

    for b, i in zip(batched, individual):
        assert b.final_money == i.final_money
        assert b.shop_draw_order == i.shop_draw_order


def test_run_batch_requires_matching_lengths():
    tape = native.TapePolicy([])
    cfg = native.EpisodeConfig()
    with pytest.raises(Exception):
        native.run_batch([(tape, tape)], [cfg], [1, 2], n_threads=2)
    with pytest.raises(Exception):
        native.run_batch([(tape, tape)], [cfg, cfg], [1], n_threads=2)


def test_run_batch_supports_per_episode_configs():
    tape = native.TapePolicy([])
    short = native.EpisodeConfig()
    short.episode_steps = 24
    long = native.EpisodeConfig()
    long.episode_steps = 48
    results = native.run_batch([(tape, tape), (tape, tape)], [short, long], [1, 1], n_threads=2)
    assert len(results[0].daily_money) == 1
    assert len(results[1].daily_money) == 2


def test_advance_turns_matches_a_full_run_episode_prefix():
    """Branching-search primitive: snapshot a state, advance it, and confirm that matches
    running a fresh episode for the same number of turns from scratch (both start from the
    same seed/config, so they should agree turn for turn)."""
    tape = native.TapePolicy([])
    cfg = native.EpisodeConfig()

    state = native.GameState(cfg.board_size, cfg.turns_per_day, 100, cfg.starting_money, 5)
    market = native.MarketTownState()
    native.advance_turns(state, market, tape, tape, 50, cfg.market)

    reference = native.EpisodeConfig()
    reference.episode_steps = 50
    expected = native.run_episode(tape, tape, reference, 5)

    assert state.step == 50
    assert [state.farm_money(0), state.farm_money(1)] == pytest.approx(list(expected.final_money))


def test_advance_turns_branches_diverge_under_different_policies():
    cfg = native.EpisodeConfig()
    base_state = native.GameState(cfg.board_size, cfg.turns_per_day, 100, cfg.starting_money, 9)
    base_market = native.MarketTownState()
    native.advance_turns(base_state, base_market, native.TapePolicy([]), native.TapePolicy([]), 24, cfg.market)

    branch_a = base_state.copy()
    market_a = base_market.copy()

    def hire_once(state, market, player):
        turn = native.PlayerTurn()
        turn.unit_actions = [native.UnitAction(native.Op.PASS)]
        turn.market_orders = [native.MarketOrder(native.MarketOp.HIRE)]
        return turn

    # A CallbackPolicy rather than a TapePolicy here: TapePolicy indexes by the *absolute*
    # state.step (by design -- see policy.hpp -- since a tape is normally a whole-episode plan
    # replayed from turn 0, and branching re-uses the same tape object rather than a fresh
    # zero-indexed one), so a length-1 tape at state.step == 24 would never fire.
    native.advance_turns(branch_a, market_a, native.CallbackPolicy(hire_once), native.TapePolicy([]), 1, cfg.market)

    branch_b = base_state.copy()
    market_b = base_market.copy()
    native.advance_turns(branch_b, market_b, native.TapePolicy([]), native.TapePolicy([]), 1, cfg.market)

    assert branch_a.farm_n_units(0) == 2  # hired one hand
    assert branch_b.farm_n_units(0) == 1  # untouched
    assert branch_a.farm_money(0) != branch_b.farm_money(0)  # paid the hire cost
    # The original snapshot is untouched by either branch.
    assert base_state.farm_n_units(0) == 1


def test_tape_policy_plants_and_harvests_a_carrot():
    """Same scenario as sim/smoke_main.cpp's carrot lifecycle: plant on turn 1, water through
    the bonus window (ages 0-3), harvest on day 3. Expected yield 3, matching model.yields and
    the how-to-play.md numbers. A CallbackPolicy observes GameState at the harvest turn (before
    HARVEST is applied) to check yield_units directly -- EpisodeResult only samples at day
    boundaries, which isn't fine-grained enough for this."""
    observed = {}

    def farmer(state, market, player):
        step = state.step
        turn = native.PlayerTurn()
        if step == 0:
            turn.unit_actions = [native.UnitAction(native.Op.PASS)]
            turn.market_orders = [native.MarketOrder(native.MarketOp.BUY_SEED, native.Item.CARROT, 1)]
        elif step == 1:
            turn.unit_actions = [native.UnitAction(native.Op.PLANT, native.Item.CARROT)]
            turn.market_orders = []
        elif step in (2, 24, 48, 73):
            turn.unit_actions = [native.UnitAction(native.Op.WATER)]
            turn.market_orders = []
        elif step == 74:
            observed["pre_harvest_tile"] = state.tile_info(player, 4, 4)
            turn.unit_actions = [native.UnitAction(native.Op.HARVEST)]
            turn.market_orders = []
        else:
            turn.unit_actions = [native.UnitAction(native.Op.PASS)]
            turn.market_orders = []
        return turn

    policy = native.CallbackPolicy(farmer)
    idle = native.TapePolicy([])
    cfg = native.EpisodeConfig()
    cfg.episode_steps = 96
    result = native.run_episode(policy, idle, cfg, 1)

    assert observed["pre_harvest_tile"]["kind"] == "PLANT"
    assert observed["pre_harvest_tile"]["yield_units"] == 3
    assert result.final_money[0] == pytest.approx(3000.0 - 20)  # CARROT seed cost


def test_callback_policy_is_invoked_and_can_read_state():
    calls = []

    def callback(state, market, player):
        calls.append((state.step, player))
        turn = native.PlayerTurn()
        turn.unit_actions = [native.UnitAction(native.Op.PASS)]
        turn.market_orders = []
        return turn

    policy = native.CallbackPolicy(callback)
    idle = native.TapePolicy([])
    cfg = native.EpisodeConfig()
    cfg.episode_steps = 5
    native.run_episode(policy, idle, cfg, 0)

    assert len(calls) == 5
    assert [c[0] for c in calls] == [0, 1, 2, 3, 4]
    assert all(c[1] == 0 for c in calls)


def test_game_state_tile_info_reports_plant_fields():
    gs = native.GameState()
    info = gs.tile_info(0, 4, 4)
    assert info["kind"] == "EMPTY"


def test_market_price_matches_how_to_play_table():
    assert native.market_price(native.Item.WHEAT, 10000) == 25
    assert native.market_price(native.Item.WHEAT, 10000 - 400) == 45

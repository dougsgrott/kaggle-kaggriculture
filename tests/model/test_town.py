"""town.py against the real engine's `_town_consume`, driven purely by both agents PASSing so
market inventory only moves from town demand (no player SELL/BUY_PRODUCT in the mix)."""

from kaggle_environments import make

from kaggriculture.model.constants import PRODUCTS
from kaggriculture.model.town import drain_at_step


def test_shop_and_center_drain_match_a_pass_pass_episode():
    env = make("kaggriculture", configuration={"episodeSteps": 200}, debug=True)
    env.run(["pass", "pass"])

    prev = None
    checked_ticks = 0
    for step in env.steps[:-1]:
        obs0 = step[0].observation
        inv = dict(obs0["market"]["inventory"])
        shops = list(obs0["town"]["unlocked_shops"])
        turn = obs0.get("step", 0)
        if prev is not None:
            prev_inv, prev_shops, prev_turn = prev
            predicted = drain_at_step(prev_shops, prev_turn)
            for item in PRODUCTS:
                assert prev_inv[item] - inv[item] == predicted.get(item, 0), (prev_turn, item)
            if predicted:
                checked_ticks += 1
        prev = (inv, shops, turn)

    assert checked_ticks > 0  # sanity: at least one shop/center tick actually fired

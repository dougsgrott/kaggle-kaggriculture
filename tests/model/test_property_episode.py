"""Property test: model/ against 50 real kaggle_environments episodes (issue 004's acceptance
criterion). Player 0 runs a scripted, fully-known carrot lifecycle against a `random` opponent,
so market/town/weed RNG genuinely varies by seed while the crop schedule stays under our
control. Two things are checked at every step:
  - price parity: obs.market.prices matches price() computed from obs.market.inventory
  - yield parity: the carrot tile's final harvest matches yields.one_time_crop_yield()
"""

from kaggle_environments import make

from kaggriculture.model.constants import PRODUCTS
from kaggriculture.model.price import price
from kaggriculture.model.yields import one_time_crop_yield

N_SEEDS = 50
EPISODE_STEPS = 150

# Plant on turn 1 (turn 0 buys the seed), water every day through the bonus window (ages 0-3),
# harvest on day 4 -- see tests/model/test_price.py's sibling for the isolated version of this.
CARROT_SCRIPT = {
    0: {"market": [["BUY_SEED", "CARROT", 1]]},
    1: {"farmer": ["PLANT", "CARROT"]},
    2: {"farmer": ["WATER"]},
    24: {"farmer": ["WATER"]},
    48: {"farmer": ["WATER"]},
    72: {"farmer": ["WATER"]},
    96: {"farmer": ["HARVEST"]},
}
WATERED_AGES = {0, 1, 2, 3}


def _scripted_carrot_agent(obs):
    action = CARROT_SCRIPT.get(obs.get("step", 0), {})
    return {"farmer": action.get("farmer", ["PASS"]), "hands": [], "market": action.get("market", [])}


def test_price_and_yield_parity_over_50_seeds():
    expected_carrots = one_time_crop_yield("CARROT", WATERED_AGES)

    for seed in range(N_SEEDS):
        env = make("kaggriculture", configuration={"episodeSteps": EPISODE_STEPS, "seed": seed}, debug=True)
        env.run([_scripted_carrot_agent, "random"])

        for step in env.steps:
            obs0 = step[0].observation
            inv = obs0["market"]["inventory"]
            quoted = obs0["market"]["prices"]
            for item in PRODUCTS:
                assert price(item, inv[item]) == quoted[item], (seed, obs0.get("step"), item)

        final_shed = env.steps[-1][0].observation["private"]["shed"]
        assert final_shed["CARROT"] == expected_carrots, seed

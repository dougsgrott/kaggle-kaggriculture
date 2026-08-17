"""The six documented doc-vs-engine discrepancies from issue 002 / discussion 732450, encoded
so a future engine update that "fixes" one of them (reverting to the old documented behaviour)
fails loudly.

Four of the six are yield/timing facts, covered in test_yields.py against yields.py directly:
  1. CARE bonus increments by +1, not +2            -> test_care_bonus_increments_by_1_not_2
  4. Melon's bonus window has two dead days (11-12) -> test_melon_bonus_window_has_two_dead_days
  5. Strawberry is capped at 4 productions          -> test_strawberry_capped_at_4_productions...
  6. Yield-per-day table uses a consistent formula  -> test_yield_per_tile_per_day_matches_...

(A bonus fifth timing fact not in the original six but from the same discussion — planting-day
watering has no grace period — is also covered there, test_planting_day_watering_discrepancy_....)

The other two are farm/market mechanics with no yields.py equivalent, so they're checked
directly against the real vendored engine here:
  2. SELL is accepted for FERTILIZER despite the docs calling it buy-only.
  3. DIG is a no-op on an occupied coop/pasture (only empty structures can be dug up).
"""

from kaggle_environments import make

from kaggriculture.model.constants import PRODUCTS


def _run(script: dict, episode_steps: int = 10):
    def scripted(obs):
        action = script.get(obs.get("step", 0), {})
        return {"farmer": action.get("farmer", ["PASS"]), "hands": [], "market": action.get("market", [])}

    env = make("kaggriculture", configuration={"episodeSteps": episode_steps}, debug=True)
    env.run([scripted, "pass"])
    return env


def test_fertilizer_is_sellable_despite_buy_only_docs():
    assert "FERTILIZER" in PRODUCTS  # the structural reason SELL FERTILIZER isn't rejected

    env = _run(
        {
            0: {"market": [["BUY_PRODUCT", "FERTILIZER", 1]]},
            1: {"market": [["SELL", "FERTILIZER", 1]]},
        }
    )
    money_after_buy = env.steps[1][0].observation["farms"][0]["money"]
    money_after_sell = env.steps[2][0].observation["farms"][0]["money"]
    shed_after_sell = env.steps[2][0].observation["private"]["shed"]["FERTILIZER"]

    assert money_after_sell > money_after_buy  # the SELL was accepted and paid out
    assert shed_after_sell == 0  # and it actually left the shed


def test_dig_fails_on_occupied_structure():
    env = _run(
        {
            0: {"farmer": ["BUILD_COOP"], "market": [["BUY_ANIMAL", "GOOSE", 1]]},
            1: {"farmer": ["PICKUP", "GOOSE", 1]},
            2: {"farmer": ["PLACE", "GOOSE"]},
            3: {"farmer": ["DIG"]},
        }
    )
    farm = env.steps[4][0].observation["farms"][0]
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]

    assert tile["kind"] == "COOP"
    assert tile["animal"] == "GOOSE"  # DIG did not remove it

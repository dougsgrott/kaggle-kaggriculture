# Engine version

Resolved via `uv add "kaggle-environments>=1.32.7"` on 2026-08-17.

- **Installed:** `kaggle-environments==1.32.7`
- **Source path:** `.venv/lib/python3.12/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`
  (not yet vendored — that's [002](../issues/002-vendor-engine-source.md))
- **Smoke test:** `make("kaggriculture")` + `env.run(["starter", "random"])` completes 720 steps,
  both seats report `status == "DONE"`. Codified in `tests/test_engine_version.py`.

## `CROPS`

| Crop | seed | first_yield_day | max_yield_day | interval | max_yield | ongoing |
|---|---|---|---|---|---|---|
| WHEAT | 10 | 2 | 4 | 0 | 6 | False |
| CARROT | 20 | 2 | 3 | 0 | 4 | False |
| TOMATO | 50 | 8 | 8 | 1 | 4 | True |
| STRAWBERRY | 100 | 10 | 10 | 2 | 4 | True |
| MELON | 80 | 10 | 12 | 0 | 6 | False |

## `ANIMALS`

| Animal | cost | structure | first_yield_day | interval | max_held | product |
|---|---|---|---|---|---|---|
| GOOSE | 300 | COOP | 4 | 1 | 4 | EGG |
| COW | 400 | PASTURE | 8 | 2 | 6 | MILK |
| SHEEP | 500 | PASTURE | 6 | 3 | 6 | WOOL |

## `MARKET_PARAMS`

Pricing model: `price(inv) = base + sign * amp * f(|inv - I0|)`, `sign = +1` below `I0`
(scarcity), `-1` above (glut); `amp = target * base / f(T)`; `T` is the production capacity of
one 5x5 field over 24 days at optimal watering, no fertilizer (animal `T` pre-discounted 30% for
wheat-feed overhead). `I0 = 10000`, `PRICE_FLOOR = 1`.

`f` in `{linear, sq, sqrt, log, log10, hinge}`; `hinge` is linear in `x/T` below the knee, then
adds `HINGE_GAIN=8.0 * max(0, x/T - 1)^2` above it — calm until genuinely scarce, then runs away.
This shape (PR #1399, 2026-08-15) is now on **CARROT, TOMATO, EGG** below-target.

| Item | base | T | below_func | below_target | above_func | above_target |
|---|---|---|---|---|---|---|
| WHEAT | 25 | 400 | sqrt | 0.80 | log | 0.20 |
| CARROT | 35 | 450 | **hinge** | 1.00 | sqrt | 0.70 |
| TOMATO | 60 | 200 | **hinge** | 0.40 | sqrt | 0.60 |
| STRAWBERRY | 120 | 100 | sqrt | 0.70 | linear | 1.60 |
| MELON | 250 | 300 | log | 0.20 | sq | 3.60 |
| EGG | 50 | 332 | **hinge** | 0.40 | log | 0.20 |
| MILK | 160 | 122 | sqrt | 0.60 | linear | 1.60 |
| WOOL | 200 | 105 | log | 0.20 | sq | 3.20 |
| FERTILIZER | 100 | 200 | linear | 0.40 | linear | 0.40 |

## Open question

`configuration.marketParams` supports sparse per-resource overrides merged onto these defaults
(`_resolve_market_params`, kaggriculture.py:77) — worth checking what the actual competition
configuration sets before trusting these as the live values end-to-end.

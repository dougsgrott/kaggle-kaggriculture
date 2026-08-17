# Engine version

Resolved via `uv add "kaggle-environments>=1.32.7"` on 2026-08-17.

- **Installed:** `kaggle-environments==1.32.7`
- **Vendored:** `vendor/kaggriculture.py` + `vendor/kaggriculture.json`, copied verbatim from the
  installed package on 2026-08-17 ([002](../issues/002-vendor-engine-source.md)). Provenance
  (wheel sha256, upload timestamp) is recorded in the vendored file's header — there is no git
  tag for 1.32.x releases on `Kaggle/kaggle-environments` to point at instead.
- **Smoke test:** `make("kaggriculture")` + `env.run(["starter", "random"])` completes 720 steps,
  both seats report `status == "DONE"`. Codified in `tests/test_engine_version.py`.

## Changelog: PR #1394 and PR #1399

PyPI release timestamps pin the boundaries precisely: PR #1394 landed between `1.32.5`
(2026-08-06) and `1.32.6` (2026-08-07); PR #1399 landed between `1.32.6` and `1.32.7`
(2026-08-15). Diffed by downloading each wheel from PyPI and comparing
`kaggriculture.py`/`kaggriculture.json` directly — no changelog or PR description was available,
so this is the ground truth for what actually shipped.

### PR #1394 (`1.32.5` → `1.32.6`) — town-center demand and shop unlocks

- **Town-center demand cut from a scaling schedule to a flat 1/day.** Previously
  `TOWN_CENTER_DEMAND_SCHEDULE = [(20, 4), (10, 2), (0, 1)]` pulled 4 units/product on days
  20-23, 2 on days 10-19, 1 before that (`market["inventory"][item] -= center_mult`). Now every
  tick pulls exactly 1 unit/product regardless of day, and `townCenterSellInterval`'s default
  doubled from 12 to 24 turns — with `turnsPerDay=24` this means the town center buys once/day,
  flat, for the whole season. This is a large cut to a demand sink that used to ramp up 4x by the
  endgame; late-game gluts on town-center goods are worse post-patch than pre-patch models would
  predict.
- **Shop unlocks now sample with replacement, capped at `MAX_SHOP_INSTANCES = 8`.** Previously
  each unlock drew from `[s for s in SHOPS if s not in town["unlocked_shops"]]` — each of the (7?)
  shop types could appear at most once. Now `town["unlocked_shops"].append(rng.choice(sorted(SHOPS)))`
  runs unconditionally up to 8 total instances, so the same shop can unlock multiple times and a
  given game may never see all shop types. Each duplicate instance still consumes independently
  (its own `townShopSellInterval` tick), so demand for popular shop types can compound instead of
  being capped at one tick per type.
- **Unrelated same-release fix:** `PLACE` (animal/shed-drop) now resolves *before* the `LOCKED`
  tile guard, not after. Three of the four shed-access tiles start `LOCKED`; under the old
  ordering a hand standing on one of them could never `PLACE` into the shed. This looks like a
  bugfix bundled into the same release, not part of #1394's stated scope, but it changes legal
  action sequences near the shed and is worth knowing about.

### PR #1399 (`1.32.6` → `1.32.7`) — hinge scarcity curves

- **New `"hinge"` shape function**, `_shape(func, x, T)`: linear in `x/T` below the knee (`u <= 1`),
  then adds `HINGE_GAIN=8.0 * max(0, u - 1)^2` above it. `f(T) == 1` by construction so `target`
  keeps meaning the same thing across shapes. Every `_shape` call site now threads `T` through
  (`_shape(f, T, T)`, `_shape(f, I0 - inventory, T)`), a signature change from the old two-arg
  `_shape(func, x)` — any code that imported/copied `_shape` needs updating, not just the params
  table.
- **`CARROT`, `TOMATO`, `EGG` switched to `hinge` below target** (previously `log`/`linear`/`linear`
  respectively). `CARROT.below_target` also jumped 0.20 → 1.00 and `TOMATO.below_target` stayed at
  0.40 but under a much steeper curve. Below-target scarcity pricing on these three goods is now
  calm until genuinely scarce, then runs away quadratically — public notebooks modeling them with
  the old smooth curves (or the old params) will misprice sell timing.
- No changes to `above_func`/`above_target` for these three, nor to any WHEAT/STRAWBERRY/MELON/
  MILK/WOOL/FERTILIZER row.

### Net effect vs. the pre-#1394 baseline (`1.32.5`)

Both PRs push toward *less forgiving* endgame selling: town-center demand no longer scales up in
the back half of the game, and three below-target curves went from smooth to hinge-shaped (calm
near equilibrium, punishing once genuinely scarce). Any strategy tuned against `1.32.2`-era
constants (per `CLAUDE.md` non-negotiable #2) will be miscalibrated on both axes.

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

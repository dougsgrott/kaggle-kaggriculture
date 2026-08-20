# Terminal liquidation: wind-down days, the concurrent-selling mechanic, and denial-selling

[017](../issues/017-terminal-liquidation.md)'s own acceptance text asks for "a stated wind-down
day per product, with the reasoning" — this is that writeup, plus the two mechanical findings
(concurrent selling, denial-selling) the issue's scope asked to be worked out and handed to
issue 025's ablation suite rather than solved here.

## 1. The mechanic everything else here depends on

Verified directly against `vendor.kaggriculture._process_market` (hand-seeded farm/private/market
state, not inferred from reading the code) — see
`tests/search/test_liquidation.py::test_simulate_concurrent_sale_matches_the_vendor_engine_exactly`.

| scenario | player A revenue | player B revenue |
|---|---|---|
| both SELL 40 WOOL, same turn | $4,026 | $4,026 |
| A SELLs 50, B SELLs 20, same turn | $4,558 | $3,428 |
| A SELLs 50 alone (B sells 0) | $7,655 | — |

Two things fall out of this table:

1. **Equal-sized simultaneous orders get an identical price sequence.** `_process_market` quotes
   both players from the *same* shared inventory each round and commits both before moving to the
   next round — there is no "whoever's listed first wins" effect from player index or action-dict
   order, contrary to what the issue's own original framing worried about.
2. **What actually matters is order size.** B's smaller order finishes first and keeps the price
   trajectory both were seeing while both were active; A's *remaining* 30 units then face a market
   growing at half the combined rate, alone — worse than if A had sold 50 units solo the whole
   time. Concretely: A's revenue for the SAME 50 units drops from $7,655 (alone) to $4,558 (with B
   concurrently dumping 20) — a **40% cut, from the opponent's presence alone**, on production A
   would have owned regardless.

`search.liquidation.simulate_concurrent_sale(item, base_inv, my_units, opponent_units)` is the
reusable tool this table came from.

## 2. Wind-down day per product

`search.liquidation.wind_down_day(item, n_days, committed_units)`: the latest day a NEW planting
(crop) or purchase (animal) can start and still be worth it, gated by (a) finishing before season
end and (b) clearing positive $/action at the price a fresh unit would fetch given what's already
committed to that product this season (`search.schedule`'s own `_effective_price` formula,
recomputed here since the original is module-private).

Run against `build_schedule()`'s own default plan (`experiments/exp-013-terminal-liquidation/`):

| product | committed units | wind-down day | reasoning |
|---|---|---|---|
| WHEAT | 0 | 27 | `n_days - 1 - first_yield_day` (2) — production lead time is the binding constraint, not saturation |
| CARROT | 213 | 27 | same — 213 committed units is nowhere near enough to saturate CARROT's much larger absorption pool |
| MELON | 180 | 19 | same — even 180 committed melon units (this schedule's own biggest single commitment) doesn't saturate it enough to make a fresh planting unprofitable |
| TOMATO | 0 | 21 | lead time only |
| STRAWBERRY | 0 | 19 | lead time only |
| GOOSE | 0 | 25 | lead time only |
| COW | 0 | 21 | lead time only |
| SHEEP | 108 (via WOOL) | 23 | lead time only |

**The finding: for this schedule, every product's wind-down day is set by production lead time
(`first_yield_day`), not by market saturation.** `wind_down_day` only returns `None` (never worth
it, at any day) once committed production is pushed far past what's realistic for one schedule —
checked directly: MELON needs ~5,000 committed units before `_effective_price` crushes it enough
to return `None` (`test_wind_down_day_is_none_once_saturated`), two orders of magnitude past what
any schedule in this repo has actually produced. This is a reassuring result for the midgame (013's
own committed-production tracking is already conservative enough that "should I still be planting
this?" rarely needs a special end-of-season answer beyond "can it still mature in time") — but
it's a schedule-dependent fact, not a game-mechanical one, and would look different for a schedule
that had genuinely saturated a product's absorption pool by day 20.

## 3. Denial-selling

Issue scope: "work out whether pre-crashing a product you hold little of, to deny the opponent, is
ever correct here — hand the result to issue 025's denial-selling ablation." Using
`simulate_concurrent_sale`, holding the opponent's assumed dump fixed at 60 WOOL units (their own
private holdings, unobservable — this table shows the *mechanism*, not a specific recommendation):

| my concurrent dump | my revenue | opponent's revenue | opponent's revenue lost (denial) | denial per unit I sell |
|---|---|---|---|---|
| 0 (baseline: opponent sells alone) | $0 | $7,929 | — | — |
| 5 | $993 | $6,944 | $985 | $197.0 |
| 10 | $1,934 | $6,017 | $1,912 | $191.2 |
| 20 | $3,428 | $4,568 | $3,361 | $168.1 |
| 40 | $4,026 | $4,046 | $3,883 | $97.1 |
| 60 | $4,046 | $4,046 | $3,883 | $64.7 |

**The finding: denial-selling here is not a sacrifice, it's close to a free option.** Every one of
my early units earns me real revenue (~$170-200/unit while the opponent's order is still active,
roughly market rate) *and* costs the opponent a comparable amount — I'm not burning value to hurt
them, I'm just choosing to compete for the same pool instead of staying out of it. The effect
saturates once my dump matches or exceeds the opponent's (past ~60 units here, there's no more of
their order left to crash into). This makes denial-selling attractive specifically in the endgame
(where I was going to liquidate this product for roughly the same price anyway, so there's no real
opportunity cost to timing it concurrently with a suspected opponent dump) and much less obviously
attractive mid-season (where holding for a better later price, per issue 016, is usually the
better use of the same units) — **not evaluated here**, since it depends on inferring the
opponent's holdings, which this issue has no visibility into and issue 025's ablation suite is
better positioned to test empirically (varying assumed opponent behavior across many episodes)
than to derive analytically from first principles.

## 4. Fertilizer

`FERTILIZER` has zero exogenous drain — it isn't in `TOWN_CENTER_PRODUCTS` and no `SHOPS` entry
lists it (confirmed in issue 016's own investigation). `solve_endgame` run on a pure-fertilizer
scenario with no opponent contesting it confirms the expected degenerate case: with a perfectly
flat forecast (no drain, no competition), every turn in the window is revenue-equivalent, so the
DP's tie-breaking (favors deferring when tied) pushes the whole dump to the window's last turn —
harmless (no revenue difference from selling earlier), not a meaningful "wait for a better price"
finding the way WOOL/MELON's town-drain-driven timing is. **Practical takeaway: fertilizer should
just be sold whenever convenient during the endgame — there's no scarcity dynamic to time against,
consistent with it being a free byproduct with no cost basis to protect.**

## 5. Validation

Endgame DP vs. the naive "dump everything the moment the window opens" baseline, same production
schedule, live evaluation (`experiments/exp-014-terminal-liquidation/`, `window_turns=24`,
100 seeds / 200 games): **200W-0L-0T**, Wilson CI `[0.981, 1.000]`, verdict **`better`**. Mean
money: $8,588 (DP) vs. $7,711 (naive), +11%. (Corrected 2026-08-20 — the original run,
`exp-013-terminal-liquidation`, hit a cross-episode stale-state bug in both wrapper agents that
this rerun fixes; see [017](../issues/017-terminal-liquidation.md)'s second Revision section for
what broke and why the corrected result is cleaner, not weaker.) See
[017](../issues/017-terminal-liquidation.md)'s first Revision section for the full writeup,
including why `window_turns=24` (exactly the final day) outperforms a 48-turn window.

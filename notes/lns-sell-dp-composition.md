# Composing 014's LNS plan with 016's sell-DP and 017's terminal liquidation (issue 028)

Run: `uv run python -m kaggriculture.search.composition --n-seeds 30 --threads 8` —
`experiments/exp-016-lns-sell-dp-composition/result.json`. Build time (LNS rerun + sell-DP solve):
293s.

## The composition

`terminal_liquidation_agent(schedule_agent(lns_incumbent, config, sell_plan=measured_sell_plan),
config)` — 014's LNS-optimized `Schedule` (reproduced from `exp-005-lns`'s own recorded
parameters), executed via `schedule_agent`, sell-timed by 016's DP (`solve_all_products`,
`discount=0.93`), wrapped by 017's endgame liquidation layer (`window_turns=24`). The only new
piece is `measure_arrivals`/`_track_shed_increases` (`search/composition.py`) — see below for why
`Schedule.arrivals` couldn't be used directly.

## Why `Schedule.arrivals` had to be replaced with a measurement, not just recomputed

The original concern (LNS's direct-mutation operators — `_swap_crop`, `_resize_herd`,
`_resize_crew`, `_shift_land_day` — don't recompute `arrivals` after mutating `tile_role`) turned
out to be one instance of a bigger gap. Checked `measure_arrivals` against 013's own **default,
never-LNS-touched** schedule expecting agreement — it disagreed substantially: analytical
`schedule.arrivals` credits MELON with 180 units (days 13-29), but the live agent, run through the
real engine, never plants a single MELON tile. `build_schedule()`'s greedy construction loop
re-ranks the best crop for a tile *every time it comes free* — a tile can and does grow a
different crop cycle to cycle as marginal prices shift over the season, and `arrivals` correctly
records every one of those cycles — but `Schedule.tile_role` only stores the FINAL such decision
per tile, and `schedule_agent`'s live execution has no memory of anything earlier: it just
replants whatever `tile_role` says, forever. Confirmed directly: the default schedule's final
`tile_role` snapshot is 100% CARROT across all 23 crop tiles; the MELON credit in `arrivals` is
real production from cycles that happened before those tiles got reassigned.

**`Schedule.arrivals` describes an idealized multi-crop-rotation forecast that `schedule_agent`
has never actually executed, for any schedule this repo has built, not just LNS-mutated ones.**
This doesn't invalidate issue 016's own reported result (measured through real games, not through
the analytical field) — but it likely means 016's own sell plan has dead entries for products the
live agent never actually produces. Worth a look if 016 gets more budget; out of this issue's own
scope to fix retroactively.

**The fix**: `measure_arrivals` runs the schedule through one real recording episode and tracks
`private_shed` turn by turn, attributing every *increase* to that turn's day (a decrease is a
sale, ignored). Ground truth from the engine, not the analytical field — consistent with CLAUDE.md's
non-negotiable #1. Validated with a small hand-built `agent_fn` (`tests/search/test_composition.py`)
issuing `BUY_PRODUCT` on known turns, confirming the tracker reports exactly the expected
per-day gross arrivals and correctly ignores a same-day-later sale.

## Results

| opponent | record | Wilson 95% CI | verdict |
|---|---|---|---|
| `current017` (016+017 on the *default* schedule) | 60W-0L-0T | `[0.940, 1.000]` | **better** |
| `lns014` (LNS plan, naive floor-threshold selling) | 60W-0L-0T | `[0.940, 1.000]` | **better** |
| `baseline` (issue 005) | 26W-34L-0T | `[0.316, 0.559]` | undecided |
| `beicicc__kaggriculture-c26-kaito-v23-dual-regime` (population's dominant strategy) | 0W-60L-0T | `[0.000, 0.060]` | worse |

The composition is a clean, decisive improvement over **both** of the pieces it's built from —
confirming issue 018's own finding (production-plan quality dominates) *and* that sell-timing
sophistication still adds real value once it's applied to the better plan, not just that swapping
plans alone explains everything. Against `baseline`, the gap that `current017` lost 93.3% of the
time (issue 018's payoff matrix) is now `undecided` at 43.3% win rate (26W-34L) — a large
improvement, though not (yet) a clean win. `beicicc-c26` remains undefeated against everything
this repo has built, consistent with issue 018's finding that it's a strict dominant strategy over
the whole population.

## Submitted to the ladder (2026-08-20)

Submitted directly as `lns-sell-dp-composition-v1` rather than held for issue 026 — `baseline-v1`
remains separately submitted and separately scored (Kaggle scores the latest 2; this replaces the
old `003` PASS-agent smoke test in that pair, not `baseline-v1`), so this is a genuine second
scored ladder data point, not a replacement gamble. Needed a new bundler,
`submit.build.build_composed_bundle`: freezes the LNS-optimized `Schedule` and its measured sell
plan as literal data (the ~5-minute search can't run at Kaggle's per-turn time budget), and bundles
`schedule_agent`'s and `terminal_liquidation_agent`'s own per-turn code verbatim (import-rewritten
from the real source, not hand-transcribed) — `terminal_liquidation_agent`'s endgame DP has to run
live, since it needs each game's own actual final-day holdings.

**The first submission attempt errored on Kaggle's actual grader despite every local check
passing** (dev-tree parity, a clean run through the real `kaggle_environments` engine, an 80-game
crash-free sweep) — because every local check ran the bundle from this repo's own directory tree,
which has plenty of parent directories, while Kaggle's grader runs it from
`/kaggle_simulations/agent/`, which doesn't. `liquidation.py` (copied verbatim from the dev tree)
defines `REPO_ROOT = Path(__file__).resolve().parents[3]` at module level for its own CLI's
bookkeeping — a statement that runs at import time, and `.parents[3]` doesn't exist 2 directories
above root. Confirmed via `kaggle competitions logs`; fixed by stripping the module-level
`REPO_ROOT`/`EXPERIMENTS_DIR` pair from every bundled search module, and by reproducing Kaggle's
shallow path locally before ever uploading again — now a standing regression test.

Resubmitted the same day: **`SubmissionStatus.COMPLETE`, public score 600.0, above
`baseline-v1`'s 484.5.** See `submissions/lns-sell-dp-composition-v1/README.md` for the full
story, and issue 028's Revision section for a secondary finding surfaced along the way (the two
submission attempts' LNS incumbents scored slightly differently against `baseline` locally despite
a fixed RNG seed — a possible multi-threaded-evaluation nondeterminism worth a closer look, not
chased further here).

## The baseline gap, closed (issue 029, same day)

A 5x larger LNS search budget (`--n-iters 300 --n-restarts 15`, vs. the original `60`/`6` from
`exp-005-lns`) — issue 014's own long-standing "what's next" note — turned the `undecided` 43.3%
win rate above into a clean **60W-0L-0T sweep against `baseline`, CI `[0.940, 1.000]`, verdict
`better`**: the first search-derived agent this repo has built to decisively beat `baseline`.
Also decisive against `current017` and against the same larger-budget plan with naive selling
(both 60W-0L-0T) — the sell-DP/liquidation layer keeps adding value on top of a much stronger base
plan, not just papering over a weak one. `experiments/exp-017-lns/`: 315 evaluations, 1,663s,
portfolio margin $22,504 (search-set) vs. the original run's $14,087 — the win-rate jump and the
margin improvement track closely (both roughly +60%), suggesting no diminishing returns yet in
this budget range. Submitted as `lns-sell-dp-composition-v2` (replacing `lns-sell-dp-
composition-v1` in Kaggle's latest-2-scored pair, a deliberate choice given v2's strict local
improvement over v1). Full account: `issues/029-close-baseline-gap.md`.

## What's next

- `beicicc-c26` is now the only population member this repo's best agent still loses to (0W-60L,
  unchanged across every generation tested) — the single highest-value open question left.
  Whether an even larger LNS budget (issue 014's own speculative "thousands, not tens") would
  close this gap too is untested and a natural next experiment.
- Watch the real ladder result for `lns-sell-dp-composition-v2` — the first search-derived
  (non-baseline) agent this repo has both decisively validated locally against `baseline` AND put
  on the live ladder.

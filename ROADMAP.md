# Kaggriculture — Solution Plan & Research Roadmap

_Written 2026-08-16. Living document: update the phase checkboxes and the research-line table as
results land. Project instructions live in [`CLAUDE.md`](CLAUDE.md)._

**This file is the strategy; [`issues/`](issues/README.md) is the work breakdown.** Each roadmap
phase expands into numbered issue files with scope, acceptance criteria and references:

| Phase | Issues |
|---|---|
| P0 · Foundation | [001–006](issues/README.md) |
| P1 · Instruments | [007–012](issues/README.md) |
| P2 · Offline optimization | [013–020](issues/README.md) |
| P3 · Closed-loop and robustness | [021–025](issues/README.md) |
| P4 · Freeze and select | [026](issues/026-final-portfolio.md) |
| P5 · After the deadline | [027](issues/027-publication-package.md) |

## Context

This repo currently holds a **knowledge base, not a solution**. `wikikit` has mirrored the
competition into `wiki/` (12 rules pages, 83 discussion threads, 256 notebooks, 1 leaderboard
snapshot) and analysed the notebook corpus into `analysis/` (archetypes, clusters, fork
families, priority reading list). `competition.toml` encodes a carefully-tuned domain lexicon.
`src/kaggriculture/__init__.py` is a `print("Hello")` stub. Nothing plays the game.

The goal is to build the actual competitive agent plus the research apparatus around it.

**Decisions locked at the outset:**
- Objective: **balanced** — a strong rank *and* publishable research artifacts.
- Publishing: **private until Sept 30**, publish writeups after. (The current rank-1/rank-2
  teams run nothing public; the public meta has a documented ELO ceiling of ~3117–3131.)
- Offline sim: **C++ port with Python bindings**, validated step-for-step against the real
  engine. This box is 20 vCPU / 7 GB RAM / no GPU — a search machine, not an RL machine.

**Hard facts that drive everything below**

| | |
|---|---|
| Deadline | **2026-09-30** (45 days from 2026-08-16); games run to ~Oct 15, then Bradley-Terry final |
| Prizes | Places **1–10 each get $5,000** — top-10 is the target band |
| LB at start | rank 1 = 3208.4, rank 10 = 2965.2, rank 50 ≈ 2828 (4,784 teams) |
| Game | 2-player, 720 turns (24 turns × 30 days), 1 farmer action/turn + hired hands, ≤10 market orders/turn, most coins wins |
| Submissions | 5/day, **only latest 2 active**, ≤100 MiB, `main.py` at root, runs on 1.6 vCPU / 6.5 GiB |
| Engine | `kaggle-environments >= 1.32.7` — **balance changed twice in the 10 days before this plan** (PR #1394 Aug 6, PR #1399 Aug 15) |

---

## The one thing most competitors get wrong

The engine changed under the meta. PR #1394 (Aug 6) cut town-center demand 2x→1x/day and made
shop draws sampled *with replacement*; PR #1399 (Aug 15) put `hinge` scarcity curves on
tomato/carrot/egg so those become viable in 50% / 26% / 22% of games depending on the random
shop draw. Meanwhile **56 public notebooks paste the pre-change `MARKET_PARAMS` table verbatim**
and the top public lineage pins `kaggle-environments==1.32.2`. A large fraction of the ladder is
tuned for a game that no longer exists (see `analysis/nb_clean/dariushafshar__agent-tuned-for-a-dead-game.py`).

Two consequences: (a) the biggest single edge available is simply being correctly calibrated to
1.32.7; (b) **shop composition is now a per-episode scenario variable** that nobody's open-loop
tape can react to. Research line R6 is built on this.

---

## Strategic read of the game

Distilled from `wiki/competition/pages/how-to-play.md` and the high-signal discussions:

1. **Actions are the binding constraint, not land or money.** 720 farmer-turns; market orders are
   effectively free (10/turn). Profit-per-action ranking (thread 734033): melon 142, sheep+CARE
   100, cow+CARE 79, strawberry 27, wheat 15. Land caps at 100 tiles but nobody buys the 4th
   quadrant (thread 734308) because they cannot staff it.
2. **CARE is the most under-used lever.** Feed-only sheep = 32 $/action; with daily CARE = 100,
   because the banked bonus pays out in full on the 3-day production tick.
3. **Selling is where the ladder is won.** Premium goods (strawberry, melon, milk, wool) use
   `above_target > 1`, so a modest glut drives them to the $1 floor. Thread 734412 measured
   strawberry season output at **$100,445 sold into the town's demand hole vs $4,173 dumped flat**
   — a 24x swing on timing alone.
4. **The market is shared and public.** The opponent's farm is fully visible; their sells move
   your prices. This makes denial-selling, front-running and mirror-detection real strategies.
5. **The ladder is open-loop.** 22 public notebooks literally replay a recorded action tape
   (`TRACE_ACTIONS`); the top public agent is a zlib+base85 blob of `_ACTIONS` plus weed-repair
   and front-run patches. Kaito Fukami's framing (thread 734212) is the right one: *the submitted
   agent may be open-loop, but the research process must be closed-loop.*
6. **Rating is path-dependent.** Thread 734000: two byte-identical submissions landed 1,400 points
   apart. Thread 734074: a 92%-win-rate agent stuck at 2,450. Submission timing and early-game
   luck matter; this must be managed as an explicit policy, not discovered late.
7. **Measuring an edge is expensive.** A true 55% win rate needs ~380 games to exclude a coin flip
   (`analysis/nb_clean/busyaprime__how-many-games-to-rank-a-kaggriculture-agent.py`). Local
   evaluation without variance control will lie to us. This is why R1 (fast sim) is phase one.

---

## Architecture

```
kaggriculture/
├── CLAUDE.md                    # project instructions
├── ROADMAP.md                   # this file
├── competition.toml             # wikikit profile — already tuned, leave alone
├── wiki_hooks.py                # wikikit hooks
├── wiki/  analysis/             # generated by wikikit, gitignored
├── vendor/
│   └── kaggriculture.py         # pinned upstream engine source (ground truth)
├── src/kaggriculture/
│   ├── engine/                  # thin wrapper over kaggle_environments
│   ├── sim/                     # C++ port + pybind11 bindings  <- R1
│   │   ├── sim.hpp/.cpp         # rules, market curve, town, weeds
│   │   ├── bindings.cpp
│   │   └── validate.py          # step-for-step diff vs real engine
│   ├── model/                   # pure-Python domain model: CROPS, MARKET_PARAMS,
│   │                            #   price(inv), profit-per-action, town demand
│   ├── agents/                  # policies (baseline, scripted, scenario, adaptive)
│   ├── search/                  # LNS, CMA-ES, beam, DP sell-scheduler, EGTA  <- R2-R4
│   ├── eval/                    # arena, paired seeds, both-seat, CI reporting <- R7
│   └── submit/                  # bundle -> main.py / submission.tar.gz
├── experiments/exp-NNN-slug/    # one dir per run: config, logs, result.json
├── notes/                       # research log; one file per hypothesis
└── submissions/                 # frozen artifact + ladder outcome per submission
```

Key reuse (all public, all permitted under rules 2.6 / 3.6.b):
- `analysis/nb_clean/nikital7__4000x-environment-speedup-kaggriculture.py` — the trace-export
  ground-truth validator for a C++ port. Its `OPS`/`ITEMS`/`MOPS` enum ordering and its
  "emit the env's actual configuration, don't assume defaults" lesson are directly reusable.
- `analysis/nb_clean/bovard__kaggriculture-getting-started.py` — official starter (`melon_maxxer`),
  the correct observation-handling contract.
- `analysis/nb_clean/boatlee__v16-rc5-*.py` — the public reference lineage; read as a *target to
  beat and a source of tactical primitives* (`_weed_repair_action`, `_rank_sell_slots`,
  `_front_run`, `_terminal_liquidation`), not as a base to fork.
- `analysis/nb_clean/raykkretzschmar__kaggriculture-rank-your-agent.py` and
  `busyaprime__how-many-games-to-rank-a-kaggriculture-agent.py` — evaluation methodology.
- Kaggle dataset `kaggle/kaggriculture-episodes-index` — daily top-episode replays (up to 20 GB/day)
  for imitation learning and meta-tracking.

---

## Roadmap

### P0 — Foundation (Aug 16–18)

Get on the board immediately; path-dependence means a rating trajectory needs time to converge.

- [x] 1. `uv add kaggle-environments` (pin `>=1.32.7`), verify `make("kaggriculture")` runs.
- [x] 2. Vendor the engine source to `vendor/kaggriculture.py`, record the exact version and PR
      level. Point `competition.toml`'s commented `[reference]` section at it (unlocks the
      clone/novel machinery that currently reports `0 novel` for all 256 notebooks).
- [x] 3. Accept competition rules on Kaggle; verify `kaggle competitions list --group entered`.
- [x] 4. Build `src/kaggriculture/model/` from the vendored source — not from the docs. Thread
      732450 catalogues six places where the docs and engine disagree.
- [x] 5. Ship **submission #1**: a corrected, 1.32.7-calibrated melon+CARE heuristic. Purpose is a
      rating trajectory and an end-to-end submit path, not a good score. (`baseline-v1`, Kaggle
      submission 55582453, 2026-08-17.)
- [ ] 6. Fill the wikikit gaps: `wikikit synth discussions`, `wikikit synth leaderboard`,
      `wikikit players`, `wikikit index` (the digests `wiki/index.md` links to don't exist yet).
      Install `wikikit cron` so the leaderboard snapshots daily — one snapshot means no Δ signal.

### P1 — Instruments (Aug 18–26)

Nothing downstream is trustworthy without these.

- [x] 7. **R1: C++ simulator.** Port `kaggriculture.py` to `src/kaggriculture/sim/`, pybind11
      bindings. Acceptance gate: `validate.py` reproduces money + market-inventory trajectories
      **exactly** for ≥200 episodes across seeds × the `{starter, random, pass}` agent pairs. Any
      divergence is a bug in the port, never in the engine. Target ≥1,000x real-env throughput on
      20 cores. **Gate passed: 207/207 episodes, exact, 2026-08-17. The sim is trusted for P2/P3.**
      - [x] [007](issues/007-sim-core-port.md): the mechanical half — farm/tile/unit state, the
            full per-unit action set, deterministic day refresh, the turn-loop skeleton. Compiles
            clean; a standalone smoke check (not the parity gate) proves it links and runs.
      - [x] [008](issues/008-sim-market-town-port.md): market curve, order lockstep, HIRE/BUY_LAND,
            town demand, weed-spawn RNG. Includes a from-scratch bit-exact port of CPython's
            `random.Random` (`pyrandom.hpp`), verified against live Python output — needed because
            weed-spawn and shop-unlock draws share one RNG stream per day, which is also why 008
            doesn't integrate through 007's hooks (see 008's Revision note). Compiles clean;
            smoke checks (not the parity gate) prove the price table, buy/sell round-trip, HIRE
            cost curve, and town drain all match by-hand-traced vendor values.
      - [x] [009](issues/009-sim-bindings-runner.md): pybind11 bindings (`run_episode`,
            `run_batch`, `TapePolicy`, `CallbackPolicy`, `advance_turns` for branching search),
            wired into `pyproject.toml` (switched build backend to setuptools+pybind11 — no cmake
            on this machine). Verified in an isolated clean-checkout copy. Measured **~3,500x**
            single-thread / **~31,500x** at 20 threads over the real engine, 116 MB peak RSS for
            50k episodes — both acceptance numbers cleared by a wide margin.
      - [x] [010](issues/010-sim-parity-validator.md): `export_trace.py` + `validate.py`, the
            exact-parity gate this item's acceptance criterion depends on. **207/207 episodes
            passed exactly** (23 seeds × 9 `{starter,random,pass}` pairings, both seat orders,
            every step's money and market inventory). Verified the gate actually catches bugs:
            injected a real off-by-one into `sim.cpp` (weed threshold 2→3), confirmed the sweep
            caught and localized it to the correct step, reverted. Found along the way: vendor's
            `random_agent` seeds itself from OS entropy, not the episode seed — a `random`-
            involving trace isn't reproducible across separate exports (harmless here; worth
            knowing before 011/012 assume otherwise).
- [x] 8. **R7 (part 1): evaluation harness.** Paired-seed, both-seat, common-random-numbers arena
      with Wilson CIs and a sequential stopping rule. It must answer "is B better than A" with a
      stated error rate, and must refuse to answer when n is too small.
      [011](issues/011-eval-arena.md): `src/kaggriculture/eval/{stats,arena,agents}.py`. CRN turns
      out to mean "same episode seed for both arms," not independently-pinnable shop/weed streams
      — they share one RNG stream and the weed draw count depends on the policies' own actions
      (see 011's Revision note). Verified against real data: `baseline` vs `pass`/`starter`/
      `random` through this harness lands at ~26,140 vs 3,000 over a season, matching issue 005's
      real-engine submission numbers; SPRT decides `better` in 31 games (well under the ~380 a
      fixed-n 55%-effect test would need); byte-identical policies (including non-trivial ones —
      `baseline` vs `baseline`) correctly land at an undecided 50%-centered CI once ties score 0.5
      rather than counting as non-wins.
- [ ] 9. **Scenario taxonomy.** Enumerate the shop-draw distribution (8 instances, sampled with
      replacement from 8 shop types) and cluster it into a handful of demand regimes. This is the
      input to R6 and it did not exist before PR #1394.

### P2 — Offline optimization (Aug 26 – Sep 10) — the main scoring push

Run R2–R4 in parallel; they decompose cleanly.

- [ ] 10. **R2: production plan search.** Treat the 720-turn action sequence as the decision
      variable. Greedy constructive schedule → Large Neighbourhood Search (destroy/repair on
      day-blocks) → CMA-ES over a compact parametric policy (crew size per phase, crop mix, herd
      targets, quadrant purchase timing). 20 cores × a fast sim makes this the highest-EV line.
- [ ] 11. **R3: sell-schedule optimization as a separate problem.** Given a production plan and a
      model of town drain, choosing sell times/quantities is a near-separable DP over the price
      curve (`price(inv) = base ± amp·f(|inv − I0|)`, floored at $1). Solve it exactly rather than
      with the heuristics the public lineage uses. This is where the measured 24x lives (thread
      734412). Include **terminal liquidation**: unsold shed inventory scores zero, and at the $1
      floor units are not added to market inventory, so the floor stays responsive — endgame
      dumping has structure worth exploiting.
- [ ] 12. **R4: opponent coupling / EGTA.** Build a population of policies (ours across
      generations, plus reconstructions of the public lineage and the top-5 opening clusters from
      thread 733924), compute the empirical payoff matrix, and iterate best responses. Directly
      tests Kaito's "stable attractor" hypothesis and prevents overfitting to the current top-30 —
      his stated failure mode is losing to *older* meta generations still on the ladder.
- [ ] 13. **Submission cadence:** 1–2 subs/day from the frozen champion; never burn all 5.

### P3 — Closed-loop and robustness (Sep 10–23)

- [ ] 14. **R6: scenario-conditional policy** — the outside-the-box bet. Shop composition is drawn
      with replacement and revealed progressively (one unlock every 3 days, capped at 8).
      Precompute an optimized plan per demand regime offline, then switch regimes online as
      `unlocked_shops` reveals itself. Every open-loop tape on the ladder is committed to one
      regime; this is a structural edge that PR #1394 created and nobody has taken.
- [ ] 15. **R5: learning where it pays.**
      - Behaviour cloning of *openings* from the daily top-episodes dataset → an opening book,
        then hand off to search for the mid/endgame.
      - Evolution strategies on the fast sim (no GPU makes ES/CMA-ES strictly better than PPO
        here; thread 734952 is the community wrestling with the same constraint).
      - Inference-time rollout/MCTS over macro-actions, budgeted to 1.6 vCPU.
- [ ] 16. **Tactical primitives**, each validated as an ablation with the R7 harness:
      denial-selling (crash a premium good you have no exposure to, to deny opponent revenue);
      front-running the opponent's visible sell schedule; fertilizer-as-cash (animals produce it
      free and the engine *does* accept `SELL FERTILIZER`, contra the docs); wheat/fertilizer
      buy-low-sell-high carry against town drain; shed-capacity (100) overflow scheduling;
      4th-quadrant feasibility given a large hired crew; seat asymmetry in concurrent order
      processing.

### P4 — Freeze and select (Sep 23–30)

- [ ] 17. Stop new research Sep 25. Final week is portfolio selection only.
- [ ] 18. Only the **latest 2 submissions** are scored, so the last two uploads are the entire
      final entry. Choose a diversified pair (e.g. one scenario-adaptive, one robust open-loop)
      rather than two near-identical agents, and submit them early enough to accumulate episodes.
- [ ] 19. Ladder-timing policy driven by R7's path-dependence study.

### P5 — After the deadline (Oct 1–15)

- [ ] 20. Monitor the convergence period; publish the research writeups and the fast simulator.
      Candidate posts: the port + validation methodology, the 1.32.7 recalibration finding, the
      sell-schedule DP result, and the EGTA payoff-matrix study of the public meta.

---

## Research lines (index)

| | Line | Payoff | Risk |
|---|---|---|---|
| R1 | Exact C++ simulator + validator | Enables everything else | Port divergence; mitigated by the exact-trajectory gate |
| R2 | Open-loop plan search (LNS + CMA-ES) | Highest expected rank gain | Overfits to a fixed opponent |
| R3 | Sell-schedule DP against the price curve | Measured 24x revenue swing | Needs an accurate town-drain model |
| R4 | EGTA / best-response iteration | Robustness; tests the attractor hypothesis | Compute-hungry |
| R5 | BC opening book + ES; inference-time rollout | Novelty; the corpus has ~zero ML | No GPU; 1.6 vCPU at inference |
| R6 | Scenario-conditional regime switching | Structural edge no public agent has | Depends on regimes being separable |
| R7 | Ladder science: variance, path-dependence, timing | Makes every other result believable | Pure infrastructure, no direct points |

---

## Verification

- **Engine parity (P0/P1, hard gate):** `uv run python -m kaggriculture.sim.validate --seeds 23`
  — the C++ sim must reproduce per-step money and market inventory bit-for-bit against
  `kaggle_environments` for ≥200 episodes across seeds and `{starter, random, pass}` pairings.
  Non-exact output blocks all downstream work. **Passed 2026-08-17: 207/207.** Re-run after any
  change to `sim/` or any `kaggle-environments` version bump — a pass is a snapshot, not a grant.
- **Model parity:** unit tests asserting `model.price(item, inv)` and the profit-per-action table
  match values computed by the vendored engine, and that the `P(I0−T)/P(I0+T)/P(I0+2T)` column of
  the how-to-play price table reproduces exactly under 1.32.7.
- **Agent smoke:** `env.run(["submissions/<tag>/main.py", "starter"])` completes 720 steps with
  `status == "DONE"` for both seats and zero illegal actions in the logs, at both seats.
- **Statistical:** `eval.arena` on champion-vs-challenger reports a win rate with a Wilson CI that
  excludes 50% before anything is promoted or submitted.
- **Submission path:** upload, confirm the validation episode passes, then
  `kaggle competitions episodes <SUB_ID>` shows accruing games and
  `kaggle competitions leaderboard kaggriculture -s` shows the rating.
- **End-to-end:** rank trajectory across submissions, logged in `submissions/`, with the
  R7 path-dependence controls applied (same agent submitted at different times as a control).

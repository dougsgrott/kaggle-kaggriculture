# CMA-ES policy vs. the LNS plan

[015](../issues/015-cmaes-policy-tuning.md)'s acceptance criterion: "Beats the LNS plan under
`eval.arena` on a held-out seed set and against opponents not in the training portfolio, or the
negative result is written up here."

**Hypothesis**: a CMA-ES-tuned reactive policy (`search.policy` + `search.cmaes`, ~20 parameters,
opponent portfolio `pass,starter,baseline`, 194 evaluations) would beat issue 014's LNS-refined
plan on a disjoint holdout seed set.

**Result: disproved, at full sample size — but the story is more interesting than a flat no.**

| sample | result | Wilson CI | verdict |
|---|---|---|---|
| 20 holdout seeds (40 games) | 22W-18L-0T | `[0.398, 0.693]` | `undecided` |
| 50 holdout seeds (100 games) | 34W-66L-0T | `[0.255, 0.437]` | **`worse`** |

The 40-game sample alone would have supported "roughly even, leaning toward CMA-ES" — exactly the
kind of false signal the issue's own acceptance text warns about ("under `eval.arena` ... at full
sample size"). Only at 100 games does the CI clear 50% cleanly, and it clears in the *other*
direction: the CMA-ES policy reliably **loses** to the LNS plan, not wins.

**What the same run found instead, decisively:** the CMA-ES policy beats issue 005's baseline —
something neither 013's greedy schedule (0W-40L) nor 014's LNS-refined version of it (013's
4W-56L → 014's 10W-50L, still `worse`) managed. Holdout result: **32W-8L-0T** (40 games), Wilson CI
`[0.652, 0.895]`, verdict `better`. See [015](../issues/015-cmaes-policy-tuning.md)'s Revision
section for the full numbers, the parameters found, and why a *reactive* policy (this one) can
compete for MELON's tiny absorption pool in a way an *open-loop* plan (013/014's) structurally
can't — it sees the real, live price every turn instead of planning against a projected one.

**Working theory for why CMA-ES < LNS here, given CMA-ES > baseline:** the LNS plan is optimized
directly against `baseline` as part of its own portfolio, using a much larger, more targeted
search budget on a lower-dimensional problem (one 720-turn tape, not a policy that has to
generalize across arbitrary market states). The parametric policy's ~20 weights have to hold up
across every possible board state a live game can reach, which is a harder generalization target
than fitting one fixed tape — and 194 candidate evaluations is a small budget for a 20-dimensional
search (the issue's own "20-50 dimensions" framing implies substantially more). This is a
plausible, not confirmed, explanation; distinguishing "CMA-ES needs a bigger budget" from
"open-loop plans are structurally better suited to short episodes" would need a follow-up run at
10x this budget, which wasn't in scope for this issue's own time allowance.

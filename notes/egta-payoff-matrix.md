# Empirical payoff matrix over the policy population (issue 018)

Run: `uv run python -m kaggriculture.eval.egta --n-seeds 15 --n-public 10 --threads 8
--fp-iterations 500` — `experiments/exp-015-egta/` (`payoff_matrix.json`, `result.json`).
18 population members, 153 pairs, 30 games/pair (15 seeds, both seats), 1,361s total.

## 1. The population

**Our own lineage** (9 members, `eval.population.own_lineage_policies`): `pass`, `starter`,
`random`, `baseline` (005), `greedy013` (013's default schedule), `lns014` (014's LNS incumbent,
rerun with exp-005-lns's own recorded parameters so it's the same incumbent that experiment
reported), `dp016` (013's default schedule + 016's sell-timing DP), `cmaes015` (015's trained
reactive policy, params loaded from `exp-006-cmaes`), `current017` (016's schedule+sell-plan
wrapped by 017's terminal-liquidation layer — "our current best," but just one more population
member here, not privileged over retired generations, per Kaito Fukami's own argument in 734212).

**The public pool** (9 members): `analysis/nb_clean/` holds 256 wikikit-cleaned public
notebooks — vendored and public per CLAUDE.md. Issue 018's own scope asked to mine
`kaggle/kaggriculture-episodes-index` (a Kaggle dataset of daily top replays) to fingerprint
opening signatures; this machine has no Kaggle credentials configured, and since we already have
every candidate's actual *source*, running each one ourselves and reading its own engine-exact
state at turn 47 (crop/animal mix, hands, money spent, quadrants unlocked) is strictly more
reliable fingerprinting than reverse-engineering an opening from a replay would be — see
`eval.population`'s module docstring for the full reasoning. Of 103 statically-safe candidates
(defines `agent()`, doesn't import `requests`/`socket`/`subprocess` or touch `/kaggle/input`), 30
survived an actual smoke-test run (most failures are notebook cruft unrelated to the agent logic
itself — matplotlib backends, `IPython.display`, a self-validation cell that rewrites and
re-imports its own `main.py`; a stub-injecting import sandbox in `eval.population` recovers most
of these, but not all). The 30 survivors cluster into 9 distinct opening signatures (single-link
clustering, threshold 0.25 on the feature vector); one medoid representative per cluster, largest
clusters first. The largest cluster (7 near-identical notebooks) is exactly the "33 notebooks
share the same opening verbatim" lineage discussion 733924 describes; its representative,
`beicicc__kaggriculture-c26-kaito-v23-dual-regime`, turns out to matter a lot (§3).

**The vendored reference agent itself failed the smoke test.** `vendor/public-soil`'s own
`boatlee__v16-rc5-*.py` (issue 002's chosen reference) ends with a Kaggle-notebook-only cell that
reads back its own `main.py` from disk (`main_path.read_text()` before ever writing it) — a
`%%writefile`-cell artifact wikikit's cleaning couldn't reconstruct, not a bug in the agent logic.
It's excluded from this population for that reason. Its lineage is still represented: multiple
`beicicc__kaggriculture-c*` notebooks in the largest cluster are explicit derivatives of it
(`c22`-`c31` are literally named "exact-byte-control", "exact-reproducibility-control" — attempts
to reproduce it byte-for-byte), and one of them (`c26`) is this population's dominant strategy.

## 2. Population diagnostics

| | count |
|---|---|
| public candidates discovered (define `agent()`, no risky imports) | 103 |
| survived the smoke test | 30 |
| distinct opening clusters | 9 |
| population size (9 own + 9 public representatives) | 18 |

## 3. Is the meta transitive, cyclic, or converging?

**There is a dominant strategy: `beicicc__kaggriculture-c26-kaito-v23-dual-regime`** beats every
other one of the 17 other population members decisively, **100% win rate against all nine of our
own generations, including `current017`** ("our current best"). 800 of 816 possible triples were
fully decisive (all three pairwise verdicts landed `better`/`worse`, not `undecided`) — the
population is overwhelmingly transitive.

**But not entirely: an 8-triple rock-paper-scissors structure exists, and it is entirely inside
our own lineage plus two outside points.** Every cycle has the same shape:

```
{baseline, ayeshasummaiyya__fields-of-fortune-strategic-farming-agent}
    > {current017, dp016, greedy013, lns014}
    > cmaes015
    > {baseline, ayeshasummaiyya__fields-of-fortune-strategic-farming-agent}
```

i.e. `cmaes015` beats `baseline` and `ayeshasummaiyya` decisively (both 100%: `baseline vs
cmaes015` is 0.0 win rate for baseline, CI `[0.000, 0.114]`), each of `current017`/`dp016`/
`greedy013`/`lns014` beats `cmaes015` decisively (67-83% win rate for the schedule-based agent),
and `baseline`/`ayeshasummaiyya` each beat every one of those schedule-based agents decisively
(`baseline vs current017`: 93.3%, CI `[0.787, 0.982]`). This is a genuine, non-noise cycle, not an
artifact of thin sampling — every edge in it is individually decisive at 30 games.

## 4. The real finding: production-plan quality dominates sell-timing sophistication, and they've
   never been combined

The single most useful number in this matrix: **`lns014` (013's default schedule replaced by
014's LNS-optimized one, but with 013's own naive floor-threshold selling) beats `current017`
(013's default schedule, but with 016's sell-timing DP *and* 017's endgame liquidation layered on
top) 87% of the time** (`current017 vs lns014`: 0.133, CI `[0.053, 0.297]`, verdict `worse`). It
also beats `dp016` 100% of the time and `greedy013` 100% of the time.

This isn't a contradiction of 016's or 017's own reported results — both were measured against
*the same production plan on both sides* (016's Revision: "the ONLY difference between the two
agents is the sell strategy"), and under that controlled comparison the sell-timing layer wins
cleanly. What this matrix adds is the comparison 016/017 never ran: sell-timing sophistication on
013's plan **against** production-plan quality alone (014's LNS incumbent, no sell-timing at all).
Production quality wins by a wide margin. Issue 014's own Revision flagged this exact gap and
named the fix ("wire in 016's sell-DP as an operator... confirmed fast enough (0.27s/solve) but
not actually wired in") — this matrix is the first empirical evidence of how much that gap costs.
**The highest-leverage next experiment this repo has queued is not a new search line: it's running
016's sell-DP and 017's endgame layer on top of 014's LNS-optimized schedule instead of 013's
default one**, closing the loop between the three already-built pieces rather than building a
fourth.

## 5. What beats the current top public reference, and what it beats

`beicicc__kaggriculture-c26-kaito-v23-dual-regime` (the dominant strategy, §3) is undefeated in
this population. Nothing in this run beats it. It is the priority target for future search-line
opponent portfolios and for direct study (what does its "dual regime" logic actually do
differently) — a natural issue for whichever line picks up next.

One notch down, `beicicc__kaggriculture-c07-public-v12-tape` (the `public-v12-tape` cluster,
4 members) is beaten only by the other `beicicc` clusters (`c11-moon-sales`, `c22-exact-
reproducibility`, `c26-kaito-v23`) and beats everything else in the population, including
`baseline`, `cmaes015`, and all of our own schedule-based lineage.

`baseline` (issue 005, still our shipped agent) is beaten by six population members: four
`beicicc` clusters and `cmaes015`. It beats every other own-lineage member and three of the nine
public representatives (`anshu042006`, `dheerajkannaujiya`, `dreldirdirifadol`).

## 6. Fictitious play: converges, to the dominant strategy

500 iterations, starting from a uniform mix over all 18 members. The pure best-response sequence
converges to a fixed point (`beicicc__kaggriculture-c26-kaito-v23-dual-regime`, held for the last
50+ rounds) — expected once a matrix has a strict dominant strategy (verified structurally on a
synthetic 3-member case in `tests/eval/test_egta.py` before trusting it here): the best response
to *any* mix that still contains positive weight on beatable strategies is always the dominant
one, so it becomes self-reinforcing from the very first round. Estimated game value: 0.501 (a
balanced population overall, unsurprising with a single 18-0 dominant member and an otherwise
mostly-transitive structure). **This run does not test Kaito Fukami's stronger "stable attractor"
claim** (734212) — that would need generating genuinely new best-response strategies each round
(a "double oracle" with the oracle step), which this issue's budget didn't stretch to; what it
does confirm is his *weaker*, directly falsifiable claim that a repeated-best-response process
against a fixed population converges rather than cycling indefinitely — true here, cleanly.

## 7. Feeding this back

- **Opponent portfolios** (issues 014/015's `--opponents` flags, currently `pass,starter,baseline`
  by default): should include `beicicc__kaggriculture-c26-kaito-v23-dual-regime` and `cmaes015` at
  minimum — both decisively beat `baseline`, the sole non-trivial opponent those portfolios
  currently train against. See the pointers added to 014/015/016/017 for the specific follow-up.
- **The LNS+sell-DP composition gap** (§4) is the actual highest-leverage next step, not a new
  search line.
- **Population size and per-cell sample size are both deliberately modest** (18 members, 30 games
  a cell) given this issue's compute budget — a cell's `undecided` verdict (e.g. `dp016 vs
  greedy013`, 56.7% at `n=30`, CI `[0.392, 0.726]`) reflects genuinely limited statistical power
  at this sample size, not necessarily a true near-tie; a dedicated head-to-head at issue 016's own
  n=100 scale resolved that specific pair cleanly. Rerunning this matrix at a larger `--n-seeds`
  and/or `--n-public` is a legitimate next step if this line gets more budget, not a correction of
  anything wrong with the method itself.

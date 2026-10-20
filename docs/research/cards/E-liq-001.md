---
id: E-liq-001
mechanism: volume-flow; microstructure
timeframe: 1h
target: high-low
model_class: nig-timenet
reopens:
not_covered_by:
---
# Experiment card — E-liq-001: does telling the model how liquid the coin is (A) or removing illiquid coin-days (B) help the 1h high/low model?

Pre-registered 2026-10-19, status: **pre-registered, NOT run**. To be built and run AFTER N2 (the single `Normalizer`
interface); N1 (`price_norm_mode=logret_std`, PR #37) is already in. Rules: `docs/research/RESEARCH_RULES.md`,
`docs/research/MODEL_DEV_PROTOCOL.md`, `CLAUDE.md` ("One component, one interface"). Docs only: no code is written by this card.

Context (settled, not re-tested here): dead zero-volume tails are trimmed in Drive (trim-tails-001, 167/860 coins); long
padded mid-history gaps (>= 168 h flat zero-volume runs) are masked by `CONFIG['stale_bars']={'min_run_hours': 168}`
(stale-bars-002: 1h train zero-target share 2.48% -> 1.87%, val 2.61%, test 2.59%); short quiet stretches are kept (real
market state). NOT handled: **thin liquidity** — bars that move but on tiny volume, giving wide, noisy wicks (the
high/low targets then measure the lack of depth, not the market).

Two parts, two registered components, two separate experiments (one change each; never run together; the A+B
combination is a third experiment that needs its own card):

| Part | Component (registered, unknown names raise) | Kind |
|---|---|---|
| A | features `rel_volume`, `zero_vol_share_w` | model input (feature registry of the indicator/feature step, found by Grep at build time) |
| B | sample filter `min_liquidity_30d` | `register_sample_filter` entry (`data/windows.py:42`, `CONFIG['sample_filters']`) |

Both are **stateless causal functions of the coin's own past** (no fitted state, so nothing to leak and nothing to store
with the dataset). Anything with learned state that they feed (the scaling of `rel_volume` in the model input) goes through
the N2 `Normalizer` like every other feature; no second scaler is written here.

## Fixed definitions (numbers are fixed now, not tuned later)

- `qv_1h` = quote volume of the 1h bar (USD-like quote; `quote_volume` column). Bar time t = bar open; "strictly past"
  means information from bars with open time <= t-1h only.
- **A1 `rel_volume`** at bar t = `clip(log1p(qv_1h[t-1h] / M30[t-1h]), 0, log1p(20))` where `M30` = median of `qv_1h` over the
  720 bars (30 days) ending at t-1h, requiring >= 360 non-NaN bars (else NaN). The numerator is the last CLOSED bar relative to the
  prediction time and the median window excludes bar t. A zero `M30` gives NaN (such rows are already under B's rule or stale-bars).
  Clip constants 0 and log1p(20) ~ 3.045 are fixed here (ratio 20x median).
- **A2 `zero_vol_share_w`** at bar t = share of bars with `qv_1h == 0` among the T input-window bars (T = project default
  `window_sizes`, 32 at the time of writing), values in [0,1], no scaling.
- **B `min_liquidity_30d`**: for a coin-day D (UTC), `DM30(D)` = median over the 30 COMPLETED UTC days D-30..D-1 of the daily quote
  volume (sum of the day's 24 hourly `qv_1h`). Every sample whose target bar falls on day D is kept iff `DM30(D) >= 1,000,000`
  USD per day; fewer than 20 completed days of history, or NaN, means EXCLUDED (fail closed). The threshold is identical in
  train / val / test / live, and is fixed by the pre-step below; it is never tuned on val or test.
- **Common sample set (control for "one change")**: every run of this card — baseline and variants — uses only samples that
  have a defined `M30` (>= 360 non-NaN bars before t) AND a defined `DM30` (the first ~30 days of each coin are dropped for
  all runs). So A and B differ from the baseline only in the feature / the exclusion, not in a changed warm-up.

## Pre-step (descriptive, CPU only, run before the model; result recorded in the card's Result section)

Purpose: confirm the fixed threshold is a sensible cut, not to choose it. On the trimmed data + stale_bars 168 h, open coins only,
train period only for the fit-like statistics:
1. Distribution of `DM30` over coin-days: quantiles (1, 5, 10, 25, 50, 75, 90, 99%), per split and per year.
2. Share of coin-days, of samples, and of coins (fully excluded / partially excluded) removed at 250k, **1M**, 4M USD/day
   (descriptive sensitivity only; 1M is the registered value).
3. Zero-target share and NIG-floor hit rate (from the baseline run if available, otherwise the 1h zero-target share of the
   high/low reg targets) split by `DM30` bucket, to check the mechanism exists at all.
4. Why 1,000,000 USD/day (a priori, from market structure): ~42k USD per hour on average, i.e. about 700 USD per minute; in that regime a marketable order of a
   few thousand USD can already move price by tens of basis points (an assumption, to be checked by pre-step item 3), so wicks reflect
   depth rather than information. It is also a round number that sits well below liquid majors
   (hundreds of millions) so it does not select "only large caps".
- **Pre-registered validity guard** (decided now): B is *testable* iff, on train, the fraction of samples excluded by 1M lies in
  [3%, 30%] AND >= 100 coins keep >= 50% of their samples AND val and test each retain >= `min_split_samples` x 10. If the guard
  fails, part B is recorded "not testable at this threshold" and NO other threshold is substituted in this card (a new
  threshold = a new card with its own pre-step). Part A has no such guard other than non-degenerate features (feature std > 0 on train).

## Experiment table

| Field | Value |
|---|---|
| Date / author | 2026-10-19 / builder session (liq-card) |
| Baseline (commit, checkpoint, dataset fingerprint) | **Primary: the E-norm-001 winner** (logret_std or window_scale, whichever the accepted E-norm-001 verdict names). **If E-norm-001 fails or is "unexplained": `price_norm_mode=window_scale`** (the project default). Both are stated here so the choice is not made after seeing results; the baseline's commit, checkpoint and dataset fingerprint are filled in the first line of the Result section BEFORE the variant runs. Baseline re-run on the common sample set above, same seeds. |
| The ONE change | Experiment A: add exactly the two features A1+A2 to the model input (same N2 normalizer for them). Experiment B (separate run): add the `min_liquidity_30d` sample filter, identical on all splits. |
| Failure being addressed (level, location, registry id) | data / representation: the 1h high/low targets of thin coin-days are dominated by depth noise (wide wicks on tiny volume); no registry id yet (F-ids for volume-flow cover signal screens, not this); stale-bars handles only the dead/padded case |
| Data / splits / holdout | trimmed data (trim-tails-001) + `stale_bars` `min_run_hours=168`, open coins only, **no sealed bucket / holdout not opened**; splits = project default (`split_mode=global_time`, `train_pct=0.80`, `val_pct=0.10`, embargo = window + horizon, `holdout_start=None`). Same splits for baseline and both experiments. |
| Seeds / budget | >= 3 seeds (the same 3 for baseline, A and B, e.g. 100, 101, 102; add more if the spread requires, for all arms alike); epochs/hardware as the baseline config (Colab T4) |

## Verified cause and mechanism
- Verified cause this change attacks: **not yet verified** — the pre-step (item 3) and the liquidity-quintile report are the localisation
  test. Hypothesis H-A: the net cannot tell a thin bar from a normal one (the input has volume levels only in
  window-normalised form), so its high/low quantiles are too wide-and-noisy on thin coin-days. Hypothesis H-B: the thin coin-days
  are noise samples whose targets carry no tradable information; removing them raises label quality. Competing cause: the
  difficulty is plain volatility (NATR), not liquidity — tested by reporting the liquidity-quintile results after controlling for
  NATR_14 tercile.
- Why it should help: A gives the model the missing conditioning variable (relative activity / quietness) so it can widen quantiles
  where depth is thin and keep them tight elsewhere. B removes samples whose targets are depth-driven.
- Diagnostic that must move (predictions, layer = outputs + data):
  - **A**: pinball loss of the high and low quantile heads and the IC of the high/low reg heads **in the bottom DM30 quintile
    improve** vs baseline (high/low pinball down, IC up), while the **top quintile is unchanged** (within seed spread);
    permutation importance of `rel_volume` > its shuffled-column null; NIG floor-hit rate in the bottom quintile drops.
  - **B**: the zero-target share (1h, train/val/test) and the NIG alpha-floor hit rate **drop** vs baseline on the common set
    (prediction: zero-target share below the 1.87% train stale-bars-002 level, e.g. <= 1.2%; floor-hit rate lower), and the per-sample
    pinball / IC on the REMAINING samples is not worse. Null for all: seed spread (>= 3 seeds) and, for B, a random exclusion of the same
    number of samples (same coin-day counts) that must not produce the same movement.
- Output effect expected on val: A: high/low pinball -1% or better overall (the effect is concentrated in ~20% of the samples);
  B: IC of the high/low heads on val (measured on the SAME filtered val set for the baseline) +0.005 or better. Sizes are
  predictions, not thresholds; the accept rule below is the decision.

## Accept / reject rule (val; test only compares an accepted variant; sealed holdout not touched)
Each experiment (A, B) is judged alone against the baseline, on the common sample set; B is judged on the **filtered** val set for
both arms, and additionally reported on the unfiltered val set (what live would face if the coin is still traded: the filter's
cost is the coverage lost, reported as share of coin-days).
- **Accept** if: (a) the mean val high/low pinball (or the project's primary val metric for the head) improves by MORE than the seed spread
  (|mean difference| > 2 x the pooled seed std over >= 3 seeds, same sign in >= 2 of 3 individual seeds), AND (b) the predicted diagnostic
  moved beyond its null (A: bottom-quintile improvement exceeds top-quintile change and seed spread; B: zero-target share and floor-hit rate
  drop beyond seed spread and beyond the random-exclusion null), AND (c) no other layer-health metric degrades beyond spread (probe skill,
  calibration, gradient balance).
- **Reject** if: the output does not improve beyond seed spread, or it improves but the diagnostic did not move (then recorded as
  **unexplained**, not adopted), or the top quintile degrades beyond spread (A) / the remaining-sample metric degrades (B).
- B also requires the validity guard in the pre-step; if it fails, B is "not testable", not rejected, and no failure row is written
  (a note is added in the Result section).

## Failure handling
- A rejected part gets its own `docs/research/failure_registry.csv` row (append-only, byte-exact append, next free `F-` id) in the same
  commit as the result: scope tags `volume-flow;microstructure`, `1h`, `high-low`, `nig-timenet`; failure level and location
  (data / outputs by liquidity quintile), the verified cause (if the pre-step + NATR control support one; else `closed-unverified-cause`),
  invariants, and `reopen_if`, with candidate clauses: `R1: a liquidity measure with intraday resolution (order-book depth / spread,
  not derivable from OHLCV) | R2: the universe changes so the bottom-quintile DM30 share is > 2x the share measured in the pre-step |
  R3: the pre-step shows the effect only below a different threshold (new card)`.
- An "unexplained" result is recorded in the study log with artifact links and not adopted.
- The experiment never re-runs a closed failure: before building, `python tools/experiment_registry.py check` must print `card OK` (below).

---
## Result (added after the run)
- Pre-step output (DM30 quantiles, excluded shares, guard verdict):
- Baseline identity (E-norm-001 winner or window_scale; commit, checkpoint, dataset fingerprint):
- A: output metrics (mean +- seed spread); bottom/top quintile diagnostic; verdict:
- B: output metrics (mean +- seed spread); zero-target share, floor-hit rate, random-exclusion null; coverage lost; verdict:
- If rejected: registry row `F-____` added.

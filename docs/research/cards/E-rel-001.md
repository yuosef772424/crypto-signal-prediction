---
id: E-rel-001
mechanism: cross-sectional;representation
timeframe: 4h
target: relative-direction;return
model_class: nig-timenet;regression;ic-screen
reopens:
not_covered_by: F-0030=timeframe 1d and a direction-AUC question about encoders/capacity, whereas this card compares two TARGET definitions on the 4h line and judges them by per-timestamp cross-sectional rank IC of the regression heads (no encoder or capacity change); F-0031=that row closed a 1d market-neutral P&L construction (economics level, claim on pnl), this card makes no P&L claim and stops at the target/information level, and it reports the unclipped mean-vs-median jump-tail check that F-0031 identified as its verified cause; F-0033=repairs of the F-0031 rank portfolio by volnorm and stops on 1d (economics), none of which this card tries; F-0035=1d cross-coin attention panel (different model_class and architecture question), this card changes only the label
---
# Experiment card — E-rel-001: absolute vs within-timestamp relative target on the current 4h model line

Pre-registered 2026-10-20, BEFORE any run (**NOT run; the owner has not yet decided absolute vs relative, so nothing is
applied now**). Rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`. Answers the open owner
decision in `docs/research/plan/STATUS.md` ("absolute vs relative targets") and item 1 of
`docs/research/notes/external_research_2026-10-20.md` (citations there are unverified; none is relied on here).

## What is already known (so this is not a blind repeat)
- Relative targets were tried before on **1d** with NIG-TimeNet and closed at the **economics** level, not the
  information level: rank IC 0.093 was real (t = 11.9, permutation p = 0.001, train-test gap ~0.01), but the net daily
  rank portfolio at 0.08 % was <= 0 for every construction (F-0031, F-0033). Verified cause: the model ranks the
  **median** while mean returns are dominated by positive jump tails in the same coins (clipping at +-10 % flips the
  portfolio from -0.027 % to +0.228 %/day). Direction AUC ceiling ~0.55 on the 18 OHLCV features (F-0030).
- The hypothesis "the absolute `close` target itself causes the sign instability" was **partly refuted**
  (`close_target_sign_instability_synthesis.md`, note at the top): the "independent" features were proxies of one 24-day
  momentum, and the expected close improvement on the relative target did not appear. This card therefore does **not**
  predict that close becomes tradable; it predicts a narrower, falsifiable effect (below).
- Pilot on the owner's data, **exploration only, not evidence** (10 coins, 1d, 2020-2026, market = equal-weight mean of the
  10): the common market component explains 55 % of the variance of the absolute daily return; mean pairwise correlation
  0.55 (absolute) vs -0.10 (after removing the per-day mean); 24-day momentum -> next-day pooled correlation is positive
  in 4/7 years on the absolute target and 6/7 on the relative one, with a largest value of only 0.08. Cross-sectional
  rank IC is **identical** for both by construction (subtracting a per-timestamp constant never changes the ranking), so
  the two arms must be compared on a metric where training can differ: what the model learns, not how it is scored.

## Fields

| Field | Value |
|---|---|
| Date / author | 2026-10-20 / planner session (owner asked for the card; not run) |
| Baseline (commit, checkpoint, dataset fingerprint) | master at the commit that carries this card; `price_norm_mode='window_scale'` (byte-identical to the old pipeline, pinned by `tests/test_price_norm_mode.py`); same trimmed 4h dataset, window 32, stride 32, `align_windows_to_grid`, targets high/low/close, horizon 1, regression heads only; dataset fingerprint recorded in the bundle (`SYNCED_FROM.txt` per run) |
| The ONE change | `TargetSettings.target_mode`: `'return'` (arm A, baseline, absolute return anchored to the last same-type price as today) vs `'relative'` = `'return+relative'` (arm B: the same return minus the per-timestamp **median** over coins, existing `retarget_splits(center='median', min_group=5)`; nothing else changes: model, heads, losses, seeds, epochs, splits, clip) |
| Failure being addressed (level, location, registry id) | data/representation: the label of every coin at a timestamp carries a common market component (55 % of variance in the pilot) that no per-coin feature can explain, which lowers the effective independent label information (Kish: m / (1 + (m-1)·rho) with rho = mean cross-coin target correlation; ~3.5 effective coins of 200 at rho ~0.28, see the external-research note). No registry id yet: closes as a row only if rejected |
| Data / splits / holdout | as the E-norm-001 owner run (4h, ~200 coins, trimmed data + `stale_bars={'min_run_hours':168}`, rows with `y_close_reg == 0` kept, `global_time` split); the sealed buckets H1-H6 and the time seal (bars >= 2026-10-01) are **not** read; no holdout is opened by this card |
| Seeds / budget | >= 3 seeds per arm, the same seed list, ONE commit for all runs; epochs and hardware as the baseline's latest run (Colab), recorded in the bundle |

## Prerequisites (must exist before Stage 1; none is built by this card)
1. `E-norm-001` is decided (or this card runs on `window_scale`, as written). `price_norm_mode='logret_std'` **raises**
   with any `target_mode` other than the default (E-norm-001 implementation note), so combining the two needs its own
   `feat` card; it is out of scope here and listed as the follow-up if B wins.
2. The T2 dependence-aware comparison tool (STATUS "Model line"): per-timestamp paired differential d_t, block bootstrap
   over **whole timestamps** plus Newey-West, and a per-target temporal design effect. Without it the accept rule below
   cannot be evaluated and Stage 1 does not start.
3. Owner rule "one component, one interface": the relative transform must be a registered component
   (`fit` on train only -> state, `transform`, `inverse`). The existing `+relative` option is a stateless per-timestamp
   median. Its `inverse` (recomposing an absolute forecast) needs a **separate market forecast**, which does not exist.
   This card does **not** claim an absolute forecast from arm B; if B wins, consolidating the transform and defining the
   market-forecast component is a separate `refactor`/`feat` PR before any consumer outside this experiment.

## Verified cause and mechanism
- Verified cause this change attacks: **not yet verified** at the model level. What is measured: the common component
  share (pilot 55 %, 1d, 10 coins; the 4h ~200-coin value is Stage 0). The model-level claim "the shared component
  hurts learning" is the hypothesis under test, with the intervention being this card's single change (RESEARCH_RULES §4).
- Why the change should help: in arm A the loss on every sample is dominated by the shared market move, so the
  gradient mostly fits the day's market direction, which per-coin features cannot know; in arm B that component is
  gone and the remaining signal is cross-sectional (which coin does better than the others).
- Diagnostics that must move (null: the same statistic across the >= 3 seeds of arm A, plus the shuffled-label control):
  1. **Stage 0 (no training):** on the train split, share of target variance explained by the per-timestamp median falls
     from the pilot's ~55 % (to be re-measured at 4h) to < 5 %; mean cross-coin target correlation rho falls from ~0.3-0.5
     to <= 0.05; Kish effective coins per timestamp rises accordingly.
  2. **Stage 0 linear reference** (`simple_baseline`, ridge on the same features, same splits): val per-timestamp
     cross-sectional rank IC against the **relative** realised return is higher for the ridge fitted on the relative
     target than for the ridge fitted on the absolute target, and its share of calendar quarters with IC > 0 is higher.
     If this does not move, the shared component is not the bottleneck and Stage 1 is not run.
  3. **Stage 1 (model):** same two statistics for the NIG-TimeNet regression heads (high, low, close), plus the
     layer-wise linear probe against the relative return at the first layer and at the head input
     (`model/layer_report.py`): the head-input probe skill of B exceeds A's beyond the seed spread.
- Output effect expected on val: cross-sectional rank IC of the regression heads up by about 0.01-0.03 for B over A on
  the **relative** realised return (the metric both arms are scored on); high/low heads more than close (the close head
  carries most of the F-0030 ceiling). Close direction stays near chance.

## Accept / reject rule (val; fixed now)
Common scoring for both arms: per-timestamp cross-sectional Spearman IC between each head's prediction and the realised
**raw (unclipped)** return, then the mean over timestamps. Scoring on raw returns removes the arm-dependent label
definition from the comparison, and also from the F-0031 failure mode (median vs mean).
- **Accept** B if ALL hold: (a) mean IC of B exceeds A by more than the larger seed spread (max-min over seeds of either
  arm) on at least two of the three heads; (b) the T2 paired d_t block-bootstrap interval (whole timestamps,
  Newey-West) for B-A excludes 0 for the pooled head mean; (c) the share of val quarters with IC > 0 is >= 70 % for B and
  higher than A's; (d) the Stage 0 diagnostics moved as predicted; (e) the head-input probe moved as predicted.
- **Economics bound (reported with every result, never a P&L claim):** top-minus-bottom decile of B's score, gross
  mean per period on raw returns, unclipped AND clipped at +-10 %, against the 0.14 % round-trip taker cost at 1x and
  2x (ROADMAP contracts), plus the jump rate (>20 % next bar) by score decile. An accepted B whose unclipped gross is
  below cost is recorded as "information level only", and nothing is traded or forwarded to R6 on that basis.
- **Reject** if (a) fails, or the T2 interval includes 0, while the Stage 0 shared-component diagnostic did move (then
  the shared component was not the bottleneck: close the failure with a registry row at level `data`, with the verified
  cause and `reopen_if` naming a different information source), or Stage 0 diagnostic 2 did not move (Stage 1 never ran:
  close with the same row, `cause_test` = the Stage 0 table).
- Otherwise ("IC rose but the probe or Stage 0 diagnostic did not", or the reverse): **unexplained**, not adopted.
- Every variant counts toward multiple testing: the tests here are 2 arms x 3 heads, one pre-registered pooled rule; no
  other target variant (volnorm, rank, mean-centering, beta-neutral) is run under this card. Each would be its own card.

## Stages and cost
- **Stage 0 (cheap, no training, build session or Colab, minutes):** the target-dependence table and the ridge
  comparison above, on the train/val splits only; mean-centering vs median-centering reported as a **sensitivity row**
  (not a second arm). The script is used more than once, so it goes to `tools/` with a card and a test (CLAUDE.md).
- **Stage 1 (Colab, owner):** 2 arms x >= 3 seeds, only if Stage 0 diagnostic 2 moved.
- Not run now; the owner's go-ahead and the prerequisites decide when.

---
## Result (added after the run)
- Output metrics (mean +- seed spread):
- Diagnostic movement vs prediction (with null):
- Failure level and location, if failed:
- Verdict: accepted / rejected / unexplained — artifact links:
- If rejected: registry row `F-____` added (claim, scope tags, verified cause, invariants, reopen_if).

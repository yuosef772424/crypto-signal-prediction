---
id: E-tgt-002
mechanism: loss-objective;target-normalization;sample-weighting
timeframe: 4h
target: direction;high-low;return
model_class: nig-timenet
reopens:
not_covered_by: F-0033=F-0033 tested volnorm+RELATIVE (cross-sectionally demeaned) targets on a 1d market-neutral rank portfolio and found the raw-return ranking weakened, while this card uses the ABSOLUTE volnorm target (no cross-sectional centring) on the per-coin 4h model, judged by val IC with P&L only reported; F-0035=F-0035 closed the cross-coin attention PANEL model with a daily IC objective (one UTC day = one sample), while this card changes the loss and target of the MAIN per-coin NIG-TimeNet (one window = one sample, pooled batches), no panel and no attention; F-0031=F-0031 is an economics failure of a 1d relative rank portfolio dominated by jump tails, while this card is a 4h absolute-target model experiment accepted on val IC (not on a portfolio) and pre-registers the jump-tail risk of arm B clipping as a predicted failure mode (see Risks)
---
# Experiment card — E-tgt-002: IC-aligned objective for the main model (volnorm target, clip 4, Huber, IC + anti-shrink terms, magnitude weights)

Pre-registered 2026-10-31, BEFORE any run (NOT run). Rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`.
Four arms in a chain A -> B -> C -> D, each adds ONE change to the previous arm (B's change is the owner's bundle "target + its
point loss": the volnorm target needs a point loss in its units, so B is the target change with its Huber term).

| Field | Value |
|---|---|
| Date / author | 2026-10-31 / builder session (owner experiment E-tgt-002, owner refinements of the same day) |
| Baseline (commit, checkpoint, dataset fingerprint) | the branch carrying the four feat commits of this card (on `claude/crypto-project-hardening-yptgm9` f227042); arm A = `RunSettings` defaults (pipeline target `reg_target_mode='return'`, NIG loss, anti_memorization on) with close trained and the IC selection metric; dataset fingerprint recorded in each run bundle |
| The ONE change per arm | A (control): current default target + NIG loss. B = A + `target_mode="volnorm"` (return / window volatility, past-only: window IQR / last close, `workflow/retarget.py`) clipped at +-4 (`TargetSettings.clip=4.0`, default would be 10) + `huber_weight=0.1` on the three regression heads. C = B + `ic_weight=0.4` (-pearson(mu, y) over the batch) + `var_weight=0.1` ((std mu - std y)^2, anti-shrink) on the three regression heads. D = C + training sample weights `"abs_target_clip"`: w = clip(\|z\|, 0.2, 2), w = 0.2 for \|z\| < 0.3, from TRAIN labels, training loss only |
| Failure being addressed (level, location, registry id) | objective level, regression heads: the NIG point forecast mu is trained by a likelihood that can trade accuracy for uncertainty and is dominated by large-move rows; mu shrinks toward a constant ("cowardly model") so its ranking power is weak and unmeasured at selection time (best epoch chosen by `val_raw_loss`, the objective itself). No registry id |
| Data / splits / holdout | as E-norm-001 run by the owner: 4h, window 32, stride 32, `align_windows_to_grid`, targets high/low/close, horizon 1, indicators off (inputs H/L/C), trimmed data + `stale_bars={'min_run_hours':168}`, rows with `y_close_reg == 0` kept, `reg_target_scale=1`, `global_time` split, `price_norm_mode='window_scale'`; sealed holdout stays sealed. `suspended_targets=()` so close is trained (heads `close_reg` + `close_class`) |
| Seeds / budget | seeds 0, 1, 2 per arm (same list in every arm; `TrainSettings.seed`), one run folder per arm and seed (`train.run_dir`; the identity check refuses to resume a changed objective and ignores the seed, so folders must differ); epochs 60, patience 15 (defaults), Colab GPU; 12 runs |

## Settings of each arm (the run is `workflow.run.run_main(settings, kit)` / main.ipynb with `settings = arm(name, seed)`)

```python
HEADS = ("high_reg", "low_reg", "close_reg")

def arm(name, seed):
    target = {"suspended_targets": ()}                       # train close too
    loss = {"report_ic": ("high_reg", "low_reg")}            # log val_high_reg_ic / val_low_reg_ic besides the monitor
    if name in "BCD":
        target.update(target_mode="volnorm", clip=4.0)
        loss["head_terms"] = {h: {"huber_weight": 0.1} for h in HEADS}
    if name in "CD":
        loss["head_terms"] = {h: {"huber_weight": 0.1, "ic_weight": 0.4, "var_weight": 0.1} for h in HEADS}
    if name == "D":
        loss["sample_weighting"] = "abs_target_clip"
    return RunSettings().updated({"target": target, "train": {
        "seed": seed, "run_dir": f"/content/drive/MyDrive/training_runs/E-tgt-002_{name}_s{seed}",
        "early_stopping_monitor": "val_close_reg_ic", "early_stopping_mode": "max", "loss": loss}})
```

All arms select the best epoch by the SAME objective-independent metric `val_close_reg_ic` (Pearson of close mu vs label over the
full val set, `stats.metrics.pearson`; registered `objective_independent=True`), so a loss change cannot move the selection rule.
Note: in A the labels are returns, in B-D volnorm units; Pearson is scale-free per target but the label sets differ, so the
comparison metrics below are recomputed on ONE common label (raw next-bar return) from each arm's predictions.

## Verified cause and mechanism
- Cause attacked (hypothesis, not yet verified by a dedicated test): (1) raw-return labels are heavy-tailed across coins and
  time, so the NLL / mean of a pooled batch is driven by high-volatility rows; (2) the NIG likelihood lets the head lower its loss
  by widening beta instead of moving mu; (3) nothing in the objective rewards ranking. The first is attacked by B (vol units,
  clip), the second by B's Huber term on mu, the third by C (IC term, with the anti-shrink term so the scale-free IC term cannot
  collapse the spread of mu), and the residual weight of noise-sized moves by D.
- Why the changes should help: IC is invariant to the scale of mu, so a direct correlation term moves the ranking that the
  owner's metric policy judges (high/low by IC, close by direction), while the variance term keeps std(mu) near std(y).
- Diagnostic that must move ("shrinkage" diagnostic, recorded for EVERY arm): `std(mu) / std(y)` per head on val (computed on the
  model's own target units), and the share of rows with \|mu\| below 0.1 std(y). Null: the same ratio across the 3 seeds of A
  (seed spread) and on the epoch-0 (untrained) model.
- Predicted movement per arm (val):
  - B vs A: val IC up for close AND high/low; tail domination down: the share of the total squared error coming from the top 1%
    \|y\| rows falls; the shrinkage ratio roughly unchanged.
  - C vs B: val IC up further; shrinkage ratio std(mu)/std(y) up (toward 1).
  - D vs C: IC on the subset \|z\| > 0.3 up (the full-set IC may stay flat); IC on \|z\| < 0.3 may fall (expected, not a failure).
- Output effect expected on val: close IC +0.01 to +0.03 for B and again for C over the previous arm; high/low IC up by a similar
  amount; close direction still near chance (AUC 0.51-0.54; F-0031 measured close AUC 0.543 on 1d relative targets).

## Metrics reported per arm (val; mean +- seed spread; owner metric policy, `stats.metrics.evaluate_heads`)
- close: direction (AUC / hit rate of sign(mu) and of the class head) and IC; high/low: rank-IC, MAE / RMSE and skill vs the naive
  zero-change prediction, interval coverage TOGETHER with mean width (NIG Student-t intervals).
- The shrinkage diagnostic above, per head.
- Trading (reported only, NOT used to accept): close signal, one fixed \|pred\| threshold chosen on val (top-30% \|mu\| on val), net
  of 0.08% round trip per trade: profit factor, avg_win / avg_loss, net Sharpe (`stats.metrics.threshold_trades`, `profit_factor`,
  `win_loss_ratio`, `net_sharpe`). Threshold walk-forward selection waits for PR12.

## Accept / reject rule (val)
- Paired test: per val timestamp, the cross-sectional rank-IC of close mu vs the raw next-bar return (`stats.metrics.spearman` over
  the coins of that timestamp), seed-averaged predictions per arm; `tools/paired_compare.py --loss precomputed` with the per-row
  loss = minus that timestamp's IC, Newey-West lags `auto_nw`, stationary block bootstrap (n_boot 2000, seed 0).
- Accept arm X over its predecessor if: mean val close IC (over seeds) exceeds the predecessor's by more than the larger seed spread
  (max-min over seeds of either arm) AND the paired test is significant (NW p < 0.05, d_t favouring X) AND the predicted diagnostic
  of that arm moved beyond its null (B: tail share of squared error down; C: shrinkage ratio up; D: IC on \|z\| > 0.3 up).
- Reject if the IC difference is within seed noise or negative (close the failure with a registry row naming which arm and why),
  or if the diagnostic moved the wrong way.
- Otherwise ("IC up but diagnostic flat", or vice versa): "unexplained", not adopted.
- High/low IC is reported with the same paired test but is a secondary criterion (it cannot accept an arm alone).
- Net-of-cost P&L is reported, never used to accept.

## Risks (pre-registered)
- F-0031 (median vs jump tails): clipping at +-4 vol units removes exactly the jump tail that dominated mean returns in F-0031.
  Prediction: B can raise IC (median ranking) while net P&L on raw returns does NOT improve or falls; the jump-rate by mu decile
  (share of next-bar \|z\| > 4 per decile) is reported for every arm to see it. If B's IC rises and its P&L falls, that is this
  risk, not a contradiction.
- F-0033: a volnorm target weakened the raw-return ranking in its relative form; here the target is absolute, and the comparison
  is on raw-return IC precisely to detect the same weakening.
- The IC term works on pooled batches mixing coins and timestamps (batches carry no timestamp key; a per-timestamp variant needs
  data plumbing): it rewards time-series AND cross-sectional ranking together.

## Deferred (not adopted in this card)
- A direct profit loss -(tanh(out) * r): the F-0031 jump tails would dominate its gradient.
- An asymmetric quantile loss for high/low.
- A per-timestamp (cross-sectional) IC term: batches carry no timestamp key today.
- Rank-IC (Spearman) as the selection metric: Pearson over the full val set is used (stats.metrics.pearson); a Spearman
  version needs an epoch-end computation (n log n per step is too slow inside the tracker).

---
## Result (added after the run)
- Output metrics (mean +- seed spread):
- Diagnostic movement vs prediction (with null):
- Failure level and location, if failed:
- Verdict: accepted / rejected / unexplained — artifact links:
- If rejected: registry row `F-____` added (claim, scope tags, verified cause, invariants, reopen_if).

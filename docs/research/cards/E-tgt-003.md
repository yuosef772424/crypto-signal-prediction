---
id: E-tgt-003
mechanism: loss-objective;target-normalization;sample-weighting
timeframe: 4h
target: direction;high-low;return
model_class: nig-timenet
reopens:
not_covered_by: F-0033=F-0033 tested volnorm+RELATIVE (cross-sectionally demeaned) targets on a 1d market-neutral rank portfolio, while this card uses the ABSOLUTE vol-normalised log return (volret, no cross-sectional centring) on the per-coin 4h model and adds proper-scoring, decision-utility and curriculum factors that F-0033 never touched; F-0035=F-0035 closed the cross-coin attention PANEL model with a daily IC objective (one UTC day = one sample), while this card changes the loss and target of the MAIN per-coin NIG-TimeNet (one window = one sample, pooled batches), no panel and no attention; F-0031=F-0031 is an economics failure of a 1d relative rank portfolio dominated by jump tails, while this card is a 4h absolute-target model experiment whose accept rules are paired bootstrap CIs on IC and net Sharpe with the all-sample P&L (jumps INCLUDED) always reported, and it pre-registers the jump-tail risk (see Risks)
---
# Experiment card — E-tgt-003: proper scoring, decision utility, confidence and curriculum on top of the E-tgt-002 volret line

Pre-registered 2026-11-01, BEFORE any run (NOT run). Rules: `docs/research/RESEARCH_RULES.md`, `docs/research/MODEL_DEV_PROTOCOL.md`.
Staged one-factor-at-a-time arms S0 -> S1 -> S2 -> S3 -> S4 (and the side arms S5, S6): each arm adds ONE change to its named predecessor,
>= 5 seeds per arm, the SAME seed list in every arm. Every arm (every seed of every arm) counts toward the number of trials.

Owner research adopted here: proper scoring rules instead of raw evidential NLL (beta-NLL; the NIG predictive IS a Student-t), a confidence
head trained on DETACHED inputs and targets (no abstain head: it collapses to "fear of predicting"), a decision-utility term MIXED with the
standard losses and ramped from 0 (never alone), fixed loss weights on EMA-normalised losses (no Kendall on a variance head), an
easy-to-hard curriculum by |move|/vol (never focal / hard mining on noisy labels), anti-fear diagnostics.

## What is already known, and why this is not a repeat
- E-tgt-002 (card `docs/research/cards/E-tgt-002.md`, code merged, NOT run when this card was written): target `volnorm` (return / window
  IQR vol) + Huber + IC + variance terms + magnitude weights. This card does not repeat it: it changes the target definition to `volret`
  (log return / EWMA std of the window's past close log returns, decodable on ANY dataset and through the TargetCodec), the balancer
  (ema_norm), the likelihood weighting (beta-NLL), and adds a confidence head, a utility term and a curriculum. E-tgt-002's results, when
  they exist, are the S0 reference only if run on the same seeds.
- F-0031 / F-0033 (1d relative rank portfolio; jump tails dominate mean returns; volnorm+relative did not repair it) and F-0035 (panel
  model, closed): different unit of a sample, different target and no loss changes; see `not_covered_by`.

## Fields

| Field | Value |
|---|---|
| Date / author | 2026-11-01 / builder session (owner research summary of the same day) |
| Baseline (commit, checkpoint, dataset fingerprint) | the branch carrying the E-tgt-003 feat commits; S0 below; dataset fingerprint recorded in each run bundle |
| The ONE change per arm | S0 baseline. S1 = S0 + beta-NLL (beta = 0.5) on the three NIG heads. S2 = S1 + the confidence head ("tcp"). S3 = S2 + the utility term (lambda_U in {0.1, 0.25}, ramped). S4 = best of S3 + curriculum_snr, vs its control curriculum_random. S5 = best + jump_downweight (+ the event input features when they exist). S6 = best (same seeds) + the uncertainty-gated skip residual on mu |
| Failure being addressed (level, location, registry id) | objective level, regression heads: (1) the NIG likelihood trades accuracy for uncertainty on heavy-tailed rows; (2) uncertainty and confidence are not decision-grade (reported sigma larger than the move; confidence supervised for |error| magnitude, not for "direction is right"); (3) nothing in the objective is the decision the model feeds. No registry id yet |
| Data / splits / holdout | as E-tgt-002: 4h, window 32, stride 32, `align_windows_to_grid`, targets high/low/close, horizon 1, indicators off (inputs H/L/C: `close` must be an input, volret decodes the window from it), trimmed data + `stale_bars={'min_run_hours':168}`, `reg_target_scale=1`, `global_time` split, `price_norm_mode='window_scale'` (volret also runs on `logret_std`); sealed holdout stays sealed; `suspended_targets=()` so close is trained |
| Seeds / budget | seeds 0-4 (>= 5) per arm, identical in every arm (`TrainSettings.seed`); one run folder per arm and seed `/content/drive/MyDrive/training_runs/E-tgt-003_<arm>_s<seed>`; epochs 60, patience 15, Colab GPU; arms S0, S1, S2, S3 x2, S4, S4-control, S5, S6 = 10 arms x 5 seeds = 50 runs |

## Settings of each arm (the run is `workflow.run.run_main(settings, kit)` / main.ipynb with `settings = arm(name, seed)`)

```python
HEADS = ("high_reg", "low_reg", "close_reg")

def arm(name, seed, lam_u=0.1):
    """name in S0 S1 S2 S3 S4 S4c S5 S6; each arm = the previous one's loss plus ONE entry. Early stopping on val_close_reg_ic (max)."""
    terms = {h: {"huber_weight": 0.1} for h in HEADS}                           # S0: volret clip 4 + Huber on mu
    loss = {"report_ic": ("high_reg", "low_reg"), "balancing": "ema_norm"}      # fixed weights on EMA-normalised head losses
    order = ["S0", "S1", "S2", "S3", "S4", "S4c", "S5", "S6"]
    has = lambda a: order.index(name) >= order.index(a)
    if has("S1"):
        terms = {h: {"huber_weight": 0.1, "beta_nll": 0.5} for h in HEADS}      # beta-NLL on the NIG (Student-t) likelihood
    loss["head_terms"] = terms
    # S2: the "tcp" confidence head (registered head type) -- REQUIRES the tcp feat commit, NOT in this branch (see "Not built")
    if has("S3"):
        loss["utility"] = {"weight": lam_u, "warmup_epochs": 10, "ramp_epochs": 10, "cost": 0.0008}
    if name == "S4":
        loss["sample_weighting"] = "curriculum_snr"
    if name == "S4c":
        loss["sample_weighting"] = "curriculum_random"                          # the control: same sizes, random order
    if name == "S5":
        loss["sample_weighting"] = "jump_downweight"                            # k = 6 vol units, weight 0.3, never dropped
    # S6: model.config_overrides {"mu_skip": "unc_gated"} -- REQUIRES the mu_skip feat commit, NOT in this branch
    return RunSettings().updated({
        "target": {"suspended_targets": (), "target_mode": "volret", "clip": 4.0},
        "train": {"seed": seed, "run_dir": f"/content/drive/MyDrive/training_runs/E-tgt-003_{name}_s{seed}",
                  "early_stopping_monitor": "val_close_reg_ic", "early_stopping_mode": "max", "loss": loss}})
```

`tests/test_e_tgt_003_arms.py` builds and trains two epochs of the arms that exist in this branch (S0, S1, S3, S4, S4c, S5) on the synthetic
fixture. Every arm selects the best epoch by the SAME objective-independent metric `val_close_reg_ic`; since the fix of audit D1 the EMA
weights that get saved are the ones re-evaluated on val (so the restored best equals the recorded best).

## Verified cause and mechanism
- Cause attacked (hypotheses, not yet verified by a dedicated test): the raw NIG NLL is dominated by high-variance rows (S1); the
  confidence output is supervised for the magnitude of the error, which is mostly noise (S2); the objective never rewards the decision
  (S3); noise-level moves carry the same weight as clear ones early in training (S4); event jumps (news, pumps, hacks) dominate squared
  errors (S5).
- Diagnostic that must move (every arm, val, per head, computed in return space through the TargetCodec): `std(mu)/std(y)` (anti-fear),
  median sigma/|y| and the coverage of mu +- 1.645 sigma (`stats.metrics.sigma_calibration`), CRPS of the Gaussian summary
  (`stats.metrics.crps_gaussian`), the share of |position| > 0.1 (`position_share`), confidence mean / std / share < 0.55 and the
  risk-coverage AUC (`risk_coverage_auc`), the share of samples at each NIG floor (nu_min 0.1, alpha_min 2, beta_min 0.01).
- Predicted movement (val, vs the predecessor): S1 coverage of the 90% interval closer to 0.90 and CRPS down with IC not worse; S2 risk-coverage
  AUC down (confidence ranks mistakes) with direction accuracy not worse; S3 net Sharpe up, turnover not exploding, MSE up a little
  (expected, the price of the utility); S4 IC up vs the control with the same sizes (the order, not the size, is the effect); S5 IC / Sharpe
  on non-jump samples up with all-sample P&L not worse; S6 IC up with std(mu)/std(y) not collapsing.

## Metrics reported per arm (val; mean +- seed spread)
close: direction (AUC / hit rate of sign(mu)), IC and rank-IC; high / low rank-IC; interval coverage at 90 / 95% WITH width, CRPS, winkler
(`stats.metrics.evaluate_heads`); the diagnostics above; net P&L at 0.08% round trip with one |pred| threshold fixed on val
(`threshold_trades`, `net_sharpe`, turnover, profit factor) on ALL samples INCLUDING jumps (F-0031: jump tails dominated mean returns), and
the same metrics excluding samples with |future return|/vol > 6 (not built into the evaluation yet, see "Not built").

## Accept / reject rules (val; every arm vs its named predecessor; paired by seed)
- Paired bootstrap (`tools/paired_compare.py --loss precomputed`, per-timestamp loss = minus that timestamp's rank-IC for IC claims, the
  per-timestamp net P&L for Sharpe claims; stationary block bootstrap n_boot 2000, seed 0, Newey-West lags `auto_nw`): the CI of the
  difference must EXCLUDE 0 in the predicted direction, AND the arm must win in a MAJORITY of seeds (>= 3 of 5), AND the predicted
  diagnostic of the arm moved beyond its null (S1: coverage error and CRPS; S2: risk-coverage AUC; S3: net Sharpe with CI excluding 0,
  turnover and the MSE rise reported; S4: IC against the CONTROL S4c, not only against S3; S5: non-jump IC/Sharpe), AND no significant
  degradation of calibration (coverage error) or direction accuracy (paired CI of the accuracy difference includes 0 or is positive).
- Reject if the CI includes 0, or the wrong diagnostic moved, or the majority of seeds disagree; close the failure with a registry row
  naming the arm and the cause. "IC up but diagnostic flat" = unexplained, not adopted. P&L alone never accepts an arm; for S3 only the net
  Sharpe CI is the primary claim and the all-sample P&L must not be worse than S2's.
- S3 compares lambda_U 0.1 and 0.25 as two arms of the same stage; the better one (by the S3 rule) goes on; both count as trials.
- Seeds: no arm is accepted on one seed; seed noise = the max-min over seeds of the SAME arm.

## Risks (pre-registered)
- F-0031 (jump tails): the clip at +-4 vol units and the jump down-weight remove exactly the tail that dominated mean returns; IC may rise
  while all-sample P&L falls. The jump-rate by mu decile and the P&L including jumps are reported for every arm; if IC rises and P&L falls
  that is this risk, not a contradiction.
- Utility term: it trains a position from a noisy mu / sigma; the cost is in return units (0.0008) and the term is normalised by std(r), so
  lambda_U is comparable to O(1) head losses; watch the share of |position| > 0.1 (it can collapse to 0 = the fear it is meant to cure).
- The ramp starts after warmup_epochs = 10; with patience 15 an early stop before the ramp would silently test S2 again: the run bundle
  records the epoch of the best checkpoint, and a best epoch below the ramp start voids the S3 comparison for that seed.
- ema_norm changes every head's gradient share relative to the kendall default: S0 already uses it, so S0 is NOT the E-tgt-002 arm A/B.

## Not built in the branch that carries this card (arms that cannot run yet)
S2 (tcp confidence head), S6 (uncertainty-gated mu skip) and the event input features of S5 are not implemented here; their arms stay
pre-registered but are blocked until their feat commits land. The dedicated Gaussian / Student-t regression heads were not built either:
S1 uses the NIG likelihood with the beta-NLL term (`head_terms[...]["beta_nll"]`), whose predictive distribution is a Student-t.
Evaluation reporting of the confidence quality, of every metric with and without jump samples, and of the post-hoc sigma-scale calibrator is
not wired into `evaluate_heads` yet (the pure metric functions exist in `stats.metrics`).

---
## Result (added after the run)
- Output metrics (mean +- seed spread):
- Diagnostic movement vs prediction (with null):
- Failure level and location, if failed:
- Verdict: accepted / rejected / unexplained — artifact links:
- If rejected: registry row `F-____` added (claim, scope tags, verified cause, invariants, reopen_if).

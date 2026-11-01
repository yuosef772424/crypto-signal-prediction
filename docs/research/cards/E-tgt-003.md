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
ORDER = ["S0", "S1", "S2", "S3", "S4", "S4c", "S5", "S6"]

def arm(name, seed, lam_u=0.1):
    """name in S0 S1 S2 S3 S4 S4c S5 S6; each arm = the previous one's settings plus ONE entry (S4c = S3 + the control of S4, S5/S6 stack on the
    BEST of S4/S4c per the accept rule: pass `base` = that arm's name to build them on it). Early stopping on val_close_reg_ic (max), epochs 60, patience 15."""
    has = lambda a: ORDER.index(name) >= ORDER.index(a)
    terms = {h: {"huber_weight": 0.1} for h in HEADS}                            # S0: volret clip 4 + Huber on mu
    loss = {"report_ic": ("high_reg", "low_reg"), "balancing": "ema_norm"}       # fixed weights on EMA-normalised head losses
    model = {}
    if has("S1"):
        terms = {h: {"huber_weight": 0.1, "beta_nll": 0.5} for h in HEADS}       # S1: beta-NLL on the NIG (Student-t) likelihood
    loss["head_terms"] = terms
    if has("S2"):                                                                # S2: the TCP confidence head (Brier, after 10 warm-up epochs)
        model["config_overrides"] = {"tcp_head": True}                           # model builds the head tcp_close (on stop_gradient trunk features)
        loss["tcp"] = {"weight": 0.1, "warmup_epochs": 10}                       # RunSettings refuses one without the other
    if has("S3"):                                                                # S3: the decision-utility term, lam_u in {0.1, 0.25}
        loss["utility"] = {"weight": lam_u, "warmup_epochs": 10, "ramp_epochs": 10, "cost": 0.0008}
    if name == "S4":
        loss["sample_weighting"] = "curriculum_snr"
    if name == "S4c":
        loss["sample_weighting"] = "curriculum_random"                           # the control: same sizes, random order
    if name == "S5":
        loss["sample_weighting"] = "jump_downweight"                             # k = 6 vol units, weight 0.3, never dropped (+ the data line below)
    if name == "S6":                                                             # S6: the gated skip residual on mu (compare with the SAME arm without it)
        model["config_overrides"] = {**model.get("config_overrides", {}), "mu_skip": "unc_gated"}
    return RunSettings().updated({
        "model": model,
        "target": {"suspended_targets": (), "target_mode": "volret", "clip": 4.0},
        "train": {"seed": seed, "epochs": 60, "early_stopping_patience": 15,
                  "run_dir": f"/content/drive/MyDrive/training_runs/E-tgt-003_{name}_s{seed}",
                  "early_stopping_monitor": "val_close_reg_ic", "early_stopping_mode": "max", "loss": loss}})

# S5 data line (event features known at entry; default None = off; the dataset is rebuilt WITH them for S5 only, `data.CONFIG`):
#   data.CONFIG["custom_settings"] = {**data.CONFIG.get("custom_settings", {}),
#                                     "event_features": {"spike_z": 3.0, "ret_windows": [8], "gap_k": 3.0}}   # -> EVT_VOL_SPIKE, EVT_MAXRET_8, EVT_GAP
# usage: for name in ORDER: for seed in range(5): run_main(arm(name, seed), kit)      # 10 arms (S3 at lam_u 0.1 and 0.25) x 5 seeds = 50 runs
```

`tests/test_e_tgt_003_arms.py` builds and trains two epochs of the arms on the synthetic fixture. Every arm selects the best epoch by the SAME
objective-independent metric `val_close_reg_ic`; since the fix of audit D1 the EMA weights that get saved are the ones re-evaluated on val
(so the restored best equals the recorded best).

## Built in the branch that carries this card (and what is still missing)
- S2: `LossSettings.tcp` + `model config tcp_head` (`tests/test_tcp_head.py`): an MLP on stop_gradient(trunk features) -> output `tcp_close` =
  P(sign(mu_close) == sign(y_close)), Brier loss against stop_gradient(1[mu * y > 0]), weight 0 until `warmup_epochs`. Confidence quality
  is read with `evaluation.integrity.confidence_quality_report(preds, y_close)` (risk-coverage AUC and AUROC of the OLD confidence output
  and of TCP; pure function `stats.metrics.confidence_quality`).
- S5 event inputs: `data/custom.py` setting `event_features` (EVT_VOL_SPIKE, EVT_MAXRET_w, EVT_GAP; default None = off; `tests/test_event_features.py`).
- S6 (`mu_skip="unc_gated"`) is judged against the SAME arm without it, same seeds (val IC, std(mu)/std(y), coverage, the gate distribution
  `unc_gate_model(model)` and its correlation with |error|; the gate reads stop_gradient(epistemic), so it cannot push epistemic).
- S1 is additionally judged on median sigma/|y| and the coverage of mu +- 1.645 sigma BEFORE and AFTER the post-hoc `sigma_scale` calibrator
  (`evaluation.calibrators.SIGMA_CALIBRATORS`, fitted on the calibration split only).
- Not built: the dedicated Gaussian / Student-t regression heads (S1 uses the NIG likelihood with the beta-NLL term, whose predictive is a
  Student-t); the evaluation of every metric with and without jump samples and the full anti-fear diagnostics are not wired into the
  evaluation run (the pure functions exist in `stats.metrics`).

---
## Result (added after the run)
- Output metrics (mean +- seed spread):
- Diagnostic movement vs prediction (with null):
- Failure level and location, if failed:
- Verdict: accepted / rejected / unexplained — artifact links:
- If rejected: registry row `F-____` added (claim, scope tags, verified cause, invariants, reopen_if).

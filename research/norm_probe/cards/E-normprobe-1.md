---
id: E-normprobe-1
mechanism: representation
timeframe: 1h
target: direction;high-low
model_class: regression;logistic
reopens:
not_covered_by: F-0029=no neural training and no InstanceNorm layer here, a CPU-only static probe of the data-side input normalization, not the model-side per-window InstanceNorm that F-0029 closed; F-0056=symbolic tokens and wavelets as an information source, whereas this card compares numeric normalizations of the same H/L/C bars with linear probes; F-0063=chart-image CNN input, whereas this card has no image and no neural net
---
# Experiment card — E-normprobe-1: do differenced (log-return) inputs with Min-Max / Standardization keep more distinct samples and the same linear signal as the current window_scale input?

Pre-registered 2026-10-19, BEFORE any variant is computed (rules: `docs/research/RESEARCH_RULES.md`,
`docs/research/MODEL_DEV_PROTOCOL.md`). Static data-side probe: no network is trained.

| Field | Value |
|---|---|
| Date / author | 2026-10-19 / builder session (norm-diff-probe) |
| Baseline (commit, checkpoint, dataset fingerprint) | master cf395ff (`data/windows.py` + `data/normalize.py`, `price_norm_mode=window_scale`, `scaler_type=robust`); Binance spot 1h bars of BTC ETH SOL XRP DOGE ADA BNB TRX (LINK, AVAX are not in the public dataset) from `tools/fetch_crypto_dataset.py`, end 2026-09-30 23:00 |
| The ONE change | the per-bar input transform: V0 current window_scale (H/L/C minus window-median-close over window IQR, clip 5) vs V1 log-returns + global Min-Max, V2 log-returns + Standardization, V3 log-returns / trailing sigma, clip 5, V4 = V2 clipped at 5, V5 train-fitted quantile (rank) transform |
| Failure being addressed (level, location, registry id) | representation level, model input: layer health reports show low effective rank (2-12% of dim) and the NIG-TimeNet never beats the linear reference; hypothesis: the input already has few distinct samples (no registry id; F-0029 covers the model-side InstanceNorm only) |
| Data / splits / holdout | window T=32 bars (project default `window_sizes`), stride 4 on a shared hourly grid, 1h; train: targets <= 2024-12-31 23:00; val: window end >= 2025-01-02 00:00 (48 h embargo) and target <= 2026-09-30; no holdout opened. Windows touching (inputs t-47..t, target bar t+1) a gap, NaN, or a flat zero-volume run >= 168 h are dropped for ALL variants (same sample set) |
| Seeds / budget | no randomness except the bootstrap (seed 0, 300 reps, month blocks) and the NN subsample (seed 0, 15000 windows per split); CPU, minutes |

## Verified cause and mechanism
- Verified cause this change attacks: not yet verified, this IS the localisation test. Facts: window_scale maps each window
  to (x-median close)/IQR close, so the price level is erased but the three price columns H/L/C are ~0.99 correlated
  (see the comment in `data/windows.py` on exclude_from_features) and what remains is a smooth, highly redundant shape;
  the model's own InstanceNorm then standardizes every window again.
- Why the change should fix it: differenced inputs are stationary and close to white, so the flattened window should
  spread over many more directions (higher effective rank, fewer near-duplicates). Min-Max on heavy-tailed returns
  should do the opposite (a few outliers fix the range, the bulk is squeezed near one value); Standardization keeps the
  bulk at unit scale. The three comparison metrics are designed to test exactly this.
- Diagnostic that must move: effective rank ratio of the flattened window matrix (participation ratio of the covariance
  eigenvalues, the definition of `model/diagnostics.py:201`) on val: V2/V4 >> V0 (prediction: V0 below 0.15, V2 above
  0.5); near-duplicate share (NN distance < 1% of the variant's median NN distance) down vs V0; V1 below V2 in rank
  and with a non-trivial share of val values outside [0,1]. Null: bootstrap spread (month blocks) of each metric.
- Output effect expected on val: high/low linear-probe skill (Spearman IC of ridge predictions; AUC of the logistic) of
  V2-V4 within bootstrap noise of V0 or higher; close skill ~chance for every variant (F-0080 consistent).

## Accept / reject rule (val)
Metrics per variant on val (train for reference): effective rank ratio (covariance, and correlation as a scale-free
check); near-duplicate share and NN-distance median/p10/p90 (same-coin windows closer than 48 bars excluded from the
neighbour search); share of variance in top-1 / top-5 principal components; ridge probe (high/low/close return,
next-bar, relative to the same-type last price as in the project's `reg_mode='return'`) scored by Spearman IC (the
project skill, `workflow/capacity.py`) and R2; logistic probe (high/low/close up vs own last) scored by AUC; for V1 the
share of val values outside [0,1]. Ridge/logistic regularization is relative to the feature scale and chosen on the
last 20% of train, so global scale differences between variants do not decide the result.
- Accept (a variant is recommended) if: it is NOT disqualified and has the maximum effective rank ratio (val) among
  non-disqualified variants, AND its near-duplicate share is below V0's. Disqualified = the paired month-block
  bootstrap 95% CI of (variant minus V0) for the Spearman IC of the high reg probe OR of the low reg probe lies entirely
  below 0 (skill lowered by more than its bootstrap noise).
- Reject if: every non-V0 variant is disqualified or none beats V0 on both rank and near-duplicates: then the current
  normalization stays and the representation collapse is not an input-normalization effect.
- Otherwise: "unexplained" (not adopted). The close targets are reported but are NOT part of the rule.
- Not a model result: even an accepted variant only licenses a NEW data option (default = old behaviour) and a later
  trained-model experiment under the MODEL_DEV_PROTOCOL; no "architecture/normalization X is better" claim is made.

### Addendum 2026-10-19, written after the smoke test on synthetic data and BEFORE any real-data variant is computed
- Tie rule for the near-duplicate criterion: "near-duplicate share below V0's" is read as "not above V0's" (<=) when V0's
  share is exactly 0 (the criterion cannot discriminate then and is reported as uninformative); strictly below otherwise.
  The NN-distance quantiles (p10/median, p90/median) are reported as the continuous companion of that criterion.
- Effective rank is the covariance participation ratio; the correlation-matrix version and the entropy version are
  reported as checks and do not enter the rule.

---
## Result (added after the run)

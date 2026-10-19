# norm_probe — differenced (log-return) input normalizations vs the current window_scale

Card (pre-registered, committed before any variant was computed): [`cards/E-normprobe-1.md`](cards/E-normprobe-1.md).
Script: `run_norm_probe.py` (tests: `tests/test_norm_probe.py`). Results: `results/norm_probe_results.csv` (geometry + probe
point values, train and val), `results/norm_probe_bootstrap.csv` (val metrics with month-block bootstrap CI and paired delta
vs V0), `results/norm_probe_meta.json` (decision, chosen regularization). CPU only, no neural training, ~7 min.

## The current normalization (V0), where it lives
- Window length: `data/defaults.py:33` `window_sizes {"1D": 32}`, `stride` 1 (`:34`); `scaler_type: "robust"` at `data/defaults.py:151`.
- Per window, the price columns (open/high/low/close, kind `price_level`) are `(x - median(close window)) / IQR(close window)`,
  with the IQR floored at `1e-3*|median|` and `1e-8` (`data/normalize.py:231,242-258` `calc_scale_params`; applied in
  `data/normalize.py:~475` `process_windows`, called from `data/windows.py:266,336-339`); constant columns -> 0, then clip
  to `+-clip_abs=5` (`data/defaults.py:390`). Only the alternative `price_norm_mode='pct_change'` (`:399`) is differenced; it is off by default.
- The model then standardizes every window again per feature (`model/input_norm.py:12` `InstanceNorm`, built at `model/builder.py:111`),
  keeping `[mean, log std]` as a stats path (so the level that window_scale removed can only come back through that path).

## Setup actually run
Data: real, fetched (`tools/fetch_crypto_dataset.py`, Speirsy11/crypto-dataset, Binance spot 1h, end 2026-09-30 23:00; the
network was open): ADA BNB BTC DOGE ETH SOL TRX XRP (LINK and AVAX are not in that dataset; BNB and TRX replace them). Window
T=32 bars (project default), stride 4, H/L/C per bar. Train: targets <= 2024-12-31 (52,408 windows, 210,408 train bars used for
fitting); val: window end >= 2025-01-02 to 2026-09-30 (29,956 windows). Windows touching a gap/NaN or a flat zero-volume run >= 168 h
are excluded for all variants. Probes: ridge (3 return targets) and logistic (3 up/down targets), next bar, relative to the same-type
last price as the project's `reg_mode='return'`; regularization relative to feature scale, chosen on the last 20% of train.
Skill = Spearman IC (reg) / AUC (class), as in `workflow/capacity.py`. Bootstrap: 300 reps over the 21 val months.

## Results (val; train in the CSV)
| | V0 current | V1 logret+MinMax | V2 logret+Std | V3 logret/sigma14, clip5 | V4 V2 clip5 | V5 quantile |
|---|---|---|---|---|---|---|
| eff. rank ratio (cov) | 0.057 | 0.324 | 0.342 | **0.497** | 0.318 | 0.281 |
| eff. rank (corr / entropy) | 0.073 / 0.117 | 0.343 / 0.522 | 0.343 / 0.529 | 0.532 / 0.596 | 0.302 / 0.514 | 0.287 / 0.504 |
| top-1 / top-5 PC variance | 0.358 / 0.742 | 0.125 / 0.258 | 0.116 / 0.249 | 0.029 / 0.139 | 0.129 / 0.230 | 0.147 / 0.245 |
| near-dup share (<1% of median NN) | 0 | 0 | 0 | 0 | 0 | 0 |
| NN dist p10/median, p90/median | 0.47, 1.86 | 0.55, 1.84 | 0.55, 1.83 | 0.65, 1.43 | 0.55, 1.80 | 0.75, 1.36 |
| high reg IC (95% CI) | 0.557 (.540-.576) | 0.595 | 0.595 | 0.577 (.561-.592) | 0.599 | 0.536 |
| low reg IC (95% CI) | 0.549 (.529-.569) | 0.592 | 0.592 | 0.565 (.545-.587) | 0.590 | 0.540 |
| delta high IC vs V0 [95% CI] | - | +.037 [.027,.047] | +.038 [.027,.047] | +.020 [.009,.028] | +.042 [.031,.051] | -.021 [-.036,-.004] |
| delta low IC vs V0 [95% CI] | - | +.043 [.036,.053] | +.043 [.036,.053] | +.016 [.009,.024] | +.041 [.035,.051] | -.009 [-.021,.004] |
| high / low AUC | .775 / .782 | .776 / .786 | .779 / .786 | .784 / .791 | .779 / .787 | .756 / .769 |
| close IC / AUC | .032 / .527 | .019 / .511 | .020 / .517 | .026 / .525 | .021 / .520 | .029 / .527 |
| high / low R2 | .231 / .107 | .375 / .190 | .375 / .190 | .266 / .134 | .374 / .200 | .228 / .125 |
| share outside [0,1] (val) | - | 0.023% (223 of 958,592) | - | - | - | - |
| share at clip (val) | 0.16% | - | - | 0.67% | 0.24% | - |

V0's val effective rank (5.7%) reproduces the 2-12% seen in the layer health reports: the collapse is already present in the
INPUT, before any layer. Close direction is at chance for every variant (AUC 0.51-0.53, IC 0.02-0.03; consistent with F-0080).

## Reading, and what the numbers do NOT say
- Pre-registered rule result: **V3 is picked** (highest rank among non-disqualified variants; V5 is disqualified: its high-IC delta
  CI lies below 0). Every non-V0 variant raises rank 5-9x and lowers top-1 variance (0.36 -> 0.03-0.15).
- The near-duplicate criterion (1% of the median NN distance) is uninformative here: the share is 0 for every variant including V0
  (tie rule in the card addendum made it pass). The continuous companion (NN p10/median) rises from 0.47 (V0) to 0.65 (V3) and 0.75 (V5),
  so V0 does have the most "crowded" neighbourhoods, but there are no literal duplicates once overlapping windows are excluded.
- Standardization vs Min-Max: the linear skill and the rank are almost the same (V1 vs V2: IC identical to 3 decimals, rank 0.324 vs
  0.342), because a global per-channel affine map does not change a linear probe. The difference is practical: Min-Max is fixed by
  two train extremes (single bars of -35% and +31%), so the val bulk occupies about 2% of [0,1] (per-channel std 0.017-0.022; the
  high channel's 99th percentile is 0.089, the low channel sits near 0.96), and 223 val values still fall outside [0,1] (Min-Max
  does not bound unseen data); Standardization keeps unit-order spread (train std 0.7-0.85% per bar) and V4 (clip 5) bounds it.
  Evidence supports Standardization (V2/V4) over Min-Max; the V4 clip touches only 0.24% of values.
- V2/V4 give the largest magnitude skill (+0.04 IC on high/low, CI excludes 0) but V3 gives the largest rank. Unverified
  hypothesis (not tested here): the V2/V4 gain on high/low is largely the absolute volatility level that returns keep and that
  window_scale (IQR division) removes; V3 divides it away again, which explains its lower IC and higher rank. If true, the
  trade-off is "volatility level kept (V4)" vs "most distinct shapes (V3)", and V3 + a separate log-sigma input would be the
  natural next candidate.
- Limits: linear probes only (no claim about the network), one train/val split, 8 coins, stride-4 windows (heavily overlapping,
  bootstrap blocks are months), 3 features (H/L/C) not the full feature set, ridge/logistic scored on val only once. Some logistic
  fits (large C) hit the iteration cap without converging; the chosen C is on the tail-validation of train, so the AUCs are a floor.

## Recommendation (recommend only; `data/` is untouched)
Add a NEW option in `data/` (e.g. `price_norm_mode='logret_std'`, default stays `'window_scale'` so checkpoints and past results
are reproduced exactly) that feeds the price columns as per-bar log returns relative to the previous close
(`log(H/C_prev)`, `log(L/C_prev)`, `log(C/C_prev)`), standardized with mean/std fitted on the TRAIN split only and stored in the
dataset (clip 5, = V4); rank/quantile (V5) and Min-Max (V1) are not recommended. Keep V3 (`/ trailing 14-bar sigma`, clip 5) as a
second option to test, because it has the best rank and is the only variant whose AUC rises for both the high and the low
class with the CI above 0 (delta +0.009 [.006,.013] and +0.009 [.005,.013]). The next step is a model experiment under `MODEL_DEV_PROTOCOL.md` (one change: V4 or V3
vs V0, 3 seeds, predicted movement: layer effective rank above V0's 2-12% and the linear-reference gap shrinking); this probe
alone does not show that the network will use the extra rank.

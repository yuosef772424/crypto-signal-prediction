# Data-preparation audit: what per-window normalization destroys (2026-11-01)

Scope: `data/normalize.py` (kinds, `process_windows`), `data/normalizers.py`, `data/windows.py`, the model's `InstanceNorm`
(`model/input_norm.py`, read only), failure F-0029 (`failure_registry.csv`: "InstanceNorm erases the feature level"; `reopen_if`:
"only if feature levels reach the head through another channel").
Instrument: `docs/research/hardening/data_prep_probe.py` (results: `data_prep_probe_results.json`). Real pipeline code, real data:
10 Binance spot coins, 1h, 2021-01 -> 2026-10 (ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC, `tools/fetch_crypto_dataset.py`), preset
`HOURLY_W32_S8` (window 32, stride 8, grid-aligned), horizon 4 bars, the preset's 18 features. Split (time): train <= 2023-12-31
(30,120 samples), val 2024 (10,940), test 2025-01 -> 2026-10 (19,250). Probes are ridge regressions (penalty chosen on val, everything
fitted on train, scored on test, pooled over coins) - a measure of what information IS in a view, not what a network can extract.
Not a model result: no claim here about trading edge.

## 1. What each step does to a level (by construction, `data/normalize.py`)
| Kind | Transform | What survives | What is lost |
|---|---|---|---|
| `price_level` (close, EMA, bands) | `(x - median_window) / IQR_window` | shape, position inside the window range | absolute price, volatility level (everything is in window-IQR units), trend size in % |
| `price_scale` (ATR, MACD) | `x / IQR_window` | size relative to the window's own dispersion | absolute vol level |
| `log_volume`, `cumulative` | `log1p` then window median/IQR | shape | absolute dollar volume, volume regime |
| `sign_robust` (RET_*, CCI, TSI...) | `x / (1.4826 * median|x|)` | sign, relative size | the MAGNITUDE of returns (vol level) |
| `percent` (NATR, RANGE_, BBB, SUPERT_STRETCH) | `x / 10` | **absolute level** | nothing |
| `osc_*`, `unit_*`, `zscore` | fixed affine | absolute level of a bounded oscillator | - |
| model `InstanceNorm` | per window, per feature `(x - mean) / std` + stats path `[mean, log std]` | shape (z); level only through the stats path | level in the z view |

So the data layer keeps volatility level ONLY through the `percent` family; the model's InstanceNorm then erases whatever level is left in
the main path and keeps it only in the side `stats` path.

## 2. Measurements (real pipeline, real data)
**M1 - what the encoder-visible views retain** (test R^2 of recovering each level quantity from the view; `retained_r2_test`):
| Quantity at entry | normalized window (pipeline X, 32 x 18) | InstanceNorm z view (what the encoder sees) | model stats path [mean, log std] |
|---|---|---|---|
| log realized vol (32 bars) | 0.757 | **-0.036** | 0.946 |
| window return in vol units | 0.963 | 0.885 | 0.667 |
| distance to window high in vol units | 0.843 | 0.748 | 0.340 |
| distance to window low in vol units | 0.823 | 0.719 | 0.335 |
| log dollar volume z vs 256-bar mean | 0.479 | 0.350 | 0.425 |
| vol regime (log sigma32 - log sigma256) | 0.501 | 0.378 | 0.504 |

**M2 - which of the 18 features carry the volatility level** (R^2 of log vol from that feature's own 32 normalized steps, test):
NATR_14 0.642, RANGE_rel 0.640, BBB_20_2.0 0.618 (all `percent`), MKT_BETA_20 0.145; the other 14 are all <= 0 (close -0.145,
volume -0.120, RSI -0.143, POS_50 -0.143, MOM_ORTH_NATR -0.142, VOLZ_20 -0.075 ...). `close` (window_scale) and `volume` (log_volume)
carry no absolute price or volume level at all.

**M3 - predictive content of the views** (test, 19,250 samples, 22 months; IC = Spearman; `months_pos` = months with IC > 0):
| View | log|ret_4h| IC / R^2 | log range_4h IC / R^2 | direction AUC (IC) |
|---|---|---|---|---|
| normalized window (pipeline X) | 0.410 / 0.137 | 0.732 / 0.465 | 0.509 (0.016) |
| InstanceNorm z window only | **0.116** / -0.007 | **0.190** / -0.054 | 0.511 (0.019) |
| z + stats path | 0.416 / 0.153 | 0.739 / 0.573 | 0.510 (0.017) |
| level channel only (6 LVL_ columns, last step) | 0.416 / 0.155 | 0.713 / 0.542 | **0.5185** (0.032) |
| normalized window + level channel | **0.425** / 0.153 | **0.748** / 0.582 | 0.510 (0.017) |
| z + stats + level | 0.423 / 0.159 | 0.748 / 0.589 | 0.511 (0.019) |
Magnitude IC is positive in 22/22 months for every view; direction IC is positive in 13-15 of 22 months for every view (no sign stability).

**M4 - the level is mostly a coin identity**: eta^2 of log realized vol on the asset = 0.41 on the test period (41% of the variance of the
level is between coins). A model that is given the level can therefore also read "which coin" (the F-0029 memorization mechanism, now by
design); `data/level_channel.py` standardizes with TRAIN-only global statistics (not per coin) and the model-side regularizers
(`level_norm`, feature dropout, noise) are the guard - untested here.

**M5 - clipping** (`clip_abs = 5`, window_scale): `close` is clipped in 2.1% of windows (0.15% of values), `volume` 0.5% of windows, `BBB`
0.4%. These are the breakout / trend windows (close more than 5 window-IQRs from the window median): the largest moves lose their size.

**M6 - exact-zero horizon return** (`close[t+h] == close[t]`, `CONFIG['stale_bars']` at its default = off): 1.3% train / 0.6% val /
0.7% test on these 10 liquid coins (the registry records 7.1% / 17.1% on the 222-coin Drive set). The default still keeps these rows.

## 3. Findings (ranked)
1. **F-0029 is confirmed on real data at the model-input stage, and its `reopen_if` is met by the stats path, not by new data.** The z view
   alone predicts magnitude at IC 0.12 (range 0.19); adding the level lifts it to 0.42 (0.74). So the encoder's main path is blind to the
   volatility level and everything rides on the side `stats` path (R^2 0.946 for log vol). Whether a model USES it is a model question
   (`level_passthrough`, E-norm-002); the data layer cannot fix it alone.
2. **The data layer is not the main culprit for volatility: 3 `percent` features already carry it** (M2). It does erase absolute price and
   volume (`price_level`, `log_volume`), the size of returns inside `sign_robust` columns (none of RET_*/CCI is in this preset: the preset
   has only 18 features), and the trend magnitude in %.
3. **A train-fitted level channel adds little linear information on top of the existing columns**: +0.015 IC (abs return) and +0.016 IC
   (log range) over the normalized window (M3). Its value is that the level reaches the head explicitly, as standardized inputs, with
   a documented train-only state - not that it is a new source of signal.
4. **No view has stable direction information** (AUC 0.509-0.5185, IC 0.016-0.032, positive in 13-15 of 22 months). The best direction view is the
   level channel alone (AUC 0.5185): the little direction information there is rides on the volatility / liquidity level, as F-0029
   says; it is far below any cost threshold (compare F-0057, F-0080).
5. **Clipping at +-5 removes the size of exactly the biggest-move windows** (M5; `close` 2.1% of windows); and **unclassified columns fall
   back to `CUMULATIVE` silently** (`data/normalize.py` `DEFAULT_KIND`), the mechanism behind the earlier VWAP_DEVIATION / SUPERT_DIR /
   MOM_RANK mis-scalings recorded in that file: a new feature name should raise, not default.

## 4. Other defects looked for
| Check | Result |
|---|---|
| Look-ahead in features | None found: rolling features use bars <= t (`tests/test_no_lookahead.py`, `tests/test_signal_causality.py`; `ITD_TRADES_Z` uses the past days only, `tools/intraday_features.py:337`; the new LVL_ features have prefix / future-perturbation tests). |
| Target / feature alignment | `timestamp` = open time of the last window bar, entry = that bar's close, target from bars ts+1..h exactly (gap samples dropped, `data/windows.py`): consistent with `tools/bracket_path.py`. |
| Scaling fitted on all data instead of train | None in the default path: `window_scale` is per window, `reg_target_scale` a constant; `logret_std` fits on train in `finish_price_norm`. Footgun: a logret_std (and level-channel) dataset is UNFITTED until `split_data` - nothing stops feeding it to a model. |
| Per-asset vs global stats | Global train statistics (logret_std, level channel). `split_mode='per_asset'` leaks when histories differ in length (documented in `data/split.py`). |
| Cross-sectional target normalization | `cross_sectional_normalize` standardizes each target by the other coins' targets at the same timestamp: the inverse needs peers' realized targets, so it cannot be applied to a live prediction as is (read, not run). |
| Degenerate columns | `MKT_CORR_*` / `MKT_BETA_20` are constant for the reference coin itself (10% of windows = 1 of 10 coins): zeroed, harmless. |
| Stale bars | M6; default off. |

## 5. What was built (feat, default OFF): the level side-channel
`CONFIG['level_channel'] = {...}` (None = old behaviour, byte-identical datasets / splits / checkpoint fingerprints). One registered component
`LEVEL_CHANNELS['level_v1']` (`data/level_channel.py`, owner rule: `fit` on train only -> `state`; `transform_inputs` / `stamp` /
`prepare_live`; state stored as `split['level_meta']`; every path calls it: `split_data`, `split_holdout(level_meta=...)`,
`build_dataset_live(level_meta=...)`). Features (registry `LEVEL_FEATURES`, unknown names raise): `logvol`, `retvol`, `disthi`, `distlo`,
`logdvol_z`, `volreg` (columns `LVL_*`, kind `LEVEL_RAW` = `x/10`, no per-window scaling, never zeroed). Settings: `features`, `window`
(32), `long_window` (256, <= 500), `clip` (5), `component`; unknown keys raise. Enable: `update_config({"level_channel": {"window": 32,
"long_window": 256}})`, then build and split as usual. Tests: `tests/test_level_channel.py` (causality, train-only fit, default off, holdout and live need
the stored state, checkpoint fingerprint).

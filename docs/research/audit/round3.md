# Audit round 3: two-timeframe path (1h base + closed 4h context), commit f2442be

Scope: `build_hourly_4h_dataset`, `higher_tf_mode='closed'`, `x_storage_dtype='float16'`, and how main.ipynb / cross_asset feed the two arrays.
Evidence only (PROTOCOL.md). All scripts use synthetic data and the real notebook functions (`_nbload.py`).

## Expectations written before reading any rationale

- A sample whose last 1h bar opens at t is decided when that bar closes, so its information time is I = t + 1h (its features contain that bar's close, its target is the next bar).
- Every 4h bar in its window must have closed at or before I (open + 4h <= I). A bar closing exactly at I is allowed, one closing after I is a leak. Any 4h indicator, cross-asset feature or phase-2 feature attached to a 4h bar may use only data available at that bar's close.
- 4h windows are 32 consecutive bars; 4h prices are normalised with statistics of that window only.
- No val/test input bar (1h window or 4h window) may be at or after the open of a candle a train (resp. val) target was built from.
- float16 must be invisible: finite, no overflow, upcast before arithmetic. Single-TF output must equal the pre-change output.

## Clean (proven; scripts pass)

| # | What | Script | Result |
|---|---|---|---|
| R3-00 E1 | Exact-timestamp alignment: real `align_multi_timeframes_time_based`, frames whose value is the bar's own open hour. Strides 1, 3, 8, grid on/off, starts 00:00/02:00/03:00, a 3 h hole in 1h data, a missing 4h bar. | [r3_00](r3_00_clean_checks.py) | 0 windows with 4h close > t+1h; 0 with close > t (the stricter documented rule); 4h windows contiguous (32 bars, step 4h); slack I-close is 1..4h. A bar closing exactly at t+1h is skipped when t mod 4h = 3 (1h staler than allowed; not a leak). |
| R3-00b | Perturbation, end to end, real `HOURLY_4H_OVERRIDES` (43 features, MKT_*, breadth, MOM_ORTH_NATR, 15m, funding, OI, metrics), 6 coins: from a boundary T on EVERY raw stream of EVERY coin is replaced; samples with t+1h <= T must be bit-identical. T on/between 4h and 8h boundaries and mid-bar, stride 8 and stride 1 (all 4 phases). | [r3_00b](r3_00b_no_lookahead_perturbation.py) | 0 changed X_1h, 0 changed X_4h in 480 (stride 8, x3 T) and 3822..3840 (stride 1, x5 T) checked samples. Teeth: 438/438 (s8) and ~3500/3540 (s1) later samples do change; the replacement reaches 36/43 features (the 7 unreached are constant availability flags, see S3); a deliberately leaky config (legacy, `higher_tf_offset=0`) is flagged at 4/4 boundaries (X_4h changed for 24..42 samples). |
| R3-00 E2 | Purge on the shipped grid: real `split_data`, stride 8, `embargo_candles`=129. | [r3_00](r3_00_clean_checks.py) | 0 val windows contain a train target candle, 0 test windows contain a val target candle. |
| R3-00 E3 | 4h normalisation: `X_4h[close]` recomputed from the 32 raw 4h closes of the same window. | [r3_00](r3_00_clean_checks.py) | max abs diff 9.3e-4 over 153 samples (float16 rounding); normalising with the 1h window's statistics would differ by up to 5.33 (teeth). No statistic uses a bar outside the window (also implied by R3-00b). |
| R3-00 E4 | float16: finite, range, error vs an otherwise identical float32 build. | [r3_00](r3_00_clean_checks.py) | finite, max abs 5.000, max abs error 1.95e-3 (normal data) and 1.95e-3 with raw volume x1e12, trades x1e9, OI x1e9. Cast happens after `process_windows` (float32, clipped +-5). |
| R3-00c | Same indices for 1h and 4h: signature rows (X_1h=i, X_4h=i+0.5, y=i). Real `make_shuffled_dataset` (train), `_concat_splits` (eval / collect_signals / export), `pool_test_dict` (chicks), `PanelSplit` (shuffled + k-coin sampling, val/test order, `chunk_groups(k=3)`, `iter_group_batches`, `subset`). Model: float16 dict call. | [r3_00c](r3_00c_gather_alignment.py) | 0 misaligned batches (6 + 35), all gathers aligned; float16 input equals float32-upcast input exactly (max diff 0.0); tuple(1h,4h) equals dict; tuple(4h,1h) differs (max diff 3.69), so tuple order is load-bearing (all callers pass `_tfs_of()` = MODEL_TFS order). |
| R3-00d | Single-TF unchanged: old notebook from `git show 4cb2d7b`, same synthetic universe, 4 configs (1h_s8 with/without phase 2, legacy 1h+4h, stride 1). | [r3_00d](r3_00d_single_tf_unchanged.py) | dataset keys and bytes identical, `split_data` identical, `embargo_candles` 33 = 33. `tests.test_multi_tf ...test_old_presets_byte_identical` (golden_pre_multi_tf.json digests) also OK. |

## Findings (each has a repro that fails now)

### R3-01: 129h purge assumes t mod 4h = 0 (low; not reachable with the shipped stride-8 grid)

`embargo_candles` (closed mode) = 32 x 4h + horizon = 129. The newest closed 4h bar closes at t - (t mod 4h), so the 4h window starts up to 131h before t.

```
$ python docs/research/audit/r3_01_purge_short_off_4h_phase.py
shipped: stride 8, grid    embargo=129h  val phases(t mod 4h)=[0]  val windows containing a train target candle:   0   test ...: 0
stride 4, grid             embargo=129h  val phases=[0]           ... 0 / 0
stride 1                   embargo=129h  val phases=[0, 1, 2, 3]  ... 2 / 2   e.g. sample t=02-06 10:00 4h window starts 02-01 00:00
stride 3                   embargo=129h  ... 0 / 0
stride 5                   embargo=129h  ... 1 / 0   e.g. sample t=02-06 10:00 4h window starts 02-01 00:00
stride 8, grid OFF (t=09:00+8k) embargo=129h  val phases=[1]  ... 0 / 0
  stride 1 with embargo_candles=129: val 2, test 2 | 130: 1, 1 | 131: 0, 0 | 132: 0, 0 | 133: 0, 0
AssertionError: DEFECT: 5 val/test samples contain candles used as train/val targets
```
Expected: 0 for any stride/grid the pipeline accepts. Actual: samples at phase 1..3 right after the gap contain the target candle (train t + 1h) of the last train sample. Same formula is reused by main.ipynb's verification check (`window + horizon`).
The 131 in the last row is measured for this split date (train_end on a 4h boundary); other split dates were not measured.

### R3-02: legacy 1h+4h data is split with a 33h purge (low; only for datasets built before this commit)

main.ipynb section 3 accepts `MODEL_TFS=["1h","4h"]` on a dataset without `higher_tf_mode='closed'` and only prints a warning about the alignment rule. `embargo_candles` widens only for `closed`.
```
$ python docs/research/audit/r3_02_legacy_two_tf_embargo_33h.py
embargo_candles = 33h | 4h window = 32 bars = 128h | val samples 80: windows containing a train target candle = 12 (e.g. sample t=02-02 16:00, 4h window starts 01-28 08:00)
AssertionError: DEFECT: legacy two-timeframe split purges 33h but the 4h window spans 128h+
```
Expected 0. Actual 12 of 80.

### R3-03: `audit_normalization` returns std=inf on float16 X (low; diagnostic gate)

numpy reduces float16 in float16; `v.std()` overflows. `audit_normalization` is the documented pre-training gate (`# audit_normalization(dataset[f"X_{base_tf}"], ...)` in the pipeline usage cells).
```
$ python docs/research/audit/r3_03_audit_normalization_float16_std_inf.py
X_1h (7362, 32, 43) float16: std=inf for 23/43 features; verdict differs from the float32 audit for 23
feature   std_f16   std_f32   verdict_f16     verdict_f32
close     inf       0.744896  high dispersion  ok
volume    inf       0.787996  high dispersion  ok
RSI_14    0.244141  0.244136  ok               ok
AssertionError: DEFECT: audit_normalization on float16 X gives std=inf / different verdicts than on float32
```
Expected: identical statistics for float16 and float32 storage. Actual: inf and 23 flipped verdicts (real file: ~14M values per feature, so all features overflow). main.ipynb's own `normalization_audit` casts to float64 first (read, not run).

## Suspicions (not proven)

- S1: `x_storage_dtype='float16'` has no guard when `clip_abs` is falsy (`process_windows` skips the clip, `astype(float16)` turns values > 65504 into inf). Shipped `clip_abs=5.0`; not triggered.
- S2: live path (`build_dataset_live`, zero-volume dummy tail, `_keep_full_bars`) was not exercised with `higher_tf_mode='closed'`.
- S3: `*_available` flags are constant 1 in the synthetic universe, so R3-00b does not exercise look-ahead through availability flags (they are asof-at-close by construction in `tools/intraday_features.py`; read, not perturbed).
- S4: `_finalize_accumulator` (per_asset split mode) builds an empty split as float32 even when X is float16 (0 rows, cosmetic).

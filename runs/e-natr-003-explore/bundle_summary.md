# Bundle summary: e-natr-003-explore

- status: **OK**
- repo_ref: `49ecd4acc9a0a51f151fe7bc8d75ff77b2c35cef`
- entry: `research/natr_range/run_natr_calibrate.py `
- note: Prepared, NOT queued (R3, E-natr-003 Stage A, exploration only). Reads the 10 OPEN coins from Drive history_15m through holdout_guard.load_unsealed (window end 2026-10-01); no sealed bucket is opened and no holdout log row is written. Fits on 2020-2024 only, evaluates 2025-01..2026-09. frozen_stage_
- exit code: 0   duration: 32.9 s
- problems: none

## manifest.json
```json
{
 "accept_rule_evaluations": 1,
 "baseline_computations": 2,
 "card": "research/natr_range/cards/E-natr-003.md",
 "config": {
  "base_end": "2024-01-01",
  "calib_end": "2025-01-01",
  "coins": [
   "ADAUSDT",
   "BCHUSDT",
   "BNBUSDT",
   "BTCUSDT",
   "DOGEUSDT",
   "ETHUSDT",
   "SOLUSDT",
   "TRXUSDT",
   "XRPUSDT",
   "ZECUSDT"
  ],
  "data_dir": "/content/drive/MyDrive/history_15m",
  "end": "2026-10-01",
  "fit_start": "2020-01-01",
  "load_start": "2019-10-01",
  "min_coins": 8,
  "oof_first_year": 2021,
  "oof_last_year": 2024,
  "oof_start_year": 2020,
  "pattern": "{symbol}.csv.gz",
  "source_minutes": 15,
  "stale_min_run_hours": 168,
  "val_end": "2026-10-01"
 },
 "git_sha": "49ecd4acc9a0a51f151fe7bc8d75ff77b2c35cef",
 "input_fingerprint": "e28fdf71ffc5bbdce8716a42aaa56142a1922ba3c87a5ee7cef7bbb3ce3d6f98",
 "n_configurations": 21,
 "outputs": {
  "corrections.csv": "2785ab35368ae9e0c188c7e89a48db1324055fab6e7727495967c9131d08e9d8",
  "deciles.csv": "2ef255b22ab7cab0d79df28397b41aab40ca523fee99aea33c70b13230959795",
  "frozen_stage_b.json": "5a04d54fc48dd2cad5e5bca1aa0fd96ecf93bee3c68307a98eedc3b2f86ad330",
  "months.csv": "cdea39c478d3d50690f7daf9aaf0f3768d691aa27a6caf241adced3c2269e6bf",
  "oof_diagnostic.csv": "b2cb8d62088411ae756a7878be4e1bd392ce4861c4bc1743d392c0fe4f31ac32",
  "results.json": "0fe1cdd6ee21e68d604c754eee69052282b202c9a44d477037e309ca34df0851",
  "summary.csv": "837c5bbae296fb26f6642d09147cc5760025d15145bfb936650956f1f7685d13",
  "year_deciles.csv": "a1797887ba378f8aa56b0133c93e20086cef98335a20c263c4764128c06229f0"
 },
 "row_counts": {
  "ADAUSDT_rows": 58321,
  "ADAUSDT_source_bars": 233344,
  "BCHUSDT_rows": 59049,
  "BCHUSDT_source_bars": 236256,
  "BNBUSDT_rows": 58081,
  "BNBUSDT_source_bars": 232384,
  "BTCUSDT_rows": 59049,
  "BTCUSDT_source_bars": 236256,
  "DOGEUSDT_rows": 54456,
  "DOGEUSDT_source_bars": 217884,
  "ETHUSDT_rows": 59049,
  "ETHUSDT_source_bars": 236256,
  "SOLUSDT_rows": 52752,
  "SOLUSDT_source_bars": 211076,
  "TRXUSDT_rows": 58583,
  "TRXUSDT_source_bars": 234400,
  "XRPUSDT_rows": 58798,
  "XRPUSDT_source_bars": 235263,
  "ZECUSDT_rows": 58079,
  "ZECUSDT_source_bars": 232384
 },
 "runtime_seconds": 24.061,
 "truncated": false
}
```

## corrections.csv (first 11 of 11 lines)
```
decile,n_calib,c10,c50,c90
0,12267,1.020942,1.0,0.965193
1,10979,1.001893,1.0,1.016046
2,10910,1.008296,1.0,0.991202
3,9731,0.995055,1.0,0.996588
4,8781,1.004673,1.0,0.977196
5,8199,0.996963,1.0,0.997359
6,7494,0.976765,1.0,0.987994
7,7327,0.977879,1.0,1.004004
8,6994,0.972058,1.0,1.002658
9,5158,0.948482,1.0,1.020483
```

## deciles.csv (first 12 of 21 lines)
```
arm,decile,n,coverage,baseline_coverage,pinball_model,pinball_baseline,pinball_ratio
control,0,30301,0.806244,0.732913,0.057541,0.064432,0.893053
control,1,20382,0.80316,0.745707,0.089217,0.098351,0.907132
control,2,17517,0.804932,0.75264,0.106931,0.116749,0.915905
control,3,15873,0.805204,0.75304,0.125232,0.135499,0.924232
control,4,15224,0.802286,0.763531,0.13684,0.147486,0.927818
control,5,14154,0.801964,0.76247,0.152501,0.165584,0.92099
control,6,13027,0.802334,0.769556,0.171729,0.18673,0.919664
control,7,11480,0.805662,0.778397,0.196485,0.217079,0.905132
control,8,8898,0.796921,0.776467,0.256354,0.288101,0.889806
control,9,5304,0.762632,0.765837,0.413763,0.500146,0.827285
corrected,0,30301,0.789215,0.732913,0.057534,0.064432,0.892941
```

## months.csv (first 12 of 22 lines)
```
month,ic
2025-01,0.695989
2025-02,0.707434
2025-03,0.689577
2025-04,0.706246
2025-05,0.700104
2025-06,0.686488
2025-07,0.727654
2025-08,0.661203
2025-09,0.69301
2025-10,0.720539
2025-11,0.741926
```

## oof_diagnostic.csv (first 12 of 41 lines)
```
fold,decile,n,coverage
2021,0,502,0.794821
2021,1,2221,0.822602
2021,2,4353,0.824029
2021,3,5680,0.821831
2021,4,7143,0.834523
2021,5,8234,0.836653
2021,6,10275,0.839805
2021,7,12473,0.839894
2021,8,15024,0.838325
2021,9,21695,0.826043
2022,0,18125,0.7696
```

## summary.csv (first 2 of 2 lines)
```
coverage,overall_ok,deciles_ok,pinball_deciles_better,pinball_ok,ic_months,ic_positive_months,ic_ok,n_evaluable_deciles,pinball_needed,enough_deciles_ok,top_decile_year_ok,top_decile_closer_than_control_ok,coins_ok,localization_static_ok,rules_1_to_4_ok,accepted,localization,pooled_oof_top_coverage,
0.801768,True,True,10,True,21,21,True,10,8,True,True,True,True,False,True,False,drift,0.813727,152160,0,0.755074,0.135589,0.150386,0.172865,0.179627,0.176093,0.18117,152160,0,0.755074,0.135601,0.150386,0.172865,0.179627,0.176093,0.18117
```

## year_deciles.csv (first 12 of 41 lines)
```
year,decile,n,coverage,arm
2025,0,13271,0.814709,control
2025,1,10041,0.81217,control
2025,2,9313,0.813701,control
2025,3,9166,0.807222,control
2025,4,9964,0.805801,control
2025,5,9679,0.811138,control
2025,6,8658,0.806191,control
2025,7,7720,0.804275,control
2025,8,5866,0.795943,control
2025,9,3922,0.755992,control
2026,0,17030,0.799648,control
```

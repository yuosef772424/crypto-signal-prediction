# Bundle summary: e-natr-004-explore

- status: **OK**
- repo_ref: `e32d8a6d9da2fd54153a754585485894b5ef6c3d`
- entry: `research/natr_range/run_natr_rolling.py `
- note: Prepared, NOT queued (R3, E-natr-004 Stage A, exploration only). Reads the 10 OPEN coins from Drive history_15m through holdout_guard.load_unsealed (window end 2026-10-01); no sealed bucket is opened and the forward data (>= 2026-10-01) is not read. Base fit 2020-2023, edges 2020-2024, rolling month
- exit code: 0   duration: 32.8 s
- problems: none

## manifest.json
```json
{
 "accept_rule_evaluations": 1,
 "baseline_computations": 3,
 "card": "research/natr_range/cards/E-natr-004.md",
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
  "pattern": "{symbol}.csv.gz",
  "source_minutes": 15,
  "stale_min_run_hours": 168,
  "val_end": "2026-10-01"
 },
 "git_sha": "e32d8a6d9da2fd54153a754585485894b5ef6c3d",
 "input_fingerprint": "e28fdf71ffc5bbdce8716a42aaa56142a1922ba3c87a5ee7cef7bbb3ce3d6f98",
 "n_configurations": 9,
 "outputs": {
  "deciles.csv": "a22c275effbcc3fcff2da11491dc5ee4a1efc9dc9500ecb910a9de4438b7451d",
  "frozen_stage_b.json": "7493a719559b636962273c1211060fa472cc46e1307db74ac06805b2b6b6c8e7",
  "monthly_factors.csv": "aed07e619f1600638d6d97b7b7ccc61de90f96707d71092eaf6655ec49629a57",
  "months.csv": "cdea39c478d3d50690f7daf9aaf0f3768d691aa27a6caf241adced3c2269e6bf",
  "quarters.csv": "b49a876cdb88f40bf3851d925b16530f109f1de6ada9de51d9326db62e3d1eff",
  "results.json": "bea4d4dfaee68ddbf6cdf0e81f090e7cafdcbbdf5dd1bf7c0356f7687c136cd7",
  "summary.csv": "95e487baf74fdb41ac7c6788a335eccf83b42e02522549f57f240d99c33f1687",
  "year_deciles.csv": "693cd82ce674c475bde8212e199d207da8791249cb10f59510fc1b41c1685477"
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
 "runtime_seconds": 20.935,
 "truncated": false
}
```

## deciles.csv (first 12 of 31 lines)
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
static,0,30301,0.789215,0.732913,0.057534,0.064432,0.892941
```

## monthly_factors.csv (first 12 of 211 lines)
```
month,decile,n,fallback,c10,c90
2025-01,0,12267,False,1.020942,0.965193
2025-01,1,10979,False,1.001893,1.016046
2025-01,2,10910,False,1.008296,0.991202
2025-01,3,9731,False,0.995055,0.996588
2025-01,4,8781,False,1.004673,0.977196
2025-01,5,8199,False,0.996963,0.997359
2025-01,6,7494,False,0.976765,0.987994
2025-01,7,7327,False,0.977879,1.004004
2025-01,8,6994,False,0.972058,1.002658
2025-01,9,5158,False,0.948482,1.020483
2025-02,0,11938,False,1.02148,0.965401
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

## quarters.csv (first 12 of 22 lines)
```
quarter,n,coverage,arm
2025Q1,1462,0.736662,control
2025Q2,416,0.766827,control
2025Q3,141,0.794326,control
2025Q4,1903,0.765633,control
2026Q1,520,0.765385,control
2026Q2,478,0.828452,control
2026Q3,384,0.744792,control
2025Q1,1462,0.774966,static
2025Q2,416,0.814904,static
2025Q3,141,0.822695,static
2025Q4,1903,0.797162,static
```

## summary.csv (first 2 of 2 lines)
```
coverage,overall_ok,deciles_ok,pinball_deciles_better,pinball_ok,ic_months,ic_positive_months,ic_ok,n_evaluable_deciles,pinball_needed,enough_deciles_ok,top_decile_year_ok,top_decile_closer_than_control_ok,coins_ok,quarters_ok,n_evaluable_quarters,stability_ok,rules_1_to_4_ok,accepted,stability_max_
0.797503,True,True,10,True,21,21,True,10,8,True,True,True,True,True,6,True,True,True,0.021701,0.004181,0.135601,0.150386,0.135589,0.150386,0.135605,0.150386
```

## year_deciles.csv (first 12 of 61 lines)
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

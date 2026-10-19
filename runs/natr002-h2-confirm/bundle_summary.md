# Bundle summary: natr002-h2-confirm

- status: **OK**
- repo_ref: `cf395ff5466d19024162f634c55ddea848e04c31`
- entry: `research/natr_range/run_natr_confirm.py `
- note: E-natr-002 H2 confirmation (sparse-decile amendment, frozen sha a441aec5). Runs on history_15m after trim-tails-001. Opens H2 once.
- exit code: 0   duration: 28.3 s
- problems: none

## manifest.json
```json
{
 "bucket": "H2",
 "card": "research/natr_range/cards/E-natr-002.md",
 "config": {
  "card": "research/natr_range/cards/E-natr-002.md",
  "data_dir": "/content/drive/MyDrive/history_15m",
  "frozen_rules": "research/natr_range/frozen_rules.json",
  "load_start": "2024-10-01",
  "pattern": "{symbol}.csv.gz"
 },
 "frozen_rules_sha256": "a441aec550f8323a9bcf87a423dfdced1ec84bf5d5e00cc96e5add9c0a7d65da",
 "git_sha": "cf395ff5466d19024162f634c55ddea848e04c31",
 "input_fingerprint": "a9ae4ff07291d13c8f855dd599825a40db07b9cf604955261fb02ab3687490f9",
 "n_configurations": 1,
 "outputs": {
  "access_log_row.csv": "54b796abc6e5cae0c18f83fa32852fde05cb93f21b3374effba1137ea8614b4f",
  "deciles.csv": "4925447ca9b68d6659e69d024b9a35a60fae36c254c7188a4bd33afccafa514b",
  "months.csv": "8e1f7acc96d006829107df83ca428a809e7fc6bd5437001bf4581b852f310972",
  "results.json": "2f4ad7cbd3ed2c1e1c04c056c00b977c35299f6532983ffa983e85e16d741830",
  "slices.csv": "db75e86a7dd2b195fb0238ffc35693e5f4090ec04bbbae8dbbc2a1c2a132682e",
  "summary.csv": "c4277e6cee58cb392882ac8cba3d580dadf690ee6fcfe609057609d3b315aa70"
 },
 "row_counts": {
  "1000000MOGUSDT_rows": 15216,
  "A2ZUSDT_rows": 5598,
  "AINUSDT_rows": 10554,
  "AWEUSDT_rows": 11757,
  "AXLUSDT_rows": 15216,
  "BROCCOLIF3BUSDT_rows": 13219,
  "BUSDT_rows": 11724,
  "CUSDT_rows": 10436,
  "DRIFTUSDT_rows": 15216,
  "EPICUSDT_rows": 13413,
  "ERAUSDT_rows": 10381,
  "ETHWUSDT_rows": 15216,
  "FLMUSDT_rows": 7785,
  "GHSTUSDT_rows": 9633,
  "GRIFFAINUSDT_rows": 15087,
  "HOMEUSDT_rows": 11273,
  "JOEUSDT_rows": 15216,
  "KAITOUSDT_rows": 13912,
  "MYROUSDT_rows": 7617,
  "OMGUSDT_rows": 729,
  "ORDERUSDT_rows": 8680,
  "PUFFERUSDT_rows": 7301,
  "PUNDIXUSDT_rows": 12259,
  "QUSDT_rows": 9261,
  "RENDERUSDT_rows": 15216,
  "SCRUSDT_rows": 15216,
  "STEEMUSDT_rows": 15216,
  "STGUSDT_rows": 15153,
  "SWARMSUSDT_rows": 14968,
  "SYNUSDT_rows": 15216,
  "SYRUPUSDT_rows": 12092,
  "UMAUSDT_rows": 15216,
  "VIRTUALUSDT_rows": 15216,
  "WAXPUSDT_rows": 15216,
  "YFIUSDT_rows": 15216,
  "ZORAUSDT_rows": 10194
 },
 "runtime_seconds": 18.429,
 "truncated": false
}
```

## access_log_row.csv (first 2 of 2 lines)
```
opened_at_utc,bucket,card_path,card_sha256,frozen_rules_sha256,git_head
2026-10-19T20:33:24Z,H2,research/natr_range/cards/E-natr-002.md,fdc9f6c9c83196f9c4aa66ea02f2b020d6784126d5c9cd76c481281f8312a7d7,a441aec550f8323a9bcf87a423dfdced1ec84bf5d5e00cc96e5add9c0a7d65da,cf395ff5466d19024162f634c55ddea848e04c31
```

## deciles.csv (first 11 of 11 lines)
```
decile,n,coverage,baseline_coverage,pinball_model,pinball_baseline,pinball_ratio
0,2309,0.82893,0.72932,0.072538,0.078042,0.929466
1,12649,0.838801,0.749624,0.086564,0.093421,0.926607
2,19932,0.824102,0.743829,0.109167,0.1173,0.930666
3,28494,0.824454,0.753036,0.123623,0.132076,0.935999
4,37081,0.817939,0.754079,0.140346,0.15002,0.935514
5,45905,0.813659,0.756127,0.160613,0.171515,0.936432
6,58087,0.810336,0.76141,0.180649,0.192972,0.936144
7,70437,0.799253,0.761773,0.21144,0.226657,0.932863
8,81172,0.786688,0.76588,0.26964,0.29152,0.924944
9,84768,0.737767,0.767778,0.502876,0.559216,0.899252
```

## months.csv (first 12 of 22 lines)
```
month,ic
2025-01,0.636847
2025-02,0.716919
2025-03,0.656256
2025-04,0.671924
2025-05,0.700916
2025-06,0.62532
2025-07,0.675635
2025-08,0.588073
2025-09,0.638631
2025-10,0.654718
2025-11,0.596008
```

## slices.csv (first 12 of 21 lines)
```
group,year,decile,n,coverage,pinball_ratio
others,2025,0,502,0.850598,0.914626
others,2025,1,4349,0.841113,0.906709
others,2025,2,8335,0.826515,0.913866
others,2025,3,12601,0.828664,0.927724
others,2025,4,16775,0.823666,0.92772
others,2025,5,21814,0.81819,0.929153
others,2025,6,30327,0.815544,0.933447
others,2025,7,40582,0.808265,0.933964
others,2025,8,51691,0.803177,0.928451
others,2025,9,55054,0.765884,0.901768
others,2026,0,1807,0.822911,0.933345
```

## summary.csv (first 2 of 2 lines)
```
coverage,overall_ok,deciles_ok,pinball_deciles_better,pinball_ok,ic_months,ic_positive_months,ic_ok,accepted,n_evaluable_deciles,pinball_needed,enough_deciles_ok,sparse_deciles,coins_ok,n_val_rows,n_val_rows_dropped_no_baseline,baseline_coverage,pinball_model,pinball_baseline,up_p50_pinball_model,up
0.793691,True,False,10,True,21,21,True,False,10,8,True,[],True,440834,0,0.760506,0.248256,0.270262,0.314178,0.322629,0.296793,0.304619,36,0
```

# Bundle summary: stale-bars-001

- status: **OK**
- repo_ref: `de20d964c25b23b9b7ade30b5e91291efb7a0ee7`
- entry: `tools/stale_bars_colab_job.py `
- note: Stale/zero-target audit (claude/zero-target-fix de20d96): zero share per split OFF/ON, dead coins.
- exit code: 0   duration: 809.9 s
- problems: none

## manifest.json
```json
{
 "config": {
  "data_dir": "history_15m",
  "floor": 2,
  "horizon": 1,
  "max_coins": null,
  "min_run": null,
  "quantile": 0.99,
  "timeframes": [
   "1h",
   "4h",
   "1D"
  ],
  "train_pct": 0.8,
  "val_pct": 0.1
 },
 "git_sha": "de20d964c25b23b9b7ade30b5e91291efb7a0ee7",
 "input_fingerprint": "a6d367aa71be0a14ad8b7705fa9d3475320c55253010a7c9e00aae5c26025a1d",
 "n_configurations": 3,
 "outputs": {
  "coins_1D.csv": "16e75e36a37db358b9a7931d1464b49d27f6667af9d1ffed5ea66c14081d15ac",
  "coins_1h.csv": "961eae0b0602ecd66b595a8fcff8d8bc0cd59f13fd67da3fc45c8418db3fde64",
  "coins_4h.csv": "8330e3bb988c9f14410a13496340965a7a587646b949f1d89941e3b9e0a6fd2a",
  "splits_pooled_1D.csv": "473a9a68a15d87e7b48645834e1cfa4b0d4f198f545708524e4861abc34009a9",
  "splits_pooled_1h.csv": "c8d4d6cec4e6792966b339f94a178ee03030fa0f5ae5425c6c7ee10096fea93d",
  "splits_pooled_4h.csv": "53c68b26c0a14ac7815f5c85b2782e8143f6520bf24d715a2602d8d7404f2985",
  "summary.json": "86ba261eb1aede08b9ccc530db57877c51c7958b80fd2579c7977788d933cbe5"
 },
 "row_counts": {
  "coins_1D.csv": 860,
  "coins_1h.csv": 860,
  "coins_4h.csv": 860,
  "splits_pooled_1D.csv": 3,
  "splits_pooled_1h.csv": 3,
  "splits_pooled_4h.csv": 3
 },
 "runtime_seconds": 800.117
}
```

## coins_1D.csv (first 12 of 861 lines)
```
coin,n_bars,last_real_ts,last_ts,tail_bars,dead,longest_flat_run,stale_bars,stale_share,n_rows,zero_target_rows,zero_rows_from_stale
0GUSDT,375,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,374,0,0
1000000BOBUSDT,479,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,478,1,0
1000000MOGUSDT,689,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,688,2,0
1000BONKUSDT,1040,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,1039,0,0
1000BTTCUSDT,76,2022-04-11 00:00:00+00:00,2022-04-11 00:00:00+00:00,0,False,0,0,0.0,75,0,0
1000CATUSDT,706,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,705,0,0
1000CHEEMSUSDT,671,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,670,0,0
1000FLOKIUSDT,1240,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,1239,2,0
1000LUNCUSDT,1479,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,1478,5,0
1000PEPEUSDT,1241,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,1240,0,0
1000RATSUSDT,1017,2026-09-26 00:00:00+00:00,2026-09-26 00:00:00+00:00,0,False,0,0,0.0,1016,2,0
```

## coins_1h.csv (first 12 of 861 lines)
```
coin,n_bars,last_real_ts,last_ts,tail_bars,dead,longest_flat_run,stale_bars,stale_share,n_rows,zero_target_rows,zero_rows_from_stale
0GUSDT,8985,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,0,0,0.0,8984,125,0
1000000BOBUSDT,11488,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,0,0,0.0,11487,336,0
1000000MOGUSDT,16524,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,0,0,0.0,16523,323,0
1000BONKUSDT,24946,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,1,0,0.0,24945,126,0
1000BTTCUSDT,1807,2022-04-11 08:00:00+00:00,2022-04-11 09:00:00+00:00,1,False,1,0,0.0,1806,78,0
1000CATUSDT,16932,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,1,0,0.0,16931,379,0
1000CHEEMSUSDT,16093,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,0,0,0.0,16092,133,0
1000FLOKIUSDT,29744,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,1,0,0.0,29743,416,0
1000LUNCUSDT,35483,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,1,0,0.0,35482,594,0
1000PEPEUSDT,29768,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,1,0,0.0,29767,55,0
1000RATSUSDT,24394,2026-09-26 23:00:00+00:00,2026-09-26 23:00:00+00:00,0,False,1,0,0.0,24393,253,0
```

## coins_4h.csv (first 12 of 861 lines)
```
coin,n_bars,last_real_ts,last_ts,tail_bars,dead,longest_flat_run,stale_bars,stale_share,n_rows,zero_target_rows,zero_rows_from_stale
0GUSDT,2247,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,2246,11,0
1000000BOBUSDT,2872,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,2871,26,0
1000000MOGUSDT,4131,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,4130,53,0
1000BONKUSDT,6237,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,6236,21,0
1000BTTCUSDT,453,2022-04-11 08:00:00+00:00,2022-04-11 08:00:00+00:00,0,False,0,0,0.0,452,4,0
1000CATUSDT,4233,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,4232,47,0
1000CHEEMSUSDT,4024,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,4023,18,0
1000FLOKIUSDT,7436,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,7435,52,0
1000LUNCUSDT,8871,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,8870,69,0
1000PEPEUSDT,7442,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,7441,5,0
1000RATSUSDT,6099,2026-09-26 20:00:00+00:00,2026-09-26 20:00:00+00:00,0,False,0,0,0.0,6098,31,0
```

## splits_pooled_1D.csv (first 4 of 4 lines)
```
min_run,split,n_rows_off,zero_share_off,n_rows_on,zero_share_on,stale_row_share,zero_rows_from_stale_share
853,train,525612,0.070111,513395,0.047984,0.023243,0.331497
853,val,65825,0.164861,64697,0.150301,0.017136,0.103944
853,test,66327,0.159196,65402,0.147304,0.013946,0.087603
```

## splits_pooled_1h.csv (first 4 of 4 lines)
```
min_run,split,n_rows_off,zero_share_off,n_rows_on,zero_share_on,stale_row_share,zero_rows_from_stale_share
6645,train,12637374,0.084888,11844504,0.023631,0.06274,0.73909
6645,val,1579810,0.181881,1422240,0.091242,0.09974,0.548379
6645,test,1579718,0.177416,1444564,0.100455,0.085556,0.482231
```

## splits_pooled_4h.csv (first 4 of 4 lines)
```
min_run,split,n_rows_off,zero_share_off,n_rows_on,zero_share_on,stale_row_share,zero_rows_from_stale_share
5058,train,3158782,0.075969,3085382,0.053987,0.023237,0.305867
5058,val,395038,0.171424,388282,0.157007,0.017102,0.099765
5058,test,395097,0.166281,389595,0.154507,0.013926,0.083748
```

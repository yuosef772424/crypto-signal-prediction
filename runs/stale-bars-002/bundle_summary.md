# Bundle summary: stale-bars-002

- status: **OK**
- repo_ref: `de3a49d9fc9912582ee438ecd33bcc63530b6673`
- entry: `tools/stale_bars_colab_job.py `
- note: NOT QUEUED. Re-run of stale-bars-001 with a time-based threshold (168 h = 7 days: 168 bars at 1h, 42 at 4h, 7 at 1D) instead of derive_min_run (6645/5058/853 bars). Check: 168 h hits no live traded coin (only 10 relisted coins with padded gaps >= 639 h; next-longest live run is 63 h).
- exit code: 0   duration: 740.4 s
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
  "min_run_hours": 168,
  "quantile": 0.99,
  "timeframes": [
   "1h",
   "4h",
   "1D"
  ],
  "train_pct": 0.8,
  "val_pct": 0.1
 },
 "git_sha": "de3a49d9fc9912582ee438ecd33bcc63530b6673",
 "input_fingerprint": "adacdf8cee59c75fe1fa21bdab6de10d1c2c4c45706bb42220c7c7223204bff8",
 "n_configurations": 3,
 "outputs": {
  "coins_1D.csv": "59d98e5b73c81f407e76dae1cd30bf2ffd896e298feb6adbdbb44adc5ed1d89e",
  "coins_1h.csv": "39578091740c7682e041d9700f97169c7344c5bae5cb09af16f9ab6ddd958b48",
  "coins_4h.csv": "438173056bc730b26c4014baeddcce15806143f83ff336f8f17b6bd22ae5f3e1",
  "splits_pooled_1D.csv": "0ce96a2c9744241a6c5089c6dfe2ff14b3238ce30378e03a491de8a99dcfccc4",
  "splits_pooled_1h.csv": "baf1df9d79ddf30520d054f173364b0ae55a22cdb56856e22d5bf634c400739c",
  "splits_pooled_4h.csv": "2ecf37d79793e095d84e864636aa1158f5a3b08e2beb31cb9a2b3ca673423f11",
  "summary.json": "22a83cb868508f87b40b7fdff722ba46b2dce06c7f1141f2c18601730016682e"
 },
 "row_counts": {
  "coins_1D.csv": 860,
  "coins_1h.csv": 860,
  "coins_4h.csv": 860,
  "splits_pooled_1D.csv": 3,
  "splits_pooled_1h.csv": 3,
  "splits_pooled_4h.csv": 3
 },
 "runtime_seconds": 732.701
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
1000BTTCUSDT,1806,2022-04-11 08:00:00+00:00,2022-04-11 08:00:00+00:00,0,False,0,0,0.0,1805,77,0
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
7,train,483426,0.009513,480403,0.003303,0.006253,0.654925
7,val,60732,0.00489,60732,0.00489,0.0,0.0
7,test,60492,0.004166,60492,0.004166,0.0,0.0
```

## splits_pooled_1h.csv (first 4 of 4 lines)
```
min_run,split,n_rows_off,zero_share_off,n_rows_on,zero_share_on,stale_row_share,zero_rows_from_stale_share
168,train,11615735,0.02481,11543280,0.01869,0.006238,0.251382
168,val,1452415,0.026133,1452415,0.026133,0.0,0.0
168,test,1452112,0.025883,1452112,0.025883,0.0,0.0
```

## splits_pooled_4h.csv (first 4 of 4 lines)
```
min_run,split,n_rows_off,zero_share_off,n_rows_on,zero_share_on,stale_row_share,zero_rows_from_stale_share
42,train,2903767,0.015509,2885652,0.009333,0.006238,0.401998
42,val,362912,0.012923,362912,0.012923,0.0,0.0
42,test,363163,0.012628,363163,0.012628,0.0,0.0
```

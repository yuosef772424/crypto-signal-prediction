# Run profile (ok)

wall 2031.6s, run steps total 1997.3s, process peak RSS 9423 MB; python 3.13.16, tensorflow 2.21.0, GPUs ['/physical_device:GPU:0'], CPUs 2, RAM 12.67 GB
top by seconds: train_model, report:verification, report:candle; top by RSS peak: make_splits, report:chicks, report:signals

## Steps (sorted by seconds; kind run = counted in the share, detail = inside a run step, probe = added by the profiler)

| step | kind | seconds | share of run | RSS start MB | RSS peak MB | RSS end MB | GPU peak MB | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| train_model | run | 520.40 | 26.1% | 5060 | 5591 | 5080 | 310 | epochs=2 |
| report:verification | run | 456.44 | 22.9% | 5217 | 6216 | 5741 | 576 | produced_output=True |
| report:candle | run | 431.61 | 21.6% | 5767 | 6146 | 5720 | 575 | produced_output=True |
| report:chicks | run | 338.28 | 16.9% | 5720 | 7082 | 6712 | 608 | produced_output=True |
| report:market_neutral | run | 99.36 | 5.0% | 5741 | 6131 | 5767 | 439 | produced_output=True |
| report:signals | run | 93.05 | 4.7% | 6712 | 7038 | 6141 | 439 | produced_output=True |
| make_splits | run | 36.55 | 1.8% | 2819 | 9407 | 4516 | 0 |  |
| signal_diagnostics | probe | 17.61 | - | 6141 | 6153 | 6153 | 140 | splits=val,test |
| predict_val | probe | 14.28 | - | 5414 | 5415 | 5134 | 254 | batches=668 |
| report:gap | run | 9.90 | 0.5% | 5134 | 6453 | 5217 | 458 | produced_output=True |
| build_training_system | detail | 4.70 | - | 5060 | 5115 | 5115 | 220 |  |
| load_dataset | run | 4.41 | 0.2% | 1247 | 2819 | 2819 | 0 |  |
| make_datasets | run | 4.22 | 0.2% | 4573 | 5278 | 5060 | 137 | batch_size=512; train_rows=2739860 |
| build_model | run | 3.10 | 0.2% | 4516 | 4573 | 4573 | 0 | params=102328 |
| input_pipeline_only | probe | 2.16 | - | 5080 | 5513 | 5414 | 139 | batches=200; first_batch_s=1.730; ms_per_batch=1.097 |
| prepare_chicks | run | 0.01 | 0.0% | 5080 | 5080 | 5080 | 139 |  |
| make_training_config | run | 0.00 | 0.0% | 4573 | 4573 | 4573 | 0 |  |
| apply_dataset_config | run | 0.00 | 0.0% | 2819 | 2819 | 2819 | 0 |  |
| apply_project_config | run | 0.00 | 0.0% | 1151 | 1247 | 1247 | 0 |  |
| plan_model | run | 0.00 | 0.0% | 4516 | 4516 | 4516 | 0 |  |
| retarget | run | 0.00 | 0.0% | 4516 | 4516 | 4516 | 0 |  |

## Training

| epoch | epoch s | steps | step ms (steady) | steps/s | first step s | validation s | other s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 253.68 | 5351 | 42.88 | 23.32 | 11.877 | 11.46 | 0.95 |
| 1 | 240.06 | 5351 | 43.02 | 23.24 | 2.080 | 8.99 | 0.85 |

## Confidence / uncertainty vs the error (within-asset Spearman, permutation null; see tools/signal_diagnostics.py)

- **val**
  -  high confidence  within-asset rho +0.653 (z +336.4), pooled +0.706: INVERTED (opposite sign)
  -  high uncertainty within-asset rho -0.676 (z -343.1), pooled -0.722: INVERTED (opposite sign)
  -   low confidence  within-asset rho +0.671 (z +347.5), pooled +0.741: INVERTED (opposite sign)
  -   low uncertainty within-asset rho -0.696 (z -354.6), pooled -0.756: INVERTED (opposite sign)
  - close confidence  within-asset rho -0.242 (z -122.5), pooled -0.337: informative (expected sign)
  - close uncertainty within-asset rho +0.251 (z +125.7), pooled +0.344: informative (expected sign)
- **test**
  -  high confidence  within-asset rho +0.720 (z +410.4), pooled +0.748: INVERTED (opposite sign)
  -  high uncertainty within-asset rho -0.739 (z -414.2), pooled -0.760: INVERTED (opposite sign)
  -   low confidence  within-asset rho +0.727 (z +420.9), pooled +0.770: INVERTED (opposite sign)
  -   low uncertainty within-asset rho -0.753 (z -427.4), pooled -0.786: INVERTED (opposite sign)
  - close confidence  within-asset rho -0.240 (z -133.6), pooled -0.313: informative (expected sign)
  - close uncertainty within-asset rho +0.246 (z +139.5), pooled +0.319: informative (expected sign)

steady step 42.95 ms (23.28 steps/s); input pipeline alone 1.10 ms/batch; input_bound_ratio 0.026 (near 1 = input-bound). `other s` = gaps between steps + epoch-end callbacks (checkpoint saves...).

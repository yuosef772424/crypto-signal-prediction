# Run profile (ok)

wall 2713.8s, run steps total 2693.5s, process peak RSS 9432 MB; python 3.13.16, tensorflow 2.21.0, GPUs ['/physical_device:GPU:0'], CPUs 2, RAM 12.67 GB
top by seconds: report:chicks, train_model, report:verification; top by RSS peak: make_splits, report:signals, report:chicks

## Steps (sorted by seconds; kind run = counted in the share, detail = inside a run step, probe = added by the profiler)

| step | kind | seconds | share of run | RSS start MB | RSS peak MB | RSS end MB | GPU peak MB | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| report:chicks | run | 776.48 | 28.8% | 5553 | 7014 | 6919 | 313 | produced_output=True |
| train_model | run | 610.68 | 22.7% | 5066 | 5598 | 5204 | 303 | epochs=2 |
| report:verification | run | 544.26 | 20.2% | 5187 | 6200 | 5722 | 431 | produced_output=True |
| report:candle | run | 483.63 | 18.0% | 5369 | 5921 | 5553 | 431 | produced_output=True |
| report:market_neutral | run | 115.87 | 4.3% | 5722 | 5973 | 5369 | 431 | produced_output=True |
| report:signals | run | 98.00 | 3.6% | 6919 | 7311 | 6917 | 431 | produced_output=True |
| make_splits | run | 39.21 | 1.5% | 2829 | 9428 | 4521 | 0 |  |
| predict_val | probe | 17.34 | - | 5537 | 5538 | 5196 | 258 | batches=668 |
| report:gap | run | 11.63 | 0.4% | 5196 | 6437 | 5187 | 458 | produced_output=True |
| make_datasets | run | 5.83 | 0.2% | 4579 | 5285 | 5066 | 137 | batch_size=512; train_rows=2739860 |
| build_training_system | detail | 5.49 | - | 5066 | 5124 | 5124 | 220 |  |
| load_dataset | run | 5.31 | 0.2% | 1257 | 2829 | 2829 | 0 |  |
| input_pipeline_only | probe | 2.69 | - | 5204 | 5636 | 5537 | 139 | batches=200; first_batch_s=2.220; ms_per_batch=1.262 |
| build_model | run | 2.59 | 0.1% | 4521 | 4579 | 4579 | 0 | params=102328 |
| prepare_chicks | run | 0.01 | 0.0% | 5204 | 5204 | 5204 | 139 |  |
| make_training_config | run | 0.00 | 0.0% | 4579 | 4579 | 4579 | 0 |  |
| apply_dataset_config | run | 0.00 | 0.0% | 2829 | 2829 | 2829 | 0 |  |
| apply_project_config | run | 0.00 | 0.0% | 1161 | 1257 | 1257 | 0 |  |
| plan_model | run | 0.00 | 0.0% | 4521 | 4521 | 4521 | 0 |  |
| retarget | run | 0.00 | 0.0% | 4521 | 4521 | 4521 | 0 |  |

## Training

| epoch | epoch s | steps | step ms (steady) | steps/s | first step s | validation s | other s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 314.71 | 5351 | 53.22 | 18.79 | 14.684 | 14.16 | 1.11 |
| 1 | 289.48 | 5351 | 51.56 | 19.39 | 2.331 | 12.53 | 1.05 |

steady step 52.39 ms (19.09 steps/s); input pipeline alone 1.26 ms/batch; input_bound_ratio 0.024 (near 1 = input-bound). `other s` = gaps between steps + epoch-end callbacks (checkpoint saves...).

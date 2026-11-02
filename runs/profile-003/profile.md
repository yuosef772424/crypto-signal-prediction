# Run profile (ok)

wall 1899.6s, run steps total 1886.3s, process peak RSS 9433 MB; python 3.13.16, tensorflow 2.21.0, GPUs ['/physical_device:GPU:0'], CPUs 2, RAM 12.67 GB
top by seconds: report:verification, report:chicks, report:candle; top by RSS peak: make_splits, report:signals, report:chicks

## Steps (sorted by seconds; kind run = counted in the share, detail = inside a run step, probe = added by the profiler)

| step | kind | seconds | share of run | RSS start MB | RSS peak MB | RSS end MB | GPU peak MB | notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| report:verification | run | 455.69 | 24.2% | 5185 | 6128 | 5667 | 431 | produced_output=True |
| report:chicks | run | 446.48 | 23.7% | 5645 | 6452 | 6179 | 308 | produced_output=True |
| report:candle | run | 427.70 | 22.7% | 5647 | 6021 | 5645 | 431 | produced_output=True |
| train_model | run | 412.05 | 21.8% | 5051 | 5508 | 5383 | 297 | epochs=2 |
| report:signals | run | 72.19 | 3.8% | 6179 | 6547 | 5822 | 431 | produced_output=True |
| make_splits | run | 35.51 | 1.9% | 2830 | 9427 | 4526 | 0 |  |
| report:market_neutral | run | 14.51 | 0.8% | 5667 | 5918 | 5647 | 431 | produced_output=False |
| predict_val | probe | 10.84 | - | 5716 | 5717 | 5131 | 227 | batches=668 |
| load_dataset | run | 8.31 | 0.4% | 1258 | 2830 | 2830 | 0 |  |
| report:gap | run | 6.95 | 0.4% | 5131 | 6373 | 5185 | 455 | produced_output=True |
| build_training_system | detail | 5.60 | - | 5051 | 5118 | 5118 | 218 |  |
| make_datasets | run | 4.00 | 0.2% | 4582 | 5272 | 5051 | 134 | batch_size=512; train_rows=2739860 |
| build_model | run | 2.94 | 0.2% | 4526 | 4582 | 4582 | 0 | params=93586 |
| input_pipeline_only | probe | 2.09 | - | 5383 | 5815 | 5716 | 136 | batches=200; first_batch_s=1.712; ms_per_batch=0.964 |
| prepare_chicks | run | 0.01 | 0.0% | 5383 | 5383 | 5383 | 136 |  |
| make_training_config | run | 0.00 | 0.0% | 4582 | 4582 | 4582 | 0 |  |
| apply_dataset_config | run | 0.00 | 0.0% | 2830 | 2830 | 2830 | 0 |  |
| apply_project_config | run | 0.00 | 0.0% | 1162 | 1258 | 1258 | 0 |  |
| plan_model | run | 0.00 | 0.0% | 4526 | 4526 | 4526 | 0 |  |
| retarget | run | 0.00 | 0.0% | 4526 | 4526 | 4526 | 0 |  |

## Training

| epoch | epoch s | steps | step ms (steady) | steps/s | first step s | validation s | other s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 196.47 | 5351 | 33.08 | 30.23 | 9.228 | 9.48 | 0.79 |
| 1 | 209.52 | 5351 | 35.55 | 28.13 | 2.023 | 8.08 | 11.24 |

steady step 34.31 ms (29.14 steps/s); input pipeline alone 0.96 ms/batch; input_bound_ratio 0.028 (near 1 = input-bound). `other s` = gaps between steps + epoch-end callbacks (checkpoint saves...).

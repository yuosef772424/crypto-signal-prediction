# Paired model comparison per timestamp (`tools/paired_compare.py`, T2)

Compare two model arms on the same evaluation set without treating coin-time rows as independent. Plain textbook statistics
(no scipy); the external-research note's citations stay unverified and nothing here relies on them.

## Unit and sign
- The unit is the **timestamp**. `per_timestamp_loss` averages the per-row loss (`abs` or `sq`, or a precomputed per-row loss column)
  over the coins of each timestamp, one value per (timestamp, arm). Rows where y or any arm is non-finite are dropped for all arms
  (same coin set) and counted.
- `d_t = loss_A - loss_B`. **Negative mean = A is better.** `paired_differential` aligns the two arms on the timestamps both have,
  reports how many were dropped per side and raises below `min_overlap`.

## Inference on mean(d)
| Piece | What it does | Caveat |
|---|---|---|
| `newey_west_mean(d, lags=)` | Bartlett-kernel HAC se, t, two-sided p (normal approximation) | `lags`: int, `"nw_default"` = floor(4 (T/100)^(2/9)), or `"andrews"` (AR(1) plug-in) |
| `block_bootstrap_mean(d, block_len=, n_boot=, seed=, kind=)` | resamples whole timestamps in contiguous blocks; percentile CI; two-sided p for H0 mean=0 from the centered resampled means | `kind` = `"stationary"` (Politis-Romano, circular, geometric lengths) or `"moving"`; `block_len` = int, `"auto_nw"` (= NW default lag) or `"auto_andrews"` |
| `temporal_design_effect(x, lags=, clamp=)` | deff = 1 + 2 sum_k (1-k/(L+1)) rho_k, n_eff = T/deff | `clamp="conservative"` clips deff to [1, T]; `"none"` only guards deff >= 1/T. `deff_raw` is always returned |
| `design_effect_by_target(df, targets, ...)` | deff / n_eff table per target for a per-timestamp series: `series="mean"` (a loss column averaged per timestamp) or `"ic"` (per-timestamp Spearman IC) | |
| `compare_arms(df, targets, ...)` + `format_report` | one row per target: n_timestamps, mean A/B/d, NW t/p, bootstrap CI/p, deff, n_eff | data first, text second |

"Contiguous" means adjacent in the sorted aligned timestamp list. With a stride > 1 these are blocks of samples, not wall-clock
spans. Every option is explicit; unknown names raise.

## Measured behaviour (synthetic, pinned seeds; reproduced in `tests/test_paired_compare.py`)
- AR(1), rho = 0.8: the population deff is (1+rho)/(1-rho) = 9, but the Bartlett-weighted estimator is biased low at a finite lag L:
  T=20000 gives 7.5 (L=23), 8.6 (L=100), 9.3 (L=200). The NW default rule (L=5 at T=500) gives ~4. **Use a long lag, or the
  `"andrews"` rule, when persistence is strong, and read deff as a lower bound.**
- 95 % CI coverage of the true mean 0, T=500, 300 simulations: naive iid se 0.44; NW `nw_default` 0.81, NW `andrews` 0.89;
  stationary bootstrap `auto_nw` 0.82, `auto_andrews` 0.88. HAC and block bootstrap are much better than iid but still under-cover
  at rho = 0.8 with T=500 (expected for rules of thumb); at the real ~440 timestamps, treat borderline p-values as borderline.

## Using it
```python
from tools.paired_compare import compare_arms, format_report   # or: sys.path.insert(0, "tools"); import paired_compare
res = compare_arms(df, {"high": ("y_high", "pred_high_a", "pred_high_b")}, ts_col="timestamp", loss="sq",
                   lags="andrews", block_len="auto_andrews", n_boot=5000, seed=0, kind="stationary",
                   min_overlap=100, clamp="conservative")
print(format_report(res))
```
The deff of the per-target IC / loss series is the temporal correction the effective-sample count in `workflow/capacity.py`
(`effective_sample_size`) does not yet apply; `design_effect_by_target` estimates it. Wiring it into the capacity report is not
part of this tool.

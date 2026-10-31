# Audit of external "model staleness / rolling retrain" experiment (Perplexity, repo /home/user/crypto-decay)

Auditor: independent, read-only. Nothing from the repo was executed; I only read code/CSVs and recomputed numbers with my own pandas scripts (/tmp/aud, not in repo). Raw klines (`data/raw/klines_*.parquet`) are Git-LFS pointers (133 bytes), so raw OHLCV could NOT be inspected; everything below that touches raw data is inferred from `universe.csv` and the per-row prediction files (`results/*_preds.csv.gz`, 76-80k rows each, columns symbol,date,month,y,pred,close). The repo has ONE git commit (2026-10-31), so the claimed chronology (selection before sealed look) cannot be verified from history.

## 1. Verdict summary

| Claim (as reported) | Verified? | Evidence | Caveat |
|---|---|---|---|
| Data 2021-01..2026-09, nothing >= 2026-10-01 | Plausible, not provable | `src/download_data.py:32` END="2026-09"; universe months 2021-01..2026-09 (69 months x 100 rows); preds max date 2026-09-29 (y needs 09-30 close) | raw parquet unavailable (LFS pointer) |
| 183 sealed coins excluded | **NO - violated** | No exclusion logic anywhere in `src/` (grep "exclud" = 0 hits). 126 of the 183 symbols are in `universe.csv` (1,186 of 6,900 universe rows); 20.2% of rows in `expA_1d_lgbm_s0_preds`, 21.0% in 1h, 20.7% in sealed | Headline ICs survive removing them (see section 2) |
| Top-100/month incl. delisted (no survivorship) | Mostly | 573 unique symbols; 362-386 symbols appear in eval preds; rank uses same-month quote volume | Membership uses the SAME month's full volume (mild look-ahead in universe selection); stablecoins/EUR/PAXG pairs are NOT filtered (17 symbols, ~4.8 per month) |
| 1d train-once IC negative (LGBM -0.006, MLP -0.028, GRU -0.040) | Yes (numbers reproduce) | Recomputed seed 0: LGBM -0.0055 (NW t -1.2), GRU -0.0394 (t -8.0), MLP -0.016 (seed 0; seed mean -0.024), Ridge -0.0586 (t -6.2) | LGBM is statistically ZERO (t -1.2; -0.002, t -0.4 without stablecoins). Only GRU/MLP/Ridge are reliably negative |
| Monthly retrain IC +0.013 | IC yes; value no | k1_expand IC 0.0126 (t 2.65, 19/27 months positive); 3 seeds 0.0130 +/- 0.0016. Holm p<=0.005 vs train-once (`holm_expB_family.csv`) | Net return after 10bp ~0 (section 4); effect is "negative -> small positive", not evidence of a tradable edge |
| 1h GRU IC up to +0.046 | Yes but seed-driven | GRU seeds: 0.017 / 0.046 / 0.044 (NW t 3.9 / 9.3 / 8.8); MLP seeds 0.006 / -0.029 / 0.044 | Not an "IC up to"; it is a seed lottery. Net return of 1h GRU on dev: -21% ann, Sharpe 0.07 |
| 1h IC not inflated by overlapping 24h targets | Yes | One sample per coin per day at 00:00 UTC (`pipeline.py:100-101`), so 24h targets do not overlap; 0 duplicate (date,symbol); NW lag 24 t even larger (GRU s0: 4.98 vs 3.91 at lag 1) | Report's t uses NW lag 1-2 only (`build_main_tables.py:179`); lag>=24 not used but not needed |
| Train/test gap >= horizon + window | 1h yes, 1d NO | 1h: 56h = 24+32 (`pipeline.py:23`). 1d: 22 days < 1 + 32 (net window) and < 1 + 30 (relvol_30/dist_30 lookbacks). `nonoverlap_verification.csv` itself shows `gap_ok=False` for 1d | Label overlap is covered (horizon = 1 day); only feature-window overlap with train. Low leakage risk, but protocol not met and the "21-day lookback" statement in `pipeline.py:24` is wrong (30-day features) |
| Normalization fit on train only per fold | Yes | `run_experiments.py:69-70` (ridge), `:87` (nets, train windows only); LGBM unscaled; per-fold retrain each time | Early-stopping validation slice (last 15% of train) has no purge from the fitted part (minor) |
| Sealed period never used for selection | Partly / unverifiable | `run_sealed.py:49` hard-codes "best_rolling" = k=3 expanding although dev winner was k=1 (their own later rule); k3 was run on sealed first, THEN `sealed_selected.py` ran k=1 on sealed | Sealed was read twice (report says "read once"). Selection rule uses dev only per their text and k1 vs k3 differ by less than seed noise; contamination risk small but not excluded |
| Seeds >= 3 | Yes for models; no for tests | 3 seeds for lgbm/mlp/gru and rolling configs; ridge/naive0 1 seed; permutation test (exp C) on seed 0 only; Holm/bootstrap on seed 0 only | Seeds are not independent evidence on the same 821 days |
| Block bootstrap / NW / multiple-comparison | Implemented | `metrics.py:29-41` NW; `:160-170` moving-block bootstrap (block 10); `:173-184` Holm over 12 comparisons | Bootstrap p resolution is 1/2000 (Holm "0.0" is a floor). Slope CI iid-bootstrap over 27 monthly points, no HAC. Holm family excludes the other ~50 comparisons/tables the report discusses (model-vs-model, 18 permutation setups, decay slopes) |
| Net-of-cost results | Implemented, simplistic | `metrics.py:120-155`: long-only top 20%, equal weight, cost = 2 x turnover x 10bp | See section 4 |
| Experiments A/B/C/D + calibration complete | A, B, C yes; D, calibration partial | Section 6 | |
| Inversion of the sign wins (Ridge +126%, GRU +97%) = short-term reversal | **NOT supported** | Inverted baskets are static (turnover 0.19 / 0.03). Ridge's bottom-20% basket is EURUSDT, USDCUSDT, FDUSDUSDT (821/821 days), TRXUSDT, BTCUSDT; GRU's is BCH/BNB/BTC/ETH/LTC (821/821 days) | Long-only basket of stablecoins + majors, not a reversal signal. True dollar-neutral top-minus-bottom is negative for all models (Ridge gross -67%, GRU -37%, LGBM -34%) |

## 2. Data hygiene

- Time window: universe 2021-01..2026-09; dev preds 2024-01-01..2026-03-31; sealed preds 2026-04-01..2026-09-29. `download_data.py` iterates monthly archives to 2026-09 and normalises ms/us timestamps (`:86-89`). Max bar timestamp could not be read from raw (LFS pointers). [evidence for preds; inference for raw]
- Excluded symbols: intersection of `docs/research/hardening/external_excluded_symbols.txt` (183) with `universe.csv` symbols = 126; with eval rows = 20% of all rows in every prediction file (dev 1d: 90 distinct excluded symbols, dev 1h: 97, sealed: 51-52). The protocol was breached by the tool, perhaps because it never received the list. Impact check on the headline numbers (my recomputation, seed 0, NW t lag 1):
  - 1d LGBM train-once: all -0.0055 -> without excluded -0.0060 (unchanged).
  - 1d GRU: -0.0394 -> -0.0390; 1d k1_expand: +0.0126 -> +0.0135 (t 2.55).
  - 1h GRU s0: 0.0169 -> 0.0164 (t 3.6); 1h LGBM s0: 0.0089 -> 0.0057 (t 1.2, loses significance); 1h k1_expand: 0.0112 -> 0.0153 (t 3.2).
  So no headline conclusion flips, but the 1h LGBM IC is not robust to the exclusion, and the whole data cut is non-compliant. Note removing them also removes them from the universe, which would change the top-100 composition; I only dropped rows.
- Stablecoins and fiat/gold pairs are in the universe (AEUR, BFUSD, BUSD, EURI, EUR, FDUSD, GBP, PAXG, PAX, RLUSD, TUSD, USD1, USDC, USDE, USDP, UST, XUSD: 4.8% of universe rows). They dominate the "bottom" of predictions (zero-volatility coins) and distort both Spearman IC and the portfolios. Without them: 1d LGBM train-once -0.0019 (t -0.4); 1d GRU -0.0435; 1d k1_expand +0.0148 (t 3.2); 1h GRU s0 +0.0190.
- Survivorship: delisted coins are included via S3 listing (573 symbols) [evidence: universe.csv]. Universe membership for month M is ranked on month-M quote volume, i.e. includes information from the rest of month M at selection time; train rows are filtered the same way. [inference: mild selection bias toward coins that spiked in volume]
- Target construction: 1d y = per-symbol `pct_change().shift(-1)` (`pipeline.py:65`); the shift crosses listing gaps (a missing day would give a multi-day return). Eval dates beyond y availability are dropped.
- One duplicate (date,symbol) row in `expA_1d_lgbm_s0` (disclosed; dedup applied in tables).

## 3. Leakage

- Features (`pipeline.py:28-30, 77-78`): rolling means/std/max/min of closes up to bar t, RSI, BTC residual vol, all computed on bars up to and including t. 1h sample = bar opened 00:00 UTC (its close is known at 01:00), target = close(t) -> close(t+24h), so entry price and features are consistent (no use of the open of the next bar). No target-derived feature found. [evidence, code read]
- Windows for MLP/GRU: last 32 bars ending at t, price rescaled by the window's last close (causal), volume log1p, z-scored with train stats (`run_experiments.py:79-88`). Clean.
- Purge: 1h compliant (56h). 1d: 22 days only; label overlap impossible (1-day horizon), feature overlap with train possible for 10 days. Report discloses 22 vs 33 but wrongly calls the feature lookback 21 days (it is 30). Direction of bias: would slightly flatter nothing in train-once (test is far from train); negligible for headline.
- Rolling training: each fold trains on months strictly before the fold, minus the embargo. Training rows with y NaN dropped. No future information in training found.
- Not leakage but relevant: nets clip y to [-1,1] only for training; IC is evaluated on unclipped y.

## 4. Statistics (recomputed)

| Series (seed 0, dev 821 days) | IC | NW t (lag 1) | NW t (lag 24) | months IC>0 |
|---|---|---|---|---|
| 1d LGBM train-once | -0.0055 | -1.22 | -1.09 | 10/27 |
| 1d GRU train-once | -0.0394 | -7.89 | -8.85 | 1/27 |
| 1d k1_expand LGBM | +0.0126 | +2.65 | +2.61 | 19/27 |
| 1h GRU train-once | +0.0169 | +3.91 | +4.98 | 23/27 |
| 1h LGBM train-once | +0.0089 | +2.04 | +2.42 | 18/27 |
| 1h k1_expand LGBM | +0.0112 | +2.56 | +3.09 | 18/27 |

These match the report's `main_table_*.csv` (e.g. 1d k1_expand 0.0130, t 2.85 averaged over seeds). The t values treat 821 days as the sample and a single seed path; averaging t over seeds (`build_main_tables.py:213`) is not a valid t of the seed-averaged signal.

- Multiple testing: Holm over 12 comparisons (done, correct implementation). Not covered: the ~18 setups x 27 months in the permutation tests, model-vs-model contrasts, decay slopes, the choice of "best" config. Permutation test: 4/27 months p<0.05 for k1_expand (expected 1.35; 1 survives Bonferroni 0.05/27).
- Decay slope: OLS on 27 monthly ICs, p 0.09-0.39 for 1d models. Nothing significant: the evidence for "staleness" as a monotone decay is weak. The report says "staleness exists but gradual" at p=0.09; that is not supported.
- Paired "train-once vs expanding" difference is real in IC terms (+0.018, block bootstrap, Holm p<=0.005) but it is a difference between a model whose IC is ~0 / negative and a model trained on 3 years more data in the SAME regime (2024-26) it is tested in. It does not isolate staleness vs "more recent data"; sliding 12/24-month windows (equally recent data, less history) show NO improvement (-0.003..-0.005 IC, p>0.28). This actually points to the benefit coming from MORE history, not from recency.

## 5. Economics (net of 10bp per side)

- Cost model: long-only top-20% equal weight, daily rebalance, cost = 2 x (1 - overlap) x 10bp. Reasonable but long-only, so the net result is mostly market beta + basket choice; no market-neutral or benchmark-relative variant was reported in the main tables.
- 1d dev (3-seed means, `main_table_1d.csv`): LGBM train-once net -25% ann (Sharpe 0.15, turnover 0.64); k1_expand +0.1% ann (Sharpe 0.45, turnover 0.61); k3_expand -14%; k6_expand -19%. Benchmarks: naive0 (a static basket by row order; zero predictions) +45% ann, equal-weight universe +43% ann (my calc). Cost drag at turnover 0.61 is ~44%/yr. **A +0.013 IC does not survive costs and does not beat a static basket.**
- The "best" net configs are sliding windows (k3_slide12 +48%, Sharpe 0.88; k6_slide24 +46%) whose IC is NEGATIVE (-0.003..-0.005): the profit is basket composition (low-vol/large caps), not forecasting skill. The report notes IC and net ranks disagree but still interprets them.
- 1d GRU/MLP "net +21%/+27%" have turnover 0.02-0.03: near-constant ranking = fixed basket (BTC/ETH/BNB/LTC/BCH for GRU bottom). They match naive0 / equal-weight beta, not skill.
- 1h dev: all configs net negative (-21% to -62% ann, turnover 0.46-0.73); gross edges are far below the 10-30bp/day drag. [evidence]
- Sealed 2026-04..09: naive0 Sharpe 2.98 (1d) / 3.15 (1h) exceeds the selected model's 1.83 / 2.04; the model is below a zero-skill static basket in a rising market. The report states this honestly.
- Sign flip (dev 1d): Ridge inverted +126% net, GRU +97% (matches `expA_1d_inversion_econ.csv`; I reproduce 1.26 and 0.97) but baskets are static stablecoin/major baskets (section 1). Dollar-neutral top-minus-bottom is negative (gross) for Ridge -67%, GRU -37%, LGBM -34%, k1_expand +1%; the reversed long-short would be positive gross for Ridge/GRU (+67%/+37%) but that depends on the same low-turnover basket structure and was not tested net, market-neutral and stablecoin-free. Short-term-reversal as a tradable signal: NOT shown. A positive 1d IC from sign-flipping Ridge/GRU (IC +0.04..+0.06, t~7-8, 25-26/27 months) is a real rank-correlation fact, however, and consistent with reversal in the cross-section (unmodified by costs).
- Stability of negative 1d IC: LGBM negative in 17/27 months with worst 3 months = 36% of seed-0 negative mass (report's 86.5% is on seed-averaged months; I could not reproduce that figure from preds; not material). Ridge/MLP/GRU negative in 23-26 of 27 months: systematic, not driven by a few periods.

## 6. Completeness vs claims

- A (train-once curve): complete for 1d and 1h, 5 models, 3 seeds (ridge/naive0 1 seed). T0 fixed at 2024-01 (a single origin, so "staleness curve" is one path, 27 points).
- B (rolling k=1,3,6; expanding / sliding 12, 24): partial grid. Only 6 configs: expanding k=1,3,6 and sliding 12 (k=1,3) and 24 (k=6). The full cross (k x {expand,12,24}) was not run; only LGBM (no Ridge/GRU rolling). Sliding 24 appears only with k=6.
- C (permutation null): done, 500 permutations, seed 0 only, shuffling y within day across coins.
- D (PSI / adversarial AUC): done for LGBM train-once only, 27 monthly points, Spearman |r| < 0.12 (1d). 4-fold CV AUC on a shuffled sample is not time-aware. Weak evidence; no use in rolling decisions.
- Calibration/probabilistic: 1d only, MLP multi-head, 5 folds over dev, 5 seeds; no sealed calibration, no 1h. Probabilistic metric is NLL/CRPS of a Gaussian head on clipped targets. The finding "heads are conservative, not optimistic" is plausible but rests on clipped targets + weight decay.
- Baselines: Naive0 (constant, ties broken by row order, so it is a static basket, not a "no-skill" return), Ridge, LGBM present.
- Sealed: 3 seeds, one-shot claim contradicted (two configs read, see section 1).
- Missing: dollar-neutral evaluation, benchmark-relative net return, MLP/GRU in rolling, unified significance across the whole grid.

## 7. [evidence] vs [inference]

[evidence] (checked in code/CSV)
- Excluded symbols present (126/183; ~20% of rows); stablecoins present (4.8% of universe).
- Headline ICs and their t values reproduce; seeds, NW and block bootstrap/Holm are implemented.
- 1h samples are daily and non-overlapping; 56h gap; NW lag 24 does not weaken them.
- 1d LGBM train-once is not distinguishable from zero; Ridge/MLP/GRU are systematically negative.
- k1_expand IC +0.0126..0.0130, net ~0%, turnover 0.61; all 1h configs net negative.
- Sliding windows do not improve IC; expanding does.
- Inverted-sign "wins" come from static baskets of stablecoins and majors.
- Sealed read twice; `run_sealed.py` froze k=3, selection later made k=1.

[inference]
- Raw bar timestamps are < 2026-10-01 (cannot read the LFS parquet).
- Most of the expanding-window gain is "more history" rather than "recency/staleness" (sliding-window null supports it, but it was not tested directly with equal-length sliding windows at multiple lengths).
- Universe-by-same-month-volume look-ahead mildly flatters long-only top-volume baskets.
- Selection contamination from the sealed period is probably small (k1 vs k3 difference below seed noise) but the history is a single commit and cannot be audited.

## 8. What we should re-test locally (if anything)

Nothing here justifies adopting a strategy or a model change. The only (weak) item worth a local card: the 1d finding that Ridge/GRU-type models trained on 2021-2023 have a stable NEGATIVE cross-sectional rank-IC (about -0.04 to -0.06, t ~ -7, 23-26/27 months negative) on 2024-2026 while expanding-window LightGBM turns slightly positive (+0.013, t 2.7, vs equal-length sliding windows ~ -0.003). That is a small, regime-dependent reversal/momentum flip at the 1-day cross-sectional level. If tested locally it needs: our own universe (the 183 sealed coins excluded, stablecoins/fiat/gold excluded), dollar-neutral top-minus-bottom evaluation net of 10bp with turnover, expanding vs equal-length sliding windows over several lengths (to separate recency from history), seed-averaged signals, and the sealed period untouched. Any card must first pass `python tools/experiment_registry.py check <card>` against `docs/research/failure_registry.csv` (1d next-day cross-sectional signals on these features have likely already been closed there). The 1h results (24h target sampled daily) are effectively another 1-day horizon problem, seed-unstable (GRU IC 0.017-0.046, MLP -0.029..+0.044) and net-negative after cost, so no card is justified.

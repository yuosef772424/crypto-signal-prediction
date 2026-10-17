# gate_features — E-gate-001 (phase G2): gate-style derived technical states as model inputs

Card: `cards/E-gate-001.md` (committed before any result). Features: `tools/gate_features.py` (`gate-style-v1, unverified`).
Run: `run_gate001.py` + `config_e_gate_001.json`; bundle `results/e_gate_001_run1/` (manifest.json, 9 CSVs + results.json, 26 KB).

## Verdict: direction = information yes, economics NO (F-0080); magnitude = no (F-0081)
Run: 24.6 min, 616 configurations (direction 48, magnitude 48, null refits 520), not truncated, 691,414 rows.

| Direction (state set, best val config) | 4h | 24h | threshold |
|---|---|---|---|
| model / k | GBM / 78 | GBM / 10 | |
| val AUC | 0.5447 | 0.5335 | >= 0.53 PASS / PASS |
| null (20 label shifts, max over 12 configs): max / mean+2.33sd | 0.5139 / 0.5159 | 0.5038 / 0.5074 | val above both: PASS / PASS |
| log-loss gain over intercept, years 2021..2025 positive | 4 of 5 (2025 negative) | 4 of 5 (2024 negative) | >= 4: PASS / PASS |
| k-sweep peak vs null peaks (same model) | 0.5447 vs max 0.5139 | 0.5335 vs max 0.5026 | above every null: PASS / PASS |
| top-decile gross per trade, val | 0.134% | 0.181% | > 0.14% (1x) and > 0.28% (2x): FAIL / FAIL at 2x (24h passes 1x only) |
| compare 2025-01..2026-09 gross | -0.014% | +0.046% | reported only |

Costs (top decile, 0.14% round trip), val / compare:
- 4h: net 1x -0.006% / -0.154%; net 2x -0.146% / -0.294%; best 5% removed 1x -0.235% / -0.377%, 2x -0.375% / -0.517%; day-clustered t of net 1x +3.5 / -0.3; coins (>= 20 trades) with net 1x > 0: 4/10 / 0/10.
- 24h: net 1x +0.041% / -0.094%; net 2x -0.099% / -0.234%; best 5% removed 1x -0.467% / -0.634%, 2x -0.607% / -0.774%; t +2.9 / +1.8; coins net 1x > 0: 5/10 / 2/10. Per-coin table: `results/e_gate_001_run1/percoin.csv`.

Magnitude (state set + NATR columns vs best NATR-only = ridge on NATR_14 of 4 timeframes), val pinball improvement: 4h -0.31%, 24h -0.51% (needs >= +2%), positive years 1/5 and 2/5, null max -2.5% / -4.0%: FAIL.

Other findings: old (raw indicator values) vs new (states) are equal inside the same model (best val AUC 0.5452 vs 0.5447 at 4h, 0.5362 vs 0.5335 at 24h): the F-0027 R2 question is answered, states do not beat raw values. Learning curve 4h rises 0.534 / 0.542 / 0.545 (25/50/100%), 24h is flat. Effective samples about 2,800-3,000 (about 36-38 per state feature). The signal weakens in 2025+ (compare AUC 0.516 / 0.520). A gate-like `signal` (sign of the aggregate) was not tested alone; it is only one of the 78 columns (note for bot2: no standalone verdict, and the economics gate failed for the model built from all states, so no `get_tech_states` spec is proposed).

## Process notes
- Commit 890ef61 is labelled `feat` but contains only the failing tests (history is not rewritten). Commit 5a76852 (`fix`) restores a correct MACD-histogram assertion that 13648ff had removed (the original expectation sign -1 was wrong: for price 100*0.997**n the histogram is +0.008176).
- `manifest.json` records `git_sha` 5a76852; the runner file itself was still uncommitted when the run started and is committed byte-identically in the exp commit.
- Annual 2021-2025 checks use expanding refits (2021 and 2022 are out-of-sample), an interpretation recorded in the card.
- Reproduce: `python tools/fetch_crypto_dataset.py --out /tmp/g2data --interval 5m --coins ADAUSDT BCHUSDT BNBUSDT BTCUSDT DOGEUSDT ETHUSDT SOLUSDT TRXUSDT XRPUSDT ZECUSDT --end "2026-09-30 23:55"` then `python research/gate_features/run_gate001.py --config research/gate_features/config_e_gate_001.json --out <dir>` (config `data_dir` is `/tmp/g2data`, the build session path). Input fingerprint (manifest, sha256 of the per-coin file/row/first/last list): `344211ab1a77aedfe3c1e40cdf8eba7369bd3f3252bbcb45a85d7cf8add31e16`. Output sha256 values are in the manifest.

## H5 confirmation: NOT run, NOT queued
`h5_job.json` is a prepared specification only. The economics gate failed on val, so the card's accept rule gives no reason to open bucket H5 (opening it burns the bucket). It would also need a 15m-data adaptation of the runner (decision table `base_minutes=15`, 15m target grid), which does not exist yet.

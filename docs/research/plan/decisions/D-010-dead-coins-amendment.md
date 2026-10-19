# D-010: R0 lock amendment, dead coins in the sealed buckets

**Status:** Accepted (owner approved trim-tails-001, 2026-10-19). Recorded as a separate note because the lock files under
`docs/research/plan/lock/` are never changed (`holdout.json` sha256 70322ba699d5068e4fbe7f8b014f85492aeebf15d3c743551fbf1062cd889b2c is unchanged; salt unchanged).

**Context.** Some `history_15m` files were padded with flat zero-volume bars after a coin died, so they looked alive until 2026-09 and passed the
"Fresh" test (`last_bar >= 2026-09-01`) of the lock. The H2 run exposed it (7 of 36 coins had dead tails).

**trim-tails-001 (2026-10-19, `tools/trim_dead_tails.py`, owner-approved).** Trimmed the dead zero-volume tails of 167 of 860 coins in Drive
`history_15m`, in place: 5,106,818 15m bars. The tails are kept in `history_15m_tails` with a manifest and are restorable. The tool read only
the volume column; no price or research use. Bundle: `runs/trim-tails-001/` on `origin/colab-results`.

**Dead coins among the sealed (42 of 183, from `holdout.json` buckets intersected with `trimmed_coins.csv`).**

| Bucket | Dead / total | Coins |
|---|---|---|
| H1 | 7 / 36 | ACX BAL MBOX MILK OBOL PONKE QUICK |
| H2 | 7 / 36 | A2Z FLM GHST MYRO OMG PUFFER STG |
| H3 | 0 / 3 | none |
| H4 | 10 / 36 | BAKE COMBO DEGO FIO ICX KDA PERP SC UXLINK ZKJ |
| H5 | 12 / 36 | CUDIS DGB D GLMR KLAY LEVER NEIROETH NFP ORBS REEF VINE WAVES |
| H6 | 6 / 36 | 1000WHY CHESS NULS OM STMX XEM |

(All symbols carry the `USDT` suffix in the files.) The lock selected them because the padded files looked alive.

**Decision.**
1. Bucket membership is NOT changed (the lock is fixed). Coins with a dead tail still contribute their live part; coins without a dead tail per bucket: H1 29, H2 29, H3 3, H4 26, H5 24, H6 30, so the effective live history is shorter than the lock counts suggest.
2. Future bucket runs evaluate only live data: the trimmed files, plus `CONFIG['stale_bars'] = {'min_run_hours': 168}` once PR #30 is merged.
3. A run states the number of coins and rows actually used (as E-natr-002 did: 36/36 coins, with the dead tails removed).

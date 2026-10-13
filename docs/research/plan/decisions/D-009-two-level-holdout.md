# D-009: Two-level holdout (level A untouched, level B daily-only touched)

**Status:** Accepted (owner, 2026-10-13), decided before any manifest of `history_15m` exists. The salt
`b4f3790287a2e12d6e5cb3fa20b2ea81998566e5` is unchanged. Amends the eligibility part of the R0 reviewer decisions.

**Context.** The strict reading ("not touched at any timeframe") leaves a very small pool: a count-only preview on the
860-coin snapshot gave 38 eligible coins, 15 sealed, 2 to 3 coins per bucket, before the `history_1d` rule. R5 needs 30
sealed coins. Most coins were read only at daily resolution, or only through funding/OI/metrics or listing studies, which
cannot have taught anything about 15m path behaviour.

**Decision.**
1. A coin read at any timeframe below 1d (1m, 5m, 15m, 1h, 4h) by any study is touched at level X and excluded.
2. A coin touched only by daily-or-slower or non-price studies (`history_1d` rules, daily cross-sections,
   funding/OI/metrics, listings, categories, CoinMetrics daily) is level B and eligible. A coin never touched is level A
   and eligible.
3. History must be at least 365 days of 15m bars before 2026-10-01 (was 2 years). Last bar at or after 2026-09-01;
   stablecoin and leveraged-token exclusions unchanged; time seal at 2026-10-01 for all coins unchanged.
4. Sealed = 40 % of eligible (A and B together), ranked by `sha256(salt + symbol)`, as before.
5. Buckets stay round-robin by the same hash, except H3 (R2, funding/basis) accepts only coins with no
   funding/OI/basis/metrics touch; a coin refused by a bucket goes to the next bucket in order.
6. `holdout.json` records the level (A or B) of each coin and its touching sources.
7. The CoinMetrics proxy rule `first_bar_before:2021-01-01` is dropped: a daily-only touch never excludes a coin. Its rows
   stay in `touched.csv` as level-B evidence (the 44 assets are not listed in the repo, so they stay unresolved).

**Consequences.** The pool is larger. A level-B coin has been seen by daily-or-slower studies, so a confirmation on it is
weaker evidence than on a level-A coin; every confirmation card states the level mix of its bucket (reported from
`holdout.json`). Under this decision a registry universe that read 1h bars for coins named only as a count (F-0022 and
F-0032 for part of the `history_1d` universe) is not resolvable and is treated as level B.

# Lock: sealed-holdout specification (phase R0)

This file is the **lock specification**. The commit that adds this file defines the salt:

> `salt` = the 40-character lowercase hexadecimal SHA of the commit that first adds
> `docs/research/plan/lock/README.md` (`git log --diff-filter=A --format=%H -- docs/research/plan/lock/README.md`).

It is committed and pushed **before** the Colab manifest of `history_15m` exists, so nobody can choose the rules after seeing
which coins they select. The ROADMAP calls this the "R0 card"; R0 is infrastructure with no experiment card, so this
specification replaces it. The salt is never changed. The pull request that carries this commit must be merged with a merge
commit (never squashed or rebased), otherwise the salt commit leaves the history of `master`.

Decisions are fixed by the reviewer (2026-10-13); this file only states them exactly so that code and tests can follow them.

## 1. Files

| File | Written by | Changes after merge |
|---|---|---|
| `README.md` (this file) | R0 build | never |
| `touched.csv` (touched-data ledger) | `tools/touched_ledger.py`, from `master` | never |
| `manifest/` (the Colab bundle, verbatim) | `tools/build_lock.py ingest` | never |
| `holdout.json`, `holdout.sha256` | `tools/build_lock.py build` | never |
| `holdout_access_log.csv` | `tools/holdout_guard.open_holdout` | rows are appended only |

`holdout.sha256` holds the SHA-256 of `holdout.json` (`sha256sum` format). `tools/holdout_guard.py` refuses to work when the
two disagree, and refuses (fails closed) when `holdout.json` does not exist yet.

## 2. Manifest

`tools/data_manifest.py` runs on Colab over the Drive folder `history_15m` and lists, per file `<SYMBOL>.csv.gz`:
`symbol` (file name without `.csv.gz`, exact spelling), `first_bar` and `last_bar` (bar **open** time, UTC), `rows`
(data rows), `bytes`, `sha256` of the file. It also lists the file names of the other Drive folders named in its config
(for the folder rules of section 4). The bundle it prints is committed verbatim; it is only accepted when `complete` is
true and every section matches the `sha256` in its `manifest.json`.

## 3. Eligibility

A manifest symbol is **eligible** when all of these hold; every failed test is recorded as a reason in `holdout.json`.

1. **Readable:** the file was parsed (`rows > 0`, `first_bar` and `last_bar` present).
2. **History:** `min(last_bar, 2026-09-30T23:45:00Z) - first_bar >= 730 days` (two years of 15m history before the time seal).
3. **Fresh:** `last_bar >= 2026-09-01T00:00:00Z`.
4. **Not a stablecoin:** the base asset (the symbol without a trailing `USDT`) is none of
   `USDC USDP TUSD FDUSD BUSD DAI USDD PYUSD USDE USD1 RLUSD EURI EUR AEUR USDS BFUSD XUSD UST USTC SUSD GUSD LUSD`.
5. **Not a leveraged token:** the base asset is not `<X>UP`, `<X>DOWN`, `<X>BULL` or `<X>BEAR` where `<X>` is itself the
   base asset of another manifest symbol.
6. **Not an index contract:** the base asset is not `BTCDOM` or `DEFI` (Binance index perpetuals, not coins).
7. **Not touched at any timeframe:** the symbol appears in no resolved row of `touched.csv` (section 4).

## 4. Touched-data ledger (`touched.csv`)

Columns: `symbol_or_rule,kind,timeframe,period,source,note`.

- `kind = symbol`: that symbol was read in the stated source (any timeframe counts).
- `kind = rule`: a universe the repo describes without listing its symbols. `symbol_or_rule` is one of
  - `folder:<name>`: every symbol whose file is listed under that Drive folder in the manifest bundle;
  - `first_bar_before:<YYYY-MM-DD>`: every manifest symbol whose `first_bar` is before that date.
  Strict side: a rule counts its whole universe as touched. A rule whose folder is missing from the bundle makes the
  build fail; it is never skipped.
- `kind = time_seal_read`: bars with open time at or after 2026-10-01 that were read after the seal (the fresh window of
  PR #15, the H07 forward ledgers). They are recorded as reads of time-sealed bars by those sources; they do not change
  eligibility, and the sealed-bar reads are listed in `holdout.json`.

Sources: the `universe` and `period` columns of `docs/research/failure_registry.csv` (rows F-0001..F-0079), the
`research/*/README.md` data sections, `research/edge_discovery` (`data_tools/metrics_downloaded.txt`, `listing_events.csv`,
`00_PREREGISTRATION.md`), `docs/research/scripts/entry_range_eval/results/frozen.json`, the symbol lists in the notebooks
(`COINS_BY_CATEGORY`, `REAL_ASSET_NAMES`, the H002 five assets), the ROADMAP data contract (the ten Speirsy11 coins at every
timeframe), and the open branch `origin/claude/quirky-knuth-rpt5lg` (PR #13; it adds no universe beyond `master`).
`tools/touched_ledger.py` regenerates the file from those inputs; each row names its source.

## 5. Sealed set and buckets

1. Let `E` be the eligible symbols and `n = len(E)`.
2. Rank `E` ascending by `sha256(salt + symbol)` (hexadecimal digest of the UTF-8 bytes of the 40-character salt followed by
   the symbol; ties broken by symbol).
3. The first `floor(2n / 5)` symbols (40 % of `n`, integer arithmetic) are **sealed**. The rest of `E` is the open pool.
4. The sealed symbols, in rank order, are assigned to buckets round-robin: rank `i` (from 0) goes to bucket `H(i mod 6 + 1)`.

| Bucket | Phase |
|---|---|
| H1 | R1b confluence confirm |
| H2 | R3 NATR range tool |
| H3 | R2 funding/basis |
| H4 | R5 E-disc S1/S3 confirm |
| H5 | G2 gate-style features |
| H6 | reserve (R6) |

`holdout.json` lists the counts of every step (the funnel), every ineligible symbol with its reasons, the open pool, the
sealed symbols with their buckets, the sha256 of the bundle sections and of `touched.csv`, and the salt.

## 6. Time seal

For every coin, sealed or not, bars with **open time >= 2026-10-01T00:00:00Z** are forward data (R4 and later re-checks).
A request is the half-open interval `[start, end)` of bar open times and is refused when `end > 2026-10-01T00:00:00Z`.

## 7. Access rules

- `tools/holdout_guard.py` is the only way to read data in a research phase: `load_unsealed(...)` and
  `assert_unsealed(symbols, start, end)` raise for a sealed symbol or a request that reaches a time-sealed bar.
- A sealed bucket is opened once, by the single frozen confirmation step of its phase: `open_holdout(card_path, bucket)`.
  The card must carry `confirmation: true` and `frozen_rules_sha256: <64 hex>` in its front matter (and, if it names a
  `holdout_bucket`, the same bucket). A second open of the same bucket raises. The call appends one row to
  `holdout_access_log.csv` before it returns: `opened_at_utc,bucket,card_path,card_sha256,frozen_rules_sha256,git_head`.
- Even an opened bucket never returns a bar at or after the time seal.
- The open pool is for exploration and selection; reading it makes those coins touched for later purposes.

## 8. Order of work

1. This specification is committed and pushed (the salt).
2. The guard, the manifest script and the lock builder are built with tests on fixtures.
3. The owner runs `tools/data_manifest.py` on Colab and pastes the bundle into the build session.
4. `tools/build_lock.py ingest` and `build` produce `manifest/`, `holdout.json` and `holdout.sha256`; they are committed.
5. After merge, only the access log grows.

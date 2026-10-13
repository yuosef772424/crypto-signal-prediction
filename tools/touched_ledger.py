"""
PURPOSE:  Generates the touched-data ledger docs/research/plan/lock/touched.csv (columns symbol_or_rule, kind, timeframe,
          period, source, note) from master as of 2026-10-13: every universe in docs/research/failure_registry.csv
          (F-0001..F-0079), the research/*/README.md data sections, research/edge_discovery's symbol lists, the notebooks'
          symbol lists, the ten Speirsy11 coins at every timeframe, and the reads of time-sealed bars made by PR #15 and the
          H07 forward ledgers. A universe the repo describes without listing symbols becomes a `rule` row. Each touch row
          carries a D-009 `touch_class`: intraday (any timeframe below 1d: level X, excluded), daily (daily-or-slower price
          studies: level B) or derivs (funding/OI/metrics/listings: level B, and H3 refuses the coin).
TAGS:     touched data, ledger, holdout lock, universe, registry, sealed coins, strict side, provenance, R0
PITFALLS: The hourly OI/metrics files behind the 195 perps and the listings are `derivs` by D-009 (b) although edge_discovery
          derived an hourly price proxy from them; the history_1d rule is `daily` although F-0022/F-0032 read hourly bars for
          part of it (not resolvable to symbols). The CoinMetrics 44 assets are not listed in the repo: an `unresolved:` rule,
          no proxy (D-009 h). Registry rows after F-0079 are outside this ledger by design (it is frozen with the lock); an unmapped row up to
          F-0079 raises. The notebook symbol lists (COINS_BY_CATEGORY, REAL_ASSET_NAMES, the H002 five) are copied here as
          literals because the notebooks are being restructured; tests check them against frozen.json and the file lists.

Usage:
    python tools/touched_ledger.py --out docs/research/plan/lock/touched.csv      # refuses to overwrite
    python tools/touched_ledger.py --check docs/research/plan/lock/touched.csv    # exit 1 when the file differs
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLUMNS = ["symbol_or_rule", "kind", "touch_class", "timeframe", "period", "source", "note"]
LAST_REGISTRY_ID = "F-0079"
DATA_TOOLS = os.path.join(ROOT, "research", "edge_discovery", "data_tools")


def _words(text):
    return sorted(text.split())


def _ids(first, last, skip=()):
    return [f"F-{i:04d}" for i in range(first, last + 1) if f"F-{i:04d}" not in skip]


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return sorted(set(f.read().split()))


def _listing_symbols():
    with open(os.path.join(ROOT, "research", "edge_discovery", "listing_events.csv"), encoding="utf-8", newline="") as f:
        return sorted({r["sym"] for r in csv.DictReader(f)})


# crypto_data_pipeline_v6.ipynb, cell 9: COINS_BY_CATEGORY (all seven categories; the 83-coin universe of F-0034..F-0040).
COINS_BY_CATEGORY = """
    BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT AVAXUSDT DOTUSDT DOGEUSDT TRXUSDT LINKUSDT LTCUSDT ATOMUSDT
    UNIUSDT TONUSDT ICPUSDT ETCUSDT FILUSDT APTUSDT XLMUSDT INJUSDT NEARUSDT XMRUSDT OPUSDT AAVEUSDT COMPUSDT SNXUSDT
    SUSHIUSDT LDOUSDT RUNEUSDT FXSUSDT PENDLEUSDT DYDXUSDT TAOUSDT FETUSDT RNDRUSDT AKTUSDT NMRUSDT GRTUSDT PHBUSDT
    AIUSDT SANDUSDT MANAUSDT IMXUSDT GALAUSDT AXSUSDT ENJUSDT YGGUSDT MAGICUSDT PORTALUSDT PIXELUSDT 1000SHIBUSDT
    1000PEPEUSDT SUIUSDT SEIUSDT ALGOUSDT EGLDUSDT ARBUSDT STRKUSDT ZETAUSDT KASUSDT TIAUSDT HBARUSDT EOSUSDT XTZUSDT
    VETUSDT THETAUSDT QNTUSDT CHZUSDT HOTUSDT BATUSDT ZILUSDT ONTUSDT IOTXUSDT ONEUSDT CELOUSDT CFXUSDT ANKRUSDT
    STORJUSDT DENTUSDT GASUSDT NEOUSDT DASHUSDT
"""
# pandas_ta_full_survey.ipynb, cell 18: REAL_ASSET_NAMES (the 14 coins of F-0004 / the pandas_ta batches).
REAL_ASSET_NAMES = """
    BTCUSDT ETHUSDT XRPUSDT ADAUSDT DOGEUSDT DOTUSDT LTCUSDT ATOMUSDT AVAXUSDT LINKUSDT AAVEUSDT TRXUSDT ZECUSDT JASMYUSDT
"""
# hypothesis_h002_classification_head.ipynb, cell 2: "5 coins from history_1d (SOLUSDT, ILVUSDT, PYTHUSDT, RPLUSDT, DRIFTUSDT)".
H002_FIVE = "SOLUSDT ILVUSDT PYTHUSDT RPLUSDT DRIFTUSDT"
SPEIRSY10 = "ADA BCH BNB BTC DOGE ETH SOL TRX XRP ZEC"
# research/edge_discovery/00_PREREGISTRATION.md H18: coins named as untouched; LTC/LINK/AVAX/DOT turned out absent from the data.
H18_ABSENT = "LTCUSDT LINKUSDT AVAXUSDT DOTUSDT"

SRC_10 = ("Speirsy11 10 coins (ROADMAP data contract, tools/fetch_crypto_dataset.py); registry F-0046,F-0053..F-0079; "
          "research/{edge_discovery,strategy_discovery,hedge_recovery,indicator_strategy,h07_*}")

#: key, registry ids, symbols or rule, timeframe, period, source, note
GROUPS = [
    {"key": "speirsy10", "touch_class": "intraday", "registry_ids": ["F-0046"] + _ids(53, 79), "symbols": sorted(c + "USDT" for c in SPEIRSY10.split()),
     "timeframe": "all (1m..1w)", "period": "2017-08..2026-10-08", "source": SRC_10,
     "note": "touched at every timeframe; exploration and selection only, never confirmation"},
    {"key": "h002_5", "touch_class": "daily", "registry_ids": ["F-0001", "F-0002", "F-0003", "F-0028"], "symbols": _words(H002_FIVE),
     "timeframe": "1d", "period": "full history_1d (12 windows 2021-11..2022-03; WIDEHIST 2022-02..2026-09)",
     "source": "F-0001,F-0002,F-0003,F-0028; hypothesis_h002 notebook (SOL ILV PYTH RPL DRIFT)", "note": ""},
    {"key": "survey_14", "touch_class": "daily", "registry_ids": ["F-0004"], "symbols": _words(REAL_ASSET_NAMES),
     "timeframe": "1d", "period": "history_1d, 12 rolling windows (64-bar windows in batch 4)",
     "source": "F-0004; pandas_ta_full_survey notebook REAL_ASSET_NAMES (14 coins)", "note": ""},
    {"key": "history_1d", "touch_class": "daily", "registry_ids": _ids(5, 33, skip=("F-0028",)) + ["F-0035"], "rule": "folder:history_1d",
     "timeframe": "1d", "period": "2019-09..2026-09 (30+38 monthly windows, 267 test days to 2026-09-17)",
     "source": "F-0005..F-0033 except F-0028, and F-0035 (50/47/222/507 coins of Drive history_1d; F-0022 timeframes_1h = the same 50)",
     "note": "only some of these coins are named in the repo, so the whole Drive folder is level B (daily); F-0022 (50 coins, "
             "timeframes_1h) and F-0032 (hourly candles for double-touch days) read hourly bars for coins that cannot be resolved"},
    {"key": "hourly_83", "touch_class": "intraday", "registry_ids": ["F-0034", "F-0036", "F-0037", "F-0038", "F-0039", "F-0040"],
     "symbols": _words(COINS_BY_CATEGORY), "timeframe": "1h;15m;1d",
     "period": "2020-01..2026-09-26 (val 2025-06..2025-11, test 2025-11-22..2026-09-26)",
     "source": "F-0034..F-0040 (83-coin COINS_BY_CATEGORY, entry_range_eval frozen.json 81 coins)", "note": ""},
    {"key": "perps_195", "touch_class": "derivs", "registry_ids": _ids(41, 48, skip=("F-0046",)), "symbols": _read_lines(os.path.join(DATA_TOOLS, "metrics_downloaded.txt")),
     "timeframe": "1h", "period": "2024-01..2026-09 (DISC 2024-01..2025-03, VAL 2025-04..2025-12)",
     "source": "F-0041..F-0045,F-0047,F-0048; research/edge_discovery data_tools/metrics_downloaded.txt (195 perps)", "note": ""},
    {"key": "listings", "touch_class": "derivs", "registry_ids": ["F-0049"], "symbols": _listing_symbols(), "timeframe": "1h;1d",
     "period": "2024-01..2026-06 listings (DISC to 2025-03-31, VAL to 2025-12-31)",
     "source": "F-0049; research/edge_discovery listing_events.csv (= data_tools/newlist_pending.txt, 413 perps)",
     "note": "12_listing.py reads every metrics_new file"},
    {"key": "coinmetrics", "touch_class": "daily", "registry_ids": ["F-0050", "F-0052"], "rule": "unresolved:coinmetrics-44-assets-daily",
     "timeframe": "1d", "period": "2013..2026-05 (fit <= 2023-06-28, test 2024-07..2026-05; HOLD 2023..2026)",
     "source": "F-0050,F-0052; CoinMetrics community daily (44 assets, H13/H15)",
     "note": "the 44 assets are not listed in the repo (18_h13_pr7_signal.py globs a CoinMetrics folder); D-009 h: no proxy, "
             "daily-only touches never exclude a coin"},
    {"key": "hl_copy", "touch_class": "derivs", "registry_ids": ["F-0051"], "symbols": [], "timeframe": "1h", "period": "2026-06-01..2026-10-01",
     "source": "F-0051; research/edge_discovery 19_hl_copy.py (19 Hyperliquid accounts)",
     "note": "prices of the top-25 coins by position come from metrics/metrics_new, i.e. inside perps_195 + listings"},
]
#: extra symbol rows that belong to no registry row
EXTRA = [
    {"touch_class": "intraday", "symbols": _words(H18_ABSENT), "timeframe": "1h;4h", "period": "2018..2026-09",
     "source": "research/edge_discovery 00_PREREGISTRATION.md H18 (coins named as untouched)",
     "note": "named in the universe but absent from the data; counted on the strict side (level X)"},
]
#: reads of bars at or after the time seal (2026-10-01), recorded as reads of time-sealed bars
TIME_SEAL_READS = [
    {"symbols": sorted(c + "USDT" for c in SPEIRSY10.split()), "timeframe": "1d;1h", "period": "2026-10-01..2026-10-08",
     "source": "PR #15 research/strategy_discovery (oos.py, survivor_checks.py, forward.py: fresh window)",
     "note": "fresh bars read after the seal"},
    {"symbols": ["BTCUSDT", "ETHUSDT"], "timeframe": "1d", "period": "2026-10-01..2026-10-03",
     "source": "research/h07_forward ledger.csv + ledger_v3_10coins.csv + ledger_v4_ens_10coins.csv",
     "note": "forward ledger = the intended R4 access, logged here"},
    {"symbols": sorted(c + "USDT" for c in SPEIRSY10.split() if c not in ("BTC", "ETH")), "timeframe": "1d",
     "period": "2026-10-01..2026-10-03", "source": "research/h07_forward ledger_v3_10coins.csv + ledger_v4_ens_10coins.csv",
     "note": "forward ledger = the intended R4 access, logged here"},
    {"symbols": ["HL_TOP25_BY_POSITION"], "timeframe": "1h", "period": "2026-10-01 (at most the first hourly bar)",
     "source": "F-0051 research/edge_discovery 19_hl_copy.py (price end of the snapshot window)",
     "note": "the coins are not listed in the repo; possible one-hour overlap with the time seal"},
]


def check_registry(ids, mapped_ids=None):
    """Every registry id up to LAST_REGISTRY_ID must belong to exactly one group."""
    mapped = set(mapped_ids) if mapped_ids is not None else {i for g in GROUPS for i in g["registry_ids"]}
    missing = sorted(i for i in ids if i <= LAST_REGISTRY_ID and i not in mapped)
    if missing:
        raise ValueError(f"registry rows without a universe group in tools/touched_ledger.py: {missing}")


def generate():
    """The ledger rows (dicts), deterministic."""
    with open(os.path.join(ROOT, "docs", "research", "failure_registry.csv"), encoding="utf-8", newline="") as f:
        check_registry([r["id"] for r in csv.DictReader(f)])
    rows = []
    for g in GROUPS:
        keys = [("rule", g["rule"])] if "rule" in g else [("symbol", s) for s in g["symbols"]]
        for kind, key in keys:
            rows.append({"symbol_or_rule": key, "kind": kind, "touch_class": g["touch_class"], "timeframe": g["timeframe"],
                         "period": g["period"], "source": g["source"], "note": g["note"]})
    for e in EXTRA:
        for s in e["symbols"]:
            rows.append({"symbol_or_rule": s, "kind": "symbol", "touch_class": e["touch_class"], "timeframe": e["timeframe"],
                         "period": e["period"], "source": e["source"], "note": e["note"]})
    for e in TIME_SEAL_READS:
        for s in e["symbols"]:
            rows.append({"symbol_or_rule": s, "kind": "time_seal_read", "touch_class": "", "timeframe": e["timeframe"],
                         "period": e["period"], "source": e["source"], "note": e["note"]})
    return rows


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="write the ledger here (refuses to overwrite)")
    g.add_argument("--check", help="compare this file with the generator output")
    a = ap.parse_args(argv)
    rows = generate()
    if a.out:
        if os.path.exists(a.out):
            raise FileExistsError(f"{a.out} exists; the ledger never changes after the lock")
        write_csv(rows, a.out)
        print(f"wrote {a.out}: {len(rows)} rows")
        return 0
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        write_csv(rows, p)
        same = open(p, "rb").read() == open(a.check, "rb").read()
    print("identical" if same else "DIFFERS")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())

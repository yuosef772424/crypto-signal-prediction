"""
PURPOSE:  E-natr-001 run (phase R3): is a log-log quantile regression on NATR_14 a calibrated forecast of the next 1h / 1d range
          (p10/p50/p90, percent of close)? Fits on train, evaluates on val against per-coin trailing-90-bar quantiles, applies the
          fixed accept rule of research/natr_range/cards/E-natr-001.md and writes the run-contract bundle.
TAGS:     NATR range forecast, E-natr-001, R3, calibration report, coverage, pinball, run contract, exploration
PITFALLS: Reads data ONLY through tools/holdout_guard.load_unsealed (window end <= 2026-10-01). Unknown config keys raise;
          data_dir is required. Stops at --max-minutes and then marks the manifest `truncated: true` (not evidence). This run
          never opens a sealed bucket. Run from the repo root.

Usage:
    python research/natr_range/run_natr_range.py --config research/natr_range/config_e_natr_001.json --out <dir> [--max-minutes 50]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import colab_runner as cr           # noqa: E402
import holdout_guard as hg          # noqa: E402
import natr_range as nr             # noqa: E402

TIMEFRAMES = {"1h": 60, "1d": 1440}
DEFAULTS = {
    "data_dir": None, "coins": ["ADAUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT", "ZECUSDT"],
    "pattern": "ohlc_{symbol}_5m.parquet", "source_minutes": 5, "load_start": "2017-01-01", "end": "2026-10-01",
    "train_start": "2020-01-01", "train_end": "2025-01-01", "val_end": "2026-10-01", "timeframes": ["1h", "1d"],
}


class Deadline(Exception):
    pass


def load_config(path):
    cfg = json.load(open(path))
    bad = set(cfg) - set(DEFAULTS)
    if bad:
        raise KeyError(f"unknown config keys: {sorted(bad)}")
    out = {**DEFAULTS, **cfg}
    if not out["data_dir"]:
        raise KeyError("data_dir is required")
    unknown = set(out["timeframes"]) - set(TIMEFRAMES)
    if unknown:
        raise KeyError(f"unknown timeframes: {sorted(unknown)}")
    return out


def load_coin(data_dir, symbol, pattern, start, end):
    """Bars of one coin through the holdout guard (raises for a sealed symbol or a window past the time seal)."""
    return hg.load_unsealed(data_dir, symbol, start, end, pattern=pattern)


def run(cfg, out_dir, max_minutes):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rows = {tf: [] for tf in cfg["timeframes"]}
    inputs, counts = [], {}
    truncated = False
    try:
        for sym in cfg["coins"]:
            if time.time() - t0 > max_minutes * 60:
                raise Deadline
            d5 = load_coin(cfg["data_dir"], sym, cfg["pattern"], cfg["load_start"], cfg["end"])
            inputs.append(cfg["pattern"].format(symbol=sym))
            counts[f"{sym}_source_bars"] = int(len(d5))
            for tf in cfg["timeframes"]:
                bars = nr.resample_bars(d5, TIMEFRAMES[tf], cfg["source_minutes"])
                r = nr.build_rows(bars, TIMEFRAMES[tf])
                r["coin"] = sym
                rows[tf].append(r)
                counts[f"{sym}_{tf}_rows"] = int(len(r))
            del d5
        tables = {"summary": [], "deciles": [], "months": [], "slices": []}
        results = {}
        for tf in cfg["timeframes"]:
            if time.time() - t0 > max_minutes * 60:
                raise Deadline
            allr = pd.concat(rows[tf])
            tr = allr[(allr["tgt_open"] >= pd.Timestamp(cfg["train_start"], tz="UTC")) & (allr["tgt_open"] < pd.Timestamp(cfg["train_end"], tz="UTC"))]
            va = allr[(allr["tgt_open"] >= pd.Timestamp(cfg["train_end"], tz="UTC")) & (allr["tgt_open"] < pd.Timestamp(cfg["val_end"], tz="UTC"))]
            ev = nr.evaluate_timeframe(tr, va)
            counts[f"{tf}_train_rows"], counts[f"{tf}_val_rows"] = int(len(tr)), int(len(va))
            for name in ("deciles", "months", "slices"):
                df = ev[name].copy()
                df.insert(0, "timeframe", tf)
                tables[name].append(df)
            ex = {k: v for k, v in ev["extras"].items() if not isinstance(v, dict)}
            tables["summary"].append(pd.DataFrame([{"timeframe": tf, **ev["verdict"], **ex, "n_crossed_val": ev["n_crossed"]}]))
            results[tf] = {"verdict": ev["verdict"], "extras": ev["extras"], "n_crossed_val": ev["n_crossed"], "edges": [float(e) for e in ev["edges"]],
                           "models": {f"{tg}_{nr.qname(t)}": {"a": ab[0], "b": ab[1]} for (tg, t), ab in ev["models"].items()}}
    except Deadline:
        truncated = True
        tables, results = {k: [] for k in ("summary", "deciles", "months", "slices")}, {}
    for name, parts in tables.items():
        if parts:
            pd.concat(parts).round(6).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    minutes = (time.time() - t0) / 60
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump({"card": "research/natr_range/cards/E-natr-001.md", "truncated": truncated, "timeframes": results, "runtime_min": round(minutes, 2)},
                  fh, indent=1, sort_keys=True, default=float)
    n_fits = len(cfg["timeframes"]) * len(nr.TARGETS) * len(nr.TAUS)
    manifest = cr.write_manifest(out_dir, cfg, inputs=inputs, input_root=cfg["data_dir"], row_counts=counts, runtime_seconds=time.time() - t0,
                                 n_configurations=n_fits, repo_dir=ROOT,
                                 extra={"truncated": truncated, "card": "research/natr_range/cards/E-natr-001.md",
                                        "accept_rule_evaluations": len(cfg["timeframes"]), "baseline_computations": n_fits})
    return manifest, results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=50)
    a = ap.parse_args(argv)
    manifest, results = run(load_config(a.config), a.out, a.max_minutes)
    print(json.dumps(manifest["outputs"], indent=1))
    print("truncated:", manifest["truncated"], "fits:", manifest["n_configurations"], "minutes:", round(manifest["runtime_seconds"] / 60, 2))
    for tf, r in results.items():
        print(tf, json.dumps(r["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

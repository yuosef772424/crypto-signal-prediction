"""
PURPOSE:  Colab run-contract entry that runs tools/stale_bars_audit.py on the Drive history (history_15m) at several timeframes: per-coin
          tail lengths, coins flagged dead, and the exact-zero close-return share per split with CONFIG['stale_bars'] OFF (all rows)
          and ON (stale rows removed). Answers: are the exact-zero y_close_reg shares (train 7.1%, val 17.1%) caused by dead/forward-filled
          tails, or by genuine zero-return bars of live coins?
TAGS:     colab job, stale bars, delisted coins, zero share, close_reg, run contract, history_15m, audit
PITFALLS: Reads Drive (needs_drive=true); data_dir is relative to CSP_DRIVE_ROOT unless absolute. Every output <= 300 KB (checked, the run
          fails otherwise). min_run null = derived from the data per timeframe (derive_min_run); min_run_hours (optional, exclusive with min_run) = time-based,
          ceil(H / bar hours) per timeframe. The value used is written to summary.json.
          Split cuts are pooled-timestamp quantiles (approximation of the pipeline's global_time split, no embargo). Not queued: see job.json.

Usage:  python tools/stale_bars_colab_job.py --config cfg.json --out DIR --max-minutes N
cfg = {"data_dir": "history_15m", "timeframes": ["1h","4h","1D"], "horizon": 1, "min_run": null, "min_run_hours": null (optional), "quantile": 0.99, "floor": 2,
       "train_pct": 0.8, "val_pct": 0.1, "max_coins": null}
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colab_runner  # noqa: E402
import stale_bars_audit as sba  # noqa: E402

CONFIG_KEYS = ("data_dir", "timeframes", "horizon", "min_run", "quantile", "floor", "train_pct", "val_pct", "max_coins")
OPTIONAL_KEYS = ("min_run_hours",)     # absent == null; at most one of min_run / min_run_hours is non-null
MAX_BYTES = 300 * 1024


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    allowed = set(CONFIG_KEYS) | set(OPTIONAL_KEYS)
    missing, unknown = sorted(set(CONFIG_KEYS) - set(cfg)), sorted(set(cfg) - allowed)
    if missing or unknown:
        raise SystemExit(f"config keys: missing {missing}, unknown {unknown}; allowed {sorted(allowed)}")
    cfg.setdefault("min_run_hours", None)
    if cfg["min_run"] is not None and cfg["min_run_hours"] is not None:
        raise SystemExit("config: min_run and min_run_hours are mutually exclusive (set at most one)")
    if cfg["min_run_hours"] is not None:
        sba.check_min_run_hours(cfg["min_run_hours"])
    if not cfg["timeframes"] or not all(isinstance(t, str) for t in cfg["timeframes"]):
        raise SystemExit("timeframes must be a non-empty list of strings")
    return cfg


def run(cfg, out_dir, deadline):
    root = os.environ.get("CSP_DRIVE_ROOT", "")
    data_dir = cfg["data_dir"] if os.path.isabs(cfg["data_dir"]) else os.path.join(root, cfg["data_dir"])
    if not os.path.isdir(data_dir):
        raise SystemExit(f"data_dir not found: {data_dir}")
    files = sorted(f for f in os.listdir(data_dir) if f.endswith((".csv", ".csv.gz")))
    summary, rows = {}, {}
    for tf in cfg["timeframes"]:
        if time.time() > deadline:
            raise SystemExit("max-minutes reached before all timeframes finished")
        frames = sba.load_dir(data_dir, tf, cfg["max_coins"])
        if cfg["min_run_hours"] is not None:
            min_run = sba.min_run_from_hours(cfg["min_run_hours"], tf)
        elif cfg["min_run"] is not None:
            min_run = cfg["min_run"]
        else:
            min_run = sba.derive_min_run(frames, cfg["quantile"], cfg["floor"])
        coins, per = sba.audit_frames(frames, min_run=min_run, horizon=cfg["horizon"],
                                      splits=sba.split_bounds(frames, cfg["horizon"], cfg["train_pct"], cfg["val_pct"]))
        z = per.groupby("coin")[["zero_target_rows", "zero_rows_from_stale", "n_rows"]].sum()
        coins = coins.merge(z, left_on="coin", right_index=True, how="left")
        keep = ["coin", "n_bars", "last_real_ts", "last_ts", "tail_bars", "dead", "longest_flat_run", "stale_bars", "stale_share",
                "n_rows", "zero_target_rows", "zero_rows_from_stale"]
        coins[keep].round(6).to_csv(os.path.join(out_dir, f"coins_{tf}.csv"), index=False)
        pooled = sba.pool_splits(per)
        pooled.insert(0, "min_run", min_run)
        pooled.round(6).to_csv(os.path.join(out_dir, f"splits_pooled_{tf}.csv"), index=False)
        dead = coins[coins["dead"]].sort_values("tail_bars", ascending=False)
        summary[tf] = {"min_run": int(min_run), "min_run_hours": cfg["min_run_hours"], "n_coins": int(len(coins)), "n_dead": int(len(dead)),
                       "dead_coins_by_tail": dict(zip(dead["coin"].head(40), dead["tail_bars"].head(40).astype(int))),
                       "tail_bars_quantiles": {str(q): float(coins["tail_bars"].quantile(q)) for q in (0.5, 0.9, 0.99, 1.0)},
                       "rows_total": int(per["n_rows"].sum()), "zero_rows_total": int(per["zero_target_rows"].sum()),
                       "zero_rows_from_stale": int(per["zero_rows_from_stale"].sum())}
        rows[f"coins_{tf}.csv"] = len(coins)
        rows[f"splits_pooled_{tf}.csv"] = len(pooled)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for f in os.listdir(out_dir):
        if os.path.getsize(os.path.join(out_dir, f)) > MAX_BYTES:
            raise SystemExit(f"output {f} exceeds 300 KB")
    return files, data_dir, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=50)
    ns = ap.parse_args(argv)
    cfg = load_config(ns.config)
    t0 = time.time()
    os.makedirs(ns.out, exist_ok=True)
    files, data_dir, rows = run(cfg, ns.out, t0 + ns.max_minutes * 60)
    colab_runner.write_manifest(ns.out, config=cfg, inputs=files, input_root=data_dir, row_counts=rows,
                                runtime_seconds=time.time() - t0, n_configurations=len(cfg["timeframes"]))
    print(f"stale-bars job ok: {len(files)} files, timeframes {cfg['timeframes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

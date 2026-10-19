"""
PURPOSE:  Colab run-contract entry that runs tools/trim_dead_tails.py on the Drive folder history_15m: trims the zero-volume tails in
          place (dry_run=false) or only counts them (dry_run=true), and writes summary.json + trimmed_coins.csv.
TAGS:     colab job, dead tail, trim, history_15m, run contract, drive, dry run
PITFALLS: dry_run=false MODIFIES Drive files (the removed rows go to <data_dir>_tails, restorable with `trim_dead_tails.py restore`);
          run the dry-run job first. data_dir is relative to CSP_DRIVE_ROOT unless absolute. Every output <= 300 KB (checked).
          Errors (all-zero files, mismatches) do not abort the run: they are counted in summary.json and listed in trimmed_coins.csv
          (status column); the process exits 1 if any exist. In dry-run the csv lists the coins that would be trimmed.

Usage:  python tools/trim_dead_tails_colab_job.py --config cfg.json --out DIR --max-minutes N
cfg = {"data_dir": "history_15m", "dry_run": true}
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colab_runner  # noqa: E402
import trim_dead_tails as tdt  # noqa: E402

CONFIG_KEYS = ("data_dir", "dry_run")
MAX_BYTES = 300 * 1024
CSV_COLS = ["coin", "status", "error", "file", "original_sha256", "trimmed_sha256", "n_bars_before", "n_tail_bars",
            "first_tail_ts", "last_tail_ts", "last_real_ts", "trimmed_utc"]


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if set(cfg) != set(CONFIG_KEYS):
        raise SystemExit(f"config keys must be exactly {CONFIG_KEYS}, got {sorted(cfg)}")
    if not isinstance(cfg["dry_run"], bool) or not isinstance(cfg["data_dir"], str):
        raise SystemExit("config: data_dir must be a string and dry_run a boolean")
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=50)
    ns = ap.parse_args(argv)
    cfg = load_config(ns.config)
    t0 = time.time()
    os.makedirs(ns.out, exist_ok=True)
    root = os.environ.get("CSP_DRIVE_ROOT", "")
    data_dir = cfg["data_dir"] if os.path.isabs(cfg["data_dir"]) else os.path.join(root, cfg["data_dir"])
    results = tdt.trim_dir(data_dir, cfg["dry_run"], deadline=t0 + ns.max_minutes * 60)
    rows = [r for r in results if r["status"] not in ("live", "already_trimmed")]
    errors = [r for r in results if r["status"] == "error"]
    with open(os.path.join(ns.out, "trimmed_coins.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summary = {"dry_run": cfg["dry_run"], "coins_scanned": len(results),
               "coins_trimmed": sum(r["status"] in ("trimmed", "would_trim") for r in results),
               "total_tail_bars": sum(r["n_tail_bars"] for r in results if r["status"] != "error"),
               "errors": len(errors), "error_coins": {r["coin"]: r["error"] for r in errors}}
    with open(os.path.join(ns.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for f in os.listdir(ns.out):
        if os.path.getsize(os.path.join(ns.out, f)) > MAX_BYTES:
            raise SystemExit(f"output {f} exceeds 300 KB")
    colab_runner.write_manifest(ns.out, config=cfg, inputs=tdt.list_files(data_dir), input_root=data_dir,
                                row_counts={"trimmed_coins.csv": len(rows)}, runtime_seconds=time.time() - t0)
    print(f"trim-tails job ok: {summary}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

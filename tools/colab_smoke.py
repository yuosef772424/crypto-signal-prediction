"""
PURPOSE:  Smoke test for the Colab runner (colab_runner.ipynb): a tiny run-contract script. Reads --config {rows, seed}, writes
          smoke.csv (rows of a deterministic LCG sequence) and manifest.json (git SHA, config, input fingerprint, sha256 of every
          output) through tools/colab_runner.write_manifest. It is the model for any real run script that will be queued.
TAGS:     colab, smoke test, run contract, manifest, runner, job.json, example run script
PITFALLS: Unknown config keys raise (no silent defaults). The output directory is created if missing; --sleep only exists so the
          wall-clock guard can be tested. Needs no data, no network, no Drive.

Usage:
    python tools/colab_smoke.py --config cfg.json --out /tmp/smoke --max-minutes 1 [--sleep 0]
    cfg.json = {"rows": 5, "seed": 7}
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colab_runner  # noqa: E402

CONFIG_KEYS = ("rows", "seed")


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    unknown = sorted(set(cfg) - set(CONFIG_KEYS))
    if unknown:
        raise SystemExit(f"unknown config key(s): {unknown}; allowed: {list(CONFIG_KEYS)}")
    for key in CONFIG_KEYS:
        if not isinstance(cfg.get(key), int) or isinstance(cfg.get(key), bool):
            raise SystemExit(f"config key {key!r} must be an integer")
    if cfg["rows"] < 1:
        raise SystemExit("config key 'rows' must be >= 1")
    return cfg


def sequence(rows, seed):
    x = seed % 2147483647
    for i in range(rows):
        x = (x * 48271 + 11) % 2147483647
        yield i, x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=50)
    ap.add_argument("--sleep", type=float, default=0.0)
    ns = ap.parse_args(argv)
    cfg = load_config(ns.config)
    t0 = time.time()
    os.makedirs(ns.out, exist_ok=True)
    deadline = t0 + ns.max_minutes * 60
    if ns.sleep > 0:
        time.sleep(min(ns.sleep, max(0.0, deadline - time.time())))
    with open(os.path.join(ns.out, "smoke.csv"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("i,value\n")
        for i, v in sequence(cfg["rows"], cfg["seed"]):
            fh.write(f"{i},{v}\n")
    colab_runner.write_manifest(ns.out, config=cfg, row_counts={"smoke.csv": cfg["rows"]}, runtime_seconds=time.time() - t0,
                                n_configurations=1)
    print(f"smoke ok: {cfg['rows']} rows, seed {cfg['seed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

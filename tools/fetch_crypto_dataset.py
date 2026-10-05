"""
PURPOSE:  Download Binance spot OHLCV from the public GitHub dataset Speirsy11/crypto-dataset (monthly parquet per coin and
          interval, stored in Git LFS) and merge it into one parquet per coin: <out>/ohlc_<SYMBOL>_<interval>.parquet with a
          UTC DatetimeIndex and open/high/low/close/volume. Works where Binance itself is blocked (Claude Code cloud) because
          it only needs github.com via git and media.githubusercontent.com.
TAGS:     download history, ohlc, 5m, spot, github dataset, Speirsy11, LFS, media.githubusercontent, research data, H19
PITFALLS: Listing uses a blob-less git clone (`git ls-tree --name-only` only — `-l` would fetch every blob); existing output
          files are skipped (delete to refresh); `--end` trims to a pre-registered test window so a newer snapshot does not
          silently extend it (H19 used 2026-09-30 23:55).

Usage:
    python tools/fetch_crypto_dataset.py --out /home/user/research --interval 5m \\
        --coins BTCUSDT ETHUSDT SOLUSDT ADAUSDT BCHUSDT BNBUSDT DOGEUSDT TRXUSDT XRPUSDT ZECUSDT --end "2026-09-30 23:55"
"""
import argparse
import io
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

REPO_URL = "https://github.com/Speirsy11/crypto-dataset"
MEDIA = "https://media.githubusercontent.com/media/Speirsy11/crypto-dataset/main/"
COLUMNS = ["open", "high", "low", "close", "volume"]


def list_files(clone_dir=None):
    """Every path in the dataset repo, from a blob-less clone (cheap: tree objects only)."""
    clone_dir = clone_dir or os.path.join(tempfile.gettempdir(), "crypto-dataset-tree")
    if not os.path.isdir(os.path.join(clone_dir, ".git")):
        subprocess.check_call(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout",
                               REPO_URL, clone_dir])
    out = subprocess.check_output(["git", "-C", clone_dir, "ls-tree", "-r", "--name-only", "HEAD"], text=True)
    return [p for p in out.split("\n") if p.endswith(".parquet")]


def _get(path, retries=5):
    err = None
    for k in range(retries):
        try:
            r = requests.get(MEDIA + path, timeout=120)
            r.raise_for_status()
            return pd.read_parquet(io.BytesIO(r.content))
        except Exception as e:                                        # noqa: BLE001
            err = e
            time.sleep(2 ** k)
    raise RuntimeError(f"{path}: {err}")


def fetch(coins, interval="5m", out=".", end=None, workers=8, clone_dir=None, verbose=True):
    """Download and merge each coin; returns {symbol: output path}. Existing outputs are kept as they are."""
    os.makedirs(out, exist_ok=True)
    paths = list_files(clone_dir)
    done = {}
    for sym in coins:
        dst = os.path.join(out, f"ohlc_{sym}_{interval}.parquet")
        if os.path.exists(dst):
            done[sym] = dst
            if verbose:
                print("skip (exists)", dst, flush=True)
            continue
        ps = sorted(p for p in paths if f"interval_id={interval}/symbol_id={sym}/" in p)
        if not ps:
            raise ValueError(f"{sym} {interval}: no files in {REPO_URL}")
        with ThreadPoolExecutor(workers) as ex:
            frames = list(ex.map(_get, ps))
        d = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")[COLUMNS]
        if end is not None:
            d = d[d.index <= pd.Timestamp(end, tz="UTC")]
        d.to_parquet(dst)
        done[sym] = dst
        if verbose:
            print(sym, len(ps), "files", d.shape, d.index.min(), "->", d.index.max(), flush=True)
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Download Speirsy11/crypto-dataset OHLCV into one parquet per coin.")
    ap.add_argument("--coins", nargs="+", required=True)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--out", default=".")
    ap.add_argument("--end", default=None, help="inclusive UTC end, e.g. '2026-09-30 23:55'")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    fetch(a.coins, a.interval, a.out, a.end, a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())

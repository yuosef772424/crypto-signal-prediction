#!/usr/bin/env python3
"""
PURPOSE:  Export the prepared dataset that main.ipynb trains on (`<name>_latest.pkl.gz` or an npy_dir `<name>_latest.dataset`) as a contract artifact for
          the PyTorch project nig-timenet-torch (docs/DATA_ARTIFACT_CONTRACT.md there): the same npy_dir layout, with every
          window of a SEALED coin (tools/holdout_guard.py lock) and every window whose label bar is at or after the time seal
          (2026-10-01) removed, and an `export` block added to manifest.json that records the source commit, the coin list, the
          sealed coins excluded, the max bar time and a sha256 of every written file. A new tool only: it reads the dataset and
          the lock, changes no existing code, and its output is a separate directory.
TAGS:     export, torch artifact, nig-timenet-torch, npy_dir, data contract, manifest, holdout_guard, sealed coins, time seal, source commit, T2, D1
PITFALLS: Fails closed: no/invalid lock, a dirty working tree, an unknown per-window key or an asset_bounds that does not tile
          the rows all raise. The sealed list is the whole lock (`export.sealed_excluded`), not only the coins found in the
          dataset (`export.sealed_found`), so the consumer can see that the list was applied in full. `max_bar_time` is the
          latest LABEL bar open time (last window bar + forecast_horizon * base-tf bar), the conservative one; `max_window_time`
          is the last bar inside a window. Windows are DROPPED, never clipped, so no kept window carries a label from the
          sealed period; split dates inside the dataset are left as they are. A per-window array is any array whose first axis
          equals len(last_candles); a list/array in meta.pkl of that length is refused (the tool cannot slice what it does not
          know). Output is written to `<out>.tmp` and renamed; manifest.json is written last (completion marker, like
          data/storage.py::save_dataset_dir, which this tool cannot import: the data package lives in a shared namespace).
          The manifest keeps the npy_dir keys unchanged (format/version/id/created/key_order/arrays), so the old reader still
          opens the directory; `export.schema` is "torch_artifact/1". A `--dataset` ending `.pkl.gz` is read the way
          data/storage.py::load_data_from_drive reads the old format (gzip + pickle, the whole dict in RAM) and split the way
          data/storage.py::save_dataset_dir splits it (numpy arrays without object dtype -> <key>.npy, everything else -> meta.pkl,
          key_order = the dict's key order), so the same content gives the same file hashes from either format; the manifest
          records `export.source_format` ("pkl.gz" | "npy_dir"). Any other file is refused.

Usage:
    python tools/export_torch_artifact.py --dataset /content/drive/MyDrive/crypto_model/preprocessed_data/preprocessing_output_latest.pkl.gz \\
        --out /content/drive/MyDrive/crypto_model/torch_artifacts/preprocessing_output_1
"""
import argparse
import datetime as _dt
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import holdout_guard as hg  # noqa: E402
from core.timeframe import normalize_tf  # noqa: E402

SCHEMA = "torch_artifact/1"
LAST_TS_COL = 3                       # core.schema.TS_COL: last_candles column 3 = bar open time, ns since epoch (float64)
CHUNK_ROWS = 20_000
REQUIRED_META = ("asset_bounds", "base_tf", "forecast_horizon")


class ExportError(Exception):
    """The dataset cannot be exported under the contract (the message says why)."""


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def source_state(root=ROOT):
    """(HEAD sha, dirty) of the repo that holds this tool; dirty = tracked files differ from HEAD."""
    sha = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    return sha, dirty


def read_npy_dir(path):
    """-> (npy_dir manifest, arrays {key: memmap}, meta {key: value}). Mirrors data/storage.py::load_dataset_dir."""
    man_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(man_path):
        raise ExportError(f"{path}: no manifest.json (incomplete or not an npy_dir dataset)")
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    if man.get("format") != "npy_dir" or man.get("version") != 1:
        raise ExportError(f"{path}: unsupported manifest format/version {man.get('format')!r}/{man.get('version')!r}")
    with open(os.path.join(path, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)               # our own pipeline's output (same trust as data/storage.py)
    arrays = {}
    for k, spec in man["arrays"].items():
        a = np.load(os.path.join(path, f"{k}.npy"), mmap_mode="r", allow_pickle=False)
        if list(a.shape) != list(spec["shape"]) or str(a.dtype) != spec["dtype"]:
            raise ExportError(f"{path}: {k}.npy is {a.dtype}{list(a.shape)}, manifest says {spec['dtype']}{spec['shape']}")
        arrays[k] = a
    return man, arrays, meta


def read_pkl_gz(path):
    """-> (npy_dir-style manifest, arrays {key: ndarray}, meta). The old `.pkl.gz` format of data/storage.py::load_data_from_drive,
    split as save_dataset_dir would split it. Everything is loaded into RAM (what the old reader does)."""
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)               # our own pipeline's output (same trust as data/storage.py)
    if not isinstance(data, dict):
        raise ExportError(f"{path}: the pickle holds a {type(data).__name__}, not a dataset dict")
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray) and not v.dtype.hasobject}
    meta = {k: v for k, v in data.items() if k not in arrays}
    man = {"format": "pkl.gz", "version": 1, "id": None, "created": None, "key_order": list(data),
           "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}
    return man, arrays, meta


def read_dataset(path):
    """Dispatch on what ``path`` is: a directory -> npy_dir, a file ending .pkl.gz -> the old format, else refused."""
    if os.path.isdir(path):
        return read_npy_dir(path)
    if os.path.isfile(path) and str(path).endswith(".pkl.gz"):
        return read_pkl_gz(path)
    raise ExportError(f"{path}: not an npy_dir directory and not a .pkl.gz file (or it does not exist)")


def bar_delta(tf):
    """Timedelta of a timeframe string ('15m', '1h', '1d'); anything pandas cannot read as a positive duration raises."""
    try:
        d = pd.Timedelta(normalize_tf(tf))
    except (ValueError, TypeError) as e:
        raise ExportError(f"base_tf {tf!r} is not a timeframe this tool can convert to a duration") from e
    if d <= pd.Timedelta(0):
        raise ExportError(f"base_tf {tf!r} is not a positive duration")
    return d


def _iso(ns):
    return pd.Timestamp(int(ns), tz="UTC").isoformat()


def plan_rows(arrays, meta, lock):
    """Which rows to keep. -> dict(keep=int64 row indices, n_rows, sealed_found, dropped_sealed, dropped_time, bounds)."""
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing or "last_candles" not in arrays:
        raise ExportError(f"dataset lacks required keys: {missing + ([] if 'last_candles' in arrays else ['last_candles'])}")
    n = arrays["last_candles"].shape[0]
    bounds = meta["asset_bounds"]
    pos = 0
    for b in bounds:
        if b["start"] != pos or b["end"] < b["start"]:
            raise ExportError(f"asset_bounds do not tile the rows at {b['name']}: start {b['start']} != {pos}")
        pos = b["end"]
    if pos != n:
        raise ExportError(f"asset_bounds cover {pos} rows, last_candles has {n}")
    ts = np.asarray(arrays["last_candles"][:, LAST_TS_COL], dtype="float64")
    if not np.isfinite(ts).all():
        raise ExportError("last_candles has a non-finite timestamp")
    ts = ts.astype("int64")
    label_ns = ts + int(meta["forecast_horizon"]) * int(bar_delta(meta["base_tf"]).value)
    seal_ns = int(lock.time_seal.value)
    time_ok = label_ns < seal_ns
    unsealed = np.ones(n, dtype=bool)
    sealed_found = []
    for b in bounds:
        if hg._sealed_hits([b["name"]], lock):                     # same matching as the guard (case, trailing USDT)
            unsealed[b["start"]:b["end"]] = False
            sealed_found.append(b["name"])
    keep = np.flatnonzero(unsealed & time_ok)
    return {"keep": keep, "n_rows": n, "sealed_found": sealed_found,
            "dropped_sealed": int((~unsealed).sum()),
            "dropped_time": int((unsealed & ~time_ok).sum()),
            "ts": ts, "label_ns": label_ns, "bounds": bounds}


def _check_per_window_meta(meta, n):
    for k, v in meta.items():
        if k == "asset_bounds":
            continue
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n:
            raise ExportError(f"meta key {k!r} has length N={n}: it looks per-window and this tool cannot slice it")


def _write_chunked(dest, arr, keep):
    out = np.lib.format.open_memmap(dest, mode="w+", dtype=arr.dtype, shape=(len(keep),) + arr.shape[1:])
    for s in range(0, len(keep), CHUNK_ROWS):
        out[s:s + CHUNK_ROWS] = arr[keep[s:s + CHUNK_ROWS]]
    out.flush()
    del out


def export(dataset_dir, out_dir, *, lock=None, source=None, now=None):
    """Write the contract artifact to ``out_dir``; returns its manifest dict. ``source`` = (sha, dirty), default from git."""
    lock = hg.load_lock() if lock is None else lock
    sha, dirty = source if source is not None else source_state()
    if dirty:
        raise ExportError("the working tree has uncommitted changes to tracked files: commit first, the manifest records a commit")
    man, arrays, meta = read_dataset(dataset_dir)
    plan = plan_rows(arrays, meta, lock)
    keep = plan["keep"]
    if len(keep) == 0:
        raise ExportError("no window is left after removing sealed coins and the sealed period")
    _check_per_window_meta(meta, plan["n_rows"])

    new_bounds, pos = [], 0
    for b in plan["bounds"]:
        cnt = int(((keep >= b["start"]) & (keep < b["end"])).sum())
        if cnt:
            nb = dict(b)
            nb["start"], nb["end"] = pos, pos + cnt
            new_bounds.append(nb)
            pos += cnt
    out_dir = os.path.abspath(out_dir)
    tmp = out_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    n_old = plan["n_rows"]
    files = {}
    new_arrays = {}
    for k, a in arrays.items():
        if a.shape[0] == n_old:
            _write_chunked(os.path.join(tmp, f"{k}.npy"), a, keep)
            new_arrays[k] = {"shape": [len(keep)] + list(a.shape[1:]), "dtype": str(a.dtype)}
        else:
            raise ExportError(f"array {k!r} has first axis {a.shape[0]} != N={n_old}: not a per-window array, cannot export")
    meta = dict(meta)
    meta["asset_bounds"] = new_bounds
    with open(os.path.join(tmp, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    for k in sorted(new_arrays):
        files[f"{k}.npy"] = _sha256_file(os.path.join(tmp, f"{k}.npy"))
    files["meta.pkl"] = _sha256_file(os.path.join(tmp, "meta.pkl"))

    kept_label_ns, kept_ts = plan["label_ns"][keep], plan["ts"][keep]
    when = (now or _dt.datetime.now(_dt.timezone.utc)).isoformat()
    new_man = {"format": "npy_dir", "version": 1, "id": uuid.uuid4().hex, "created": when,
               "key_order": list(man["key_order"]), "arrays": new_arrays,
               "export": {
                   "schema": SCHEMA,
                   "source_commit": sha,
                   "source_repo": "yuosef772424/crypto-signal-prediction",
                   "source_format": man["format"],
                   "source_dataset_id": man.get("id"),
                   "source_dataset_created": man.get("created"),
                   "coins": [b["name"] for b in new_bounds],
                   "sealed_excluded": sorted(lock.sealed),
                   "sealed_found": sorted(plan["sealed_found"]),
                   "lock_sha256": _sha256_file(os.path.join(lock.lock_dir, "holdout.json")),
                   "time_seal": lock.time_seal.isoformat(),
                   "max_bar_time": _iso(kept_label_ns.max()),
                   "max_window_time": _iso(kept_ts.max()),
                   "rows": len(keep), "rows_source": n_old,
                   "dropped_sealed_rows": plan["dropped_sealed"], "dropped_time_rows": plan["dropped_time"],
                   "files": files,
                   "data_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}}
    with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(new_man, f)                                     # last = completion marker
    shutil.rmtree(out_dir, ignore_errors=True)
    os.rename(tmp, out_dir)
    return new_man


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export the training dataset as a nig-timenet-torch contract artifact.")
    ap.add_argument("--dataset", required=True, help="preprocessing_output_latest.pkl.gz (what main.ipynb trains on) or an npy_dir directory (.dataset)")
    ap.add_argument("--out", required=True, help="output directory (replaced if it exists)")
    args = ap.parse_args(argv)
    try:
        m = export(args.dataset, args.out)
    except (ExportError, hg.GuardError) as e:
        print(f"export refused: {e}", file=sys.stderr)
        return 1
    e = m["export"]
    print(f"exported {e['rows']}/{e['rows_source']} windows, {len(e['coins'])} coins, max_bar_time {e['max_bar_time']}, "
          f"sealed found {len(e['sealed_found'])}/{len(e['sealed_excluded'])}, data_sha256 {e['data_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
PURPOSE:  Trim the dead (zero-volume) tail of each coin's 15m history file IN PLACE (delisted / migrated coins are padded with flat
          zero-volume bars up to the end of the data). The removed rows are saved to <data_dir>_tails/<same file name> (same format,
          same compression) so the original is exactly restorable (`restore`). Per-coin audit row in <data_dir>_tails/manifest.csv.
TAGS:     dead tail, zero volume, delisted coins, trim at source, history_15m, drive, restore, sha256, stale bars, padded bars,
          survivorship, data cleaning
PITFALLS: Only bars with volume == 0 at the very END are removed: walk back from the last bar, stop at the first volume > 0 (or an
          unparseable / empty volume, or a blank line: conservative, nothing past it is touched). Zero-volume bars in mid-history stay.
          Works on the TEXT of the file (header + lines, no float round trip), so the trimmed + tail files concatenate back to the
          original text byte for byte. Equality check = sha256 of the DECOMPRESSED content (original_content_sha256), not of the file
          bytes: gzip bytes depend on compressor/level/mtime, so a re-compressed .csv.gz is not guaranteed byte-identical (a plain
          .csv is: original_sha256 == restored file hash is also checked then, and reported for .gz as file_bytes_identical).
          original_sha256 is the hash of the original FILE and is never overwritten by a re-run. A file whose every bar has volume 0
          is skipped and recorded as an error (all_zero_volume), never emptied. Each replace is atomic (temp file + os.replace); the
          tail file is written before the original is replaced, and any mismatch leaves the original untouched. Not for concurrent
          writers. A coin already in the manifest whose file changed since the trim, or an existing tail file with different
          content, is an error (no action).

Usage:  python tools/trim_dead_tails.py trim <data_dir> [--dry-run]
        python tools/trim_dead_tails.py restore <data_dir> (<COIN> | --all)
Colab:  tools/trim_dead_tails_colab_job.py (job specs tools/jobs/trim_tails_001*.json).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import io
import os
import sys
from typing import Dict, List, Optional

MANIFEST_COLS = ["coin", "file", "original_sha256", "original_content_sha256", "trimmed_sha256", "n_bars_before", "n_tail_bars",
                 "first_tail_ts", "last_tail_ts", "last_real_ts", "trimmed_utc"]
TS_COLS = ("timestamp", "datetime_utc")
SUFFIXES = (".csv.gz", ".csv")


def tails_dir(data_dir: str) -> str:
    return os.path.normpath(data_dir) + "_tails"


def coin_of(name: str) -> str:
    return name.split(".")[0]


def list_files(data_dir: str) -> List[str]:
    return sorted(f for f in os.listdir(data_dir) if f.endswith(SUFFIXES))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_content(path: str) -> bytes:
    """Decompressed bytes of a .csv / .csv.gz file."""
    with open(path, "rb") as fh:
        raw = fh.read()
    return gzip.decompress(raw) if path.endswith(".gz") else raw


def write_atomic(path: str, content: bytes) -> None:
    """Write `content` (compressed if path ends with .gz; mtime 0 so the bytes are deterministic) via temp file + os.replace."""
    data = gzip.compress(content, compresslevel=6, mtime=0) if path.endswith(".gz") else content
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _fields(line: bytes) -> List[str]:
    return next(csv.reader([line.decode("utf-8").rstrip("\r\n")]))


def split_dead_tail(content: bytes) -> dict:
    """Locate the dead tail of one file's text. Returns {header, lines, cut, n_bars, ts_idx, error}: lines[cut:] are the dead rows
    (cut == len(lines) -> none). error set -> nothing to do (no_volume_column / empty / all_zero_volume)."""
    lines = content.splitlines(keepends=True)
    if not lines:
        return {"error": "empty_file"}
    header, rows = lines[0], lines[1:]
    cols = [c.strip().lower() for c in _fields(header)]
    if "volume" not in cols:
        return {"error": "no_volume_column"}
    vi = cols.index("volume")
    ti = next((cols.index(c) for c in TS_COLS if c in cols), None)
    cut = len(rows)
    while cut > 0:
        try:
            f = _fields(rows[cut - 1])
            if float(f[vi]) != 0.0:       # NaN != 0 too: an unreadable value stops the walk and is never trimmed
                break
        except (ValueError, IndexError, UnicodeDecodeError, StopIteration):
            break
        cut -= 1
    if cut == 0 and rows:
        return {"error": "all_zero_volume"}
    return {"header": header, "rows": rows, "cut": cut, "n_bars": len(rows), "ts_idx": ti, "error": None}


def _ts(line: bytes, ti: Optional[int]) -> str:
    return _fields(line)[ti] if ti is not None else ""


def read_manifest(tdir: str) -> Dict[str, dict]:
    p = os.path.join(tdir, "manifest.csv")
    if not os.path.isfile(p):
        return {}
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["coin"]: r for r in csv.DictReader(fh)}


def write_manifest(tdir: str, rows: Dict[str, dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=MANIFEST_COLS, lineterminator="\n")
    w.writeheader()
    for c in sorted(rows):
        w.writerow({k: rows[c].get(k, "") for k in MANIFEST_COLS})
    write_atomic(os.path.join(tdir, "manifest.csv"), buf.getvalue().encode("utf-8"))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trim_file(data_dir: str, name: str, manifest: Dict[str, dict], dry_run: bool = False) -> dict:
    """Trim one coin file. Returns a result dict: status in {live, trimmed, would_trim, already_trimmed, error}."""
    coin, path = coin_of(name), os.path.join(data_dir, name)
    tdir = tails_dir(data_dir)
    tpath = os.path.join(tdir, name)
    res = {"coin": coin, "file": name, "status": "live", "n_tail_bars": 0, "error": ""}
    try:
        content = read_content(path)
        sp = split_dead_tail(content)
        if sp["error"]:
            return {**res, "status": "error", "error": sp["error"]}
        rows, cut, ti = sp["rows"], sp["cut"], sp["ts_idx"]
        res["n_bars_before"] = sp["n_bars"]
        if cut == len(rows):
            if coin in manifest:
                res["status"] = "already_trimmed"
            return res
        tail = rows[cut:]
        res.update(n_tail_bars=len(tail), first_tail_ts=_ts(tail[0], ti), last_tail_ts=_ts(tail[-1], ti),
                   last_real_ts=_ts(rows[cut - 1], ti))
        if dry_run:
            res["status"] = "would_trim"
            return res
        file_sha = sha256_file(path)
        if coin in manifest and manifest[coin]["trimmed_sha256"] != file_sha:
            return {**res, "status": "error", "error": "in_manifest_but_file_changed_since_trim"}
        trimmed = sp["header"] + b"".join(rows[:cut])
        tail_content = sp["header"] + b"".join(tail)
        # safety: trimmed + tail (minus its repeated header) must reproduce the original text exactly
        rebuilt = trimmed + tail_content[len(sp["header"]):]
        if rebuilt != content:
            return {**res, "status": "error", "error": "rebuild_mismatch"}
        if os.path.exists(tpath) and read_content(tpath) != tail_content:
            return {**res, "status": "error", "error": "tail_file_exists_with_different_content"}
        os.makedirs(tdir, exist_ok=True)
        write_atomic(tpath, tail_content)
        if read_content(tpath) != tail_content:
            return {**res, "status": "error", "error": "tail_readback_mismatch"}
        # the temp trimmed file is verified before it replaces the original
        tmp = path + ".trim.tmp"
        data = gzip.compress(trimmed, compresslevel=6, mtime=0) if name.endswith(".gz") else trimmed
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        back = gzip.decompress(open(tmp, "rb").read()) if name.endswith(".gz") else open(tmp, "rb").read()
        if back != trimmed or back + tail_content[len(sp["header"]):] != content:
            os.remove(tmp)
            return {**res, "status": "error", "error": "trimmed_readback_mismatch"}
        trimmed_sha = sha256_file(tmp)
        os.replace(tmp, path)
        res.update(status="trimmed", original_sha256=file_sha, original_content_sha256=sha256_bytes(content),
                   trimmed_sha256=trimmed_sha, trimmed_utc=_now())
        manifest[coin] = {"coin": coin, "file": name, **{k: res[k] for k in MANIFEST_COLS[2:] if k in res}}
        write_manifest(tdir, manifest)
        return res
    except Exception as e:  # noqa: BLE001 - one bad file must not stop the others; recorded as an error row
        return {**res, "status": "error", "error": f"{type(e).__name__}: {e}"}


def trim_dir(data_dir: str, dry_run: bool = False, deadline: Optional[float] = None) -> List[dict]:
    if not os.path.isdir(data_dir):
        raise SystemExit(f"data_dir not found: {data_dir}")
    manifest = read_manifest(tails_dir(data_dir))
    out = []
    import time
    for name in list_files(data_dir):
        if deadline is not None and time.time() > deadline:
            raise SystemExit("max-minutes reached before all files were processed (finished coins stay trimmed; re-run is idempotent)")
        out.append(trim_file(data_dir, name, manifest, dry_run))
    return out


def restore_coin(data_dir: str, coin: str) -> dict:
    tdir = tails_dir(data_dir)
    manifest = read_manifest(tdir)
    if coin not in manifest:
        return {"coin": coin, "status": "error", "error": "not_in_manifest"}
    row = manifest[coin]
    name = row["file"]
    path, tpath = os.path.join(data_dir, name), os.path.join(tdir, name)
    try:
        if sha256_file(path) != row["trimmed_sha256"]:
            return {"coin": coin, "status": "error", "error": "file_changed_since_trim"}
        trimmed, tail = read_content(path), read_content(tpath)
        header_len = len(trimmed.splitlines(keepends=True)[0])
        if tail[:header_len] != trimmed[:header_len]:
            return {"coin": coin, "status": "error", "error": "header_mismatch"}
        content = trimmed + tail[header_len:]
        if sha256_bytes(content) != row["original_content_sha256"]:
            return {"coin": coin, "status": "error", "error": "content_sha256_mismatch"}
        write_atomic(path, content)
        identical = sha256_file(path) == row["original_sha256"]
        if not name.endswith(".gz") and not identical:
            return {"coin": coin, "status": "error", "error": "file_sha256_mismatch_after_restore"}
        os.replace(tpath, tpath + ".restored")
        del manifest[coin]
        write_manifest(tdir, manifest)
        return {"coin": coin, "status": "restored", "file_bytes_identical": identical, "error": ""}
    except Exception as e:  # noqa: BLE001
        return {"coin": coin, "status": "error", "error": f"{type(e).__name__}: {e}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("trim")
    t.add_argument("data_dir")
    t.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("restore")
    r.add_argument("data_dir")
    r.add_argument("coin", nargs="?")
    r.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "restore":
        if bool(a.coin) == bool(a.all):
            ap.error("restore: give exactly one of <COIN> or --all")
        coins = sorted(read_manifest(tails_dir(a.data_dir))) if a.all else [a.coin]
        results = [restore_coin(a.data_dir, c) for c in coins]
        for x in results:
            print(x["coin"], x["status"], x.get("error", ""))
        return 1 if any(x["status"] == "error" for x in results) else 0
    results = trim_dir(a.data_dir, getattr(a, "dry_run", False)) if a.cmd == "trim" else None
    if results is None:
        ap.error("command required: trim | restore")
    for x in results:
        if x["status"] != "live":
            print(x["coin"], x["status"], x["n_tail_bars"], x.get("error", ""))
    print(f"scanned {len(results)}, tail bars {sum(x['n_tail_bars'] for x in results)}, "
          f"errors {sum(x['status'] == 'error' for x in results)}")
    return 1 if any(x["status"] == "error" for x in results) else 0


if __name__ == "__main__":
    sys.exit(main())

"""
PURPOSE:  Colab script that lists the Drive folder history_15m (one row per <SYMBOL>.csv.gz: first/last bar open time, data
          rows, bytes, file sha256) and the file names of the other Drive folders named in its config (for the folder rules
          of the touched-data ledger). It follows the run contract: --config (unknown keys raise), --out, --max-minutes,
          manifest.json with the sha256 of every output, outputs <= 300 KB, and a printed paste bundle that
          tools/build_lock.py ingests. Standard library only, so one raw file runs on Colab.
TAGS:     data manifest, history_15m, drive listing, file sha256, paste bundle, colab, lock, R0, input fingerprint
PITFALLS: Everything is written as files in --out (manifest.json lists the sha256 of every output; a listing larger than
          one file is split into history_15m_manifest.partNN.csv files, each under the 300 KB contract limit), so a runner can
          push the folder as is; the stdout bundle is the same content for copy-and-paste. It reads only headers, first and last bars and counts rows (no prices are analysed). An unreadable file is listed
          with its error, never skipped. A run that hits --max-minutes says complete=false and exits 1; tools/build_lock.py
          refuses it. The pasted bundle is verified against manifest.json, so any edit made while copying is detected.

Usage (Colab; see tools/data_manifest_colab_cell.py for the paste-ready cell):
    python tools/data_manifest.py --config cfg.json --out /content/manifest_out --max-minutes 50
cfg.json: {"history_15m_dir": "/content/drive/MyDrive/history_15m", "extra_folders": {"history_1d": "/content/drive/MyDrive/history_1d"}}
"""
import argparse
import csv
import datetime as _dt
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time

SCHEMA = "data_manifest/1"
OUT_CSV = "history_15m_manifest.csv"
OUT_FOLDERS = "folders_manifest.csv"
FILE_LIMIT = 300_000                  # run contract: every output file at most 300 KB
DEFAULT_FILE_BYTES = 250_000          # split target
CSV_COLUMNS = ["symbol", "first_bar", "last_bar", "rows", "bytes", "sha256", "error"]
FOLDER_COLUMNS = ["folder", "symbol"]
CONFIG_KEYS = {"history_15m_dir": None, "extra_folders": {}, "suffix": ".csv.gz"}      # None = required
DATA_EXTS = (".csv.gz", ".csv", ".parquet")
BEGIN = "=== PASTE BUNDLE BEGIN (data_manifest) ==="
END = "=== PASTE BUNDLE END (data_manifest) ==="


def load_config(cfg):
    """A dict or the path of a JSON file. Unknown keys and missing required keys raise ValueError."""
    if not isinstance(cfg, dict):
        with open(cfg, encoding="utf-8") as f:
            cfg = json.load(f)
    unknown = sorted(set(cfg) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"unknown config key(s) {unknown}; allowed: {sorted(CONFIG_KEYS)}")
    out = {}
    for k, default in CONFIG_KEYS.items():
        if k in cfg:
            out[k] = cfg[k]
        elif default is None:
            raise ValueError(f"config key '{k}' is required")
        else:
            out[k] = default
    if not isinstance(out["extra_folders"], dict) or not all(isinstance(v, str) for v in out["extra_folders"].values()):
        raise ValueError("extra_folders must be {folder name: directory path}")
    if not isinstance(out["history_15m_dir"], str) or not out["suffix"].startswith("."):
        raise ValueError("history_15m_dir must be a path string and suffix must start with '.'")
    return out


def _iso(ms):
    return _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_text(s):
    s = s.strip().replace("T", " ").rstrip("Z")
    return _dt.datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path, suffix):
    """One manifest row for <SYMBOL><suffix>. Bar times are bar OPEN times (UTC); the file is read once, line by line."""
    name = os.path.basename(path)
    row = {"symbol": name[: -len(suffix)] if name.endswith(suffix) else name, "first_bar": "", "last_bar": "", "rows": 0,
           "bytes": os.path.getsize(path), "sha256": _sha256(path), "error": ""}
    try:
        n, first, last = 0, None, None
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            cols = f.readline().strip().split(",")
            ti = cols.index("timestamp") if "timestamp" in cols else None
            di = cols.index("datetime_utc") if "datetime_utc" in cols else None
            if ti is None and di is None:
                raise ValueError("no timestamp/datetime_utc column in the header")
            k = (ti if ti is not None else di) + 1
            for line in f:
                if not line.strip():
                    continue
                n += 1
                if first is None:
                    first = line
                last = line
        if n:
            def when(line):
                parts = line.split(",", k)
                return _iso(int(parts[ti])) if ti is not None else _iso_text(parts[di])
            row.update(rows=n, first_bar=when(first), last_bar=when(last))
    except Exception as e:                                       # noqa: BLE001 - recorded, never skipped
        row.update(rows=0, first_bar="", last_bar="", error=re.sub(r"\s+", " ", f"{type(e).__name__}: {e}")[:200])
    return row


def _stem(name):
    for e in DATA_EXTS:
        if name.endswith(e):
            return name[: -len(e)]
    return name


def _git_sha():
    try:
        r = subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except Exception:                                            # noqa: BLE001
        return "unknown"


def _write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([r[c] for c in columns] if isinstance(r, dict) else r)


def output_names(names, base):
    """Output files of one listing: `<stem>.csv` or `<stem>.part01.csv`, `<stem>.part02.csv`... (sorted)."""
    stem = base[: -len(".csv")]
    pat = re.compile(r"^" + re.escape(stem) + r"(\.part\d{2})?\.csv$")
    return sorted(n for n in names if pat.match(n))


def _write_parts(out_dir, base, columns, rows, max_bytes):
    """Write the rows into one file, or into `.partNN.csv` files of at most max_bytes each. Returns the file names."""
    def render(r):
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerow([r[c] for c in columns] if isinstance(r, dict) else r)
        return buf.getvalue()
    head = render(columns)
    chunks, cur, size = [], [], len(head.encode("utf-8"))
    for r in rows:
        line = render(r)
        n = len(line.encode("utf-8"))
        if len(head.encode("utf-8")) + n > max_bytes:
            raise ValueError(f"{base}: one row ({n} bytes) plus the header does not fit the {max_bytes} byte file limit")
        if cur and size + n > max_bytes:
            chunks.append(cur)
            cur, size = [], len(head.encode("utf-8"))
        cur.append(line)
        size += n
    chunks.append(cur)
    stem = base[: -len(".csv")]
    names = [base] if len(chunks) == 1 else [f"{stem}.part{i:02d}.csv" for i in range(1, len(chunks) + 1)]
    for name, lines in zip(names, chunks):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8", newline="") as f:
            f.write(head + "".join(lines))
    return names


def run(cfg, out_dir, max_minutes=50, clock=time.time, max_file_bytes=DEFAULT_FILE_BYTES):
    """List the folders and write history_15m_manifest*.csv, folders_manifest*.csv and manifest.json into out_dir."""
    if max_file_bytes > FILE_LIMIT:
        raise ValueError(f"max_file_bytes {max_file_bytes} is over the {FILE_LIMIT} byte limit of the run contract")
    cfg = load_config(cfg)
    for label, d in [("history_15m_dir", cfg["history_15m_dir"])] + [(f"extra_folders[{k}]", v) for k, v in cfg["extra_folders"].items()]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{label}: {d} is not a directory")
    suffix = cfg["suffix"]
    names = sorted(os.listdir(cfg["history_15m_dir"]))
    files = [n for n in names if n.endswith(suffix)]
    ignored = [n for n in names if not n.endswith(suffix)]
    if not files:
        raise ValueError(f"{cfg['history_15m_dir']}: no *{suffix} files")
    folders = {k: sorted(n for n in os.listdir(v) if n.endswith(DATA_EXTS)) for k, v in cfg["extra_folders"].items()}
    for k, v in folders.items():
        if not v:
            raise ValueError(f"extra folder {k}: no data files ({'/'.join(DATA_EXTS)})")
    fp = hashlib.sha256()
    for n in files:
        fp.update(f"history_15m\t{n}\t{os.path.getsize(os.path.join(cfg['history_15m_dir'], n))}\n".encode())
    for k in sorted(folders):
        for n in folders[k]:
            fp.update(f"{k}\t{n}\t{os.path.getsize(os.path.join(cfg['extra_folders'][k], n))}\n".encode())
    os.makedirs(out_dir, exist_ok=True)
    for n in output_names(os.listdir(out_dir), OUT_CSV) + output_names(os.listdir(out_dir), OUT_FOLDERS):
        os.remove(os.path.join(out_dir, n))                         # outputs of an older run must not be mistaken for this one
    t0 = clock()
    rows, complete = [], True
    for n in files:
        if clock() - t0 >= max_minutes * 60:
            complete = False
            break
        rows.append(file_row(os.path.join(cfg["history_15m_dir"], n), suffix))
    written = _write_parts(out_dir, OUT_CSV, CSV_COLUMNS, rows, max_file_bytes)
    written += _write_parts(out_dir, OUT_FOLDERS, FOLDER_COLUMNS, [[k, _stem(n)] for k in cfg["extra_folders"] for n in folders[k]],
                            max_file_bytes)
    outputs = {n: _sha256(os.path.join(out_dir, n)) for n in written}
    manifest = {"schema": SCHEMA, "git_sha": _git_sha(), "config": cfg, "input_fingerprint": fp.hexdigest(),
                "complete": complete, "max_minutes": max_minutes, "n_files_total": len(files), "n_files_listed": len(rows),
                "ignored_files": ignored, "n_bars_listed": sum(r["rows"] for r in rows),
                "rows": {"history_15m_files": len(rows), "folders": {k: len(v) for k, v in folders.items()}},
                "n_configurations_evaluated": 0, "runtime_seconds": round(clock() - t0, 3), "outputs": outputs}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def bundle_text(out_dir):
    """The paste bundle: manifest.json then every output file verbatim, each under a '### name' line."""
    parts = [BEGIN + "\n"]
    for name in ["manifest.json"] + sorted(json.load(open(os.path.join(out_dir, "manifest.json")))["outputs"]):
        parts.append(f"### {name}\n")
        body = open(os.path.join(out_dir, name), encoding="utf-8", newline="").read()
        parts.append(body if body.endswith("\n") else body + "\n")
    parts.append(END + "\n")
    return "".join(parts)


def parse_bundle(text):
    """Split a pasted bundle; verify every section against manifest.json. Returns {manifest, manifest_text, sections}."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    try:
        b = next(i for i, l in enumerate(lines) if l.strip() == BEGIN)
        e = next(i for i, l in enumerate(lines) if i > b and l.strip() == END)
    except StopIteration:
        raise ValueError("no complete paste bundle (BEGIN/END lines) found") from None
    sections, cur = {}, None
    for l in lines[b + 1:e]:
        if l.startswith("### "):
            cur = l[4:].strip()
            if cur in sections:
                raise ValueError(f"section {cur} appears twice")
            sections[cur] = []
        elif cur is not None:
            sections[cur].append(l)
    bodies = {k: "\n".join(v) + "\n" for k, v in sections.items()}
    if "manifest.json" not in bodies:
        raise ValueError("bundle has no manifest.json section")
    manifest_text = bodies.pop("manifest.json")
    manifest = json.loads(manifest_text)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"unknown bundle schema {manifest.get('schema')!r}")
    if set(bodies) != set(manifest["outputs"]):
        raise ValueError(f"sections {sorted(bodies)} differ from manifest.outputs {sorted(manifest['outputs'])}")
    for name, body in bodies.items():
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != manifest["outputs"][name]:
            raise ValueError(f"section {name} does not match the sha256 in manifest.json (text altered while copying?)")
    return {"manifest": manifest, "manifest_text": manifest_text, "sections": bodies}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="JSON file; unknown keys raise")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--max-minutes", type=float, default=50, help="stop cleanly after this many minutes (default 50)")
    a = ap.parse_args(argv)
    m = run(a.config, a.out, max_minutes=a.max_minutes)
    print(bundle_text(a.out), end="")
    print(f"listed {m['n_files_listed']}/{m['n_files_total']} files in {m['runtime_seconds']} s; complete={m['complete']}", file=sys.stderr)
    return 0 if m["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

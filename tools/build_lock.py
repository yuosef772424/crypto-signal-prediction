"""
PURPOSE:  Builds the sealed-holdout lock of phase R0, deterministically: the pasted Colab manifest of history_15m
          (tools/data_manifest.py bundle) + the touched-data ledger (docs/research/plan/lock/touched.csv) + the salt ->
          docs/research/plan/lock/holdout.json and holdout.sha256. `ingest` stores the pasted bundle verbatim; `build` applies
          the eligibility tests, ranks eligible coins by sha256(salt + symbol), seals floor(2n/5) of them and assigns buckets
          H1..H6 round-robin (rules: docs/research/plan/lock/README.md).
TAGS:     lock, sealed holdout, salt, eligibility, touched ledger, buckets, deterministic build, manifest ingest, R0
PITFALLS: The salt must be the commit that first added lock/README.md (checked with git unless --no-repo-check, which is for
          fixtures only). A pull request carrying that commit must be merge-committed, never squashed. A folder rule whose folder
          is missing from the bundle, an incomplete manifest, an edited CSV or a duplicate symbol makes the build fail. The output
          has no timestamps, so the same inputs give the same bytes; `write_lock` never overwrites an existing holdout.json.

Usage:
    python tools/build_lock.py ingest --bundle pasted.txt --dir docs/research/plan/lock/manifest
    python tools/build_lock.py build --salt <40-hex> --manifest-dir docs/research/plan/lock/manifest \\
        --touched docs/research/plan/lock/touched.csv --out docs/research/plan/lock
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter

import data_manifest as dm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(ROOT, "docs", "research", "plan", "lock")
SPEC_PATH = "docs/research/plan/lock/README.md"
TIME_SEAL = "2026-10-01T00:00:00Z"
HISTORY_END = dt.datetime(2026, 9, 30, 23, 45, tzinfo=dt.timezone.utc)      # last 15m bar before the time seal
FRESH_LIMIT = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
MIN_HISTORY = dt.timedelta(days=730)
BUCKETS = ("H1", "H2", "H3", "H4", "H5", "H6")
BUCKET_PHASE = {"H1": "R1b", "H2": "R3", "H3": "R2", "H4": "R5", "H5": "G2", "H6": "reserve (R6)"}
STABLE_BASES = frozenset("USDC USDP TUSD FDUSD BUSD DAI USDD PYUSD USDE USD1 RLUSD EURI EUR AEUR USDS BFUSD XUSD UST USTC SUSD GUSD LUSD".split())
INDEX_BASES = frozenset({"BTCDOM", "DEFI"})
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
REASON_ORDER = ("unreadable", "short_history", "stale", "stablecoin", "leveraged", "index_contract", "touched")
TOUCHED_COLUMNS = ["symbol_or_rule", "kind", "timeframe", "period", "source", "note"]
KINDS = ("symbol", "rule", "time_seal_read")
_SALT = re.compile(r"^[0-9a-f]{40}$")
_RULE_FOLDER = re.compile(r"^folder:[A-Za-z0-9_]+$")
_RULE_BEFORE = re.compile(r"^first_bar_before:(\d{4}-\d{2}-\d{2})$")
WARN_BUCKET_MIN, WARN_SEALED_MIN = 5, 30


def n_sealed(n):
    """40 % of n, rounded down, in integer arithmetic."""
    return (2 * n) // 5


def rank_hash(salt, symbol):
    return hashlib.sha256((salt + symbol).encode("utf-8")).hexdigest()


def _sha256_file(path):
    return dm._sha256(path)


def _when(s):
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def _base(symbol):
    return symbol[:-4] if symbol.endswith("USDT") else symbol


# --------------------------------------------------------------------------- inputs
def load_manifest(mdir):
    """Verified manifest folder -> (manifest dict, rows by symbol, folders {name: set(symbol)}, input hashes)."""
    mp = os.path.join(mdir, "manifest.json")
    manifest = json.load(open(mp, encoding="utf-8"))
    if manifest.get("schema") != dm.SCHEMA:
        raise ValueError(f"{mp}: unknown schema {manifest.get('schema')!r}")
    if manifest.get("complete") is not True:
        raise ValueError(f"{mp}: the manifest is incomplete (complete != true); rerun data_manifest.py with more minutes")
    if set(manifest["outputs"]) != {dm.OUT_CSV, dm.OUT_FOLDERS}:
        raise ValueError(f"{mp}: unexpected outputs {sorted(manifest['outputs'])}")
    for name, digest in manifest["outputs"].items():
        if _sha256_file(os.path.join(mdir, name)) != digest:
            raise ValueError(f"{os.path.join(mdir, name)}: sha256 differs from manifest.json (file edited?)")
    rows = {}
    with open(os.path.join(mdir, dm.OUT_CSV), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["symbol"] in rows:
                raise ValueError(f"duplicate symbol {r['symbol']} in {dm.OUT_CSV}")
            rows[r["symbol"]] = r
    if not rows:
        raise ValueError(f"{dm.OUT_CSV} lists no symbol")
    folders = {}
    with open(os.path.join(mdir, dm.OUT_FOLDERS), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            folders.setdefault(r["folder"], set()).add(r["symbol"].strip().upper())
    hashes = {"manifest_json_sha256": _sha256_file(mp), "history_15m_manifest_sha256": manifest["outputs"][dm.OUT_CSV],
              "folders_manifest_sha256": manifest["outputs"][dm.OUT_FOLDERS],
              "manifest_git_sha": manifest.get("git_sha"), "manifest_input_fingerprint": manifest.get("input_fingerprint")}
    return manifest, rows, folders, hashes


def parse_touched(path):
    """touched.csv -> list of validated row dicts (symbols upper-cased). Any malformed row raises ValueError."""
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header != TOUCHED_COLUMNS:
            raise ValueError(f"{path}: header must be {TOUCHED_COLUMNS}, got {header}")
        out = []
        for i, raw in enumerate(rd, start=2):
            if len(raw) != len(TOUCHED_COLUMNS):
                raise ValueError(f"{path}:{i}: expected {len(TOUCHED_COLUMNS)} columns, got {len(raw)}")
            r = dict(zip(TOUCHED_COLUMNS, raw))
            if r["kind"] not in KINDS:
                raise ValueError(f"{path}:{i}: unknown kind {r['kind']!r}")
            if not r["source"].strip():
                raise ValueError(f"{path}:{i}: every row needs a source")
            key = r["symbol_or_rule"].strip()
            if not key:
                raise ValueError(f"{path}:{i}: empty symbol_or_rule")
            if r["kind"] == "rule":
                if not (_RULE_FOLDER.match(key) or _RULE_BEFORE.match(key)):
                    raise ValueError(f"{path}:{i}: rule must be folder:<name> or first_bar_before:<YYYY-MM-DD>, got {key!r}")
                r["symbol_or_rule"] = key
            else:
                if ":" in key:
                    raise ValueError(f"{path}:{i}: a {r['kind']} row names a symbol, got {key!r}")
                r["symbol_or_rule"] = key.upper()
            out.append(r)
    return out


def resolve_touched(touched, rows, folders):
    """{manifest symbol: sorted distinct source labels} from symbol rows and rules (strict: whole universe)."""
    hit = {}
    for r in touched:
        if r["kind"] == "symbol":
            if r["symbol_or_rule"] in rows:
                hit.setdefault(r["symbol_or_rule"], set()).add(r["source"])
        elif r["kind"] == "rule":
            m = _RULE_BEFORE.match(r["symbol_or_rule"])
            if m:
                for s, row in rows.items():
                    if row["first_bar"] and row["first_bar"][:10] < m.group(1):
                        hit.setdefault(s, set()).add(r["source"])
            else:
                name = r["symbol_or_rule"].split(":", 1)[1]
                if not folders.get(name):
                    raise ValueError(f"rule {r['symbol_or_rule']}: folder {name!r} is missing or empty in the manifest bundle "
                                     f"(a rule is never skipped; add the folder to data_manifest extra_folders)")
                for s in folders[name]:
                    if s in rows:
                        hit.setdefault(s, set()).add(r["source"])
    return {s: sorted(v) for s, v in hit.items()}


# --------------------------------------------------------------------------- the rules
def reasons_for(symbol, row, bases, touched_sources):
    why = []
    readable = not row["error"] and row["first_bar"] and row["last_bar"] and int(row["rows"] or 0) > 0
    if not readable:
        why.append("unreadable")
    else:
        first, last = _when(row["first_bar"]), _when(row["last_bar"])
        if min(last, HISTORY_END) - first < MIN_HISTORY:
            why.append("short_history")
        if last < FRESH_LIMIT:
            why.append("stale")
    base = _base(symbol)
    if base in STABLE_BASES:
        why.append("stablecoin")
    if any(base.endswith(x) and base[: -len(x)] in bases for x in LEVERAGED_SUFFIXES):
        why.append("leveraged")
    if base in INDEX_BASES:
        why.append("index_contract")
    if symbol in touched_sources:
        why.append("touched")
    return why


def verify_salt(salt, repo_root=ROOT):
    """The salt must be the commit that first added lock/README.md."""
    out = subprocess.run(["git", "-C", repo_root, "log", "--diff-filter=A", "--format=%H", "--", SPEC_PATH],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise ValueError(f"git log failed in {repo_root}: {out.stderr.strip()}")
    commits = out.stdout.split()
    if not commits or commits[-1] != salt:
        raise ValueError(f"salt {salt} is not the commit that first adds {SPEC_PATH} (git says {commits[-1] if commits else 'none'})")


def build(manifest_dir, touched_path, salt, *, repo_check=True, repo_root=ROOT):
    """The holdout.json dictionary (see lock/README.md). Deterministic in its inputs."""
    if not _SALT.match(salt or ""):
        raise ValueError("salt must be 40 lowercase hexadecimal characters (the SHA of the spec commit)")
    if repo_check:
        verify_salt(salt, repo_root)
    manifest, rows, folders, hashes = load_manifest(manifest_dir)
    touched = parse_touched(touched_path)
    hashes["touched_csv_sha256"] = _sha256_file(touched_path)
    hashes["n_touched_rows"] = len(touched)
    bases = {_base(s) for s in rows}
    tsrc = resolve_touched(touched, rows, folders)
    ineligible = {}
    for s in sorted(rows):
        why = reasons_for(s, rows[s], bases, tsrc)
        if why:
            ineligible[s] = why
    eligible = sorted(set(rows) - set(ineligible))
    ranked = sorted(eligible, key=lambda s: (rank_hash(salt, s), s))
    k = n_sealed(len(eligible))
    sealed_order = ranked[:k]
    sealed = {s: f"H{i % 6 + 1}" for i, s in enumerate(sealed_order)}
    buckets = {b: sorted(s for s, bb in sealed.items() if bb == b) for b in BUCKETS}
    failed = Counter(w for ws in ineligible.values() for w in ws)
    warnings = [f"bucket {b} has {len(v)} sealed symbols (< {WARN_BUCKET_MIN}); {BUCKET_PHASE[b]} will confirm on very few coins"
                for b, v in buckets.items() if len(v) < WARN_BUCKET_MIN]
    if k < WARN_SEALED_MIN:
        warnings.append(f"total sealed symbols {k} < {WARN_SEALED_MIN}: the R5 reopen condition (30 sealed coins) cannot be met")
    return {
        "version": 1, "salt": salt, "time_seal": TIME_SEAL, "spec": SPEC_PATH, "inputs": hashes,
        "funnel": {"manifest_symbols": len(rows), "ineligible": len(ineligible), "eligible": len(eligible),
                   "sealed": k, "open_pool": len(eligible) - k,
                   "failed_test": {r: failed.get(r, 0) for r in REASON_ORDER},
                   "bucket_sizes": {b: len(v) for b, v in buckets.items()}},
        "warnings": warnings, "bucket_phase": dict(BUCKET_PHASE),
        "ineligible": ineligible, "touched_sources": tsrc,
        "eligible": eligible, "open_pool": sorted(set(eligible) - set(sealed)),
        "sealed_order": sealed_order, "sealed": sealed, "buckets": buckets,
        "rank_sha256": {s: rank_hash(salt, s) for s in eligible},
        "time_seal_reads": [{c: r[c] for c in TOUCHED_COLUMNS} for r in touched if r["kind"] == "time_seal_read"],
    }


def dumps(d):
    return json.dumps(d, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write_lock(d, out_dir):
    """holdout.json + holdout.sha256 (sha256sum format). Never overwrites."""
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "holdout.json")
    if os.path.exists(jp) or os.path.exists(os.path.join(out_dir, "holdout.sha256")):
        raise FileExistsError(f"{jp}: the lock exists and never changes")
    body = dumps(d).encode("utf-8")
    with open(jp, "wb") as f:
        f.write(body)
    with open(os.path.join(out_dir, "holdout.sha256"), "w", encoding="utf-8", newline="") as f:
        f.write(hashlib.sha256(body).hexdigest() + "  holdout.json\n")


def ingest(text, dest_dir):
    """Store the pasted bundle verbatim (after verifying every section against its manifest.json)."""
    parsed = dm.parse_bundle(text)
    if os.path.exists(os.path.join(dest_dir, "manifest.json")):
        raise FileExistsError(f"{dest_dir}: a manifest is already stored; it is never replaced")
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, "manifest.json"), "w", encoding="utf-8", newline="") as f:
        f.write(parsed["manifest_text"])
    for name, body in parsed["sections"].items():
        with open(os.path.join(dest_dir, name), "w", encoding="utf-8", newline="") as f:
            f.write(body)
    return parsed["manifest"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_in = sub.add_parser("ingest", help="store a pasted data_manifest bundle verbatim")
    a_in.add_argument("--bundle", required=True)
    a_in.add_argument("--dir", default=os.path.join(LOCK_DIR, "manifest"))
    a_b = sub.add_parser("build", help="write holdout.json and holdout.sha256")
    a_b.add_argument("--salt", required=True)
    a_b.add_argument("--manifest-dir", default=os.path.join(LOCK_DIR, "manifest"))
    a_b.add_argument("--touched", default=os.path.join(LOCK_DIR, "touched.csv"))
    a_b.add_argument("--out", default=LOCK_DIR)
    a_b.add_argument("--no-repo-check", action="store_true", help="skip the git check of the salt (fixtures only)")
    a_b.add_argument("--dry-run", action="store_true", help="print the funnel, write nothing")
    a = ap.parse_args(argv)
    if a.cmd == "ingest":
        m = ingest(open(a.bundle, encoding="utf-8").read(), a.dir)
        print(f"stored {a.dir}: {m['n_files_listed']} files listed, complete={m['complete']}")
        return 0
    d = build(a.manifest_dir, a.touched, a.salt, repo_check=not a.no_repo_check)
    f = d["funnel"]
    print("funnel:", json.dumps(f, sort_keys=True))
    for w in d["warnings"]:
        print("WARNING:", w, file=sys.stderr)
    if not a.dry_run:
        write_lock(d, a.out)
        print(f"wrote {os.path.join(a.out, 'holdout.json')} and holdout.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())

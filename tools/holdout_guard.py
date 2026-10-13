"""
PURPOSE:  The only door to market data in a research phase after R0. Refuses (raises) a request for a sealed symbol, or for
          any bar with open time >= 2026-10-01 (the time seal, every coin), and opens each sealed bucket H1..H6 exactly once,
          through `open_holdout(card_path, bucket)`, which appends a row to holdout_access_log.csv. The sealed set comes from
          docs/research/plan/lock/holdout.json (built by tools/build_lock.py, spec in lock/README.md).
TAGS:     holdout, sealed coins, time seal, lock, data loader guard, confirmation, access log, evaluation integrity, R0
PITFALLS: Fails closed: with no holdout.json (or a holdout.sha256 that does not match it) every loader raises LockError.
          Requests are half-open [start, end) over bar OPEN times and `end` is required (no silent default). A symbol is
          matched after upper-casing, with and without a trailing USDT. An opened bucket still never returns a bar at or
          after the seal, and a refused card does not burn the bucket (only a successful open writes a log row). The plain
          loader keeps refusing sealed symbols after their bucket was opened: confirmation uses the handle.

Usage:
    import holdout_guard as hg
    hg.assert_unsealed(["BTCUSDT", "ETHUSDT"], "2024-01-01", "2026-10-01")
    df = hg.load_unsealed("/content/drive/MyDrive/history_15m", "BTCUSDT", "2024-01-01", "2026-10-01")
    h = hg.open_holdout("research/<study>/cards/E-xxx.md", "H1")   # card front matter: confirmation: true, frozen_rules_sha256: <64 hex>
    df = h.load(data_dir, h.symbols[0], "2024-01-01", "2026-10-01")
"""
import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field

import pandas as pd

try:                                                            # POSIX only; the lock is advisory elsewhere
    import fcntl
except ImportError:                                             # pragma: no cover
    fcntl = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(ROOT, "docs", "research", "plan", "lock")
TIME_SEAL = pd.Timestamp("2026-10-01", tz="UTC")
BUCKETS = ("H1", "H2", "H3", "H4", "H5", "H6")
LOG_COLUMNS = ["opened_at_utc", "bucket", "card_path", "card_sha256", "frozen_rules_sha256", "git_head"]
DEFAULT_PATTERN = "{symbol}.csv.gz"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class GuardError(Exception):
    """Base class of every refusal."""


class LockError(GuardError):
    """holdout.json is missing, edited (sha256 mismatch) or inconsistent: nothing may be read."""


class SealedAccessError(GuardError):
    """A sealed symbol, or a bar at or after the time seal, was requested."""


class HoldoutOpenError(GuardError):
    """A bucket cannot be opened: bad card, unknown/empty bucket, or already opened."""


@dataclass(frozen=True)
class Lock:
    salt: str
    time_seal: pd.Timestamp
    sealed: dict = field(repr=False)          # symbol -> bucket
    buckets: dict = field(repr=False)         # bucket -> tuple of symbols (sorted)
    lock_dir: str = LOCK_DIR


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lock(lock_dir=None):
    """Read and verify holdout.json (sha256 must match holdout.sha256, time seal must be 2026-10-01, buckets consistent)."""
    lock_dir = str(lock_dir or LOCK_DIR)
    jp, sp = os.path.join(lock_dir, "holdout.json"), os.path.join(lock_dir, "holdout.sha256")
    if not (os.path.isfile(jp) and os.path.isfile(sp)):
        raise LockError(f"{lock_dir}: holdout.json / holdout.sha256 not found - the lock is not built yet; refusing to read data")
    want = open(sp, encoding="utf-8").read().split()
    if not want or want[0] != _sha256_file(jp):
        raise LockError(f"{jp}: sha256 does not match holdout.sha256 - the lock file was changed")
    d = json.load(open(jp, encoding="utf-8"))
    for k in ("salt", "time_seal", "sealed", "buckets"):
        if k not in d:
            raise LockError(f"{jp}: key '{k}' is missing")
    seal = pd.Timestamp(d["time_seal"])
    seal = seal.tz_localize("UTC") if seal.tzinfo is None else seal.tz_convert("UTC")
    if seal != TIME_SEAL:
        raise LockError(f"{jp}: time_seal {d['time_seal']} is not {TIME_SEAL.isoformat()}")
    sealed = {str(s): str(b) for s, b in d["sealed"].items()}
    buckets = {b: tuple(sorted(d["buckets"].get(b, []))) for b in BUCKETS}
    from_map = {b: tuple(sorted(s for s, bb in sealed.items() if bb == b)) for b in BUCKETS}
    if set(d["buckets"]) - set(BUCKETS) or buckets != from_map:
        raise LockError(f"{jp}: 'buckets' and 'sealed' disagree")
    return Lock(salt=d["salt"], time_seal=seal, sealed=sealed, buckets=buckets, lock_dir=lock_dir)


def _lock(lock):
    return lock if isinstance(lock, Lock) else load_lock(lock)


def _ts(x, name):
    if x is None:
        raise TypeError(f"{name} is required (no silent default)")
    t = pd.Timestamp(x)
    if pd.isna(t):
        raise ValueError(f"{name}: not a timestamp: {x!r}")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _norm(symbol):
    return str(symbol).strip().upper()


def _sealed_hits(symbols, lock):
    hits = []
    for s in symbols:
        n = _norm(s)
        if n in lock.sealed:
            hits.append(n)
        elif not n.endswith("USDT") and n + "USDT" in lock.sealed:
            hits.append(n + "USDT")
    return hits


def _check_window(start, end, lock):
    start, end = _ts(start, "start"), _ts(end, "end")
    if start >= end:
        raise ValueError(f"empty window: start {start} >= end {end}")
    if end > lock.time_seal:
        raise SealedAccessError(f"window [{start}, {end}) reaches bars at or after the time seal {lock.time_seal.isoformat()}")
    return start, end


def assert_unsealed(symbols, start, end, *, lock=None):
    """Raise SealedAccessError if any symbol is sealed or [start, end) reaches a bar with open time >= the time seal."""
    lock = _lock(lock)
    syms = [symbols] if isinstance(symbols, str) else list(symbols)
    if not syms:
        raise ValueError("no symbols given")
    _check_window(start, end, lock)
    hits = _sealed_hits(syms, lock)
    if hits:
        raise SealedAccessError(f"sealed symbol(s) requested: {sorted(set(hits))}")


def _read_bars(path):
    """Bars indexed by UTC bar OPEN time. CSV(.gz): `timestamp` (ms) or `datetime_utc`/`open_time`; parquet: DatetimeIndex."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{path}: parquet needs a DatetimeIndex")
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    else:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
        else:
            col = next((c for c in ("datetime_utc", "open_time") if c in df.columns), None)
            if col is None:
                raise ValueError(f"{path}: no timestamp/datetime_utc/open_time column")
            idx = pd.DatetimeIndex(pd.to_datetime(df[col], utc=True))
    out = df.set_axis(idx.rename("open_time")).sort_index()
    return out


def _load(data_dir, symbol, start, end, pattern, lock):
    start, end = _check_window(start, end, lock)
    path = os.path.join(str(data_dir), pattern.format(symbol=str(symbol).strip()))
    df = _read_bars(path)
    df = df[(df.index >= start) & (df.index < end)]
    if len(df) and df.index.max() >= lock.time_seal:             # defence in depth: cannot happen after _check_window
        raise SealedAccessError("a bar at or after the time seal was about to be returned")
    return df


def load_unsealed(data_dir, symbol, start, end, *, pattern=DEFAULT_PATTERN, lock=None):
    """Bars of one unsealed symbol with open time in [start, end), end <= the time seal. Sealed symbols raise first."""
    lock = _lock(lock)
    assert_unsealed([symbol], start, end, lock=lock)
    return _load(data_dir, symbol, start, end, pattern, lock)


def _parse_card(path):
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as e:
        raise HoldoutOpenError(f"{path}: cannot read the card ({e})")
    m = re.match(r"\s*---\n(.*?)\n---", text, re.S)
    if not m:
        raise HoldoutOpenError(f"{path}: no front matter (--- ... ---) at the top of the card")
    meta = {}
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].rstrip()
        if line.strip() and ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("'\"")
    return meta, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_head():
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:                                           # noqa: BLE001
        return "unknown"


@dataclass(frozen=True)
class HoldoutHandle:
    """Result of a successful open_holdout: the bucket's symbols and a loader restricted to them."""
    bucket: str
    symbols: tuple
    card_sha256: str
    frozen_rules_sha256: str
    _lock: Lock = field(repr=False, default=None)

    def load(self, data_dir, symbol, start, end, *, pattern=DEFAULT_PATTERN):
        if _norm(symbol) not in self.symbols:
            raise SealedAccessError(f"{symbol} is not in bucket {self.bucket}")
        return _load(data_dir, symbol, start, end, pattern, self._lock)


def open_holdout(card_path, bucket, *, lock=None):
    """Open sealed bucket `bucket` once for the confirmation step described by the card; log it; return a HoldoutHandle.

    Refuses: unknown or empty bucket; a card without front matter, without `confirmation: true`, without a 64-hex
    `frozen_rules_sha256` (or whose `frozen_rules_file`, if named, hashes differently), or naming another
    `holdout_bucket`; and a bucket that already has a row in holdout_access_log.csv (any earlier open, any card)."""
    lock = _lock(lock)
    if bucket not in BUCKETS:
        raise HoldoutOpenError(f"unknown bucket {bucket!r}; expected one of {BUCKETS}")
    symbols = lock.buckets[bucket]
    if not symbols:
        raise HoldoutOpenError(f"bucket {bucket} holds no sealed symbol")
    meta, card_sha = _parse_card(str(card_path))
    if meta.get("confirmation", "").lower() != "true":
        raise HoldoutOpenError(f"{card_path}: front matter must say 'confirmation: true'")
    rules = meta.get("frozen_rules_sha256", "")
    if not _HEX64.match(rules):
        raise HoldoutOpenError(f"{card_path}: front matter needs 'frozen_rules_sha256: <64 lowercase hex>' (the frozen rules)")
    if meta.get("holdout_bucket") not in (None, bucket):
        raise HoldoutOpenError(f"{card_path}: card names bucket {meta['holdout_bucket']}, asked for {bucket}")
    if "frozen_rules_file" in meta:
        rf = meta["frozen_rules_file"]
        rf = rf if os.path.isabs(rf) else os.path.join(ROOT, rf)
        if not os.path.isfile(rf) or _sha256_file(rf) != rules:
            raise HoldoutOpenError(f"{card_path}: frozen_rules_file {meta['frozen_rules_file']} is missing or does not hash to frozen_rules_sha256")
    log = os.path.join(lock.lock_dir, "holdout_access_log.csv")
    if not os.path.isfile(log):
        raise LockError(f"{log}: access log not found")
    shown = os.path.relpath(str(card_path), ROOT).replace(os.sep, "/") if str(card_path).startswith(ROOT) else str(card_path)
    with open(log, "a+", encoding="utf-8", newline="") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            rows = list(csv.DictReader(f))
            f.seek(0)
            if f.readline().rstrip("\r\n") != ",".join(LOG_COLUMNS):
                raise LockError(f"{log}: header is not {LOG_COLUMNS}")
            if any(r["bucket"] == bucket for r in rows):
                raise HoldoutOpenError(f"bucket {bucket} was already opened (see holdout_access_log.csv); it opens once")
            f.seek(0, 2)
            csv.writer(f, lineterminator="\n").writerow(
                [pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"), bucket, shown, card_sha, rules, _git_head()])
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_UN)
    return HoldoutHandle(bucket=bucket, symbols=symbols, card_sha256=card_sha, frozen_rules_sha256=rules, _lock=lock)

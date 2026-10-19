"""
PURPOSE:  Audit (and mask) stale bars of per-coin OHLCV frames: the last real trade, the flat/zero-volume tail of a delisted or dead
          coin, runs of identical OHLC with zero volume, and the share of target rows (close(t+h)/close(t)-1) that come from such
          bars, per train/val/test period. Also the single definition behind CONFIG['stale_bars'] (data/windows.py loads
          `stale_bar_mask` from here).
TAGS:     stale bars, delisted coins, dead coins, forward-fill, zero volume, flat bars, exact-zero targets, close_reg, zero share,
          last real trade, stale_bars, data quality audit, survivorship
PITFALLS: "Real trade" = volume > 0 AND a price change (high > low, or close != previous close). A bar with volume > 0 and no price
          change is NOT stale (genuine zero return of an illiquid live coin); it only counts as stale if it lies in the dead tail.
          A tail shorter than min_run is NOT dropped (a live coin's last few flat bars). Dropping uses the whole coin history (last
          real trade), so it is a dataset-cleaning step, never a feature. Runs are counted on consecutive ROWS of the frame given
          (resample first: the base-timeframe frame is what the pipeline sees). Target rows here are positional (row i -> i+h);
          the pipeline additionally requires gap-free windows, so its counts are a subset. min_run must come from the data:
          see derive_min_run, or set it in time (min_run_hours -> min_run_from_hours: the same wall-clock span at every
          timeframe; derive_min_run's quantile is dominated by a few relisted coins with multi-year padded gaps, see research/stale_bars).

Usage (library):  mask = stale_bar_mask(df, min_run); coins, splits = audit_frames(frames, min_run=6, horizon=1, splits=...)
Usage (CLI):      python tools/stale_bars_audit.py --data-dir <dir of <SYMBOL>.csv[.gz]> --timeframe 4h --out <dir> [--min-run N]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

OHLC = ("open", "high", "low", "close")
OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def check_min_run(min_run) -> int:
    """min_run: an int >= 2 (not bool). Raises ValueError otherwise (no silent defaults)."""
    if isinstance(min_run, bool) or not isinstance(min_run, (int, np.integer)) or int(min_run) < 2:
        raise ValueError(f"min_run must be an integer >= 2, got {min_run!r}")
    return int(min_run)


def check_min_run_hours(hours) -> float:
    """hours: a finite number > 0 (not bool). Raises ValueError otherwise (no silent defaults)."""
    if isinstance(hours, bool) or not isinstance(hours, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(hours)) or float(hours) <= 0:
        raise ValueError(f"min_run_hours must be a finite number > 0, got {hours!r}")
    return float(hours)


def bar_hours(timeframe) -> float:
    """Hours in one bar of a fixed-length pandas offset string ('15m', '1h', '4h', '1D')."""
    try:
        h = pd.Timedelta(str(timeframe)).total_seconds() / 3600.0
    except (ValueError, TypeError) as e:
        raise ValueError(f"timeframe {timeframe!r} is not a fixed-length offset") from e
    if not h > 0:
        raise ValueError(f"timeframe {timeframe!r} has no positive length")
    return h


def min_run_from_hours(hours, timeframe) -> int:
    """Time-based threshold -> bars of `timeframe`: ceil(hours / bar_hours), never below 2. The same H means the same
    wall-clock span at every timeframe (168 h -> 168 bars of 1h, 42 of 4h, 7 of 1D)."""
    hours = check_min_run_hours(hours)
    return max(2, int(math.ceil(hours / bar_hours(timeframe) - 1e-9)))


STALE_BARS_KEYS = ("min_run", "min_run_hours")


def resolve_min_run(cfg, timeframe=None) -> int:
    """CONFIG['stale_bars'] dict -> min_run in bars. Exactly one of {'min_run': int >= 2} / {'min_run_hours': number > 0};
    unknown keys, both keys, or an empty dict raise ValueError. 'min_run_hours' needs `timeframe` (the base timeframe)."""
    if not isinstance(cfg, dict) or len(cfg) != 1 or not set(cfg) <= set(STALE_BARS_KEYS):
        raise ValueError(f"stale_bars: exactly one of {STALE_BARS_KEYS} (min_run int >= 2 | min_run_hours number > 0), got {cfg!r}")
    if "min_run" in cfg:
        return check_min_run(cfg["min_run"])
    if timeframe is None:
        raise ValueError("stale_bars: 'min_run_hours' needs the base timeframe to convert hours to bars")
    return min_run_from_hours(cfg["min_run_hours"], timeframe)


def _cols(df: pd.DataFrame):
    return [df[c].to_numpy(dtype="float64") for c in (*OHLC, "volume")]


def flat_zero_volume(df: pd.DataFrame) -> np.ndarray:
    """Bars with volume == 0 and open == high == low == close (NaN -> False)."""
    o, h, l, c, v = _cols(df)
    with np.errstate(invalid="ignore"):
        return (v == 0) & (o == h) & (h == l) & (l == c)


def real_trade_mask(df: pd.DataFrame) -> np.ndarray:
    """Bars with volume > 0 and a price change (high > low, or close != previous close)."""
    o, h, l, c, v = _cols(df)
    prev = np.r_[np.nan, c[:-1]]
    with np.errstate(invalid="ignore"):
        moved = (h > l) | ((c != prev) & ~np.isnan(prev))
        return (v > 0) & moved


def _runs(mask: np.ndarray):
    """(start, length) of maximal True runs."""
    if not mask.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return starts, ends - starts


def tail_start(df: pd.DataFrame) -> int:
    """Row position right after the last real trade (0 if there is none; len(df) if the last bar is real)."""
    real = np.where(real_trade_mask(df))[0]
    return int(real[-1]) + 1 if len(real) else 0


def stale_bar_mask(df: pd.DataFrame, min_run: int) -> np.ndarray:
    """True for stale bars: the tail after the last real trade (if it is >= min_run bars) and every run of >= min_run
    consecutive flat zero-volume bars."""
    min_run = check_min_run(min_run)
    n = len(df)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    ts = tail_start(df)
    if n - ts >= min_run:
        out[ts:] = True
    starts, lens = _runs(flat_zero_volume(df))
    for s, ln in zip(starts, lens):
        if ln >= min_run:
            out[s:s + ln] = True
    return out


def stale_row_mask(stale: np.ndarray, horizon: int) -> np.ndarray:
    """Entry rows i (0 .. n-horizon-1) whose entry bar or any of the next `horizon` bars is stale."""
    n = len(stale)
    if n <= horizon:
        return np.zeros(0, dtype=bool)
    cs = np.r_[0, np.cumsum(stale.astype(np.int64))]
    i = np.arange(n - horizon)
    return (cs[i + horizon + 1] - cs[i]) > 0


def derive_min_run(frames: Mapping[str, pd.DataFrame], quantile: float = 0.99, floor: int = 2) -> int:
    """Data-derived threshold: one more than the `quantile` of the lengths of flat zero-volume runs found OUTSIDE dead tails
    (genuine pauses of live coins), never below `floor`. No such runs -> floor."""
    floor = check_min_run(floor)
    lens_all = []
    for df in frames.values():
        ts = tail_start(df)
        starts, lens = _runs(flat_zero_volume(df))
        lens_all.extend(int(ln) for s, ln in zip(starts, lens) if s < ts)
    if not lens_all:
        return floor
    return max(floor, int(math.ceil(float(np.quantile(lens_all, quantile)))) + 1)


def audit_coin(df: pd.DataFrame, name: str, *, min_run: int, horizon: int,
               splits: Mapping[str, Tuple[pd.Timestamp, pd.Timestamp]]):
    """-> (coin_row: dict, [split_row: dict]). `splits`: name -> (start, end) on the ENTRY timestamp, start inclusive, end exclusive."""
    min_run = check_min_run(min_run)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be an int >= 1, got {horizon!r}")
    df = df.sort_index()
    n = len(df)
    stale = stale_bar_mask(df, min_run)
    flat = flat_zero_volume(df)
    ts_tail = tail_start(df)
    starts, lens = _runs(flat)
    real = np.where(real_trade_mask(df))[0]
    coin = {
        "coin": name, "n_bars": n,
        "first_ts": df.index[0] if n else pd.NaT, "last_ts": df.index[-1] if n else pd.NaT,
        "last_real_ts": df.index[real[-1]] if len(real) else pd.NaT,
        "tail_bars": n - ts_tail, "dead": bool(n - ts_tail >= min_run),
        "flat_zero_volume_bars": int(flat.sum()), "n_flat_runs": int((lens >= min_run).sum()),
        "longest_flat_run": int(lens.max()) if len(lens) else 0,
        "stale_bars": int(stale.sum()), "stale_share": float(stale.mean()) if n else 0.0,
    }
    rows = []
    if n > horizon:
        c = df["close"].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore"):
            zero = c[horizon:] == c[:-horizon]
        srow = stale_row_mask(stale, horizon)
        ts = df.index[:n - horizon]
        for sname, (a, b) in splits.items():
            sel = (ts >= a) & (ts < b)
            k = int(sel.sum())
            z, s = zero[sel], srow[sel]
            clean = int((~s).sum())
            rows.append({
                "coin": name, "split": sname, "n_rows": k, "zero_target_rows": int(z.sum()),
                "zero_target_share": float(z.mean()) if k else float("nan"),
                "stale_rows": int(s.sum()), "stale_share": float(s.mean()) if k else float("nan"),
                "zero_rows_from_stale": int((z & s).sum()), "clean_rows": clean,
                "zero_share_clean": float((z & ~s).sum() / clean) if clean else float("nan")})
    return coin, rows


def audit_frames(frames: Mapping[str, pd.DataFrame], *, min_run: int, horizon: int,
                 splits: Mapping[str, Tuple[pd.Timestamp, pd.Timestamp]]):
    """-> (coins DataFrame, per-coin-per-split DataFrame)."""
    coins, srows = [], []
    for name, df in frames.items():
        c, r = audit_coin(df, name, min_run=min_run, horizon=horizon, splits=splits)
        coins.append(c)
        srows.extend(r)
    return pd.DataFrame(coins), pd.DataFrame(srows)


def pool_splits(splits_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled over coins, per split: zero share with the option OFF (all rows) and ON (stale rows removed)."""
    g = splits_df.groupby("split", sort=False)[["n_rows", "zero_target_rows", "stale_rows", "zero_rows_from_stale",
                                                 "clean_rows"]].sum()
    out = pd.DataFrame({
        "n_rows_off": g["n_rows"], "zero_share_off": g["zero_target_rows"] / g["n_rows"],
        "n_rows_on": g["clean_rows"], "zero_share_on": (g["zero_target_rows"] - g["zero_rows_from_stale"]) / g["clean_rows"],
        "stale_row_share": g["stale_rows"] / g["n_rows"],
        "zero_rows_from_stale_share": g["zero_rows_from_stale"] / g["zero_target_rows"].where(g["zero_target_rows"] > 0)})
    return out.reset_index()


def split_bounds(frames: Mapping[str, pd.DataFrame], horizon: int, train_pct: float = 0.8, val_pct: float = 0.1):
    """Pooled-timestamp quantile cuts (an approximation of the pipeline's global_time split, without the embargo gaps)."""
    ts = np.sort(np.concatenate([f.sort_index().index[:max(len(f) - horizon, 0)].as_unit("ns").asi8 for f in frames.values()]))
    t1 = pd.Timestamp(ts[int(len(ts) * train_pct)], tz="UTC")
    t2 = pd.Timestamp(ts[int(len(ts) * (train_pct + val_pct))], tz="UTC")
    lo, hi = pd.Timestamp(ts[0], tz="UTC"), pd.Timestamp(ts[-1], tz="UTC") + pd.Timedelta(1, "ns")
    return {"train": (lo, t1), "val": (t1, t2), "test": (t2, hi)}


def load_ohlcv(path: str, timeframe: Optional[str] = None) -> pd.DataFrame:
    """Read one history CSV(.gz) (timestamp ms / datetime_utc + open..volume) and resample to `timeframe` (None = as is)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    if "datetime_utc" in df.columns:
        idx = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.set_index(idx)[[*OHLC, "volume"]]
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if timeframe:
        df = df.resample(timeframe).agg(OHLCV_AGG)
        df = df[df["close"].notna()]
    return df


def load_dir(data_dir: str, timeframe: Optional[str] = None, max_coins: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    names = sorted(f for f in os.listdir(data_dir) if f.endswith((".csv", ".csv.gz")))
    if max_coins:
        names = names[:max_coins]
    out = {}
    for f in names:
        out[f.split(".")[0]] = load_ohlcv(os.path.join(data_dir, f), timeframe)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeframe", default=None, help="pandas offset to resample to, e.g. 1h, 4h, 1D")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--min-run", type=int, default=None, help="default: derive_min_run on the data")
    ap.add_argument("--min-run-hours", type=float, default=None,
                    help="time-based threshold in hours (converted per --timeframe); exclusive with --min-run")
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--train-pct", type=float, default=0.8)
    ap.add_argument("--val-pct", type=float, default=0.1)
    a = ap.parse_args(argv)
    frames = load_dir(a.data_dir, a.timeframe)
    if a.min_run is not None and a.min_run_hours is not None:
        ap.error("--min-run and --min-run-hours are mutually exclusive")
    if a.min_run_hours is not None:
        if not a.timeframe:
            ap.error("--min-run-hours needs --timeframe")
        min_run = min_run_from_hours(a.min_run_hours, a.timeframe)
    else:
        min_run = a.min_run if a.min_run is not None else derive_min_run(frames, a.quantile)
    coins, per = audit_frames(frames, min_run=min_run, horizon=a.horizon,
                              splits=split_bounds(frames, a.horizon, a.train_pct, a.val_pct))
    os.makedirs(a.out, exist_ok=True)
    coins.to_csv(os.path.join(a.out, "coins.csv"), index=False)
    pool_splits(per).to_csv(os.path.join(a.out, "splits_pooled.csv"), index=False)
    print(f"min_run={min_run}; dead coins: {int(coins['dead'].sum())}/{len(coins)}")
    print(pool_splits(per).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

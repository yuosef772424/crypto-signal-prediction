"""Round-2 audit helper: a synthetic 'Drive' (1h + 15m klines, funding, OI, futures_metrics) and a
builder that runs the real notebook functions on it. No Colab, no network.

Why not reuse round-1 fixtures: round 1 ran with phase-2 archives absent, so the 43-feature path
(15m intraday, funding/OI/metrics) was never exercised end to end.
"""
import contextlib
import gzip
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _nbload import ROOT, load_pipeline  # noqa: E402

KL_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume",
           "taker_buy_quote_volume"]


def make_coin_15m(rng, start, end, s0=100.0, vol=0.004):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else start,
                        pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else end,
                        freq="15min", inclusive="left")
    n = len(idx)
    lv = np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.2)
    r = rng.normal(0, vol, n) * lv
    close = s0 * np.exp(np.cumsum(r))
    op = np.r_[s0, close[:-1]]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    v = rng.uniform(50, 150, n) * lv
    tb = v * rng.uniform(0.4, 0.6, n)
    df = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close, "volume": v,
                       "quote_volume": v * close, "trades": rng.integers(100, 400, n).astype(float),
                       "taker_buy_volume": tb, "taker_buy_quote_volume": tb * close}, index=idx)
    return df


def to_1h(k15):
    g = k15.resample("1h")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    return out.dropna(how="all")


def write_archives(root, coins, k15, start, end, rng, funding=True, oi=True, metrics=True,
                   metrics_start=None, only=None):
    """Write history_15m / funding_rate / open_interest / futures_metrics under root in the vision-fetch formats."""
    root = os.fspath(root)
    for d in ("history_15m", "funding_rate", "open_interest", "futures_metrics"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for c in coins:
        if only is not None and c not in only:
            continue
        k = k15[c].copy()
        o = k.reset_index().rename(columns={"index": "datetime_utc"})
        o.insert(0, "timestamp", (k.index.as_unit("ns").asi8 // 1_000_000))   # ms, like the vision archive
        o.columns = ["timestamp", "datetime_utc"] + list(k.columns)
        with gzip.open(os.path.join(root, "history_15m", f"{c}.csv.gz"), "wt") as f:
            o.to_csv(f, index=False)
        if funding:
            fi = pd.date_range(start, end, freq="8h", tz="UTC")
            fr = pd.DataFrame({"timestamp": [str(t) for t in fi],
                               "funding_rate": rng.normal(1e-4, 5e-5, len(fi))})
            with gzip.open(os.path.join(root, "funding_rate", f"{c}.csv.gz"), "wt") as f:
                fr.to_csv(f, index=False)
        hi_ = pd.date_range(metrics_start or start, end, freq="1h", tz="UTC")
        if oi:
            oidf = pd.DataFrame({"timestamp": [f"{t:%Y-%m-%d %H:%M:%S}+00:00" for t in hi_],   # same text format as fetch_history_vision_colab
                                 "open_interest": 1e6 * np.exp(np.cumsum(rng.normal(0, .01, len(hi_))))})
            with gzip.open(os.path.join(root, "open_interest", f"{c}.csv.gz"), "wt") as f:
                oidf.to_csv(f, index=False)
        if metrics:
            m = pd.DataFrame({"timestamp": [f"{t:%Y-%m-%d %H:%M:%S}+00:00" for t in hi_],
                              "sum_open_interest": 1e6 * np.exp(np.cumsum(rng.normal(0, .01, len(hi_)))),
                              "sum_open_interest_value": 1e8,
                              "count_toptrader_long_short_ratio": np.exp(rng.normal(0, .2, len(hi_))),
                              "sum_toptrader_long_short_ratio": np.exp(rng.normal(0, .2, len(hi_))),
                              "count_long_short_ratio": np.exp(rng.normal(0, .2, len(hi_))),
                              "sum_taker_long_short_vol_ratio": np.exp(rng.normal(0, .2, len(hi_)))})
            with gzip.open(os.path.join(root, "futures_metrics", f"{c}.csv.gz"), "wt") as f:
                m.to_csv(f, index=False)


def fresh_ns():
    """Fresh pipeline namespace with the 1h w32/s8 preset applied and phase-2 rooted at a synthetic dir."""
    os.chdir(ROOT)      # _phase2_module looks for tools/intraday_features.py under cwd
    ns = load_pipeline()
    ns["apply_hourly_preset"](ns["HOURLY_W32_S8_OVERRIDES"])
    return ns


def build(ns, root, h1, coins, phase2=True, checkpoint_dir=None, extra_cfg=None, max_workers=1):
    """Run build_dataset_from_loader on in-memory 1h frames `h1` (dict coin->df)."""
    C = ns["CONFIG"]
    ns["_PHASE2_CACHE"].clear()
    C["phase2_data"]["data_root"] = os.fspath(root)
    C["phase2_data"]["use_intraday_15m"] = "auto" if phase2 else False
    C["phase2_data"]["use_futures_metrics"] = "auto" if phase2 else False
    C["feature_order"] = None
    for k, v in (extra_cfg or {}).items():
        C[k] = v
    with contextlib.redirect_stdout(io.StringIO()):
        ds = ns["build_dataset_from_loader"](
            [{"name": c} for c in coins], lambda f, n: h1[n].copy(), ns["make_resample_fn"](C),
            max_workers=max_workers, checkpoint_dir=checkpoint_dir, config=C)
    return ds

"""Round-3 audit helper: synthetic multi-coin universe (15m klines + funding/OI/metrics archives) for the 1h+4h preset,
and a perturbation harness ("replace everything from T on with different values, do earlier samples change?").

Why a perturbation harness and not a value-encoding one: the end-to-end X is normalised, so bar times cannot be read back
from it; what can be tested end to end is dependence. Exact-timestamp checks (which 4h bar closes when) live in the
scripts that feed the real alignment function frames whose value IS the bar open time.
"""
import contextlib
import gzip
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _nbload import ROOT, load_pipeline  # noqa: E402
from _r2_synth import build, make_coin_15m, to_1h, write_archives  # noqa: E402

H_NS = 3_600_000_000_000
LOUD = ("open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")


def universe(seed=5, ncoin=6, start="2025-03-01", end="2025-05-01"):
    rng = np.random.default_rng(seed)
    coins = ["BTCUSDT", "ETHUSDT"] + [f"C{i}USDT" for i in range(ncoin - 2)]
    k15 = {c: make_coin_15m(rng, start, end, s0=100 * (i + 1)) for i, c in enumerate(coins)}
    return coins, k15, start, end


def perturb_k15(k15, T, seed=99):
    """Every 15m row opening at or after T gets a different (random) price factor and volume."""
    r = np.random.default_rng(seed)
    out = {}
    for c, k in k15.items():
        k = k.copy()
        m = k.index >= T
        f = np.exp(r.normal(0, 0.03, m.sum()))
        for col in LOUD:
            k.loc[m, col] = k.loc[m, col].to_numpy() * f
        for col in ("volume", "trades", "taker_buy_volume"):
            k.loc[m, col] = k.loc[m, col].to_numpy() * r.uniform(0.2, 5, m.sum())
        out[c] = k
    return out


def _edit_archives(root, coins, fn):
    for d in ("funding_rate", "open_interest", "futures_metrics"):
        for c in coins:
            p = os.path.join(root, d, f"{c}.csv.gz")
            with gzip.open(p, "rt") as f:
                df = pd.read_csv(f)
            ts = pd.to_datetime(df["timestamp"], utc=True)
            fn(df, ts)
            with gzip.open(p, "wt") as f:
                df.to_csv(f, index=False)


def perturb_archives(root, T, coins, seed=7):
    r = np.random.default_rng(seed)

    def fn(df, ts):
        m = (ts >= T).to_numpy()
        for col in df.columns:
            if col != "timestamp":
                df.loc[m, col] = df.loc[m, col].to_numpy() * np.exp(r.normal(0, 0.5, m.sum()))
    _edit_archives(root, coins, fn)


def scale_archives(root, coins, factor=1e9):
    """Open-interest style level columns x factor (huge raw values) — for the float16 overflow check."""
    def fn(df, ts):
        for col in df.columns:
            if col.startswith("sum_open_interest") or col == "open_interest":
                df[col] = df[col] * factor
    _edit_archives(root, coins, fn)


def make_ns(stride=8, over=None, extra=None):
    """Fresh pipeline namespace with the real shipped 1h+4h preset (HOURLY_4H_OVERRIDES) applied, then `over`."""
    os.chdir(ROOT)
    ns = load_pipeline()
    ns["apply_hourly_preset"](ns["HOURLY_4H_OVERRIDES"])
    ns["CONFIG"]["stride"] = stride
    for k, v in (over or {}).items():
        ns["CONFIG"][k] = v
    if over:
        ns["refresh_features"]()
    return ns


def run(ns, coins, k15, start, end, T=None, phase2=True, scale_k15=None, oi_factor=None):
    """Build the dataset from the synthetic universe; from T on (if given) all coins' raw data are replaced."""
    tmp = tempfile.mkdtemp()
    k = perturb_k15(k15, T) if T is not None else k15
    if scale_k15:
        k = {c: v.assign(**{col: v[col] * f for col, f in scale_k15.items()}) for c, v in k.items()}
    write_archives(tmp, coins, k, start, end, np.random.default_rng(11))
    if T is not None:
        perturb_archives(tmp, T, coins)
    if oi_factor:
        scale_archives(tmp, coins, oi_factor)
    h1 = {c: to_1h(k[c]) for c in coins}
    return build(ns, tmp, h1, coins, phase2=phase2)


def keyed(ds, ts_col):
    lc = np.asarray(ds["last_candles"])
    ts = lc[:, ts_col].astype("int64")
    names = np.empty(len(ts), dtype=object)
    for b in ds["asset_bounds"]:
        names[b["start"]:b["end"]] = b["name"]
    return {(names[i], int(ts[i])): i for i in range(len(ts))}


def compare_before(A, B, ts_col, T, tfs=("1h", "4h")):
    """Samples whose information time (last 1h bar open + 1h) is <= T must be bit-identical in A and B.
    Returns (checked, {tf: changed}, first_bad, after_T_samples, after_T_changed_4h)."""
    KA, KB = keyed(A, ts_col), keyed(B, ts_col)
    Tn = pd.Timestamp(T).value
    checked, changed, first_bad = 0, {tf: 0 for tf in tfs}, None
    for key, i in KA.items():
        if key[1] + H_NS <= Tn and key in KB:
            j = KB[key]
            checked += 1
            for tf in tfs:
                if not np.array_equal(A[f"X_{tf}"][i], B[f"X_{tf}"][j]):
                    changed[tf] += 1
                    first_bad = first_bad or (key[0], pd.Timestamp(key[1], tz="UTC"), tf)
    after = [(k, i) for k, i in KA.items() if k[1] >= Tn and k in KB]
    after_changed = sum(1 for k, i in after if not np.array_equal(A["X_4h"][i], B["X_4h"][KB[k]]))
    return checked, changed, first_bad, len(after), after_changed


# ── exact-timestamp helpers: frames whose ONE column is the bar's own OPEN time in hours since epoch ──────────────────
def hour_frames(start, n_hours, drop_1h=(), drop_4h=()):
    """{'1h': frame, '4h': frame}: value = open time in hours (exact in float32). drop_* remove bars (data holes)."""
    i1 = pd.date_range(start, periods=n_hours, freq="1h", tz="UTC")
    i1 = i1.difference(pd.DatetimeIndex(drop_1h))
    i4 = pd.date_range(pd.Timestamp(start, tz="UTC").floor("4h"), periods=n_hours // 4, freq="4h", tz="UTC")
    i4 = i4.difference(pd.DatetimeIndex(drop_4h))

    def f(idx):
        return pd.DataFrame({"v": idx.as_unit("ns").asi8 // H_NS}, index=idx).astype("float64")
    return {"1h": f(i1), "4h": f(i4)}


def closed_cfg(ns, stride, mode="closed", **kw):
    from copy import deepcopy
    cfg = deepcopy(ns["CONFIG"])
    cfg.update(tf_order=["1h", "4h"], base_tf="1h", window_sizes={"1h": 32, "4h": 32}, stride=stride,
               align_windows_to_grid=True, higher_tf_mode=mode, forecast_horizon=1, embargo_candles=None,
               holdout_start=None, min_split_samples=1, keep_asset_test_separate=False)
    cfg.update(kw)
    return cfg


def aligned_hour_dataset(ns, cfg, dfs):
    """Run the REAL alignment on hour-valued frames and wrap the result as a dataset dict the REAL split_data accepts.
    X_1h / X_4h[...,0] then hold the real open times (hours) of every bar in every window."""
    out, ends = ns["align_multi_timeframes_time_based"](dfs, ["1h", "4h"], cfg["window_sizes"], cfg["stride"], cfg)
    n = len(ends)
    ts = np.array([e.value for e in ends], dtype="float64")
    lc = np.zeros((n, len(ns["LAST_COLUMNS"])))
    lc[:, ns["TS_COL"]] = ts
    return {"X_1h": out["1h"], "X_4h": out["4h"], "base_params": np.zeros((n, 2), "float32"), "last_candles": lc,
            "y_t": np.zeros(n, "float32"), "timeframes": ["1h", "4h"], "targets": ["t"], "base_tf": "1h",
            "window_sizes": dict(cfg["window_sizes"]), "forecast_horizon": cfg["forecast_horizon"],
            "stride": cfg["stride"], "higher_tf_mode": cfg["higher_tf_mode"]}


def overlap_with_train_targets(ns, ds, split, train, horizon=1):
    """How many samples of `split` have any input bar (1h window or 4h window) at or after the OPEN of a train sample's
    target candle (train t + 1h .. + horizon), i.e. whose input window contains a candle a train label was built from."""
    ts_col = ns["TS_COL"]
    tr_t = np.asarray(train["last_candles"])[:, ts_col] / H_NS               # hours
    if not len(tr_t):
        return 0, None
    last_target_open = tr_t.max() + 1                                         # open of the newest train target candle
    first_open = np.minimum(split["X_1h"][:, 0, 0], split["X_4h"][:, 0, 0]).astype("float64")
    bad = first_open <= last_target_open
    worst = None
    if bad.any():
        i = int(np.argmax(bad))
        worst = (pd.Timestamp(np.asarray(split["last_candles"])[i, ts_col], tz="UTC"),
                 pd.Timestamp(int(split["X_4h"][i, 0, 0]) * H_NS, tz="UTC"))
    return int(bad.sum()), worst

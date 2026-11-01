"""
PURPOSE:  Path-resolved 1:1 bracket backtest of a model's signals. For every sample: direction = sign of the model's close prediction
          (or P(up) - 0.5 of a class head); enter at the signal candle's close (option: the next open); take-profit and stop at EQUAL
          distance (fixed %, k x ATR known at entry, or k x the model's predicted range = min of the predicted up/down room); hold over
          the NEXT N bars of the model timeframe until the first barrier is touched, else exit at the last bar's close. The touch ORDER is
          read bar by bar (and, when both barriers fall inside one bar, from finer bars if given; unresolved bars count stop-first and
          their share is reported). Outputs: win rate vs break-even, avg win / loss, profit factor, net return per trade and per day,
          day-clustered t, per-coin consistency, long vs short, by confidence decile (does the hit rate rise with confidence?), and a
          random-direction null on the same samples. Works from a PredictionTable (stats/prediction_table.py, read-only use) or any
          DataFrame with asset, timestamp, pred columns.
TAGS:     bracket, 1:1, take profit, stop loss, path resolved, touch order, finer bars, 15m, ATR stop, predicted range, timeout exit,
          break-even win rate, profit factor, confidence decile, accuracy vs confidence, random-direction null, day-clustered t,
          per-coin consistency, long short split, PredictionTable, backtest, trades, costs, slippage
PITFALLS: `timestamp` = the OPEN time of the signal bar (window end bar, as in the pipeline's last_candles); its close is the entry and the
          next N bars are timestamp + k x timeframe, k = 1..N: a sample whose signal bar or any of the N bars is missing in the candles is
          SKIPPED (reported, never filled). Entry at the close assumes a fill at that price (the next bar may open elsewhere: a gap
          beyond the stop fills at the open; a gap beyond the target fills at the target, conservative). Both barriers in one bar and no
          finer bar resolving it (or the finer bar itself touching both) = STOP FIRST (the honest lower bound; ambiguous='tp_first' is the
          upper bound) and the share of such exits is `ambiguous_share`: trust a result only if it holds with that share small. Costs:
          `cost_rt` is paid on every trade; `stop_slip` extra on stop exits only. ATR is Wilder (ewm alpha 1/period) of the candles up to
          and including the signal bar (causal), an approximation of pandas_ta NATR_14: pass a `natr_pct` column to override.
          predicted-range sizing needs `pred_high` / `pred_low` prices (signals_from_table decodes out_high / out_low with `reg_scale`
          relative to the signal bar's high / low, as tools/bracket_eval.frame_from_split does). The confidence decile table answers
          "when confident, is it right": hit = the sign of the horizon close return equals the direction (path-independent), also the
          net. Choosing a sizing / N on the test split is data snooping: pick on val, confirm once on test (docs/research/RESEARCH_RULES.md).
          Related: tools/bracket_eval.py (single horizon, order unknown), F-0032 / F-0037 / F-0057 / F-0062 in failure_registry.csv.

Usage:
    import sys; sys.path.insert(0, "tools"); import bracket_path as bp
    sig = bp.signals_from_table(table, head="close")            # table: stats.prediction_table PredictionTable
    res = bp.run_backtest(sig, candles, timeframe="1h", hold_bars=6, sizing="atr", size_value=1.5,
                          fine_candles=candles_15m, cost_rt=0.0008, stop_slip=0.0002)
    print(bp.format_report(res)); res.trades; res.by_confidence; res.null
CLI:
    python tools/bracket_path.py --table preds.csv.gz --candles-dir data_1h --timeframe 1h --hold-bars 6 \\
        --sizing atr --size-value 1.5 --fine-dir data_15m --fine-timeframe 15m --split-role test
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from core.schema import TARGET_COLUMNS  # noqa: E402  (high, low, close)
from stats import metrics as _m  # noqa: E402  (tier 1: the one implementation of profit factor, win/loss ratio, spearman, wilson)
from stats.prediction_table import to_utc  # noqa: E402
from bracket_eval import day_clustered_t  # noqa: E402  (the repo's day-clustered t, one implementation)

COST_RT = 0.0008          # 0.08% round trip (owner's number)
STOP_SLIP = 0.0002        # extra adverse slippage on stop exits, fraction of price
HOLD_BARS = 6
ATR_PERIOD = 14
SIZINGS = ("fixed", "atr", "pred_range")
ENTRIES = ("close", "next_open")
AMBIGUOUS = ("stop_first", "tp_first")
_EPS = 1e-12
OHLC = ("open", "high", "low", "close")

# exit reasons
TP, SL, TIMEOUT = "tp", "sl", "timeout"


# --------------------------------------------------------------------------------------------------------- inputs
def _check_choice(name: str, value: str, allowed: Tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name}: one of {list(allowed)}, not {value!r}")


def _tf_ns(timeframe: str) -> int:
    ns = int(pd.Timedelta(str(timeframe)).value)
    if ns <= 0:
        raise ValueError(f"timeframe {timeframe!r} is not positive")
    return ns


def _ts_ns(values: Any, what: str) -> np.ndarray:
    return to_utc(values, what=what).asi8.astype("int64")


def prep_candles(df: pd.DataFrame, what: str = "candles") -> pd.DataFrame:
    """Candles as asset, ts (int64 ns, the bar OPEN time), open, high, low, close; sorted; a duplicate (asset, ts) raises."""
    need = ["asset", "timestamp", *OHLC]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{what}: missing column(s) {miss}; need {need}")
    out = pd.DataFrame({"asset": df["asset"].astype(str).to_numpy(), "ts": _ts_ns(df["timestamp"], f"{what}.timestamp")})
    for c in OHLC:
        out[c] = pd.to_numeric(df[c]).to_numpy(dtype="float64")
    out = out.sort_values(["asset", "ts"], kind="mergesort").reset_index(drop=True)
    if out.duplicated(["asset", "ts"]).any():
        raise ValueError(f"{what}: duplicate (asset, timestamp) rows")
    return out


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Finer candles (asset, timestamp, OHLC) -> ``timeframe`` bars, left-labelled and left-closed (the bar OPEN time, as the pipeline).
    A bar is kept only if it is complete (holds every fine bar the interval needs): the incomplete edge bars are dropped."""
    f = prep_candles(df, "fine candles")
    tf = pd.Timedelta(str(timeframe))
    parts = []
    for a, g in f.groupby("asset", sort=True):
        s = g.set_index(pd.to_datetime(g["ts"], unit="ns", utc=True))
        step = pd.Series(s.index).diff().dropna().mode()
        step = step.iloc[0] if len(step) else tf
        need = max(int(round(tf / step)), 1)
        r = s.resample(tf, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "ts": "count"}).rename(columns={"ts": "n"})
        r = r[r["n"] >= need].drop(columns="n")
        r.insert(0, "asset", a)
        r.insert(1, "timestamp", r.index)
        parts.append(r.reset_index(drop=True))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["asset", "timestamp", *OHLC])


def load_candles_dir(path: str, interval: str, assets=None, suffix: str = "") -> pd.DataFrame:
    """Per-coin files ``<path>/ohlc_<SYMBOL>_<interval>.parquet`` (tools/fetch_crypto_dataset.py: UTC DatetimeIndex + open/high/low/close)
    -> one frame. ``suffix``: stripped from the SYMBOL to get the asset name (e.g. ``USDT``); ``assets``: keep only these names."""
    frames = []
    for fn in sorted(os.listdir(path)):
        if not (fn.startswith("ohlc_") and fn.endswith(f"_{interval}.parquet")):
            continue
        sym = fn[len("ohlc_"):-len(f"_{interval}.parquet")]
        name = sym[:-len(suffix)] if suffix and sym.endswith(suffix) else sym
        if assets is not None and name not in set(assets):
            continue
        d = pd.read_parquet(os.path.join(path, fn))
        d = d.reset_index() if "timestamp" not in d.columns else d
        if "timestamp" not in d.columns:
            d = d.rename(columns={d.columns[0]: "timestamp"})
        d["asset"] = name
        frames.append(d[["asset", "timestamp", *OHLC]])
    if not frames:
        raise FileNotFoundError(f"no ohlc_*_{interval}.parquet in {path}")
    return pd.concat(frames, ignore_index=True)


def signals_from_table(table: pd.DataFrame, *, head: str = "close", candles: Optional[pd.DataFrame] = None, reg_scale: float = 1.0,
                       split_role: Optional[str] = None, confidence: Optional[str] = None) -> pd.DataFrame:
    """The signals frame (asset, timestamp, pred, confidence [, pred_high, pred_low]) of a PredictionTable (read only, never modified).

    pred: ``out_<head>_p_up - 0.5`` when the class head output exists (direction = its sign, confidence = |p - 0.5|), else ``out_<head>``
    (the regression output; direction = its sign, confidence = |pred|). ``confidence``: the name of another ``out_*`` column to use
    instead (e.g. ``out_close_confidence``). ``candles`` + ``reg_scale``: also decode ``out_high`` / ``out_low`` (returns x reg_scale
    relative to the signal bar's own high / low) into ``pred_high`` / ``pred_low`` for ``sizing='pred_range'``. ``split_role``: keep only
    that role. Raises if neither output exists."""
    from stats.prediction_table import validate_table
    validate_table(table)
    t = table if split_role is None else table[table["split_role"] == split_role]
    p_col, r_col = f"out_{head}_p_up", f"out_{head}"
    if p_col in t.columns:
        pred = t[p_col].to_numpy(float) - 0.5
        conf = np.abs(pred)
        source = p_col
    elif r_col in t.columns:
        pred = t[r_col].to_numpy(float)
        conf = np.abs(pred)
        source = r_col
    else:
        raise ValueError(f"table has neither {p_col!r} nor {r_col!r}; out_* columns: {[c for c in t.columns if c.startswith('out_')]}")
    if confidence is not None:
        if confidence not in t.columns:
            raise ValueError(f"confidence column {confidence!r} not in the table")
        conf = t[confidence].to_numpy(float)
        source = confidence
    sig = pd.DataFrame({"asset": t["asset"].to_numpy(), "timestamp": t["timestamp"].to_numpy(), "pred": pred, "confidence": conf})
    sig.attrs["confidence_source"] = source
    if candles is not None and {"out_high", "out_low"} <= set(t.columns):
        c = prep_candles(candles)
        key = pd.DataFrame({"asset": sig["asset"].astype(str).to_numpy(), "ts": _ts_ns(sig["timestamp"], "timestamp")})
        m = key.merge(c, on=["asset", "ts"], how="left")
        sig["pred_high"] = m["high"].to_numpy() * (1.0 + t["out_high"].to_numpy(float) / float(reg_scale))
        sig["pred_low"] = m["low"].to_numpy() * (1.0 + t["out_low"].to_numpy(float) / float(reg_scale))
    return sig.reset_index(drop=True)


def _prep_signals(sig: pd.DataFrame) -> pd.DataFrame:
    miss = [c for c in ("asset", "timestamp", "pred") if c not in sig.columns]
    if miss:
        raise ValueError(f"signals: missing column(s) {miss}; need asset, timestamp, pred (PredictionTable: use signals_from_table)")
    out = pd.DataFrame({"asset": sig["asset"].astype(str).to_numpy(), "ts": _ts_ns(sig["timestamp"], "signals.timestamp"),
                        "pred": pd.to_numeric(sig["pred"]).to_numpy(dtype="float64")})
    if out.duplicated(["asset", "ts"]).any():
        raise ValueError("signals: duplicate (asset, timestamp) rows (one prediction per sample; filter fold / role first)")
    conf = sig["confidence"] if "confidence" in sig.columns else None
    out["confidence"] = pd.to_numeric(conf).to_numpy(dtype="float64") if conf is not None else np.nan
    for c in ("pred_high", "pred_low", "natr_pct"):
        if c in sig.columns:
            out[c] = pd.to_numeric(sig[c]).to_numpy(dtype="float64")
    return out


# ----------------------------------------------------------------------------------------------------- the path book
class _Book:
    """Per-asset sorted candle arrays; ``paths`` gathers the next N bars of each signal by exact time (never by position)."""

    def __init__(self, candles: pd.DataFrame):
        self.by_asset: Dict[str, Dict[str, np.ndarray]] = {}
        for a, g in candles.groupby("asset", sort=False):
            self.by_asset[a] = {k: g[k].to_numpy() for k in ("ts", *OHLC)}

    def atr_frac(self, asset: str, ts: np.ndarray, period: int) -> np.ndarray:
        """ATR / close at the signal bar (Wilder, causal, NaN until ``period`` bars exist)."""
        d = self.by_asset.get(asset)
        if d is None:
            return np.full(len(ts), np.nan)
        h, l, c = d["high"], d["low"], d["close"]
        pc = np.r_[np.nan, c[:-1]]
        tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
        atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy().copy()
        atr[:period] = np.nan
        j = np.searchsorted(d["ts"], ts)
        ok = (j < len(d["ts"]))
        j2 = np.minimum(j, len(d["ts"]) - 1)
        ok &= d["ts"][j2] == ts
        return np.where(ok, atr[j2] / c[j2], np.nan)

    def paths(self, asset: str, ts: np.ndarray, n_bars: int, tf_ns: int):
        """Arrays (n,), (n, N)x4 for one asset: signal close / high / low, then open/high/low/close of bars ts+k*tf (NaN if absent)."""
        n = len(ts)
        sig = {k: np.full(n, np.nan) for k in TARGET_COLUMNS}
        arr = {k: np.full((n, n_bars), np.nan) for k in OHLC}
        d = self.by_asset.get(asset)
        if d is None:
            return sig, arr
        T = d["ts"]

        def locate(target):
            j = np.minimum(np.searchsorted(T, target), len(T) - 1)
            return j, T[j] == target

        j, ok = locate(ts)
        for k in TARGET_COLUMNS:
            sig[k] = np.where(ok, d[k][j], np.nan)
        for b in range(n_bars):
            j, ok = locate(ts + (b + 1) * tf_ns)
            for k in OHLC:
                arr[k][:, b] = np.where(ok, d[k][j], np.nan)
        return sig, arr


class _Fine:
    """Finer bars per asset, to resolve the touch order inside one model-timeframe bar."""

    def __init__(self, fine: pd.DataFrame):
        self.by_asset = {a: {k: g[k].to_numpy() for k in ("ts", *OHLC)} for a, g in fine.groupby("asset", sort=False)}

    def first_touch(self, asset: str, start: int, end: int, side: float, entry: float, dist: float) -> int:
        """+1 target first, -1 stop first, 0 unresolved, inside [start, end) from the fine bars. A fine bar touching both = unresolved."""
        d = self.by_asset.get(asset)
        if d is None:
            return 0
        lo, hi = np.searchsorted(d["ts"], [start, end])
        for i in range(lo, hi):
            o = side * (d["open"][i] / entry - 1.0)
            if o <= -dist + _EPS:
                return -1
            if o >= dist - _EPS:
                return 1
            if side > 0:
                fav, adv = d["high"][i] / entry - 1.0, 1.0 - d["low"][i] / entry
            else:
                fav, adv = 1.0 - d["low"][i] / entry, d["high"][i] / entry - 1.0
            tp, sl = fav >= dist - _EPS, adv >= dist - _EPS
            if tp and sl:
                return 0
            if tp:
                return 1
            if sl:
                return -1
        return 0


def _simulate_side(side: float, entry: np.ndarray, dist: np.ndarray, arr: Dict[str, np.ndarray], bar_start: np.ndarray, tf_ns: int,
                   asset_of: np.ndarray, fine: Optional[_Fine], ambiguous: str) -> Dict[str, np.ndarray]:
    """Path outcome of trading every sample in direction ``side`` (+1 / -1): gross return (fraction of entry), reason, bar index, flags."""
    n, N = arr["close"].shape
    e = entry[:, None]
    rel = lambda a: a / e - 1.0  # noqa: E731
    o_rel = side * rel(arr["open"])
    if side > 0:
        fav, adv = rel(arr["high"]), -rel(arr["low"])
    else:
        fav, adv = -rel(arr["low"]), rel(arr["high"])
    d = dist[:, None]
    gross = np.full(n, np.nan)
    reason = np.full(n, TIMEOUT, dtype=object)
    bar = np.full(n, N - 1, dtype="int64")
    done = np.zeros(n, bool)
    both_bars = np.zeros(n, bool)       # the exit bar had both barriers inside it (not a gap)
    fine_resolved = np.zeros(n, bool)
    ambiguous_exit = np.zeros(n, bool)
    for b in range(N):
        active = ~done
        gap_sl = active & (o_rel[:, b] <= -dist + _EPS)
        gap_tp = active & ~gap_sl & (o_rel[:, b] >= dist - _EPS)
        rest = active & ~gap_sl & ~gap_tp
        t_hit = rest & (fav[:, b] >= dist - _EPS)
        s_hit = rest & (adv[:, b] >= dist - _EPS)
        both = t_hit & s_hit
        first_tp = t_hit & ~s_hit
        first_sl = s_hit & ~t_hit
        for i in np.flatnonzero(both):
            both_bars[i] = True
            r = 0
            if fine is not None:
                st = int(bar_start[i]) + (b + 1) * tf_ns
                r = fine.first_touch(asset_of[i], st, st + tf_ns, side, entry[i], dist[i])
            if r != 0:
                fine_resolved[i] = True
            else:
                ambiguous_exit[i] = True
                r = -1 if ambiguous == "stop_first" else 1
            (first_tp if r > 0 else first_sl)[i] = True
        gross[gap_sl] = o_rel[gap_sl, b]
        gross[gap_tp] = dist[gap_tp]
        gross[first_tp] = dist[first_tp]
        gross[first_sl] = -dist[first_sl]
        reason[gap_sl | first_sl] = SL
        reason[gap_tp | first_tp] = TP
        hit = gap_sl | gap_tp | first_tp | first_sl
        bar[hit] = b
        done |= hit
    last = side * rel(arr["close"][:, N - 1:N])[:, 0]
    gross = np.where(done, gross, last)
    # both-bars / ambiguity only count for the bar the trade actually exited on
    return {"gross": gross, "reason": reason, "bar": bar, "both": both_bars & done, "fine": fine_resolved & done,
            "ambiguous": ambiguous_exit & done, "horizon_ret": last}


def _net(sim: Dict[str, np.ndarray], cost_rt: float, stop_slip: float) -> np.ndarray:
    return sim["gross"] - cost_rt - np.where(sim["reason"] == SL, stop_slip, 0.0)


# ------------------------------------------------------------------------------------------------------- summaries
def _daily(net: np.ndarray, ts: pd.Series) -> Tuple[float, float, float]:
    d = pd.DataFrame({"r": net, "d": pd.DatetimeIndex(ts).floor("1D")}).groupby("d")["r"].agg(["sum", "count"])
    return float(d["sum"].mean()), float(d["count"].mean()), float(len(d))


def _summarize(tr: pd.DataFrame, cost_rt: float, stop_slip: float) -> Dict[str, Any]:
    n = len(tr)
    if n == 0:
        return {"n": 0}
    net = tr["net"].to_numpy(float)
    wins, losses = net[net > 0], net[net < 0]
    d = tr["dist"].to_numpy(float)
    be = float(np.mean((d + cost_rt + stop_slip) / (2.0 * d + stop_slip)))
    tp, sl = int((tr["reason"] == TP).sum()), int((tr["reason"] == SL).sum())
    per_day, trades_per_day, n_days = _daily(net, tr["ts"])
    k_win = int((net > 0).sum())
    lo, hi = _m.wilson_interval(k_win, n)
    return {
        "n": n, "win_rate": k_win / n, "win_rate_wilson": (lo, hi), "breakeven_win": be,
        "tp_share": tp / n, "sl_share": sl / n, "timeout_share": float((tr["reason"] == TIMEOUT).mean()),
        "win_rate_decided": (tp / (tp + sl)) if tp + sl else float("nan"),
        "avg_win_bp": float(wins.mean() * 1e4) if len(wins) else float("nan"),
        "avg_loss_bp": float(losses.mean() * 1e4) if len(losses) else float("nan"),
        "profit_factor": _m.profit_factor(net), "win_loss_ratio": _m.win_loss_ratio(net),
        "net_per_trade_bp": float(net.mean() * 1e4), "gross_per_trade_bp": float(tr["gross"].mean() * 1e4),
        "net_per_day_bp": per_day * 1e4, "trades_per_day": trades_per_day, "n_days": int(n_days),
        "t_day_clustered": day_clustered_t(net, tr["ts"]),
        "horizon_dir_acc": float((tr["horizon_ret"] > 0).mean()), "mean_bars_held": float((tr["bar"] + 1).mean()),
        "mean_dist_bp": float(d.mean() * 1e4),
    }


def _group(tr: pd.DataFrame, by: str, cost_rt: float, stop_slip: float) -> pd.DataFrame:
    rows = []
    for k, g in tr.groupby(by, sort=True):
        s = _summarize(g, cost_rt, stop_slip)
        rows.append({by: k, **{c: s.get(c, np.nan) for c in ("n", "win_rate", "breakeven_win", "net_per_trade_bp", "profit_factor",
                                                             "horizon_dir_acc", "t_day_clustered")}})
    return pd.DataFrame(rows)


def _confidence_table(tr: pd.DataFrame, bins: int, cost_rt: float, stop_slip: float):
    c = tr["confidence"]
    if c.notna().sum() < max(bins * 5, 20) or c.nunique() < 2:
        return None, {}
    q = pd.qcut(c.rank(method="first"), bins, labels=False)
    rows = []
    for k in sorted(q.dropna().unique()):
        g = tr[q == k]
        s = _summarize(g, cost_rt, stop_slip)
        rows.append({"decile": int(k) + 1, "n": s["n"], "conf_lo": float(g["confidence"].min()), "conf_hi": float(g["confidence"].max()),
                     "hit_rate": s["horizon_dir_acc"], "win_rate": s["win_rate"], "win_rate_decided": s["win_rate_decided"],
                     "net_per_trade_bp": s["net_per_trade_bp"], "profit_factor": s["profit_factor"]})
    t = pd.DataFrame(rows)
    top, bot = t.iloc[-1], t.iloc[0]
    k_top, k_bot = round(top["hit_rate"] * top["n"]), round(bot["hit_rate"] * bot["n"])
    extra = {
        "hit_vs_confidence_spearman": _m.spearman(c.to_numpy(float), (tr["horizon_ret"] > 0).to_numpy(float)),
        "net_vs_confidence_spearman": _m.spearman(c.to_numpy(float), tr["net"].to_numpy(float)),
        "decile_hit_slope_spearman": _m.spearman(t["decile"].to_numpy(float), t["hit_rate"].to_numpy(float)),
        "top_minus_bottom_hit": float(top["hit_rate"] - bot["hit_rate"]),
        "top_hit_wilson": _m.wilson_interval(k_top, int(top["n"])), "bottom_hit_wilson": _m.wilson_interval(k_bot, int(bot["n"])),
    }
    return t, extra


def _null(net_long: np.ndarray, net_short: np.ndarray, real_mean: float, reps: int, seed: int) -> Dict[str, float]:
    """Random directions on the SAME samples (same sizing, same paths): the mean net of each draw."""
    if reps <= 0:
        return {}
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for r in range(reps):
        means[r] = np.where(rng.random(len(net_long)) < 0.5, net_long, net_short).mean()
    sd = float(means.std(ddof=1)) if reps > 1 else float("nan")
    return {"reps": reps, "null_mean_bp": float(means.mean() * 1e4), "null_sd_bp": sd * 1e4,
            "null_p": float((means >= real_mean).mean()),
            "z": float((real_mean - means.mean()) / sd) if sd and sd > 0 else float("nan")}


# --------------------------------------------------------------------------------------------------------- the run
@dataclasses.dataclass
class BracketPathResult:
    """``trades``: one row per taken trade; ``summary``: dict of the headline numbers; the other tables are the breakdowns."""
    trades: pd.DataFrame
    summary: Dict[str, Any]
    by_coin: pd.DataFrame
    by_side: pd.DataFrame
    by_confidence: Optional[pd.DataFrame]
    confidence_stats: Dict[str, Any]
    null: Dict[str, float]
    config: Dict[str, Any]


def run_backtest(signals: pd.DataFrame, candles: pd.DataFrame, *, timeframe: str, hold_bars: int = HOLD_BARS, sizing: str = "fixed",
                 size_value: float = 0.01, entry: str = "close", fine_candles: Optional[pd.DataFrame] = None,
                 cost_rt: float = COST_RT, stop_slip: float = STOP_SLIP, ambiguous: str = "stop_first", atr_period: int = ATR_PERIOD,
                 min_dist: Optional[float] = None, max_dist: Optional[float] = None, confidence_bins: int = 10, null_reps: int = 200,
                 seed: int = 0) -> BracketPathResult:
    """Run the path-resolved 1:1 bracket backtest.

    ``signals``: asset, timestamp (the signal bar's OPEN time), pred (direction = its sign; 0 / NaN = no trade) and optionally
    ``confidence``, ``pred_high`` / ``pred_low`` (prices), ``natr_pct`` (percent; overrides the ATR computed from candles).
    ``candles``: asset, timestamp (bar open), open, high, low, close at ``timeframe``. ``sizing``: 'fixed' (distance = ``size_value`` as a
    fraction), 'atr' (``size_value`` x ATR/close at the signal bar), 'pred_range' (``size_value`` x min(pred_high/entry - 1,
    1 - pred_low/entry)). ``entry``: 'close' of the signal bar or the 'next_open'. ``hold_bars``: the N bars after the signal bar.
    ``fine_candles``: finer bars (e.g. 15m) that resolve a bar where both barriers are inside; without them such a bar is stop-first.
    ``min_dist`` / ``max_dist``: explicit distance clip (default none; a non-positive or non-finite distance skips the trade)."""
    _check_choice("sizing", sizing, SIZINGS)
    _check_choice("entry", entry, ENTRIES)
    _check_choice("ambiguous", ambiguous, AMBIGUOUS)
    if isinstance(hold_bars, bool) or not isinstance(hold_bars, (int, np.integer)) or hold_bars < 1:
        raise ValueError(f"hold_bars must be an int >= 1, got {hold_bars!r}")
    if not (cost_rt >= 0 and stop_slip >= 0):
        raise ValueError("cost_rt and stop_slip must be >= 0")
    if not size_value > 0:
        raise ValueError("size_value must be > 0")
    tf_ns = _tf_ns(timeframe)
    sig = _prep_signals(signals)
    if sizing == "pred_range" and not {"pred_high", "pred_low"} <= set(sig.columns):
        raise ValueError("sizing='pred_range' needs pred_high and pred_low columns (signals_from_table(..., candles=, reg_scale=))")
    book = _Book(prep_candles(candles))
    fine = _Fine(prep_candles(fine_candles, "fine candles")) if fine_candles is not None else None

    n_all = len(sig)
    side = np.where(np.isfinite(sig["pred"]), np.sign(sig["pred"]), 0.0)
    pieces = []
    skipped = {"no_direction": int((side == 0).sum()), "no_candles": 0, "missing_bars": 0, "bad_size": 0}
    for a, idx in sig.groupby("asset", sort=False).indices.items():
        g = sig.iloc[idx]
        ts = g["ts"].to_numpy()
        sg, arr = book.paths(a, ts, int(hold_bars), tf_ns)
        has_sig = np.isfinite(sg["close"])
        full = np.all(np.isfinite(arr["close"]) & np.isfinite(arr["open"]) & np.isfinite(arr["high"]) & np.isfinite(arr["low"]), axis=1)
        if a not in book.by_asset:
            skipped["no_candles"] += int((side[idx] != 0).sum())
            continue
        px = sg["close"] if entry == "close" else arr["open"][:, 0]
        e_ok = np.isfinite(px) & (px > 0)
        if sizing == "fixed":
            dist = np.full(len(g), float(size_value))
        elif sizing == "atr":
            nat = g["natr_pct"].to_numpy() / 100.0 if "natr_pct" in g.columns else book.atr_frac(a, ts, atr_period)
            dist = size_value * nat
        else:
            up = g["pred_high"].to_numpy() / px - 1.0
            dn = 1.0 - g["pred_low"].to_numpy() / px
            dist = size_value * np.minimum(up, dn)
        if min_dist is not None:
            dist = np.where(np.isfinite(dist), np.maximum(dist, min_dist), dist)
        if max_dist is not None:
            dist = np.where(np.isfinite(dist), np.minimum(dist, max_dist), dist)
        good_size = np.isfinite(dist) & (dist > 0)
        trade = side[idx] != 0
        path_ok = has_sig & full & e_ok
        skipped["missing_bars"] += int((trade & ~path_ok).sum())
        skipped["bad_size"] += int((trade & path_ok & ~good_size).sum())
        keep = trade & path_ok & good_size
        if not keep.any():
            continue
        k = np.flatnonzero(keep)
        sub = {kk: v[k] for kk, v in arr.items()}
        assets_k = np.full(len(k), a, dtype=object)
        e_k, d_k = px[k], dist[k]
        sims = {s: _simulate_side(float(s), e_k, d_k, sub, ts[k], tf_ns, assets_k, fine, ambiguous) for s in (1, -1)}
        chosen = side[idx][k]
        pick = lambda key: np.where(chosen > 0, sims[1][key], sims[-1][key])  # noqa: E731
        df = pd.DataFrame({
            "asset": a, "ts": pd.to_datetime(ts[k], unit="ns", utc=True), "side": chosen, "confidence": g["confidence"].to_numpy()[k],
            "entry": e_k, "dist": d_k, "gross": pick("gross"), "reason": pick("reason"), "bar": pick("bar"),
            "both": pick("both"), "fine": pick("fine"), "ambiguous": pick("ambiguous"), "horizon_ret": pick("horizon_ret"),
            "net_long": _net(sims[1], cost_rt, stop_slip), "net_short": _net(sims[-1], cost_rt, stop_slip),
        })
        pieces.append(df)
    cols = ["asset", "ts", "side", "confidence", "entry", "dist", "gross", "reason", "bar", "both", "fine", "ambiguous", "horizon_ret",
            "net_long", "net_short"]
    allt = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=cols)
    allt["net"] = np.where(allt["side"] > 0, allt["net_long"], allt["net_short"]) if len(allt) else []
    allt = allt.sort_values(["ts", "asset"], kind="mergesort").reset_index(drop=True)
    summary = _summarize(allt, cost_rt, stop_slip)
    summary.update(n_signals=n_all, skipped=skipped, coverage=(len(allt) / n_all) if n_all else float("nan"))
    cfg = {"timeframe": timeframe, "hold_bars": int(hold_bars), "sizing": sizing, "size_value": float(size_value), "entry": entry,
           "cost_rt": cost_rt, "stop_slip": stop_slip, "ambiguous": ambiguous, "fine": fine is not None, "atr_period": atr_period,
           "null_reps": null_reps, "seed": seed}
    if len(allt) == 0:
        return BracketPathResult(allt, summary, pd.DataFrame(), pd.DataFrame(), None, {}, {}, cfg)
    both = int(allt["both"].sum())
    summary.update(both_touched_exits=both, resolved_by_fine=int(allt["fine"].sum()), ambiguous_exits=int(allt["ambiguous"].sum()),
                   ambiguous_share=float(allt["ambiguous"].mean()),
                   ambiguous_share_of_both=(float(allt["ambiguous"].sum() / both) if both else float("nan")))
    summary["side_share_long"] = float((allt["side"] > 0).mean())
    by_coin = _group(allt, "asset", cost_rt, stop_slip)
    summary["coins_positive"] = f"{int((by_coin['net_per_trade_bp'] > 0).sum())}/{len(by_coin)}"
    summary["coin_net_median_bp"] = float(by_coin["net_per_trade_bp"].median())
    allt["side_name"] = np.where(allt["side"] > 0, "long", "short")
    by_side = _group(allt, "side_name", cost_rt, stop_slip)
    conf_tab, conf_stats = _confidence_table(allt, int(confidence_bins), cost_rt, stop_slip)
    nul = _null(allt["net_long"].to_numpy(float), allt["net_short"].to_numpy(float), float(allt["net"].mean()), int(null_reps), int(seed))
    return BracketPathResult(allt.drop(columns=["side_name"]), summary, by_coin, by_side, conf_tab, conf_stats, nul, cfg)


# -------------------------------------------------------------------------------------------------------- reporting
def format_report(res: BracketPathResult) -> str:
    s, c = res.summary, res.config
    L = [f"bracket 1:1 path-resolved | tf={c['timeframe']} hold={c['hold_bars']} bars sizing={c['sizing']}({c['size_value']}) "
         f"entry={c['entry']} cost_rt={c['cost_rt']:.4%} stop_slip={c['stop_slip']:.4%} fine_bars={c['fine']} ambiguous={c['ambiguous']}"]
    if s.get("n", 0) == 0:
        return L[0] + f"\nno trades; skipped: {s.get('skipped')}"
    lo, hi = s["win_rate_wilson"]
    L += [f"trades {s['n']} of {s['n_signals']} signals (skipped {s['skipped']}); days {s['n_days']}; mean dist {s['mean_dist_bp']:.0f}bp",
          f"win rate {s['win_rate']:.3f} [{lo:.3f},{hi:.3f}] vs break-even {s['breakeven_win']:.3f} | tp {s['tp_share']:.3f} sl {s['sl_share']:.3f} "
          f"timeout {s['timeout_share']:.3f} | decided-win {s['win_rate_decided']:.3f}",
          f"avg win {s['avg_win_bp']:.1f}bp avg loss {s['avg_loss_bp']:.1f}bp PF {s['profit_factor']:.3f} W/L {s['win_loss_ratio']:.3f}",
          f"net/trade {s['net_per_trade_bp']:+.2f}bp (gross {s['gross_per_trade_bp']:+.2f}) net/day {s['net_per_day_bp']:+.1f}bp "
          f"({s['trades_per_day']:.1f} trades/day) t(day-clustered) {s['t_day_clustered']:.2f}",
          f"coins positive {s['coins_positive']} (median {s['coin_net_median_bp']:+.1f}bp) | horizon direction accuracy {s['horizon_dir_acc']:.3f}",
          f"ambiguous exits (both barriers in one bar, unresolved -> {c['ambiguous']}): {s['ambiguous_exits']} = {s['ambiguous_share']:.2%} of trades; "
          f"both-touched exits {s['both_touched_exits']}, resolved by finer bars {s['resolved_by_fine']}"]
    if res.null:
        n = res.null
        L.append(f"random-direction null ({n['reps']} draws, same samples): {n['null_mean_bp']:+.2f}bp sd {n['null_sd_bp']:.2f} | "
                 f"model {s['net_per_trade_bp']:+.2f}bp z {n['z']:.2f} p(null>=model) {n['null_p']:.3f}")
    L += ["by side:", res.by_side.round(4).to_string(index=False)]
    if res.by_confidence is not None:
        cs = res.confidence_stats
        L += ["by confidence decile (1 = least confident):", res.by_confidence.round(4).to_string(index=False),
              f"hit~confidence spearman {cs['hit_vs_confidence_spearman']:+.3f}; decile slope {cs['decile_hit_slope_spearman']:+.2f}; "
              f"top-bottom hit {cs['top_minus_bottom_hit']:+.3f}; net~confidence spearman {cs['net_vs_confidence_spearman']:+.3f}"]
    L += ["per coin (net bp):", res.by_coin.round(4).to_string(index=False)]
    return "\n".join(L)


def _read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Path-resolved 1:1 bracket backtest of a PredictionTable / signals file.")
    ap.add_argument("--table", required=True, help="PredictionTable (.csv/.csv.gz with .meta.json) or a csv/parquet with asset,timestamp,pred")
    ap.add_argument("--head", default="close", help="head whose out_<head>_p_up / out_<head> gives the direction")
    ap.add_argument("--split-role", default=None, help="keep only this role of a PredictionTable (e.g. test)")
    ap.add_argument("--confidence-col", default=None, help="out_* column to use as confidence instead of |p_up-0.5| / |pred|")
    ap.add_argument("--reg-scale", type=float, default=1.0, help="reg_target_scale to decode out_high/out_low (pred_range sizing)")
    ap.add_argument("--candles", default=None, help="candles file (asset,timestamp,open,high,low,close) at --timeframe")
    ap.add_argument("--candles-dir", default=None, help="dir of ohlc_<SYMBOL>_<timeframe>.parquet (tools/fetch_crypto_dataset.py)")
    ap.add_argument("--fine", default=None, help="finer candles file (asset,timestamp,OHLC) for touch order")
    ap.add_argument("--fine-dir", default=None, help="dir of ohlc_<SYMBOL>_<fine-timeframe>.parquet")
    ap.add_argument("--fine-timeframe", default="15m")
    ap.add_argument("--symbol-suffix", default="", help="stripped from file SYMBOLs to get asset names (e.g. USDT)")
    ap.add_argument("--timeframe", required=True, help="model timeframe, e.g. 1h")
    ap.add_argument("--hold-bars", type=int, default=HOLD_BARS)
    ap.add_argument("--sizing", choices=SIZINGS, default="fixed")
    ap.add_argument("--size-value", type=float, default=0.01, help="fixed: fraction; atr / pred_range: the multiplier k")
    ap.add_argument("--entry", choices=ENTRIES, default="close")
    ap.add_argument("--cost-rt", type=float, default=COST_RT)
    ap.add_argument("--stop-slip", type=float, default=STOP_SLIP)
    ap.add_argument("--ambiguous", choices=AMBIGUOUS, default="stop_first")
    ap.add_argument("--min-dist", type=float, default=None)
    ap.add_argument("--max-dist", type=float, default=None)
    ap.add_argument("--null-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None, help="write the summary + config as json")
    ap.add_argument("--out-trades", default=None, help="write the trades as csv(.gz)")
    a = ap.parse_args(argv)

    if a.table.endswith(".csv") or a.table.endswith(".csv.gz"):
        from stats.prediction_table import read_table
        try:
            raw = read_table(a.table)
        except (OSError, ValueError):                       # a plain csv without the sidecar / not a PredictionTable
            raw = pd.read_csv(a.table)
    else:
        raw = _read_any(a.table)
    assets = sorted(raw["asset"].astype(str).unique())
    if a.candles_dir:
        candles = load_candles_dir(a.candles_dir, a.timeframe, assets, a.symbol_suffix)
    elif a.candles:
        candles = _read_any(a.candles)
    elif a.fine_dir or a.fine:
        candles = None
    else:
        ap.error("give --candles, --candles-dir, or a fine source to resample from")
    fine = None
    if a.fine_dir:
        fine = load_candles_dir(a.fine_dir, a.fine_timeframe, assets, a.symbol_suffix)
    elif a.fine:
        fine = _read_any(a.fine)
    if candles is None:
        candles = resample_ohlc(fine, a.timeframe)
    if "pred" in raw.columns:
        sig = raw
    else:
        sig = signals_from_table(raw, head=a.head, candles=candles if a.sizing == "pred_range" else None, reg_scale=a.reg_scale,
                                 split_role=a.split_role, confidence=a.confidence_col)
    res = run_backtest(sig, candles, timeframe=a.timeframe, hold_bars=a.hold_bars, sizing=a.sizing, size_value=a.size_value, entry=a.entry,
                       fine_candles=fine, cost_rt=a.cost_rt, stop_slip=a.stop_slip, ambiguous=a.ambiguous, min_dist=a.min_dist,
                       max_dist=a.max_dist, null_reps=a.null_reps, seed=a.seed)
    print(format_report(res))
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump({"config": res.config, "summary": res.summary, "null": res.null, "confidence": res.confidence_stats}, f, indent=2,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if a.out_trades:
        res.trades.to_csv(a.out_trades, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

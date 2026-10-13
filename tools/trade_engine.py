"""
PURPOSE: Shared trade engine for the strategy-discovery studies: loads 1h spot OHLCV and resamples it to 4h / 1d, computes the
         indicators (EMA, RSI, MACD, Bollinger, ADX/DI, ATR, Donchian), builds named signals from triggers plus optional gates, and
         simulates single-position trades with an ATR stop, an ATR target and a time exit. Outcomes are in R (initial stop distance),
         net of the round-trip cost. Lifted from research/indicator_strategy/engine.py so that studies do not import each other.
TAGS:    trade engine, indicator screen, RSI, MACD, Bollinger breakout, ADX trend, Donchian, EMA200 pullback, ATR stop, R multiple,
         timeframe resample, trade simulator, gate, regime, cost in R, shared tool
PITFALLS: Indicators at bar t use closes up to t; signals at bar t fill at the OPEN of bar t+1. Stop is checked before target
         inside a bar (conservative); a gap through the stop fills at the open. One position per coin at a time. Wilder smoothing
         is ewm(alpha=1/n), an approximation of the textbook recursion. The indicator_strategy copy is left unchanged on purpose.
"""
import numpy as np
import pandas as pd

COST = 0.0012            # round trip per trade (as in the earlier studies)
ATR_STOP = 2.0           # stop distance in ATR14 units (card default)
ATR_TGT = 3.0            # target distance in ATR14 units
MAX_BARS = {"4h": 30, "1d": 30}   # time exit in bars


def load(path, tf):
    """1h OHLCV -> tf bars (UTC, left-labelled). tf in {'4h', '1d'}."""
    df = pd.read_parquet(path).sort_index()
    rule = {"4h": "4h", "1d": "1D"}[tf]
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["close"])
    return out


def indicators(df):
    """Adds the indicators used by the signals. All are causal (value at t uses bars <= t)."""
    c, h, l = df.close, df.high, df.low
    d = df.copy()
    d["ema20"] = c.ewm(span=20, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    d["bb_up"], d["bb_lo"] = mid + 2 * sd, mid - 2 * sd
    prev_c = c.shift()
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d["natr"] = d["atr"] / c * 100.0
    up_move, dn_move = h.diff(), -l.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_s = d["atr"]
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_s
    d["plus_di"], d["minus_di"] = plus_di, minus_di
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    d["don_hi"] = h.rolling(20).max().shift(1)          # prior 20-bar channel (no look-ahead)
    d["don_lo"] = l.rolling(20).min().shift(1)
    return d


def triggers(d):
    """Named trigger events at bar t (crossings use t and t-1). Returns dict name -> (long, short) bool arrays."""
    p = lambda s: s.shift(1)
    out = {}
    out["rsi_mr"] = ((p(d.rsi) < 30) & (d.rsi >= 30), (p(d.rsi) > 70) & (d.rsi <= 70))
    out["macd_cross"] = ((p(d.macd_hist) <= 0) & (d.macd_hist > 0), (p(d.macd_hist) >= 0) & (d.macd_hist < 0))
    out["bb_break"] = ((p(d.close) <= p(d.bb_up)) & (d.close > d.bb_up), (p(d.close) >= p(d.bb_lo)) & (d.close < d.bb_lo))
    out["adx_trend"] = ((p(d.adx) < 25) & (d.adx >= 25) & (d.plus_di > d.minus_di),
                        (p(d.adx) < 25) & (d.adx >= 25) & (d.minus_di > d.plus_di))
    out["donchian"] = ((d.close > d.don_hi), (d.close < d.don_lo))
    out["ema_pullback"] = ((d.ema20 > d.ema50) & (d.low <= d.ema20) & (d.close > d.ema20),
                           (d.ema20 < d.ema50) & (d.high >= d.ema20) & (d.close < d.ema20))
    return {k: (np.asarray(a, bool), np.asarray(b, bool)) for k, (a, b) in out.items()}


def gates(d):
    """Named gates (boolean arrays for a long direction; the short direction uses the mirror where it applies)."""
    return {
        "adx25": (d.adx >= 25).values,
        "adx20": (d.adx >= 20).values,
        "ema200_up": (d.close > d.ema200).values,
        "ema200_dn": (d.close < d.ema200).values,
        "not_adx40": (d.adx < 40).values,
    }


def build_signal(d, trigger, gate_names=()):
    """trigger name + gate names -> (long, short) arrays. Gates named *_up/*_dn apply to one side only."""
    tr = triggers(d)[trigger]
    g = gates(d)
    long, short = tr[0].copy(), tr[1].copy()
    for name in gate_names:
        if name == "long_only":                      # removes the short side entirely
            short[:] = False
        elif name.endswith("_up"):
            long &= g[name]
        elif name.endswith("_dn"):
            short &= g[name]
        else:
            long &= g[name]
            short &= g[name]
    return long, short


def simulate(d, long, short, tf, atr_stop=ATR_STOP, atr_tgt=ATR_TGT, cost=COST, max_bars=None):
    """One position per coin. Entry at the open after a signal. Returns a DataFrame of trades with R outcomes."""
    O, H, L, C = d.open.values, d.high.values, d.low.values, d.close.values
    atr = d.atr.values
    sig = np.where(long, 1, np.where(short, -1, 0))
    max_bars = MAX_BARS[tf] if max_bars is None else max_bars   # default keeps the old behaviour
    n = len(C)
    trades, pos = [], None
    for k in range(1, n):
        if pos is None and sig[k - 1] != 0 and np.isfinite(atr[k - 1]) and k < n:
            dd = int(sig[k - 1])
            e = O[k]
            risk = atr_stop * atr[k - 1]
            pos = dict(d=dd, entry=e, entry_bar=k, stop=e - dd * risk, tgt=e + dd * atr_tgt * atr[k - 1], risk=risk,
                       feat_bar=k - 1)
        if pos is None:
            continue
        dd, stop, tgt = pos["d"], pos["stop"], pos["tgt"]
        exit_px, reason = None, None
        if dd == 1:
            if L[k] <= stop:
                exit_px, reason = min(O[k], stop), "stop"
            elif H[k] >= tgt:
                exit_px, reason = max(O[k], tgt), "target"
        else:
            if H[k] >= stop:
                exit_px, reason = max(O[k], stop), "stop"
            elif L[k] <= tgt:
                exit_px, reason = min(O[k], tgt), "target"
        if exit_px is None and k - pos["entry_bar"] >= max_bars:
            exit_px, reason = C[k], "time"
        if exit_px is None and k == n - 1:
            exit_px, reason = C[k], "eod"
        if exit_px is not None:
            s = pos["risk"] / pos["entry"]                       # stop distance as a fraction of price
            gross_r = dd * (exit_px / pos["entry"] - 1.0) / s
            fb = pos["feat_bar"]
            trades.append(dict(entry_time=d.index[pos["entry_bar"]], exit_time=d.index[k], side=dd,
                               gross_r=gross_r, net_r=gross_r - cost / s, reason=reason, bars=k - pos["entry_bar"],
                               risk_pct=s * 100.0, adx=d.adx.values[fb], natr=d.natr.values[fb],
                               rsi=d.rsi.values[fb], trend_up=bool(d.close.values[fb] > d.ema200.values[fb])
                               if np.isfinite(d.ema200.values[fb]) else np.nan))
            pos = None
    return pd.DataFrame(trades)


def metrics(t):
    """Summary of a trade DataFrame in R units."""
    if t is None or len(t) == 0:
        return dict(n=0)
    w, l_ = t.net_r[t.net_r > 0].sum(), -t.net_r[t.net_r <= 0].sum()
    return dict(n=len(t), win=float((t.net_r > 0).mean()), mean_net_r=float(t.net_r.mean()),
                mean_gross_r=float(t.gross_r.mean()), pf=float(w / l_) if l_ > 0 else np.inf,
                median_risk_pct=float(t.risk_pct.median()), cost_r=float((COST / (t.risk_pct / 100)).median()))


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--coins", nargs="+", default=["BTCUSDT"])
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--trigger", default="ema_pullback")
    ap.add_argument("--gates", nargs="*", default=[])
    a = ap.parse_args()
    for coin in a.coins:
        base = load(os.path.join(a.data, f"ohlc_{coin}_1h.parquet"), a.tf)
        d = indicators(base)
        lg, sh = build_signal(d, a.trigger, a.gates)
        t = simulate(d, lg, sh, a.tf)
        print(coin, a.trigger, a.gates, metrics(t))

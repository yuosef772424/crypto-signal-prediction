"""
PURPOSE:  Gate-style derived technical STATES (not raw indicators) computed from our own OHLCV, causally, for use as model
          inputs (phase G2, card research/gate_features/cards/E-gate-001.md). Per timeframe (15m/1h/4h/1d): ma_alignment
          A/B (SMA and EMA), rsi_zone, macd cross + histogram sign, KDJ zone + cross, SAR side, BOLL %B zone + bandwidth
          percentile bucket, ADX bucket + DI dominance, WR zone, CCI zone, and bullish-minus-bearish aggregates; across
          timeframes one aggregate and a gate-like `signal` = sign of it. Also the matching RAW indicator values (the "old
          set" of the F-0027 R2 comparison). Every definition is versioned `gate-style-v1, unverified` until phase G
          recovers gate's exact rules (then a new card changes one definition at a time).
TAGS:     gate, technical states, ma_alignment, rsi_zone, macd_signal, KDJ, SAR, BOLL, ADX, WR, CCI, multi-timeframe signal,
          derived features, categorical states, closed bars, no look-ahead, G2, gate-style-v1
PITFALLS: Missing or insufficient history is NaN, NEVER 0 (the gate bugs): a state is NaN until every input it needs exists,
          an aggregate is NaN if ANY member is NaN, flat/zero-range inputs give NaN. States are in natural coordinates
          (rsi_zone +1 = overbought); only the aggregates orient them: trend states keep their sign, oscillator zones
          (rsi, kdj, boll, wr, cci) are contrarian (oversold = bullish) - a v1 candidate choice. A higher-timeframe bar is used
          only once closed (open + length <= decision time) and not if it is older than one bar length (gap): an incomplete
          5m->tf bar is dropped, not filled. SAR is path dependent, so its first 30 bars are NaN. Decision row = the 1h bar
          OPEN time t; features are known at t + 1h and the entry is the open of the next bar.

Usage:
    import gate_features as gf
    dt = gf.decision_table(df5m)              # df5m: UTC DatetimeIndex (bar open), open/high/low/close/volume
    states_1h = dt.filter(like="60__")
    raw, st = gf.tf_features(gf.resample_ohlcv(df5m, 240))
"""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

VERSION = "gate-style-v1, unverified"
TIMEFRAMES = (15, 60, 240, 1440)                      # minutes
SAR_WARMUP = 30

TREND_STATES = ("sma_A", "sma_B", "ema_A", "ema_B", "macd_cross", "macd_hist_sign", "kdj_cross", "sar_side")
OSC_STATES = ("rsi_zone", "kdj_zone", "boll_zone", "wr_zone", "cci_zone")      # contrarian in the aggregates
STATE_COLUMNS = ("sma_A", "sma_B", "ema_A", "ema_B", "rsi_zone", "macd_cross", "macd_hist_sign", "kdj_zone", "kdj_cross",
                 "sar_side", "boll_zone", "boll_bw_bucket", "adx_bucket", "di_dom", "wr_zone", "cci_zone",
                 "agg_trend", "agg_osc", "agg_all")
RAW_COLUMNS = ("ma_r_sma_7_30", "ma_r_sma_30_120", "ma_r_sma_120_200", "ma_r_ema_7_30", "ma_r_ema_30_120", "ma_r_ema_120_200",
               "rsi14", "macd_line", "macd_signal", "macd_hist", "kdj_k", "kdj_d", "kdj_j", "sar_dist", "boll_pctb",
               "boll_bw", "boll_bw_pct", "adx14", "pdi14", "mdi14", "wr14", "cci20")
NATR_COLUMN = "natr14"


def resample_ohlcv(df, minutes, base_minutes=5):
    """UTC bars of `minutes`, label = bar open; a bar needs ALL its base bars (incomplete bars are dropped, not filled)."""
    need = minutes // base_minutes
    key = df.index.floor(f"{minutes}min")
    g = df.groupby(key)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    cnt = g["close"].count()
    out = out[cnt == need]
    out.index.name = df.index.name
    return out


# ------------------------------------------------------------------ building blocks
def _wilder(x, n):
    """Wilder smoothing: seed = mean of the first n valid values, then y = y_prev*(n-1)/n + x/n. NaN before the seed."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = np.flatnonzero(~np.isnan(x))
    if len(ok) < n:
        return out
    s = ok[0]
    if np.isnan(x[s:]).any():                              # never fill: restart is not supported, stay NaN
        end = s + np.flatnonzero(np.isnan(x[s:]))[0]
        x = x[:end]
    if len(x) - s < n:
        return out
    seed = x[s:s + n].mean()
    out[s + n - 1] = seed
    rest = x[s + n:]
    if len(rest):
        y, _ = lfilter([1.0 / n], [1.0, -(1.0 - 1.0 / n)], rest, zi=[(1.0 - 1.0 / n) * seed])
        out[s + n:s + n + len(rest)] = y
    return out


def _cross_state(diff, within=3):
    """+1 golden / -1 dead cross (diff crosses 0, <=0 -> >0 and the reverse) within the last `within` bars (most recent wins),
    0 none; NaN unless diff is known on all `within`+1 bars."""
    d = pd.Series(diff)
    prev = d.shift(1)
    up = ((d > 0) & (prev <= 0)).astype(float)
    dn = ((d < 0) & (prev >= 0)).astype(float)
    n = len(d)
    out = np.zeros(n)
    for lag in range(within - 1, -1, -1):                  # oldest first, newest overrides
        u, w = up.shift(lag).to_numpy(), dn.shift(lag).to_numpy()
        out = np.where(u == 1, 1.0, np.where(w == 1, -1.0, out))
    known = d.notna() & prev.notna()
    for lag in range(1, within):
        known &= d.shift(lag).notna() & d.shift(lag + 1).notna()
    return np.where(known.to_numpy(), out, np.nan)


def _sign(x):
    x = np.asarray(x, float)
    return np.where(np.isnan(x), np.nan, np.sign(x))


def _ma_alignment(a7, a30, a120, a200=None):
    """+1 if a7 > a30 > a120 (> a200 for B), -1 for the exact reverse, else 0; NaN if any input is NaN."""
    cols = [a7, a30, a120] + ([a200] if a200 is not None else [])
    bull = np.ones(len(a7), bool)
    bear = np.ones(len(a7), bool)
    for hi, lo in zip(cols[:-1], cols[1:]):
        bull &= (hi > lo)
        bear &= (hi < lo)
    out = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
    nan = np.zeros(len(a7), bool)
    for c in cols:
        nan |= np.isnan(c)
    return np.where(nan, np.nan, out)


def _roll_rank(x, win):
    """Share of the last `win` values (incl. the current one) that are <= the current; NaN if any is NaN or fewer than win."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < win:
        return out
    v = np.lib.stride_tricks.sliding_window_view(x, win)
    for s in range(0, len(v), 50000):
        c = v[s:s + 50000]
        out[win - 1 + s:win - 1 + s + len(c)] = np.where(np.isnan(c).any(axis=1), np.nan, (c <= c[:, -1:]).mean(axis=1))
    return out


def _roll_mad(x, win):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < win:
        return out
    v = np.lib.stride_tricks.sliding_window_view(x, win)
    for s in range(0, len(v), 50000):
        c = v[s:s + 50000]
        out[win - 1 + s:win - 1 + s + len(c)] = np.abs(c - c.mean(axis=1, keepdims=True)).mean(axis=1)
    return out


def _kdj(h, l, c, n=9):
    hh = pd.Series(h).rolling(n).max().to_numpy()
    ll = pd.Series(l).rolling(n).min().to_numpy()
    rng = hh - ll
    rsv = np.where(rng > 0, (c - ll) / np.where(rng > 0, rng, 1) * 100, np.nan)
    k = np.full(len(c), np.nan)
    d = np.full(len(c), np.nan)
    pk = pd_ = np.nan
    for i in range(len(c)):
        if np.isnan(rsv[i]):
            pk = pd_ = np.nan
            continue
        if np.isnan(pk):
            pk = pd_ = 50.0
        pk = pk * 2 / 3 + rsv[i] / 3
        pd_ = pd_ * 2 / 3 + pk / 3
        k[i], d[i] = pk, pd_
    return k, d, 3 * k - 2 * d


def _sar(h, l, c, af0=0.02, afmax=0.2):
    n = len(c)
    sar = np.full(n, np.nan)
    side = np.full(n, np.nan)
    if n < 3:
        return sar, side
    long = c[1] >= c[0]
    prev = l[0] if long else h[0]
    ep = h[0] if long else l[0]
    af = af0
    for i in range(1, n):
        s = prev + af * (ep - prev)
        if long:
            s = min(s, l[i - 1], l[i - 2] if i >= 2 else l[i - 1])
            if l[i] < s:
                long, s, ep, af = False, ep, l[i], af0
            elif h[i] > ep:
                ep, af = h[i], min(af + af0, afmax)
        else:
            s = max(s, h[i - 1], h[i - 2] if i >= 2 else h[i - 1])
            if h[i] > s:
                long, s, ep, af = True, ep, h[i], af0
            elif l[i] < ep:
                ep, af = l[i], min(af + af0, afmax)
        sar[i] = s
        side[i] = 1.0 if long else -1.0
        prev = s
    sar[:SAR_WARMUP] = np.nan
    side[:SAR_WARMUP] = np.nan
    return sar, side


def _zone(x, lo, hi):
    """-1 below lo, +1 above hi, else 0 (inclusive thresholds: x <= lo -> -1, x >= hi -> +1 are set by the caller)."""
    x = np.asarray(x, float)
    return np.where(np.isnan(x), np.nan, np.where(x < lo, -1.0, np.where(x > hi, 1.0, 0.0)))


def _agg(st, names, sign):
    """Sum of oriented members (bullish +1 minus bearish -1); NaN if any member is NaN."""
    tot = np.zeros(len(st))
    for nme in names:
        tot = tot + sign * st[nme].to_numpy()
    return tot


# ------------------------------------------------------------------ one timeframe
def tf_features(ohlc):
    """(raw, states) DataFrames aligned with `ohlc` (index = bar open). Row i uses bars <= i only."""
    o, h, l, c = (ohlc[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    idx = ohlc.index
    cs = pd.Series(c, index=idx)
    raw, st = {}, {}
    sma = {n: cs.rolling(n).mean().to_numpy() for n in (7, 30, 120, 200)}
    ema = {n: cs.ewm(span=n, adjust=False, min_periods=n).mean().to_numpy() for n in (7, 30, 120, 200)}
    for nm, m in (("sma", sma), ("ema", ema)):
        raw[f"ma_r_{nm}_7_30"] = m[7] / m[30] - 1
        raw[f"ma_r_{nm}_30_120"] = m[30] / m[120] - 1
        raw[f"ma_r_{nm}_120_200"] = m[120] / m[200] - 1
        st[f"{nm}_A"] = _ma_alignment(m[7], m[30], m[120])
        st[f"{nm}_B"] = _ma_alignment(m[7], m[30], m[120], m[200])
    # RSI 14 (Wilder)
    dc = np.r_[np.nan, np.diff(c)]
    ag, al = _wilder(np.clip(dc, 0, None) + 0 * dc, 14), _wilder(np.clip(-dc, 0, None) + 0 * dc, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = np.where(al == 0, np.where(ag == 0, np.nan, 100.0), 100 - 100 / (1 + ag / np.where(al == 0, 1, al)))
    rsi = np.where(np.isnan(ag) | np.isnan(al), np.nan, rsi)
    raw["rsi14"] = rsi
    st["rsi_zone"] = np.where(np.isnan(rsi), np.nan, np.where(rsi < 30, -1.0, np.where(rsi > 70, 1.0, 0.0)))
    # MACD (12, 26, 9)
    e12 = cs.ewm(span=12, adjust=False, min_periods=12).mean()
    e26 = cs.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = e12 - e26
    sig = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = (macd - sig).to_numpy()
    raw["macd_line"], raw["macd_signal"], raw["macd_hist"] = (macd / cs).to_numpy(), (sig / cs).to_numpy(), hist / c
    st["macd_cross"] = _cross_state(hist)
    st["macd_hist_sign"] = _sign(hist)
    # KDJ (9, 3, 3)
    k, d, j = _kdj(h, l, c)
    raw["kdj_k"], raw["kdj_d"], raw["kdj_j"] = k, d, j
    st["kdj_zone"] = np.where(np.isnan(k), np.nan, np.where(k < 20, -1.0, np.where(k > 80, 1.0, 0.0)))
    st["kdj_cross"] = _cross_state(k - d)
    # SAR
    sar, side = _sar(h, l, c)
    raw["sar_dist"] = (c - sar) / c
    st["sar_side"] = side
    # BOLL (20, 2)
    mid = cs.rolling(20).mean().to_numpy()
    sd = cs.rolling(20).std(ddof=0).to_numpy()
    up, lo = mid + 2 * sd, mid - 2 * sd
    width = up - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        pctb = np.where(width > 0, (c - lo) / np.where(width > 0, width, 1), np.nan)
        bw = np.where(width > 0, width / mid, np.nan)
    raw["boll_pctb"], raw["boll_bw"] = pctb, bw
    bwp = _roll_rank(bw, 120)
    raw["boll_bw_pct"] = bwp
    st["boll_zone"] = _zone(pctb, 0.0, 1.0)
    st["boll_bw_bucket"] = np.where(np.isnan(bwp), np.nan, np.where(bwp < 0.2, -1.0, np.where(bwp > 0.8, 1.0, 0.0)))
    # ADX / DI (14)
    up_m = np.r_[np.nan, h[1:] - h[:-1]]
    dn_m = np.r_[np.nan, l[:-1] - l[1:]]
    pdm = np.where(up_m > dn_m, np.clip(up_m, 0, None), 0.0)
    mdm = np.where(dn_m > up_m, np.clip(dn_m, 0, None), 0.0)
    pdm, mdm = np.where(np.isnan(up_m), np.nan, pdm), np.where(np.isnan(up_m), np.nan, mdm)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = np.nan
    atr = _wilder(tr, 14)
    raw[NATR_COLUMN] = atr / c * 100
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * _wilder(pdm, 14) / atr
        mdi = 100 * _wilder(mdm, 14) / atr
        dx = np.where(pdi + mdi > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), np.nan)
    adx = _wilder(dx, 14)
    raw["adx14"], raw["pdi14"], raw["mdi14"] = adx, pdi, mdi
    st["adx_bucket"] = np.where(np.isnan(adx), np.nan, np.where(adx < 20, 0.0, np.where(adx <= 40, 1.0, 2.0)))
    st["di_dom"] = np.where(np.isnan(pdi) | np.isnan(mdi), np.nan, np.sign(pdi - mdi))
    # WR (14), CCI (20)
    hh = pd.Series(h).rolling(14).max().to_numpy()
    ll = pd.Series(l).rolling(14).min().to_numpy()
    rng = hh - ll
    wr = np.where(rng > 0, -100 * (hh - c) / np.where(rng > 0, rng, 1), np.nan)
    raw["wr14"] = wr
    st["wr_zone"] = np.where(np.isnan(wr), np.nan, np.where(wr <= -80, -1.0, np.where(wr >= -20, 1.0, 0.0)))
    tp = (h + l + c) / 3
    mad = _roll_mad(tp, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        cci = np.where(mad > 0, (tp - pd.Series(tp).rolling(20).mean().to_numpy()) / (0.015 * np.where(mad > 0, mad, 1)), np.nan)
    raw["cci20"] = cci
    st["cci_zone"] = _zone(cci, -100.0, 100.0)
    # aggregates (bullish count - bearish count)
    sdf = pd.DataFrame(st)
    adx_trend = np.where(sdf["adx_bucket"] >= 1, sdf["di_dom"], 0.0)            # DI dominance counts when the trend has strength
    adx_trend = np.where(sdf["adx_bucket"].isna() | sdf["di_dom"].isna(), np.nan, adx_trend)
    st["agg_trend"] = _agg(sdf, TREND_STATES, 1.0) + adx_trend
    st["agg_osc"] = _agg(sdf, OSC_STATES, -1.0)
    st["agg_all"] = st["agg_trend"] + st["agg_osc"]
    raw_df = pd.DataFrame(raw, index=idx)[list(RAW_COLUMNS) + [NATR_COLUMN]]
    st_df = pd.DataFrame(st, index=idx)[list(STATE_COLUMNS)]
    return raw_df, st_df


# ------------------------------------------------------------------ decision table
def decision_table(df5, timeframes=TIMEFRAMES, base_minutes=5):
    """One row per closed 1h bar (index = bar OPEN time t; decision at t + 1h, entry at the next open). Columns
    `<tf>__<name>` (raw and states of the last CLOSED tf bar as of t + 1h), `overall_agg` (sum of agg_all over the
    timeframes, NaN if any is NaN) and `signal` = sign(overall_agg)."""
    hours = resample_ohlcv(df5, 60, base_minutes)
    left = pd.DataFrame({"decision": hours.index + pd.Timedelta(hours=1)}, index=hours.index)
    parts = []
    for tf in timeframes:
        bars = resample_ohlcv(df5, tf, base_minutes)
        raw, st = tf_features(bars)
        feat = pd.concat([raw, st], axis=1)
        feat.columns = [f"{tf}__{c}" for c in feat.columns]
        feat["__avail"] = bars.index + pd.Timedelta(minutes=tf)
        m = pd.merge_asof(left.reset_index(drop=True), feat.sort_values("__avail"), left_on="decision", right_on="__avail",
                          direction="backward", tolerance=pd.Timedelta(minutes=tf) - pd.Timedelta(1, "ns"))
        m = m.drop(columns=["decision", "__avail"])
        m.index = left.index
        parts.append(m)
    out = pd.concat(parts, axis=1)
    agg = sum(out[f"{tf}__agg_all"].to_numpy() for tf in timeframes)
    out["overall_agg"] = agg
    out["signal"] = _sign(agg)
    return out

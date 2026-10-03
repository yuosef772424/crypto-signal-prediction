"""
PURPOSE:  H17: 10 classic chart setups x 2 sides x 5 HTF filters x TP 1-3R (600 rules) on 15m/1h with exits resolved
          on the 5m path.
TAGS:     H17, chart patterns, engulfing, pin bar, sweep, Donchian, inside bar, London breakout, HTF filter, R
          multiple, numba
PITFALLS: Zero rules pass (best t=1.79 of 600); cost in R is 0.21-0.38R on 15m so tiny timeframes cannot win; heavy
          runtime; needs ohlc_<SYMBOL>_5m.parquet in /home/user/research (built from github Speirsy11/crypto-dataset
          via LFS; not in repo)

H17: chart-reading setups on 15m/1h with HTF filters (Addendum F). Exits resolved on the 5m path."""
import numpy as np, pandas as pd, sys
from numba import njit
D = '/home/user/research'
TAKER, MAKER = 0.0007, 0.0002
SEGS = {'DISC': ('2017-08-01', '2021-12-31 23:59'), 'VAL': ('2022-01-01', '2023-12-31 23:59'), 'HOLD': ('2024-01-01', '2026-09-30')}

@njit
def outcomes(h5, l5, o5, c5, ent_idx, atr, ks, tmax):
    """R-multiple (net) for long/short with stop 1*ATR and target k*ATR; entry at o5[ent_idx]."""
    n = len(ent_idx); nk = len(ks)
    R = np.full((n, 2, nk), np.nan)
    for j in range(n):
        e = ent_idx[j]; a = atr[j]
        if e < 0 or not (a > 0): continue
        p0 = o5[e]
        for d in range(2):
            sgn = 1.0 if d == 0 else -1.0
            sl = p0 - sgn * a
            for q in range(nk):
                tp = p0 + sgn * ks[q] * a
                res = np.nan
                end = min(e + tmax, len(h5) - 1)
                for i in range(e, end + 1):
                    hit_sl = (l5[i] <= sl) if sgn > 0 else (h5[i] >= sl)
                    hit_tp = (h5[i] >= tp * (1 + 0.0002)) if sgn > 0 else (l5[i] <= tp * (1 - 0.0002))
                    if hit_sl:
                        res = -1.0 - (TAKER + TAKER) * p0 / a; break
                    if hit_tp and i > e:
                        res = ks[q] - (TAKER + MAKER) * p0 / a; break
                if np.isnan(res):
                    res = sgn * (c5[end] - p0) / a - (TAKER + TAKER) * p0 / a
                R[j, d, q] = res
    return R

def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def atrf(b, n=14):
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def build(sym, tf):
    m5 = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    rule = {'15m': '15min', '1h': '1h'}[tf]
    b = m5.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    A = atrf(b); o, h, l, c, v = b.open, b.high, b.low, b.close, b.volume
    body = (c - o).abs(); rng = h - l
    S = {}
    S['P1_engulf'] = ((c > o) & (c.shift() < o.shift()) & (c >= o.shift()) & (o <= c.shift()),
                      (c < o) & (c.shift() > o.shift()) & (c <= o.shift()) & (o >= c.shift()))
    lw = np.minimum(o, c) - l; uw = h - np.maximum(o, c)
    S['P2_pinbar'] = ((lw >= 2 * body) & (lw >= 0.6 * rng), (uw >= 2 * body) & (uw >= 0.6 * rng))
    lo20 = l.shift().rolling(20).min(); hi20 = h.shift().rolling(20).max()
    S['P3_sweep'] = ((l < lo20) & (c > lo20), (h > hi20) & (c < hi20))
    S['P4_donchian'] = (c > hi20, c < lo20)
    comp = (A / atrf(b, 100)).rolling(500).rank(pct=True) < 0.2
    hi10 = h.shift().rolling(10).max(); lo10 = l.shift().rolling(10).min()
    S['P5_compress'] = (comp & (c > hi10), comp & (c < lo10))
    inside_prev = (h.shift() <= h.shift(2)) & (l.shift() >= l.shift(2))
    S['P6_insidebo'] = (inside_prev & (c > h.shift(2)), inside_prev & (c < l.shift(2)))
    e20, e50 = ema(c, 20), ema(c, 50)
    S['P7_pullback'] = ((e20 > e50) & (l <= e20) & (c >= e20), (e20 < e50) & (h >= e20) & (c <= e20))
    vs = v > 3 * v.shift().rolling(20).mean(); big = rng > 2 * A
    S['P8_capit'] = (vs & big & (c > (h + l) / 2) & (c < o.shift(3)), vs & big & (c < (h + l) / 2) & (c > o.shift(3)))
    d3 = (c < c.shift()) & (c.shift() < c.shift(2)) & (c.shift(2) < c.shift(3)) & ((c.shift(3) - c) > 2 * A)
    u3 = (c > c.shift()) & (c.shift() > c.shift(2)) & (c.shift(2) > c.shift(3)) & ((c - c.shift(3)) > 2 * A)
    S['P9_exhaust'] = (d3, u3)
    day = b.index.floor('1D'); hr = b.index.hour
    asia = b[(hr < 7)].groupby(day[hr < 7]).agg(ah=('high', 'max'), al=('low', 'min'))
    ah = pd.Series(day.map(asia.ah), index=b.index); al = pd.Series(day.map(asia.al), index=b.index)
    win = (hr >= 7) & (hr < 10)
    upx = win & (c > ah); dnx = win & (c < al)
    first_up = upx & (upx.groupby(day).cumsum() == 1); first_dn = dnx & (dnx.groupby(day).cumsum() == 1)
    S['P10_london'] = (first_up, first_dn)
    # HTF (completed bars only): value known at signal close t = last completed 4h / 1d bar
    h4 = m5.resample('4h').agg({'close': 'last'}).dropna(); h4['ema'] = ema(h4.close, 50)
    h4.index = h4.index + pd.Timedelta('4h')                      # available at bar end
    d1 = m5.resample('1D').agg({'close': 'last'}).dropna(); d1['ema'] = ema(d1.close, 20)
    d1['er'] = (d1.close - d1.close.shift(10)).abs() / d1.close.diff().abs().rolling(10).sum()
    d1.index = d1.index + pd.Timedelta('1D')
    tclose = b.index + pd.Timedelta(rule)                         # signal known at bar close
    H4 = h4.reindex(tclose, method='ffill'); D1 = d1.reindex(tclose, method='ffill')
    up4 = (H4.close > H4.ema).values; upd = (D1.close > D1.ema).values; rngd = (D1.er < 0.3).values
    F = {'F0': (np.ones(len(b), bool), np.ones(len(b), bool)), 'F1': (up4, ~up4), 'F2': (~up4, up4),
         'F3': (upd, ~upd), 'F4': (rngd, rngd)}
    # entry = first 5m bar of the next signal bar
    ent_t = tclose
    ent_idx = m5.index.get_indexer(ent_t, method='bfill')
    ent_idx[ent_idx < 0] = -1
    tmax = {'15m': 288, '1h': 864}[tf]                             # 24h / 72h of 5m bars
    R = outcomes(m5.high.values, m5.low.values, m5.open.values, m5.close.values, ent_idx.astype(np.int64),
                 A.values, np.array([1.0, 2.0, 3.0]), tmax)
    return b.index, S, F, R

def cluster_t(r, days):
    g = pd.DataFrame({'r': r, 'd': days}).groupby('d').r.agg(['sum', 'count'])
    m = r.mean(); se = np.sqrt(((g['sum'] - g['count'] * m) ** 2).sum()) / len(r)
    return m / se if se > 0 else np.nan

if __name__ == '__main__':
    rows = []; store = {}
    for tf in ['15m', '1h']:
        per = []
        for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
            idx, S, F, R = build(sym, tf); per.append((sym, idx, S, F, R)); print('built', sym, tf, len(idx), flush=True)
        for p in per[0][2]:
            for d, dn in enumerate(['long', 'short']):
                for f in ['F0', 'F1', 'F2', 'F3', 'F4']:
                    for q, k in enumerate([1, 2, 3]):
                        recs = []
                        for sym, idx, S, Fd, R in per:
                            m = S[p][d].fillna(False).values & Fd[f][d] & ~np.isnan(R[:, d, q])
                            recs.append(pd.DataFrame({'t': idx[m], 'r': R[m, d, q], 'sym': sym}))
                        T = pd.concat(recs)
                        for sg, (a, b_) in SEGS.items():
                            x = T[(T.t >= a) & (T.t <= b_)]
                            if len(x) < 30: continue
                            rows.append(dict(tf=tf, setup=p, dir=dn, filt=f, tp=k, seg=sg, n=len(x), expR=x.r.mean(),
                                             win=(x.r > 0).mean(), t=cluster_t(x.r.values, x.t.dt.floor('1D').values),
                                             min_coin=x.groupby('sym').r.mean().min(), totR=x.r.sum()))
    Rz = pd.DataFrame(rows); Rz.to_csv('chart_rules_all.csv', index=False)
    disc = Rz[Rz.seg == 'DISC']
    print('configs evaluated on DISC:', len(disc))
    print('baseline-ish: median DISC expR', round(disc.expR.median(), 3))
    pd.set_option('display.width', 250)
    print('\nTop 15 DISC by t:'); print(disc.sort_values('t', ascending=False).head(15).round(3).to_string())
    sel = disc[(disc.expR > 0) & (disc.t > 3.9) & (disc.min_coin > 0)]
    print(f'\nDISC survivors (t>3.9, all coins >0): {len(sel)}')
    if len(sel):
        key = ['tf', 'setup', 'dir', 'filt', 'tp']
        V = Rz[Rz.seg == 'VAL'].set_index(key); Hh = Rz[Rz.seg == 'HOLD'].set_index(key)
        out = sel.set_index(key)[['n', 'expR', 't', 'win']].join(V[['n', 'expR', 't']], rsuffix='_VAL')
        out['VAL_pass'] = (out.expR_VAL > 0) & (out.t_VAL > 2)
        out = out.join(Hh[['n', 'expR', 't']], rsuffix='_HOLD')
        out.loc[~out.VAL_pass, ['n_HOLD', 'expR_HOLD', 't_HOLD']] = np.nan     # HOLD shown only if VAL passed
        print(out.round(3).to_string())

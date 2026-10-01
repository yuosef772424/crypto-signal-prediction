"""H18: maker-entry trend pullback on untouched coins (Addendum G / G.1)."""
import numpy as np, pandas as pd, sys
from numba import njit
D = '/home/user/research'
MAKER, TAKER, EPS = 0.0002, 0.0007, 0.0002
COINS = ['ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'DOGEUSDT', 'TRXUSDT', 'XRPUSDT', 'ZECUSDT']
SEGS = {'2018-21': ('2017-01-01', '2021-12-31 23:59'), '2022-23': ('2022-01-01', '2023-12-31 23:59'), '2024-26': ('2024-01-01', '2026-12-31')}

@njit
def trades(h5, l5, c5, start, end, lim, dirn, stopd, m, tmax):
    n = len(start); ei = np.full(n, -1); xi = np.full(n, -1); R = np.full(n, np.nan)
    for j in range(n):
        d = dirn[j]
        if d == 0 or not (stopd[j] > 0) or start[j] < 0: continue
        L = lim[j]; f = -1
        for i in range(start[j], end[j]):
            if (d == 1 and l5[i] <= L * (1 - EPS)) or (d == -1 and h5[i] >= L * (1 + EPS)):
                f = i; break
        if f < 0: continue
        sl = L - d * stopd[j]; tp = L + d * m * stopd[j]
        cost_sl = (MAKER + TAKER) * L / stopd[j]; cost_tp = (2 * MAKER) * L / stopd[j]
        last = min(f + tmax, len(h5) - 1); r = np.nan; x = last
        for i in range(f, last + 1):
            if (d == 1 and l5[i] <= sl) or (d == -1 and h5[i] >= sl):
                r = -1.0 - cost_sl; x = i; break
            if i > f and ((d == 1 and h5[i] >= tp * (1 + EPS)) or (d == -1 and l5[i] <= tp * (1 - EPS))):
                r = m - cost_tp; x = i; break
        if np.isnan(r):
            r = d * (c5[last] - L) / stopd[j] - cost_sl
        ei[j] = f; xi[j] = x; R[j] = r
    return ei, xi, R

def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def run_coin(sym, tf, k, m, htf):
    m5 = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    rule = {'15m': '15min', '1h': '1h'}[tf]
    b = m5.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean(); e20, e50 = ema(b.close, 20), ema(b.close, 50)
    dirn = np.where((e20 > e50) & (b.close > e20), 1, np.where((e20 < e50) & (b.close < e20), -1, 0)).astype(np.int64)  # G.2: limit rests away from market
    if htf:
        h4 = m5.close.resample('4h').last().dropna(); up4 = (h4 > ema(h4, 50)); up4.index = up4.index + pd.Timedelta('4h')
        u = up4.reindex(b.index + pd.Timedelta(rule), method='ffill').values
        dirn = np.where((dirn == 1) & (u == True), 1, np.where((dirn == -1) & (u == False), -1, 0)).astype(np.int64)
    tclose = b.index + pd.Timedelta(rule)
    start = m5.index.get_indexer(tclose, method='bfill'); end = m5.index.get_indexer(tclose + pd.Timedelta(rule), method='bfill')
    end = np.where(end < 0, len(m5), end)
    ok = atr.notna().values & (start >= 0)
    dirn = np.where(ok, dirn, 0)
    tmax = {'15m': 288, '1h': 864}[tf]
    ei, xi, R = trades(m5.high.values, m5.low.values, m5.close.values, start.astype(np.int64), end.astype(np.int64),
                       e20.values, dirn, (k * atr).fillna(0).values, float(m), tmax)
    sel = ei >= 0
    T = pd.DataFrame({'ei': ei[sel], 'xi': xi[sel], 'r': R[sel]}).sort_values('ei')
    acc = []; opens = []                                      # cap 3 concurrent (scaling-in)
    for e, x in zip(T.ei.values, T.xi.values):
        opens = [z for z in opens if z > e]
        if len(opens) < 3: acc.append(True); opens.append(x)
        else: acc.append(False)
    T = T[np.array(acc)]
    T['t'] = m5.index[T.ei.values]; T['sym'] = sym
    return T

def cluster_t(r, days):
    g = pd.DataFrame({'r': r, 'd': days}).groupby('d').r.agg(['sum', 'count'])
    mu = r.mean(); se = np.sqrt(((g['sum'] - g['count'] * mu) ** 2).sum()) / len(r)
    return mu / se if se > 0 else np.nan

if __name__ == '__main__':
    coins = COINS if (len(sys.argv) < 2 or sys.argv[1] != 'majors') else ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    rows = []
    for tf in ['15m', '1h']:
        for k in [1, 2, 3]:
            for m in [2, 3]:
                for htf in [False, True]:
                    T = pd.concat([run_coin(s, tf, k, m, htf) for s in coins])
                    r = dict(tf=tf, k=k, m=m, htf=htf, n=len(T), expR=T.r.mean(), win=(T.r > 0).mean(),
                             t=cluster_t(T.r.values, T.t.dt.floor('1D').values),
                             coins_pos=int((T.groupby('sym').r.mean() > 0).sum()), totR=T.r.sum())
                    for sg, (a, b_) in SEGS.items():
                        x = T[(T.t >= a) & (T.t <= b_)]; r[f'exp_{sg}'] = x.r.mean(); r[f't_{sg}'] = cluster_t(x.r.values, x.t.dt.floor('1D').values) if len(x) > 30 else np.nan
                    rows.append(r); print(tf, k, m, htf, round(r['expR'], 3), round(r['t'], 2), flush=True)
    Rz = pd.DataFrame(rows); pd.set_option('display.width', 250)
    print(Rz.round(3).to_string())
    Rz['ACCEPT'] = (Rz.expR > 0) & (Rz.t > 3.0) & (Rz.coins_pos >= (5 if len(coins) == 7 else 3)) & (Rz[[f'exp_{s}' for s in SEGS]] > 0).all(axis=1)
    print('\nACCEPTED:', int(Rz.ACCEPT.sum())); print(Rz[Rz.ACCEPT].round(3).to_string())
    Rz.to_csv(f'pullback_maker_{"majors" if len(coins) == 3 else "untouched"}.csv', index=False)

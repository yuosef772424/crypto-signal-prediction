"""H23 (Addendum L): limits at predicted high / low of the current candle, exit at its close."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
D = '/home/user/research'
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'DOGEUSDT', 'TRXUSDT', 'XRPUSDT', 'ZECUSDT']
TR, VA = '2021-12-31 23:59', '2022-12-31 23:59'
EPS, COST = 0.0002, 0.0009

def frame(sym, tf):
    m5 = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    b = m5.resample({'15m': '15min', '1h': '1h', '4h': '4h'}[tf]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    b = b[b.high > b.low]
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    r = np.log(b.close).diff(); sig = r.rolling(100).std().shift(1); z = (r / sig).clip(-8, 8)
    F = pd.DataFrame(index=b.index)                                   # features at close of k-1 -> shifted onto row k
    F['B_z'] = z; F['B_z4'] = z.rolling(4).sum(); F['B_z16'] = z.rolling(16).sum(); F['B_z32'] = z.rolling(32).sum()
    F['B_vr'] = np.log(r.rolling(20).std() / r.rolling(100).std())
    lv = np.log(b.volume.replace(0, np.nan)); F['B_vz'] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    F['B_rng'] = (b.high - b.low) / atr; F['B_pos'] = (b.close - b.low) / (b.high - b.low)
    F['B_uw'] = (b.high - b[['open', 'close']].max(axis=1)) / atr; F['B_lw'] = (b[['open', 'close']].min(axis=1) - b.low) / atr
    F['B_atrpct'] = atr / b.close
    Y = z.fillna(0).cumsum()
    for j in range(1, 6):
        h = 2 ** (j - 1); d = Y.rolling(h).mean() - Y.shift(h).rolling(h).mean(); F[f'W_d{j}'] = d; F[f'W_e{j}'] = (d ** 2).rolling(16).mean()
    F['hour'] = b.index.hour; F['dow'] = b.index.dayofweek
    F = F.shift(1); F['hour'] = b.index.hour; F['dow'] = b.index.dayofweek     # time-of-day of candle k is known at its open
    F['atr'] = atr.shift(1); F['O'] = b.open; F['H'] = b.high; F['L'] = b.low; F['C'] = b.close
    F['h'] = (b.high - b.open) / F.atr; F['l'] = (b.low - b.open) / F.atr; F['sym'] = sym
    return F.replace([np.inf, -np.inf], np.nan).dropna()

def pnl(df, hh, ll):
    sell = df.O + np.maximum(hh, 0.05) * df.atr; buy = df.O + np.minimum(ll, -0.05) * df.atr
    fs = df.H >= sell * (1 + EPS); fb = df.L <= buy * (1 - EPS)
    s = pd.DataFrame({'t': df.index[fs], 'sym': df.sym[fs].values, 'net': ((sell - df.C) / sell)[fs].values - COST, 'side': -1})
    l = pd.DataFrame({'t': df.index[fb], 'sym': df.sym[fb].values, 'net': ((df.C - buy) / buy)[fb].values - COST, 'side': 1})
    both = (fs & fb).mean()
    return pd.concat([s, l]), fs.mean(), fb.mean(), both

def ctstat(x, t):
    g = pd.DataFrame({'r': x, 'd': pd.to_datetime(t).floor('1D')}).groupby('d').r.agg(['sum', 'count'])
    mu = x.mean(); se = np.sqrt(((g['sum'] - g['count'] * mu) ** 2).sum()) / len(x); return mu / se

rows = []
for tf in ['4h', '1h', '15m']:
    P = pd.concat([frame(s, tf) for s in COINS]).sort_index()
    X = [c for c in P if c[:2] in ('B_', 'W_')] + ['hour', 'dow']
    tr, te = P.loc[:TR], P.loc[VA:].iloc[1:]
    if tf == '15m': tr = tr.iloc[::2]
    for q in [0.3, 0.5, 0.7]:
        mh = HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, random_state=0).fit(tr[X], tr.h)
        ml = HistGradientBoostingRegressor(loss='quantile', quantile=1 - q, max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, random_state=0).fit(tr[X], tr.l)
        for name, hh, ll in [('model', mh.predict(te[X]), ml.predict(te[X])),
                             ('ATR-band', np.full(len(te), tr.h.quantile(q)), np.full(len(te), tr.l.quantile(1 - q)))]:
            T, f_s, f_b, both = pnl(te, hh, ll)
            rows.append(dict(tf=tf, q=q, levels=name, trades=len(T), fill_sell=f_s, fill_buy=f_b, both_fill=both,
                             net_bp=T.net.mean() * 1e4, gross_bp=(T.net.mean() + COST) * 1e4, win=(T.net > 0).mean(),
                             t=ctstat(T.net.values, T.t.values), coins_pos=int((T.groupby('sym').net.mean() > 0).sum()),
                             net_2324=T[T.t < '2025-01-01'].net.mean() * 1e4, net_2526=T[T.t >= '2025-01-01'].net.mean() * 1e4,
                             sell_bp=T[T.side == -1].net.mean() * 1e4, buy_bp=T[T.side == 1].net.mean() * 1e4))
        print(tf, q, 'done', flush=True)
R = pd.DataFrame(rows)
base = R[R.levels == 'ATR-band'].set_index(['tf', 'q']).net_bp
R['beats_band'] = R.apply(lambda r: r.net_bp > base.loc[(r.tf, r.q)], axis=1)
R['ACCEPT'] = (R.levels == 'model') & (R.net_bp > 0) & (R.t > 3) & (R.coins_pos >= 7) & (R.net_2324 > 0) & (R.net_2526 > 0) & R.beats_band
pd.set_option('display.width', 250); print(R.round(3).to_string()); R.to_csv('wick_capture_results.csv', index=False)

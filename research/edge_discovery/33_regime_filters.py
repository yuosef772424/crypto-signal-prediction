"""H24 (Addendum M): ADX / NATR / ATR-ratio regime filters on fade (H23) and breakout (H23-R). Select on 2022, confirm on 2023-26."""
import numpy as np, pandas as pd
src = open('31_wick_capture.py').read().split("rows = []")[0]; exec(src)       # frame(), ctstat(), COINS, TR, VA
from sklearn.ensemble import HistGradientBoostingRegressor

def indicators(sym, tf):
    m5 = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    b = m5.resample({'15m': '15min', '1h': '1h', '4h': '4h'}[tf]).agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    b = b[b.high > b.low]
    up = b.high.diff(); dn = -b.low.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    w = lambda x: pd.Series(x, index=b.index).ewm(alpha=1 / 14, adjust=False).mean()
    atrw = w(tr); pdi = 100 * w(pdm) / atrw; ndi = 100 * w(ndm) / atrw
    adx = w((100 * (pdi - ndi).abs() / (pdi + ndi)).values)
    atr = tr.rolling(14).mean(); natr = atr / b.close
    I = pd.DataFrame({'ADX': adx, 'NATRp': natr.rolling(500, min_periods=200).rank(pct=True), 'ATRr': atr / tr.rolling(100).mean()})
    return I.shift(1)                                                             # known at close k-1

def trades(df, hh, ll, kind):
    up = df.O + np.maximum(hh, 0.05) * df.atr; lo = df.O + np.minimum(ll, -0.05) * df.atr
    fu = df.H >= up * (1 + EPS); fl = df.L <= lo * (1 - EPS)
    if kind == 'fade':
        cost = 0.0009; a = ((up - df.C) / up) - cost; b = ((df.C - lo) / lo) - cost
    else:
        cost = 0.0014; bx = up * (1 + EPS); sx = lo * (1 - EPS); a = ((df.C - bx) / bx) - cost; b = ((sx - df.C) / sx) - cost
    return pd.concat([pd.DataFrame({'t': df.index[fu], 'sym': df.sym[fu].values, 'net': a[fu].values}),
                      pd.DataFrame({'t': df.index[fl], 'sym': df.sym[fl].values, 'net': b[fl].values})])

FILT = {'F1 ADX<20': lambda d: d.ADX < 20, 'F2 ADX>25': lambda d: d.ADX > 25, 'F3 NATR high': lambda d: d.NATRp > 0.7,
        'F4 NATR low': lambda d: d.NATRp < 0.3, 'F5 ATR expanding': lambda d: d.ATRr > 1.2, 'F6 ATR contracting': lambda d: d.ATRr < 0.8}
rows = []
for tf in ['4h', '1h', '15m']:
    P = pd.concat([frame(s, tf).join(indicators(s, tf)) for s in COINS]).sort_index().dropna(subset=['ADX', 'NATRp', 'ATRr'])
    X = [c for c in P if c[:2] in ('B_', 'W_')] + ['hour', 'dow']
    tr, va, te = P.loc[:TR], P.loc[TR:VA].iloc[1:], P.loc[VA:].iloc[1:]
    if tf == '15m': tr = tr.iloc[::2]
    for q in [0.5, 0.7]:
        mh = HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, random_state=0).fit(tr[X], tr.h)
        ml = HistGradientBoostingRegressor(loss='quantile', quantile=1 - q, max_iter=150, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, random_state=0).fit(tr[X], tr.l)
        pred = {k: (mh.predict(d[X]), ml.predict(d[X])) for k, d in (('va', va), ('te', te))}
        for kind in ['fade', 'breakout']:
            filters = dict(FILT)
            filters['F7 theory combo'] = (lambda d: (d.ADX < 20) & (d.ATRr < 1)) if kind == 'fade' else (lambda d: (d.ADX > 25) & (d.ATRr > 1.2) & (d.NATRp > 0.5))
            for fn, f in filters.items():
                r = dict(tf=tf, q=q, strategy=kind, filter=fn)
                for seg, d in (('va', va), ('te', te)):
                    m = f(d).values; hh, ll = pred[seg]
                    T = trades(d[m], hh[m], ll[m], kind)
                    r[f'{seg}_n'] = len(T); r[f'{seg}_net'] = T.net.mean() * 1e4 if len(T) else np.nan
                    r[f'{seg}_t'] = ctstat(T.net.values, T.t.values) if len(T) > 30 else np.nan
                    if seg == 'te' and len(T):
                        r['te_coins'] = int((T.groupby('sym').net.mean() > 0).sum())
                        r['te_2324'] = T[T.t < '2025-01-01'].net.mean() * 1e4; r['te_2526'] = T[T.t >= '2025-01-01'].net.mean() * 1e4
                rows.append(r)
        print(tf, q, 'done', flush=True)
R = pd.DataFrame(rows)
R['selected_on_2022'] = (R.va_net > 0) & (R.va_t > 2)
R['CONFIRMED'] = R.selected_on_2022 & (R.te_net > 0) & (R.te_t > 2.5) & (R.te_coins >= 7) & (R.te_2324 > 0) & (R.te_2526 > 0)
pd.set_option('display.width', 250)
print('\nbest 15 by 2022 (selection data):'); print(R.sort_values('va_net', ascending=False).head(15).round(2).to_string())
print('\nselected on 2022:', int(R.selected_on_2022.sum()), '| confirmed on 2023-26:', int(R.CONFIRMED.sum()))
print(R[R.selected_on_2022].round(2).to_string())
print('\nmean net by filter (TEST, all tf/q):'); print(R.groupby(['strategy', 'filter'])[['va_net', 'te_net']].mean().round(2).to_string())
R.to_csv('regime_filters_results.csv', index=False)

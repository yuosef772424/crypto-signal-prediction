"""Diagnostic for H17 (post-hoc, descriptive): same grid at ZERO cost + cost expressed in R per timeframe."""
import numpy as np, pandas as pd, importlib
M = importlib.import_module('22_chart_rules'.replace('.py', '')) if False else None
import runpy
g = runpy.run_path('22_chart_rules.py', run_name='lib')
g['TAKER'] = 0.0; g['MAKER'] = 0.0
# rebuild numba function with zero-cost globals
src = open('22_chart_rules.py').read().split("if __name__ == '__main__':")[0].replace('TAKER, MAKER = 0.0007, 0.0002', 'TAKER, MAKER = 0.0, 0.0')
ns = {}; exec(compile(src, 'zc', 'exec'), ns)
rows = []
for tf in ['15m', '1h']:
    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        idx, S, F, R = ns['build'](sym, tf)
        m5 = pd.read_parquet(f"{ns['D']}/ohlc_{sym}_5m.parquet")
        b = m5.resample({'15m': '15min', '1h': '1h'}[tf]).agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        atr = ns['atrf'](b.assign(open=b.close))
        cost_R = (0.0014 * b.close / atr).median()                 # round-trip taker cost in R (stop = 1 ATR)
        for p in S:
            for d, dn in enumerate(['long', 'short']):
                for q, k in enumerate([1, 2, 3]):
                    m = S[p][d].fillna(False).values & ~np.isnan(R[:, d, q])
                    x = pd.Series(R[m, d, q], index=idx[m])
                    for sg, (a, b_) in ns['SEGS'].items():
                        y = x.loc[a:b_]
                        if len(y) > 30: rows.append(dict(tf=tf, sym=sym, setup=p, dir=dn, tp=k, seg=sg, n=len(y), grossR=y.mean(), cost_R=cost_R))
G = pd.DataFrame(rows)
print('median round-trip cost in R (stop = 1 ATR):'); print(G.groupby(['tf', 'sym']).cost_R.first().round(3).to_string())
P = G.groupby(['tf', 'setup', 'dir', 'tp', 'seg']).apply(lambda z: np.average(z.grossR, weights=z.n)).unstack('seg')
print('\nGROSS expectancy (R, zero cost, F0 no filter), pooled coins — best 12 by DISC:')
print(P.sort_values('DISC', ascending=False).head(12).round(3).to_string())
print('\nshare of setups with gross DISC>0:', round((P.DISC > 0).mean(), 2), '| gross DISC>0 AND VAL>0 AND HOLD>0:', int(((P > 0).all(axis=1)).sum()), 'of', len(P))

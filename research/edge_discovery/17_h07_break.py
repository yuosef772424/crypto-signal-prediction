"""Attempts to break H07-OOS (post-hoc stress, descriptive): 1-day execution lag, 2x/4x costs, DSR, sub-periods."""
import numpy as np, pandas as pd
from scipy.stats import norm
from tsbt import stats
exec(open('16_h07_oos.py').read().split("A, B = ")[0])          # reuse loader + run()
A, B = '2018-01-01', '2023-12-31'
def dsr(x, N):
    sr = x.mean() / x.std(); T = len(x); g = 0.5772156649
    sr0 = np.sqrt(1 / T) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e))) if N > 1 else 0
    return norm.cdf((sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - x.skew() * sr + (x.kurt() + 2) / 4 * sr ** 2))
sig = (np.log(px).diff(30) > 0).astype(float)
bh = pd.DataFrame(1.0, index=px.index, columns=px.columns)
rows = []
for name, s_, c in [('base', sig, .0012), ('lag1', sig.shift(1), .0012), ('lag2', sig.shift(2), .0012),
                    ('cost2x', sig, .0024), ('cost4x', sig, .0048), ('lag1_cost2x', sig.shift(1), .0024),
                    ('BUYHOLD', bh, .0012)]:
    x = run(s_, ['BTC', 'ETH'], c).loc[A:B]; st = stats(x)
    st.update(test=name, DSR_N1=dsr(x, 1), DSR_N44=dsr(x, 44), DSR_N750=dsr(x, 750)); rows.append(st)
D = pd.DataFrame(rows)
print(D[['test', 'sharpe', 'ann_ret', 'maxdd', 't', 'DSR_N1', 'DSR_N44', 'DSR_N750']].round(3).to_string())
x = run(sig, ['BTC', 'ETH']).loc[A:B]
print('exposure (share of days invested):', round((sig.loc[A:B].mean()).mean(), 2))
print('remove best 5% days -> sharpe', round(stats(x[x < x.quantile(.95)])['sharpe'], 2),
      '| buyhold same:', round(stats(run(bh, ['BTC','ETH']).loc[A:B].pipe(lambda y: y[y < y.quantile(.95)]))['sharpe'], 2))

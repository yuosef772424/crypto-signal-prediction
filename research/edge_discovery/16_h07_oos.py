"""H07-OOS: frozen TSMOM30 long-only BTC+ETH on untouched 2018-2023 (Addendum C). CoinMetrics daily PriceUSD."""
import numpy as np, pandas as pd
from tsbt import stats
CM = '/home/user/research/coinmetrics'
def load(a):
    d = pd.read_csv(f'{CM}/{a}.csv', usecols=['time', 'PriceUSD'], low_memory=False)
    return pd.Series(pd.to_numeric(d.PriceUSD, errors='coerce').values, index=pd.to_datetime(d.time)).dropna()
px = pd.DataFrame({'BTC': load('btc'), 'ETH': load('eth')}).loc['2017-01-01':'2023-12-31']
vol = np.log(px).diff().rolling(30, min_periods=20).std()
r = px.pct_change().shift(-1)                     # weights at day d earn d -> d+1
def run(raw, cols, cost=0.0012):
    w = (raw[cols] * (0.02 / vol[cols]).clip(upper=3)) / len(cols)
    w = w.fillna(0)
    turn = (w - w.shift(1).fillna(0)).abs().sum(axis=1)
    return (w * r[cols]).sum(axis=1, min_count=1) - turn * cost / 2
A, B = '2018-01-01', '2023-12-31'
rows = []
for cols in [['BTC', 'ETH'], ['BTC'], ['ETH']]:
    for N in [30, 7, 14, 21, 45, 60, 90]:
        sig = (np.log(px).diff(N) > 0).astype(float)
        n = run(sig, cols).loc[A:B]
        s = stats(n); s.update(univ='+'.join(cols), strat=f'TSMOM{N}'); rows.append(s)
    n = run(pd.DataFrame(1.0, index=px.index, columns=px.columns), cols).loc[A:B]
    s = stats(n); s.update(univ='+'.join(cols), strat='BUYHOLD'); rows.append(s)
D = pd.DataFrame(rows)
print(D[['univ', 'strat', 'days', 'sharpe', 'ann_ret', 'ann_vol', 'maxdd', 'total', 't', 'skew']].round(3).to_string())
# by calendar year, primary config
sig = (np.log(px).diff(30) > 0).astype(float)
ts = run(sig, ['BTC', 'ETH']).loc[A:B]; bh = run(pd.DataFrame(1.0, index=px.index, columns=px.columns), ['BTC', 'ETH']).loc[A:B]
Y = pd.DataFrame({'TSMOM30_ret': ts.groupby(ts.index.year).sum(), 'BUYHOLD_ret': bh.groupby(bh.index.year).sum(),
                  'TSMOM30_sharpe': ts.groupby(ts.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(365)),
                  'BUYHOLD_sharpe': bh.groupby(bh.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(365))})
print(Y.round(2).to_string())

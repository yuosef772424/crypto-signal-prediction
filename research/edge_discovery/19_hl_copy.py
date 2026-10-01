"""H14 (exploratory): do Hyperliquid top traders' visible positions predict returns? Copy with 1h delay.
Data: github.com/kushagra93/hl-top-traders hourly snapshots 2026-06-01 -> 2026-10-01 (19 top all-time-PnL accounts).
No holdout exists (4 months) -> exploratory only; cannot be 'accepted'."""
import numpy as np, pandas as pd, os
from tsbt import stats
D = '/home/user/research'
P = pd.read_parquet(f'{D}/hl_positions.parquet')
P['szi'] = pd.to_numeric(P.szi); P['val'] = pd.to_numeric(P.positionValue)
P['ts'] = pd.to_datetime(P.ts, utc=True, format='ISO8601').dt.floor('h'); P['usd'] = np.sign(P.szi) * P.val
P = P[~P.coin.str.contains(':')]                                          # crypto perps only
px = pd.read_parquet(f'{D}/panel_px.parquet')
def price(c):
    k = f'{c}USDT'
    if k in px: return px[k]
    for d in ['metrics_new', 'metrics']:
        f = f'{D}/{d}/{k}.csv.gz'
        if os.path.exists(f):
            m = pd.read_csv(f); m['ts'] = pd.to_datetime(m.timestamp, utc=True).dt.floor('h')
            m = m.drop_duplicates('ts').set_index('ts'); return (m.sum_open_interest_value / m.sum_open_interest).where(m.sum_open_interest > 0)
    return None
coins = P.groupby('coin').usd.apply(lambda s: s.abs().sum()).sort_values(ascending=False).index[:25]
PX = pd.DataFrame({c: price(c) for c in coins if price(c) is not None}).loc['2026-05-25':]
PX = PX.ffill(limit=2); r1 = PX.pct_change()                      # r1[t] = return t-1 -> t
print('priced coins:', list(PX.columns), '| price end', PX.index.max())
N = P[P.coin.isin(PX.columns)].pivot_table(index='ts', columns='coin', values='usd', aggfunc='sum').fillna(0)
N = N.reindex(pd.date_range(N.index.min(), PX.index.max(), freq='h', tz='UTC')).ffill(limit=6)   # hold last snapshot <=6h
def copy(W, lag=1, cost=0.0007):
    W = W.div(W.abs().sum(axis=1), axis=0).fillna(0)               # gross 1
    Wx = W.shift(lag)                                              # decided at t, entered t+lag
    R = r1.reindex(Wx.index).shift(-1)                             # earn t -> t+1
    turn = (Wx - Wx.shift(1)).abs().sum(axis=1)
    return ((Wx * R).sum(axis=1) - turn * cost / 2).dropna()
def hstats(x):
    s = stats(x.resample('D').sum()); return {k: s[k] for k in ['days', 'sharpe', 'ann_ret', 'maxdd', 't']}
rows = []
for lag in [1, 2, 6, 24]:
    s = hstats(copy(N, lag)); s.update(test=f'copy_net_all lag{lag}h'); rows.append(s)
s = hstats(copy(np.sign(N), 1)); s.update(test='copy_sign_all lag1h'); rows.append(s)
for c in ['BTC', 'ETH', 'SOL', 'HYPE']:
    if c in N: s = hstats(copy(N[[c]], 1)); s.update(test=f'copy_{c} lag1h'); rows.append(s)
bh = r1[['BTC']].reindex(N.index).shift(-1)['BTC'].dropna(); s = hstats(bh); s.update(test='BTC buy&hold'); rows.append(s)
# per trader copy (their own book)
for a, g in P[P.coin.isin(PX.columns)].groupby('address'):
    W = g.pivot_table(index='ts', columns='coin', values='usd', aggfunc='sum').fillna(0)
    W = W.reindex(N.index).ffill(limit=6).reindex(columns=PX.columns).fillna(0)
    if W.abs().sum(axis=1).gt(0).sum() < 24 * 20: continue
    s = hstats(copy(W, 1)); s.update(test=f'trader {a[:8]} lag1h (gross $M {g.val.mean()/1e6:.1f})'); rows.append(s)
print(pd.DataFrame(rows).set_index('test').round(3).to_string())
# flow signal: 24h change in aggregate net USD -> next 24h return, pooled across coins, daily non-overlapping
dN = N.diff(24); f24 = PX.pct_change(24).shift(-25).reindex(N.index)  # return t+1 -> t+25 (1h latency)
X = dN.iloc[::24].stack(); Y = f24.iloc[::24].stack()
J = pd.concat([X.rename('dN'), Y.rename('fwd')], axis=1).dropna(); J = J[J.dN != 0]
J['sgn'] = np.sign(J.dN); d = J.groupby(level=0).apply(lambda g: (g.sgn * g.fwd).mean())
print(f'\nflow: 24h position change -> next-24h return (follow sign): n_days={len(d)} mean={d.mean()*1e4:.1f}bp t={d.mean()/d.std()*np.sqrt(len(d)):.2f} hit={(J.sgn*J.fwd>0).mean():.3f}')
# --- decomposition: market direction vs coin selection ---
G = N.div(N.abs().sum(axis=1), axis=0)
print('\nshare of hours net short (gross-normalised sum<0):', round((G.sum(axis=1) < 0).mean(), 2), '| mean net exposure', round(G.sum(axis=1).mean(), 2))
rel = N.sub(N.mean(axis=1), axis=0)                                # market-neutral tilt across coins
s = hstats(copy(rel, 1)); print('copy relative tilt (market-neutral) lag1h:', {k: round(v, 3) for k, v in s.items()})
# spot holdings of the same accounts (hedge check): fraction of perp-short notional by accounts that also hold spot
F = pd.read_parquet(f'{D}/hl_fills.parquet'); spot_holders = set(F[F.coin.str.startswith('@')].address)
sh = P[(P.usd < 0)]; print('share of short notional held by accounts that also trade spot:',
      round(sh[sh.address.isin(spot_holders)].val.sum() / sh.val.sum(), 2))

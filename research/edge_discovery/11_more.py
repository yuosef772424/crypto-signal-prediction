import numpy as np, pandas as pd
from lib import *; from features import build
P = load(); L = P['L']; U = P['univ']; F = build(P)
r = L.diff(); a, b = DISC
alts = [c for c in L.columns if c not in ('BTCUSDT', 'ETHUSDT')]
# 1) lead-lag: does BTC's last-hour return predict alts' next-hour return beyond alts' own last-hour return?
btc = r['BTCUSDT']; ra = r[alts].where(U[alts])
y = (L.shift(-2) - L.shift(-1))[alts].where(U[alts])        # next hour, 1h latency
for seg_, (s, e) in [('DISC', DISC), ('VAL', VAL)]:
    beta_res = []
    Yb = y.loc[s:e]; Xb = btc.loc[s:e]; Xo = ra.loc[s:e]
    m = Yb.mean(axis=1); mb = Xb
    # market-level: EW alt next-hour on BTC last hour
    d = pd.DataFrame({'y': m, 'btc': mb, 'own': Xo.mean(axis=1)}).dropna()
    A = np.column_stack([np.ones(len(d)), d.btc, d.own]); co, *_ = np.linalg.lstsq(A, d.y, rcond=None)
    res = d.y - A @ co; se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * res.var())
    print(f'lead-lag {seg_}: alt_next ~ btc_last + alt_last  coef_btc={co[1]:.4f} t={co[1]/se[1]:.2f} coef_own={co[2]:.4f} t={co[2]/se[2]:.2f}')
# 2) failed breakout: new 7d high at t-k..t-1 then back below the prior 7d high now -> next 24h
hi = L.rolling(168).max().shift(1)
brk = (L > hi)
failed = brk.shift(6).fillna(False) & (L < hi.shift(6))       # broke out 6h ago, now back below that old high
from events import dedup, event_stats
for seg_, (s, e) in [('DISC', DISC), ('VAL', VAL)]:
    E = dedup(failed.loc[s:e].fillna(False), 72)
    for H in [12, 24, 72]:
        st = event_stats(E, L, U, H, side=-1, mkt_adj=True, cost=0.0015)
        print(f'failed-breakout SHORT {seg_} H={H}', {k: round(v, 3) for k, v in st.items()})
# 3) market-level positioning: BTC retail ratio extreme & aggregate (median z across top50)
lsz = F['ls_acc_z'].where(U & P['big']).median(axis=1)
mret = r.where(U & P['big']).mean(axis=1); mL = mret.cumsum()
for H in [24, 72]:
    f = (mL.shift(-1 - H) - mL.shift(-1))
    for seg_, (s, e) in [('DISC', DISC), ('VAL', VAL)]:
        z = lsz.loc[s:e].iloc[::H]; ff = f.loc[s:e].iloc[::H]
        d = pd.DataFrame({'z': z, 'f': ff}).dropna()
        hi_, lo_ = d[d.z > 1].f, d[d.z < -1].f
        print(f'agg retail L/S z H={H} {seg_}: corr={d.corr().iloc[0,1]:.3f} n={len(d)} | z>1: {hi_.mean()*1e4:.0f}bp n={len(hi_)} | z<-1: {lo_.mean()*1e4:.0f}bp n={len(lo_)}')

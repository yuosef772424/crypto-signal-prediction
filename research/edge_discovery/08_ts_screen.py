"""
PURPOSE:  H08: pooled time-series predictability of each feature's own z-score vs vol-normalised forward return (93
          tests, hourly top-50).
TAGS:     H08, time-series screen, pooled z*y, IC by quarter, ts_screen_disc.csv
PITFALLS: Max |t| = 2.7 is noise level; needs /home/user/research/panel_*.parquet (built from Drive futures_metrics
          via data_tools; not in repo, see RESUME.md)
"""
import numpy as np, pandas as pd
from lib import *; from features import build
P = load(); L = P['L']; U = P['univ'] & P['big']; F = build(P)
def tsz(x, w=720): return (x - x.rolling(w, min_periods=240).mean()) / x.rolling(w, min_periods=240).std()
rows = []
a, b = DISC
skip = {'size', 'vol24', 'vol168'}
for H in [1, 4, 24]:
    Y = (L.shift(-1 - H) - L.shift(-1))
    Yv = Y / F['vol168'] / np.sqrt(H)                        # vol-normalised target
    for name, X in F.items():
        if name in skip: continue
        Z = tsz(X).clip(-4, 4).where(U)
        m = (Z * Yv).loc[a:b].iloc[::H].mean(axis=1)         # avg over coins of z*y at each timestamp
        # separate market-timing component: avg z times avg y
        q = m.groupby([m.index.year, m.index.quarter]).mean()
        rows.append(dict(feat=name, H=H, ic=m.mean(), t=tstat(m), q_pos=(q > 0).sum(), q_n=len(q)))
    # also for BTC alone
D = pd.DataFrame(rows); D['abs_t'] = D.t.abs()
pd.set_option('display.width', 200)
print(D.sort_values('abs_t', ascending=False).head(30).round(4).to_string()); D.to_csv('ts_screen_disc.csv', index=False)
print('tests', len(D))

"""
PURPOSE:  H02 test: quintile long-short backtests on price/positioning features at H=24/72/168h, net of costs, with
          turnover.
TAGS:     H02, quintiles, long-short, turnover, cost_big cost_small, combo signal, discovery
PITFALLS: needs /home/user/research/panel_*.parquet (built from Drive futures_metrics via data_tools; not in repo, see
          RESUME.md)
"""
import numpy as np, pandas as pd, sys
from lib import *
P = load(); L = P['L']; U = P['univ']; r1 = L.diff()
S = {'vol24': r1.rolling(24, min_periods=18).std(), 'vol72': r1.rolling(72, min_periods=50).std(),
     'dist_lo_168': L - L.rolling(168).min(), 'mom_72': L.diff(72), 'mom_168': L.diff(168)}
rk = lambda d: d.where(U).rank(axis=1, pct=True)
S['combo'] = (rk(S['vol72']) + rk(S['dist_lo_168']) + rk(S['mom_72'])) / 3
def quint_bt(sig, H, a, b, cost_big=0.0012, cost_small=0.0020, nq=5):
    """rebalance every H hours; hold H hours; equal weight; returns per-period quintile returns (raw), turnover."""
    idx = L.loc[a:b].index[::H]
    R = (L.shift(-1 - H) - L.shift(-1)).loc[idx]           # entry t+1, exit t+1+H
    R = np.expm1(R)
    s = sig.loc[idx].where(U.loc[idx])
    q = s.rank(axis=1, pct=True).apply(lambda x: np.ceil(x * nq))
    out = {}
    for k in range(1, nq + 1):
        out[k] = R.where(q == k).mean(axis=1)
    mkt = R.where(U.loc[idx]).mean(axis=1)
    Q = pd.DataFrame(out); Q['mkt'] = mkt
    # turnover of bottom & top quintile membership
    cst = np.where(P['big'].loc[idx], cost_big, cost_small)
    cst = pd.DataFrame(cst, index=idx, columns=L.columns)
    tc = {}
    for k in [1, nq]:
        m = (q == k).astype(float); w = m.div(m.sum(axis=1), axis=0).fillna(0)
        tc[k] = ((w - w.shift(1).fillna(0)).abs() * cst).sum(axis=1)  # one-way traded weight × RT cost/2*2
    return Q, tc
pd.set_option('display.width', 200)
for H in [24, 72, 168]:
    print(f'\n===== H={H}h, DISCOVERY')
    for name, sig in S.items():
        Q, tc = quint_bt(sig, H, *DISC)
        ls = Q[1] - Q[5]; net = ls - (tc[1] + tc[5]) / 1.0 * 0.5 * 2 / 2  # each side pays its own turnover*cost (cost is RT)
        ann = 365 * 24 / H
        line = ' '.join(f'Q{k}:{Q[k].mean()*1e4:6.1f}' for k in range(1, 6))
        print(f'{name:12s} {line} | mkt {Q.mkt.mean()*1e4:6.1f} | L-S {ls.mean()*1e4:6.1f}bp t={tstat(ls):5.2f} '
              f'| net {net.mean()*1e4:6.1f}bp SR_ann={net.mean()/net.std()*np.sqrt(ann):5.2f} | short-only(Q5-mkt) {(Q[5]-Q.mkt).mean()*1e4:6.1f}')

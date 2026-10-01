import numpy as np, pandas as pd
from lib import *
P = load(); L = P['L']; U = P['univ']; r1 = L.diff()
vol168 = r1.rolling(168, min_periods=120).std()
X = {'vol24': r1.rolling(24, min_periods=18).std(), 'mom_24': L.diff(24), 'mom_168': L.diff(168),
     'mom_720': L.diff(720), 'dist_lo_168': L - L.rolling(168).min(),
     'ls_acc': np.log(P['ls_acc']), 'oi_z30': None, 'smr': np.log(P['tt_pos']) - np.log(P['ls_acc'])}
loiv = np.log(P['oiv']); X['oi_z30'] = (loiv - loiv.rolling(720).mean()) / loiv.rolling(720).std()
X['size'] = loiv  # OI value as size proxy
print('pairwise XS rank corr (discovery, daily samples)')
a, b = DISC
idx = L.loc[a:b].index[::24]
def xsr(df): return df.loc[idx].where(U.loc[idx]).rank(axis=1, pct=True)
R = {k: xsr(v) for k, v in X.items()}
names = list(R)
C = pd.DataFrame(index=names, columns=names, dtype=float)
for i in names:
    for j in names:
        C.loc[i, j] = R[i].corrwith(R[j], axis=1).mean()
print(C.round(2).to_string())
# Fama-MacBeth on ranks, H=24, daily non-overlapping
for H in [24, 72]:
    Y = xs_demean(fwd(L, H), U).loc[idx]
    coefs = []
    for t in idx[::max(1, H // 24)]:
        df = pd.DataFrame({k: R[k].loc[t] for k in names}); df['y'] = Y.loc[t]
        df = df.dropna()
        if len(df) < 40: continue
        A = np.column_stack([np.ones(len(df))] + [df[k] - 0.5 for k in names])
        beta, *_ = np.linalg.lstsq(A, df.y.values, rcond=None)
        coefs.append(pd.Series(beta[1:], index=names, name=t))
    Cf = pd.DataFrame(coefs)
    print(f'\nFama-MacBeth H={H}h  (coef = return of top-vs-bottom rank, bps)')
    print(pd.DataFrame({'coef_bps': Cf.mean() * 1e4, 't': Cf.mean() / Cf.std() * np.sqrt(len(Cf))}).round(2).to_string())

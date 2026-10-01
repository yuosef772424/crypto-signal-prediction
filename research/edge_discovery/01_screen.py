import numpy as np, pandas as pd, itertools, sys
from lib import *
P = load(); L = P['L']; U = P['univ']
r1 = L.diff()
vol24 = r1.rolling(24, min_periods=18).std()
vol168 = r1.rolling(168, min_periods=120).std()
F = {}
for h in [1, 4, 24, 72, 168, 720]:
    F[f'mom_{h}'] = L.diff(h)
    F[f'mom_{h}_vn'] = L.diff(h) / (vol168 * np.sqrt(h))
F['vol24'] = vol24; F['vol_ratio'] = np.log(vol24 / vol168)
F['dist_hi_168'] = L - L.rolling(168).max(); F['dist_lo_168'] = L - L.rolling(168).min()
F['dist_hi_720'] = L - L.rolling(720).max()
lo = np.log(P['oi']); loiv = np.log(P['oiv'])
for h in [1, 4, 24, 72]:
    F[f'doi_{h}'] = lo.diff(h)
F['oi_z30'] = (loiv - loiv.rolling(720).mean()) / loiv.rolling(720).std()
F['oi_to_px_24'] = lo.diff(24) - L.diff(24)       # OI growth beyond price
for k in ['ls_acc', 'tt_acc', 'tt_pos']:
    x = np.log(P[k]); F[k] = x; F[f'{k}_d24'] = x.diff(24)
    F[f'{k}_z'] = (x - x.rolling(720).mean()) / x.rolling(720).std()
F['smart_minus_retail'] = np.log(P['tt_pos']) - np.log(P['ls_acc'])
F['smr_d24'] = F['smart_minus_retail'].diff(24)
tk = np.log(P['taker'].clip(1e-3, 1e3))
F['taker_1'] = tk; F['taker_24'] = tk.rolling(24).mean(); F['taker_z'] = (tk.rolling(4).mean() - tk.rolling(720).mean()) / tk.rolling(720).std()
F['mom24_x_doi24'] = np.sign(L.diff(24)) * lo.diff(24)        # +: trend confirmed by new positions
F['mom24_x_dls24'] = np.sign(L.diff(24)) * F['ls_acc_d24']
np.save('/tmp/fnames.npy', list(F))
rows = []
a, b = DISC
for H in [1, 4, 24, 72]:
    Y = fwd(L, H); Yx = xs_demean(Y, U)
    for name, X in F.items():
        Xs = seg(X, a, b); 
        for tgt, YY in [('xs', Yx)]:
            ic = rank_ic(Xs, seg(YY, a, b), seg(U, a, b), step=H)
            q = ic.groupby(ic.index.quarter.astype(str) + ic.index.year.astype(str)).mean()
            rows.append(dict(feat=name, H=H, tgt=tgt, ic=ic.mean(), t=tstat(ic), n=len(ic),
                             q_pos=(q > 0).sum(), q_n=len(q)))
        # time-series (market timing) IC: pooled across coins of own-feature vs own fwd return, per-coin demeaned feature
        Xd = Xs.where(seg(U, a, b)); Xd = (Xd - Xd.rolling(720, min_periods=200).mean())
        # use equal-weight market return as target for ts-test on cross-sectional average feature
mt = pd.DataFrame(rows); mt['abs_t'] = mt.t.abs()
mt.to_csv('screen_disc_xs.csv', index=False)
pd.set_option('display.width', 200)
print(mt.sort_values('abs_t', ascending=False).head(45).round(4).to_string())
print('n tests', len(mt))

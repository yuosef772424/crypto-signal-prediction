"""
PURPOSE:  Builds the dict of hourly price/positioning features (momentum, vol, distance to extremes, OI change,
          long/short ratios, taker flow) from the panel.
TAGS:     features, build, momentum, volatility, dist_hi/lo, OI change, ls_acc, tt_pos, taker_z, smart-minus-retail,
          feature panel
PITFALLS: All features are trailing (causal) but must be paired with fwd() entry at t+1; z-scores use 720h windows.
"""
import numpy as np, pandas as pd
def build(P):
    L = P['L']; r1 = L.diff()
    vol24 = r1.rolling(24, min_periods=18).std(); vol168 = r1.rolling(168, min_periods=120).std()
    F = {}
    for h in [1, 4, 24, 72, 168, 720]:
        F[f'mom_{h}'] = L.diff(h)
    F['vol24'] = vol24; F['vol168'] = vol168; F['vol_ratio'] = np.log(vol24 / vol168)
    F['dist_hi_168'] = L - L.rolling(168).max(); F['dist_lo_168'] = L - L.rolling(168).min()
    F['dist_hi_720'] = L - L.rolling(720).max()
    lo = np.log(P['oi']); loiv = np.log(P['oiv'])
    for h in [1, 4, 24, 72]:
        F[f'doi_{h}'] = lo.diff(h)
    F['oi_z30'] = (loiv - loiv.rolling(720).mean()) / loiv.rolling(720).std()
    F['oi_to_px_24'] = lo.diff(24) - L.diff(24)
    for k in ['ls_acc', 'tt_acc', 'tt_pos']:
        x = np.log(P[k]); F[k] = x; F[f'{k}_d24'] = x.diff(24)
        F[f'{k}_z'] = (x - x.rolling(720).mean()) / x.rolling(720).std()
    F['smr'] = np.log(P['tt_pos']) - np.log(P['ls_acc']); F['smr_d24'] = F['smr'].diff(24)
    tk = np.log(P['taker'].clip(1e-3, 1e3))
    F['taker_1'] = tk; F['taker_24'] = tk.rolling(24).mean()
    F['taker_z'] = (tk.rolling(4).mean() - tk.rolling(720).mean()) / tk.rolling(720).std()
    F['mom24_x_doi24'] = np.sign(L.diff(24)) * lo.diff(24)
    F['size'] = loiv
    return F

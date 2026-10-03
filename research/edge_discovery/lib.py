"""
PURPOSE:  Shared loaders and eval helpers for the study: load() panel, DISC/VAL/HOLD split constants, fwd(),
          xs_demean(), rank_ic(), tstat().
TAGS:     load panel, DISC VAL HOLD dates, forward return, universe mask, top-50 big, rank_ic, t-stat, shared helpers,
          EDGE_DATA
PITFALLS: Split dates are frozen by the pre-registration (never change); HOLD 2026 is already consumed; universe uses
          OI value >= $10M one hour earlier (look-ahead safe).

Shared data/feature/eval helpers for the edge-discovery study (see 00_PREREGISTRATION.md)."""
import numpy as np, pandas as pd, os
DATA = os.environ.get('EDGE_DATA', '/home/user/research')
DISC = ('2024-01-01', '2025-03-31 23:00')
VAL = ('2025-04-01', '2025-12-31 23:00')
HOLD = ('2026-01-01', '2026-09-30')

def load():
    P = {k: pd.read_parquet(f'{DATA}/panel_{k}.parquet') for k in ['px','oi','oiv','tt_acc','tt_pos','ls_acc','taker']}
    for k in P:  # zeros are missing values in Binance metrics
        P[k] = P[k].where(P[k] > 0)
    P['px'] = P['px'].ffill(limit=2)
    P['L'] = np.log(P['px'])
    # tradable universe: OI value >= $10M one hour earlier, and price present
    P['univ'] = (P['oiv'].shift(1) >= 10e6) & P['px'].notna()
    rank = P['oiv'].shift(1).rank(axis=1, ascending=False)
    P['big'] = rank <= 50
    return P

def fwd(L, H):
    """log return from entry at t+1 to t+1+H (one hour latency)."""
    return L.shift(-1 - H) - L.shift(-1)

def xs_demean(X, univ):
    X = X.where(univ)
    return X.sub(X.mean(axis=1), axis=0)

def rank_ic(F, Y, univ, step=1, min_n=20):
    """per-timestamp Spearman IC between F and Y within universe; returns Series."""
    F = F.where(univ & Y.notna()); Y = Y.where(F.notna())
    F = F.iloc[::step]; Y = Y.iloc[::step]
    rf = F.rank(axis=1); ry = Y.rank(axis=1)
    rf = rf.sub(rf.mean(axis=1), axis=0); ry = ry.sub(ry.mean(axis=1), axis=0)
    num = (rf * ry).sum(axis=1); den = np.sqrt((rf**2).sum(axis=1) * (ry**2).sum(axis=1))
    n = F.notna().sum(axis=1)
    return (num / den).where(n >= min_n).dropna()

def tstat(s):
    s = s.dropna(); return s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan

def seg(df, a, b): return df.loc[a:b]

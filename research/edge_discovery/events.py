"""
PURPOSE:  Event-study helpers: dedup() cooldown per coin and event_stats() with day-clustered t, market adjustment and
          costs.
TAGS:     event study, dedup, cooldown, event_stats, day-clustered t, market adjusted, win rate, per-coin events
PITFALLS: Per-coin events cluster in a few market-wide days: only day-clustered t is meaningful, raw means mislead
          (H04).
"""
import numpy as np, pandas as pd
def dedup(E, cool):
    """keep first event per coin, then suppress further events for `cool` hours."""
    A = E.values.copy(); n, m = A.shape
    for j in range(m):
        last = -10**9
        for i in np.flatnonzero(A[:, j]):
            if i - last < cool: A[i, j] = False
            else: last = i
    return pd.DataFrame(A, index=E.index, columns=E.columns)

def event_stats(E, L, U, H, side=1, mkt_adj=False, cost=0.0015, a=None, b=None):
    """E boolean panel at signal time t; trade entry t+1, exit t+1+H; side=+1 long, -1 short."""
    if a: E = E.loc[a:b]
    E = E & U.reindex(E.index)
    R = (L.shift(-1 - H) - L.shift(-1)).reindex(E.index)
    R = np.expm1(R)
    if mkt_adj:
        R = R.sub(np.expm1((L.shift(-1 - H) - L.shift(-1)).reindex(E.index)).where(U.reindex(E.index)).mean(axis=1), axis=0)
    tr = (side * R).where(E).stack().dropna() - cost
    if len(tr) < 10: return None
    day = tr.index.get_level_values(0).floor('D')
    daily = tr.groupby(day).mean()
    w = tr[tr > 0]; l = tr[tr <= 0]
    return dict(n=len(tr), days=len(daily), mean_bp=tr.mean()*1e4, med_bp=tr.median()*1e4, win=(tr > 0).mean(),
                avg_win_bp=w.mean()*1e4 if len(w) else np.nan, avg_loss_bp=l.mean()*1e4 if len(l) else np.nan,
                payoff=(w.mean()/-l.mean()) if len(w) and len(l) else np.nan, t_day=daily.mean()/daily.std()*np.sqrt(len(daily)),
                pf=w.sum()/-l.sum() if len(l) else np.nan)

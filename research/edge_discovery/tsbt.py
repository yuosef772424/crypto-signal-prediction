import numpy as np, pandas as pd
def daily_panel(P):
    """daily closes at 00:00 UTC (the hourly snapshot); universe = OI value >= 10M at that time."""
    px = P['px'].resample('D').first()       # value at 00:00
    oiv = P['oiv'].resample('D').first()
    return px, oiv
def stats(ret, per_year=365):
    ret = ret.dropna(); 
    if len(ret) < 5: return {}
    eq = (1 + ret).cumprod(); dd = (eq / eq.cummax() - 1).min()
    neg = ret[ret < 0]
    ls = (ret < 0).astype(int); streak = (ls.groupby((ls != ls.shift()).cumsum()).cumsum()).max()
    return dict(days=len(ret), ann_ret=ret.mean()*per_year, ann_vol=ret.std()*np.sqrt(per_year),
                sharpe=ret.mean()/ret.std()*np.sqrt(per_year), sortino=ret.mean()/neg.std()*np.sqrt(per_year) if len(neg)>1 else np.nan,
                maxdd=dd, total=eq.iloc[-1]-1, t=ret.mean()/ret.std()*np.sqrt(len(ret)), skew=ret.skew(), worst_day=ret.min(),
                longest_losing_days=streak)
def run_weights(W, px, cost_rt):
    """W: target weights decided at day d close (00:00 of d), executed ~01:00 -> approximate with next-day close-to-close.
    returns daily portfolio returns net of costs. cost_rt: per-unit-turnover round-trip cost / 2 per side."""
    r = px.pct_change().shift(-1)          # return from d to d+1 belongs to weights set at d
    gross = (W * r).sum(axis=1, min_count=1)
    turn = (W - W.shift(1).fillna(0)).abs().sum(axis=1)
    net = gross - turn * cost_rt / 2
    return gross, net, turn

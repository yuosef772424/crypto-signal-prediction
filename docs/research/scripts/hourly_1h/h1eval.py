"""Evaluation for the 1h dataset: groups = 32h buckets (anchored at epoch), reusing cross_asset.report where it fits."""
import os
import sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from cross_asset.report import daily_ic, vol_quintile_ic, partial_ic, _t
from sklearn.metrics import roc_auc_score
BUCKET_NS = 32 * 3600 * 10**9
TRAIN_END = pd.Timestamp('2025-06-24', tz='UTC'); VAL_END = pd.Timestamp('2025-11-21', tz='UTC'); GAP = pd.Timedelta(hours=33)

def split_masks(ts_ns):
    ts = pd.to_datetime(np.asarray(ts_ns).astype('int64'), utc=True)
    return {'train': np.asarray(ts <= TRAIN_END), 'val': np.asarray((ts > TRAIN_END + GAP) & (ts <= VAL_END)),
            'test': np.asarray(ts > VAL_END + GAP)}

def prep(df, pvol):
    """df: asset,timestamp(ns),entry,last_high,last_low,fut_close,fut_high,fut_low,p_up_*; pvol: past vol known at entry."""
    d = df.copy()
    d['ts_raw'] = d['timestamp'].astype('int64')
    d['timestamp'] = (d['ts_raw'] // BUCKET_NS) * BUCKET_NS          # report.py groups by 'timestamp'
    d['pvol'] = np.asarray(pvol)
    d['r'] = d['fut_close'] / d['entry'] - 1
    raw = {'close': d['r'], 'high': d['fut_high'] / d['last_high'] - 1, 'low': d['fut_low'] / d['last_low'] - 1}
    for t, v in raw.items():
        d[f'own_{t}'] = (v > 0).astype(int)
        med = v.groupby(d['timestamp']).transform('median')
        d[f'rel_{t}'] = (v - med).clip(-1, 1)
        d[f'cls_{t}'] = (v > med).astype(int)
    return d

def _dec(d, score, n=10):
    return d.groupby('timestamp')[score].transform(lambda x: pd.qcut(x.rank(method='first'), n, labels=False)) + 1

def portfolio(d, score, cost_pct=0.08, min_coins=20):
    """Rank-weighted market-neutral book per bucket (weights as report.mn_portfolio, q=0.5), held 1 bar.
    Positions are 1h and the coin's next sample is 32h later, so the whole book is opened and closed each period:
    turnover = 1 (0.5 long + 0.5 short, round trip) -> net = gross - cost_pct."""
    rows = []
    for _, g in d.groupby('timestamp'):
        if len(g) < min_coins: continue
        c = g[score].rank(method='first').to_numpy(); c = c - (len(g) + 1) / 2
        pos, neg = c.clip(min=0), (-c).clip(min=0)
        w = 0.5 * pos / pos.sum() - 0.5 * neg / neg.sum()
        rows.append(np.dot(w, g['r'].to_numpy()))
    x = np.array(rows) * 100
    gm, gt, n = _t(x); nm, nt, _ = _t(x - cost_pct)
    return gm, gt, nm, nt, n

def evaluate(d, score='p_up_close', min_coins=10):
    m = {}
    for t in ('high', 'low', 'close'):
        s = f'p_up_{t}' if f'p_up_{t}' in d else None
        if s is None: continue
        m[f'auc_own_{t}'] = roc_auc_score(d[f'own_{t}'], d[s])
        m[f'auc_rel_{t}'] = roc_auc_score(d[f'cls_{t}'], d[s])
    m['ic'], m['ic_t'], m['n_groups'] = daily_ic(d, score, 'r', min_coins)
    m['ic_rel'], _, _ = daily_ic(d, score, 'rel_close', min_coins)
    m['volq_ic'], m['volq_ic_t'], _ = vol_quintile_ic(d, score, 'r')
    m['partial_ic'], m['partial_ic_t'], _ = partial_ic(d, score, 'r', min_coins)
    m['score_vs_lowvol'], _, _ = daily_ic(d.assign(nv=-d['pvol']), score, 'nv', min_coins)
    m['mn_gross'], m['mn_gross_t'], m['mn_net'], m['mn_net_t'], m['mn_n'] = portfolio(d, score)
    dec = _dec(d, score)
    up = d.assign(dec=dec).groupby('dec')['own_close'].mean() * 100
    m['up_dec1'], m['up_dec10'] = up.iloc[0], up.iloc[-1]
    m['up_gap'] = up.iloc[-1] - up.iloc[0]
    m['up_mono'] = pd.Series(up.index).corr(up.reset_index(drop=True), method='spearman')
    m['up_by_dec'] = ' '.join(f'{v:.1f}' for v in up)
    # pooled deciles (not per bucket) as secondary
    pdec = pd.qcut(d[score].rank(method='first'), 10, labels=False) + 1
    upp = d.assign(dec=pdec).groupby('dec')['own_close'].mean() * 100
    m['up_pooled_gap'] = upp.iloc[-1] - upp.iloc[0]
    m['up_pooled_mono'] = pd.Series(upp.index).corr(upp.reset_index(drop=True), method='spearman')
    return m

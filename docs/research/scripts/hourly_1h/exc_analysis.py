"""Vol vs direction separation for excursion predictions. Usage: python exc_analysis.py [extra preds pkl with cols <m>_up,<m>_dn]"""
import os
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from cross_asset.report import _t
e = pd.read_pickle('exc_preds.pkl')
for extra in sys.argv[1:]:
    x = pd.read_pickle(extra); e = e.merge(x, on=['asset', 'timestamp'], how='left')
models = sorted({c[:-3] for c in e.columns if c.endswith('_up') and c[:-3] in ('vol', 'full', 'nn', 'pa0', 'pa1', 'pb0', 'pb1')})
e['up_c'] = e.up_exc.clip(lower=0); e['dn_c'] = e.dn_exc.clip(lower=0)
e['asym'] = e.up_c - e.dn_c
def grp_ic(d, s, y):
    ic = d.groupby('bucket').apply(lambda g: g[s].corr(g[y], method='spearman') if len(g) >= 10 else np.nan)
    return _t(ic)
def partial(d, s, ctrl, y):
    def f(g):
        if len(g) < 10: return np.nan
        z = np.c_[np.ones(len(g)), g[ctrl].rank()]
        a, b = (v - z @ np.linalg.lstsq(z, v, rcond=None)[0] for v in (g[s].rank().to_numpy(), g[y].rank().to_numpy()))
        return np.corrcoef(a, b)[0, 1]
    return _t(d.groupby('bucket').apply(f))
def auc_in_q(d, s, ctrl, y, nq=5):
    q = pd.qcut(d[ctrl].rank(method='first'), nq, labels=False)
    v = [roc_auc_score(g[y], g[s]) for _, g in d.groupby(q) if 0 < g[y].mean() < 1]
    return np.mean(v)
rows = {}
for sp in ('val', 'test'):
    d = e[e.split == sp].copy()
    for side, yc in (('up', 'up_c'), ('dn', 'dn_c')):
        k_vol = d.pvol  # per-row vol-scaled threshold: 1x window 1h realized vol
        d[f'{side}_gt05'] = (d[yc] > 0.005).astype(int); d[f'{side}_gt10'] = (d[yc] > 0.01).astype(int)
        d[f'{side}_gtv'] = (d[yc] > k_vol).astype(int)
    for m in models:
        if d[f'{m}_up'].isna().all(): continue
        d[f'{m}_skew'] = np.log(d[f'{m}_up'] + 1e-3) - np.log(d[f'{m}_dn'] + 1e-3)
        r = {}
        for side, yc in (('up', 'up_c'), ('dn', 'dn_c')):
            s = f'{m}_{side}'
            r[f'IC_{side}'] = grp_ic(d, s, yc)[0]
            r[f'IC_{side}_pooled'] = d[s].corr(d[yc], method='spearman')
            for k in ('gt05', 'gt10', 'gtv'):
                r[f'AUC_{side}>{k[2:]}'] = roc_auc_score(d[f'{side}_{k}'], d[s])
            if m != 'vol':
                r[f'pIC_{side}|vol'], r[f'pIC_{side}|vol_t'], _ = partial(d, s, f'vol_{side}', yc)
                r[f'AUCq_{side}>10|volQ'] = auc_in_q(d, s, f'vol_{side}', f'{side}_gt10')
        # asymmetry = direction test
        r['skewIC_asym'], r['skewIC_asym_t'], _ = grp_ic(d, f'{m}_skew', 'asym')
        r['skewIC_r'], r['skewIC_r_t'], _ = grp_ic(d, f'{m}_skew', 'r')
        dec = d.groupby('bucket')[f'{m}_skew'].transform(lambda x: pd.qcut(x.rank(method='first'), 10, labels=False)) + 1
        up = d.assign(dec=dec).groupby('dec').apply(lambda g: (g.r > 0).mean() * 100)
        mr = d.assign(dec=dec).groupby('dec').r.mean() * 100
        r['skew_upshare_d1'], r['skew_upshare_d10'] = up.iloc[0], up.iloc[-1]
        r['skew_upshare_mono'] = pd.Series(range(10)).corr(up.reset_index(drop=True), method='spearman')
        r['skew_r%_d1'], r['skew_r%_d10'] = mr.iloc[0], mr.iloc[-1]
        rows[(m, sp)] = r
T = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(T.astype(float).round(4).to_string())
T.to_pickle('exc_analysis.pkl')

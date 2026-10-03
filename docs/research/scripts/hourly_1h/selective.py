"""Selective direction (next-bar close up/down). Discover on VAL (chicks detect_success_failure_patterns + simple
confidence/agreement rules + tolerant coin filter), freeze, evaluate once on TEST with Bonferroni over all candidates."""
import os
import json, sys, numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from cross_asset.report import _t
from sklearn.tree import DecisionTreeClassifier
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')
ns = {'np': np, 'pd': pd}
exec('from typing import Dict, List, Optional\n' + ''.join(json.load(open(f'{REPO}/chicks_v4_5_input_output_patterns.ipynb'))['cells'][28]['source']), ns)
detect = ns['detect_success_failure_patterns']
e = pd.read_pickle('exc_preds.pkl')
from base_data import load
d, df, closes = load()
fo = d['feature_order']
extra = pd.DataFrame({'asset': df.asset, 'timestamp': df.timestamp, 'rsi': d['X_1h'][:, -1, fo.index('RSI_14')],
                      'trend31': closes[:, -1] / closes[:, 0] - 1, 'ret4': closes[:, -1] / closes[:, -5] - 1})
lg = pd.read_pickle('sig_logreg_own.pkl')[['asset', 'timestamp', 'p_up_close']]
e = e.merge(extra, on=['asset', 'timestamp']).merge(lg, on=['asset', 'timestamp'])
nn = 'nn_heads.pkl'
try:
    h = pd.read_pickle(nn); e = e.merge(h, on=['asset', 'timestamp'], how='left'); HAVE_NN = e.nn_p_high.notna().all()
except FileNotFoundError:
    HAVE_NN = False
e['p'] = e.p_up_close; e['call'] = (e.p > 0.5).astype(int); e['actual'] = (e.r > 0).astype(int)
e['correct'] = (e.call == e.actual).astype(int); e['confidence'] = (e.p - 0.5).abs()
e['pred_range'] = e.full_up + e.full_dn; e['hour'] = pd.to_datetime(e.timestamp.astype('int64')).dt.hour
e['month'] = pd.to_datetime(e.timestamp.astype('int64')).dt.strftime('%Y-%m')
feats = ['confidence', 'pred_range', 'pvol', 'trend31', 'ret4', 'rsi', 'hour']
if HAVE_NN:
    e['heads_agree'] = (((e.nn_p_high > 0.5) & (e.nn_p_low > 0.5) & (e.call == 1)) | ((e.nn_p_high < 0.5) & (e.nn_p_low < 0.5) & (e.call == 0))).astype(int)
    feats.append('heads_agree')
V, T = e[e.split == 'val'].copy(), e[e.split == 'test'].copy()
# tolerant coin filter from VAL: drop coins whose val accuracy is clearly below 50% (binomial p<0.05), keep >=70% coins
acc = V.groupby('asset').correct.agg(['mean', 'size', 'sum'])
acc['p'] = [binomtest(int(s), int(n), 0.5, alternative='less').pvalue for s, n in zip(acc['sum'], acc['size'])]
drop = acc[(acc.p < 0.05)].sort_values('mean').index[: int(0.3 * len(acc))]
print(f'coin filter: {len(drop)} of {len(acc)} coins dropped (val acc {acc.loc[drop, "mean"].round(3).tolist()})')
res = detect(V[feats + ['correct', 'asset']], feature_cols=feats, verbose=False)
print('VAL tree acc %.3f | balanced %.3f | naive %.3f' % (res['tree_accuracy'], res['tree_balanced_accuracy'], res['tree_naive_baseline_accuracy']))
print(res['feature_importance'].head(6).to_string())
# identical tree (same hyper-parameters as detect_success_failure_patterns, without the asset one-hot) to freeze its leaves
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(10, len(V) // 50), class_weight='balanced', random_state=42).fit(V[feats], V.correct)
V['leaf'], T['leaf'] = tree.apply(V[feats]), tree.apply(T[feats])
leaf_acc = V.groupby('leaf').correct.agg(['mean', 'size'])
good = leaf_acc[(leaf_acc['mean'] >= 0.55)].index.tolist()
print('val leaves with acc>=0.55:', leaf_acc.loc[good].round(3).to_dict('index'))

def trade(sub, cost):
    """1h trade in the called direction: entry at P, exit at next close, round-trip cost (%)."""
    g = np.where(sub.call == 1, sub.r, -sub.r) * 100
    nb = T.groupby('bucket').size()
    per = (pd.Series(g - cost).groupby(sub.bucket.to_numpy()).sum().reindex(nb.index, fill_value=0) / nb)
    mon = pd.Series(g - cost).groupby(sub.month.to_numpy()).mean()
    return g.mean(), (g - cost).mean(), _t(per)[1], f'{(mon > 0).sum()}/{len(mon)}'
cands = {}
for q in (0.5, 0.2, 0.1, 0.05):
    thr = V.confidence.quantile(1 - q)                                   # threshold frozen on VAL
    cands[f'conf_top{int(q*100)}%'] = lambda x, thr=thr: x.confidence >= thr
    if HAVE_NN:
        cands[f'conf_top{int(q*100)}%+agree'] = lambda x, thr=thr: (x.confidence >= thr) & (x.heads_agree == 1)
if good:
    cands['tree_good_leaves'] = lambda x: x.leaf.isin(good)
rows = []
K = 2 * len(cands)                                                        # x2: with/without coin filter
for name, f in cands.items():
    for filt in (False, True):
        for sp, D in (('val', V), ('test', T)):
            m = f(D) & (~D.asset.isin(drop) if filt else True)
            s = D[m]
            if len(s) < 20: continue
            base = max(s.actual.mean(), 1 - s.actual.mean())
            k = int(s.correct.sum()); n = len(s)
            p = binomtest(k, n, base, alternative='greater').pvalue
            gm, nm, nt, mp = trade(s, 0.08) if sp == 'test' else (np.nan,) * 4
            rows.append(dict(rule=name, coin_filter=filt, split=sp, n=n, cover=n / len(D), acc=s.correct.mean(), base=base,
                             p_bonf=min(1, p * K), gross_trade=gm, net_trade=nm, net_t=nt, months_pos=mp))
R = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print('candidates K =', K); print(R.round(4).to_string())
R.to_pickle('selective_res.pkl')

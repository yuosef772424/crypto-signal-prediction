"""Evaluate all signal files with the same metrics (h1eval), on identical (asset,timestamp) rows."""
import sys, glob, os, numpy as np, pandas as pd
from base_data import load
from h1eval import prep, evaluate, split_masks
d, df, closes = load()
key = df[['asset', 'timestamp', 'pvol']].copy(); key['timestamp'] = key['timestamp'].astype('int64')
runs = {}
def add(name, vpath, tpath):
    runs[name] = {}
    for sp, p in (('val', vpath), ('test', tpath)):
        s = pd.read_csv(p) if p.endswith('.gz') else pd.read_pickle(p)
        s['timestamp'] = s['timestamp'].astype('int64')
        runs[name][sp] = s
for lab in ('logreg_cls', 'lgbm_cls', 'logreg_own', 'lgbm_own'):
    s = pd.read_pickle(f'sig_{lab}.pkl'); s['timestamp'] = s['timestamp'].astype('int64')
    M = split_masks(s.timestamp.values)
    runs[lab] = {'val': s[M['val']], 'test': s[M['test']]}
rsi = df.copy(); rsi['timestamp'] = rsi.timestamp.astype('int64')
for t in ('high', 'low', 'close'):
    rsi[f'p_up_{t}'] = -d['X_1h'][:, -1, d['feature_order'].index('RSI_14')]
M = split_masks(rsi.timestamp.values); runs['rule_-RSI'] = {'val': rsi[M['val']], 'test': rsi[M['test']]}
for spec in sys.argv[1:]:
    name, folder = spec.split('=')
    add(name, os.path.join(folder, 'signals_val.csv.gz'), os.path.join(folder, 'signals_test.csv.gz'))
res = {}
ref = None
for name, r in runs.items():
    for sp in ('val', 'test'):
        s = r[sp].drop(columns=[c for c in ('pvol',) if c in r[sp]]).merge(key, on=['asset', 'timestamp'], how='inner')
        if sp == 'test':
            print(f'{name}: test rows {len(s)} (file {len(r[sp])})')
        s = s.sort_values(['timestamp', 'asset']).reset_index(drop=True)
        res[(name, sp)] = evaluate(prep(s, s['pvol'].values))
T = pd.DataFrame(res).T
T.to_pickle('all_res.pkl')
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 50)
cols = ['auc_own_high', 'auc_own_low', 'auc_own_close', 'auc_rel_high', 'auc_rel_low', 'auc_rel_close', 'ic', 'ic_t', 'volq_ic', 'volq_ic_t',
        'partial_ic', 'partial_ic_t', 'score_vs_lowvol', 'mn_gross', 'mn_gross_t', 'mn_net', 'mn_net_t', 'up_dec1', 'up_dec10', 'up_gap', 'up_mono', 'n_groups']
for sp in ('test', 'val'):
    print(f'==== {sp}'); print(T.xs(sp, level=1)[cols].astype(float).round(4).T.to_string())
print(T.xs('test', level=1)['up_by_dec'].to_string())

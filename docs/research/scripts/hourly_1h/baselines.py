import time, numpy as np, pandas as pd
from base_data import load
from h1eval import split_masks, prep, evaluate
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
d, df, closes = load()
X = d['X_1h']; fo = d['feature_order']
M = split_masks(df['timestamp'].values)
for k, v in M.items(): 
    ts = pd.to_datetime(df.timestamp[v].astype('int64'), utc=True)
    print(k, v.sum(), ts.min(), ts.max(), 'coins', df.asset[v].nunique())
full = prep(df, df['pvol'])
# bucket stats
for k in ('train','val','test'):
    g = full[M[k]].groupby('timestamp').size()
    print(k, 'buckets', len(g), 'coins/bucket min/med/max', g.min(), g.median(), g.max(), 'dup asset-bucket', full[M[k]].duplicated(['asset','timestamp']).sum())
print('class balance (own) by split:', {k: full.loc[M[k], ['own_high','own_low','own_close']].mean().round(3).to_dict() for k in M})
print('class balance (rel, 32h bucket):', {k: full.loc[M[k], ['cls_high','cls_low','cls_close']].mean().round(3).to_dict() for k in M})

# ---- trivial rules (leakage / mechanics) on test + val ----
last = X[:, -1, :]
rules = {
  'close_vs_window_median (-(last_close-bp0)/bp1)': -(df.entry - d['base_params'][:, 0]) / d['base_params'][:, 1],
  'last_bar_return (reversal: -r_last)': -(closes[:, -1] / closes[:, -2] - 1),
  'upper_wick -(last_high/last_close-1)': -(df.last_high / df.entry - 1),
  'lower_wick -(last_close/last_low-1)': -(df.entry / df.last_low - 1),
  'window vol (pvol)': df.pvol,
}
for i, f in enumerate(fo):
    rules[f'lastfeat {f}'] = last[:, i]
rows = []
for name, s in rules.items():
    s = np.asarray(s, float)
    if np.nanstd(s) == 0: rows.append((name, 'constant', '', '', '')); continue
    r = [name]
    for t in ('high', 'low', 'close'):
        a = roc_auc_score(full.loc[M['test'], f'own_{t}'], s[M['test']])
        r.append(round(max(a, 1 - a), 3) * (1 if a >= 0.5 else -1))
    a = roc_auc_score(full.loc[M['val'], 'own_close'], s[M['val']]); r.append(round(a, 3))
    rows.append(tuple(r))
R = pd.DataFrame(rows, columns=['rule', 'test AUC own_high (sign=dir)', 'own_low', 'own_close', 'val own_close'])
print(R.to_string())
R.to_csv('rules.csv', index=False)

# ---- models on last-step features ----
keep = [i for i in range(len(fo)) if last[M['train'], i].std() > 0]
print('dropped constant features:', [fo[i] for i in range(len(fo)) if i not in keep])
F = np.c_[last[:, keep], df.pvol.values, closes[:, -1] / closes[:, -2] - 1, df.last_high / df.entry - 1, df.entry / df.last_low - 1]
fnames = [fo[i] for i in keep] + ['pvol', 'r_last', 'uwick', 'lwick']
res = {}
def run(label, mk, target_prefix):
    out = df.copy()
    t0 = time.time()
    for t in ('high', 'low', 'close'):
        y = full[f'{target_prefix}_{t}'].values
        m = mk(); m.fit(F[M['train']], y[M['train']])
        out[f'p_up_{t}'] = m.predict_proba(F)[:, 1]
    sig = prep(out, df.pvol)
    for sp in ('val', 'test'):
        res[(label, sp)] = evaluate(sig[M[sp]].reset_index(drop=True))
    res[(label, 'test')]['sec'] = round(time.time() - t0)
    out.loc[M['val'] | M['test']].to_pickle(f'sig_{label}.pkl')
from sklearn.pipeline import make_pipeline
for tp in ('own', 'cls'):
    run(f'logreg_{tp}', lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)), tp)
    run(f'lgbm_{tp}', lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0), tp)
# majority
out = df.copy()
for t in ('high','low','close'):
    out[f'p_up_{t}'] = 0.5
sig = prep(out, df.pvol)
T = pd.DataFrame({k: v for k, v in res.items()}).T
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(T.drop(columns=['up_by_dec']).round(4).to_string())
print(T['up_by_dec'].to_string())
T.to_pickle('baselines_res.pkl')
maj = {sp: {t: max(full.loc[M[sp], f'own_{t}'].mean(), 1 - full.loc[M[sp], f'own_{t}'].mean()) for t in ('high','low','close')} for sp in ('val','test')}
print('majority accuracy (own):', maj)

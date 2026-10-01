"""Own-direction high/low class AUC on VAL/TEST: H > last_high ('high up'), L > last_low ('low up').
Compares NN/panel heads with trivial mechanics (wick of the last bar) and a vol-only LightGBM classifier."""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from base_data import load
from h1eval import split_masks
d, df, closes = load()
df['own_high'] = (df.fut_high > df.last_high).astype(int); df['own_low'] = (df.fut_low > df.last_low).astype(int)
df['timestamp'] = df.timestamp.astype('int64')
M = split_masks(df.timestamp.values)
X = d['X_1h']; fo = d['feature_order']; F = lambda n: X[:, :, fo.index(n)]
lr = np.diff(np.log(np.maximum(closes, 1e-12)), axis=1)
vol = np.c_[df.pvol, lr[:, -8:].std(1), np.abs(lr[:, -4:]).mean(1), F('NATR_14')[:, -1], F('RANGE_rel')[:, -1], F('RANGE_rel')[:, -4:].mean(1),
            F('BBB_20_2.0')[:, -1], F('VOL_TERM_3_30')[:, -1], (df.last_high - df.last_low) / df.entry]
scores = {'wick_rule': {'high': -(df.last_high / df.entry - 1), 'low': (1 - df.last_low / df.entry) * -1 * -1}}
scores['wick_rule']['low'] = -(df.entry / df.last_low - 1)   # big lower wick -> next low below last_low
for t in ('high', 'low'):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, min_child_samples=200, subsample=0.8, subsample_freq=1, verbose=-1, random_state=0)
    m.fit(vol[M['train']], df[f'own_{t}'][M['train']]); scores.setdefault('vol_only_lgbm', {})[t] = m.predict_proba(vol)[:, 1]
base = df[['asset', 'timestamp', 'own_high', 'own_low']]
rows = {}
for name, sc in scores.items():
    for sp in ('val', 'test'):
        rows[(name, sp)] = {f'AUC_{t}': roc_auc_score(df[f'own_{t}'][M[sp]], np.asarray(sc[t])[M[sp]]) for t in ('high', 'low')}
for lab in ('lgbm_own', 'logreg_own'):
    s = pd.read_pickle(f'sig_{lab}.pkl'); s['timestamp'] = s.timestamp.astype('int64')
    s = s.merge(base, on=['asset', 'timestamp']); Ms = split_masks(s.timestamp.values)
    for sp in ('val', 'test'):
        rows[(lab, sp)] = {f'AUC_{t}': roc_auc_score(s[f'own_{t}'][Ms[sp]], s[f'p_up_{t}'][Ms[sp]]) for t in ('high', 'low')}
for spec in sys.argv[1:]:
    name, folder = spec.split('=')
    for sp in ('val', 'test'):
        s = pd.read_csv(os.path.join(folder, f'signals_{sp}.csv.gz')); s['timestamp'] = s.timestamp.astype('int64')
        s = s.merge(base, on=['asset', 'timestamp'])
        rows[(name, sp)] = {f'AUC_{t}': roc_auc_score(s[f'own_{t}'], s[f'p_up_{t}']) for t in ('high', 'low')}
        rows[(name, sp)]['n'] = len(s)
print(pd.DataFrame(rows).T.round(4).to_string())

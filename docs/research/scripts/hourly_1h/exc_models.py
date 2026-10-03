"""Excursion targets up_exc = H/P-1, dn_exc = 1-L/P (P = last close). LightGBM vol-only vs full features.
Writes exc_preds.pkl: per-row pred_up/pred_dn for each model (val+test rows), plus realized columns."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from base_data import load
from h1eval import split_masks, BUCKET_NS
d, df, closes = load()
X = d['X_1h']; fo = d['feature_order']; F = lambda n: X[:, :, fo.index(n)]
P = df.entry.values
df['up_exc'] = df.fut_high / df.entry - 1
df['dn_exc'] = 1 - df.fut_low / df.entry
df['r'] = df.fut_close / df.entry - 1
lr = np.diff(np.log(np.maximum(closes, 1e-12)), axis=1)
vol = pd.DataFrame({
    'pvol': df.pvol, 'pvol8': lr[:, -8:].std(1), 'absr4': np.abs(lr[:, -4:]).mean(1),
    'natr': F('NATR_14')[:, -1], 'range_last': F('RANGE_rel')[:, -1], 'range_m4': F('RANGE_rel')[:, -4:].mean(1),
    'range_m32': F('RANGE_rel').mean(1), 'bbb': F('BBB_20_2.0')[:, -1], 'volterm': F('VOL_TERM_3_30')[:, -1],
    'lastbar_hl': (df.last_high - df.last_low) / df.entry, 'hour': pd.to_datetime(df.timestamp.astype('int64')).dt.hour})
full = vol.copy()
for i, n in enumerate(fo):
    if X[:, :, i].std() > 0: full[f'l_{n}'] = X[:, -1, i]
full['uwick'] = df.last_high / df.entry - 1; full['lwick'] = 1 - df.last_low / df.entry
for k in (1, 4, 8, 31): full[f'ret{k}'] = closes[:, -1] / closes[:, -1 - k] - 1
full['vol_z'] = F('volume')[:, -1]
M = split_masks(df.timestamp.values)
out = df[['asset', 'timestamp', 'entry', 'last_high', 'last_low', 'fut_close', 'fut_high', 'fut_low', 'pvol', 'up_exc', 'dn_exc', 'r']].copy()
out['bucket'] = (out.timestamp.astype('int64') // BUCKET_NS) * BUCKET_NS
EPS = 1e-3
for name, feats in (('vol', vol), ('full', full)):
    t0 = time.time()
    for tgt in ('up_exc', 'dn_exc'):
        y = np.log(np.maximum(df[tgt].values, 0) + EPS)   # gap below/above last close -> 0 excursion
        m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0)
        m.fit(feats[M['train']], y[M['train']])
        out[f'{name}_{tgt[:2]}'] = np.exp(m.predict(feats)) - EPS
    print(name, 'features', feats.shape[1], 'sec', round(time.time() - t0))
out['split'] = np.where(M['val'], 'val', np.where(M['test'], 'test', 'train'))
out[out.split != 'train'].to_pickle('exc_preds.pkl')

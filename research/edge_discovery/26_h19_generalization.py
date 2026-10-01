import re, numpy as np, pandas as pd, sys
sys.argv = ['x']
exec(open('25_image_cnn.py').read().split("if __name__ == '__main__':")[0])
# --- CNN: parse log -> per config, per seed: val AUC at chosen epoch (min val_loss), all-epoch AUC range
log = open('/home/user/research/h19_log.txt').read().splitlines()
cfg = None; rec = {}
for ln in log:
    m = re.match(r'\[(\w+) (C\d)\]', ln)
    if m: cfg = (m.group(1), m.group(2)); rec[cfg] = {}; continue
    m = re.match(r'\s+seed(\d) ep(\d+) val_loss ([\d.]+) val_auc ([\d.]+)', ln)
    if m and cfg: rec[cfg].setdefault(int(m.group(1)), []).append((float(m.group(3)), float(m.group(4))))
R = pd.read_csv('image_cnn_results.csv')
rows = []
for (tf, f), seeds in rec.items():
    if not ((R.tf == tf) & (R.filt == f)).any(): continue
    chosen = [min(v)[1] for v in seeds.values()]            # AUC at min val_loss epoch
    allauc = [a for v in seeds.values() for _, a in v]
    cnn = R[(R.tf == tf) & (R.filt == f) & (R.model == 'cnn')].iloc[0]
    rows.append(dict(tf=tf, filt=f, model='CNN images', train_auc=np.nan, val_auc=np.mean(chosen), test_auc=cnn.auc,
                     seed_spread=max(chosen) - min(chosen), epoch_range=max(allauc) - min(allauc), coins_pos=cnn.coins_pos,
                     net_2324=cnn.net_2324, net_2526=cnn.net_2526))
    # --- logistic: recompute train/val/test AUC on the identical samples
    Ps = [prep(s, tf) for s in COINS]
    tr = split_idx(Ps, f, '2017-01-01', '2021-12-31 23:59', 2 if tf == '1h' else 1)
    va = split_idx(Ps, f, '2022-01-01', '2022-12-31 23:59', 1); te = split_idx(Ps, f, '2023-01-01', '2026-12-31', 1)
    def XY(idx): return (np.concatenate([numeric(Ps[c], idx[idx[:, 0] == c, 1]) for c in range(10)]),
                         np.concatenate([Ps[c]['fwd'][idx[idx[:, 0] == c, 1]] > 0 for c in range(10)]))
    Xtr, ytr = XY(tr); Xva, yva = XY(va); Xte, yte = XY(te)
    sc = StandardScaler().fit(Xtr); lr = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(Xtr), ytr)
    a = lambda X, y: roc_auc_score(y, lr.predict_proba(sc.transform(X))[:, 1])
    lg = R[(R.tf == tf) & (R.filt == f) & (R.model == 'logit')].iloc[0]
    rows.append(dict(tf=tf, filt=f, model='Logistic numbers', train_auc=a(Xtr, ytr), val_auc=a(Xva, yva), test_auc=a(Xte, yte),
                     seed_spread=0.0, epoch_range=0.0, coins_pos=lg.coins_pos, net_2324=lg.net_2324, net_2526=lg.net_2526))
G = pd.DataFrame(rows); G['val_to_test_drop'] = G.val_auc - G.test_auc
pd.set_option('display.width', 250); print(G.round(3).to_string())
print('\nmean over configs:'); print(G.groupby('model')[['val_auc', 'test_auc', 'val_to_test_drop', 'seed_spread', 'epoch_range', 'coins_pos']].mean().round(3).to_string())
G.to_csv('h19_generalization.csv', index=False)

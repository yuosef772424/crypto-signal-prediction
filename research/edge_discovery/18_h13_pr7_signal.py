"""H13: PR #7 relative-direction signal (linear reference) -> daily long/short decile P&L (Addendum C)."""
import numpy as np, pandas as pd, glob, os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tsbt import stats
CM = '/home/user/research/coinmetrics'
px, vol = {}, {}
for f in sorted(glob.glob(f'{CM}/*.csv')):
    a = os.path.basename(f)[:-4]
    d = pd.read_csv(f, low_memory=False)
    if 'PriceUSD' not in d: continue
    d = d.set_index(pd.to_datetime(d['time']))
    s = pd.to_numeric(d['PriceUSD'], errors='coerce'); s = s[(s.index >= '2018-01-01') & (s > 0)].dropna()
    if len(s) < 700: continue
    px[a] = s
    if 'volume_reported_spot_usd_1d' in d: vol[a] = pd.to_numeric(d['volume_reported_spot_usd_1d'], errors='coerce')
px = pd.DataFrame(px).sort_index(); vol = pd.DataFrame(vol).reindex(index=px.index, columns=px.columns)
print('universe:', px.shape[1], 'assets', px.index[0].date(), '->', px.index[-1].date())
# --- PR #7 build_features (same logic) ---
lr = np.log(px).diff(); lr = lr.where(lr.abs() < 1.5); mkt = lr.mean(axis=1); btc = lr['btc']
def rsi(r, n=14):
    up = r.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-r.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return up / (up + dn + 1e-12)
rows = []
fut = lr.shift(-1); dirt = fut.sub(fut.median(axis=1), axis=0)
for a in px.columns:
    r = lr[a]; lv = np.log(vol[a].where(vol[a] > 0))
    f = pd.DataFrame({'RET_1': r, 'RET_5': r.rolling(5).sum(), 'RET_20': r.rolling(20).sum(), 'VOL_20': r.rolling(20).std(),
                      'NATR_PROXY_14': r.abs().rolling(14).mean(), 'RSI_14': rsi(r) - .5,
                      'MA_DIST_20': px[a] / px[a].rolling(20).mean() - 1,
                      'VOLUME_Z': (lv - lv.rolling(20).mean()) / (lv.rolling(20).std() + 1e-6),
                      'MKT_RET_1': mkt, 'BTC_RET_1': btc, 'REL_RET_1': r - mkt})
    f['y'] = dirt[a]; f['fwd'] = np.expm1(fut[a]); f['asset'] = a
    rows.append(f)
D = pd.concat(rows).replace([np.inf, -np.inf], np.nan)
D.index.name = 'time'
X = [c for c in D.columns if c not in ('y', 'fwd', 'asset')]
D = D.dropna(subset=X + ['y']).sort_index()
tr = D.loc[:'2023-06-28']; te = D.loc['2024-07-01':]
sc = StandardScaler().fit(tr[X]); m = LogisticRegression(max_iter=2000).fit(sc.transform(tr[X]), tr.y > 0)
te = te.assign(p=m.predict_proba(sc.transform(te[X]))[:, 1])
print('test AUC (dir):', round(roc_auc_score(te.y > 0, te.p), 4), '| n test rows', len(te))
print('coefs:', dict(zip(X, m.coef_[0].round(3))))
def port(te, q=0.1, cost=0.0012):
    W = []
    for d, g in te.groupby(level=0):
        if len(g) < 20: continue
        k = max(1, int(round(len(g) * q)))
        g = g.sort_values('p'); w = pd.Series(0.0, index=g.asset.values)
        w[g.asset.iloc[-k:].values] = 1 / k; w[g.asset.iloc[:k].values] = -1 / k
        W.append(w.rename(d))
    W = pd.DataFrame(W).fillna(0)
    R = te.reset_index().pivot_table(index='time', columns='asset', values='fwd').reindex(index=W.index, columns=W.columns).fillna(0)
    turn = (W - W.shift(1).fillna(0)).abs().sum(axis=1)
    return (W * R).sum(axis=1) - turn * cost / 2, turn, W, R
res = []
for q in [0.1, 0.2]:
    for c in [0.0012, 0.0024, 0.0]:
        n, turn, W, R = port(te, q, c); s = stats(n)
        h = len(n) // 2; s.update(q=q, cost=c, turn=turn.mean(), H1=n.iloc[:h].mean()*1e4, H2=n.iloc[h:].mean()*1e4,
                                   mean_bp=n.mean()*1e4, long_bp=(W.clip(lower=0)*R).sum(axis=1).mean()*1e4,
                                   short_bp=(W.clip(upper=0)*R).sum(axis=1).mean()*1e4)
        res.append(s)
print(pd.DataFrame(res)[['q', 'cost', 'days', 'mean_bp', 't', 'sharpe', 'maxdd', 'turn', 'H1', 'H2', 'long_bp', 'short_bp', 'skew']].round(3).to_string())

"""H19: chart-image CNN on 1h/4h with sample filters (Addendum H). Resumable: results appended per config."""
import numpy as np, pandas as pd, os, sys, time, json
import torch, torch.nn as nn
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
D = os.environ.get('DATA_DIR', '/home/user/research'); OUT = os.environ.get('OUT_CSV', 'image_cnn_results.csv')
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'DOGEUSDT', 'TRXUSDT', 'XRPUSDT', 'ZECUSDT']
W, PH, VH = 32, 48, 12; IMH, IMW = PH + VH, 3 * W
COST = 0.0007
torch.set_num_threads(4)

def ema(x, n): return x.ewm(span=n, adjust=False).mean()
def prep(sym, tf):
    m5 = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    b = m5.resample({'1h': '1h', '4h': '4h'}[tf]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    b = b[b.high > b.low]
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    a14, a100 = tr.rolling(14).mean(), tr.rolling(100).mean()
    e20, e50 = ema(b.close, 20), ema(b.close, 50)
    H = 8 if tf == '1h' else 6
    fwd = np.log(b.open.shift(-1 - H) / b.open.shift(-1))            # entry next open, exit H bars later
    F = {'C0': np.ones(len(b), bool), 'C1': (a14 / a100 > 1.2).values, 'C2': ((e20 - e50).abs() > 1.5 * a14).values,
         'C3': (b.volume > 2 * b.volume.shift().rolling(20).mean()).values}
    ok = (np.arange(len(b)) >= max(W, 100)) & fwd.notna().values & a100.notna().values
    return dict(t=b.index, o=b.open.values, h=b.high.values, l=b.low.values, c=b.close.values, v=b.volume.values,
                e=e20.values, fwd=fwd.values, ok=ok, F=F)

@njit
def render(o, h, l, c, v, e, ends):
    n = len(ends); X = np.zeros((n, IMH, IMW), np.uint8)
    for k in range(n):
        t = ends[k]; s = t - W + 1
        lo = l[s]; hi = h[s]; vm = 0.0
        for j in range(s, t + 1):
            if l[j] < lo: lo = l[j]
            if h[j] > hi: hi = h[j]
            if v[j] > vm: vm = v[j]
        rg = hi - lo
        if rg <= 0: continue
        for j in range(W):
            i = s + j; cb = 3 * j
            yo = PH - 1 - int(round((o[i] - lo) / rg * (PH - 1))); yh = PH - 1 - int(round((h[i] - lo) / rg * (PH - 1)))
            yl = PH - 1 - int(round((l[i] - lo) / rg * (PH - 1))); yc = PH - 1 - int(round((c[i] - lo) / rg * (PH - 1)))
            X[k, yo, cb] = 1; X[k, yc, cb + 2] = 1
            for y in range(yh, yl + 1): X[k, y, cb + 1] = 1
            if e[i] >= lo and e[i] <= hi:
                ye = PH - 1 - int(round((e[i] - lo) / rg * (PH - 1)))
                X[k, ye, cb] = 1; X[k, ye, cb + 1] = 1; X[k, ye, cb + 2] = 1
            if vm > 0:
                hv = int(round(v[i] / vm * (VH - 1)))
                for y in range(IMH - 1 - hv, IMH): X[k, y, cb + 1] = 1
    return X

def numeric(P, ends):                                        # same window as numbers, normalised like the image
    out = np.zeros((len(ends), 5 * W), np.float32)
    for k, t in enumerate(ends):
        s = t - W + 1; lo = P['l'][s:t + 1].min(); hi = P['h'][s:t + 1].max(); rg = hi - lo
        for q, key in enumerate(['o', 'h', 'l', 'c']): out[k, q * W:(q + 1) * W] = (P[key][s:t + 1] - lo) / rg
        vv = P['v'][s:t + 1]; out[k, 4 * W:] = vv / (vv.max() + 1e-12)
    return out

class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, (5, 3), padding=(2, 1)), nn.BatchNorm2d(o), nn.LeakyReLU(0.01), nn.MaxPool2d(2))
        s.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64)); s.d = nn.Dropout(0.5)
        s.o = nn.Linear(64 * (IMH // 8) * (IMW // 8), 1)
    def forward(s, x): return s.o(s.d(s.f(x).flatten(1))).squeeze(1)

def batches(Ps, idx, bs, shuffle, rng=None):
    order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    for i in range(0, len(order), bs):
        sel = idx[order[i:i + bs]]; xs = []; ys = []
        for ci in np.unique(sel[:, 0]):
            m = sel[sel[:, 0] == ci]; P = Ps[ci]
            xs.append(render(P['o'], P['h'], P['l'], P['c'], P['v'], P['e'], m[:, 1].astype(np.int64)))
            ys.append((P['fwd'][m[:, 1]] > 0).astype(np.float32))
        yield torch.from_numpy(np.concatenate(xs)[:, None].astype(np.float32)), torch.from_numpy(np.concatenate(ys)), np.concatenate([sel[sel[:, 0] == ci] for ci in np.unique(sel[:, 0])])

def predict(model, Ps, idx):
    model.eval(); ps = []; ids = []
    with torch.no_grad():
        for x, y, s in batches(Ps, idx, 1024, False):
            ps.append(torch.sigmoid(model(x)).numpy()); ids.append(s)
    return np.concatenate(ids), np.concatenate(ps)

def split_idx(Ps, filt, a, b, stride):
    rows = []
    for ci, P in enumerate(Ps):
        m = P['ok'] & P['F'][filt] & (P['t'] >= a) & (P['t'] <= b)
        ii = np.where(m)[0][::stride]; rows.append(np.c_[np.full(len(ii), ci), ii])
    return np.concatenate(rows).astype(np.int64)

def evaluate(Ps, ids, p, lo_q, hi_q, cost):
    rec = []
    for (ci, t), pp in zip(ids, p):
        if pp >= hi_q or pp <= lo_q:
            d = 1 if pp >= hi_q else -1; P = Ps[ci]
            rec.append((P['t'][t], COINS[ci], np.expm1(P['fwd'][t]) if d == 1 else -np.expm1(P['fwd'][t])))
    T = pd.DataFrame(rec, columns=['t', 'sym', 'gross'])
    T['net'] = T.gross - 2 * cost
    return T

def ctstat(x, t):
    g = pd.DataFrame({'r': x, 'd': pd.to_datetime(t).floor('1D')}).groupby('d').r.agg(['sum', 'count'])
    mu = x.mean(); se = np.sqrt(((g['sum'] - g['count'] * mu) ** 2).sum()) / len(x)
    return mu / se if se > 0 else np.nan

def run(tf, filt):
    Ps = [prep(s, tf) for s in COINS]
    tr = split_idx(Ps, filt, '2017-01-01', '2021-12-31 23:59', 2 if tf == '1h' else 1)
    va = split_idx(Ps, filt, '2022-01-01', '2022-12-31 23:59', 1)
    te = split_idx(Ps, filt, '2023-01-01', '2026-12-31', 1)
    print(f'[{tf} {filt}] train {len(tr)} val {len(va)} test {len(te)}', flush=True)
    pv_all, pt_all = [], []
    for seed in [0, 1]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = CNN(); opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
        best = 1e9; bad = 0; state = None
        for ep in range(8):
            model.train(); t0 = time.time()
            for x, y, _ in batches(Ps, tr, 256, True, rng):
                opt.zero_grad(); l = lossf(model(x), y); l.backward(); opt.step()
            iv, pv = predict(model, Ps, va)
            yv = np.array([Ps[c]['fwd'][t] > 0 for c, t in iv]); vl = -np.mean(yv * np.log(pv + 1e-7) + (1 - yv) * np.log(1 - pv + 1e-7))
            print(f'  seed{seed} ep{ep} val_loss {vl:.4f} val_auc {roc_auc_score(yv, pv):.4f} ({time.time() - t0:.0f}s)', flush=True)
            if vl < best - 1e-4: best = vl; bad = 0; state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 2: break
        model.load_state_dict(state)
        iv, pv = predict(model, Ps, va); it, pt = predict(model, Ps, te)
        pv_all.append(pd.Series(pv, index=pd.MultiIndex.from_arrays(iv.T))); pt_all.append(pd.Series(pt, index=pd.MultiIndex.from_arrays(it.T)))
    pv = pd.concat(pv_all, axis=1).mean(axis=1); pt = pd.concat(pt_all, axis=1).mean(axis=1)
    res = []
    for name, (vv, tt) in {'cnn': (pv, pt)}.items():
        lo_q, hi_q = np.quantile(vv.values, [0.2, 0.8])
        ids = np.array(list(tt.index)); T = evaluate(Ps, ids, tt.values, lo_q, hi_q, COST)
        yt = np.array([Ps[c]['fwd'][t] > 0 for c, t in ids]); res.append((name, T, roc_auc_score(yt, tt.values)))
    # numeric logistic baseline on the same samples
    Xtr = np.concatenate([numeric(Ps[c], tr[tr[:, 0] == c, 1]) for c in range(len(Ps))]); ytr = np.concatenate([Ps[c]['fwd'][tr[tr[:, 0] == c, 1]] > 0 for c in range(len(Ps))])
    sc = StandardScaler().fit(Xtr); lr = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(Xtr), ytr)
    def lp(idx):
        ids = np.concatenate([idx[idx[:, 0] == c] for c in range(len(Ps))])
        X = np.concatenate([numeric(Ps[c], idx[idx[:, 0] == c, 1]) for c in range(len(Ps))]); return ids, lr.predict_proba(sc.transform(X))[:, 1]
    ivl, pvl = lp(va); itl, ptl = lp(te); lo_q, hi_q = np.quantile(pvl, [0.2, 0.8])
    T = evaluate(Ps, itl, ptl, lo_q, hi_q, COST); yt = np.array([Ps[c]['fwd'][t] > 0 for c, t in itl])
    res.append(('logit', T, roc_auc_score(yt, ptl)))
    rows = []
    for name, T, auc in res:
        r = dict(tf=tf, filt=filt, model=name, auc=auc, trades=len(T), gross_bp=T.gross.mean() * 1e4, net_bp=T.net.mean() * 1e4,
                 maker_net_bp=(T.gross.mean() - 0.0004) * 1e4, hit=(T.gross > 0).mean(), t=ctstat(T.net.values, T.t.values),
                 coins_pos=int((T.groupby('sym').net.mean() > 0).sum()),
                 net_2324=T[T.t < '2025-01-01'].net.mean() * 1e4, net_2526=T[T.t >= '2025-01-01'].net.mean() * 1e4)
        rows.append(r); T.to_parquet(f'{D}/h19_trades_{tf}_{filt}_{name}.parquet')
    return rows

if __name__ == '__main__':
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT); done = set(zip(d.tf, d.filt))
    for tf in ['4h', '1h']:
        for filt in ['C0', 'C1', 'C2', 'C3']:
            if (tf, filt) in done: continue
            rows = run(tf, filt)
            pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            print(pd.DataFrame(rows).round(3).to_string(), flush=True)

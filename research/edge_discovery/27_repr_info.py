"""H20 step 1 (Addendum I): do tokens / causal wavelet / recurrence (RQA) add information beyond returns & vol?"""
import numpy as np, pandas as pd
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss
D = '/home/user/research'
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'DOGEUSDT', 'TRXUSDT', 'XRPUSDT', 'ZECUSDT']
TR, VA = '2021-12-31 23:59', '2022-12-31 23:59'

@njit
def rqa(z, W, m):
    n = len(z); out = np.full((n, 6), np.nan); S = W - m + 1
    for t in range(W + m, n):
        X = np.empty((S, m))
        for i in range(S):
            for k in range(m): X[i, k] = z[t - W + 1 + i + k]          # state i ends at bar t-W+1+i+m-1
        Dm = np.zeros((S, S)); vals = np.empty(S * (S - 1) // 2); q = 0
        for i in range(S):
            for j in range(i + 1, S):
                d = 0.0
                for k in range(m): d += (X[i, k] - X[j, k]) ** 2
                d = np.sqrt(d); Dm[i, j] = d; Dm[j, i] = d; vals[q] = d; q += 1
        eps = np.sort(vals)[int(0.2 * len(vals))]
        rec = 0; diag_pts = 0; nlines = 0; ltot = 0; hist = np.zeros(S + 1)
        for off in range(1, S):                                     # diagonals of the upper triangle
            run = 0
            for i in range(S - off + 1):
                if i < S - off and Dm[i, i + off] < eps:
                    rec += 1; run += 1
                else:
                    if run >= 2: diag_pts += run; nlines += 1; ltot += run; hist[run] += 1
                    run = 0
        vert = 0
        for j in range(S):                                           # vertical lines (laminarity)
            run = 0
            for i in range(S + 1):
                if i < S and i != j and Dm[i, j] < eps: run += 1
                else:
                    if run >= 2: vert += run
                    run = 0
        ent = 0.0
        if nlines > 0:
            for L in range(2, S + 1):
                if hist[L] > 0:
                    p = hist[L] / nlines; ent -= p * np.log(p)
        last = S - 1; cnt = 0; fut = 0.0
        for j in range(S - 1):
            if Dm[last, j] < eps:
                cnt += 1; fut += z[t - W + 1 + j + m]                    # the bar that followed past state j
        out[t, 0] = diag_pts / rec if rec > 0 else 0.0
        out[t, 1] = vert / (2 * rec) if rec > 0 else 0.0
        out[t, 2] = ltot / nlines if nlines > 0 else 0.0
        out[t, 3] = ent
        out[t, 4] = cnt / (S - 1)
        out[t, 5] = fut / cnt if cnt > 0 else 0.0
    return out

def coin_frame(sym, tf):
    m5 = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    b = m5.resample({'1h': '1h', '4h': '4h'}[tf]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    b = b[b.high > b.low]
    r = np.log(b.close).diff(); sig = r.rolling(100).std().shift(1); z = (r / sig).clip(-8, 8)
    H = 8 if tf == '1h' else 6
    fwd = np.log(b.open.shift(-1 - H) / b.open.shift(-1))
    F = pd.DataFrame(index=b.index)
    F['B_z'] = z; F['B_z4'] = z.rolling(4).sum(); F['B_z16'] = z.rolling(16).sum(); F['B_z32'] = z.rolling(32).sum()
    F['B_vr'] = np.log(r.rolling(20).std() / r.rolling(100).std())
    lv = np.log(b.volume.replace(0, np.nan)); F['B_vz'] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    Y = z.fillna(0).cumsum()
    for j in range(1, 6):
        h = 2 ** (j - 1); d = Y.rolling(h).mean() - Y.shift(h).rolling(h).mean()
        F[f'W_d{j}'] = d; F[f'W_e{j}'] = (d ** 2).rolling(16).mean()
    R = rqa(z.fillna(0).values, 32, 3)
    for k, nme in enumerate(['R_det', 'R_lam', 'R_L', 'R_ent', 'R_rec_now', 'R_analog']): F[nme] = R[:, k]
    F['z'] = z; F['fwd'] = fwd; F['y'] = (fwd > 0).astype(float); F['sym'] = sym
    return F.replace([np.inf, -np.inf], np.nan).dropna()

def tokens(P, edges):
    tok = np.digitize(P.z.values, edges)                              # 0..8
    P = P.assign(tok=tok); P['ctx'] = P.groupby('sym').tok.transform(lambda s: s.shift(2) * 81 + s.shift(1) * 9 + s)
    P['nxt'] = P.groupby('sym').tok.shift(-1)
    return P.dropna(subset=['ctx'])

def evaluate_trades(te_p, va_p, te):
    lo, hi = np.quantile(va_p, [0.2, 0.8]); g = np.expm1(te.fwd.values)
    long = te_p >= hi; short = te_p <= lo
    ret = np.r_[g[long], -g[short]]; return ret.mean() * 1e4 - 14, long.sum() + short.sum()

rows = []; lang = []
for tf in ['4h', '1h']:
    P = pd.concat([coin_frame(s, tf) for s in COINS]).sort_index()
    tr = P.loc[:TR]; edges = np.quantile(tr.z, np.linspace(0, 1, 10)[1:-1])
    P = tokens(P, edges); tr = P.loc[:TR]; va = P.loc[TR:VA].iloc[1:]; te = P.loc[VA:].iloc[1:]
    # --- language test: next-token cross-entropy (trigram vs unigram), fit on train
    cnt = np.ones((729, 9)); np.add.at(cnt, (tr.ctx.values.astype(int), tr.nxt.dropna().reindex(tr.index).fillna(4).values.astype(int)), 1)
    uni = np.bincount(tr.tok, minlength=9) + 1.0; uni /= uni.sum(); tri = cnt / cnt.sum(1, keepdims=True)
    tt = te.dropna(subset=['nxt'])
    ce_tri = -np.mean(np.log2(tri[tt.ctx.astype(int), tt.nxt.astype(int)])); ce_uni = -np.mean(np.log2(uni[tt.nxt.astype(int)]))
    lang.append(dict(tf=tf, ce_unigram_bits=ce_uni, ce_trigram_bits=ce_tri, reduction_pct=100 * (ce_uni - ce_tri) / ce_uni))
    # --- token label feature: smoothed P(up | ctx) from train
    up = tr.groupby('ctx').y.agg(['sum', 'count']); a = 20.0; base = tr.y.mean()
    pu = (up['sum'] + a * base) / (up['count'] + a); lo_ = lambda c: np.log(c.map(pu).fillna(base) / (1 - c.map(pu).fillna(base)))
    for X in (tr, va, te): X['T_ctx'] = lo_(X.ctx).values
    sets = {'B': [c for c in P if c.startswith('B_')]}
    sets['B+T'] = sets['B'] + ['T_ctx']; sets['B+W'] = sets['B'] + [c for c in P if c.startswith('W_')]
    sets['B+R'] = sets['B'] + [c for c in P if c.startswith('R_')]; sets['B+T+W+R'] = sets['B'] + ['T_ctx'] + [c for c in P if c[:2] in ('W_', 'R_')]
    base_coin = None
    for nm, cols in sets.items():
        sc = StandardScaler().fit(tr[cols]); lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(tr[cols]), tr.y)
        pv = lr.predict_proba(sc.transform(va[cols]))[:, 1]; pt = lr.predict_proba(sc.transform(te[cols]))[:, 1]
        coin_auc = te.assign(p=pt).groupby('sym').apply(lambda g: roc_auc_score(g.y, g.p))
        if nm == 'B': base_coin = coin_auc
        net, ntr = evaluate_trades(pt, pv, te)
        rows.append(dict(tf=tf, set=nm, test_auc=roc_auc_score(te.y, pt), test_logloss=log_loss(te.y, pt),
                         coins_auc_up=int((coin_auc > base_coin).sum()) if nm != 'B' else np.nan, net_bp=net, trades=ntr))
R = pd.DataFrame(rows); B = R[R.set == 'B'].set_index('tf')
R['auc_gain'] = R.apply(lambda r: r.test_auc - B.loc[r.tf, 'test_auc'], axis=1)
R['net_gain_bp'] = R.apply(lambda r: r.net_bp - B.loc[r.tf, 'net_bp'], axis=1)
pd.set_option('display.width', 250)
print('=== language test (next token) ==='); print(pd.DataFrame(lang).round(4).to_string())
print('\n=== label prediction, test 2023-2026 ==='); print(R.round(4).to_string())
q = R[R.set != 'B'].copy(); q['qual'] = (q.auc_gain >= 0.005) & (q.coins_auc_up >= 7) & (q.net_gain_bp > 0)
print('\nqualifies in BOTH timeframes:', q.groupby('set').qual.all().to_dict())
R.to_csv('repr_info_test.csv', index=False); pd.DataFrame(lang).to_csv('repr_token_language.csv', index=False)

"""H22 (Addendum K): direction from the 4 envelope outputs (implied close identities) + entry-candle details."""
import numpy as np, pandas as pd
src = open('29_path_envelope.py').read().split("skill, trades = [], []")[0]; exec(src)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def candle(m5, tf):
    b = m5.resample({'1h': '1h', '4h': '4h'}[tf]).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    b = b[b.high > b.low]
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    a = tr.rolling(14).mean()
    return pd.DataFrame({'C_body': (b.close - b.open) / a, 'C_uw': (b.high - b[['open', 'close']].max(axis=1)) / a,
                         'C_lw': (b[['open', 'close']].min(axis=1) - b.low) / a}, index=b.index)

def trade(sel, M5, tf, tag):
    out = []
    U_ = sel.Uh.clip(lower=0.05); Dn_ = sel.Dh.clip(upper=-0.05)
    sel = sel.assign(tp=np.where(sel.d == 1, sel.E + 0.8 * U_ * sel.atr, sel.E + 0.8 * Dn_ * sel.atr),
                     sl=np.where(sel.d == 1, sel.E + Dn_ * sel.atr, sel.E + U_ * sel.atr))
    for s, g in sel.groupby('sym'):
        m5 = M5[s]
        R, ty = sim(m5.high.values, m5.low.values, m5.open.values, m5.close.values, g.i0.values.astype(np.int64),
                    g.i1.values.astype(np.int64), g.d.values.astype(np.int64), g.tp.values, g.sl.values)
        out.append(pd.DataFrame({'tf': tf, 'variant': tag, 'sym': s, 't': g.index, 'net': R}))
    return pd.concat(out)

skill, T = [], []
for tf in ['4h', '1h']:
    parts = [frame(s, tf) for s in COINS]; M5 = {s: m for (f, m), s in zip(parts, COINS)}
    P = pd.concat([f.join(candle(m, tf)) for f, m in parts]).sort_index()
    X = [c for c in P if c[:2] in ('B_', 'W_')]
    tr, va, te = P.loc[:TR], P.loc[TR:VA].iloc[1:].copy(), P.loc[VA:].iloc[1:].copy()
    for y in TGT:
        best = None
        for it in [50, 150, 400]:
            m = HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(tr[X], tr[y])
            mse = np.mean((m.predict(va[X]) - va[y]) ** 2)
            if best is None or mse < best[0]: best = (mse, m)
        va[y + 'h'] = best[1].predict(va[X]); te[y + 'h'] = best[1].predict(te[X])
    for d_ in (va, te):
        d_['Uh_'] = d_['Uh']; d_['Uh'] = d_['Uh_']; d_['Dh'] = d_['Dnh']
        d_['R1'] = d_.Dnh + d_.CLh; d_['R2'] = d_.Uh + d_.CHh; d_['S'] = (d_.R1 + d_.R2) / 2
        d_['agree'] = np.sign(d_.R1) == np.sign(d_.R2); d_['asym'] = d_.Uh + d_.Dnh
    retz = te.ret / (te.atr / te.E)
    skill += [dict(tf=tf, signal='R1 = Dn+CL', spearman=spearmanr(te.R1, te.ret).statistic),
              dict(tf=tf, signal='R2 = U+CH', spearman=spearmanr(te.R2, te.ret).statistic),
              dict(tf=tf, signal='S = (R1+R2)/2', spearman=spearmanr(te.S, te.ret).statistic),
              dict(tf=tf, signal='corr(R1,R2) (agreement)', spearman=spearmanr(te.R1, te.R2).statistic),
              dict(tf=tf, signal='S | R1,R2 agree', spearman=spearmanr(te.S[te.agree], te.ret[te.agree]).statistic)]
    Z = ['Uh', 'Dnh', 'CLh', 'CHh', 'R1', 'R2', 'asym', 'C_body', 'C_uw', 'C_lw', 'B_pos', 'B_rng', 'B_vz']
    sc = StandardScaler().fit(va[Z]); lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(va[Z]), va.ret > 0)
    va['st2'] = lr.predict_proba(sc.transform(va[Z]))[:, 1]; te['st2'] = lr.predict_proba(sc.transform(te[Z]))[:, 1]
    skill.append(dict(tf=tf, signal='stage2 (outputs + entry candle), AUC', spearman=roc_auc_score(te.ret > 0, te.st2)))
    skill.append(dict(tf=tf, signal='stage2 spearman vs return', spearman=spearmanr(te.st2, te.ret).statistic))
    for tag, col, mask in [('a: S', 'S', None), ('b: S, R1&R2 agree', 'S', 'agree'), ('c: stage2', 'st2', None)]:
        vv = va if mask is None else va[va[mask]]; tt = te if mask is None else te[te[mask]]
        lo, hi = np.quantile(vv[col], [0.1, 0.9])
        sel = tt[(tt[col] >= hi) | (tt[col] <= lo)].copy(); sel['d'] = np.where(sel[col] >= hi, 1, -1)
        T.append(trade(sel, M5, tf, tag))
S = pd.DataFrame(skill); T = pd.concat(T)
pd.set_option('display.width', 250)
print('=== directional skill on TEST 2023-2026 (Spearman with realised close return; AUC where noted) ==='); print(S.round(4).to_string())
rows = []
for (tf, v), g in T.groupby(['tf', 'variant']):
    rows.append(dict(tf=tf, variant=v, trades=len(g), net_bp=g.net.mean() * 1e4, win=(g.net > 0).mean(), t=ctstat(g.net.values, g.t.values),
                     coins_pos=int((g.groupby('sym').net.mean() > 0).sum()), net_2324=g[g.t < '2025-01-01'].net.mean() * 1e4,
                     net_2526=g[g.t >= '2025-01-01'].net.mean() * 1e4))
Q = pd.DataFrame(rows); Q['pass'] = (Q.net_bp > 0) & (Q.t > 2.5) & (Q.coins_pos >= 7) & (Q.net_2324 > 0) & (Q.net_2526 > 0)
print('\n=== trades ==='); print(Q.round(3).to_string())
S.to_csv('envelope_direction_skill.csv', index=False); Q.to_csv('envelope_direction_trades.csv', index=False)

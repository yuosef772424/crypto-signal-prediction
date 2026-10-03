"""H21 (Addendum J): owner's 4-output path-envelope model -> TP/SL trades on the 5m path."""
import numpy as np, pandas as pd
from numba import njit
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
D = '/home/user/research'
COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BCHUSDT', 'BNBUSDT', 'DOGEUSDT', 'TRXUSDT', 'XRPUSDT', 'ZECUSDT']
TR, VA = '2021-12-31 23:59', '2022-12-31 23:59'
TAKER, MAKER = 0.0007, 0.0002
TGT = ['U', 'Dn', 'CL', 'CH']

def frame(sym, tf):
    m5 = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    rule = {'1h': '1h', '4h': '4h'}[tf]; H = 8 if tf == '1h' else 6
    b = m5.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
    b = b[b.high > b.low]
    tr = pd.concat([b.high - b.low, (b.high - b.close.shift()).abs(), (b.low - b.close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    r = np.log(b.close).diff(); sig = r.rolling(100).std().shift(1); z = (r / sig).clip(-8, 8)
    F = pd.DataFrame(index=b.index)
    F['B_z'] = z; F['B_z4'] = z.rolling(4).sum(); F['B_z16'] = z.rolling(16).sum(); F['B_z32'] = z.rolling(32).sum()
    F['B_vr'] = np.log(r.rolling(20).std() / r.rolling(100).std())
    lv = np.log(b.volume.replace(0, np.nan)); F['B_vz'] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    F['B_rng'] = (b.high - b.low) / atr; F['B_pos'] = (b.close - b.low) / (b.high - b.low)
    F['B_atrpct'] = atr / b.close
    Y = z.fillna(0).cumsum()
    for j in range(1, 6):
        h = 2 ** (j - 1); d = Y.rolling(h).mean() - Y.shift(h).rolling(h).mean()
        F[f'W_d{j}'] = d; F[f'W_e{j}'] = (d ** 2).rolling(16).mean()
    E = b.open.shift(-1)
    mx = b.high[::-1].rolling(H).max()[::-1].shift(-1); mn = b.low[::-1].rolling(H).min()[::-1].shift(-1); cH = b.close.shift(-H)
    F['U'] = (mx - E) / atr; F['Dn'] = (mn - E) / atr; F['CL'] = (cH - mn) / atr; F['CH'] = (cH - mx) / atr
    F['ret'] = cH / E - 1; F['atr'] = atr; F['E'] = E
    tclose = b.index + pd.Timedelta(rule)
    F['i0'] = m5.index.get_indexer(tclose, method='bfill')                      # first 5m bar of bar t+1
    F['i1'] = m5.index.get_indexer(tclose + H * pd.Timedelta(rule), method='bfill') - 1
    F['sym'] = sym
    F = F.replace([np.inf, -np.inf], np.nan).dropna()
    return F[(F.i0 >= 0) & (F.i1 > F.i0)], m5

@njit
def sim(h5, l5, o5, c5, i0, i1, d, tp, sl):
    n = len(i0); R = np.zeros(n); typ = np.zeros(n, np.int8)
    for k in range(n):
        p0 = o5[i0[k]]; res = 0.0; ty = 3
        for i in range(i0[k], i1[k] + 1):
            if (d[k] == 1 and l5[i] <= sl[k]) or (d[k] == -1 and h5[i] >= sl[k]):
                res = d[k] * (sl[k] / p0 - 1) - 2 * TAKER; ty = 2; break
            if i > i0[k] and ((d[k] == 1 and h5[i] >= tp[k] * 1.0002) or (d[k] == -1 and l5[i] <= tp[k] * 0.9998)):
                res = d[k] * (tp[k] / p0 - 1) - TAKER - MAKER; ty = 1; break
        if ty == 3: res = d[k] * (c5[i1[k]] / p0 - 1) - 2 * TAKER
        R[k] = res; typ[k] = ty
    return R, typ

def ctstat(x, t):
    g = pd.DataFrame({'r': x, 'd': pd.to_datetime(t).floor('1D')}).groupby('d').r.agg(['sum', 'count'])
    mu = x.mean(); se = np.sqrt(((g['sum'] - g['count'] * mu) ** 2).sum()) / len(x); return mu / se

skill, trades = [], []
for tf in ['4h', '1h']:
    parts = [frame(s, tf) for s in COINS]; M5 = {s: m for (f, m), s in zip(parts, COINS)}
    P = pd.concat([f for f, m in parts]).sort_index(); X = [c for c in P if c[:2] in ('B_', 'W_')]
    tr, va, te = P.loc[:TR], P.loc[TR:VA].iloc[1:], P.loc[VA:].iloc[1:]
    pred = {}
    for y in TGT:
        best = None
        for it in [50, 150, 400]:
            m = HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0).fit(tr[X], tr[y])
            mse = np.mean((m.predict(va[X]) - va[y]) ** 2)
            if best is None or mse < best[0]: best = (mse, m)
        m = best[1]; pv, pt = m.predict(va[X]), m.predict(te[X]); pred[y] = (pv, pt)
        base = tr[y].mean(); r2 = 1 - np.mean((pt - te[y]) ** 2) / np.mean((base - te[y]) ** 2)
        skill.append(dict(tf=tf, target=y, spearman=spearmanr(pt, te[y]).statistic, r2_vs_symmetric_vol=r2, iters=m.max_iter))
    Av, At = pred['U'][0] + pred['Dn'][0], pred['U'][1] + pred['Dn'][1]
    skill.append(dict(tf=tf, target='ASYM U+Dn (directional)', spearman=spearmanr(At, te.U + te.Dn).statistic,
                      r2_vs_symmetric_vol=np.nan, iters=np.nan))
    skill.append(dict(tf=tf, target='ASYM vs close return', spearman=spearmanr(At, te.ret).statistic, r2_vs_symmetric_vol=np.nan, iters=np.nan))
    lo, hi = np.quantile(Av, [0.1, 0.9])
    te = te.assign(Uh=pred['U'][1], Dh=pred['Dn'][1], A=At)
    sel = te[(te.A >= hi) | (te.A <= lo)].copy(); sel['d'] = np.where(sel.A >= hi, 1, -1)
    U_ = sel.Uh.clip(lower=0.05); Dn_ = sel.Dh.clip(upper=-0.05)
    sel['tp'] = np.where(sel.d == 1, sel.E + 0.8 * U_ * sel.atr, sel.E + 0.8 * Dn_ * sel.atr)
    sel['sl'] = np.where(sel.d == 1, sel.E + Dn_ * sel.atr, sel.E + U_ * sel.atr)
    for s, g in sel.groupby('sym'):
        m5 = M5[s]
        R, ty = sim(m5.high.values, m5.low.values, m5.open.values, m5.close.values, g.i0.values.astype(np.int64),
                    g.i1.values.astype(np.int64), g.d.values.astype(np.int64), g.tp.values, g.sl.values)
        trades.append(pd.DataFrame({'tf': tf, 'sym': s, 't': g.index, 'net': R, 'type': ty, 'd': g.d.values}))
S = pd.DataFrame(skill); T = pd.concat(trades)
pd.set_option('display.width', 250)
print('=== prediction skill on TEST 2023-2026 ==='); print(S.round(4).to_string())
rows = []
for tf, g in T.groupby('tf'):
    rows.append(dict(tf=tf, trades=len(g), net_bp=g.net.mean() * 1e4, win=(g.net > 0).mean(), t=ctstat(g.net.values, g.t.values),
                     tp_share=(g.type == 1).mean(), sl_share=(g.type == 2).mean(), coins_pos=int((g.groupby('sym').net.mean() > 0).sum()),
                     net_2324=g[g.t < '2025-01-01'].net.mean() * 1e4, net_2526=g[g.t >= '2025-01-01'].net.mean() * 1e4,
                     long_bp=g[g.d == 1].net.mean() * 1e4, short_bp=g[g.d == -1].net.mean() * 1e4))
Q = pd.DataFrame(rows); Q['ACCEPT'] = (Q.net_bp > 0) & (Q.t > 2.5) & (Q.coins_pos >= 7) & (Q.net_2324 > 0) & (Q.net_2526 > 0)
print('\n=== trading rule (TP 0.8*predicted extreme, SL at predicted opposite extreme) ==='); print(Q.round(3).to_string())
S.to_csv('path_envelope_skill.csv', index=False); Q.to_csv('path_envelope_trades_summary.csv', index=False)

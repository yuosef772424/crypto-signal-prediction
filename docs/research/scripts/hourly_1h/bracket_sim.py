"""Maker-only "low-entry bracket" + two-sided quote sims on 1h next-bar OHLC (no 15m available -> bounds).
Per opportunity (coin, 32h-bucket): entry limit, TP limit, optional stop; horizon = next 1h bar.
LB (reported/decisive): SL counts whenever touched (entry precedes the extreme beyond it), TP never assumed filled
when order vs. entry is unknown -> forced taker exit at close. UB: TP filled whenever H > TP (and SL not touched)."""
import os
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from cross_asset.report import _t
from base_data import load
MK, TK, SLIP = 0.0002, 0.0005, 0.0002
e = pd.read_pickle('exc_preds.pkl')
d, df, closes = load()
rsi = pd.DataFrame({'asset': df.asset, 'timestamp': df.timestamp, 'rsi': d['X_1h'][:, -1, d['feature_order'].index('RSI_14')]})
lg = pd.read_pickle('sig_logreg_own.pkl')[['asset', 'timestamp', 'p_up_close']]
e = e.merge(rsi, on=['asset', 'timestamp']).merge(lg, on=['asset', 'timestamp'])
e['score_logreg'] = e.p_up_close; e['score_rsi'] = -e.rsi
e['month'] = pd.to_datetime(e.timestamp.astype('int64')).dt.strftime('%Y-%m')
rng = np.random.default_rng(0); e['rand'] = rng.random(len(e))
P, H, L, C = (e[c].to_numpy() for c in ('entry', 'fut_high', 'fut_low', 'fut_close'))
PU, PD = e.full_up.to_numpy(), e.full_dn.to_numpy()

def side_vec(kind, frac=0.5):
    """+1 long, -1 short, 0 no trade. frac: share of each bucket traded per side (0.5 = everyone)."""
    if kind == 'long': return np.ones(len(e))
    if kind == 'short': return -np.ones(len(e))
    col, inv = {'logreg': ('score_logreg', 1), 'logreg_inv': ('score_logreg', -1), 'rsi': ('score_rsi', 1),
                'rsi_inv': ('score_rsi', -1), 'random': ('rand', 1)}[kind]
    pr = e.groupby('bucket')[col].rank(pct=True).to_numpy()
    s = np.where(pr > 1 - frac, 1, np.where(pr <= frac, -1, 0))
    return s * inv

def bracket(side, entry, b, stop, ub=False):
    """entry: ('a', a) range-scaled or ('o', pct) fixed; stop: None | ('p', s) frac of pred range | ('f', pct)."""
    long = side > 0
    off = entry[1] * np.where(long, PD, PU) if entry[0] == 'a' else np.full(len(P), entry[1] / 100)
    E = np.where(long, P * (1 - off), P * (1 + off))
    filled = np.where(long, L < E, H > E) & (side != 0)
    TP = np.where(long, P * (1 + b * PU), P * (1 - b * PD))
    if stop is None: sl_hit = np.zeros(len(P), bool); SL = E
    else:
        s = stop[1] * np.where(long, PD, PU) if stop[0] == 'p' else np.full(len(P), stop[1] / 100)
        SL = np.where(long, E * (1 - s), E * (1 + s))
        sl_hit = np.where(long, L <= SL, H >= SL)
    tp_touch = np.where(long, H > TP, L < TP)
    tp_hit = tp_touch & ~sl_hit if ub else np.zeros(len(P), bool)
    exit_px = np.where(sl_hit, np.where(long, SL * (1 - SLIP), SL * (1 + SLIP)), np.where(tp_hit, TP, C))
    fee = MK + np.where(tp_hit, MK, TK)
    gross = np.where(long, exit_px / E - 1, 1 - exit_px / E)
    net = np.where(filled, gross - fee, 0.0); gross = np.where(filled, gross, 0.0)
    return filled, net, gross, sl_hit & filled, tp_hit & filled

def stats(filled, net, gross, active, sub=None):
    m = np.ones(len(P), bool) if sub is None else sub
    f, n, g, a = filled[m], net[m], gross[m], active[m]
    per = pd.Series(n).groupby(e.bucket.to_numpy()[m]).mean() * 100
    pg = pd.Series(g).groupby(e.bucket.to_numpy()[m]).mean() * 100
    mean, t, nb = _t(per)
    wins, losses = n[f][n[f] > 0], -n[f][n[f] <= 0]
    be = losses.mean() / (wins.mean() + losses.mean()) if len(wins) and len(losses) else np.nan
    mon = pd.Series(n).groupby(e.month.to_numpy()[m]).sum()
    coin = pd.Series(n).groupby(e.asset.to_numpy()[m]).sum()
    top5 = coin.sort_values(ascending=False).index[:5]
    ex = ~np.isin(e.asset.to_numpy()[m], top5)
    _, t_ex5, _ = _t(pd.Series(n[ex]).groupby(e.bucket.to_numpy()[m][ex]).mean())
    return dict(fill=f.sum() / max(a.sum(), 1), n_fill=int(f.sum()), win=(n[f] > 0).mean() if f.any() else np.nan,
                be_win=be, net_trade=n[f].mean() * 100 if f.any() else np.nan, gross_trade=g[f].mean() * 100 if f.any() else np.nan,
                per=mean, t=t, gross_per=pg.mean(), gross_t=_t(pg)[1], months_pos=f'{(mon > 0).sum()}/{len(mon)}', t_ex_top5=t_ex5)

if __name__ == '__main__':
    entries = [('a', a) for a in (0.25, 0.5, 0.75, 1.0)] + [('o', o) for o in (0.0, 0.05, 0.1, 0.2, 0.3)]
    stops = [None, ('p', 0.25), ('p', 0.5), ('f', 0.2), ('f', 0.3)]
    dirs = ['logreg', 'logreg_inv', 'rsi', 'rsi_inv', 'long', 'short', 'random']
    rows = []
    for dk in dirs:
        sv = side_vec(dk)
        for en, b, st in itertools.product(entries, (0.5, 0.75, 1.0), stops):
            fl, net, gr, slh, tph = bracket(sv, en, b, st)
            for sp in ('val', 'test'):
                sub = (e.split == sp).to_numpy()
                r = stats(fl, net, gr, sv != 0, sub)
                rows.append(dict(dir=dk, entry=f'{en[0]}{en[1]}', b=b, stop='none' if st is None else f'{st[0]}{st[1]}', split=sp, **r))
    R = pd.DataFrame(rows); R.to_pickle('bracket_grid.pkl')
    print('configs searched:', len(R) // 2)
    V = R[R.split == 'val'].set_index(['dir', 'entry', 'b', 'stop']); T = R[R.split == 'test'].set_index(['dir', 'entry', 'b', 'stop'])
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    best = V.t.idxmax(); print('VAL-best config:', best)
    print('val :', V.loc[best].to_dict()); print('test:', T.loc[best].to_dict())
    print('\nbest val per direction (and its test):')
    for dk in dirs:
        k = V.loc[dk].t.idxmax(); print(dk, k, '| val per %.4f t %.2f | test per %.4f t %.2f gross_t %.2f fill %.2f win %.2f be %.2f months+ %s' % (
            V.loc[(dk,) + k].per, V.loc[(dk,) + k].t, T.loc[(dk,) + k].per, T.loc[(dk,) + k].t, T.loc[(dk,) + k].gross_t,
            T.loc[(dk,) + k].fill, T.loc[(dk,) + k].win, T.loc[(dk,) + k].be_win, T.loc[(dk,) + k].months_pos))
    print('\nsame (entry,b,stop) as val-best, other directions (test):')
    print(T.xs(best[1:], level=[1, 2, 3], drop_level=False)[['fill', 'n_fill', 'win', 'be_win', 'net_trade', 'gross_trade', 'per', 't', 'gross_per', 'gross_t', 'months_pos', 't_ex_top5']].round(4).to_string())
    print('\nhow many val configs with t>2:', (V.t > 2).sum(), '| of those, test t>2:', (T.loc[V.index[V.t > 2]].t > 2).sum())
    print('max test t over all configs (not a valid selection):', T.t.max().round(2), T.t.idxmax())

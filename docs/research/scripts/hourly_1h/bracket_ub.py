import itertools, numpy as np, pandas as pd
import bracket_sim as B
entries = [('a', a) for a in (0.25, 0.5, 0.75, 1.0)] + [('o', o) for o in (0.0, 0.05, 0.1, 0.2, 0.3)]
stops = [None, ('p', 0.25), ('p', 0.5), ('f', 0.2), ('f', 0.3)]
rows = []
for dk in ['logreg', 'logreg_inv', 'rsi', 'rsi_inv', 'long', 'short', 'random']:
    sv = B.side_vec(dk)
    for en, b, st in itertools.product(entries, (0.5, 0.75, 1.0), stops):
        fl, net, gr, slh, tph = B.bracket(sv, en, b, st, ub=True)
        for sp in ('val', 'test'):
            r = B.stats(fl, net, gr, sv != 0, (B.e.split == sp).to_numpy())
            rows.append(dict(dir=dk, entry=f'{en[0]}{en[1]}', b=b, stop='none' if st is None else f'{st[0]}{st[1]}', split=sp, **r))
R = pd.DataFrame(rows); R.to_pickle('bracket_grid_ub.pkl')
V = R[R.split == 'val'].set_index(['dir', 'entry', 'b', 'stop']); T = R[R.split == 'test'].set_index(['dir', 'entry', 'b', 'stop'])
best = V.t.idxmax(); print('UB VAL-best', best, 'val t %.2f per %.4f' % (V.loc[best].t, V.loc[best].per))
cols = ['fill', 'n_fill', 'win', 'be_win', 'net_trade', 'gross_trade', 'per', 't', 'gross_t', 'months_pos', 't_ex_top5']
print(T.xs(best[1:], level=[1, 2, 3], drop_level=False)[cols].round(4).to_string())
print('val configs t>2:', (V.t > 2).sum(), '| test t>2 among them:', (T.loc[V.index[V.t > 2]].t > 2).sum(), '| max test t', T.t.max().round(2))

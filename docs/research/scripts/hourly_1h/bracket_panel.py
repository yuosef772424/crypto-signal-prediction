"""Low-entry bracket with the panel A_ic outputs: range = panel up/dn, direction = panel skew (and inverse/random/-RSI).
Selection on VAL per seed; TEST reported once. LB resolution (see bracket_sim)."""
import itertools, numpy as np, pandas as pd
import bracket_sim as B
px = pd.read_pickle('panel_exc.pkl')
e = B.e.merge(px, on=['asset', 'timestamp'], how='left')
assert len(e) == len(B.e) and e.pa0_up.notna().all()
B.e = e
entries = [('a', a) for a in (0.25, 0.5, 0.75, 1.0)] + [('o', o) for o in (0.0, 0.1, 0.2, 0.3)]
stops = [None, ('p', 0.25), ('p', 0.5), ('f', 0.2), ('f', 0.3)]
out = []
for seed in ('pa0', 'pa1'):
    B.PU, B.PD = e[f'{seed}_up'].to_numpy(), e[f'{seed}_dn'].to_numpy()
    e['score_panel'] = np.log(e[f'{seed}_up'] + 1e-3) - np.log(e[f'{seed}_dn'] + 1e-3)
    dirs = {}
    for frac in (0.5, 0.2):
        pr = e.groupby('bucket').score_panel.rank(pct=True).to_numpy()
        s = np.where(pr > 1 - frac, 1, np.where(pr <= frac, -1, 0))
        dirs[f'panel_top{int(frac*100)}'] = s; dirs[f'panel_top{int(frac*100)}_inv'] = -s
    dirs['random'] = B.side_vec('random'); dirs['rsi'] = B.side_vec('rsi')
    rows = []
    for dk, sv in dirs.items():
        for en, b, st in itertools.product(entries, (0.5, 0.75, 1.0), stops):
            fl, net, gr, _, _ = B.bracket(sv, en, b, st)
            for sp in ('val', 'test'):
                r = B.stats(fl, net, gr, sv != 0, (e.split == sp).to_numpy())
                rows.append(dict(seed=seed, dir=dk, entry=f'{en[0]}{en[1]}', b=b, stop='none' if st is None else f'{st[0]}{st[1]}', split=sp, **r))
    R = pd.DataFrame(rows); out.append(R)
    V = R[R.split == 'val'].set_index(['dir', 'entry', 'b', 'stop']); T = R[R.split == 'test'].set_index(['dir', 'entry', 'b', 'stop'])
    print(seed, 'configs', len(V), '| val t>2:', (V.t > 2).sum())
    for dk in dirs:
        k = V.loc[dk].t.idxmax(); v, t = V.loc[(dk,) + k], T.loc[(dk,) + k]
        print(f'  {dk:16s} {k} val t {v.t:.2f} | test per {t.per:.4f} t {t.t:.2f} gross_t {t.gross_t:.2f} fill {t.fill:.2f} win {t.win:.2f} be {t.be_win:.2f} m+ {t.months_pos} n {t.n_fill}')
pd.concat(out).to_pickle('bracket_panel.pkl')

"""H20 descriptive follow-up: is the 1h token 'language' about DIRECTION or only about MAGNITUDE (vol clustering)?"""
import numpy as np, pandas as pd, runpy
g = runpy.run_path('27_repr_info.py', run_name='lib') if False else None
src = open('27_repr_info.py').read().split("rows = []; lang = []")[0]; ns = {}; exec(src, ns)
TR, VA = ns['TR'], ns['VA']
def ce(tr_ctx, tr_nxt, te_ctx, te_nxt, nctx, nsym):
    cnt = np.ones((nctx, nsym)); np.add.at(cnt, (tr_ctx, tr_nxt), 1); tri = cnt / cnt.sum(1, keepdims=True)
    uni = np.bincount(tr_nxt, minlength=nsym) + 1.0; uni /= uni.sum()
    a = -np.mean(np.log2(tri[te_ctx, te_nxt])); b = -np.mean(np.log2(uni[te_nxt])); return 100 * (b - a) / b
out = []
for tf in ['1h', '4h']:
    P = pd.concat([ns['coin_frame'](s, tf) for s in ns['COINS']]).sort_index()
    tr = P.loc[:TR]
    for kind in ['sign (up/down)', 'magnitude (|z| 3 bins)', 'full 9 levels']:
        if kind.startswith('sign'): tok = (P.z > 0).astype(int); k = 2
        elif kind.startswith('mag'): e = np.quantile(tr.z.abs(), [1/3, 2/3]); tok = pd.Series(np.digitize(P.z.abs(), e), index=P.index); k = 3
        else: e = np.quantile(tr.z, np.linspace(0, 1, 10)[1:-1]); tok = pd.Series(np.digitize(P.z, e), index=P.index); k = 9
        Q = P.assign(t=tok.values); s = Q.groupby('sym').t
        Q['ctx'] = s.shift(2) * k * k + s.shift(1) * k + Q.t; Q['nxt'] = s.shift(-1); Q = Q.dropna(subset=['ctx', 'nxt'])
        a, b = Q.loc[:TR], Q.loc[VA:].iloc[1:]
        out.append(dict(tf=tf, token=kind, ce_reduction_pct_test=ce(a.ctx.astype(int).values, a.nxt.astype(int).values,
                                                                   b.ctx.astype(int).values, b.nxt.astype(int).values, k ** 3, k)))
print(pd.DataFrame(out).round(3).to_string())

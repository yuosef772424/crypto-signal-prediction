"""POST-HOC robustness for H07 (labelled: run after the single HOLDOUT look; descriptive only)."""
import numpy as np, pandas as pd
from lib import *; from tsbt import *
P = load(); px, oiv = daily_panel(P)
ok = (oiv >= 10e6) & px.notna()
vol = np.log(px).diff().rolling(30, min_periods=20).std()
segs = [('DISC', DISC), ('VAL', VAL), ('HOLD', HOLD), ('ALL', (DISC[0], HOLD[1]))]
def run(raw, U, cost):
    w = raw.where(U).fillna(0) * (0.02 / vol).clip(upper=3)
    w = w.div(U.sum(axis=1).replace(0, np.nan), axis=0)
    return run_weights(w, px, cost)[1]
rows = []
for uname, cols in [('BTC', ['BTCUSDT']), ('BTC+ETH', ['BTCUSDT', 'ETHUSDT'])]:
    U = ok & px.columns.isin(cols)
    for N in [7, 14, 21, 30, 45, 60, 90]:
        for cost in [0.0012, 0.0024]:
            n = run(np.sign(np.log(px).diff(N)).clip(lower=0), U, cost)
            for sg, (a, b) in segs:
                s = stats(n.loc[a:b]); rows.append(dict(univ=uname, N=N, cost=cost, seg=sg, sharpe=s['sharpe'], maxdd=s['maxdd']))
    b = run(pd.DataFrame(1.0, index=px.index, columns=px.columns), U, 0.0012)
    for sg, (a, c) in segs:
        s = stats(b.loc[a:c]); rows.append(dict(univ=uname, N=0, cost=0.0012, seg=sg, sharpe=s['sharpe'], maxdd=s['maxdd']))
D = pd.DataFrame(rows)
print(D.pivot_table(index=['univ', 'N', 'cost'], columns='seg', values='sharpe').round(2).to_string())
print(D[D.seg == 'ALL'].pivot_table(index=['univ', 'N', 'cost'], values='maxdd').round(2).T.to_string())

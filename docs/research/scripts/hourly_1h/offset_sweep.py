"""Maker entry offset sweep (long and short, all coins, no direction model): fill rate, adverse-selection gap
(close-to-entry return given fill vs unconditional close-to-last-close return), and per-trade/per-period net with
exit = limit at predicted extreme (LB: never assumed filled) -> taker exit at close."""
import numpy as np, pandas as pd
import bracket_sim as B
from cross_asset.report import _t
e = B.e; te = (e.split == 'test').to_numpy()
rows = []
for side in (1, -1):
    sv = np.full(len(e), side)
    for en in [('o', o) for o in (0.0, 0.05, 0.1, 0.2, 0.3)] + [('a', a) for a in (0.25, 0.5, 0.75, 1.0)]:
        fl, net, gr, _, _ = B.bracket(sv, en, 1.0, None)
        off = en[1] * np.where(side > 0, B.PD, B.PU) if en[0] == 'a' else np.full(len(e), en[1] / 100)
        E = B.P * (1 - side * off)
        cond = side * (B.C / E - 1)            # return from fill price to close, when filled
        unc = side * (B.C / B.P - 1)           # same side entered at last close, no condition
        f = fl & te
        per = pd.Series(net[te]).groupby(e.bucket.to_numpy()[te]).mean() * 100
        mon = pd.Series(net[te]).groupby(e.month.to_numpy()[te]).sum()
        rows.append(dict(side='long' if side > 0 else 'short', entry=f'{en[0]}{en[1]}', fill=f.sum() / te.sum(),
                         cond_ret=cond[f].mean() * 100, uncond_ret=unc[te].mean() * 100, gap=(cond[f].mean() - unc[te].mean()) * 100,
                         win=(net[f] > 0).mean(), net_trade=net[f].mean() * 100, per=per.mean(), t=_t(per)[1],
                         months_pos=f'{(mon > 0).sum()}/{len(mon)}'))
pd.set_option('display.width', 200)
print(pd.DataFrame(rows).round(4).to_string())

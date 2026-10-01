"""Single descriptive HOLDOUT look for H06/H07 — specs frozen in 00_PREREGISTRATION.md Addendum B."""
import numpy as np, pandas as pd
from scipy.stats import norm
from lib import *; from tsbt import *
P = load(); L = P['L']
segs = [('DISC', DISC), ('VAL', VAL), ('HOLD', HOLD)]

print('== H06: long 20:00 -> 22:00 UTC ==')
rows = []
for sym in ['BTCUSDT', 'ETHUSDT']:
    l = L[sym]
    r = (l.shift(-2) - l)[l.index.hour == 20].dropna()        # log return 20:00 -> 22:00, one per day
    r = np.expm1(r)
    for sg, (a, b) in segs:
        x = r.loc[a:b]
        rows.append(dict(sym=sym, seg=sg, n=len(x), gross_bp=x.mean()*1e4, t=x.mean()/x.std()*np.sqrt(len(x)),
                         win=(x > 0).mean(), net_maker_bp=(x.mean()-0.0004)*1e4, net_taker_bp=(x.mean()-0.0012)*1e4,
                         ann_sharpe_gross=x.mean()/x.std()*np.sqrt(365)))
print(pd.DataFrame(rows).round(2).to_string())

print('\n== H07: TSMOM30_LO_BTC+ETH vs BUYHOLD ==')
px, oiv = daily_panel(P)
ok = (oiv >= 10e6) & px.notna()
U = ok & px.columns.isin(['BTCUSDT', 'ETHUSDT'])
vol = np.log(px).diff().rolling(30, min_periods=20).std()
def run(raw):
    w = raw.where(U).fillna(0) * (0.02 / vol).clip(upper=3)
    w = w.div(U.sum(axis=1).replace(0, np.nan), axis=0)
    return run_weights(w, px, 0.0012)[1]
nets = {'TSMOM30_LO': run(np.sign(np.log(px).diff(30)).clip(lower=0)),
        'BUYHOLD': run(pd.DataFrame(1.0, index=px.index, columns=px.columns))}
def dsr(sr_ann, T, N, skew, kurt):
    sr = sr_ann / np.sqrt(365)
    g = 0.5772156649
    sr0 = np.sqrt(1/T) * ((1-g)*norm.ppf(1-1/N) + g*norm.ppf(1-1/(N*np.e)))   # per-period, var(SR_hat)≈1/T
    return norm.cdf((sr - sr0) * np.sqrt(T-1) / np.sqrt(1 - skew*sr + (kurt-1)/4*sr**2))
rows = []
for k, n in nets.items():
    for sg, (a, b) in segs:
        x = n.loc[a:b].dropna(); s = stats(x)
        s.update(strategy=k, seg=sg, kurt=x.kurt()+3)
        if k == 'TSMOM30_LO':
            s['DSR_N44'] = dsr(s['sharpe'], len(x), 44, s['skew'], s['kurt'])
            s['DSR_N750'] = dsr(s['sharpe'], len(x), 750, s['skew'], s['kurt'])
        rows.append(s)
D = pd.DataFrame(rows)
cols = ['strategy','seg','days','sharpe','ann_ret','ann_vol','maxdd','total','t','skew','DSR_N44','DSR_N750']
print(D[cols].round(3).to_string())

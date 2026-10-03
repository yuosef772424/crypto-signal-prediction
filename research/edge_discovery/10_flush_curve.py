"""
PURPOSE:  H05 dose-response: forward market return by aggregate OI-change bin after market crash hours, plus the
          deep-flush event list.
TAGS:     H05, dose-response, OI flush, market z<-2, flush curve, event list
PITFALLS: Monotonic in DISC but 2025 deep-flush events were mixed; needs /home/user/research/panel_*.parquet (built
          from Drive futures_metrics via data_tools; not in repo, see RESUME.md)
"""
import numpy as np, pandas as pd
from lib import *
P = load(); L = P['L']; U = P['univ'] & P['big']
r = L.diff(); mret = r.where(U).mean(axis=1); mvol = mret.rolling(168, min_periods=120).std()
doi = np.log(P['oi']).diff().where(U); w = P['oiv'].shift(1).where(U)
aoi = (doi * w).sum(axis=1) / w.sum(axis=1)
mL = mret.cumsum(); z = mret / mvol
f12 = mL.shift(-13) - mL.shift(-1)
df = pd.DataFrame({'z': z, 'aoi': aoi, 'f12': f12, 'f24': mL.shift(-25) - mL.shift(-1), 'f4': mL.shift(-5) - mL.shift(-1)}).dropna()
for seg_, (a, b) in [('DISC', DISC), ('VAL', VAL), ('DISC+VAL', (DISC[0], VAL[1]))]:
    d = df.loc[a:b]; d = d[d.z < -2]
    d['aoi_bin'] = pd.cut(d.aoi, [-1, -0.02, -0.01, -0.005, 0, 0.005, 1])
    g = d.groupby('aoi_bin', observed=True).agg(n=('f12', 'size'), f4=('f4', 'mean'), f12=('f12', 'mean'), f24=('f24', 'mean'), win12=('f12', lambda x: (x > 0).mean()))
    g[['f4','f12','f24']] *= 1e4
    print(f'\n{seg_}: hours with market z<-2, by aggregate OI change in that hour'); print(g.round(2).to_string())
# event list for deep flush
d = df.loc[DISC[0]:VAL[1]]; ev = d[(d.z < -3) & (d.aoi < -0.015)]
keep = []; last = None
for t in ev.index:
    if last is None or t - last >= pd.Timedelta(hours=24): keep.append(t); last = t
e = ev.loc[keep]; e[['f4','f12','f24']] = np.expm1(e[['f4','f12','f24']]) * 1e4
print(e.round(3).to_string())

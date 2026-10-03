"""
PURPOSE:  H05: market-wide crash hour plus aggregate OI flush -> rebound, event table over thresholds (DISC and VAL).
TAGS:     H05, market crash, OI flush, rebound, dip buying, aggregate OI change, cooldown
PITFALLS: H05 worked in 2024 only (bull-market regime) and failed in 2025; needs /home/user/research/panel_*.parquet
          (built from Drive futures_metrics via data_tools; not in repo, see RESUME.md)
"""
import numpy as np, pandas as pd
from lib import *
P = load(); L = P['L']; U = P['univ'] & P['big']
r = L.diff()
mret = r.where(U).mean(axis=1)                     # EW top-50 hourly log ret
mvol = mret.rolling(168, min_periods=120).std()
# aggregate contracts change: OI value change minus price change, OI-value weighted
doi = np.log(P['oi']).diff().where(U)
w = P['oiv'].shift(1).where(U); aoi = (doi * w).sum(axis=1) / w.sum(axis=1)
btc = r['BTCUSDT']
mL = mret.cumsum()
def fwdm(H): return mL.shift(-1 - H) - mL.shift(-1)
def show(mask, label, Hs=(4, 12, 24, 72), cool=24):
    idx = mask[mask].index; keep = []; last = None
    for t in idx:
        if last is None or (t - last) >= pd.Timedelta(hours=cool): keep.append(t); last = t
    out = {}
    for seg_, (a, b) in [('DISC', DISC), ('VAL', VAL)]:
        k = [t for t in keep if pd.Timestamp(a, tz='UTC') <= t <= pd.Timestamp(b, tz='UTC')]
        row = {'n': len(k)}
        for H in Hs:
            f = np.expm1(fwdm(H).loc[k]); row[f'{H}h_bp'] = f.mean()*1e4; row[f'{H}h_win'] = (f > 0).mean()
            row[f'{H}h_t'] = f.mean() / f.std() * np.sqrt(len(f)) if len(f) > 2 else np.nan
        out[seg_] = row
    print(label); print(pd.DataFrame(out).T.round(2).to_string()); return keep
z = mret / mvol
for zt in [-3, -4]:
    for oit in [0, -0.01, -0.02]:
        show((z < zt) & (aoi < oit), f'\n== market 1h z<{zt} & aggOI change<{oit}')
show(z < -3, '\n== market 1h z<-3 (any OI)')
# baseline unconditional
print('\nunconditional 24h mean bp DISC/VAL:', (np.expm1(fwdm(24)).loc[DISC[0]:DISC[1]].mean()*1e4).round(1), (np.expm1(fwdm(24)).loc[VAL[0]:VAL[1]].mean()*1e4).round(1))

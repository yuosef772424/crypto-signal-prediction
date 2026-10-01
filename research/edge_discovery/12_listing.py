"""H12 post-listing drift (pre-registered: 00_PREREGISTRATION.md Addendum A / A.1)."""
import numpy as np, pandas as pd, glob, os, sys
D = '/home/user/research'
px = pd.read_parquet(f'{D}/panel_px.parquet'); btc = np.log(px['BTCUSDT'].ffill())
ex = set(open('data_tools/tradfi_exclude.txt').read().split())
rows = []
for f in sorted(glob.glob(f'{D}/metrics_new/*.csv.gz')):
    s = os.path.basename(f)[:-7]
    if s in ex: continue
    d = pd.read_csv(f); d['ts'] = pd.to_datetime(d.timestamp, utc=True).dt.floor('h')
    d = d.drop_duplicates('ts').set_index('ts').sort_index()
    d = d[(d.sum_open_interest > 0) & (d.sum_open_interest_value > 0)]
    if len(d) < 24 * 31: continue
    p = np.log(d.sum_open_interest_value / d.sum_open_interest)
    t0 = p.index[0]
    if t0 < pd.Timestamp('2024-01-15', tz='UTC'): continue           # not a new listing inside data
    te = t0 + pd.Timedelta(hours=24)
    pe = p.asof(te); be = btc.asof(te)
    r = dict(sym=s, first=t0, entry=te, oiv_entry=float(d.sum_open_interest_value.asof(te)))
    for k in [7, 30, 60]:
        tx = te + pd.Timedelta(days=k)
        if tx > p.index[-1] or tx > btc.dropna().index[-1]: r[f'r{k}'] = np.nan; continue
        r[f'r{k}'] = (p.asof(tx) - pe) - (btc.asof(tx) - be)     # coin minus BTC (log)
    rows.append(r)
E = pd.DataFrame(rows)
def seg(t):
    if t <= pd.Timestamp('2025-03-31 23:00', tz='UTC'): return 'DISC'
    if t <= pd.Timestamp('2025-12-31 23:00', tz='UTC'): return 'VAL'
    return 'HOLD'
E['seg'] = E['first'].map(seg); E['month'] = E['first'].dt.strftime('%Y-%m')
E.to_csv('listing_events.csv', index=False)
COST = 0.003
show_hold = '--holdout' in sys.argv
out = []
for sg in ['DISC', 'VAL'] + (['HOLD'] if show_hold else []):
    for k in [7, 30, 60]:
        x = E[E.seg == sg].dropna(subset=[f'r{k}'])
        if len(x) < 5: continue
        short = -np.expm1(x[f'r{k}']) - COST          # short coin / long BTC, simple-return approx per $1
        m = short.groupby(x.month).mean()             # cluster by listing month
        out.append(dict(seg=sg, days=k, n=len(x), months=len(m), mean=short.mean(), median=short.median(),
                        win=(short > 0).mean(), t_month=m.mean() / m.std() * np.sqrt(len(m)) if len(m) > 2 else np.nan,
                        avg_win=short[short > 0].mean(), avg_loss=short[short <= 0].mean(), worst=short.min(),
                        p5=short.quantile(.05)))
print(pd.DataFrame(out).round(3).to_string())
print('events by segment:', E.seg.value_counts().to_dict())

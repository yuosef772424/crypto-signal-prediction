"""H12 with stop-loss (Addendum A.2). Usage: python 13_listing_stops.py [--holdout]"""
import numpy as np, pandas as pd, glob, os, sys
D = '/home/user/research'
px = pd.read_parquet(f'{D}/panel_px.parquet'); btc = px['BTCUSDT'].ffill()
ex = set(open('data_tools/tradfi_exclude.txt').read().split())
COST, SLIP = 0.003, 0.02
def seg(t):
    if t <= pd.Timestamp('2025-03-31 23:00', tz='UTC'): return 'DISC'
    if t <= pd.Timestamp('2025-12-31 23:00', tz='UTC'): return 'VAL'
    return 'HOLD'
trades = []
for f in sorted(glob.glob(f'{D}/metrics_new/*.csv.gz')):
    s = os.path.basename(f)[:-7]
    if s in ex: continue
    d = pd.read_csv(f); d['ts'] = pd.to_datetime(d.timestamp, utc=True).dt.floor('h')
    d = d.drop_duplicates('ts').set_index('ts').sort_index()
    d = d[(d.sum_open_interest > 0) & (d.sum_open_interest_value > 0)]
    if len(d) < 24 * 31: continue
    p = (d.sum_open_interest_value / d.sum_open_interest)
    t0 = p.index[0]
    if t0 < pd.Timestamp('2024-01-15', tz='UTC'): continue
    te = t0 + pd.Timedelta(hours=24); pe = p.asof(te); be = btc.asof(te)
    for H in [30, 60]:
        tx = te + pd.Timedelta(days=H)
        if tx > p.index[-1] or tx > btc.dropna().index[-1]: continue
        path = p.loc[te:tx]
        for S in [None, 0.5, 1.0]:
            exit_t, exit_p = tx, p.asof(tx); stopped = False
            if S is not None:
                hit = path[path >= pe * (1 + S)]
                if len(hit):
                    exit_t = hit.index[0]; exit_p = hit.iloc[0] * (1 + SLIP); stopped = True
            coin = exit_p / pe - 1; bh = btc.asof(exit_t) / be - 1
            ret = -coin + bh - COST                  # short coin, long BTC, per $1 notional each
            trades.append(dict(sym=s, entry=te, seg=seg(t0), H=H, S=str(S), ret=ret, stopped=stopped,
                               days=(exit_t - te).total_seconds() / 86400, month=t0.strftime('%Y-%m')))
T = pd.DataFrame(trades); T.to_csv('listing_trades.csv', index=False)
segs = ['DISC', 'VAL'] + (['HOLD'] if '--holdout' in sys.argv else [])
rows = []
for (sg, H, S), x in T[T.seg.isin(segs)].groupby(['seg', 'H', 'S']):
    m = x.groupby('month').ret.mean()
    w, l = x.ret[x.ret > 0], x.ret[x.ret <= 0]
    rows.append(dict(seg=sg, H=H, S=S, n=len(x), mean=x.ret.mean(), median=x.ret.median(), win=(x.ret > 0).mean(),
                     avg_win=w.mean(), avg_loss=l.mean(), payoff=w.mean() / -l.mean(), pf=w.sum() / -l.sum(),
                     worst=x.ret.min(), stop_rate=x.stopped.mean(), t_month=m.mean() / m.std() * np.sqrt(len(m)), months=len(m)))
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(3).to_string())

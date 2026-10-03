import numpy as np, pandas as pd
from lib import *; from features import build; from events import *
P = load(); L = P['L']; U = P['univ']; F = build(P)
r1 = L.diff(); z1 = r1 / F['vol168']; z24 = L.diff(24) / (F['vol168'] * np.sqrt(24))
doi1 = F['doi_1']; doi24 = F['doi_24']
ev = {
 'crash1h_z<-4':           z1 < -4,
 'crash1h_z<-4 & OIflush': (z1 < -4) & (doi1 < -0.03),
 'crash1h_z<-4 & OIup':    (z1 < -4) & (doi1 > 0.01),
 'pump1h_z>4':             z1 > 4,
 'pump1h_z>4 & OIflush':   (z1 > 4) & (doi1 < -0.03),
 'pump1h_z>4 & OIup':      (z1 > 4) & (doi1 > 0.03),
 'crash24_z<-3':           z24 < -3,
 'crash24_z<-3 & OIflush24': (z24 < -3) & (doi24 < -0.10),
 'pump24_z>3':             z24 > 3,
 'pump24_z>3 & OIup24':    (z24 > 3) & (doi24 > 0.15),
 'pump24_z>3 & OIdown24':  (z24 > 3) & (doi24 < -0.05),
 'retail_long_z>2':        F['ls_acc_z'] > 2,
 'retail_short_z<-2':      F['ls_acc_z'] < -2,
 'smart_long_z>2':         F['tt_pos_z'] > 2,
 'smart_short_z<-2':       F['tt_pos_z'] < -2,
 'taker_buy_z>3':          F['taker_z'] > 3,
 'taker_sell_z<-3':        F['taker_z'] < -3,
 'breakout_30d_high':      (L >= L.rolling(720).max()) & (L.shift(1) < L.rolling(720).max().shift(1)),
 'breakdown_30d_low':      (L <= L.rolling(720).min()) & (L.shift(1) > L.rolling(720).min().shift(1)),
 'oi_z30>2.5':             F['oi_z30'] > 2.5,
}
rows = []
for name, E in ev.items():
    Ed = dedup(E.fillna(False).loc[DISC[0]:DISC[1]], 72)
    for H in [4, 24, 72]:
        for side in [1, -1]:
            for madj in [False, True]:
                s = event_stats(Ed, L, U, H, side=side, mkt_adj=madj, cost=0.0015)
                if s: rows.append(dict(event=name, H=H, side=side, madj=madj, **s))
D = pd.DataFrame(rows)
D.to_csv('events_disc.csv', index=False)
pd.set_option('display.width', 250)
# only show long side rows (short = mirror minus costs) and both adj types
show = D[D.side == 1].copy()
print(show.round(3).sort_values('t_day').to_string())
print('tests:', len(D))

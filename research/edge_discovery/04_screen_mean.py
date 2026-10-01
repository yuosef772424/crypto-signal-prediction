import numpy as np, pandas as pd
from lib import *; from features import build
P = load(); L = P['L']; U = P['univ']; F = build(P)
rows = []
a, b = DISC
for H in [24, 72]:
    idx = L.loc[a:b].index[::H]
    R = np.expm1((L.shift(-1 - H) - L.shift(-1)).loc[idx]).where(U.loc[idx])
    mkt = R.mean(axis=1)
    for name, X in F.items():
        q = X.loc[idx].where(U.loc[idx] & R.notna()).rank(axis=1, pct=True)
        top = R.where(q > 0.8).mean(axis=1) - mkt; bot = R.where(q <= 0.2).mean(axis=1) - mkt
        # pooled trade-level skew of top leg
        tl = (R.where(q > 0.8).sub(mkt, axis=0)).stack()
        rows.append(dict(feat=name, H=H, top_bp=top.mean()*1e4, t_top=tstat(top), bot_bp=bot.mean()*1e4, t_bot=tstat(bot),
                         ls_bp=(top-bot).mean()*1e4, t_ls=tstat(top-bot), med_top_bp=tl.median()*1e4, skew_top=tl.skew()))
D = pd.DataFrame(rows); D['maxt'] = D[['t_top','t_bot','t_ls']].abs().max(axis=1)
pd.set_option('display.width', 220)
print(D.sort_values('maxt', ascending=False).round(2).head(30).to_string())
D.to_csv('screen_disc_mean.csv', index=False)

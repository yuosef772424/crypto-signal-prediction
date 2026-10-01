import numpy as np, pandas as pd
from lib import *
P = load(); L = P['L']; U = P['univ']
r = L.diff()
btc = r['BTCUSDT']; eth = r['ETHUSDT']
alt = r.where(U & P['big']).mean(axis=1)  # EW top-50
out = []
for name, s in [('BTC', btc), ('ETH', eth), ('ALT50', alt)]:
    for segname, (a, b) in [('DISC', DISC), ('VAL', VAL)]:
        x = s.loc[a:b]
        g = x.groupby(x.index.hour)
        out.append(pd.DataFrame({'asset': name, 'seg': segname, 'hour': g.mean().index, 'bp': g.mean().values*1e4,
                                 't': (g.mean()/g.std()*np.sqrt(g.count())).values}))
D = pd.concat(out)
W = D.pivot_table(index='hour', columns=['asset', 'seg'], values=['bp', 't']).round(2)
pd.set_option('display.width', 250); print(W.to_string())
# consistency: corr of hourly means DISC vs VAL
for a in ['BTC','ETH','ALT50']:
    d = D[(D.asset==a)&(D.seg=='DISC')].bp.values; v = D[(D.asset==a)&(D.seg=='VAL')].bp.values
    print(a, 'corr(hour profile DISC, VAL)=', np.corrcoef(d, v)[0,1].round(3))
# day of week
for name, s in [('BTC', btc), ('ALT50', alt)]:
    for segname, (a, b) in [('DISC', DISC), ('VAL', VAL)]:
        x = s.loc[a:b].resample('D').sum(); g = x.groupby(x.index.dayofweek)
        print(name, segname, 'DoW bp:', (g.mean()*1e4).round(0).to_dict(), 't:', (g.mean()/g.std()*np.sqrt(g.count())).round(1).to_dict())

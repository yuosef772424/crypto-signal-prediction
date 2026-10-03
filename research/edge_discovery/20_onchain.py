"""H15: on-chain & stablecoin-liquidity signals (Addendum D)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from tsbt import stats
CM = '/home/user/research/coinmetrics'
def cm(a, cols):
    d = pd.read_csv(f'{CM}/{a}.csv', low_memory=False); d.index = pd.to_datetime(d.time)
    return d[cols].apply(pd.to_numeric, errors='coerce')
segs = {'DISC': ('2013-01-01', '2019-12-31'), 'VAL': ('2020-01-01', '2022-12-31'), 'HOLD': ('2023-01-01', '2026-05-24')}
st = sum(cm(s, ['SplyCur'])['SplyCur'].fillna(0) for s in ['usdt', 'usdc', 'dai', 'busd']).replace(0, np.nan)
def signals(a):
    b = cm(a, ['PriceUSD', 'CapMVRVCur', 'FlowInExNtv', 'FlowOutExNtv', 'SplyExNtv', 'SplyCur', 'HashRate', 'AdrActCnt']).loc['2011-01-01':]
    S = pd.DataFrame(index=b.index)
    lm = np.log(b.CapMVRVCur); S['S1_mvrv'] = lm.expanding(365).rank(pct=True)
    S['S2_exflow'] = (b.FlowInExNtv - b.FlowOutExNtv).rolling(30).sum() / b.SplyCur
    S['S3_exsupply'] = (b.SplyExNtv / b.SplyCur).diff(90)
    S['S4_hashribbon'] = b.HashRate.rolling(30).mean() / b.HashRate.rolling(60).mean() - 1
    S['S5_stable'] = np.log(st.reindex(b.index)).diff(30).where(st.reindex(b.index) > 1e8)
    S['S6_activity'] = np.log(b.AdrActCnt.rolling(30).mean() / b.AdrActCnt.rolling(365).mean())
    return b.PriceUSD, S.shift(1)                                  # one extra day of publication lag
SIGN = {'S1_mvrv': -1, 'S2_exflow': -1, 'S3_exsupply': -1, 'S4_hashribbon': 1, 'S5_stable': 1, 'S6_activity': 1}
for a in ['btc', 'eth']:
    px, S = signals(a)
    f30 = np.log(px).shift(-30) - np.log(px)
    r = px.pct_change().shift(-1)
    vol = np.log(px).diff().rolling(30, min_periods=20).std()
    rows = []
    for k, sg in SIGN.items():
        x = S[k] * sg                                              # oriented: higher = more bullish (theory)
        fav = (x > x.expanding(180).median()).astype(float).where(x.notna())
        for seg, (lo, hi) in segs.items():
            me = pd.concat([x, f30], axis=1).loc[lo:hi].resample('ME').last().dropna()
            if len(me) < 8: rows.append(dict(sig=k, seg=seg, n=len(me))); continue
            rho = spearmanr(me.iloc[:, 0], me.iloc[:, 1]).statistic
            t = rho * np.sqrt((len(me) - 2) / max(1e-9, 1 - rho ** 2))
            w = fav.loc[lo:hi].fillna(0); turn = w.diff().abs().fillna(0)
            strat = (w * r.loc[lo:hi] - turn * 0.0006).dropna(); bh = r.loc[lo:hi].dropna()
            rows.append(dict(sig=k, seg=seg, n=len(me), rho=rho, t=t, sharpe=stats(strat)['sharpe'],
                             bh_sharpe=stats(bh)['sharpe'], maxdd=stats(strat)['maxdd'], bh_maxdd=stats(bh)['maxdd'],
                             in_mkt=w.mean()))
    D = pd.DataFrame(rows)
    print(f'===== {a.upper()} =====')
    print(D.pivot(index='sig', columns='seg', values=['rho', 't']).round(2).to_string())
    print(D.pivot(index='sig', columns='seg', values=['sharpe', 'bh_sharpe', 'in_mkt']).round(2).to_string())
    D.to_csv(f'onchain_{a}.csv', index=False)

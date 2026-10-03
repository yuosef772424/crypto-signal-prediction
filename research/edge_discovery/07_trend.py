import numpy as np, pandas as pd
from lib import *; from tsbt import *
P = load(); px, oiv = daily_panel(P)
ok = (oiv.shift(0) >= 10e6) & px.notna()
rk = oiv.rank(axis=1, ascending=False)
lr = np.log(px).diff()
vol = lr.rolling(30, min_periods=20).std()
res = []
def evaluate(name, raw_pos, univ, cost=0.0012, risk=0.02):
    pos = raw_pos.where(univ).fillna(0)
    w = pos * (risk / (vol * np.sqrt(1))).clip(upper=3)      # each coin 2% daily vol target (inverse vol)
    n = univ.sum(axis=1).replace(0, np.nan)
    w = w.div(n, axis=0)                                      # equal risk across universe
    g, net, turn = run_weights(w, px, cost)
    for seg_, (a, b) in [('DISC', DISC), ('VAL', VAL)]:
        s = stats(net.loc[a:b]); s.update(name=name, seg=seg_, turn=turn.loc[a:b].mean()); res.append(s)
univs = {'BTC': ok & (px.columns == 'BTCUSDT'), 'BTC+ETH': ok & px.columns.isin(['BTCUSDT','ETHUSDT']), 'top20': ok & (rk <= 20), 'top50': ok & (rk <= 50)}
for uname, U in univs.items():
    for N in [7, 14, 30, 60]:
        s = np.sign(np.log(px).diff(N))
        evaluate(f'TSMOM{N}_LS_{uname}', s, U)
        evaluate(f'TSMOM{N}_LO_{uname}', s.clip(lower=0), U)
    # Donchian ensemble long-only (Zarattini style): state per N, average
    st = []
    for N in [5, 10, 20, 30, 60, 90]:
        hi = px.rolling(N).max(); lo = px.rolling(max(2, N // 2)).min()
        sig = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
        sig[px >= hi] = 1; sig[px <= lo] = 0
        st.append(sig.ffill().fillna(0))
    don = sum(st) / len(st)
    evaluate(f'DONCH_ENS_LO_{uname}', don, U)
    st2 = []
    for N in [5, 10, 20, 30, 60, 90]:
        hi = px.rolling(N).max(); lo = px.rolling(N).min()
        sig = pd.DataFrame(np.nan, index=px.index, columns=px.columns); sig[px >= hi] = 1; sig[px <= lo] = -1
        st2.append(sig.ffill().fillna(0))
    evaluate(f'DONCH_ENS_LS_{uname}', sum(st2) / len(st2), U)
    evaluate(f'BUYHOLD_{uname}', pd.DataFrame(1.0, index=px.index, columns=px.columns), U)
D = pd.DataFrame(res)
pd.set_option('display.width', 250)
cols = ['name','seg','sharpe','ann_ret','ann_vol','maxdd','t','skew','turn']
W = D.pivot(index='name', columns='seg', values=['sharpe','ann_ret','maxdd']).round(2)
print(W.to_string()); D.to_csv('trend_disc_val.csv', index=False)

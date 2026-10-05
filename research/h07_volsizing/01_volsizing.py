"""
PURPOSE:  E-h07vs-1: does an ML forecast of next-day realized volatility (instead of the trailing 30-day std) improve the
          frozen H07 trend filter (TSMOM30 long-only BTC+ETH, weight 0.02/sigma clipped at 3)? Compares trail30 / HAR-RV
          / gradient boosting on forecast QLIKE and on the H07 backtest (Sharpe, max drawdown, risk-targeting error).
TAGS:     H07, TSMOM30, volatility forecast, realized variance, HAR-RV, gradient boosting, position sizing, vol targeting,
          QLIKE, block bootstrap, E-h07vs-1
PITFALLS: Needs ohlc_BTCUSDT_5m.parquet / ohlc_ETHUSDT_5m.parquet in DATA_DIR (tools/fetch_crypto_dataset.py, trimmed
          2026-09-30). Daily close = last 5m close of the UTC day (Binance spot, not CoinMetrics as in 16_h07_oos.py).
          The weight set at day d uses data up to d's close and earns d -> d+1; the forecast target is RV of day d+1.
          TEST rows are computed but the card's rule decides on VAL first.

Usage: DATA_DIR=/home/user/research python research/h07_volsizing/01_volsizing.py
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

D = os.environ.get('DATA_DIR', '/home/user/research')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'volsizing_results.csv' if os.environ.get('RV_BAR', '5min') == '5min'
                   else f"volsizing_results_rv{os.environ['RV_BAR']}.csv")
COINS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}
TRAIN = ('2018-01-01', '2021-12-31')
VAL = ('2022-01-01', '2023-12-31')
TEST = ('2024-01-01', '2026-09-30')
COST_RT, TARGET, CAP, N = 0.0012, 0.02, 3.0, 30


#: Bar size for the realized variance (env RV_BAR): '5min' (default, the pre-registered run), '1h' or '4h' (robustness).
RV_BAR = os.environ.get('RV_BAR', '5min')


def daily(sym):
    m = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    if RV_BAR != '5min':
        m = m.resample(RV_BAR).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                                    'volume': 'sum'}).dropna()
    lr = np.log(m['close']).diff()
    day = m.index.floor('1D')
    out = pd.DataFrame({
        'close': m['close'].groupby(day).last(),
        'high': m['high'].groupby(day).max(), 'low': m['low'].groupby(day).min(),
        'rv': (lr ** 2).groupby(day).sum(), 'nbars': m['close'].groupby(day).size()})
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    full = int(pd.Timedelta('1D') / pd.Timedelta(RV_BAR))
    return out[out.nbars >= int(np.ceil(0.87 * full))]   # >= ~87% of the day's bars


def features(d, other):
    f = pd.DataFrame(index=d.index)
    lrv = np.log(d.rv)
    f['lrv1'], f['lrv5'], f['lrv22'] = lrv, np.log(d.rv.rolling(5).mean()), np.log(d.rv.rolling(22).mean())
    park = np.log(d.high / d.low) ** 2 / (4 * np.log(2))
    f['lpk1'], f['lpk5'] = np.log(park), np.log(park.rolling(5).mean())
    ret = np.log(d.close).diff()
    f['ar1'], f['ar5'] = ret.abs(), ret.abs().rolling(5).mean()
    f['trend'] = (np.log(d.close).diff(N) > 0).astype(float)
    f['other_lrv1'] = np.log(other.rv).reindex(d.index)
    f['dow'] = d.index.dayofweek
    f['y'] = lrv.shift(-1)                             # next day's log realized variance
    return f


def qlike(log_var_pred, log_var_true):
    """QLIKE on variances: mean(true/pred - log(true/pred) - 1); 0 is perfect."""
    q = np.exp(log_var_true - log_var_pred)
    return float(np.nanmean(q - np.log(q) - 1))


def forecasts(f, seeds=(0, 1, 2)):
    """Daily sigma forecasts per model (fitted on TRAIN only)."""
    tr = f.loc[TRAIN[0]:TRAIN[1]].dropna()
    X_cols = [c for c in f.columns if c != 'y']
    har_cols = ['lrv1', 'lrv5', 'lrv22']
    har = LinearRegression().fit(tr[har_cols], tr['y'])
    ok = f[X_cols].notna().all(axis=1)
    pred = {'har': pd.Series(np.nan, index=f.index), 'gbm': pd.Series(np.nan, index=f.index)}
    pred['har'][ok] = har.predict(f.loc[ok, har_cols])
    g = np.zeros(ok.sum())
    for s in seeds:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=s)
        g += m.fit(tr[X_cols], tr['y']).predict(f.loc[ok, X_cols]) / len(seeds)
    pred['gbm'][ok] = g
    return pred, har


def backtest(px, sig, sigma):
    """H07 weights = sig * clip(TARGET/sigma, <=CAP) / n_assets; returns net daily returns."""
    w = (sig * (TARGET / sigma).clip(upper=CAP)) / sig.shape[1]
    w = w.fillna(0.0)
    r = px.pct_change().shift(-1)
    turn = (w - w.shift(1).fillna(0)).abs().sum(axis=1)
    gross = (w * r).sum(axis=1, min_count=1)
    return (gross - turn * COST_RT / 2).dropna(), w


def stats(x):
    x = x.dropna()
    eq = (1 + x).cumprod()
    return dict(days=len(x), sharpe=x.mean() / x.std() * np.sqrt(365), ann_ret=x.mean() * 365,
                ann_vol=x.std() * np.sqrt(365), maxdd=float((eq / eq.cummax() - 1).min()), total=float(eq.iloc[-1] - 1))


def block_boot_diff(a, b, block=20, reps=2000, seed=0):
    """Stationary block bootstrap of Sharpe(a) - Sharpe(b) on paired daily returns -> (2.5%, 97.5%)."""
    a, b = np.asarray(a), np.asarray(b)
    n, rng, out = len(a), np.random.default_rng(seed), []
    sh = lambda x: x.mean() / x.std() * np.sqrt(365)  # noqa: E731
    for _ in range(reps):
        idx, i = [], rng.integers(n)
        while len(idx) < n:
            idx.append(i)
            i = rng.integers(n) if rng.random() < 1 / block else (i + 1) % n
        idx = np.array(idx[:n])
        out.append(sh(a[idx]) - sh(b[idx]))
    return np.percentile(out, [2.5, 97.5])


def main():
    dd = {k: daily(s) for k, s in COINS.items()}
    idx = dd['BTC'].index.intersection(dd['ETH'].index)
    dd = {k: v.reindex(idx) for k, v in dd.items()}
    px = pd.DataFrame({k: v.close for k, v in dd.items()})
    sig = (np.log(px).diff(N) > 0).astype(float)
    sigmas, rows_fc = {m: pd.DataFrame(index=idx) for m in ('trail30', 'har', 'gbm')}, []
    for k, other in (('BTC', 'ETH'), ('ETH', 'BTC')):
        f = features(dd[k], dd[other])
        pred, har = forecasts(f)
        # H07 baseline exactly as 16_h07_oos.py: 30-day std of daily log returns (min 20) -> as a log-variance forecast
        pred['trail30'] = 2 * np.log(np.log(px[k]).diff().rolling(N, min_periods=20).std())
        for m, p in pred.items():
            sigmas[m][k] = np.sqrt(np.exp(p))
            for name, (a, b) in (('val', VAL), ('test', TEST)):
                sl = slice(a, b)
                ok = p.loc[sl].notna() & f['y'].loc[sl].notna()
                rows_fc.append(dict(coin=k, model=m, split=name, qlike=qlike(p.loc[sl][ok], f['y'].loc[sl][ok]),
                                    corr=float(np.corrcoef(p.loc[sl][ok], f['y'].loc[sl][ok])[0, 1])))
    fc = pd.DataFrame(rows_fc)
    rows = []
    net = {}
    rv_next = pd.DataFrame({k: np.sqrt(v.rv).shift(-1) for k, v in dd.items()})       # realized vol of the held day
    for m in ('trail30', 'har', 'gbm'):
        n, w = backtest(px, sig, sigmas[m])
        net[m] = n
        # risk-targeting error: per active, unclipped coin-day |log(position risk / per-coin target)|
        risk = (w * rv_next)
        target = TARGET / px.shape[1]
        unclipped = (TARGET / sigmas[m]) < CAP
        for name, (a, b) in (('val', VAL), ('test', TEST)):
            s = stats(n.loc[a:b])
            mask = (w.loc[a:b] > 0) & unclipped.loc[a:b] & rv_next.loc[a:b].notna()
            err = np.log(risk.loc[a:b][mask] / target).abs().stack().mean()
            rows.append(dict(model=m, split=name, **s, risk_target_err=float(err)))
    bt = pd.DataFrame(rows)
    for name, (a, b) in (('val', VAL), ('test', TEST)):
        for m in ('har', 'gbm'):
            x = pd.concat([net[m].loc[a:b], net['trail30'].loc[a:b]], axis=1).dropna()
            lo, hi = block_boot_diff(x.iloc[:, 0].values, x.iloc[:, 1].values)
            bt.loc[(bt.model == m) & (bt.split == name), ['dsharpe_lo', 'dsharpe_hi']] = [lo, hi]
    bhn = (px.pct_change().shift(-1) * 0.5).sum(axis=1, min_count=1).dropna()        # 50/50 BTC/ETH, no scaling
    for name, (a, b) in (('val', VAL), ('test', TEST)):
        rows.append(dict(model='buyhold_1x', split=name, **stats(bhn.loc[a:b])))
    out = pd.concat([fc.assign(kind='forecast'), bt.assign(kind='backtest'),
                     pd.DataFrame(rows[-2:]).assign(kind='benchmark')], ignore_index=True)
    out.to_csv(OUT, index=False)
    pd.set_option('display.width', 220)
    print(fc.round(4).to_string(index=False))
    print(bt.round(3).to_string(index=False))
    print(pd.DataFrame(rows[-2:]).round(3).to_string(index=False))


if __name__ == '__main__':
    main()

"""
PURPOSE:  E-h07comb-1: four pre-registered extensions of the frozen H07 trend filter — V1 multi-speed ensemble
          (10/20/30/60d), V2 fast exit (30d AND 10d), V3 10-coin universe, V4 = V1 on 10 coins — against V0 (H07) on VAL
          2018-2023 with a Bonferroni block-bootstrap rule, then TEST 2024-01..2026-09 for accepted variants only.
TAGS:     H07, TSMOM, trend following, ensemble, fast exit, universe, diversification, block bootstrap, Bonferroni,
          vol-matched drawdown, give-back, E-h07comb-1
PITFALLS: Needs ohlc_<SYM>_5m.parquet for the 10 coins in DATA_DIR (tools/fetch_crypto_dataset.py, trimmed 2026-09-30).
          Daily close = last 5m close of the UTC day; a coin trades once it has 60 days of history. TEST numbers are
          printed for every variant for transparency, but the decision uses VAL first (card rule).

Usage: DATA_DIR=/home/user/research python research/h07_combos/01_combos.py
"""
import os

import numpy as np
import pandas as pd

D = os.environ.get('DATA_DIR', '/home/user/research')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combos_results.csv')
COINS = ['BTC', 'ETH', 'SOL', 'ADA', 'BCH', 'BNB', 'DOGE', 'TRX', 'XRP', 'ZEC']
VAL, TEST = ('2018-01-01', '2023-12-31'), ('2024-01-01', '2026-09-30')
TARGET, CAP, COST_SIDE, ALPHA = 0.02, 3.0, 0.0006, 0.05 / 4


def closes():
    px = {}
    for c in COINS:
        m = pd.read_parquet(f'{D}/ohlc_{c}USDT_5m.parquet')['close']
        d = m.groupby(m.index.floor('1D')).last()
        d.index = d.index.tz_localize(None)
        px[c] = d
    return pd.DataFrame(px).sort_index()


def weights(px, sig):
    """sig (0..1) * 0.02/std30 capped at 3, divided by the number of coins with data (>= 60 days) that day."""
    lr = np.log(px).diff()
    vol = lr.rolling(30, min_periods=20).std()
    live = px.notna() & (px.notna().cumsum() >= 60)
    n = live.sum(axis=1).replace(0, np.nan)
    w = (sig.where(live) * (TARGET / vol).clip(upper=CAP)).div(n, axis=0)
    return w.fillna(0.0)


def run(px, w):
    r = px.pct_change().shift(-1)
    turn = (w - w.shift(1).fillna(0)).abs().sum(axis=1)
    return ((w * r).sum(axis=1, min_count=1).fillna(0) - turn * COST_SIDE)


def mom(px, n):
    return (np.log(px).diff(n) > 0).astype(float).where(px.notna())


def variants(px):
    two = px[['BTC', 'ETH']]
    ens = lambda p: sum(mom(p, n) for n in (10, 20, 30, 60)) / 4  # noqa: E731
    return {'V0_h07': (two, mom(two, 30)),
            'V1_ensemble': (two, ens(two)),
            'V2_fast_exit': (two, mom(two, 30) * mom(two, 10)),
            'V3_10coins': (px, mom(px, 30)),
            'V4_ens_10coins': (px, ens(px))}


def stats(x):
    eq = (1 + x).cumprod()
    return dict(sharpe=x.mean() / x.std() * np.sqrt(365), ann_ret=x.mean() * 365, ann_vol=x.std() * np.sqrt(365),
                maxdd=float((eq / eq.cummax() - 1).min()), total=float(eq.iloc[-1] - 1))


def matched_maxdd(x, ref):
    y = x * (ref.std() / x.std())
    eq = (1 + y).cumprod()
    return float((eq / eq.cummax() - 1).min())


def boot_dsharpe(a, b, alpha, block=20, reps=2000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    n, rng, out = len(a), np.random.default_rng(seed), np.empty(reps)
    sh = lambda x: x.mean() / x.std() * np.sqrt(365)  # noqa: E731
    p = 1.0 / block
    for k in range(reps):
        starts = rng.integers(n, size=n)
        jump = rng.random(n) < p
        idx = np.empty(n, dtype=np.int64)
        idx[0] = starts[0]
        for i in range(1, n):
            idx[i] = starts[i] if jump[i] else (idx[i - 1] + 1) % n
        out[k] = sh(a[idx]) - sh(b[idx])
    return np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def give_back(px, w_exit, w_pos, days=10):
    """Mean daily return of the `w_pos` position over the `days` before each exit of `w_exit` (V0 weight > 0 -> 0)."""
    vals = []
    for c in w_exit.columns:
        ex = w_exit.index[(w_exit[c].shift(1) > 0) & (w_exit[c] == 0)]
        r = w_pos[c] * px[c].pct_change().shift(-1)
        for t in ex:
            vals.append(r.loc[:t].iloc[-days - 1:-1].mean())
    return float(np.nanmean(vals)) if vals else float('nan')


def main():
    px = closes()
    V = variants(px)
    nets, ws = {}, {}
    for k, (p, s) in V.items():
        ws[k] = weights(p, s)
        nets[k] = run(p, ws[k])
    rows = []
    for split, (a, b) in (('val', VAL), ('test', TEST)):
        ref = nets['V0_h07'].loc[a:b]
        for k, n in nets.items():
            x = n.loc[a:b]
            row = dict(variant=k, split=split, **stats(x), maxdd_volmatched=matched_maxdd(x, ref),
                       avg_exposure=float(ws[k].loc[a:b].sum(axis=1).mean()),
                       turnover=float((ws[k] - ws[k].shift(1).fillna(0)).abs().sum(axis=1).loc[a:b].mean()))
            if k != 'V0_h07':
                z = pd.concat([x, ref], axis=1).dropna()
                row['dsharpe'] = row['sharpe'] - stats(ref)['sharpe']
                row['dsharpe_lo'], row['dsharpe_hi'] = boot_dsharpe(z.iloc[:, 0].values, z.iloc[:, 1].values, ALPHA)
            rows.append(row)
    R = pd.DataFrame(rows)
    a, b = VAL
    vx = lambda w: w.loc[a:b]  # noqa: E731
    two = px[['BTC', 'ETH']].loc[a:b]
    diag = {f'{k}_giveback_bp': give_back(two, vx(ws['V0_h07']), vx(ws[k])) * 1e4
            for k in ('V0_h07', 'V1_ensemble', 'V2_fast_exit')}
    per_coin = pd.DataFrame({c: run(px[[c]], weights(px[[c]], mom(px[[c]], 30))) for c in COINS}).loc[a:b]
    per_coin = per_coin.loc[:, per_coin.abs().sum() > 0]
    cm = per_coin.corr().values
    diag['V3_mean_pairwise_corr'] = float(cm[np.triu_indices_from(cm, 1)].mean())
    R.to_csv(OUT, index=False)
    pd.set_option('display.width', 250)
    print(R.round(3).to_string(index=False))
    print({k: round(v, 3) for k, v in diag.items()})


if __name__ == '__main__':
    main()

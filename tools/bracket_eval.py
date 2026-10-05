"""
PURPOSE:  Evaluate a model's trades with a symmetric bracket (stop distance = take-profit distance) on all samples and then
          on filtered subsets (ATR/NATR, ADX, SuperTrend, predicted high/low room, any pandas expression), each against a
          random-direction null on the SAME samples, with day-clustered t, per-coin consistency, and lower/upper bounds for
          bars where both barriers were touched (order unknown from bar high/low).
TAGS:     bracket, stop loss, take profit, symmetric, filters, selective trading, predicted high low, tp_room, sl_room,
          break-even hit rate, random-direction null, day-clustered t, evaluation, trades
PITFALLS: With only the horizon's max high / min low the order of touches is unknown: 'stop_first' (default) is the honest
          lower bound, 'tp_first' the upper bound — trust a filter only if the LB is positive. Entry is the signal candle's
          close (last_close); costs are round-trip. Choosing the best of many filters on test is data snooping: pick filters
          on val, confirm once on test (docs/research/RESEARCH_RULES.md). Related closed failures: F-0032, F-0037, F-0057,
          F-0062 in docs/research/failure_registry.csv.

Usage (e.g. in main.ipynb after model.predict on the test split):
    import sys; sys.path.insert(0, "tools"); import bracket_eval as be
    df = be.frame_from_split(test_split, preds=model.predict(x_test), reg_scale=dataset["reg_target_scale"],
                             extra=entry_table)                       # entry_table: asset, ts, NATR_14, ADX_14, ...
    table = be.evaluate_filters(df, {"all": None, "natr>1": "NATR_14 > 1", "adx>20": "ADX_14 > 20",
                                     "room": "tp_room >= bracket and sl_room < bracket"}, bracket=0.01)
"""
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # core/ sits next to tools/ (the script is imported with only tools/ on the path)
from core.schema import LAST_COLUMNS  # noqa: E402,F401  columns of the pipeline's last_candles array (single source: core/schema.py)
COST_RT = 0.0014                                    # 0.07% per side taker, as in the edge-discovery studies


def breakeven_hit_rate(bracket: float, cost_rt: float = COST_RT) -> float:
    """Hit rate needed when every trade ends at ±bracket: p·b − (1−p)·b − c = 0  →  p = 0.5 + c / (2b)."""
    return 0.5 + cost_rt / (2.0 * bracket)


def _first_col(a) -> np.ndarray:
    a = np.asarray(a, dtype='float64')
    return a[:, 0] if a.ndim > 1 else a


def frame_from_split(split, preds: Optional[Dict[str, np.ndarray]] = None, reg_scale: float = 1.0,
                     extra: Optional[pd.DataFrame] = None, class_head: str = 'close') -> pd.DataFrame:
    """One row per sample: asset, ts, entry and future path from ``last_candles``; model outputs if ``preds`` given.

    ``split``: a pipeline split dict (``last_candles``) or a per-asset dict of them (test with keep_asset_test_separate).
    ``preds``: ``model.predict`` output dict. Uses ``y_<class_head>_class_logits`` (probability of up) for ``p_up`` and
    ``y_high`` / ``y_low`` / ``y_close`` (return × reg_scale, reg_target_mode='return') for predicted prices: the high and
    low targets are relative to the last candle's own high/low, so pred_high = last_high·(1+r).
    ``extra``: a table with asset, ts and any raw entry-candle columns (NATR_14, ADX_14, SUPERT_D...) joined on (asset, ts).
    """
    parts = split.items() if 'last_candles' not in split else [('all', split)]
    frames = []
    for name, s in parts:
        lc = np.asarray(s['last_candles'], dtype='float64')
        d = pd.DataFrame(lc, columns=list(LAST_COLUMNS))
        d.insert(0, 'asset', name)
        d['ts'] = pd.to_datetime(d.pop('timestamp').astype('int64'), utc=True)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df['entry'] = df['last_close']
    if preds is not None:
        n = len(df)
        get = lambda k: _first_col(preds[k])[:n] if k in preds else None  # noqa: E731
        p = get(f'y_{class_head}_class_logits')
        if p is not None:
            df['p_up'] = p
        for t, anchor in (('high', 'last_high'), ('low', 'last_low'), ('close', 'last_close')):
            r = get(f'y_{t}')
            if r is not None:
                df[f'pred_{t}'] = df[anchor] * (1.0 + r / float(reg_scale))
    if extra is not None:
        df = df.merge(extra, on=['asset', 'ts'], how='left', validate='one_to_one')
    return df


def add_rooms(df: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    """Side-aware predicted room from entry: ``up_room``/``dn_room`` (fractions), ``tp_room`` toward the trade, ``sl_room``
    against it, ``asym`` = up_room − dn_room. Needs pred_high / pred_low."""
    out = df.copy()
    if {'pred_high', 'pred_low'} <= set(out.columns):
        out['up_room'] = out['pred_high'] / out['entry'] - 1.0
        out['dn_room'] = 1.0 - out['pred_low'] / out['entry']
        out['asym'] = out['up_room'] - out['dn_room']
        long = np.asarray(direction) > 0
        out['tp_room'] = np.where(long, out['up_room'], out['dn_room'])
        out['sl_room'] = np.where(long, out['dn_room'], out['up_room'])
    return out


def bracket_returns(df: pd.DataFrame, direction: np.ndarray, bracket, cost_rt: float = COST_RT,
                    ambiguous: str = 'stop_first') -> np.ndarray:
    """Net return per trade with a symmetric bracket ±``bracket`` (fraction, scalar or per-row) around ``entry``.

    Hit only the target → +b; only the stop → −b; both in the horizon → by ``ambiguous`` ('stop_first' lower bound,
    'tp_first' upper bound); neither → exit at future_close. direction 0 = no trade (NaN)."""
    if ambiguous not in ('stop_first', 'tp_first'):
        raise ValueError(f"ambiguous: 'stop_first' | 'tp_first', not {ambiguous!r}")
    e = df['entry'].to_numpy(float)
    b = np.broadcast_to(np.asarray(bracket, dtype=float), e.shape)
    d = np.asarray(direction, dtype=float)
    up = df['future_high_max'].to_numpy(float) / e - 1.0
    dn = 1.0 - df['future_low_min'].to_numpy(float) / e
    close = df['future_close'].to_numpy(float) / e - 1.0
    fav = np.where(d > 0, up, dn)          # excursion toward the trade
    adv = np.where(d > 0, dn, up)          # excursion against it
    hit_tp, hit_sl = fav >= b, adv >= b
    gross = d * close
    gross = np.where(hit_tp & ~hit_sl, b, gross)
    gross = np.where(hit_sl & ~hit_tp, -b, gross)
    both = hit_tp & hit_sl
    gross = np.where(both, -b if ambiguous == 'stop_first' else b, gross)
    return np.where(d == 0, np.nan, gross - cost_rt)


def day_clustered_t(x: np.ndarray, ts) -> float:
    """t of the mean with errors clustered by UTC day (trades on the same day are not independent)."""
    x = np.asarray(x, float)
    g = pd.DataFrame({'r': x, 'd': pd.DatetimeIndex(pd.to_datetime(ts)).floor('1D')}).groupby('d').r.agg(['sum', 'count'])
    mu = x.mean()
    se = np.sqrt(((g['sum'] - g['count'] * mu) ** 2).sum()) / len(x)
    return float(mu / se) if se > 0 else float('nan')


def evaluate_filters(df: pd.DataFrame, filters: Dict[str, Optional[str]], bracket=0.01, direction=None,
                     cost_rt: float = COST_RT, null_reps: int = 200, seed: int = 0, min_trades: int = 30) -> pd.DataFrame:
    """One row per filter (``None`` = all samples; else a ``DataFrame.eval`` expression over df's columns plus up_room,
    dn_room, tp_room, sl_room, asym and ``bracket``). ``bracket``: fraction, or a column name (e.g. per-row k·NATR).
    ``direction``: array of ±1/0, or None → sign(p_up − 0.5), else sign(asym).

    Columns: n, coverage, win (share of trades with LB return > 0), breakeven_hit, net_bp_lb / net_bp_ub (ambiguous bars as stop / as
    target), t_lb (day-clustered), coins_pos (assets with mean LB > 0), null_bp (random directions on the same samples,
    mean), null_p (share of random-direction runs with mean ≥ the model's LB)."""
    if direction is None:
        if 'p_up' in df:
            direction = np.sign(df['p_up'].to_numpy(float) - 0.5)
        elif {'pred_high', 'pred_low'} <= set(df.columns):
            direction = np.sign(add_rooms(df, np.ones(len(df)))['asym'].to_numpy(float))
        else:
            raise ValueError("direction: pass it, or provide p_up or pred_high/pred_low")
    direction = np.asarray(direction, dtype=float)
    base = add_rooms(df, direction)
    b = base[bracket].to_numpy(float) if isinstance(bracket, str) else np.full(len(base), float(bracket))
    base['bracket'] = b
    rng = np.random.default_rng(seed)
    rows = []
    for name, expr in filters.items():
        m = np.ones(len(base), bool) if expr is None else np.array(base.eval(expr), dtype=bool)
        m &= direction != 0
        sub, dsub, bsub = base[m], direction[m], b[m]
        row = {'filter': name, 'expr': expr or 'all samples', 'n': int(m.sum()), 'coverage': float(m.mean())}
        if row['n'] < min_trades:
            rows.append({**row, 'note': f'< {min_trades} trades'})
            continue
        lb = bracket_returns(sub, dsub, bsub, cost_rt, 'stop_first')
        ub = bracket_returns(sub, dsub, bsub, cost_rt, 'tp_first')
        null = np.array([np.nanmean(bracket_returns(sub, rng.choice([-1.0, 1.0], len(sub)), bsub, cost_rt))
                         for _ in range(null_reps)])
        per_coin = pd.Series(lb).groupby(sub['asset'].to_numpy()).mean()
        row.update(win=float((lb > 0).mean()), breakeven_hit=float(np.mean(breakeven_hit_rate(bsub, cost_rt))),
                   net_bp_lb=float(np.nanmean(lb) * 1e4), net_bp_ub=float(np.nanmean(ub) * 1e4),
                   t_lb=day_clustered_t(lb, sub['ts']), coins_pos=f"{int((per_coin > 0).sum())}/{len(per_coin)}",
                   null_bp=float(null.mean() * 1e4), null_p=float((null >= np.nanmean(lb)).mean()))
        rows.append(row)
    return pd.DataFrame(rows)

"""
PURPOSE:  Calibrated next-bar range forecast from NATR_14 (phase R3, card E-natr-001): causal NATR and next-range targets
          (range, up and down excursion, percent of the decision close), an exact pooled quantile regression
          log(range_{t+1}) = a + b log(NATR_t) at tau 0.1/0.5/0.9, the per-coin trailing-90-bar empirical-quantile baseline,
          NATR deciles fixed on train, coverage / pinball / monthly rank IC, and the fixed accept rule (80 +/- 3 pp overall,
          80 +/- 6 pp per decile, pinball better in >= 8/10 deciles, monthly IC > 0 in >= 90 % of months).
TAGS:     NATR, range forecast, quantile regression, calibration, coverage, pinball loss, trailing quantile baseline,
          volatility clustering, H003, R3, E-natr-001, bot2 get_range_forecast
PITFALLS: A bar exists only if ALL its source members exist and is used only after it closed; row t = decision at the close
          of bar t, target = bar t+1 divided by the close of bar t. The baseline needs 90 consecutive valid bars (NaN
          otherwise) and rows without it are dropped from model AND baseline in the comparison. Deciles come from train only.
          Forecast quantiles are sorted across tau (rearrangement); the crossed-row count is returned. A target is floored at
          EPS_PCT before the log. Thresholds are constants of the card, not parameters: do not tune them.
"""
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

TAUS = (0.1, 0.5, 0.9)
TARGETS = ("range", "up", "down")
EPS_PCT = 1e-4                       # percent; floor before the log
NATR_PERIOD = 14
BASELINE_WINDOW = 90
# accept rule (card E-natr-001, ROADMAP R3): constants, not parameters
COVER_TARGET, OVERALL_TOL, DECILE_TOL = 0.80, 0.03, 0.06
MIN_PINBALL_DECILES = 8
MIN_IC_MONTH_SHARE_PCT = 90          # integer percent: positives * 100 >= 90 * months
BTC_ETH = ("BTCUSDT", "ETHUSDT")


def qname(tau):
    return f"q{int(round(tau * 100))}"


# ------------------------------------------------------------------ bars and rows
def resample_bars(df, minutes, source_minutes=5):
    """UTC resample of source bars to `minutes`; label = bar open; a bar needs ALL its members (else dropped)."""
    k = minutes // source_minutes
    if k * source_minutes != minutes:
        raise ValueError("minutes must be a multiple of source_minutes")
    if k == 1:
        return df.copy()
    key = df.index.floor(f"{minutes}min")
    g = df.groupby(key)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum() if "volume" in df.columns else np.nan, "_n": g["close"].size()})
    out = out[out["_n"] == k].drop(columns="_n")
    out.index.name = df.index.name
    return out


def _grid(bars, step_minutes):
    return bars.reindex(pd.date_range(bars.index[0], bars.index[-1], freq=f"{step_minutes}min"))


def natr_pct(bars, step_minutes, period=NATR_PERIOD):
    """NATR (percent of close), ATR = Wilder smoothing of the true range on the regular grid; NaN where bars are missing."""
    g = _grid(bars, step_minutes)
    cp = g["close"].shift(1)
    tr = pd.concat([g["high"] - g["low"], (g["high"] - cp).abs(), (g["low"] - cp).abs()], axis=1).max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period, ignore_na=True).mean()
    return 100.0 * atr / g["close"]


def build_rows(bars, step_minutes, period=NATR_PERIOD, window=BASELINE_WINDOW):
    """Rows indexed by decision-bar open t: natr, y_range/y_up/y_down (percent of C_t, bar t+1), tgt_open, baseline columns
    b_<target>_q10/q50/q90 (empirical quantiles of the target-series values of bars t-window+1..t, known at the close of t)."""
    g = _grid(bars, step_minutes)
    cp = g["close"].shift(1)
    series = {"range": 100.0 * (g["high"] - g["low"]) / cp,
              "up": 100.0 * (g["high"] - cp).clip(lower=0.0) / cp,
              "down": 100.0 * (cp - g["low"]).clip(lower=0.0) / cp}
    rows = pd.DataFrame({"natr": natr_pct(bars, step_minutes, period)})
    for tg, s in series.items():
        rows[f"y_{tg}"] = s.shift(-1)
    rows["tgt_open"] = g.index + pd.Timedelta(minutes=step_minutes)
    for tg, s in series.items():
        roll = s.rolling(window, min_periods=window)
        for t in TAUS:
            rows[f"b_{tg}_{qname(t)}"] = roll.quantile(t)
    rows = rows[rows["natr"].notna() & rows["y_range"].notna() & rows["y_up"].notna() & rows["y_down"].notna()]
    return rows[rows["natr"] > 0]


# ------------------------------------------------------------------ model
def pinball(y, q, tau):
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.maximum(tau * d, (tau - 1.0) * d)


def fit_quantile(x, y, tau):
    """Exact linear quantile regression y = a + b x (two parameters): profile over b, a = tau-quantile of y - b x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def cost(b):
        r = y - b * x
        a = np.quantile(r, tau)
        d = r - a
        return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))

    res = minimize_scalar(cost, bounds=(-10.0, 10.0), method="bounded", options={"xatol": 1e-10, "maxiter": 500})
    b = float(res.x)
    return float(np.quantile(y - b * x, tau)), b


def fit_models(train):
    """{(target, tau): (a, b)} for log y = a + b log natr, fitted on `train` rows only."""
    lx = np.log(train["natr"].to_numpy(dtype=float))
    models = {}
    for tg in TARGETS:
        ly = np.log(np.maximum(train[f"y_{tg}"].to_numpy(dtype=float), EPS_PCT))
        for t in TAUS:
            models[(tg, t)] = fit_quantile(lx, ly, t)
    return models


def predict_quantiles(models, target, natr):
    """(n, 3) forecast quantiles in percent, sorted across tau; second value = number of rows that were crossed."""
    lx = np.log(np.asarray(natr, dtype=float))
    raw = np.column_stack([np.exp(models[(target, t)][0] + models[(target, t)][1] * lx) for t in TAUS])
    crossed = int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
    return np.sort(raw, axis=1), crossed


# ------------------------------------------------------------------ metrics
def decile_edges(natr_train):
    return np.quantile(np.asarray(natr_train, dtype=float), np.arange(1, 10) / 10.0)


def assign_decile(natr, edges):
    return np.searchsorted(np.asarray(edges), np.asarray(natr, dtype=float), side="right")


def coverage(y, lo, hi):
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi)))) if len(y) else float("nan")


def monthly_ic(index, x, y):
    """Spearman(x, y) per calendar month of `index` (pooled rows); Series indexed by month Period."""
    idx = pd.DatetimeIndex(index)
    month = idx.tz_convert(None).to_period("M") if idx.tz is not None else idx.to_period("M")
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float), "m": month})
    return df.groupby("m").apply(lambda d: d["x"].rank().corr(d["y"].rank()), include_groups=False)


def evaluate_rule(cov, cov_dec, pin_model, pin_base, ic):
    """The fixed accept rule of E-natr-001 (one timeframe). Inputs: overall coverage (0..1), 10 decile coverages,
    10 decile pinball losses of model and baseline, the monthly ICs."""
    cov_dec = np.asarray(cov_dec, dtype=float)
    pm, pb = np.asarray(pin_model, dtype=float), np.asarray(pin_base, dtype=float)
    ic = np.asarray(ic, dtype=float)
    eps = 1e-12
    overall_ok = bool(np.isfinite(cov) and abs(cov - COVER_TARGET) <= OVERALL_TOL + eps)
    deciles_ok = bool(len(cov_dec) == 10 and np.isfinite(cov_dec).all() and (np.abs(cov_dec - COVER_TARGET) <= DECILE_TOL + eps).all())
    better = int(np.sum(np.isfinite(pm) & np.isfinite(pb) & (pm < pb)))
    pinball_ok = better >= MIN_PINBALL_DECILES
    pos = int(np.sum(ic > 0))
    ic_ok = bool(len(ic) > 0 and pos * 100 >= MIN_IC_MONTH_SHARE_PCT * len(ic))
    return {"coverage": float(cov), "overall_ok": overall_ok, "deciles_ok": deciles_ok, "pinball_deciles_better": better,
            "pinball_ok": pinball_ok, "ic_months": int(len(ic)), "ic_positive_months": pos, "ic_ok": ic_ok,
            "accepted": bool(overall_ok and deciles_ok and pinball_ok and ic_ok)}


def _baseline_matrix(df, target):
    return df[[f"b_{target}_{qname(t)}" for t in TAUS]].to_numpy(dtype=float)


def _mean_pinball(y, q):
    return float(np.mean([pinball(y, q[:, i], t).mean() for i, t in enumerate(TAUS)]))


def evaluate_timeframe(train, val):
    """Fit on `train` rows, evaluate on `val` rows (both from build_rows, optional `coin` column). Returns the models, edges,
    decile / month / slice tables, extras and the accept-rule verdict."""
    models = fit_models(train)
    edges = decile_edges(train["natr"])
    need = [f"b_{tg}_{qname(t)}" for tg in TARGETS for t in TAUS]
    v = val.dropna(subset=need).copy()
    q, ncross = predict_quantiles(models, "range", v["natr"])
    base = _baseline_matrix(v, "range")
    y = v["y_range"].to_numpy(dtype=float)
    dec = assign_decile(v["natr"], edges)
    covered = (y >= q[:, 0]) & (y <= q[:, 2])
    bcovered = (y >= base[:, 0]) & (y <= base[:, 2])
    rows = []
    for d in range(10):
        m = dec == d
        pm = _mean_pinball(y[m], q[m]) if m.any() else float("nan")
        pb = _mean_pinball(y[m], base[m]) if m.any() else float("nan")
        rows.append({"decile": d, "n": int(m.sum()), "coverage": float(covered[m].mean()) if m.any() else float("nan"),
                     "baseline_coverage": float(bcovered[m].mean()) if m.any() else float("nan"),
                     "pinball_model": pm, "pinball_baseline": pb, "pinball_ratio": pm / pb if m.any() else float("nan")})
    deciles = pd.DataFrame(rows)
    ic = monthly_ic(v["tgt_open"], v["natr"], y)
    months = pd.DataFrame({"month": ic.index.astype(str), "ic": ic.to_numpy()})
    verdict = evaluate_rule(float(covered.mean()), deciles["coverage"], deciles["pinball_model"], deciles["pinball_baseline"], ic.to_numpy())
    # slices: decile x {BTC/ETH, others} x year (reported, not rules)
    group = np.where(v["coin"].isin(BTC_ETH), "BTC_ETH", "others") if "coin" in v.columns else np.full(len(v), "all")
    year = pd.DatetimeIndex(v["tgt_open"]).year.to_numpy()
    srows = []
    for gname in sorted(set(group)):
        for yr in sorted(set(year)):
            for d in range(10):
                m = (group == gname) & (year == yr) & (dec == d)
                if not m.any():
                    continue
                pm, pb = _mean_pinball(y[m], q[m]), _mean_pinball(y[m], base[m])
                srows.append({"group": gname, "year": int(yr), "decile": d, "n": int(m.sum()), "coverage": float(covered[m].mean()),
                              "pinball_ratio": pm / pb})
    extras = {"n_val_rows": int(len(v)), "n_val_rows_dropped_no_baseline": int(len(val) - len(v)), "baseline_coverage": float(bcovered.mean()),
              "pinball_model": _mean_pinball(y, q), "pinball_baseline": _mean_pinball(y, base)}
    for tg in ("up", "down"):
        qq, _ = predict_quantiles(models, tg, v["natr"])
        yy = v[f"y_{tg}"].to_numpy(dtype=float)
        bb = _baseline_matrix(v, tg)
        extras[f"{tg}_p50_pinball_model"] = float(pinball(yy, qq[:, 1], 0.5).mean())
        extras[f"{tg}_p50_pinball_baseline"] = float(pinball(yy, bb[:, 1], 0.5).mean())
    qt, nct = predict_quantiles(models, "range", train["natr"])
    ytr = train["y_range"].to_numpy(dtype=float)
    ctr = (ytr >= qt[:, 0]) & (ytr <= qt[:, 2])
    tyear = pd.DatetimeIndex(train["tgt_open"]).year.to_numpy()
    extras["train_coverage"] = float(ctr.mean())
    extras["train_coverage_by_year"] = {int(yr): float(ctr[tyear == yr].mean()) for yr in sorted(set(tyear))}
    extras["train_n"] = int(len(train))
    extras["n_crossed_train"] = nct
    return {"models": models, "edges": edges, "deciles": deciles, "months": months, "slices": pd.DataFrame(srows), "verdict": verdict,
            "n_crossed": ncross, "extras": extras}

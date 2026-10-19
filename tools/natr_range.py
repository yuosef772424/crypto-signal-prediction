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
# E-natr-003 (high-NATR calibration): constants of the card, not parameters
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")   # nothing at or after this open time may be used for fitting (exploration val starts here)
TOP_DECILE = 9
CORR_MIN_N = 200                                    # a decile with fewer calibration rows keeps factor 1 (reported)
LOC_STATIC_POOLED_MAX, LOC_STATIC_FOLD_MAX, LOC_STATIC_MIN_FOLDS = 0.78, 0.79, 3
LOC_DRIFT_POOLED_MIN, LOC_VAL_UNDER = 0.79, 0.78
# E-natr-004 (rolling recalibration): constants of the card, not parameters
ROLL_WINDOW_MONTHS = 12                             # trailing window of realized rows before each monthly refit
ROLL_MIN_N = 200                                    # a decile with fewer window rows uses the POOLED factor (reported)
ROLL_STEP_MINUTES = 60                              # a row's target bar ends tgt_open + this; it must end before the refit month starts
STAB_MAX_LN, STAB_MEAN_LN = 0.05, 0.02              # rule 9: max / mean |ln(c_m / c_{m-1})| between consecutive refits
QUARTER_MIN_N, QUARTER_MIN_EVALUABLE = 200, 5       # rule 8


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


def fit_models(train, targets=TARGETS):
    """{(target, tau): (a, b)} for log y = a + b log natr, fitted on `train` rows only (`targets` default = all three)."""
    lx = np.log(train["natr"].to_numpy(dtype=float))
    models = {}
    for tg in targets:
        ly = np.log(np.maximum(train[f"y_{tg}"].to_numpy(dtype=float), EPS_PCT))
        for t in TAUS:
            models[(tg, t)] = fit_quantile(lx, ly, t)
    return models


def predict_quantiles(models, target, natr, corrections=None):
    """(n, 3) forecast quantiles in percent, sorted across tau; second value = number of rows that were crossed.
    `corrections` (default None = E-natr-001 behaviour): {target: {"edges": 9 NATR cut points, "factors": (10, 3) array}}; the raw
    quantile of a row in NATR decile d is multiplied by factors[d] (before the rearrangement) for that target."""
    lx = np.log(np.asarray(natr, dtype=float))
    raw = np.column_stack([np.exp(models[(target, t)][0] + models[(target, t)][1] * lx) for t in TAUS])
    c = (corrections or {}).get(target)
    if c is not None and c.get("row_factors") is not None:           # E-natr-004: one (c10, c50, c90) triple per row (rolling refits)
        rf = np.asarray(c["row_factors"], dtype=float)
        if rf.shape != raw.shape:
            raise ValueError(f"row_factors shape {rf.shape} does not match the forecast rows {raw.shape}")
        raw = raw * rf
    elif c is not None:
        raw = raw * np.asarray(c["factors"], dtype=float)[assign_decile(np.asarray(natr, dtype=float), c["edges"])]
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


def evaluate_frozen(models, edges, val, corrections=None):
    """Evaluate FROZEN `models` and decile `edges` on `val` rows (build_rows output, optional `coin` column): decile / month /
    slice tables, extras and the accept-rule verdict. Nothing is fitted here (used by the H2 confirmation). `corrections`: see
    predict_quantiles (default None = unchanged)."""
    need = [f"b_{tg}_{qname(t)}" for tg in TARGETS for t in TAUS]
    v = val.dropna(subset=need).copy()
    q, ncross = predict_quantiles(models, "range", v["natr"], corrections)
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
        qq, _ = predict_quantiles(models, tg, v["natr"], corrections)
        yy = v[f"y_{tg}"].to_numpy(dtype=float)
        bb = _baseline_matrix(v, tg)
        extras[f"{tg}_p50_pinball_model"] = float(pinball(yy, qq[:, 1], 0.5).mean())
        extras[f"{tg}_p50_pinball_baseline"] = float(pinball(yy, bb[:, 1], 0.5).mean())
    return {"models": models, "edges": edges, "deciles": deciles, "months": months, "slices": pd.DataFrame(srows), "verdict": verdict,
            "n_crossed": ncross, "extras": extras}


def evaluate_timeframe(train, val):
    """Fit on `train` rows, evaluate on `val` rows (both from build_rows, optional `coin` column). Returns the models, edges,
    decile / month / slice tables, extras (with the in-sample train coverage) and the accept-rule verdict."""
    models = fit_models(train)
    edges = decile_edges(train["natr"])
    res = evaluate_frozen(models, edges, val)
    extras = res["extras"]
    qt, nct = predict_quantiles(models, "range", train["natr"])
    ytr = train["y_range"].to_numpy(dtype=float)
    ctr = (ytr >= qt[:, 0]) & (ytr <= qt[:, 2])
    tyear = pd.DatetimeIndex(train["tgt_open"]).year.to_numpy()
    extras["train_coverage"] = float(ctr.mean())
    extras["train_coverage_by_year"] = {int(yr): float(ctr[tyear == yr].mean()) for yr in sorted(set(tyear))}
    extras["train_n"] = int(len(train))
    extras["n_crossed_train"] = nct
    return res


# ------------------------------------------------------------------ E-natr-003: high-NATR calibration (train-only)
def drop_stale_runs(df, min_run_hours=168, source_minutes=15):
    """Self-contained stale-bar mask: drops source bars inside runs of >= min_run_hours of CONSECUTIVE flat zero-volume bars
    (volume == 0 and high == low; padded history of dead / migrated coins). Shorter zero-volume stretches are real quiet bars and
    stay. A run breaks at a missing timestamp. Dropped rows leave a hole, so every 1h bar that touches them is not built.
    Needs a `volume` column; the original bars are not modified."""
    if "volume" not in df.columns:
        raise KeyError("drop_stale_runs needs a volume column")
    need = int(round(min_run_hours * 60 / source_minutes))
    flat = ((df["volume"] == 0) & (df["high"] == df["low"])).to_numpy()
    step = pd.Timedelta(minutes=source_minutes)
    gap = np.r_[True, np.asarray((df.index[1:] - df.index[:-1]) != step)]
    run_id = np.cumsum(gap | np.r_[True, flat[1:] != flat[:-1]])    # a new run starts at a gap or where flat / non-flat changes
    key = pd.Series(run_id, index=df.index)
    size = key.groupby(key).transform("size").to_numpy()
    return df[~(flat & (size >= need))]


def _assert_fit_window(rows, end, what):
    end = pd.Timestamp(end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end
    if end > TRAIN_END:
        raise ValueError(f"{what}: fit window end {end} is after the train end {TRAIN_END} (no 2025+ data may be used for fitting)")
    if len(rows) and pd.DatetimeIndex(rows["tgt_open"]).max() >= end:
        raise ValueError(f"{what}: rows with target open time >= {end} (the fit window end) are not allowed")


def fit_models_until(rows, end):
    """fit_models on `rows`, refusing any row whose target open time is at or after `end` (<= TRAIN_END)."""
    _assert_fit_window(rows, end, "fit_models_until")
    return fit_models(rows)


def fit_decile_corrections(models, edges, calib, fit_end, min_n=CORR_MIN_N):
    """Per-NATR-decile multiplicative split-conformal (CQR-style, ratio form) correction of the range p10 / p90, TRAIN ONLY.
    On the calibration rows of decile d: c90_d = the ceil((n+1)*0.9)-th smallest of y / q90, c10_d = the floor((n+1)*0.1)-th smallest of
    y / q10 (raw, rearranged forecasts of `models`); factors row = [c10_d, 1, c90_d]. A decile with n < min_n keeps [1, 1, 1].
    `fit_end` (<= TRAIN_END) is required and every calibration row must open before it. Returns {"range": {"edges", "factors", "n"}}."""
    _assert_fit_window(calib, fit_end, "fit_decile_corrections")
    q, _ = predict_quantiles(models, "range", calib["natr"])
    y = calib["y_range"].to_numpy(dtype=float)
    dec = assign_decile(calib["natr"], edges)
    factors = np.ones((10, 3))
    ns = np.zeros(10, dtype=int)
    for d in range(10):
        m = dec == d
        n = int(m.sum())
        ns[d] = n
        if n < min_n:
            continue
        k_hi = min(n, int(math.ceil((n + 1) * 0.9)))
        k_lo = max(1, int(math.floor((n + 1) * 0.1)))
        factors[d, 2] = np.sort(y[m] / q[m, 2])[k_hi - 1]
        factors[d, 0] = np.sort(y[m] / q[m, 0])[k_lo - 1]
    return {"range": {"edges": np.asarray(edges, dtype=float), "factors": factors, "n": ns}}


def decile_year_coverage(models, edges, val, corrections=None):
    """DataFrame year x decile: n and [p10, p90] coverage of the (optionally corrected) range forecast on `val` rows."""
    q, _ = predict_quantiles(models, "range", val["natr"], corrections)
    y = val["y_range"].to_numpy(dtype=float)
    cov = (y >= q[:, 0]) & (y <= q[:, 2])
    dec = assign_decile(val["natr"], edges)
    year = pd.DatetimeIndex(val["tgt_open"]).year.to_numpy()
    rows = []
    for yr in sorted(set(year)):
        for d in range(10):
            m = (year == yr) & (dec == d)
            rows.append({"year": int(yr), "decile": d, "n": int(m.sum()), "coverage": float(cov[m].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def oof_decile_diagnostic(rows, first_year=2021, last_year=2024, start_year=2020):
    """Forward-chain out-of-fold decile coverage inside the train period: fold Y fits (models + decile edges) on target years
    start_year..Y-1 and covers year Y. Rows at or after TRAIN_END are refused. Columns: fold, decile, n, coverage."""
    _assert_fit_window(rows, TRAIN_END, "oof_decile_diagnostic")
    year = pd.DatetimeIndex(rows["tgt_open"]).year.to_numpy()
    out = []
    for fy in range(first_year, last_year + 1):
        fit, test = rows[(year >= start_year) & (year < fy)], rows[year == fy]
        if len(fit) == 0 or len(test) == 0:
            continue
        models, edges = fit_models(fit, targets=("range",)), decile_edges(fit["natr"])
        t = decile_year_coverage(models, edges, test)
        out.append(t.rename(columns={"year": "fold"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["fold", "decile", "n", "coverage"])


def classify_localization(oof, control_val_top):
    """Pre-registered localization of the top-decile under-coverage (card E-natr-003). `oof` = oof_decile_diagnostic output,
    `control_val_top` = {val year: top-decile coverage of the UNCORRECTED base model}.
    static: pooled OOF top-decile coverage <= 78.0 % AND >= 3 folds <= 79.0 %;  drift: pooled OOF >= 79.0 % AND the control's top decile is
    < 78.0 % in at least one val year;  not_reproduced: pooled OOF >= 79.0 % and no val year < 78.0 %;  mixed: anything else."""
    t = oof[oof["decile"] == TOP_DECILE]
    n = float(t["n"].sum())
    pooled = float((t["coverage"] * t["n"]).sum() / n) if n > 0 else float("nan")
    folds_low = int((t["coverage"] <= LOC_STATIC_FOLD_MAX + 1e-12).sum())
    val_low = any(v < LOC_VAL_UNDER for v in control_val_top.values())
    if np.isfinite(pooled) and pooled <= LOC_STATIC_POOLED_MAX + 1e-12 and folds_low >= LOC_STATIC_MIN_FOLDS:
        label = "static"
    elif np.isfinite(pooled) and pooled >= LOC_DRIFT_POOLED_MIN - 1e-12:
        label = "drift" if val_low else "not_reproduced"
    else:
        label = "mixed"
    return {"label": label, "pooled_oof_top_coverage": pooled, "folds_at_or_below_79": folds_low, "n_folds": int(len(t)),
            "oof_top_by_fold": {int(f): float(c) for f, c in zip(t["fold"], t["coverage"])},
            "control_val_top_by_year": {int(k): float(v) for k, v in control_val_top.items()}}


# ------------------------------------------------------------------ E-natr-004: rolling (adaptive) decile recalibration
def _utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _tail_factors(y, q):
    """(c10, c90): the floor((n+1)*0.1)-th / ceil((n+1)*0.9)-th smallest of y / q10 and y / q90 (the E-natr-003 formula), n >= 1."""
    n = len(y)
    k_hi = min(n, int(math.ceil((n + 1) * 0.9)))
    k_lo = max(1, int(math.floor((n + 1) * 0.1)))
    return float(np.sort(y / q[:, 0])[k_lo - 1]), float(np.sort(y / q[:, 2])[k_hi - 1])


def rolling_window(rows, month_start, window_months=ROLL_WINDOW_MONTHS, step_minutes=ROLL_STEP_MINUTES):
    """Realized rows usable for the refit at `month_start`: target open in [month_start - window_months, month_start) AND the target bar has
    ended (tgt_open + step_minutes <= month_start). Rows with a non-finite natr or y_range are dropped."""
    m = _utc(month_start)
    start = m - pd.DateOffset(months=int(window_months))
    t = pd.DatetimeIndex(rows["tgt_open"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    ok = (t >= start) & (t + pd.Timedelta(minutes=step_minutes) <= m)
    sub = rows[np.asarray(ok)]
    return sub[np.isfinite(sub["natr"].to_numpy(dtype=float)) & np.isfinite(sub["y_range"].to_numpy(dtype=float))]


def fit_month_factors(models, edges, window, month_start, min_n=ROLL_MIN_N, step_minutes=ROLL_STEP_MINUTES):
    """Factors (10, 3) = [c10_d, 1, c90_d] for the refit at `month_start` from the realized `window` rows (base `models`, fixed `edges`).
    STRICTLY-PAST GUARD: raises ValueError if any window row has tgt_open + step_minutes > month_start. A decile with fewer than `min_n`
    window rows uses the pooled factors (all window rows); fewer than `min_n` rows in total raises. Returns {"factors", "n", "fallback", "n_window"}."""
    m = _utc(month_start)
    t = pd.DatetimeIndex(window["tgt_open"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    if len(t) and (t + pd.Timedelta(minutes=step_minutes)).max() > m:
        raise ValueError(f"fit_month_factors: a window row's target ends after the refit month start {m} (look-ahead)")
    if len(window) < min_n:
        raise ValueError(f"fit_month_factors: {len(window)} window rows < {min_n}")
    q, _ = predict_quantiles(models, "range", window["natr"])
    y = window["y_range"].to_numpy(dtype=float)
    dec = assign_decile(window["natr"], edges)
    pooled = _tail_factors(y, q)
    factors = np.ones((10, 3))
    ns = np.zeros(10, dtype=int)
    fb = np.zeros(10, dtype=bool)
    for d in range(10):
        sel = dec == d
        ns[d] = int(sel.sum())
        if ns[d] < min_n:
            fb[d] = True
            factors[d, 0], factors[d, 2] = pooled
        else:
            factors[d, 0], factors[d, 2] = _tail_factors(y[sel], q[sel])
    return {"factors": factors, "n": ns, "fallback": fb, "n_window": int(len(window))}


def rolling_row_factors(models, edges, rows, val, window_months=ROLL_WINDOW_MONTHS, min_n=ROLL_MIN_N, step_minutes=ROLL_STEP_MINUTES):
    """For each calendar month with rows in `val`, refit on `rolling_window(rows, month_start)` and give every val row of that month the
    factors of its NATR decile. Returns (row_factors (len(val), 3), table month / decile / n / fallback / c10 / c90). `rows` are all realized
    rows (build_rows output); rows of the month itself are never in its window. Pass the result as {"range": {"row_factors": ...}}."""
    t = pd.DatetimeIndex(val["tgt_open"])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    month = t.tz_convert(None).to_period("M")
    dec = assign_decile(val["natr"], edges)
    out = np.ones((len(val), 3))
    recs = []
    for per in sorted(set(month)):
        start = per.start_time.tz_localize("UTC")
        f = fit_month_factors(models, edges, rolling_window(rows, start, window_months, step_minutes), start, min_n, step_minutes)
        sel = np.asarray(month == per)
        out[sel] = f["factors"][dec[sel]]
        for d in range(10):
            recs.append({"month": str(per), "decile": d, "n": int(f["n"][d]), "fallback": bool(f["fallback"][d]),
                         "c10": float(f["factors"][d, 0]), "c90": float(f["factors"][d, 2])})
    return out, pd.DataFrame(recs)


def factor_stability(table):
    """Rule 9 inputs from the `rolling_row_factors` table: |ln(c_m / c_{m-1})| of c10 and c90 per decile between consecutive refit months."""
    ch = []
    months = sorted(table["month"].unique())
    for a, b in zip(months[:-1], months[1:]):
        ta, tb = table[table["month"] == a].set_index("decile"), table[table["month"] == b].set_index("decile")
        for c in ("c10", "c90"):
            ch.extend(np.abs(np.log(tb[c].to_numpy(dtype=float) / ta[c].to_numpy(dtype=float))).tolist())
    ch = np.asarray(ch, dtype=float)
    return {"n_changes": int(len(ch)), "max_abs_ln": float(ch.max()) if len(ch) else float("nan"),
            "mean_abs_ln": float(ch.mean()) if len(ch) else float("nan")}


def quarter_top_coverage(models, edges, val, corrections=None):
    """DataFrame quarter x (n, coverage) of the top NATR decile's [p10, p90] coverage by calendar quarter of the target open."""
    q, _ = predict_quantiles(models, "range", val["natr"], corrections)
    y = val["y_range"].to_numpy(dtype=float)
    cov = (y >= q[:, 0]) & (y <= q[:, 2])
    top = assign_decile(val["natr"], edges) == TOP_DECILE
    t = pd.DatetimeIndex(val["tgt_open"])
    quarter = (t.tz_convert(None) if t.tz is not None else t).to_period("Q")
    rows = []
    for qq in sorted(set(quarter)):
        m = top & np.asarray(quarter == qq)
        rows.append({"quarter": str(qq), "n": int(m.sum()), "coverage": float(cov[m].mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)

"""
PURPOSE:  Reverse-engineer how a provider built its technical indicators, and test that reverse.
          (1) identity_checks(row): exact algebraic identities a correct indicator row must satisfy
              (KDJ J = 3K - 2D, MACD histogram sign/definition, Bollinger symmetry, oscillator ranges,
              all-zero rows read as missing data).
          (2) fit_indicator(...): take a provider's output series, try every parameterised candidate
              definition (pandas-ta-classic as the reference implementation) and report which one
              reproduces the output and with what relative error.
          (3) insufficient_history(...): a moving average longer than the available history must be null, not 0.
TAGS:     indicator, reverse engineering, rsi, macd, bollinger, kdj, williams, cci, adx, moving average,
          gate-cli, data quality, pandas-ta-classic
PITFALLS: Provider kline rows come NEWEST FIRST; call to_ascending() before any fit. Zero values are treated
          as missing (the provider uses 0 as a fill value), so a true 0 is not fit. The reference is
          pandas-ta-classic, pinned in requirements-ci.txt (0.8.32); a fit is relative to that version and
          to the candidate grid, so "no candidate matches" is a result, not proof of a bug in the provider.
"""
import math

import pandas as pd
import pandas_ta_classic as ta

RSI_LENGTHS = (6, 9, 12, 14, 21, 24)
MA_LENGTHS = (7, 30, 120, 200)
OSC_LENGTHS = (9, 14, 20)
MACD_PARAMS = ((12, 26, 9), (6, 13, 5), (5, 35, 5))
BB_LENGTHS = (20,)

OHLCV_COLUMNS = ("open", "high", "low", "close")


def to_ascending(rows, time_key="time"):
    """Provider rows (dicts with OHLC keys) -> DataFrame sorted oldest first.

    Accepts the gate-style keys (open_price, high_price, low_price, close_price) or plain open/high/low/close.
    """
    frame = pd.DataFrame(rows)
    renames = {"open_price": "open", "high_price": "high", "low_price": "low", "close_price": "close"}
    frame = frame.rename(columns=renames)
    if time_key in frame.columns:
        frame = frame.sort_values(time_key, kind="stable")
    return frame.reset_index(drop=True)


def _col(frame, prefix):
    """First column whose name starts with `prefix` (pandas-ta-classic names columns like 'BBM_20_2.0')."""
    for name in frame.columns:
        if str(name).startswith(prefix):
            return frame[name]
    raise KeyError(f"no column starting with {prefix!r} in {list(frame.columns)}")


def candidates_for(ohlcv, family):
    """Parameterised candidate definitions for one indicator family.

    Returns {candidate_name: pd.Series}. `ohlcv` is oldest-first with open/high/low/close columns.
    """
    o, h, l, c = (ohlcv[k] for k in OHLCV_COLUMNS)
    out = {}
    if family == "rsi":
        for n in RSI_LENGTHS:
            out[f"rsi_{n}"] = ta.rsi(c, length=n)
    elif family == "sma":
        for n in MA_LENGTHS:
            out[f"sma_{n}"] = ta.sma(c, length=n)
    elif family == "ema":
        for n in MA_LENGTHS:
            out[f"ema_{n}"] = ta.ema(c, length=n)
    elif family == "willr":
        for n in OSC_LENGTHS:
            out[f"willr_{n}"] = ta.willr(h, l, c, length=n)
    elif family == "cci":
        for n in OSC_LENGTHS:
            out[f"cci_{n}"] = ta.cci(h, l, c, length=n)
    elif family == "adx":
        for n in OSC_LENGTHS:
            out[f"adx_{n}"] = _col(ta.adx(h, l, c, length=n), "ADX")
    elif family == "bbands_mid":
        for n in BB_LENGTHS:
            out[f"bb_mid_{n}"] = _col(ta.bbands(c, length=n, std=2), "BBM")
    else:
        raise ValueError(f"unknown indicator family {family!r}")
    return out


def macd_candidates(ohlcv):
    """{name: (macd_line, signal_line)} for each (fast, slow, signal) in MACD_PARAMS."""
    c = ohlcv["close"]
    out = {}
    for fast, slow, sig in MACD_PARAMS:
        frame = ta.macd(c, fast=fast, slow=slow, signal=sig)
        out[f"macd_{fast}_{slow}_{sig}"] = (_col(frame, "MACD_"), _col(frame, "MACDs_"))
    return out


def fit_indicator(observed, candidates, last_n=60, tol=1e-6):
    """Match one provider series against candidate definitions.

    observed: pd.Series aligned with the candidates (same index, oldest first).
    Only the last `last_n` rows where observed is finite and non-zero are compared.
    Returns dict: best, rel_err, n_compared, matched (rel_err <= tol), ranking (all candidates, best first).
    """
    obs = pd.to_numeric(pd.Series(observed).reset_index(drop=True), errors="coerce")
    usable = obs.notna() & (obs != 0)
    ranking = []
    for name, cand in candidates.items():
        cand = pd.to_numeric(pd.Series(cand).reset_index(drop=True), errors="coerce")
        mask = usable & cand.notna()
        idx = mask[mask].index[-last_n:]
        if len(idx) == 0:
            ranking.append((name, math.inf, 0))
            continue
        o = obs.loc[idx]
        k = cand.loc[idx]
        rel = ((o - k).abs() / o.abs().clip(lower=1.0)).max()
        ranking.append((name, float(rel), int(len(idx))))
    ranking.sort(key=lambda r: r[1])
    best, rel, n = ranking[0]
    return {"best": best, "rel_err": rel, "n_compared": n, "matched": rel <= tol, "ranking": ranking}


def fit_macd(observed_macd, observed_signal, ohlcv, last_n=60, tol=1e-6):
    """Same as fit_indicator for MACD, scoring the line and the signal together."""
    scored = []
    for name, (line, sig) in macd_candidates(ohlcv).items():
        a = fit_indicator(observed_macd, {name: line}, last_n, tol)["ranking"][0][1]
        b = fit_indicator(observed_signal, {name: sig}, last_n, tol)["ranking"][0][1]
        scored.append((name, max(a, b)))
    scored.sort(key=lambda r: r[1])
    best, rel = scored[0]
    return {"best": best, "rel_err": rel, "matched": rel <= tol, "ranking": scored}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def identity_checks(row, tol=1e-4):
    """Algebraic identities a single provider indicator row must satisfy. Returns a list of findings (empty = ok).

    Keys are the gate-style names. Missing keys are skipped (the caller decides whether that matters).
    """
    findings = []
    get = {k: _num(v) for k, v in row.items()}

    osc = [get.get(k) for k in ("rsi", "wr", "cci", "adx", "signal_value_k", "signal_value_d", "signal_value_j")]
    if all(v == 0 for v in osc if v is not None) and any(v is not None for v in osc):
        findings.append("missing: all oscillators are 0 (provider fill value, not a measurement)")

    k, d, j = get.get("signal_value_k"), get.get("signal_value_d"), get.get("signal_value_j")
    if None not in (k, d, j):
        if abs(j - (3 * k - 2 * d)) > tol * max(1.0, abs(j)):
            findings.append(f"kdj: j={j} != 3k-2d={3 * k - 2 * d}")

    m, dea, diff = get.get("macd"), get.get("macd_dea"), get.get("macd_difference")
    if None not in (m, dea, diff) and (m or dea or diff):
        std, summed = m - dea, m + dea
        if abs(diff - std) <= tol * max(1.0, abs(diff)):
            findings.append("macd: macd_difference = macd - dea (standard histogram)")
        elif abs(diff - summed) <= tol * max(1.0, abs(diff)):
            findings.append("macd: macd_difference = macd + dea (NOT the histogram; check the definition)")
        else:
            findings.append(f"macd: macd_difference={diff} matches neither macd-dea={std} nor macd+dea={summed}")

    up, mid, lo = get.get("boll_upper_band"), get.get("boll_middle_band"), get.get("boll_lower_band")
    if None not in (up, mid, lo) and (up or mid or lo):
        if abs((up - mid) - (mid - lo)) > tol * max(1.0, abs(mid)):
            findings.append("bollinger: bands are not symmetric around the middle")

    for key, lo_b, hi_b in (("rsi", 0, 100), ("wr", -100, 0), ("adx", 0, 100), ("di_plus", 0, 100), ("di_minus", 0, 100)):
        v = get.get(key)
        if v is not None and not (lo_b - tol <= v <= hi_b + tol):
            findings.append(f"range: {key}={v} outside [{lo_b}, {hi_b}]")
    return findings


def insufficient_history(observed_value, n_candles, length):
    """True when a moving average of `length` cannot exist with `n_candles` bars, but the provider returned a value.

    A null is the correct answer in that case; 0 is a fill value and must be flagged.
    """
    if n_candles >= length:
        return False
    return observed_value is not None and _num(observed_value) is not None

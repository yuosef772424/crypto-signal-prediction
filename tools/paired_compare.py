"""
PURPOSE:  Paired comparison of two model arms PER TIMESTAMP (card T2): one loss per (timestamp, arm) averaged over the coins
          of that timestamp, the paired differential d_t = loss_a - loss_b on the timestamps both arms have, a block bootstrap
          over WHOLE timestamps (Politis-Romano stationary and moving block), a Newey-West (Bartlett HAC) mean test, and the
          temporal design effect deff = 1 + 2 sum_k (1 - k/(L+1)) rho_k with n_eff = T / deff, per target and for d_t.
          `compare_arms` returns one summary table (data); `format_report` renders it (separate thin function).
TAGS:     paired comparison, loss differential, Newey-West, HAC, block bootstrap, stationary bootstrap, moving block bootstrap,
          design effect, effective sample size, n_eff, per-timestamp loss, autocorrelation, T2, compare models
PITFALLS: The unit is the TIMESTAMP: coin-time rows are never resampled or counted as independent (coins are averaged inside a
          timestamp first). d_t = loss_a - loss_b, so NEGATIVE mean = arm A is better. "Contiguous block" means adjacent in the
          sorted aligned timestamp list (after dropping timestamps one arm lacks), not adjacent in wall-clock time: with a
          stride > 1 the blocks are blocks of samples. Non-finite rows are dropped JOINTLY over all arms of a target (same coin
          set for both arms) and counted. The bootstrap p-value is two-sided on the centered resampled means, (1 + #{|m*-m|>=|m|})
          / (B + 1). NW p-values use the normal approximation (no t table, no scipy). Block-length / lag rules ("auto_nw",
          "nw_default", "auto_andrews") are heuristics for a mean; with strong persistence they under-cover, so state the rule
          next to the result. deff is clamped (see temporal_design_effect), the raw value is always returned too. Defaults
          do not exist: lags, block_len, n_boot, seed, kind, loss, min_overlap are all explicit; unknown names raise.
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # stats/ sits next to tools/ (this module is imported with only tools/ on the path)
from stats.metrics import pearson  # noqa: E402

LOSSES = ("abs", "sq", "precomputed")
BOOT_KINDS = ("stationary", "moving")
LAG_RULES = ("nw_default", "andrews")          # accepted by `lags=` (or an int >= 0)
BLOCK_RULES = ("auto_nw", "auto_andrews")      # accepted by `block_len=` (or an int >= 1)
CLAMPS = ("conservative", "none")
SERIES = ("mean", "ic")
_CHUNK_CELLS = 4_000_000                       # bootstrap index matrix is built in chunks of at most this many cells


# ── result containers ─────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, eq=False)
class PerTimestampLosses:
    table: pd.DataFrame          # index = sorted unique timestamps, columns = arms, value = coin-mean loss
    n_rows: int                  # input rows
    n_rows_nonfinite: int        # rows dropped because y or any arm value/loss was non-finite
    n_timestamps: int


@dataclass(frozen=True, eq=False)
class PairedDifferential:
    d: pd.Series                 # d_t = loss_a - loss_b on the common timestamps (sorted index)
    n_a: int                     # timestamps arm A has (finite)
    n_b: int
    n_overlap: int
    n_dropped_a: int             # in A but not in B
    n_dropped_b: int             # in B but not in A


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_lo: float
    ci_hi: float
    p_value: float               # two-sided, H0: mean(d) = 0
    kind: str
    block_len: int
    n_boot: int
    seed: int
    ci_level: float
    n: int


@dataclass(frozen=True)
class NeweyWestResult:
    mean: float
    se: float
    t: float
    p_value: float               # two-sided, normal approximation
    lags: int
    n: int


@dataclass(frozen=True)
class DesignEffect:
    deff_raw: float              # 1 + 2 sum_k w_k rho_k, unclamped
    deff: float                  # after the clamp
    n_eff: float                 # T / deff
    n: int
    lags: int
    clamp: str
    clamped: bool
    rho1: float


@dataclass(frozen=True, eq=False)
class CompareResult:
    table: pd.DataFrame          # one row per target
    params: dict


# ── small helpers ─────────────────────────────────────────────────────────────────────────────────────────────────
def _norm_two_sided_p(z):
    return float("nan") if not np.isfinite(z) else math.erfc(abs(z) / math.sqrt(2.0))


def _as_1d(x, name):
    a = np.asarray(x, dtype="float64")
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if len(a) < 2:
        raise ValueError(f"{name} needs at least 2 observations, got {len(a)}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (drop them first; paired_differential does it and counts them)")
    return a


def _pick(value, allowed, name):
    if value not in allowed:
        raise ValueError(f"unknown {name} {value!r}; allowed: {allowed}")
    return value


def _autocov(x, max_lag):
    """gamma_k = (1/T) sum_{t>k} (x_t - m)(x_{t-k} - m), k = 0..max_lag (biased, 1/T: the Bartlett sum stays non-negative)."""
    xc = x - x.mean()
    T = len(x)
    return np.array([float(np.dot(xc[k:], xc[:T - k])) / T for k in range(max_lag + 1)])


def nw_default_lag(T):
    """Newey-West (1994) rule of thumb, floor(4 (T/100)^(2/9))."""
    if int(T) < 2:
        raise ValueError(f"T must be >= 2, got {T}")
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def andrews_lag(x):
    """Andrews (1991) AR(1) plug-in bandwidth for the Bartlett kernel: floor(1.1447 (alpha T)^(1/3)),
    alpha = 4 r^2 / (1 - r^2)^2 with r the lag-1 autocorrelation (clipped to [0, 0.97])."""
    x = _as_1d(x, "x")
    g = _autocov(x, 1)
    if g[0] <= 0:
        raise ValueError("x has zero variance")
    r = float(np.clip(g[1] / g[0], 0.0, 0.97))
    alpha = 4.0 * r * r / (1.0 - r * r) ** 2
    return int(min(math.floor(1.1447 * (alpha * len(x)) ** (1.0 / 3.0)), len(x) - 1))


def resolve_lags(x, lags):
    """lags: int >= 0, or a named rule in LAG_RULES. Never silent: anything else raises."""
    x = _as_1d(x, "x")
    if isinstance(lags, (int, np.integer)) and not isinstance(lags, bool):
        if lags < 0 or lags >= len(x):
            raise ValueError(f"lags must be in [0, T-1={len(x) - 1}], got {lags}")
        return int(lags)
    _pick(lags, LAG_RULES, "lags rule")
    return min(nw_default_lag(len(x)), len(x) - 1) if lags == "nw_default" else andrews_lag(x)


def resolve_block_len(x, block_len):
    """block_len: int >= 1, or a named rule in BLOCK_RULES ("auto_nw": max(1, NW default lag); "auto_andrews": max(1, Andrews lag))."""
    x = _as_1d(x, "x")
    if isinstance(block_len, (int, np.integer)) and not isinstance(block_len, bool):
        if block_len < 1 or block_len > len(x):
            raise ValueError(f"block_len must be in [1, T={len(x)}], got {block_len}")
        return int(block_len)
    _pick(block_len, BLOCK_RULES, "block_len rule")
    b = nw_default_lag(len(x)) if block_len == "auto_nw" else andrews_lag(x)
    return int(min(max(1, b), len(x)))


# ── 1. per-timestamp loss ─────────────────────────────────────────────────────────────────────────────────────────
def per_timestamp_loss(df, *, ts_col, y_col=None, pred_cols=None, loss_cols=None, loss):
    """One loss per (timestamp, arm), the mean over the coins of that timestamp.
    Either `pred_cols` (dict arm -> prediction column, with `y_col`, loss in {"abs","sq"}) or `loss_cols`
    (dict arm -> per-row loss column, loss="precomputed"). Exactly one of the two. Rows where y or any arm is non-finite are
    dropped for ALL arms (same coin set per timestamp) and counted."""
    _pick(loss, LOSSES, "loss")
    if (pred_cols is None) == (loss_cols is None):
        raise ValueError("pass exactly one of pred_cols / loss_cols")
    if pred_cols is not None:
        if loss == "precomputed":
            raise ValueError("loss='precomputed' needs loss_cols, not pred_cols")
        if y_col is None:
            raise ValueError("pred_cols needs y_col")
        arms = dict(pred_cols)
    else:
        if loss != "precomputed":
            raise ValueError("loss_cols needs loss='precomputed'")
        arms = dict(loss_cols)
    if not arms:
        raise ValueError("no arms given")
    need = [ts_col] + ([y_col] if pred_cols is not None else []) + list(arms.values())
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in df: {missing}")
    sub = df[list(dict.fromkeys(need))]                     # unique columns (two arms may point at the same column)
    vals = np.column_stack([df[c].to_numpy(dtype="float64") for c in arms.values()])
    ok = np.isfinite(vals).all(axis=1)
    if pred_cols is not None:
        y = sub[y_col].to_numpy(dtype="float64")
        ok &= np.isfinite(y)
        err = vals - y[:, None]
        vals = np.abs(err) if loss == "abs" else err * err
    if sub[ts_col].isna().any():
        raise ValueError(f"{ts_col} contains missing timestamps")
    g = pd.DataFrame(vals[ok], columns=list(arms.keys()))
    g[ts_col] = sub[ts_col].to_numpy()[ok]
    table = g.groupby(ts_col, sort=True).mean()
    table.index.name = ts_col
    return PerTimestampLosses(table=table, n_rows=int(len(df)), n_rows_nonfinite=int((~ok).sum()), n_timestamps=int(len(table)))


# ── 2. paired differential ────────────────────────────────────────────────────────────────────────────────────────
def paired_differential(loss_a, loss_b, *, min_overlap):
    """d_t = loss_a - loss_b on the timestamps both series have (pd.Series indexed by timestamp). Non-finite values count as
    missing. Reports how many timestamps were dropped from each side; raises if overlap < min_overlap."""
    if not isinstance(min_overlap, (int, np.integer)) or isinstance(min_overlap, bool) or min_overlap < 2:
        raise ValueError(f"min_overlap must be an int >= 2, got {min_overlap!r}")
    for nm, s in (("loss_a", loss_a), ("loss_b", loss_b)):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{nm} must be a pd.Series indexed by timestamp")
        if s.index.has_duplicates:
            raise ValueError(f"{nm} has duplicate timestamps (aggregate per timestamp first: per_timestamp_loss)")
    a = loss_a[np.isfinite(loss_a.to_numpy(dtype="float64"))]
    b = loss_b[np.isfinite(loss_b.to_numpy(dtype="float64"))]
    common = a.index.intersection(b.index).sort_values()
    if len(common) < min_overlap:
        raise ValueError(f"timestamp overlap {len(common)} < min_overlap {min_overlap} (A has {len(a)}, B has {len(b)})")
    d = (a.loc[common] - b.loc[common]).astype("float64")
    d.name = "d"
    return PairedDifferential(d=d, n_a=int(len(a)), n_b=int(len(b)), n_overlap=int(len(common)),
                              n_dropped_a=int(len(a) - len(common)), n_dropped_b=int(len(b) - len(common)))


# ── 3. block bootstrap ────────────────────────────────────────────────────────────────────────────────────────────
def _boot_means(x, b, n_boot, kind, rng):
    T = len(x)
    out = np.empty(n_boot)
    step = max(1, _CHUNK_CELLS // T)
    pos = np.arange(T)
    for i0 in range(0, n_boot, step):
        m = min(step, n_boot - i0)
        if kind == "stationary":                           # Politis-Romano 1994, circular, mean block length b
            starts = rng.integers(0, T, size=(m, T))
            new = rng.random((m, T)) < 1.0 / b
            new[:, 0] = True
            last = np.maximum.accumulate(np.where(new, pos[None, :], 0), axis=1)
            base = np.take_along_axis(starts, last, axis=1)
            idx = (base + pos[None, :] - last) % T
        else:                                              # moving block (Kunsch 1989 / Liu-Singh 1992), non-circular, length b
            nb = -(-T // b)
            starts = rng.integers(0, T - b + 1, size=(m, nb))
            idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(m, -1)[:, :T]
        out[i0:i0 + m] = x[idx].mean(axis=1)
    return out


def block_bootstrap_mean(d, *, block_len, n_boot, seed, kind, ci_level=0.95):
    """Bootstrap of mean(d) resampling WHOLE timestamps in contiguous blocks (d is one value per timestamp, already ordered in
    time). kind: "stationary" (random block lengths, mean `block_len`) or "moving". block_len: int or "auto_nw"/"auto_andrews".
    CI: percentile of the resampled means. p: two-sided, H0 mean=0, from the centered resampled means."""
    x = _as_1d(d, "d")
    _pick(kind, BOOT_KINDS, "kind")
    if not isinstance(n_boot, (int, np.integer)) or isinstance(n_boot, bool) or n_boot < 1:
        raise ValueError(f"n_boot must be an int >= 1, got {n_boot!r}")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError(f"seed must be an int, got {seed!r}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    b = resolve_block_len(x, block_len)
    rng = np.random.default_rng(int(seed))
    bm = _boot_means(x, b, int(n_boot), kind, rng)
    m = float(x.mean())
    lo, hi = np.percentile(bm, [100 * (1 - ci_level) / 2, 100 * (1 + ci_level) / 2])
    p = (1 + int((np.abs(bm - m) >= abs(m) - 1e-15).sum())) / (int(n_boot) + 1)
    return BootstrapResult(mean=m, ci_lo=float(lo), ci_hi=float(hi), p_value=float(p), kind=kind, block_len=b,
                           n_boot=int(n_boot), seed=int(seed), ci_level=float(ci_level), n=len(x))


# ── 4. Newey-West ─────────────────────────────────────────────────────────────────────────────────────────────────
def _long_run_var(x, L):
    g = _autocov(x, L)
    w = 1.0 - np.arange(1, L + 1) / (L + 1.0)
    return float(g[0] + 2.0 * np.dot(w, g[1:]))


def newey_west_mean(d, *, lags):
    """Mean of d with the Newey-West (Bartlett-kernel HAC) standard error, t and two-sided normal p-value (H0 mean = 0).
    lags: int >= 0 or "nw_default" (floor(4 (T/100)^(2/9))) / "andrews". se = sqrt(lrv / T), lrv = g0 + 2 sum_k (1-k/(L+1)) g_k.
    A constant series (se = 0) returns t = p = NaN."""
    x = _as_1d(d, "d")
    L = resolve_lags(x, lags)
    lrv = _long_run_var(x, L)
    m = float(x.mean())
    se = math.sqrt(max(lrv, 0.0) / len(x))
    t = m / se if se > 0 else float("nan")
    return NeweyWestResult(mean=m, se=se, t=t, p_value=_norm_two_sided_p(t), lags=L, n=len(x))


# ── 5. temporal design effect ─────────────────────────────────────────────────────────────────────────────────────
def temporal_design_effect(x, *, lags, clamp):
    """Kish-style design effect of a per-timestamp series (loss_t, d_t, IC_t...): deff = 1 + 2 sum_{k=1..L} (1 - k/(L+1)) rho_k,
    n_eff = T / deff. clamp="conservative": deff is clipped to [1, T] (n_eff in [1, T]: negative autocorrelation never inflates
    the sample); clamp="none": only guarded to deff >= 1/T so that n_eff stays finite (n_eff <= T^2). `deff_raw` is always
    returned and `clamped` says whether the clamp changed it. A zero-variance series raises."""
    xs = _as_1d(x, "x")
    _pick(clamp, CLAMPS, "clamp")
    L = resolve_lags(xs, lags)
    g = _autocov(xs, L)
    if g[0] <= 0:
        raise ValueError("series has zero variance: design effect undefined")
    rho = g[1:] / g[0]
    w = 1.0 - np.arange(1, L + 1) / (L + 1.0)
    raw = float(1.0 + 2.0 * np.dot(w, rho)) if L > 0 else 1.0
    T = len(xs)
    lo, hi = (1.0, float(T)) if clamp == "conservative" else (1.0 / T, float("inf"))
    deff = float(min(max(raw, lo), hi))
    return DesignEffect(deff_raw=raw, deff=deff, n_eff=float(T / deff), n=T, lags=L, clamp=clamp, clamped=bool(deff != raw),
                        rho1=float(rho[0]) if L > 0 else 0.0)


def _ic_series(df, ts_col, y_col, pred_col, min_assets):
    sub = df[[ts_col, y_col, pred_col]].copy()
    sub = sub[np.isfinite(sub[y_col].to_numpy(dtype="float64")) & np.isfinite(sub[pred_col].to_numpy(dtype="float64"))]
    sub["_ry"] = sub.groupby(ts_col)[y_col].rank()
    sub["_rp"] = sub.groupby(ts_col)[pred_col].rank()
    out = {}
    for t, g in sub.groupby(ts_col, sort=True):
        if len(g) < min_assets:
            continue
        a, b = g["_ry"].to_numpy(), g["_rp"].to_numpy()
        r = pearson(a, b, nan_policy="propagate", min_n=1)   # Pearson of the in-timestamp ranks; a constant side -> NaN -> skipped
        if np.isnan(r):
            continue
        out[t] = r
    return pd.Series(out, dtype="float64")


def design_effect_by_target(df, targets, *, ts_col, lags, clamp, series, min_assets=None):
    """Small table (one row per target) of the temporal design effect of a per-timestamp series.
    series="mean": targets = {name: column}; the series is that column averaged over the coins of each timestamp (use it on a loss
    column or any per-row statistic). series="ic": targets = {name: (y_col, pred_col)}; the series is the per-timestamp
    cross-sectional Spearman IC (timestamps with < min_assets finite coins are skipped; min_assets required then)."""
    _pick(series, SERIES, "series")
    if series == "ic" and (not isinstance(min_assets, (int, np.integer)) or min_assets < 3):
        raise ValueError("series='ic' needs min_assets (int >= 3)")
    rows = {}
    for name, spec in targets.items():
        if series == "mean":
            s = df[[ts_col, spec]].replace([np.inf, -np.inf], np.nan).dropna().groupby(ts_col, sort=True)[spec].mean()
        else:
            s = _ic_series(df, ts_col, spec[0], spec[1], int(min_assets))
        de = temporal_design_effect(s.to_numpy(), lags=lags, clamp=clamp)
        rows[name] = {"n_timestamps": de.n, "lags": de.lags, "rho1": de.rho1, "deff_raw": de.deff_raw, "deff": de.deff,
                      "n_eff": de.n_eff, "clamped": de.clamped}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("target")


# ── 6. compare_arms / report ──────────────────────────────────────────────────────────────────────────────────────
def compare_arms(df, targets, *, ts_col, loss, lags, block_len, n_boot, seed, kind, min_overlap, clamp, ci_level=0.95):
    """One summary table, one row per target, for arm A vs arm B (d_t = loss_A - loss_B, negative = A better).
    targets: {name: (y_col, pred_col_a, pred_col_b)} with loss in {"abs","sq"}, or {name: (loss_col_a, loss_col_b)} with
    loss="precomputed". Per target: n_timestamps (overlap) and the timestamps dropped per side, mean loss of A and B, mean d,
    Newey-West se/t/p (NaN t/p if d_t is constant), bootstrap CI/p (same seed for every target, so the resampled timestamps match across targets), the rules'
    resolved lags / block_len, and deff / n_eff of d_t (NaN if d_t is constant). Returns CompareResult (data); see format_report."""
    _pick(loss, LOSSES, "loss")
    rows = {}
    for name, spec in targets.items():
        if loss == "precomputed":
            if len(spec) != 2:
                raise ValueError(f"target {name!r}: loss='precomputed' expects (loss_col_a, loss_col_b)")
            ptl = per_timestamp_loss(df, ts_col=ts_col, loss_cols={"a": spec[0], "b": spec[1]}, loss=loss)
        else:
            if len(spec) != 3:
                raise ValueError(f"target {name!r}: expects (y_col, pred_col_a, pred_col_b)")
            ptl = per_timestamp_loss(df, ts_col=ts_col, y_col=spec[0], pred_cols={"a": spec[1], "b": spec[2]}, loss=loss)
        # per_timestamp_loss drops rows jointly over both arms, so both arms hold the same timestamps; the general
        # misalignment case (arms scored on different timestamps) goes through paired_differential directly.
        pd_ = paired_differential(ptl.table["a"], ptl.table["b"], min_overlap=min_overlap)
        d = pd_.d.to_numpy()
        nw = newey_west_mean(d, lags=lags)
        bs = block_bootstrap_mean(d, block_len=block_len, n_boot=n_boot, seed=seed, kind=kind, ci_level=ci_level)
        if np.ptp(d) == 0:                                  # identical arms: d_t is constant, deff undefined -> NaN, not a crash
            de = DesignEffect(float("nan"), float("nan"), float("nan"), len(d), nw.lags, clamp, False, float("nan"))
        else:
            de = temporal_design_effect(d, lags=lags, clamp=clamp)
        rows[name] = {"n_timestamps": pd_.n_overlap, "n_rows_nonfinite": ptl.n_rows_nonfinite,
                      "n_dropped_a": pd_.n_dropped_a, "n_dropped_b": pd_.n_dropped_b,
                      "mean_a": float(ptl.table["a"].mean()), "mean_b": float(ptl.table["b"].mean()), "mean_d": nw.mean,
                      "nw_se": nw.se, "nw_t": nw.t, "nw_p": nw.p_value, "nw_lags": nw.lags,
                      "boot_lo": bs.ci_lo, "boot_hi": bs.ci_hi, "boot_p": bs.p_value, "boot_block_len": bs.block_len,
                      "deff": de.deff, "deff_raw": de.deff_raw, "n_eff": de.n_eff}
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("target")
    params = {"loss": loss, "lags": lags, "block_len": block_len, "n_boot": int(n_boot), "seed": int(seed), "kind": kind,
              "min_overlap": int(min_overlap), "clamp": clamp, "ci_level": float(ci_level), "sign": "d = loss_a - loss_b; negative = A better"}
    return CompareResult(table=table, params=params)


def format_report(result):
    """Plain-text rendering of a CompareResult (no computation)."""
    p = result.params
    head = (f"paired comparison per timestamp | loss={p['loss']} | {p['kind']} bootstrap B={p['n_boot']} seed={p['seed']} "
            f"block_len={p['block_len']} | NW lags={p['lags']} | clamp={p['clamp']} | {p['sign']}")
    cols = ["n_timestamps", "mean_d", "nw_t", "nw_p", "boot_lo", "boot_hi", "boot_p", "deff", "n_eff"]
    return head + "\n" + result.table[cols].to_string(float_format=lambda v: f"{v:.4g}")


def _load(path):
    return pd.read_parquet(path) if str(path).endswith((".parquet", ".pq")) else pd.read_csv(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Paired model comparison per timestamp (every option explicit).")
    ap.add_argument("data", help="csv or parquet with one row per (timestamp, coin)")
    ap.add_argument("--ts-col", required=True)
    ap.add_argument("--target", action="append", required=True,
                    help="name:y_col:pred_a:pred_b (loss abs/sq) or name:loss_a:loss_b (loss precomputed); repeatable")
    ap.add_argument("--loss", required=True, choices=LOSSES)
    ap.add_argument("--lags", required=True, help=f"int or one of {LAG_RULES}")
    ap.add_argument("--block-len", required=True, help=f"int or one of {BLOCK_RULES}")
    ap.add_argument("--n-boot", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--kind", required=True, choices=BOOT_KINDS)
    ap.add_argument("--min-overlap", required=True, type=int)
    ap.add_argument("--clamp", required=True, choices=CLAMPS)
    a = ap.parse_args(argv)
    targets = {}
    for t in a.target:
        parts = t.split(":")
        targets[parts[0]] = tuple(parts[1:])
    num = lambda s: int(s) if s.lstrip("-").isdigit() else s
    res = compare_arms(_load(a.data), targets, ts_col=a.ts_col, loss=a.loss, lags=num(a.lags), block_len=num(a.block_len),
                       n_boot=a.n_boot, seed=a.seed, kind=a.kind, min_overlap=a.min_overlap, clamp=a.clamp)
    print(format_report(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

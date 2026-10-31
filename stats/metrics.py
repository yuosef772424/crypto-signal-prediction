"""
PURPOSE:  The ONE implementation of each metric: pearson / spearman (rank-IC) / partial_spearman, auc, wilson_interval, pinball_loss / mean_pinball, ece_quantile, each with an explicit tie/NaN policy chosen by a keyword (no hidden default per call site).
TAGS:     metrics, spearman, pearson, rank ic, information coefficient, auc, mann-whitney, wilson interval, pinball loss, ece, expected calibration error, tie policy, nan policy, average rank, single owner
PITFALLS: Ties get AVERAGE ranks everywhere (pandas `rank()` == scipy `rankdata`). NaN policy is a keyword, never implied: correlations take nan_policy in {"drop_nonfinite" (default: drop pairs with NaN or +-inf), "drop" (drop pairs with NaN only; the pandas `Series.corr` behaviour), "propagate" (any NaN -> NaN)}; auc takes {"propagate" (default), "drop", "legacy_rank_skipna" (reproduces model/diagnostics: NaN scores get NaN ranks that the positive-class rank sum then skips or poisons)}. A degenerate input (too few valid pairs, zero variance, non-finite result) returns `degenerate` (default NaN; signal_eval uses 0.0 = "no signal"); `var_tol` > 0 widens "zero variance" from exactly 0 to std < var_tol. Wilson is clipped to [0, 1] by default (`clip=False` reproduces the unclipped selective_eval form). Inputs are plain arrays: grouping (per day / per month / per coin) stays in the caller, which calls these per group. Pure numpy/pandas; stats imports at most `core`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["NAN_POLICIES", "AUC_NAN_POLICIES", "rank_average", "pearson", "spearman", "partial_spearman", "auc",
           "wilson_interval", "pinball_loss", "mean_pinball", "ece_quantile"]

NAN_POLICIES = ("drop_nonfinite", "drop", "propagate")
AUC_NAN_POLICIES = ("propagate", "drop", "legacy_rank_skipna")


def _float_1d(a) -> np.ndarray:
    return np.asarray(a, dtype="float64").reshape(-1)


def rank_average(a) -> np.ndarray:
    """Average ranks (1-based; ties share the mean rank). NaN stays NaN at its position (pandas `rank()` semantics)."""
    return pd.Series(np.asarray(a, dtype="float64")).rank().to_numpy()


def _pair(a, b, nan_policy: str):
    """(a, b) as float64 1-D arrays after the NaN policy, or None when 'propagate' met a NaN."""
    if nan_policy not in NAN_POLICIES:
        raise ValueError(f"nan_policy {nan_policy!r} unknown; use one of {NAN_POLICIES}")
    a, b = _float_1d(a), _float_1d(b)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} != {b.shape}")
    if nan_policy == "propagate":
        return None if (np.isnan(a).any() or np.isnan(b).any()) else (a, b)
    ok = (np.isfinite(a) & np.isfinite(b)) if nan_policy == "drop_nonfinite" else (~np.isnan(a) & ~np.isnan(b))
    return (a, b) if ok.all() else (a[ok], b[ok])


def _constant(x: np.ndarray, var_tol: float) -> bool:
    sd = np.std(x)
    return bool(sd == 0 or sd < var_tol)


def _corr(a, b, rank: bool, nan_policy: str, min_n: int, var_tol: float, degenerate: float) -> float:
    pair = _pair(a, b, nan_policy)
    if pair is None:
        return float("nan")
    a, b = pair
    if len(a) < min_n:
        return float(degenerate)
    with np.errstate(all="ignore"):
        if _constant(a, var_tol) or _constant(b, var_tol):
            return float(degenerate)
        if rank:
            a, b = rank_average(a), rank_average(b)
        r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else float(degenerate)


def pearson(a, b, *, nan_policy: str = "drop_nonfinite", min_n: int = 3, var_tol: float = 0.0,
            degenerate: float = float("nan")) -> float:
    """Pearson correlation of two 1-D arrays. Fewer than `min_n` valid pairs, a constant side (std == 0 or < var_tol)
    or a non-finite result -> `degenerate`. nan_policy 'propagate' returns NaN whenever either side holds a NaN."""
    return _corr(a, b, False, nan_policy, min_n, var_tol, degenerate)


def spearman(a, b, *, nan_policy: str = "drop_nonfinite", min_n: int = 3, var_tol: float = 0.0,
             degenerate: float = float("nan")) -> float:
    """Spearman rank correlation = Pearson of the average ranks (ties share the mean rank) of the pairs left after the NaN
    policy. The zero-variance test runs on the raw values (equivalent to testing the ranks, and what signal_eval did)."""
    return _corr(a, b, True, nan_policy, min_n, var_tol, degenerate)


def partial_spearman(a, b, control) -> float:
    """Spearman of `a` and `b` after removing (least squares, with intercept) the average RANK of `control` from the ranks of
    both. No NaN handling: the caller passes finite arrays (a NaN rank propagates like the old per-day code)."""
    ra, rb, rc = rank_average(a), rank_average(b), rank_average(control)
    z = np.c_[np.ones(len(ra)), rc]
    res = [x - z @ np.linalg.lstsq(z, x, rcond=None)[0] for x in (ra, rb)]
    return float(np.corrcoef(res[0], res[1])[0, 1])


def auc(y, score, *, nan_policy: str = "propagate") -> float:
    """Area under the ROC curve = Mann-Whitney U / (n1 n0) with AVERAGE ranks for tied scores. `y` is cast to bool (truthy =
    positive). One class only -> NaN. nan_policy: 'propagate' (a NaN score -> NaN), 'drop' (drop the NaN-score pairs first),
    'legacy_rank_skipna' (model/diagnostics behaviour: class sizes count every row, NaN scores are skipped by the ranking,
    so the result is NaN only when a positive row is NaN)."""
    if nan_policy not in AUC_NAN_POLICIES:
        raise ValueError(f"nan_policy {nan_policy!r} unknown; use one of {AUC_NAN_POLICIES}")
    y = np.asarray(y).astype(bool).reshape(-1)
    s = _float_1d(score)
    if y.shape != s.shape:
        raise ValueError(f"shapes differ: {y.shape} != {s.shape}")
    if nan_policy == "drop":
        ok = ~np.isnan(s)
        y, s = y[ok], s[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    if nan_policy == "propagate" and np.isnan(s).any():
        return float("nan")
    return float((rank_average(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wilson_interval(k, n, z: float = 1.96, *, clip: bool = True) -> tuple[float, float]:
    """Wilson score interval (default 95%) of the proportion k/n -> (low, high). n <= 0 -> (NaN, NaN). Valid for small n and
    proportions near 0/1 (unlike Wald). `clip=True` bounds it to [0, 1]; `clip=False` returns the raw formula."""
    n = float(n)
    if n <= 0:
        return float("nan"), float("nan")
    p = float(k) / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    if clip:
        return float(max(0.0, centre - half)), float(min(1.0, centre + half))
    return float(centre - half), float(centre + half)


def pinball_loss(y, q, tau) -> np.ndarray:
    """Element-wise pinball (quantile) loss max(tau d, (tau - 1) d), d = y - q. NaN propagates (the caller filters)."""
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.maximum(tau * d, (tau - 1.0) * d)


def mean_pinball(y, q, taus) -> float:
    """Mean over the quantile levels `taus` of the mean pinball loss of column i of the (n, len(taus)) matrix `q`."""
    return float(np.mean([pinball_loss(y, q[:, i], t).mean() for i, t in enumerate(taus)]))


def ece_quantile(p, y, n_bins: int = 10) -> float:
    """Expected calibration error with QUANTILE bins: sum over bins of weight * |mean p - mean y|. Pairs with a non-finite p or
    y are dropped; nothing left -> NaN. Bins = pd.qcut into min(n_bins, #unique p) equal-count bins (duplicate edges dropped);
    when qcut cannot split (e.g. one unique p) equal-width pd.cut(n_bins) is used instead."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return float("nan")
    try:
        bins = pd.qcut(p, q=min(n_bins, len(np.unique(p))), duplicates="drop")
    except Exception:
        bins = pd.cut(p, bins=n_bins)
    g = pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True)
    return float(sum(len(x) / len(p) * abs(x["p"].mean() - x["y"].mean()) for _, x in g))

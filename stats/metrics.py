"""
PURPOSE:  The ONE implementation of each metric: pearson / spearman (rank-IC) / partial_spearman, auc, wilson_interval, pinball_loss / mean_pinball, ece_quantile, each with an explicit tie/NaN policy chosen by a keyword (no hidden default per call site).
TAGS:     metrics, evaluate_heads, per-head policy, qss, quantile skill score, quantile coverage, interval coverage, winkler, profit factor, net sharpe, turnover cost, threshold trades, spearman, pearson, rank ic, information coefficient, auc, mann-whitney, wilson interval, pinball loss, ece, expected calibration error, tie policy, nan policy, average rank, single owner
PITFALLS: Ties get AVERAGE ranks everywhere (pandas `rank()` == scipy `rankdata`). NaN policy is a keyword, never implied: correlations take nan_policy in {"drop_nonfinite" (default: drop pairs with NaN or +-inf), "drop" (drop pairs with NaN only; the pandas `Series.corr` behaviour), "propagate" (any NaN -> NaN)}; auc takes {"propagate" (default), "drop", "legacy_rank_skipna" (reproduces model/diagnostics: NaN scores get NaN ranks that the positive-class rank sum then skips or poisons)}. A degenerate input (too few valid pairs, zero variance, non-finite result) returns `degenerate` (default NaN; signal_eval uses 0.0 = "no signal"); `var_tol` > 0 widens "zero variance" from exactly 0 to std < var_tol. Wilson is clipped to [0, 1] by default (`clip=False` reproduces the unclipped selective_eval form). Inputs are plain arrays: grouping (per day / per month / per coin) stays in the caller, which calls these per group. Pure numpy/pandas; stats imports at most `core`. Quantile / interval / trading metrics (qss, quantile_coverage, interval_coverage, winkler_score, threshold_trades, profit_factor, net_sharpe) drop non-finite rows and return NaN when nothing is left; `evaluate_heads` applies the owner's per-head policy (2026-10-31): direction metrics ONLY for core.schema.DIRECTION_TARGET (close), every regression head gets rank-IC + MAE/RMSE + skill vs the zero (no-change) prediction, intervals always report coverage TOGETHER with mean width (coverage alone is gamed by wide intervals). Trading helpers take a threshold, they never optimise one. Head roles come from core.heads (HeadSpec.role / ROLES), never from name literals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.heads import ROLES
from core.schema import DIRECTION_TARGET, RANGE_TARGETS

__all__ = ["confidence_quality", "crps_gaussian", "spread_ratio", "confidence_summary", "position_share", "risk_coverage_auc", "sigma_calibration", "qss", "quantile_coverage", "interval_coverage", "mean_interval_width", "winkler_score", "threshold_trades",
           "profit_factor", "win_loss_ratio", "net_sharpe", "evaluate_heads", "NAN_POLICIES", "AUC_NAN_POLICIES", "rank_average", "pearson", "spearman", "partial_spearman", "auc",
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


# ───────────────────────────── quantile / interval metrics (additive) ─────────────────────────────
def _finite_pairs(*arrays):
    arrs = [_float_1d(a) for a in arrays]
    n = {len(a) for a in arrs}
    if len(n) != 1:
        raise ValueError(f"lengths differ: {sorted(n)}")
    ok = np.ones(len(arrs[0]), bool)
    for a in arrs:
        ok &= np.isfinite(a)
    return [a[ok] for a in arrs]


def quantile_coverage(y, q) -> float:
    """Share of rows with y <= q (the empirical coverage of a predicted quantile; ideally tau). Non-finite rows dropped; none -> NaN."""
    y, q = _finite_pairs(y, q)
    return float(np.mean(y <= q)) if len(y) else float("nan")


def qss(y, q, taus, baseline=None) -> float:
    """Quantile skill score = 1 - mean pinball(model) / mean pinball(baseline). `q`: (n, len(taus)) predicted quantiles; `baseline`: the
    reference (n, len(taus)) matrix, default = the constant empirical quantiles of y (the unconditional forecast). 1 = perfect, 0 = no
    better than the baseline, < 0 = worse. Rows with a non-finite y or q are dropped; a zero baseline loss -> NaN."""
    y = _float_1d(y)
    q = np.asarray(q, dtype="float64")
    taus = tuple(taus)
    if q.ndim != 2 or q.shape != (len(y), len(taus)):
        raise ValueError(f"q must be (n, len(taus)) = ({len(y)}, {len(taus)}), got {q.shape}")
    ok = np.isfinite(y) & np.isfinite(q).all(axis=1)
    if baseline is not None:
        baseline = np.asarray(baseline, dtype="float64")
        if baseline.shape != q.shape:
            raise ValueError(f"baseline shape {baseline.shape} != q shape {q.shape}")
        ok &= np.isfinite(baseline).all(axis=1)
    y, q = y[ok], q[ok]
    if len(y) == 0:
        return float("nan")
    base = np.tile(np.quantile(y, taus), (len(y), 1)) if baseline is None else baseline[ok]
    den = mean_pinball(y, base, taus)
    return float(1.0 - mean_pinball(y, q, taus) / den) if den > 0 else float("nan")


def interval_coverage(y, lo, hi) -> float:
    """Share of rows with lo <= y <= hi (a crossed interval, lo > hi, covers nothing). Non-finite rows dropped; none -> NaN."""
    y, lo, hi = _finite_pairs(y, lo, hi)
    return float(np.mean((y >= lo) & (y <= hi))) if len(y) else float("nan")


def mean_interval_width(lo, hi) -> float:
    """Mean of hi - lo over the finite rows (report it with the coverage: a wide interval covers anything). None -> NaN."""
    lo, hi = _finite_pairs(lo, hi)
    return float(np.mean(hi - lo)) if len(lo) else float("nan")


def winkler_score(y, lo, hi, alpha: float) -> float:
    """Mean Winkler (interval) score of a (1 - alpha) interval: width + (2 / alpha) * distance outside the interval (lower = better;
    it punishes both width and misses, so it cannot be gamed by either). 0 < alpha < 1. Non-finite rows dropped; none -> NaN."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    y, lo, hi = _finite_pairs(y, lo, hi)
    if not len(y):
        return float("nan")
    return float(np.mean((hi - lo) + (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))))


def crps_gaussian(y, mu, sigma) -> float:
    """Mean continuous ranked probability score of N(mu, sigma^2) forecasts (closed form: sigma * (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)),
    z = (y - mu) / sigma; lower = better; a proper score of the whole predictive distribution, in the units of y). sigma must be > 0.
    Non-finite rows dropped; none -> NaN."""
    y, mu, sigma = _finite_pairs(y, mu, sigma)
    if not len(y):
        return float("nan")
    if np.any(sigma <= 0):
        raise ValueError("crps_gaussian needs sigma > 0")
    from scipy.stats import norm
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))))


# ───────────────────────────── anti-fear diagnostics (a model that predicts 'nothing to be safe') ─────────────────────────────
def spread_ratio(mu, y) -> float:
    """std(mu) / std(y) over the finite rows: ~0 = the head collapsed to a constant ('fear of predicting'), ~1 = it moves like the label.
    NaN when std(y) is 0 or there are no rows."""
    mu, y = _finite_pairs(mu, y)
    sy = float(np.std(y)) if len(y) else 0.0
    return float(np.std(mu)) / sy if sy > 0 else float("nan")


def confidence_summary(conf, floor: float = 0.55) -> dict:
    """mean / std of a confidence P(direction correct) and the share below ``floor`` (0.55: close to a coin flip)."""
    (c,) = _finite_pairs(conf)
    if not len(c):
        return {"mean": float("nan"), "std": float("nan"), "share_below": float("nan")}
    return {"mean": float(np.mean(c)), "std": float(np.std(c)), "share_below": float(np.mean(c < floor))}


def position_share(mu, sigma, k: float = 1.0, threshold: float = 0.1, eps: float = 1e-6) -> float:
    """Share of rows whose utility position |tanh(k * mu / (sigma + eps))| exceeds ``threshold`` (0 = the model never takes a position)."""
    mu, sigma = _finite_pairs(mu, sigma)
    return float(np.mean(np.abs(np.tanh(k * mu / (sigma + eps))) > threshold)) if len(mu) else float("nan")


def risk_coverage_auc(conf, correct) -> float:
    """Area under the risk-coverage curve of a confidence score (lower = better): rows are sorted by decreasing confidence and the error
    rate of the top-n rows is averaged over n = 1..N. A useless confidence gives the overall error rate; a perfect one approaches 0."""
    conf, correct = _finite_pairs(conf, correct)
    if not len(conf):
        return float("nan")
    err = 1.0 - (np.asarray(correct) > 0.5).astype("float64")[np.argsort(-conf, kind="stable")]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def confidence_quality(scores, mu, y) -> dict:
    """Quality of confidence scores as predictors of "the direction of mu is right" (correct = mu * y > 0): for each named score in
    ``scores`` (``{name: array}``) the risk-coverage AUC (lower = better; a useless score gives 1 - accuracy) and the AUROC of the score
    against ``correct`` (0.5 = useless), plus the overall ``accuracy``. Compare the old confidence output with the TCP head with this."""
    mu, y = _finite_pairs(mu, y)
    correct = (mu * y > 0).astype("float64")
    out = {"accuracy": float(correct.mean()) if len(correct) else float("nan")}
    for name, sc in scores.items():
        sc = np.asarray(sc, dtype="float64").reshape(-1)
        if sc.shape != correct.shape:
            raise ValueError(f"confidence_quality: score {name!r} has {sc.shape[0]} rows, mu/y (finite pairs) have {correct.shape[0]}")
        out[name] = {"risk_coverage_auc": risk_coverage_auc(sc, correct), "auroc": auc(correct, sc, nan_policy="drop")}
    return out


def sigma_calibration(y, mu, sigma, z: float = 1.645) -> float:
    """Coverage of the central interval mu +- z * sigma (z = 1.645: nominal 90%): the share of rows inside; compare with the nominal level."""
    y, mu, sigma = _finite_pairs(y, mu, sigma)
    return float(np.mean(np.abs(y - mu) <= z * sigma)) if len(y) else float("nan")


# ───────────────────────────── trading metrics (additive; no threshold optimisation here) ─────────────────────────────
def threshold_trades(pred, ret, threshold: float, round_trip_cost: float = 0.0) -> np.ndarray:
    """Per-trade NET returns of the rule "trade only when |pred| > threshold, side = sign(pred), hold one period": side * ret - cost.
    `round_trip_cost` is a fraction of notional paid once per trade (entry + exit), in the units of `ret`. Rows with a non-finite pred
    or ret are not traded. `threshold` is given, never searched (choosing it on the same data is a separate, validated step)."""
    if threshold < 0 or round_trip_cost < 0:
        raise ValueError("threshold and round_trip_cost must be >= 0")
    pred, ret = _finite_pairs(pred, ret)
    take = np.abs(pred) > threshold
    return np.sign(pred[take]) * ret[take] - float(round_trip_cost)


def profit_factor(net) -> float:
    """Sum of winning trades / |sum of losing trades| (inf when there are wins and no loss; NaN when there is neither a win nor a loss)."""
    net = _float_1d(net)
    net = net[np.isfinite(net)]
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    if losses > 0:
        return float(gains / losses)
    return float("inf") if gains > 0 else float("nan")


def win_loss_ratio(net) -> float:
    """Average winning trade / |average losing trade| (NaN when there is no win or no loss)."""
    net = _float_1d(net)
    net = net[np.isfinite(net)]
    wins, losses = net[net > 0], net[net < 0]
    if not len(wins) or not len(losses):
        return float("nan")
    return float(wins.mean() / -losses.mean())


def net_sharpe(ret, position, cost_per_turnover: float = 0.0, periods_per_year: float | None = None) -> float:
    """Sharpe of the NET per-period return position * ret - cost_per_turnover * |position change| (the first position is bought from 0, so
    its entry is charged). mean / std(ddof=1), times sqrt(periods_per_year) when given. Fewer than 3 finite periods or zero std -> NaN."""
    ret, pos = _float_1d(ret), _float_1d(position)
    if ret.shape != pos.shape:
        raise ValueError(f"shapes differ: {ret.shape} != {pos.shape}")
    net = pos * ret - cost_per_turnover * np.abs(np.diff(pos, prepend=0.0))
    net = net[np.isfinite(net)]
    if len(net) < 3 or np.std(net, ddof=1) == 0:
        return float("nan")
    ann = np.sqrt(periods_per_year) if periods_per_year else 1.0
    return float(net.mean() / net.std(ddof=1) * ann)


# ───────────────────────────── per-head evaluation policy (owner decision 2026-10-31) ─────────────────────────────
def _regression_row(p, y) -> dict:
    p, y = _finite_pairs(p, y)
    n = int(len(y))
    if n == 0:
        return {"n": 0, "rank_ic": float("nan"), "mae": float("nan"), "rmse": float("nan"), "mae_naive": float("nan"),
                "rmse_naive": float("nan"), "mae_skill": float("nan"), "rmse_skill": float("nan")}
    mae, rmse = float(np.mean(np.abs(y - p))), float(np.sqrt(np.mean((y - p) ** 2)))
    mae0, rmse0 = float(np.mean(np.abs(y))), float(np.sqrt(np.mean(y ** 2)))      # naive = predict 0 (no change)
    return {"n": n, "rank_ic": spearman(p, y), "mae": mae, "rmse": rmse, "mae_naive": mae0, "rmse_naive": rmse0,
            "mae_skill": float(1.0 - mae / mae0) if mae0 > 0 else float("nan"),
            "rmse_skill": float(1.0 - rmse / rmse0) if rmse0 > 0 else float("nan")}


def _direction_row(p, y, score) -> dict:
    p, y, score = _finite_pairs(p, y, score)
    moved = y != 0                                    # a zero return has no direction
    p, y, score = p[moved], y[moved], score[moved]
    if not len(y):
        return {"n_direction": 0, "hit_rate": float("nan"), "direction_auc": float("nan")}
    return {"n_direction": int(len(y)), "hit_rate": float(np.mean(np.sign(p) == np.sign(y))),
            "direction_auc": auc(y > 0, score)}


def evaluate_heads(preds, targets, heads, *, direction_target: str = DIRECTION_TARGET, direction_scores=None,
                   intervals=None) -> dict:
    """Apply the per-head evaluation policy. `preds` / `targets`: {target: array} of point predictions (mu) and realised values in the
    same units; `heads`: a core.heads HeadSet (or any iterable of HeadSpec; only `.target` and `.role` are read, role in core.heads.ROLES).
    Every target with a "regression" head gets: n, rank_ic, mae, rmse and the skill vs the NAIVE ZERO prediction (mae_skill = 1 - MAE_model /
    MAE_naive, naive = predicting 0, i.e. no change; rmse_skill likewise). Direction metrics (hit_rate = share of sign(pred) == sign(true) on
    rows with a non-zero true value; direction_auc = AUC of the score for true > 0) are computed ONLY for `direction_target` (core.schema.
    DIRECTION_TARGET = close) and only when it has a regression head; high / low never get them. `direction_scores`: optional
    {target: score} (e.g. the classification head's p_up) used for direction_auc instead of the point prediction.
    `intervals`: {name: {"target": t, "low": array, "high": array, "alpha": optional nominal miss rate}}; when None and the heads include
    core.schema.RANGE_TARGETS (low, high) and the direction target is available, one interval "range" = [pred low, pred high] vs the true
    direction target is evaluated. Each interval reports coverage AND mean_width (never coverage alone), crossed_share (low > high) and,
    given alpha, the Winkler score. Returns {"heads": {target: {...}}, "intervals": {name: {...}}}."""
    hs = getattr(heads, "heads", heads)
    reg_targets = []
    for h in hs:
        if h.role not in ROLES:
            raise ValueError(f"head {getattr(h, 'name', h)!r}: role {h.role!r} not in core.heads.ROLES {ROLES}")
        if h.role == "regression" and h.target not in reg_targets:
            reg_targets.append(h.target)
    if not reg_targets:
        raise ValueError("evaluate_heads: the head set has no regression head")
    for t in reg_targets:
        if t not in preds or t not in targets:
            raise KeyError(f"evaluate_heads: head target {t!r} needs both preds and targets")
    scores = dict(direction_scores or {})
    out = {"heads": {}, "intervals": {}}
    for t in reg_targets:
        row = _regression_row(preds[t], targets[t])
        if t == direction_target:
            row.update(_direction_row(preds[t], targets[t], scores.get(t, preds[t])))
        out["heads"][t] = row
    if intervals is None:
        lo_t, hi_t = RANGE_TARGETS
        intervals = ({"range": {"target": direction_target, "low": preds[lo_t], "high": preds[hi_t]}}
                     if lo_t in reg_targets and hi_t in reg_targets and direction_target in targets else {})
    for name, spec in intervals.items():
        y, lo, hi = targets[spec["target"]], spec["low"], spec["high"]
        lo_f, hi_f = _finite_pairs(lo, hi)
        row = {"target": spec["target"], "coverage": interval_coverage(y, lo, hi), "mean_width": mean_interval_width(lo, hi),
               "crossed_share": float(np.mean(lo_f > hi_f)) if len(lo_f) else float("nan")}
        if spec.get("alpha") is not None:
            row["alpha"] = float(spec["alpha"])
            row["winkler"] = winkler_score(y, lo, hi, spec["alpha"])
        out["intervals"][name] = row
    return out

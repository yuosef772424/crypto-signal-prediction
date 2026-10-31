"""Old-vs-new proof for the consumers migrated to stats/ (hardening audit C7, PR 10): each class pins the migrated consumer to a
VERBATIM copy of its pre-migration definition on fixed random inputs with ties and NaNs (assert_array_equal = bit-equal unless the
test says rtol). The legacy copies live here only; they are the frozen reference, never imported by code.
    python -m pytest tests/test_stats_migration.py -q
"""
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "docs" / "research" / "audit"))


def same(testcase, new, old, msg=""):
    """Bit-equal for floats/arrays/Series, NaN == NaN."""
    new, old = np.asarray(new, dtype="float64"), np.asarray(old, dtype="float64")
    np.testing.assert_array_equal(new, old, msg)


def close(testcase, new, old, msg=""):
    """rtol=1e-12 (documented): the old code went through scipy.spearmanr (pandas Series.corr), whose last ulp differs from
    np.corrcoef of the average ranks that stats.spearman uses. NaN positions must still match exactly."""
    new, old = np.asarray(new, dtype="float64"), np.asarray(old, dtype="float64")
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=0, equal_nan=True, err_msg=msg)


def panel_frame(seed, n_days=14, n_coins=30, levels=5, nan_frac=0.08):
    """Long frame (timestamp, asset, score, target, pvol) with tie-heavy values, NaNs, and a few tiny / constant days."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        k = int(rng.integers(3, n_coins + 1)) if d % 4 else n_coins
        score = rng.integers(0, levels, k).astype("float64")
        target = (score + rng.integers(-2, 3, k)).astype("float64")
        pvol = rng.integers(0, levels + 2, k).astype("float64")
        if d == 5:
            score[:] = 1.0                                  # constant day
        rows.append(pd.DataFrame({"timestamp": d, "asset": np.arange(k), "score": score, "target": target, "pvol": pvol}))
    df = pd.concat(rows, ignore_index=True)
    for c in ("score", "target"):
        df.loc[rng.random(len(df)) < nan_frac, c] = np.nan
    return df


# ───────────────────────── cross_asset/report.py (daily_ic, vol_quintile_ic, partial_ic) ─────────────────────────
def _legacy_t(x):
    x = pd.Series(x).dropna()
    return (float(x.mean()), float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan,
            int(len(x)))


def _legacy_daily_ic(d, score, target="r", min_coins=10):
    g = d.groupby("timestamp")
    ic = g.apply(lambda x: x[score].corr(x[target], method="spearman") if len(x) >= min_coins else np.nan)
    return _legacy_t(ic)


def _legacy_vol_quintile_ic(d, score, target="r", n_q=5, min_cell=5):
    def per_day(x):
        if len(x) < n_q * min_cell:
            return np.nan
        q = pd.qcut(x["pvol"].rank(method="first"), n_q, labels=False)
        v = [c[score].corr(c[target], method="spearman") for _, c in x.groupby(q) if len(c) >= min_cell]
        return np.nanmean(v) if v else np.nan
    return _legacy_t(d.groupby("timestamp").apply(per_day))


def _legacy_partial_ic(d, score, target="r", min_coins=10):
    def per_day(x):
        if len(x) < min_coins:
            return np.nan
        z = np.c_[np.ones(len(x)), x["pvol"].rank().to_numpy()]
        res = [a - z @ np.linalg.lstsq(z, a, rcond=None)[0] for a in (x[score].rank().to_numpy(), x[target].rank().to_numpy())]
        return np.corrcoef(res[0], res[1])[0, 1]
    return _legacy_t(d.groupby("timestamp").apply(per_day))


class CrossAssetReportMigration(unittest.TestCase):
    def setUp(self):
        from cross_asset import report
        self.report = report

    def test_daily_ic_vol_quintile_ic_match_the_old_pandas_versions(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(6):
                d = panel_frame(seed)
                for min_coins in (3, 10):
                    close(self, self.report.daily_ic(d, "score", "target", min_coins), _legacy_daily_ic(d, "score", "target", min_coins))
                close(self, self.report.vol_quintile_ic(d, "score", "target"), _legacy_vol_quintile_ic(d, "score", "target"))

    def test_partial_ic_matches_the_old_lstsq_version_on_nan_free_days(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(6):
                d = panel_frame(seed, nan_frac=0.0)
                same(self, self.report.partial_ic(d, "score", "target", 3), _legacy_partial_ic(d, "score", "target", 3))


# ───────────────────────── cross_asset/train.py (group_ic) ─────────────────────────
def _legacy_group_ic(ps, score, target, min_n=10):
    df = pd.DataFrame({"g": ps.day_key, "s": np.asarray(score, "float64"), "r": np.asarray(target, "float64")})
    df = df[df.groupby("g")["g"].transform("size") >= min_n]
    if df.empty:
        return float("nan")
    rk = df.groupby("g")[["s", "r"]].rank()
    rk["g"] = df["g"]
    ic = rk.groupby("g").apply(lambda x: x["s"].corr(x["r"]))
    return float(ic.mean())


class CrossAssetTrainMigration(unittest.TestCase):
    def test_group_ic_is_bit_equal_to_the_old_pandas_version(self):
        from types import SimpleNamespace
        from cross_asset import train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(8):
                d = panel_frame(seed, nan_frac=0.1 if seed % 2 else 0.0)
                ps = SimpleNamespace(day_key=d["timestamp"].to_numpy())
                for min_n in (3, 10, 100):
                    same(self, train.group_ic(ps, d["score"].to_numpy(), d["target"].to_numpy(), min_n),
                         _legacy_group_ic(ps, d["score"].to_numpy(), d["target"].to_numpy(), min_n))


# ───────────────────────── evaluation/targets.py (wilson_ci) ─────────────────────────
def _legacy_wilson_ci(k, n, z: float = 1.96):
    n = float(n)
    if n <= 0:
        return float('nan'), float('nan')
    p = float(k) / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


class EvaluationTargetsMigration(unittest.TestCase):
    def test_wilson_ci_is_bit_equal_to_the_old_definition(self):
        import _nbload
        ns = _nbload.exec_evaluation_module("targets", {"__name__": "audit_nb"})
        new = ns["wilson_ci"]
        ks_ns = [(0, 0), (0, 1), (1, 1), (5, 10), (0, 20), (20, 20), (3, 7), (500, 1000), (np.int64(12), np.int64(40)),
                 (np.float64(2.0), 9), (4, -3)]
        rng = np.random.default_rng(0)
        for n in rng.integers(1, 3000, 300):
            ks_ns.append((int(rng.integers(0, n + 1)), int(n)))
        for k, n in ks_ns:
            for z in (1.96, 1.0, 2.576):
                same(self, new(k, n, z), _legacy_wilson_ci(k, n, z), f"k={k} n={n} z={z}")
        same(self, new(5, 10), _legacy_wilson_ci(5, 10))


# ───────────────────────── evaluation/trust_calibration.py (_ece_quantile) ─────────────────────────
def _legacy_ece_quantile(p, y, n_bins: int = 10) -> float:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) == 0:
        return float('nan')
    try:
        bins = pd.qcut(p, q=min(n_bins, len(np.unique(p))), duplicates='drop')
    except Exception:
        bins = pd.cut(p, bins=n_bins)
    g = pd.DataFrame({'p': p, 'y': y, 'b': bins}).groupby('b', observed=True)
    return float(sum(len(x) / len(p) * abs(x['p'].mean() - x['y'].mean()) for _, x in g))


class EvaluationTrustMigration(unittest.TestCase):
    def test_ece_quantile_is_bit_equal_to_the_old_definition(self):
        import _nbload
        ns = _nbload.load_evaluation()
        new = ns["_ece_quantile"]
        rng = np.random.default_rng(0)
        cases = [(np.full(100, 0.7), np.r_[np.ones(60), np.zeros(40)]), (np.array([np.nan]), np.array([1.0])),
                 (np.array([]), np.array([])), (np.array([0.5]), np.array([1.0])), (np.array([0.2, 0.8]), np.array([0.0, 1.0]))]
        for seed in range(12):
            n = int(rng.integers(5, 800))
            p = rng.random(n)
            if seed % 3 == 0:
                p = np.round(p * 5) / 5                                   # heavy ties -> duplicate quantile edges
            if seed % 4 == 1:
                p[rng.random(n) < 0.1] = np.nan
            y = (rng.random(n) < p).astype(float)
            if seed % 5 == 2:
                y[rng.random(n) < 0.1] = np.nan
            if seed % 6 == 3:
                p[rng.random(n) < 0.02] = np.inf
            cases.append((p, y))
        for p, y in cases:
            for n_bins in (1, 5, 10, 20):
                same(self, new(p, y, n_bins), _legacy_ece_quantile(p, y, n_bins), f"n_bins={n_bins} p={p[:5]}")


# ───────────────────────── legacy copies shared by the model / workflow AUC + Spearman consumers ─────────────────────────
def _legacy_rank(a):
    return pd.Series(np.asarray(a, dtype=np.float64)).rank().to_numpy()


def _legacy_diag_auc(y, s):                       # model/diagnostics.py::_auc
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    return float((_legacy_rank(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _legacy_diag_spearman(a, b):                  # model/diagnostics.py::_spearman
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(_legacy_rank(a), _legacy_rank(b))[0, 1])


def _legacy_wf_auc(y, s):                         # workflow/generalization.py::_auc
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_cases():
    """(y, s) pairs: random, tie-heavy, single-class, NaN-in-negatives, NaN-in-positives, constant."""
    rng = np.random.default_rng(0)
    out = []
    for seed in range(10):
        n = int(rng.integers(4, 300))
        y = (rng.random(n) < 0.4).astype(int) if seed % 5 else np.ones(n, int)
        s = rng.normal(size=n) if seed % 2 else rng.integers(0, 4, n).astype("float64")
        out.append((y, s))
    out.append((np.array([0, 1, 0, 1]), np.full(4, 0.3)))
    return out


def auc_cases_with_nan():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    neg_nan = np.array([0.1, np.nan, 0.35, 0.8, 0.2, 0.6, 0.55, 0.3])
    pos_nan = np.array([0.1, 0.4, np.nan, 0.8, 0.2, 0.6, 0.55, 0.3])
    return [(y, neg_nan), (y, pos_nan)]


def spearman_cases():
    rng = np.random.default_rng(1)
    out = []
    for seed in range(10):
        n = int(rng.integers(1, 200))
        a = rng.integers(0, 5, n).astype("float64") if seed % 2 else rng.normal(size=n)
        b = a + rng.integers(-1, 2, n)
        out.append((a, b))
    out.append((np.ones(20), np.arange(20.0)))
    out.append((np.arange(20.0), np.ones(20)))
    a = np.arange(20.0)
    a[3] = np.nan
    out.append((a, np.arange(20.0)[::-1].copy()))
    return out


class WorkflowCapacityMigration(unittest.TestCase):
    def test_skill_is_bit_equal_to_the_old_auc_and_spearman(self):
        import _nbload
        ns = {"__name__": "audit_nb", "np": np, "pd": pd}
        _nbload.workflow_package().load_into(ns, only=("capacity",))
        skill = ns["_skill"]
        for y, s in auc_cases() + auc_cases_with_nan():
            same(self, skill("class", y, s), _legacy_wf_auc(y > 0, s))
        for a, b in spearman_cases():
            same(self, skill("reg", b, a), _legacy_diag_spearman(a, b))


class ModelLayerReportMigration(unittest.TestCase):
    def test_metric_is_bit_equal_to_the_old_diagnostics_auc_and_spearman(self):
        import _nbload
        ns = _nbload.load_model()
        metric = ns["_metric"]
        for y, s in auc_cases() + auc_cases_with_nan():
            same(self, metric("class", y.astype(float) * 2 - 1, s), _legacy_diag_auc(y, s))
        for a, b in spearman_cases():
            same(self, metric("reg", b, a), _legacy_diag_spearman(a, b))


class ModelDiagnosticsMigration(unittest.TestCase):
    def test_perm_metric_is_bit_equal_to_the_old_private_auc_and_spearman(self):
        import _nbload
        ns = _nbload.load_model()
        self.assertNotIn("_auc", ns)
        self.assertNotIn("_spearman", ns)
        perm = ns["_perm_metric"]
        for (y, s), (a, b) in zip(auc_cases() + auc_cases_with_nan(), spearman_cases() + spearman_cases()):
            n = min(len(y), len(a))
            yy, ss, aa, bb = y[:n], s[:n], a[:n], b[:n]
            ys = {"y_high_class": yy * 2 - 1, "y_high_reg": bb}
            out = {"c": ss, "r": aa}
            got = perm(out, ys, {"high_class": "c", "high_reg": "r"})
            same(self, got["high_class"], _legacy_diag_auc(yy > 0, ss))
            same(self, got["high_reg"], _legacy_diag_spearman(aa, bb))


class WorkflowGeneralizationMigration(unittest.TestCase):
    def test_trivial_range_baseline_auc_is_bit_equal_to_the_old_scipy_auc(self):
        import _nbload
        from core.schema import LAST_COLUMN_INDEX as LC
        ns = {"__name__": "audit_nb", "np": np, "pd": pd}
        _nbload.workflow_package().load_into(ns, only=("generalization",))
        self.assertNotIn("_auc", ns)
        rng = np.random.default_rng(5)
        for seed in range(8):
            n = int(rng.integers(5, 400))
            lc = np.zeros((n, len(LC)))
            lo = rng.integers(90, 100, n).astype("float64")                       # tie-heavy prices; some zero ranges
            hi = lo + rng.integers(0, 4, n)
            cl = np.minimum(hi, lo + rng.integers(0, 4, n))
            lc[:, LC["last_low"]], lc[:, LC["last_high"]], lc[:, LC["last_close"]] = lo, hi, cl
            y = np.where(rng.random(n) < 0.5, 1.0, -1.0) if seed else np.ones(n)
            p = np.where(hi - lo > 0, (cl - lo) / np.where(hi - lo > 0, hi - lo, 1), 0.5)
            for sign in (None, 1, -1):
                got = ns["trivial_range_baseline"](lc, y, "high", sign)
                if sign is None:
                    raw = _legacy_wf_auc(y > 0, p)
                    want = (float("nan"), 1) if not np.isfinite(raw) else ((raw, 1) if raw >= 0.5 else (1.0 - raw, -1))
                else:
                    want = (_legacy_wf_auc(y > 0, sign * p), sign)
                same(self, got[0], want[0])
                self.assertEqual(got[1], want[1])


# ───────────────────────── signal_eval/core.py (compute_ic) ─────────────────────────
def _legacy_compute_ic(predictions, actuals, method="spearman", min_samples=10):
    predictions = np.asarray(predictions, dtype="float64")
    actuals = np.asarray(actuals, dtype="float64")
    if predictions.shape != actuals.shape:
        raise ValueError(f"shape {predictions.shape} != {actuals.shape}")
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method {method!r}")
    mask = np.isfinite(predictions) & np.isfinite(actuals)
    n = int(mask.sum())
    if n < min_samples:
        raise ValueError(f"{n} < {min_samples}")
    p, a = predictions[mask], actuals[mask]
    if np.std(p) < 1e-12 or np.std(a) < 1e-12:
        return 0.0
    if method == "spearman":
        p = pd.Series(p).rank().to_numpy()
        a = pd.Series(a).rank().to_numpy()
    corr = np.corrcoef(p, a)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


class SignalEvalCoreMigration(unittest.TestCase):
    def test_compute_ic_is_bit_equal_to_the_old_definition(self):
        import signal_eval
        ns = {"__name__": "audit_nb", "np": np, "pd": pd}
        signal_eval.exec_module("common", ns)
        signal_eval.exec_module("core", ns)
        new = ns["compute_ic"]
        rng = np.random.default_rng(7)
        cases = []
        for seed in range(30):
            n = int(rng.integers(1, 120))
            a = rng.integers(0, 4, n).astype("float64") if seed % 2 else rng.normal(size=n)
            b = a + rng.normal(size=n) * (seed % 3)
            if seed % 4 == 1:
                a[rng.random(n) < 0.15] = np.nan
            if seed % 5 == 2:
                b[rng.random(n) < 0.1] = np.inf
            cases.append((a, b))
        cases += [(np.ones(30), np.arange(30.0)), (np.arange(30.0), np.arange(30.0) * 1e-14), (np.arange(30.0) * 1e-14, np.arange(30.0))]
        for a, b in cases:
            for method in ("spearman", "pearson"):
                for min_samples in (0, 1, 10):
                    try:
                        want = _legacy_compute_ic(a, b, method, min_samples)
                    except ValueError:
                        with self.assertRaises(ValueError):
                            new(a, b, method, min_samples)
                        continue
                    same(self, new(a, b, method, min_samples), want, f"{method} min={min_samples}")


# ───────────────────────── tools/paired_compare.py (_ic_series) ─────────────────────────
def _legacy_ic_series(df, ts_col, y_col, pred_col, min_assets):
    sub = df[[ts_col, y_col, pred_col]].copy()
    sub = sub[np.isfinite(sub[y_col].to_numpy(dtype="float64")) & np.isfinite(sub[pred_col].to_numpy(dtype="float64"))]
    sub["_ry"] = sub.groupby(ts_col)[y_col].rank()
    sub["_rp"] = sub.groupby(ts_col)[pred_col].rank()
    out = {}
    for t, g in sub.groupby(ts_col, sort=True):
        if len(g) < min_assets:
            continue
        a, b = g["_ry"].to_numpy(), g["_rp"].to_numpy()
        if a.std() == 0 or b.std() == 0:
            continue
        out[t] = float(np.corrcoef(a, b)[0, 1])
    return pd.Series(out, dtype="float64")


class PairedCompareMigration(unittest.TestCase):
    def test_ic_series_is_bit_equal_to_the_old_definition(self):
        import paired_compare as pc
        for seed in range(8):
            d = panel_frame(seed, nan_frac=0.1)
            d.loc[d.index[::17], "target"] = np.inf
            for min_assets in (3, 8, 25):
                got = pc._ic_series(d, "timestamp", "target", "score", min_assets)
                want = _legacy_ic_series(d, "timestamp", "target", "score", min_assets)
                pd.testing.assert_series_equal(got, want, check_exact=True)


# ───────────────────────── workflow/candle_baseline.py (_day_bootstrap_auc_diff) ─────────────────────────
def _legacy_day_bootstrap_auc_diff(y, s_a, s_b, days, n_boot=300, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(days, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    point = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    return point, lo, hi


class WorkflowCandleBaselineMigration(unittest.TestCase):
    def test_day_bootstrap_auc_diff_matches_sklearn_version_to_1e12(self):
        import _nbload
        ns = {"__name__": "audit_nb", "np": np, "pd": pd}
        _nbload.workflow_package().load_into(ns, only=("candle_baseline",))
        new = ns["_day_bootstrap_auc_diff"]
        rng = np.random.default_rng(3)
        for seed in range(5):
            n = int(rng.integers(200, 600))
            days = rng.integers(0, 40, n)
            y = (rng.random(n) < 0.45).astype(int)
            s_a = np.round(rng.normal(size=n) + 0.5 * y, 1)                     # ties
            s_b = rng.normal(size=n)
            close(self, new(y, s_a, s_b, days, n_boot=60, seed=seed), _legacy_day_bootstrap_auc_diff(y, s_a, s_b, days, 60, seed))
        y1 = np.ones(10, int)
        with self.assertRaises(ValueError):
            new(y1, np.arange(10.0), np.arange(10.0), np.arange(10), n_boot=5)
        with self.assertRaises(ValueError):
            new(np.r_[np.zeros(5, int), np.ones(5, int)], np.r_[np.nan, np.arange(9.0)], np.arange(10.0), np.arange(10), n_boot=5)


if __name__ == "__main__":
    unittest.main()

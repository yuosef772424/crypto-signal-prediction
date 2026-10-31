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


if __name__ == "__main__":
    unittest.main()

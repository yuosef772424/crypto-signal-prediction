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


if __name__ == "__main__":
    unittest.main()

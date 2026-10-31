"""stats/ (Tier 1): one implementation per metric with an explicit tie/NaN policy (hardening audit C7, PR 10).
    python -m pytest tests/test_stats_metrics.py -q
"""
import ast
import math
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import check_deps  # noqa: E402
from stats import metrics as sm  # noqa: E402


def _tied_nan(seed=0, n=60, nan_frac=0.1, levels=6):
    """Tie-heavy float arrays with NaNs: values drawn from a few levels so ties abound."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, levels, n).astype("float64") + 0.0
    b = (a + rng.integers(-2, 3, n)).astype("float64")
    a[rng.random(n) < nan_frac] = np.nan
    b[rng.random(n) < nan_frac] = np.nan
    return a, b


class CorrelationTests(unittest.TestCase):
    def test_spearman_matches_scipy_with_ties_after_dropping_nan_pairs(self):
        from scipy.stats import spearmanr
        for seed in range(8):
            a, b = _tied_nan(seed)
            ok = np.isfinite(a) & np.isfinite(b)
            want = spearmanr(a[ok], b[ok])[0]
            for pol in ("drop_nonfinite", "drop"):
                np.testing.assert_allclose(sm.spearman(a, b, nan_policy=pol), want, rtol=1e-12, atol=0)

    def test_pearson_matches_numpy(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=50), rng.normal(size=50)
        self.assertEqual(sm.pearson(a, b), float(np.corrcoef(a, b)[0, 1]))

    def test_nan_policies_are_explicit(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
        b = np.array([1.0, 2.5, 2.0, 4.0, 5.0])
        self.assertTrue(math.isnan(sm.spearman(a, b, nan_policy="propagate")))
        self.assertTrue(np.isfinite(sm.spearman(a, b, nan_policy="drop")))
        # +-inf: dropped by drop_nonfinite, kept (ranked) by drop
        c = np.array([1.0, 2.0, 3.0, 4.0, np.inf])
        self.assertEqual(sm.spearman(c, b, nan_policy="drop"), sm.spearman(np.array([1, 2, 3, 4, 9.0]), b))
        self.assertEqual(sm.spearman(c, b, nan_policy="drop_nonfinite"), sm.spearman(c[:4], b[:4], min_n=3))
        with self.assertRaises(ValueError):
            sm.spearman(a, b, nan_policy="bogus")
        with self.assertRaises(ValueError):
            sm.spearman(a, b[:3])

    def test_degenerate_inputs_return_the_declared_value(self):
        const, x = np.ones(10), np.arange(10.0)
        self.assertTrue(math.isnan(sm.spearman(const, x)))
        self.assertEqual(sm.spearman(const, x, degenerate=0.0), 0.0)
        self.assertTrue(math.isnan(sm.spearman(x[:2], x[:2])))  # fewer than min_n=3
        self.assertAlmostEqual(sm.spearman(x[:2], x[:2], min_n=2), 1.0, places=12)
        self.assertEqual(sm.spearman(x * 1e-14, x, var_tol=1e-12, degenerate=0.0), 0.0)
        self.assertTrue(math.isnan(sm.pearson(x, const)))

    def test_ties_get_average_ranks(self):
        np.testing.assert_array_equal(sm.rank_average([3, 1, 1, 2]), [4.0, 1.5, 1.5, 3.0])
        np.testing.assert_array_equal(sm.rank_average([1.0, np.nan, 3.0]), [1.0, np.nan, 2.0])

    def test_partial_spearman_removes_the_control(self):
        rng = np.random.default_rng(2)
        c = rng.normal(size=200)
        a, b = c + 0.1 * rng.normal(size=200), c + 0.1 * rng.normal(size=200)
        self.assertGreater(sm.spearman(a, b), 0.9)
        self.assertLess(abs(sm.partial_spearman(a, b, c)), 0.3)


class AucTests(unittest.TestCase):
    def test_matches_sklearn_with_ties(self):
        from sklearn.metrics import roc_auc_score
        for seed in range(8):
            rng = np.random.default_rng(seed)
            y = rng.integers(0, 2, 80)
            s = rng.integers(0, 5, 80).astype("float64")
            np.testing.assert_allclose(sm.auc(y, s), roc_auc_score(y, s), rtol=1e-12)

    def test_single_class_is_nan_and_known_value(self):
        self.assertTrue(math.isnan(sm.auc([1, 1, 1], [0.1, 0.2, 0.3])))
        self.assertEqual(sm.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(sm.auc([0, 1], [0.5, 0.5]), 0.5)

    def test_nan_policies(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        s = np.array([0.1, 0.4, 0.35, 0.8, np.nan, 0.6])
        self.assertTrue(math.isnan(sm.auc(y, s)))                                  # propagate
        keep = ~np.isnan(s)
        self.assertEqual(sm.auc(y, s, nan_policy="drop"), sm.auc(y[keep], s[keep]))
        legacy = sm.auc(y, s, nan_policy="legacy_rank_skipna")                    # NaN on a negative row: finite, counts it in n0
        self.assertTrue(np.isfinite(legacy))
        s2 = s.copy()
        s2[5] = np.nan                                                            # NaN on a positive row: poisons the rank sum
        self.assertTrue(math.isnan(sm.auc(y, s2, nan_policy="legacy_rank_skipna")))
        with self.assertRaises(ValueError):
            sm.auc(y, s, nan_policy="bogus")


class IntervalAndLossTests(unittest.TestCase):
    def test_wilson_known_values_clip_and_empty(self):
        lo, hi = sm.wilson_interval(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertTrue(all(math.isnan(v) for v in sm.wilson_interval(0, 0)))
        self.assertEqual(sm.wilson_interval(10, 10)[1], 1.0)
        raw = sm.wilson_interval(10, 10, clip=False)
        self.assertAlmostEqual(raw[1], 1.0, places=12)
        self.assertEqual(sm.wilson_interval(0, 10)[0], 0.0)

    def test_pinball_and_mean_pinball(self):
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(sm.pinball_loss(y, 2.0, 0.9), [0.1, 0.0, 0.9])
        np.testing.assert_allclose(sm.pinball_loss(y, 2.0, 0.1), [0.9, 0.0, 0.1])
        q = np.c_[np.full(3, 1.0), np.full(3, 2.0)]
        want = float(np.mean([sm.pinball_loss(y, 1.0, 0.1).mean(), sm.pinball_loss(y, 2.0, 0.9).mean()]))
        self.assertEqual(sm.mean_pinball(y, q, (0.1, 0.9)), want)
        self.assertTrue(np.isnan(sm.pinball_loss([np.nan], [1.0], 0.5)).all())

    def test_ece_quantile_perfect_empty_and_constant_p(self):
        rng = np.random.default_rng(3)
        p = rng.random(4000)
        y = (rng.random(4000) < p).astype(float)
        self.assertLess(sm.ece_quantile(p, y), 0.05)
        self.assertGreater(sm.ece_quantile(p, 1.0 - y), 0.3)
        self.assertTrue(math.isnan(sm.ece_quantile([np.nan], [1.0])))
        self.assertAlmostEqual(sm.ece_quantile(np.full(10, 0.7), np.r_[np.ones(5), np.zeros(5)]), 0.2, places=12)


class TierOneTests(unittest.TestCase):
    def test_stats_imports_only_core_and_numeric_libraries(self):
        allowed = {"numpy", "pandas", "scipy", "core", "__future__"}
        for f in sorted((ROOT / "stats").glob("*.py")):
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
                for m in mods:
                    self.assertIn(m.split(".")[0], allowed, f"{f.name} imports {m}")

    def test_importing_stats_loads_no_repo_package(self):
        code = ("import sys; sys.path.insert(0, %r); before = set(sys.modules); import stats.metrics; "
                "new = sorted(set(sys.modules) - before); "
                "bad = [m for m in new if m.split('.')[0] in ('data','model','trainer','evaluation','signal_eval','workflow',"
                "'discovery','cross_asset','tools','tensorflow')]; assert not bad, bad; print('ok')" % str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "ok"), out.stderr)

    def test_zone_registration_and_direction(self):
        self.assertIn("stats", check_deps.CODE_PACKAGES)
        self.assertEqual(check_deps.zone_of("stats/metrics.py"), "stats")
        self.assertIsNone(check_deps.violation_for("stats", "core"))
        self.assertEqual(check_deps.violation_for("stats", "data"), "stats-may-import-only-core")
        self.assertEqual(check_deps.violation_for("stats", "tests"), "code-must-not-import-tests")
        for pkg in check_deps.CODE_PACKAGES:  # every package above tier 0 may import stats
            if pkg not in ("core", "stats"):
                self.assertIsNone(check_deps.violation_for(pkg, "stats"), pkg)
        self.assertEqual(check_deps.violation_for("core", "stats"), "core-must-not-import-repo")

    def test_check_deps_flags_a_repo_import_inside_stats(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "stats").mkdir()
            (root / "data").mkdir()
            (root / "stats" / "bad.py").write_text('"""card"""\nimport json\nfrom data import windows\nfrom core import schema\n', encoding="utf-8")
            (root / "core").mkdir()
            (root / "core" / "schema.py").write_text("x = 1\n", encoding="utf-8")
            (root / "data" / "windows.py").write_text("x = 1\n", encoding="utf-8")
            repo = check_deps.Repo(root, files=["stats/bad.py", "core/schema.py", "data/windows.py"])
            bad = [f"{rel}:{ln}" for rel in repo.py_files for ln, sev, rule, _t in check_deps.check_file(repo, rel)
                   if rule == "stats-may-import-only-core" and sev == "error"]
            self.assertEqual(bad, ["stats/bad.py:3"])
        lines, _n, errors, _w, _a = check_deps.run(check_deps.Repo(ROOT))
        self.assertEqual(errors, 0, lines)


if __name__ == "__main__":
    unittest.main()

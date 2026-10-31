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


class QuantileAndIntervalTests(unittest.TestCase):
    def test_quantile_coverage_and_qss(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=4000)
        taus = (0.1, 0.5, 0.9)
        ideal = np.tile(np.quantile(y, taus), (len(y), 1))
        for i, t in enumerate(taus):
            self.assertAlmostEqual(sm.quantile_coverage(y, ideal[:, i]), t, places=2)
        self.assertAlmostEqual(sm.qss(y, ideal, taus), 0.0, places=12)               # the baseline itself has zero skill
        sharp = np.c_[y - 0.01, y, y + 0.01]
        self.assertGreater(sm.qss(y, sharp, taus), 0.95)
        self.assertLess(sm.qss(y, ideal + 5.0, taus), 0.0)
        # explicit baseline; NaN rows dropped; shape errors raise
        y2, q2 = y.copy(), ideal.copy()
        y2[0], q2[1, 0] = np.nan, np.nan
        self.assertAlmostEqual(sm.qss(y2, q2, taus, baseline=ideal), 0.0, places=2)
        with self.assertRaises(ValueError):
            sm.qss(y, ideal[:, :2], taus)
        self.assertTrue(math.isnan(sm.qss([np.nan], [[1.0, 2.0, 3.0]], taus)))
        self.assertTrue(math.isnan(sm.quantile_coverage([], [])))

    def test_interval_coverage_width_and_winkler_cannot_be_gamed(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
        lo, hi = np.array([-1.0, 0.0, 3.0, 4.0, 0.0]), np.array([1.0, 2.0, 1.0, 5.0, 1.0])     # 3rd row crossed, 4th misses
        self.assertEqual(sm.interval_coverage(y, lo, hi), 0.5)
        self.assertEqual(sm.mean_interval_width(lo, hi), np.mean(hi - lo))
        # widening the interval raises coverage but not the Winkler score once it is wide enough
        rng = np.random.default_rng(1)
        yy = rng.normal(size=2000)
        tight = sm.winkler_score(yy, np.full(2000, -1.645), np.full(2000, 1.645), 0.1)
        wide = sm.winkler_score(yy, np.full(2000, -50.0), np.full(2000, 50.0), 0.1)
        self.assertGreater(sm.interval_coverage(yy, np.full(2000, -50.0), np.full(2000, 50.0)), 0.999)
        self.assertLess(tight, wide)
        self.assertAlmostEqual(sm.winkler_score([0.0], [-1.0], [1.0], 0.1), 2.0)           # inside: just the width
        self.assertAlmostEqual(sm.winkler_score([2.0], [-1.0], [1.0], 0.1), 2.0 + 20.0 * 1.0)
        with self.assertRaises(ValueError):
            sm.winkler_score(y, lo, hi, 1.5)
        with self.assertRaises(ValueError):
            sm.interval_coverage([1.0, 2.0], [0.0], [3.0])


class TradingMetricTests(unittest.TestCase):
    def test_threshold_trades_only_trades_above_the_threshold_and_charges_the_round_trip(self):
        pred = np.array([0.5, -0.5, 0.05, -0.05, np.nan, 0.5])
        ret = np.array([0.02, -0.03, 0.10, 0.10, 0.10, np.nan])
        net = sm.threshold_trades(pred, ret, 0.1, round_trip_cost=0.001)
        np.testing.assert_allclose(net, [0.02 - 0.001, 0.03 - 0.001])                       # long wins, short wins; small preds / NaN rows skipped
        self.assertEqual(len(sm.threshold_trades(pred, ret, 10.0)), 0)
        np.testing.assert_allclose(sm.threshold_trades(pred, ret, 0.0)[:2], [0.02, 0.03])
        with self.assertRaises(ValueError):
            sm.threshold_trades(pred, ret, -0.1)

    def test_profit_factor_and_win_loss_ratio(self):
        net = np.array([0.03, 0.01, -0.02, np.nan, -0.02])
        self.assertAlmostEqual(sm.profit_factor(net), 0.04 / 0.04)
        self.assertAlmostEqual(sm.win_loss_ratio(net), 0.02 / 0.02)
        self.assertAlmostEqual(sm.profit_factor([0.06, -0.02]), 3.0)
        self.assertAlmostEqual(sm.win_loss_ratio([0.06, 0.02, -0.02]), 2.0)
        self.assertEqual(sm.profit_factor([0.01, 0.02]), float("inf"))
        self.assertTrue(math.isnan(sm.profit_factor([])))
        self.assertTrue(math.isnan(sm.profit_factor([0.0, 0.0])))
        self.assertTrue(math.isnan(sm.win_loss_ratio([0.01, 0.02])))
        self.assertEqual(sm.profit_factor([-0.01]), 0.0)

    def test_net_sharpe_charges_turnover(self):
        rng = np.random.default_rng(2)
        ret = rng.normal(0.001, 0.01, 500)
        pos = np.sign(rng.normal(size=500))                                                 # flips often: high turnover
        gross = sm.net_sharpe(ret, pos)
        self.assertAlmostEqual(gross, float(np.mean(pos * ret) / np.std(pos * ret, ddof=1)), places=12)
        self.assertLess(sm.net_sharpe(ret, pos, cost_per_turnover=0.002), gross)
        flat = np.ones(500)                                                                 # buy once: the entry is the only charge
        net = flat * ret - np.r_[0.01, np.zeros(499)]
        self.assertAlmostEqual(sm.net_sharpe(ret, flat, cost_per_turnover=0.01), float(net.mean() / net.std(ddof=1)), places=12)
        self.assertAlmostEqual(sm.net_sharpe(ret, flat, periods_per_year=252), sm.net_sharpe(ret, flat) * math.sqrt(252), places=12)
        self.assertTrue(math.isnan(sm.net_sharpe([0.1, 0.2], [1, 1])))
        self.assertTrue(math.isnan(sm.net_sharpe(np.zeros(10), np.zeros(10))))
        with self.assertRaises(ValueError):
            sm.net_sharpe(ret, pos[:3])


class EvaluateHeadsTests(unittest.TestCase):
    def setUp(self):
        from core.heads import HeadSet
        self.heads = HeadSet.default(("high", "low", "close"))
        rng = np.random.default_rng(5)
        n = 600
        self.n = n
        close = rng.normal(0, 0.02, n)
        self.targets = {"close": close, "high": np.abs(close) + 0.01 + np.abs(rng.normal(0, 0.005, n)),
                        "low": -np.abs(close) - 0.01 - np.abs(rng.normal(0, 0.005, n))}
        self.preds = {"close": 0.5 * close + rng.normal(0, 0.01, n),
                      "high": self.targets["high"] + rng.normal(0, 0.004, n),
                      "low": self.targets["low"] + rng.normal(0, 0.004, n)}

    def test_direction_metrics_exist_only_for_close(self):
        r = sm.evaluate_heads(self.preds, self.targets, self.heads)
        self.assertEqual(sorted(r["heads"]), ["close", "high", "low"])
        for t in ("high", "low"):
            self.assertFalse({"hit_rate", "direction_auc", "n_direction"} & set(r["heads"][t]), t)
        c = r["heads"]["close"]
        y, p = self.targets["close"], self.preds["close"]
        self.assertEqual(c["hit_rate"], float(np.mean(np.sign(p) == np.sign(y))))
        self.assertEqual(c["direction_auc"], sm.auc(y > 0, p))
        self.assertGreater(c["direction_auc"], 0.6)

    def test_regression_metrics_and_naive_zero_skill(self):
        r = sm.evaluate_heads(self.preds, self.targets, self.heads)
        for t in ("high", "low", "close"):
            y, p = self.targets[t], self.preds[t]
            h = r["heads"][t]
            self.assertEqual(h["n"], self.n)
            self.assertAlmostEqual(h["mae"], float(np.mean(np.abs(y - p))), places=14)
            self.assertAlmostEqual(h["rmse"], float(np.sqrt(np.mean((y - p) ** 2))), places=14)
            self.assertAlmostEqual(h["mae_naive"], float(np.mean(np.abs(y))), places=14)            # naive = predict 0
            self.assertAlmostEqual(h["mae_skill"], 1.0 - h["mae"] / h["mae_naive"], places=14)
            self.assertAlmostEqual(h["rank_ic"], sm.spearman(p, y), places=14)
        self.assertGreater(r["heads"]["high"]["mae_skill"], 0.5)
        zero = sm.evaluate_heads({**self.preds, "close": np.zeros(self.n)}, self.targets, self.heads)
        self.assertEqual(zero["heads"]["close"]["mae_skill"], 0.0)                                  # predicting 0 has no skill by definition
        self.assertTrue(math.isnan(zero["heads"]["close"]["rank_ic"]))

    def test_range_interval_reports_coverage_with_width(self):
        r = sm.evaluate_heads(self.preds, self.targets, self.heads)
        iv = r["intervals"]["range"]
        lo, hi, y = self.preds["low"], self.preds["high"], self.targets["close"]
        self.assertEqual(iv["coverage"], float(np.mean((y >= lo) & (y <= hi))))
        self.assertEqual(iv["mean_width"], float(np.mean(hi - lo)))
        self.assertEqual(iv["crossed_share"], 0.0)
        self.assertNotIn("winkler", iv)
        wide = sm.evaluate_heads({**self.preds, "low": self.preds["low"] - 1.0, "high": self.preds["high"] + 1.0}, self.targets, self.heads)
        self.assertGreaterEqual(wide["intervals"]["range"]["coverage"], iv["coverage"])
        self.assertGreater(wide["intervals"]["range"]["mean_width"], iv["mean_width"] + 1.9)       # the gamed coverage shows in the width

    def test_explicit_intervals_alpha_and_direction_scores(self):
        n = self.n
        q = {"nig": {"target": "close", "low": self.preds["close"] - 0.03, "high": self.preds["close"] + 0.03, "alpha": 0.1}}
        score = np.linspace(0, 1, n)
        r = sm.evaluate_heads(self.preds, self.targets, self.heads, intervals=q, direction_scores={"close": score})
        self.assertEqual(list(r["intervals"]), ["nig"])
        self.assertIn("winkler", r["intervals"]["nig"])
        self.assertEqual(r["intervals"]["nig"]["winkler"], sm.winkler_score(self.targets["close"], q["nig"]["low"], q["nig"]["high"], 0.1))
        self.assertEqual(r["heads"]["close"]["direction_auc"], sm.auc(self.targets["close"] > 0, score))

    def test_roles_come_from_head_specs_and_unknown_inputs_raise(self):
        from core.heads import HeadSet
        only_close = HeadSet.default(("close",))
        r = sm.evaluate_heads({"close": self.preds["close"]}, {"close": self.targets["close"]}, only_close)
        self.assertEqual(list(r["heads"]), ["close"])
        self.assertEqual(r["intervals"], {})                                                       # no low/high heads: no interval
        class_only = HeadSet.default(("close",), class_only=True)                                  # no regression head at all
        with self.assertRaises(ValueError):
            sm.evaluate_heads({}, {}, class_only)
        with self.assertRaises(KeyError):
            sm.evaluate_heads({"close": self.preds["close"]}, {"close": self.targets["close"]}, self.heads)
        r2 = sm.evaluate_heads(self.preds, self.targets, list(self.heads.heads))                    # any iterable of HeadSpec
        self.assertEqual(r2["heads"]["high"], r["heads"].get("high", r2["heads"]["high"]))

    def test_nan_rows_are_dropped_not_propagated(self):
        p = {k: v.copy() for k, v in self.preds.items()}
        p["close"][::7] = np.nan
        r = sm.evaluate_heads(p, self.targets, self.heads)
        self.assertEqual(r["heads"]["close"]["n"], int(np.isfinite(p["close"]).sum()))
        self.assertTrue(np.isfinite(r["heads"]["close"]["rank_ic"]))


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

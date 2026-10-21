"""tools/paired_compare.py (card T2): per-timestamp loss, paired differential, block bootstrap, Newey-West, temporal design effect,
compare_arms. Synthetic data with pinned seeds only; no SIMD-sensitive fingerprints are pinned.
    python -m pytest tests/test_paired_compare.py -q
"""
import math
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import paired_compare as pc  # noqa: E402


def ar1(rng, T, rho, burn=200):
    e = rng.standard_normal(T + burn)
    x = np.zeros(T + burn)
    for t in range(1, T + burn):
        x[t] = rho * x[t - 1] + e[t]
    return x[burn:]


def panel(T=120, coins=6, seed=0, shift=0.0):
    """Long panel: y, pred_a, pred_b per (timestamp, coin); arm B has error scale 1, arm A scale 1 (+ shift on the mean error)."""
    rng = np.random.default_rng(seed)
    ts = np.repeat(np.arange(T) * 14_400_000_000_000, coins)
    y = rng.standard_normal(T * coins)
    return pd.DataFrame({"ts": ts, "coin": np.tile(np.arange(coins), T), "y": y,
                         "pa": y + rng.standard_normal(T * coins) + shift, "pb": y + rng.standard_normal(T * coins)})


class TestPerTimestampLoss(unittest.TestCase):
    def test_coin_mean_inside_timestamp(self):
        df = pd.DataFrame({"ts": [1, 1, 2, 2, 2], "y": [0., 0., 0., 0., 0.], "pa": [1., 3., 2., 2., 5.]})
        r = pc.per_timestamp_loss(df, ts_col="ts", y_col="y", pred_cols={"a": "pa"}, loss="abs")
        self.assertEqual(list(r.table.index), [1, 2])
        np.testing.assert_allclose(r.table["a"].to_numpy(), [2.0, 3.0])
        r2 = pc.per_timestamp_loss(df, ts_col="ts", y_col="y", pred_cols={"a": "pa"}, loss="sq")
        np.testing.assert_allclose(r2.table["a"].to_numpy(), [5.0, 11.0])
        self.assertEqual(r.n_timestamps, 2)

    def test_nonfinite_rows_dropped_jointly_and_counted(self):
        df = pd.DataFrame({"ts": [1, 1, 2], "y": [0., 0., 0.], "pa": [1., np.nan, 2.], "pb": [1., 5., 2.]})
        r = pc.per_timestamp_loss(df, ts_col="ts", y_col="y", pred_cols={"a": "pa", "b": "pb"}, loss="abs")
        self.assertEqual(r.n_rows_nonfinite, 1)
        self.assertEqual(float(r.table.loc[1, "b"]), 1.0)          # B's 5.0 is dropped too: same coin set for both arms

    def test_shuffling_coins_inside_timestamp_changes_nothing(self):
        df = panel()
        sh = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
        kw = dict(loss="abs", lags=3, block_len=4, n_boot=300, seed=5, kind="stationary", min_overlap=10, clamp="conservative")
        t = {"x": ("y", "pa", "pb")}
        a = pc.compare_arms(df, t, ts_col="ts", **kw).table
        b = pc.compare_arms(sh, t, ts_col="ts", **kw).table
        pd.testing.assert_frame_equal(a, b, rtol=1e-9, atol=1e-12)

    def test_option_errors(self):
        df = panel(T=5)
        with self.assertRaises(ValueError):
            pc.per_timestamp_loss(df, ts_col="ts", y_col="y", pred_cols={"a": "pa"}, loss="huber")
        with self.assertRaises(ValueError):
            pc.per_timestamp_loss(df, ts_col="ts", y_col="y", pred_cols={"a": "pa"}, loss_cols={"a": "pa"}, loss="abs")
        with self.assertRaises(ValueError):
            pc.per_timestamp_loss(df, ts_col="ts", y_col="y", loss_cols={"a": "pa"}, loss="abs")
        with self.assertRaises(KeyError):
            pc.per_timestamp_loss(df, ts_col="ts", y_col="y", pred_cols={"a": "nope"}, loss="abs")


class TestPairedDifferential(unittest.TestCase):
    def test_alignment_and_drop_counts(self):
        a = pd.Series([1., 2., 3., 4.], index=[1, 2, 3, 4])
        b = pd.Series([1., 1., 1., np.nan, 9.], index=[2, 3, 4, 5, 6])
        r = pc.paired_differential(a, b, min_overlap=2)
        self.assertEqual(list(r.d.index), [2, 3, 4])
        np.testing.assert_allclose(r.d.to_numpy(), [1.0, 2.0, 3.0])
        self.assertEqual((r.n_a, r.n_b, r.n_overlap, r.n_dropped_a, r.n_dropped_b), (4, 4, 3, 1, 1))   # A: ts 1; B: ts 6 (ts 5 is NaN)

    def test_raises_below_min_overlap(self):
        a = pd.Series(np.arange(10.), index=range(10))
        b = pd.Series(np.arange(10.), index=range(8, 18))
        with self.assertRaises(ValueError):
            pc.paired_differential(a, b, min_overlap=5)
        self.assertEqual(pc.paired_differential(a, b, min_overlap=2).n_overlap, 2)

    def test_bad_inputs(self):
        s = pd.Series([1., 2., 3.], index=[1, 2, 3])
        with self.assertRaises(ValueError):
            pc.paired_differential(s, pd.Series([1., 2.], index=[1, 1]), min_overlap=2)
        with self.assertRaises(TypeError):
            pc.paired_differential(s.to_numpy(), s, min_overlap=2)
        with self.assertRaises(ValueError):
            pc.paired_differential(s, s, min_overlap=1)


class TestNeweyWest(unittest.TestCase):
    def test_lags0_equals_iid_se(self):
        x = np.random.default_rng(0).standard_normal(400)
        r = pc.newey_west_mean(x, lags=0)
        self.assertAlmostEqual(r.se, math.sqrt(x.var() / len(x)), places=12)   # biased var (1/T), by construction
        self.assertAlmostEqual(r.t, x.mean() / r.se, places=12)

    def test_default_lag_rule(self):
        self.assertEqual(pc.nw_default_lag(100), 4)
        self.assertEqual(pc.nw_default_lag(500), int(math.floor(4 * 5 ** (2 / 9))))
        x = np.random.default_rng(1).standard_normal(500)
        self.assertEqual(pc.newey_west_mean(x, lags="nw_default").lags, pc.nw_default_lag(500))

    def test_p_value_matches_normal(self):
        r = pc.newey_west_mean(np.random.default_rng(2).standard_normal(300) + 0.2, lags=2)
        self.assertAlmostEqual(r.p_value, math.erfc(abs(r.t) / math.sqrt(2)), places=12)

    def test_constant_series_nan(self):
        r = pc.newey_west_mean(np.ones(50), lags=2)
        self.assertTrue(math.isnan(r.t) and math.isnan(r.p_value))

    def test_nonfinite_and_short_raise(self):
        with self.assertRaises(ValueError):
            pc.newey_west_mean(np.array([1., np.nan, 2.]), lags=1)
        with self.assertRaises(ValueError):
            pc.newey_west_mean(np.array([1.]), lags=0)
        with self.assertRaises(ValueError):
            pc.newey_west_mean(np.arange(10.), lags=10)


class TestIid(unittest.TestCase):
    def test_nw_and_bootstrap_agree_and_deff_near_one(self):
        d = np.random.default_rng(10).standard_normal(600) + 0.1
        nw = pc.newey_west_mean(d, lags="nw_default")
        for kind in pc.BOOT_KINDS:
            bs = pc.block_bootstrap_mean(d, block_len=1, n_boot=4000, seed=1, kind=kind)
            self.assertLess(abs(math.log(bs.p_value / nw.p_value)), 0.6)
            self.assertAlmostEqual(bs.mean, nw.mean, places=12)
            self.assertLessEqual(bs.ci_lo, bs.mean)
            self.assertGreaterEqual(bs.ci_hi, bs.mean)
        de = pc.temporal_design_effect(d, lags="nw_default", clamp="conservative")
        self.assertTrue(0.7 < de.deff_raw < 1.5, de.deff_raw)
        self.assertGreaterEqual(de.deff, 1.0)
        self.assertLessEqual(de.n_eff, 600.0)


class TestAr1(unittest.TestCase):
    RHO = 0.8

    def test_deff_matches_bartlett_formula_and_approaches_theory(self):
        T, L = 20000, 100
        x = ar1(np.random.default_rng(7), T, self.RHO)
        de = pc.temporal_design_effect(x, lags=L, clamp="conservative")
        k = np.arange(1, L + 1)
        formula = 1 + 2 * np.sum((1 - k / (L + 1)) * self.RHO ** k)           # population value at this truncation
        self.assertLess(abs(de.deff_raw - formula) / formula, 0.12)
        self.assertLess(abs(de.deff_raw - 9.0) / 9.0, 0.2)                       # (1+rho)/(1-rho) = 9
        self.assertAlmostEqual(de.n_eff, T / de.deff, places=9)
        self.assertFalse(de.clamped)

    def test_andrews_rule_deff_is_large(self):
        x = ar1(np.random.default_rng(8), 2000, self.RHO)
        self.assertGreater(pc.temporal_design_effect(x, lags="andrews", clamp="none").deff_raw, 4.0)

    def test_naive_se_too_small_and_hac_coverage(self):
        rng = np.random.default_rng(123)
        T, N = 500, 300
        cov = {"iid": 0, "nw_andrews": 0, "boot_andrews": 0}
        for i in range(N):
            x = ar1(rng, T, self.RHO)
            cov["iid"] += abs(x.mean()) <= 1.96 * x.std(ddof=1) / math.sqrt(T)
            r = pc.newey_west_mean(x, lags="andrews")
            cov["nw_andrews"] += abs(r.mean) <= 1.96 * r.se
            b = pc.block_bootstrap_mean(x, block_len="auto_andrews", n_boot=300, seed=i, kind="stationary")
            cov["boot_andrews"] += b.ci_lo <= 0.0 <= b.ci_hi
        cov = {k: v / N for k, v in cov.items()}
        self.assertLess(cov["iid"], 0.65, cov)                  # nominal 0.95: the naive iid se badly under-covers
        self.assertGreater(cov["nw_andrews"], 0.80, cov)
        self.assertGreater(cov["boot_andrews"], 0.78, cov)
        self.assertGreater(cov["nw_andrews"], cov["iid"] + 0.2)

    def test_bootstrap_se_above_iid_se(self):
        x = ar1(np.random.default_rng(9), 800, self.RHO)
        b = pc.block_bootstrap_mean(x, block_len=25, n_boot=1000, seed=0, kind="moving")
        self.assertGreater((b.ci_hi - b.ci_lo) / 3.92, 1.8 * x.std(ddof=1) / math.sqrt(800))


class TestDesignEffectClamp(unittest.TestCase):
    def test_negative_autocorrelation_clamped_conservatively(self):
        x = np.array([1., -1.] * 200) + np.random.default_rng(0).standard_normal(400) * 0.05
        c = pc.temporal_design_effect(x, lags=3, clamp="conservative")
        n = pc.temporal_design_effect(x, lags=3, clamp="none")
        self.assertLess(c.deff_raw, 1.0)
        self.assertEqual(c.deff, 1.0)
        self.assertTrue(c.clamped)
        self.assertEqual(c.n_eff, 400.0)
        self.assertAlmostEqual(n.deff, max(n.deff_raw, 1 / 400), places=12)
        self.assertGreater(n.n_eff, 400.0)

    def test_zero_variance_raises_and_unknown_clamp(self):
        with self.assertRaises(ValueError):
            pc.temporal_design_effect(np.ones(20), lags=2, clamp="none")
        with self.assertRaises(ValueError):
            pc.temporal_design_effect(np.arange(20.), lags=2, clamp="soft")

    def test_design_effect_by_target(self):
        df = panel(T=300, coins=8, seed=1)
        df["l1"] = (df.pa - df.y).abs()
        df["l2"] = (df.pb - df.y).abs()
        t = pc.design_effect_by_target(df, {"a": "l1", "b": "l2"}, ts_col="ts", lags=4, clamp="conservative", series="mean")
        self.assertEqual(list(t.index), ["a", "b"])
        self.assertTrue((t["n_timestamps"] == 300).all())
        ic = pc.design_effect_by_target(df, {"a": ("y", "pa")}, ts_col="ts", lags=4, clamp="conservative", series="ic", min_assets=4)
        self.assertEqual(int(ic.loc["a", "n_timestamps"]), 300)
        with self.assertRaises(ValueError):
            pc.design_effect_by_target(df, {"a": ("y", "pa")}, ts_col="ts", lags=4, clamp="conservative", series="ic")
        with self.assertRaises(ValueError):
            pc.design_effect_by_target(df, {"a": "l1"}, ts_col="ts", lags=4, clamp="conservative", series="median")


class TestBootstrapContract(unittest.TestCase):
    def setUp(self):
        self.d = np.random.default_rng(4).standard_normal(200) * 0.5 + 0.3

    def test_determinism_per_seed(self):
        for kind in pc.BOOT_KINDS:
            a = pc.block_bootstrap_mean(self.d, block_len=7, n_boot=500, seed=11, kind=kind)
            b = pc.block_bootstrap_mean(self.d, block_len=7, n_boot=500, seed=11, kind=kind)
            c = pc.block_bootstrap_mean(self.d, block_len=7, n_boot=500, seed=12, kind=kind)
            self.assertEqual(a, b)
            self.assertNotEqual((a.ci_lo, a.ci_hi), (c.ci_lo, c.ci_hi))

    def test_chunking_does_not_change_result(self):
        old = pc._CHUNK_CELLS
        try:
            a = pc.block_bootstrap_mean(self.d, block_len=7, n_boot=500, seed=3, kind="moving")
            pc._CHUNK_CELLS = 5000
            b = pc.block_bootstrap_mean(self.d, block_len=7, n_boot=500, seed=3, kind="moving")
        finally:
            pc._CHUNK_CELLS = old
        # chunking changes the draw order of the rng, so only statistical equality is required
        self.assertAlmostEqual(a.ci_lo, b.ci_lo, delta=0.05)

    def test_resamples_whole_blocks(self):
        rng = np.random.default_rng(0)
        x = np.arange(50.)
        idx_mean = pc._boot_means(x, 5, 3, "moving", rng)
        self.assertEqual(len(idx_mean), 3)
        # block_len == T moving bootstrap returns the sample mean every time
        np.testing.assert_allclose(pc._boot_means(x, 50, 4, "moving", rng), x.mean())

    def test_unknown_options_raise(self):
        d = self.d
        with self.assertRaises(ValueError):
            pc.block_bootstrap_mean(d, block_len=5, n_boot=10, seed=0, kind="circular")
        with self.assertRaises(ValueError):
            pc.block_bootstrap_mean(d, block_len="auto", n_boot=10, seed=0, kind="moving")
        with self.assertRaises(ValueError):
            pc.block_bootstrap_mean(d, block_len=0, n_boot=10, seed=0, kind="moving")
        with self.assertRaises(ValueError):
            pc.block_bootstrap_mean(d, block_len=len(d) + 1, n_boot=10, seed=0, kind="moving")
        with self.assertRaises(ValueError):
            pc.block_bootstrap_mean(d, block_len=5, n_boot=0, seed=0, kind="moving")
        with self.assertRaises(ValueError):
            pc.block_bootstrap_mean(d, block_len=5, n_boot=10, seed=0.5, kind="moving")
        with self.assertRaises(TypeError):
            pc.block_bootstrap_mean(d, block_len=5, n_boot=10, kind="moving")           # seed has no default
        with self.assertRaises(ValueError):
            pc.newey_west_mean(d, lags="auto")
        with self.assertRaises(ValueError):
            pc.compare_arms(panel(T=30), {"x": ("y", "pa", "pb")}, ts_col="ts", loss="mae", lags=2, block_len=3, n_boot=10, seed=0,
                            kind="moving", min_overlap=5, clamp="none")

    def test_auto_rules_resolve(self):
        self.assertEqual(pc.block_bootstrap_mean(self.d, block_len="auto_nw", n_boot=20, seed=0, kind="stationary").block_len,
                         max(1, pc.nw_default_lag(len(self.d))))
        self.assertGreaterEqual(pc.block_bootstrap_mean(self.d, block_len="auto_andrews", n_boot=20, seed=0, kind="moving").block_len, 1)


class TestCompareArms(unittest.TestCase):
    KW = dict(ts_col="ts", loss="sq", lags="nw_default", block_len="auto_nw", n_boot=400, seed=2, kind="stationary",
              min_overlap=20, clamp="conservative")

    def test_detects_worse_arm_and_columns(self):
        df = panel(T=300, coins=10, seed=3, shift=1.0)          # arm A has a +1 bias: higher squared loss
        res = pc.compare_arms(df, {"close": ("y", "pa", "pb"), "same": ("y", "pb", "pb")}, **self.KW)
        t = res.table
        self.assertGreater(t.loc["close", "mean_d"], 0.5)
        self.assertLess(t.loc["close", "nw_p"], 0.01)
        self.assertLess(t.loc["close", "boot_p"], 0.01)
        self.assertGreater(t.loc["close", "boot_lo"], 0.0)
        self.assertEqual(t.loc["same", "mean_d"], 0.0)
        for c in ("n_timestamps", "mean_d", "nw_t", "nw_p", "boot_lo", "boot_hi", "boot_p", "deff", "n_eff"):
            self.assertIn(c, t.columns)
        self.assertEqual(int(t.loc["close", "n_timestamps"]), 300)
        self.assertIn("close", pc.format_report(res))

    def test_precomputed_losses(self):
        df = panel(T=60, coins=3, seed=4)
        df["la"], df["lb"] = (df.pa - df.y).abs(), (df.pb - df.y).abs()
        a = pc.compare_arms(df, {"x": ("la", "lb")}, **{**self.KW, "loss": "precomputed"}).table
        b = pc.compare_arms(df, {"x": ("y", "pa", "pb")}, **{**self.KW, "loss": "abs"}).table
        pd.testing.assert_frame_equal(a, b)
        with self.assertRaises(ValueError):
            pc.compare_arms(df, {"x": ("la", "lb")}, **self.KW)         # 2-tuple with loss="sq"

    def test_few_timestamps_raise(self):
        with self.assertRaises(ValueError):
            pc.compare_arms(panel(T=10), {"x": ("y", "pa", "pb")}, **self.KW)

    def test_determinism(self):
        df = panel(T=80, seed=5)
        a = pc.compare_arms(df, {"x": ("y", "pa", "pb")}, **self.KW).table
        b = pc.compare_arms(df, {"x": ("y", "pa", "pb")}, **self.KW).table
        pd.testing.assert_frame_equal(a, b)

    def test_cli(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "p.csv")
            panel(T=60, seed=6).to_csv(p, index=False)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = pc.main([p, "--ts-col", "ts", "--target", "x:y:pa:pb", "--loss", "abs", "--lags", "nw_default", "--block-len", "4",
                              "--n-boot", "200", "--seed", "1", "--kind", "moving", "--min-overlap", "20", "--clamp", "none"])
        self.assertEqual(rc, 0)
        self.assertIn("mean_d", buf.getvalue())


if __name__ == "__main__":
    unittest.main()

"""E-normprobe-1 transforms (research/norm_probe/run_norm_probe.py): V0 pinned to the real process_windows, train-only
fit, no look-ahead, clip, flat zero-volume exclusion. Synthetic data only, CPU, seconds:
    python -m unittest tests.test_norm_probe -v
"""
import importlib.util
import os
import unittest

import numpy as np
import pandas as pd

from tests import test_audit_round2 as r2

_p = os.path.join(os.path.dirname(__file__), "..", "research", "norm_probe", "run_norm_probe.py")
_spec = importlib.util.spec_from_file_location("run_norm_probe", _p)
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

_CACHE = {}


def _built(n_coins=2):
    if n_coins not in _CACHE:
        panel = rp.synthetic_panel(n_coins, seed=1)
        smp = rp.build_samples(panel)
        _CACHE[n_coins] = (panel, smp, rp.make_inputs(smp))
    return _CACHE[n_coins]


class V0PinnedToPipeline(unittest.TestCase):
    def test_v0_equals_process_windows(self):
        _, smp, _ = _built()
        ns = r2._ns()
        idx = np.arange(0, len(smp["Cw"]), 997)[:12]
        H, L, C = smp["Hw"][idx], smp["Lw"][idx], smp["Cw"][idx]
        raw = np.stack([H, L, C], axis=2)
        centers, scales = zip(*[ns["calc_scale_params"](c, "robust") for c in C])
        real = ns["process_windows"](raw.copy(), ["high", "low", "close"], np.array(centers, "float32"),
                                     np.array(scales, "float32"), "robust", ns["CONFIG"])
        np.testing.assert_allclose(rp.v0_current(H, L, C), real, atol=1e-4)

    def test_v0_constant_window_is_zero(self):
        c = np.full((1, rp.T), 100.0)
        self.assertTrue((rp.v0_current(c, c, c) == 0).all())


class FitAndLookahead(unittest.TestCase):
    def test_train_bars_only_up_to_train_end_and_stats_ignore_val(self):
        panel, smp, (X, tf) = _built()
        panel2 = {k: v.copy() for k, v in panel.items()}
        for d in panel2.values():                                  # distort everything after the train end
            m = d.index > rp.TRAIN_END
            d.loc[m, ["open", "high", "low", "close"]] *= 3.0 * np.exp(np.arange(m.sum()) * 1e-4)[:, None]
        smp2 = rp.build_samples(panel2)
        np.testing.assert_array_equal(smp["train_bars"], smp2["train_bars"])
        _, tf2 = rp.make_inputs(smp2)
        np.testing.assert_array_equal(tf["V1"].lo, tf2["V1"].lo)
        np.testing.assert_array_equal(tf["V1"].hi, tf2["V1"].hi)
        np.testing.assert_array_equal(tf["V2"].mu, tf2["V2"].mu)
        np.testing.assert_array_equal(tf["V2"].sd, tf2["V2"].sd)
        np.testing.assert_array_equal(tf["V5"].q, tf2["V5"].q)

    def test_no_lookahead_future_bars_do_not_change_inputs(self):
        panel, smp, (X, _) = _built()
        cut = pd.Timestamp("2024-06-01", tz="UTC")
        panel2 = {k: v.copy() for k, v in panel.items()}
        for d in panel2.values():
            m = d.index > cut
            d.loc[m, ["open", "high", "low", "close"]] *= 7.0
        smp2 = rp.build_samples(panel2)
        X2, _ = rp.make_inputs(smp2, train_bars=smp["train_bars"])   # same fitted stats: only the inputs may differ
        times = pd.date_range(rp.GRID_START, rp.END, freq="1h")
        t_end = times[smp["e"]]
        key = {(c, e): i for i, (c, e) in enumerate(zip(smp["coin"], smp["e"]))}
        key2 = {(c, e): i for i, (c, e) in enumerate(zip(smp2["coin"], smp2["e"]))}
        checked = 0
        for (c, e), i in key.items():
            if t_end[i] <= cut and (c, e) in key2:
                j = key2[(c, e)]
                for v in rp.VARIANTS:
                    np.testing.assert_allclose(X[v][i], X2[v][j], rtol=0, atol=0)
                checked += 1
        self.assertGreater(checked, 1000)

    def test_sigma_is_strictly_past(self):
        rc = np.random.default_rng(0).normal(0, 0.01, 100)
        s = rp.trailing_sigma(rc)
        rc2 = rc.copy()
        rc2[60] = 5.0
        s2 = rp.trailing_sigma(rc2)
        np.testing.assert_array_equal(s[:61], s2[:61])             # bar 60's own return does not touch sigma[60]
        self.assertFalse(np.array_equal(s[61], s2[61]))


class ClipAndRanges(unittest.TestCase):
    def test_clip_bounds_and_standardization(self):
        _, smp, (X, tf) = _built()
        for v in ("V3", "V4", "V0"):
            self.assertLessEqual(np.abs(X[v]).max(), rp.CLIP + 1e-12)
        tr = smp["train_bars"]
        z = tf["V2"].transform(tr[None, :, :])[0]
        np.testing.assert_allclose(z.mean(0), 0, atol=1e-9)
        np.testing.assert_allclose(z.std(0), 1, atol=1e-9)
        m = tf["V1"].transform(tr[None, :, :])[0]
        np.testing.assert_allclose([m.min(), m.max()], [0, 1], atol=1e-12)
        q = tf["V5"].transform(tr[None, :, :])[0]
        self.assertTrue(q.min() >= 0 and q.max() <= 1)

    def test_minmax_extrapolates_outside_unit_interval(self):
        mm = rp.MinMax().fit(np.array([[-1.0, -1, -1], [1, 1, 1]]))
        out = mm.transform(np.array([[[2.0, 0, -3]]]))
        self.assertGreater(out[0, 0, 0], 1)
        self.assertLess(out[0, 0, 2], 0)


class FlatRunExclusion(unittest.TestCase):
    def test_flat_zero_volume_mask_threshold(self):
        n = 600
        d = pd.DataFrame({"high": np.ones(n), "low": np.ones(n), "volume": np.zeros(n)})
        d.loc[:100, "volume"] = 5.0            # bars 101..599 flat zero-volume
        d.loc[200:250, "low"] = 0.5            # break the run: 101..199 (99 bars) and 251..599 (349 bars)
        m = rp.flat_zero_volume_mask(d)
        self.assertFalse(m[:200].any())        # 99 bars < 168 -> not masked
        self.assertFalse(m[200:251].any())
        self.assertTrue(m[251:].all())         # 349 >= 168 -> masked
        d.loc[251:420, "volume"] = 1.0         # leaves 421..599 = 179 bars >= 168
        self.assertTrue(rp.flat_zero_volume_mask(d)[421:].all())
        d.loc[421:431, "volume"] = 1.0         # leaves 432..599 = 168 bars: exactly the threshold -> masked
        self.assertTrue(rp.flat_zero_volume_mask(d)[432:].all())
        d.loc[432:433, "volume"] = 1.0         # 166 bars -> not masked
        self.assertFalse(rp.flat_zero_volume_mask(d).any())

    def test_windows_touching_a_long_flat_run_are_dropped(self):
        panel, smp, _ = _built()
        p2 = {k: v.copy() for k, v in panel.items()}
        d = p2["S0"]
        lo, hi = 20000, 20200                                       # 200 flat zero-volume bars
        d.iloc[lo:hi, d.columns.get_loc("volume")] = 0.0
        d.iloc[lo:hi, d.columns.get_loc("high")] = d.iloc[lo, d.columns.get_loc("close")]
        d.iloc[lo:hi, d.columns.get_loc("low")] = d.iloc[lo, d.columns.get_loc("close")]
        s2 = rp.build_samples(p2)
        bad = (s2["coin"] == 0) & (s2["e"] >= lo - 1) & (s2["e"] - rp.NEED_BACK <= hi)
        self.assertFalse(bad.any())
        ok = (smp["coin"] == 0) & (smp["e"] >= lo - 1) & (smp["e"] - rp.NEED_BACK <= hi)
        self.assertTrue(ok.any())                                   # they existed before


class Metrics(unittest.TestCase):
    def test_eff_rank_bounds_and_nn_exclusion(self):
        rng = np.random.default_rng(0)
        iso = rng.normal(size=(5000, 20))
        self.assertGreater(rp.eff_rank_ratio(iso)[0], 0.9)
        low = rng.normal(size=(5000, 1)) @ rng.normal(size=(1, 20)) + 1e-3 * rng.normal(size=(5000, 20))
        self.assertLess(rp.eff_rank_ratio(low)[0], 0.1)
        X = rng.normal(size=(300, 5))
        X[1] = X[0]
        z = np.zeros(300, int)
        near = rp.nn_distances(X, z, np.arange(300), n_sub=300, excl=48)         # adjacent same-coin windows: excluded
        self.assertGreater(near[0], 1e-3)
        far = rp.nn_distances(X, z, np.arange(300) * 100, n_sub=300, excl=48)    # far apart in time: a real duplicate
        self.assertLess(far[0], 1e-9)


if __name__ == "__main__":
    unittest.main()

"""price_norm_mode in {'logret_std', 'logret_sigma'}: OHLC inputs as log returns vs the previous close (train-only
standardization / strictly-past sigma), targets log(price / last_close) in the same space, one inverse for evaluation.
The default 'window_scale' must stay byte-identical. Cells run the real pipeline on tiny synthetic data, no Drive:
    python -m unittest tests.test_price_norm_mode -v
"""
import hashlib
import os
import subprocess
import sys
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from tests import _simd
from tests import test_audit_round2 as r2

# sha256 over sorted X_*/y_*/base_params/last_candles bytes of the fixture below, measured on master BEFORE this change.
# Byte-exact per numpy SIMD path (tests/_simd.py): the synthetic fixture and the volume channel use float64 exp / float32
# log1p, whose last bits differ between X86_V4 (AVX512) and X86_V3 (AVX2) -- GitHub runners are a lottery of both. The V4
# value is the original pin (unchanged); the V3 value is the SAME code with numpy's X86_V4 path disabled.
DEFAULT_FINGERPRINTS = {
    "x86_v4": "baf6195e0b2ed1ad20b3074af5d17c858b7c3a80d7c5d37135a9bee1322977b3",
    "x86_v3": "3132e0dc088c143d74b5698c524627b69a94e938574547e2ab5bb6a8f7ab0a71",
}
DEFAULT_FINGERPRINT = DEFAULT_FINGERPRINTS["x86_v4"]
OHLC = ("open", "high", "low", "close")
SPLITS = {"train_end": "2025-02-01", "val_end": "2025-02-15"}


def _frames(seed0=11, end="2025-03-01"):
    """Two synthetic coins; open is jittered off the previous close (the shared fixture has open == previous close,
    a constant zero log-return channel)."""
    out = {}
    for k, name in enumerate(("AAAUSDT", "BBBUSDT")):
        f = r2._ohlcv("2025-01-01", end, seed0 + k)
        f["open"] = f["open"] * (1 + np.random.default_rng(100 + k).normal(0, 0.002, len(f)))
        out[name] = f
    return out


def _cfg(mode=None, with_ohlc=True, **over):
    cfg = r2._cfg(split_dates=dict(SPLITS), **over)
    if mode is not None:
        cfg["price_norm_mode"] = mode
    if with_ohlc:
        cfg["exclude_from_features"] = [c for c in cfg["exclude_from_features"] if c not in ("open", "high", "low")]
    return cfg


def _build(cfg, frames=None):
    return r2._build(cfg, frames or _frames())


def _prepare(cfg):
    """prepare_single_asset directly (errors are not swallowed by the per-asset worker of build_dataset)."""
    ns = r2._ns()
    dfs = r2._quiet(ns["resample_timeframes"], r2._ohlcv("2025-01-01", "2025-03-01", 11), ["1h"], config=cfg)
    return r2._quiet(ns["prepare_single_asset"], dfs, config=cfg)


def _split(ds, cfg):
    return r2._quiet(r2._ns()["split_data"], ds, config=cfg)


def _channels(ds):
    fo = list(ds["feature_order"])
    return [fo.index(c) for c in OHLC if c in fo]


def _perturb(frames, t):
    out = {}
    for k, f in frames.items():
        f = f.copy()
        rng = np.random.default_rng(5)
        m = f.index >= t
        mult = np.exp(np.cumsum(rng.normal(0, 0.02, int(m.sum()))))
        for c in ("open", "high", "low", "close"):
            f.loc[m, c] = f.loc[m, c].to_numpy() * mult
        out[k] = f
    return out


def _by_ts(ds, key="X_1h"):
    ts = ds["last_candles"][:, 3]
    return {int(t): i for i, t in enumerate(ts)}


def _default_dataset():
    return _build(r2._cfg(), {"AAAUSDT": r2._ohlcv("2025-01-01", "2025-03-01", 11),
                              "BBBUSDT": r2._ohlcv("2025-01-01", "2025-03-01", 12)})


def _fingerprint(ds):
    h = hashlib.sha256()
    for k in sorted(k for k in ds if k.startswith(("X_", "y_")) or k in ("base_params", "last_candles")):
        h.update(k.encode())
        h.update(np.ascontiguousarray(ds[k]).tobytes())
    return h.hexdigest()


class DefaultUnchangedTests(unittest.TestCase):
    def test_x86_v3_pin_holds_on_any_x86_machine(self):
        """Regression (flaky CI: the pin was only the AVX512 value): a fresh interpreter with numpy's X86_V4 path disabled
        must give the X86_V3 pin, whatever CPU the runner has. Skipped off x86."""
        if _simd.simd_path() is None:
            self.skipTest("not an x86 CPU with X86_V3")
        env = dict(os.environ, NPY_DISABLE_CPU_FEATURES="X86_V4")
        code = "from tests.test_price_norm_mode import _default_dataset, _fingerprint; print(_fingerprint(_default_dataset()))"
        out = subprocess.run([sys.executable, "-c", code], cwd=r2.ROOT, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.split()[-1], DEFAULT_FINGERPRINTS["x86_v3"])

    def test_default_fingerprint_and_no_new_keys(self):
        ds = _default_dataset()
        h = _fingerprint(ds)
        pin = _simd.pick(DEFAULT_FINGERPRINTS)
        if pin is None:
            self.skipTest("no recorded fingerprint for this CPU (neither X86_V4 nor X86_V3)")
        self.assertEqual(h, pin)
        self.assertNotIn("price_norm_meta", ds)
        self.assertNotIn("price_norm_state", ds)
        self.assertNotIn("price_norm_mode", ds)

    def test_default_split_has_no_meta(self):
        cfg = _cfg()
        tr, va, te = _split(_build(cfg), cfg)
        self.assertNotIn("price_norm_meta", tr)

    def test_unknown_mode_raises(self):
        ns = r2._ns()
        with self.assertRaises(ValueError):
            ns["price_norm_mode"](dict(ns["CONFIG"], price_norm_mode="logret_zscore"))
        with self.assertRaises(ValueError):
            _prepare(_cfg("logret_zscore"))

    def test_new_modes_listed(self):
        ns = r2._ns()
        for m in ("window_scale", "pct_change", "logret_std", "logret_sigma"):
            self.assertEqual(ns["price_norm_mode"](dict(ns["CONFIG"], price_norm_mode=m)), m)


class NewModeInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _cfg("logret_std")
        cls.frames = _frames()
        cls.ds = _build(cls.cfg, cls.frames)

    def test_dataset_is_marked_unfitted_and_channels_are_raw_logret(self):
        ds = self.ds
        self.assertEqual(ds["price_norm_mode"], "logret_std")
        self.assertEqual(ds["price_norm_state"], "unfitted")
        self.assertEqual(len(_channels(ds)), 4)
        # expected value for one sample/asset from the raw frame
        f = self.frames["AAAUSDT"]
        ts = pd.Timestamp(int(ds["last_candles"][0, 3]), tz="UTC")
        pos = f.index.get_loc(ts)
        j = list(ds["feature_order"]).index("high")
        expect = np.log(f["high"].to_numpy()[pos - 31:pos + 1] / f["close"].to_numpy()[pos - 32:pos])
        np.testing.assert_allclose(ds["X_1h"][0, :, j], expect, rtol=1e-4, atol=1e-7)

    def test_non_price_columns_equal_default_mode(self):
        cfg0 = _cfg(None)
        d0 = _build(cfg0, self.frames)
        i0, i1 = _by_ts(d0), _by_ts(self.ds)
        common = sorted(set(i0) & set(i1))
        self.assertGreater(len(common), 100)
        fo = list(self.ds["feature_order"])
        keep = [j for j, n in enumerate(fo) if n not in OHLC]
        a = d0["X_1h"][[i0[t] for t in common]][:, :, keep]
        b = self.ds["X_1h"][[i1[t] for t in common]][:, :, keep]
        np.testing.assert_array_equal(a, b)

    def test_split_fits_on_train_only_and_standardizes(self):
        tr, va, te = _split(self.ds, self.cfg)
        meta = tr["price_norm_meta"]
        self.assertEqual(meta["mode"], "logret_std")
        st = meta["inputs"]["1h"]
        j = _channels(self.ds)
        x = tr["X_1h"][:, :, j].astype("float64")
        # train channels: mean ~ 0, std ~ 1 (clip 5 trims a hair)
        np.testing.assert_allclose(x.mean(axis=(0, 1)), 0.0, atol=0.05)
        np.testing.assert_allclose(x.std(axis=(0, 1)), 1.0, atol=0.05)
        self.assertLessEqual(np.abs(tr["X_1h"][:, :, j]).max(), 5.0 + 1e-6)
        for s in (va, te["AAAUSDT"]):
            self.assertEqual(s["price_norm_meta"]["inputs"]["1h"]["mean"], st["mean"])
        self.assertEqual(set(meta["targets"]), {"high", "low", "close"})
        self.assertEqual(self.ds["price_norm_state"], "unfitted")     # source dataset not mutated

    def test_perturbing_val_test_bars_keeps_stats(self):
        tr, _, _ = _split(self.ds, self.cfg)
        pert = _perturb(self.frames, "2025-02-15 00:00")
        ds2 = _build(self.cfg, pert)
        tr2, _, _ = _split(ds2, self.cfg)
        self.assertEqual(tr["price_norm_meta"], tr2["price_norm_meta"])
        np.testing.assert_array_equal(tr["X_1h"], tr2["X_1h"])

    def test_split_twice_is_idempotent(self):
        a = _split(self.ds, self.cfg)[0]
        b = _split(self.ds, self.cfg)[0]
        np.testing.assert_array_equal(a["X_1h"], b["X_1h"])

    def test_rolling_splits_fit_per_fold(self):
        ns = r2._ns()
        cfg = dict(self.cfg, rolling_train_span="14D")
        folds = r2._quiet(ns["rolling_splits"], self.ds, test_span="5D", val_span="5D", train_span="14D",
                          initial_train_span="14D", step="5D", max_windows=2, config=cfg, verbose=False)
        self.assertGreaterEqual(len(folds), 1)
        for tr, va, te in folds:
            self.assertEqual(tr["price_norm_meta"]["mode"], "logret_std")


class NoLookAheadTests(unittest.TestCase):
    def _compare(self, mode):
        cfg = _cfg(mode, window_sizes={"1h": 32}) if mode == "logret_sigma" else _cfg(mode)
        frames = _frames()
        t_cut = pd.Timestamp("2025-02-10 00:00", tz="UTC")
        pert = _perturb(frames, t_cut)
        if mode == "logret_sigma":
            cfg = dict(cfg, targets=["high", "low", "close"], enabled_heads={f"{t}_reg": False for t in ("high", "low", "close")})
            cfg2 = deepcopy(cfg)
        else:
            cfg2 = deepcopy(cfg)
        d1, d2 = _build(cfg, frames), _build(cfg2, pert)
        i1, i2 = _by_ts(d1), _by_ts(d2)
        j = _channels(d1)
        # window end strictly before the first changed bar and before its 1-bar target
        ends = [t for t in sorted(set(i1) & set(i2)) if t < (t_cut - pd.Timedelta("2h")).value]
        self.assertGreater(len(ends), 50)
        a = d1["X_1h"][[i1[t] for t in ends]][:, :, j]
        b = d2["X_1h"][[i2[t] for t in ends]][:, :, j]
        np.testing.assert_array_equal(a, b)
        # a window whose last bar is the first changed bar differs only through its own newest bars
        late = [t for t in sorted(set(i1) & set(i2)) if t >= t_cut.value]
        self.assertTrue(late)

    def test_logret_std_inputs_do_not_see_later_bars(self):
        self._compare("logret_std")

    def test_logret_sigma_inputs_and_sigma_do_not_see_later_bars(self):
        self._compare("logret_sigma")

    def test_trailing_sigma_is_past_only(self):
        ns = r2._ns()
        rng = np.random.default_rng(1)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        s1 = ns["trailing_close_sigma"](c)
        c2 = c.copy()
        c2[120:] *= 1.5
        s2 = ns["trailing_close_sigma"](c2)
        # sigma[t] uses returns of bars t-14..t-1, so it is unchanged for t <= 120 (return of bar 120 enters at t = 121)
        np.testing.assert_array_equal(s1[:121], s2[:121])
        self.assertTrue(np.isnan(s1[:14]).all())
        self.assertTrue(np.isfinite(s1[15:]).all())
        self.assertGreaterEqual(np.nanmin(s1), 1e-4)


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _cfg("logret_std")
        cls.ds = _build(cls.cfg)
        cls.tr, cls.va, cls.te = _split(cls.ds, cls.cfg)
        cls.meta = cls.tr["price_norm_meta"]

    def test_target_is_standardized_log_return_vs_last_close(self):
        ns = r2._ns()
        lc = self.tr["last_candles"]
        for t, col in (("high", 6), ("low", 5), ("close", 4)):
            y = self.tr["y"][f"y_{t}_reg"].astype("float64")
            raw = np.log(lc[:, col] / lc[:, 2])
            m, s = self.meta["targets"][t]["mean"], self.meta["targets"][t]["std"]
            np.testing.assert_allclose(y, (np.clip(raw, -1, 1) - m) / s, atol=2e-5)
            self.assertAlmostEqual(float(y.mean()), 0.0, delta=0.02)
            self.assertAlmostEqual(float(y.std()), 1.0, delta=0.02)

    def test_round_trip_is_exact(self):
        ns = r2._ns()
        lc = self.tr["last_candles"]
        rng = np.random.default_rng(0)
        for t, col in (("high", 6), ("low", 5), ("close", 4)):
            price = lc[:, col] * (1 + rng.normal(0, 0.01, len(lc)))
            z = ns["normalize_price_targets"](price, self.meta, lc[:, 2], head=t)
            back = ns["denormalize_price_targets"](z, self.meta, lc[:, 2], head=t)
            np.testing.assert_allclose(back, price, rtol=0, atol=1e-9 * np.max(price))
            lr = ns["denormalize_price_targets"](z, self.meta, lc[:, 2], head=t, output="logret")
            np.testing.assert_allclose(lr, np.log(price / lc[:, 2]), atol=1e-9)

    def test_denormalize_dataset_targets_recovers_future_prices(self):
        ns = r2._ns()
        lc = self.va["last_candles"]
        for t, col in (("high", 6), ("low", 5), ("close", 4)):
            y = self.va["y"][f"y_{t}_reg"]
            price = ns["denormalize_price_targets"](y, self.va["price_norm_meta"], lc[:, 2], head=t)
            np.testing.assert_allclose(price, lc[:, col], rtol=2e-5)

    def test_unknown_head_and_output_raise(self):
        ns = r2._ns()
        with self.assertRaises(ValueError):
            ns["denormalize_price_targets"](np.zeros(2), self.meta, np.ones(2), head="mid")
        with self.assertRaises(ValueError):
            ns["denormalize_price_targets"](np.zeros(2), self.meta, np.ones(2), head="high", output="bps")

    def test_class_targets_unchanged_vs_default_mode(self):
        d0 = _build(_cfg(None))
        i0, i1 = _by_ts(d0), _by_ts(self.ds)
        common = sorted(set(i0) & set(i1))
        for t in ("high", "low", "close"):
            np.testing.assert_array_equal(d0[f"y_{t}_class"][[i0[x] for x in common]],
                                          self.ds[f"y_{t}_class"][[i1[x] for x in common]])

    def test_unsupported_combinations_raise(self):
        with self.assertRaisesRegex(ValueError, "per-sample sigma"):
            _prepare(_cfg("logret_sigma"))                         # regression heads need per-sample sigma
        with self.assertRaisesRegex(ValueError, "reg_target_mode"):
            _prepare(_cfg("logret_std", reg_target_mode="window_scale"))
        with self.assertRaisesRegex(ValueError, "reg_target_scale"):
            _prepare(_cfg("logret_std", reg_target_scale=100.0))

    def test_retarget_and_invert_refuse_new_mode(self):
        ns = r2._ns()
        with self.assertRaises(ValueError):
            ns["invert_reg_predictions"](np.zeros(3), "high_reg", last_candles=self.tr["last_candles"][:3],
                                         config=self.cfg)


    def test_invert_reg_predictions_with_meta_matches_public_inverse(self):
        ns = r2._ns()
        lc = self.va["last_candles"]
        y = self.va["y"]["y_high_reg"]
        a = ns["invert_reg_predictions"](y, "high_reg", last_candles=lc, config=self.cfg,
                                         price_norm_meta=self.va["price_norm_meta"])
        np.testing.assert_allclose(a, lc[:, 6], rtol=2e-5)

    def test_cross_sectional_and_live_refuse_new_mode(self):
        ns = r2._ns()
        with self.assertRaises(ValueError):
            ns["cross_sectional_normalize"](self.ds, config=self.cfg, verbose=False)
        with self.assertRaises(ValueError):
            ns["build_dataset_live"](["BTCUSDT"], config=self.cfg)

    def test_holdout_needs_train_meta(self):
        ns = r2._ns()
        cfg = dict(self.cfg, holdout_start="2025-02-20", open_holdout=True)
        with self.assertRaisesRegex(ValueError, "price_norm_meta"):
            r2._quiet(ns["split_holdout"], self.ds, config=cfg)
        out = r2._quiet(ns["split_holdout"], self.ds, config=cfg, price_norm_meta=self.meta)
        sp = next(iter(out.values()))
        self.assertEqual(sp["price_norm_meta"], self.meta)

    def test_sigma_meta_has_no_targets_and_denormalize_raises(self):
        ns = r2._ns()
        cfg = _cfg("logret_sigma", enabled_heads={f"{t}_reg": False for t in ("high", "low", "close")})
        ds = _build(cfg)
        tr, va, te = _split(ds, cfg)
        self.assertEqual(tr["price_norm_meta"]["mode"], "logret_sigma")
        with self.assertRaises(ValueError):
            ns["denormalize_price_targets"](np.zeros(2), tr["price_norm_meta"], np.ones(2), head="high")
        self.assertLessEqual(np.abs(tr["X_1h"][:, :, _channels(ds)]).max(), 5.0 + 1e-6)


class EvaluationWiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _cfg("logret_std")
        cls.ds = _build(cls.cfg)
        cls.tr, cls.va, cls.te = _split(cls.ds, cls.cfg)
        cls.ev = r2._nbload.load_evaluation()

    def _specs(self, split, direct=False):
        ev = self.ev
        out = []
        for s in ev["DEFAULT_PRICE_TARGETS"]:
            if direct:
                out.append(__import__("dataclasses").replace(s, relative_to_entry=True, price_index=2, entry_col=2, reg_scale=1.0))
            else:
                st = split["price_norm_meta"]["targets"][s.name]
                out.append(s.with_log_norm(st["mean"], st["std"]))
        return out

    def _run(self, specs, raw, split):
        ev = self.ev
        dec = ev["decode_predictions_v4"](raw, specs, split["base_params"], split["last_candles"])
        y_true = {f"y_{s.name}": np.asarray(split["y"][f"y_{s.name}_reg"]) for s in specs}
        return dec, ev["evaluate_predictions_v4"](dec, specs, y_true=y_true, base_params=split["base_params"])

    def test_perfect_predictor_metrics_match_price_space(self):
        sp = self.va
        lc = sp["last_candles"]
        raw = {t: np.asarray(sp["y"][f"y_{t}_reg"], dtype="float64") for t in ("high", "low", "close")}
        dec, met = self._run(self._specs(sp), raw, sp)
        # the same perfect predictor written directly in price space (return vs last close)
        raw_d = {t: lc[:, c] / lc[:, 2] - 1.0 for t, c in (("high", 6), ("low", 5), ("close", 4))}
        dec_d, met_d = self._run(self._specs(sp, direct=True), raw_d, sp)
        for t in ("high", "low", "close"):
            np.testing.assert_allclose(dec[t]["pred_real"], dec_d[t]["pred_real"], rtol=3e-5)
            np.testing.assert_allclose(dec[t]["pred_real"], lc[:, {"high": 6, "low": 5, "close": 4}[t]], rtol=3e-5)
            self.assertLess(np.nanmax(met[t]["error_pct"]), 1e-2)
            for k in ("win_rate", "n_valid"):
                self.assertAlmostEqual(float(met[t][k]), float(met_d[t][k]), places=6)
            self.assertAlmostEqual(float(met[t]["correlation"]), float(met_d[t]["correlation"]), places=4)

    def test_imperfect_predictor_equals_manual_price_formula(self):
        sp = self.va
        lc = sp["last_candles"]
        rng = np.random.default_rng(3)
        raw = {t: np.asarray(sp["y"][f"y_{t}_reg"], dtype="float64") + rng.normal(0, 0.5, len(lc))
               for t in ("high", "low", "close")}
        dec, met = self._run(self._specs(sp), raw, sp)
        for t, col in (("high", 6), ("low", 5), ("close", 4)):
            m, s = sp["price_norm_meta"]["targets"][t]["mean"], sp["price_norm_meta"]["targets"][t]["std"]
            manual = lc[:, 2] * np.exp(raw[t] * s + m)
            np.testing.assert_allclose(dec[t]["pred_real"], manual, rtol=1e-12)
            np.testing.assert_allclose(met[t]["error_abs"], np.abs(lc[:, col] - manual), rtol=1e-9)

    def test_public_inverse_matches_decode(self):
        ns = r2._ns()
        sp = self.va
        lc = sp["last_candles"]
        raw = {t: np.asarray(sp["y"][f"y_{t}_reg"], dtype="float64") * 0.7 for t in ("high", "low", "close")}
        dec, _ = self._run(self._specs(sp), raw, sp)
        for t in raw:
            np.testing.assert_allclose(dec[t]["pred_real"],
                                       ns["denormalize_price_targets"](raw[t], sp["price_norm_meta"], lc[:, 2], head=t),
                                       rtol=1e-12)

    def test_default_specs_unchanged(self):
        for s in self.ev["DEFAULT_PRICE_TARGETS"]:
            self.assertIsNone(getattr(s, "log_norm", None))
            self.assertNotIn("log_norm", s.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()

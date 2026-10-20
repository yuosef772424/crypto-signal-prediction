"""One registered Normalizer interface behind every price_norm mode ('window_scale', 'logret_std', 'logret_sigma').
Pins (sha256 fingerprints measured on the N1 head BEFORE the refactor) that dataset + split outputs are byte-identical,
that fitting sees train only, that state is a plain serializable dict, that targets round-trip, and that evaluation
decode and the Normalizer share one inverse formula:
    python -m unittest tests.test_normalizer -v
"""
import json
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from tests import _simd
from tests import test_audit_round2 as r2
from tests._normalizer_fixture import NO_REG, fingerprint_dataset, fingerprint_splits, mode_cfg, run_mode
from tests.test_price_norm_mode import _build, _by_ts, _cfg, _frames, _perturb, _split

MODES = ("window_scale", "logret_std", "logret_sigma")
# sha256 (dataset, splits) per mode, measured on the N1 head (f4b74a1) before the refactor.
PINNED = {
    'x86_v4': {
        "window_scale": ("1312b17794abb23c88e33fb31b7742c684ac6f8439c8bad14b0e6c61f5b56519",
                         "41c2365d2a9f9ad88e37f861af7bca6cd64b2c5617c8180480f6d975b984cfba"),
        "pct_change": ("090e33b6593f3dd3ec3074a06444239b74de6c2b3bac49cb5f1883886cad1b88",
                       "246d0a8a700ef9eabbd5b84787aa816d799a0f8c94cacbbcda439fa8af5ce429"),
        "logret_std": ("4dd2d7fba1f81abc1cf7b6d3537aa4f616643f8cf274bf9922efbde6126e0b6d",
                       "d0a7cb0dff6d056c108d41897a17636fe91393ab9290c0fcef754b76ed02e910"),
        "logret_sigma": ("2975988066bfef26346dd2ce03b071e7de514f5d3652396e8e60be387e51eba7",
                         "30cf24146ada5a2ff90bd1450ff4074cc28021f5e883da0c1ef8591b76b05a77"),
    },
    'x86_v3': {
        "window_scale": ("99377c7845af9a4345bcdbf4e24edb3e171ed8c199e8fae4a174d5679c57b434",
                         "cd53d8e2039ef9f7489bbae1fe0d001a46b9bbbe18076333fe4450a9982540ca"),
        "pct_change": ("451093fecb39ec81fe160c68c8e659a34093d8b26c7ab5a633ff7e395bab8f96",
                       "8404938e7671b991c847d9e4df58d8105b9323da3d85c8bd7c0bd894eeb8627c"),
        "logret_std": ("30e4f4dae22d8d8e39dd8eb33755163e5114201058d9dece1cd108ba351db12f",
                       "570f7e731c83ec3ed0c2bd7f118fa7ed8237d08292adfe0416d82a04d8804a2f"),
        "logret_sigma": ("53423c33381014fcfda07504e9eced169266ee5580606793812a7d682db29a6e",
                         "e4dc16d48225a42c52c0790ef495e50d351aa88e639b677d88c7603fbb0c1274"),
    },
}
_RUNS = {}


def _run(mode):
    if mode not in _RUNS:
        if mode == "pct_change":
            cfg = _cfg("pct_change", with_ohlc=False)
            ds = _build(cfg)
            _RUNS[mode] = (cfg, ds, _split(ds, cfg))
        else:
            _RUNS[mode] = run_mode(mode)
    return _RUNS[mode]


def _nz(mode):
    return r2._ns()["get_normalizer"](mode)


def _raw_train(ds, cfg):
    """The UNFITTED train split (the dataset with its mode mark dropped is split without any stamping)."""
    raw = dict(ds)
    raw.pop("price_norm_mode", None)
    raw.pop("price_norm_state", None)
    return _split(raw, cfg)[0]


def _fit(mode, ds, cfg):
    return _nz(mode).fit(_raw_train(ds, cfg), feature_order=ds["feature_order"], timeframes=ds["timeframes"])


class RegistryTests(unittest.TestCase):
    def test_registry_lists_exactly_three(self):
        self.assertEqual(set(r2._ns()["NORMALIZERS"]), set(MODES))

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            r2._ns()["get_normalizer"]("logret_zscore")
        with self.assertRaises(ValueError):
            r2._ns()["get_normalizer"]("pct_change")      # legacy window-level alias, resolved by normalizer_for_mode only

    def test_names_match_registry_keys(self):
        for k, nz in r2._ns()["NORMALIZERS"].items():
            self.assertEqual(nz.name, k)


class FingerprintTests(unittest.TestCase):
    """transform output byte-for-byte equal to the pre-refactor pipeline (dataset inputs and stamped splits)."""

    def test_pinned_fingerprints(self):
        pins = _simd.pick(PINNED)
        if pins is None:
            self.skipTest(f"no recorded pin for this CPU's numpy SIMD path ({_simd.simd_path()})")
        for mode, (d, s) in pins.items():
            with self.subTest(mode=mode):
                cfg, ds, sp = _run(mode)
                self.assertEqual(fingerprint_dataset(ds), d)
                self.assertEqual(fingerprint_splits(*sp), s)

    def test_transform_inputs_equals_stamped_split(self):
        cfg, ds, (tr, va, te) = _run("logret_std")
        nz = _nz("logret_std")
        state = tr["price_norm_meta"]
        i_ds, i_tr = _by_ts(ds), _by_ts(tr)
        common = sorted(set(i_ds) & set(i_tr))
        self.assertGreater(len(common), 50)
        raw = ds["X_1h"][[i_ds[t] for t in common]]
        out = nz.transform_inputs({"X_1h": raw}, state)["X_1h"]
        np.testing.assert_array_equal(out, tr["X_1h"][[i_tr[t] for t in common]])

    def test_transform_inputs_identity_for_window_level_modes(self):
        for mode in ("window_scale", "logret_sigma"):
            cfg, ds, _ = _run(mode)
            nz = _nz(mode)
            X = ds["X_1h"]
            np.testing.assert_array_equal(nz.transform_inputs({"X_1h": X}, nz.fit({}, feature_order=ds["feature_order"],
                                                                                   timeframes=ds["timeframes"]))["X_1h"], X)


class StateTests(unittest.TestCase):
    def test_state_is_plain_json_dict_and_stored(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                cfg, ds, (tr, va, te) = _run(mode)
                st = _fit(mode, ds, cfg)
                self.assertIsInstance(st, dict)
                self.assertEqual(json.loads(json.dumps(st)), st)
                self.assertEqual(st["mode"], mode)
                if mode != "window_scale":              # window_scale stores nothing: byte-identical default datasets
                    self.assertEqual(tr["price_norm_meta"]["mode"], mode)
                    self.assertEqual(json.loads(json.dumps(tr["price_norm_meta"])), tr["price_norm_meta"])
                else:
                    self.assertNotIn("price_norm_meta", tr)

    def test_roundtripped_state_gives_same_transform_and_targets(self):
        cfg, ds, (tr, va, te) = _run("logret_std")
        nz = _nz("logret_std")
        st = tr["price_norm_meta"]
        st2 = json.loads(json.dumps(st))
        raw = {"X_1h": ds["X_1h"][:40]}
        np.testing.assert_array_equal(nz.transform_inputs(raw, st)["X_1h"], nz.transform_inputs(raw, st2)["X_1h"])
        y = {"y_high_reg": ds["y_high_reg"][:40]}
        np.testing.assert_array_equal(nz.normalize_targets(y, st)["y_high_reg"], nz.normalize_targets(y, st2)["y_high_reg"])


class TrainOnlyFitTests(unittest.TestCase):
    def test_perturbing_val_test_leaves_state_identical(self):
        frames = _frames()
        pert = _perturb(frames, pd.Timestamp("2025-02-10 00:00", tz="UTC"))
        for mode in MODES:
            with self.subTest(mode=mode):
                cfg = mode_cfg(mode)
                d1, d2 = _build(cfg, frames), _build(deepcopy(cfg), pert)
                t1, _, _ = _split(d1, cfg)
                t2, _, _ = _split(d2, cfg)
                s1, s2 = _fit(mode, d1, cfg), _fit(mode, d2, cfg)
                self.assertEqual(s1, s2)
                if mode == "logret_std":
                    self.assertEqual(t1["price_norm_meta"], t2["price_norm_meta"])
                    self.assertEqual(t1["price_norm_meta"], s1)       # split_data stamps exactly what fit returns

    def test_perturbing_a_train_bar_changes_logret_std_state(self):
        frames = _frames()
        pert = _perturb(frames, pd.Timestamp("2025-01-20 00:00", tz="UTC"))
        cfg = mode_cfg("logret_std")
        t1, _, _ = _split(_build(cfg, frames), cfg)
        t2, _, _ = _split(_build(deepcopy(cfg), pert), cfg)
        self.assertNotEqual(t1["price_norm_meta"], t2["price_norm_meta"])


class TargetRoundTripTests(unittest.TestCase):
    def test_inverse_of_normalize_is_identity(self):
        rng = np.random.default_rng(7)
        lc = 100 * np.exp(rng.normal(0, 0.1, 64))
        price = lc * np.exp(rng.normal(0, 0.02, 64))
        for mode in MODES:
            with self.subTest(mode=mode):
                cfg, ds, (tr, va, te) = _run(mode)
                nz = _nz(mode)
                st = _fit(mode, ds, cfg)
                if mode == "logret_sigma":              # no regression heads: normalize/inverse raise, never guess
                    with self.assertRaises(ValueError):
                        nz.inverse_targets(np.zeros(3), st, np.ones(3), head="high")
                    continue
                z = nz.normalize_prices(price, st, lc, head="high")
                np.testing.assert_allclose(nz.inverse_targets(z, st, lc, head="high"), price, rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(nz.inverse_targets(z, st, lc, head="y_high_reg"), price, rtol=1e-9, atol=1e-9)

    def test_logret_output_logret_is_raw_target(self):
        cfg, ds, (tr, va, te) = _run("logret_std")
        nz = _nz("logret_std")
        st = tr["price_norm_meta"]
        raw = np.asarray(ds["y_high_reg"][:50], dtype="float64")
        z = nz.normalize_targets({"y_high_reg": ds["y_high_reg"][:50]}, st)["y_high_reg"]
        np.testing.assert_allclose(nz.inverse_targets(z, st, np.ones(50), head="high_reg", output="logret"), raw, atol=1e-6)


class OneFormulaTests(unittest.TestCase):
    def test_evaluation_decode_equals_normalizer_inverse(self):
        cfg, ds, (tr, va, te) = _run("logret_std")
        ev = r2._nbload.load_evaluation()
        nz = _nz("logret_std")
        sp = va
        lc = sp["last_candles"]
        rng = np.random.default_rng(3)
        raw = {t: np.asarray(sp["y"][f"y_{t}_reg"], dtype="float64") + rng.normal(0, 0.5, len(lc))
               for t in ("high", "low", "close")}
        specs = [s.with_log_norm(sp["price_norm_meta"]["targets"][s.name]["mean"], sp["price_norm_meta"]["targets"][s.name]["std"])
                 for s in ev["DEFAULT_PRICE_TARGETS"]]
        dec = ev["decode_predictions_v4"](raw, specs, sp["base_params"], lc)
        for t in raw:
            np.testing.assert_array_equal(dec[t]["pred_real"],
                                          nz.inverse_targets(raw[t], sp["price_norm_meta"], lc[:, 2], head=t))
            np.testing.assert_array_equal(
                nz.inverse_targets(raw[t], sp["price_norm_meta"], lc[:, 2], head=t),
                r2._ns()["denormalize_price_targets"](raw[t], sp["price_norm_meta"], lc[:, 2], head=t))

    def test_invert_reg_predictions_goes_through_normalizer(self):
        ns = r2._ns()
        cfg, ds, (tr, va, te) = _run("window_scale")
        lc = va["last_candles"]
        preds = np.asarray(va["y"]["y_high_reg"], dtype="float64")
        got = ns["invert_reg_predictions"](preds, "high_reg", last_candles=lc, config=cfg)
        nz = _nz("window_scale")
        st = nz.state_from_config(cfg)
        np.testing.assert_array_equal(got, nz.inverse_targets(preds, st, lc[:, 0], head="high"))


if __name__ == "__main__":
    unittest.main()

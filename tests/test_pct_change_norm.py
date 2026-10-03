"""price_norm_mode='pct_change': أعمدة price_level = تغيّر نسبي هندسي % عن الشمعة السابقة، وفكّه إلى الأسعار الحقيقية.

تُنفَّذ خلايا خط الأنابيب نفسها ببيانات تركيبية صغيرة بلا Drive:
    python -m unittest tests.test_pct_change_norm -v
"""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from tests import test_audit_round2 as r2


def _pw(cfg, w, cols, centers=None, scales=None):
    ns = r2._ns()
    n = w.shape[0]
    c = np.zeros(n, "float32") if centers is None else centers
    s = np.ones(n, "float32") if scales is None else scales
    return ns["process_windows"](w.copy(), cols, c, s, "robust", cfg)


def _walk(n_win=6, T=32, n_col=3, seed=0, sigma=0.01):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, sigma, (n_win, T, n_col)), axis=1))


class EncodeDecodeTests(unittest.TestCase):
    def test_example_values_and_roundtrip(self):
        ns = r2._ns()
        x = np.array([100.0, 99.0, 103.95, 103.5342])
        r = ns["pct_change_encode"](x)
        np.testing.assert_allclose(r, [0.0, -1.0, 5.0, -0.4], atol=1e-9)
        np.testing.assert_allclose(ns["pct_change_decode"](r, x[-1]), x, rtol=1e-12)
        np.testing.assert_allclose(ns["pct_change_decode"](r, x[0], anchor_at="first"), x, rtol=1e-12)

    def test_batch_axis_and_bad_values(self):
        ns = r2._ns()
        W = _walk()[:, :, 0]                                            # (N, T)
        r = ns["pct_change_encode"](W, axis=1)
        np.testing.assert_allclose(ns["pct_change_decode"](r, W[:, -1], axis=1), W, rtol=1e-12)
        bad = ns["pct_change_encode"](np.array([0.0, 5.0, np.nan, 7.0]))
        self.assertTrue(np.isfinite(bad).all())                         # سابق صفري أو غير منتهٍ ← 0 لا inf/nan
        with self.assertRaises(ValueError):
            ns["pct_change_decode"](r, W[:, -1], anchor_at="middle", axis=1)


class ProcessWindowsTests(unittest.TestCase):
    COLS = ["close", "EMA_10", "RSI_14"]

    def test_batch_matches_single_and_decodes(self):
        ns = r2._ns()
        cfg = dict(ns["CONFIG"], price_norm_mode="pct_change")
        W = _walk()
        W[:, :, 2] = np.linspace(20, 80, W.shape[1])                    # RSI بمداه الطبيعي
        out = _pw(cfg, W, self.COLS, np.full(len(W), 100.0, "float32"))
        single = np.stack([ns["process_window"](W[i].copy(), self.COLS, 100.0, 1.0, config=cfg) for i in range(len(W))])
        np.testing.assert_allclose(out, single, atol=1e-6)
        for j in (0, 1):                                                # كلا عمودي price_level يُفكّان من آخر قيمة
            back = ns["pct_change_decode"](out[:, :, j], W[:, -1, j], axis=1)
            np.testing.assert_allclose(back, W[:, :, j], rtol=1e-5)
        # بقية الأنواع لا تتغيّر بين الوضعين
        default = _pw(ns["CONFIG"], W, self.COLS, np.full(len(W), 100.0, "float32"))
        np.testing.assert_array_equal(out[:, :, 2], default[:, :, 2])

    def test_pct_clip_not_clip_abs(self):
        ns = r2._ns()
        w = np.full((1, 8, 1), 100.0)
        w[0, 4:, 0] = 107.0                                             # +7% — فوق clip_abs=5
        w[0, 6:, 0] = 107.0 * 3                                         # +200% — فوق price_pct_clip
        out = _pw(dict(ns["CONFIG"], price_norm_mode="pct_change", clip_abs=5.0, price_pct_clip=100.0), w, ["close"])
        np.testing.assert_allclose(out[0, :, 0], [0, 0, 0, 0, 7.0, 0, 100.0, 0], atol=1e-5)

    def test_default_mode_unchanged_and_unknown_mode_raises(self):
        ns = r2._ns()
        W = _walk(seed=3)
        c, s = np.full(len(W), 100.0, "float32"), np.full(len(W), 2.0, "float32")
        base = {k: v for k, v in ns["CONFIG"].items() if k not in ("price_norm_mode", "price_pct_clip")}
        np.testing.assert_array_equal(_pw(base, W, self.COLS, c, s),
                                      _pw(dict(base, price_norm_mode="window_scale"), W, self.COLS, c, s))
        self.assertEqual(ns["DEFAULT_CONFIG"]["price_norm_mode"], "window_scale")
        for bad in ({"price_norm_mode": "pct"}, {"price_norm_mode": "pct_change", "price_pct_clip": 0}):
            with self.assertRaises(ValueError):
                _pw(dict(base, **bad), W, self.COLS, c, s)

    def test_audit_uses_pct_limit(self):
        ns = r2._ns()
        w = np.full((1, 8, 1), 100.0)
        w[0, 4:, 0] = 110.0                                             # +10% — «متطرف» بمقياس clip_abs، عادي كنقاط مئوية
        cfg = dict(ns["CONFIG"], price_norm_mode="pct_change")
        table = ns["audit_normalization"](_pw(cfg, w, ["close"]), ["close"], verbose=False, config=cfg)
        self.assertEqual(table.verdict.iloc[0], "✅")


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = {c: r2._ohlcv("2025-01-01", "2025-02-15", s) for s, c in enumerate(("AAAUSDT", "BBBUSDT"))}
        cls.cfg = r2._cfg(price_norm_mode="pct_change")
        cls.ds = r2._build(cls.cfg, cls.frames)
        cls.ds_old = r2._build(r2._cfg(), cls.frames)

    def test_dataset_carries_mode_only_when_not_default(self):
        self.assertEqual((self.ds["price_norm_mode"], self.ds["price_pct_clip"]), ("pct_change", 100.0))
        self.assertNotIn("price_norm_mode", self.ds_old)
        self.assertNotIn("price_pct_clip", self.ds_old)
        np.testing.assert_array_equal(self.ds["last_candles"], self.ds_old["last_candles"])
        for t in self.ds["targets"]:                                    # الأهداف لا تتغيّر
            np.testing.assert_array_equal(self.ds[f"y_{t}"], self.ds_old[f"y_{t}"])

    def test_decode_price_columns_equal_raw_prices(self):
        ns = r2._ns()
        T = self.cfg["window_sizes"]["1h"]
        ts = pd.to_datetime(self.ds["last_candles"][:, 3].astype("int64"), utc=True)
        cols = [c for c in ("close", "high", "low") if c in self.ds["feature_order"]]
        self.assertIn("close", cols)
        for col in cols:
            dec = ns["decode_price_window"](self.ds, col)
            self.assertEqual(dec.shape, (len(ts), T))
            for b in self.ds["asset_bounds"]:
                raw = self.frames[b["name"]][col]
                for i in range(b["start"], b["end"], 7):
                    want = raw.loc[ts[i] - pd.Timedelta(hours=T - 1):ts[i]].to_numpy()
                    np.testing.assert_allclose(dec[i], want, rtol=2e-6, err_msg=f"{col} {b['name']} {i}")

    def test_decode_refuses_without_mode_or_anchor(self):
        ns = r2._ns()
        with self.assertRaises(ValueError):
            ns["decode_price_window"](self.ds_old, "close")               # وضع window_scale: لا فكّ
        split = {"X_1h": self.ds["X_1h"], "last_candles": self.ds["last_candles"], "base_tf": "1h"}
        with self.assertRaises(ValueError):
            ns["decode_price_window"](split, "close", feature_order=self.ds["feature_order"])   # قسم بلا الوضع
        dec = ns["decode_price_window"](split, "close", feature_order=self.ds["feature_order"], mode="pct_change")
        np.testing.assert_array_equal(dec, ns["decode_price_window"](self.ds, "close"))
        ema = [f for f in self.ds["feature_order"] if ns["classify_feature"](f) == ns["PRICE_LEVEL"]
               and f not in ("open", "high", "low", "close")]
        if ema:
            with self.assertRaises(ValueError):
                ns["decode_price_window"](self.ds, ema[0])              # لا مرساة مخزَّنة ← anchor إلزامي
        rsi = [f for f in self.ds["feature_order"] if ns["classify_feature"](f) != ns["PRICE_LEVEL"]]
        with self.assertRaises(ValueError):
            ns["decode_price_window"](self.ds, rsi[0], anchor=np.ones(len(self.ds["last_candles"])))

    def test_consecutive_samples_decode_consistently(self):
        ns = r2._ns()
        chk = ns["pct_decode_consistency"](self.ds)
        self.assertGreater(chk["pairs"], 100)
        self.assertLess(chk["max_rel_err"], 1e-5)
        self.assertEqual(chk["clipped_frac"], 0.0)

    def test_fingerprint_changes_only_for_pct_mode(self):
        ns = r2._ns()
        fp = ns["_checkpoint_fingerprint"]
        base = {k: v for k, v in r2._cfg().items() if k not in ("price_norm_mode", "price_pct_clip")}
        f0 = fp(base, ["close"], 0)
        self.assertEqual(f0, fp(dict(base, price_norm_mode="window_scale", price_pct_clip=50.0), ["close"], 0))
        f1 = fp(dict(base, price_norm_mode="pct_change"), ["close"], 0)
        self.assertNotEqual(f0, f1)
        self.assertNotEqual(f1, fp(dict(base, price_norm_mode="pct_change", price_pct_clip=50.0), ["close"], 0))

    def test_preset_changes_only_price_norm(self):
        ns = r2._ns()
        pct, s8 = deepcopy(ns["PCT_CHANGE_OVERRIDES"]), ns["HOURLY_W32_S8_OVERRIDES"]
        self.assertEqual({k: v for k, v in pct.items() if k not in ("price_norm_mode", "price_pct_clip")}, s8)
        self.assertEqual(pct["price_norm_mode"], "pct_change")
        self.assertNotEqual(ns["HOURLY_PCT_NAME"], ns["HOURLY_W32_S8_NAME"])


if __name__ == "__main__":
    unittest.main()

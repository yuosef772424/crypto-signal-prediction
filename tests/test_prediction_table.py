"""prediction_table / prediction_table_summary (workflow/reports.py): الحقيقة المفكوكة بجانب التنبؤ وبقية المخرجات.

مخرجات النموذج هنا مصطنعة (تنبؤ مثالي = الهدف المُدرَّب نفسه) فيثبت الاختبار الفكّ لا جودة نموذج: عائد بمقياس، magnitude،
logret_std، وقاموس أصول (test). بلا Drive ولا TensorFlow:
    python -m unittest tests.test_prediction_table -v
"""
import unittest

import numpy as np

from tests import test_price_norm_mode as pn
from tests import test_reg_target_scale as H

TARGETS = ("high", "low", "close")


def _outputs(split, scale_noise=0.0, ale=0.01, epi=0.002, conf=0.7):
    """مخرجات نموذج مثالية: y_t = y_t_reg المُدرَّب، واحتمال الفئة 0.9/0.1 حسب التسمية الحقيقية."""
    out = {}
    for t in TARGETS:
        y = np.asarray(split["y"][f"y_{t}_reg"], dtype="float64")
        out[f"y_{t}"] = y + scale_noise
        out[f"y_{t}_aleatoric"] = np.full(len(y), ale)
        out[f"y_{t}_epistemic"] = np.full(len(y), epi)
        out[f"y_{t}_confidence"] = np.full(len(y), conf)
        out[f"y_{t}_class_logits"] = np.where(np.asarray(split["y"][f"y_{t}_class"]) > 0, 0.9, 0.1)
    return out


class _StubModel:
    """model.predict لا يهمّه X: يُرجع مخرجات القسم ذي الحجم n (أحجام الأقسام هنا مختلفة)."""

    def __init__(self, by_n):
        self.by_n = by_n

    def predict(self, x, batch_size=None, verbose=0):
        return self.by_n[len(x)]


class PredictionTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = H._load_workflow(dict(H._ns()))

    def _scaled_split(self, n, seed, scale):
        s = H._split(self.ns, n, seed)
        for k in list(s["y"]):
            if k.endswith("_reg"):
                s["y"][k] = s["y"][k] * scale
        s["reg_target_scale"] = scale
        return s

    def test_return_mode_decodes_scale_and_aligns_columns(self):
        sp = self._scaled_split(40, 0, 100.0)
        df = self.ns["prediction_table"](None, sp, "1h", TARGETS, outputs=_outputs(sp))
        lc, cols = sp["last_candles"], self.ns["LAST_COLUMNS"]
        for t, fut, last in (("high", "future_high_max", "last_high"), ("low", "future_low_min", "last_low"),
                             ("close", "future_close", "last_close")):
            truth = lc[:, cols.index(fut)] / lc[:, cols.index(last)] - 1
            np.testing.assert_allclose(df[f"true_{t}"], truth, atol=1e-6)       # الحقيقة بوحدة العائد لا × المقياس
            np.testing.assert_allclose(df[f"pred_{t}"], df[f"true_{t}"])
            np.testing.assert_allclose(df[f"err_{t}"], 0.0, atol=1e-12)
            np.testing.assert_allclose(df[f"ale_{t}"], 0.01 / 100.0)             # عدم اليقين يُقسم على المقياس أيضاً
            np.testing.assert_allclose(df[f"epi_{t}"], 0.002 / 100.0)
            np.testing.assert_allclose(df[f"z_{t}"], 0.0, atol=1e-9)
            np.testing.assert_allclose(df[f"conf_{t}"], 0.7)
            self.assertTrue((df[f"hit_{t}"] == 1).all())
            np.testing.assert_array_equal(df[f"cls_{t}"], (sp["y"][f"y_{t}_class"] > 0).astype(int))
        self.assertEqual(list(df["asset"].unique()), ["all"])
        self.assertTrue(np.issubdtype(df["timestamp"].dtype, np.datetime64))
        np.testing.assert_allclose(df["entry"], lc[:, cols.index("last_close")])

    def test_error_is_prediction_minus_truth(self):
        sp = self._scaled_split(30, 1, 100.0)
        df = self.ns["prediction_table"](None, sp, "1h", ("close",), outputs=_outputs(sp, scale_noise=0.5))
        np.testing.assert_allclose(df["err_close"], 0.5 / 100.0)                   # +0.5 بوحدة التدريب = +0.005 عائداً
        self.assertTrue((df["z_close"] > 0).all())

    def test_asset_dict_concatenates_with_names(self):
        a, b = self._scaled_split(24, 2, 1.0), self._scaled_split(25, 3, 1.0)
        model = _StubModel({24: _outputs(a), 25: _outputs(b)})
        df = self.ns["prediction_table"](model, {"AAA": a, "BBB": b}, "1h", TARGETS)
        self.assertEqual(len(df), 49)
        self.assertEqual(df["asset"].value_counts().to_dict(), {"AAA": 24, "BBB": 25})
        with self.assertRaises(ValueError):
            self.ns["prediction_table"](model, {"AAA": a, "BBB": b}, "1h", TARGETS, outputs=_outputs(a))
        with self.assertRaises(ValueError):                                         # price_targets صريحة، لا افتراض
            self.ns["prediction_table"](model, a, "1h", None, outputs=_outputs(a))

    def test_class_only_outputs_drop_regression_columns(self):
        sp = self._scaled_split(20, 4, 1.0)
        out = {k: v for k, v in _outputs(sp).items() if k.endswith("_class_logits")}
        df = self.ns["prediction_table"](None, sp, "1h", TARGETS, outputs=out)
        self.assertIn("p_close", df)
        self.assertFalse(any(c.startswith(("true_", "pred_", "ale_", "conf_")) for c in df.columns))

    def test_magnitude_truth_is_excursion_not_signed_return(self):
        ns = H._main_ns(1.0)
        _, va, _ = H._rt(ns, mode="magnitude")
        df = ns["prediction_table"](None, va, "1h", TARGETS, outputs=_outputs(va))
        lc, cols = va["last_candles"], ns["LAST_COLUMNS"]
        last = lc[:, cols.index("last_close")]
        np.testing.assert_allclose(df["true_close"], np.abs(lc[:, cols.index("future_close")] / last - 1), atol=1e-6)
        np.testing.assert_allclose(df["true_high"], np.maximum(lc[:, cols.index("future_high_max")] / last - 1, 0), atol=1e-6)
        np.testing.assert_allclose(df["true_low"], np.maximum(-(lc[:, cols.index("future_low_min")] / last - 1), 0), atol=1e-6)
        self.assertTrue((df[["true_high", "true_low", "true_close"]] >= 0).all().all())

    def test_summary_reports_honest_uncertainty_and_confidence_bins(self):
        sp = self._scaled_split(400, 5, 1.0)
        out = _outputs(sp, scale_noise=0.0)
        rng = np.random.default_rng(0)
        err = rng.normal(0, 0.01, 400)
        out["y_close"] = out["y_close"] + err
        out["y_close_aleatoric"] = np.abs(err) + 1e-4                              # عدم يقين يتتبّع الخطأ المطلق تماماً
        out["y_close_confidence"] = 1.0 - np.abs(err) / np.abs(err).max()          # ثقة تنخفض مع الخطأ
        out["y_close_class_logits"] = rng.uniform(0.0, 1.0, 400)
        df = self.ns["prediction_table"](None, sp, "1h", ("close",), outputs=out)
        res = self.ns["prediction_table_summary"](df, ("close",), verbose=False)
        ov = res["overview"].loc["close"]
        self.assertGreater(ov["spearman_ale_abs_err"], 0.99)
        self.assertLess(ov["spearman_conf_abs_err"], -0.99)
        self.assertEqual(set(res), {"overview", "close_by_conf", "close_by_margin"})
        self.assertEqual(int(res["close_by_conf"]["n"].sum()), 400)
        self.assertAlmostEqual(ov["class_base"], max(df["cls_close"].mean(), 1 - df["cls_close"].mean()))


class LogretDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = H._load_workflow(dict(H._ns()))
        cfg = pn._cfg("logret_std")
        cls.tr, cls.va, cls.te = pn._split(pn._build(cfg), cfg)
        cls.meta = cls.tr["price_norm_meta"]

    def test_logret_truth_pred_and_uncertainty_units(self):
        va = self.va
        out = _outputs(va, ale=1.0, epi=0.5)
        df = self.ns["prediction_table"](None, va, "1h", TARGETS, outputs=out)
        lc = va["last_candles"]
        for t, col in (("high", 6), ("low", 5), ("close", 4)):
            np.testing.assert_allclose(df[f"true_{t}"], np.log(lc[:, col] / lc[:, 2]), atol=1e-4)   # log(سعر/آخر_إغلاق)
            np.testing.assert_allclose(df[f"pred_{t}"], df[f"true_{t}"], atol=1e-9)
            std = self.meta["targets"][t]["std"]
            np.testing.assert_allclose(df[f"ale_{t}"], 1.0 * std, rtol=1e-6)                       # وحدة معيارية ← × std
            np.testing.assert_allclose(df[f"epi_{t}"], 0.5 * std, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

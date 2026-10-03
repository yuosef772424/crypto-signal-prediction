"""استقرار NIG + معايرة الثقة post-hoc + تغطية كل صفوف الاختبار مع فاصل Wilson (chicks + model_v2).

تُنفَّذ خلايا الدفاتر نفسها (لا نسخاً):
    python -m unittest tests.test_nig_calibration -v
"""
import io
import contextlib
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_reg_target_scale import _cell, _ns, _split  # noqa: E402

_MV = {}


def _model_ns():
    """model_v2 (كل خلاياه عدا الاختبار الذاتي الذي يعمل عند الاستيراد)."""
    if not _MV:
        import tensorflow as tf  # noqa: F401
        mv = {"__name__": "audit_nb"}
        with open(os.path.join(ROOT, "model_v2 (1).ipynb"), encoding="utf-8") as f:
            cells = json.load(f)["cells"]
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            lines = [ln for ln in "".join(cell["source"]).splitlines()
                     if not ln.lstrip().startswith(("!", "%")) and ln.strip() != "run_model_selftests()"]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile("\n".join(lines), "model_v2", "exec"), mv)
        _MV.update(mv)
    return _MV


_CH = {}


def _chicks():
    """نطاق chicks مع خليتَي test_all_assets_v4 وrun_full_analysis (يتخطّاهما التحميل العام لنصّ الاستدعاء فيهما)."""
    if not _CH:
        ns = dict(_ns())
        for idx in (18, 36):
            exec(compile(_cell("chicks_v4_5_input_output_patterns.ipynb", idx), f"chicks#cell{idx}", "exec"), ns)
        _CH.update(ns)
    return dict(_CH)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ١) استقرار NIG
# ══════════════════════════════════════════════════════════════════════════════════════════════════
class NIGStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf, cls.mv = tf, _model_ns()

    def _unc(self, nu, alpha, beta, unc_max):
        tf = self.tf
        layer = self.mv["NIGUncertainty"](unc_max=unc_max)
        epi, ale = layer([tf.constant(nu, tf.float32), tf.constant(alpha, tf.float32), tf.constant(beta, tf.float32)])
        return epi.numpy(), ale.numpy()

    def test_normal_samples_unchanged(self):
        """القيم الطبيعية لا تتغيّر: √(β/(α−1)) و√(β/(ν(α−1))) حرفياً (القصّ لا يمسّها)."""
        nu, alpha, beta = [[1.0], [0.5]], [[3.0], [2.5]], [[0.25], [0.04]]
        epi, ale = self._unc(nu, alpha, beta, 20.0)
        np.testing.assert_allclose(ale, np.sqrt(np.array(beta) / (np.array(alpha) - 1)), rtol=1e-6)
        np.testing.assert_allclose(epi, np.sqrt(np.array(beta) / (np.array(nu) * (np.array(alpha) - 1))), rtol=1e-6)
        e2, a2 = self._unc(nu, alpha, beta, None)            # بلا قصّ: نفس القيم
        np.testing.assert_allclose(a2, ale, rtol=1e-7)
        np.testing.assert_allclose(e2, epi, rtol=1e-7)

    def test_extreme_parameters_finite_and_bounded(self):
        nu = np.array([[0.1], [1e-9], [1e9], [0.1], [0.1], [0.1], [np.nan]])
        alpha = np.array([[1.0], [1.0], [1.0 + 1e-7], [2.0], [1e9], [2.0], [2.0]])
        beta = np.array([[1e9], [1e30], [1e3], [np.inf], [1e-12], [3e38], [1.0]])
        for cap in (20.0, 5.0):
            epi, ale = self._unc(nu, alpha, beta, cap)
            self.assertTrue(np.isfinite(epi).all() and np.isfinite(ale).all(), (epi, ale))
            self.assertLessEqual(float(epi.max()), cap)
            self.assertLessEqual(float(ale.max()), cap)
            self.assertGreaterEqual(float(min(epi.min(), ale.min())), 0.0)

    def test_head_alpha_never_below_alpha_min_and_unbounded_beta_was_the_blowup(self):
        """الجذر: alpha ≥ alpha_min = 2 دائماً (softplus + 2) فلا يصل α → 1 من الرأس؛ الانفجار من β غير المحدود (softplus خطي)."""
        tf, mv = self.tf, self.mv
        h = tf.keras.Input((8,))
        head = mv["NIGHead"](hidden=8, dropout=0.0)
        _, nu, alpha, beta = head(h)
        epi_raw, ale_raw = mv["NIGUncertainty"](unc_max=None)([nu, alpha, beta])
        epi, ale = mv["NIGUncertainty"](unc_max=20.0)([nu, alpha, beta])
        m = tf.keras.Model(h, [alpha, ale_raw, epi_raw, ale, epi])
        head.out.kernel.assign(tf.zeros_like(head.out.kernel))
        head.out.bias.assign(tf.constant([0.0, 0.0, -1e9, 1e9]))        # alpha logit → −∞، beta logit → +∞
        a, ale_r, epi_r, ale_c, epi_c = (x.numpy() for x in m(np.ones((3, 8), "float32")))
        self.assertGreaterEqual(float(a.min()), 2.0)                     # α لا تقترب من 1
        self.assertGreater(float(ale_r.max()), 1e4)                      # بلا قصّ: عدم اليقين بعشرات الآلاف
        self.assertTrue(np.isfinite(ale_c).all() and np.isfinite(epi_c).all())
        self.assertLessEqual(float(max(ale_c.max(), epi_c.max())), 20.0)

    def test_alpha_min_near_one_is_floored(self):
        """لو خُفِّضت alpha_min إلى 1 (α → 1) فالأرضية تمنع β/(α−1) → ∞ حتى بلا قصّ: √(0.01/0.01) = 1."""
        tf, mv = self.tf, self.mv
        h = tf.keras.Input((4,))
        head = mv["NIGHead"](hidden=4, dropout=0.0, alpha_min=1.0)
        _, nu, alpha, beta = head(h)
        epi, ale = mv["NIGUncertainty"](unc_max=None)([nu, alpha, beta])
        m = tf.keras.Model(h, [alpha, ale, epi])
        head.out.kernel.assign(tf.zeros_like(head.out.kernel))
        head.out.bias.assign(tf.constant([0.0, 0.0, -1e9, -1e9]))
        a, al, ep = (x.numpy() for x in m(np.ones((2, 4), "float32")))
        np.testing.assert_allclose(a, 1.0)
        self.assertTrue(np.isfinite(al).all() and np.isfinite(ep).all())
        np.testing.assert_allclose(al, 1.0, rtol=1e-5)

    def test_full_model_knob_and_bounded_on_extreme_input(self):
        mv = _model_ns()
        heads = {t: ["nig_regression", "binary_classification"] for t in ("high", "low")}
        cfg = dict(mv["ANTI_MEMORIZATION_CONFIG"], price_targets=("high", "low"), head_types=heads, d_model=16,
                   num_layers=1, head_hidden=8, class_head_hidden=8)
        model = mv["build_model_fn"](16, 4, config=cfg)
        self.assertEqual(model.get_layer("unc_high").unc_max, mv["MODEL_CONFIG"]["unc_max"])
        model5 = mv["build_model_fn"](16, 4, config=dict(cfg, unc_max=5.0))
        self.assertEqual(model5.get_layer("unc_low").unc_max, 5.0)
        x = np.random.default_rng(0).normal(0, 1, (6, 16, 4)).astype("float32")
        x[0] *= 1e6
        x[1] *= 1e30
        out = model5(x, training=False)
        for t in ("high", "low"):
            for k in ("aleatoric", "epistemic"):
                v = out[f"y_{t}_{k}"].numpy()
                self.assertTrue(np.isfinite(v).all(), (t, k))
                self.assertLessEqual(float(v.max()), 5.0)


class ChicksNIGTests(unittest.TestCase):
    def test_numpy_matches_layer_formula_and_bounds(self):
        ns = _ns()
        nu, alpha, beta = np.array([1.0, 0.5, 0.1]), np.array([3.0, 2.5, 1.0]), np.array([0.25, 0.04, 1e9])
        epi, ale, clipped = ns["nig_uncertainty_bounded"](nu, alpha, beta, 20.0)
        np.testing.assert_allclose(ale[:2], np.sqrt(beta[:2] / (alpha[:2] - 1)))
        np.testing.assert_allclose(epi[:2], np.sqrt(beta[:2] / (nu[:2] * (alpha[:2] - 1))))
        self.assertEqual(clipped.tolist(), [False, False, True])
        self.assertEqual((ale[2], epi[2]), (20.0, 20.0))
        e, a, c = ns["nig_uncertainty_bounded"]([0.1], [2.0], [np.nan], 20.0)       # غير منتهٍ → الحدّ
        self.assertEqual((float(e[0]), float(a[0]), bool(c[0])), (20.0, 20.0, True))

    def test_predict_batch_bounds_even_a_model_with_old_unbounded_layer(self):
        """نموذج مُدرَّب بطبقة قديمة (epistemic/aleatoric خارجان ضخمان/∞) يُعاد حسابهما محدودَين من ν/α/β دون إعادة تدريب."""
        ns = _chicks()
        n = 6
        beta = np.full((n, 1), 0.04)
        beta[0], beta[1] = 1e9, np.inf
        alpha = np.full((n, 1), 2.0)
        alpha[2] = 1.0 + 1e-9                                                     # α ≈ 1
        rng = np.random.default_rng(0)

        def model(x, training=False):
            out = {}
            for t in ("high", "low", "close"):
                out.update({f"y_{t}": rng.normal(0, 0.01, (n, 1)), f"y_{t}_nu": np.full((n, 1), 0.5),
                            f"y_{t}_alpha": alpha, f"y_{t}_beta": beta,
                            f"y_{t}_epistemic": np.sqrt(beta / 1e-8), f"y_{t}_aleatoric": np.sqrt(beta / 1e-8),
                            f"y_{t}_confidence": rng.uniform(0.3, 0.9, (n, 1))})
            return out

        raw = ns["predict_batch_v4"](model, (np.zeros((n, 4, 2), "float32"),), ns["DEFAULT_PRICE_TARGETS"])
        for t in ("high", "low", "close"):
            for k in ("epistemic", "aleatoric"):
                v = raw[f"{t}_{k}"]
                self.assertTrue(np.isfinite(v).all())
                self.assertLessEqual(float(v.max()), ns["NIG_UNC_MAX"])
            self.assertEqual(raw[f"{t}_unc_clipped"][:3].tolist(), [True, True, False])
            self.assertFalse(raw[f"{t}_unc_clipped"][3:].any())
            self.assertAlmostEqual(float(raw[f"{t}_aleatoric"][2, 0]), 2.0, places=5)   # α ≈ 1: الأرضية (0.01) لا ∞
            np.testing.assert_allclose(raw[f"{t}_aleatoric"][3:, 0], np.sqrt(0.04 / 1.0), rtol=1e-6)   # الطبيعي سليم
        raw2 = ns["predict_batch_v4"](model, (np.zeros((n, 4, 2), "float32"),), ns["DEFAULT_PRICE_TARGETS"], unc_max=1.0)
        self.assertLessEqual(float(raw2["high_aleatoric"].max()), 1.0)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ٢) معايرة الثقة
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _df(n, seed, conf_fn, p_fn=None, acc_fn=None, target="close", class_fn=None):
    """DataFrame مسطّح مصطنع: confidence، correct، (p_up، price_change بحيث صحة رأس التصنيف = class_fn(margin))."""
    rng = np.random.default_rng(seed)
    conf = conf_fn(rng, n)
    correct = (rng.random(n) < (acc_fn(conf) if acc_fn else 0.55)).astype(int)
    d = {"target": target, "confidence": conf, "correct": correct}
    if p_fn is not None:
        p_up = p_fn(rng, n)
        margin = np.abs(p_up - 0.5)
        ok = rng.random(n) < class_fn(margin)
        up = np.where(ok, p_up >= 0.5, p_up < 0.5)
        d.update(p_up=p_up, price_change=np.where(up, 1.0, -1.0))
    return pd.DataFrame(d)


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _ns()

    def test_uninformative_confidence_falls_back_to_class_margin(self):
        """الثقة عشوائية مبالَغ فيها (0.72 مقابل دقة 0.55): يُكشف ذلك ويُعايَر |p_up−0.5| بدلها، وECE يهبط، والخام محفوظ."""
        ns = self.ns
        mk = lambda seed: _df(6000, seed, lambda r, n: r.uniform(0.5, 0.95, n),
                              p_fn=lambda r, n: r.uniform(0.05, 0.95, n), class_fn=lambda m: 0.5 + 0.6 * m)
        val, test = mk(1), mk(2)
        cal = ns["fit_confidence_calibrators"](val, method="isotonic", verbose=False)
        self.assertEqual(cal["close"]["signal"], "class_margin")
        self.assertNotEqual(cal["close"]["diag"]["verdict"], "informative")
        before = test["confidence"].copy()
        cdf = ns["apply_confidence_calibration"](test, cal)
        pd.testing.assert_series_equal(cdf["confidence"], before)                   # الخام لا يُمسّ
        rep = ns["calibration_report"](cdf, verbose=False).iloc[0]
        self.assertLess(rep["ece_after"], 0.05)
        self.assertLess(rep["ece_after"], rep["ece_before"])
        self.assertTrue(((cdf["confidence_cal"] >= 0) & (cdf["confidence_cal"] <= 1)).all())

    def test_informative_confidence_is_calibrated_isotonic_and_platt(self):
        ns = self.ns
        mk = lambda seed: _df(6000, seed, lambda r, n: r.uniform(0.3, 0.95, n), acc_fn=lambda c: 0.3 + 0.35 * c)
        val, test = mk(3), mk(4)
        for method in ("isotonic", "platt"):
            cal = ns["fit_confidence_calibrators"](val, method=method, verbose=False)
            self.assertEqual(cal["close"]["signal"], "confidence", method)
            rep = ns["calibration_report"](ns["apply_confidence_calibration"](test, cal), verbose=False).iloc[0]
            self.assertLess(rep["ece_after"], rep["ece_before"], method)
            self.assertLess(rep["ece_after"], 0.04, method)

    def test_constant_confidence_no_class_head_uses_base_rate(self):
        ns = self.ns
        val = _df(500, 5, lambda r, n: np.full(n, 0.72))
        test = _df(500, 6, lambda r, n: np.full(n, 0.72))
        cal = ns["fit_confidence_calibrators"](val, verbose=False)
        self.assertEqual(cal["close"]["diag"]["verdict"], "constant")
        self.assertEqual(cal["close"]["signal"], "base_rate")
        cdf = ns["apply_confidence_calibration"](test, cal)
        np.testing.assert_allclose(cdf["confidence_cal"], val["correct"].mean())

    def test_fit_uses_val_only(self):
        """المعاير دالة في val وحده: تغيير تسميات test لا يغيّر خريطته."""
        ns = self.ns
        val = _df(2000, 7, lambda r, n: r.uniform(0.3, 0.95, n), acc_fn=lambda c: 0.3 + 0.35 * c)
        cal = ns["fit_confidence_calibrators"](val, verbose=False)
        grid = np.linspace(0.3, 0.95, 20)
        a = cal["close"]["predict"](grid)
        test = _df(2000, 8, lambda r, n: r.uniform(0.3, 0.95, n), acc_fn=lambda c: 0.9 - 0.5 * c)
        ns["apply_confidence_calibration"](test, cal)
        np.testing.assert_array_equal(a, cal["close"]["predict"](grid))

    def test_pav_fallback_matches_sklearn_isotonic(self):
        from sklearn.isotonic import IsotonicRegression
        ns = self.ns
        rng = np.random.default_rng(9)
        x = rng.uniform(0, 1, 500)
        y = (rng.random(500) < x).astype(float)
        f = ns["_pav_isotonic"](x, y)
        sk = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y)
        q = np.linspace(0, 1, 50)
        np.testing.assert_allclose(f(q), sk.predict(q), atol=1e-9)

    def test_ece_handles_constant_input(self):
        self.assertAlmostEqual(self.ns["_ece_quantile"](np.full(100, 0.7), np.r_[np.ones(60), np.zeros(40)]), 0.1, places=2)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# ٣) تغطية كل صفوف الاختبار + Wilson
# ══════════════════════════════════════════════════════════════════════════════════════════════════
def _fake_model(seed=0):
    rng = np.random.default_rng(seed)

    def model(x, training=False):
        xb = x[0] if isinstance(x, (tuple, list)) else x
        n = len(xb)
        out = {}
        for t in ("high", "low", "close"):
            out.update({f"y_{t}": rng.normal(0, 0.01, (n, 1)), f"y_{t}_nu": rng.uniform(0.5, 2, (n, 1)),
                        f"y_{t}_alpha": rng.uniform(2, 3, (n, 1)), f"y_{t}_beta": rng.uniform(1e-4, 1e-3, (n, 1)),
                        f"y_{t}_confidence": rng.uniform(0.3, 0.9, (n, 1)),
                        f"y_{t}_class_logits": rng.uniform(0.05, 0.95, (n, 1))})
        return out
    return model


def _test_dict(ns, sizes, seed0=10):
    td = {}
    for i, n in enumerate(sizes):
        s = _split(ns, n, seed0 + i)
        td[f"C{i}"] = {"X_1h": s["X_1h"], "base_params": s["base_params"], "last_candles": s["last_candles"],
                       "y": {t: s["y"][f"y_{t}_reg"] for t in ("high", "low", "close")}}
    return td


class CoverageWilsonTests(unittest.TestCase):
    def test_wilson_values(self):
        w = _ns()["wilson_ci"]
        lo, hi = w(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        lo, hi = w(0, 20)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1611, places=3)
        self.assertTrue(np.isnan(w(0, 0)[0]))
        lo2, hi2 = w(500, 1000)
        self.assertLess(hi2 - lo2, hi - lo)                                         # n أكبر ← فاصل أضيق

    def test_all_rows_evaluated_by_default_and_old_filter_optional(self):
        ns = _chicks()
        td = _test_dict(ns, [5, 25, 40])
        specs = ns["DEFAULT_PRICE_TARGETS"]
        res = ns["test_all_assets_v4"](_fake_model(), td, ["1h"], specs, verbose=False, run_verification=False)
        cov = res["coverage"]
        self.assertEqual((cov["test_rows"], cov["evaluated_rows"], cov["evaluated_assets"]), (70, 70, 3))
        self.assertEqual(cov["skipped_assets"], [])
        ci = res["asset_ci"]
        self.assertEqual(set(ci["asset"]), {"C0", "C1", "C2"})
        self.assertEqual(sorted(ci[ci.target == "close"]["n"]), [5, 25, 40])
        for _, r in ci.iterrows():
            self.assertLessEqual(r["ci95_lo_%"], r["acc_%"] + 1e-9)
            self.assertGreaterEqual(r["ci95_hi_%"], r["acc_%"] - 1e-9)
        small = ci[(ci.target == "close") & (ci.n == 5)].iloc[0]
        big = ci[(ci.target == "close") & (ci.n == 40)].iloc[0]
        self.assertGreater(small["ci95_hi_%"] - small["ci95_lo_%"], big["ci95_hi_%"] - big["ci95_lo_%"])
        old = ns["test_all_assets_v4"](_fake_model(), td, ["1h"], specs, verbose=False, run_verification=False,
                                       eval_all_rows=False)
        self.assertEqual(old["coverage"]["evaluated_rows"], 65)
        self.assertEqual(old["coverage"]["skipped_assets"], [("C0", 5)])
        # n_display عرض فقط: لا يغيّر المجمّع
        a = ns["test_all_assets_v4"](_fake_model(), td, ["1h"], specs, verbose=False, run_verification=False, n_display=0)
        self.assertEqual(a["coverage"]["evaluated_rows"], 70)

    def test_full_analysis_with_val_calibration_and_ret_columns(self):
        ns = _chicks()
        td, val = _test_dict(ns, [30, 30]), _test_dict(ns, [60], seed0=50)
        specs = [ns["dataclasses"].replace(s, class_key=f"y_{s.name}_class_logits") for s in ns["DEFAULT_PRICE_TARGETS"]] \
            if "dataclasses" in ns else ns["DEFAULT_PRICE_TARGETS"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = ns["run_full_analysis"](_fake_model(), td, ["1h"], specs, make_plots=False, verbose=True,
                                          run_integrity_check=False, run_trade_selection=False,
                                          out_dir="/tmp/_nig_cal_test", val_dict={"VAL": val["C0"]})
        flat = res["flat_df"]
        self.assertEqual(len(flat[flat.target == "close"]), 60)                     # كل صفوف الاختبار
        for c in ("aleatoric_ret", "epistemic_ret", "uncertainty_ret", "confidence", "unc_clipped"):
            self.assertIn(c, flat.columns)
        np.testing.assert_allclose(flat["aleatoric_ret"], flat["aleatoric"] / (flat["entry"].abs() + 1e-7))
        cal = res["trust_report"]["calibration"]
        self.assertEqual(set(cal["target"]), {"high", "low", "close"})
        self.assertTrue({"ece_before", "ece_after"} <= set(cal.columns))
        self.assertIn("confidence_cal", res["trust_report"]["calibrated_df"].columns)
        self.assertIn("accuracy_ci95_lo", res["trust_report"]["by_group"].columns)
        self.assertIn("معايرة الثقة post-hoc", buf.getvalue())
        self.assertIn("ECE قبل ← بعد", buf.getvalue())
        self.assertIsNotNone(res["calibrators"])
        pd_res = res["pattern_discovery"]
        feats = next(iter(pd_res["per_head"].values()))["features"]
        # عدم اليقين النسبي لا بوحدة السعر
        self.assertIn("aleatoric_ret", feats)
        self.assertNotIn("aleatoric", feats)


if __name__ == "__main__":
    unittest.main()

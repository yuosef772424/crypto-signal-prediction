"""وضع الهدف entry_range (main القسم ٣-ب): كل الأهداف من سعر الدخول P = last_close.

    high = up_exc = H/P − 1، low = dn_exc = 1 − L/P، close = pos = (C − L)/(H − L) (0.5 حين H = L)
    التسميات 1/0: up_exc > c، dn_exc > c، pos > 0.5 — والمقياس يُضرب في high/low فقط.

تُنفَّذ خلايا الدفترين نفسها (لا نسخاً منها) كما في test_reg_target_scale، ببيانات تركيبية فيها شمعة مداها صفر
وشمعة فجوة هبوط (قمة الأفق تحت سعر الدخول):
    python -m unittest tests.test_entry_range -v
"""
import json
import os
import re
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_reg_target_scale import _cell, _ns  # noqa: E402 — نطاق خط الأنابيب وchicks المشترك (يُحمَّل مرّة)

C = 0.002   # ENTRY_RANGE_CLASS_THRESHOLD الافتراضي
ZERO, GAP = 0, 1   # صفّا الحالتين الحدّيتين في كل قسم


def _split(ns, n, seed, scale):
    """قسم بشكل split_data: last_candles حقيقية، وأهداف 'return' خط الأنابيب (× scale) — entry_range يعيد حسابها."""
    rng = np.random.default_rng(seed)
    cols = ns["LAST_COLUMNS"]
    P = 100 * np.exp(rng.normal(0, 0.3, n))
    H = P * (1 + np.abs(rng.normal(0.004, 0.01, n)))
    L = P * (1 - np.abs(rng.normal(0.004, 0.01, n)))
    Cl = L + rng.uniform(0, 1, n) * (H - L)
    H[ZERO] = L[ZERO] = Cl[ZERO] = P[ZERO] * 1.003               # مدى صفر (وفوق الدخول بـ 0.3%)
    H[GAP], L[GAP], Cl[GAP] = P[GAP] * 0.98, P[GAP] * 0.95, P[GAP] * 0.97   # فجوة هبوط: H < P
    lc = np.zeros((n, len(cols)))
    for k, v in (("last_close", P), ("last_high", P * 1.01), ("last_low", P * 0.99), ("future_close", Cl),
                 ("future_high_max", H), ("future_low_min", L)):
        lc[:, cols.index(k)] = v
    lc[:, cols.index("timestamp")] = (np.arange(n) // 4) * 8 * 3600 * 10**9 + 1_700_000_000 * 10**9
    y = {}
    for t, fut, last in (("high", H, P * 1.01), ("low", L, P * 0.99), ("close", Cl, P)):
        ret = fut / last - 1
        y[f"y_{t}_reg"] = (ret * scale).astype("float32")
        y[f"y_{t}_class"] = (ret > 0).astype("float32")
    return {"X_1h": np.zeros((n, 4, 2), "float32"), "last_candles": lc, "y": y,
            "base_params": np.c_[P, P * 0.02].astype("float32"), "reg_target_scale": scale}


def _main_ns(scale, price_targets=("high", "low", "close")):
    """خلايا main (٣ الختم، ٣-ب، ٦ chicks، ٧-ب collect_signals) على بيانات بمقياس scale."""
    ns = dict(_ns())
    tr, va = _split(ns, 64, 0, scale), _split(ns, 32, 1, scale)
    te = {"AAA": _split(ns, 24, 2, scale), "BBB": _split(ns, 24, 3, scale)}
    ns.update(REG_TARGET_SCALE=scale, MODEL_TF="1h", PRICE_TARGETS=list(price_targets),
              split_data=lambda dataset, config=None: (tr, va, te), dataset={})
    for idx in (8, 10):
        exec(compile(_cell("main.ipynb", idx), f"main#cell{idx}", "exec"), ns)
    return ns


def _expected(split):
    lc, cols = split["last_candles"], _ns()["LAST_COLUMNS"]
    P, H, L, Cl = (lc[:, cols.index(k)] for k in ("last_close", "future_high_max", "future_low_min", "future_close"))
    pos = np.where(H > L, (Cl - L) / np.where(H > L, H - L, 1.0), 0.5)
    return {"high": H / P - 1, "low": 1 - L / P, "close": pos}, {"P": P, "high": H, "low": L, "close": Cl}


def _retarget(ns, **kw):
    return ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode="entry_range", verbose=False, **kw)


class EntryRangeTargetTests(unittest.TestCase):
    def test_targets_on_synthetic_candles(self):
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            for split in (_retarget(ns)[0], *_retarget(ns)[2].values()):
                want, _ = _expected(split)
                np.testing.assert_allclose(split["y"]["y_high_reg"], want["high"] * scale, rtol=1e-6, atol=1e-7 * scale)
                np.testing.assert_allclose(split["y"]["y_low_reg"], want["low"] * scale, rtol=1e-6, atol=1e-7 * scale)
                np.testing.assert_allclose(split["y"]["y_close_reg"], want["close"], rtol=1e-6, atol=1e-7)   # بلا مقياس
                self.assertEqual(split["target_mode"], "entry_range")
                self.assertEqual(split["reg_target_scale"], scale)
                self.assertEqual(split["reg_target_scales"], {"high": scale, "low": scale, "close": 1.0})
                self.assertEqual(ns["reg_scale_of"](split, "close"), 1.0)
                self.assertEqual(ns["reg_scale_of"](split, "high"), scale)
                # الحالتان الحدّيتان صراحةً
                self.assertAlmostEqual(float(split["y"]["y_close_reg"][ZERO]), 0.5, places=6)   # مدى صفر
                self.assertAlmostEqual(float(split["y"]["y_high_reg"][ZERO]) / scale, 0.003, places=6)
                self.assertAlmostEqual(float(split["y"]["y_high_reg"][GAP]) / scale, -0.02, places=6)  # فجوة: سالب
                self.assertAlmostEqual(float(split["y"]["y_low_reg"][GAP]) / scale, 0.05, places=6)
                self.assertAlmostEqual(float(split["y"]["y_close_reg"][GAP]), 2 / 3, places=5)

    def test_classes_match_definitions(self):
        ns = _main_ns(100.0)
        for c in (C, 0.01):
            tr = _retarget(ns, class_threshold=None if c == C else c)[0]
            want, _ = _expected(tr)
            y = tr["y"]
            self.assertEqual(set(np.unique(np.r_[y["y_high_class"], y["y_low_class"], y["y_close_class"]])), {0.0, 1.0})
            np.testing.assert_array_equal(y["y_high_class"], (want["high"] > c).astype("float32"))
            np.testing.assert_array_equal(y["y_low_class"], (want["low"] > c).astype("float32"))
            np.testing.assert_array_equal(y["y_close_class"], (want["close"] > 0.5).astype("float32"))
            self.assertEqual(tr["class_threshold"], c)
            # مدى صفر فوق الدخول بـ 0.3%: صعود > c=0.002 فقط، لا هبوط، والموقع 0.5 ليس > 0.5؛ فجوة الهبوط: العكس
            zero = [1.0 if c < 0.003 else 0.0, 0.0, 0.0]
            self.assertEqual([y[f"y_{t}_class"][ZERO] for t in ("high", "low", "close")], zero)
            self.assertEqual([y[f"y_{t}_class"][GAP] for t in ("high", "low", "close")], [0.0, 1.0, 1.0])

    def test_close_recomputed_even_when_suspended(self):
        """إعادة الخلية بعد القسم ٤ (PRICE_TARGETS بلا close) لا تُبقي y_close بوضع سابق تحت ختم entry_range."""
        ns = _main_ns(100.0, price_targets=("high", "low"))
        tr = _retarget(ns)[0]
        np.testing.assert_allclose(tr["y"]["y_close_reg"], _expected(tr)[0]["close"], rtol=1e-6, atol=1e-7)

    def test_relative_composition_refused(self):
        ns = _main_ns(1.0)
        with self.assertRaises(ValueError):
            ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode="entry_range+relative", verbose=False)

    def test_roundtrip_targets_to_prices(self):
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            for split in _retarget(ns)[2].values():
                _, px = _expected(split)
                got = ns["entry_range_to_prices"](px["P"], scale=scale,
                                                  **{t: split["y"][f"y_{t}_reg"] for t in ("high", "low", "close")})
                for t in ("high", "low", "close"):
                    np.testing.assert_allclose(got[t], px[t], rtol=1e-6, err_msg=f"{t} scale={scale}")

    def test_collect_signals_prices_and_stamps(self):
        """collect_signals: mu بوحدة العائد (close موقع بلا قسمة) وأسعار pred_* = الأسعار المستقبلية حين mu = الهدف."""
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            ns["train"], ns["val"], ns["test"] = _retarget(ns)
            for idx in (19, 26):
                exec(compile(_cell("main.ipynb", idx), f"main#cell{idx}", "exec"), ns)
            te = ns["test"]
            y = {t: np.concatenate([s["y"][f"y_{t}_reg"] for s in te.values()]) for t in ("high", "low", "close")}
            n = len(y["high"])
            out = {}
            for t in ("high", "low", "close"):
                out.update({f"y_{t}": y[t], f"y_{t}_nu": np.full(n, 1.5), f"y_{t}_alpha": np.full(n, 2.5),
                            f"y_{t}_beta": np.full(n, 1e-3), f"y_{t}_class_logits": np.full(n, 0.5)})
            df = ns["collect_signals"](None, te, "1h", outputs=out)
            np.testing.assert_allclose(df["pred_high"], df["fut_high"], rtol=1e-6)
            np.testing.assert_allclose(df["pred_low"], df["fut_low"], rtol=1e-6)
            np.testing.assert_allclose(df["pred_close"], df["fut_close"], rtol=1e-6)
            np.testing.assert_allclose(df["mu_close"], y["close"], rtol=1e-6)            # موقع، لا ÷ المقياس
            np.testing.assert_allclose(df["mu_high"], y["high"] / scale, rtol=1e-6)
            self.assertEqual(set(df["target_mode"]), {"entry_range"})
            self.assertTrue({"y_high_class", "y_low_class", "y_close_class"} <= set(df.columns))

    def test_chicks_decode_roundtrip(self):
        """EVAL_TARGET_SPECS لـ entry_range: فكّ chicks للأهداف الحقيقية يعيد الأسعار المستقبلية (close بـ range_of)."""
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            ns["train"], ns["val"], ns["test"] = _retarget(ns)
            exec(compile(_cell("main.ipynb", 19), "main#cell19", "exec"), ns)
            src = _cell("main.ipynb", 20)
            exec(compile(src[src.index("import dataclasses"):], "main#cell20", "exec"), ns)
            self.assertEqual([s.name for s in ns["EVAL_TARGET_SPECS"]], ["high", "low", "close"])
            split = ns["test"]["AAA"]
            lc = split["last_candles"]
            raw = {t: split["y"][f"y_{t}_reg"] for t in ("high", "low", "close")}
            dec = ns["decode_predictions_v4"](raw, ns["EVAL_TARGET_SPECS"], None, lc[:, :4])   # بلا أعمدة المستقبل
            _, px = _expected(split)
            for t in ("high", "low", "close"):
                np.testing.assert_allclose(dec[t]["pred_real"], px[t], rtol=1e-6, err_msg=f"{t} scale={scale}")
            # verify_decoding بأعمدة المستقبل: الهدف الحقيقي المفكوك يطابق السعر الفعلي لكل هدف
            dec = ns["decode_predictions_v4"](raw, ns["EVAL_TARGET_SPECS"], None, lc)
            rep = ns["verify_decoding"](raw, dec, ns["EVAL_TARGET_SPECS"], None,
                                        {f"y_{t}": v for t, v in raw.items()}, lc, verbose=False)
            for t in ("high", "low", "close"):
                self.assertEqual(rep[t]["status"], "ok", (t, scale, rep[t]))
                self.assertTrue(rep[t]["true_value_consistent"], (t, scale))
            # عدم اليقين عرضٌ موجب حتى لـ low (reg_scale سالب) ولـ close (مدى متوقَّع)
            ones = np.ones(len(lc))
            raw_u = dict(raw, **{f"{t}_{k}": ones for t in ("low", "close") for k in ("epistemic", "aleatoric", "confidence")})
            dec = ns["decode_predictions_v4"](raw_u, ns["EVAL_TARGET_SPECS"], None, lc)
            self.assertTrue(np.all(dec["low"]["uncertainty_real"] > 0))
            self.assertTrue(np.all(dec["close"]["uncertainty_real"] >= 0))

    def test_other_modes_unchanged(self):
        """الأوضاع الأخرى: نفس صيغها (عائد من سعر النوع، ±1، قصّ ثم ضرب) — لا ختم entry_range عليها."""
        ns = _main_ns(100.0)
        cols = ns["LAST_COLUMNS"]
        for mode in ("return", "return_close", "magnitude", "relative"):
            tr = ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode=mode, verbose=False)[0]
            lc = tr["last_candles"]
            col = lambda k: lc[:, cols.index(k)]
            self.assertEqual(tr["target_mode"], mode)
            self.assertNotIn("class_threshold", tr)
            self.assertEqual(tr["reg_target_scales"], {"high": 100.0, "low": 100.0, "close": 100.0})
            for t, fut, last in (("high", "future_high_max", "last_high"), ("low", "future_low_min", "last_low"),
                                 ("close", "future_close", "last_close")):
                if mode == "return":
                    r = col(fut) / col(last) - 1
                elif mode == "return_close":
                    r = col(fut) / col("last_close") - 1
                elif mode == "magnitude":
                    r0 = col(fut) / col("last_close") - 1
                    r = {"close": np.abs(r0), "high": np.maximum(r0, 0), "low": np.maximum(-r0, 0)}[t]
                else:
                    import pandas as pd
                    r0 = col(fut) / col(last) - 1
                    r = r0 - pd.Series(r0).groupby(col("timestamp")).transform("median").to_numpy()
                np.testing.assert_allclose(tr["y"][f"y_{t}_reg"], np.clip(r, -1, 1) * 100.0, rtol=1e-5, atol=1e-5)
                if mode != "magnitude":
                    np.testing.assert_array_equal(tr["y"][f"y_{t}_class"], np.where(r > 0, 1.0, -1.0))
                else:
                    self.assertEqual(set(np.unique(tr["y"][f"y_{t}_class"])) - {1.0, -1.0}, set())
        s = ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode="scaled", verbose=False)[0]
        self.assertEqual(s["reg_target_scales"], {"high": 1.0, "low": 1.0, "close": 1.0})
        # العودة من entry_range إلى وضع آخر في الجلسة نفسها: لا يبقى ختمه ولا تسمياته 1/0
        er = _retarget(ns)
        back = ns["retarget_splits"](*er, mode="return", verbose=False)[0]
        self.assertNotIn("class_threshold", back)
        self.assertEqual(set(np.unique(back["y"]["y_close_class"])), {-1.0, 1.0})


class EntryRangeConsumerTests(unittest.TestCase):
    """المستهلكون خارج الدفتر: cross_asset (اللوحة والتقرير) وأداة التقييم."""

    def _panel(self, scale):
        from cross_asset.data import panel_split_from
        ns = _main_ns(scale)
        te = _retarget(ns)[2]
        return panel_split_from(te, "1h", targets=("high", "low", "close"), name="test"), te

    def test_export_signals_prices(self):
        from cross_asset.train import asym_score, export_signals
        for scale in (1.0, 100.0):
            ps, _ = self._panel(scale)
            self.assertEqual(ps.target_mode, "entry_range")
            np.testing.assert_array_equal(ps.target_scale, [scale, scale, 1.0])
            mu = ps.yreg.astype("float64")                                   # النموذج يتنبأ بالهدف تماماً
            df = export_signals(ps, np.zeros_like(mu), mu, "test", targets=ps.targets)
            np.testing.assert_allclose(df["pred_high"], df["fut_high"], rtol=1e-6)
            np.testing.assert_allclose(df["pred_low"], df["fut_low"], rtol=1e-6)
            np.testing.assert_allclose(df["pred_close"], df["fut_close"], rtol=1e-6)
            np.testing.assert_allclose(df["mu_close"], ps.yreg[:, 2], rtol=1e-6)
            self.assertEqual(set(df["target_mode"]), {"entry_range"})
            up, dn = df["fut_high"] / df["entry"] - 1, 1 - df["fut_low"] / df["entry"]
            np.testing.assert_allclose(asym_score(ps, mu), up - dn, rtol=1e-5, atol=1e-7)

    def test_export_signals_other_modes_unchanged(self):
        from cross_asset.data import panel_split_from
        from cross_asset.train import export_signals
        ns = _main_ns(100.0)
        te = ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode="return", verbose=False)[2]
        ps = panel_split_from(te, "1h", targets=("high", "low", "close"), name="test")
        self.assertEqual(ps.target_scale, 100.0)                          # رقم واحد كما كان
        df = export_signals(ps, np.zeros((ps.n, 3)), ps.yreg.astype("float64"), "test", targets=ps.targets)
        np.testing.assert_allclose(df["pred_high"], df["fut_high"], rtol=1e-5)
        np.testing.assert_allclose(df["pred_low"], df["fut_low"], rtol=1e-5)
        self.assertNotIn("pred_close", df)
        self.assertEqual(set(df["target_mode"]), {"return"})

    def test_report_labels_from_stamp(self):
        """ملف إشارات بلا y_*_class: التسميات من عمود target_mode المختوم = تعريف entry_range."""
        from cross_asset.report import prepare
        from cross_asset.train import export_signals
        ps, te = self._panel(100.0)
        df = export_signals(ps, np.zeros((ps.n, 3)), ps.yreg.astype("float64"), "test", targets=ps.targets)
        df["rid"] = np.arange(len(df))
        d = prepare(df.drop(columns=[f"y_{t}_class" for t in ps.targets])).sort_values("rid")
        for t in ps.targets:
            self.assertEqual(df[f"y_{t}_class"].nunique(), 2, t)          # الاختبار يميّز فعلاً
            np.testing.assert_array_equal(d[f"cls_{t}"].to_numpy(), df[f"y_{t}_class"].to_numpy(), err_msg=t)

    def test_tool_target_modes_match_notebook(self):
        from tools.evaluate_trained_model import TARGET_MODE_CHOICES
        src = _cell("main.ipynb", 10)
        bases = eval(re.search(r"^TARGET_MODES = (\(.*?\))", src, re.M).group(1))
        no_rel = eval(re.search(r"^_NO_RELATIVE_BASES = (\(.*?\))", src, re.M).group(1))
        want = set(bases) | {"relative"} | {f"{b}+relative" for b in bases if b not in no_rel}
        self.assertEqual(set(TARGET_MODE_CHOICES), want)

    def test_suspended_targets_follow_mode(self):
        """القسم ٤: close مُفعَّل في entry_range (موقع في المدى)، ومعلّق في غيره (اتجاه)."""
        src = _cell("main.ipynb", 13)
        src = src[:src.index("# ⚠️ OrderedMeans")]
        for mode, want in ((None, ["high", "low"]), ("return", ["high", "low"]),
                           ("entry_range", ["high", "low", "close"])):
            ns = {"dataset": {"window_sizes": {"1h": 32}, "feature_order": ["a"]}, "MODEL_TF": "1h",
                  "CONFIG": {"targets": ["high", "low", "close"]}, "TARGET_MODE": mode}
            exec(compile(src, "main#cell13", "exec"), ns)
            self.assertEqual(ns["PRICE_TARGETS"], want, mode)


if __name__ == "__main__":
    unittest.main()

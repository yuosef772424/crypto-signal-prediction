"""وضع الهدف entry_range (main القسم ٣-ب): الاتجاه في رؤوس التصنيف، والمقدار من سعر الدخول P = last_close في الانحدار.

    التصنيف (1/0) اتجاه من نوع الهدف نفسه كخط الأنابيب: القمة القادمة > آخر قمة، القاع > آخر قاع، الإغلاق > آخر إغلاق.
    high_reg = max(H/P − 1, 0)، low_reg = max(1 − L/P, 0)، close_reg = |C/P − 1| (ENTRY_CLOSE_REG = "abs_return")
    أو موقع الإغلاق في المدى (C − L)/(H − L) بلا مقياس ("range_pos")؛ القصّ بوحدة العائد ثم × المقياس. لا قيمة سالبة.

تُنفَّذ خلايا الدفترين نفسها (لا نسخاً منها) كما في test_reg_target_scale، ببيانات تركيبية فيها فجوة صعود (كل الأفق فوق
الدخول) وفجوة هبوط (كله تحته) وشمعة مداها صفر وحركة تتجاوز القصّ:
    python -m unittest tests.test_entry_range -v
"""
import json
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_reg_target_scale import _cell, _ns, _nbload  # noqa: E402 — نطاق خط الأنابيب وchicks المشترك (يُحمَّل مرّة)

# صفوف الحالات الحدّية في كل قسم: فجوة صعود (L > P)، فجوة هبوط (H < P)، مدى صفر (فوق P)، حركة أكبر من القصّ (+200%)
UP_GAP, DOWN_GAP, ZERO, HUGE = 0, 1, 2, 3
REGS = ("abs_return", "range_pos")


def _split(ns, n, seed, scale):
    """قسم بشكل split_data: last_candles حقيقية، وأهداف خط الأنابيب الافتراضية (عائد نوعه × scale، تصنيف 1/0)."""
    rng = np.random.default_rng(seed)
    cols = ns["LAST_COLUMNS"]
    P = 100 * np.exp(rng.normal(0, 0.3, n))
    H = P * (1 + np.abs(rng.normal(0.004, 0.01, n)))
    L = P * (1 - np.abs(rng.normal(0.004, 0.01, n)))
    Cl = L + rng.uniform(0, 1, n) * (H - L)
    H[UP_GAP], L[UP_GAP], Cl[UP_GAP] = P[UP_GAP] * 1.004, P[UP_GAP] * 1.001, P[UP_GAP] * 1.003
    H[DOWN_GAP], L[DOWN_GAP], Cl[DOWN_GAP] = P[DOWN_GAP] * 0.98, P[DOWN_GAP] * 0.95, P[DOWN_GAP] * 0.97
    H[ZERO] = L[ZERO] = Cl[ZERO] = P[ZERO] * 1.003
    H[HUGE], L[HUGE], Cl[HUGE] = P[HUGE] * 3.0, P[HUGE] * 0.999, P[HUGE] * 3.0
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
    """خلايا main (٣ الختم، ٣-ب) على بيانات بمقياس scale."""
    ns = dict(_ns())
    tr, va = _split(ns, 64, 0, scale), _split(ns, 32, 1, scale)
    te = {"AAA": _split(ns, 24, 2, scale), "BBB": _split(ns, 24, 3, scale)}
    ns.update(REG_TARGET_SCALE=scale, MODEL_TF="1h", PRICE_TARGETS=list(price_targets),
              split_data=lambda dataset, config=None: (tr, va, te), dataset={})
    for idx in ("splits", "retarget"):
        exec(compile(_cell("main.ipynb", idx), f"main#{idx}", "exec"), ns)
    return ns


def _prices(split):
    lc, cols = split["last_candles"], _ns()["LAST_COLUMNS"]
    g = lambda k: lc[:, cols.index(k)]
    return {"P": g("last_close"), "H": g("future_high_max"), "L": g("future_low_min"), "C": g("future_close"),
            "last_high": g("last_high"), "last_low": g("last_low")}


def _expected(split, close_reg="abs_return"):
    """أهداف الانحدار بوحدة العائد (قبل × المقياس، بعد القصّ عند 1) وتصنيفات نوعها."""
    p = _prices(split)
    pos = np.where(p["H"] > p["L"], (p["C"] - p["L"]) / np.where(p["H"] > p["L"], p["H"] - p["L"], 1.0), 0.5)
    reg = {"high": np.clip(np.maximum(p["H"] / p["P"] - 1, 0), 0, 1), "low": np.clip(np.maximum(1 - p["L"] / p["P"], 0), 0, 1),
           "close": np.clip(np.abs(p["C"] / p["P"] - 1), 0, 1) if close_reg == "abs_return" else pos}
    cls = {"high": p["H"] > p["last_high"], "low": p["L"] > p["last_low"], "close": p["C"] > p["P"]}
    return reg, cls


def _reachable(P, H, L):
    """أسعار high/low التي يمكن للهدف أن يعيدها: الهدف مصفَّر عند الفجوة (max(·، 0)) ومقصوص عند 1 (+100%)."""
    return P * (1 + np.clip(np.maximum(H / P - 1, 0), 0, 1)), P * (1 - np.clip(np.maximum(1 - L / P, 0), 0, 1))


def _retarget(ns, **kw):
    return ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode="entry_range", verbose=False, **kw)


def _splits_of(out):
    return (out[0], out[1], *out[2].values())


class EntryRangeTargetTests(unittest.TestCase):
    def test_targets_are_nonnegative_magnitudes_from_entry(self):
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            for split in _splits_of(_retarget(ns)):
                want, _ = _expected(split)
                for t in ("high", "low", "close"):
                    y = split["y"][f"y_{t}_reg"]
                    self.assertTrue(np.all(y >= 0), f"{t} scale={scale}: قيمة سالبة")
                    np.testing.assert_allclose(y, want[t] * scale, rtol=1e-6, atol=1e-7 * scale, err_msg=t)
                self.assertEqual(split["target_mode"], "entry_range")
                self.assertEqual(split["entry_close_reg"], "abs_return")
                self.assertEqual(split["reg_target_scale"], scale)
                self.assertEqual(split["reg_target_scales"], {"high": scale, "low": scale, "close": scale})
                self.assertNotIn("class_threshold", split)
                self.assertEqual(ns["reg_scale_of"](split, "close"), scale)
                self.assertEqual(ns["entry_close_reg_of"](split), "abs_return")
                y = {t: split["y"][f"y_{t}_reg"] / scale for t in ("high", "low", "close")}
                # فجوة الصعود (L > P): low = 0 لا −0.1%، وhigh وclose موجبان. فجوة الهبوط (H < P): high = 0، low وclose من P
                self.assertAlmostEqual(float(y["low"][UP_GAP]), 0.0, places=7)
                self.assertAlmostEqual(float(y["high"][UP_GAP]), 0.004, places=6)
                self.assertAlmostEqual(float(y["close"][UP_GAP]), 0.003, places=6)
                self.assertAlmostEqual(float(y["high"][DOWN_GAP]), 0.0, places=7)
                self.assertAlmostEqual(float(y["low"][DOWN_GAP]), 0.05, places=6)
                self.assertAlmostEqual(float(y["close"][DOWN_GAP]), 0.03, places=6)   # |−3%|: الاتجاه ليس هنا
                # القصّ بوحدة العائد ثم المقياس: +200% ← 1 × scale (لا 2 × scale)
                self.assertAlmostEqual(float(y["high"][HUGE]), 1.0, places=6)
                self.assertAlmostEqual(float(y["close"][HUGE]), 1.0, places=6)

    def test_close_reg_range_pos_option(self):
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            for how in ("kw", "global"):
                if how == "global":
                    ns["ENTRY_CLOSE_REG"] = "range_pos"
                out = _retarget(ns, **({"close_reg": "range_pos"} if how == "kw" else {}))
                for split in _splits_of(out):
                    want, _ = _expected(split, "range_pos")
                    y = split["y"]
                    np.testing.assert_allclose(y["y_close_reg"], want["close"], rtol=1e-6, atol=1e-7)   # بلا مقياس
                    self.assertTrue(np.all((y["y_close_reg"] >= 0) & (y["y_close_reg"] <= 1)))
                    for t in ("high", "low"):
                        self.assertTrue(np.all(y[f"y_{t}_reg"] >= 0))
                        np.testing.assert_allclose(y[f"y_{t}_reg"], want[t] * scale, rtol=1e-6, atol=1e-7 * scale)
                    self.assertEqual(split["entry_close_reg"], "range_pos")
                    self.assertEqual(split["reg_target_scales"], {"high": scale, "low": scale, "close": 1.0})
                    self.assertEqual(ns["reg_scale_of"](split, "close"), 1.0)
                    self.assertAlmostEqual(float(y["y_close_reg"][ZERO]), 0.5, places=6)        # مدى صفر
                    self.assertAlmostEqual(float(y["y_close_reg"][DOWN_GAP]), 2 / 3, places=5)  # (C−L)/(H−L)
        ns = _main_ns(100.0)
        with self.assertRaises(ValueError):
            _retarget(ns, close_reg="pos")

    def test_defaults_and_removed_threshold(self):
        ns = _main_ns(1.0)
        self.assertEqual(ns["ENTRY_CLOSE_REG"], "abs_return")
        self.assertEqual(tuple(ns["ENTRY_CLOSE_REGS"]), REGS)
        self.assertNotIn("ENTRY_RANGE_CLASS_THRESHOLD", ns)
        with open(os.path.join(ROOT, "main.ipynb"), encoding="utf-8") as f:
            for i, c in enumerate(json.load(f)["cells"]):
                self.assertNotIn("ENTRY_RANGE_CLASS_THRESHOLD", "".join(c["source"]), f"main cell {i}")
                self.assertNotIn("class_threshold", "".join(c["source"]), f"main cell {i}")
        for fn in sorted(os.listdir(os.path.join(ROOT, "workflow"))):          # main's code now lives in workflow/
            if fn.endswith(".py"):
                with open(os.path.join(ROOT, "workflow", fn), encoding="utf-8") as f:
                    text = f.read()
                self.assertNotIn("ENTRY_RANGE_CLASS_THRESHOLD", text, f"workflow/{fn}")
                self.assertNotIn("class_threshold", text, f"workflow/{fn}")
        import cross_asset.report as report
        self.assertFalse(hasattr(report, "ENTRY_RANGE_CLASS_THRESHOLD"))

    def test_classes_are_same_type_pipeline_directions(self):
        for close_reg in REGS:
            ns = _main_ns(100.0)
            pipeline_y = ns["train"]["y"]        # أهداف خط الأنابيب الافتراضية قبل إعادة الحساب
            out = _retarget(ns, close_reg=close_reg)
            tr = out[0]
            _, want = _expected(tr)
            y = tr["y"]
            self.assertEqual(set(np.unique(np.r_[y["y_high_class"], y["y_low_class"], y["y_close_class"]])), {0.0, 1.0})
            for t in ("high", "low", "close"):
                np.testing.assert_array_equal(y[f"y_{t}_class"], want[t].astype("float32"), err_msg=t)
                np.testing.assert_array_equal(y[f"y_{t}_class"], pipeline_y[f"y_{t}_class"], err_msg=f"{t} vs pipeline")
            # فجوة الصعود: القمة تحت آخر قمة (1.004P < 1.01P) فلا صعود high رغم أنها فوق الدخول؛ القاع والإغلاق فوق سابقيهما
            self.assertEqual([y[f"y_{t}_class"][UP_GAP] for t in ("high", "low", "close")], [0.0, 1.0, 1.0])
            self.assertEqual([y[f"y_{t}_class"][DOWN_GAP] for t in ("high", "low", "close")], [0.0, 0.0, 0.0])
            self.assertEqual([y[f"y_{t}_class"][HUGE] for t in ("high", "low", "close")], [1.0, 1.0, 1.0])

    def test_close_recomputed_even_when_suspended(self):
        """إعادة الخلية بعد القسم ٤ (PRICE_TARGETS بلا close) لا تُبقي y_close بوضع سابق تحت ختم entry_range."""
        ns = _main_ns(100.0, price_targets=("high", "low"))
        tr = _retarget(ns)[0]
        np.testing.assert_allclose(tr["y"]["y_close_reg"], _expected(tr)[0]["close"] * 100, rtol=1e-6, atol=1e-5)

    def test_relative_composition_refused(self):
        ns = _main_ns(1.0)
        with self.assertRaises(ValueError):
            ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode="entry_range+relative", verbose=False)

    def test_roundtrip_targets_to_prices(self):
        """high وlow يعودان إلى سعر الأفق تماماً (ما دام الهدف لم يُصفَّر بـ max(·،0) ولم يُقصّ)، وclose تماماً بالإشارة الحقيقية."""
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            for split in _retarget(ns)[2].values():
                p = _prices(split)
                y = {t: split["y"][f"y_{t}_reg"].astype("float64") / scale for t in ("high", "low", "close")}
                up = p["C"] > p["P"]
                got = ns["entry_range_to_prices"](p["P"], high=y["high"], low=y["low"], close=y["close"],
                                                  p_close_up=np.where(up, 0.9, 0.1))
                ok_h = (p["H"] > p["P"]) & (p["H"] / p["P"] - 1 < 1)
                ok_l = (p["L"] < p["P"]) & (1 - p["L"] / p["P"] < 1)
                ok_c = np.abs(p["C"] / p["P"] - 1) < 1
                self.assertGreater(ok_h.sum(), 10)
                np.testing.assert_allclose(got["high"][ok_h], p["H"][ok_h], rtol=1e-6, err_msg=f"high scale={scale}")
                np.testing.assert_allclose(got["low"][ok_l], p["L"][ok_l], rtol=1e-6, err_msg=f"low scale={scale}")
                np.testing.assert_allclose(got["close"][ok_c], p["C"][ok_c], rtol=1e-6, err_msg=f"close scale={scale}")
                # الفجوة: الهدف صفر فيعود السعر إلى سعر الدخول (لا إلى قمة تحته ولا قاع فوقه)
                np.testing.assert_allclose(got["high"][DOWN_GAP], p["P"][DOWN_GAP], rtol=1e-6)
                np.testing.assert_allclose(got["low"][UP_GAP], p["P"][UP_GAP], rtol=1e-6)
                # الاتجاهان معاً، والصحيح منهما هو الإغلاق الفعلي
                np.testing.assert_allclose(got["close_up"], p["P"] * (1 + y["close"]), rtol=1e-12)
                np.testing.assert_allclose(got["close_down"], p["P"] * (1 - y["close"]), rtol=1e-12)
                true_side = np.where(up, got["close_up"], got["close_down"])
                np.testing.assert_allclose(true_side[ok_c], p["C"][ok_c], rtol=1e-6)
                # p_close_up = 0.5 بالضبط ← الاتجاه صعود؛ وبلا p_close_up لا "close"
                tie = ns["entry_range_to_prices"](p["P"], close=y["close"], p_close_up=np.full(len(up), 0.5))
                np.testing.assert_array_equal(tie["close"], tie["close_up"])
                self.assertNotIn("close", ns["entry_range_to_prices"](p["P"], close=y["close"]))

    def test_roundtrip_range_pos(self):
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            for split in _retarget(ns, close_reg="range_pos")[2].values():
                p = _prices(split)
                y = {t: split["y"][f"y_{t}_reg"].astype("float64") / ns["reg_scale_of"](split, t)
                     for t in ("high", "low", "close")}
                got = ns["entry_range_to_prices"](p["P"], close_reg="range_pos", **y)
                ok = (p["H"] > p["P"]) & (p["L"] < p["P"]) & (p["H"] / p["P"] < 2)
                self.assertGreater(ok.sum(), 10)
                np.testing.assert_allclose(got["close"][ok], p["C"][ok], rtol=1e-6, err_msg=f"scale={scale}")
                self.assertNotIn("close_up", got)                                 # لا اتجاه في هذا التعريف
                self.assertNotIn("close", ns["entry_range_to_prices"](p["P"], close=y["close"], close_reg="range_pos"))

    def test_real_price_predictions(self):
        """real_price_predictions (٧-أ): النموذج يُخرج الهدف المقاس ورأس التصنيف ← السعر الفعلي، لكل تعريف close."""
        class _T:
            def __init__(self, a):
                self.a = np.asarray(a)

            def numpy(self):
                return self.a

        for close_reg in REGS:
            for scale in (1.0, 100.0):
                ns = _main_ns(scale)
                ns["train"], ns["val"], ns["test"] = _retarget(ns, close_reg=close_reg)
                exec(compile(_cell("main.ipynb", "reports"), "main#reports", "exec"), ns)
                split = ns["test"]["AAA"]
                p = _prices(split)
                out = {f"y_{t}": _T(split["y"][f"y_{t}_reg"].reshape(-1, 1)) for t in ("high", "low", "close")}
                out["y_close_class_logits"] = _T(np.where(p["C"] > p["P"], 0.9, 0.1).reshape(-1, 1))
                ns["model"] = lambda x, training=False: out
                ok = (p["H"] > p["P"]) & (p["L"] < p["P"]) & (np.abs(p["C"] / p["P"] - 1) < 1) & (p["H"] / p["P"] < 2)
                for t, want in (("high", p["H"]), ("low", p["L"]), ("close", p["C"])):
                    got = ns["real_price_predictions"]("AAA", t)
                    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-6, err_msg=f"{t} {close_reg} {scale}")
                if close_reg == "abs_return":
                    self.assertEqual(len(ns["real_price_predictions"]("AAA", "close_up")), len(p["P"]))
                    del out["y_close_class_logits"]
                    with self.assertRaises(ValueError):
                        ns["real_price_predictions"]("AAA", "close")           # الاتجاه بلا رأس تصنيف: لا تخمين

    def test_collect_signals_prices_and_stamps(self):
        """collect_signals: mu بوحدة العائد، وأسعار pred_* من P، وpred_close بإشارة p_up_close، وختم التعريف."""
        for close_reg in REGS:
            for scale in (1.0, 100.0):
                ns = _main_ns(scale)
                ns["train"], ns["val"], ns["test"] = _retarget(ns, close_reg=close_reg)
                for idx in ("chicks_bridge", "selective_eval"):
                    exec(compile(_cell("main.ipynb", idx), f"main#{idx}", "exec"), ns)
                te = ns["test"]
                y = {t: np.concatenate([s["y"][f"y_{t}_reg"] for s in te.values()]) for t in ("high", "low", "close")}
                cls = np.concatenate([s["y"]["y_close_class"] for s in te.values()])
                n = len(y["high"])
                out = {}
                for t in ("high", "low", "close"):
                    out.update({f"y_{t}": y[t], f"y_{t}_nu": np.full(n, 1.5), f"y_{t}_alpha": np.full(n, 2.5),
                                f"y_{t}_beta": np.full(n, 1e-3), f"y_{t}_class_logits": np.full(n, 0.5)})
                out["y_close_class_logits"] = np.where(cls > 0, 0.9, 0.1)
                df = ns["collect_signals"](None, te, "1h", outputs=out)
                P, H, L, C = (df[k].to_numpy() for k in ("entry", "fut_high", "fut_low", "fut_close"))
                want_h, want_l = _reachable(P, H, L)
                np.testing.assert_allclose(df["pred_high"], want_h, rtol=1e-6)
                np.testing.assert_allclose(df["pred_low"], want_l, rtol=1e-6)
                np.testing.assert_allclose(df["mu_high"], y["high"] / scale, rtol=1e-6)
                self.assertEqual(set(df["target_mode"]), {"entry_range"})
                self.assertEqual(set(df["entry_close_reg"]), {close_reg})
                self.assertTrue({"y_high_class", "y_low_class", "y_close_class"} <= set(df.columns))
                ok = (np.abs(C / P - 1) < 1) & (H > P) & (L < P) & (H / P < 2)
                self.assertGreater(ok.sum(), 10)
                np.testing.assert_allclose(df["pred_close"][ok], C[ok], rtol=1e-6)
                if close_reg == "abs_return":
                    np.testing.assert_allclose(df["mu_close"], y["close"] / scale, rtol=1e-6)
                    np.testing.assert_allclose(df["pred_close_up"], P * (1 + df["mu_close"]), rtol=1e-12)
                    np.testing.assert_allclose(df["pred_close_down"], P * (1 - df["mu_close"]), rtol=1e-12)
                    self.assertTrue(np.all(df["pred_close_up"] >= P) and np.all(df["pred_close_down"] <= P))
                else:
                    np.testing.assert_allclose(df["mu_close"], y["close"], rtol=1e-6)            # موقع، لا ÷ المقياس
                    self.assertNotIn("pred_close_up", df.columns)
                # p_up_close = 0.5 بالضبط ← الاتجاه صعود (≥ 0.5)
                if close_reg == "abs_return":
                    out["y_close_class_logits"] = np.full(n, 0.5)
                    df5 = ns["collect_signals"](None, te, "1h", outputs=out)
                    np.testing.assert_array_equal(df5["pred_close"], df5["pred_close_up"])

    def test_chicks_decode_roundtrip(self):
        """EVAL_TARGET_SPECS لـ entry_range: high/low من P دائماً، وclose في الحالتين: موقعاً في المدى (range_pos)
        أو مقداراً باتجاه رأس التصنيف (abs_return — تفاصيله في EntryRangeChicksCloseTests)."""
        for close_reg, names in (("abs_return", ["high", "low", "close"]), ("range_pos", ["high", "low", "close"])):
            for scale in (1.0, 100.0):
                ns = _main_ns(scale)
                ns["train"], ns["val"], ns["test"] = _retarget(ns, close_reg=close_reg)
                for idx in ("chicks_bridge", "text:RETURN_PRICE_TARGETS"):
                    src = _cell("main.ipynb", idx)
                    src = src[src.index("import dataclasses"):] if idx.startswith("text:") else src
                    exec(compile(src, f"main#{idx}", "exec"), ns)
                self.assertEqual(ns["CHICKS_TARGETS"], names)
                self.assertEqual([s.name for s in ns["EVAL_TARGET_SPECS"]], names)
                split = ns["test"]["AAA"]
                lc, p = split["last_candles"], _prices(split)
                raw = {t: split["y"][f"y_{t}_reg"] for t in names}
                if close_reg == "abs_return":
                    raw["close_p_up"] = np.where(p["C"] > p["P"], 0.9, 0.1)     # اتجاه close من رأس التصنيف
                dec = ns["decode_predictions_v4"](raw, ns["EVAL_TARGET_SPECS"], None, lc[:, :4])   # بلا أعمدة المستقبل
                want_h, want_l = _reachable(p["P"], p["H"], p["L"])
                np.testing.assert_allclose(dec["high"]["pred_real"], want_h, rtol=1e-6)
                np.testing.assert_allclose(dec["low"]["pred_real"], want_l, rtol=1e-6)
                if close_reg == "range_pos":
                    ok = (p["H"] > p["P"]) & (p["L"] < p["P"]) & (p["H"] / p["P"] < 2)
                else:
                    ok = np.abs(p["C"] / p["P"] - 1) < 1
                np.testing.assert_allclose(dec["close"]["pred_real"][ok], p["C"][ok], rtol=1e-6)
                # verify_decoding بأعمدة المستقبل: الفكّ سليم، والهدف المقصوص/المصفَّر يظهر كعدم اتساق بنسبته الفعلية
                dec = ns["decode_predictions_v4"](raw, ns["EVAL_TARGET_SPECS"], None, lc)
                rep = ns["verify_decoding"](raw, dec, ns["EVAL_TARGET_SPECS"], None,
                                            {f"y_{t}": v for t, v in raw.items()}, lc, verbose=False)
                checks = [("high", want_h, p["H"]), ("low", want_l, p["L"])]
                if close_reg == "abs_return":   # مقصوص عند ±100% كغيره: الحركة +200% لا تُعاد حرفياً
                    checks.append(("close", p["P"] * (1 + np.where(p["C"] > p["P"], 1, -1)
                                                       * np.clip(np.abs(p["C"] / p["P"] - 1), 0, 1)), p["C"]))
                for t, fut, true in checks:
                    self.assertTrue(rep[t]["roundtrip_ok"], (t, scale))
                    self.assertAlmostEqual(rep[t]["true_value_inconsistent_frac"],
                                           float(np.mean(np.abs(fut - true) / true > 1e-3)), places=9)
                if close_reg == "range_pos":
                    self.assertEqual(rep["close"]["status"], "ok", (scale, rep["close"]))
                # عدم اليقين عرضٌ موجب حتى لـ low (reg_scale سالب)
                ones = np.ones(len(lc))
                raw_u = dict(raw, **{f"{t}_{k}": ones for t in ("low",) for k in ("epistemic", "aleatoric", "confidence")})
                dec = ns["decode_predictions_v4"](raw_u, ns["EVAL_TARGET_SPECS"], None, lc)
                self.assertTrue(np.all(dec["low"]["uncertainty_real"] > 0))

    def test_other_modes_unchanged(self):
        """الأوضاع الأخرى: نفس صيغها (عائد من سعر النوع، ±1، قصّ ثم ضرب) — لا ختم entry_range عليها."""
        ns = _main_ns(100.0)
        cols = ns["LAST_COLUMNS"]
        for mode in ("return", "return_close", "magnitude", "relative"):
            tr = ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode=mode, verbose=False)[0]
            lc = tr["last_candles"]
            col = lambda k: lc[:, cols.index(k)]
            self.assertEqual(tr["target_mode"], mode)
            self.assertNotIn("entry_close_reg", tr)
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
        self.assertNotIn("entry_close_reg", back)
        self.assertEqual(set(np.unique(back["y"]["y_close_class"])), {-1.0, 1.0})
        # entry_range يطابق magnitude في الانحدار (المقدار نفسه) ويختلف عنه في التسمية (اتجاه لا عتبة وسيط)
        mag = ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode="magnitude", verbose=False)[0]
        for t in ("high", "low", "close"):
            np.testing.assert_allclose(er[0]["y"][f"y_{t}_reg"], mag["y"][f"y_{t}_reg"], rtol=1e-6, atol=1e-6)


class EntryRangeChicksCloseTests(unittest.TestCase):
    """chicks في entry_range مع abs_return: close = P·(1 + s·mu/scale) باتجاه رأس التصنيف، وتحليل الأنماط لكل رأس."""

    @staticmethod
    def _chicks_ns(scale, close_reg="abs_return"):
        ns = _main_ns(scale)
        ns["train"], ns["val"], ns["test"] = _retarget(ns, close_reg=close_reg)
        for idx in ("chicks_bridge", "text:RETURN_PRICE_TARGETS"):
            src = _cell("main.ipynb", idx)
            exec(compile(src[src.index("import dataclasses"):] if idx.startswith("text:") else src, f"main#{idx}", "exec"), ns)
        # test_all_assets_v4 (chicks ٨): تحميل الدفتر العام يتخطّاها (تحوي نصّ الاستدعاء)، وهنا هي المسار المختبَر
        _nbload.exec_evaluation_module("all_assets", ns, quiet=False)      # chicks cell 18
        return ns

    @staticmethod
    def _model(ns, p_up, noise_seed=0):
        """نموذج مزيّف يُخرج y_* (المقدار الحقيقي) وy_*_class_logits (احتمال) لكل صفّ يُعرَّف بمعرّفه في X[:, 0, 0]."""
        ids = np.concatenate([np.arange(len(s["last_candles"])) + 1000 * i for i, s in enumerate(ns["test"].values())])
        for i, s in enumerate(ns["test"].values()):
            X = np.zeros_like(s["X_1h"])
            X[:, 0, 0] = np.arange(len(X)) + 1000 * i
            s["X_1h"] = X
        table = {}
        for i, s in enumerate(ns["test"].values()):
            for j in range(len(s["last_candles"])):
                table[j + 1000 * i] = {**{f"y_{t}": s["y"][f"y_{t}_reg"][j] for t in ("high", "low", "close")},
                                       **{f"y_{t}_class_logits": p_up[t][i][j] for t in ("high", "low", "close")}}

        def model(x, training=False):
            xb = x[0] if isinstance(x, (tuple, list)) else x         # predict_batch_v4 يمرّر tuple بنافذة لكل إطار
            rows = [table[int(v)] for v in np.asarray(xb)[:, 0, 0]]
            return {k: np.array([r[k] for r in rows], "float32").reshape(-1, 1) for k in rows[0]}
        model.table = table
        return model

    def test_close_price_uses_sign_from_p_up(self):
        for scale in (1.0, 100.0):
            ns = self._chicks_ns(scale)
            self.assertEqual(ns["CHICKS_TARGETS"], ["high", "low", "close"])
            spec = {s.name: s for s in ns["EVAL_TARGET_SPECS"]}["close"]
            self.assertTrue(spec.signed_by_class and spec.relative_to_entry)
            self.assertEqual(spec.class_key, "y_close_class_logits")
            split = ns["test"]["AAA"]
            p = _prices(split)
            mu = split["y"]["y_close_reg"].astype("float64")
            for name, p_up in (("truth", np.where(p["C"] > p["P"], 0.9, 0.1)), ("all_up", np.full(len(mu), 0.9)),
                               ("all_down", np.full(len(mu), 0.1)), ("boundary", np.full(len(mu), 0.5))):
                s_sign = np.where(p_up >= 0.5, 1.0, -1.0)                     # 0.5 بالضبط ← صعود
                dec = ns["decode_predictions_v4"]({"close": mu, "close_p_up": p_up}, [spec], None, split["last_candles"])
                want = p["P"] * (1.0 + s_sign * mu / scale)
                np.testing.assert_allclose(dec["close"]["pred_real"], want, rtol=1e-12, err_msg=f"{name} {scale}")
                np.testing.assert_allclose(dec["close"]["p_up"], p_up)
                # مطابق لصيغة main الوحيدة (entry_range_to_prices) لا نسخة منها
                np.testing.assert_allclose(dec["close"]["pred_real"], ns["entry_range_to_prices"](
                    p["P"], close=mu / scale, close_reg="abs_return", p_close_up=p_up)["close"], rtol=1e-12)
            # مقدار مشترك، إشارة عكسية: الفرق عن P متناظر
            up = ns["decode_predictions_v4"]({"close": mu, "close_p_up": np.full(len(mu), 0.9)}, [spec], None, split["last_candles"])
            dn = ns["decode_predictions_v4"]({"close": mu, "close_p_up": np.full(len(mu), 0.1)}, [spec], None, split["last_candles"])
            np.testing.assert_allclose(up["close"]["pred_real"] + dn["close"]["pred_real"], 2 * p["P"], rtol=1e-12)
            # بلا احتمال لا تخمين
            with self.assertRaises(ValueError):
                ns["decode_predictions_v4"]({"close": mu}, [spec], None, split["last_candles"])

    def test_class_output_reaches_chicks_through_model_call(self):
        """المسار الكامل: build_chicks_test_dict ← test_all_assets_v4 ← predict_batch_v4 يقرأ y_close_class_logits من النموذج."""
        ns = self._chicks_ns(100.0)
        rng = np.random.default_rng(3)
        ps = {t: [rng.uniform(0.05, 0.95, len(s["last_candles"])) for s in ns["test"].values()] for t in ("high", "low", "close")}
        for i, s in enumerate(ns["test"].values()):                     # close: احتمال يوافق الاتجاه الفعلي بضجيج 25%
            truth = s["y"]["y_close_class"] > 0
            flip = rng.random(len(truth)) < 0.25
            ps["close"][i] = np.where(truth ^ flip, 0.8, 0.2)
        model = self._model(ns, ps)
        td = ns["build_chicks_test_dict"](ns["test"], "1h")
        self.assertEqual(set(td["AAA"]["y"]), {"high", "low", "close"})
        for i, (a, s) in enumerate(td.items()):
            s["X_1h"] = ns["test"][a]["X_1h"]
        res = ns["test_all_assets_v4"](model, td, ["1h"], ns["EVAL_TARGET_SPECS"], verbose=False, batch_size=256)
        ver = res["verification_summary"].query("target == 'close'").set_index("asset")
        self.assertTrue(ver["roundtrip_ok"].all())
        for i, (a, row) in enumerate(res["per_asset_results"].iterrows()):
            split = ns["test"][row["asset"]]
            pr = _prices(split)
            c = row["close"]
            want_sign = np.where(ps["close"][list(ns["test"]).index(row["asset"])] >= 0.5, 1.0, -1.0)
            mu = split["y"]["y_close_reg"].astype("float64")
            np.testing.assert_allclose(c["pred_real"], pr["P"] * (1 + want_sign * mu / 100.0), rtol=1e-6)
            np.testing.assert_allclose(c["true_real"], pr["C"])
            # فكّ الحقيقة (مقدار × اتجاه فعلي) يعيد السعر إلا ما قُصّ عند ±100% (+200% هنا) — نسبته الفعلية لا أكثر
            fut = pr["P"] * (1 + np.where(pr["C"] > pr["P"], 1, -1) * np.clip(np.abs(pr["C"] / pr["P"] - 1), 0, 1))
            self.assertAlmostEqual(ver.loc[row["asset"], "true_value_inconsistent_frac"],
                                   float(np.mean(np.abs(fut - pr["C"]) / pr["C"] > 1e-3)), places=9)
            # الاتجاه الصائب = اتجاه رأس التصنيف مقابل الاتجاه الفعلي (Win% = دقة الرأس)
            np.testing.assert_array_equal(c["direction_correct"], (want_sign == np.where(pr["C"] > pr["P"], 1, -1)).astype(int))
        # رأس التصنيف غائب من النموذج ← خطأ صريح لا اتجاه مخمَّن
        del model.table
        bad = lambda x, training=False: {k: v for k, v in model(x).items() if k != "y_close_class_logits"}
        with self.assertRaises(KeyError):
            ns["predict_batch_v4"](bad, (td["AAA"]["X_1h"],), ns["EVAL_TARGET_SPECS"])

    def test_pattern_discovery_runs_per_head(self):
        ns = self._chicks_ns(100.0)
        rng = np.random.default_rng(4)
        ps = {t: [rng.uniform(0.05, 0.95, len(s["last_candles"])) for s in ns["test"].values()] for t in ("high", "low", "close")}
        model = self._model(ns, ps)
        td = ns["build_chicks_test_dict"](ns["test"], "1h")
        for a, s in td.items():
            s["X_1h"] = ns["test"][a]["X_1h"]
        res = ns["test_all_assets_v4"](model, td, ["1h"], ns["EVAL_TARGET_SPECS"], verbose=False, batch_size=256)
        flat = ns["build_flat_dataframe"](res["per_asset_results"], ns["EVAL_TARGET_SPECS"])
        self.assertEqual(set(flat["target"]), {"high", "low", "close"})
        self.assertTrue(flat["p_up"].notna().all())
        # الفوارق البنيوية بين الرؤوس التي كانت تُضلّل الشجرة المجمَّعة
        self.assertTrue((flat[flat.target == "high"]["predicted_change_pct"] >= 0).all())
        self.assertTrue((flat[flat.target == "low"]["predicted_change_pct"] <= 0).all())
        got = ns["detect_success_failure_patterns"](flat, verbose=False)
        self.assertEqual(got["mode"], "per_head")
        self.assertEqual(sorted(got["per_head"]), ["close", "high", "low"])
        self.assertEqual(sorted(got["summary"]["head"]), ["close", "high", "low"])
        for h, r in got["per_head"].items():
            self.assertEqual(r["head"], h)
            self.assertEqual(r["n"], int((flat.target == h).sum()))
            self.assertIn("p_up", r["features"])
            self.assertIn("p_up", set(r["feature_importance"]["الخاصية"]))
            y = flat[flat.target == h]["correct"]
            self.assertAlmostEqual(r["tree_naive_baseline_accuracy"], max(y.mean(), 1 - y.mean()))
            self.assertTrue(0.0 <= r["tree_balanced_accuracy"] <= 1.0)
        row = got["summary"].set_index("head")
        self.assertEqual(set(row.columns) >= {"naive_baseline_accuracy", "tree_balanced_accuracy"}, True)
        # الخيار المجمَّع: نتيجة واحدة بالحقول القديمة، ولا p_up حين يغيب عن رأس (هنا حاضر في الكل)
        pooled = ns["detect_success_failure_patterns"](flat, per_head=False, verbose=False)
        self.assertEqual(pooled["mode"], "pooled")
        self.assertNotIn("per_head", pooled)
        self.assertEqual(pooled["n"], len(flat))
        # رأس واحد أو بلا عمود target: الحقول القديمة على المستوى الأعلى (يقرؤها selective.py)
        one = ns["detect_success_failure_patterns"](flat[flat.target == "close"].drop(columns="target"), verbose=False)
        self.assertEqual(list(one["per_head"]), ["all"])
        for k in ("tree_accuracy", "tree_balanced_accuracy", "tree_naive_baseline_accuracy", "feature_importance", "tree_rules"):
            self.assertIn(k, one)
        # رأس بلا عيّنات كافية يُسجَّل تخطّيه ولا يُفشل الباقي
        few = pd.concat([flat[flat.target != "low"], flat[flat.target == "low"].head(5)])
        r = ns["detect_success_failure_patterns"](few, verbose=False)
        self.assertEqual(sorted(r["per_head"]), ["close", "high"])
        self.assertEqual(list(r["skipped"]), ["low"])

    def test_other_modes_unchanged(self):
        ns = _main_ns(100.0)                                        # target_mode = None (return)
        for idx in ("chicks_bridge", "text:RETURN_PRICE_TARGETS"):
            src = _cell("main.ipynb", idx)
            exec(compile(src[src.index("import dataclasses"):] if idx.startswith("text:") else src, f"main#{idx}", "exec"), ns)
        self.assertEqual(ns["CHICKS_TARGETS"], ["high", "low", "close"])
        for s in ns["EVAL_TARGET_SPECS"]:
            self.assertIsNone(s.class_key, s.name)
            self.assertFalse(s.signed_by_class, s.name)
        # مواصفة بلا class_key: تتجاهل p_up لو وُجد في raw ولا تغيّر الفكّ
        split = ns["test"]["AAA"]
        lc = split["last_candles"]
        raw = {t: np.random.default_rng(1).normal(0, 1, len(lc)) for t in ("high", "low", "close")}
        a = ns["decode_predictions_v4"](raw, ns["EVAL_TARGET_SPECS"], None, lc)
        b = ns["decode_predictions_v4"](dict(raw, close_p_up=np.zeros(len(lc))), ns["EVAL_TARGET_SPECS"], None, lc)
        for t in raw:
            np.testing.assert_array_equal(a[t]["pred_real"], b[t]["pred_real"])
            self.assertNotIn("direction_sign", a[t])
        # signed_by_class بلا class_key أو بلا فكّ نسبة لسعر الدخول: مرفوض
        with self.assertRaises(ValueError):
            ns["TargetSpec"](name="close", signed_by_class=True, relative_to_entry=True)
        with self.assertRaises(ValueError):
            ns["TargetSpec"](name="close", signed_by_class=True, class_key="y_close_class_logits")
        # p_up كله NaN (أوضاع لا تقرؤه) ← خاصية غير مرشّحة، والباقي كما كان؛ وحضوره الكامل ← مرشّحة
        rng = np.random.default_rng(2)
        n = 80
        df = pd.DataFrame({"asset": "A", "target": "close", "confidence": rng.random(n), "uncertainty": rng.random(n),
                           "aleatoric": rng.random(n), "epistemic": rng.random(n), "predicted_change_pct": rng.normal(0, 1, n),
                           "correct": rng.integers(0, 2, n), "pct_error": rng.random(n), "p_up": np.nan})
        r = ns["detect_success_failure_patterns"](df, verbose=False)
        self.assertEqual(r["features"], ["confidence", "uncertainty", "aleatoric", "epistemic", "predicted_change_pct"])
        df["p_up"] = rng.random(n)
        self.assertEqual(ns["detect_success_failure_patterns"](df, verbose=False)["features"][-1], "p_up")
        # feature_cols صريحة تُحترم كما هي (لا p_up تلقائياً)
        r = ns["detect_success_failure_patterns"](df, feature_cols=["confidence"], verbose=False)
        self.assertEqual(r["features"], ["confidence"])

    def test_range_pos_close_unchanged(self):
        ns = self._chicks_ns(100.0, "range_pos")
        close = {s.name: s for s in ns["EVAL_TARGET_SPECS"]}["close"]
        self.assertEqual(close.range_of, ("high", "low"))
        self.assertFalse(close.signed_by_class)


class EntryRangeConsumerTests(unittest.TestCase):
    """المستهلكون خارج الدفتر: cross_asset (اللوحة والتقرير) وأداة التقييم."""

    def _panel(self, scale, close_reg="abs_return"):
        from cross_asset.data import panel_split_from
        ns = _main_ns(scale)
        te = _retarget(ns, close_reg=close_reg)[2]
        return panel_split_from(te, "1h", targets=("high", "low", "close"), name="test"), te

    def test_export_signals_prices(self):
        from cross_asset.train import asym_score, export_signals
        for close_reg in REGS:
            for scale in (1.0, 100.0):
                ps, _ = self._panel(scale, close_reg)
                self.assertEqual(ps.target_mode, "entry_range")
                self.assertEqual(ps.entry_close_reg, close_reg)
                np.testing.assert_array_equal(ps.target_scale, scale if close_reg == "abs_return" else [scale, scale, 1.0])
                mu = ps.yreg.astype("float64")                                # النموذج يتنبأ بالهدف تماماً
                logit = np.where(ps.ycls > 0, 3.0, -3.0)                      # واثق باتجاه صحيح لكل رأس
                df = export_signals(ps, logit, mu, "test", targets=ps.targets)
                P, H, L, C = (df[k].to_numpy() for k in ("entry", "fut_high", "fut_low", "fut_close"))
                want_h, want_l = _reachable(P, H, L)
                np.testing.assert_allclose(df["pred_high"], want_h, rtol=1e-6)
                np.testing.assert_allclose(df["pred_low"], want_l, rtol=1e-6)
                ok = (np.abs(C / P - 1) < 1) & (H > P) & (L < P) & (H / P < 2)
                self.assertGreater(ok.sum(), 10)
                np.testing.assert_allclose(df["pred_close"][ok], C[ok], rtol=1e-6)
                self.assertEqual(set(df["target_mode"]), {"entry_range"})
                self.assertEqual(set(df["entry_close_reg"]), {close_reg})
                if close_reg == "abs_return":
                    np.testing.assert_allclose(df["mu_close"], ps.yreg[:, 2] / scale, rtol=1e-6)
                    np.testing.assert_allclose(df["pred_close_up"], P * (1 + df["mu_close"]), rtol=1e-12)
                    np.testing.assert_allclose(df["pred_close_down"], P * (1 - df["mu_close"]), rtol=1e-12)
                else:
                    np.testing.assert_allclose(df["mu_close"], ps.yreg[:, 2], rtol=1e-6)         # موقع بلا ÷ المقياس
                    self.assertNotIn("pred_close_up", df.columns)
                up, dn = np.maximum(H / P - 1, 0), np.maximum(1 - L / P, 0)
                np.testing.assert_allclose(asym_score(ps, mu), np.clip(up, 0, 1) - np.clip(dn, 0, 1), rtol=1e-5, atol=1e-7)

    def test_export_signals_other_modes_unchanged(self):
        from cross_asset.data import panel_split_from
        from cross_asset.train import export_signals
        ns = _main_ns(100.0)
        te = ns["retarget_splits"](ns["train"], ns["val"], ns["test"], mode="return", verbose=False)[2]
        ps = panel_split_from(te, "1h", targets=("high", "low", "close"), name="test")
        self.assertEqual(ps.target_scale, 100.0)                          # رقم واحد كما كان
        self.assertIsNone(ps.entry_close_reg)
        df = export_signals(ps, np.zeros((ps.n, 3)), ps.yreg.astype("float64"), "test", targets=ps.targets)
        keep = np.arange(len(df)) % 24 != HUGE     # صف +200% مقصوص عند ±1 في كل الأوضاع (الأقسام بـ 24 صفاً لكل عملة)
        np.testing.assert_allclose(df["pred_high"][keep], df["fut_high"][keep], rtol=1e-5)
        np.testing.assert_allclose(df["pred_low"][keep], df["fut_low"][keep], rtol=1e-5)
        self.assertNotIn("pred_close", df)
        self.assertNotIn("entry_close_reg", df)
        self.assertEqual(set(df["target_mode"]), {"return"})

    def test_report_labels_from_stamp(self):
        """ملف إشارات بلا y_*_class: التسميات من عمود target_mode المختوم = اتجاه من نوع الهدف كخط الأنابيب."""
        from cross_asset.report import prepare
        from cross_asset.train import export_signals
        ps, te = self._panel(100.0)
        df = export_signals(ps, np.zeros((ps.n, 3)), ps.yreg.astype("float64"), "test", targets=ps.targets)
        df["rid"] = np.arange(len(df))
        d = prepare(df.drop(columns=[f"y_{t}_class" for t in ps.targets])).sort_values("rid")
        same_type = {"high": df["fut_high"] > df["last_high"], "low": df["fut_low"] > df["last_low"],
                     "close": df["fut_close"] > df["entry"]}
        for t in ps.targets:
            self.assertEqual(df[f"y_{t}_class"].nunique(), 2, t)          # الاختبار يميّز فعلاً
            np.testing.assert_array_equal(d[f"cls_{t}"].to_numpy(), df[f"y_{t}_class"].to_numpy(), err_msg=t)
            np.testing.assert_array_equal(d[f"cls_{t}"].to_numpy(), same_type[t].astype(int).to_numpy(), err_msg=t)

    def test_tool_target_modes_match_notebook(self):
        from tools.evaluate_trained_model import ENTRY_CLOSE_REG_CHOICES, TARGET_MODE_CHOICES
        ns = {"__name__": "t"}                                       # TARGET_MODES now comes from core/schema.py via workflow/retarget.py
        _nbload.workflow_package().load_into(ns, only=("retarget",))
        bases, no_rel = ns["TARGET_MODES"], ns["_NO_RELATIVE_BASES"]
        self.assertEqual(tuple(ns["ENTRY_CLOSE_REGS"]), tuple(ENTRY_CLOSE_REG_CHOICES))
        want = set(bases) | {"relative"} | {f"{b}+relative" for b in bases if b not in no_rel}
        self.assertEqual(set(TARGET_MODE_CHOICES), want)
        self.assertEqual(set(ENTRY_CLOSE_REG_CHOICES), set(REGS))

    def test_suspended_targets_follow_mode(self):
        """القسم ٤: close مُفعَّل في entry_range (مقدار|موقع)، ومعلّق في غيره (اتجاه)."""
        src = _cell("main.ipynb", "model_build")
        src = src[:src.index("# ⚠️ OrderedMeans")]
        for mode, want in ((None, ["high", "low"]), ("return", ["high", "low"]),
                           ("entry_range", ["high", "low", "close"])):
            ns = {"dataset": {"window_sizes": {"1h": 32}, "feature_order": ["a"]}, "MODEL_TF": "1h",
                  "CONFIG": {"targets": ["high", "low", "close"]}, "TARGET_MODE": mode,
                  "workflow": _nbload.workflow_package()}
            exec(compile(src, "main#model_build", "exec"), ns)
            self.assertEqual(ns["PRICE_TARGETS"], want, mode)


if __name__ == "__main__":
    unittest.main()

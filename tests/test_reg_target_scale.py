"""reg_target_scale في دفتر main وchicks: y_*_reg = عائد × المقياس، وكل تحويل إلى سعر يقسم عليه أولاً.

تُنفَّذ خلايا الدفترين نفسها (لا نسخاً منها) فوق نطاق خط الأنابيب، ببيانات تركيبية صغيرة بلا Drive:
    python -m unittest tests.test_reg_target_scale -v
اختبارات خط الأنابيب نفسه (البناء، القصّ قبل الضرب، الحفظ في البيانات) في run_pipeline_selftests.
"""
import json
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402

_NS = None


def _cell(nb_name, idx):
    """Source of a notebook code cell (magics dropped). ``idx``: 0-based cell index, or for notebooks whose code moved to a package
    (main.ipynb) a section key: a workflow module name (the cell whose ``workflow.load_into(...)`` line loads it) or
    ``"text:<substring>"`` (the first code cell containing it). Keys survive cell insertions/removals, unlike indices."""
    with open(os.path.join(ROOT, nb_name), encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    if isinstance(idx, str):
        marker = idx[5:] if idx.startswith("text:") else f'"{idx}"'
        hits = [c for c in cells if c["cell_type"] == "code" and (
            marker in "".join(c["source"]) if idx.startswith("text:") else
            any(ln.startswith("workflow.load_into(") and marker in ln for ln in "".join(c["source"]).splitlines()))]
        assert hits, f"no code cell of {nb_name} matches {idx!r}"
        cell = hits[0]
    else:
        cell = cells[idx]
    src = "".join(cell["source"])
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%")))


def _split(ns, n, seed):
    """قسم بنفس شكل split_data: last_candles حقيقية وأهداف return بمقياس 1."""
    rng = np.random.default_rng(seed)
    cols = ns["LAST_COLUMNS"]
    lc = np.zeros((n, len(cols)))
    close = 100 * np.exp(rng.normal(0, 0.3, n))
    lc[:, cols.index("last_close")] = close
    lc[:, cols.index("last_high")] = close * 1.01
    lc[:, cols.index("last_low")] = close * 0.99
    lc[:, cols.index("timestamp")] = (np.arange(n) // 4) * 8 * 3600 * 10**9 + 1_700_000_000 * 10**9
    r = rng.normal(0, 0.012, n)
    lc[:, cols.index("future_close")] = close * (1 + r)
    lc[:, cols.index("future_high_max")] = close * 1.01 * (1 + np.abs(r))
    lc[:, cols.index("future_low_min")] = close * 0.99 * (1 - np.abs(r))
    y = {}
    for t, fut, last in (("high", "future_high_max", "last_high"), ("low", "future_low_min", "last_low"),
                         ("close", "future_close", "last_close")):
        ret = lc[:, cols.index(fut)] / lc[:, cols.index(last)] - 1
        y[f"y_{t}_reg"], y[f"y_{t}_class"] = ret.astype("float32"), (ret > 0).astype("float32")
    return {"X_1h": np.zeros((n, 4, 2), "float32"), "last_candles": lc, "y": y,
            "base_params": np.c_[close, close * 0.02].astype("float32")}


def _ns():
    """نطاق خط الأنابيب + خلايا main (٣، ٣-ب، ٦، ٧-ب) + تعريفات chicks — مرّة واحدة لكل الاختبارات."""
    global _NS
    if _NS is None:
        ns = _nbload.load_pipeline()
        _nbload.load_evaluation(ns=ns, exclude=("all_assets", "live", "full_analysis"))   # = chicks cells 18, 20, 36 (skipped as before)
        ns["workflow"] = _nbload.workflow_package()      # main's cells start with workflow.load_into(globals(), only=...)
        _NS = ns
    return _NS


def _main_ns(scale):
    """يشغّل خلايا main على بيانات مبنية بالمقياس scale (y = عائد × scale)."""
    base = _ns()
    ns = dict(base)
    tr, va = _split(base, 64, 0), _split(base, 32, 1)
    te = {"AAA": _split(base, 24, 2), "BBB": _split(base, 24, 3)}
    for s in (tr, va, *te.values()):
        for k in list(s["y"]):
            if k.endswith("_reg"):
                s["y"][k] = s["y"][k] * scale
    ns.update(REG_TARGET_SCALE=scale, MODEL_TF="1h", PRICE_TARGETS=["high", "low", "close"],
              split_data=lambda dataset, config=None: (tr, va, te), dataset={})
    for idx in ("splits", "retarget", "chicks_bridge", "selective_eval"):   # ختم الأقسام، retarget_splits، build_chicks_test_dict، collect_signals
        exec(compile(_cell("main.ipynb", idx), f"main#{idx}", "exec"), ns)
    return ns


class MainRegScaleTests(unittest.TestCase):
    def test_splits_stamped_from_dataset(self):
        ns = _main_ns(100.0)
        self.assertEqual(ns["reg_scale_of"](ns["train"]), 100.0)
        self.assertEqual(ns["reg_scale_of"](ns["test"]), 100.0)
        self.assertEqual(ns["reg_scale_of"]({"y": {}}), 1.0)        # قسم قديم بلا المفتاح

    def test_retarget_scales_return_modes_after_clip(self):
        ns1, ns100 = _main_ns(1.0), _main_ns(100.0)
        for mode in ("return", "relative", "magnitude"):
            a = ns1["retarget_splits"](ns1["train"], ns1["val"], ns1["test"], mode=mode, verbose=False)[0]
            b = ns100["retarget_splits"](ns100["train"], ns100["val"], ns100["test"], mode=mode, verbose=False)[0]
            self.assertEqual(b["reg_target_scale"], 100.0)
            np.testing.assert_allclose(b["y"]["y_close_reg"] / 100.0, a["y"]["y_close_reg"], rtol=1e-5, atol=1e-8)
            np.testing.assert_array_equal(b["y"]["y_close_class"], a["y"]["y_close_class"])
        s = ns100["retarget_splits"](ns100["train"], ns100["val"], ns100["test"], mode="scaled", verbose=False)[0]
        self.assertEqual(s["reg_target_scale"], 1.0)                   # وحدة التقلّب: لا ضرب
        # القصّ بوحدة العائد قبل الضرب: قفزة +300% تُقصّ عند 1 ثم تصبح 100
        tr = dict(ns100["train"])
        tr["last_candles"] = tr["last_candles"].copy()
        lc_cols = ns100["LAST_COLUMNS"]
        tr["last_candles"][0, lc_cols.index("future_close")] = 4 * tr["last_candles"][0, lc_cols.index("last_close")]
        c = ns100["retarget_splits"](tr, ns100["val"], ns100["test"], mode="return", verbose=False)[0]
        self.assertAlmostEqual(float(c["y"]["y_close_reg"][0]), 100.0, places=4)

    def test_chicks_relative_future_prices_invariant(self):
        ns1, ns100 = _main_ns(1.0), _main_ns(100.0)
        a = ns1["retarget_splits"](ns1["train"], ns1["val"], ns1["test"], mode="relative", verbose=False)[2]
        b = ns100["retarget_splits"](ns100["train"], ns100["val"], ns100["test"], mode="relative", verbose=False)[2]
        da, db = ns1["build_chicks_test_dict"](a, "1h"), ns100["build_chicks_test_dict"](b, "1h")
        for asset in da:
            np.testing.assert_allclose(db[asset]["last_candles"], da[asset]["last_candles"], rtol=1e-6)

    def test_chicks_decode_roundtrip_with_scale(self):
        """سعر ← هدف ← فكّ chicks == السعر المستقبلي، بمقياس 1 و100 (نفس specs التي يبنيها main)."""
        for scale in (1.0, 100.0):
            ns = _main_ns(scale)
            src = _cell("main.ipynb", "text:RETURN_PRICE_TARGETS")
            exec(compile(src[src.index("import dataclasses"):], "main#specs", "exec"), ns)   # بلا %run chicks
            split = ns["test"]["AAA"]
            raw = {t: split["y"][f"y_{t}_reg"] for t in ns["PRICE_TARGETS"]}
            lc = split["last_candles"].copy()
            want = {t: lc[:, s.future_col].copy() for t, s in ((s.name, s) for s in ns["EVAL_TARGET_SPECS"])}
            dec = ns["decode_predictions_v4"](raw, ns["EVAL_TARGET_SPECS"], None, lc[:, :4])   # بلا أعمدة المستقبل
            for t in ns["PRICE_TARGETS"]:
                np.testing.assert_allclose(dec[t]["pred_real"], want[t], rtol=1e-5, err_msg=f"{t} scale={scale}")

    def test_collect_signals_identical_for_scaled_model_output(self):
        ns1, ns100 = _main_ns(1.0), _main_ns(100.0)
        rng = np.random.default_rng(0)
        n = sum(len(s["last_candles"]) for s in ns1["test"].values())
        out = {}
        for t in ("high", "low", "close"):
            out.update({f"y_{t}": rng.normal(0, 0.01, n), f"y_{t}_nu": rng.uniform(1, 2, n),
                        f"y_{t}_alpha": rng.uniform(2, 3, n), f"y_{t}_beta": rng.uniform(1e-4, 1e-3, n)})
        # نفس التوزيع بوحدة الهدف المضروبة: mu × s، beta × s² (عرض Student-t يتناسب مع s)
        scaled = {k: v * (100.0 if k.count("_") == 1 else 1e4 if k.endswith("_beta") else 1.0) for k, v in out.items()}
        a = ns1["collect_signals"](None, ns1["test"], "1h", outputs=out, price_targets=ns1["PRICE_TARGETS"])
        b = ns100["collect_signals"](None, ns100["test"], "1h", outputs=scaled, price_targets=ns100["PRICE_TARGETS"])
        for c in ("pred_high", "pred_low", "mu_close", "wst_close"):
            np.testing.assert_allclose(b[c].to_numpy(), a[c].to_numpy(), rtol=1e-9, err_msg=c)

    def test_invert_reg_predictions_reads_scale(self):
        ns = _main_ns(100.0)
        split = ns["test"]["AAA"]
        got = ns["invert_reg_predictions"](split["y"]["y_close_reg"], "close_reg", last_candles=split["last_candles"],
                                           config=ns["CONFIG"], scale=ns["reg_scale_of"](split))
        np.testing.assert_allclose(got, split["last_candles"][:, ns["LAST_COLUMNS"].index("future_close")], rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

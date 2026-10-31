"""Output-name consumers (model diagnostics / layer reports, workflow reports, selective eval, generalization, capacity, candle baseline,
permutation control, chicks bridge, tools/bracket_eval) keep producing the SAME numbers after their `y_<t>_...` literals were replaced by
names derived from core/heads.py (audit C1, PR 8).
    python -m pytest tests/test_head_consumers.py -q

tests/golden_head_consumers.json holds a fingerprint of every function's output, recorded from the code BEFORE the migration (branch
head 717afc2) on deterministic inputs: a model whose weight matrices are set from a seeded numpy generator (no initializer / CPU
dependence), the synthetic datasets of tests/run_fixture.py. Re-record only for a deliberate change of a report:
    RECORD_HEAD_CONSUMERS=1 python -m pytest tests/test_head_consumers.py -q
PITFALLS: fingerprints are (shape, sums) / per-column sums compared with a loose relative tolerance (1e-3): float32 model outputs differ in
the last bits between CPUs, any change of logic (a wrong key -> a KeyError or a different head) moves a sum by far more. The permutation
control trains two models for one epoch: only its table structure is exact, its numbers are compared with rtol 0.2.
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_fixture as rf  # noqa: E402
import _nbload  # noqa: E402

GOLDEN_PATH = os.path.join(ROOT, "tests", "golden_head_consumers.json")
RECORD = os.environ.get("RECORD_HEAD_CONSUMERS") == "1"
RTOL = 1e-3
T_LEN, N_FEAT = 8, 4


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _num(v):
    f = float(v)
    return None if f != f else f


def fp(x):
    """JSON fingerprint of a report: DataFrames by column sums / strings, arrays by (shape, sum, sum of squares)."""
    if isinstance(x, pd.DataFrame):
        out = {"columns": [str(c) for c in x.columns], "n": len(x),
               "index": [str(i) for i in x.index] if not isinstance(x.index, pd.RangeIndex) else None, "num": {}, "obj": {}}
        for c in x.columns:
            col = x[c]
            if col.dtype.kind in "fiub":
                v = col.to_numpy(dtype="float64")
                out["num"][str(c)] = [_num(np.nansum(v)), _num(np.nansum(v * v)), int(np.isnan(v).sum())]
            else:
                vals = [str(v) for v in col.tolist()]
                out["obj"][str(c)] = vals if len(vals) <= 40 else {"n": len(vals), "unique": sorted(set(vals))[:40],
                                                                  "md5": hashlib.md5("\x00".join(vals).encode()).hexdigest()}
        return out
    if isinstance(x, pd.Series):
        return fp(x.to_frame("v"))
    if isinstance(x, dict):
        return {str(k): fp(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [fp(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "fiub":
            v = x.astype("float64")
            return {"shape": list(x.shape), "sum": _num(np.nansum(v)), "sq": _num(np.nansum(v * v))}
        return {"shape": list(x.shape), "obj": [str(v) for v in x.ravel().tolist()]}
    if isinstance(x, (np.floating, float)):
        return _num(x)
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if x is None or isinstance(x, (bool, int, str)):
        return x
    return type(x).__name__


def det_weights(model, seed):
    """Weight matrices from a seeded generator (biases / scales keep their deterministic initial values)."""
    rng = np.random.default_rng(seed)
    model.set_weights([rng.normal(0.0, 0.2, w.shape).astype("float32") if w.ndim >= 2 else w for w in model.get_weights()])
    return model


# ------------------------------------------------------------------------------------------------ model-level (diagnostics / layer report)
def _model_cases():
    import tensorflow as tf
    import test_model_diagnostics as tmd
    ns = tmd._ns()
    x, y = tmd._data(160, 5)
    out = {}
    models = {}
    for enc in ("transformer", "gru"):
        tf.keras.utils.set_random_seed(0)
        models[enc] = det_weights(ns["build_model_fn"](T_LEN, N_FEAT, config=tmd._model_cfg(ns, enc)), 11)
    rep = _quiet(ns["diagnose_model"], models["transformer"], x, y)
    out["diagnose_all_transformer"] = fp({k: v for k, v in rep.items() if k != "sel"})
    out["diagnose_sel"] = fp(rep["sel"])
    out["verdicts_transformer"] = fp(ns["model_health_verdicts"](rep))
    out["diagnose_permutation_gru"] = fp(_quiet(ns["diagnose_model"], models["gru"], x, y, sections=("attention", "sensitivity", "samples"),
                                                sensitivity="permutation"))
    out["diagnose_no_labels"] = fp(_quiet(ns["diagnose_model"], models["gru"], x, None))
    cfg_cls = dict(tmd._model_cfg(ns, "gru"), head_types={t: ["binary_classification"] for t in ("high", "low")})
    tf.keras.utils.set_random_seed(0)
    m_cls = det_weights(ns["build_model_fn"](T_LEN, N_FEAT, config=cfg_cls), 12)
    out["diagnose_class_only"] = fp({k: v for k, v in _quiet(ns["diagnose_model"], m_cls, x, y).items() if k != "sel"})
    cfg_nig = dict(tmd._model_cfg(ns, "gru"), head_types={t: ["nig_regression"] for t in ("high", "low")})
    tf.keras.utils.set_random_seed(0)
    m_nig = det_weights(ns["build_model_fn"](T_LEN, N_FEAT, config=cfg_nig), 13)
    out["diagnose_nig_only"] = fp({k: v for k, v in _quiet(ns["diagnose_model"], m_nig, x, y).items() if k != "sel"})
    xv, yv = tmd._data(120, 3)
    probe = _quiet(ns["layer_probe_report"], models["transformer"], x, y, xv, yv, max_train=160, max_val=120, n_rand=1)
    out["layer_probe_report"] = fp(probe)
    out["layer_probe_verdict"] = fp(ns["layer_probe_verdict"](probe))
    out["layer_probe_report_class_only"] = fp(_quiet(ns["layer_probe_report"], m_cls, x, y, xv, yv, max_train=160, max_val=120, n_rand=1))
    cmp_ = _quiet(ns["layer_compare_report"], models["transformer"], x, y, max_n=160, n_perm=3, n_rand=1, min_group=5)
    out["layer_compare_report"] = fp(cmp_)
    out["layer_compare_report_class_only"] = fp(_quiet(ns["layer_compare_report"], m_cls, x, y, max_n=160, n_perm=3, n_rand=1, min_group=5))
    return out


# ------------------------------------------------------------------------------------------------ workflow-level (run fixture)
_DATASETS = {}


def _dataset():
    if "ds" not in _DATASETS:
        _DATASETS["ds"] = rf.build_dataset(_nbload.load_pipeline(), two_tf=False)
    import copy
    return copy.deepcopy(_DATASETS["ds"])


def _run(update):
    ns = rf.run_namespace()
    kit = ns["Toolkit"].from_namespace(ns)
    settings = ns["RunSettings"]().updated({"train": {"run_main_training": False}, **update})
    res = ns["run_main"](settings, kit, dataset=_dataset(), summary=False)
    det_weights(res.model, 21)
    return ns, res


def _workflow_cases():
    import bracket_eval as be
    out = {}
    scenarios = {"return3": {"target": {"suspended_targets": ()}},
                 "entry_abs": {"target": {"target_mode": "entry_range", "entry_close_reg": "abs_return"}},
                 "entry_pos": {"target": {"target_mode": "entry_range", "entry_close_reg": "range_pos"}}}
    for name, upd in scenarios.items():
        ns, res = _run(upd)
        pt, tf = res.plan.price_targets, res.info.model_tf
        out[f"{name}/price_targets"] = list(pt)
        out[f"{name}/collect_val"] = fp(ns["collect_signals"](res.model, res.val, tf, price_targets=pt))
        out[f"{name}/collect_test"] = fp(ns["collect_signals"](res.model, res.test, tf, price_targets=pt))
        out[f"{name}/class_accuracy"] = fp(ns["classification_accuracy_report"](res.model, res.test, pt, tf))
        for asset in sorted(res.test)[:2]:
            for target in (*pt, *(("close_up", "close_down") if name.startswith("entry") else ())):
                try:
                    r = ns["real_price_predictions"](res.model, res.test, asset, target, tf)
                    out[f"{name}/real_price/{asset}/{target}"] = fp(r)
                except Exception as e:                                  # noqa: BLE001  (a refusal is part of the contract)
                    out[f"{name}/real_price/{asset}/{target}"] = f"{type(e).__name__}: {str(e)[:200]}"
        gap = _quiet(ns["generalization_gap_report"], res.model, res.train, res.val, res.test, model_tf=tf, max_n=300, verbose=False,
                     price_targets=pt, trivial_baseline=True)
        out[f"{name}/generalization_gap"] = fp(gap if isinstance(gap, pd.DataFrame) else gap)
        X = ns["model_x"](res.val, tf)
        heads = ns["_head_targets"](res.val["y"])
        out[f"{name}/capacity_model_skills"] = fp(ns["_model_skills"](res.model, X, res.val["y"], heads))
        preds = res.model.predict(X, batch_size=256, verbose=0)
        for cls in sorted({t for t in pt}):
            out[f"{name}/bracket_frame/{cls}"] = fp(be.frame_from_split(res.val, preds=preds, reg_scale=float(res.info.reg_target_scale),
                                                                       class_head=cls))
        if name.startswith("entry"):
            specs = res.chicks.eval_target_specs
            out[f"{name}/entry_range_target_spec"] = rf.jsonable([ns["entry_range_target_spec"](s, res.val) for s in specs])
        else:
            sel = _quiet(ns["selective_evaluation"], res.model, res.val, res.test, tf, min_n=20, verbose=False, price_targets=pt)
            out[f"{name}/selective_evaluation"] = fp({k: v for k, v in sel.items()})
            out[f"{name}/candle_baseline"] = fp(_quiet(ns["candle_baseline_report"], res.model, res.train, res.val, res.test, model_tf=tf,
                                                       targets=("high", "low"), max_train=300, n_boot=20, verbose=False))
            if name == "return3":
                work = tempfile.mkdtemp(prefix="head_consumers_")
                ctl = _quiet(ns["run_label_permutation_control"], res.train, res.val, res.test, epochs=1, mode="global",
                             max_train_samples=200, seed=0, n_boot=20, run_root=work, run_tag="t", verbose=0,
                             config=res.main_config, model_builder=res.model_builder, model_tf=res.info.model_tfs, price_targets=pt)
                out[f"{name}/permutation_control_summary"] = fp(ctl["summary"])
    return out


_CASES = {}


def cases():
    if not _CASES:
        _CASES.update(_model_cases())
        _CASES.update(_workflow_cases())
        if RECORD:
            with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
                json.dump(_CASES, f, ensure_ascii=False, indent=1, sort_keys=True)
    return _CASES


class HeadConsumersKeepTheirOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.got = json.loads(json.dumps(cases()))
        cls.gold = json.load(open(GOLDEN_PATH, encoding="utf-8"))

    def test_same_case_names(self):
        self.assertEqual(sorted(self.got), sorted(self.gold))

    def _check(self, prefix, rtol=RTOL):
        keys = [k for k in self.gold if k.startswith(prefix)]
        self.assertTrue(keys, prefix)
        for k in keys:
            diffs = rf.close(self.got[k], self.gold[k], k, rtol)
            self.assertEqual(diffs, [], "\n".join(diffs[:15]))

    def test_diagnostics(self):
        self._check("diagnose_")
        self._check("verdicts_")

    def test_layer_report(self):
        self._check("layer_")

    def test_return_mode_reports(self):
        for k in [k for k in self.gold if k.startswith("return3/") and "permutation" not in k]:
            self.assertEqual(rf.close(self.got[k], self.gold[k], k, RTOL), [], k)

    def test_entry_range_reports(self):
        for k in [k for k in self.gold if k.startswith("entry_")]:
            self.assertEqual(rf.close(self.got[k], self.gold[k], k, RTOL), [], k)

    def test_permutation_control(self):
        k = "return3/permutation_control_summary"
        # its formatted text cells ('-0.43558 [-0.48340, -0.39667]') are compared exactly as strings, so a last-digit
        # float difference between machines (seen on CI) failed the test; round the numbers inside text cells to 3 decimals
        # (consistent with the loose rtol 0.2 of the numeric cells)
        self.assertEqual(rf.close(_round_text(self.got[k]), _round_text(self.gold[k]), k, 0.2), [], k)


def _round_text(x):
    """Copy of a fingerprint with every decimal number inside a string rounded to 3 places."""
    import re
    if isinstance(x, str):
        return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.3f}", x)
    if isinstance(x, dict):
        return {k: _round_text(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round_text(v) for v in x]
    return x


if __name__ == "__main__":
    unittest.main()

"""Evaluation asks the model ONCE for all coins (audit PR 2, class C5): the number of model calls does not grow with the number of
coins, and the pooled path equals the old one-call-per-coin path.

The old path (one model call per coin: `predict_with_evaluation_v4` / `predict_latest_v4` / `model(x)` in a loop) is kept ONLY here, as
the reference (`legacy_*` below); production has the single path of evaluation/pooled.py.

1. Call counts (fast, synthetic coins and a fake model): same count for 3 and 30 coins, = ceil(rows / batch_size) for every site.
2. Equivalence on the run_fixture data (real model of `workflow.run.run_main`, no training) per target mode:
   raw outputs allclose (rtol 1e-5, atol 1e-5: batch composition changes float reduction order; see RAW_ATOL), discrete outputs and row order exact,
   and the numpy decode/evaluate path BYTE-identical given equal raw outputs.
    python -m pytest tests/test_evaluation_predict_calls.py -q
"""
import math
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

import run_fixture as rf  # noqa: E402
import test_run_steps as trs  # noqa: E402

#: raw model outputs, pooled vs per coin. Spec was rtol 1e-5 / atol 1e-6, but the measured float32 noise is ABSOLUTE (activations are O(1), so
#: the output of any size carries ~1e-6 noise: 2.6e-6 worst on a 0.027 output) => atol 1e-5 (4x the worst measured), rtol 1e-5 kept.
RAW_RTOL, RAW_ATOL = 1e-5, 1e-5
#: decoded / derived values (price = raw * iqr + median, move_% = (pred - entry)/entry ...): the same noise scaled by the decode (worst
#: measured abs 6e-5 on move_% / confidence_%, rel 4.6e-4 only on values near 0)
DER_RTOL, DER_ATOL = 1e-4, 2e-4
TFS = ["1h"]
TARGETS = ["high", "low", "close"]
#: worst differences seen by the equivalence tests (printed by the last test; reported in the PR description)
WORST = {}


class CountingModel:
    """Wraps a model; counts `__call__`s (the only way evaluation reaches the model) and the rows they carried."""

    def __init__(self, model):
        self.model, self.calls, self.rows = model, 0, 0

    def __call__(self, x, training=False):
        self.calls += 1
        first = x[0] if isinstance(x, (tuple, list)) else (next(iter(x.values())) if isinstance(x, dict) else x)
        self.rows += int(first.shape[0])
        return self.model(x, training=training)

    def __getattr__(self, name):          # output_names, inputs, ... of the wrapped model
        return getattr(self.model, name)


class FakeModel:
    """Deterministic per-row model (output = f(mean of the row's window)), so slicing / row order errors change the values."""
    output_names = ["y_high", "y_low", "y_close"]

    def __call__(self, x, training=False):
        x = x[0] if isinstance(x, (tuple, list)) else x
        m = tf.reduce_mean(tf.cast(x, tf.float32), axis=(1, 2))[:, None]
        return {"y_high": m * 1.0, "y_low": m * 2.0, "y_close": m * 3.0, "y_close_class_logits": tf.sigmoid(m)}


def synthetic_test_dict(n_coins, seed=0, rows=(4, 9)):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_coins):
        n = int(rng.integers(rows[0], rows[1] + 1))
        out[f"S{i:03d}USDT"] = {
            "X_1h": rng.normal(size=(n, 8, 3)).astype("float32"),
            "base_params": np.c_[rng.uniform(50, 150, n), rng.uniform(1, 5, n)].astype("float64"),
            "last_candles": None,
            "y": {t: rng.normal(100, 5, n) for t in TARGETS},
        }
    return out


def legacy_per_asset_results(ns, model, test_dict, tfs, specs, batch_size=256, range_frac=0.2):
    """The pre-refactor evaluation loop of test_all_assets_v4: one predict_with_evaluation_v4 (= one predict_batch_v4) per coin."""
    out = {}
    for name, d in test_dict.items():
        X = tuple(d[f"X_{tf}"] for tf in tfs)
        y = {f"y_{s.name}": d["y"][s.name] for s in specs if s.name in d["y"]}
        out[name] = ns["predict_with_evaluation_v4"](
            model=model, X_inputs=X, target_specs=specs, base_params=d["base_params"], last_candles=d.get("last_candles"),
            y_true=y, verbose=False, n_display=0, batch_size=batch_size, direction_tolerance=range_frac, run_verification=True)
    return out


def legacy_latest_all_assets(ns, model, test_dict, tfs, specs, n_display, batch_size=256):
    """The pre-refactor predict_latest_all_assets: one predict_latest_v4 per coin."""
    frames = []
    for name, d in test_dict.items():
        X = tuple(d[f"X_{tf}"] for tf in tfs)
        ts = ns["_find_timestamps"](d, None, None)
        y = {f"y_{s.name}": d["y"][s.name] for s in specs if "y" in d and s.name in d["y"]} or None
        frames.append(ns["predict_latest_v4"](model, X, d["base_params"], d.get("last_candles"), specs, n_display=n_display,
                                              timestamps=ts, y_true=y, batch_size=batch_size, asset=name, verbose=False,
                                              legend=False))
    return pd.concat(frames, ignore_index=True)


def legacy_classification_report(model, test, price_targets, model_tf, ns):
    """The pre-refactor classification_accuracy_report: one `model(x)` per coin."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    rows = []
    for asset, split in test.items():
        out = model(ns["model_x"](split, model_tf), training=False)
        for t in price_targets:
            y_true = ns["_to_unit_label"](np.asarray(split["y"][f"y_{t}_class"]).ravel())
            y_prob = out[f"y_{t}_class_logits"].numpy().ravel()
            y_pred = (y_prob >= 0.5).astype("float32")
            auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan")
            rows.append({"asset": asset, "target": t, "n": len(y_true), "accuracy": accuracy_score(y_true, y_pred), "auc": auc})
    return pd.DataFrame(rows)


# ── comparison helpers ──────────────────────────────────────────────────────────────────────────────────────────────
def assert_identical(tc, a, b, path=""):
    """Byte-identical: same keys / types / shapes / dtypes, arrays equal bit for bit (NaN == NaN)."""
    if isinstance(a, dict):
        tc.assertIsInstance(b, dict, path)
        tc.assertEqual(sorted(map(str, a)), sorted(map(str, b)), path)
        for k in a:
            assert_identical(tc, a[k], b[k], f"{path}/{k}")
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    elif isinstance(a, (list, tuple)):
        tc.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_identical(tc, x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        tc.assertEqual((a.shape, a.dtype), (b.shape, b.dtype), path)
        tc.assertTrue(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"), f"{path}: arrays differ")
    elif isinstance(a, float) and math.isnan(a):
        tc.assertTrue(isinstance(b, float) and math.isnan(b), path)
    else:
        tc.assertEqual(a, b, path)


def max_diffs(a, b):
    """(max abs diff, max rel diff) over two float arrays (rel = |a-b| / max(|b|, tiny))."""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    if a.size == 0:
        return 0.0, 0.0
    d = np.abs(a - b)
    return float(np.nanmax(d)), float(np.nanmax(d / np.maximum(np.abs(b), 1e-12)))


def assert_close_tree(tc, new, old, mode, tag, path="", rtol=DER_RTOL, atol=DER_ATOL):
    """Floats allclose (worst abs/rel difference recorded in WORST[(mode, tag)]), everything discrete (bool/int/str/object arrays,
    scalars, keys, shapes = row order) exact."""
    if isinstance(old, dict):
        tc.assertEqual(sorted(map(str, new)), sorted(map(str, old)), path)
        for k in old:
            assert_close_tree(tc, new[k], old[k], mode, tag, f"{path}/{k}", rtol, atol)
    elif isinstance(old, pd.DataFrame):
        tc.assertEqual(list(new.columns), list(old.columns))
        tc.assertEqual(new.shape, old.shape)
        for c in old.columns:
            assert_close_tree(tc, new[c].to_numpy(), old[c].to_numpy(), mode, tag, f"{path}.{c}", rtol, atol)
    elif isinstance(old, (list, tuple)):
        tc.assertEqual(len(new), len(old), path)
        for i, (x, y) in enumerate(zip(new, old)):
            assert_close_tree(tc, x, y, mode, tag, f"{path}[{i}]", rtol, atol)
    elif isinstance(old, np.ndarray):
        new = np.asarray(new)
        tc.assertEqual(new.shape, old.shape, path)
        if old.dtype.kind in "fc":
            ad, rd = max_diffs(new, old)
            w = WORST.setdefault((mode, tag), [0.0, 0.0])
            w[0], w[1] = max(w[0], ad), max(w[1], rd)
            np.testing.assert_allclose(new, old, rtol=rtol, atol=atol, equal_nan=True, err_msg=path)
        else:
            bad = np.flatnonzero((new != old).reshape(len(old), -1).any(axis=1)) if old.ndim else []
            same = pd.Series(new.ravel()).equals(pd.Series(old.ravel())) if old.dtype.kind in "Mm" else np.array_equal(new, old)
            tc.assertTrue(same, f"{path}: discrete values differ at rows {list(bad)[:20]}")
    elif isinstance(old, (float, np.floating)):
        w = WORST.setdefault((mode, tag), [0.0, 0.0])
        ad = abs(float(new) - float(old))
        w[0], w[1] = max(w[0], ad), max(w[1], ad / max(abs(float(old)), 1e-12))
        np.testing.assert_allclose(new, old, rtol=rtol, atol=atol, equal_nan=True, err_msg=path)
    else:
        tc.assertEqual(new, old, path)


# ── 1. call counts ──────────────────────────────────────────────────────────────────────────────────────────────────
class ModelCallCountsAreIndependentOfCoinCount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = rf.run_namespace()

    def _count(self, fn, n_coins, **kw):
        td = synthetic_test_dict(n_coins)
        m = CountingModel(FakeModel())
        rf._quiet(fn, m, td, **kw)
        return m.calls, m.rows, sum(len(d["base_params"]) for d in td.values())

    def test_test_all_assets_v4_is_one_call_for_3_and_30_coins(self):
        f = lambda m, td, **k: self.ns["test_all_assets_v4"](m, td, TFS, TARGETS, verbose=False, n_display=0, **k)  # noqa: E731
        (c3, r3, t3), (c30, r30, t30) = self._count(f, 3), self._count(f, 30)
        self.assertEqual((c3, c30), (1, 1))
        self.assertEqual((r3, r30), (t3, t30))                        # every row went through exactly once

    def test_predict_latest_all_assets_is_one_call_for_3_and_30_coins(self):
        f = lambda m, td, **k: self.ns["predict_latest_all_assets"](m, td, TFS, TARGETS, n_display=3, verbose=False, **k)  # noqa: E731
        (c3, r3, _), (c30, r30, _) = self._count(f, 3), self._count(f, 30)
        self.assertEqual((c3, c30), (1, 1))
        self.assertEqual((r3, r30), (9, 90))                          # last 3 rows of each coin

    def test_pooled_table_and_classification_report_are_one_call(self):
        for n in (3, 30):
            td = synthetic_test_dict(n)
            for d in td.values():
                d["y"].update({f"y_{t}_class": np.where(np.arange(len(d["base_params"])) % 2, 1.0, -1.0) for t in ("close",)})
            m = CountingModel(FakeModel())
            pooled, y = self.ns["pool_test_dict"](td, "1h")
            rf._quiet(self.ns["predict_pooled_batch_by_asset"], m, pooled, "1h", target_specs=TARGETS, y_true_pooled=y,
                      n_display=2, verbose=False)
            self.assertEqual(m.calls, 1, f"pooled table, {n} coins")
            m = CountingModel(FakeModel())
            self.ns["classification_accuracy_report"](m, td, ["close"], "1h")
            self.assertEqual(m.calls, 1, f"classification report, {n} coins")

    def test_calls_are_ceil_rows_over_batch_size(self):
        td = synthetic_test_dict(30)
        total = sum(len(d["base_params"]) for d in td.values())
        for bs in (16, 50, 1000):
            m = CountingModel(FakeModel())
            rf._quiet(self.ns["test_all_assets_v4"], m, td, TFS, TARGETS, verbose=False, n_display=0, batch_size=bs)
            self.assertEqual(m.calls, math.ceil(total / bs), bs)

    def test_the_old_loop_grew_with_the_coin_count(self):
        """What the refactor removed: the legacy loop calls the model once per coin (the guard above would fail on it)."""
        specs = self.ns["resolve_targets"](TARGETS)
        for n in (3, 30):
            m = CountingModel(FakeModel())
            legacy_per_asset_results(self.ns, m, synthetic_test_dict(n), TFS, specs)
            self.assertEqual(m.calls, n)

    def test_pooled_rows_land_on_their_own_coin(self):
        """Row order: each coin's evaluated predictions are the fake model's function of ITS OWN windows."""
        td = synthetic_test_dict(7, seed=3)
        res = rf._quiet(self.ns["test_all_assets_v4"], FakeModel(), td, TFS, ["close"], verbose=False, n_display=0,
                        range_frac=0.2)
        per = res["per_asset_results"]
        self.assertEqual(list(per["asset"]), list(td))
        for _, row in per.iterrows():
            d = td[row["asset"]]
            m = d["X_1h"].astype("float32").mean(axis=(1, 2))
            expect = m * 3.0 * d["base_params"][:, 1] + d["base_params"][:, 0]    # decode: raw * iqr + median
            np.testing.assert_allclose(np.asarray(row["close"]["pred_real"]).ravel(), expect, rtol=1e-5)


# ── 2. equivalence on the run_fixture data, per target mode ────────────────────────────────────────────────────────
SCENARIOS = {"return": "default_1tf", "relative": "relative_group_am_off", "entry_range_two_tf": "entry_range_abs_two_tf"}


class PooledPathEqualsPerCoinPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = rf.run_namespace()
        cls.kit = cls.ns["Toolkit"].from_namespace(cls.ns)
        cls.runs = {}
        for mode, scen in SCENARIOS.items():
            sc = trs.GOLDEN["scenarios"][scen]["scenario"]
            settings = trs.settings_for(cls.ns, sc, trs.GOLDEN["defaults"]["train"]["run_dir"])
            res = rf._quiet(cls.ns["run_main"], settings, cls.kit, dataset=trs._dataset(sc["two_tf"], sc["holdout"]), summary=False)
            cls.runs[mode] = res

    def _inputs(self, mode):
        res = self.runs[mode]
        specs = res.chicks.eval_target_specs
        tfs = list(res.info.model_tfs)
        return res, res.model, res.chicks.test_dict, tfs, specs

    def test_each_mode_has_a_real_multi_coin_test_dict(self):
        for mode in SCENARIOS:
            res, model, td, tfs, specs = self._inputs(mode)
            self.assertGreaterEqual(len(td), 5, mode)
            self.assertTrue(specs, mode)

    def test_raw_outputs_equal_and_decode_is_byte_identical_given_equal_raw(self):
        ns = self.ns
        for mode in SCENARIOS:
            with self.subTest(mode=mode):
                res, model, td, tfs, specs = self._inputs(mode)
                old_raw = {}
                for name, d in td.items():
                    old_raw[name] = ns["predict_batch_v4"](model, tuple(d[f"X_{tf}"] for tf in tfs), specs, 256)
                pooled_x, bounds = ns["pool_x"](td, tfs)
                new_raw_pooled, _ = ns["predict_pooled_raw"](model, pooled_x, tfs, specs, 256)
                self.assertEqual([b["name"] for b in bounds], list(td))                   # row order = test_dict order
                for b, new_slice in ns["slice_by_asset"](new_raw_pooled, bounds):
                    n = len(td[b["name"]]["base_params"])
                    self.assertEqual(b["end"] - b["start"], n)
                    assert_close_tree(self, new_slice, old_raw[b["name"]], mode, "raw", rtol=RAW_RTOL, atol=RAW_ATOL)
                # given EQUAL raw outputs the numpy path is byte-identical: the old per-coin call vs feeding that same raw in
                for name, d in td.items():
                    X = tuple(d[f"X_{tf}"] for tf in tfs)
                    y = {f"y_{s.name}": d["y"][s.name] for s in specs if s.name in d["y"]}
                    kw = dict(model=model, X_inputs=X, target_specs=specs, base_params=d["base_params"],
                              last_candles=d.get("last_candles"), y_true=y, verbose=False, n_display=0, run_verification=True)
                    via_model = ns["predict_with_evaluation_v4"](**kw)
                    via_raw = ns["predict_with_evaluation_v4"](**kw, raw_preds=old_raw[name])
                    assert_identical(self, via_raw, via_model, f"{mode}/{name}")

    def test_test_all_assets_v4_matches_the_per_coin_loop(self):
        ns = self.ns
        for mode in SCENARIOS:
            with self.subTest(mode=mode):
                res, model, td, tfs, specs = self._inputs(mode)
                new = rf._quiet(ns["test_all_assets_v4"], model, td, tfs, specs, verbose=False, n_display=0)
                old = legacy_per_asset_results(ns, model, td, tfs, specs)
                per = new["per_asset_results"]
                self.assertEqual(list(per["asset"]), list(td))
                self.assertEqual(list(per["n_samples"]), [len(d["base_params"]) for d in td.values()])
                for i, name in enumerate(td):
                    r_old = {k: v for k, v in old[name].items() if k != "_verification"}
                    r_new = {k: per.iloc[i][k] for k in r_old}
                    assert_close_tree(self, r_new, r_old, mode, "test_all_assets_v4")
                self.assertEqual(new["coverage"]["evaluated_rows"], sum(len(d["base_params"]) for d in td.values()))

    def test_predict_latest_all_assets_matches_the_per_coin_loop(self):
        ns = self.ns
        for mode in SCENARIOS:
            with self.subTest(mode=mode):
                res, model, td, tfs, specs = self._inputs(mode)
                for n_display in (1, 5, 10 ** 6):
                    new = rf._quiet(ns["predict_latest_all_assets"], model, td, tfs, specs, n_display=n_display, verbose=False)
                    old = legacy_latest_all_assets(ns, model, td, tfs, specs, n_display)
                    assert_close_tree(self, new, old, mode, "predict_latest_all_assets")

    def test_pooled_table_matches_the_per_coin_loop(self):
        """predict_pooled_batch_by_asset (live / pooled table) vs one predict_latest_v4 per coin (the old wiring self-test check)."""
        ns = self.ns
        for mode in SCENARIOS:
            with self.subTest(mode=mode):
                res, model, td, tfs, specs = self._inputs(mode)
                pooled, y = ns["pool_test_dict"](td, tfs)
                self.assertEqual([b["name"] for b in pooled["asset_bounds"]], list(td))
                new = rf._quiet(ns["predict_pooled_batch_by_asset"], model, pooled, tfs, specs, n_display=10 ** 6,
                                y_true_pooled=y, verbose=False)
                old = legacy_latest_all_assets(ns, model, td, tfs, specs, 10 ** 6)
                assert_close_tree(self, new, old, mode, "predict_pooled_batch_by_asset")

    def test_classification_accuracy_report_matches_the_per_coin_loop(self):
        ns = self.ns
        for mode in SCENARIOS:
            with self.subTest(mode=mode):
                res, model, td, tfs, specs = self._inputs(mode)
                if "y_high_class_logits" not in model.output:
                    self.skipTest("no classification heads")
                test, pts, model_tf = res.test, list(res.plan.price_targets), res.info.model_tfs
                new = ns["classification_accuracy_report"](model, test, pts, model_tf)
                old = legacy_classification_report(model, test, pts, model_tf, ns)
                assert_close_tree(self, new, old, mode, "classification_accuracy_report")

    def test_class_only_model_classification_report(self):
        sc = trs.GOLDEN["scenarios"]["entry_range_pos_class_only"]["scenario"]
        settings = trs.settings_for(self.ns, sc, trs.GOLDEN["defaults"]["train"]["run_dir"])
        res = rf._quiet(self.ns["run_main"], settings, self.kit, dataset=trs._dataset(sc["two_tf"], sc["holdout"]), summary=False)
        pts, model_tf = list(res.plan.price_targets), res.info.model_tfs
        cm = CountingModel(res.model)
        new = self.ns["classification_accuracy_report"](cm, res.test, pts, model_tf)
        self.assertEqual(cm.calls, math.ceil(sum(len(d["base_params"]) for d in res.test.values()) / 256))
        assert_close_tree(self, new, legacy_classification_report(res.model, res.test, pts, model_tf, self.ns),
                          "entry_range_class_only", "classification_accuracy_report")

    def test_the_fixture_models_are_called_once_per_site(self):
        for mode in SCENARIOS:
            res, model, td, tfs, specs = self._inputs(mode)
            total = sum(len(d["base_params"]) for d in td.values())
            cm = CountingModel(model)
            rf._quiet(self.ns["test_all_assets_v4"], cm, td, tfs, specs, verbose=False, n_display=0)
            self.assertEqual(cm.calls, math.ceil(total / 256), mode)

    @classmethod
    def tearDownClass(cls):
        for (mode, tag), (ad, rd) in sorted(WORST.items()):
            print(f"[pooled-vs-per-coin] {mode:22s} {tag:30s} max abs diff {ad:.3e}  max rel diff {rd:.3e}")


if __name__ == "__main__":
    unittest.main()

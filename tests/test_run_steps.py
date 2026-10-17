"""The workflow/run.py steps (RunSettings -> load, CONFIG, split, retarget, model plan, training config, datasets, training, chicks inputs)
reproduce what main.ipynb's old cells computed.

tests/golden_run_settings.json holds the fingerprints of main.ipynb's cells executed in order (commit 1b1b14f, before RunSettings
existed) on the synthetic datasets of tests/run_fixture.py, for six settings variants: single/two timeframes, default / relative /
entry_range (both close definitions) / magnitude / explicit return targets, anti-memorization on/off, class-only heads, a sealed
holdout, an explicit time split, one trained epoch. Here the same variants are built as RunSettings and run through the steps; every
split, target, config, model structure, training config, batch, chicks input and decode spec must equal the recorded one.
    python -m pytest tests/test_run_steps.py -q
"""
import copy
import dataclasses
import gzip
import json
import os
import pickle
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

import run_fixture as rf  # noqa: E402
import _nbload  # noqa: E402

GOLDEN = json.load(open(os.path.join(ROOT, "tests", "golden_run_settings.json"), encoding="utf-8"))
_DATASETS = {}


def _dataset(two_tf, holdout):
    key = (bool(two_tf), holdout)
    if key not in _DATASETS:
        _DATASETS[key] = rf.build_dataset(_nbload.load_pipeline(), two_tf=two_tf, holdout=holdout)
    return copy.deepcopy(_DATASETS[key])


def settings_for(ns, sc, run_dir):
    """The RunSettings of a recorded scenario (the old notebook globals / evaluate_trained_model patches, as settings)."""
    upd = {"target": {"target_mode": sc.get("target_mode"), "entry_close_reg": sc.get("entry_close_reg", "abs_return"),
                      "group_freq": sc.get("group_freq")},
           "model": {"anti_memorization": sc.get("anti_memorization", True), "class_only": sc.get("class_only", False)},
           "data": {"model_tfs": sc.get("model_tfs")},
           "train": {"run_main_training": bool(sc.get("train")), "run_dir": run_dir, "fit_verbose": 0}}
    if sc.get("split_dates"):
        a, b = sc["split_dates"].split(",")
        upd["data"]["split_dates"] = {"train_end": a, "val_end": b}
    for k in ("epochs", "batch_size"):
        if sc.get(k):
            upd["train"][k] = sc[k]
    if sc.get("patience"):
        upd["train"]["early_stopping_patience"] = sc["patience"]
    return ns["RunSettings"]().updated(upd)


def capture(ns, res):
    xb = res.val_ds.as_numpy_iterator().next()
    hist = res.history.history if res.history is not None else None
    return rf.capture_state(
        config=ns["CONFIG"], model_tf=res.info.model_tf, model_tfs=res.info.model_tfs, reg_target_scale=res.info.reg_target_scale,
        train=res.train, val=res.val, test=res.test, price_targets=res.plan.price_targets,
        suspended_targets=res.plan.suspended_targets, model_overrides=res.plan.model_overrides,
        model_seq_len=res.plan.model_seq_len, model_n_features=res.plan.model_n_features, model=res.model,
        main_config=res.main_config, val_batch=xb, chicks_targets=res.chicks.chicks_targets,
        chicks_market_neutral=res.chicks.market_neutral, test_dict=res.chicks.test_dict,
        eval_target_specs=res.chicks.eval_target_specs, history=hist)


def _norm(state, work):
    """JSON image with the temp work dir replaced (the recorder and this test trained in different temp dirs)."""
    return json.loads(json.dumps(state).replace(work, "<WORK>"))


class RunStepsMatchTheOldCells(unittest.TestCase):
    def _scenario(self, name):
        gold = copy.deepcopy(GOLDEN["scenarios"][name])
        sc = gold.pop("scenario")
        work = tempfile.mkdtemp(prefix="run_steps_")
        ns = rf.run_namespace()
        settings = settings_for(ns, sc, os.path.join(work, "run") if sc.get("train") else GOLDEN["defaults"]["train"]["run_dir"])
        kit = ns["Toolkit"].from_namespace(ns)
        res = ns["run_main"](settings, kit, dataset=_dataset(sc["two_tf"], sc["holdout"]), summary=False)
        got = _norm(capture(ns, res), work)
        if "history" in gold:
            # one epoch of real training: same seeds => same numbers on the machine of the recording; across CPUs/threads the
            # last digits move, so the training history is compared loosely, everything structural exactly
            g_hist, n_hist = gold.pop("history"), got.pop("history")
            self.assertEqual(sorted(g_hist), sorted(n_hist))
            for k in g_hist:
                np.testing.assert_allclose(n_hist[k], g_hist[k], rtol=0.15, atol=0.05, err_msg=k)
        diffs = rf.close(got, gold)
        self.assertEqual(diffs, [], "\n".join(diffs[:25]))
        return ns, settings, kit, res

    def test_default_single_timeframe(self):
        self._scenario("default_1tf")

    def test_relative_with_group_freq_and_old_architecture(self):
        self._scenario("relative_group_am_off")

    def test_entry_range_range_pos_class_only(self):
        self._scenario("entry_range_pos_class_only")

    def test_magnitude_two_timeframes_chicks_unsupported(self):
        _, _, _, res = self._scenario("magnitude_two_tf")
        self.assertIsNone(res.chicks.test_dict)

    def test_explicit_return_trained_with_holdout_and_split_dates(self):
        ns, settings, kit, res = self._scenario("return_trained_holdout")
        self.assertIsNotNone(res.trainer)

    def test_entry_range_abs_return_two_timeframes_with_holdout(self):
        self._scenario("entry_range_abs_two_tf")


class StepContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = rf.run_namespace()
        cls.kit = cls.ns["Toolkit"].from_namespace(cls.ns)
        cls.S = cls.ns["RunSettings"]

    def _loaded(self, settings=None, two_tf=False, holdout=None):
        ns = rf.run_namespace()
        kit = ns["Toolkit"].from_namespace(ns)
        settings = settings or self.S()
        ds = _dataset(two_tf, holdout)
        info = rf._quiet(ns["apply_dataset_config"], settings, ds, kit)
        return ns, kit, settings, ds, info

    def test_toolkit_names_the_missing_entry_points(self):
        ns = dict(self.ns)
        del ns["split_data"], ns["DEFAULT_PRICE_TARGETS"]
        with self.assertRaises(NameError) as cm:
            self.ns["Toolkit"].from_namespace(ns)
        self.assertIn("split_data", str(cm.exception))
        self.assertIn("DEFAULT_PRICE_TARGETS", str(cm.exception))
        replaced = dataclasses.replace(self.kit, load_data_from_drive=lambda **k: 1)          # a field can be swapped (tools / tests)
        self.assertEqual(replaced.load_data_from_drive(), 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.kit.split_data = None

    def test_apply_dataset_config_writes_config_and_validates_timeframes(self):
        ns, kit, settings, ds, info = self._loaded(holdout="2025-03-06")
        self.assertEqual((info.model_tf, info.model_tfs, info.reg_target_scale), ("1h", ("1h",), 100.0))
        self.assertEqual(kit.config["split_dates"], ds["split_dates"])         # copied from the data (CONFIG had none)
        self.assertEqual(kit.config["holdout_start"], "2025-03-06")
        self.assertEqual((kit.config["stride"], kit.config["tf_order"], kit.config["reg_target_scale"]), (8, ["1h"], 100.0))
        ns, kit, settings, ds, info = self._loaded(self.S().updated({"project": {"config_overrides": {"project_name": "p"}}}))
        for bad, why in ((("1h", "4h"), "missing tf"), (("4h",), "wrong base")):
            with self.assertRaises(ValueError, msg=why):
                rf._quiet(ns["apply_dataset_config"], self.S().updated({"data": {"model_tfs": bad}}), ds, kit)
        # 2 timeframes: the first must be the data's base timeframe
        ns, kit, settings, ds, info = self._loaded(two_tf=True)
        with self.assertRaises(ValueError):
            rf._quiet(ns["apply_dataset_config"], self.S().updated({"data": {"model_tfs": ("4h", "1h")}}), ds, kit)
        info2 = rf._quiet(ns["apply_dataset_config"], self.S().updated({"data": {"model_tfs": ("1h", "4h")}}), ds, kit)
        self.assertEqual((info2.model_tf, info2.model_tfs), ("1h", ("1h", "4h")))

    def test_project_config_overrides_are_applied_and_unknown_keys_raise(self):
        ns, kit, settings, ds, info = self._loaded()
        rf._quiet(ns["apply_project_config"], self.S().updated({"project": {"config_overrides": {"project_name": "zz"}}}), kit)
        self.assertEqual(kit.config["project_name"], "zz")
        with self.assertRaises(Exception):
            rf._quiet(ns["apply_project_config"], self.S().updated({"project": {"config_overrides": {"no_such_key": 1}}}), kit)

    def test_load_dataset_uses_the_loader_then_the_path_fallback(self):
        ns = rf.run_namespace()
        ds = _dataset(False, None)
        calls = []

        def fake_loader(**kw):
            calls.append(kw)
            return ds
        kit = dataclasses.replace(ns["Toolkit"].from_namespace(ns), load_data_from_drive=fake_loader)
        s = self.S().updated({"data": {"filename_base": "x", "format": "pkl.gz", "mmap": False, "local_dir": False}})
        self.assertIs(rf._quiet(ns["load_dataset"], s, kit), ds)
        self.assertEqual(calls, [{"filename_base": "x", "fmt": "pkl.gz", "mmap": False, "local_dir": False}])
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "d.pkl.gz")
        with gzip.open(path, "wb") as f:
            pickle.dump(ds, f)
        got = rf._quiet(ns["load_dataset"], self.S().updated({"data": {"path": path}}), kit)
        self.assertEqual(sorted(got), sorted(ds))
        self.assertEqual(len(calls), 1)                                         # an explicit path never goes through the loader
        with self.assertRaises(FileNotFoundError):
            rf._quiet(ns["load_dataset"], self.S().updated({"data": {"path": os.path.join(tmp, "missing.pkl.gz")}}), kit)

    def test_retarget_reverts_to_return_and_stamps_the_dataset_scale(self):
        ns, kit, settings, ds, info = self._loaded()
        train, val, test = rf._quiet(ns["make_splits"], settings, ds, info, kit)
        self.assertEqual(ns["target_mode_of"](train), None)
        s_rel = self.S().updated({"target": {"target_mode": "relative"}})
        rt, rv, rte = rf._quiet(ns["retarget"], s_rel, train, val, test, info)
        self.assertEqual((ns["target_mode_of"](rt), ns["reg_scale_of"](rt)), ("relative", 100.0))
        self.assertEqual(ns["target_mode_of"](train), None)                   # the inputs are not modified
        back = rf._quiet(ns["retarget"], self.S(), rt, rv, rte, info)         # target_mode None after another mode: back to 'return'
        self.assertEqual(ns["target_mode_of"](back[0]), "return")
        same = rf._quiet(ns["retarget"], self.S(), train, val, test, info)    # untouched pipeline targets stay as they are
        self.assertIs(same[0], train)

    def test_split_dates_setting_overrides_the_data_and_the_split_is_time_ordered(self):
        s = self.S().updated({"data": {"split_dates": {"train_end": "2025-02-05", "val_end": "2025-02-20"}}})
        ns, kit, settings, ds, info = self._loaded(s)
        train, val, test = rf._quiet(ns["make_splits"], s, ds, info, kit)
        self.assertEqual(kit.config["split_dates"], {"train_end": "2025-02-05", "val_end": "2025-02-20"})
        ts = lambda sp: np.asarray(sp["last_candles"])[:, ns["TS_COL"]]          # noqa: E731
        te = np.concatenate([ts(v) for v in test.values()])
        self.assertLess(ts(train).max(), ts(val).min())
        self.assertLess(ts(val).max(), te.min())
        for part in (train, val, *test.values()):
            self.assertEqual(part["reg_target_scale"], 100.0)

    def test_plan_model_rules(self):
        ns, kit, settings, ds, info = self._loaded()
        p = ns["plan_model"](settings, ds, info, kit)
        self.assertEqual((p.price_targets, p.suspended_targets), (("high", "low"), ("close",)))
        self.assertEqual((p.seq_len, p.n_features, p.model_seq_len), (16, 18, 16))
        self.assertFalse(p.model_overrides["enforce_order"])
        e = ns["plan_model"](self.S().updated({"target": {"target_mode": "entry_range"}}), ds, info, kit)
        self.assertEqual((e.price_targets, e.suspended_targets), (("high", "low", "close"), ()))
        c = ns["plan_model"](self.S().updated({"model": {"class_only": True, "anti_memorization": False}}), ds, info, kit)
        self.assertEqual(c.model_overrides["head_types"], {"high": ["binary_classification"], "low": ["binary_classification"]})
        self.assertNotIn("d_model", c.model_overrides)                           # no anti-memorization architecture
        self.assertIn("d_model", p.model_overrides)
        # a real_price-mode data file (reg_target_mode != return) keeps OrderedMeans only for the pipeline's own targets
        kit.config["reg_target_mode"] = "window_scale"
        self.assertTrue(ns["plan_model"](self.S().updated({"model": {"anti_memorization": False}}), ds, info, kit)
                        .model_overrides["enforce_order"])
        self.assertFalse(ns["plan_model"](self.S().updated({"target": {"target_mode": "relative"}}), ds, info, kit)
                         .model_overrides["enforce_order"])

    def test_untrained_run_returns_the_built_model_and_a_resumable_builder(self):
        ns, kit, settings, ds, info = self._loaded()
        res = ns["run_main"](settings.updated({"train": {"run_main_training": False}}), kit, dataset=ds, summary=False)
        self.assertIsNone(res.trainer)
        self.assertIsNone(res.history)
        again = res.model_builder()
        self.assertEqual([l.name for l in again.layers], [l.name for l in res.model.layers])   # same architecture every call
        self.assertEqual(again.count_params(), res.model.count_params())

    def test_chicks_run_is_skipped_for_unsupported_target_modes(self):
        ns, kit, settings, ds, info = self._loaded(self.S().updated({"target": {"target_mode": "magnitude"}}))
        s = self.S().updated({"target": {"target_mode": "magnitude"}, "train": {"run_main_training": False}})
        res = ns["run_main"](s, kit, dataset=ds, summary=False)
        self.assertIsNone(res.chicks.test_dict)
        self.assertIsNone(rf._quiet(ns["run_chicks"], s, kit, model=res.model, val=res.val, chicks=res.chicks, info=res.info))

    def test_run_reports_refuses_unknown_report_names(self):
        ns, kit, settings, ds, info = self._loaded()
        with self.assertRaises(ValueError):
            ns["run_reports"](settings, ["nope"], kit, model=None, train=None, val=None, test=None, dataset=ds, info=info,
                              plan=None, chicks=None)


class WiringSelfTest(unittest.TestCase):
    def test_run_wiring_selftest_passes_without_any_notebook_global(self):
        """Section 8 on synthetic data (real tiny training, chicks, every section-7 report, the target modes, multi-timeframe, panel).
        The old cells ran it with the notebook's PRICE_TARGETS / REG_TARGET_SCALE and failed when they were the real run's (close
        suspended, scale 100); it builds its own targets now. Recorded at 1b1b14f: True (95 s) when those globals were 3 targets / 1.0."""
        ns = rf.run_namespace()
        kit = ns["Toolkit"].from_namespace(ns)
        self.assertTrue(rf._quiet(ns["run_wiring_selftest"], kit, verbose=False))


if __name__ == "__main__":
    unittest.main()

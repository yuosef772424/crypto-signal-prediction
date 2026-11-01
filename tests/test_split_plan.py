"""Split leakage control (audit C6 / PR 12): `label_end` on every sample, the SplitPlan, the opt-in calibration split, the purity of the roles.

    python -m pytest -q tests/test_split_plan.py

Groups (added commit by commit):
  * CalibrationSplitTests  the opt-in calibration split (train_tail / val_head): role "calibration", embargo on both sides, the other roles
                           unchanged where they should be, strict arguments, per_asset refused, the run wiring (RunSettings -> make_splits_with_
                           calibration -> run_chicks fits the calibrators on the "calibration" role; the role check of the Calibrator itself is
                           tests/test_calibrator_role.py);
  * PurityTests            for EVERY pair of roles no label of an earlier role is read at or after the later role's first feature bar (default
                           plan, both calibration plans, holdout, rolling windows); a zero-embargo plan and the documented-leaky per_asset
                           split are caught;
  * DefaultPlanTests       the default SplitPlan (data/split_plan.py) selects, byte for byte, the rows `split_data` / `rolling_splits` return
                           (also pinned by golden_split_arrays.json, recorded before the plan existed); a plan whose gap is shorter than the
                           embargo is refused;
  * LabelEndTests          every split (train / val / test, per-asset dict, holdout, rolling window) carries `label_end` = timestamp +
                           horizon bars, the time of the last bar the label reads (checked against the raw candles), and the split ARRAYS are
                           the ones recorded before `label_end` existed (tests/golden_split_arrays.json, tests/split_golden.py).
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import contextlib  # noqa: E402
import io  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _nbload  # noqa: E402
import split_golden as sg  # noqa: E402
from core import roles  # noqa: E402
from core.schema import LABEL_END_KEY, TS_COL  # noqa: E402
from run_fixture import _quiet, build_dataset, ohlcv  # noqa: E402

_PIPE = {}


def pipeline():
    if not _PIPE:
        _PIPE["ns"] = _nbload.load_pipeline()
    return _PIPE["ns"]


def members(split):
    """The splits inside a split or a `{asset: split}` dict."""
    return [split] if "y" in split else list(split.values())


class SplitArraysAreUnchanged(unittest.TestCase):
    """The row selection of every public split function equals the one recorded before label_end / SplitPlan (golden_split_arrays.json)."""

    @classmethod
    def setUpClass(cls):
        cls.fps = sg.scenarios(pipeline())
        cls.golden = sg.load_golden()

    def test_every_scenario_equals_the_recorded_golden(self):
        self.assertEqual(sorted(self.fps), sorted(self.golden))
        for name in self.golden:
            with self.subTest(name):
                self.assertEqual(sg.close(self.fps[name], self.golden[name]), [])


class LabelEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = pipeline()
        cls.ds = build_dataset(cls.ns)
        cls.cfg = dict(cls.ns["CONFIG"], split_dates=dict(sg.SPLIT_DATES))
        cls.horizon = int(cls.ds["forecast_horizon"])

    def _check(self, split):
        for m in members(split):
            le = m[LABEL_END_KEY]
            self.assertEqual(le.dtype, np.int64)
            self.assertEqual(le.shape, (len(m["last_candles"]),))
            bar = int(pd.Timedelta(self.ds["base_tf"]).value)
            np.testing.assert_array_equal(le - m["last_candles"][:, TS_COL].astype("int64"), np.full(len(le), self.horizon * bar))

    def test_every_producer_stamps_label_end(self):
        ns = self.ns
        for sep in (True, False):
            tr, va, te = _quiet(ns["split_data"], self.ds, config=self.cfg, keep_asset_test_separate=sep)
            for s in (tr, va, te):
                self._check(s)
        for s in _quiet(ns["split_data"], self.ds, mode="per_asset", config=dict(ns["CONFIG"]), keep_asset_test_separate=True):
            self._check(s)
        dsh = build_dataset(ns, holdout=sg.HOLDOUT)
        self._check(_quiet(ns["split_holdout"], dsh, config=dict(self.cfg, holdout_start=sg.HOLDOUT, open_holdout=True)))
        for tr, va, te in _quiet(ns["rolling_splits"], self.ds, test_span="4D", val_span="4D", train_span="14D", step="4D", max_windows=2,
                                 config=dict(ns["CONFIG"]), verbose=False):
            for s in (tr, va, te):
                self._check(s)

    def test_label_end_is_the_bar_the_label_reads(self):
        """future_close of a sample IS the close of the bar stamped label_end (checked on coin C0 against the raw candles)."""
        tr, va, te = _quiet(self.ns["split_data"], self.ds, config=self.cfg, keep_asset_test_separate=True)
        raw = ohlcv(70, 40, "2025-01-01")
        close_at = pd.Series(raw["close"].to_numpy(), index=raw.index.as_unit("ns").asi8)
        for split in (va, te["C0USDT"]):
            idx = split[LABEL_END_KEY]
            lc = split["last_candles"]
            keep = np.isin(idx, close_at.index.to_numpy())
            self.assertTrue(keep.any())
            # rows of C0 only: the other coins have other candles, so only rows whose label_end exists in C0 and whose entry matches are compared
            ent = close_at.reindex(split["last_candles"][:, TS_COL].astype("int64")).to_numpy()
            c0 = np.isclose(ent, lc[:, 2], rtol=1e-5) & keep
            self.assertTrue(c0.any())
            np.testing.assert_allclose(close_at.reindex(idx[c0]).to_numpy(), lc[c0, 4], rtol=1e-5)

    def test_label_end_is_never_inside_the_embargo_of_the_next_split(self):
        """The embargo gap exists so a train label cannot be read by val: train.label_end.max() < val.timestamp.min()."""
        tr, va, te = _quiet(self.ns["split_data"], self.ds, config=self.cfg, keep_asset_test_separate=False)
        self.assertLess(tr[LABEL_END_KEY].max(), va["last_candles"][:, TS_COL].min())
        self.assertLess(va[LABEL_END_KEY].max(), te["last_candles"][:, TS_COL].min())

    def test_restamping_raises_and_arrays_are_shared(self):
        tr, _, _ = _quiet(self.ns["split_data"], self.ds, config=self.cfg, keep_asset_test_separate=False)
        with self.assertRaises(ValueError):
            self.ns["stamp_label_end"](self.ds, tr, self.cfg)
        bare = {k: v for k, v in tr.items() if k != LABEL_END_KEY}
        out = self.ns["stamp_label_end"](self.ds, bare, self.cfg)
        self.assertIs(out["X_1h"], bare["X_1h"])
        self.assertNotIn(LABEL_END_KEY, bare)
        self.assertEqual(roles.role_of(out), "train")


class DefaultPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = pipeline()
        cls.ds = build_dataset(cls.ns)
        cls.cfg = dict(cls.ns["CONFIG"], split_dates=dict(sg.SPLIT_DATES))

    def _rows_equal(self, split, mask):
        """The split IS the dataset rows of ``mask``: every array of the split equals ``ds[array][mask]`` byte for byte."""
        mask = np.asarray(mask)
        for k in ("base_params", "last_candles", "X_1h"):
            a, b = np.asarray(split[k]), np.asarray(self.ds[k])[mask]
            self.assertTrue(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), k)
        for t in self.ds["targets"]:
            a, b = np.asarray(split["y"][f"y_{t}"] if f"y_{t}" in split["y"] else split["y"][t]), np.asarray(self.ds[f"y_{t}"])[mask]
            self.assertEqual(a.tobytes(), b.tobytes(), t)

    def test_default_plan_selects_the_rows_split_data_returns(self):
        ns, ds = self.ns, self.ds
        for cfg, label in ((self.cfg, "no holdout"), (dict(self.cfg, holdout_start=sg.HOLDOUT, open_holdout=True, min_split_samples=8), "holdout")):
            with self.subTest(label):
                plan = ns["resolve_split_plan"](ds, config=cfg)
                ts = ns["sample_timestamps"](ds)
                masks = plan.masks(ts)
                self.assertEqual(list(masks), ["train", "val", "test"] + (["holdout"] if label == "holdout" else []))
                tr, va, te = _quiet(ns["split_data"], ds, config=cfg, keep_asset_test_separate=False)
                for split, role in ((tr, "train"), (va, "val"), (te, "test")):
                    self._rows_equal(split, masks[role])
                if label == "holdout":
                    ho = _quiet(ns["split_holdout"], ds, config=cfg, keep_asset_test_separate=False)
                    self._rows_equal(ho, masks["holdout"])
                    self.assertFalse((masks["test"] & masks["holdout"]).any())
                plan.check()

    def test_rolling_windows_are_plans_too(self):
        ns, ds = self.ns, self.ds
        cfg = dict(ns["CONFIG"])
        sched = ns["rolling_split_schedule"](ds, test_span="4D", val_span="4D", train_span="14D", step="4D", max_windows=3, config=cfg)
        wins = _quiet(ns["rolling_splits"], ds, test_span="4D", val_span="4D", train_span="14D", step="4D", max_windows=3,
                      keep_asset_test_separate=False, config=cfg, verbose=False)
        ts, gap = ns["sample_timestamps"](ds), ns["embargo_duration"](ds, cfg)
        for w, (tr, va, te) in zip(sched, wins):
            plan = ns["SplitPlan"].from_dates(w["train_end"], w["val_end"], gap, train_start=w["train_start"], test_end=w["test_end"]).check()
            m = plan.masks(ts)
            for split, role in ((tr, "train"), (va, "val"), (te, "test")):
                self._rows_equal(split, m[role])

    def test_plan_check_refuses_a_gap_shorter_than_the_embargo(self):
        SplitPlan = self.ns["SplitPlan"]
        gap = self.ns["embargo_duration"](self.ds, self.cfg)
        te, ve = pd.Timestamp(sg.SPLIT_DATES["train_end"], tz="UTC"), pd.Timestamp(sg.SPLIT_DATES["val_end"], tz="UTC")
        SplitPlan.from_dates(te, ve, gap).check()
        leaky = SplitPlan(embargo=gap, train_end=te, val=(te + gap / 2, ve), test_lo=ve + gap)
        with self.assertRaisesRegex(ValueError, "shorter than the embargo"):
            leaky.check()
        with self.assertRaisesRegex(ValueError, "empty interval"):
            SplitPlan(embargo=gap, train_end=te, val=(te + gap, te), test_lo=te + 2 * gap).check()


ROLE_ORDER = ("train", "calibration", "val", "test", "holdout")


def arrays_of(split):
    """(timestamps ns, label_end ns) of every sample of a split or a {asset: split} dict."""
    ms = members(split)
    return (np.concatenate([m["last_candles"][:, TS_COL].astype("int64") for m in ms]),
            np.concatenate([m[LABEL_END_KEY] for m in ms]))


def purity_problems(by_role, embargo_ns, horizon_ns):
    """Leakage between roles. ``by_role``: {role: (timestamps ns, label_end ns)}. For every pair (A earlier in time than B) the label of every
    A sample (read up to label_end) must be over before B's first FEATURE bar: max(label_end_A) < min(ts_B) - (embargo - horizon), where
    embargo - horizon is the widest input window (embargo = window + horizon bars). Equivalent to max(ts_A) < min(ts_B) - embargo.
    Returns the list of violations (empty = pure)."""
    names = [r for r in ROLE_ORDER if r in by_role]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = int(by_role[b][0].min()) - (embargo_ns - horizon_ns) - int(by_role[a][1].max())
            if gap <= 0:
                out.append(f"{a} -> {b}: the labels of {a} reach {-gap} ns into the first feature window of {b}")
    return out


def rows_of(split):
    """The set of sample identities (the last_candles rows) of a split."""
    return set(map(tuple, np.asarray(split["last_candles"])[:, :4].tolist()))


def by_role_of(splits):
    """{role: arrays} from stamped splits (the role comes from the stamp, never from the position)."""
    out = {}
    for sp in splits:
        role = roles.role_of(sp)
        assert role not in out, role
        out[role] = arrays_of(sp)
    return out


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class CalibrationSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = pipeline()
        cls.ds = build_dataset(cls.ns)
        cls.cfg = dict(cls.ns["CONFIG"], split_dates=dict(sg.SPLIT_DATES))
        cls.base = _quiet(cls.ns["split_data"], cls.ds, config=cls.cfg, keep_asset_test_separate=False)
        cls.gap = int(cls.ns["embargo_duration"](cls.ds, cls.cfg).value)

    def _carve(self, source, share, sep=False):
        return _quiet(self.ns["split_data_with_calibration"], self.ds, {"source": source, "share": share}, config=self.cfg,
                      keep_asset_test_separate=sep)

    def test_train_tail_carves_the_end_of_train_and_leaves_val_and_test_untouched(self):
        tr, cal, va, te = self._carve("train_tail", 0.2)
        b_tr, b_va, b_te = self.base
        self.assertEqual([roles.role_of(x) for x in (tr, cal, va, te)], ["train", "calibration", "val", "test"])
        for new, old in ((va, b_va), (te, b_te)):                                        # val / test: the very same rows
            np.testing.assert_array_equal(new["last_candles"], old["last_candles"])
            np.testing.assert_array_equal(new["X_1h"], old["X_1h"])
        self.assertLess(len(tr["last_candles"]), len(b_tr["last_candles"]))
        n_tr = len(b_tr["last_candles"])
        self.assertTrue(0.12 * n_tr < len(cal["last_candles"]) < 0.25 * n_tr)             # about the requested share
        ts_tr, ts_cal = tr["last_candles"][:, TS_COL], cal["last_candles"][:, TS_COL]
        self.assertGreaterEqual(ts_cal.min() - ts_tr.max(), self.gap)                     # embargo on the train side
        self.assertGreaterEqual(va["last_candles"][:, TS_COL].min() - ts_cal.max(), self.gap)   # and on the val side
        self.assertTrue(rows_of(tr) | rows_of(cal) <= rows_of(b_tr))                       # carved from the old train, nothing else
        self.assertFalse(rows_of(tr) & rows_of(cal))

    def test_val_head_carves_the_start_of_val_and_leaves_train_and_test_untouched(self):
        tr, cal, va, te = self._carve("val_head", 0.5)
        b_tr, b_va, b_te = self.base
        for new, old in ((tr, b_tr), (te, b_te)):
            np.testing.assert_array_equal(new["last_candles"], old["last_candles"])
        self.assertEqual(roles.role_of(cal), "calibration")
        ts_cal, ts_va = cal["last_candles"][:, TS_COL], va["last_candles"][:, TS_COL]
        self.assertGreaterEqual(ts_cal.min() - tr["last_candles"][:, TS_COL].max(), self.gap)
        self.assertGreaterEqual(ts_va.min() - ts_cal.max(), self.gap)
        self.assertTrue(rows_of(va) | rows_of(cal) <= rows_of(b_va))                       # carved from the old val, nothing else
        self.assertFalse(rows_of(va) & rows_of(cal))

    def test_calibration_has_label_end_prefixed_targets_and_the_trains_normalization(self):
        tr, cal, va, te = self._carve("train_tail", 0.2, sep=True)
        for m in members(cal):
            self.assertEqual(set(m) - set(tr), set())                                    # the same keys as train
        self.assertIn("X_1h", cal)
        self.assertTrue(all(k.startswith("y_") for k in cal["y"]))
        self.assertIsInstance(te, dict)
        self.assertEqual(roles.role_of(te), "test")
        self.assertNotIn("y", te)

    def test_arguments_are_strict_and_per_asset_is_refused(self):
        f = self.ns["split_data_with_calibration"]
        for bad in ({"source": "test_head", "share": 0.1}, {"source": "train_tail", "share": 0.0}, {"source": "train_tail", "share": 0.9},
                    {"source": "train_tail", "share": True}):
            with self.assertRaises(ValueError, msg=str(bad)):
                _quiet(f, self.ds, bad, config=self.cfg)
        with self.assertRaises(TypeError):
            _quiet(f, self.ds, {"source": "train_tail", "share": 0.1, "extra": 1}, config=self.cfg)       # unknown key
        with self.assertRaises(TypeError):
            _quiet(f, self.ds, "train_tail", config=self.cfg)
        with self.assertRaisesRegex(ValueError, "global_time"):
            _quiet(f, self.ds, {"source": "train_tail", "share": 0.1}, config=dict(self.cfg, split_mode="per_asset"))
        with self.assertRaisesRegex(ValueError, "الحدّ الأدنى|min"):                      # a carve that leaves a role too small is refused
            _quiet(f, self.ds, {"source": "val_head", "share": 0.5}, config=dict(self.cfg, min_split_samples=200))

    def test_run_wiring_fits_the_calibrators_on_the_calibration_role(self):
        from tests import run_fixture as rf
        ns = rf.run_namespace()
        kit = ns["Toolkit"].from_namespace(ns)
        ds = build_dataset(_nbload.load_pipeline())
        cal = {"source": "train_tail", "share": 0.2}
        base = ns["RunSettings"]().updated({"train": {"run_main_training": False}})
        on = base.updated({"data": {"calibration_split": cal}, "evaluation": {"calibration_source": "calibration"}})
        ns["apply_project_config"](on, kit)
        info = rf._quiet(ns["apply_dataset_config"], on, ds, kit)
        with self.assertRaisesRegex(ValueError, "make_splits_with_calibration"):       # make_splits would silently drop the split: refused
            rf._quiet(ns["make_splits"], on, ds, info, kit)
        res_off = rf._quiet(ns["run_main"], base, kit, dataset=ds, summary=False)
        self.assertIsNone(res_off.calibration)
        res = rf._quiet(ns["run_main"], on, kit, dataset=ds, summary=False)
        self.assertEqual(roles.role_of(res.calibration), "calibration")
        self.assertLess(len(res.train["last_candles"]), len(res_off.train["last_candles"]))
        np.testing.assert_array_equal(res.val["last_candles"], res_off.val["last_candles"])
        np.testing.assert_array_equal(res.test[next(iter(res.test))]["last_candles"], res_off.test[next(iter(res_off.test))]["last_candles"])
        out = rf._quiet(ns["run_chicks"], on, kit, model=res.model, val=res.val, chicks=res.chicks, info=res.info, calibration=res.calibration)
        self.assertEqual(out["calibrators"].provenance["fit_role"], "calibration")
        with self.assertRaisesRegex(ValueError, "calibration split"):
            rf._quiet(ns["run_chicks"], on, kit, model=res.model, val=res.val, chicks=res.chicks, info=res.info)    # source asks for it, none given
        out_val = rf._quiet(ns["run_chicks"], base, kit, model=res.model, val=res.val, chicks=res.chicks, info=res.info)
        self.assertEqual(out_val["calibrators"].provenance["fit_role"], "val")
        self.assertNotEqual(ns["run_dir_for"](on, 1.0, ["1h"]), ns["run_dir_for"](base, 1.0, ["1h"]))     # another train = another run folder


class PurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = pipeline()
        cls.ds = build_dataset(cls.ns)
        cls.cfg = dict(cls.ns["CONFIG"], split_dates=dict(sg.SPLIT_DATES))
        cls.embargo = int(cls.ns["embargo_duration"](cls.ds, cls.cfg).value)
        cls.horizon = int(cls.ds["forecast_horizon"]) * int(pd.Timedelta(cls.ds["base_tf"]).value)

    def _assert_pure(self, splits, label):
        self.assertEqual(purity_problems(by_role_of(splits), self.embargo, self.horizon), [], label)

    def test_default_plan_and_both_calibration_plans_are_pure(self):
        ns, ds, cfg = self.ns, self.ds, self.cfg
        for sep in (False, True):
            self._assert_pure(_quiet(ns["split_data"], ds, config=cfg, keep_asset_test_separate=sep), f"default sep={sep}")
        for src, share in (("train_tail", 0.2), ("val_head", 0.5)):
            self._assert_pure(_quiet(ns["split_data_with_calibration"], ds, {"source": src, "share": share}, config=cfg,
                                     keep_asset_test_separate=False), src)

    def test_two_timeframe_plan_is_pure(self):
        ns = self.ns
        ds2 = build_dataset(ns, two_tf=True)
        embargo = int(ns["embargo_duration"](ds2, self.cfg).value)
        horizon = int(ds2["forecast_horizon"]) * int(pd.Timedelta(ds2["base_tf"]).value)
        for sp in (_quiet(ns["split_data"], ds2, config=self.cfg, keep_asset_test_separate=False),
                   _quiet(ns["split_data_with_calibration"], ds2, {"source": "train_tail", "share": 0.2}, config=self.cfg,
                          keep_asset_test_separate=False)):
            self.assertEqual(purity_problems(by_role_of(sp), embargo, horizon), [])

    def test_holdout_is_pure_against_every_other_role(self):
        ns = self.ns
        dsh = build_dataset(ns, holdout=sg.HOLDOUT)
        cfg = dict(self.cfg, holdout_start=sg.HOLDOUT, open_holdout=True, min_split_samples=8)
        tr, va, te = _quiet(ns["split_data"], dsh, config=cfg, keep_asset_test_separate=False)
        ho = _quiet(ns["split_holdout"], dsh, config=cfg, keep_asset_test_separate=False)
        self._assert_pure([tr, va, te, ho], "holdout")
        cal = _quiet(ns["split_data_with_calibration"], dsh, {"source": "train_tail", "share": 0.2}, config=cfg, keep_asset_test_separate=False)
        self._assert_pure(list(cal) + [ho], "holdout + calibration")

    def test_every_rolling_window_is_pure(self):
        for w, (tr, va, te) in enumerate(_quiet(self.ns["rolling_splits"], self.ds, test_span="4D", val_span="4D", train_span="14D",
                                                step="4D", max_windows=3, keep_asset_test_separate=False, config=dict(self.ns["CONFIG"]),
                                                verbose=False)):
            self._assert_pure([tr, va, te], f"window {w}")

    def test_a_zero_embargo_plan_and_the_per_asset_split_are_caught(self):
        """The purity check must have teeth: a plan without embargo, and the documented-leaky per_asset mode, both fail it."""
        ns, ds = self.ns, self.ds
        ts = ns["sample_timestamps"](ds)
        lc = ds["last_candles"]
        lab = ns["label_end_values"](ds, lc, self.cfg)
        te, ve = pd.Timestamp(sg.SPLIT_DATES["train_end"], tz="UTC"), pd.Timestamp(sg.SPLIT_DATES["val_end"], tz="UTC")
        leaky = ns["SplitPlan"].from_dates(te, ve, pd.Timedelta(0))
        by_role = {r: (lc[m, TS_COL].astype("int64"), lab[np.asarray(m)]) for r, m in leaky.masks(ts).items()}
        problems = purity_problems(by_role, self.embargo, self.horizon)
        self.assertTrue(any("train -> val" in p for p in problems), problems)
        leaky.check()                                                                 # self-consistent (its own embargo is 0) ...
        with self.assertRaisesRegex(ValueError, "shorter than the embargo"):
            leaky.check(min_embargo=pd.Timedelta(self.embargo, "ns"))                 # ... but refused against the embargo the data needs
        sound = ns["SplitPlan"].from_dates(te, ve, pd.Timedelta(self.embargo, "ns"))
        by_role = {r: (lc[m, TS_COL].astype("int64"), lab[np.asarray(m)]) for r, m in sound.masks(ts).items()}
        self.assertEqual(purity_problems(by_role, self.embargo, self.horizon), [])
        per_asset = _quiet(ns["split_data"], ds, mode="per_asset", config=dict(ns["CONFIG"]), keep_asset_test_separate=False)
        self.assertTrue(purity_problems(by_role_of(per_asset), self.embargo, self.horizon),
                        "per_asset leaks when the coins' histories differ (documented): the purity test must see it")

    def test_carve_plans_keep_the_embargo_on_both_sides(self):
        ns, ds = self.ns, self.ds
        ts = ns["sample_timestamps"](ds)
        base = ns["resolve_split_plan"](ds, config=self.cfg)
        for src, share in (("train_tail", 0.2), ("val_head", 0.5)):
            plan = base.with_calibration(ns["CalibrationCarve"](src, share), ts).check()
            self.assertIsNotNone(plan.calibration)
            with self.assertRaises(ValueError):
                plan.with_calibration(ns["CalibrationCarve"](src, share), ts)          # one calibration split per plan
        with self.assertRaises(ValueError):
            ns["CalibrationCarve"]("middle", 0.1)


if __name__ == "__main__":
    unittest.main()

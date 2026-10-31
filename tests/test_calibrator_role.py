"""Data roles (core/roles.py) and the role-checked Calibrator component (evaluation/calibrators.py), audit C6 / PR 6.

    python -m pytest -q tests/test_calibrator_role.py

Three groups:
  * RoleStampTests       the role is an ADDITIVE stamp on every split (arrays byte-identical), and the stamp helpers refuse
                         unstamped / mixed / re-labelled data;
  * CalibratorRoleTests  fitting on train / holdout / test / unstamped data raises; val and calibration are accepted; the state is
                         serializable and records its provenance; a state fitted on a forbidden role is refused by transform;
  * CalibratorEquivalenceTests  the component's output equals, EXACTLY, what the old code path produced on the same val table
                         (tests/golden_calibrator_outputs.json was frozen from 6c30e22 before the component existed).
"""
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402
from core import roles  # noqa: E402
from tests.run_fixture import _quiet, build_dataset  # noqa: E402

_PIPE = {}


def _pipeline():
    if not _PIPE:
        _PIPE["ns"] = _nbload.load_pipeline()
    return _PIPE["ns"]


def _same_arrays(a, b, path=""):
    """Two split dicts are equal key by key; every array is the same bytes (dtype, shape, content), nested dicts recursed."""
    assert set(a) == set(b), f"{path}: keys differ {sorted(set(a) ^ set(b))}"
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, dict):
            _same_arrays(x, y, f"{path}/{k}")
        elif isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(), f"{path}/{k}: array bytes differ"
        else:
            assert repr(x) == repr(y), f"{path}/{k}: {x!r} != {y!r}"


class RoleStampTests(unittest.TestCase):
    def test_declared_roles(self):
        self.assertEqual(roles.ROLES, ("train", "calibration", "val", "test", "holdout"))
        self.assertEqual(roles.NEVER_FIT_ROLES, ("train", "holdout"))
        self.assertTrue(set(roles.CALIBRATOR_ROLES) <= set(roles.ROLES) - set(roles.NEVER_FIT_ROLES))
        with self.assertRaises(roles.RoleError):
            roles.check_role("validation")

    def test_stamp_is_a_shallow_copy_and_never_relabels(self):
        split = {"y": {"a": np.arange(3)}, "X_1h": np.zeros((3, 2))}
        out = roles.stamp_role(split, "val")
        self.assertNotIn(roles.ROLE_KEY, split)                                   # the original is untouched
        self.assertIs(out["X_1h"], split["X_1h"])                                 # arrays shared, not copied
        self.assertEqual(roles.role_of(out), "val")
        self.assertEqual(roles.stamp_role(out, "val"), out)                       # same role: fine
        with self.assertRaises(roles.RoleError):
            roles.stamp_role(out, "calibration")                                  # re-labelling is how a leak is made
        with self.assertRaises(roles.RoleError):
            roles.role_of(split)                                                  # unstamped: no silent default

    def test_asset_dict_stamps_members_not_the_container(self):
        assets = {"A": {"y": {"a": np.arange(2)}}, "B": {"y": {"a": np.arange(2)}}}
        out = roles.stamp_role(assets, "test")
        self.assertEqual(sorted(out), ["A", "B"])                                 # no 'split_role' key among the assets
        self.assertTrue(all(v[roles.ROLE_KEY] == "test" for v in out.values()))
        self.assertEqual(roles.role_of(out), "test")
        mixed = {"A": roles.stamp_role(assets["A"], "test"), "B": roles.stamp_role(assets["B"], "val")}
        with self.assertRaises(roles.RoleError):
            roles.role_of(mixed)

    def test_dataframe_stamp(self):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        out = roles.stamp_role(df, "calibration")
        self.assertEqual(roles.role_of(out), "calibration")
        self.assertNotIn(roles.ROLE_KEY, df.attrs)
        with self.assertRaises(roles.RoleError):
            roles.role_of(df)

    def test_require_role(self):
        tr = roles.stamp_role({"y": {}}, "train")
        with self.assertRaises(roles.RoleError):
            roles.require_role(tr, roles.CALIBRATOR_ROLES, what="calibrator")
        self.assertEqual(roles.require_role(roles.stamp_role({"y": {}}, "val"), roles.CALIBRATOR_ROLES, what="c"), "val")
        # `never` wins even when the role is also listed as allowed
        with self.assertRaisesRegex(roles.RoleError, "never"):
            roles.require_role(tr, ("train",), what="c", never=roles.NEVER_FIT_ROLES)

    def test_split_data_stamps_roles_and_leaves_arrays_byte_identical(self):
        ns = _pipeline()
        ds = build_dataset(ns)
        stamped = _quiet(ns["split_data"], ds, config=ns["CONFIG"])
        self.assertEqual([roles.role_of(s) for s in stamped], ["train", "val", "test"])
        # the old behaviour = the same call with the stamp function switched off
        real = ns["stamp_split_roles"]
        ns["stamp_split_roles"] = lambda train, val, test: (train, val, test)
        try:
            old = _quiet(ns["split_data"], ds, config=ns["CONFIG"])
        finally:
            ns["stamp_split_roles"] = real
        for new, ref in zip(stamped, old):
            members_new = list(new.values()) if "y" not in new else [new]
            members_old = list(ref.values()) if "y" not in ref else [ref]
            self.assertEqual(len(members_new), len(members_old))
            for n_, o_ in zip(members_new, members_old):
                self.assertEqual(set(n_) - set(o_), {roles.ROLE_KEY})             # the ONLY difference is the new key
                self.assertEqual(set(o_) - set(n_), set())
                _same_arrays({k: v for k, v in n_.items() if k != roles.ROLE_KEY}, o_)

    def test_rolling_splits_and_holdout_are_stamped(self):
        ns = _pipeline()
        ds = build_dataset(ns, holdout="2025-02-25")
        cfg = dict(ns["CONFIG"], open_holdout=True, holdout_start="2025-02-25")
        ho = _quiet(ns["split_holdout"], ds, config=cfg)
        self.assertEqual(roles.role_of(ho), "holdout")
        wins = _quiet(ns["rolling_splits"], ds, test_span="4D", val_span="4D", train_span="14D", step="4D", max_windows=2,
                      config=dict(ns["CONFIG"]), verbose=False)
        self.assertGreaterEqual(len(wins), 1)
        for tr, va, te in wins:
            self.assertEqual([roles.role_of(s) for s in (tr, va, te)], ["train", "val", "test"])


# ══════════════════════════════════════════════════════════════════════════════════════════════════
# the Calibrator component
# ══════════════════════════════════════════════════════════════════════════════════════════════════
GOLDEN = json.load(open(os.path.join(ROOT, "tests", "golden_calibrator_outputs.json"), encoding="utf-8"))
_EV = {}


def _ev():
    """The evaluation namespace (the calibrator math needs no TensorFlow, but the loader executes every module)."""
    if not _EV:
        _EV["ns"] = _nbload.load_evaluation()
    return _EV["ns"]


def _tables():
    from tests.calibration_fixture import TEST_SEED, VAL_SEED, multi_target_table
    return multi_target_table(VAL_SEED), multi_target_table(TEST_SEED)


def _same(a, b):
    """Exact equality of nested dict/list/float structures where NaN equals NaN (diag has NaN for a constant confidence)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and a != a:
        return isinstance(b, float) and b != b
    return a == b


class CalibratorRoleTests(unittest.TestCase):
    def setUp(self):
        self.ns = _ev()
        self.val, self.test = _tables()

    def test_fit_refuses_train_and_holdout_always_and_test_and_unstamped(self):
        fit = self.ns["CALIBRATORS"]["isotonic"].fit
        for role in ("train", "holdout"):
            with self.assertRaisesRegex(roles.RoleError, "never"):
                fit(roles.stamp_role(self.val, role), verbose=False)
        with self.assertRaisesRegex(roles.RoleError, "requires"):
            fit(roles.stamp_role(self.val, "test"), verbose=False)             # test is for reporting, not for fitting a map
        with self.assertRaisesRegex(roles.RoleError, "no role stamp"):
            fit(self.val, verbose=False)                                       # unstamped: no silent default
        # the legacy wrapper is the same check, not a second one
        with self.assertRaises(roles.RoleError):
            self.ns["fit_confidence_calibrators"](roles.stamp_role(self.val, "train"), verbose=False)
        with self.assertRaises(roles.RoleError):
            self.ns["fit_confidence_calibrators"](self.val, verbose=False)

    def test_fit_accepts_val_and_calibration_and_records_provenance(self):
        for method in ("isotonic", "platt"):
            for role in ("val", "calibration"):
                st = self.ns["CALIBRATORS"][method].fit(roles.stamp_role(self.val, role), verbose=False)
                self.assertEqual(st.method, method)
                self.assertEqual(st.provenance["fit_role"], role)
                self.assertEqual(st.provenance["fit_rows"], len(self.val.dropna(subset=["confidence", "correct"])))
        a = self.ns["fit_confidence_calibrators"](roles.stamp_role(self.val, "val"), verbose=False)
        b = self.ns["fit_confidence_calibrators"](roles.stamp_role(self.val, "val"), verbose=False)
        c = self.ns["fit_confidence_calibrators"](roles.stamp_role(self.test, "val"), verbose=False)
        self.assertEqual(a.provenance["fingerprint"], b.provenance["fingerprint"])
        self.assertNotEqual(a.provenance["fingerprint"], c.provenance["fingerprint"])

    def test_transform_and_load_state_refuse_a_state_fitted_on_a_forbidden_role_or_without_provenance(self):
        cal = self.ns["CALIBRATORS"]["isotonic"]
        st = cal.fit(roles.stamp_role(self.val, "val"), verbose=False)
        cal.transform(st, self.test)                                           # fine
        with self.assertRaisesRegex(roles.RoleError, "no provenance"):
            cal.transform(dict(st), self.test)                                 # a raw dict of closures cannot show its source
        with self.assertRaises(roles.RoleError):
            self.ns["apply_confidence_calibration"](self.test, dict(st))
        forged = self.ns["CalibrationState"](st, method="isotonic", provenance=dict(st.provenance, fit_role="train"))
        with self.assertRaisesRegex(roles.RoleError, "fitted on role 'train'"):
            cal.transform(forged, self.test)
        dumped = cal.dump_state(st)
        dumped["provenance"] = dict(dumped["provenance"], fit_role="holdout")
        with self.assertRaises(roles.RoleError):
            cal.load_state(dumped)
        with self.assertRaises(ValueError):
            self.ns["CALIBRATORS"]["platt"].transform(st, self.test)           # a state belongs to the method that fitted it

    def test_registry_contract_and_unknown_names(self):
        from core.registry import ContractError, DuplicateNameError
        reg = self.ns["CALIBRATORS"]
        self.assertEqual(reg.names(), ("isotonic", "platt"))
        self.assertEqual([e.version for e in reg.entries()], [1, 1])
        reg.check_all()
        with self.assertRaises(ValueError) as cm:
            self.ns["fit_confidence_calibrators"](roles.stamp_role(self.val, "val"), method="svm", verbose=False)
        self.assertIn("isotonic", str(cm.exception))
        cls = self.ns["ConfidenceCalibrator"]
        with self.assertRaises(ContractError):                                 # a calibrator that may fit on train is refused AT REGISTRATION
            reg.register("bad", cls("bad", allowed_roles=("train", "val")), version=1)
        with self.assertRaises(ContractError):
            reg.register("bad2", cls("bad2", allowed_roles=("holdout",)), version=1)
        with self.assertRaises(ContractError):
            reg.register("bad3", cls("other", allowed_roles=("val",)), version=1)   # obj.name must equal the registered name
        self.assertEqual(reg.names(), ("isotonic", "platt"))                   # failed registrations leave it unchanged
        with self.assertRaises(DuplicateNameError):
            reg.register("isotonic", cls("isotonic"), version=1)

    def test_one_implementation_of_the_calibration_math(self):
        """No second implementation left behind: an isotonic fit call and the calibrator functions exist in evaluation/calibrators.py only."""
        import ast
        calls, defs = {}, {}
        names = ("_fit_map", "_pav_isotonic", "fit_confidence_calibrators", "apply_confidence_calibration", "diagnose_confidence")
        for pkg in ("core", "cross_asset", "tools", "data", "model", "trainer", "evaluation", "signal_eval", "workflow", "discovery"):
            for dirpath, _, files in os.walk(os.path.join(ROOT, pkg)):
                for f in files:
                    if not f.endswith(".py"):
                        continue
                    path = os.path.join(dirpath, f)
                    rel = os.path.relpath(path, ROOT)
                    for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
                        if isinstance(node, ast.Call):
                            fn = node.func
                            if (fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")) == "IsotonicRegression":
                                calls.setdefault(rel, []).append(node.lineno)
                        elif isinstance(node, ast.FunctionDef) and node.name in names:
                            defs.setdefault(node.name, []).append(rel)
        self.assertEqual(sorted(calls), ["evaluation/calibrators.py"], calls)
        self.assertEqual(defs, {n: ["evaluation/calibrators.py"] for n in names}, defs)

    def test_run_full_analysis_refuses_unstamped_or_forbidden_val_dict(self):
        from tests.test_nig_calibration import _chicks, _fake_model, _test_dict
        ns = _chicks()
        td, val = _test_dict(ns, [30, 30]), _test_dict(ns, [60], seed0=50)
        specs = ns["DEFAULT_PRICE_TARGETS"]

        def run(**kw):
            return _quiet(ns["run_full_analysis"], _fake_model(), td, ["1h"], specs, make_plots=False, verbose=False,
                          run_integrity_check=False, run_trade_selection=False, out_dir="/tmp/_cal_role_test", **kw)
        with self.assertRaisesRegex(roles.RoleError, "no role"):
            run(val_dict={"VAL": val["C0"]})                                   # no stamp, no calibration_role
        with self.assertRaises(roles.RoleError):
            run(val_dict={"VAL": val["C0"]}, calibration_role="train")
        with self.assertRaises(roles.RoleError):
            run(val_dict={"VAL": val["C0"]}, calibration_role="holdout")
        stamped = roles.stamp_role({"VAL": val["C0"]}, "val")                   # the stamp of the data is enough
        res = run(val_dict=stamped)
        self.assertEqual(res["calibrators"].provenance["fit_role"], "val")
        with self.assertRaisesRegex(roles.RoleError, "contradicts"):
            run(val_dict=stamped, calibration_role="calibration")
        with self.assertRaises(roles.RoleError):                                # prebuilt calibrators must carry provenance too
            run(calibrators=dict(res["calibrators"]))
        self.assertEqual(run(calibrators=res["calibrators"])["calibrators"].provenance["fit_role"], "val")


class CalibratorEquivalenceTests(unittest.TestCase):
    """Output of the component == output of the old code path (golden frozen from 6c30e22) on the same val table."""

    def setUp(self):
        self.ns = _ev()
        self.val, self.test = _tables()

    def _check(self, method, exact):
        from tests.calibration_fixture import GRID
        gold = GOLDEN["methods"][method]
        cal = self.ns["fit_confidence_calibrators"](roles.stamp_role(self.val, "val"), method=method, verbose=False)

        def close(a, b, what):
            if exact:
                np.testing.assert_array_equal(a, b, what)
            else:
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=0, err_msg=what)
        self.assertEqual(sorted(cal), sorted(gold["groups"]))
        for g, e in cal.items():
            want = gold["groups"][g]
            for k in ("method", "signal", "label", "n_val", "base_rate", "diag"):
                self.assertTrue(_same(e[k], want[k]), f"{method}/{g}/{k}: {e[k]!r} != {want[k]!r}")   # exact: same table, same math
            self.assertEqual(set(e), {"method", "signal", "label", "n_val", "base_rate", "diag", "predict"})   # legacy entry shape
            close(e["predict"](GRID), np.array(want["grid"], dtype=float), f"{method}/{g} predict")
        cdf = self.ns["apply_confidence_calibration"](self.test, cal)
        for c in ("confidence_cal", "calib_raw_p", "calib_correct"):
            close(cdf[c].to_numpy(dtype=float), np.array([np.nan if v is None else v for v in gold["applied"][c]]), f"{method}/{c}")
        self.assertEqual(cdf["calib_signal"].tolist(), gold["applied_signal"])
        rep = self.ns["calibration_report"](cdf, verbose=False)
        self.assertEqual(rep["target"].tolist(), gold["report_target"])
        for c, want in gold["report"].items():
            if isinstance(want[0], str):                                           # string columns (target, signal)
                self.assertEqual(rep[c].tolist(), want, c)
                continue
            close(rep[c].to_numpy(dtype=float), np.array([np.nan if v is None else v for v in want]), f"{method}/report/{c}")

    def test_isotonic_exactly_equal_to_the_old_path(self):
        self._check("isotonic", exact=True)

    def test_platt_equal_to_the_old_path(self):
        # lbfgs (sklearn LogisticRegression) can move the last digits across CPUs/BLAS builds: rtol 1e-9, everything else exact
        self._check("platt", exact=False)

    def test_the_golden_covers_every_signal_path(self):
        sig = {e["signal"] for e in GOLDEN["methods"]["isotonic"]["groups"].values()}
        self.assertEqual(sig, {"confidence", "class_margin", "base_rate"})

    def test_state_round_trips_through_json(self):
        from tests.calibration_fixture import GRID
        for method in ("isotonic", "platt"):
            cal = self.ns["CALIBRATORS"][method]
            st = cal.fit(roles.stamp_role(self.val, "val"), verbose=False)
            back = cal.load_state(json.loads(json.dumps(cal.dump_state(st))))   # JSON-safe: no closures
            self.assertEqual(back.provenance, st.provenance)
            self.assertEqual(sorted(back), sorted(st))
            for g in st:
                np.testing.assert_allclose(back[g]["predict"](GRID), st[g]["predict"](GRID), rtol=0, atol=1e-15, err_msg=f"{method}/{g}")
                self.assertTrue(_same({k: v for k, v in back[g].items() if k != "predict"}, {k: v for k, v in st[g].items() if k != "predict"}))
            pd.testing.assert_frame_equal(self.ns["apply_confidence_calibration"](self.test, st),
                                          self.ns["apply_confidence_calibration"](self.test, back), atol=1e-15, rtol=0)

    def test_pav_fallback_path_is_serializable_and_equal(self):
        """Without sklearn the PAV fallback is used; its state round-trips and equals the closure it came from."""
        from tests.calibration_fixture import GRID
        saved = sys.modules.get("sklearn.isotonic", "absent")
        sys.modules["sklearn.isotonic"] = None                                  # `from sklearn.isotonic import ...` -> ImportError
        try:
            cal = self.ns["CALIBRATORS"]["isotonic"]
            st = cal.fit(roles.stamp_role(self.val, "val"), verbose=False)
        finally:
            if saved == "absent":
                sys.modules.pop("sklearn.isotonic", None)
            else:
                sys.modules["sklearn.isotonic"] = saved
        kinds = {st[g]["predict"].params["kind"] for g in st if st[g]["signal"] != "base_rate"}
        self.assertEqual(kinds, {"pav_clip"})
        back = cal.load_state(json.loads(json.dumps(cal.dump_state(st))))
        for g in st:
            np.testing.assert_array_equal(back[g]["predict"](GRID), st[g]["predict"](GRID))


class CalibrationSourceSettingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _nbload.workflow_package().load_into({"__name__": "audit_nb"}, only=("settings",))

    def test_default_is_val_and_calibration_is_not_built_yet(self):
        E = self.ns["EvalSettings"]
        self.assertEqual(E().calibration_source, "val")
        self.assertEqual(E(calibration_source="val"), E())
        with self.assertRaisesRegex(NotImplementedError, "C6"):
            E(calibration_source="calibration")                                  # no calibration split exists: refused, not faked
        with self.assertRaises(ValueError):
            E(calibration_source="train")                                        # unknown: strict
        with self.assertRaises(NotImplementedError):
            self.ns["RunSettings"]().updated({"evaluation": {"calibration_source": "calibration"}})


class RunChicksCalibrationTests(unittest.TestCase):
    """workflow.run.run_chicks (main.ipynb section 6) on the synthetic pipeline: the calibrators are fitted on the stamped val split."""

    @classmethod
    def setUpClass(cls):
        from tests import run_fixture as rf
        cls.rf = rf
        cls.ns = rf.run_namespace()
        cls.kit = cls.ns["Toolkit"].from_namespace(cls.ns)
        cls.ds = build_dataset(_nbload.load_pipeline())
        cls.settings = cls.ns["RunSettings"]().updated({"train": {"run_main_training": False}})
        cls.res = cls.ns["run_main"](cls.settings, cls.kit, dataset=cls.ds, summary=False)

    def _run_chicks(self, settings, val):
        return _quiet(self.ns["run_chicks"], settings, self.kit, model=self.res.model, val=val, chicks=self.res.chicks, info=self.res.info)

    def test_calibrators_are_fitted_on_the_val_role_and_unstamped_val_is_refused(self):
        self.assertEqual(roles.role_of(self.res.val), "val")                   # split_data stamped it
        out = self._run_chicks(self.settings, self.res.val)
        self.assertEqual(out["calibrators"].provenance["fit_role"], "val")
        unstamped = {k: v for k, v in self.res.val.items() if k != roles.ROLE_KEY}
        with self.assertRaisesRegex(roles.RoleError, "no role stamp"):
            self._run_chicks(self.settings, unstamped)                          # hand-built / rebuilt val: no silent default
        train_as_val = roles.stamp_role(unstamped, "train")
        with self.assertRaisesRegex(roles.RoleError, "never"):
            self._run_chicks(self.settings, train_as_val)                       # a train split passed as val: refused by the calibrator
        off = self.settings.updated({"evaluation": {"calibrate_confidence": False}})
        self.assertIsNone(self._run_chicks(off, unstamped)["calibrators"])      # no calibration requested: nothing to refuse


if __name__ == "__main__":
    unittest.main()

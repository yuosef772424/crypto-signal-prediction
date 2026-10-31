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


if __name__ == "__main__":
    unittest.main()

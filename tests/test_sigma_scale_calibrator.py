"""SigmaScaleCalibrator (evaluation/calibrators.py, SIGMA_CALIBRATORS): one post-hoc scale so that mu +- z*s*sigma covers the nominal share.

Guards: an over-wide sigma gets s < 1 and an over-confident one s > 1, and the calibrated coverage on the fit data equals nominal; fit refuses
train / holdout / test / unstamped tables; transform and load_state refuse a state fitted on a forbidden role; the raw sigma is untouched.
    python -m pytest tests/test_sigma_scale_calibrator.py -q
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _nbload  # noqa: E402
from core import roles  # noqa: E402
from stats.metrics import sigma_calibration  # noqa: E402


def _table(sigma_mult, role="calibration", n=4000, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n)
    y = mu + rng.normal(0, 1.0, n)                      # the true noise std is 1
    df = pd.DataFrame({"y": y, "mu": mu, "sigma": np.full(n, sigma_mult)})
    return roles.stamp_role(df, role) if role else df


class Calibrator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _nbload.load_evaluation()
        cls.cal = cls.ns["SIGMA_CALIBRATORS"].get("sigma_scale")

    def test_scale_hits_nominal_both_ways(self):
        for mult, side in ((3.0, "narrow"), (0.4, "widen")):
            df = _table(mult)
            st = self.cal.fit(df)
            self.assertEqual(st["provenance"]["fit_role"], "calibration")
            self.assertAlmostEqual(st["scale"] * mult, 1.0, delta=0.06)               # the calibrated sigma ~ the true noise std
            out = self.cal.transform(st, df)
            self.assertAlmostEqual(sigma_calibration(out["y"], out["mu"], out["sigma_cal"]), 0.90, delta=0.01)
            self.assertTrue((out["sigma"] == mult).all())                              # raw untouched
            self.assertEqual(side == "narrow", st["scale"] < 1.0)
        self.assertLess(sigma_calibration(df["y"], df["mu"], df["sigma"]), 0.5)        # the raw over-confident sigma covered far less

    def test_roles_are_enforced(self):
        for role in ("train", "holdout", "test", None):
            with self.assertRaises(roles.RoleError, msg=str(role)):
                self.cal.fit(_table(1.0, role))
        self.cal.fit(_table(1.0, "val"))
        st = self.cal.fit(_table(1.0))
        bad = {**st, "provenance": {"fit_role": "train", "fit_rows": 10}}
        with self.assertRaises(roles.RoleError):
            self.cal.transform(bad, _table(1.0))
        with self.assertRaises(roles.RoleError):
            self.cal.load_state(bad)
        with self.assertRaises(roles.RoleError):
            self.cal.transform({"scale": 2.0}, _table(1.0))                          # no provenance
        self.assertEqual(self.cal.load_state(self.cal.dump_state(st))["scale"], st["scale"])

    def test_registry_contract_and_bad_input(self):
        self.assertEqual(self.ns["SIGMA_CALIBRATORS"].check_all(), [])
        with self.assertRaises(ValueError):
            self.ns["SIGMA_CALIBRATORS"].get("sigma_scal")
        with self.assertRaises(ValueError):
            self.cal.fit(_table(1.0), nominal=1.5)
        with self.assertRaises(ValueError):
            self.cal.fit(_table(1.0, n=10))


if __name__ == "__main__":
    unittest.main()

"""TargetSettings.clip: the +- clip of the regression targets retarget_splits recomputes for target_mode (E-tgt-002 arm B: volnorm
clipped at +-4 instead of the mode's default +-10).

Guards: None = the old call exactly (same arrays as passing no clip); a clip value reaches retarget_splits (|y_reg| <= clip, and the
rows inside the clip are unchanged); a clip without target_mode / a non-positive or non-finite clip raise at construction; a clip
changes the run folder name, None does not.
    python -m pytest tests/test_target_clip.py -q
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from tests import run_fixture as rf  # noqa: E402
from tests.test_run_steps import _dataset  # noqa: E402


class TargetClip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = rf.run_namespace()
        cls.ns, cls.S = ns, ns["RunSettings"]
        kit = ns["Toolkit"].from_namespace(ns)
        settings = cls.S()
        ds = _dataset(False, None)
        cls.info = rf._quiet(ns["apply_dataset_config"], settings, ds, kit)
        cls.splits = rf._quiet(ns["make_splits"], settings, ds, cls.info, kit)

    def _retarget(self, target):
        return rf._quiet(self.ns["retarget"], self.S().updated({"target": target}), *self.splits, self.info)

    def _parts(self, xs, ys):
        """(part of xs, same part of ys) over train, val and every test asset."""
        for x, y in zip(xs, ys):
            for (_, a), (_, b) in zip(self.ns["_split_parts"](x), self.ns["_split_parts"](y)):
                yield a, b

    def test_none_is_the_old_call(self):
        old = rf._quiet(self.ns["retarget_splits"], *self.splits, mode="volnorm", reg_target_scale=self.info.reg_target_scale,
                        close_reg="abs_return", group_freq=None)
        new = self._retarget({"target_mode": "volnorm"})
        for a, b in self._parts(old, new):
            for k in a["y"]:
                np.testing.assert_array_equal(a["y"][k], b["y"][k], err_msg=k)

    def test_clip_reaches_the_targets(self):
        d10 = self._retarget({"target_mode": "volnorm"})
        d4 = self._retarget({"target_mode": "volnorm", "clip": 4})
        for a, b in self._parts(d10, d4):
            for k in a["y"]:
                if not k.endswith("_reg"):
                    continue
                self.assertLessEqual(float(np.max(np.abs(b["y"][k]))), 4.0 + 1e-6, k)
                inside = np.abs(a["y"][k]) <= 4.0
                np.testing.assert_array_equal(a["y"][k][inside], b["y"][k][inside], err_msg=k)
                np.testing.assert_array_equal(np.clip(a["y"][k], -4, 4), b["y"][k], err_msg=k)

    def test_bad_values_raise(self):
        for bad in ({"clip": 4}, {"target_mode": "volnorm", "clip": 0}, {"target_mode": "volnorm", "clip": -1},
                    {"target_mode": "volnorm", "clip": float("inf")}, {"target_mode": "volnorm", "clip": True}):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.S().updated({"target": bad})

    def test_run_dir(self):
        rd = self.ns["run_dir_for"]
        base = rd(self.S().updated({"target": {"target_mode": "volnorm"}}), 1.0, ("4h",))
        self.assertEqual(rd(self.S().updated({"target": {"target_mode": "volnorm", "clip": 4}}), 1.0, ("4h",)),
                         base.replace("_volnorm", "_volnorm_clip4"))
        self.assertTrue(base.endswith("_volnorm_am"))


if __name__ == "__main__":
    unittest.main()

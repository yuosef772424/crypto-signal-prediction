"""TargetCodec registry (stats/target_codec.py; audit PR13 / C4): one registered codec per target mode in use, both directions.

Guards: (1) every codec decodes its own retargeted labels back to the true simple return fut/last - 1 (unclipped rows) through the ONE
function decode_return; (2) relative_median: inverse = relative return, decode_absolute with the stamped ex-post median = the true
return; (3) modes without a codec / relative variants of vol modes / splits without the stamp raise; (4) the TensorFlow ops path gives the
numpy result; (5) forward of the old modes is untouched (tests/test_run_steps.py, test_target_clip.py pin the arrays).
    python -m pytest tests/test_target_codec.py -q
"""
import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

from stats.target_codec import CODECS, codec_name_for_mode, decode_absolute, decode_return  # noqa: E402
from tests import test_volret as tv  # noqa: E402
from tests import test_audit_round2 as r2  # noqa: E402

_COLS = {"high": (0, 6), "low": (1, 5), "close": (2, 4)}      # target -> (last col, future col) of last_candles


def _parts(sp):
    return tv._parts(sp)


def _true(p, t):
    last, fut = _COLS[t]
    return p["last_candles"][:, fut] / p["last_candles"][:, last] - 1.0


class Decode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tv._mode("window_scale")
        cls.splits = tv._mode("window_scale")[2]

    def _retarget(self, mode, **kw):
        kw.setdefault("clip", 4.0)
        return tv._retarget(copy.deepcopy(self.splits), mode, **kw)

    def _roundtrip(self, mode, rtol=2e-3, atol=2e-5):
        out = self._retarget(mode)
        for sp in out:
            for p in _parts(sp):
                for t in ("high", "low", "close"):
                    z = p["y"][f"y_{t}_reg"].astype("float64")
                    dec = decode_return(p, z, t)
                    true = _true(p, t)
                    inside = np.abs(z) < 3.9 * p.get("reg_target_scale", 1.0)
                    self.assertTrue(inside.any())
                    np.testing.assert_allclose(dec[inside], true[inside], rtol=rtol, atol=atol, err_msg=f"{mode}/{t}")

    def test_return(self):
        self._roundtrip("return", rtol=1e-5)

    def test_volnorm(self):
        self._roundtrip("volnorm")

    def test_volret(self):
        self._roundtrip("volret")

    def test_volnorm_stamps_target_vol(self):
        for sp in self._retarget("volnorm"):
            for p in _parts(sp):
                self.assertEqual(len(p["target_vol"]), len(p["last_candles"]))

    def test_relative_median(self):
        out = self._retarget("relative", min_group=1)
        for sp in out:
            for p in _parts(sp):
                z = p["y"]["y_close_reg"].astype("float64")
                med = p["target_group_median_close"]
                rel = decode_return(p, z, "close")
                inside = np.abs(z) < 0.99
                np.testing.assert_allclose(decode_absolute(p, z, med, "close")[inside], _true(p, "close")[inside], atol=2e-6)
                np.testing.assert_allclose((rel + med)[inside], _true(p, "close")[inside], atol=2e-6)

    def test_stale_stamps_do_not_survive_a_mode_change(self):
        out = self._retarget("relative", min_group=1)
        back = r2._quiet(tv._ns()["retarget_splits"], *out, mode="volnorm")
        for sp in back:
            for p in _parts(sp):
                self.assertFalse([k for k in p if k.startswith("target_group_median_")])
                self.assertIn("target_vol", p)


class Refusals(unittest.TestCase):
    def test_names(self):
        self.assertEqual(sorted(CODECS.names()), ["relative_median", "return", "volnorm", "volret"])
        self.assertEqual([codec_name_for_mode(m) for m in (None, "return", "relative", "return+relative", "volnorm", "volret")],
                         ["return", "return", "relative_median", "relative_median", "volnorm", "volret"])

    def test_modes_without_a_codec_raise(self):
        for m in ("scaled", "magnitude", "return_close", "entry_range", "volnorm+relative", "volret+relative", "nope"):
            with self.assertRaises(ValueError, msg=m):
                codec_name_for_mode(m)

    def test_missing_stamp_and_dict_raise(self):
        split = {"y": {}, "target_mode": "volnorm", "reg_target_scale": 1.0}
        with self.assertRaisesRegex(ValueError, "target_vol"):
            decode_return(split, np.zeros(2))
        with self.assertRaisesRegex(ValueError, "one split"):
            decode_return({"A": split}, np.zeros(2))
        with self.assertRaises(ValueError):
            decode_absolute({"y": {}, "target_mode": "return"}, np.zeros(2), np.zeros(2))

    def test_default_split_is_return_with_scale(self):
        np.testing.assert_allclose(decode_return({"y": {}, "reg_target_scale": 100.0}, np.array([1.0, -2.0])), [0.01, -0.02])


class TensorFlowOps(unittest.TestCase):
    def test_volret_inverse_with_tf_ops(self):
        import tensorflow as tf
        split = {"y": {}, "target_mode": "volret", "target_vol": np.array([0.01, 0.02], "float32")}
        z = np.array([1.5, -2.0], "float32")
        want = decode_return(split, z.astype("float64"))
        got = decode_return({**split, "target_vol": tf.constant(split["target_vol"])}, tf.constant(z), ops=tf.math).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

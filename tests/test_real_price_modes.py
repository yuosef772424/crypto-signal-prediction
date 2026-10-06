"""
PURPOSE:  real_price_predictions (main §7-a) inverts each section 3-b target mode to the real future price, or refuses.
TAGS:     real_price_predictions, scaled, inverse, target_mode, retarget_splits, return, return_close, decode
PITFALLS: 'scaled' is (future - last same-kind price) / IQR in main (not the old base_params-centre form PR #8 assumed);
          targets are clipped at +-10, so only unclipped samples can round-trip exactly.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_entry_range import _main_ns  # noqa: E402 — main cells 8/10 on synthetic splits
from tests.test_reg_target_scale import _ns, _rt  # noqa: E402

FUTURE = {"high": "future_high_max", "low": "future_low_min", "close": "future_close"}


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def numpy(self):
        return self.a


def _with_mode(scale, mode):
    ns = _main_ns(scale)
    ns["train"], ns["val"], ns["test"] = _rt(ns, mode=mode)
    split = ns["test"]["AAA"]
    out = {f"y_{t}": _T(split["y"][f"y_{t}_reg"].reshape(-1, 1)) for t in FUTURE}   # a perfect model
    ns["model"] = lambda x, training=False: out
    return ns, split


class RealPricePredictionsModes(unittest.TestCase):
    def test_scaled_returns_the_future_price(self):
        cols = _ns()["LAST_COLUMNS"]
        for scale in (1.0, 100.0):
            ns, split = _with_mode(scale, "scaled")
            for t, fut in FUTURE.items():
                y = np.asarray(split["y"][f"y_{t}_reg"], dtype="float64") / ns["reg_scale_of"](split, t)
                ok = np.abs(y) < 9.999                                   # clipped samples cannot round-trip
                self.assertGreater(ok.sum(), 0)
                want = split["last_candles"][:, cols.index(fut)]
                got = ns["real_price_predictions"](ns["model"], ns["test"], "AAA", t, "1h")
                np.testing.assert_allclose(got[ok], want[ok], rtol=1e-5, err_msg=f"{t} scale={scale}")
                # teeth: the generic same-kind-return inversion (what ran before) gets scaled targets wrong
                wrong = ns["invert_reg_predictions"](y, f"{t}_reg", last_candles=split["last_candles"],
                                                     config=ns["CONFIG"], scale=1.0)
                self.assertFalse(np.allclose(wrong[ok], want[ok], rtol=1e-3), t)

    def test_return_mode_unchanged(self):
        cols = _ns()["LAST_COLUMNS"]
        ns, split = _with_mode(1.0, "return")
        for t, fut in FUTURE.items():
            y = np.asarray(split["y"][f"y_{t}_reg"], dtype="float64")
            ok = np.abs(y) < 0.999                                       # return targets are clipped at +-1
            want = split["last_candles"][:, cols.index(fut)]
            np.testing.assert_allclose(ns["real_price_predictions"](ns["model"], ns["test"], "AAA", t, "1h")[ok], want[ok], rtol=1e-5, err_msg=t)

    def test_unsupported_modes_raise_instead_of_a_wrong_price(self):
        for mode in ("return_close", "magnitude", "relative"):
            ns, _ = _with_mode(1.0, mode)
            with self.assertRaises(ValueError, msg=mode):
                ns["real_price_predictions"](ns["model"], ns["test"], "AAA", "close", "1h")


if __name__ == "__main__":
    unittest.main()

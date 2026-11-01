"""`pandas_tf`: the timeframe name `'1d'` reaches pandas as `'1D'` (Pandas4Warning: `'d'` is deprecated) in data/split.py, sources.py, align.py,
and every output is identical to the canonical `'1D'` path.

    python -m pytest -q tests/test_pandas_tf.py
"""
import os
import sys
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _nbload  # noqa: E402
from tests.run_fixture import ohlcv  # noqa: E402

DEPRECATED_D = r"'d' is deprecated"


class PandasTfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _nbload.load_pipeline()

    def test_only_a_numeric_d_suffix_is_rewritten(self):
        f = self.ns["pandas_tf"]
        self.assertEqual([f(t) for t in ("1d", "7d", "1D", "4h", "15m", "1h", "1W", "d", "")], ["1D", "7D", "1D", "4h", "15m", "1h", "1W", "d", ""])
        self.assertEqual(f(None), None)

    def test_split_embargo_helper_gives_the_same_timedelta_without_the_warning(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=DEPRECATED_D)
            self.assertEqual(self.ns["_tf_to_timedelta"]("1d"), pd.Timedelta(days=1))
            self.assertEqual(self.ns["_tf_to_timedelta"]("4h"), pd.Timedelta(hours=4))

    def test_resample_and_alignment_with_1d_equal_the_1D_path_and_do_not_warn(self):
        ns = self.ns
        df = ohlcv(40, 3, "2025-01-01")
        cfg = dict(ns["CONFIG"], higher_tf_mode="closed")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=DEPRECATED_D)
            low = ns["resample_timeframes"](df, ["1h", "1d"], base_tf="1h", with_features=False, config=cfg)
            ends_low = ns["window_end_indices"](low["1h"].index, 8, 4, "1h", cfg)
            ends_d = ns["window_end_indices"](low["1d"].index, 3, 2, "1d", cfg)
        up = ns["resample_timeframes"](df, ["1h", "1D"], base_tf="1h", with_features=False, config=cfg)
        pd.testing.assert_frame_equal(low["1d"], up["1D"])
        pd.testing.assert_frame_equal(low["1h"], up["1h"])
        np.testing.assert_array_equal(ends_d, ns["window_end_indices"](up["1D"].index, 3, 2, "1D", cfg))
        self.assertGreater(len(ends_low), 0)
        self.assertLess(len(low["1d"]), len(low["1h"]))                       # a real daily frame, not an echo of the hourly one


if __name__ == "__main__":
    unittest.main()

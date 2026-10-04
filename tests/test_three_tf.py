"""ثلاثة فريمات (1h أساسي + 4h + 1d) بالإعدادات وحدها: tf_order + window_sizes + higher_tf_mode='closed'، بلا كود خاص.
    python -m unittest tests.test_three_tf -v
"""
import unittest

import numpy as np
import pandas as pd

from tests import test_audit_round2 as r2


class ThreeTFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = r2._cfg(tf_order=["1h", "4h", "1d"], base_tf="1h", window_sizes={"1h": 32, "4h": 16, "1d": 10},
                          higher_tf_mode="closed")
        cls.frames = {c: r2._ohlcv("2024-10-01", "2025-03-01", s) for s, c in enumerate(("AAAUSDT", "BBBUSDT"))}
        cls.ds = r2._build(cls.cfg, cls.frames)

    def test_shapes_and_meta(self):
        n = len(self.ds["last_candles"])
        self.assertGreater(n, 100)
        self.assertEqual(self.ds["timeframes"], ["1h", "4h", "1d"])
        for tf, w in self.cfg["window_sizes"].items():
            self.assertEqual(self.ds[f"X_{tf}"].shape[:2], (n, w))
            self.assertTrue(np.isfinite(self.ds[f"X_{tf}"]).all(), tf)

    def test_embargo_covers_longest_context_and_split_runs(self):
        ns = r2._ns()
        self.assertGreaterEqual(ns["embargo_candles"](self.ds, self.cfg), 10 * 24)   # سياق 1d = 10 أيام بساعات
        tr, va, te = r2._quiet(ns["split_data"], self.ds, config=self.cfg)
        ts = lambda s: pd.to_datetime(np.concatenate([p["last_candles"][:, 3] for p in  # noqa: E731
                                                      ([s] if "last_candles" in s else s.values())]).astype("int64"))
        self.assertLess(ts(tr).max(), ts(va).min())
        self.assertLess(ts(va).max(), ts(te).min())


if __name__ == "__main__":
    unittest.main()

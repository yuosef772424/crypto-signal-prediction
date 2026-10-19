"""research/gate_features/run_gate001.py helpers: strict config, ranking, trade statistics."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "gate_features"))
import run_gate001 as rg  # noqa: E402


class TestRun(unittest.TestCase):
    def test_unknown_config_key_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"data_dir": "x", "bogus": 1}, f)
        with self.assertRaises(KeyError):
            rg.load_config(f.name)

    def test_rank_columns_finds_informative_column(self):
        rng = np.random.default_rng(0)
        x = rng.integers(-1, 2, 5000).astype(float)
        X = np.column_stack([rng.normal(size=5000), x])
        y = (x > 0).astype(float) + rng.normal(0, 0.1, 5000)
        r = rg.rank_columns(X, y)
        self.assertGreater(r[1], 0.5)
        self.assertLess(r[0], 0.05)

    def test_trade_stats_cost_arithmetic(self):
        n = 1000
        p = np.where(np.arange(n) % 2 == 0, 0.9, 0.1)
        ret = np.where(p > 0.5, 0.01, -0.01)               # perfect foresight: gross 1 % per trade
        coin = np.array(["A"] * n)
        day = np.arange(n) // 10
        s, _ = rg.trade_stats(p, ret, coin, day, 0.0014)
        self.assertAlmostEqual(s["gross_pct"], 1.0)
        self.assertAlmostEqual(s["net1x_pct"], 1.0 - 0.14)
        self.assertAlmostEqual(s["net2x_pct"], 1.0 - 0.28)


if __name__ == "__main__":
    unittest.main()

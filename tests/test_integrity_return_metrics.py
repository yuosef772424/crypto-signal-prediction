"""evaluation/integrity.py: return-space IC / AUC / persistence-baseline MAE next to the price-level metrics (audit D3).

On price levels any model that copies the entry price has corr ~ 0.997; in return space it has none. The old keys are unchanged.
    python -m pytest tests/test_integrity_return_metrics.py -q
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

import _nbload  # noqa: E402


def _ns():
    return _nbload.load_evaluation()


class ReturnMetrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 400
        self.entry = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)) + rng.normal(0, 1.0, n))     # levels vary a lot across rows
        self.ret = rng.normal(0, 0.01, n)
        self.true = self.entry * (1 + self.ret)

    def test_copying_the_entry_price_looks_perfect_on_levels_and_empty_in_returns(self):
        n = _ns()
        pred = self.entry.copy()
        old = n["compute_point_metrics"](self.true, pred)
        new = n["compute_return_metrics"](self.true, pred, self.entry)
        self.assertGreater(old["corr"], 0.99)                                  # the level metric is fooled
        self.assertEqual(new["ic"], 0.0)                                       # no return information (constant prediction)
        self.assertAlmostEqual(new["mae_improvement_%"], 0.0, places=6)        # = the persistence baseline by definition
        self.assertAlmostEqual(new["persistence_mae"], float(np.mean(np.abs(self.ret))), places=9)

    def test_a_real_signal_shows_in_returns(self):
        n = _ns()
        rng = np.random.default_rng(1)
        pred = self.entry * (1 + 0.5 * self.ret + rng.normal(0, 0.005, len(self.ret)))
        m = n["compute_return_metrics"](self.true, pred, self.entry)
        self.assertGreater(m["ic"], 0.5)
        self.assertGreater(m["rank_ic"], 0.5)
        self.assertGreater(m["auc"], 0.7)
        self.assertGreater(m["mae_improvement_%"], 0.0)

    def test_old_keys_unchanged(self):
        n = _ns()
        m = n["compute_point_metrics"](np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 2.0]))
        self.assertEqual(sorted(m), ["corr", "mae", "rmse"])


if __name__ == "__main__":
    unittest.main()

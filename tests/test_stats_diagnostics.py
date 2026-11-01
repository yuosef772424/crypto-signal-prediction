"""stats.metrics: crps_gaussian and the anti-fear diagnostics (spread_ratio, confidence_summary, position_share, risk_coverage_auc,
sigma_calibration): closed forms / hand-computed values, degenerate inputs, NaN policy.
    python -m pytest tests/test_stats_diagnostics.py -q
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from stats import metrics as m  # noqa: E402


class Crps(unittest.TestCase):
    def test_closed_form_and_monte_carlo(self):
        rng = np.random.default_rng(0)
        mu, sigma, y = 0.3, 2.0, 1.1
        draws = rng.normal(mu, sigma, 400_000)
        mc = float(np.mean(np.abs(draws - y)) - 0.5 * np.mean(np.abs(draws[:200_000] - draws[200_000:])))
        self.assertAlmostEqual(m.crps_gaussian([y], [mu], [sigma]), mc, delta=0.01)
        self.assertAlmostEqual(m.crps_gaussian([0.0], [0.0], [1.0]), (2 / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi)), places=9)

    def test_proper_and_edge_cases(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1.0, 5000)
        good = m.crps_gaussian(y, np.zeros(5000), np.ones(5000))
        self.assertLess(good, m.crps_gaussian(y, np.zeros(5000), 0.3 * np.ones(5000)))      # over-confident is worse
        self.assertLess(good, m.crps_gaussian(y, np.zeros(5000), 3.0 * np.ones(5000)))      # under-confident is worse
        self.assertTrue(np.isnan(m.crps_gaussian([], [], [])))
        self.assertAlmostEqual(m.crps_gaussian([1.0, np.nan], [0.0, 0.0], [1.0, 1.0]), m.crps_gaussian([1.0], [0.0], [1.0]))
        with self.assertRaises(ValueError):
            m.crps_gaussian([1.0], [0.0], [0.0])


class AntiFear(unittest.TestCase):
    def test_spread_ratio(self):
        y = np.array([-1.0, 1.0, -1.0, 1.0])
        self.assertAlmostEqual(m.spread_ratio(y, y), 1.0)
        self.assertAlmostEqual(m.spread_ratio(np.zeros(4), y), 0.0)
        self.assertTrue(np.isnan(m.spread_ratio(y, np.zeros(4))))

    def test_confidence_and_position(self):
        c = m.confidence_summary(np.array([0.5, 0.6, 0.7, 0.54]))
        self.assertAlmostEqual(c["mean"], 0.585)
        self.assertAlmostEqual(c["share_below"], 0.5)
        self.assertTrue(np.isnan(m.confidence_summary([])["mean"]))
        self.assertAlmostEqual(m.position_share([0.0, 1.0, -1.0, 0.01], [1.0] * 4), 0.5)      # tanh(1)=.76 and tanh(.01)=.01 < .1
        self.assertEqual(m.position_share(np.zeros(5), np.ones(5)), 0.0)

    def test_risk_coverage_auc(self):
        correct = np.array([1, 1, 0, 0, 1, 0])
        perfect = m.risk_coverage_auc(np.array([.9, .8, .3, .2, .7, .1]), correct)     # all correct rows ranked first -> only late errors
        useless = m.risk_coverage_auc(np.array([.1, .2, .9, .8, .3, .7]), correct)     # confident when wrong
        self.assertLess(perfect, 0.5 * (1 - correct.mean()) + 1e-9)
        self.assertLess(perfect, useless)
        self.assertAlmostEqual(m.risk_coverage_auc(np.array([1.0, 0.0]), np.array([1, 0])), 0.25)   # errors: [0, 1] -> (0 + 0.5) / 2

    def test_sigma_calibration(self):
        rng = np.random.default_rng(2)
        y = rng.normal(0, 1.0, 100_000)
        self.assertAlmostEqual(m.sigma_calibration(y, np.zeros_like(y), np.ones_like(y)), 0.90, delta=0.01)
        self.assertLess(m.sigma_calibration(y, np.zeros_like(y), 0.5 * np.ones_like(y)), 0.6)


if __name__ == "__main__":
    unittest.main()

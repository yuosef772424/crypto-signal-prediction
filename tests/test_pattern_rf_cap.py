"""evaluation.patterns (hardening S03/S04 evidence: profile-003 `report:chicks` 446 s, one 200-tree RandomForest of 106 s per head on 342 k rows):
the importance forest samples at most `rf_max_samples` rows per tree. A table not larger than the cap (and `rf_max_samples=None`) trains exactly the
old forest; above it the importance ranking of the informative features is kept within a stated tolerance (measured at 342 k rows: cap 50 000 ->
x5 faster, max |d importance| 0.015, importances sum to 1).
    python -m pytest tests/test_pattern_rf_cap.py -q
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "docs", "research", "audit")):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from unittest import mock  # noqa: E402

import _nbload  # noqa: E402

FEATURES = ["confidence", "uncertainty", "aleatoric", "epistemic", "predicted_change_pct"]


def frame(n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in FEATURES})
    logit = 1.6 * df["confidence"] - 1.1 * df["predicted_change_pct"].abs() + 0.7 * df["uncertainty"] * df["confidence"]
    df["correct"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["asset"] = rng.choice([f"A{i}" for i in range(10)], n)
    df["target"] = "high"
    return df


class Recorder(RandomForestClassifier):
    seen = []

    def __init__(self, **kw):
        Recorder.seen.append(kw)
        super().__init__(**kw)


class PatternForestCap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _nbload.load_evaluation()
        cls.detect = staticmethod(cls.ns["detect_success_failure_patterns"])

    def run_detect(self, df, **kw):
        Recorder.seen = []
        with mock.patch("sklearn.ensemble.RandomForestClassifier", Recorder):
            out = self.detect(df, per_head=False, verbose=False, **kw)
        self.assertEqual(len(Recorder.seen), 1)
        return out, Recorder.seen[0]

    def test_a_table_not_larger_than_the_cap_trains_the_old_forest(self):
        df = frame(1000)
        out, kw = self.run_detect(df)
        self.assertNotIn("max_samples", kw)
        out_none, kw_none = self.run_detect(df, rf_max_samples=None)
        self.assertNotIn("max_samples", kw_none)
        pd.testing.assert_frame_equal(out["feature_importance"], out_none["feature_importance"])

    def test_a_larger_table_caps_the_tree_sample(self):
        _, kw = self.run_detect(frame(1000), rf_max_samples=300)
        self.assertEqual(kw["max_samples"], 300)
        _, kw = self.run_detect(frame(1000), rf_max_samples=None)
        self.assertNotIn("max_samples", kw)
        self.assertEqual(self.ns["PATTERN_RF_MAX_SAMPLES"], 50_000)

    def test_the_informative_features_keep_their_importance_within_tolerance(self):
        df = frame(60_000, seed=3)
        full, _ = self.run_detect(df, rf_max_samples=None)
        capped, _ = self.run_detect(df, rf_max_samples=20_000)
        a = full["feature_importance"].set_index("الخاصية")["الأهمية"]
        b = capped["feature_importance"].set_index("الخاصية")["الأهمية"].reindex(a.index)
        self.assertAlmostEqual(float(b.sum()), 1.0, places=6)
        self.assertLessEqual(float((a - b).abs().max()), 0.03)
        informative = {"confidence", "predicted_change_pct", "uncertainty"}
        self.assertEqual(set(a.sort_values(ascending=False).index[:3]), informative)
        self.assertEqual(set(b.sort_values(ascending=False).index[:3]), informative)


if __name__ == "__main__":
    unittest.main()

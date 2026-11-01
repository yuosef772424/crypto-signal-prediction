"""Event feature group (data/custom.py event_features; default None = off): runs the pipeline self-tests `_test_event_features` (default off,
spike / gap / max-return values, causality, unknown key raises, names) and the normalize-kind lock of the three columns.
    python -m pytest tests/test_event_features.py -q
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402


class Events(unittest.TestCase):
    def test_group(self):
        ns = _nbload.load_pipeline()
        ns["_test_event_features"]()
        self.assertIsNone(ns["DEFAULT_CUSTOM_SETTINGS"]["event_features"])
        kinds = {c: ns["classify_feature"](c) for c in ("EVT_VOL_SPIKE", "EVT_GAP", "EVT_MAXRET_8")}
        self.assertEqual(kinds, {"EVT_VOL_SPIKE": ns["BINARY_FLAG"], "EVT_GAP": ns["BINARY_FLAG"], "EVT_MAXRET_8": ns["ZSCORE"]})


if __name__ == "__main__":
    unittest.main()

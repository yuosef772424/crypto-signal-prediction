"""tools/bracket_eval.py (حدّ ربح = حدّ خسارة، فلاتر، صفر مقابل اتجاه عشوائي) وentry_feature_table في خط الأنابيب.
    python -m unittest tests.test_bracket_eval -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

from tests import test_audit_round2 as r2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bracket_eval as be  # noqa: E402


def _df(rows):
    """rows: (entry, fut_high, fut_low, fut_close)."""
    a = np.array(rows, float)
    ts = pd.date_range("2025-01-01", periods=len(a), freq="1h", tz="UTC")
    return pd.DataFrame({"asset": "A", "ts": ts, "entry": a[:, 0], "future_high_max": a[:, 1],
                         "future_low_min": a[:, 2], "future_close": a[:, 3]})


class BracketTests(unittest.TestCase):
    def test_outcomes_long_short_both_none(self):
        df = _df([(100, 101.5, 99.8, 101), (100, 100.2, 98.5, 99), (100, 101.5, 98.5, 100), (100, 100.5, 99.5, 100.3)])
        long = be.bracket_returns(df, np.ones(4), 0.01, cost_rt=0.0)
        np.testing.assert_allclose(long, [0.01, -0.01, -0.01, 0.003])            # both touched → stop first (LB)
        ub = be.bracket_returns(df, np.ones(4), 0.01, cost_rt=0.0, ambiguous="tp_first")
        np.testing.assert_allclose(ub, [0.01, -0.01, 0.01, 0.003])
        short = be.bracket_returns(df, -np.ones(4), 0.01, cost_rt=0.001)
        np.testing.assert_allclose(short, [-0.011, 0.009, -0.011, -0.004])
        self.assertTrue(np.isnan(be.bracket_returns(df, np.zeros(4), 0.01)).all())  # no trade
        self.assertAlmostEqual(be.breakeven_hit_rate(0.01, 0.0014), 0.57)

    def test_filters_null_and_rooms(self):
        rng = np.random.default_rng(0)
        n = 4000
        move = rng.normal(0, 0.01, n)
        df = _df(np.c_[np.full(n, 100), 100 * (1 + np.abs(move) + 0.002), 100 * (1 - np.abs(move) - 0.002),
                       100 * (1 + move)])
        df["NATR_14"] = rng.uniform(0.2, 2, n)
        df["p_up"] = np.where(df["NATR_14"] > 1, 0.5 + 0.4 * np.sign(move), rng.uniform(0, 1, n))  # skill only if NATR>1
        df["pred_high"] = df["future_high_max"]
        df["pred_low"] = df["future_low_min"]
        t = be.evaluate_filters(df, {"all": None, "natr>1": "NATR_14 > 1", "room": "tp_room >= bracket",
                                     "tiny": "NATR_14 > 99"}, bracket=0.005, cost_rt=0.0, null_reps=50)
        t = t.set_index("filter")
        self.assertGreater(t.loc["natr>1", "net_bp_lb"], t.loc["all", "net_bp_lb"])
        self.assertLess(t.loc["natr>1", "null_p"], 0.05)
        self.assertLess(t.loc["natr>1", "null_bp"], t.loc["natr>1", "net_bp_lb"])  # skill beats random direction
        self.assertLessEqual(t.loc["natr>1", "null_bp"], 0.0)                 # random direction cannot profit (LB)
        self.assertGreaterEqual(t.loc["natr>1", "net_bp_ub"], t.loc["natr>1", "net_bp_lb"])
        self.assertIn("note", t.columns)
        self.assertTrue(str(t.loc["tiny", "note"]).startswith("<"))
        self.assertEqual(t.loc["all", "coverage"], 1.0)

    def test_frame_from_split_maps_predictions(self):
        lc = np.array([[101.0, 99.0, 100.0, pd.Timestamp("2025-01-01", tz="UTC").value, 100.5, 99.5, 101.2]])
        split = {"AAA": {"last_candles": lc}}
        preds = {"y_close_class_logits": np.array([[0.7]]), "y_high": np.array([[2.0]]), "y_low": np.array([[-1.0]])}
        df = be.frame_from_split(split, preds, reg_scale=100.0)
        self.assertEqual(df.loc[0, "asset"], "AAA")
        self.assertAlmostEqual(df.loc[0, "pred_high"], 101.0 * 1.02)          # relative to last_high
        self.assertAlmostEqual(df.loc[0, "pred_low"], 99.0 * 0.99)            # relative to last_low
        self.assertAlmostEqual(df.loc[0, "p_up"], 0.7)
        self.assertEqual(df.loc[0, "entry"], 100.0)


class EntryFeatureTableTests(unittest.TestCase):
    def test_raw_values_at_entry(self):
        ns = r2._ns()
        cfg = r2._cfg()
        frames = {c: r2._ohlcv("2025-01-01", "2025-02-15", s) for s, c in enumerate(("AAAUSDT", "BBBUSDT"))}
        ds = r2._build(cfg, frames)
        tab = r2._quiet(ns["entry_feature_table"], ds, ["NATR_14", "ADX_14"],
                        load_asset_fn=lambda fid, name: frames[name].copy(), config=cfg)
        self.assertEqual(len(tab), len(ds["last_candles"]))
        self.assertFalse(tab[["NATR_14", "ADX_14"]].isna().any().any())
        f = r2._quiet(ns["resample_timeframes"], frames["BBBUSDT"], ["1h"], config=cfg)["1h"]
        row = tab[tab.asset == "BBBUSDT"].iloc[5]
        self.assertAlmostEqual(row["ADX_14"], f.loc[row["ts"], "ADX_14"])
        with self.assertRaises(ValueError):
            r2._quiet(ns["entry_feature_table"], ds, ["NOPE"], load_asset_fn=lambda fid, name: frames[name].copy(),
                      config=cfg)


if __name__ == "__main__":
    unittest.main()

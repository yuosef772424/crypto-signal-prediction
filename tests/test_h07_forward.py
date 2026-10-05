"""tools/h07_forward.py: قاعدة H07 المجمّدة، والسجلّ لا يُعاد كتابته (يُضاف فقط)، بلا شبكة.
    python -m unittest tests.test_h07_forward -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import h07_forward as hf  # noqa: E402


def _px(days=120, seed=0, end="2026-10-10"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=end, periods=days, freq="D")
    return pd.DataFrame({"BTC": 60000 * np.exp(np.cumsum(rng.normal(0.002, 0.03, days))),
                         "ETH": 3000 * np.exp(np.cumsum(rng.normal(-0.002, 0.04, days)))}, index=idx)


class H07ForwardTests(unittest.TestCase):
    def test_weights_match_frozen_formula(self):
        px = _px()
        w = hf.h07_weights(px)
        lp = np.log(px)
        d = px.index[-1]
        vol = lp.diff().iloc[-30:].std()
        sig = (lp.iloc[-1] - lp.iloc[-31] > 0).astype(float)
        np.testing.assert_allclose(w.loc[d], (sig * (0.02 / vol).clip(upper=3) / 2).values)

    def test_ledger_starts_at_first_decision_and_pnl_is_next_day(self):
        px = _px()
        led = hf.build_ledger(px)
        self.assertEqual(led.date.iloc[0], "2026-09-30")
        self.assertTrue(np.isnan(led.pnl.iloc[-1]))                        # last day: no next close yet
        r = px.pct_change().shift(-1).loc["2026-10-02"]
        row = led.set_index("date").loc["2026-10-02"]
        prev = led.set_index("date").loc["2026-10-01"]
        turn = abs(row.BTC_w - prev.BTC_w) + abs(row.ETH_w - prev.ETH_w)
        self.assertAlmostEqual(row.pnl, row.BTC_w * r.BTC + row.ETH_w * r.ETH - turn * 0.0006)

    def test_append_only_and_revision_guard(self):
        px = _px()
        old = hf.build_ledger(px.iloc[:-3])
        led = hf.merge_ledger(old, hf.build_ledger(px))
        self.assertEqual(len(led), len(hf.build_ledger(px)))
        self.assertFalse(np.isnan(led.pnl.iloc[len(old) - 1]))             # yesterday's pnl filled in
        revised = px.copy()
        revised.iloc[-5, 0] *= 1.01                                          # a past close changes
        with self.assertRaises(ValueError):
            hf.merge_ledger(led, hf.build_ledger(revised))

    def test_summary(self):
        s = hf.summary(hf.build_ledger(_px()))
        self.assertEqual(s["position_date"], "2026-10-10")
        self.assertGreater(s["days"], 5)


if __name__ == "__main__":
    unittest.main()

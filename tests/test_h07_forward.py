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


class ShadowVariantTests(unittest.TestCase):
    def _px10(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range(end="2026-10-10", periods=150, freq="D")
        return pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (150, 10)), axis=0)), index=idx,
                            columns=hf.ALL_COINS)

    def test_shadow_ledgers_and_ensemble_signal(self):
        px = self._px10()
        v4 = hf.build_ledger(px, "v4_ens_10coins")
        self.assertEqual(list(v4.columns), hf.columns_for(hf.ALL_COINS))
        lp = np.log(px)
        d = px.index[-1]
        frac = np.mean([(lp.iloc[-1] - lp.iloc[-1 - n] > 0).astype(float) for n in (10, 20, 30, 60)], axis=0)
        vol = lp.diff().iloc[-30:].std()
        want = frac * (0.02 / vol).clip(upper=3) / 10
        np.testing.assert_allclose(v4.iloc[-1][[f"{c}_w" for c in hf.ALL_COINS]].astype(float).values, want.values)
        self.assertEqual(hf.ledger_path("v3_10coins", "/x/ledger.csv"), "/x/ledger_v3_10coins.csv")
        self.assertEqual(hf.ledger_path("h07", "/x/ledger.csv"), "/x/ledger.csv")

    def test_column_mismatch_refused(self):
        px = self._px10()
        with self.assertRaises(ValueError):
            hf.merge_ledger(hf.build_ledger(px, "h07"), hf.build_ledger(px, "v3_10coins"))

"""tools/trade_engine.py: الدخول عند افتتاح الشمعة التالية (لا تسرّب للمستقبل)، وR بعد التكلفة عند الوقف والهدف،
وإعادة أخذ العيّنات إلى 4h و1d، ومرشح الشراء فقط.
    python -m unittest tests.test_trade_engine -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import trade_engine as E  # noqa: E402


def _frame(n, open_, high, low, close):
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    d = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)
    d = d.assign(atr=2.0, natr=2.0, adx=30.0, rsi=50.0, ema200=close, ema20=close, ema50=close,
                 bb_up=close + 1, bb_lo=close - 1, plus_di=1.0, minus_di=0.0, don_hi=close - 1, don_lo=close + 1,
                 macd_hist=0.0)
    return d


class TradeEngineTests(unittest.TestCase):
    def test_entry_at_next_open_and_stop_gives_minus_one_r_net_of_cost(self):
        n = 20
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 100.0)
        open_[11] = 100.0
        low[12] = 90.0                       # long stop = 100 - 2*ATR(2) = 96, so bar 12 hits it
        d = _frame(n, open_, high, low, close)
        long = np.zeros(n, bool)
        short = np.zeros(n, bool)
        long[10] = True                      # signal at the close of bar 10, entry at the open of bar 11
        t = E.simulate(d, long, short, "1d")
        self.assertEqual(len(t), 1)
        row = t.iloc[0]
        self.assertEqual(row.reason, "stop")
        self.assertEqual(row.entry_time, d.index[11])
        s = 4.0 / 100.0                      # stop distance as a fraction of the entry price
        self.assertAlmostEqual(row.gross_r, -1.0, places=6)
        self.assertAlmostEqual(row.net_r, -1.0 - E.COST / s, places=6)

    def test_target_gives_one_and_a_half_r_gross(self):
        n = 20
        open_ = np.full(n, 100.0)
        high = np.full(n, 101.0)
        low = np.full(n, 99.0)
        close = np.full(n, 100.0)
        high[12] = 107.0                     # long target = 100 + 3*ATR(2) = 106, touched on bar 12
        d = _frame(n, open_, high, low, close)
        long = np.zeros(n, bool)
        long[10] = True
        t = E.simulate(d, long, np.zeros(n, bool), "1d")
        row = t.iloc[0]
        self.assertEqual(row.reason, "target")
        self.assertAlmostEqual(row.gross_r, 1.5, places=6)   # 3 ATR over 2 ATR

    def test_resample_1d_is_left_labelled_and_aggregates_ohlcv(self):
        idx = pd.date_range("2024-01-01 00:00", periods=48, freq="1h", tz="UTC")
        h = pd.DataFrame({"open": np.arange(48.0), "high": np.arange(48.0) + 1, "low": np.arange(48.0) - 1,
                          "close": np.arange(48.0), "volume": 1.0}, index=idx)
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_trade_engine.parquet")
        try:
            h.to_parquet(path)
            d = E.load(path, "1d")
        finally:
            if os.path.exists(path):
                os.remove(path)
        self.assertEqual(len(d), 2)
        self.assertEqual(d.index[0], pd.Timestamp("2024-01-01", tz="UTC"))
        self.assertEqual(d.open.iloc[0], 0.0)                 # first open of day one
        self.assertEqual(d.close.iloc[0], 23.0)               # last close of day one
        self.assertEqual(d.high.iloc[0], 24.0)                # max high of day one
        self.assertEqual(d.low.iloc[0], -1.0)                 # min low of day one


if __name__ == "__main__":
    unittest.main()

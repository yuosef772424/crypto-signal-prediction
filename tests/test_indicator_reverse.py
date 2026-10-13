"""tools/indicator_reverse.py: reverse-engineering of indicator definitions and identity checks.

Synthetic OHLCV is generated with a known definition; the fit must recover that definition (round trip).
Identity checks use the BTC 1h row and the ASTEROID all-zero row quoted in the provider report.
    python -m pytest tests/test_indicator_reverse.py -q
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd
import pandas_ta_classic as ta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import indicator_reverse as ir  # noqa: E402


def synthetic_ohlcv(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 0.005, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


BTC_1H_ROW = {  # report, BTCUSDT 1h, 2026-10-13T15:00Z (values as quoted)
    "rsi": 53.5126, "wr": -41.913925, "cci": 44.339504, "adx": 10.617199, "di_plus": 22.94012,
    "di_minus": 16.898596, "macd": 35.05824, "macd_dea": 79.93519, "macd_difference": 114.99343,
    "signal_value_k": 36.01609, "signal_value_d": 45.793995, "signal_value_j": 16.460289,
    "boll_upper_band": 84408.86, "boll_middle_band": 83102.87, "boll_lower_band": 81796.88,
}

ASTEROID_ZERO_ROW = {  # report, ASTEROID 1h, indicators all 0 with 120 candles
    "adx": 0, "cci": 0, "rsi": 0, "wr": 0, "macd": 0, "macd_dea": 0, "macd_difference": 0,
    "signal_value_k": 0, "signal_value_d": 0, "signal_value_j": 0,
    "boll_upper_band": 0, "boll_middle_band": 0, "boll_lower_band": 0,
}


class FitRecoversDefinition(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_ohlcv()

    def test_rsi_14_recovered(self):
        observed = ta.rsi(self.df["close"], length=14)
        res = ir.fit_indicator(observed, ir.candidates_for(self.df, "rsi"))
        self.assertEqual(res["best"], "rsi_14")
        self.assertTrue(res["matched"])

    def test_rsi_9_not_mistaken_for_14(self):
        observed = ta.rsi(self.df["close"], length=9)
        res = ir.fit_indicator(observed, ir.candidates_for(self.df, "rsi"))
        self.assertEqual(res["best"], "rsi_9")
        self.assertTrue(res["matched"])

    def test_ema_30_recovered(self):
        observed = ta.ema(self.df["close"], length=30)
        res = ir.fit_indicator(observed, ir.candidates_for(self.df, "ema"))
        self.assertEqual(res["best"], "ema_30")
        self.assertTrue(res["matched"])

    def test_sma_7_recovered(self):
        observed = ta.sma(self.df["close"], length=7)
        res = ir.fit_indicator(observed, ir.candidates_for(self.df, "sma"))
        self.assertEqual(res["best"], "sma_7")
        self.assertTrue(res["matched"])

    def test_willr_14_recovered(self):
        observed = ta.willr(self.df["high"], self.df["low"], self.df["close"], length=14)
        res = ir.fit_indicator(observed, ir.candidates_for(self.df, "willr"))
        self.assertEqual(res["best"], "willr_14")
        self.assertTrue(res["matched"])

    def test_macd_12_26_9_recovered(self):
        frame = ta.macd(self.df["close"], fast=12, slow=26, signal=9)
        line = frame[[c for c in frame.columns if c.startswith("MACD_")][0]]
        sig = frame[[c for c in frame.columns if c.startswith("MACDs_")][0]]
        res = ir.fit_macd(line, sig, self.df)
        self.assertEqual(res["best"], "macd_12_26_9")
        self.assertTrue(res["matched"])

    def test_zeros_in_observed_are_ignored(self):
        observed = ta.rsi(self.df["close"], length=14).copy()
        observed.iloc[-5:] = 0  # provider fill value at the tail
        res = ir.fit_indicator(observed, ir.candidates_for(self.df, "rsi"))
        self.assertEqual(res["best"], "rsi_14")
        self.assertTrue(res["matched"])

    def test_wrong_series_does_not_match(self):
        rng = np.random.default_rng(1)
        observed = pd.Series(rng.uniform(0, 100, len(self.df)))
        res = ir.fit_indicator(observed, ir.candidates_for(self.df, "rsi"))
        self.assertFalse(res["matched"])


class IdentityChecks(unittest.TestCase):
    def test_kdj_identity_holds_on_btc_row(self):
        findings = ir.identity_checks(BTC_1H_ROW)
        self.assertFalse([f for f in findings if f.startswith("kdj")], findings)

    def test_bollinger_symmetric_on_btc_row(self):
        findings = ir.identity_checks(BTC_1H_ROW)
        self.assertFalse([f for f in findings if f.startswith("bollinger")], findings)

    def test_macd_difference_is_sum_not_histogram(self):
        findings = ir.identity_checks(BTC_1H_ROW)
        self.assertTrue(any("macd + dea" in f for f in findings), findings)

    def test_all_zero_row_reads_as_missing(self):
        findings = ir.identity_checks(ASTEROID_ZERO_ROW)
        self.assertTrue(any(f.startswith("missing") for f in findings), findings)

    def test_out_of_range_rsi_is_flagged(self):
        findings = ir.identity_checks({"rsi": 120.0})
        self.assertTrue(any(f.startswith("range: rsi") for f in findings), findings)


class InsufficientHistory(unittest.TestCase):
    def test_ma200_with_120_candles_zero_is_flagged(self):
        self.assertTrue(ir.insufficient_history(0, n_candles=120, length=200))

    def test_ma200_with_120_candles_null_is_correct(self):
        self.assertFalse(ir.insufficient_history(None, n_candles=120, length=200))

    def test_ma200_with_enough_history_is_not_flagged(self):
        self.assertFalse(ir.insufficient_history(83577.55, n_candles=250, length=200))


class GateOrdering(unittest.TestCase):
    def test_newest_first_rows_are_reversed(self):
        rows = [
            {"time": 3, "open_price": 3.0, "high_price": 3.5, "low_price": 2.5, "close_price": 3.2},
            {"time": 1, "open_price": 1.0, "high_price": 1.5, "low_price": 0.5, "close_price": 1.2},
            {"time": 2, "open_price": 2.0, "high_price": 2.5, "low_price": 1.5, "close_price": 2.2},
        ]
        frame = ir.to_ascending(rows)
        self.assertEqual(list(frame["time"]), [1, 2, 3])
        self.assertEqual(list(frame["close"]), [1.2, 2.2, 3.2])


if __name__ == "__main__":
    unittest.main()

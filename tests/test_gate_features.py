"""tools/gate_features.py (gate-style-v1, unverified): causal categorical states, NaN never 0, closed bars only.
    python -m pytest tests/test_gate_features.py -q
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gate_features as gf  # noqa: E402


def _walk(n, seed=0, start="2024-01-01", freq="5min", drift=0.0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": 1.0}, index=idx)


class TestVersion(unittest.TestCase):
    def test_version_label(self):
        self.assertEqual(gf.VERSION, "gate-style-v1, unverified")


class TestResample(unittest.TestCase):
    def test_label_is_open_and_incomplete_bar_dropped(self):
        df = _walk(12 * 3 + 5)                       # 3 full hours + 5 bars of the 4th
        h = gf.resample_ohlcv(df, 60)
        self.assertEqual(len(h), 3)
        self.assertEqual(h.index[0], df.index[0])
        self.assertAlmostEqual(h["open"].iloc[1], df["open"].iloc[12])
        self.assertAlmostEqual(h["close"].iloc[1], df["close"].iloc[23])
        self.assertAlmostEqual(h["high"].iloc[1], df["high"].iloc[12:24].max())

    def test_gap_bar_dropped_not_filled(self):
        df = _walk(12 * 4)
        df = df.drop(df.index[15])                    # one missing 5m bar inside hour 2
        h = gf.resample_ohlcv(df, 60)
        self.assertEqual(len(h), 3)
        self.assertNotIn(df.index[12], h.index)


class TestKnownShapes(unittest.TestCase):
    def setUp(self):
        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        up = 100 * 1.003 ** np.arange(n)
        self.up = pd.DataFrame({"open": up / 1.001, "high": up * 1.001, "low": up / 1.002, "close": up, "volume": 1.0}, index=idx)
        dn = 100 * 0.997 ** np.arange(n)
        self.dn = pd.DataFrame({"open": dn * 1.001, "high": dn * 1.002, "low": dn / 1.001, "close": dn, "volume": 1.0}, index=idx)

    def test_uptrend_states(self):
        raw, st = gf.tf_features(self.up)
        last = st.iloc[-1]
        for c in ("sma_A", "sma_B", "ema_A", "ema_B", "sar_side", "macd_hist_sign"):
            self.assertEqual(last[c], 1, c)
        self.assertEqual(last["rsi_zone"], 1)          # overbought (natural coordinates)
        self.assertEqual(last["di_dom"], 1)
        self.assertEqual(last["adx_bucket"], 2)
        self.assertGreater(raw["rsi14"].iloc[-1], 99)

    def test_downtrend_states(self):
        raw, st = gf.tf_features(self.dn)
        last = st.iloc[-1]
        for c in ("sma_A", "sma_B", "ema_A", "ema_B", "sar_side", "macd_hist_sign", "di_dom"):
            self.assertEqual(last[c], -1, c)
        self.assertEqual(last["rsi_zone"], -1)
        self.assertLess(raw["rsi14"].iloc[-1], 1)

    def test_aggregate_orientation(self):
        _, up = gf.tf_features(self.up)
        _, dn = gf.tf_features(self.dn)
        self.assertGreater(up["agg_trend"].iloc[-1], 0)
        self.assertLess(dn["agg_trend"].iloc[-1], 0)
        # contrarian oscillators: an overbought market is bearish on the oscillator side
        self.assertLess(up["agg_osc"].iloc[-1], 0)
        self.assertGreater(dn["agg_osc"].iloc[-1], 0)
        s = up.iloc[-1]
        self.assertEqual(s["agg_all"], s["agg_trend"] + s["agg_osc"])

    def test_rsi_matches_wilder_reference(self):
        raw, _ = gf.tf_features(_walk(500, freq="1h", seed=3))
        c = _walk(500, freq="1h", seed=3)["close"].to_numpy()
        d = np.diff(c)
        g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
        ag, al = g[:14].mean(), l[:14].mean()
        ref = {14: 100 - 100 / (1 + ag / al)}
        for i in range(14, len(d)):
            ag = (ag * 13 + g[i]) / 14
            al = (al * 13 + l[i]) / 14
            ref[i + 1] = 100 - 100 / (1 + ag / al)
        for i in (14, 100, 300, 499):
            self.assertAlmostEqual(raw["rsi14"].iloc[i], ref[i], places=6)

    def test_kdj_matches_manual_recursion(self):
        df = _walk(200, freq="1h", seed=4)
        raw, _ = gf.tf_features(df)
        hh = df["high"].rolling(9).max().to_numpy()
        ll = df["low"].rolling(9).min().to_numpy()
        rsv = (df["close"].to_numpy() - ll) / (hh - ll) * 100
        k = d = 50.0
        for i in range(8, 200):
            k = k * 2 / 3 + rsv[i] / 3
            d = d * 2 / 3 + k / 3
        self.assertAlmostEqual(raw["kdj_k"].iloc[-1], k, places=6)
        self.assertAlmostEqual(raw["kdj_d"].iloc[-1], d, places=6)
        self.assertAlmostEqual(raw["kdj_j"].iloc[-1], 3 * k - 2 * d, places=6)


class TestNaNNeverZero(unittest.TestCase):
    def test_short_history_is_nan(self):
        raw, st = gf.tf_features(_walk(60, freq="1h"))
        self.assertTrue(st["sma_B"].isna().all())      # needs 200 bars
        self.assertTrue(st["sma_A"].isna().all())      # needs 120 bars
        self.assertTrue(st["boll_bw_bucket"].isna().all())
        self.assertTrue(st["agg_all"].isna().all())
        self.assertTrue(st["macd_cross"].iloc[:30].isna().all())
        self.assertTrue(st["rsi_zone"].iloc[:13].isna().all())
        self.assertFalse(st["rsi_zone"].iloc[-1:].isna().any())

    def test_aggregate_nan_if_any_member_nan(self):
        _, st = gf.tf_features(_walk(150, freq="1h"))   # sma_A ready (120) but sma_B (200) not
        self.assertTrue(st["agg_trend"].isna().all())
        self.assertTrue(st["sma_A"].iloc[-1] in (-1, 0, 1))

    def test_flat_series_gives_nan_not_zero_for_undefined_ratios(self):
        idx = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
        flat = pd.DataFrame({"open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 1.0}, index=idx)
        raw, st = gf.tf_features(flat)
        self.assertTrue(st["boll_zone"].isna().all())   # zero band width: %B undefined
        self.assertTrue(raw["wr14"].isna().all())


class TestNoLookAhead(unittest.TestCase):
    def test_decision_table_ignores_future_bars(self):
        df = _walk(12 * 24 * 60, seed=7)                # 60 days of 5m
        base = gf.decision_table(df)
        cut = df.index[12 * 24 * 45 + 7]                # inside an hour, 45 days in
        pert = df.copy()
        rng = np.random.default_rng(1)
        m = pert.index >= cut
        for c in ("open", "high", "low", "close"):
            pert.loc[m, c] = pert.loc[m, c] * (1 + rng.normal(0, 0.05, m.sum()))
        alt = gf.decision_table(pert)
        known = base.index[base.index + pd.Timedelta(hours=1) <= cut]
        self.assertGreater(len(known), 1000)
        a, b = base.loc[known], alt.loc[known]
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        self.assertFalse(base.loc[base.index[-5:]].equals(alt.loc[alt.index[-5:]]))   # but the future rows do differ

    def test_decision_row_uses_only_closed_higher_bars(self):
        df = _walk(12 * 24 * 30, seed=2)
        dt = gf.decision_table(df)
        h = gf.resample_ohlcv(df, 60)
        self.assertEqual(dt.index.freq, None) if False else None
        t = dt.index[500]                                # decision at close of hour bar t
        d4 = gf.resample_ohlcv(df, 240)
        last4 = d4.index[d4.index + pd.Timedelta(hours=4) <= t + pd.Timedelta(hours=1)][-1]
        raw4, _ = gf.tf_features(d4)
        self.assertAlmostEqual(dt.loc[t, "240__rsi14"], raw4.loc[last4, "rsi14"], places=9)
        raw1, _ = gf.tf_features(h)
        self.assertAlmostEqual(dt.loc[t, "60__rsi14"], raw1.loc[t, "rsi14"], places=9)


class TestOverallSignal(unittest.TestCase):
    def test_signal_is_sign_of_overall_aggregate(self):
        dt = gf.decision_table(_walk(12 * 24 * 400, seed=5, freq="5min"))
        ok = dt["overall_agg"].notna()
        self.assertGreater(ok.sum(), 100)
        np.testing.assert_array_equal(dt.loc[ok, "signal"].to_numpy(), np.sign(dt.loc[ok, "overall_agg"]).to_numpy())
        self.assertTrue(dt.loc[~ok, "signal"].isna().all())


if __name__ == "__main__":
    unittest.main()

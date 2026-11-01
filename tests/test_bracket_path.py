"""tools/bracket_path.py: path-resolved 1:1 bracket backtest on synthetic paths with known outcomes.

    python -m pytest -q tests/test_bracket_path.py

Groups: barrier outcomes (TP / SL / timeout / gap / order across bars / short), ambiguity (both barriers in one bar: stop-first, tp-first
bound, finer bars resolving it), costs and slippage, sizing (fixed / ATR / predicted range), entry option, skipped samples, the random
direction null, the confidence decile table, the PredictionTable adapter, argument validation.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bracket_path as bp  # noqa: E402
from stats import prediction_table as pt  # noqa: E402

T0 = pd.Timestamp("2025-01-01", tz="UTC")
H = pd.Timedelta("1h")


def candles(bars, asset="A", t0=T0, step=H):
    """bars: list of (open, high, low, close); the first one is the signal bar."""
    rows = [(asset, t0 + i * step, *b) for i, b in enumerate(bars)]
    return pd.DataFrame(rows, columns=["asset", "timestamp", "open", "high", "low", "close"])


def sig(pred=1.0, asset="A", ts=T0, **extra):
    return pd.DataFrame({"asset": [asset], "timestamp": [ts], "pred": [pred], **{k: [v] for k, v in extra.items()}})


FLAT = (100, 100.2, 99.8, 100)           # inside +-1%
SIGBAR = (100, 100.2, 99.8, 100)


def run(bars, pred=1.0, hold=3, **kw):
    kw.setdefault("cost_rt", 0.0)
    kw.setdefault("stop_slip", 0.0)
    kw.setdefault("null_reps", 0)
    kw.setdefault("sizing", "fixed")
    kw.setdefault("size_value", 0.01)
    return bp.run_backtest(sig(pred), candles([SIGBAR] + bars), timeframe="1h", hold_bars=hold, **kw)


class OutcomeTests(unittest.TestCase):
    def test_long_tp_sl_timeout(self):
        tp = run([(100, 101.5, 99.9, 101), FLAT, FLAT]).trades.iloc[0]
        self.assertEqual((tp.reason, tp.bar), ("tp", 0))
        self.assertAlmostEqual(tp.gross, 0.01)
        sl = run([(100, 100.1, 98.9, 99), FLAT, FLAT]).trades.iloc[0]
        self.assertEqual(sl.reason, "sl")
        self.assertAlmostEqual(sl.gross, -0.01)
        to = run([FLAT, FLAT, (100, 100.5, 99.5, 100.3)]).trades.iloc[0]
        self.assertEqual((to.reason, to.bar), ("timeout", 2))
        self.assertAlmostEqual(to.gross, 0.003)

    def test_short_mirrors_long(self):
        r = run([(100, 100.1, 98.5, 99), FLAT, FLAT], pred=-2.0).trades.iloc[0]
        self.assertEqual((r.reason, r.side), ("tp", -1))
        self.assertAlmostEqual(r.gross, 0.01)
        r = run([(100, 101.5, 99.9, 101), FLAT, FLAT], pred=-0.1).trades.iloc[0]
        self.assertEqual(r.reason, "sl")
        self.assertAlmostEqual(r.gross, -0.01)
        r = run([FLAT, FLAT, (100, 100.5, 99.5, 100.3)], pred=-1).trades.iloc[0]
        self.assertAlmostEqual(r.gross, -0.003)

    def test_order_across_bars_is_resolved_by_the_path(self):
        # target on bar 2, stop only on bar 3: a single-horizon max/min view would call it ambiguous, the path says TP
        r = run([FLAT, (100, 101.2, 99.9, 101), (101, 101.1, 98.0, 98)]).trades.iloc[0]
        self.assertEqual((r.reason, r.bar), ("tp", 1))
        self.assertFalse(r.ambiguous)
        # stop on bar 2, target on bar 3
        r = run([FLAT, (100, 100.1, 98.8, 99), (99, 102, 98.9, 101)]).trades.iloc[0]
        self.assertEqual((r.reason, r.bar), ("sl", 1))

    def test_gap_through_stop_fills_at_the_open(self):
        r = run([(98, 98.5, 97.5, 98), FLAT, FLAT]).trades.iloc[0]
        self.assertEqual(r.reason, "sl")
        self.assertAlmostEqual(r.gross, -0.02)
        self.assertFalse(r.ambiguous)

    def test_gap_through_target_fills_at_the_target(self):
        r = run([(102, 103, 100.5, 102), FLAT, FLAT]).trades.iloc[0]
        self.assertEqual(r.reason, "tp")
        self.assertAlmostEqual(r.gross, 0.01)

    def test_zero_prediction_is_no_trade(self):
        res = run([FLAT, FLAT, FLAT], pred=0.0)
        self.assertEqual(res.summary["n"], 0)
        self.assertEqual(res.summary["skipped"]["no_direction"], 1)


class AmbiguityTests(unittest.TestCase):
    BOTH = (100, 101.5, 98.5, 100)

    def test_both_touched_is_stop_first_and_flagged(self):
        res = run([self.BOTH, FLAT, FLAT])
        r = res.trades.iloc[0]
        self.assertEqual(r.reason, "sl")
        self.assertTrue(r.ambiguous)
        self.assertEqual(res.summary["ambiguous_share"], 1.0)
        self.assertEqual(res.summary["both_touched_exits"], 1)

    def test_tp_first_is_the_upper_bound(self):
        r = run([self.BOTH, FLAT, FLAT], ambiguous="tp_first").trades.iloc[0]
        self.assertEqual(r.reason, "tp")
        self.assertAlmostEqual(r.gross, 0.01)
        self.assertTrue(r.ambiguous)

    def _fine(self, bars):
        return candles(bars, step=pd.Timedelta("15min"), t0=T0 + H)       # the 4 fine bars of model bar 1

    def test_finer_bars_resolve_the_order(self):
        up_first = self._fine([(100, 101.5, 99.9, 101), (101, 101.2, 98.5, 99), FLAT, FLAT])
        r = run([self.BOTH, FLAT, FLAT], fine_candles=up_first)
        t = r.trades.iloc[0]
        self.assertEqual(t.reason, "tp")
        self.assertTrue(t.fine and not t.ambiguous)
        self.assertEqual((r.summary["resolved_by_fine"], r.summary["ambiguous_share"]), (1, 0.0))
        dn_first = self._fine([(100, 100.1, 98.5, 99), (99, 101.5, 98.9, 101), FLAT, FLAT])
        t = run([self.BOTH, FLAT, FLAT], fine_candles=dn_first).trades.iloc[0]
        self.assertEqual(t.reason, "sl")
        self.assertTrue(t.fine and not t.ambiguous)

    def test_finer_bar_touching_both_stays_ambiguous(self):
        t = run([self.BOTH, FLAT, FLAT], fine_candles=self._fine([self.BOTH, FLAT, FLAT, FLAT])).trades.iloc[0]
        self.assertTrue(t.ambiguous)
        self.assertEqual(t.reason, "sl")

    def test_missing_finer_bars_stay_ambiguous(self):
        other = candles([FLAT] * 4, asset="B", step=pd.Timedelta("15min"), t0=T0 + H)
        t = run([self.BOTH, FLAT, FLAT], fine_candles=other).trades.iloc[0]
        self.assertTrue(t.ambiguous)

    def test_short_with_fine_bars(self):
        fine = self._fine([(100, 100.1, 98.5, 99), (99, 101.5, 98.9, 101), FLAT, FLAT])      # down first -> short takes profit
        t = run([self.BOTH, FLAT, FLAT], pred=-1, fine_candles=fine).trades.iloc[0]
        self.assertEqual(t.reason, "tp")
        self.assertTrue(t.fine)


class CostTests(unittest.TestCase):
    def test_costs_and_stop_slippage(self):
        win = run([(100, 101.5, 99.9, 101), FLAT, FLAT], cost_rt=0.0008, stop_slip=0.0003).trades.iloc[0]
        self.assertAlmostEqual(win.net, 0.01 - 0.0008)
        loss = run([(100, 100.1, 98.9, 99), FLAT, FLAT], cost_rt=0.0008, stop_slip=0.0003).trades.iloc[0]
        self.assertAlmostEqual(loss.net, -0.01 - 0.0008 - 0.0003)
        to = run([FLAT, FLAT, (100, 100.5, 99.5, 100.3)], cost_rt=0.0008, stop_slip=0.0003).trades.iloc[0]
        self.assertAlmostEqual(to.net, 0.003 - 0.0008)

    def test_breakeven_win_rate(self):
        s = run([(100, 101.5, 99.9, 101), FLAT, FLAT], cost_rt=0.0008, stop_slip=0.0).summary
        self.assertAlmostEqual(s["breakeven_win"], 0.5 + 0.0008 / 0.02)         # p = 0.5 + c / 2d
        s = run([(100, 101.5, 99.9, 101), FLAT, FLAT], cost_rt=0.0008, stop_slip=0.0004).summary
        self.assertAlmostEqual(s["breakeven_win"], (0.01 + 0.0008 + 0.0004) / (0.02 + 0.0004))


class SizingTests(unittest.TestCase):
    def test_atr_sizing_uses_bars_up_to_the_signal_bar_only(self):
        n = 40
        pre = [(100, 100.5, 99.5, 100)] * n                      # TR = 1, ATR = 1, close 100 -> 1%
        future = [(100, 103, 97, 100), (100, 100.1, 99.9, 100)]  # a huge future bar must not leak into the ATR
        c = candles(pre + future)
        ts = T0 + (n - 1) * H
        res = bp.run_backtest(sig(1.0, ts=ts), c, timeframe="1h", hold_bars=2, sizing="atr", size_value=2.0, cost_rt=0, stop_slip=0, null_reps=0)
        t = res.trades.iloc[0]
        self.assertAlmostEqual(t.dist, 0.02)

    def test_natr_column_overrides(self):
        res = bp.run_backtest(sig(1.0, natr_pct=0.8), candles([SIGBAR, FLAT, FLAT, FLAT]), timeframe="1h", hold_bars=3, sizing="atr",
                              size_value=1.5, cost_rt=0, stop_slip=0, null_reps=0)
        self.assertAlmostEqual(res.trades.iloc[0].dist, 0.012)

    def test_atr_not_available_early_skips(self):
        res = run([FLAT, FLAT, FLAT], sizing="atr", size_value=1.0)
        self.assertEqual(res.summary["n"], 0)
        self.assertEqual(res.summary["skipped"]["bad_size"], 1)

    def test_pred_range_is_min_of_the_rooms_and_same_for_both_directions(self):
        c = candles([SIGBAR, (100, 100.5, 99.5, 100), FLAT, FLAT])
        for p in (1.0, -1.0):
            res = bp.run_backtest(sig(p, pred_high=102.0, pred_low=99.0), c, timeframe="1h", hold_bars=3, sizing="pred_range", size_value=1.0,
                                  cost_rt=0, stop_slip=0, null_reps=0)
            self.assertAlmostEqual(res.trades.iloc[0].dist, 0.01)
        with self.assertRaises(ValueError):
            bp.run_backtest(sig(1.0), c, timeframe="1h", sizing="pred_range")

    def test_min_and_max_dist_clip_only_when_given(self):
        a = run([FLAT, FLAT, FLAT], min_dist=0.02).trades.iloc[0].dist
        b = run([FLAT, FLAT, FLAT], max_dist=0.005).trades.iloc[0].dist
        self.assertAlmostEqual(a, 0.02)
        self.assertAlmostEqual(b, 0.005)


class EntryAndSkipTests(unittest.TestCase):
    def test_next_open_entry(self):
        bars = [(101, 102.2, 100.9, 102), FLAT, FLAT]
        close = run(bars, entry="close").trades.iloc[0]
        nxt = run(bars, entry="next_open").trades.iloc[0]
        self.assertEqual(close.entry, 100)
        self.assertEqual(nxt.entry, 101)
        self.assertEqual(close.reason, "tp")          # the open 101 is already at the +1% target of entry 100
        self.assertEqual(nxt.reason, "tp")
        self.assertAlmostEqual(nxt.dist, 0.01)

    def test_missing_future_bar_or_signal_bar_or_coin_skips_never_fills(self):
        c = candles([SIGBAR, FLAT, FLAT])                                    # needs 3 future bars
        res = bp.run_backtest(sig(1.0), c, timeframe="1h", hold_bars=3, cost_rt=0, stop_slip=0, null_reps=0, size_value=0.01)
        self.assertEqual((res.summary["n"], res.summary["skipped"]["missing_bars"]), (0, 1))
        res = bp.run_backtest(sig(1.0, asset="Z"), candles([SIGBAR, FLAT, FLAT, FLAT]), timeframe="1h", hold_bars=3, null_reps=0)
        self.assertEqual(res.summary["skipped"]["no_candles"], 1)
        hole = candles([SIGBAR, FLAT, FLAT, FLAT]).drop(index=2)             # a gap in the middle
        res = bp.run_backtest(sig(1.0), hole, timeframe="1h", hold_bars=3, null_reps=0)
        self.assertEqual(res.summary["n"], 0)

    def test_duplicate_signal_raises(self):
        s = pd.concat([sig(), sig()])
        with self.assertRaises(ValueError):
            bp.run_backtest(s, candles([SIGBAR, FLAT]), timeframe="1h", hold_bars=1)

    def test_validation(self):
        c = candles([SIGBAR, FLAT, FLAT, FLAT])
        for kw in ({"sizing": "kelly"}, {"entry": "mid"}, {"ambiguous": "coin"}, {"hold_bars": 0}, {"cost_rt": -1}, {"size_value": 0}):
            with self.assertRaises(ValueError, msg=str(kw)):
                bp.run_backtest(sig(), c, timeframe="1h", **{"hold_bars": 3, **kw})
        with self.assertRaises(ValueError):
            bp.run_backtest(sig().drop(columns="pred"), c, timeframe="1h")


def _walk(n_coins=4, n_bars=700, seed=0, vol=0.004):
    rng = np.random.default_rng(seed)
    cs = []
    for k in range(n_coins):
        r = rng.normal(0, vol, n_bars)
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.r_[100.0, close[:-1]]
        hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n_bars)))
        lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n_bars)))
        cs.append(pd.DataFrame({"asset": f"C{k}", "timestamp": T0 + np.arange(n_bars) * H, "open": open_, "high": hi, "low": lo, "close": close}))
    return pd.concat(cs, ignore_index=True)


class NullAndConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.c = _walk()
        rows = [(a, T0 + i * H) for a in self.c["asset"].unique() for i in range(30, 600, 3)]
        self.s = pd.DataFrame(rows, columns=["asset", "timestamp"])
        self.rng = np.random.default_rng(1)
        fwd = self.c.set_index(["asset", "timestamp"])["close"]
        self.truth = np.array([np.sign(fwd[(a, ts + 6 * H)] - fwd[(a, ts)]) for a, ts in zip(self.s["asset"], self.s["timestamp"])])

    def _res(self, s, **kw):
        kw.setdefault("null_reps", 300)
        return bp.run_backtest(s, self.c, timeframe="1h", hold_bars=6, sizing="fixed", size_value=0.006, seed=3, **kw)

    def test_random_signals_are_not_distinguishable_from_the_null(self):
        res = self._res(self.s.assign(pred=self.rng.choice([-1.0, 1.0], len(self.s))))
        self.assertGreater(res.null["null_p"], 0.02)
        self.assertLess(res.null["null_p"], 0.98)
        expected = 0.5 * (res.trades.net_long + res.trades.net_short).mean() * 1e4
        self.assertAlmostEqual(res.null["null_mean_bp"], expected, delta=3.0)
        self.assertLess(res.summary["net_per_trade_bp"], 0)                      # costs make a coin flip a loser

    def test_foresight_beats_the_null(self):
        # direction = sign of the realised close 6 bars later (look-ahead on purpose: the null must be far below it)
        res = self._res(self.s.assign(pred=self.truth))
        self.assertGreater(res.summary["net_per_trade_bp"], res.null["null_mean_bp"] + 20)
        self.assertEqual(res.null["null_p"], 0.0)
        self.assertGreater(res.summary["t_day_clustered"], 3)
        self.assertGreater(res.summary["horizon_dir_acc"], 0.99)
        self.assertEqual(res.summary["coins_positive"], "4/4")
        self.assertGreater(res.summary["profit_factor"], 1.0)

    def test_confidence_deciles_rise_when_confidence_is_informative(self):
        conf = self.rng.random(len(self.s))
        right = self.rng.random(len(self.s)) < (0.5 + 0.45 * conf)            # the more confident, the more often right
        res = self._res(self.s.assign(pred=np.where(right, self.truth, -self.truth), confidence=conf), null_reps=0)
        t = res.by_confidence
        self.assertEqual(len(t), 10)
        self.assertGreater(res.confidence_stats["hit_vs_confidence_spearman"], 0.2)
        self.assertGreater(res.confidence_stats["top_minus_bottom_hit"], 0.3)
        self.assertGreater(t["hit_rate"].iloc[-1], t["hit_rate"].iloc[0])
        self.assertEqual(t["n"].sum(), res.summary["n"])

    def test_uninformative_confidence_is_flat(self):
        res = self._res(self.s.assign(pred=self.rng.choice([-1.0, 1.0], len(self.s)), confidence=self.rng.random(len(self.s))), null_reps=0)
        self.assertLess(abs(res.confidence_stats["hit_vs_confidence_spearman"]), 0.1)

    def test_long_short_split_and_by_coin_tables(self):
        res = self._res(self.s.assign(pred=self.rng.choice([-1.0, 1.0], len(self.s))), null_reps=0)
        self.assertEqual(set(res.by_side["side_name"]), {"long", "short"})
        self.assertEqual(res.by_side["n"].sum(), res.summary["n"])
        self.assertEqual(len(res.by_coin), 4)
        self.assertIsNone(res.by_confidence)                                    # no confidence column -> no decile table

    def test_report_text(self):
        txt = bp.format_report(self._res(self.s.assign(pred=self.rng.choice([-1.0, 1.0], len(self.s))), null_reps=20))
        for word in ("win rate", "break-even", "PF", "random-direction null", "ambiguous", "per coin"):
            self.assertIn(word, txt)


class AdapterTests(unittest.TestCase):
    def _table(self, outputs, n=4):
        ts = [T0 + i * H for i in range(n)]
        return pt.build_table(asset="A", timestamp=ts, outputs=outputs, split_role="test", train_end=T0 - H, fold=0, identity_hash="h",
                              horizon_bars=6, timeframe="1h")

    def test_class_head_wins_over_regression_and_table_is_untouched(self):
        t = self._table({"close": [0.1, -0.2, 0.3, -0.4], "close_p_up": [0.9, 0.2, 0.6, 0.1]})
        before = t.copy()
        s = bp.signals_from_table(t, head="close")
        np.testing.assert_allclose(s["pred"], [0.4, -0.3, 0.1, -0.4])
        np.testing.assert_allclose(s["confidence"], [0.4, 0.3, 0.1, 0.4])
        self.assertEqual(s.attrs["confidence_source"], "out_close_p_up")
        pd.testing.assert_frame_equal(t, before)

    def test_regression_only_and_custom_confidence(self):
        t = self._table({"close": [0.1, -0.2, 0.3, -0.4], "close_confidence": [1, 2, 3, 4]})
        s = bp.signals_from_table(t, head="close", confidence="out_close_confidence")
        np.testing.assert_allclose(s["pred"], [0.1, -0.2, 0.3, -0.4])
        np.testing.assert_allclose(s["confidence"], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            bp.signals_from_table(t, head="nothing")

    def test_predicted_range_decoding_and_split_role(self):
        t = self._table({"close": [1, 1, 1, 1], "high": [0.02] * 4, "low": [-0.01] * 4})
        c = candles([(100, 101, 99, 100)] * 4)
        s = bp.signals_from_table(t, head="close", candles=c, reg_scale=1.0)
        np.testing.assert_allclose(s["pred_high"], 101 * 1.02)
        np.testing.assert_allclose(s["pred_low"], 99 * 0.99)
        self.assertEqual(len(bp.signals_from_table(t, head="close", split_role="val")), 0)

    def test_end_to_end_from_a_table(self):
        t = self._table({"close": [1.0, 1.0, 1.0]}, n=3)
        res = bp.run_backtest(bp.signals_from_table(t, head="close"), candles([FLAT] * 12), timeframe="1h", hold_bars=3, size_value=0.01, null_reps=5)
        self.assertEqual(res.summary["n"], 3)

    def test_resample_ohlc_keeps_complete_bars_only(self):
        fine = candles([(100 + i, 101 + i, 99 + i, 100.5 + i) for i in range(10)], step=pd.Timedelta("15min"))
        r = bp.resample_ohlc(fine, "1h")
        self.assertEqual(len(r), 2)                                          # 10 fine bars -> 2 complete hours
        row = r.iloc[0]
        self.assertEqual((row.open, row.high, row.low, row.close), (100, 104, 99, 103.5))


class CliTests(unittest.TestCase):
    def test_cli_from_written_table_and_candles_dir(self):
        import contextlib
        import io
        import tempfile
        c = _walk(n_coins=2, n_bars=120)
        ts = [T0 + i * H for i in range(20, 100, 4)]
        out = {"close": np.random.default_rng(0).normal(size=len(ts)), "close_p_up": np.random.default_rng(1).random(len(ts))}
        tabs = [pt.build_table(asset=a, timestamp=ts, outputs=out, split_role="test", train_end=T0, fold=0, identity_hash="h", horizon_bars=6,
                               timeframe="1h") for a in ("C0", "C1")]
        with tempfile.TemporaryDirectory() as d:
            tpath = os.path.join(d, "t.csv.gz")
            pt.write_table(tpath, pd.concat(tabs, ignore_index=True))
            for a, g in c.groupby("asset"):
                g.set_index("timestamp")[["open", "high", "low", "close"]].to_parquet(os.path.join(d, f"ohlc_{a}USDT_1h.parquet"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = bp.main(["--table", tpath, "--candles-dir", d, "--symbol-suffix", "USDT", "--timeframe", "1h", "--hold-bars", "6",
                              "--size-value", "0.006", "--split-role", "test", "--null-reps", "20", "--out-json", os.path.join(d, "o.json")])
            self.assertEqual(rc, 0)
            self.assertIn("random-direction null", buf.getvalue())
            self.assertTrue(os.path.exists(os.path.join(d, "o.json")))


if __name__ == "__main__":
    unittest.main()

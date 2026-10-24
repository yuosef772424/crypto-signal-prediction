"""tools/stop_grid_sim.py: hand-computed fills/exits/PnL for every exit mode, gap fill at the open, slippage, intrabar orderings
(LB <= UB), rebuild triggers, cancel-opposite, caps, no look-ahead in the gates and in the simulator, the null baselines
(gross ~ 0, net ~ -cost on synthetic martingale data with fixed seeds), the metrics and the coin selection.
SYNTHETIC data only: nothing here is evidence about real markets.
    python -m pytest tests/test_stop_grid_sim.py -q
"""
import gzip
import os
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stop_grid_sim as sg  # noqa: E402

MIN = sg.MS_MIN
T0 = 1_700_000_000_000 // sg.MS_DAY * sg.MS_DAY          # a UTC midnight


def bars(rows, ts=None):
    """rows: (o, h, l, c) per bar; consecutive minutes from a UTC midnight unless ts is given."""
    a = np.array(rows, dtype="float64")
    t = (T0 + np.arange(len(a), dtype="int64") * MIN) if ts is None else np.asarray(ts, dtype="int64")
    return t, a[:, 0], a[:, 1], a[:, 2], a[:, 3]


# one level per side, 1 percent spacing around 100: buy stop 101, sell stop 99; SL 2 percent, TP 2 percent; no slippage
BASE = dict(n_levels=1, spacing=0.01, sl=0.02, tp=0.02, act=0.01, gap=0.005, rebuild="once", daily_loss_cap_bp=None,
            taker_bp=5.0, maker_bp=2.0, slip_bps=0.0, max_positions=10)


def cfg(**kw):
    d = dict(BASE)
    d.update(kw)
    return sg.GridConfig(**d)


FLAT = (100.0, 100.0, 100.0, 100.0)


class ExitModes(unittest.TestCase):
    def test_tp_long(self):
        data = bars([FLAT, (100, 101.5, 100, 101.5), (101.5, 103.1, 101.4, 103.0)])
        for od in sg.ORDERINGS:
            d = sg.simulate(*data, cfg(exit_mode="tp"), od)
            self.assertEqual(len(d), 1)
            r = d.iloc[0]
            self.assertEqual(r["exit"], "tp")
            self.assertAlmostEqual(r["gross_bp"], 200.0, places=6)               # 103.02 / 101 - 1
            self.assertAlmostEqual(r["gross_noslip_bp"], 200.0, places=6)
            self.assertAlmostEqual(r["net_bp"], 200.0 - 5.0 - 2.0, places=6)     # taker entry + maker TP
            self.assertEqual(r["side"], 1)

    def test_tp_short(self):
        data = bars([FLAT, (100, 100, 98.5, 98.5), (98.5, 98.6, 96.9, 97.0)])
        d = sg.simulate(*data, cfg(exit_mode="tp"), "hl")
        self.assertEqual(len(d), 1)
        r = d.iloc[0]
        self.assertEqual(r["side"], -1)
        self.assertAlmostEqual(r["gross_bp"], (1 - 97.02 / 99.0) * 1e4, places=6)  # short 99 -> TP 99*0.98
        self.assertAlmostEqual(r["net_bp"], r["gross_bp"] - 7.0, places=6)

    def test_stop_loss_and_cancel_opposite(self):
        data = bars([FLAT, (100, 101.2, 100, 101.2), (101.2, 101.3, 98.5, 98.6)])
        d = sg.simulate(*data, cfg(exit_mode="tp", cancel_opposite_on_fill=True), "hl")
        self.assertEqual(len(d), 1)                       # the sell stop was cancelled when the buy stop filled
        r = d.iloc[0]
        self.assertEqual(r["exit"], "sl")
        self.assertAlmostEqual(r["gross_bp"], (98.98 / 101.0 - 1) * 1e4, places=6)
        self.assertAlmostEqual(r["net_bp"], r["gross_bp"] - 10.0, places=6)      # taker both sides
        d2 = sg.simulate(*data, cfg(exit_mode="tp", cancel_opposite_on_fill=False), "hl")
        self.assertEqual(len(d2), 2)                      # default: both sides stay live; the short (99) is closed at the end
        self.assertEqual(sorted(d2["side"]), [-1, 1])
        self.assertEqual(d2["exit"].tolist(), ["sl", "eod"])

    def test_trailing_long_and_orderings(self):
        # entry 101; activation 101*1.01 = 102.01; gap 0.5 percent
        data = bars([FLAT, (100, 101.5, 100, 101.5), (101.5, 103.0, 102.0, 102.5), (102.5, 104.0, 102.5, 104.0)])
        c = cfg(exit_mode="trail")
        hl = sg.simulate(*data, c, "hl")
        lh = sg.simulate(*data, c, "lh")
        # O-H-L-C: up to 103 (activates, trail 102.485), down to 102.0 hits the trail
        self.assertEqual(hl.iloc[0]["exit"], "trail")
        self.assertAlmostEqual(hl.iloc[0]["gross_bp"], (103.0 * 0.995 / 101.0 - 1) * 1e4, places=6)
        self.assertAlmostEqual(hl.iloc[0]["net_bp"], hl.iloc[0]["gross_bp"] - 10.0, places=6)
        # O-L-H-C: down to 102 first (nothing), up to 103 (activates), close 102.5 (above the trail), then 104: end of data close
        self.assertEqual(lh.iloc[0]["exit"], "eod")
        self.assertAlmostEqual(lh.iloc[0]["gross_bp"], (104.0 / 101.0 - 1) * 1e4, places=6)
        res = sg.run_both(*data, c)
        self.assertEqual(res["LB_order"], "hl")
        self.assertEqual(res["UB_order"], "lh")
        self.assertLess(res["LB"]["net_bp"], res["UB"]["net_bp"])

    def test_trailing_short(self):
        # short 99, activation 99*0.99 = 98.01, trail gap 0.5 percent above the low
        data = bars([FLAT, (100, 100, 98.5, 98.5), (98.5, 98.5, 97.0, 97.2), (97.2, 97.6, 97.2, 97.5)])
        d = sg.simulate(*data, cfg(exit_mode="trail"), "hl")
        self.assertEqual(len(d), 1)
        r = d.iloc[0]
        self.assertEqual(r["side"], -1)
        self.assertEqual(r["exit"], "trail")             # low 97 -> trail 97*1.005 = 97.485 hit on the way up in bar 3
        self.assertAlmostEqual(r["gross_bp"], (1 - 97.485 / 99.0) * 1e4, places=6)

    def test_trail_not_activated_stays_on_initial_stop(self):
        data = bars([FLAT, (100, 101.5, 100, 101.5), (101.5, 101.9, 98.0, 98.2)])   # never reaches 102.01; SL 98.98
        d = sg.simulate(*data, cfg(exit_mode="trail", cancel_opposite_on_fill=True), "hl")
        self.assertEqual(d.iloc[0]["exit"], "sl")
        self.assertAlmostEqual(d.iloc[0]["gross_bp"], (98.98 / 101.0 - 1) * 1e4, places=6)

    def test_split_half_tp_half_trailing(self):
        data = bars([FLAT, (100, 101.5, 100, 101.5), (101.5, 103.1, 101.4, 103.0)])
        d = sg.simulate(*data, cfg(exit_mode="split", cancel_opposite_on_fill=True), "hl")
        self.assertEqual(len(d), 1)                      # one parent position, two legs
        r = d.iloc[0]
        tp_leg = 0.5 * (103.02 / 101.0 - 1) * 1e4
        trail_leg = 0.5 * (103.1 * 0.995 / 101.0 - 1) * 1e4
        self.assertAlmostEqual(r["gross_bp"], tp_leg + trail_leg, places=6)
        fees = 5.0 + 0.5 * 2.0 + 0.5 * 5.0               # taker entry (whole), maker on the TP half, taker on the trailing half
        self.assertAlmostEqual(r["net_bp"], tp_leg + trail_leg - fees, places=6)
        self.assertEqual(sorted(r["exit"].split("+")), ["tp", "trail"])


class Fills(unittest.TestCase):
    def test_gap_fills_at_open_adverse(self):
        data = bars([FLAT, (102.0, 102.5, 101.8, 102.2), (102.2, 104.1, 102.0, 104.0)])      # bar 1 opens above the 101 trigger
        d = sg.simulate(*data, cfg(exit_mode="tp", cancel_opposite_on_fill=True), "hl")
        r = d.iloc[0]
        self.assertAlmostEqual(r["gross_bp"], (102.0 * 1.02 / 102.0 - 1) * 1e4, places=6)     # entry 102 (open), TP 104.04
        self.assertEqual(r["entry_bar"], 1)
        # slippage on a gap fill is added to the open: entry 102 * 1.001
        d2 = sg.simulate(*data, cfg(exit_mode="tp", cancel_opposite_on_fill=True, slip_bps=10.0), "hl")
        self.assertAlmostEqual(d2.iloc[0]["gross_bp"], (104.04 / (102.0 * 1.001) - 1) * 1e4, places=6)
        self.assertAlmostEqual(d2.iloc[0]["gross_noslip_bp"], 200.0, places=6)

    def test_gap_down_through_stop_fills_at_open(self):
        data = bars([FLAT, (100, 101.5, 100, 101.5), (97.0, 97.5, 96.5, 97.0)])               # long 101, SL 98.98, gap to 97
        d = sg.simulate(*data, cfg(exit_mode="tp", cancel_opposite_on_fill=True), "hl")
        self.assertAlmostEqual(d.iloc[0]["gross_bp"], (97.0 / 101.0 - 1) * 1e4, places=6)

    def test_slip_range_fraction(self):
        data = bars([FLAT, (100, 101.5, 100, 101.5), (101.5, 103.1, 101.4, 103.0)])
        d = sg.simulate(*data, cfg(exit_mode="tp", cancel_opposite_on_fill=True, slip_range_frac=0.1), "hl")
        # bar 1 range 1.5 -> entry 101 + 0.15; the TP limit gets no slippage
        self.assertAlmostEqual(d.iloc[0]["gross_bp"], (103.02 / 101.15 - 1) * 1e4, places=6)
        self.assertAlmostEqual(d.iloc[0]["gross_noslip_bp"], 200.0, places=6)

    def test_stop_exit_slippage(self):
        data = bars([FLAT, (100, 101.2, 100, 101.2), (101.2, 101.3, 98.5, 98.6)])
        d = sg.simulate(*data, cfg(exit_mode="tp", cancel_opposite_on_fill=True, slip_bps=10.0), "hl")
        e, x = 101.0 * 1.001, 98.98 * (1 - 0.001)
        self.assertAlmostEqual(d.iloc[0]["gross_bp"], (x / e - 1) * 1e4, places=6)
        self.assertAlmostEqual(d.iloc[0]["gross_noslip_bp"], (98.98 / 101.0 - 1) * 1e4, places=6)
        self.assertLess(d.iloc[0]["gross_bp"], d.iloc[0]["gross_noslip_bp"])

    def test_empty_when_nothing_touches(self):
        data = bars([FLAT] + [(100, 100.5, 99.5, 100)] * 5)
        d = sg.simulate(*data, cfg(), "hl")
        self.assertEqual(len(d), 0)
        s = sg.summarize(d, 6, 1.0)
        self.assertEqual(s["trades"], 0)


class Triggers(unittest.TestCase):
    def test_once_builds_once(self):
        data = bars([FLAT] + [(100, 100.5, 99.5, 100)] * 4 + [(100, 103.5, 100, 103.5)] + [(103.5, 103.6, 103.4, 103.5)] * 4)
        _, st = sg.simulate(*data, cfg(rebuild="once", exit_mode="tp"), "hl", return_stats=True)
        self.assertEqual(st["builds"], 1)

    def test_breakout_rebuilds_around_new_price(self):
        # n=1, s=1 percent, X=1: outermost 101, rebuild when a close > 102 or < 98
        rows = [FLAT, (100, 101.5, 100, 101.5), (101.5, 103.2, 101.5, 103.0), (103.0, 103.0, 101.9, 102.0), (102.0, 102.0, 102.0, 102.0)]
        c = cfg(rebuild="breakout", breakout_x=1.0, exit_mode="tp", cancel_opposite_on_fill=True)
        d, st = sg.simulate(*bars(rows), c, "hl", return_stats=True)
        self.assertEqual(st["builds"], 2)             # bar 3 rebuilds around close[2] = 103: sells at 101.97 -> fills in bar 3
        self.assertIn(-1, d["side"].tolist())
        short = d[d["side"] == -1].iloc[0]
        self.assertEqual(short["entry_bar"], 3)
        self.assertAlmostEqual(short["gross_noslip_bp"], (1 - 102.0 / 101.97) * 1e4, places=4)   # eod close 102
        # same path with 'once': no rebuild, no short
        d1, st1 = sg.simulate(*bars(rows), cfg(rebuild="once", exit_mode="tp", cancel_opposite_on_fill=True), "hl", return_stats=True)
        self.assertEqual(st1["builds"], 1)
        self.assertNotIn(-1, d1["side"].tolist())

    def test_flat_rebuild_after_cooldown(self):
        # long fills in bar 1, TP in bar 2 (flat), cooldown 2 -> rebuild at bar 4
        rows = [FLAT, (100, 101.5, 100, 101.5), (101.5, 103.1, 101.5, 103.0)] + [(103, 103, 103, 103)] * 6
        _, st = sg.simulate(*bars(rows), cfg(rebuild="flat", flat_cooldown=2, exit_mode="tp", cancel_opposite_on_fill=True), "hl",
                            return_stats=True)
        self.assertEqual(st["builds"], 2)
        _, st = sg.simulate(*bars(rows[:4]), cfg(rebuild="flat", flat_cooldown=2, exit_mode="tp", cancel_opposite_on_fill=True),
                            "hl", return_stats=True)
        self.assertEqual(st["builds"], 1)             # data ends before the cooldown has elapsed
        # no fill, no rebuild (flat is armed by a fill)
        _, st = sg.simulate(*bars([FLAT] + [(100, 100.2, 99.8, 100)] * 8), cfg(rebuild="flat", flat_cooldown=2), "hl",
                            return_stats=True)
        self.assertEqual(st["builds"], 1)

    def test_time_rebuild(self):
        rows = [FLAT] + [(100, 100.2, 99.8, 100)] * 10             # bars 1..10
        _, st = sg.simulate(*bars(rows), cfg(rebuild="time", time_every=3), "hl", return_stats=True)
        self.assertEqual(st["builds"], 4)                          # bars 1, 4, 7, 10

    def test_gate_blocks_building_until_true(self):
        rows = [FLAT, (100, 101.5, 100, 101.5), (101.5, 101.6, 101.4, 101.5), (101.5, 103.1, 101.5, 103.0), (103, 103, 103, 103)]
        gate = np.array([False, False, False, True, True])
        d, st = sg.simulate(*bars(rows), cfg(exit_mode="tp", rebuild="once"), "hl", gate=gate, return_stats=True)
        # the first possible grid is built at bar 3 around close[2] = 101.5 (buy stop 102.515): the 101-fill of bar 1 never exists
        self.assertEqual(st["builds"], 1)
        self.assertTrue((d["entry_bar"] >= 3).all())
        d0, _ = sg.simulate(*bars(rows), cfg(exit_mode="tp", rebuild="once"), "hl", gate=np.zeros(5, bool), return_stats=True)
        self.assertEqual(len(d0), 0)


class Caps(unittest.TestCase):
    def test_max_positions_cancels_the_rest(self):
        rows = [FLAT, (100, 103.5, 100, 103.5), (103.5, 103.6, 103.4, 103.5)]
        c3 = cfg(n_levels=3, spacing=0.01, exit_mode="tp", cancel_opposite_on_fill=True, max_positions=3)
        d, st = sg.simulate(*bars(rows), c3, "hl", return_stats=True)
        self.assertEqual(len(d), 3)
        self.assertEqual(st["skipped_cap"], 0)
        d, st = sg.simulate(*bars(rows), c3.replace(max_positions=1), "hl", return_stats=True)
        self.assertEqual(len(d), 1)
        self.assertEqual(st["skipped_cap"], 2)

    def test_daily_loss_cap_halts_the_day_and_resumes_next_day(self):
        rows = [FLAT, (100, 101.2, 100, 101.2), (101.2, 101.3, 98.5, 98.6),     # long stopped: net -210 bp -> halt (cap 100)
                (98.6, 100.5, 98.0, 100.0), (100, 100, 100, 100),               # same day: nothing may be built or filled
                (100, 101.5, 100.0, 101.5), (101.5, 101.5, 101.5, 101.5)]       # next day: a fresh grid around 100 fills 101
        ts = [T0 + i * MIN for i in range(5)] + [T0 + sg.MS_DAY + 1 * MIN, T0 + sg.MS_DAY + 2 * MIN]
        ts[4] = T0 + 4 * MIN
        # the new day starts at bar 5: the grid is gone (halt), so a new one is built at bar 5 around close[4] = 100
        c = cfg(exit_mode="tp", cancel_opposite_on_fill=True, daily_loss_cap_bp=100.0, rebuild="time", time_every=2)
        d, st = sg.simulate(*bars(rows, ts), c, "hl", return_stats=True)
        self.assertEqual(st["halted_days"], 1)
        self.assertEqual(len(d), 2)                                             # the stopped long + one fill on day 2
        self.assertTrue(d.iloc[1]["entry_bar"] == 5)
        d_nocap, _ = sg.simulate(*bars(rows, ts), c.replace(daily_loss_cap_bp=None), "hl", return_stats=True)
        self.assertGreater(len(d_nocap), 2)                                     # without the cap the same-day bars trade again


class NoLookAhead(unittest.TestCase):
    def test_gates_use_only_past_bars(self):
        n, k = 4000, 2501
        ts, o, h, l, c = sg.synth_ohlc(n, 3)
        ts2, o2, h2, l2, c2 = sg.synth_ohlc(n, 99)           # a completely different future
        for gates, tf in (("adx", "1m"), ("adx", "5m"), ("vol", "1m"), ("cost", "1m")):
            cf = sg.GridConfig(gates=(gates,), adx_tf=tf, adx_thr=15.0, vol_ratio=1.0, cost_k=0.5)
            g1 = sg.gate_array(ts, o, h, l, c, cf)
            f = lambda a, b: np.r_[a[:k], b[k:]]            # noqa: E731  identical up to bar k-1, different afterwards
            g2 = sg.gate_array(ts, f(o, o2), f(h, h2), f(l, l2), f(c, c2), cf)
            np.testing.assert_array_equal(g1[:k + 1], g2[:k + 1], err_msg=f"{gates}/{tf}")   # gate[k] sees bars <= k-1 only
            self.assertTrue(g1.any() and not g1.all(), f"{gates}/{tf} gate should vary on this data")

    def test_gate_values_do_not_depend_on_the_current_bar(self):
        ts, o, h, l, c = sg.synth_ohlc(1500, 5)
        a = sg.adx_signal(ts, h, l, c, 14, "1m")
        h2, l2, c2 = h.copy(), l.copy(), c.copy()
        h2[700], l2[700], c2[700] = h[700] * 1.5, l[700] * 0.5, c[700] * 1.2           # only bar 700 changes
        b = sg.adx_signal(ts, h2, l2, c2, 14, "1m")
        np.testing.assert_array_equal(a[:701], b[:701])
        self.assertFalse(np.array_equal(a[701:710], b[701:710]))

    def test_simulator_is_causal(self):
        n, k = 6000, 4000
        ts, o, h, l, c = sg.synth_ohlc(n, 11)
        ts2, o2, h2, l2, c2 = sg.synth_ohlc(n, 12)
        f = lambda a, b: np.r_[a[:k], b[k:]]                # noqa: E731
        for cf in (sg.GridConfig(exit_mode="trail", rebuild="breakout", gates=("vol",), vol_ratio=1.0),
                   sg.GridConfig(exit_mode="split", rebuild="time", time_every=20),
                   sg.GridConfig(exit_mode="tp", rebuild="flat")):
            for od in sg.ORDERINGS:
                a = sg.simulate(ts, o, h, l, c, cf, od)
                b = sg.simulate(ts, f(o, o2), f(h, h2), f(l, l2), f(c, c2), cf, od)
                pd.testing.assert_frame_equal(a[a["exit_bar"] < k].reset_index(drop=True),
                                              b[b["exit_bar"] < k].reset_index(drop=True))
                self.assertGreater((a["exit_bar"] < k).sum(), 20)


class Orderings(unittest.TestCase):
    def test_lb_not_above_ub_and_trades_exist(self):
        ts, o, h, l, c = sg.synth_ohlc(30_000, 4)
        for cf in (sg.GridConfig(), sg.GridConfig(exit_mode="tp", rebuild="flat"), sg.GridConfig(exit_mode="split", rebuild="time")):
            r = sg.run_both(ts, o, h, l, c, cf)
            self.assertLessEqual(r["LB"]["net_bp"], r["UB"]["net_bp"])
            self.assertGreater(r["LB"]["trades"], 100)
            self.assertNotEqual(r["LB_order"], r["UB_order"])

    def test_config_validation(self):
        for bad in (dict(exit_mode="x"), dict(rebuild="y"), dict(gates=("adxx",)), dict(adx_tf="3m"), dict(spacing=0.0),
                    dict(max_positions=0)):
            with self.assertRaises(ValueError):
                sg.GridConfig(**bad)
        with self.assertRaises(TypeError):
            sg.GridConfig(typo=1)
        c = sg.GridConfig(taker_bp=5, maker_bp=2, slip_bps=1).scaled_costs(2.0)
        self.assertEqual((c.taker_bp, c.maker_bp, c.slip_bps), (10.0, 4.0, 2.0))

    def test_speed(self):
        ts, o, h, l, c = sg.synth_ohlc(100_000, 6)
        t = time.time()
        sg.run_both(ts, o, h, l, c, sg.GridConfig(gates=("adx", "vol", "cost")))
        self.assertLess(time.time() - t, 30.0)           # ~0.3 s expected; 1 year of 1m bars is 5x this


class Nulls(unittest.TestCase):
    """Martingale data: gross before fees ~ 0, net ~ -cost. Stated tolerances on the pooled trades of 4 fixed-seed series of
    120k bars (per ordering): |day-clustered t| of the gross without slippage < 3.0 and |mean| < 3 bp (a free overshoot of
    ~1 bp per fill, the bug this guards against, shows as t > 5), and the mean net bp per trade between -cost_hi and
    -cost_lo, i.e. the analytic cost +-3 bp (stop exits cost 5+5 taker + 2 slippage = 12 bp, a maker TP 5+2+1 = 8 bp)."""

    @classmethod
    def setUpClass(cls):
        cls.data = [sg.synth_ohlc(120_000, s) for s in (21, 22, 23, 24)]

    def pooled(self, cf, make=None):
        parts = []
        for i, (ts, o, h, l, c) in enumerate(self.data):
            if make is not None:
                o, h, l, c = make(ts, o, h, l, c, 100 + i)
            for od in sg.ORDERINGS:
                d = sg.simulate(ts, o, h, l, c, cf, od)
                d["order"] = od
                d["entry_ts"] = d["entry_ts"] + i * 10_000 * sg.MS_DAY          # distinct days per series (clustering)
                parts.append(d)
        return pd.concat(parts, ignore_index=True)

    def check(self, d, cost_lo, cost_hi):
        for od, x in d.groupby("order"):
            n = len(x)
            self.assertGreater(n, 5000)
            t = sg.day_clustered_t(x["gross_noslip_bp"].to_numpy(), x["entry_ts"].to_numpy())
            self.assertLess(abs(t), 3.0, f"{od}: gross_noslip t={t:.2f}")
            self.assertLess(abs(x["gross_noslip_bp"].mean()), 3.0, od)
            self.assertGreater(x["net_bp"].mean(), -cost_hi, od)
            self.assertLess(x["net_bp"].mean(), -cost_lo, od)
            self.assertLess(x["gross_bp"].mean(), x["gross_noslip_bp"].mean() + 0.2)      # slippage only ever costs

    def test_random_walk_null_stop_exits(self):
        self.check(self.pooled(sg.GridConfig(exit_mode="trail", rebuild="breakout")), 9.0, 15.0)

    def test_random_walk_null_tp(self):
        self.check(self.pooled(sg.GridConfig(exit_mode="tp", rebuild="flat")), 7.0, 13.0)

    def test_sign_flip_null_and_shuffled_null(self):
        cf = sg.GridConfig(exit_mode="split", rebuild="time", time_every=30)
        self.check(self.pooled(cf, lambda ts, o, h, l, c, s: sg.sign_flip_null(ts, o, h, l, c, s)), 8.0, 15.0)
        self.check(self.pooled(cf, lambda ts, o, h, l, c, s: sg.random_walk_null(ts, o, h, l, c, s)), 8.0, 15.0)

    def test_null_constructions(self):
        ts, o, h, l, c = self.data[0]
        n = 20_000
        ts, o, h, l, c = ts[:n], o[:n], h[:n], l[:n], c[:n]
        fo, fh, fl, fc = sg.sign_flip_null(ts, o, h, l, c, 1)
        self.assertTrue(np.all(fh >= np.maximum(fo, fc) - 1e-9) and np.all(fl <= np.minimum(fo, fc) + 1e-9))
        r = np.diff(np.log(c))
        fr = np.diff(np.log(fc))
        np.testing.assert_allclose(np.abs(fr), np.abs(r), rtol=1e-6, atol=1e-12)      # the volatility structure is kept
        day = (ts[1:] // sg.MS_DAY)
        for d in np.unique(day)[:5]:                                                    # whole days flip together
            m = day == d
            ratio = np.sign(fr[m] * r[m])
            self.assertTrue(np.all(ratio == ratio[0]))
        self.assertTrue((np.sign(fr * r) < 0).any() and (np.sign(fr * r) > 0).any())
        ro, rh, rl, rc = sg.random_walk_null(ts, o, h, l, c, 2)
        self.assertTrue(np.all(rh >= np.maximum(ro, rc) - 1e-9) and np.all(rl <= np.minimum(ro, rc) + 1e-9))
        self.assertAlmostEqual(float(np.mean(np.diff(np.log(rc)) > 0)), 0.5, delta=0.02)


class Metrics(unittest.TestCase):
    def test_day_clustered_t(self):
        x = np.array([1.0, -2.0, 3.0, 0.5, -1.0, 2.0])
        t_each_day = (T0 + np.arange(6) * sg.MS_DAY)
        mu = x.mean()
        self.assertAlmostEqual(sg.day_clustered_t(x, t_each_day), mu / (np.sqrt(((x - mu) ** 2).sum()) / 6), places=9)
        same_day = np.full(6, T0)
        se = np.sqrt((x.sum() - 6 * mu) ** 2) / 6                  # one cluster: the cluster sum minus its expected sum = 0
        self.assertTrue(np.isnan(sg.day_clustered_t(x, same_day)) or se == 0.0)

    def test_max_drawdown_and_time_in_market(self):
        self.assertEqual(sg.max_drawdown_bp([1, 2, -5, 1, -1, 3], np.arange(6)), 5.0)
        self.assertEqual(sg.max_drawdown_bp([-1, -1], [1, 0]), 2.0)
        tr = pd.DataFrame({"entry_ts": [T0, T0 + 5 * MIN], "entry_bar": [2, 6], "exit_bar": [4, 6], "side": [1, 1], "level": [1, 1],
                           "gross_bp": [10.0, -4.0], "gross_noslip_bp": [11.0, -3.0], "net_bp": [3.0, -11.0], "exit": ["tp", "sl"]})
        s = sg.summarize(tr, 10, 2.0)
        self.assertAlmostEqual(s["time_in_market"], 4 / 10)       # bars 2,3,4 and 6
        self.assertAlmostEqual(s["fills_per_day"], 1.0)
        self.assertAlmostEqual(s["win_rate"], 0.5)
        self.assertAlmostEqual(s["net_bp"], -4.0)
        self.assertEqual(sg.sign_consistency({"a": 1.0, "b": -2.0, "c": 3.0, "d": float("nan")}), (2, 3))


def _write_symbol(dirpath, name, df, kind="csv.gz", sub=False):
    if sub:
        os.makedirs(os.path.join(dirpath, name), exist_ok=True)
        path = os.path.join(dirpath, name, f"{name}_part.csv")
        df.to_csv(path, index=False)
        return
    path = os.path.join(dirpath, f"{name}.{kind}")
    if kind == "csv.gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


class Selection(unittest.TestCase):
    def build(self, tmp, tail_boost=None):
        n = 20_000                                            # ~13.9 days
        vols = {"AAA": 0.02, "BBB": 0.05, "CCC": 0.04, "DDD": 0.03, "EEE": 0.07, "FFF": 0.06, "GGG": 0.01, "ILLIQ": 0.09, "SHORT": 0.08}
        for i, (s, v) in enumerate(vols.items()):
            f = sg.synth_frame(n, 100 + i, daily_vol=v, qv_per_min=5_000.0 if s == "ILLIQ" else 80_000.0, start_ms=T0)
            if s == "SHORT":
                f = f.iloc[: int(n * 0.55)]                   # fine inside SELECT but gone before DEV/TEST: must be excluded
            if tail_boost == s:
                k = int(n * 0.5)
                f.loc[k:, ["high", "low"]] = f.loc[k:, ["high", "low"]].to_numpy() * np.array([[1.05, 0.95]])
            _write_symbol(tmp, s, f, kind="csv.gz" if i % 2 == 0 else "csv", sub=(s == "FFF"))
        return vols

    def test_list_symbols_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp)
            syms = sg.list_symbols(tmp)
            self.assertEqual(set(syms), {"AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "ILLIQ", "SHORT"})
            d = sg.read_symbol(syms["FFF"], ["close"], T0 + 10 * MIN, T0 + 20 * MIN)
            self.assertEqual(len(d), 10)

    def test_selection_top_by_range_with_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.build(tmp)
            sel, table, sp = sg.select_coins(tmp, top=3, min_quote_volume=20_000, min_coverage=0.9, verbose=False)
            self.assertEqual(sel, ["EEE", "FFF", "BBB"])                        # by median range; ILLIQ (0.09 vol) excluded by volume
            row = table.set_index("symbol")
            self.assertIn("quote volume", row.loc["ILLIQ", "why"])
            self.assertIn("span", row.loc["SHORT", "why"])
            self.assertFalse(bool(row.loc["ILLIQ", "eligible"]))
            self.assertGreater(row.loc["EEE", "score"], row.loc["BBB", "score"])
            # splits: 40 / 30 / 30 of the common range, contiguous, in order
            (a, b), (b2, c), (c2, d) = sp.select, sp.dev, sp.test
            self.assertEqual((b, c), (b2, c2))
            span = d - a
            self.assertAlmostEqual((b - a) / span, 0.4, delta=0.001)
            self.assertAlmostEqual((c - b) / span, 0.3, delta=0.001)
            self.assertAlmostEqual((d - c) / span, 0.3, delta=0.001)

    def test_selection_ignores_dev_and_test_data(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            self.build(t1)
            self.build(t2, tail_boost="GGG")          # GGG (least volatile) becomes huge in the last half: DEV/TEST only
            s1, _, _ = sg.select_coins(t1, top=3, verbose=False)
            s2, _, _ = sg.select_coins(t2, top=3, verbose=False)
            self.assertEqual(s1, s2)

    def test_coverage_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = 10_000
            _write_symbol(tmp, "FULL", sg.synth_frame(n, 1, 0.03, start_ms=T0))
            _write_symbol(tmp, "OTHER", sg.synth_frame(n, 2, 0.02, start_ms=T0))
            holes = sg.synth_frame(n, 3, 0.09, start_ms=T0).iloc[::2]          # every other minute missing: coverage 0.5
            _write_symbol(tmp, "HOLES", holes)
            sel, table, _ = sg.select_coins(tmp, top=3, verbose=False)
            self.assertEqual(sel, ["FULL", "OTHER"])
            self.assertIn("coverage", table.set_index("symbol").loc["HOLES", "why"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("pyarrow"), "pyarrow not installed")
    def test_parquet_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = sg.synth_frame(500, 1, start_ms=T0)
            os.makedirs(os.path.join(tmp, "XXX"))
            f.to_parquet(os.path.join(tmp, "XXX", "a.parquet"))
            d = sg.read_symbol(sg.list_symbols(tmp)["XXX"])
            self.assertEqual(len(d), 500)

PIN_CFGS = {
    "trail": dict(spacing=0.001, sl=0.005, exit_mode="trail", rebuild="breakout"),
    "tp_time": dict(spacing=0.002, sl=0.01, exit_mode="tp", rebuild="time", time_every=30),
    "split_flat": dict(spacing=0.001, sl=0.005, exit_mode="split", rebuild="flat", slip_range_frac=0.1),
    "gated_cap": dict(spacing=0.001, sl=0.005, exit_mode="trail", rebuild="breakout", gates=("vol",), vol_ratio=1.0,
                      daily_loss_cap_bp=150.0, max_positions=3),
    "adx_cancel": dict(spacing=0.0015, sl=0.007, exit_mode="split", rebuild="once", gates=("adx",), adx_tf="5m", adx_thr=15.0,
                       cancel_opposite_on_fill=True),
}


def _pin_data(seed, vol):
    f = sg.synth_frame(6 * 1440, seed, daily_vol=vol)
    return f, f["timestamp"].to_numpy("int64"), *(f[k].to_numpy(float) for k in ("open", "high", "low", "close"))


class SignalTf5m(unittest.TestCase):
    """signal_tf='5m': every gate indicator on UTC-aligned 5m bars, a 1m bar sees the last COMPLETED 5m bar. SYNTHETIC data only."""
    KW = dict(gates=("adx", "vol", "cost"), signal_tf="5m", adx_tf="5m", adx_n=2, atr_fast=2, atr_slow=3, cost_window=15,
              adx_thr=10.0, vol_ratio=0.0, cost_k=0.1, taker_bp=1.0, slip_bps=0.0)

    @staticmethod
    def series(n=60, seed=1, off=0):
        rng = np.random.default_rng(seed)
        c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) * (1 + rng.random(n) * 0.002)
        l = np.minimum(o, c) * (1 - rng.random(n) * 0.002)
        ts = T0 + off * MIN + np.arange(n, dtype="int64") * MIN
        return ts, o, h, l, c

    def test_resample_alignment_and_last_completed_rule(self):
        ts, o, h, l, c = self.series(30, off=2)               # starts at minute 2: first bucket holds only minutes 2,3,4
        h5, l5, c5, inv = sg.resample_5m(ts, h, l, c)
        self.assertEqual(list(inv[:4]), [0, 0, 0, 1])          # minutes 2,3,4 | 5..9
        self.assertEqual(len(h5), 7)                           # minutes 2..31 -> buckets 0..6 (last one: minutes 30,31)
        self.assertAlmostEqual(h5[1], h[3:8].max())
        self.assertAlmostEqual(l5[1], l[3:8].min())
        self.assertAlmostEqual(c5[1], c[7])                    # last close of the bucket
        v = sg.signal_values(ts, h, l, c, sg.GridConfig(**self.KW))
        # manual: bucket j (j>=1) sees the indicators of bucket j-1 (completed); first bucket sees nothing
        tr5 = np.maximum(h5 - l5, np.maximum(np.abs(h5 - np.r_[np.nan, c5[:-1]]), np.abs(l5 - np.r_[np.nan, c5[:-1]])))
        tr5[0] = h5[0] - l5[0]
        ratio = pd.Series(tr5).rolling(2).mean() / pd.Series(tr5).rolling(3).mean()
        med = pd.Series((h5 - l5) / c5 * 1e4).rolling(3).median()
        adx5 = sg._adx_values(h5, l5, c5, 2)
        for name, ser in (("vol", ratio.to_numpy()), ("cost", med.to_numpy()), ("adx", adx5)):
            exp = np.r_[np.nan, ser[:-1]][inv]
            np.testing.assert_allclose(v[name], exp, equal_nan=True, err_msg=name)
        self.assertTrue(np.isnan(v["vol"][:3]).all())          # nothing completed yet inside the first bucket

    def test_gate_series_piecewise_constant_over_5m_blocks(self):
        ts, o, h, l, c = sg.synth_ohlc(3000, 11, daily_vol=0.06)
        g = sg.gate_array(ts, o, h, l, c, sg.GridConfig(**{**self.KW, "adx_n": 14, "atr_fast": 14, "atr_slow": 100, "cost_window": 60}))
        blocks = ts // (5 * MIN)
        self.assertTrue(0 < g.mean() < 1)
        for b in np.unique(blocks):
            self.assertEqual(len(set(g[blocks == b].tolist())), 1, f"gate flips inside 5m block {b}")
        self.assertTrue((np.diff(g.astype(int)) != 0).any())

    def test_causal_future_bars_and_forming_bar_do_not_change_past(self):
        ts, o, h, l, c = sg.synth_ohlc(2000, 5, daily_vol=0.06)
        conf = sg.GridConfig(**{**self.KW, "adx_n": 14, "atr_fast": 14, "atr_slow": 100, "cost_window": 60}, spacing=0.0015,
                             rebuild="breakout")
        base_g = sg.gate_array(ts, o, h, l, c, conf)
        base = sg.simulate(ts, o, h, l, c, conf, "hl")
        for k in (1003, 1202):                                    # 1003 and 1202 sit inside a 5m bucket (not at its start)
            o2, h2, l2, c2 = (a.copy() for a in (o, h, l, c))
            o2[k:] *= 1.07; h2[k:] *= 1.09; l2[k:] *= 0.93; c2[k:] *= 1.05
            g2 = sg.gate_array(ts, o2, h2, l2, c2, conf)
            first_in_bucket = k - int((ts[k] // MIN) % 5)           # earlier minutes of the same bucket
            np.testing.assert_array_equal(g2[:k], base_g[:k])      # also the minutes before k INSIDE the forming bar
            self.assertGreaterEqual(k - first_in_bucket, 0)
            d2 = sg.simulate(ts, o2, h2, l2, c2, conf, "hl")
            a = base[base["exit_bar"] < k].reset_index(drop=True)
            b = d2[d2["exit_bar"] < k].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b)
        # the bars of the CURRENT bucket never feed the gate seen inside it
        k = 1003
        bucket_start = k - int((ts[k] // MIN) % 5)
        o2, h2, l2, c2 = (a.copy() for a in (o, h, l, c))
        h2[k] *= 1.5; l2[k] *= 0.5; c2[k] *= 1.3
        g2 = sg.gate_array(ts, o2, h2, l2, c2, conf)
        np.testing.assert_array_equal(g2[bucket_start:k + 1], base_g[bucket_start:k + 1])

    def test_invalid_values_and_conflict_raise(self):
        with self.assertRaises(ValueError):
            sg.GridConfig(signal_tf="15m")
        with self.assertRaises(ValueError):
            sg.GridConfig(signal_tf="5m", adx_tf="1m")
        self.assertEqual(sg.GridConfig().signal_tf, "1m")
        sg.GridConfig(signal_tf="5m", adx_tf="5m")

    def test_default_signal_tf_equals_explicit_1m_and_deterministic_5m(self):
        ts, o, h, l, c = sg.synth_ohlc(4000, 21, daily_vol=0.06)
        kw = dict(gates=("adx", "vol", "cost"), adx_thr=15.0, vol_ratio=0.5, cost_k=0.2, taker_bp=1.0, slip_bps=0.0, spacing=0.0015)
        d0 = sg.simulate(ts, o, h, l, c, sg.GridConfig(**kw), "hl")
        d1 = sg.simulate(ts, o, h, l, c, sg.GridConfig(**kw, signal_tf="1m"), "hl")
        pd.testing.assert_frame_equal(d0, d1)
        c5 = sg.GridConfig(**kw, signal_tf="5m", adx_tf="5m")
        r1 = sg.simulate(ts, o, h, l, c, c5, "hl")
        r2 = sg.simulate(ts, o, h, l, c, c5, "hl")
        pd.testing.assert_frame_equal(r1, r2)
        self.assertGreater(len(r1), 0)

    def test_gaps_incomplete_bucket_and_missing_bucket(self):
        ts, o, h, l, c = self.series(40)
        keep = np.ones(40, bool)
        keep[[7, 8, 9]] = False                                   # bucket 1 keeps minutes 5,6 only
        keep[20:25] = False                                       # bucket 4 entirely absent
        t, h_, l_, c_ = ts[keep], h[keep], l[keep], c[keep]
        h5, l5, c5, inv = sg.resample_5m(t, h_, l_, c_)
        self.assertEqual(len(h5), 7)
        self.assertAlmostEqual(c5[1], c[6])                       # built from the minutes present
        self.assertAlmostEqual(h5[1], h[5:7].max())
        v = sg.signal_values(t, h_, l_, c_, sg.GridConfig(**self.KW))
        # the bucket after the absent one sees the last PRESENT bucket (no forward-filled phantom bar)
        j = np.flatnonzero(np.diff(inv) > 0)
        self.assertEqual(len(j), 6)
        self.assertTrue(np.isfinite(v["cost"][inv == 4]).all() and np.isfinite(v["cost"][inv == 5]).all())
        self.assertTrue(np.isnan(v["cost"][inv == 0]).all() and np.isnan(v["cost"][inv == 1]).all())


class TestPinnedMetricsAndRecording(unittest.TestCase):
    """Summary metrics and the full trade table of several configs were captured BEFORE the recording hooks were added
    (tests/stop_grid_sim_pinned_metrics.json): the hooks must change nothing, and recording on/off must agree."""

    def test_pinned_summary_and_trade_hash(self):
        import hashlib
        import json
        pinned = json.load(open(os.path.join(ROOT, "tests", "stop_grid_sim_pinned_metrics.json")))
        got = {}
        for seed, vol in ((11, 0.04), (12, 0.07)):
            f, ts, o, h, l, c = _pin_data(seed, vol)
            for name, kw in PIN_CFGS.items():
                r = sg.run_both(ts, o, h, l, c, sg.GridConfig(**kw))
                for od in sg.ORDERINGS:
                    d = r["trades"][od]
                    s = sg.summarize(d, len(ts), r["n_days"])
                    s["hash"] = hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()[:16]
                    got[f"{seed}|{name}|{od}"] = json.loads(json.dumps(s, default=lambda x: None if x != x else x))
        self.assertEqual(sorted(got), sorted(pinned))
        for k in pinned:
            self.assertEqual(got[k], pinned[k], k)

    def test_recording_does_not_change_trades_and_adds_columns(self):
        f, ts, o, h, l, c = _pin_data(11, 0.04)
        for name, kw in PIN_CFGS.items():
            cfg_ = sg.GridConfig(**kw)
            for od in sg.ORDERINGS:
                base = sg.simulate(ts, o, h, l, c, cfg_, od)
                ev = []
                rec = sg.simulate(ts, o, h, l, c, cfg_, od, record=True, events=ev)
                pd.testing.assert_frame_equal(rec[list(base.columns)], base)
                self.assertEqual(len(rec), len(base))
                if len(rec):
                    self.assertLess(np.abs(rec["net_bp"] - (rec["gross_bp"] - rec["fee_bp"])).max(), 1e-9)
                    self.assertLess(np.abs(rec["slip_bp"] - (rec["gross_noslip_bp"] - rec["gross_bp"])).max(), 1e-9)
                    self.assertTrue((rec["mfe_bp"] >= 0).all() and (rec["mae_bp"] <= 0).all())
                    self.assertTrue((rec["exit_ts"] >= rec["entry_ts"]).all())
                n_entry = sum(1 for e in ev if e[1] == "filled" and e[6].startswith("entry"))
                self.assertEqual(n_entry, len(base))

    def test_event_cap_and_hand_computed_events(self):
        t, o, h, l, c = bars([(100, 100, 100, 100), (100, 101.5, 100, 101.2), (101.2, 101.2, 98.9, 99.0)])
        ev = []
        d = sg.simulate(t, o, h, l, c, cfg(), "hl", record=True, events=ev)
        kinds = [e[1] for e in ev]
        self.assertEqual(kinds.count("placed"), 2)
        self.assertEqual(kinds.count("rebuilt"), 1)
        self.assertIn("filled", kinds)
        self.assertEqual(float(d["trigger_px"].iloc[0]), 101.0)
        self.assertEqual(float(d["entry_px"].iloc[0]), 101.0)
        self.assertEqual(int(d["concurrent"].iloc[0]), 1)
        capped = []
        sg.simulate(t, o, h, l, c, cfg(), "hl", events=capped, event_max=2)
        self.assertEqual(len(capped), 2)
        a = sg.annotate_trades(d, {"ts": t, "o": o, "h": h, "l": l, "c": c}, cfg(), "X", "hl")
        self.assertEqual(len(a), len(d))
        self.assertTrue(a["entry_quote_volume"].isna().all())
        self.assertEqual(len(sg.events_frame(ev, t, "X", "hl")), len(ev))


if __name__ == "__main__":
    unittest.main()

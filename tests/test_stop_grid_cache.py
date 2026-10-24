"""tools/stop_grid_sim.py SignalCache: caching must be bit-identical to recomputing (gate series incl. signal_tf='5m', run_both,
annotate_trades, null baselines), reuse series across configs, and refuse another coin's bars; the runner's evaluate / null_runs
give the same rows with and without the cache and for any --jobs. SYNTHETIC data only.
    python -m pytest tests/test_stop_grid_cache.py -q
"""
import importlib.util
import multiprocessing as mp
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stop_grid_sim as sg  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_stop_grid_1m_cache_test", os.path.join(ROOT, "research", "stop_grid_1m", "run_stop_grid_1m.py"))
rn = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rn                                  # pickling the worker functions (jobs > 1) needs the module importable by name
_spec.loader.exec_module(rn)

KW5 = dict(gates=("adx", "vol", "cost"), signal_tf="5m", adx_tf="5m", adx_n=7, atr_fast=6, atr_slow=30, cost_window=30, adx_thr=10.0,
           vol_ratio=0.5, cost_k=0.2)
CFGS = [
    sg.GridConfig(spacing=0.001, rebuild="breakout"),
    sg.GridConfig(spacing=0.001, rebuild="flat", gates=("adx",), adx_tf="5m", adx_thr=15.0),
    sg.GridConfig(spacing=0.002, rebuild="time", time_every=30, exit_mode="split", gates=("vol",), vol_ratio=1.0),
    sg.GridConfig(spacing=0.001, rebuild="breakout", exit_mode="tp", gates=("cost",), cost_k=0.3),
    sg.GridConfig(spacing=0.0015, rebuild="flat", exit_mode="split", **KW5),
    sg.GridConfig(spacing=0.001, rebuild="breakout", **{**KW5, "gates": ("vol", "cost")}),
    sg.GridConfig(spacing=0.001, rebuild="breakout", gates=("adx", "vol", "cost"), adx_thr=15.0, vol_ratio=0.8, cost_k=0.3),
    sg.GridConfig(spacing=0.001, rebuild="breakout", gates=("adx",), adx_tf="5m", signal_tf="1m", adx_thr=20.0),
]


def data(n=4 * 1440, seed=5, vol=0.05):
    return sg.synth_ohlc(n, seed, daily_vol=vol)


class GateCache(unittest.TestCase):
    def test_cached_gate_equals_recomputed_for_every_config_and_cost_scale(self):
        ts, o, h, l, c = data()
        cc = sg.SignalCache(ts, o, h, l, c)
        for cfg in CFGS:
            for mult in (1.0, 2.0, 0.5):                  # the cost threshold changes the gate, not the cached indicator
                cf = cfg.scaled_costs(mult)
                ref = sg.gate_array(ts, o, h, l, c, cf)
                got = sg.gate_array(ts, o, h, l, c, cf, cache=cc)
                np.testing.assert_array_equal(got, ref)
                self.assertTrue(got is None or not got.flags.writeable)
        self.assertGreater(cc.hits, 0)

    def test_signal_values_identical_and_series_reused(self):
        ts, o, h, l, c = data()
        cc = sg.SignalCache(ts, o, h, l, c)
        for cfg in CFGS:
            ref = sg.signal_values(ts, h, l, c, cfg)
            got = sg.signal_values(ts, h, l, c, cfg, cache=cc)
            self.assertEqual(sorted(ref), sorted(got))
            for k in ref:
                np.testing.assert_array_equal(got[k], ref[k], err_msg=k)
        misses = cc.misses
        a = sg.signal_values(ts, h, l, c, CFGS[0].replace(spacing=0.005, sl=0.02, exit_mode="tp"), cache=cc)   # no gate parameter differs
        b = sg.signal_values(ts, h, l, c, CFGS[0], cache=cc)
        self.assertEqual(cc.misses, misses)
        self.assertTrue(all(a[k] is b[k] for k in a))
        # a different gate parameter is a different series
        v = sg.signal_values(ts, h, l, c, CFGS[0].replace(adx_n=5), which=("adx",), cache=cc)
        self.assertEqual(cc.misses, misses + 1)
        self.assertFalse(np.array_equal(v["adx"], a["adx"], equal_nan=True))

    def test_clear_and_rebuild_give_the_same_series(self):
        ts, o, h, l, c = data()
        cc = sg.SignalCache(ts, o, h, l, c)
        g1 = sg.gate_array(ts, o, h, l, c, CFGS[4], cache=cc).copy()
        cc.clear()
        self.assertEqual((cc.hits, cc.misses), (0, 0))
        g2 = sg.gate_array(ts, o, h, l, c, CFGS[4], cache=cc)
        np.testing.assert_array_equal(g1, g2)

    def test_cache_refuses_other_bars(self):
        ts, o, h, l, c = data(seed=5)
        ts2, o2, h2, l2, c2 = data(seed=6)
        cc = sg.SignalCache(ts, o, h, l, c)
        with self.assertRaises(ValueError):
            sg.gate_array(ts2, o2, h2, l2, c2, CFGS[1], cache=cc)
        with self.assertRaises(ValueError):
            sg.signal_values(ts, h2, l, c, CFGS[1], cache=cc)
        self.assertTrue(cc.matches(ts.copy(), o.copy(), h.copy(), l.copy(), c.copy()))   # equal bars in other objects are fine
        self.assertEqual(sg.gate_array(ts, o, h, l, c, sg.GridConfig(), cache=cc), None)


class SimCache(unittest.TestCase):
    def test_run_both_with_and_without_cache_identical(self):
        ts, o, h, l, c = data()
        cc = sg.SignalCache(ts, o, h, l, c)
        for cfg in CFGS:
            a = sg.run_both(ts, o, h, l, c, cfg)
            b = sg.run_both(ts, o, h, l, c, cfg, cache=cc)
            for od in sg.ORDERINGS:
                pd.testing.assert_frame_equal(a["trades"][od], b["trades"][od], check_exact=True)
            for k in ("LB", "UB"):
                self.assertEqual(repr(a[k]), repr(b[k]))
            self.assertEqual((a["LB_order"], a["UB_order"]), (b["LB_order"], b["UB_order"]))
        self.assertGreater(a["LB"]["trades"] + b["LB"]["trades"], 0)

    def test_annotate_trades_and_nulls_identical(self):
        ts, o, h, l, c = data()
        cc = sg.SignalCache(ts, o, h, l, c)
        cfg = CFGS[4]
        d = sg.simulate(ts, o, h, l, c, cfg, "hl", record=True)
        arrs = {"ts": ts, "o": o, "h": h, "l": l, "c": c}
        pd.testing.assert_frame_equal(sg.annotate_trades(d, arrs, cfg, "X", "lb"), sg.annotate_trades(d, arrs, cfg, "X", "lb", cache=cc),
                                      check_exact=True)
        for seed in (1, 2, 3):
            for fn in (sg.sign_flip_null, sg.random_walk_null):
                for x, y in zip(fn(ts, o, h, l, c, seed), fn(ts, o, h, l, c, seed, cache=cc)):
                    np.testing.assert_array_equal(x, y)

    def test_runner_evaluate_cached_equals_uncached_and_gates_for(self):
        ts, o, h, l, c = data(3 * 1440)
        win = {"AAA": {"ts": ts, "o": o, "h": h, "l": l, "c": c}}
        for cfg in CFGS[:6]:
            r1, t1 = rn.evaluate(cfg, win, "cachetest", cache_gates=True)
            r2, t2 = rn.evaluate(cfg, win, "cachetest", cache_gates=False)
            self.assertEqual(repr(r1), repr(r2))
            pd.testing.assert_frame_equal(t1, t2, check_exact=True)
            if cfg.gates:
                np.testing.assert_array_equal(rn.gates_for(("cachetest", "AAA"), win["AAA"], cfg), sg.gate_array(ts, o, h, l, c, cfg))
        # a stale slot name with different bars is rebuilt, never served
        ts2, o2, h2, l2, c2 = data(3 * 1440, seed=9)
        win2 = {"AAA": {"ts": ts2, "o": o2, "h": h2, "l": l2, "c": c2}}
        r3, _ = rn.evaluate(CFGS[1], win2, "cachetest", cache_gates=True)
        r4, _ = rn.evaluate(CFGS[1], win2, "cachetest", cache_gates=False)
        self.assertEqual(repr(r3), repr(r4))

    @unittest.skipUnless("fork" in mp.get_all_start_methods(), "needs fork")
    def test_null_runs_serial_equals_parallel_and_prefix(self):
        test = {}
        for j, s in enumerate(("AAA", "BBB")):
            ts, o, h, l, c = data(2 * 1440, seed=20 + j)
            test[s] = {"ts": ts, "o": o, "h": h, "l": l, "c": c, "qv": np.ones(len(c))}
        cfg = CFGS[2]
        a = rn.null_runs(cfg, test, 3, jobs=1)
        b = rn.null_runs(cfg, test, 3, jobs=2)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        small = rn.null_runs(cfg, test, 2, jobs=1)                 # fewer draws = the same first draws
        pd.testing.assert_frame_equal(small, a[a["draw"] < 2].reset_index(drop=True), check_exact=True)
        # and identical to the original uncached construction
        ref = []
        for kind, fn in (("sign_flip", sg.sign_flip_null), ("random_walk", sg.random_walk_null)):
            for k in range(3):
                w = {}
                for j, (coin, x) in enumerate(test.items()):
                    oo, hh, ll, cc_ = fn(x["ts"], x["o"], x["h"], x["l"], x["c"], 1000 + 100 * k + j)
                    w[coin] = {"ts": x["ts"], "o": oo, "h": hh, "l": ll, "c": cc_}
                r, _ = rn.evaluate(cfg, w, None, cache_gates=False)
                ref.append({"null": kind, "draw": k, **{y: r[y] for y in ("trades_lb", "gross_lb", "gross_noslip_lb", "net_lb", "net_ub", "t_lb")}})
        pd.testing.assert_frame_equal(a, pd.DataFrame(ref), check_exact=True)


if __name__ == "__main__":
    unittest.main()

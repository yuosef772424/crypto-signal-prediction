"""Optional numba engine of tools/stop_grid_sim.py (tools/stop_grid_numba.py): parity with the python engine (same trades, exit
reasons, order; floats within rtol 1e-9; summary metrics too), the engine switch and its errors. The JIT tests are skipped when
numba is not installed (CI); the python path and the error paths are always tested. SYNTHETIC data only.
    python -m pytest tests/test_stop_grid_numba.py -q          # or: python tools/stop_grid_bench.py --parity
"""
import os
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stop_grid_bench as bench  # noqa: E402
import stop_grid_sim as sg  # noqa: E402

HAVE_NUMBA = sg.numba_available()


class EngineSwitch(unittest.TestCase):
    def test_default_is_python_and_explicit_python_is_identical(self):
        ts, o, h, l, c = bench.make_data(2 * 1440, 3)
        cfg = sg.GridConfig(spacing=0.001, rebuild="flat")
        a = sg.simulate(ts, o, h, l, c, cfg, "hl")
        b = sg.simulate(ts, o, h, l, c, cfg, "hl", engine="python")
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        self.assertGreater(len(a), 0)

    def test_unknown_engine_raises(self):
        ts, o, h, l, c = bench.make_data(300, 3)
        with self.assertRaises(ValueError):
            sg.simulate(ts, o, h, l, c, sg.GridConfig(), "hl", engine="cython")
        with self.assertRaises(ValueError):
            sg.run_both(ts, o, h, l, c, sg.GridConfig(), engine="gpu")

    def test_missing_numba_raises_clearly_never_falls_back(self):
        ts, o, h, l, c = bench.make_data(300, 3)
        with mock.patch.dict(sys.modules, {"numba": None, "stop_grid_numba": None}):
            with self.assertRaises(RuntimeError) as cm:
                sg.simulate(ts, o, h, l, c, sg.GridConfig(), "hl", engine="numba")
            self.assertIn("numba", str(cm.exception))
            self.assertFalse(sg.numba_available())
            with self.assertRaises(RuntimeError):
                sg.require_engine("numba")
        self.assertEqual(sg.require_engine("python"), "python")

    def test_parity_command_exits_2_without_numba(self):
        with mock.patch.object(sg, "numba_available", return_value=False):
            self.assertEqual(bench.run_parity(days=1, seeds=(1,), verbose=False), 2)

    def test_bench_helpers_detect_a_mismatch(self):
        a = pd.DataFrame({"x": [1, 2], "y": [1.0, 2.0], "e": ["tp", "sl"]})
        self.assertTrue(bench.frames_match(a, a.copy())[0])
        for col, val in (("x", 3), ("y", 2.0000001), ("e", "tp+sl")):
            b = a.copy()
            b.loc[1, col] = val
            self.assertFalse(bench.frames_match(a, b)[0], col)
        self.assertFalse(bench.frames_match(a, a.iloc[:1])[0])
        self.assertTrue(bench.frames_match(a, a.assign(y=a["y"] * (1 + 1e-12)))[0])


@unittest.skipUnless(HAVE_NUMBA, "numba not installed")
class NumbaParity(unittest.TestCase):
    def test_parity_all_configs_orderings_seeds(self):
        for seed in (11, 12):
            for gaps in (True, False):
                ts, o, h, l, c = bench.make_data(5 * 1440, seed, gaps=gaps, daily_vol=0.04 + 0.03 * (seed - 11))
                for name, kw in bench.PARITY_CFGS.items():
                    cfg = sg.GridConfig(**kw)
                    for order in sg.ORDERINGS + sg.DIAG_ORDERINGS:
                        for record in (False, True):
                            bad, _ = bench.check_pair(ts, o, h, l, c, cfg, order, record, f"{seed}|{gaps}|{name}|{order}|{record}")
                            self.assertEqual(bad, [])

    def test_pinned_configs_trade_table_identical_and_columns(self):
        for seed, vol in ((11, 0.04), (12, 0.07)):
            f = sg.synth_frame(6 * 1440, seed, daily_vol=vol)
            ts, o, h, l, c = (f[k].to_numpy(dtype) for k, dtype in (("timestamp", "int64"), ("open", float), ("high", float), ("low", float), ("close", float)))
            for name in ("trail", "tp_time", "split_flat", "gated_cap", "adx_cancel"):
                cfg = sg.GridConfig(**bench.PARITY_CFGS[name])
                a = sg.run_both(ts, o, h, l, c, cfg)
                b = sg.run_both(ts, o, h, l, c, cfg, engine="numba")
                for od in sg.ORDERINGS:
                    pd.testing.assert_frame_equal(a["trades"][od], b["trades"][od], check_exact=False, rtol=1e-9)
                    self.assertEqual(list(a["trades"][od].columns), list(b["trades"][od].columns))
                    self.assertEqual(a["trades"][od]["exit"].tolist(), b["trades"][od]["exit"].tolist())
                    self.assertEqual(((a["trades"][od]["exit_bar"] - a["trades"][od]["entry_bar"])).tolist(),
                                     ((b["trades"][od]["exit_bar"] - b["trades"][od]["entry_bar"])).tolist())
                for k in ("LB", "UB"):
                    ok, why = bench.summaries_match(a[k], b[k])
                    self.assertTrue(ok, why)
                self.assertEqual((a["LB_order"], a["UB_order"]), (b["LB_order"], b["UB_order"]))

    def test_no_trades_and_tiny_inputs(self):
        ts, o, h, l, c = bench.make_data(300, 3)
        flat = np.full(300, 100.0)
        for engine in ("python", "numba"):
            d = sg.simulate(ts, flat, flat, flat, flat, sg.GridConfig(), "hl", engine=engine)
            self.assertEqual(len(d), 0)
            self.assertEqual(list(d.columns), ["entry_ts", *sg.TRADE_COLUMNS])
        a = sg.simulate(ts[:2], o[:2], h[:2], l[:2], c[:2], sg.GridConfig(), "hl")
        b = sg.simulate(ts[:2], o[:2], h[:2], l[:2], c[:2], sg.GridConfig(), "hl", engine="numba")
        pd.testing.assert_frame_equal(a, b)

    def test_events_log_is_not_supported_and_raises(self):
        ts, o, h, l, c = bench.make_data(300, 3)
        with self.assertRaises(ValueError):
            sg.simulate(ts, o, h, l, c, sg.GridConfig(), "hl", engine="numba", events=[])

    def test_explicit_gate_and_cache_path(self):
        ts, o, h, l, c = bench.make_data(3 * 1440, 4)
        cfg = sg.GridConfig(**bench.PARITY_CFGS["all_gates_5m"])
        cc = sg.SignalCache(ts, o, h, l, c)
        a = sg.run_both(ts, o, h, l, c, cfg)
        b = sg.run_both(ts, o, h, l, c, cfg, engine="numba", cache=cc)
        for od in sg.ORDERINGS:
            pd.testing.assert_frame_equal(a["trades"][od], b["trades"][od], rtol=1e-9)

    def test_parity_cli_exit_code(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "stop_grid_bench.py"), "--parity", "--parity-days", "2", "--parity-seeds", "1"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PARITY OK", r.stdout)


if __name__ == "__main__":
    unittest.main()

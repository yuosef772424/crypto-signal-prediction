"""tools/natr_range.py + research/natr_range/run_natr_range.py (E-natr-001): causal NATR range quantile forecast, exact quantile
regression, trailing-90 baseline, deciles from train only, the fixed accept rule, run contract.
    python -m pytest tests/test_natr_range.py -q
"""
import importlib.util
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import natr_range as nr  # noqa: E402


def _bars(n, seed=0, start="2024-01-01", freq="1h", vol=0.01):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": 1.0}, index=idx)


class TestResample(unittest.TestCase):
    def test_label_is_open_and_incomplete_bar_dropped(self):
        d5 = _bars(48, freq="5min", start="2024-01-01 00:00")          # 4 full hours
        d5 = d5.drop(d5.index[17])                                      # hour 01:00 loses one member
        h = nr.resample_bars(d5, 60)
        self.assertEqual(list(h.index.hour), [0, 2, 3])
        self.assertAlmostEqual(h["high"].iloc[0], d5["high"].iloc[:12].max())
        self.assertAlmostEqual(h["close"].iloc[0], d5["close"].iloc[11])
        self.assertAlmostEqual(h["open"].iloc[0], d5["open"].iloc[0])

    def test_15m_source(self):
        d15 = _bars(8, freq="15min", start="2024-01-01 00:00")
        h = nr.resample_bars(d15, 60, source_minutes=15)
        self.assertEqual(len(h), 2)
        self.assertAlmostEqual(h["low"].iloc[1], d15["low"].iloc[4:8].min())


class TestNatrAndRows(unittest.TestCase):
    def test_natr_constant_range(self):
        idx = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
        b = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx)
        n = nr.natr_pct(b, 60)
        self.assertTrue(n.iloc[:13].isna().all())
        self.assertAlmostEqual(n.iloc[14], 2.0, places=9)
        self.assertAlmostEqual(n.iloc[-1], 2.0, places=9)

    def test_rows_target_formula(self):
        idx = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
        b = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx)
        b.loc[idx[30], ["high", "low", "close"]] = [104.0, 98.0, 100.0]            # next bar of decision row 29
        r = nr.build_rows(b, 60)
        row = r.loc[idx[29]]
        self.assertAlmostEqual(row["y_range"], 6.0)                 # 100*(104-98)/C_29 with C_29 = 100
        self.assertAlmostEqual(row["y_up"], 4.0)
        self.assertAlmostEqual(row["y_down"], 2.0)
        self.assertEqual(row["tgt_open"], idx[30])

    def test_no_lookahead_in_rows(self):
        b = _bars(400, seed=1)
        r1 = nr.build_rows(b, 60)
        b2 = b.copy()
        b2.iloc[301:] = b2.iloc[301:] * 1.5                          # change everything after bar 300
        r2 = nr.build_rows(b2, 60)
        t = b.index[299]                                             # decision row 299 uses bars <= 300 only
        pd.testing.assert_series_equal(r1.loc[t].drop("tgt_open"), r2.loc[t].drop("tgt_open"))
        t = b.index[250]
        pd.testing.assert_series_equal(r1.loc[t].drop("tgt_open"), r2.loc[t].drop("tgt_open"))

    def test_baseline_is_trailing_90_known_at_t(self):
        b = _bars(400, seed=2)
        r = nr.build_rows(b, 60, window=90)
        s = 100 * (b["high"] - b["low"]) / b["close"].shift(1)       # target-series value of bar s
        i = 200
        expect = np.quantile(s.iloc[i - 89:i + 1].to_numpy(), 0.9)
        self.assertAlmostEqual(r.loc[b.index[i], "b_range_q90"], expect, places=10)
        self.assertTrue(np.isnan(r.loc[b.index[60], "b_range_q90"]) if b.index[60] in r.index else True)

    def test_gap_drops_rows(self):
        b = _bars(300, seed=3).drop(index=lambda ix: ix[[150]])
        r = nr.build_rows(b, 60)
        self.assertNotIn(b.index[0] + pd.Timedelta(hours=149), r.index)    # target bar 150 missing
        self.assertNotIn(b.index[0] + pd.Timedelta(hours=150), r.index)    # decision bar 150 missing


class TestQuantileFit(unittest.TestCase):
    def test_recovers_known_quantile_line(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 60000)
        ly = 0.3 + 0.8 * x + 0.2 * rng.normal(0, 1, 60000)
        from scipy.stats import norm
        for tau in (0.1, 0.5, 0.9):
            a, b = nr.fit_quantile(x, ly, tau)
            self.assertAlmostEqual(b, 0.8, delta=0.02)
            self.assertAlmostEqual(a, 0.3 + 0.2 * norm.ppf(tau), delta=0.01)

    def test_is_a_minimiser(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 5000)
        y = 0.5 * x + rng.standard_t(4, 5000)
        a, b = nr.fit_quantile(x, y, 0.9)
        base = nr.pinball(y, a + b * x, 0.9).mean()
        for da, db in ((0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05), (0.03, 0.03)):
            self.assertLessEqual(base, nr.pinball(y, a + da + (b + db) * x, 0.9).mean() + 1e-12)

    def test_pinball_values(self):
        self.assertAlmostEqual(float(nr.pinball(np.array([2.0]), np.array([1.0]), 0.9)[0]), 0.9)
        self.assertAlmostEqual(float(nr.pinball(np.array([0.0]), np.array([1.0]), 0.9)[0]), 0.1)

    def test_predict_sorted_and_crossings_counted(self):
        models = {("range", 0.1): (1.0, 0.0), ("range", 0.5): (0.5, 0.0), ("range", 0.9): (2.0, 0.0)}   # p10 > p50: crossed
        q, ncross = nr.predict_quantiles(models, "range", np.array([1.0, 2.0]))
        self.assertTrue((np.diff(q, axis=1) >= 0).all())
        self.assertEqual(ncross, 2)


class TestDecilesAndMetrics(unittest.TestCase):
    def test_edges_from_train_only(self):
        rng = np.random.default_rng(0)
        tr = rng.lognormal(0, 0.5, 5000)
        e = nr.decile_edges(tr)
        self.assertEqual(len(e), 9)
        d = nr.assign_decile(np.r_[tr, 1e9, 1e-9], e)
        self.assertEqual(d[-2], 9)
        self.assertEqual(d[-1], 0)
        np.testing.assert_array_equal(e, nr.decile_edges(tr))             # val never enters the edges
        counts = np.bincount(d[:-2], minlength=10)
        self.assertTrue((abs(counts - 500) <= 2).all())

    def test_coverage(self):
        y = np.array([1, 2, 3, 4.0])
        self.assertAlmostEqual(nr.coverage(y, np.full(4, 2.0), np.full(4, 3.0)), 0.5)

    def test_monthly_ic(self):
        idx = pd.date_range("2025-01-01", periods=90, freq="D", tz="UTC")
        x = np.arange(90.0)
        y = x.copy()
        y[60:] = -x[60:]                                                   # March reversed
        m = nr.monthly_ic(idx, x, y)
        self.assertEqual(list(m.index.strftime("%Y-%m")), ["2025-01", "2025-02", "2025-03"])
        self.assertAlmostEqual(m.iloc[0], 1.0)
        self.assertAlmostEqual(m.iloc[2], -1.0)


class TestAcceptRule(unittest.TestCase):
    def _ok(self):
        return dict(cov=0.80, cov_dec=[0.80] * 10, pin_model=[1.0] * 10, pin_base=[1.1] * 10, ic=[0.2] * 21)

    def test_pass(self):
        v = nr.evaluate_rule(**self._ok())
        self.assertTrue(v["accepted"])
        self.assertEqual(v["pinball_deciles_better"], 10)

    def test_overall_band_edges(self):
        a = self._ok()
        a["cov"] = 0.7701
        self.assertTrue(nr.evaluate_rule(**a)["overall_ok"])
        a["cov"] = 0.7699
        v = nr.evaluate_rule(**a)
        self.assertFalse(v["overall_ok"])
        self.assertFalse(v["accepted"])
        a["cov"] = 0.8301
        self.assertFalse(nr.evaluate_rule(**a)["accepted"])

    def test_one_decile_out(self):
        a = self._ok()
        a["cov_dec"] = [0.80] * 9 + [0.7399]
        v = nr.evaluate_rule(**a)
        self.assertFalse(v["deciles_ok"])
        self.assertFalse(v["accepted"])
        a["cov_dec"] = [0.74, 0.86] + [0.80] * 8
        self.assertTrue(nr.evaluate_rule(**a)["deciles_ok"])

    def test_pinball_needs_8_of_10(self):
        a = self._ok()
        a["pin_model"] = [1.0] * 8 + [1.2, 1.2]
        self.assertTrue(nr.evaluate_rule(**a)["pinball_ok"])
        a["pin_model"] = [1.0] * 7 + [1.2] * 3
        v = nr.evaluate_rule(**a)
        self.assertFalse(v["pinball_ok"])
        self.assertEqual(v["pinball_deciles_better"], 7)
        a["pin_model"] = [1.1] * 10                                          # equal is not better
        self.assertFalse(nr.evaluate_rule(**a)["pinball_ok"])

    def test_ic_needs_90_percent_of_months(self):
        a = self._ok()
        a["ic"] = [0.2] * 19 + [-0.1] * 2                                   # 19 of 21 = 90.5 %
        self.assertTrue(nr.evaluate_rule(**a)["ic_ok"])
        a["ic"] = [0.2] * 18 + [-0.1] * 3                                   # 18 of 21 = 85.7 %
        self.assertFalse(nr.evaluate_rule(**a)["ic_ok"])
        a["ic"] = [0.2] * 9 + [0.0]                                         # zero is not > 0 : 9 of 10 = 90 %
        self.assertTrue(nr.evaluate_rule(**a)["ic_ok"])
        a["ic"] = [0.2] * 8 + [0.0, 0.0]
        self.assertFalse(nr.evaluate_rule(**a)["ic_ok"])

    def test_missing_decile_cell_fails(self):
        a = self._ok()
        a["cov_dec"] = [0.8] * 9 + [float("nan")]
        self.assertFalse(nr.evaluate_rule(**a)["accepted"])


class TestEvaluateTimeframe(unittest.TestCase):
    @staticmethod
    def _rows(n_coin=4, n=3000, seed=0, sigma=0.35, shift=0.0):
        """Synthetic rows whose truth is exactly the model family: log y = 0.1 + 0.9 log natr + sigma*eps."""
        parts = []
        rng = np.random.default_rng(seed)
        for c in range(n_coin):
            idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
            ln = rng.normal(0.0, 0.5, n)
            lgy = 0.1 + 0.9 * ln + shift + sigma * rng.normal(0, 1, n)
            d = pd.DataFrame({"natr": np.exp(ln), "y_range": np.exp(lgy), "y_up": np.exp(lgy) / 2, "y_down": np.exp(lgy) / 2,
                              "tgt_open": idx + pd.Timedelta(hours=1), "coin": f"C{c}"}, index=idx)
            for t in nr.TAUS:
                for tg in nr.TARGETS:
                    d[f"b_{tg}_{nr.qname(t)}"] = np.quantile(d[f"y_{tg}"].iloc[:500], t)      # a crude constant baseline
            parts.append(d)
        return pd.concat(parts)

    def test_calibrated_model_passes_and_is_reported(self):
        rows = self._rows()
        tr = rows[rows["tgt_open"] < "2024-04-15"]
        va = rows[rows["tgt_open"] >= "2024-04-15"]
        res = nr.evaluate_timeframe(tr, va)
        self.assertTrue(res["verdict"]["overall_ok"])
        self.assertTrue(res["verdict"]["deciles_ok"], res["deciles"][["decile", "n", "coverage"]])
        self.assertTrue(res["verdict"]["pinball_ok"])
        self.assertEqual(len(res["deciles"]), 10)
        self.assertEqual(res["n_crossed"] >= 0, True)
        self.assertIn(("range", 0.9), res["models"])

    def test_distribution_shift_breaks_coverage(self):
        a = self._rows(seed=1)
        tr = a[a["tgt_open"] < "2024-04-15"]
        b = self._rows(seed=2, shift=0.8)
        va = b[b["tgt_open"] >= "2024-04-15"]
        res = nr.evaluate_timeframe(tr, va)
        self.assertFalse(res["verdict"]["overall_ok"])
        self.assertFalse(res["verdict"]["accepted"])


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_natr_range", os.path.join(ROOT, "research", "natr_range", "run_natr_range.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestRunner(unittest.TestCase):
    def test_unknown_config_key_raises(self):
        import tempfile
        m = _load_runner()
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "c.json")
            json.dump({"data_dir": td, "bogus": 1}, open(p, "w"))
            with self.assertRaises(KeyError):
                m.load_config(p)
            json.dump({"coins": ["BTCUSDT"]}, open(p, "w"))
            with self.assertRaises(KeyError):                               # data_dir is required
                m.load_config(p)

    def test_guard_refuses_time_seal_window(self):
        import tempfile
        import holdout_guard as hg
        m = _load_runner()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(hg.SealedAccessError):
                m.load_coin(td, "BTCUSDT", "ohlc_{symbol}_5m.parquet", "2024-01-01", "2026-10-02")

    def test_smoke_run_writes_manifest(self):
        import tempfile
        m = _load_runner()
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as out:
            for sym in ("BTCUSDT", "ETHUSDT"):
                idx = pd.date_range("2024-01-01", "2025-03-31 23:55", freq="5min", tz="UTC")
                c = 100 * np.exp(np.cumsum(rng.normal(0, 0.0015, len(idx))))
                o = np.r_[c[0], c[:-1]]
                df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995, "close": c, "volume": 1.0}, index=idx)
                df.to_parquet(os.path.join(td, f"ohlc_{sym}_5m.parquet"))
            cfg = {"data_dir": td, "coins": ["BTCUSDT", "ETHUSDT"], "train_start": "2024-01-01", "train_end": "2024-09-01", "val_end": "2025-04-01"}
            p = os.path.join(td, "cfg.json")
            json.dump(cfg, open(p, "w"))
            rc = m.main(["--config", p, "--out", out, "--max-minutes", "5"])
            self.assertEqual(rc, 0)
            man = json.load(open(os.path.join(out, "manifest.json")))
            self.assertEqual(man["n_configurations"], 18)
            self.assertFalse(man["truncated"])
            self.assertEqual(len(man["input_fingerprint"]), 64)
            for rel in ("summary.csv", "deciles.csv", "months.csv", "slices.csv", "results.json"):
                self.assertIn(rel, man["outputs"])
            res = json.load(open(os.path.join(out, "results.json")))
            self.assertEqual(sorted(res["timeframes"]), ["1d", "1h"])
            self.assertIn("accepted", res["timeframes"]["1h"]["verdict"])
            total = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
            self.assertLess(total, 300 * 1024)


if __name__ == "__main__":
    unittest.main()

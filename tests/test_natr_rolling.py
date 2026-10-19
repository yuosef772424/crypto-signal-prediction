"""E-natr-004 (rolling recalibration of the high-NATR band): no look-ahead, pooled fallback, synthetic drift where rolling restores
coverage and static does not, stability, runner on fake Drive-format data, card and job. No Drive, no sealed data.
    python -m pytest tests/test_natr_rolling.py -q
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import colab_runner as cr  # noqa: E402
import holdout_guard as hg  # noqa: E402
import natr_range as nr  # noqa: E402
from tests.test_holdout_guard import write_lock  # noqa: E402
from tests.test_natr_calibrate import _hourly_5  # noqa: E402

STUDY = os.path.join(ROOT, "research", "natr_range")
CARD = os.path.join(STUDY, "cards", "E-natr-004.md")
UTC = "UTC"


def _load_script():
    sys.path.insert(0, STUDY)
    spec = importlib.util.spec_from_file_location("run_natr_rolling", os.path.join(STUDY, "run_natr_rolling.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def drift_rows(n=300000, seed=3, rate=0.04):
    """Rows (natr, y_range, tgt_open) 2020-01..2026-09 whose top-decile log-noise sigma is 0.3 until 2024-01 and then grows by `rate` per year:
    a band calibrated on 2024 goes stale (a drift), a trailing-window band follows it."""
    rng = np.random.default_rng(seed)
    lx = rng.uniform(np.log(0.2), np.log(3.0), n)
    x90 = np.quantile(lx, 0.9)
    t0, t1 = pd.Timestamp("2020-01-01", tz=UTC), pd.Timestamp("2026-09-30", tz=UTC)
    tgt = t0 + pd.to_timedelta(rng.integers(0, int((t1 - t0).total_seconds() // 3600), n), unit="h")
    yrs = np.clip((tgt - pd.Timestamp("2024-01-01", tz=UTC)).total_seconds() / (365.25 * 86400), 0, None)
    sigma = np.where(lx > x90, 0.3 + rate * yrs, 0.3)
    y = np.exp(0.1 + 0.9 * lx + sigma * rng.standard_normal(n))
    return pd.DataFrame({"natr": np.exp(lx), "y_range": y, "tgt_open": tgt}).sort_values("tgt_open").reset_index(drop=True)


class TestRolling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r = drift_rows()
        cls.rows = r
        cls.base = r[r["tgt_open"] < "2024-01-01"]
        cls.cal = r[(r["tgt_open"] >= "2024-01-01") & (r["tgt_open"] < "2025-01-01")]
        cls.val = r[r["tgt_open"] >= "2025-01-01"]
        cls.models = nr.fit_models(cls.base.assign(y_up=cls.base["y_range"], y_down=cls.base["y_range"]), targets=("range",))
        cls.edges = nr.decile_edges(r[r["tgt_open"] < "2025-01-01"]["natr"])
        cls.static = nr.fit_decile_corrections(cls.models, cls.edges, cls.cal, fit_end="2025-01-01")
        cls.row_f, cls.table = nr.rolling_row_factors(cls.models, cls.edges, r, cls.val)

    # ---- no look-ahead
    def test_window_is_strictly_past_and_target_has_ended(self):
        m = pd.Timestamp("2025-06-01", tz=UTC)
        w = nr.rolling_window(self.rows, m)
        t = pd.DatetimeIndex(w["tgt_open"])
        self.assertLessEqual((t + pd.Timedelta(hours=1)).max(), m)
        self.assertGreaterEqual(t.min(), pd.Timestamp("2024-06-01", tz=UTC))
        edge = pd.DataFrame({"natr": [1.0, 1.0, 1.0], "y_range": [1.0] * 3,
                             "tgt_open": pd.to_datetime(["2025-05-31 22:00", "2025-05-31 23:00", "2025-06-01 00:00"], utc=True)})
        got = pd.DatetimeIndex(nr.rolling_window(edge, m)["tgt_open"])
        self.assertEqual(len(got), 2)                     # the 23:00 bar ends exactly at the month start (kept), the 00:00 bar is month m itself

    def test_fit_raises_on_any_row_ending_after_month_start(self):
        m = pd.Timestamp("2025-06-01", tz=UTC)
        w = nr.rolling_window(self.rows, m)
        leak = pd.concat([w, self.rows[self.rows["tgt_open"] == pd.Timestamp("2025-06-01 00:00", tz=UTC)].head(1)])
        with self.assertRaises(ValueError):
            nr.fit_month_factors(self.models, self.edges, leak, m)
        straddle = w.copy()
        straddle.iloc[0, straddle.columns.get_loc("tgt_open")] = pd.Timestamp("2025-05-31 23:30", tz=UTC)   # ends 00:30 after the start
        with self.assertRaises(ValueError):
            nr.fit_month_factors(self.models, self.edges, straddle, m)
        nr.fit_month_factors(self.models, self.edges, w, m)                                                  # the clean window works

    def test_factors_of_month_m_do_not_depend_on_rows_of_m_or_later(self):
        v = self.val[self.val["tgt_open"] < "2025-09-01"]
        f_full, _ = nr.rolling_row_factors(self.models, self.edges, self.rows, v)
        cut = self.rows[self.rows["tgt_open"] < "2025-07-01"]
        v6 = v[v["tgt_open"] < "2025-07-01"]
        f_cut, _ = nr.rolling_row_factors(self.models, self.edges, cut, v6)
        np.testing.assert_array_equal(f_full[: len(v6)], f_cut)     # deleting every row from July on leaves Jan..Jun factors identical

    def test_first_month_equals_static_calibration_slice(self):
        jan = self.table[self.table["month"] == "2025-01"].sort_values("decile")
        np.testing.assert_allclose(jan["c10"], self.static["range"]["factors"][:, 0], rtol=1e-12)
        np.testing.assert_allclose(jan["c90"], self.static["range"]["factors"][:, 2], rtol=1e-12)

    # ---- fallback rule
    def test_sparse_decile_uses_pooled_factor_and_is_flagged(self):
        m = pd.Timestamp("2025-01-01", tz=UTC)
        w = nr.rolling_window(self.rows, m)
        dec = nr.assign_decile(w["natr"], self.edges)
        few = w[(dec != 4) | (np.arange(len(w)) % 500 == 0)]
        f = nr.fit_month_factors(self.models, self.edges, few, m)
        self.assertLess(f["n"][4], 200)
        self.assertTrue(f["fallback"][4])
        self.assertEqual(int(f["fallback"].sum()), 1)
        q, _ = nr.predict_quantiles(self.models, "range", few["natr"])
        y = few["y_range"].to_numpy()
        pooled = (nr._tail_factors(y, q))
        self.assertEqual((f["factors"][4, 0], f["factors"][4, 2]), pooled)
        self.assertEqual(f["factors"][4, 1], 1.0)
        self.assertFalse(f["fallback"][9])

    def test_too_few_rows_in_total_raises(self):
        m = pd.Timestamp("2025-01-01", tz=UTC)
        with self.assertRaises(ValueError):
            nr.fit_month_factors(self.models, self.edges, nr.rolling_window(self.rows, m).head(150), m)

    # ---- drift: rolling restores coverage, static does not
    def test_drift_rolling_restores_top_decile_coverage_static_does_not(self):
        late = self.val["tgt_open"] >= "2025-10-01"
        v = self.val
        rolling = {"range": {"row_factors": self.row_f}}
        qs, qr = nr.quarter_top_coverage(self.models, self.edges, v, self.static), nr.quarter_top_coverage(self.models, self.edges, v, rolling)
        qc = nr.quarter_top_coverage(self.models, self.edges, v)
        self.assertEqual(len(qs), 7)
        s, r, c = qs["coverage"].iloc[-4:].mean(), qr["coverage"].iloc[-4:].mean(), qc["coverage"].iloc[-4:].mean()
        self.assertLess(s, 0.74)                          # the static band went stale
        self.assertGreater(r, 0.75)                       # the rolling band follows the drift
        self.assertGreater(r - s, 0.03)
        self.assertLess(c, s)                             # control is worse still
        self.assertTrue(late.any())

    def test_stability_is_small_for_slow_drift_and_flags_a_jump(self):
        st = nr.factor_stability(self.table)
        self.assertLessEqual(st["max_abs_ln"], nr.STAB_MAX_LN)
        self.assertLessEqual(st["mean_abs_ln"], nr.STAB_MEAN_LN)
        self.assertEqual(st["n_changes"], 2 * 10 * (self.table["month"].nunique() - 1))
        jump = self.table.copy()
        jump.loc[(jump["month"] == "2025-05") & (jump["decile"] == 9), "c90"] *= 1.2
        self.assertGreater(nr.factor_stability(jump)["max_abs_ln"], nr.STAB_MAX_LN)

    def test_row_factors_default_and_shape(self):
        n = self.val["natr"].to_numpy()
        old, _ = nr.predict_quantiles(self.models, "range", n)
        none, _ = nr.predict_quantiles(self.models, "range", n, None)
        ones, _ = nr.predict_quantiles(self.models, "range", n, {"range": {"row_factors": np.ones((len(n), 3))}})
        np.testing.assert_array_equal(old, none)
        np.testing.assert_array_equal(old, ones)
        with self.assertRaises(ValueError):
            nr.predict_quantiles(self.models, "range", n, {"range": {"row_factors": np.ones((5, 3))}})


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, "data")
        write_lock(os.path.join(self.tmp, "lock"), sealed={"SEALAUSDT": "H1"})
        self.lock = hg.load_lock(os.path.join(self.tmp, "lock"))
        for i, s in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
            _hourly_5(self.data, s, "2022-01-01", "2025-03-01", i)
        self.cfg = {"data_dir": self.data, "coins": ["AAAUSDT", "BBBUSDT", "CCCUSDT"], "load_start": "2021-12-01", "fit_start": "2022-01-01",
                    "base_end": "2023-07-01", "calib_end": "2024-07-01", "val_end": "2025-03-01", "min_coins": 2}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_writes_bundle_three_arms_no_fit_leak(self):
        m = _load_script()
        out = os.path.join(self.tmp, "out")
        manifest, res = m.run(self.cfg, out, 10, lock=self.lock)
        for f in ("manifest.json", "results.json", "summary.csv", "deciles.csv", "year_deciles.csv", "quarters.csv", "monthly_factors.csv", "months.csv", "frozen_stage_b.json"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        self.assertFalse(manifest["truncated"])
        d = pd.read_csv(os.path.join(out, "deciles.csv"))
        self.assertEqual(sorted(d["arm"].unique()), ["control", "rolling", "static"])
        self.assertIn("accepted", res["verdict"])
        self.assertLess(res["rolling_first_month_vs_static_max_abs_diff"], 1e-9)   # window of the first refit == the static calibration slice
        ft = pd.read_csv(os.path.join(out, "monthly_factors.csv"))
        self.assertEqual(ft["month"].min(), "2024-07")          # val starts at calib_end in this fake window
        self.assertEqual(res["hyper_parameters"]["window_months"], 12)
        fr = json.load(open(os.path.join(out, "frozen_stage_b.json")))
        self.assertEqual(len(fr["edges"]), 9)
        self.assertEqual(fr["hyper_parameters"]["min_rows_per_decile"], 200)

    def test_windows_unknown_keys_and_sealed_coin_refused(self):
        m = _load_script()
        for key, val in (("calib_end", "2025-03-01"), ("base_end", "2024-08-01")):
            with self.assertRaises(ValueError):
                m.run({**self.cfg, key: val}, os.path.join(self.tmp, "o"), 10, lock=self.lock)
        with self.assertRaises(KeyError):
            m.run({**self.cfg, "window_months": 6}, os.path.join(self.tmp, "o"), 10, lock=self.lock)     # hyper-parameters are not config keys
        _hourly_5(self.data, "SEALAUSDT", "2022-01-01", "2022-06-01", 9)
        with self.assertRaises(Exception) as cm:
            m.run({**self.cfg, "coins": ["SEALAUSDT"]}, os.path.join(self.tmp, "o"), 10, lock=self.lock)
        self.assertIn("seal", str(cm.exception).lower())

    def test_verdict_rules_8_and_9(self):
        m = _load_script()
        ok = {"accepted": True, "overall_ok": True, "ic_ok": True, "deciles_ok": True, "pinball_ok": True, "enough_deciles_ok": True}
        top = {2025: {"coverage": 0.80, "n": 500}, 2026: {"coverage": 0.80, "n": 500}}
        ctrl = {2025: 0.70, 2026: 0.70}
        qs = pd.DataFrame({"quarter": [f"q{i}" for i in range(7)], "n": [300] * 7, "coverage": [0.79] * 7})
        stab = {"n_changes": 100, "max_abs_ln": 0.03, "mean_abs_ln": 0.01}
        self.assertTrue(m.rolling_verdict(ok, top, ctrl, qs, stab, 10, 8)["accepted"])
        bad_q = qs.copy()
        bad_q.loc[3, "coverage"] = 0.73
        self.assertFalse(m.rolling_verdict(ok, top, ctrl, bad_q, stab, 10, 8)["accepted"])
        sparse_q = qs.copy()
        sparse_q.loc[3, ["n", "coverage"]] = [150, 0.60]                     # a sparse quarter is reported, not ruled
        self.assertTrue(m.rolling_verdict(ok, top, ctrl, sparse_q, stab, 10, 8)["accepted"])
        few = qs.copy()
        few["n"] = [300, 300, 300, 300, 100, 100, 100]                        # only 4 evaluable quarters: not evaluable
        self.assertFalse(m.rolling_verdict(ok, top, ctrl, few, stab, 10, 8)["accepted"])
        self.assertFalse(m.rolling_verdict(ok, top, ctrl, qs, {**stab, "max_abs_ln": 0.06}, 10, 8)["accepted"])
        self.assertFalse(m.rolling_verdict(ok, top, ctrl, qs, {**stab, "mean_abs_ln": 0.03}, 10, 8)["accepted"])
        self.assertFalse(m.rolling_verdict(ok, top, {2025: 0.80, 2026: 0.70}, qs, stab, 10, 8)["accepted"])   # not closer than control
        self.assertFalse(m.rolling_verdict(ok, top, ctrl, qs, stab, 7, 8)["accepted"])


class TestCardAndJob(unittest.TestCase):
    def test_card_cites_reopen_conditions_and_registry_check_passes(self):
        import experiment_registry as er
        meta = er.parse_card(CARD)
        self.assertIn("F-0084:R2", meta["reopens"])
        self.assertNotIn("confirmation", meta)
        self.assertFalse(er.check_card(CARD, er.load_registry()))

    def test_job_valid_not_queued(self):
        job = json.load(open(os.path.join(STUDY, "e_natr_004_job.json")))
        self.assertEqual(job["job_id"], "e-natr-004-explore")
        self.assertTrue(job["needs_drive"])
        self.assertEqual(job["entry"], "research/natr_range/run_natr_rolling.py")
        self.assertEqual(cr.validate_job(dict(job))["job_id"], "e-natr-004-explore")
        self.assertEqual(len(job["repo_ref"]), 40)


if __name__ == "__main__":
    unittest.main()

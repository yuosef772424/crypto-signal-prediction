"""E-natr-003 (high-NATR calibration of the 1h range forecast): per-decile train-only correction, train-only enforcement, stale-run mask,
localization rule, runner on fake data, card and job. No Drive, no sealed data.
    python -m pytest tests/test_natr_calibrate.py -q
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

STUDY = os.path.join(ROOT, "research", "natr_range")
CARD = os.path.join(STUDY, "cards", "E-natr-003.md")


def _load_script():
    sys.path.insert(0, STUDY)
    spec = importlib.util.spec_from_file_location("run_natr_calibrate", os.path.join(STUDY, "run_natr_calibrate.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def hetero_rows(n=120000, seed=3):
    """Rows (natr, y_range, tgt_open) whose log-noise doubles above the 90th NATR percentile: a log-log quantile line fitted on
    the pooled data under-covers the top decile (static miscalibration), by construction."""
    rng = np.random.default_rng(seed)
    lx = rng.uniform(np.log(0.2), np.log(3.0), n)
    x90 = np.quantile(lx, 0.9)
    sigma = np.where(lx > x90, 0.7, 0.3)
    y = np.exp(0.1 + 0.9 * lx + sigma * rng.standard_normal(n))
    t0, t1 = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
    tgt = t0 + pd.to_timedelta(rng.integers(0, int((t1 - t0).total_seconds() // 3600), n), unit="h")
    return pd.DataFrame({"natr": np.exp(lx), "y_range": y, "tgt_open": tgt})


def _split(rows):
    base = rows[rows["tgt_open"] < "2024-01-01"]
    cal = rows[(rows["tgt_open"] >= "2024-01-01") & (rows["tgt_open"] < "2025-01-01")]
    val = rows[rows["tgt_open"] >= "2025-01-01"]
    return base, cal, val


class TestCorrection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = hetero_rows()
        cls.base, cls.cal, cls.val = _split(rows)
        cls.models = nr.fit_models(cls.base.assign(y_up=cls.base["y_range"], y_down=cls.base["y_range"]))
        cls.edges = nr.decile_edges(pd.concat([cls.base, cls.cal])["natr"])
        cls.corr = nr.fit_decile_corrections(cls.models, cls.edges, cls.cal, fit_end="2025-01-01")

    def test_uncorrected_top_decile_undercovers_and_correction_restores_every_decile(self):
        ctrl = nr.decile_year_coverage(self.models, self.edges, self.val)
        fixed = nr.decile_year_coverage(self.models, self.edges, self.val, self.corr)
        top_c, top_f = ctrl[ctrl["decile"] == 9]["coverage"].iloc[0], fixed[fixed["decile"] == 9]["coverage"].iloc[0]
        self.assertLess(top_c, 0.76)                         # the constructed miscalibration is there
        self.assertLess(abs(top_f - 0.80), 0.02)             # and the correction removes it out of sample
        self.assertTrue((np.abs(fixed["coverage"] - 0.80) < 0.03).all(), fixed["coverage"].tolist())

    def test_factors_shape_and_median_untouched(self):
        f = self.corr["range"]["factors"]
        self.assertEqual(f.shape, (10, 3))
        self.assertTrue((f[:, 1] == 1.0).all())
        self.assertGreater(f[9, 2], f[0, 2])                 # wider upper band needed at the top than at the bottom
        self.assertEqual(int(self.corr["range"]["n"].sum()), len(self.cal))

    def test_defaults_reproduce_old_behaviour(self):
        n = self.val["natr"].to_numpy()
        old, c0 = nr.predict_quantiles(self.models, "range", n)
        ones = {"range": {"edges": self.edges, "factors": np.ones((10, 3))}}
        same, c1 = nr.predict_quantiles(self.models, "range", n, ones)
        none, c2 = nr.predict_quantiles(self.models, "range", n, None)
        np.testing.assert_array_equal(old, same)
        np.testing.assert_array_equal(old, none)
        self.assertEqual((c0, c1), (c0, c2))
        other, _ = nr.predict_quantiles(self.models, "up", n, ones)              # correction is keyed by target
        np.testing.assert_array_equal(other, nr.predict_quantiles(self.models, "up", n)[0])

    def test_sparse_decile_keeps_factor_one(self):
        few = self.cal.copy()
        few = few[(nr.assign_decile(few["natr"], self.edges) != 3) | (np.arange(len(few)) % 1000 == 0)]
        c = nr.fit_decile_corrections(self.models, self.edges, few, fit_end="2025-01-01")
        self.assertLess(c["range"]["n"][3], 200)
        np.testing.assert_array_equal(c["range"]["factors"][3], [1.0, 1.0, 1.0])

    def test_train_only_enforced(self):
        late = pd.concat([self.cal, self.val.head(5)])
        with self.assertRaises(ValueError):
            nr.fit_decile_corrections(self.models, self.edges, late, fit_end="2025-01-01")        # a 2025 row in the calibration slice
        with self.assertRaises(ValueError):
            nr.fit_decile_corrections(self.models, self.edges, self.cal, fit_end="2025-06-01")    # the window itself reaches into val
        with self.assertRaises(ValueError):
            nr.fit_models_until(self.val, "2025-01-01")
        with self.assertRaises(ValueError):
            nr.oof_decile_diagnostic(pd.concat([self.cal, self.val.head(5)]))
        self.assertEqual(len(nr.fit_models_until(self.base, "2024-01-01")), 9)


class TestStaleRuns(unittest.TestCase):
    def _bars(self, n=3000):
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(0)
        px = 100 + np.cumsum(rng.normal(0, 0.1, n))
        return pd.DataFrame({"open": px, "high": px + 0.1, "low": px - 0.1, "close": px, "volume": 1.0}, index=idx)

    @staticmethod
    def _flat(d, a, b):
        d.iloc[a:b, d.columns.get_loc("volume")] = 0.0
        for c in ("open", "high", "low", "close"):
            d.iloc[a:b, d.columns.get_loc(c)] = 50.0

    def test_long_flat_run_dropped_short_kept(self):
        d = self._bars()
        self._flat(d, 100, 100 + 700)         # 175 h >= 168 h
        self._flat(d, 1500, 1500 + 600)       # 150 h < 168 h
        out = nr.drop_stale_runs(d, 168, 15)
        self.assertEqual(len(out), len(d) - 700)
        self.assertNotIn(d.index[200], out.index)
        self.assertIn(d.index[1600], out.index)

    def test_zero_volume_but_moving_price_is_kept_and_gap_breaks_run(self):
        d = self._bars()
        d.iloc[100:900, d.columns.get_loc("volume")] = 0.0                     # zero volume but high != low: not flat
        self.assertEqual(len(nr.drop_stale_runs(d, 168, 15)), len(d))
        d = self._bars()
        self._flat(d, 100, 800)
        d = d.drop(d.index[450])                                              # a missing timestamp splits it into 350 + 349 bars
        self.assertEqual(len(nr.drop_stale_runs(d, 168, 15)), len(d))

    def test_input_unchanged_and_volume_required(self):
        d = self._bars()
        before = d.copy()
        nr.drop_stale_runs(d)
        pd.testing.assert_frame_equal(d, before)
        with self.assertRaises(KeyError):
            nr.drop_stale_runs(d.drop(columns="volume"))


def _oof(top):
    rows = []
    for fold, c in zip((2021, 2022, 2023, 2024), top):
        for dec in range(10):
            rows.append({"fold": fold, "decile": dec, "n": 1000, "coverage": c if dec == 9 else 0.80})
    return pd.DataFrame(rows)


class TestLocalization(unittest.TestCase):
    def test_static(self):
        r = nr.classify_localization(_oof([0.77, 0.78, 0.76, 0.79]), {2025: 0.77, 2026: 0.79})
        self.assertEqual(r["label"], "static")
        self.assertEqual(r["folds_at_or_below_79"], 4)

    def test_pooled_low_but_too_few_folds_is_mixed(self):
        r = nr.classify_localization(_oof([0.70, 0.80, 0.81, 0.78]), {2025: 0.77, 2026: 0.79})   # pooled 77.25 %, 2 folds <= 79 %
        self.assertEqual(r["label"], "mixed")

    def test_drift_and_not_reproduced(self):
        self.assertEqual(nr.classify_localization(_oof([0.80, 0.81, 0.79, 0.80]), {2025: 0.77, 2026: 0.79})["label"], "drift")
        self.assertEqual(nr.classify_localization(_oof([0.80, 0.81, 0.79, 0.80]), {2025: 0.79, 2026: 0.80})["label"], "not_reproduced")

    def test_between_thresholds_is_mixed(self):
        self.assertEqual(nr.classify_localization(_oof([0.785] * 4), {2025: 0.77, 2026: 0.79})["label"], "mixed")

    def test_oof_folds_use_past_years_only(self):
        rows = hetero_rows(60000)
        rows = rows[rows["tgt_open"] < "2025-01-01"]
        t = nr.oof_decile_diagnostic(rows)
        self.assertEqual(sorted(t["fold"].unique()), [2021, 2022, 2023, 2024])
        self.assertEqual(len(t), 40)


def _hourly_5(path_dir, symbol, start, end, seed):
    """Fake 15m history_15m file (csv.gz, ms timestamps)."""
    import gzip
    idx = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
    rng = np.random.default_rng(seed)
    vol = 0.002 * (1 + 2 * (rng.random(len(idx)) > 0.97))
    px = 100 * np.exp(np.cumsum(rng.normal(0, 1, len(idx)) * vol))
    hi = px * (1 + np.abs(rng.normal(0, vol / 2)))
    lo = px * (1 - np.abs(rng.normal(0, vol / 2)))
    df = pd.DataFrame({"timestamp": idx.as_unit("ms").asi8, "datetime_utc": idx.strftime("%Y-%m-%d %H:%M:%S"),
                       "open": px, "high": hi, "low": lo, "close": px, "volume": 1.0})
    os.makedirs(path_dir, exist_ok=True)
    with gzip.open(os.path.join(path_dir, f"{symbol}.csv.gz"), "wt") as f:
        df.to_csv(f, index=False)


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lock_dir = os.path.join(self.tmp, "lock")
        self.data = os.path.join(self.tmp, "data")
        write_lock(self.lock_dir, sealed={"SEALAUSDT": "H1"})
        self.lock = hg.load_lock(self.lock_dir)
        for i, s in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
            _hourly_5(self.data, s, "2022-01-01", "2025-03-01", i)
        self.cfg = {"data_dir": self.data, "coins": ["AAAUSDT", "BBBUSDT", "CCCUSDT"], "load_start": "2021-12-01", "fit_start": "2022-01-01",
                    "base_end": "2023-07-01", "calib_end": "2024-07-01", "val_end": "2025-03-01", "oof_first_year": 2023, "oof_last_year": 2024,
                    "oof_start_year": 2022, "min_coins": 2}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_writes_bundle_without_leaking_val_into_fit(self):
        m = _load_script()
        out = os.path.join(self.tmp, "out")
        manifest, res = m.run(self.cfg, out, 10, lock=self.lock)
        for f in ("manifest.json", "results.json", "summary.csv", "deciles.csv", "year_deciles.csv", "oof_diagnostic.csv", "corrections.csv", "frozen_stage_b.json"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        self.assertFalse(manifest["truncated"])
        self.assertIn(res["localization"]["label"], ("static", "drift", "not_reproduced", "mixed"))
        self.assertEqual(res["n_fit_rows_at_or_after_2025"], 0)
        self.assertIn("accepted", res["verdict"])
        fr = json.load(open(os.path.join(out, "frozen_stage_b.json")))
        self.assertEqual(len(fr["edges"]), 9)
        self.assertEqual(np.asarray(fr["correction_factors"]).shape, (10, 3))
        if res["localization"]["label"] != "static":
            self.assertFalse(res["verdict"]["accepted"])

    def test_windows_reaching_2025_for_fitting_refused_before_loading(self):
        m = _load_script()
        for key, val in (("calib_end", "2025-03-01"), ("base_end", "2024-08-01")):
            cfg = {**self.cfg, key: val}
            with self.assertRaises(ValueError):
                m.run(cfg, os.path.join(self.tmp, "o"), 10, lock=self.lock)

    def test_sealed_coin_and_unknown_keys_refused(self):
        m = _load_script()
        with self.assertRaises(KeyError):
            m.run({**self.cfg, "nope": 1}, os.path.join(self.tmp, "o"), 10, lock=self.lock)
        _hourly_5(self.data, "SEALAUSDT", "2022-01-01", "2022-06-01", 9)
        with self.assertRaises(Exception) as cm:
            m.run({**self.cfg, "coins": ["SEALAUSDT"]}, os.path.join(self.tmp, "o"), 10, lock=self.lock)
        self.assertIn("seal", str(cm.exception).lower())

    def test_verdict_requires_static_localization_and_year_rule(self):
        m = _load_script()
        ok = {"accepted": True, "overall_ok": True, "ic_ok": True, "deciles_ok": True, "pinball_ok": True, "enough_deciles_ok": True}
        top = {2025: {"coverage": 0.80, "n": 500}, 2026: {"coverage": 0.80, "n": 500}}
        ctrl = {2025: 0.70, 2026: 0.70}
        v = m.stage_a_verdict(ok, top, ctrl, {"label": "static"}, n_coins=10, min_coins=8)
        self.assertTrue(v["accepted"])
        self.assertFalse(m.stage_a_verdict(ok, top, ctrl, {"label": "drift"}, 10, 8)["accepted"])
        self.assertFalse(m.stage_a_verdict(ok, {**top, 2026: {"coverage": 0.73, "n": 500}}, ctrl, {"label": "static"}, 10, 8)["accepted"])
        self.assertFalse(m.stage_a_verdict(ok, {**top, 2026: {"coverage": 0.80, "n": 150}}, ctrl, {"label": "static"}, 10, 8)["accepted"])
        self.assertFalse(m.stage_a_verdict(ok, top, {2025: 0.80, 2026: 0.70}, {"label": "static"}, 10, 8)["accepted"])   # not closer than control
        self.assertFalse(m.stage_a_verdict(ok, top, ctrl, {"label": "static"}, 7, 8)["accepted"])


class TestCardAndJob(unittest.TestCase):
    def test_card_front_matter_cites_the_reopen_condition(self):
        import experiment_registry as er
        meta = er.parse_card(CARD)
        self.assertIn("F-0083:R1", meta["reopens"])
        self.assertNotIn("confirmation", meta)               # Stage A is exploration; Stage B gets its own confirmation card

    def test_job_valid_not_queued(self):
        job = json.load(open(os.path.join(STUDY, "e_natr_003_job.json")))
        self.assertEqual(job["job_id"], "e-natr-003-explore")
        self.assertTrue(job["needs_drive"])
        self.assertEqual(job["entry"], "research/natr_range/run_natr_calibrate.py")
        norm = cr.validate_job(dict(job))
        self.assertEqual(norm["job_id"], "e-natr-003-explore")


if __name__ == "__main__":
    unittest.main()

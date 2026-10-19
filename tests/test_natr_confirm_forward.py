"""E-natr-004-confirm: frozen files vs card, config guards, verdict rules (sufficiency, extension, rules 1-9), runner smoke on fake
forward data through load_forward, job spec. No Drive, no sealed data.
    python -m pytest tests/test_natr_confirm_forward.py -q
"""
import copy
import hashlib
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
CARD = os.path.join(STUDY, "cards", "E-natr-004-confirm.md")
FROZEN = os.path.join(STUDY, "frozen", "frozen_stage_b.json")
JOB = os.path.join(STUDY, "e_natr_004_confirm_job.json")


def _load():
    sys.path.insert(0, STUDY)
    spec = importlib.util.spec_from_file_location("run_natr_confirm_forward", os.path.join(STUDY, "run_natr_confirm_forward.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class TestFrozen(unittest.TestCase):
    def test_card_hashes_match_files(self):
        meta, _ = hg._parse_card(CARD)
        self.assertEqual(meta["frozen_config_sha256"], _sha(FROZEN))
        self.assertEqual(meta["frozen_code_sha256"], _sha(os.path.join(ROOT, "tools", "natr_range.py")))
        self.assertEqual(meta["confirmation"], "true")
        self.assertEqual(meta["forward_start"], "2026-10-01")

    def test_frozen_is_the_stage_a_output(self):
        a = os.path.join(STUDY, "results", "e_natr_004_explore", "frozen_stage_b.json")
        self.assertEqual(_sha(FROZEN), _sha(a))
        self.assertEqual(json.load(open(FROZEN))["hyper_parameters"]["window_months"], nr.ROLL_WINDOW_MONTHS)

    def test_card_passes_registry_check(self):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "experiment_registry.py"), "check", CARD], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_job_valid_not_queued_and_pinned(self):
        job = json.load(open(JOB))
        self.assertEqual(job["job_id"], "e-natr-004-confirm")
        self.assertIn("not_before: 2027-01-01", job["note"])
        self.assertEqual(job["entry"], "research/natr_range/run_natr_confirm_forward.py")
        self.assertTrue(os.path.exists(os.path.join(ROOT, job["entry"])))
        self.assertNotIn("queue", job.get("state", ""))


class TestConfig(unittest.TestCase):
    def test_unknown_key_missing_data_dir_and_bad_end(self):
        m = _load()
        with self.assertRaises(KeyError):
            m.complete_config({"data_dir": "x", "bogus": 1})
        with self.assertRaises(KeyError):
            m.complete_config({})
        with self.assertRaises(ValueError):
            m.complete_config({"data_dir": "x", "forward_end": "2027-02-01"})
        self.assertEqual(m.complete_config({"data_dir": "x", "forward_end": "2027-04-01"})["forward_end"], "2027-04-01")


def _sv(**kw):
    d = {"coverage": 0.80, "overall_ok": True, "deciles_ok": True, "pinball_ok": True, "ic_ok": True, "enough_deciles_ok": True,
         "n_evaluable_deciles": 10, "accepted": True}
    d.update(kw)
    return d


def _q(n=300, cov=0.8):
    return pd.DataFrame({"quarter": ["2026Q4"], "n": [n], "coverage": [cov]})


STAB = {"n_changes": 40, "max_abs_ln": 0.01, "mean_abs_ln": 0.004}


class TestVerdict(unittest.TestCase):
    def call(self, sv=None, top=None, ctrl=0.76, q=None, stab=STAB, coins=10, end="2027-01-01"):
        m = _load()
        return m.confirm_verdict(sv or _sv(), top or {"n": 500, "coverage": 0.80}, ctrl, q if q is not None else _q(), stab, coins, 8, end)

    def test_all_pass_is_confirmed(self):
        v = self.call()
        self.assertEqual(v["status"], "confirmed")
        self.assertTrue(v["accepted"])

    def test_top_decile_below_200_is_insufficient_then_final_after_extension(self):
        v = self.call(top={"n": 150, "coverage": 0.8})
        self.assertEqual(v["status"], "insufficient_extend_3_months")
        self.assertFalse(v["accepted"])
        v2 = self.call(top={"n": 150, "coverage": 0.8}, end="2027-04-01")
        self.assertEqual(v2["status"], "insufficient_final")

    def test_fewer_than_six_evaluable_deciles_is_insufficient(self):
        self.assertEqual(self.call(sv=_sv(enough_deciles_ok=False, n_evaluable_deciles=5, accepted=False))["status"], "insufficient_extend_3_months")

    def test_sufficient_window_that_fails_a_rule_is_rejected_not_extended(self):
        self.assertEqual(self.call(sv=_sv(overall_ok=False, accepted=False))["status"], "rejected")
        self.assertEqual(self.call(top={"n": 500, "coverage": 0.70})["status"], "rejected")
        self.assertEqual(self.call(ctrl=0.80)["status"], "rejected")              # not closer than control
        self.assertEqual(self.call(coins=7)["status"], "rejected")
        self.assertEqual(self.call(q=_q(300, 0.90))["status"], "rejected")
        self.assertEqual(self.call(stab={"n_changes": 4, "max_abs_ln": 0.06, "mean_abs_ln": 0.01})["status"], "rejected")
        self.assertEqual(self.call(stab={"n_changes": 4, "max_abs_ln": 0.04, "mean_abs_ln": 0.03})["status"], "rejected")

    def test_no_evaluable_quarter_is_rejected_by_rule_8(self):
        self.assertEqual(self.call(q=_q(100, 0.8))["status"], "rejected")


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lock_dir = os.path.join(self.tmp, "lock")
        self.data = os.path.join(self.tmp, "data")
        write_lock(self.lock_dir, sealed={"SEALAUSDT": "H1"})
        self.lock = hg.load_lock(self.lock_dir)
        for i, s in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
            _hourly_5(self.data, s, "2025-06-01", "2027-01-01", i)
        self.cfg = {"data_dir": self.data, "coins": ["AAAUSDT", "BBBUSDT", "CCCUSDT"], "min_coins": 2}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_refuses_before_forward_end_and_sealed_coin_and_hash_drift(self):
        m = _load()
        with self.assertRaises(ValueError):
            m.run(self.cfg, os.path.join(self.tmp, "o"), 5, lock=self.lock, now=pd.Timestamp("2026-12-31", tz="UTC"))
        with self.assertRaises(hg.SealedAccessError):
            m.run({**self.cfg, "coins": ["SEALAUSDT"]}, os.path.join(self.tmp, "o"), 5, lock=self.lock, now=pd.Timestamp("2027-01-02", tz="UTC"))
        self.assertEqual(len(open(os.path.join(self.lock_dir, "holdout_access_log.csv")).read().splitlines()), 1)     # header only

    def test_smoke_writes_bundle_logs_one_row_and_never_evaluates_pre_seal(self):
        m = _load()
        out = os.path.join(self.tmp, "out")
        manifest, res = m.run(self.cfg, out, 10, lock=self.lock, now=pd.Timestamp("2027-01-02", tz="UTC"))
        for f in ("manifest.json", "results.json", "summary.csv", "deciles.csv", "quarters.csv", "monthly_factors.csv", "months.csv", "access_log_row.csv"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        lines = open(os.path.join(self.lock_dir, "holdout_access_log.csv")).read().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn(",FWD,", lines[1])
        self.assertIn(res["verdict"]["status"], ("confirmed", "rejected", "insufficient_extend_3_months"))
        self.assertGreaterEqual(pd.Timestamp(res["first_eval_open"]), hg.FORWARD_START)
        ft = pd.read_csv(os.path.join(out, "monthly_factors.csv"))
        self.assertEqual(ft["month"].min(), "2026-09")                      # stability anchor, factors only
        self.assertEqual(sorted(ft["month"].unique()), ["2026-09", "2026-10", "2026-11", "2026-12"])


if __name__ == "__main__":
    unittest.main()

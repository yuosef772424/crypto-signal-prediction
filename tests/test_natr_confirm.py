"""R3 H2 confirmation preparation (NOT run, NOT queued): nr.evaluate_frozen, the frozen rules file, the confirmation card and
research/natr_range/run_natr_confirm.py on a fake lock and fake 15m data (no Drive).
    python -m pytest tests/test_natr_confirm.py -q
"""
import copy
import csv
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
from tests import test_natr_range as tnr  # noqa: E402
from tests.test_holdout_guard import write_bars, write_card, write_lock  # noqa: E402

STUDY = os.path.join(ROOT, "research", "natr_range")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(STUDY, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestEvaluateFrozen(unittest.TestCase):
    def test_same_numbers_as_evaluate_timeframe(self):
        rows = tnr.TestEvaluateTimeframe._rows()
        tr, va = rows[rows["tgt_open"] < "2024-04-15"], rows[rows["tgt_open"] >= "2024-04-15"]
        full = nr.evaluate_timeframe(tr, va)
        fz = nr.evaluate_frozen(full["models"], full["edges"], va)
        self.assertEqual(full["verdict"], fz["verdict"])
        pd.testing.assert_frame_equal(full["deciles"], fz["deciles"])
        self.assertNotIn("train_coverage", fz["extras"])


class TestFrozenFilesInRepo(unittest.TestCase):
    def test_frozen_rules_match_run1_and_card(self):
        fr_path = os.path.join(STUDY, "frozen_rules.json")
        raw = open(fr_path, "rb").read()
        sha = __import__("hashlib").sha256(raw).hexdigest()
        meta, _ = hg._parse_card(os.path.join(STUDY, "cards", "E-natr-002.md"))
        self.assertEqual(meta["confirmation"], "true")
        self.assertEqual(meta["holdout_bucket"], "H2")
        self.assertEqual(meta["frozen_rules_sha256"], sha)
        self.assertEqual(meta["frozen_rules_file"], "research/natr_range/frozen_rules.json")
        fr = json.loads(raw)
        run1 = json.load(open(os.path.join(STUDY, "results", "e_natr_001_run1", "results.json")))["timeframes"]["1h"]
        self.assertEqual(fr["models"], run1["models"])
        self.assertEqual(fr["edges"], run1["edges"])
        self.assertEqual(fr["timeframe"], "1h")
        self.assertEqual(fr["rule"], {"cover_target": nr.COVER_TARGET, "overall_tol": nr.OVERALL_TOL, "decile_tol": nr.DECILE_TOL,
                                      "min_pinball_deciles": nr.MIN_PINBALL_DECILES, "ic_month_share_pct": nr.MIN_IC_MONTH_SHARE_PCT})

    def test_job_is_valid_and_not_queued(self):
        job = json.load(open(os.path.join(STUDY, "h2_job.json")))
        self.assertIn("DO-NOT-QUEUE", job["job_id"])
        job["repo_ref"] = "a" * 40
        job["job_id"] = "natr001-h2-confirm"
        norm = cr.validate_job(job)
        self.assertEqual(norm["entry"], "research/natr_range/run_natr_confirm.py")
        self.assertTrue(norm["needs_drive"])
        self.assertLessEqual(norm["max_minutes"], 20)

    def test_card_passes_registry_check(self):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "experiment_registry.py"), "check",
                            os.path.join(STUDY, "cards", "E-natr-002.md")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


def _frozen(path, **over):
    base = json.load(open(os.path.join(STUDY, "frozen_rules.json")))
    base.update({"eval_start": "2025-01-01", "eval_end": "2025-03-01", "min_coins": 1})
    base.update(over)
    with open(path, "w") as f:
        json.dump(base, f, indent=1, sort_keys=True)
        f.write("\n")
    return __import__("hashlib").sha256(open(path, "rb").read()).hexdigest()


class TestConfirmRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lock_dir = os.path.join(self.tmp, "lock")
        self.data_dir = os.path.join(self.tmp, "data")
        write_lock(self.lock_dir, sealed={"SEALAUSDT": "H1", "SEALBUSDT": "H2", "SEALCUSDT": "H2"})
        write_bars(self.data_dir, "SEALBUSDT", start="2024-10-01", end="2025-03-01")
        self.frozen = os.path.join(self.tmp, "frozen.json")
        self.sha = _frozen(self.frozen)
        self.card = write_card(os.path.join(self.tmp, "card.md"), confirmation="true", holdout_bucket="H2",
                               frozen_rules_sha256=self.sha, frozen_rules_file=self.frozen)
        self.m = _load("run_natr_confirm")
        self.cfg = {"data_dir": self.data_dir, "card": self.card, "frozen_rules": self.frozen}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _log_rows(self):
        return list(csv.DictReader(open(os.path.join(self.lock_dir, "holdout_access_log.csv"))))

    def test_runs_once_writes_bundle_and_logs_open(self):
        out = os.path.join(self.tmp, "out")
        manifest, res = self.m.run(self.cfg, out, 5, lock=hg.load_lock(self.lock_dir))
        rows = self._log_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bucket"], "H2")
        self.assertEqual(rows[0]["frozen_rules_sha256"], self.sha)
        for rel in ("summary.csv", "deciles.csv", "months.csv", "slices.csv", "results.json", "access_log_row.csv"):
            self.assertIn(rel, manifest["outputs"])
        self.assertEqual(res["skipped_coins"], {"SEALCUSDT": "file not found"})
        self.assertEqual(res["coins_used"], ["SEALBUSDT"])
        self.assertIn("accepted", res["verdict"])
        self.assertIn("coins_ok", res["verdict"])
        with self.assertRaises(hg.HoldoutOpenError):                       # a second open of the bucket is refused
            self.m.run(self.cfg, os.path.join(self.tmp, "out2"), 5, lock=hg.load_lock(self.lock_dir))

    def test_wrong_frozen_hash_fails_before_opening(self):
        bad = write_card(os.path.join(self.tmp, "bad.md"), confirmation="true", holdout_bucket="H2",
                         frozen_rules_sha256="c" * 64, frozen_rules_file=self.frozen)
        with self.assertRaises(hg.HoldoutOpenError):
            self.m.run({**self.cfg, "card": bad}, os.path.join(self.tmp, "o"), 5, lock=hg.load_lock(self.lock_dir))
        self.assertEqual(self._log_rows(), [])

    def test_frozen_rule_constants_must_match_code(self):
        _frozen(self.frozen, rule={"cover_target": 0.8, "overall_tol": 0.10, "decile_tol": 0.06, "min_pinball_deciles": 8, "ic_month_share_pct": 90})
        sha = __import__("hashlib").sha256(open(self.frozen, "rb").read()).hexdigest()
        card = write_card(os.path.join(self.tmp, "c2.md"), confirmation="true", holdout_bucket="H2", frozen_rules_sha256=sha, frozen_rules_file=self.frozen)
        with self.assertRaises(ValueError):
            self.m.run({**self.cfg, "card": card}, os.path.join(self.tmp, "o"), 5, lock=hg.load_lock(self.lock_dir))
        self.assertEqual(self._log_rows(), [])

    def test_unknown_config_key_raises(self):
        p = os.path.join(self.tmp, "c.json")
        json.dump({**self.cfg, "bogus": 1}, open(p, "w"))
        with self.assertRaises(KeyError):
            self.m.load_config(p)


if __name__ == "__main__":
    unittest.main()

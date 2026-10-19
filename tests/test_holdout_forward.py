"""tools/holdout_guard.load_forward: the reviewed forward-read path (bars with open time >= 2026-10-01, unsealed coins only, a
confirmation card whose frozen-config sha256 matches, one access-log row per call). Existing functions keep refusing the seal.
    python -m pytest tests/test_holdout_forward.py -q
"""
import hashlib
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import holdout_guard as hg  # noqa: E402
from tests.test_holdout_guard import write_bars, write_card, write_lock  # noqa: E402


class Fwd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lock_dir = os.path.join(self.tmp, "lock")
        self.data = os.path.join(self.tmp, "data")
        write_lock(self.lock_dir)
        for s in ("BTCUSDT", "ETHUSDT", "SEALAUSDT"):
            write_bars(self.data, s, "2026-09-29", "2026-10-03")
        self.frozen = os.path.join(self.tmp, "frozen.json")
        open(self.frozen, "w").write('{"x": 1}\n')
        self.sha = hashlib.sha256(open(self.frozen, "rb").read()).hexdigest()
        self.card = self.make_card()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_card(self, name="c.md", **over):
        f = {"confirmation": "true", "forward_confirmation": "true", "forward_start": "2026-10-01",
             "frozen_config_file": self.frozen, "frozen_config_sha256": self.sha}
        f.update(over)
        return write_card(os.path.join(self.tmp, name), **{k: v for k, v in f.items() if v is not None})

    def rows(self):
        import csv
        return list(csv.DictReader(open(os.path.join(self.lock_dir, "holdout_access_log.csv"), encoding="utf-8", newline="")))

    def load(self, coins=("BTCUSDT", "ETHUSDT"), start="2026-10-01", end="2026-10-03", card=None):
        return hg.load_forward(self.data, list(coins), start, end, card or self.card, lock=self.lock_dir)


class ForwardTests(Fwd):
    def test_reads_only_forward_bars_and_logs_exactly_one_row(self):
        out = self.load()
        self.assertEqual(set(out), {"BTCUSDT", "ETHUSDT"})
        for df in out.values():
            self.assertGreater(len(df), 0)
            self.assertGreaterEqual(df.index.min(), hg.FORWARD_START)
            self.assertLess(df.index.max(), __import__("pandas").Timestamp("2026-10-03", tz="UTC"))
        r = self.rows()
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["bucket"], "FWD")
        self.assertEqual(r[0]["frozen_rules_sha256"], self.sha)
        self.assertEqual(r[0]["card_sha256"], hashlib.sha256(open(self.card, "rb").read()).hexdigest())

    def test_refuses_sealed_coin_and_logs_nothing(self):
        with self.assertRaises(hg.SealedAccessError):
            self.load(coins=("BTCUSDT", "SEALAUSDT"))
        with self.assertRaises(hg.SealedAccessError):
            self.load(coins=("sealb",))
        self.assertEqual(self.rows(), [])

    def test_refuses_pre_start_bar(self):
        for start in ("2026-09-30", "2026-09-30 23:45", "2020-01-01"):
            with self.assertRaises(hg.SealedAccessError, msg=start):
                self.load(start=start)
        self.assertEqual(self.rows(), [])

    def test_refuses_card_hash_mismatch(self):
        open(self.frozen, "w").write('{"x": 2}\n')
        with self.assertRaises(hg.HoldoutOpenError):
            self.load()
        self.assertEqual(self.rows(), [])

    def test_refuses_bad_cards(self):
        for over in ({"confirmation": "false"}, {"forward_confirmation": None}, {"frozen_config_sha256": "zz"},
                     {"frozen_config_file": None}, {"forward_start": "2026-11-01"}):
            with self.assertRaises(hg.HoldoutOpenError, msg=str(over)):
                self.load(card=self.make_card(**over))
        with self.assertRaises(hg.HoldoutOpenError):
            self.load(card=os.path.join(self.tmp, "missing.md"))
        self.assertEqual(self.rows(), [])

    def test_empty_inputs_and_empty_window_raise(self):
        with self.assertRaises(ValueError):
            self.load(coins=())
        with self.assertRaises(ValueError):
            self.load(start="2026-10-02", end="2026-10-02")
        with self.assertRaises(TypeError):
            hg.load_forward(self.data, ["BTCUSDT"], None, "2026-10-02", self.card, lock=self.lock_dir)

    def test_each_call_appends_one_row_and_keeps_header(self):
        self.load()
        self.load(coins=("BTCUSDT",))
        self.assertEqual(len(self.rows()), 2)
        first = open(os.path.join(self.lock_dir, "holdout_access_log.csv"), encoding="utf-8").readline().rstrip("\n")
        self.assertEqual(first, ",".join(hg.LOG_COLUMNS))

    def test_existing_functions_still_refuse_the_seal(self):
        with self.assertRaises(hg.SealedAccessError):
            hg.load_unsealed(self.data, "BTCUSDT", "2026-09-30", "2026-10-02", lock=self.lock_dir)
        with self.assertRaises(hg.SealedAccessError):
            hg.assert_unsealed(["BTCUSDT"], "2026-10-01", "2026-10-02", lock=self.lock_dir)
        self.assertEqual(self.rows(), [])


if __name__ == "__main__":
    unittest.main()

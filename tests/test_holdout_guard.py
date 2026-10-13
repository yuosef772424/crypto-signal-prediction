"""tools/holdout_guard.py: a sealed symbol or a time-sealed bar raises through every public loader, and a bucket opens once.
Everything runs on small fixtures in a temporary lock folder (no Drive, no network).
    python -m unittest tests.test_holdout_guard -v
"""
import csv
import gzip
import hashlib
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
import holdout_guard as hg  # noqa: E402

SALT = "a" * 40
SEALED = {"SEALAUSDT": "H1", "SEALBUSDT": "H2", "SEALCUSDT": "H1"}
SEAL = pd.Timestamp("2026-10-01", tz="UTC")


def write_lock(lock_dir, sealed=None, salt=SALT, tamper=False, seal="2026-10-01T00:00:00Z"):
    sealed = SEALED if sealed is None else sealed
    os.makedirs(lock_dir, exist_ok=True)
    buckets = {f"H{i}": sorted(s for s, b in sealed.items() if b == f"H{i}") for i in range(1, 7)}
    body = json.dumps({"version": 1, "salt": salt, "time_seal": seal, "sealed": sealed, "buckets": buckets},
                      indent=1, sort_keys=True) + "\n"
    with open(os.path.join(lock_dir, "holdout.json"), "w", encoding="utf-8") as f:
        f.write(body)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    with open(os.path.join(lock_dir, "holdout.sha256"), "w", encoding="utf-8") as f:
        f.write(("0" * 64 if tamper else digest) + "  holdout.json\n")
    with open(os.path.join(lock_dir, "holdout_access_log.csv"), "w", encoding="utf-8", newline="") as f:
        f.write(",".join(hg.LOG_COLUMNS) + "\n")


def write_bars(data_dir, symbol, start="2026-09-29", end="2026-10-02", gz=True):
    """15m bars, `timestamp` in ms (bar open time) like history_15m."""
    idx = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
    rng = np.random.default_rng(1)
    px = 100 + np.cumsum(rng.normal(0, 0.1, len(idx)))
    df = pd.DataFrame({"timestamp": idx.as_unit("ms").asi8, "datetime_utc": idx.strftime("%Y-%m-%d %H:%M:%S"),
                       "open": px, "high": px + 0.1, "low": px - 0.1, "close": px, "volume": 1.0})
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{symbol}.csv.gz")
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)
    return path


def write_card(path, **fields):
    lines = ["---"] + [f"{k}: {v}" for k, v in fields.items()] + ["---", "", "# card", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


GOOD = {"confirmation": "true", "frozen_rules_sha256": "b" * 64}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lock_dir = os.path.join(self.tmp, "lock")
        self.data_dir = os.path.join(self.tmp, "data")
        write_lock(self.lock_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def card(self, name="card.md", **overrides):
        fields = dict(GOOD)
        fields.update(overrides)
        fields = {k: v for k, v in fields.items() if v is not None}
        return write_card(os.path.join(self.tmp, name), **fields)

    def log_rows(self):
        with open(os.path.join(self.lock_dir, "holdout_access_log.csv"), encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))


class LockTests(Base):
    def test_missing_lock_fails_closed(self):
        with self.assertRaises(hg.LockError):
            hg.assert_unsealed(["BTCUSDT"], "2026-01-01", "2026-02-01", lock=os.path.join(self.tmp, "nowhere"))
        with self.assertRaises(hg.LockError):
            hg.load_unsealed(self.data_dir, "BTCUSDT", "2026-01-01", "2026-02-01", lock=os.path.join(self.tmp, "nowhere"))

    def test_sha256_mismatch_is_refused(self):
        write_lock(self.lock_dir, tamper=True)
        with self.assertRaises(hg.LockError):
            hg.load_lock(self.lock_dir)

    def test_edited_json_is_refused(self):
        p = os.path.join(self.lock_dir, "holdout.json")
        with open(p, "a", encoding="utf-8") as f:
            f.write(" ")
        with self.assertRaises(hg.LockError):
            hg.load_lock(self.lock_dir)

    def test_wrong_time_seal_is_refused(self):
        write_lock(self.lock_dir, seal="2026-11-01T00:00:00Z")
        with self.assertRaises(hg.LockError):
            hg.load_lock(self.lock_dir)

    def test_bucket_lists_must_match_sealed_map(self):
        p = os.path.join(self.lock_dir, "holdout.json")
        d = json.load(open(p, encoding="utf-8"))
        d["buckets"]["H3"] = ["SEALAUSDT"]                                  # listed twice
        body = json.dumps(d, indent=1, sort_keys=True) + "\n"
        open(p, "w", encoding="utf-8").write(body)
        open(os.path.join(self.lock_dir, "holdout.sha256"), "w").write(hashlib.sha256(body.encode()).hexdigest() + "  holdout.json\n")
        with self.assertRaises(hg.LockError):
            hg.load_lock(self.lock_dir)

    def test_lock_exposes_sealed_set(self):
        lock = hg.load_lock(self.lock_dir)
        self.assertEqual(set(lock.sealed), set(SEALED))
        self.assertEqual(lock.buckets["H1"], ("SEALAUSDT", "SEALCUSDT"))
        self.assertEqual(lock.salt, SALT)


class AssertUnsealedTests(Base):
    def test_sealed_symbol_raises_any_spelling(self):
        for s in ("SEALAUSDT", "sealausdt", " SEALAUSDT ", "SEALA"):
            with self.assertRaises(hg.SealedAccessError, msg=s):
                hg.assert_unsealed([s], "2026-01-01", "2026-02-01", lock=self.lock_dir)

    def test_one_sealed_symbol_among_many_raises_and_is_named(self):
        with self.assertRaises(hg.SealedAccessError) as cm:
            hg.assert_unsealed(["BTCUSDT", "SEALBUSDT", "ETHUSDT"], "2026-01-01", "2026-02-01", lock=self.lock_dir)
        self.assertIn("SEALBUSDT", str(cm.exception))

    def test_unsealed_symbols_pass(self):
        hg.assert_unsealed(["BTCUSDT", "ETHUSDT"], "2026-01-01", "2026-02-01", lock=self.lock_dir)
        hg.assert_unsealed("BTCUSDT", "2026-01-01", "2026-02-01", lock=self.lock_dir)         # a single string

    def test_time_seal_applies_to_every_coin(self):
        with self.assertRaises(hg.SealedAccessError):
            hg.assert_unsealed(["BTCUSDT"], "2026-09-01", "2026-10-01 00:15", lock=self.lock_dir)
        with self.assertRaises(hg.SealedAccessError):
            hg.assert_unsealed(["BTCUSDT"], "2026-10-02", "2026-10-03", lock=self.lock_dir)

    def test_end_at_the_seal_is_allowed_half_open(self):
        hg.assert_unsealed(["BTCUSDT"], "2026-09-01", "2026-10-01", lock=self.lock_dir)
        hg.assert_unsealed(["BTCUSDT"], pd.Timestamp("2026-09-01", tz="UTC"), SEAL, lock=self.lock_dir)

    def test_naive_timestamps_are_utc_and_offsets_are_converted(self):
        hg.assert_unsealed(["BTCUSDT"], "2026-09-01", "2026-10-01T01:00:00+02:00", lock=self.lock_dir)     # = 2026-09-30 23:00Z
        with self.assertRaises(hg.SealedAccessError):
            hg.assert_unsealed(["BTCUSDT"], "2026-09-01", "2026-10-01T00:00:00-02:00", lock=self.lock_dir)   # = 2026-10-01 02:00Z

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            hg.assert_unsealed(["BTCUSDT"], "2026-02-01", "2026-01-01", lock=self.lock_dir)        # start >= end
        with self.assertRaises(TypeError):
            hg.assert_unsealed(["BTCUSDT"], "2026-01-01", None, lock=self.lock_dir)                # no silent default
        with self.assertRaises(ValueError):
            hg.assert_unsealed([], "2026-01-01", "2026-02-01", lock=self.lock_dir)


class LoadUnsealedTests(Base):
    def test_sealed_symbol_raises_before_any_file_is_opened(self):
        # the file does not even exist: the refusal must be the seal, not FileNotFoundError
        with self.assertRaises(hg.SealedAccessError):
            hg.load_unsealed(self.data_dir, "SEALAUSDT", "2026-01-01", "2026-02-01", lock=self.lock_dir)

    def test_request_beyond_seal_raises_for_unsealed_coin(self):
        write_bars(self.data_dir, "BTCUSDT")
        with self.assertRaises(hg.SealedAccessError):
            hg.load_unsealed(self.data_dir, "BTCUSDT", "2026-09-29", "2026-10-02", lock=self.lock_dir)

    def test_returns_only_the_requested_half_open_window(self):
        write_bars(self.data_dir, "BTCUSDT")
        df = hg.load_unsealed(self.data_dir, "BTCUSDT", "2026-09-30", "2026-10-01", lock=self.lock_dir)
        self.assertEqual(str(df.index.tz), "UTC")
        self.assertEqual(df.index[0], pd.Timestamp("2026-09-30 00:00", tz="UTC"))
        self.assertEqual(df.index[-1], pd.Timestamp("2026-09-30 23:45", tz="UTC"))
        self.assertEqual(len(df), 96)
        self.assertTrue((df.index < SEAL).all())
        for c in ("open", "high", "low", "close", "volume"):
            self.assertIn(c, df.columns)

    def test_file_with_post_seal_bars_never_leaks_them(self):
        write_bars(self.data_dir, "BTCUSDT", end="2026-10-05")
        df = hg.load_unsealed(self.data_dir, "BTCUSDT", "2026-09-01", SEAL, lock=self.lock_dir)
        self.assertTrue((df.index < SEAL).all())
        self.assertEqual(df.index[-1], pd.Timestamp("2026-09-30 23:45", tz="UTC"))

    def test_parquet_files_are_read_through_the_same_guard(self):
        idx = pd.date_range("2026-09-30", "2026-10-02", freq="1h", tz="UTC", inclusive="left")
        df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
        os.makedirs(self.data_dir, exist_ok=True)
        df.to_parquet(os.path.join(self.data_dir, "ohlc_BTCUSDT_1h.parquet"))
        out = hg.load_unsealed(self.data_dir, "BTCUSDT", "2026-09-30", SEAL, pattern="ohlc_{symbol}_1h.parquet",
                               lock=self.lock_dir)
        self.assertEqual(len(out), 24)
        with self.assertRaises(hg.SealedAccessError):
            hg.load_unsealed(self.data_dir, "SEALAUSDT", "2026-09-30", SEAL, pattern="ohlc_{symbol}_1h.parquet",
                             lock=self.lock_dir)

    def test_missing_file_for_an_unsealed_coin_is_a_plain_error(self):
        with self.assertRaises(FileNotFoundError):
            hg.load_unsealed(self.data_dir, "BTCUSDT", "2026-09-01", "2026-09-02", lock=self.lock_dir)

    def test_end_is_required(self):
        write_bars(self.data_dir, "BTCUSDT")
        with self.assertRaises(TypeError):
            hg.load_unsealed(self.data_dir, "BTCUSDT", "2026-09-30")


class OpenHoldoutTests(Base):
    def test_good_card_opens_once_and_logs_one_row(self):
        h = hg.open_holdout(self.card(), "H1", lock=self.lock_dir)
        self.assertEqual(h.bucket, "H1")
        self.assertEqual(h.symbols, ("SEALAUSDT", "SEALCUSDT"))
        rows = self.log_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bucket"], "H1")
        self.assertEqual(rows[0]["frozen_rules_sha256"], "b" * 64)
        self.assertEqual(rows[0]["card_sha256"], hashlib.sha256(open(self.card(), "rb").read()).hexdigest())
        self.assertTrue(rows[0]["opened_at_utc"].endswith("Z"))

    def test_second_open_of_the_same_bucket_raises_even_with_another_card(self):
        hg.open_holdout(self.card("one.md"), "H1", lock=self.lock_dir)
        with self.assertRaises(hg.HoldoutOpenError):
            hg.open_holdout(self.card("one.md"), "H1", lock=self.lock_dir)
        with self.assertRaises(hg.HoldoutOpenError):
            hg.open_holdout(self.card("two.md", frozen_rules_sha256="c" * 64), "H1", lock=self.lock_dir)
        self.assertEqual(len(self.log_rows()), 1)

    def test_other_buckets_still_open(self):
        hg.open_holdout(self.card("one.md"), "H1", lock=self.lock_dir)
        hg.open_holdout(self.card("two.md"), "H2", lock=self.lock_dir)
        self.assertEqual([r["bucket"] for r in self.log_rows()], ["H1", "H2"])

    def test_refused_cards_do_not_burn_the_bucket(self):
        bad = [self.card("a.md", confirmation=None), self.card("b.md", confirmation="false"),
               self.card("c.md", frozen_rules_sha256=None), self.card("d.md", frozen_rules_sha256="abc"),
               self.card("e.md", frozen_rules_sha256="G" * 64), self.card("f.md", confirmation="yes")]
        for p in bad:
            with self.assertRaises(hg.HoldoutOpenError, msg=p):
                hg.open_holdout(p, "H1", lock=self.lock_dir)
        self.assertEqual(self.log_rows(), [])
        hg.open_holdout(self.card("ok.md"), "H1", lock=self.lock_dir)                   # still available

    def test_card_without_front_matter_is_refused(self):
        p = os.path.join(self.tmp, "plain.md")
        open(p, "w", encoding="utf-8").write("# no front matter\nconfirmation: true\n")
        with self.assertRaises(hg.HoldoutOpenError):
            hg.open_holdout(p, "H1", lock=self.lock_dir)

    def test_missing_card_file_is_refused(self):
        with self.assertRaises(hg.HoldoutOpenError):
            hg.open_holdout(os.path.join(self.tmp, "none.md"), "H1", lock=self.lock_dir)

    def test_unknown_or_empty_bucket_is_refused(self):
        for b in ("H7", "h1", "", "H0"):
            with self.assertRaises(hg.HoldoutOpenError, msg=b):
                hg.open_holdout(self.card(), b, lock=self.lock_dir)
        with self.assertRaises(hg.HoldoutOpenError):                                    # H5 has no sealed symbol here
            hg.open_holdout(self.card(), "H5", lock=self.lock_dir)
        self.assertEqual(self.log_rows(), [])

    def test_card_that_names_another_bucket_is_refused(self):
        with self.assertRaises(hg.HoldoutOpenError):
            hg.open_holdout(self.card(holdout_bucket="H2"), "H1", lock=self.lock_dir)
        hg.open_holdout(self.card("ok.md", holdout_bucket="H1"), "H1", lock=self.lock_dir)

    def test_frozen_rules_file_hash_is_verified_when_named(self):
        rules = os.path.join(self.tmp, "rules.json")
        open(rules, "w", encoding="utf-8").write('{"rule": 1}')
        digest = hashlib.sha256(open(rules, "rb").read()).hexdigest()
        with self.assertRaises(hg.HoldoutOpenError):
            hg.open_holdout(self.card("x.md", frozen_rules_file=rules), "H1", lock=self.lock_dir)      # hash is "bbbb..."
        hg.open_holdout(self.card("y.md", frozen_rules_file=rules, frozen_rules_sha256=digest), "H1", lock=self.lock_dir)

    def test_log_is_append_only_header_kept(self):
        hg.open_holdout(self.card("one.md"), "H1", lock=self.lock_dir)
        hg.open_holdout(self.card("two.md"), "H2", lock=self.lock_dir)
        with open(os.path.join(self.lock_dir, "holdout_access_log.csv"), encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], ",".join(hg.LOG_COLUMNS))
        self.assertEqual(len(lines), 3)

    def test_handle_loads_only_its_bucket_and_never_after_the_seal(self):
        write_bars(self.data_dir, "SEALAUSDT", end="2026-10-05")
        write_bars(self.data_dir, "SEALBUSDT")
        h = hg.open_holdout(self.card(), "H1", lock=self.lock_dir)
        df = h.load(self.data_dir, "SEALAUSDT", "2026-09-29", SEAL)
        self.assertTrue((df.index < SEAL).all())
        self.assertEqual(df.index[-1], pd.Timestamp("2026-09-30 23:45", tz="UTC"))
        with self.assertRaises(hg.SealedAccessError):
            h.load(self.data_dir, "SEALAUSDT", "2026-09-29", "2026-10-03")                # time seal still applies
        with self.assertRaises(hg.SealedAccessError):
            h.load(self.data_dir, "SEALBUSDT", "2026-09-29", SEAL)                        # another bucket's coin
        with self.assertRaises(hg.SealedAccessError):
            h.load(self.data_dir, "BTCUSDT", "2026-09-29", SEAL)                          # not a sealed coin at all

    def test_sealed_symbol_stays_refused_by_the_plain_loader_after_opening(self):
        write_bars(self.data_dir, "SEALAUSDT")
        hg.open_holdout(self.card(), "H1", lock=self.lock_dir)
        with self.assertRaises(hg.SealedAccessError):
            hg.load_unsealed(self.data_dir, "SEALAUSDT", "2026-09-29", SEAL, lock=self.lock_dir)


class RepoLockStateTests(unittest.TestCase):
    """The committed access log (when present) is a header plus appended rows only."""

    def test_committed_access_log_has_the_header(self):
        p = os.path.join(ROOT, "docs", "research", "plan", "lock", "holdout_access_log.csv")
        self.assertTrue(os.path.exists(p), "holdout_access_log.csv must be committed with its header")
        with open(p, encoding="utf-8") as f:
            self.assertEqual(f.readline().rstrip("\n"), ",".join(hg.LOG_COLUMNS))


if __name__ == "__main__":
    unittest.main()

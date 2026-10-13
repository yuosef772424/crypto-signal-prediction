"""tools/build_lock.py: pasted manifest + touched.csv + salt -> holdout.json + holdout.sha256, deterministically.
Fixture manifests only (no Drive).
    python -m unittest tests.test_build_lock -v
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import build_lock as bl  # noqa: E402
import data_manifest as dm  # noqa: E402
import holdout_guard as hg  # noqa: E402

SALT = "0123456789abcdef0123456789abcdef01234567"
GOOD = ("2023-01-01T00:00:00Z", "2026-09-26T23:45:00Z")
AFTER = "2026-10-08T23:45:00Z"                                              # a last bar after the seal (the span stops at the seal)
TOUCH_HEADER = ["symbol_or_rule", "kind", "timeframe", "period", "source", "note"]


def fixture_rows(n_ok=40):
    """(symbol, first, last, error) rows covering every ineligibility reason plus n_ok clean symbols."""
    rows = [(f"OK{i:02d}USDT",) + GOOD + ("",) for i in range(n_ok)]
    rows += [("SHORTUSDT", "2025-06-01T00:00:00Z", GOOD[1], ""),            # < 2 years before the seal
             ("EDGEUSDT", "2024-10-01T00:00:00Z", AFTER, ""),               # 729.99 days to the seal: too short
             ("JUST2YUSDT", "2024-09-30T23:45:00Z", AFTER, ""),             # exactly 730 days to 2026-09-30T23:45: eligible
             ("STALEUSDT", GOOD[0], "2026-08-31T23:45:00Z", ""),
             ("FRESHUSDT", GOOD[0], "2026-09-01T00:00:00Z", ""),            # last bar exactly at the limit: eligible
             ("USDCUSDT",) + GOOD + ("",), ("ETHUPUSDT",) + GOOD + ("",), ("ETHDOWNUSDT",) + GOOD + ("",),
             ("ETHUSDT",) + GOOD + ("",),                                    # touched below, and the base of ETHUP/ETHDOWN
             ("JUPUSDT",) + GOOD + ("",),                                    # ends in UP but J is not a symbol: eligible
             ("BTCDOMUSDT",) + GOOD + ("",),
             ("BADUSDT", "", "", "OSError: bad gzip"),
             ("TOUCHEDUSDT",) + GOOD + ("",), ("FOLDERUSDT",) + GOOD + ("",),
             ("OLDUSDT", "2019-12-01T00:00:00Z", GOOD[1], "")]              # touched by the first_bar_before rule
    return rows


def write_manifest(d, rows, folders=(("history_1d", ["FOLDERUSDT", "OTHERUSDT"]),), complete=True, extra=None):
    os.makedirs(d, exist_ok=True)
    recs = [{"symbol": s, "first_bar": f, "last_bar": l, "rows": (0 if e else 1000), "bytes": 10,
             "sha256": hashlib.sha256(s.encode()).hexdigest(), "error": e} for s, f, l, e in rows]
    dm._write_csv(os.path.join(d, dm.OUT_CSV), dm.CSV_COLUMNS, recs)
    dm._write_csv(os.path.join(d, dm.OUT_FOLDERS), dm.FOLDER_COLUMNS, [[k, s] for k, ss in folders for s in ss])
    outputs = {n: dm._sha256(os.path.join(d, n)) for n in (dm.OUT_CSV, dm.OUT_FOLDERS)}
    m = {"schema": dm.SCHEMA, "git_sha": "f" * 40, "config": {}, "input_fingerprint": "e" * 64, "complete": complete,
         "n_files_total": len(rows), "n_files_listed": len(rows), "outputs": outputs}
    m.update(extra or {})
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(m, indent=1, sort_keys=True) + "\n")


def write_touched(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(TOUCH_HEADER)
        for r in rows:
            w.writerow(r)


TOUCHED = [["ETHUSDT", "symbol", "1h", "2024", "src-eth", ""],
           ["TOUCHEDUSDT", "symbol", "1d", "2023", "src-a", ""],
           ["folder:history_1d", "rule", "1d", "2019-09..2026-09", "src-folder", "whole folder"],
           ["first_bar_before:2020-01-01", "rule", "1d", "2018..2026", "src-coinmetrics", "proxy"],
           ["OK00USDT", "time_seal_read", "1d", "2026-10-01..2026-10-03", "src-fwd", ""]]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mdir = os.path.join(self.tmp, "manifest")
        self.touched = os.path.join(self.tmp, "touched.csv")
        write_manifest(self.mdir, fixture_rows())
        write_touched(self.touched, TOUCHED)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, **kw):
        return bl.build(self.mdir, self.touched, SALT, repo_check=False, **kw)


class EligibilityTests(Base):
    def test_every_reason(self):
        d = self.build()
        why = d["ineligible"]
        self.assertEqual(why["SHORTUSDT"], ["short_history"])
        self.assertEqual(why["EDGEUSDT"], ["short_history"])
        self.assertEqual(why["STALEUSDT"], ["stale"])
        self.assertEqual(why["USDCUSDT"], ["stablecoin"])
        self.assertEqual(why["ETHUPUSDT"], ["leveraged"])
        self.assertEqual(why["ETHDOWNUSDT"], ["leveraged"])
        self.assertEqual(why["BTCDOMUSDT"], ["index_contract"])
        self.assertEqual(why["BADUSDT"], ["unreadable"])
        self.assertEqual(why["ETHUSDT"], ["touched"])
        self.assertEqual(why["TOUCHEDUSDT"], ["touched"])
        self.assertEqual(why["FOLDERUSDT"], ["touched"])              # folder rule
        self.assertEqual(why["OLDUSDT"], ["touched"])                 # first_bar_before rule
        for ok in ("JUST2YUSDT", "FRESHUSDT", "JUPUSDT", "OK00USDT"):
            self.assertNotIn(ok, why, ok)
            self.assertIn(ok, d["eligible"])

    def test_reasons_can_stack(self):
        write_manifest(self.mdir, fixture_rows() + [("TUSDUSDT", "2025-06-01T00:00:00Z", "2026-08-01T00:00:00Z", "")])
        d = self.build()
        self.assertEqual(d["ineligible"]["TUSDUSDT"], ["short_history", "stale", "stablecoin"])

    def test_time_seal_reads_do_not_change_eligibility_but_are_listed(self):
        d = self.build()
        self.assertIn("OK00USDT", d["eligible"])
        self.assertEqual(d["time_seal_reads"][0]["source"], "src-fwd")

    def test_touched_sources_are_recorded(self):
        d = self.build()
        self.assertEqual(d["touched_sources"]["FOLDERUSDT"], ["src-folder"])
        self.assertEqual(d["touched_sources"]["OLDUSDT"], ["src-coinmetrics"])
        self.assertEqual(d["touched_sources"]["ETHUSDT"], ["src-eth"])

    def test_funnel_counts(self):
        d = self.build()
        f = d["funnel"]
        self.assertEqual(f["manifest_symbols"], len(fixture_rows()))
        self.assertEqual(f["eligible"], len(d["eligible"]))
        self.assertEqual(f["sealed"] + f["open_pool"], f["eligible"])
        self.assertEqual(f["ineligible"] + f["eligible"], f["manifest_symbols"])
        self.assertEqual(f["failed_test"]["touched"], 4)
        self.assertEqual(f["failed_test"]["short_history"], 2)


class SealingTests(Base):
    def rank(self, symbols):
        return sorted(symbols, key=lambda s: (hashlib.sha256((SALT + s).encode("utf-8")).hexdigest(), s))

    def test_sealed_is_floor_two_fifths_in_hash_order(self):
        d = self.build()
        el = d["eligible"]
        n = len(el)
        k = (2 * n) // 5
        self.assertEqual(d["sealed_order"], self.rank(el)[:k])
        self.assertEqual(set(d["sealed"]), set(d["sealed_order"]))
        self.assertEqual(d["funnel"]["sealed"], k)
        self.assertEqual(sorted(d["open_pool"]), sorted(set(el) - set(d["sealed"])))

    def test_round_robin_buckets(self):
        d = self.build()
        order = d["sealed_order"]
        self.assertEqual(order, self.rank(d["eligible"])[: d["funnel"]["sealed"]])
        for i, s in enumerate(order):
            self.assertEqual(d["sealed"][s], f"H{i % 6 + 1}")
        for b, syms in d["buckets"].items():
            self.assertEqual(syms, sorted(s for s, bb in d["sealed"].items() if bb == b))
        self.assertEqual(d["bucket_phase"], {"H1": "R1b", "H2": "R3", "H3": "R2", "H4": "R5", "H5": "G2", "H6": "reserve (R6)"})

    def test_floor_uses_integer_arithmetic(self):
        for n, want in [(0, 0), (1, 0), (2, 0), (3, 1), (5, 2), (7, 2), (10, 4), (35, 14), (36, 14), (37, 14), (38, 15), (100, 40)]:
            self.assertEqual(bl.n_sealed(n), want, n)

    def test_hash_is_sha256_of_salt_plus_symbol(self):
        self.assertEqual(bl.rank_hash(SALT, "ABCUSDT"), hashlib.sha256((SALT + "ABCUSDT").encode()).hexdigest())

    def test_other_salt_gives_another_sealed_set(self):
        a = self.build()
        b = bl.build(self.mdir, self.touched, "f" * 40, repo_check=False)
        self.assertEqual(a["eligible"], b["eligible"])
        self.assertNotEqual(a["sealed_order"], b["sealed_order"])

    def test_row_order_does_not_matter_and_build_is_deterministic(self):
        a = self.build()
        self.assertEqual(bl.dumps(a), bl.dumps(self.build()))                          # same inputs, same bytes
        write_manifest(self.mdir, list(reversed(fixture_rows())))
        b = bl.build(self.mdir, self.touched, SALT, repo_check=False)
        a.pop("inputs"), b.pop("inputs")                                                # the CSV bytes (and so its hash) differ
        self.assertEqual(bl.dumps(a), bl.dumps(b))

    def test_small_buckets_warn(self):
        d = self.build()
        self.assertTrue(any("bucket" in w for w in d["warnings"]))
        self.assertTrue(any("sealed symbols" in w for w in d["warnings"]))


class InputValidationTests(Base):
    def test_incomplete_manifest_is_refused(self):
        write_manifest(self.mdir, fixture_rows(), complete=False)
        with self.assertRaises(ValueError):
            self.build()

    def test_edited_csv_is_refused(self):
        p = os.path.join(self.mdir, dm.OUT_CSV)
        open(p, "a", encoding="utf-8").write("EXTRAUSDT,2023-01-01T00:00:00Z,2026-09-26T23:45:00Z,1,1,x,\n")
        with self.assertRaises(ValueError):
            self.build()

    def test_duplicate_symbol_is_refused(self):
        write_manifest(self.mdir, fixture_rows() + [fixture_rows()[0]])
        with self.assertRaises(ValueError):
            self.build()

    def test_missing_folder_for_a_rule_fails(self):
        write_manifest(self.mdir, fixture_rows(), folders=())
        with self.assertRaises(ValueError):
            self.build()

    def test_bad_touched_files_are_refused(self):
        bad = [[["X", "weird", "", "", "s", ""]],                                  # unknown kind
               [["folder:", "rule", "", "", "s", ""]],                             # bad rule
               [["first_bar_before:2020", "rule", "", "", "s", ""]],
               [["symbol:ABC", "rule", "", "", "s", ""]],
               [["ABCUSDT", "symbol", "", "", "", ""]],                            # no source
               [["", "symbol", "", "", "s", ""]]]
        for rows in bad:
            write_touched(self.touched, rows)
            with self.assertRaises(ValueError, msg=rows):
                self.build()
        open(self.touched, "w").write("a,b\n1,2\n")
        with self.assertRaises(ValueError):
            self.build()

    def test_bad_salt_is_refused(self):
        for s in ("", "abc", "G" * 40, SALT.upper(), SALT + "0"):
            with self.assertRaises(ValueError, msg=s):
                bl.build(self.mdir, self.touched, s, repo_check=False)

    def test_touched_symbols_are_normalised(self):
        write_touched(self.touched, [[" ok01usdt ", "symbol", "1d", "x", "src", ""]])
        self.assertEqual(self.build()["ineligible"]["OK01USDT"], ["touched"])


class OutputTests(Base):
    def test_write_and_guard_accepts(self):
        out = os.path.join(self.tmp, "lock")
        d = self.build()
        bl.write_lock(d, out)
        body = open(os.path.join(out, "holdout.json"), "rb").read()
        self.assertEqual(open(os.path.join(out, "holdout.sha256")).read(),
                         hashlib.sha256(body).hexdigest() + "  holdout.json\n")
        lock = hg.load_lock(out)                                        # the guard verifies the same files
        self.assertEqual(set(lock.sealed), set(d["sealed"]))
        self.assertEqual(lock.salt, SALT)
        with self.assertRaises(hg.SealedAccessError):
            hg.assert_unsealed([next(iter(d["sealed"]))], "2026-01-01", "2026-02-01", lock=lock)

    def test_write_refuses_to_overwrite(self):
        out = os.path.join(self.tmp, "lock")
        d = self.build()
        bl.write_lock(d, out)
        with self.assertRaises(FileExistsError):
            bl.write_lock(d, out)

    def test_inputs_are_fingerprinted(self):
        d = self.build()
        self.assertEqual(d["inputs"]["touched_csv_sha256"], hashlib.sha256(open(self.touched, "rb").read()).hexdigest())
        self.assertEqual(d["inputs"]["manifest_json_sha256"], hashlib.sha256(open(os.path.join(self.mdir, "manifest.json"), "rb").read()).hexdigest())
        self.assertEqual(d["salt"], SALT)


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def bundle(self):
        src = os.path.join(self.tmp, "src")
        write_manifest(src, fixture_rows())
        # use the real writer so the bundle matches what the Colab script prints
        return dm.bundle_text(src)

    def test_ingest_writes_the_sections_verbatim_and_refuses_overwrite(self):
        text = self.bundle()
        dest = os.path.join(self.tmp, "manifest")
        bl.ingest(text, dest)
        for name in (dm.OUT_CSV, dm.OUT_FOLDERS):
            self.assertEqual(open(os.path.join(dest, name), encoding="utf-8").read(), open(os.path.join(self.tmp, "src", name), encoding="utf-8").read())
        self.assertTrue(os.path.exists(os.path.join(dest, "manifest.json")))
        with self.assertRaises(FileExistsError):
            bl.ingest(text, dest)
        touched = os.path.join(self.tmp, "touched.csv")
        write_touched(touched, TOUCHED)
        self.assertGreater(len(bl.build(dest, touched, SALT, repo_check=False)["eligible"]), 0)

    def test_altered_bundle_is_refused(self):
        with self.assertRaises(ValueError):
            bl.ingest(self.bundle().replace("OK01USDT,2023", "OK01USDT,2022", 1), os.path.join(self.tmp, "m"))


class SaltTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
        def git(*a):
            return subprocess.run(["git", "-C", self.tmp, *a], env=env, capture_output=True, text=True, check=True).stdout.strip()
        git("init", "-q")
        os.makedirs(os.path.join(self.tmp, "docs/research/plan/lock"))
        open(os.path.join(self.tmp, "other.txt"), "w").write("x")
        git("add", "."); git("commit", "-q", "-m", "first")
        self.first = git("rev-parse", "HEAD")
        open(os.path.join(self.tmp, "docs/research/plan/lock/README.md"), "w").write("spec")
        git("add", "."); git("commit", "-q", "-m", "spec")
        self.spec = git("rev-parse", "HEAD")
        open(os.path.join(self.tmp, "docs/research/plan/lock/README.md"), "a").write(" more")
        git("add", "."); git("commit", "-q", "-m", "edit the spec")
        self.later = git("rev-parse", "HEAD")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_commit_that_adds_the_readme_is_the_salt(self):
        bl.verify_salt(self.spec, self.tmp)

    def test_other_commits_are_refused(self):
        for bad in (self.first, self.later, "0" * 40):
            with self.assertRaises(ValueError, msg=bad):
                bl.verify_salt(bad, self.tmp)

    def test_build_checks_the_repo_unless_told_not_to(self):
        tmp = tempfile.mkdtemp()
        try:
            write_manifest(os.path.join(tmp, "m"), fixture_rows())
            write_touched(os.path.join(tmp, "t.csv"), TOUCHED)
            with self.assertRaises(ValueError):
                bl.build(os.path.join(tmp, "m"), os.path.join(tmp, "t.csv"), self.later, repo_root=self.tmp)
            d = bl.build(os.path.join(tmp, "m"), os.path.join(tmp, "t.csv"), self.spec, repo_root=self.tmp)
            self.assertEqual(d["salt"], self.spec)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CliTests(Base):
    def test_build_and_ingest_subcommands(self):
        out = os.path.join(self.tmp, "lock")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = bl.main(["build", "--manifest-dir", self.mdir, "--touched", self.touched, "--salt", SALT, "--out", out, "--no-repo-check"])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(out, "holdout.json")))
        self.assertIn("eligible", buf.getvalue())

    def test_build_without_repo_check_flag_checks_the_real_repo(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(ValueError):
                bl.main(["build", "--manifest-dir", self.mdir, "--touched", self.touched, "--salt", SALT,
                         "--out", os.path.join(self.tmp, "lock2")])


if __name__ == "__main__":
    unittest.main()

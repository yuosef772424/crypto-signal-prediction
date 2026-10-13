"""tools/build_lock.py: pasted manifest + touched.csv + salt -> holdout.json + holdout.sha256, deterministically.
Fixture manifests only (no Drive).
    python -m unittest tests.test_build_lock -v
"""
import contextlib
import csv
import datetime as dt
import gzip
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
TOUCH_HEADER = ["symbol_or_rule", "kind", "touch_class", "timeframe", "period", "source", "note"]


def fixture_rows(n_ok=40):
    """(symbol, first, last, error) rows covering every ineligibility reason plus n_ok clean symbols."""
    rows = [(f"OK{i:02d}USDT",) + GOOD + ("",) for i in range(n_ok)]
    rows += [("SHORTUSDT", "2026-01-01T00:00:00Z", GOOD[1], ""),            # < 365 days before the seal
             ("EDGEUSDT", "2025-10-01T00:00:00Z", AFTER, ""),               # 364 d 23:45 to the seal: too short
             ("JUST1YUSDT", "2025-09-30T23:45:00Z", AFTER, ""),             # exactly 365 days to 2026-09-30T23:45: eligible
             ("STALEUSDT", GOOD[0], "2026-08-31T23:45:00Z", ""),
             ("FRESHUSDT", GOOD[0], "2026-09-01T00:00:00Z", ""),            # last bar exactly at the limit: eligible
             ("USDCUSDT",) + GOOD + ("",), ("ETHUPUSDT",) + GOOD + ("",), ("ETHDOWNUSDT",) + GOOD + ("",),
             ("ETHUSDT",) + GOOD + ("",),                                    # intraday-touched, and the base of ETHUP/ETHDOWN
             ("JUPUSDT",) + GOOD + ("",),                                    # ends in UP but J is not a symbol: eligible
             ("BTCDOMUSDT",) + GOOD + ("",),
             ("BADUSDT", "", "", "OSError: bad gzip"),
             ("DAILYUSDT",) + GOOD + ("",), ("FOLDERUSDT",) + GOOD + ("",), ("DERIVUSDT",) + GOOD + ("",),
             ("OLDUSDT", "2019-12-01T00:00:00Z", GOOD[1], "")]              # named only by an unresolved rule: level A
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


TOUCHED = [["ETHUSDT", "symbol", "intraday", "1h", "2024", "src-eth", ""],
           ["ETHUSDT", "symbol", "daily", "1d", "2024", "src-eth-daily", ""],
           ["DAILYUSDT", "symbol", "daily", "1d", "2023", "src-a", ""],
           ["DERIVUSDT", "symbol", "derivs", "1h", "2024", "src-oi", ""],
           ["folder:history_1d", "rule", "daily", "1d", "2019-09..2026-09", "src-folder", "whole folder"],
           ["unresolved:coinmetrics-44", "rule", "daily", "1d", "2018..2026", "src-coinmetrics", "not listed"],
           ["OK00USDT", "time_seal_read", "", "1d", "2026-10-01..2026-10-03", "src-fwd", ""]]


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
        self.assertEqual(why["ETHUSDT"], ["touched_intraday"])          # level X (also has a daily row)
        for ok in ("JUST1YUSDT", "FRESHUSDT", "JUPUSDT", "OK00USDT", "DAILYUSDT", "FOLDERUSDT", "DERIVUSDT", "OLDUSDT"):
            self.assertNotIn(ok, why, ok)
            self.assertIn(ok, d["eligible"], ok)

    def test_history_is_365_days_before_the_seal(self):
        why = self.build()["ineligible"]
        self.assertIn("EDGEUSDT", why)              # 364 days 23:45
        self.assertNotIn("JUST1YUSDT", why)         # exactly 365 days
        self.assertIn("SHORTUSDT", why)

    def test_levels_a_and_b(self):
        d = self.build()
        lv = d["levels"]
        self.assertEqual(lv["OK00USDT"], "A")                 # only a time_seal_read row: not a touch
        self.assertEqual(lv["OLDUSDT"], "A")                  # an unresolved rule touches nobody
        for b in ("DAILYUSDT", "FOLDERUSDT", "DERIVUSDT"):
            self.assertEqual(lv[b], "B", b)
        self.assertEqual(set(lv), set(d["eligible"]))
        self.assertNotIn("ETHUSDT", lv)                       # level X is not eligible
        f = d["funnel"]
        self.assertEqual(f["eligible_A"] + f["eligible_B"], f["eligible"])
        self.assertEqual(f["eligible_B"], 3)

    def test_unresolved_rules_are_recorded_and_touch_nobody(self):
        d = self.build()
        self.assertEqual([r["rule"] for r in d["unresolved_rules"]], ["unresolved:coinmetrics-44"])
        self.assertNotIn("OLDUSDT", d["touched_sources"])

    def test_reasons_can_stack(self):
        write_manifest(self.mdir, fixture_rows() + [("TUSDUSDT", "2026-02-01T00:00:00Z", "2026-08-01T00:00:00Z", "")])
        d = self.build()
        self.assertEqual(d["ineligible"]["TUSDUSDT"], ["short_history", "stale", "stablecoin"])

    def test_time_seal_reads_do_not_change_eligibility_but_are_listed(self):
        d = self.build()
        self.assertIn("OK00USDT", d["eligible"])
        self.assertEqual(d["time_seal_reads"][0]["source"], "src-fwd")

    def test_touching_sources_and_classes_are_recorded(self):
        d = self.build()
        self.assertEqual(d["touched_sources"]["FOLDERUSDT"], ["src-folder"])
        self.assertEqual(d["touch_classes"]["FOLDERUSDT"], ["daily"])
        self.assertEqual(d["touched_sources"]["ETHUSDT"], ["src-eth", "src-eth-daily"])
        self.assertEqual(d["touch_classes"]["ETHUSDT"], ["daily", "intraday"])
        self.assertEqual(d["touch_classes"]["DERIVUSDT"], ["derivs"])

    def test_funnel_counts(self):
        d = self.build()
        f = d["funnel"]
        self.assertEqual(f["manifest_symbols"], len(fixture_rows()))
        self.assertEqual(f["eligible"], len(d["eligible"]))
        self.assertEqual(f["sealed"] + f["open_pool"], f["eligible"])
        self.assertEqual(f["ineligible"] + f["eligible"], f["manifest_symbols"])
        self.assertEqual(f["failed_test"]["touched_intraday"], 1)
        self.assertEqual(f["failed_test"]["short_history"], 2)


def expected_buckets(order, blocked):
    """Independent statement of the D-009 rule: nominal H(i mod 6 + 1); H3 refuses derivs-touched coins, and a refused coin goes
    to the least-filled bucket that accepts it (counts so far, in rank order), ties broken H1..H6."""
    out, count = {}, {f"H{k}": 0 for k in range(1, 7)}
    for i, sym in enumerate(order):
        nominal = f"H{i % 6 + 1}"
        if nominal == "H3" and sym in blocked:
            accepting = [b for b in count if b != "H3"]
            nominal = min(accepting, key=lambda b: (count[b], b))
        out[sym] = nominal
        count[nominal] += 1
    return out


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
        blocked = {x for x, c in d["touch_classes"].items() if "derivs" in c}
        self.assertEqual(d["sealed"], expected_buckets(order, blocked))
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


class BucketPredicateTests(unittest.TestCase):
    """H3 (R2, funding/basis) accepts only coins with no funding/OI/basis/metrics touch (D-009)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mdir = os.path.join(self.tmp, "manifest")
        self.touched = os.path.join(self.tmp, "touched.csv")
        rows = [(f"AAA{i:02d}USDT",) + GOOD + ("",) for i in range(60)] + [(f"DRV{i:02d}USDT",) + GOOD + ("",) for i in range(60)]
        write_manifest(self.mdir, rows, folders=(("history_1d", ["AAA00USDT"]),))
        write_touched(self.touched, [[f"DRV{i:02d}USDT", "symbol", "derivs", "1h", "2024", "src-oi", ""] for i in range(60)] +
                      [["folder:history_1d", "rule", "daily", "1d", "x", "src-folder", ""]])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_h3_never_holds_a_derivs_touched_coin(self):
        d = bl.build(self.mdir, self.touched, SALT, repo_check=False)
        self.assertGreater(d["funnel"]["sealed"], 30)
        blocked = {s for s, c in d["touch_classes"].items() if "derivs" in c}
        self.assertTrue(blocked & set(d["sealed"]))                            # the test is not vacuous
        self.assertEqual(set(d["buckets"]["H3"]) & blocked, set())
        self.assertEqual(d["sealed"], expected_buckets(d["sealed_order"], blocked))

    def test_refused_coin_goes_to_the_least_filled_accepting_bucket_and_others_keep_their_nominal_bucket(self):
        d = bl.build(self.mdir, self.touched, SALT, repo_check=False)
        blocked = {s for s, c in d["touch_classes"].items() if "derivs" in c}
        count = {f"H{k}": 0 for k in range(1, 7)}
        moved = 0
        for i, s in enumerate(d["sealed_order"]):
            nominal = f"H{i % 6 + 1}"
            if nominal == "H3" and s in blocked:
                want = min((b for b in count if b != "H3"), key=lambda b: (count[b], b))
                self.assertEqual(d["sealed"][s], want, s)
                moved += 1
            else:
                self.assertEqual(d["sealed"][s], nominal, s)
            count[d["sealed"][s]] += 1
        self.assertGreater(moved, 0)                                           # not vacuous
        sizes = [len(v) for k, v in d["buckets"].items() if k != "H3"]
        self.assertLessEqual(max(sizes) - min(sizes), 1 + moved // 5 + 1)       # refused coins are spread, not piled on H4

    def test_ties_break_in_bucket_order(self):
        self.assertEqual(bl.assign_buckets(["A", "B", "C"], {"C"}), {"A": "H1", "B": "H2", "C": "H4"})   # H4, H5, H6 are empty: tie -> H4
        self.assertEqual(bl.assign_buckets(["A", "B", "C", "D", "E", "F"], {"C"}),
                         {"A": "H1", "B": "H2", "C": "H4", "D": "H4", "E": "H5", "F": "H6"})
        order = [f"S{i:02d}" for i in range(12)]                     # C refused at i=2 and i=8: counts decide, not position
        got = bl.assign_buckets(order, {"S02", "S08"})
        self.assertEqual(got["S02"], "H4")
        self.assertEqual(got["S08"], "H5")                           # after S02: H4=2 (S02,S03), H5=1 (S04), H6=1 (S05) -> least filled tie H5/H6 -> H5

    def test_h3_empty_when_every_coin_is_derivs_touched(self):
        write_touched(self.touched, [[f"AAA{i:02d}USDT", "symbol", "derivs", "1h", "2024", "src-oi", ""] for i in range(60)] +
                      [[f"DRV{i:02d}USDT", "symbol", "derivs", "1h", "2024", "src-oi", ""] for i in range(60)] +
                      [["folder:history_1d", "rule", "daily", "1d", "x", "src-folder", ""]])
        d = bl.build(self.mdir, self.touched, SALT, repo_check=False)
        self.assertEqual(d["buckets"]["H3"], [])
        self.assertTrue(any("H3" in w for w in d["warnings"]))
        self.assertEqual(d["funnel"]["sealed"], sum(len(v) for v in d["buckets"].values()))

    def test_daily_only_touch_does_not_block_h3(self):
        write_touched(self.touched, [[f"AAA{i:02d}USDT", "symbol", "daily", "1d", "x", "src-d", ""] for i in range(60)] +
                      [["folder:history_1d", "rule", "daily", "1d", "x", "src-folder", ""]])
        d = bl.build(self.mdir, self.touched, SALT, repo_check=False)
        for i, s in enumerate(d["sealed_order"]):
            self.assertEqual(d["sealed"][s], f"H{i % 6 + 1}")

    def test_levels_per_bucket_are_recorded(self):
        d = bl.build(self.mdir, self.touched, SALT, repo_check=False)
        for b, syms in d["buckets"].items():
            self.assertEqual(d["funnel"]["bucket_levels"][b], {"A": sum(d["levels"][s] == "A" for s in syms),
                                                              "B": sum(d["levels"][s] == "B" for s in syms)})


class InputValidationTests(Base):
    def test_incomplete_manifest_is_refused(self):
        write_manifest(self.mdir, fixture_rows(), complete=False)
        with self.assertRaises(ValueError):
            self.build()

    def test_split_manifest_files_are_concatenated(self):
        src = os.path.join(self.tmp, "raw")
        h15, h1d = os.path.join(self.tmp, "h15"), os.path.join(self.tmp, "h1d")
        os.makedirs(h15), os.makedirs(h1d)
        for sym, first, last, _ in fixture_rows()[:30]:
            with gzip.open(os.path.join(h15, f"{sym}.csv.gz"), "wt") as f:
                f.write("timestamp,datetime_utc\n")
                f.write(f"{int(dt.datetime.strptime(first, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc).timestamp()) * 1000},x\n")
                f.write(f"{int(dt.datetime.strptime(last, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc).timestamp()) * 1000},x\n")
        open(os.path.join(h1d, "OK01USDT.csv"), "w").write("datetime_utc\n")
        m = dm.run({"history_15m_dir": h15, "extra_folders": {"history_1d": h1d}}, src, max_file_bytes=1200)
        self.assertGreater(len([n for n in m["outputs"] if n.startswith("history_15m_manifest")]), 1)
        write_touched(self.touched, [["folder:history_1d", "rule", "daily", "1d", "x", "src-folder", ""]])
        d = bl.build(src, self.touched, SALT, repo_check=False)
        self.assertEqual(d["funnel"]["manifest_symbols"], 30)
        self.assertEqual(d["levels"]["OK01USDT"], "B")

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
        bad = [[["X", "weird", "", "", "", "s", ""]],                              # unknown kind
               [["folder:", "rule", "daily", "", "", "s", ""]],                    # bad rule
               [["first_bar_before:2020", "rule", "daily", "", "", "s", ""]],
               [["symbol:ABC", "rule", "daily", "", "", "s", ""]],
               [["ABCUSDT", "symbol", "daily", "", "", "", ""]],                   # no source
               [["", "symbol", "daily", "", "", "s", ""]],
               [["ABCUSDT", "symbol", "", "", "", "s", ""]],                       # no touch class
               [["ABCUSDT", "symbol", "hourly", "", "", "s", ""]],                 # unknown touch class
               [["ABCUSDT", "time_seal_read", "daily", "", "", "s", ""]]]          # a seal read has no class
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
        write_touched(self.touched, [[" ok01usdt ", "symbol", "intraday", "1d", "x", "src", ""]])
        self.assertEqual(self.build()["ineligible"]["OK01USDT"], ["touched_intraday"])


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

    def test_ingest_from_a_run_folder_is_the_same_as_from_text(self):
        src = os.path.join(self.tmp, "src")
        write_manifest(src, fixture_rows())
        open(os.path.join(src, "runner_log.txt"), "w").write("extra file written by a runner")
        a, b = os.path.join(self.tmp, "a"), os.path.join(self.tmp, "b")
        bl.ingest_dir(src, a)
        bl.ingest(dm.bundle_text(src), b)
        self.assertEqual(sorted(os.listdir(a)), sorted(os.listdir(b)))             # the runner's extra file is not copied
        for n in os.listdir(a):
            self.assertEqual(open(os.path.join(a, n), "rb").read(), open(os.path.join(b, n), "rb").read())
        with self.assertRaises(FileExistsError):
            bl.ingest_dir(src, a)

    def test_run_folder_with_an_edited_output_or_missing_manifest_is_refused(self):
        src = os.path.join(self.tmp, "src")
        write_manifest(src, fixture_rows())
        open(os.path.join(src, dm.OUT_CSV), "a").write("X,1,2,3,4,5,\n")
        with self.assertRaises(ValueError):
            bl.ingest_dir(src, os.path.join(self.tmp, "a"))
        with self.assertRaises(ValueError):
            bl.ingest_dir(os.path.join(self.tmp, "empty"), os.path.join(self.tmp, "b"))
        os.remove(os.path.join(src, dm.OUT_FOLDERS))
        with self.assertRaises(ValueError):
            bl.ingest_dir(src, os.path.join(self.tmp, "c"))

    def test_cli_ingest_from_dir(self):
        src = os.path.join(self.tmp, "src")
        write_manifest(src, fixture_rows())
        dest = os.path.join(self.tmp, "dest")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bl.main(["ingest", "--from-dir", src, "--dir", dest]), 0)
        self.assertTrue(os.path.exists(os.path.join(dest, "manifest.json")))
        with self.assertRaises(SystemExit):                                          # exactly one of --bundle / --from-dir
            with contextlib.redirect_stderr(io.StringIO()):
                bl.main(["ingest", "--dir", dest])

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

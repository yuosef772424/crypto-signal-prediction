"""tools/data_manifest.py: the Colab manifest of Drive history_15m (symbol, first/last bar, rows, file sha256) follows the run
contract (--config with unknown keys raising, --out, --max-minutes, manifest.json with the sha256 of every output, outputs
<= 300 KB, a printed paste bundle). Runs on small fixtures.
    python -m unittest tests.test_data_manifest -v
"""
import contextlib
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import data_manifest as dm  # noqa: E402

HEADER = "timestamp,datetime_utc,open,high,low,close,volume\n"


def write_gz(path, text):
    with gzip.open(path, "wt") as f:
        f.write(text)


def bar_lines(start_ms, n, step_ms=900_000, with_ms=True):
    out = []
    for i in range(n):
        ms = start_ms + i * step_ms
        dt = __import__("datetime").datetime.fromtimestamp(ms / 1000, __import__("datetime").timezone.utc)
        out.append(f"{ms},{dt:%Y-%m-%d %H:%M:%S},1,2,0.5,1.5,10\n")
    return "".join(out)


# 2024-01-01T00:00:00Z in ms
T0 = 1704067200000


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.h15 = os.path.join(self.tmp, "history_15m")
        self.h1d = os.path.join(self.tmp, "history_1d")
        self.out = os.path.join(self.tmp, "out")
        os.makedirs(self.h15)
        os.makedirs(self.h1d)
        write_gz(os.path.join(self.h15, "AAAUSDT.csv.gz"), HEADER + bar_lines(T0, 100))
        write_gz(os.path.join(self.h15, "BBBUSDT.csv.gz"), HEADER + bar_lines(T0 + 86_400_000, 5))
        write_gz(os.path.join(self.h15, "CCCUSDT.csv.gz"), "datetime_utc,open\n2025-03-01 00:00:00,1\n2025-03-01 00:15:00,1\n")
        open(os.path.join(self.h15, "notes.txt"), "w").write("not a data file")
        for s in ("AAAUSDT", "DDDUSDT"):
            open(os.path.join(self.h1d, f"{s}.csv"), "w").write("datetime_utc,close\n")
        self.cfg = {"history_15m_dir": self.h15, "extra_folders": {"history_1d": self.h1d}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_manifest(self, cfg=None, **kw):
        return dm.run(cfg or self.cfg, self.out, **kw)


class FileRowTests(Base):
    def test_rows_first_last_and_sha256(self):
        p = os.path.join(self.h15, "AAAUSDT.csv.gz")
        r = dm.file_row(p, ".csv.gz")
        self.assertEqual(r["symbol"], "AAAUSDT")
        self.assertEqual(r["rows"], 100)
        self.assertEqual(r["first_bar"], "2024-01-01T00:00:00Z")
        self.assertEqual(r["last_bar"], "2024-01-02T00:45:00Z")
        self.assertEqual(r["sha256"], hashlib.sha256(open(p, "rb").read()).hexdigest())
        self.assertEqual(r["bytes"], os.path.getsize(p))
        self.assertEqual(r["error"], "")

    def test_datetime_utc_is_used_when_there_is_no_timestamp_column(self):
        r = dm.file_row(os.path.join(self.h15, "CCCUSDT.csv.gz"), ".csv.gz")
        self.assertEqual((r["rows"], r["first_bar"], r["last_bar"]), (2, "2025-03-01T00:00:00Z", "2025-03-01T00:15:00Z"))

    def test_corrupt_file_is_recorded_not_skipped(self):
        p = os.path.join(self.h15, "BADUSDT.csv.gz")
        open(p, "wb").write(b"this is not gzip")
        r = dm.file_row(p, ".csv.gz")
        self.assertEqual(r["symbol"], "BADUSDT")
        self.assertEqual(r["rows"], 0)
        self.assertNotEqual(r["error"], "")
        self.assertEqual(r["sha256"], hashlib.sha256(b"this is not gzip").hexdigest())

    def test_header_only_file_has_no_bars(self):
        p = os.path.join(self.h15, "EMPTYUSDT.csv.gz")
        write_gz(p, HEADER)
        r = dm.file_row(p, ".csv.gz")
        self.assertEqual(r["rows"], 0)
        self.assertEqual(r["first_bar"], "")


class ConfigTests(Base):
    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            dm.load_config({**self.cfg, "extra_folderz": {}})

    def test_missing_required_key_raises(self):
        with self.assertRaises(ValueError):
            dm.load_config({"extra_folders": {}})

    def test_defaults_and_types(self):
        c = dm.load_config({"history_15m_dir": self.h15})
        self.assertEqual(c["extra_folders"], {})
        self.assertEqual(c["suffix"], ".csv.gz")
        with self.assertRaises(ValueError):
            dm.load_config({"history_15m_dir": self.h15, "extra_folders": ["x"]})

    def test_config_file_is_read_by_path(self):
        p = os.path.join(self.tmp, "cfg.json")
        json.dump(self.cfg, open(p, "w"))
        self.assertEqual(dm.load_config(p)["history_15m_dir"], self.h15)

    def test_missing_folder_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            self.run_manifest({"history_15m_dir": os.path.join(self.tmp, "nope")})
        with self.assertRaises(FileNotFoundError):
            self.run_manifest({"history_15m_dir": self.h15, "extra_folders": {"history_1d": os.path.join(self.tmp, "nope")}})

    def test_folder_without_data_files_raises(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaises(ValueError):
            self.run_manifest({"history_15m_dir": empty})


class RunTests(Base):
    def test_outputs_manifest_and_hashes(self):
        m = self.run_manifest()
        self.assertTrue(m["complete"])
        self.assertEqual(m["n_files_listed"], 3)
        self.assertEqual(m["ignored_files"], ["notes.txt"])
        self.assertEqual(m["n_configurations_evaluated"], 0)
        self.assertEqual(set(m["outputs"]), {"history_15m_manifest.csv", "folders_manifest.csv"})
        for name, digest in m["outputs"].items():
            self.assertEqual(hashlib.sha256(open(os.path.join(self.out, name), "rb").read()).hexdigest(), digest)
        on_disk = json.load(open(os.path.join(self.out, "manifest.json")))
        self.assertEqual(on_disk["outputs"], m["outputs"])
        for k in ("git_sha", "config", "input_fingerprint", "rows", "runtime_seconds", "schema"):
            self.assertIn(k, on_disk)
        self.assertLessEqual(sum(os.path.getsize(os.path.join(self.out, f)) for f in os.listdir(self.out)), 300_000)

    def test_csv_content(self):
        self.run_manifest()
        rows = list(csv.DictReader(open(os.path.join(self.out, "history_15m_manifest.csv"), encoding="utf-8")))
        self.assertEqual([r["symbol"] for r in rows], ["AAAUSDT", "BBBUSDT", "CCCUSDT"])
        self.assertEqual(rows[0]["rows"], "100")
        frows = list(csv.DictReader(open(os.path.join(self.out, "folders_manifest.csv"), encoding="utf-8")))
        self.assertEqual([(r["folder"], r["symbol"]) for r in frows], [("history_1d", "AAAUSDT"), ("history_1d", "DDDUSDT")])

    def test_input_fingerprint_is_deterministic_and_sensitive(self):
        a = self.run_manifest()["input_fingerprint"]
        b = self.run_manifest()["input_fingerprint"]
        self.assertEqual(a, b)
        write_gz(os.path.join(self.h15, "ZZZUSDT.csv.gz"), HEADER + bar_lines(T0, 3))
        self.assertNotEqual(a, self.run_manifest()["input_fingerprint"])

    def test_max_minutes_stops_cleanly_and_says_incomplete(self):
        ticks = iter([0.0, 0.0, 0.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0])
        m = self.run_manifest(max_minutes=1, clock=lambda: next(ticks))
        self.assertFalse(m["complete"])
        self.assertLess(m["n_files_listed"], 3)
        self.assertTrue(os.path.exists(os.path.join(self.out, "manifest.json")))

    def test_a_row_that_cannot_fit_the_file_limit_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_manifest(max_file_bytes=100)

    def test_the_file_limit_cannot_exceed_the_contract(self):
        with self.assertRaises(ValueError):
            self.run_manifest(max_file_bytes=300_001)

    def test_large_listings_are_split_into_parts_each_under_the_limit(self):
        for i in range(40):
            write_gz(os.path.join(self.h15, f"P{i:02d}USDT.csv.gz"), HEADER + bar_lines(T0, 2))
        m = self.run_manifest(max_file_bytes=1500)
        parts = sorted(n for n in m["outputs"] if n.startswith("history_15m_manifest"))
        self.assertGreater(len(parts), 1)
        self.assertEqual(parts, [f"history_15m_manifest.part{i:02d}.csv" for i in range(1, len(parts) + 1)])
        rows = []
        for n in parts:
            self.assertLessEqual(os.path.getsize(os.path.join(self.out, n)), 1500)
            self.assertEqual(open(os.path.join(self.out, n), encoding="utf-8").readline().strip(), ",".join(dm.CSV_COLUMNS))
            rows += list(csv.DictReader(open(os.path.join(self.out, n), encoding="utf-8")))
        self.assertEqual([r["symbol"] for r in rows], sorted(r["symbol"] for r in rows))
        self.assertEqual(len(rows), 43)
        for name, digest in m["outputs"].items():
            self.assertEqual(hashlib.sha256(open(os.path.join(self.out, name), "rb").read()).hexdigest(), digest)
        self.assertEqual(set(dm.parse_bundle(dm.bundle_text(self.out))["sections"]), set(m["outputs"]))

    def test_rerun_with_fewer_parts_does_not_leave_stale_outputs_listed(self):
        for i in range(40):
            write_gz(os.path.join(self.h15, f"P{i:02d}USDT.csv.gz"), HEADER + bar_lines(T0, 2))
        many = self.run_manifest(max_file_bytes=1500)
        few = self.run_manifest()
        self.assertEqual(set(few["outputs"]), {"history_15m_manifest.csv", "folders_manifest.csv"})
        self.assertGreater(len(many["outputs"]), len(few["outputs"]))
        self.assertEqual(sorted(os.listdir(self.out)), sorted(list(few["outputs"]) + ["manifest.json"]))      # old parts removed

    def test_everything_is_written_as_files_in_out(self):
        m = self.run_manifest()
        self.assertEqual(sorted(os.listdir(self.out)), sorted(list(m["outputs"]) + ["manifest.json"]))

    def test_unreadable_file_still_listed_with_error(self):
        open(os.path.join(self.h15, "BADUSDT.csv.gz"), "wb").write(b"junk")
        self.run_manifest()
        rows = {r["symbol"]: r for r in csv.DictReader(open(os.path.join(self.out, "history_15m_manifest.csv"), encoding="utf-8"))}
        self.assertNotEqual(rows["BADUSDT"]["error"], "")


class BundleTests(Base):
    def test_round_trip(self):
        self.run_manifest()
        text = dm.bundle_text(self.out)
        self.assertIn("PASTE BUNDLE BEGIN", text)
        self.assertIn("PASTE BUNDLE END", text)
        parsed = dm.parse_bundle(text)
        self.assertEqual(set(parsed["sections"]), {"history_15m_manifest.csv", "folders_manifest.csv"})
        self.assertEqual(parsed["manifest"]["schema"], dm.SCHEMA)
        for name, body in parsed["sections"].items():
            self.assertEqual(body, open(os.path.join(self.out, name), encoding="utf-8").read())

    def test_surrounding_noise_and_crlf_are_tolerated(self):
        self.run_manifest()
        text = "some Colab output\n" + dm.bundle_text(self.out).replace("\n", "\r\n") + "\nmore output\n"
        self.assertEqual(set(dm.parse_bundle(text)["sections"]), {"history_15m_manifest.csv", "folders_manifest.csv"})

    def test_altered_section_is_rejected(self):
        self.run_manifest()
        text = dm.bundle_text(self.out).replace("AAAUSDT,2024", "AAAUSDT,2023", 1)
        with self.assertRaises(ValueError):
            dm.parse_bundle(text)

    def test_missing_or_extra_section_is_rejected(self):
        self.run_manifest()
        text = dm.bundle_text(self.out)
        head, rest = text.split("### folders_manifest.csv", 1)
        with self.assertRaises(ValueError):
            dm.parse_bundle(head + "\n=== PASTE BUNDLE END (data_manifest) ===\n")
        with self.assertRaises(ValueError):
            dm.parse_bundle("no bundle here")


class CliTests(Base):
    def test_main_prints_the_bundle_and_writes_outputs(self):
        cfg = os.path.join(self.tmp, "cfg.json")
        json.dump(self.cfg, open(cfg, "w"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dm.main(["--config", cfg, "--out", self.out, "--max-minutes", "5"])
        self.assertEqual(rc, 0)
        self.assertIn("PASTE BUNDLE BEGIN", buf.getvalue())
        self.assertTrue(os.path.exists(os.path.join(self.out, "manifest.json")))
        self.assertEqual(set(dm.parse_bundle(buf.getvalue())["sections"]), {"history_15m_manifest.csv", "folders_manifest.csv"})

    def test_incomplete_run_returns_nonzero(self):
        cfg = os.path.join(self.tmp, "cfg.json")
        json.dump(self.cfg, open(cfg, "w"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dm.main(["--config", cfg, "--out", self.out, "--max-minutes", "0"])
        self.assertNotEqual(rc, 0)

    def test_unknown_config_key_raises_from_main(self):
        cfg = os.path.join(self.tmp, "cfg.json")
        json.dump({**self.cfg, "bogus": 1}, open(cfg, "w"))
        with self.assertRaises(ValueError):
            dm.main(["--config", cfg, "--out", self.out])


class ColabCellTests(unittest.TestCase):
    def test_colab_cell_is_valid_python_and_its_config_is_accepted(self):
        import ast
        src = open(os.path.join(ROOT, "tools", "data_manifest_colab_cell.py"), encoding="utf-8").read()
        tree = ast.parse(src)
        ns = {}
        for node in tree.body:                                   # evaluate only the plain settings assignments
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.Dict, ast.List)):
                exec(compile(ast.Module([node], []), "cell", "exec"), ns)
        cfg = ns["CONFIG"]
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(cfg, history_15m_dir=d, extra_folders={})
            self.assertEqual(dm.load_config(cfg)["history_15m_dir"], d)                      # no unknown keys in the cell
        self.assertIn("history_15m", ns["CONFIG"]["history_15m_dir"])
        self.assertIn("history_1d", ns["CONFIG"]["extra_folders"])


if __name__ == "__main__":
    unittest.main()

"""tools/nb_cells.py (تحرير خلايا الدفاتر بلا إعادة تنسيق) وtools/fetch_crypto_dataset.py (دمج الملفات الشهرية) — بلا شبكة.
    python -m unittest tests.test_tools_reuse -v
"""
import glob
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fetch_crypto_dataset as fcd  # noqa: E402
import nb_cells  # noqa: E402


class NbCellsTests(unittest.TestCase):
    def test_every_root_notebook_round_trips(self):
        for nb in sorted(glob.glob(os.path.join(ROOT, "*.ipynb"))):
            self.assertEqual(nb_cells.main(["check", nb]), 0, nb)

    def test_extract_inject_insert(self):
        tmp = tempfile.mkdtemp()
        try:
            for name in ("main.ipynb", "crypto_data_pipeline_v6.ipynb"):           # indent=1 وcompact
                nb = os.path.join(tmp, name)
                shutil.copy(os.path.join(ROOT, name), nb)
                before = open(nb, encoding="utf-8").read()
                nb_cells.main(["extract", nb, tmp, "1", "2"])
                nb_cells.main(["inject", nb, tmp, "1", "2"])
                self.assertEqual(open(nb, encoding="utf-8").read(), before, name)  # لا تغيير = لا فرق بايتي
                f = os.path.join(tmp, "new.py")
                open(f, "w", encoding="utf-8").write("x = 1\n")
                nb_cells.main(["insert", nb, "0", "code", f, "--id", "testCell"])
                raw, d = nb_cells._load(nb)
                self.assertEqual("".join(d["cells"][1]["source"]), "x = 1")
                self.assertEqual(nb_cells._dump(d, raw), raw)
        finally:
            shutil.rmtree(tmp)


class FetchDatasetTests(unittest.TestCase):
    def test_merge_dedup_trim_and_skip(self):
        idx = pd.date_range("2026-09-29 23:50", periods=40, freq="5min", tz="UTC")
        month = {p: pd.DataFrame({"symbol": "AAAUSDT", "timestamp": t, "open": 1.0, "high": 2.0, "low": 0.5,
                                  "close": 1.5, "volume": 10.0, "source_rows": 5})
                 for p, t in (("data/interval_id=5m/symbol_id=AAAUSDT/a.parquet", idx[:25]),
                              ("data/interval_id=5m/symbol_id=AAAUSDT/b.parquet", idx[20:]))}
        old_list, old_get = fcd.list_files, fcd._get
        fcd.list_files = lambda clone_dir=None: list(month) + ["data/interval_id=5m/symbol_id=BBB/x.parquet"]
        fcd._get = lambda p, retries=5: month[p]
        tmp = tempfile.mkdtemp()
        try:
            out = fcd.fetch(["AAAUSDT"], "5m", tmp, end="2026-09-30 00:30", verbose=False)
            d = pd.read_parquet(out["AAAUSDT"])
            self.assertEqual(list(d.columns), fcd.COLUMNS)
            self.assertTrue(d.index.is_unique and d.index.is_monotonic_increasing)
            self.assertEqual(d.index.max(), pd.Timestamp("2026-09-30 00:30", tz="UTC"))
            fcd._get = lambda p, retries=5: (_ for _ in ()).throw(AssertionError("must skip existing"))
            fcd.fetch(["AAAUSDT"], "5m", tmp, verbose=False)
            with self.assertRaises(ValueError):
                fcd.fetch(["ZZZUSDT"], "5m", tmp, verbose=False)
        finally:
            fcd.list_files, fcd._get = old_list, old_get
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()

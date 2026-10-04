"""save_data_to_drive: نسخة واحدة (save_single_copy) ومستوى ضغط pkl.gz (pkl_compresslevel)، والافتراضي = نسختان بضغط 9.
    python -m unittest tests.test_save_options -v
"""
import contextlib
import gzip
import io
import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np

from tests import test_audit_round2 as r2


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class SaveOptionsTests(unittest.TestCase):
    def setUp(self):
        self.ns = r2._ns()
        self.cwd = os.getcwd()
        self.work = tempfile.mkdtemp()
        os.chdir(self.work)
        self.ns["mount_drive"] = lambda config=None: None              # خارج Colab: ./p/preprocessed_data
        self.base = os.path.join(self.work, "p", "preprocessed_data")
        rng = np.random.default_rng(0)
        self.data = {"X_1h": rng.normal(size=(50, 8, 3)).astype("float32"),
                     "y_close_reg": rng.normal(size=50).astype("float32"), "feature_order": ["a", "b", "c"]}

    def tearDown(self):
        os.chdir(self.cwd)
        self.ns.pop("mount_drive", None)

    def _save(self, cfg=None, **kw):
        return _quiet(self.ns["save_data_to_drive"], self.data, filename_base="d",
                      config={"project_name": "p", **(cfg or {})}, **kw)

    def _names(self):
        return sorted(os.listdir(self.base))

    def test_default_two_copies_level9(self):
        self._save(fmt="pkl.gz")
        names = self._names()
        self.assertEqual(len(names), 2)
        self.assertIn("d_latest.pkl.gz", names)
        for n in names:
            with open(os.path.join(self.base, n), "rb") as f:
                self.assertEqual(f.read(10)[8], 2, n)                   # بايت XFL في رأس gzip: 2 = أقصى ضغط (9، كالسابق)

    def test_single_copy_pkl_and_npy_dir(self):
        for cfg, kw, fmt in (({"save_single_copy": True}, {}, "pkl.gz"), ({}, {"single_copy": True}, "npy_dir")):
            shutil.rmtree(self.base, ignore_errors=True)
            path = self._save(cfg, fmt=fmt, **kw)
            want = "d_latest.pkl.gz" if fmt == "pkl.gz" else "d_latest.dataset"
            self.assertEqual(self._names(), [want], fmt)
            self.assertEqual(os.path.basename(path), want)
            back = _quiet(self.ns["load_data_from_drive"], filename_base="d", config={"project_name": "p"},
                          fmt=fmt, local_dir=False) if fmt == "npy_dir" else \
                _quiet(self.ns["load_data_from_drive"], filename_base="d", config={"project_name": "p"}, fmt=fmt)
            np.testing.assert_array_equal(np.asarray(back["X_1h"]), self.data["X_1h"])

    def test_compresslevel_1_is_loadable_and_validated(self):
        self._save({"pkl_compresslevel": 1, "save_single_copy": True}, fmt="pkl.gz")
        with gzip.open(os.path.join(self.base, "d_latest.pkl.gz"), "rb") as f:
            back = pickle.load(f)
        np.testing.assert_array_equal(back["X_1h"], self.data["X_1h"])
        with open(os.path.join(self.base, "d_latest.pkl.gz"), "rb") as f:
            self.assertEqual(f.read(10)[8], 4)                          # بايت XFL في رأس gzip: 4 = أسرع ضغط
        with self.assertRaises(ValueError):
            self._save(fmt="pkl.gz", compresslevel=12)

    def test_defaults_in_config(self):
        cfg = self.ns["DEFAULT_CONFIG"]
        self.assertEqual((cfg["save_single_copy"], cfg["pkl_compresslevel"]), (False, 9))


if __name__ == "__main__":
    unittest.main()

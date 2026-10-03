"""بناء مجموعة البيانات المدعوم بالقرص (CONFIG['disk_backed']):  python -m unittest tests.test_disk_backed -v

ما يُثبَت (بيانات تركيبية بلا Drive، مسار 1h+4h الحقيقي HOURLY_4H_OVERRIDES):
  * المصفوفات مطابقة بايتاً (np.array_equal على كل مفتاح، ونفس dtype وترتيب المفاتيح والبيانات الوصفية) بين disk_backed=True
    وFalse، بخيط واحد وبعدة خيوط، وأن X فعلاً memmap على القرص (بخفض حدّ SMALL_ARRAY_BYTES إلى 0 كي لا تُحمَّل للرام).
  * الحفظ بصيغة المجلد (save_dataset_dir/save_data_to_drive) والتحميل memmap (load_dataset_dir/load_data_from_drive) يُعيدان
    نفس القاموس؛ و.pkl.gz القديمة ما زالت تعمل ومختارة بـ fmt؛ واكتشاف الأحدث عند وجود الصيغتين.
  * split_data على بيانات memmap = split_data على بيانات الرام، وقسم train الكبير memmap على القرص لا نسخة رام.
  * خيار حذف scratch بعد الحفظ، وأن scratch يُفرَّغ عند كل بناء.
"""
import ast
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

import _nbload  # noqa: E402
from tests.test_multi_tf import ohlcv  # noqa: E402


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class DiskBackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _nbload.load_pipeline()
        cls.tmp = tempfile.mkdtemp(prefix="disk_backed_test_")
        cls._n_builds = 0
        cls.ns["SMALL_ARRAY_BYTES"] = 0          # كل مصفوفة غير فارغة تبقى memmap (لا تُنسَخ للرام) — يُختبَر المسار الحقيقي
        cls.ram = cls._build(False, workers=1)
        cls.disk = cls._build(True, workers=1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _build(cls, disk, workers=1, coins=8, days=50, **extra):
        ns = cls.ns
        cls._n_builds += 1                       # scratch منفصل لكل بناء: إعادة البناء تُفرِّغ مجلد البصمة نفسها (تُسقط ملفات البناء السابق)
        ns["reset_config"]()
        _quiet(ns["apply_hourly_preset"], ns["HOURLY_4H_OVERRIDES"])
        ns["update_config"]({"phase2_data": {"use_intraday_15m": False, "use_futures_metrics": False},
                             "funding_rate": {"enabled": False}, "open_interest": {"enabled": False},
                             "disk_backed": disk, "scratch_dir": os.path.join(cls.tmp, f"scratch{cls._n_builds}"), **extra})
        _quiet(ns["refresh_features"])
        starts = ["2025-01-01", "2025-01-01 03:00", "2025-01-01 05:00", "2025-01-01 11:00"]
        loader = lambda fid, name: ohlcv(days, 40 + int(name[1]), starts[int(name[1]) % 4])   # noqa: E731
        return _quiet(ns["build_dataset"], [{"name": f"C{i}USDT"} for i in range(coins)], load_asset_fn=loader,
                      resample_fn=ns["make_resample_fn"](ns["CONFIG"]), max_workers=workers, config=ns["CONFIG"])

    def assertDatasetsEqual(self, a, b):
        self.assertEqual(list(a), list(b), "ترتيب/أسماء المفاتيح")
        n_arrays = 0
        for k in a:
            if isinstance(a[k], np.ndarray):
                n_arrays += 1
                self.assertIsInstance(b[k], np.ndarray, k)
                self.assertEqual((a[k].dtype, a[k].shape), (b[k].dtype, b[k].shape), k)
                self.assertTrue(np.array_equal(a[k], b[k]), f"المفتاح {k} مختلف")
            else:
                self.assertEqual(repr(a[k]), repr(b[k]), k)
        self.assertGreaterEqual(n_arrays, 10)

    # ── التطابق ──
    def test_arrays_identical_single_worker(self):
        self.assertDatasetsEqual(self.ram, self.disk)
        self.assertEqual(self.disk["X_1h"].dtype, np.float16)
        self.assertGreater(len(self.disk["base_params"]), 500)

    def test_arrays_identical_multi_worker(self):
        self.assertDatasetsEqual(self.ram, self._build(True, workers=3))

    def test_x_is_memmap_on_disk_and_read_only(self):
        for tf in ("1h", "4h"):
            X = self.disk[f"X_{tf}"]
            self.assertIsInstance(X, np.memmap)
            self.assertTrue(os.path.exists(X.filename) and str(X.filename).startswith(self.tmp))
            with self.assertRaises(ValueError):
                X[0, 0, 0] = 1                     # للقراءة فقط
        self.assertNotIsInstance(self.ram["X_1h"], np.memmap)

    def test_parts_removed_after_merge(self):
        run_dir = os.path.dirname(os.path.dirname(str(self.disk["X_1h"].filename)))
        self.assertFalse(os.path.exists(os.path.join(run_dir, "parts")), "أجزاء العملات بقيت بعد الدمج")
        self.assertEqual(sorted(os.listdir(os.path.join(run_dir, "merged"))),
                         sorted(f"{k}.npy" for k, v in self.disk.items() if isinstance(v, np.memmap)))

    def test_scratch_run_dir_is_wiped_at_start(self):
        ns = self.ns
        root = os.path.join(self.tmp, "wipe")
        d = ns["_scratch_run_dir"]({"scratch_dir": root}, "abc123")
        self.assertEqual(os.path.basename(d), "abc123")
        stale = os.path.join(d, "stale.npy")
        open(stale, "w").close()
        d2 = ns["_scratch_run_dir"]({"scratch_dir": root}, "abc123")
        self.assertEqual(d, d2)
        self.assertFalse(os.path.exists(stale))

    def test_default_flags(self):
        ns = self.ns
        ns["reset_config"]()
        self.assertFalse(ns["CONFIG"]["disk_backed"])
        self.assertIsNone(ns["CONFIG"]["scratch_dir"])
        self.assertFalse(ns["CONFIG"]["scratch_cleanup_after_save"])
        for name in ("HOURLY_PRESET", "HOURLY_W32_S8_OVERRIDES", "HOURLY_4H_OVERRIDES"):
            self.assertTrue(ns[name]["disk_backed"], name)
        _quiet(ns["apply_hourly_preset"], ns["HOURLY_4H_OVERRIDES"])
        self.assertTrue(ns["CONFIG"]["disk_backed"])
        ns["reset_config"]()

    # ── الحفظ والتحميل ──
    def test_dir_roundtrip_loads_memmap_identical(self):
        d = os.path.join(self.tmp, "rt.dataset")
        _quiet(self.ns["save_dataset_dir"], self.disk, d)
        self.assertEqual(sorted(f for f in os.listdir(d) if f.endswith(".npy")),
                         sorted(f"{k}.npy" for k, v in self.disk.items() if isinstance(v, np.ndarray)))
        back = _quiet(self.ns["load_dataset_dir"], d)
        self.assertDatasetsEqual(self.ram, back)
        self.assertIsInstance(back["X_1h"], np.memmap)
        self.assertEqual(back["X_1h"].mode, "r")
        whole = _quiet(self.ns["load_dataset_dir"], d, mmap=False)
        self.assertDatasetsEqual(self.ram, whole)
        self.assertNotIsInstance(whole["X_1h"], np.memmap)

    def test_load_copies_to_local_dir_and_reuses_matching_copy(self):
        d = os.path.join(self.tmp, "drive_like.dataset")
        _quiet(self.ns["save_dataset_dir"], self.disk, d)
        os.makedirs(os.path.join(d, "subdir"))                         # مجلد فرعي غريب لا يُنسَخ ولا يكسر النسخ
        local = os.path.join(self.tmp, "local_copy")
        back = _quiet(self.ns["load_dataset_dir"], d, local_dir=local)
        self.assertDatasetsEqual(self.ram, back)
        self.assertEqual(os.path.dirname(str(back["X_1h"].filename)), local)
        self.assertFalse(os.path.exists(os.path.join(local, "subdir")))
        mtime = os.stat(os.path.join(local, "X_1h.npy")).st_mtime_ns
        _quiet(self.ns["load_dataset_dir"], d, local_dir=local)
        self.assertEqual(os.stat(os.path.join(local, "X_1h.npy")).st_mtime_ns, mtime, "النسخة المطابقة يجب أن تُعاد استخدامها")
        self.assertDatasetsEqual(self.ram, _quiet(self.ns["load_dataset_dir"], d, local_dir=False))

    def test_save_data_to_drive_formats_and_autodetect(self):
        ns = self.ns
        cwd = os.getcwd()
        work = tempfile.mkdtemp(dir=self.tmp)
        os.chdir(work)
        try:
            cfg = {"project_name": "proj", "save_format": "auto"}
            kw = dict(filename_base="ds", config=cfg)
            ns["mount_drive"] = lambda config=None: None                  # خارج Colab: مسار نسبي ./proj/preprocessed_data
            # auto + memmap ← مجلد
            _quiet(ns["save_data_to_drive"], self.disk, **kw)
            base = os.path.join(work, "proj", "preprocessed_data")
            self.assertTrue(os.path.isdir(os.path.join(base, "ds_latest.dataset")))
            self.assertFalse(os.path.exists(os.path.join(base, "ds_latest.pkl.gz")))
            self.assertDatasetsEqual(self.ram, _quiet(ns["load_data_from_drive"], **kw))
            # auto + ndarray عادية ← pkl.gz القديمة؛ وfmt='pkl.gz' صريحاً يعمل أيضاً لمجموعة memmap
            _quiet(ns["save_data_to_drive"], self.ram, filename_base="old", config=cfg)
            self.assertTrue(os.path.exists(os.path.join(base, "old_latest.pkl.gz")))
            self.assertDatasetsEqual(self.ram, _quiet(ns["load_data_from_drive"], filename_base="old", config=cfg))
            _quiet(ns["save_data_to_drive"], self.disk, fmt="pkl.gz", **kw)
            self.assertTrue(os.path.exists(os.path.join(base, "ds_latest.pkl.gz")))
            pk = _quiet(ns["load_data_from_drive"], fmt="pkl.gz", **kw)
            self.assertDatasetsEqual(self.ram, pk)
            self.assertNotIsInstance(pk["X_1h"], np.memmap)
            nd = _quiet(ns["load_data_from_drive"], fmt="npy_dir", **kw)
            self.assertIsInstance(nd["X_1h"], np.memmap)
            # الصيغتان معاً ← الأحدث: المجلد أُعيدت كتابته بعد pkl فيُختار (memmap)
            _quiet(ns["save_data_to_drive"], self.disk, fmt="npy_dir", **kw)
            self.assertIsInstance(_quiet(ns["load_data_from_drive"], **kw)["X_1h"], np.memmap)
            with self.assertRaises(FileNotFoundError):
                _quiet(ns["load_data_from_drive"], filename_base="missing", config=cfg)
        finally:
            os.chdir(cwd)
            ns.pop("mount_drive", None)

    def test_scratch_cleanup_after_save_option(self):
        ns = self.ns
        ds = self._build(True, workers=1, coins=3, scratch_cleanup_after_save=True)
        X = ds["X_1h"]
        run_dir = os.path.dirname(os.path.dirname(str(X.filename)))
        self.assertTrue(os.path.isdir(run_dir))
        ref = np.array(X)
        cwd = os.getcwd()
        work = tempfile.mkdtemp(dir=self.tmp)
        os.chdir(work)
        try:
            ns["mount_drive"] = lambda config=None: None
            _quiet(ns["save_data_to_drive"], ds, filename_base="c", config=dict(ns["CONFIG"], project_name="p"))
        finally:
            os.chdir(cwd)
            ns.pop("mount_drive", None)
        self.assertFalse(os.path.exists(run_dir), "scratch لم يُحذف")
        self.assertTrue(np.array_equal(np.array(ds["X_1h"]), ref), "المصفوفة المفتوحة يجب أن تبقى قابلة للقراءة")
        back = _quiet(ns["load_dataset_dir"], os.path.join(work, "p", "preprocessed_data", "c_latest.dataset"))
        self.assertTrue(np.array_equal(back["X_1h"], ref))

    # ── التقسيم ──
    def test_split_data_on_memmap_equals_ram(self):
        ns = self.ns
        ns["reset_config"]()
        cfg = ns["CONFIG"]
        _quiet(ns["apply_hourly_preset"], ns["HOURLY_4H_OVERRIDES"])
        cfg.update({"split_dates": {"train_end": "2025-01-25", "val_end": "2025-02-05"}, "holdout_start": None,
                    "keep_asset_test_separate": False, "scratch_dir": os.path.join(self.tmp, "split_scratch")})
        r = _quiet(ns["split_data"], self.ram, config=cfg)
        d = _quiet(ns["split_data"], self.disk, config=cfg)
        for sr, sd in zip(r, d):
            self.assertEqual(sorted(sr), sorted(sd))
            for k in sr:
                if k == "y":
                    for h in sr["y"]:
                        self.assertTrue(np.array_equal(sr["y"][h], sd["y"][h]), h)
                else:
                    self.assertTrue(np.array_equal(sr[k], sd[k]), k)
        self.assertIsInstance(d[0]["X_1h"], np.memmap, "train يجب أن يبقى على القرص لا نسخة رام")
        self.assertEqual(os.path.dirname(str(d[0]["X_1h"].filename)), os.path.join(self.tmp, "split_scratch", "splits"),
                         "ملفات الأقسام تُكتب تحت scratch_dir لا بجوار مصدر البيانات")
        ns["reset_config"]()

    def test_take_rows_small_result_stays_in_ram(self):
        ns = self.ns
        old = ns["SMALL_ARRAY_BYTES"]
        ns["SMALL_ARRAY_BYTES"] = 10 ** 12
        try:
            mask = np.zeros(len(self.disk["X_1h"]), dtype=bool)
            mask[::3] = True
            out = ns["_take_rows"](self.disk["X_1h"], mask)
            self.assertNotIsInstance(out, np.memmap)
            self.assertTrue(np.array_equal(out, self.ram["X_1h"][mask]))
        finally:
            ns["SMALL_ARRAY_BYTES"] = old

    def test_merge_spilled_chunks_and_dtype(self):
        ns = self.ns
        d = tempfile.mkdtemp(dir=self.tmp)
        rng = np.random.default_rng(0)
        parts_np = [rng.normal(size=(n, 5, 3)).astype("float16") for n in (7, 1, 12)]
        old = ns["_COPY_CHUNK_BYTES"]
        ns["_COPY_CHUNK_BYTES"] = 64                   # دفعات صغيرة جداً ← حدود الدفعات تُختبَر
        try:
            sp = []
            for i, a in enumerate(parts_np):
                np.save(os.path.join(d, f"p{i}.npy"), a)
                sp.append(ns["_SpilledArray"](os.path.join(d, f"p{i}.npy"), a.shape, a.dtype))
            out = ns["_merge_spilled"](sp, os.path.join(d, "out.npy"), ram_limit=0)
        finally:
            ns["_COPY_CHUNK_BYTES"] = old
        self.assertIsInstance(out, np.memmap)
        self.assertTrue(np.array_equal(out, np.concatenate(parts_np)))
        self.assertFalse(any(os.path.exists(os.path.join(d, f"p{i}.npy")) for i in range(3)), "الأجزاء لم تُحذف")


class MainLazyBatchTests(unittest.TestCase):
    """دوال تكوين الدفعات في main.ipynb (القسم ٥) مع X memmap: نفس الدفعات والقيم، float32 عند التكوين."""

    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        with open(os.path.join(ROOT, "main.ipynb"), encoding="utf-8") as f:
            cells = json.load(f)["cells"]
        src = next("".join(c["source"]) for c in cells
                   if c["cell_type"] == "code" and "def make_shuffled_dataset" in "".join(c["source"]))
        tree = ast.parse(src)
        want = {"make_shuffled_dataset", "_to_float32_inputs", "make_eval_dataset"}
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
        assert {n.name for n in body} == want, "دوال الدفعات غير موجودة في main"
        cls.ns = {"np": np, "tf": tf}
        exec(compile(ast.Module(body=body, type_ignores=[]), "main#batches", "exec"), cls.ns)
        cls.tmp = tempfile.mkdtemp(prefix="lazy_batches_")
        rng = np.random.default_rng(3)
        cls.X_ram = {"1h": rng.normal(size=(70, 6, 4)).astype("float16"), "4h": rng.normal(size=(70, 6, 4)).astype("float16")}
        cls.y = {"y_a": rng.normal(size=70).astype("float32")}
        cls.X_mm = {}
        for k, a in cls.X_ram.items():
            np.save(os.path.join(cls.tmp, f"{k}.npy"), a)
            cls.X_mm[k] = np.load(os.path.join(cls.tmp, f"{k}.npy"), mmap_mode="r")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @staticmethod
    def _collect(ds):
        return [(x, y) for x, y in ds.as_numpy_iterator()]

    def test_eval_dataset_memmap_equals_ram(self):
        ev = self.ns["make_eval_dataset"]
        a, b = self._collect(ev(self.X_ram, self.y, 16)), self._collect(ev(self.X_mm, self.y, 16))
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), 4)                                   # 70 // 16، drop_remainder
        for (xa, ya), (xb, yb) in zip(a, b):
            for k in xa:
                self.assertEqual(xb[k].dtype, np.float32)
                self.assertTrue(np.array_equal(xa[k], xb[k]))
            self.assertTrue(np.array_equal(ya["y_a"], yb["y_a"]))

    def test_eval_dataset_single_array_and_order(self):
        ev = self.ns["make_eval_dataset"]
        out = self._collect(ev(self.X_mm["1h"], self.y, 10))
        got = np.concatenate([x for x, _ in out])
        self.assertTrue(np.array_equal(got, self.X_ram["1h"].astype("float32")))   # بالترتيب، بلا خلط

    def test_shuffled_dataset_memmap_equals_ram_same_seed(self):
        mk = self.ns["make_shuffled_dataset"]
        a = self._collect(mk(self.X_ram, self.y, 16, seed=5))
        b = self._collect(mk(self.X_mm, self.y, 16, seed=5))
        for (xa, ya), (xb, yb) in zip(a, b):
            for k in xa:
                self.assertTrue(np.array_equal(xa[k], xb[k]))
            self.assertTrue(np.array_equal(ya["y_a"], yb["y_a"]))


if __name__ == "__main__":
    unittest.main()

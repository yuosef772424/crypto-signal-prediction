"""tools/export_torch_artifact.py: sealed coins and windows whose label bar reaches the time seal are removed, the manifest records
the contract fields (source commit, coins, sealed list, max bar time, file hashes), and a dirty tree / unknown per-window meta /
non-tiling asset_bounds are refused. Small fixtures in a temporary folder (no Drive, no network).
    python -m unittest tests.test_export_torch_artifact -v
"""
import gzip
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)
import export_torch_artifact as eta  # noqa: E402
import holdout_guard as hg  # noqa: E402
from tests.test_holdout_guard import write_lock  # noqa: E402

SHA = "c" * 40
H = 3600 * 10**9                                           # one 1h bar in ns
SEAL_NS = int(pd.Timestamp("2026-10-01", tz="UTC").value)


KEY_ORDER = ["last_candles", "asset_bounds", "base_tf", "forecast_horizon", "timeframes", "X_1h", "y_high_reg"]


def build_data(per_asset, *, extra_meta=None, bounds_override=None):
    """-> (arrays, meta). per_asset = [(name, [last-bar timestamp ns, ...]), ...]; 1h base, horizon 1."""
    rows = [(name, t) for name, ts in per_asset for t in ts]
    n = len(rows)
    lc = np.zeros((n, 7), dtype="float64")
    lc[:, eta.LAST_TS_COL] = [t for _, t in rows]
    bounds, pos = [], 0
    for name, ts in per_asset:
        bounds.append({"name": name, "start": pos, "end": pos + len(ts)})
        pos += len(ts)
    arrays = {"last_candles": lc,
              "X_1h": np.arange(n * 2 * 3, dtype="float32").reshape(n, 2, 3),
              "y_high_reg": np.arange(n, dtype="float32")}
    meta = {"asset_bounds": bounds_override or bounds, "base_tf": "1h", "forecast_horizon": 1, "timeframes": ["1h"]}
    meta.update(extra_meta or {})
    return arrays, meta


def write_dataset(path, per_asset, *, extra_meta=None, bounds_override=None):
    """npy_dir source (what data/storage.py::save_dataset_dir writes)."""
    arrays, meta = build_data(per_asset, extra_meta=extra_meta, bounds_override=bounds_override)
    os.makedirs(path)
    for k, a in arrays.items():
        np.save(os.path.join(path, f"{k}.npy"), a)
    with open(os.path.join(path, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    man = {"format": "npy_dir", "version": 1, "id": "srcid", "created": "2026-10-20T00:00:00",
           "key_order": KEY_ORDER,
           "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(man, f)
    return arrays


def write_pkl_gz(path, per_asset, **kw):
    """old-format source (what data/storage.py::save_data_to_drive writes): one gzip'd pickle of the whole dict."""
    arrays, meta = build_data(per_asset, **kw)
    data = {k: arrays[k] if k in arrays else meta[k] for k in KEY_ORDER}
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return arrays


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.lock_dir = os.path.join(self.tmp, "lock")
        write_lock(self.lock_dir, sealed={"SEALAUSDT": "H1", "SEALBUSDT": "H2"})
        self.lock = hg.load_lock(self.lock_dir)
        self.src = os.path.join(self.tmp, "src.dataset")
        self.out = os.path.join(self.tmp, "out.dataset")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_export(self, **kw):
        return eta.export(self.src, self.out, lock=self.lock, source=(SHA, False), **kw)


class TestExport(Base):
    def setUp(self):
        super().setUp()
        t0 = SEAL_NS - 10 * H
        self.arrays = write_dataset(self.src, [("AAAUSDT", [t0, t0 + H, t0 + 2 * H]),
                                               ("SEALAUSDT", [t0, t0 + H]),
                                               ("BBBUSDT", [t0 + 5 * H, SEAL_NS - 2 * H, SEAL_NS - H, SEAL_NS])])

    def test_sealed_coin_and_sealed_period_windows_are_removed(self):
        m = self.run_export()["export"]
        self.assertEqual(m["coins"], ["AAAUSDT", "BBBUSDT"])
        self.assertEqual(m["sealed_found"], ["SEALAUSDT"])
        self.assertEqual(m["sealed_excluded"], ["SEALAUSDT", "SEALBUSDT"])    # the whole lock list, not only the found ones
        self.assertEqual((m["rows_source"], m["dropped_sealed_rows"], m["dropped_time_rows"], m["rows"]), (9, 2, 2, 5))
        # BBB bars at seal-1h and seal: label bar at seal / after it -> dropped; seal-2h has its label at seal-1h -> kept
        _, arr, meta = eta.read_npy_dir(self.out)
        self.assertEqual(arr["X_1h"].shape[0], 5)
        self.assertEqual([(b["name"], b["start"], b["end"]) for b in meta["asset_bounds"]], [("AAAUSDT", 0, 3), ("BBBUSDT", 3, 5)])

    def test_rows_are_the_original_rows_in_order(self):
        self.run_export()
        _, arr, _ = eta.read_npy_dir(self.out)
        keep = [0, 1, 2, 5, 6]
        np.testing.assert_array_equal(arr["y_high_reg"], self.arrays["y_high_reg"][keep])
        np.testing.assert_array_equal(arr["X_1h"], self.arrays["X_1h"][keep])

    def test_max_bar_time_is_the_latest_label_bar_and_below_the_seal(self):
        m = self.run_export()["export"]
        self.assertEqual(pd.Timestamp(m["max_bar_time"]), pd.Timestamp(SEAL_NS - H, tz="UTC"))
        self.assertEqual(pd.Timestamp(m["max_window_time"]), pd.Timestamp(SEAL_NS - 2 * H, tz="UTC"))
        self.assertLess(pd.Timestamp(m["max_bar_time"]), hg.TIME_SEAL)
        self.assertEqual(pd.Timestamp(m["time_seal"]), hg.TIME_SEAL)

    def test_manifest_fields_and_hashes(self):
        man = self.run_export()
        e = man["export"]
        self.assertEqual((e["schema"], e["source_commit"], e["source_dataset_id"]), (eta.SCHEMA, SHA, "srcid"))
        self.assertEqual(e["lock_sha256"], eta._sha256_file(os.path.join(self.lock_dir, "holdout.json")))
        for name, digest in e["files"].items():
            self.assertEqual(eta._sha256_file(os.path.join(self.out, name)), digest)
        on_disk = json.load(open(os.path.join(self.out, "manifest.json")))
        self.assertEqual(on_disk, man)
        self.assertFalse(os.path.exists(self.out + ".tmp"))

    def test_output_is_still_a_valid_npy_dir(self):
        man = self.run_export()
        got, arr, meta = eta.read_npy_dir(self.out)                         # shapes and dtypes match the manifest
        self.assertEqual((got["format"], got["version"], got["key_order"]), ("npy_dir", 1, man["key_order"]))
        self.assertEqual(meta["base_tf"], "1h")

    def test_export_is_deterministic_apart_from_id_and_created(self):
        a = self.run_export()["export"]
        b = self.run_export()["export"]
        self.assertEqual(a, b)


class TestRefusals(Base):
    def test_dirty_tree_is_refused(self):
        write_dataset(self.src, [("AAAUSDT", [SEAL_NS - 5 * H])])
        with self.assertRaisesRegex(eta.ExportError, "uncommitted"):
            eta.export(self.src, self.out, lock=self.lock, source=(SHA, True))

    def test_missing_lock_fails_closed(self):
        write_dataset(self.src, [("AAAUSDT", [SEAL_NS - 5 * H])])
        with self.assertRaises(hg.LockError):
            eta.export(self.src, self.out, lock=hg.load_lock(os.path.join(self.tmp, "nolock")), source=(SHA, False))

    def test_everything_sealed_is_refused(self):
        write_dataset(self.src, [("SEALAUSDT", [SEAL_NS - 5 * H])])
        with self.assertRaisesRegex(eta.ExportError, "no window is left"):
            self.run_export()

    def test_per_window_meta_list_is_refused(self):
        write_dataset(self.src, [("AAAUSDT", [SEAL_NS - 5 * H, SEAL_NS - 4 * H])], extra_meta={"asset_of_row": ["a", "a"]})
        with self.assertRaisesRegex(eta.ExportError, "per-window"):
            self.run_export()

    def test_bounds_that_do_not_tile_are_refused(self):
        write_dataset(self.src, [("AAAUSDT", [SEAL_NS - 5 * H, SEAL_NS - 4 * H])],
                      bounds_override=[{"name": "AAAUSDT", "start": 0, "end": 1}])
        with self.assertRaisesRegex(eta.ExportError, "asset_bounds"):
            self.run_export()

    def test_missing_manifest_is_refused(self):
        os.makedirs(self.src)
        with self.assertRaisesRegex(eta.ExportError, "manifest.json"):
            self.run_export()

    def test_unreadable_base_tf_is_refused(self):
        with self.assertRaises(eta.ExportError):
            eta.bar_delta("weekly-ish")
        with self.assertRaises(eta.ExportError):
            eta.bar_delta(None)

    def test_bar_delta_goes_through_normalize_tf(self):
        import warnings
        from pandas.errors import Pandas4Warning
        with warnings.catch_warnings():
            warnings.simplefilter("error", Pandas4Warning)                 # lowercase '1d' would warn without normalize_tf
            self.assertEqual(eta.bar_delta("1d"), pd.Timedelta(days=1))
            self.assertEqual(eta.bar_delta("1h"), pd.Timedelta(hours=1))
            self.assertEqual(eta.bar_delta("15min"), pd.Timedelta(minutes=15))

    def test_sealed_match_ignores_case_and_usdt_suffix(self):
        write_dataset(self.src, [("sealb", [SEAL_NS - 5 * H]), ("AAAUSDT", [SEAL_NS - 5 * H])])
        self.assertEqual(self.run_export()["export"]["coins"], ["AAAUSDT"])


class TestPklGz(Base):
    """A `.pkl.gz` source gives the same artifact as the npy_dir with the same content."""

    def setUp(self):
        super().setUp()
        t0 = SEAL_NS - 10 * H
        self.per_asset = [("AAAUSDT", [t0, t0 + H, t0 + 2 * H]), ("SEALAUSDT", [t0, t0 + H]),
                          ("BBBUSDT", [t0 + 5 * H, SEAL_NS - 2 * H, SEAL_NS - H, SEAL_NS])]
        self.pkl = os.path.join(self.tmp, "preprocessing_output_latest.pkl.gz")
        self.out_pkl = os.path.join(self.tmp, "out_pkl.dataset")
        write_dataset(self.src, self.per_asset)
        write_pkl_gz(self.pkl, self.per_asset)

    def export_pkl(self):
        return eta.export(self.pkl, self.out_pkl, lock=self.lock, source=(SHA, False))

    def test_pkl_gz_gives_the_same_file_fingerprints_as_npy_dir(self):
        a, b = self.run_export()["export"], self.export_pkl()["export"]
        self.assertEqual(a["files"], b["files"])
        self.assertEqual(a["data_sha256"], b["data_sha256"])
        for k in ("coins", "sealed_found", "sealed_excluded", "max_bar_time", "max_window_time", "rows", "rows_source",
                  "dropped_sealed_rows", "dropped_time_rows", "lock_sha256", "time_seal", "source_commit"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual((a["source_format"], b["source_format"]), ("npy_dir", "pkl.gz"))

    def test_pkl_gz_output_is_a_valid_npy_dir_with_the_same_manifest_body(self):
        a = self.run_export()
        b = self.export_pkl()
        got, arr, meta = eta.read_npy_dir(self.out_pkl)
        self.assertEqual((got["format"], got["key_order"], got["arrays"]), ("npy_dir", a["key_order"], a["arrays"]))
        self.assertEqual(b["arrays"], a["arrays"])
        np.testing.assert_array_equal(arr["X_1h"], eta.read_npy_dir(self.out)[1]["X_1h"])
        self.assertEqual(meta["asset_bounds"], eta.read_npy_dir(self.out)[2]["asset_bounds"])

    def test_read_pkl_gz_splits_like_save_dataset_dir(self):
        man, arr, meta = eta.read_pkl_gz(self.pkl)
        self.assertEqual(sorted(arr), ["X_1h", "last_candles", "y_high_reg"])
        self.assertEqual(sorted(meta), ["asset_bounds", "base_tf", "forecast_horizon", "timeframes"])
        self.assertEqual(man["key_order"], KEY_ORDER)

    def test_non_dict_pickle_is_refused(self):
        with gzip.open(self.pkl, "wb") as f:
            pickle.dump([1, 2, 3], f)
        with self.assertRaisesRegex(eta.ExportError, "not a dataset dict"):
            self.export_pkl()

    def test_other_file_types_and_missing_paths_are_refused(self):
        other = os.path.join(self.tmp, "data.pkl")
        shutil.copy(self.pkl, other)
        for path in (other, os.path.join(self.tmp, "nothing.pkl.gz")):
            with self.assertRaisesRegex(eta.ExportError, r"not an npy_dir directory and not a \.pkl\.gz file"):
                eta.export(path, self.out_pkl, lock=self.lock, source=(SHA, False))

    def test_cli_accepts_pkl_gz(self):
        # the CLI path with the real lock needs the repo's lock and a clean tree; here only the refusal path is exercised
        self.assertEqual(eta.main(["--dataset", os.path.join(self.tmp, "nothing.pkl.gz"), "--out", self.out_pkl]), 1)


class TestCli(Base):
    def test_cli_refusal_exits_1(self):
        self.assertEqual(eta.main(["--dataset", os.path.join(self.tmp, "none"), "--out", self.out]), 1)


if __name__ == "__main__":
    unittest.main()

"""PredictionTable: one schema, one writer, one validator for model predictions (audit C5 / PR 11).

    python -m pytest -q tests/test_prediction_table.py

Groups:
  * ValidatorTests    every way a table can break its contract raises (missing / unknown column, NaN key, duplicate key, bad role, label
                      available before the prediction, a non-train prediction made before train_end); a good table passes;
  * BuildWriteTests   build_table derives label_available_at = timestamp + horizon bars, write/read round-trips and re-validates;
  * EvaluationWiring  test_all_assets_v4(prediction_table=spec) writes the table; WITHOUT the option the result is exactly the old one
                      (no new key, no file) and the numbers equal the run with the option on.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core import prediction_schema as schema  # noqa: E402
from core.schema import TS_COL  # noqa: E402
from stats import prediction_table as pt  # noqa: E402

HOUR_NS = 3_600_000_000_000
T0 = int(pd.Timestamp("2025-02-10", tz="UTC").value)


def good_table(role="test", n=6, **over):
    ts = T0 + HOUR_NS * np.arange(n)
    kw = dict(asset=np.array(["A", "A", "A", "B", "B", "B"][:n]), timestamp=np.r_[ts[:3], ts[:3]][:n].astype("float64"),
              outputs={"high": np.arange(n, dtype="float64"), "close_p_up": np.linspace(0.1, 0.9, n)}, split_role=role,
              train_end=pd.Timestamp("2025-02-09", tz="UTC"), fold=0, identity_hash="abc123", horizon_bars=4, timeframe="1h")
    kw.update(over)
    return pt.build_table(**kw)


class ValidatorTests(unittest.TestCase):
    def test_good_table_passes_and_has_the_documented_columns(self):
        df = good_table()
        self.assertEqual(list(df.columns[:len(schema.REQUIRED_COLUMNS)]), list(schema.REQUIRED_COLUMNS))
        self.assertIs(pt.validate_table(df), df)
        self.assertEqual({c for c in df.columns if c.startswith(schema.HEAD_OUTPUT_PREFIX)}, {"out_high", "out_close_p_up"})

    def test_label_available_at_is_timestamp_plus_horizon_bars(self):
        df = good_table(horizon_bars=4, timeframe="1h")
        self.assertTrue(((df["label_available_at"] - df["timestamp"]) == pd.Timedelta("4h")).all())
        d4 = good_table(horizon_bars=2, timeframe="4h")
        self.assertTrue(((d4["label_available_at"] - d4["timestamp"]) == pd.Timedelta("8h")).all())
        self.assertTrue((df["prediction_made_at"] == df["timestamp"]).all())          # historical default: no back-dating

    def test_missing_and_unknown_columns_raise(self):
        df = good_table()
        for c in schema.REQUIRED_COLUMNS:
            with self.assertRaisesRegex(pt.PredictionTableError, "missing required"):
                pt.validate_table(df.drop(columns=[c]))
        with self.assertRaisesRegex(pt.PredictionTableError, "unknown column"):
            pt.validate_table(df.assign(extra=1.0))
        with self.assertRaisesRegex(pt.PredictionTableError, "unknown column"):
            pt.validate_table(df.assign(out_=1.0))                                     # the prefix alone is not an output
        with self.assertRaisesRegex(pt.PredictionTableError, "unknown column"):
            pt.validate_table(df, head_outputs=["high"])                               # out_close_p_up is not a declared output
        pt.validate_table(df, head_outputs=["high", "close_p_up"])

    def test_nan_keys_raise(self):
        df = good_table()
        for col, bad in (("asset", None), ("timestamp", pd.NaT), ("fold", np.nan), ("prediction_made_at", pd.NaT),
                         ("train_end", pd.NaT), ("split_role", None), ("identity_hash", None)):
            broken = df.copy()
            broken[col] = broken[col].astype(object)
            broken.loc[broken.index[0], col] = bad
            with self.assertRaises(pt.PredictionTableError, msg=col):
                pt.validate_table(broken)
        with self.assertRaisesRegex(pt.PredictionTableError, "non-finite"):
            good_table(timestamp=np.array([np.nan] + [float(T0)] * 5))

    def test_duplicate_asset_timestamp_fold_raises_but_another_fold_is_fine(self):
        df = good_table()
        dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(pt.PredictionTableError, "share a key"):
            pt.validate_table(dup)
        other_fold = df.iloc[[0]].copy()
        other_fold["fold"] = 1
        pt.validate_table(pd.concat([df, other_fold], ignore_index=True))

    def test_unknown_role_and_wrong_types_raise(self):
        df = good_table()
        with self.assertRaisesRegex(pt.PredictionTableError, "unknown split_role"):
            pt.validate_table(df.assign(split_role="validation"))
        with self.assertRaisesRegex(pt.PredictionTableError, "fold"):
            pt.validate_table(df.assign(fold=df["fold"].astype("float64")))
        with self.assertRaisesRegex(pt.PredictionTableError, "not a datetime"):
            pt.validate_table(df.assign(timestamp=df["timestamp"].astype("int64")))
        with self.assertRaisesRegex(pt.PredictionTableError, "not UTC"):
            pt.validate_table(df.assign(timestamp=df["timestamp"].dt.tz_convert("Asia/Riyadh")))
        with self.assertRaises(pt.PredictionTableError):
            pt.validate_table(df.iloc[0:0])

    def test_label_must_become_available_after_the_prediction(self):
        df = good_table()
        with self.assertRaisesRegex(pt.PredictionTableError, "label_available_at <= prediction_made_at"):
            pt.validate_table(df.assign(label_available_at=df["prediction_made_at"]))
        with self.assertRaises(pt.PredictionTableError):
            pt.validate_table(df.assign(label_available_at=df["prediction_made_at"] - pd.Timedelta("1h")))
        with self.assertRaises(pt.PredictionTableError):
            good_table(horizon_bars=0)

    def test_non_train_prediction_before_train_end_is_a_leak(self):
        early = pd.Timestamp("2025-02-10", tz="UTC") + pd.Timedelta("1h")
        for role in schema.NON_TRAIN_ROLES:
            with self.assertRaisesRegex(pt.PredictionTableError, "before the producing model's train_end"):
                good_table(role=role, train_end=early)
        good_table(role="train", train_end=early)                                      # in-sample rows may be earlier than train_end
        good_table(role="val", train_end=pd.Timestamp("2025-02-10", tz="UTC"))         # made exactly at train_end: not before it

    def test_head_output_must_be_one_number_per_row(self):
        with self.assertRaisesRegex(pt.PredictionTableError, "one"):
            good_table(outputs={"high": np.zeros((6, 2))})
        with self.assertRaisesRegex(pt.PredictionTableError, "rows"):
            good_table(outputs={"high": np.zeros(5)})
        self.assertEqual(list(good_table(outputs={"high": np.zeros((6, 1))})["out_high"]), [0.0] * 6)


class BuildWriteTests(unittest.TestCase):
    def test_write_read_roundtrip_and_sidecar(self):
        df = good_table()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "preds.csv.gz")
            pt.write_table(path, df, {"commit": "deadbeef", "seed": 7})
            side = pd.read_json(pt.meta_path(path), typ="series")
            self.assertEqual(int(side["schema_version"]), schema.SCHEMA_VERSION)
            self.assertEqual(side["commit"], "deadbeef")
            self.assertEqual(side["identity_hashes"], ["abc123"])
            back = pt.read_table(path)
            pd.testing.assert_frame_equal(df.reset_index(drop=True), back, check_dtype=False)
            self.assertTrue(((back["label_available_at"] - back["timestamp"]) == pd.Timedelta("4h")).all())

    def test_writer_and_reader_refuse_bad_tables_and_files(self):
        df = good_table()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(pt.PredictionTableError):
                pt.write_table(os.path.join(d, "x.csv"), df.assign(extra=1))           # the writer validates
            with self.assertRaises(pt.PredictionTableError):
                pt.write_table(os.path.join(d, "x.parquet"), df)
            path = os.path.join(d, "ok.csv")
            pt.write_table(path, df)
            os.remove(pt.meta_path(path))
            with self.assertRaisesRegex(pt.PredictionTableError, "no sidecar"):
                pt.read_table(path)
            pt.write_table(path, df)
            raw = pd.read_csv(path)
            raw.loc[0, "prediction_made_at"] = "2025-02-01 00:00:00+00:00"              # hand edit that leaks
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(pt.PredictionTableError, "train_end"):
                pt.read_table(path)

    def test_spec_validates_its_fields(self):
        ok = dict(path="a.csv.gz", split_role="test", train_end="2025-02-09", identity_hash="h", horizon_bars=4, timeframe="1h")
        pt.PredictionTableSpec(**ok)
        for bad in ({"path": "a.parquet"}, {"split_role": "validation"}, {"identity_hash": ""}, {"horizon_bars": 0},
                    {"timeframe": "soon"}, {"fold": -1}, {"train_end": "not a date"}):
            with self.assertRaises((pt.PredictionTableError, ValueError), msg=str(bad)):
                pt.PredictionTableSpec(**{**ok, **bad})


class EvaluationWiring(unittest.TestCase):
    """test_all_assets_v4 with the table option, on a fake model and synthetic coins (the harness of test_evaluation_predict_calls)."""

    @classmethod
    def setUpClass(cls):
        import _nbload
        import tensorflow as tf
        cls.tf = tf
        cls.ns = _nbload.load_evaluation()
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        import test_evaluation_predict_calls as tepc
        cls.tepc = tepc

    def _test_dict(self, n_coins=4):
        td = self.tepc.synthetic_test_dict(n_coins, seed=5)
        for i, d in enumerate(td.values()):
            n = len(d["base_params"])
            lc = np.zeros((n, 7))
            lc[:, TS_COL] = T0 + 2 * HOUR_NS * i + HOUR_NS * np.arange(n)
            d["last_candles"] = lc
        return td

    def _run(self, td, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.ns["test_all_assets_v4"](self.tepc.FakeModel(), td, ["1h"], ["close"], verbose=False, n_display=0, **kw)

    def test_default_off_writes_nothing_and_adds_no_key(self):
        res = self._run(self._test_dict())
        self.assertNotIn("prediction_table_path", res)

    def test_option_writes_one_valid_table_and_changes_no_number(self):
        td = self._test_dict()
        base = self._run(td)
        with tempfile.TemporaryDirectory() as d:
            spec = pt.PredictionTableSpec(path=os.path.join(d, "t", "test.csv.gz"), split_role="test", train_end="2025-02-09",
                                          identity_hash="run-hash", horizon_bars=4, timeframe="1h", fold=2, meta={"who": "test"})
            res = self._run(td, prediction_table=spec)
            self.assertEqual(res["prediction_table_path"], spec.path)
            table = pt.read_table(spec.path)
            n_rows = sum(len(v["base_params"]) for v in td.values())
            self.assertEqual(len(table), n_rows)
            self.assertEqual(list(table["asset"].drop_duplicates()), list(td))        # asset order = test_dict order, rows grouped
            self.assertEqual(set(table["fold"]), {2})
            self.assertEqual(set(table["split_role"]), {"test"})
            self.assertEqual(set(table["identity_hash"]), {"run-hash"})
            self.assertIn("out_close", table.columns)
            for name, split in td.items():                                            # each row is its own coin's own raw output
                rows = table[table["asset"] == name]
                m = split["X_1h"].astype("float32").mean(axis=(1, 2))
                np.testing.assert_allclose(rows["out_close"].to_numpy(), m * 3.0, rtol=1e-5)
                np.testing.assert_array_equal(rows["timestamp"].astype("int64").to_numpy(), split["last_candles"][:, TS_COL].astype("int64"))
        for key in base:
            if key == "per_asset_results":
                pd.testing.assert_frame_equal(base[key], res[key])
        self.assertEqual(set(res) - set(base), {"prediction_table_path"})

    def test_table_needs_last_candles(self):
        td = self.tepc.synthetic_test_dict(2)
        with tempfile.TemporaryDirectory() as d:
            spec = pt.PredictionTableSpec(path=os.path.join(d, "t.csv"), split_role="test", train_end="2025-02-09",
                                          identity_hash="h", horizon_bars=4, timeframe="1h")
            with self.assertRaisesRegex(ValueError, "last_candles"):
                self._run(td, prediction_table=spec)


if __name__ == "__main__":
    unittest.main()

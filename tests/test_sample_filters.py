"""CONFIG['sample_filters']: اختيار العيّنات بشروط على شمعة الدخول (قيم خام، كل الشروط معاً)، والافتراضي = كل العيّنات.
    python -m unittest tests.test_sample_filters -v
"""
import unittest

import numpy as np
import pandas as pd

from tests import test_audit_round2 as r2


def _frame():
    return pd.DataFrame({"NATR_14": [0.5, 1.5, np.nan, 2.0, 3.0],
                         "ADX_14": [30.0, 25.0, 40.0, 10.0, 22.0],
                         "BBB_20_2.0": [1.0, 2.0, 3.0, 4.0, 5.0]})


class MaskTests(unittest.TestCase):
    def test_feature_ops_and_and_combination(self):
        ns, df = r2._ns(), _frame()
        m = ns["sample_filter_mask"]
        np.testing.assert_array_equal(m(df, [{"feature": "NATR_14", "op": ">", "value": 1.0}]),
                                      [False, True, False, True, True])           # NaN ← False
        np.testing.assert_array_equal(m(df, [{"feature": "NATR_14", "op": ">", "value": 1.0},
                                             {"feature": "ADX_14", "op": ">", "value": 20}]),
                                      [False, True, False, False, True])
        np.testing.assert_array_equal(m(df, [{"feature": "ADX_14", "op": "between", "value": [22, 30]}]),
                                      [True, True, False, False, True])
        np.testing.assert_array_equal(m(df, []), [True] * 5)

    def test_expr_and_registered_fn(self):
        ns, df = r2._ns(), _frame()
        m = ns["sample_filter_mask"]
        np.testing.assert_array_equal(m(df, [{"expr": "NATR_14 > 1 and (ADX_14 > 20 or `BBB_20_2.0` >= 4)"}]),
                                      [False, True, False, True, True])
        ns["register_sample_filter"]("adx_rising", lambda d: d["ADX_14"].diff() > 0)
        try:
            np.testing.assert_array_equal(m(df, [{"fn": "adx_rising"}]), [False, False, True, False, True])
        finally:
            ns["SAMPLE_FILTER_REGISTRY"].pop("adx_rising", None)

    def test_invalid_filters_raise(self):
        ns, df = r2._ns(), _frame()
        m = ns["sample_filter_mask"]
        for bad in ({"feature": "NOPE", "op": ">", "value": 1}, {"feature": "ADX_14", "op": "~", "value": 1},
                    {"feature": "ADX_14", "op": ">"}, {"feature": "ADX_14", "op": ">", "value": 1, "x": 2},
                    {"expr": "ADX_14 > 1", "fn": "a"}, {"fn": "unregistered"}, {}):
            with self.assertRaises(ValueError, msg=bad):
                m(df, [bad])
        with self.assertRaises(TypeError):
            m(df, ["ADX_14 > 20"])
        ns["register_sample_filter"]("short", lambda d: np.ones(2, bool))
        try:
            with self.assertRaises(ValueError):
                m(df, [{"fn": "short"}])
        finally:
            ns["SAMPLE_FILTER_REGISTRY"].pop("short", None)


class DatasetTests(unittest.TestCase):
    FILTERS = [{"feature": "NATR_14", "op": ">", "value": 0.45}, {"feature": "ADX_14", "op": ">", "value": 20}]

    @classmethod
    def setUpClass(cls):
        cls.frames = {c: r2._ohlcv("2025-01-01", "2025-03-01", s) for s, c in enumerate(("AAAUSDT", "BBBUSDT"))}
        cls.ds_all = r2._build(r2._cfg(), cls.frames)
        cls.cfg = r2._cfg(sample_filters=cls.FILTERS)
        cls.ds = r2._build(cls.cfg, cls.frames)

    def _raw(self, ds):
        """قيم NATR_14/ADX_14 الخام عند شمعة دخول كل عيّنة."""
        ns = r2._ns()
        ts = pd.to_datetime(ds["last_candles"][:, 3].astype("int64"), utc=True)
        out = []
        for b in ds["asset_bounds"]:
            f = r2._quiet(ns["resample_timeframes"], self.frames[b["name"]], ["1h"], config=self.cfg)["1h"]
            out.append(f.loc[ts[b["start"]:b["end"]], ["NATR_14", "ADX_14"]].to_numpy())
        return np.concatenate(out)

    def test_every_kept_sample_satisfies_all_conditions(self):
        raw = self._raw(self.ds)
        self.assertGreater(len(raw), 20)
        self.assertTrue(((raw[:, 0] > 0.45) & (raw[:, 1] > 20)).all())

    def test_filtered_is_exactly_the_matching_subset(self):
        raw_all = self._raw(self.ds_all)
        want = (raw_all[:, 0] > 0.45) & (raw_all[:, 1] > 20)
        self.assertTrue(0 < want.sum() < len(want))                       # الشرط يحذف فعلاً، ولا يحذف الكل
        np.testing.assert_array_equal(self.ds["last_candles"], self.ds_all["last_candles"][want])
        for t in self.ds["targets"]:
            np.testing.assert_array_equal(self.ds[f"y_{t}"], self.ds_all[f"y_{t}"][want])
        np.testing.assert_array_equal(self.ds["X_1h"], self.ds_all["X_1h"][want])

    def test_dataset_key_and_fingerprint_only_when_used(self):
        ns = r2._ns()
        self.assertEqual(self.ds["sample_filters"], self.FILTERS)
        self.assertNotIn("sample_filters", self.ds_all)
        fp = ns["_checkpoint_fingerprint"]
        base = {k: v for k, v in r2._cfg().items() if k != "sample_filters"}
        self.assertEqual(fp(base, ["close"], 0), fp(dict(base, sample_filters=[]), ["close"], 0))
        self.assertNotEqual(fp(base, ["close"], 0), fp(dict(base, sample_filters=self.FILTERS), ["close"], 0))
        self.assertEqual(ns["DEFAULT_CONFIG"]["sample_filters"], [])


if __name__ == "__main__":
    unittest.main()

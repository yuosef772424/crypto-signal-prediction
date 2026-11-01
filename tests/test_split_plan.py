"""Split leakage control (audit C6 / PR 12): `label_end` on every sample, the SplitPlan, the opt-in calibration split, the purity of the roles.

    python -m pytest -q tests/test_split_plan.py

Groups (added commit by commit):
  * LabelEndTests          every split (train / val / test, per-asset dict, holdout, rolling window) carries `label_end` = timestamp +
                           horizon bars, the time of the last bar the label reads (checked against the raw candles), and the split ARRAYS are
                           the ones recorded before `label_end` existed (tests/golden_split_arrays.json, tests/split_golden.py).
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _nbload  # noqa: E402
import split_golden as sg  # noqa: E402
from core import roles  # noqa: E402
from core.schema import LABEL_END_KEY, TS_COL  # noqa: E402
from run_fixture import _quiet, build_dataset, ohlcv  # noqa: E402

_PIPE = {}


def pipeline():
    if not _PIPE:
        _PIPE["ns"] = _nbload.load_pipeline()
    return _PIPE["ns"]


def members(split):
    """The splits inside a split or a `{asset: split}` dict."""
    return [split] if "y" in split else list(split.values())


class SplitArraysAreUnchanged(unittest.TestCase):
    """The row selection of every public split function equals the one recorded before label_end / SplitPlan (golden_split_arrays.json)."""

    @classmethod
    def setUpClass(cls):
        cls.fps = sg.scenarios(pipeline())
        cls.golden = sg.load_golden()

    def test_every_scenario_equals_the_recorded_golden(self):
        self.assertEqual(sorted(self.fps), sorted(self.golden))
        for name in self.golden:
            with self.subTest(name):
                self.assertEqual(sg.close(self.fps[name], self.golden[name]), [])


class LabelEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = pipeline()
        cls.ds = build_dataset(cls.ns)
        cls.cfg = dict(cls.ns["CONFIG"], split_dates=dict(sg.SPLIT_DATES))
        cls.horizon = int(cls.ds["forecast_horizon"])

    def _check(self, split):
        for m in members(split):
            le = m[LABEL_END_KEY]
            self.assertEqual(le.dtype, np.int64)
            self.assertEqual(le.shape, (len(m["last_candles"]),))
            bar = int(pd.Timedelta(self.ds["base_tf"]).value)
            np.testing.assert_array_equal(le - m["last_candles"][:, TS_COL].astype("int64"), np.full(len(le), self.horizon * bar))

    def test_every_producer_stamps_label_end(self):
        ns = self.ns
        for sep in (True, False):
            tr, va, te = _quiet(ns["split_data"], self.ds, config=self.cfg, keep_asset_test_separate=sep)
            for s in (tr, va, te):
                self._check(s)
        for s in _quiet(ns["split_data"], self.ds, mode="per_asset", config=dict(ns["CONFIG"]), keep_asset_test_separate=True):
            self._check(s)
        dsh = build_dataset(ns, holdout=sg.HOLDOUT)
        self._check(_quiet(ns["split_holdout"], dsh, config=dict(self.cfg, holdout_start=sg.HOLDOUT, open_holdout=True)))
        for tr, va, te in _quiet(ns["rolling_splits"], self.ds, test_span="4D", val_span="4D", train_span="14D", step="4D", max_windows=2,
                                 config=dict(ns["CONFIG"]), verbose=False):
            for s in (tr, va, te):
                self._check(s)

    def test_label_end_is_the_bar_the_label_reads(self):
        """future_close of a sample IS the close of the bar stamped label_end (checked on coin C0 against the raw candles)."""
        tr, va, te = _quiet(self.ns["split_data"], self.ds, config=self.cfg, keep_asset_test_separate=True)
        raw = ohlcv(70, 40, "2025-01-01")
        close_at = pd.Series(raw["close"].to_numpy(), index=raw.index.as_unit("ns").asi8)
        for split in (va, te["C0USDT"]):
            idx = split[LABEL_END_KEY]
            lc = split["last_candles"]
            keep = np.isin(idx, close_at.index.to_numpy())
            self.assertTrue(keep.any())
            # rows of C0 only: the other coins have other candles, so only rows whose label_end exists in C0 and whose entry matches are compared
            ent = close_at.reindex(split["last_candles"][:, TS_COL].astype("int64")).to_numpy()
            c0 = np.isclose(ent, lc[:, 2], rtol=1e-5) & keep
            self.assertTrue(c0.any())
            np.testing.assert_allclose(close_at.reindex(idx[c0]).to_numpy(), lc[c0, 4], rtol=1e-5)

    def test_label_end_is_never_inside_the_embargo_of_the_next_split(self):
        """The embargo gap exists so a train label cannot be read by val: train.label_end.max() < val.timestamp.min()."""
        tr, va, te = _quiet(self.ns["split_data"], self.ds, config=self.cfg, keep_asset_test_separate=False)
        self.assertLess(tr[LABEL_END_KEY].max(), va["last_candles"][:, TS_COL].min())
        self.assertLess(va[LABEL_END_KEY].max(), te["last_candles"][:, TS_COL].min())

    def test_restamping_raises_and_arrays_are_shared(self):
        tr, _, _ = _quiet(self.ns["split_data"], self.ds, config=self.cfg, keep_asset_test_separate=False)
        with self.assertRaises(ValueError):
            self.ns["stamp_label_end"](self.ds, tr, self.cfg)
        bare = {k: v for k, v in tr.items() if k != LABEL_END_KEY}
        out = self.ns["stamp_label_end"](self.ds, bare, self.cfg)
        self.assertIs(out["X_1h"], bare["X_1h"])
        self.assertNotIn(LABEL_END_KEY, bare)
        self.assertEqual(roles.role_of(out), "train")


if __name__ == "__main__":
    unittest.main()

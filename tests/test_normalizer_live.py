"""build_dataset_live with the logret_* normalizers: needs the train-fitted state (raises without it) and produces, for a
window, exactly the inputs the historical path gives for that window with the same state. No network: fetch_data is
patched with a synthetic frame.
    python -m unittest tests.test_normalizer_live -v
"""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from tests import test_audit_round2 as r2
from tests._normalizer_fixture import mode_cfg
from tests.test_price_norm_mode import _build, _frames, _split


def _live(mode, frames, state, cfg):
    ns = r2._ns()
    name = "AAAUSDT"
    frame = frames[name].reset_index(names="timestamp")
    old = ns.get("fetch_data")
    ns["fetch_data"] = lambda sym, interval, limit: frame.copy()
    try:
        return r2._quiet(ns["build_dataset_live"], [name], limit=len(frame), config=deepcopy(cfg), max_workers=1,
                         **({"price_norm_meta": state} if state is not None else {}))
    finally:
        if old is None:
            ns.pop("fetch_data", None)
        else:
            ns["fetch_data"] = old


class LiveNormalizerTests(unittest.TestCase):
    def test_raises_without_state(self):
        for mode in ("logret_std", "logret_sigma"):
            with self.subTest(mode=mode):
                cfg = mode_cfg(mode)
                with self.assertRaisesRegex(ValueError, "price_norm_meta"):
                    _live(mode, _frames(), None, cfg)

    def test_raises_with_wrong_mode_state(self):
        cfg = mode_cfg("logret_std")
        tr, _, _ = _split(_build(cfg), cfg)
        sig = deepcopy(tr["price_norm_meta"])
        sig["mode"] = "logret_sigma"
        with self.assertRaises(ValueError):
            _live("logret_std", _frames(), sig, cfg)

    def test_default_mode_ignores_state_param(self):
        cfg = mode_cfg("window_scale")
        a = _live("window_scale", _frames(), None, cfg)
        b = _live("window_scale", _frames(), {"mode": "window_scale", "version": 1}, cfg)
        np.testing.assert_array_equal(a["X_1h"], b["X_1h"])
        self.assertNotIn("price_norm_meta", a)

    def _equal_to_historical(self, mode):
        ns = r2._ns()
        cfg = mode_cfg(mode)
        frames = _frames()
        train_ds = _build(cfg, frames)
        tr, _, _ = _split(train_ds, cfg)
        state = tr["price_norm_meta"]
        live = _live(mode, frames, state, cfg)
        # historical path on the same frames, stride 1, same state
        cfg1 = dict(cfg, stride=1)
        hist = _build(cfg1, {"AAAUSDT": frames["AAAUSDT"]})
        nz = ns["get_normalizer"](mode)
        hist_x = nz.transform_inputs({"X_1h": hist["X_1h"]}, state)["X_1h"] if mode == "logret_std" else hist["X_1h"]
        ts_h = {int(t): i for i, t in enumerate(hist["last_candles"][:, 3])}
        ts_l = {int(t): i for i, t in enumerate(live["last_candles"][:, 3])}
        common = sorted(set(ts_h) & set(ts_l))
        self.assertGreater(len(common), 100)
        np.testing.assert_array_equal(live["X_1h"][[ts_l[t] for t in common]], hist_x[[ts_h[t] for t in common]])
        self.assertEqual(live["price_norm_meta"], state)
        # the live (last real) sample exists and is finite
        self.assertTrue(np.isfinite(live["X_1h"]).all())

    def test_logret_std_live_equals_historical(self):
        self._equal_to_historical("logret_std")

    def test_logret_sigma_live_equals_historical(self):
        self._equal_to_historical("logret_sigma")


if __name__ == "__main__":
    unittest.main()

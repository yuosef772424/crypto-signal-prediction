"""target_mode 'volret' (E-tgt-003): log(future / last) / EWMA std of the window's PAST close log returns, decoded from the stored
close channel, so a vol-normalised target runs on any dataset that stores the close (window_scale and logret_std).

Guards:
1. WHY 'volnorm' FAILED: on a logret_* dataset the retarget guard refuses it (train-standardised targets tied to the inputs, not a
   bp_scale problem); on splits without base_params it raises the bp_scale error; on window_scale it works.
2. THE DECODE: input_close_logreturns recovers the true past close log returns from the stored channel (window_scale and logret_std
   fixtures vs the raw candles); 'logret_sigma' / 'pct_change' raise.
3. THE TARGET: y_reg = clip(log(fut/last) / target_vol, +-clip), class = sign of the return, target_vol past-only (changing the
   future columns does not move it) and floored; the single decode volret_inverse returns the unclipped log return.
4. STAMPS: target_vol on every part, dropped when another mode follows; a logret dataset loses price_norm_meta (targets left its space).
5. ROUTE: run.retarget passes DatasetInfo.model_tf / close_index; no close input raises.
    python -m pytest tests/test_volret.py -q
"""
import copy
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _nbload  # noqa: E402
from core.price_norm import input_close_logreturns, volret_forward, volret_inverse  # noqa: E402
from core.schema import VOLRET_FLOOR, VOLRET_LAMBDA  # noqa: E402
from stats.volatility import ewma_std  # noqa: E402
from tests import test_audit_round2 as r2  # noqa: E402
from tests._normalizer_fixture import run_mode  # noqa: E402
from tests.test_price_norm_mode import _frames  # noqa: E402

_CACHE = {}


def _ns():
    if "ns" not in _CACHE:
        ns = r2._ns()
        _nbload.workflow_package().load_into(ns, only=("retarget",))
        _CACHE["ns"] = ns
    return _CACHE["ns"]


def _mode(mode):
    if mode not in _CACHE:
        cfg, ds, splits = run_mode(mode)
        _CACHE[mode] = (cfg, ds, splits)
    cfg, ds, splits = _CACHE[mode]
    _CACHE["ci"] = list(ds["feature_order"]).index("close")      # the position of the dataset used last
    return cfg, ds, copy.deepcopy(splits)


def _retarget(splits, mode="volret", **kw):
    kw.setdefault("base_tf", "1h")
    kw.setdefault("close_index", _CACHE["ci"])
    return r2._quiet(_ns()["retarget_splits"], *splits, mode=mode, **kw)


def _parts(sp):
    return [sp] if "y" in sp else [sp[k] for k in sorted(sp)]


def _true_vol(split, n_returns_from_window):
    """Reference: EWMA std of the raw past close log returns of each row, looked up in the synthetic candles by (timestamp, close)."""
    frames = _frames()
    out = np.empty(len(split["last_candles"]))
    lc = split["last_candles"]
    for i in range(len(lc)):
        for f in frames.values():
            pos = f.index.get_indexer([pd.Timestamp(int(lc[i, 3]), tz="UTC")])[0]
            if pos >= 0 and abs(f["close"].iloc[pos] - lc[i, 2]) < 1e-9 * lc[i, 2]:
                c = f["close"].to_numpy()[pos - 32 + 1 - (0 if n_returns_from_window == 31 else 1):pos + 1]
                out[i] = ewma_std(np.diff(np.log(c))[None, :], VOLRET_LAMBDA)[0]
                break
        else:
            raise AssertionError("row not found in the candles")
    return out


class WhyVolnormFailed(unittest.TestCase):
    def test_logret_dataset_hits_the_guard_not_bp_scale(self):
        _mode("logret_std")
        _, _, sp = _mode("logret_std")
        self.assertTrue(all("base_params" in p for s in sp for p in _parts(s)))      # bp_scale IS there
        with self.assertRaisesRegex(ValueError, "logret_"):
            r2._quiet(_ns()["retarget_splits"], *sp, mode="volnorm")

    def test_missing_base_params_message(self):
        _, _, sp = _mode("window_scale")
        for p in _parts(sp[0]):
            del p["base_params"]
        with self.assertRaisesRegex(ValueError, "base_params"):
            r2._quiet(_ns()["retarget_splits"], *sp, mode="volnorm")

    def test_window_scale_works(self):
        _, _, sp = _mode("window_scale")
        out = r2._quiet(_ns()["retarget_splits"], *sp, mode="volnorm")
        self.assertEqual(out[0]["target_mode"], "volnorm")


class Decode(unittest.TestCase):
    def test_window_scale_roundtrip_through_the_real_encoder(self):
        ns = _ns()
        rng = np.random.default_rng(0)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, (7, 32)), axis=1))
        med = np.median(closes, axis=1)
        q = np.percentile(closes, [25, 75], axis=1)
        iqr = q[1] - q[0]
        X = ns["process_windows"](closes[:, :, None], ["close"], med, iqr, "robust", config=dict(ns["CONFIG"]))
        r, ok = input_close_logreturns(X[:, :, 0].astype("float64"), "window_scale", center=med[:, None], scale=iqr[:, None],
                                       clip_abs=ns["CLIP_ABS"], log=np.log)
        self.assertTrue(ok.all())
        np.testing.assert_allclose(r, np.diff(np.log(closes), axis=1), atol=2e-6)

    def test_clipped_or_flat_rows_are_flagged(self):
        x = np.array([[0.0, 5.0, 1.0], [0.1, 0.2, 0.3]])
        _, ok = input_close_logreturns(x, "window_scale", center=np.array([[100.0], [100.0]]), scale=np.array([[2.0], [0.0]]),
                                       clip_abs=5.0, log=np.log)
        self.assertEqual(ok.tolist(), [False, False])

    def test_unreadable_modes_raise(self):
        for mode in ("logret_sigma", "pct_change", "nope"):
            with self.assertRaisesRegex(ValueError, "cannot read the past closes"):
                input_close_logreturns(np.zeros((2, 3)), mode)

    def test_logret_sigma_dataset_raises_through_retarget(self):
        _mode("window_scale")
        _, _, sp = _mode("logret_sigma")
        for s in sp:
            for p in _parts(s):
                p["price_norm_meta"] = p.get("price_norm_meta") or {}
        with self.assertRaises(ValueError):
            _retarget(sp, base_tf="1h", close_index=0)


class Target(unittest.TestCase):
    def _check(self, mode_name, n_ret):
        _mode(mode_name)
        _, _, sp = _mode(mode_name)
        out = _retarget(sp, clip=4.0)
        for before, after in zip(sp, out):
            for pb, pa in zip(_parts(before), _parts(after)):
                lc = pa["last_candles"]
                vol = pa["target_vol"].astype("float64")
                self.assertEqual(len(vol), len(lc))
                self.assertGreaterEqual(float(vol.min()), VOLRET_FLOOR)
                # rows whose window the stored channel cannot decode (clipped at CLIP_ABS) take the median vol: compared apart
                ci = _CACHE["ci"]
                clipped = (np.abs(np.asarray(pa["X_1h"][:, :, ci], dtype="float64")) >= 5.0 - 1e-5).any(axis=1) if mode_name == "window_scale" \
                    else np.zeros(len(vol), dtype=bool)
                good = ~clipped
                np.testing.assert_allclose(vol[good], _true_vol(pa, n_ret)[good], rtol=2e-3, atol=1e-6)
                if clipped.any():
                    np.testing.assert_allclose(vol[clipped], np.median(vol[good]), rtol=0.5)
                for t, col in (("high", 5 + 1), ("low", 5), ("close", 4)):
                    fut = lc[:, {"high": 6, "low": 5, "close": 4}[t]]
                    last = lc[:, {"high": 0, "low": 1, "close": 2}[t]]
                    z = np.log(fut / last) / vol
                    np.testing.assert_allclose(pa["y"][f"y_{t}_reg"], np.clip(z, -4, 4).astype("float32"), rtol=1e-5, atol=1e-5)
                    np.testing.assert_array_equal(pa["y"][f"y_{t}_class"], np.where(fut > last, 1.0, -1.0).astype("float32"))
                # the ONE decode: unclipped rows give back the log return
                z = np.log(lc[:, 4] / lc[:, 2]) / vol
                inside = np.abs(z) < 4
                np.testing.assert_allclose(volret_inverse(pa["y"]["y_close_reg"][inside].astype("float64"), vol[inside]),
                                           np.log(lc[inside, 4] / lc[inside, 2]), rtol=1e-4, atol=1e-6)
                self.assertEqual(pa["target_mode"], "volret")
                self.assertEqual(pa["reg_target_scale"], 1.0)

    def test_window_scale_dataset(self):
        self._check("window_scale", 31)

    def test_logret_std_dataset_and_meta_dropped(self):
        self._check("logret_std", 32)
        _, _, sp = _mode("logret_std")
        out = _retarget(sp)
        for s in out:
            for p in _parts(s):
                self.assertNotIn("price_norm_meta", p)
        for s in sp:                                                  # the originals are untouched
            for p in _parts(s):
                self.assertIn("price_norm_meta", p)

    def test_vol_is_past_only(self):
        _mode("window_scale")
        _, _, sp = _mode("window_scale")
        a = _retarget(sp)
        sp2 = copy.deepcopy(sp)
        for s in sp2:
            for p in _parts(s):
                p["last_candles"][:, 4:7] *= 1.37                      # the future columns only
        b = _retarget(sp2)
        for x, y in zip(a, b):
            for px, py in zip(_parts(x), _parts(y)):
                np.testing.assert_array_equal(px["target_vol"], py["target_vol"])

    def test_default_clip_is_ten_and_unknown_clip_flows(self):
        _mode("window_scale")
        _, _, sp = _mode("window_scale")
        out = _retarget(sp)
        mx = max(float(np.abs(p["y"][k]).max()) for s in out for p in _parts(s) for k in p["y"] if k.endswith("_reg"))
        self.assertLessEqual(mx, 10.0 + 1e-6)

    def test_target_vol_dropped_by_the_next_mode(self):
        _mode("window_scale")
        _, _, sp = _mode("window_scale")
        out = _retarget(sp)
        back = r2._quiet(_ns()["retarget_splits"], *out, mode="return")
        for s in back:
            for p in _parts(s):
                self.assertNotIn("target_vol", p)

    def test_missing_close_index_raises(self):
        _mode("window_scale")
        _, _, sp = _mode("window_scale")
        with self.assertRaisesRegex(ValueError, "close_index"):
            r2._quiet(_ns()["retarget_splits"], *sp, mode="volret")


class PureFunctions(unittest.TestCase):
    def test_ewma_std(self):
        r = np.array([[0.01, -0.02, 0.03], [0.0, 0.0, 0.0]])
        w = 0.94 ** np.array([2, 1, 0])
        want = np.sqrt((r[0] ** 2 * w).sum() / w.sum())
        np.testing.assert_allclose(ewma_std(r, 0.94), [want, 0.0])
        self.assertTrue(np.isnan(ewma_std(np.array([[0.1, np.nan]]), 0.94)[0]))
        for bad in (0.0, 1.0, -1):
            with self.assertRaises(ValueError):
                ewma_std(r, bad)

    def test_forward_inverse(self):
        lr, vol = np.array([0.02, -0.01]), np.array([0.01, 0.02])
        np.testing.assert_allclose(volret_inverse(volret_forward(lr, vol), vol), lr)


class Route(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import run_fixture as rf
        from tests.test_run_steps import _dataset
        cls.rf = rf
        ns = rf.run_namespace()
        cls.ns, cls.S = ns, ns["RunSettings"]
        kit = ns["Toolkit"].from_namespace(ns)
        cls.ds = _dataset(False, None)
        cls.info = rf._quiet(ns["apply_dataset_config"], cls.S(), cls.ds, kit)
        cls.splits = rf._quiet(ns["make_splits"], cls.S(), cls.ds, cls.info, kit)

    def test_info_carries_the_close_position(self):
        self.assertEqual(self.info.close_index, list(self.ds["feature_order"]).index("close"))

    def test_run_retarget_volret(self):
        s = self.S().updated({"target": {"target_mode": "volret", "clip": 4.0}})
        tr, va, te = self.rf._quiet(self.ns["retarget"], s, *self.splits, self.info)
        for sp in (tr, va, te):
            for p in _parts(sp):
                self.assertEqual(p["target_mode"], "volret")
                self.assertEqual(len(p["target_vol"]), len(p["last_candles"]))
                self.assertLessEqual(float(np.abs(p["y"]["y_close_reg"]).max()), 4.0 + 1e-6)

    def test_no_close_input_raises(self):
        info = copy.copy(self.info)
        object.__setattr__(info, "close_index", None)
        s = self.S().updated({"target": {"target_mode": "volret"}})
        with self.assertRaisesRegex(ValueError, "close_index"):
            self.rf._quiet(self.ns["retarget"], s, *self.splits, info)

    def test_settings_accept_volret_and_relative(self):
        for m in ("volret", "volret+relative"):
            self.assertEqual(self.S().updated({"target": {"target_mode": m}}).target.target_mode, m)


if __name__ == "__main__":
    unittest.main()

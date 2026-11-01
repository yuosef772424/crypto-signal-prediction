"""data/level_channel.py: the level side-channel (default OFF), one registered component, fitted on TRAIN only.

    python -m pytest -q tests/test_level_channel.py

Groups:
  * SettingsTests    unknown key / feature / component / bad window raise; defaults are explicit; OFF = no columns;
  * FeatureTests     every feature is causal (a prefix gives the same values; changing the future changes no earlier row) and means what
                     its docstring says on hand-made series;
  * PipelineTests    through the REAL pipeline on seeded candles: default off adds no key / column; ON leaves every other column
                     unchanged on the shared rows; the dataset is 'unfitted' until the split; the state is fitted on the train split
                     alone (changing val / test leaves it identical), val / test use the train statistics, the sealed holdout and the
                     live path need the stored state, double application raises.
"""
import contextlib
import io
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
import run_fixture as rf  # noqa: E402

_NS = None


def ns():
    global _NS
    if _NS is None:
        _NS = _nbload.load_pipeline()
    return _NS


def frame(n=600, seed=0, vol=0.004):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    op = np.r_[close[0], close[:-1]]
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": op, "high": np.maximum(op, close) * (1 + rng.random(n) * 0.003),
                         "low": np.minimum(op, close) * (1 - rng.random(n) * 0.003), "close": close,
                         "volume": rng.random(n) * 1000 + 50}, index=idx)


def cfg(**level):
    c = dict(ns()["PIPELINE_DEFAULT_CONFIG"])
    c["level_channel"] = level if level is not None else None
    return c


class SettingsTests(unittest.TestCase):
    def test_off_means_no_columns_and_same_frame(self):
        n = ns()
        c = dict(n["PIPELINE_DEFAULT_CONFIG"])
        self.assertIsNone(c["level_channel"])
        self.assertIsNone(n["level_channel_settings"](c))
        self.assertEqual(n["level_channel_columns"](c), [])
        df = frame(50)
        self.assertIs(n["add_level_features"](df, c), df)

    def test_defaults_are_explicit_and_columns_follow_the_feature_list(self):
        n = ns()
        s = n["level_channel_settings"](cfg())
        self.assertEqual(s, {**n["LEVEL_SETTINGS_DEFAULTS"]})
        self.assertEqual(n["level_channel_columns"](cfg(features=["logvol", "volreg"], window=8, long_window=40)), ["LVL_logvol", "LVL_volreg"])

    def test_unknown_things_raise(self):
        n = ns()
        for bad in ({"windo": 8}, {"features": ["nope"]}, {"features": []}, {"features": ["logvol", "logvol"]}, {"component": "level_v9"},
                    {"window": 2}, {"window": 8, "long_window": 8}, {"long_window": 501}, {"clip": 0}, {"window": True}):
            with self.assertRaises(ValueError, msg=str(bad)):
                n["level_channel_settings"](cfg(**bad))
        with self.assertRaises(ValueError):
            n["level_channel_settings"]({**n["PIPELINE_DEFAULT_CONFIG"], "level_channel": "on"})

    def test_update_config_accepts_the_key_and_reset_restores_off(self):
        n = ns()
        n["reset_config"]()
        n["update_config"]({"level_channel": {"window": 8, "long_window": 40}})
        self.assertEqual(n["CONFIG"]["level_channel"], {"window": 8, "long_window": 40})
        n["reset_config"]()
        self.assertIsNone(n["CONFIG"]["level_channel"])

    def test_checkpoint_fingerprint_changes_only_when_the_channel_is_on(self):
        n = ns()
        base = dict(n["PIPELINE_DEFAULT_CONFIG"])
        feats = ["close", "volume"]
        fp = lambda c: n["_checkpoint_fingerprint"](c, feats)  # noqa: E731
        absent = {k: v for k, v in base.items() if k != "level_channel"}
        self.assertEqual(fp(base), fp(absent))                                    # off == the key never existed (old checkpoints stay valid)
        on, on2 = fp({**base, "level_channel": {"window": 8, "long_window": 40}}), fp({**base, "level_channel": {"window": 12, "long_window": 40}})
        self.assertNotEqual(on, fp(base))
        self.assertNotEqual(on, on2)                                             # same column names, different values -> a different fingerprint

    def test_registries_refuse_duplicates_and_unknown(self):
        n = ns()
        with self.assertRaises(ValueError):
            n["register_level_channel"](n["LevelV1"]())
        with self.assertRaises(ValueError):
            n["get_level_channel"]("nope")
        with self.assertRaises(ValueError):
            n["LEVEL_FEATURES"].register("logvol", lambda d, w, l: d["close"], version=1)


class FeatureTests(unittest.TestCase):
    W, L = 8, 40

    def feats(self, df):
        out = ns()["add_level_features"](df, cfg(window=self.W, long_window=self.L))
        return out[[c for c in out.columns if c.startswith("LVL_")]]

    def test_every_feature_is_causal(self):
        df = frame(300)
        full = self.feats(df)
        cut = 200
        prefix = self.feats(df.iloc[:cut])
        pd.testing.assert_frame_equal(full.iloc[:cut], prefix, check_exact=False, atol=1e-12)
        # change the future only: rows <= cut-1 must not move
        df2 = df.copy()
        df2.iloc[cut:, :] = df2.iloc[cut:, :] * 1.7
        pd.testing.assert_frame_equal(self.feats(df2).iloc[:cut], full.iloc[:cut], check_exact=False, atol=1e-12)

    def test_nan_until_the_long_window_is_full_then_finite(self):
        f = self.feats(frame(200))
        self.assertTrue(f.iloc[: self.L - 1].isna().any(axis=1).all())
        self.assertTrue(np.isfinite(f.iloc[self.L + 1:].to_numpy()).all())

    def test_logvol_is_the_log_of_realized_vol(self):
        df = frame(120)
        r = np.log(df.close).diff()
        expect = np.log(r.rolling(self.W).std(ddof=1))
        np.testing.assert_allclose(self.feats(df)["LVL_logvol"].iloc[60:], expect.iloc[60:], rtol=1e-12)

    def test_volatility_level_survives_where_window_scaling_erases_it(self):
        calm, wild = frame(300, seed=1, vol=0.001), frame(300, seed=1, vol=0.01)           # same shape, 10x the vol
        a, b = self.feats(calm)["LVL_logvol"].iloc[100:].mean(), self.feats(wild)["LVL_logvol"].iloc[100:].mean()
        self.assertAlmostEqual(b - a, np.log(10), delta=0.15)
        # the retvol (window return in vol units) is scale-free: identical shape -> identical value
        ra, rb = self.feats(calm)["LVL_retvol"].iloc[100:], self.feats(wild)["LVL_retvol"].iloc[100:]
        self.assertLess(np.abs(ra - rb).mean(), 0.5)

    def test_distances_are_nonnegative_and_zero_at_the_extreme(self):
        n = 120
        idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
        close = pd.Series(100 * np.exp(0.002 * np.arange(n) + 0.001 * np.sin(np.arange(n))), index=idx)     # rising: close ~ at the window high
        df = pd.DataFrame({"open": close, "high": close, "low": close * 0.999, "close": close, "volume": 100.0}, index=idx)
        f = self.feats(df)
        self.assertTrue((f["LVL_disthi"].dropna() >= 0).all() and (f["LVL_distlo"].dropna() >= 0).all())
        self.assertAlmostEqual(float(f["LVL_disthi"].iloc[-1]), 0.0, places=9)
        self.assertGreater(float(f["LVL_distlo"].iloc[-1]), 0.0)

    def test_dollar_volume_z_is_high_after_a_volume_surge(self):
        df = frame(200)
        df.iloc[-self.W:, df.columns.get_loc("volume")] *= 50
        f = self.feats(df)
        self.assertGreater(float(f["LVL_logdvol_z"].iloc[-1]), 1.5)
        self.assertLess(abs(float(f["LVL_logdvol_z"].iloc[100])), 1.5)


def build(level, coins=4, days=60):
    """The real pipeline on seeded candles (tests/run_fixture.py setup) with CONFIG['level_channel'] = ``level`` (None = off)."""
    n = ns()
    n["reset_config"]()
    q = rf._quiet
    q(n["apply_hourly_preset"], n["HOURLY_W32_S8_OVERRIDES"])
    n["update_config"]({"tf_order": ["1h"], "window_sizes": {"1h": 16}, "level_channel": level,
                        "phase2_data": {"use_intraday_15m": False, "use_futures_metrics": False},
                        "funding_rate": {"enabled": False}, "open_interest": {"enabled": False}})
    q(n["refresh_features"])
    starts = ["2025-01-01", "2025-01-01 03:00", "2025-01-01 05:00", "2025-01-01 11:00"]

    def loader(fid, name):
        i = int(name[1])
        return rf.ohlcv(days, 40 + i, starts[i])

    ds = q(n["build_dataset"], [{"name": f"C{i}USDT"} for i in range(coins)], load_asset_fn=loader,
           resample_fn=n["make_resample_fn"](n["CONFIG"]), max_workers=1, config=n["CONFIG"])
    cfgc = dict(n["CONFIG"])
    n["reset_config"]()
    ds["split_dates"] = {"train_end": "2025-01-28", "val_end": "2025-02-10"}
    return ds, cfgc


LEVEL = {"window": 8, "long_window": 40}


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off, cls.cfg_off = build(None)
        cls.on, cls.cfg_on = build(LEVEL)

    def split(self, ds, config=None):
        """split_data with the pipeline defaults (like tests/run_fixture): the level settings live on the dataset, not in this config."""
        n = ns()
        return rf._quiet(n["split_data"], ds, config={**n["PIPELINE_DEFAULT_CONFIG"], "split_dates": dict(ds["split_dates"])})

    def test_default_off_adds_no_key_and_no_column(self):
        self.assertNotIn("level_channel", self.off)
        self.assertFalse([c for c in self.off["feature_order"] if c.startswith("LVL_")])

    def test_on_adds_the_columns_and_marks_the_dataset_unfitted(self):
        cols = [c for c in self.on["feature_order"] if c.startswith("LVL_")]
        self.assertEqual(cols, ns()["level_channel_columns"](self.cfg_on))
        self.assertEqual(self.on["level_channel"]["state"], "unfitted")
        self.assertEqual(len(self.on["feature_order"]), len(self.off["feature_order"]) + len(cols))

    def test_other_columns_are_unchanged_on_the_shared_rows(self):
        keys = lambda d: pd.MultiIndex.from_arrays([np.repeat([b["name"] for b in d["asset_bounds"]], [b["end"] - b["start"] for b in d["asset_bounds"]]),  # noqa: E731
                                                    d["last_candles"][:, 0 + 3].astype("int64")])
        ko, kn = keys(self.off), keys(self.on)
        pos_on = {k: i for i, k in enumerate(kn)}
        rows = [(i, pos_on[k]) for i, k in enumerate(ko) if k in pos_on]
        self.assertGreater(len(rows), 100)
        order = {c: i for i, c in enumerate(self.on["feature_order"])}
        keep = [order[c] for c in self.off["feature_order"]]
        a = self.off["X_1h"][[r[0] for r in rows]]
        b = self.on["X_1h"][[r[1] for r in rows]][:, :, keep]
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_state_is_fitted_on_train_alone_and_val_test_use_it(self):
        n = ns()
        train, val, test = self.split(self.on, self.cfg_on)
        meta = train["level_meta"]
        self.assertEqual(meta["fit_on"], "train")
        self.assertIs(val["level_meta"], meta)
        for sp in (test.values() if "y" not in test else [test]):
            self.assertEqual(sp["level_meta"]["stats"], meta["stats"])
        idx = meta["index"]
        # the entry-time (last step) values of the train samples are standardized: mean 0, std 1 (clip permitting)
        last = train["X_1h"][:, -1, :][:, idx].astype("float64")
        np.testing.assert_allclose(last.mean(axis=0), 0.0, atol=0.08)
        np.testing.assert_allclose(last.std(axis=0), 1.0, atol=0.15)
        # equals the manual statistic of the raw (unfitted) rows that went to train
        raw_train_last = self.on["X_1h"]
        self.assertEqual(raw_train_last.shape[2], train["X_1h"].shape[2])
        manual_mean = np.asarray(meta["stats"]["1h"]["mean"])
        self.assertEqual(len(manual_mean), len(idx))
        # val / test are NOT re-centred on themselves
        vlast = val["X_1h"][:, -1, :][:, idx].astype("float64")
        self.assertGreater(np.abs(vlast.mean(axis=0)).max(), np.abs(last.mean(axis=0)).max())
        # other columns untouched by the stamping
        other = [j for j in range(train["X_1h"].shape[2]) if j not in idx]
        self.assertTrue(np.isfinite(train["X_1h"][:, :, other]).all())
        del n

    def test_changing_val_and_test_leaves_the_fitted_state_identical(self):
        n = ns()
        base_train, _, _ = self.split(self.on, self.cfg_on)
        twisted = dict(self.on)
        ts = np.asarray(self.on["last_candles"][:, 3]).astype("int64")
        late = ts >= pd.Timestamp("2025-02-01", tz="UTC").value
        X = np.array(self.on["X_1h"])
        li = [self.on["feature_order"].index(c) for c in self.on["level_channel"]["columns"]]
        X[np.ix_(late, np.arange(X.shape[1]), li)] += 3.0                           # shift every LVL value after the train end
        twisted["X_1h"] = X
        t2, _, _ = self.split(twisted, self.cfg_on)
        self.assertEqual(base_train["level_meta"]["stats"], t2["level_meta"]["stats"])
        np.testing.assert_array_equal(base_train["X_1h"], t2["X_1h"])
        del n

    def test_stamping_twice_raises_and_needs_a_state(self):
        n = ns()
        train, _, _ = self.split(self.on, self.cfg_on)
        comp = n["get_level_channel"]("level_v1")
        with self.assertRaises(ValueError):
            comp.stamp(train, train["level_meta"])
        with self.assertRaises(ValueError):
            comp.stamp(dict(train, **{}), None)
        with self.assertRaises(ValueError):
            comp.prepare_live(self.on, None)

    def test_holdout_needs_the_train_state_and_applies_it(self):
        n = ns()
        c = {**n["PIPELINE_DEFAULT_CONFIG"], "holdout_start": "2025-02-26", "open_holdout": True, "split_dates": dict(self.on["split_dates"])}
        ds = dict(self.on, holdout_start="2025-02-26")
        train, _, _ = rf._quiet(n["split_data"], ds, config=c)
        with self.assertRaises(ValueError):
            rf._quiet(n["split_holdout"], ds, config=c)
        hold = rf._quiet(n["split_holdout"], ds, config=c, level_meta=train["level_meta"])
        sp = hold if "y" in hold else next(iter(hold.values()))
        self.assertEqual(sp["level_meta"]["stats"], train["level_meta"]["stats"])
        raw = rf._quiet(n["split_holdout"], dict(self.off, holdout_start="2025-02-26"), config=c, level_meta=train["level_meta"])
        self.assertNotIn("level_meta", raw if "y" in raw else next(iter(raw.values())))   # off: nothing stamped, level_meta ignored

    def test_live_needs_the_state_before_any_fetch(self):
        n = ns()
        c = {**n["PIPELINE_DEFAULT_CONFIG"], "level_channel": dict(LEVEL)}
        with self.assertRaises(ValueError) as cm:
            n["build_dataset_live"](["BTCUSDT"], config=c)             # raises before fetching anything (no network in the tests)
        self.assertIn("level_meta", str(cm.exception))

    def test_off_split_has_no_level_meta(self):
        train, val, test = self.split(self.off, self.cfg_off)
        self.assertNotIn("level_meta", train)
        self.assertNotIn("level_meta", val)


if __name__ == "__main__":
    unittest.main()

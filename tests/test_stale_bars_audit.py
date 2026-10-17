"""tools/stale_bars_audit.py + CONFIG['stale_bars']: dead-coin tails and long zero-volume flat runs.
Fixtures (1h): DEAD = live until bar 250 then a forward-filled tail (o=h=l=c, volume 0); ILLIQ = ~30% genuine zero-return
bars WITH volume plus short (<= 2) zero-volume flat runs; NORMAL = random walk.  Pipeline part: option OFF is byte-identical
to a config without the key; option ON removes exactly the samples whose entry/horizon bars are stale.
    python -m unittest tests.test_stale_bars_audit -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import stale_bars_audit as sba  # noqa: E402

from tests import test_audit_round2 as r2  # noqa: E402

N, LIVE, MIN_RUN = 400, 250, 6


def _walk(n, seed, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    op = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": op, "high": np.maximum(op, close) * (1 + rng.random(n) * 0.003),
                       "low": np.minimum(op, close) * (1 - rng.random(n) * 0.003), "close": close,
                       "volume": rng.random(n) * 1000 + 50}, index=idx)
    return df


def dead_coin():
    df = _walk(N, 1)
    last = df["close"].iloc[LIVE - 1]
    df.iloc[LIVE:, df.columns.get_indexer(["open", "high", "low", "close"])] = last
    df.iloc[LIVE:, df.columns.get_loc("volume")] = 0.0
    return df


def illiquid_coin():
    df = _walk(N, 2)
    rng = np.random.default_rng(5)
    for i in np.where(rng.random(N) < 0.30)[0][1:]:                  # genuine zero-return bars that DO have volume
        df.iloc[i, :4] = df["close"].iloc[i - 1]
    for s in (100, 200, 300):                                        # short zero-volume flat runs (< MIN_RUN)
        df.iloc[s:s + 2, :4] = df["close"].iloc[s - 1]
        df.iloc[s:s + 2, df.columns.get_loc("volume")] = 0.0
    return df


def normal_coin():
    return _walk(N, 3)


FRAMES = {"DEADUSDT": dead_coin(), "ILLIQUSDT": illiquid_coin(), "NORMALUSDT": normal_coin()}
T0 = FRAMES["NORMALUSDT"].index[0]
SPLITS = {"train": (T0, T0 + pd.Timedelta(hours=200)), "val": (T0 + pd.Timedelta(hours=200), T0 + pd.Timedelta(hours=300)),
          "test": (T0 + pd.Timedelta(hours=300), T0 + pd.Timedelta(hours=N))}


class MaskTests(unittest.TestCase):
    def test_dead_tail_is_stale_and_live_part_is_not(self):
        m = sba.stale_bar_mask(FRAMES["DEADUSDT"], MIN_RUN)
        self.assertEqual(int(m.sum()), N - LIVE)
        self.assertTrue(m[LIVE:].all() and not m[:LIVE].any())

    def test_illiquid_genuine_zero_return_bars_with_volume_are_kept(self):
        df = illiquid_coin()
        self.assertGreater(int((df["close"].diff() == 0).sum()), 40)    # the fixture really has many zero returns
        self.assertFalse(sba.stale_bar_mask(df, MIN_RUN).any())         # volume > 0 or run < MIN_RUN: nothing stale

    def test_normal_coin_has_nothing_stale(self):
        self.assertFalse(sba.stale_bar_mask(normal_coin(), MIN_RUN).any())

    def test_long_zero_volume_run_inside_history_is_stale_but_live_data_after_it_is_not(self):
        df = normal_coin()
        df.iloc[100:110, :4] = df["close"].iloc[99]
        df.iloc[100:110, df.columns.get_loc("volume")] = 0.0
        m = sba.stale_bar_mask(df, MIN_RUN)
        self.assertTrue(m[100:110].all())
        self.assertEqual(int(m.sum()), 10)

    def test_short_tail_of_a_live_coin_is_not_dropped(self):
        df = normal_coin()
        df.iloc[-3:, :4] = df["close"].iloc[-4]
        df.iloc[-3:, df.columns.get_loc("volume")] = 0.0
        self.assertFalse(sba.stale_bar_mask(df, MIN_RUN).any())

    def test_min_run_validation(self):
        for bad in (0, 1, -3, 2.5, True, None):
            with self.assertRaises(ValueError, msg=bad):
                sba.stale_bar_mask(normal_coin(), bad)


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coins, cls.splits = sba.audit_frames(FRAMES, min_run=MIN_RUN, horizon=1, splits=SPLITS)
        cls.coin = cls.coins.set_index("coin")

    def test_dead_coin_last_real_trade_and_tail(self):
        row = self.coin.loc["DEADUSDT"]
        self.assertEqual(pd.Timestamp(row["last_real_ts"]), FRAMES["DEADUSDT"].index[LIVE - 1])
        self.assertEqual(int(row["tail_bars"]), N - LIVE)
        self.assertTrue(bool(row["dead"]))
        self.assertEqual(int(row["stale_bars"]), N - LIVE)

    def test_live_coins_are_not_flagged(self):
        for c in ("ILLIQUSDT", "NORMALUSDT"):
            self.assertFalse(bool(self.coin.loc[c, "dead"]), c)
            self.assertEqual(int(self.coin.loc[c, "stale_bars"]), 0, c)
        self.assertGreater(int(self.coin.loc["ILLIQUSDT", "flat_zero_volume_bars"]), 0)   # short runs are reported, not stale
        self.assertEqual(int(self.coin.loc["ILLIQUSDT", "longest_flat_run"]), 2)

    def test_split_shares_follow_the_period_of_the_tail(self):
        s = self.splits.set_index(["coin", "split"])
        # tail starts at bar 250 = inside val (200..300); entry rows 249..298 touch it; test is entirely tail
        self.assertEqual(s.loc[("DEADUSDT", "train"), "stale_rows"], 0)
        self.assertGreater(s.loc[("DEADUSDT", "val"), "stale_rows"], 40)
        self.assertEqual(s.loc[("DEADUSDT", "test"), "stale_share"], 1.0)
        self.assertEqual(s.loc[("DEADUSDT", "test"), "zero_target_share"], 1.0)
        self.assertEqual(s.loc[("DEADUSDT", "test"), "zero_rows_from_stale"], s.loc[("DEADUSDT", "test"), "n_rows"])
        for c in ("ILLIQUSDT", "NORMALUSDT"):
            self.assertEqual(s.loc[(c, "test"), "stale_rows"], 0)
        # illiquid genuine zeros stay in the clean share
        self.assertGreater(s.loc[("ILLIQUSDT", "train"), "zero_target_share"], 0.15)
        self.assertEqual(s.loc[("ILLIQUSDT", "train"), "zero_share_clean"], s.loc[("ILLIQUSDT", "train"), "zero_target_share"])

    def test_derive_min_run_is_data_driven(self):
        self.assertEqual(sba.derive_min_run({"I": illiquid_coin(), "N": normal_coin()}, quantile=1.0, floor=2), 3)
        df = normal_coin()
        df.iloc[50:57, :4] = df["close"].iloc[49]
        df.iloc[50:57, df.columns.get_loc("volume")] = 0.0
        self.assertEqual(sba.derive_min_run({"X": df}, quantile=1.0, floor=2), 8)     # longest live run (7) + 1
        self.assertEqual(sba.derive_min_run({"DEAD": dead_coin()}, quantile=1.0, floor=3), 3)   # tails are not "genuine"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = r2._cfg()
        cls.no_key = {k: v for k, v in cls.base.items() if k != "stale_bars"}
        cls.off = r2._build(cls.base, FRAMES)
        cls.old = r2._build(cls.no_key, FRAMES)
        cls.on_cfg = r2._cfg(stale_bars={"min_run": MIN_RUN})
        cls.on = r2._build(cls.on_cfg, FRAMES)

    def test_default_is_off_and_output_is_byte_identical(self):
        self.assertIsNone(r2._ns()["DEFAULT_CONFIG"]["stale_bars"])
        self.assertNotIn("stale_bars", self.off)
        for k in ("last_candles", "base_params", "X_1h", "y_close_reg", "y_close_class"):
            self.assertEqual(self.off[k].tobytes(), self.old[k].tobytes(), k)

    def test_on_drops_stale_samples_and_zero_share_falls(self):
        zero = lambda d: float(np.mean(d["y_close_reg"] == 0))  # noqa: E731
        self.assertGreater(zero(self.off), 0.10)                          # dead tail + 30% illiquid zeros inflate it
        self.assertLess(zero(self.on), zero(self.off))
        print(f"\nFIXTURE zero share OFF={zero(self.off):.4f} ON={zero(self.on):.4f} n_off={len(self.off['y_close_reg'])} n_on={len(self.on['y_close_reg'])}")
        self.assertEqual(self.on["stale_bars"], {"min_run": MIN_RUN})
        # every dropped sample belongs to DEAD's flat tail; every kept sample is untouched
        def keys(d):
            ts = d["last_candles"][:, 3].astype("int64")
            return [(b["name"], int(t)) for b in d["asset_bounds"] for t in ts[b["start"]:b["end"]]]
        k_off, k_on = keys(self.off), keys(self.on)
        keep = np.isin(np.array([hash(k) for k in k_off]), np.array([hash(k) for k in k_on]))
        np.testing.assert_array_equal(self.off["y_close_reg"][keep], self.on["y_close_reg"])
        ts_off = self.off["last_candles"][:, 3].astype("int64")
        dropped = [k for k, kp in zip(k_off, keep) if not kp]
        self.assertTrue(all(c == "DEADUSDT" for c, _ in dropped))
        dead_ts = FRAMES["DEADUSDT"].index[LIVE - 1].value
        self.assertTrue(all(t >= dead_ts for _, t in dropped))
        self.assertGreater(int((~keep).sum()), 0)
        # genuine zero-return samples of the illiquid coin are all still there
        b_off = {b["name"]: b for b in self.off["asset_bounds"]}
        b_on = {b["name"]: b for b in self.on["asset_bounds"]}
        for c in ("ILLIQUSDT", "NORMALUSDT"):
            self.assertEqual(b_off[c]["end"] - b_off[c]["start"], b_on[c]["end"] - b_on[c]["start"], c)

    def test_invalid_option_raises(self):
        ns = r2._ns()
        for bad in ({"min_run": 1}, {"min_run": 3, "x": 1}, {}, 5, {"min_run": "a"}):
            with self.assertRaises(ValueError, msg=bad):
                ns["stale_bars_min_run"](r2._cfg(stale_bars=bad))

    def test_fingerprint_only_when_used(self):
        fp = r2._ns()["_checkpoint_fingerprint"]
        self.assertEqual(fp(self.base, ["close"], 0), fp(self.no_key, ["close"], 0))
        self.assertNotEqual(fp(self.base, ["close"], 0), fp(self.on_cfg, ["close"], 0))


class SplitLayoutTests(unittest.TestCase):
    """The test split has no top-level 'y' when keep_asset_test_separate=True: it is {asset: split}, by design."""

    def test_test_layout_per_asset_vs_merged(self):
        ns = r2._ns()
        ds = r2._build(r2._cfg(), FRAMES)
        _, _, test = r2._quiet(ns["split_data"], ds, config=r2._cfg(keep_asset_test_separate=True))
        self.assertNotIn("y", test)
        self.assertTrue(all("y_close_reg" in sp["y"] for sp in test.values()))
        _, _, merged = r2._quiet(ns["split_data"], ds, config=r2._cfg(keep_asset_test_separate=False))
        self.assertIn("y_close_reg", merged["y"])


if __name__ == "__main__":
    unittest.main()

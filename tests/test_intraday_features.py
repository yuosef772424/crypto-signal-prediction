"""اختبارات tools/intraday_features.py — بيانات تركيبية صغيرة (15m + تمويل + metrics)، بلا Drive ولا شبكة.

تتحقّق من: صحّة التجميع اليومي، غياب التسرّب (تغيير ما بعد إغلاق اليوم D لا يغيّر ميزات D)،
NaN + قناع توفّر قبل بداية metrics، وقراءة الطوابع ms/µs بنفس النتيجة.
"""
import gzip
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import intraday_features as itf  # noqa: E402

HEADER = ["timestamp", "datetime_utc", "open", "high", "low", "close", "volume",
          "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]


def make_klines(days=10, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed)
    n = days * 96
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.r_[100.0, close[:-1]]
    vol = rng.random(n) * 10 + 1
    price = (open_ + close) / 2
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) * 1.001, "low": np.minimum(open_, close) * 0.999,
        "close": close, "volume": vol, "quote_volume": vol * price,
        "trades": rng.integers(50, 150, n).astype(float), "taker_buy_volume": vol * rng.random(n),
        "taker_buy_quote_volume": np.nan}, index=idx)


def write_kline_csv(k, path, unit="ms"):
    mul = {"ms": 1, "us": 1000}[unit]
    df = k.copy()
    ms = (df.index.asi8 // 10 ** 6) if df.index.asi8.max() > 1e17 else df.index.asi8
    df.insert(0, "timestamp", ms * mul)
    df.insert(1, "datetime_utc", df.index.strftime("%Y-%m-%d %H:%M:%S"))
    with gzip.open(path, "wt") as f:
        df[HEADER].to_csv(f, index=False)


class TestTimestamps(unittest.TestCase):
    def test_ms_us_s_and_iso(self):
        t = pd.Timestamp("2024-03-01 12:15:00", tz="UTC")
        ms = t.value // 10 ** 6
        for vals in ([ms], [ms * 1000], [ms // 1000], ["2024-03-01 12:15:00+00:00"], ["2024-03-01 12:15:00"]):
            self.assertEqual(itf.to_utc_index(vals)[0], t, vals)
        mixed = itf.to_utc_index([ms, (ms + 900_000) * 1000])         # ملف مختلط ms ثم µs
        self.assertEqual(list(mixed), [t, t + pd.Timedelta("15min")])
        self.assertEqual(itf.epoch_unit(np.array([ms * 1000])), "us")
        # دقّة صحيحة: calc_time من أرشيف التمويل الخام قد يحمل +1ms (float64 كان يُفسدها)
        self.assertEqual(itf.to_utc_index([1785542400001])[0],
                         pd.Timestamp("2026-08-01", tz="UTC") + pd.Timedelta("1ms"))

    def test_read_klines_ms_equals_us_and_dedup(self):
        k = make_klines(3)
        with tempfile.TemporaryDirectory() as d:
            p_ms, p_us = os.path.join(d, "A.csv.gz"), os.path.join(d, "B.csv.gz")
            dup = pd.concat([k.iloc[:5], k])                           # صفوف مكرّرة وغير مرتّبة
            write_kline_csv(dup, p_ms, "ms")
            write_kline_csv(k, p_us, "us")
            a, b = itf.read_klines(p_ms), itf.read_klines(p_us)
            self.assertEqual(len(a), len(k))
            self.assertTrue(a.index.is_monotonic_increasing)
            pd.testing.assert_frame_equal(a, b)
            fa = itf.daily_intraday_features(a)
            fb = itf.daily_intraday_features(b)
            pd.testing.assert_frame_equal(fa, fb)
            self.assertIsNotNone(itf.find_file(d, "A"))
            self.assertIsNone(itf.find_file(d, "ZZZ"))


class TestIntradayAggregation(unittest.TestCase):
    def setUp(self):
        self.k = make_klines(10)
        self.f = itf.daily_intraday_features(self.k, trades_z_window=5)

    def test_values_match_manual(self):
        d = pd.Timestamp("2024-01-04", tz="UTC")
        day = self.k[(self.k.index >= d) & (self.k.index < d + pd.Timedelta("1D"))]
        row = self.f.loc[d]
        self.assertAlmostEqual(row.ITD_TAKER_BUY_RATIO, day.taker_buy_volume.sum() / day.volume.sum())
        self.assertAlmostEqual(row.ITD_TRADES_LOG, np.log1p(day.trades.sum()))
        vwap = day.quote_volume.sum() / day.volume.sum()
        self.assertAlmostEqual(row.VWAP_DEVIATION, (day.close.iloc[-1] - vwap) / vwap)
        top4 = np.sort(day.volume.to_numpy())[::-1][:4].sum() / day.volume.sum()
        self.assertAlmostEqual(row.ITD_VOL_TOPK, top4)
        r = np.diff(np.log(np.r_[day.open.iloc[0], day.close.to_numpy()]))
        self.assertAlmostEqual(row.ITD_RVOL, 100 * np.sqrt((r ** 2).sum()))
        hv = day.volume.groupby(day.index.floor("h")).sum()
        self.assertAlmostEqual(row.VOL_CONC_HHI, ((hv / hv.sum()) ** 2).sum())
        hc = day.close.groupby(day.index.floor("h")).last()
        path = np.abs(np.diff(np.log(np.r_[day.open.iloc[0], hc.to_numpy()]))).sum()
        self.assertAlmostEqual(row.EFF_RATIO_24H, abs(np.log(day.close.iloc[-1] / day.open.iloc[0])) / path)
        self.assertEqual(row.ITD_available, 1.0)
        self.assertTrue(0 <= row.EFF_RATIO_24H <= 1)
        # z مقابل الأيام السابقة فقط (نافذة 5، حدّ أدنى 5 أيام سابقة)
        self.assertTrue(np.isnan(row.ITD_TRADES_Z))                     # 3 أيام سابقة فقط
        d8 = pd.Timestamp("2024-01-08", tz="UTC")
        past = self.f.ITD_TRADES_LOG.loc[:d8].iloc[-6:-1]
        self.assertAlmostEqual(self.f.loc[d8, "ITD_TRADES_Z"],
                               (self.f.loc[d8, "ITD_TRADES_LOG"] - past.mean()) / past.std())

    def test_no_lookahead(self):
        d = pd.Timestamp("2024-01-05", tz="UTC")
        k2 = self.k.copy()
        after = k2.index >= d + pd.Timedelta("1D")
        k2.loc[after, ["close", "open", "high", "low"]] *= 3.0
        k2.loc[after, ["volume", "quote_volume", "trades", "taker_buy_volume"]] *= 7.0
        k2 = k2.drop(k2.index[after][::3])                              # وحذف بعض الشموع اللاحقة
        f2 = itf.daily_intraday_features(k2, trades_z_window=5)
        pd.testing.assert_frame_equal(self.f.loc[:d], f2.loc[:d])
        # وعلى شموع يومية: شمعة D (إغلاقها D+1 00:00) ترى ميزات D، لا D+1
        bars = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        a1 = itf.align_daily_to_bars(self.f, bars, "1D")
        a2 = itf.align_daily_to_bars(f2, bars, "1D")
        pd.testing.assert_frame_equal(a1.loc[:d], a2.loc[:d])
        pd.testing.assert_series_equal(a1.loc[d], self.f.loc[d], check_names=False)
        # فريم 4h: شمعة 2024-01-05 20:00 (إغلاقها D+1 00:00) ترى D، وشمعة 16:00 ترى D-1
        h4 = pd.date_range("2024-01-05", periods=6, freq="4h", tz="UTC")
        a4 = itf.align_daily_to_bars(self.f, h4, "4h")
        self.assertAlmostEqual(a4.iloc[-1].ITD_RVOL, self.f.loc[d].ITD_RVOL)
        self.assertAlmostEqual(a4.iloc[-2].ITD_RVOL, self.f.loc[d - pd.Timedelta("1D")].ITD_RVOL)

    def test_partial_day_masked(self):
        k2 = self.k.drop(self.k.index[(self.k.index >= "2024-01-03") & (self.k.index < "2024-01-03 12:00")])
        f2 = itf.daily_intraday_features(k2)
        row = f2.loc[pd.Timestamp("2024-01-03", tz="UTC")]
        self.assertEqual(row.ITD_available, 0.0)
        self.assertTrue(np.isnan(row.ITD_RVOL))
        # أيام كاملة الغياب تظهر صفوفاً NaN (لا تُحذف)
        k3 = self.k.drop(self.k.index[(self.k.index >= "2024-01-06") & (self.k.index < "2024-01-08")])
        f3 = itf.daily_intraday_features(k3)
        self.assertEqual(len(f3), 10)
        self.assertEqual(f3.ITD_available.sum(), 8)

    def test_legacy_file_without_taker(self):
        k = make_klines(2)
        k[["quote_volume", "trades", "taker_buy_volume"]] = np.nan
        f = itf.daily_intraday_features(k)
        self.assertTrue(f.ITD_TAKER_BUY_RATIO.isna().all())
        self.assertTrue(f.VWAP_DEVIATION.notna().all())                  # بديل السعر النموذجي
        self.assertTrue((f.ITD_available == 1).all())


class TestFundingMetrics(unittest.TestCase):
    def setUp(self):
        self.days = pd.date_range("2023-12-25", "2024-01-10", freq="D", tz="UTC")
        ft = pd.date_range("2023-12-01", "2024-01-11", freq="8h", tz="UTC")
        self.fr = pd.DataFrame({"funding_rate": np.linspace(1e-4, 3e-4, len(ft))}, index=ft)
        mt = pd.date_range("2024-01-01", "2024-01-11", freq="1h", tz="UTC")     # metrics تبدأ 2024-01-01
        rng = np.random.default_rng(1)
        self.met = pd.DataFrame({
            "sum_open_interest": np.linspace(100, 200, len(mt)),
            "sum_open_interest_value": 1.0,
            "count_toptrader_long_short_ratio": 1 + rng.random(len(mt)),
            "sum_toptrader_long_short_ratio": 1 + rng.random(len(mt)),
            "count_long_short_ratio": 1 + rng.random(len(mt)),
            "sum_taker_long_short_vol_ratio": 0.5 + rng.random(len(mt))}, index=mt)

    def test_funding_alignment_and_sums(self):
        f = itf.funding_bar_features(self.fr, self.days, "1D")
        d = pd.Timestamp("2024-01-03", tz="UTC")
        ev = self.fr.funding_rate
        # آخر حدث قبل إغلاق D (= 16:00 من D؛ تسوية D+1 00:00 لا تُنسب لـ D)
        self.assertAlmostEqual(f.loc[d, "FUND_rate"], ev.loc[d + pd.Timedelta("16h")])
        self.assertAlmostEqual(f.loc[d, "FUND_sum_1d"], ev.loc[d:d + pd.Timedelta("16h")].sum())
        self.assertAlmostEqual(f.loc[d, "FUND_sum_3d"],
                               ev.loc[d - pd.Timedelta("2D"):d + pd.Timedelta("16h")].sum())
        self.assertTrue((f.FUND_available == 1).all())
        # لا تسرّب: تغيير التمويل بعد إغلاق D لا يغيّر D
        fr2 = self.fr.copy()
        fr2.loc[fr2.index >= d + pd.Timedelta("1D"), "funding_rate"] = 0.05
        g = itf.funding_bar_features(fr2, self.days, "1D")
        pd.testing.assert_frame_equal(f.loc[:d], g.loc[:d])
        # بلا أرشيف ⇒ NaN + قناع 0
        e = itf.funding_bar_features(None, self.days, "1D")
        self.assertTrue(e.FUND_rate.isna().all() and (e.FUND_available == 0).all())

    def test_metrics_nan_before_start_and_no_lookahead(self):
        m = itf.metrics_bar_features(self.met, self.days, "1D")
        before = m.index < pd.Timestamp("2024-01-01", tz="UTC")
        self.assertTrue((m.loc[before, "MET_available"] == 0).all())
        self.assertTrue(m.loc[before, ["LSR_GLOBAL", "LSR_TOP_POS", "TAKER_LSR_1d"]].isna().all().all())
        self.assertTrue((m.loc[~before, "MET_available"] == 1).all())
        self.assertEqual(len(m), len(self.days))                          # لا صفوف محذوفة
        d = pd.Timestamp("2024-01-04", tz="UTC")
        # آخر ساعة متاحة عند إغلاق D هي الساعة 23:00 (لقطتها ≈23:55، متاحة عند 24:00)
        self.assertAlmostEqual(m.loc[d, "LSR_GLOBAL"],
                               np.log(self.met.loc[d + pd.Timedelta("23h"), "count_long_short_ratio"]))
        tk = np.log(self.met.loc[d:d + pd.Timedelta("23h"), "sum_taker_long_short_vol_ratio"]).mean()
        self.assertAlmostEqual(m.loc[d, "TAKER_LSR_1d"], tk)
        met2 = self.met.copy()
        met2.loc[met2.index >= d + pd.Timedelta("1D")] *= 5.0
        m2 = itf.metrics_bar_features(met2, self.days, "1D")
        pd.testing.assert_frame_equal(m.loc[:d], m2.loc[:d])
        oi = itf.oi_bar_features(self.met.sum_open_interest, self.days, "1D")
        oi2 = itf.oi_bar_features(met2.sum_open_interest, self.days, "1D")
        pd.testing.assert_frame_equal(oi.loc[:d], oi2.loc[:d])
        self.assertTrue((oi.loc[before, "OI_available"] == 0).all())

    def test_delisted_zeros_and_staleness(self):
        met = self.met.copy()
        cut = pd.Timestamp("2024-01-06", tz="UTC")
        met.loc[met.index >= cut, "sum_open_interest"] = 0.0             # 0E-16 بعد الشطب
        met.loc[met.index >= cut, ["count_long_short_ratio", "sum_toptrader_long_short_ratio"]] = np.nan
        m = itf.metrics_bar_features(met, self.days, "1D")
        oi = itf.oi_bar_features(met.sum_open_interest, self.days, "1D")
        late = m.index >= cut + pd.Timedelta("1D")                         # أقدم من max_age = يوم
        self.assertTrue((m.loc[late, "MET_available"] == 0).all())
        self.assertTrue((oi.loc[late, "OI_available"] == 0).all())

    def test_read_archive_formats(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "X.csv.gz")
            with gzip.open(p, "wt") as f:
                f.write("timestamp,sum_open_interest,count_long_short_ratio\n"
                        "2024-07-09 08:00:00+00:00,0E-16,\n"
                        "2024-07-09 07:00:00+00:00,5.5,1.2\n"
                        "2024-07-09 07:00:00+00:00,6.5,1.3\n")
            a = itf.read_archive(p)
            self.assertEqual(len(a), 2)
            self.assertEqual(a.index[0], pd.Timestamp("2024-07-09 07:00", tz="UTC"))
            self.assertEqual(a.iloc[0].sum_open_interest, 6.5)            # التكرار: يُبقى الأخير
            self.assertEqual(itf.infer_period(pd.date_range("2024", periods=10, freq="h")), pd.Timedelta("1h"))


if __name__ == "__main__":
    unittest.main()

"""
intraday_features.py — ميزات المرحلة ٢ من أرشيف Binance الكامل (شموع 15m + تمويل + OI + metrics).

وحدة مستقلّة (pandas/numpy فقط) يستوردها crypto_data_pipeline_v6.ipynb عبر
``_phase2_module()``، وتُختبَر مباشرة في tests/test_intraday_features.py.

صيغ الملفات الفعلية (كما يكتبها tools/fetch_history_vision_colab.py، ومُتحقَّق منها على عيّنات Drive):
    history_15m/<S>.csv.gz   timestamp (ms، وقت فتح الشمعة)، datetime_utc، open، high، low، close، volume،
                             quote_volume، trades، taker_buy_volume، taker_buy_quote_volume
                             (ملفات قديمة قد تحمل الأعمدة السبعة الأولى فقط ← أعمدة taker/trades = NaN)
    funding_rate/<S>.csv.gz  timestamp ("YYYY-mm-dd HH:MM:SS+00:00" = وقت التسوية)، funding_rate
    open_interest/<S>.csv.gz timestamp (بداية الساعة)، open_interest
    futures_metrics/<S>.csv.gz timestamp (بداية الساعة)، sum_open_interest، sum_open_interest_value،
                             count_toptrader_long_short_ratio، sum_toptrader_long_short_ratio،
                             count_long_short_ratio، sum_taker_long_short_vol_ratio
      ⚠️ صف metrics/OI بطابع الساعة h يحمل **آخر** لقطة 5m داخل [h, h+1h) (مثلاً 00:55) — فهو معلوم
         فقط عند h+period، لا عند h. لذلك وقت التوفّر = الطابع + الفترة (لا الطابع نفسه).
      ⚠️ عملات مشطوبة: صفوف بـ sum_open_interest = 0E-16 ونسب فارغة بعد الشطب ← تُعامَل NaN.

مبدأ عدم التسرّب (look-ahead): كل قيمة تُنسب لشمعة (فتحها t، إغلاقها t+L) فقط إن كان وقت توفّرها ≤ t+L:
    * شموع 15m: الشمعة معلومة عند إغلاقها (فتح + 15m)؛ ميزات اليوم D تُبنى من شموع فتحها ∈ [D, D+1)،
      وتُتاح عند D+1 00:00 = إغلاق شمعة D اليومية.
    * التمويل: حدث التسوية بطابع τ يُعدّ متاحاً بعد τ مباشرة (τ + 1ms) — أي أن تسوية D+1 00:00 تُنسب
      ليوم D+1 لا D (تحفّظ مقصود: الطوابع مقطوعة للثانية، والأرشيف الخام قد يحمل 00:00:00.001).
    * metrics/OI: متاحة عند الطابع + الفترة (ساعة افتراضياً).
    * أي قيمة أقدم من ``max_age`` عند إغلاق الشمعة تُعدّ مفقودة (NaN + قناع 0)، لا تُمدَّد بلا حدّ.
الدوال هنا تُرجع NaN حيث لا بيانات + عمود ``*_available`` (1/0). الدفتر يملأ NaN بصفر محايد (نفس
أسلوب FUND_available القائم) لأن النوافذ لا تقبل NaN.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

DAY = pd.Timedelta("1D")
KLINE_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
              "taker_buy_volume", "taker_buy_quote_volume"]
METRICS_COLS = ["sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]

#: أسماء الأعمدة التي تُنتجها هذه الوحدة (الدفتر يُدرجها في feature_order).
INTRADAY_SLOT_COLS = ["EFF_RATIO_24H", "VWAP_DEVIATION", "VOL_CONC_HHI"]     # فتحات قائمة في الدفتر
INTRADAY_NEW_COLS = ["ITD_TAKER_BUY_RATIO", "ITD_TRADES_LOG", "ITD_TRADES_Z", "ITD_RVOL",
                     "ITD_VOL_TOPK", "ITD_available"]
FUNDING_COLS = ["FUND_rate", "FUND_rate_z", "FUND_sum_1d", "FUND_sum_3d", "FUND_available"]
METRICS_FEATURE_COLS = ["LSR_TOP_ACCT", "LSR_TOP_POS", "LSR_GLOBAL", "LSR_GLOBAL_chg_1",
                        "TAKER_LSR_1d", "MET_available"]


# ═══════════════ الزمن ═══════════════
def epoch_unit(values: np.ndarray) -> str:
    """وحدة طوابع رقمية من مقدارها: s / ms / us / ns (أرشيف Binance: ms، والفوري منذ 2025: us)."""
    v = np.abs(np.asarray(values, dtype="float64"))
    v = v[np.isfinite(v) & (v > 0)]
    if not v.size:
        return "ms"
    m = float(np.median(v))
    if m >= 1e17:
        return "ns"
    if m >= 1e14:
        return "us"
    if m >= 1e11:
        return "ms"
    return "s"


def to_utc_index(values) -> pd.DatetimeIndex:
    """أي عمود طوابع (أرقام ms/us/s/ns — بوحدة تُكشف **لكل قيمة** فيُقبل ملف مختلط — أو نصوص ISO)
    ← DatetimeIndex بتوقيت UTC ودقّة ns. القيم غير الصالحة ← NaT."""
    s = pd.Series(values).reset_index(drop=True)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.5:
        # حساب صحيح (int64) لا float: ms×1e6 يتجاوز دقّة float64 (1785542400001ms كان يصير …000999936ns)
        ok = num.notna().to_numpy()
        iv = np.zeros(len(num), dtype="int64")
        if pd.api.types.is_integer_dtype(num.dtype):
            iv = num.to_numpy(dtype="int64")
        else:
            iv[ok] = np.round(num.to_numpy(dtype="float64")[ok]).astype("int64")
        a = np.abs(iv)
        ns = np.zeros(len(iv), dtype="int64")
        good = np.zeros(len(iv), dtype=bool)
        for lo, hi, mul in ((10 ** 17, None, 1), (10 ** 14, 10 ** 17, 10 ** 3),
                            (10 ** 11, 10 ** 14, 10 ** 6), (10 ** 8, 10 ** 11, 10 ** 9)):
            sel = ok & (a >= lo) & ((a < hi) if hi else True)
            ns[sel] = iv[sel] * mul
            good |= sel
        out = ns.view("datetime64[ns]").copy()
        out[~good] = np.datetime64("NaT")
        return pd.DatetimeIndex(out).tz_localize("UTC")
    idx = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return pd.DatetimeIndex(idx).as_unit("ns")


def as_utc(index) -> pd.DatetimeIndex:
    """فهرس زمني (مع/بلا tz) ← UTC ns. بلا tz يُفترض UTC."""
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return idx.as_unit("ns")


def _ns(index) -> np.ndarray:
    return as_utc(index).asi8


def tf_to_timedelta(tf: str) -> pd.Timedelta:
    """'1D' / '4h' / '15min' / '1h' ← Timedelta."""
    try:
        return pd.Timedelta(tf)
    except (ValueError, TypeError):
        return pd.Timedelta(pd.tseries.frequencies.to_offset(tf).nanos, unit="ns")


# ═══════════════ الملفات ═══════════════
def find_file(directory: Union[str, Path, None], symbol: str,
              pattern: str = "{name}.csv.gz") -> Optional[Path]:
    """<dir>/<S>.csv.gz ثم <dir>/<S>.csv (أو العكس حسب النمط). None إن لم يوجد."""
    if not directory:
        return None
    p = Path(directory) / pattern.format(name=symbol)
    alt = p.with_name(p.name[:-3]) if p.name.endswith(".gz") else p.with_name(p.name + ".gz")
    for c in (p, alt):
        if c.exists() and c.stat().st_size > 0:
            return c
    return None


def _finish(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    df.index = idx
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "timestamp"
    return df


def read_klines(path: Union[str, Path]) -> pd.DataFrame:
    """ملف شموع (csv أو csv.gz) ← DataFrame مفهرس بوقت **فتح** الشمعة (UTC)، أعمدة KLINE_COLS
    (الناقص = NaN)، بلا تكرار (يُبقى الأخير) ومرتّب. ``timestamp`` الرقمي مُفضَّل على ``datetime_utc``."""
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "timestamp" in raw.columns:
        idx = to_utc_index(raw["timestamp"])
    elif "open_time" in raw.columns:
        idx = to_utc_index(raw["open_time"])
    elif "datetime_utc" in raw.columns:
        idx = to_utc_index(raw["datetime_utc"])
    else:
        raise ValueError(f"{path}: لا عمود timestamp/open_time/datetime_utc — {list(raw.columns)}")
    if "count" in raw.columns and "trades" not in raw.columns:
        raw = raw.rename(columns={"count": "trades"})
    out = pd.DataFrame({c: pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else np.nan
                        for c in KLINE_COLS})
    out = _finish(out, idx)
    return out[(out["close"] > 0) | out["close"].isna()]


def read_archive(path: Union[str, Path], cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """أرشيف funding/OI/metrics ← DataFrame مفهرس بالطابع (UTC)، أعمدة رقمية. الطابع نص ISO أو رقم."""
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    tcol = next((c for c in ("timestamp", "calc_time", "create_time", "fundingtime", "datetime_utc")
                 if c in raw.columns), None)
    if tcol is None:
        raise ValueError(f"{path}: لا عمود طابع زمني — {list(raw.columns)}")
    if "last_funding_rate" in raw.columns and "funding_rate" not in raw.columns:
        raw = raw.rename(columns={"last_funding_rate": "funding_rate"})
    keep = [c for c in (cols or [c for c in raw.columns if c != tcol])]
    out = pd.DataFrame({c: pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else np.nan
                        for c in keep})
    return _finish(out, to_utc_index(raw[tcol]))


def infer_period(index, default: pd.Timedelta = pd.Timedelta("1h")) -> pd.Timedelta:
    """الفاصل المعتاد لأرشيف (وسيط الفروق)، محصوراً بين 5 دقائق ويوم."""
    ns = _ns(index)
    if ns.size < 3:
        return default
    d = np.diff(ns)
    d = d[d > 0]
    if not d.size:
        return default
    p = pd.Timedelta(int(np.median(d)), unit="ns")
    return min(max(p, pd.Timedelta("5min")), DAY)


# ═══════════════ المحاذاة (بلا تسرّب) ═══════════════
def asof_values(avail_ns: np.ndarray, values: np.ndarray, bar_close_ns: np.ndarray,
                max_age: Optional[pd.Timedelta] = None) -> np.ndarray:
    """لكل إغلاق شمعة: آخر قيمة **غير NaN** بوقت توفّر ≤ الإغلاق (وعمرها ≤ max_age إن حُدِّد)، وإلا NaN.
    ``avail_ns`` مرتّبة تصاعدياً."""
    values = np.asarray(values, dtype="float64")
    ok = np.isfinite(values)
    t, v = avail_ns[ok], values[ok]
    out = np.full(bar_close_ns.shape, np.nan)
    if not t.size:
        return out
    pos = np.searchsorted(t, bar_close_ns, side="right") - 1
    has = pos >= 0
    out[has] = v[pos[has]]
    if max_age is not None:
        age = np.full(bar_close_ns.shape, np.inf)
        age[has] = bar_close_ns[has] - t[pos[has]]
        out[age > max_age.value] = np.nan
    return out


def window_agg(avail_ns: np.ndarray, values: np.ndarray, bar_close_ns: np.ndarray,
               window: pd.Timedelta, how: str = "sum") -> np.ndarray:
    """تجميع القيم (غير NaN) بوقت توفّر ∈ (إغلاق − window، إغلاق]. لا قيم ⇒ NaN."""
    values = np.asarray(values, dtype="float64")
    ok = np.isfinite(values)
    t, v = avail_ns[ok], values[ok]
    hi = np.searchsorted(t, bar_close_ns, side="right")
    lo = np.searchsorted(t, bar_close_ns - window.value, side="right")
    csum = np.concatenate([[0.0], np.cumsum(v)])
    n = hi - lo
    s = csum[hi] - csum[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / n if how == "mean" else s
    return np.where(n > 0, out, np.nan)


def bar_closes(bar_index, bar_len: Union[str, pd.Timedelta]) -> np.ndarray:
    """أوقات إغلاق الشموع (ns) = فتح + طول الفريم (فهرس الدفتر = وقت الفتح)."""
    bl = tf_to_timedelta(bar_len) if isinstance(bar_len, str) else pd.Timedelta(bar_len)
    return _ns(bar_index) + bl.value


def align_daily_to_bars(daily: pd.DataFrame, bar_index, bar_len: Union[str, pd.Timedelta],
                        max_age: pd.Timedelta = DAY) -> pd.DataFrame:
    """ميزات يومية (فهرس = بداية اليوم D، متاحة عند D+1) ← على شموع أي فريم: لكل شمعة آخر يوم
    مكتمل عند إغلاقها. على فريم 1D: شمعة D تأخذ ميزات D نفسها (إغلاقها = D+1 00:00)."""
    closes = bar_closes(bar_index, bar_len)
    avail = _ns(daily.index) + DAY.value
    out = {c: asof_values(avail, daily[c].to_numpy(), closes, max_age) for c in daily.columns}
    return pd.DataFrame(out, index=bar_index)


# ═══════════════ ميزات يومية من شموع 15m ═══════════════
def daily_intraday_features(k: pd.DataFrame, bar: str = "15min", min_coverage: float = 0.75,
                            trades_z_window: int = 30, topk: int = 4,
                            hourly_for_eff_hhi: bool = True) -> pd.DataFrame:
    """شموع 15m (مخرج read_klines) ← إطار يومي مفهرس ببداية اليوم UTC. كل يوم من شموعه هو فقط
    (فتح ∈ [D, D+1))؛ لا شيء من يوم لاحق. يوم تغطيته < min_coverage من شموعه المتوقَّعة ⇒ NaN.

    الأعمدة:
        ITD_TAKER_BUY_RATIO  Σtaker_buy_volume / Σvolume            [0, 1]
        ITD_TRADES_LOG       log1p(Σtrades)
        ITD_TRADES_Z         (ITD_TRADES_LOG − متوسط الأيام السابقة) / انحرافها، نافذة trades_z_window
                             (تاريخ العملة نفسها، الأيام **السابقة** فقط)
        ITD_RVOL             100·√Σr² لعوائد 15m اللوغاريتمية داخل اليوم (أول عائد: close/open للشمعة الأولى)
        ITD_VOL_TOPK         حصة أعلى topk شموع 15m من حجم اليوم
        VWAP_DEVIATION       (إغلاق آخر شمعة − VWAP) / VWAP ؛ VWAP = Σquote_volume/Σvolume (بديل: السعر النموذجي)
        VOL_CONC_HHI         Σ(حصة حجم كل ساعة)² على 24 ساعة (كتعريف docs/research: ساعي)
        EFF_RATIO_24H        |log إغلاق آخر ساعة − log فتح اليوم| / Σ|Δlog| الساعي (مسار 24 خطوة ساعية)
        ITD_available        1 إن كان اليوم مكتمل التغطية، وإلا 0
    """
    cols = INTRADAY_SLOT_COLS + INTRADAY_NEW_COLS
    if k is None or k.empty:
        return pd.DataFrame(columns=cols, dtype="float64")
    k = k.copy()
    k.index = as_utc(k.index)
    bar_td = tf_to_timedelta(bar)
    per_day = int(round(DAY / bar_td))
    day = k.index.floor("D")
    g = k.groupby(day)

    n_bars = g["close"].count()
    vol = g["volume"].sum(min_count=1)
    qvol = g["quote_volume"].sum(min_count=1)
    tb = g["taker_buy_volume"].sum(min_count=1)
    trades = g["trades"].sum(min_count=1)
    first_open = g["open"].first()
    last_close = g["close"].last()

    with np.errstate(invalid="ignore", divide="ignore"):
        taker_ratio = (tb / vol).where(vol > 0)
        trades_log = np.log1p(trades.clip(lower=0))
        typ = (k["high"] + k["low"] + k["close"]) / 3.0
        vwap_typ = (typ * k["volume"]).groupby(day).sum(min_count=1) / vol
        vwap = (qvol / vol).where((vol > 0) & (qvol > 0), vwap_typ)
        vwap_dev = (last_close - vwap) / vwap

        lc = np.log(k["close"].where(k["close"] > 0))
        lo_ = np.log(k["open"].where(k["open"] > 0))
        prev = lc.groupby(day).shift(1)
        r = (lc - prev).fillna(lc - lo_)                 # الشمعة الأولى في اليوم: close/open
        rvol = 100.0 * np.sqrt((r ** 2).groupby(day).sum(min_count=1))

        share_top = k["volume"].groupby(day).apply(
            lambda x: np.sort(x.to_numpy(dtype="float64"))[::-1][:topk].sum()) / vol
        share_top = share_top.where(vol > 0)

        if hourly_for_eff_hhi:
            h = k.resample("1h").agg({"open": "first", "close": "last", "volume": "sum"})
            h = h[k["close"].resample("1h").count() > 0]
            hday = h.index.floor("D")
            hv = h["volume"].groupby(hday).sum()
            hhi = ((h["volume"] / hday.map(hv).to_numpy()) ** 2).groupby(hday).sum(min_count=1)
            hhi = hhi.where(hv > 0)
            hlc = np.log(h["close"].where(h["close"] > 0))
            hprev = hlc.groupby(hday).shift(1)
            hstep = (hlc - hprev).fillna(hlc - np.log(h["open"].where(h["open"] > 0))).abs()
            path = hstep.groupby(hday).sum(min_count=1)
        else:
            vshare = k["volume"] / day.map(vol).to_numpy()
            hhi = (vshare ** 2).groupby(day).sum(min_count=1).where(vol > 0)
            path = r.abs().groupby(day).sum(min_count=1)
        net = (np.log(last_close) - np.log(first_open)).abs()
        eff = (net / path).where(path > 0, 0.0).clip(0.0, 1.0)

    out = pd.DataFrame({
        "EFF_RATIO_24H": eff, "VWAP_DEVIATION": vwap_dev, "VOL_CONC_HHI": hhi,
        "ITD_TAKER_BUY_RATIO": taker_ratio, "ITD_TRADES_LOG": trades_log, "ITD_RVOL": rvol,
        "ITD_VOL_TOPK": share_top,
    })
    full = pd.date_range(out.index.min(), out.index.max(), freq="D")
    out = out.reindex(full)
    coverage = (n_bars.reindex(full).fillna(0) / per_day).clip(upper=1.0)
    ok = coverage >= min_coverage
    out = out.where(ok, np.nan)

    tl = out["ITD_TRADES_LOG"]
    past = tl.shift(1).rolling(trades_z_window, min_periods=max(5, trades_z_window // 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["ITD_TRADES_Z"] = ((tl - past.mean()) / past.std().replace(0, np.nan)).clip(-10, 10)
    out["ITD_available"] = (ok & out["ITD_RVOL"].notna()).astype("float64")
    out.index.name = "date"
    return out[cols]


# ═══════════════ التمويل ═══════════════
def funding_bar_features(fr: Optional[pd.DataFrame], bar_index, bar_len: Union[str, pd.Timedelta],
                         zscore_window: int = 90, max_age: pd.Timedelta = DAY) -> pd.DataFrame:
    """أرشيف funding (عمود funding_rate) ← على الشموع: آخر معدّل متاح قبل الإغلاق، درجته المعيارية
    المتدحرجة على أحداث التمويل نفسها (سابقة ← لا تسرّب)، ومجموع الأحداث في آخر يوم/3 أيام."""
    closes = bar_closes(bar_index, bar_len)
    out = pd.DataFrame(index=bar_index, columns=FUNDING_COLS, dtype="float64")
    if fr is None or fr.empty or "funding_rate" not in fr.columns:
        out["FUND_available"] = 0.0
        return out
    s = fr["funding_rate"].astype("float64").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    min_p = max(5, zscore_window // 3)
    mu = s.rolling(zscore_window, min_periods=min_p).mean()
    sd = s.rolling(zscore_window, min_periods=min_p).std()
    z = ((s - mu) / sd.replace(0, np.nan)).fillna(0.0)
    avail = _ns(s.index) + pd.Timedelta("1ms").value           # متاح بعد لحظة التسوية
    rate = asof_values(avail, s.to_numpy(), closes, max_age)
    out["FUND_rate"] = rate
    out["FUND_rate_z"] = np.where(np.isfinite(rate), asof_values(avail, z.to_numpy(), closes, max_age), np.nan)
    out["FUND_sum_1d"] = window_agg(avail, s.to_numpy(), closes, DAY)
    out["FUND_sum_3d"] = window_agg(avail, s.to_numpy(), closes, 3 * DAY)
    out["FUND_available"] = np.isfinite(rate).astype("float64")
    return out


# ═══════════════ الفائدة المفتوحة وmetrics ═══════════════
def _positive(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype("float64")
    return x.where(x > 0)


def oi_bar_features(oi: Optional[pd.Series], bar_index, bar_len: Union[str, pd.Timedelta],
                    change_bars: int = 1, period: Optional[pd.Timedelta] = None,
                    max_age: pd.Timedelta = DAY) -> pd.DataFrame:
    """OI (سلسلة مفهرسة ببداية فترتها) ← OI_chg_{k} = log OI(إغلاق t) − log OI(إغلاق t−k شموع)، وOI_available.
    OI ≤ 0 (عملات مشطوبة: 0E-16) = مفقود."""
    col = f"OI_chg_{int(change_bars)}"
    closes = bar_closes(bar_index, bar_len)
    out = pd.DataFrame(index=bar_index, columns=[col, "OI_available"], dtype="float64")
    if oi is None or len(oi) == 0:
        out["OI_available"] = 0.0
        return out
    s = _positive(oi).dropna().sort_index()
    period = infer_period(s.index) if period is None else period
    lvl = asof_values(_ns(s.index) + period.value, np.log(s.to_numpy()), closes, max_age)
    lvl = pd.Series(lvl, index=bar_index)
    out[col] = lvl - lvl.shift(int(change_bars))
    out["OI_available"] = lvl.notna().astype("float64")
    return out


def metrics_bar_features(met: Optional[pd.DataFrame], bar_index, bar_len: Union[str, pd.Timedelta],
                         period: Optional[pd.Timedelta] = None, max_age: pd.Timedelta = DAY) -> pd.DataFrame:
    """futures_metrics ← على الشموع (كلها لوغاريتم نِسَب؛ ≤0/فارغ = مفقود):
        LSR_TOP_ACCT      log(count_toptrader_long_short_ratio)   كبار المتداولين/حسابات — آخر قيمة
        LSR_TOP_POS       log(sum_toptrader_long_short_ratio)     كبار المتداولين/مراكز — آخر قيمة
        LSR_GLOBAL        log(count_long_short_ratio)             كل الحسابات — آخر قيمة
        LSR_GLOBAL_chg_1  تغيّر LSR_GLOBAL عن الشمعة السابقة
        TAKER_LSR_1d      متوسط log(sum_taker_long_short_vol_ratio) على لقطات آخر يوم
        MET_available     1 إن توفّرت LSR_GLOBAL أو LSR_TOP_POS عند الإغلاق
    قبل بداية الأرشيف (2024-01-01 افتراضياً) كل شيء NaN وMET_available = 0."""
    closes = bar_closes(bar_index, bar_len)
    out = pd.DataFrame(index=bar_index, columns=METRICS_FEATURE_COLS, dtype="float64")
    if met is None or met.empty:
        out["MET_available"] = 0.0
        return out
    met = met.sort_index()
    period = infer_period(met.index) if period is None else period
    avail = _ns(met.index) + period.value

    def _log(c):
        return np.log(_positive(met[c]).to_numpy()) if c in met.columns else np.full(len(met), np.nan)

    out["LSR_TOP_ACCT"] = asof_values(avail, _log("count_toptrader_long_short_ratio"), closes, max_age)
    out["LSR_TOP_POS"] = asof_values(avail, _log("sum_toptrader_long_short_ratio"), closes, max_age)
    out["LSR_GLOBAL"] = asof_values(avail, _log("count_long_short_ratio"), closes, max_age)
    out["LSR_GLOBAL_chg_1"] = out["LSR_GLOBAL"] - out["LSR_GLOBAL"].shift(1)
    out["TAKER_LSR_1d"] = window_agg(avail, _log("sum_taker_long_short_vol_ratio"), closes, DAY, how="mean")
    out["MET_available"] = (out["LSR_GLOBAL"].notna() | out["LSR_TOP_POS"].notna()).astype("float64")
    return out


# ═══════════════ واجهة لرمز واحد (يستدعيها الدفتر) ═══════════════
def load_intraday_daily(root: Union[str, Path], symbol: str, subdir: str = "history_15m",
                        pattern: str = "{name}.csv.gz", **kw) -> Optional[pd.DataFrame]:
    """<root>/<subdir>/<S>.csv.gz ← daily_intraday_features. None إن لم يوجد ملف."""
    path = find_file(Path(root) / subdir, symbol, pattern)
    if path is None:
        return None
    return daily_intraday_features(read_klines(path), **kw)


def fill_neutral(frame: pd.DataFrame, flag_cols: Iterable[str] = ()) -> pd.DataFrame:
    """NaN ← 0 (محايد) لكل الأعمدة؛ أعمدة الأعلام تبقى 0/1. (أسلوب FUND_available في الدفتر.)"""
    out = frame.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for c in flag_cols:
        if c in out.columns:
            out[c] = (out[c] > 0).astype("float64")
    return out

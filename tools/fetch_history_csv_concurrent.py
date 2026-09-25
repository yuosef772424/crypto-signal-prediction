"""
fetch_history_csv_concurrent.py
تحميل شموع كل عملات Binance Futures (USDT الدائمة) من تاريخ محدد حتى الآن، بالشكل الذي
يقرؤه خط الأنابيب (crypto_data_pipeline_v6) مباشرة — بلا أي تحويل يدوي بعده.

ما يُنتجه (كلها داخل مجلد Drive واحد عند تمرير --drive-root):
    <drive-root>/history_<interval>/<SYMBOL>.csv       ← CONFIG['drive_raw_dir'] = "history_<interval>"
    <drive-root>/crypto_data/asset_registry.csv         ← CONFIG['asset_registry_path'] (الافتراضي نفسه)
    <drive-root>/funding_rate/<SYMBOL>.csv   (--funding)        ← أرشيف FUND_rate/FUND_rate_z
    <drive-root>/open_interest/<SYMBOL>.csv  (--open-interest)  ← أرشيف OI_chg (آخر 30 يوماً فقط، تراكمي)

أعمدة ملف الشموع:
    timestamp (ms, وقت فتح الشمعة), datetime_utc, open, high, low, close, volume,
    quote_volume, trades, taker_buy_volume, taker_buy_quote_volume
    الأعمدة الأربعة الأخيرة (حجم بالدولار، عدد الصفقات، حجم المشترين المبادرين) تبقى في الملف
    لميزات تدفّق الأوامر لاحقاً؛ خط الأنابيب الحالي يقرأ OHLCV فقط ويتجاهل غيرها.

سجلّ الأصول (asset_registry.csv): صف لكل عملة — name (العمود الوحيد الذي يحتاجه خط الأنابيب)،
ثم حالة العقد، تاريخ الإدراج، أول/آخر شمعة، عدد الشموع، الفجوات، أطول فجوة، الشموع بلا حجم،
الشموع غير المنطقية (low > high...). يُعاد حسابه من الملفات الفعلية في كل تشغيل.

كل الطلبات عامة (klines وexchangeInfo وfundingRate وopenInterestHist) فلا يُحتاج مفتاح API.
إن وُجد adapters.binance_live في مجلد المشروع (كالنسخة السابقة) يُؤخذ منه عنوان الـ API فقط.

الاستخدام (كالسابق):
    export BINANCE_TESTNET=false          # اختياري؛ المفاتيح لم تعد مطلوبة

    python fetch_history_csv_concurrent.py --start 2025-01-01
    python fetch_history_csv_concurrent.py --start "2025-06-01 12:00" --interval 5m
    python fetch_history_csv_concurrent.py --start 2025-01-01 --end 2025-03-01
    python fetch_history_csv_concurrent.py --start 2025-01-01 --symbols BTCUSDT,ETHUSDT

    → data/history_<interval>_from_<start>/<SYMBOL>.csv  +  data/crypto_data/asset_registry.csv

إضافات اختيارية:
    --interval 1h --funding               # فريم إعداد الساعة في خط الأنابيب + أرشيف معدّل التمويل
    --open-interest                       # آخر 30 يوماً من الفائدة المفتوحة (تراكمي عبر التشغيلات)
    --include-delisted                    # ضمّ العقود المشطوبة التي ما زالت في exchangeInfo
    --drive-root "G:/My Drive"            # اكتب البنية مباشرة في Drive: history_<interval>/ ...
    --registry-only                       # أعد بناء سجلّ الأصول من الملفات الموجودة فقط

كل التواريخ بتوقيت UTC. إعادة تشغيل نفس الأمر تُكمل من آخر شمعة محفوظة لكل عملة.
"""

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

# ═══════════════ الإعدادات ═══════════════
START_DATE = "2017-01-01"       # تاريخ البداية (UTC) — يمكن تغييره أو تمريره عبر --start
END_DATE = ""                   # تاريخ النهاية (UTC)، فارغ = الآن
INTERVAL = "15m"                # لإعداد الساعة في خط الأنابيب مرّر --interval 1h
# المخرجات: data/history_<interval>_from_<start>/<SYMBOL>.csv
OUT_ROOT = str(Path(__file__).resolve().parent.parent / "data")
MAX_PER_REQUEST = 1500          # أقصى شموع في طلب klines واحد
MAX_CONCURRENT = 20             # طلبات متوازية لكل العملات معاً
SYMBOLS_IN_PARALLEL = 4         # عملات تُعالَج في نفس الوقت (للتحكم بالذاكرة)
WEIGHT_BUDGET_PER_MIN = 2000    # حد Binance 2400/دقيقة لكل IP — نترك هامشاً
FUNDING_REQ_PER_MIN = 90        # fundingRate: حد مستقل 500 طلب/5 دقائق لكل IP
MAX_RETRIES = 5
REQUEST_TIMEOUT = 20
RESUME = True                   # إذا كان ملف العملة موجوداً: أكمل من آخر شمعة فيه
ONLY_USDT_PERPETUAL = True      # فقط عقود USDT الدائمة (يُتجاهل عند تحديد --symbols، كالسابق)

CSV_HEADER = ["timestamp", "datetime_utc", "open", "high", "low", "close", "volume",
              "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]
LEGACY_HEADER = CSV_HEADER[:7]  # صيغة النسخة السابقة من هذا السكربت

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"


# ═══════════════ دوال مساعدة ═══════════════
def interval_to_ms(interval_str: str) -> int:
    mapping = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    try:
        return int(interval_str[:-1]) * mapping[interval_str[-1]]
    except (ValueError, KeyError):
        raise ValueError(f"فريم زمني غير مدعوم: {interval_str}")


def parse_date_ms(text: str) -> int:
    """يقبل: YYYY-MM-DD أو 'YYYY-MM-DD HH:MM' أو 'YYYY-MM-DD HH:MM:SS' (UTC)."""
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"صيغة التاريخ غير صحيحة: '{text}' — استخدم مثلاً 2025-01-01 أو '2025-01-01 12:00'")


def fmt_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def date_label(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    if (dt.hour, dt.minute, dt.second) == (0, 0, 0):
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d_%H%M")


def fmt_pandas_utc(ms: int) -> str:
    """نفس صيغة pandas عند حفظ طابع UTC — ما يكتبه ويقرؤه أرشيف funding/OI في خط الأنابيب."""
    return fmt_dt(ms) + "+00:00"


def kline_weight(limit: int) -> int:
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


class BinanceHTTPError(RuntimeError):
    def __init__(self, code: int, body: str, retry_after: Optional[float]):
        super().__init__(f"HTTP {code}: {body[:200]}")
        self.code, self.retry_after = code, retry_after


def http_get_json(base_url: str, path: str, params: Optional[dict] = None):
    url = f"{base_url}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "candle-history-fetcher"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        ra = e.headers.get("Retry-After") if e.headers else None
        raise BinanceHTTPError(e.code, e.read().decode("utf-8", "ignore"), float(ra) if ra else None)


# ═══════════════ عنوان الـ API ═══════════════
def resolve_base_url(testnet: bool) -> str:
    """كالنسخة السابقة: العنوان من كائن exchange (adapters.binance_live) إن وُجد في مجلد المشروع،
    وإلا mainnet/testnet حسب BINANCE_TESTNET. لا يُطبع أي مفتاح (كانت النسخة السابقة تطبع المفتاح والسر)."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from adapters.binance_live import BinanceFutures          # noqa: E402
        from core.exchange_api import ExchangeConfig             # noqa: E402
        ex = BinanceFutures(ExchangeConfig(api_key=os.environ.get("BINANCE_API_KEY", ""),
                                           api_secret=os.environ.get("BINANCE_API_SECRET", ""),
                                           testnet=testnet, request_timeout=REQUEST_TIMEOUT))
        p = urlparse(str(getattr(ex, "_base", "") or ""))
        if p.scheme and p.netloc:
            return f"{p.scheme}://{p.netloc}"
    except Exception:
        pass
    return TESTNET if testnet else MAINNET


# ═══════════════ قائمة العملات ═══════════════
def list_symbols(base_url: str, include_delisted: bool = False, only_usdt_perp: bool = True) -> Dict[str, dict]:
    """{symbol: {"onboard_ms", "status"}} لعقود USDT الدائمة.

    ⚠️ بلا include_delisted تُؤخذ العقود الجارية (TRADING) فقط — أي أن كل عملة شُطبت تغيب عن
    البيانات (انحياز البقاء: الاختبار التاريخي يرى الناجين فقط). exchangeInfo يُبقي بعض العقود
    المشطوبة مدةً بحالة SETTLING/CLOSE، فـ include_delisted يلتقط ما يزال مُدرَجاً منها فقط."""
    data = http_get_json(base_url, "/fapi/v1/exchangeInfo")
    out: Dict[str, dict] = {}
    for s in data.get("symbols", []):
        if only_usdt_perp and (s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT"):
            continue
        if s.get("status") != "TRADING" and not include_delisted:
            continue
        out[s["symbol"]] = {"onboard_ms": int(s.get("onboardDate", 0) or 0), "status": s.get("status", "")}
    return out


# ═══════════════ التحكم بمعدل الطلبات ═══════════════
class WeightLimiter:
    """نافذة منزلقة 60 ثانية على مجموع الـ weight (أو عدد الطلبات بوزن 1)."""

    def __init__(self, budget_per_min: int):
        self.budget, self.events, self.used = budget_per_min, deque(), 0
        self.lock = asyncio.Lock()
        self.pause_until = 0.0

    async def acquire(self, weight: int):
        while True:
            async with self.lock:
                now = time.monotonic()
                if now < self.pause_until:
                    wait = self.pause_until - now
                else:
                    while self.events and now - self.events[0][0] >= 60:
                        self.used -= self.events.popleft()[1]
                    if not self.events or self.used + weight <= self.budget:
                        self.events.append((now, weight))
                        self.used += weight
                        return
                    wait = 60 - (now - self.events[0][0]) + 0.05
            await asyncio.sleep(max(wait, 0.05))

    def pause(self, seconds: float):
        """بعد 429/418 من Binance: أوقف كل الطلبات (لا الطلب الفاشل وحده) — الحظر على مستوى الـ IP."""
        self.pause_until = max(self.pause_until, time.monotonic() + seconds)


async def request_with_retry(base_url, path, params, limiter: WeightLimiter, weight: int, req_sem):
    last_err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            async with req_sem:
                await limiter.acquire(weight)
                return await asyncio.to_thread(http_get_json, base_url, path, params)
        except BinanceHTTPError as e:
            last_err = e
            if e.code in (429, 418):
                limiter.pause(e.retry_after or 60)
            elif 400 <= e.code < 500:
                raise                                   # طلب خاطئ (رمز غير صالح...) — لا فائدة من الإعادة
            await asyncio.sleep(min(2 ** attempt, 30))
        except Exception as e:                          # شبكة/مهلة
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"فشل {path} {params.get('symbol', '')} بعد {MAX_RETRIES} محاولات: {last_err}")


# ═══════════════ ملفات CSV ═══════════════
def read_header(path: Path) -> Optional[List[str]]:
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return next(csv.reader(f), None)
    except OSError:
        return None


def read_last_timestamp(path: Path) -> Optional[int]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return None
            f.seek(max(0, size - 4096))
            lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
        return int(lines[-1].split(",")[0])
    except (ValueError, IndexError, OSError):
        return None


def kline_row(k: list) -> list:
    # Binance: [openTime, open, high, low, close, volume, closeTime, quoteVolume, trades,
    #           takerBuyBase, takerBuyQuote, ignore] — القيم نصوص كما وصلت (بلا فقد دقة)
    return [int(k[0]), fmt_dt(int(k[0])), k[1], k[2], k[3], k[4], k[5], k[7], k[8], k[9], k[10]]


def write_rows(path: Path, rows: List[list], append: bool):
    if append:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        return
    tmp = path.with_suffix(".csv.tmp")                 # كتابة ذرّية: لا ملف نصف مكتوب عند الانقطاع
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    os.replace(tmp, path)


# ═══════════════ تحميل شموع عملة واحدة ═══════════════
async def download_symbol(base_url, symbol, onboard_ms, start_ms, end_ms, interval, interval_ms,
                          out_dir: Path, limiter, req_sem):
    path = out_dir / f"{symbol}.csv"
    first_ts = max(start_ms, onboard_ms)
    resumed, note = False, ""

    if RESUME and path.exists():
        header = read_header(path)
        if header == CSV_HEADER:
            last_ts = read_last_timestamp(path)
            if last_ts is not None:
                first_ts = max(first_ts, last_ts + interval_ms)
                resumed = True
        elif header == LEGACY_HEADER:
            note = " | أُعيد تحميله كاملاً (صيغة قديمة بلا أعمدة الحجم الإضافية)"
        else:
            note = " | أُعيد تحميله كاملاً (رأس ملف غير معروف)"

    if first_ts >= end_ms:
        return "محدّث مسبقاً", 0, note

    span = MAX_PER_REQUEST * interval_ms
    tasks = [asyncio.create_task(request_with_retry(
        base_url, "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": MAX_PER_REQUEST,
         "startTime": s, "endTime": min(s + span - 1, end_ms)},
        limiter, kline_weight(MAX_PER_REQUEST), req_sem)) for s in range(first_ts, end_ms, span)]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        raise

    now_ms = int(time.time() * 1000)
    unique: Dict[int, list] = {}
    for batch in results:
        for k in batch or []:
            ts, close_time = int(k[0]), int(k[6])
            if first_ts <= ts <= end_ms and close_time < now_ms:      # الشموع المغلقة فقط
                unique[ts] = k
    rows = [kline_row(unique[t]) for t in sorted(unique)]
    if not rows:
        return "لا توجد بيانات جديدة", 0, note
    write_rows(path, rows, append=resumed)
    return ("تم (استكمال)" if resumed else "تم"), len(rows), note


# ═══════════════ أرشيف معدّل التمويل والفائدة المفتوحة ═══════════════
def _archive_last_ms(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    last = None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last = row.get("timestamp") or last
    if not last:
        return None
    try:
        return int(datetime.fromisoformat(last.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _merge_archive(path: Path, value_col: str, new_rows: Dict[str, str]) -> int:
    """يدمج بلا تكرار على timestamp ويحفظ ذرّياً — بنفس أعمدة أرشيف خط الأنابيب."""
    rows: Dict[str, str] = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("timestamp"):
                    rows[r["timestamp"]] = r.get(value_col, "")
    before = len(rows)
    rows.update(new_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", value_col])
        for ts in sorted(rows):
            w.writerow([ts, rows[ts]])
    os.replace(tmp, path)
    return len(rows) - before


async def download_funding(base_url, symbol, onboard_ms, start_ms, end_ms, funding_dir: Path,
                           limiter, req_sem) -> int:
    path = funding_dir / f"{symbol}.csv"
    last = _archive_last_ms(path)
    cursor = max(start_ms, onboard_ms, (last + 1) if last is not None else 0)
    new: Dict[str, str] = {}
    while cursor < end_ms:
        batch = await request_with_retry(base_url, "/fapi/v1/fundingRate",
                                         {"symbol": symbol, "startTime": cursor, "endTime": end_ms,
                                          "limit": 1000}, limiter, 1, req_sem)
        if not batch:
            break
        for r in batch:
            new[fmt_pandas_utc(int(r["fundingTime"]))] = r["fundingRate"]
        nxt = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000 or nxt <= cursor:
            break
        cursor = nxt
    return _merge_archive(path, "funding_rate", new) if new else 0


async def download_open_interest(base_url, symbol, oi_dir: Path, period: str, limiter, req_sem) -> int:
    """Binance يُعيد آخر 30 يوماً فقط — شغّله دورياً ليتراكم أرشيف أطول (كما في خط الأنابيب)."""
    new: Dict[str, str] = {}
    end = int(time.time() * 1000)
    begin = end - 30 * 86_400_000
    cursor = begin
    while cursor < end:
        batch = await request_with_retry(base_url, "/futures/data/openInterestHist",
                                         {"symbol": symbol, "period": period, "limit": 500,
                                          "startTime": cursor, "endTime": end}, limiter, 1, req_sem)
        if not batch:
            break
        for r in batch:
            new[fmt_pandas_utc(int(r["timestamp"]))] = r["sumOpenInterest"]
        nxt = int(batch[-1]["timestamp"]) + 1
        if len(batch) < 500 or nxt <= cursor:
            break
        cursor = nxt
    return _merge_archive(oi_dir / f"{symbol}.csv", "open_interest", new) if new else 0


# ═══════════════ سجلّ الأصول ═══════════════
def file_stats(path: Path, interval_ms: int) -> dict:
    """إحصاءات جودة من الملف الفعلي (لا من آخر تحميل فقط)."""
    n = gaps = missing = zero_vol = bad = 0
    first = last = prev = None
    max_gap = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None) or []
        idx = {c: i for i, c in enumerate(header)}
        for row in reader:
            try:
                ts = int(row[0])
                o, h, l, c, v = (float(row[idx[k]]) for k in ("open", "high", "low", "close", "volume"))
            except (ValueError, IndexError, KeyError):
                bad += 1
                continue
            n += 1
            first = ts if first is None else first
            last = ts
            if prev is not None and ts - prev > interval_ms:
                gaps += 1
                missing += (ts - prev) // interval_ms - 1
                max_gap = max(max_gap, ts - prev)
            prev = ts
            if v == 0:
                zero_vol += 1
            if not (l <= min(o, c) and max(o, c) <= h and l > 0):
                bad += 1
    return {"n_candles": n, "first_candle_utc": fmt_dt(first) if first else "",
            "last_candle_utc": fmt_dt(last) if last else "", "n_gaps": gaps, "missing_candles": missing,
            "max_gap_hours": round(max_gap / 3_600_000, 2), "zero_volume_candles": zero_vol,
            "bad_ohlc_rows": bad, "extended_columns": read_header(path) == CSV_HEADER}


def update_registry(registry_path: Path, out_dir: Path, interval: str, interval_ms: int,
                    symbols_meta: Dict[str, dict]) -> int:
    """يُعيد بناء سجلّ الأصول من كل ملفات out_dir، مع الإبقاء على أعمدة إضافية موجودة سابقاً
    (مثل file_id أو category) ودمجها حسب name."""
    existing: Dict[str, dict] = {}
    extra_cols: List[str] = []
    if registry_path.exists():
        with open(registry_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            extra_cols = [c for c in (reader.fieldnames or []) if c]
            for r in reader:
                if r.get("name"):
                    existing[r["name"]] = r
    for p in sorted(out_dir.glob("*.csv")):
        name = p.stem
        row = existing.get(name, {"name": name})
        meta = symbols_meta.get(name, {})
        row.update({"status": meta.get("status", row.get("status", "")),
                    "onboard_date_utc": fmt_dt(meta["onboard_ms"]) if meta.get("onboard_ms")
                    else row.get("onboard_date_utc", ""),
                    "interval": interval, "file": p.name})
        row.update(file_stats(p, interval_ms))
        existing[name] = row
    base_cols = ["name", "status", "onboard_date_utc", "first_candle_utc", "last_candle_utc", "interval",
                 "n_candles", "n_gaps", "missing_candles", "max_gap_hours", "zero_volume_candles",
                 "bad_ohlc_rows", "extended_columns", "file"]
    cols = base_cols + [c for c in extra_cols if c not in base_cols]
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for name in sorted(existing):
            w.writerow(existing[name])
    os.replace(tmp, registry_path)
    return len(existing)


# ═══════════════ نقطة البداية ═══════════════
def parse_args():
    p = argparse.ArgumentParser(description="تحميل شموع كل عملات Binance Futures من تاريخ محدد حتى الآن")
    p.add_argument("--start", default=START_DATE, help="تاريخ البداية UTC، مثال: 2025-01-01")
    p.add_argument("--end", default=END_DATE, help="تاريخ النهاية UTC (افتراضي: الآن)")
    p.add_argument("--interval", default=INTERVAL, help="الفريم الزمني، مثال: 1m 5m 1h")
    p.add_argument("--symbols", default="", help="عملات محددة مفصولة بفاصلة (افتراضي: كل العملات)")
    p.add_argument("--symbols-file", default="", help="ملف نصي: عملة في كل سطر")
    p.add_argument("--out-dir", default="", help="مجلد المخرجات (افتراضي: تلقائي داخل data/)")
    p.add_argument("--drive-root", default="", help="اختياري: جذر Drive (مثل 'G:/My Drive') لبنية خط الأنابيب")
    p.add_argument("--registry", default="", help="مسار asset_registry.csv (افتراضي: <root>/crypto_data/)")
    p.add_argument("--funding", action="store_true", help="حمّل أيضاً تاريخ معدّل التمويل الكامل")
    p.add_argument("--open-interest", action="store_true", help="أضف آخر 30 يوماً من الفائدة المفتوحة للأرشيف")
    p.add_argument("--oi-period", default="1h", help="دقة الفائدة المفتوحة (5m..1d)")
    p.add_argument("--include-delisted", action="store_true",
                   help="ضمّ العقود غير الجارية التي ما زالت في exchangeInfo (يخفّف انحياز البقاء)")
    p.add_argument("--registry-only", action="store_true", help="أعد بناء السجلّ من الملفات الموجودة فقط")
    p.add_argument("--testnet", action="store_true", default=os.environ.get("BINANCE_TESTNET", "").lower() == "true")
    p.add_argument("--base-url", default="", help="تجاوز عنوان الـ API (للاختبار)")
    return p.parse_args()


async def main():
    args = parse_args()
    base_url = args.base_url or resolve_base_url(args.testnet)
    interval, interval_ms = args.interval, interval_to_ms(args.interval)
    start_ms = parse_date_ms(args.start)
    end_ms = parse_date_ms(args.end) if args.end else int(time.time() * 1000)
    if start_ms >= end_ms:
        print("[خطأ] تاريخ البداية يجب أن يكون قبل تاريخ النهاية.")
        return

    if args.drive_root:                 # بنية خط الأنابيب مباشرة في Drive
        root = Path(args.drive_root)
        default_out = root / f"history_{interval}"
    else:                               # كالسابق: data/history_<interval>_from_<start>
        root = Path(OUT_ROOT)
        default_out = root / f"history_{interval}_from_{date_label(start_ms)}"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    registry_path = Path(args.registry) if args.registry else root / "crypto_data" / "asset_registry.csv"
    funding_dir, oi_dir = root / "funding_rate", root / "open_interest"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"الاتصال: {base_url}")
    print("[1] جلب قائمة العملات...")
    has_wanted = bool(args.symbols.strip() or args.symbols_file)
    try:
        all_symbols = await asyncio.to_thread(list_symbols, base_url, args.include_delisted or has_wanted,
                                              ONLY_USDT_PERPETUAL and not has_wanted)
    except Exception as e:
        print(f"[خطأ] تعذّر جلب exchangeInfo: {e}")
        return

    if args.registry_only:
        n = update_registry(registry_path, out_dir, interval, interval_ms, all_symbols)
        print(f"[✓] سجلّ الأصول: {registry_path} ({n} عملة)")
        return

    wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file:
        wanted += [ln.strip().upper() for ln in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
                   if ln.strip() and not ln.startswith("#")]
    if wanted:
        unknown = [s for s in wanted if s not in all_symbols]
        if unknown:
            print(f"[تنبيه] عملات غير موجودة وسيتم تجاهلها: {', '.join(unknown)}")
        symbols = {s: all_symbols[s] for s in dict.fromkeys(wanted) if s in all_symbols}
    else:
        symbols = dict(sorted(all_symbols.items()))
    if not symbols:
        print("[خطأ] لا توجد عملات للتحميل.")
        return

    span = MAX_PER_REQUEST * interval_ms
    est_requests = sum(math.ceil((end_ms - max(start_ms, m["onboard_ms"])) / span)
                       for m in symbols.values() if max(start_ms, m["onboard_ms"]) < end_ms)
    print(f"[2] {len(symbols)} عملة | {fmt_dt(start_ms)} → {fmt_dt(end_ms)} UTC | الفريم: {interval}")
    print(f"    الشموع: {out_dir}")
    print(f"    السجلّ: {registry_path}")
    print(f"    تقدير: ~{est_requests:,} طلب ≈ "
          f"{est_requests * kline_weight(MAX_PER_REQUEST) / WEIGHT_BUDGET_PER_MIN:,.0f} دقيقة (حسب حد الـ weight)")

    kl_limiter, fr_limiter = WeightLimiter(WEIGHT_BUDGET_PER_MIN), WeightLimiter(FUNDING_REQ_PER_MIN)
    req_sem, sym_sem = asyncio.Semaphore(MAX_CONCURRENT), asyncio.Semaphore(SYMBOLS_IN_PARALLEL)
    total, done, total_candles = len(symbols), 0, 0
    failed: List[str] = []

    async def worker(sym: str, meta: dict):
        nonlocal done, total_candles
        async with sym_sem:
            parts = []
            try:
                status, n, note = await download_symbol(base_url, sym, meta["onboard_ms"], start_ms, end_ms,
                                                        interval, interval_ms, out_dir, kl_limiter, req_sem)
                total_candles += n
                parts.append(f"{status} — {n:,} شمعة{note}")
                if args.funding:
                    k = await download_funding(base_url, sym, meta["onboard_ms"], start_ms, end_ms,
                                               funding_dir, fr_limiter, req_sem)
                    parts.append(f"تمويل +{k:,}")
                if args.open_interest:
                    k = await download_open_interest(base_url, sym, oi_dir, args.oi_period, fr_limiter, req_sem)
                    parts.append(f"فائدة مفتوحة +{k:,}")
            except Exception as e:
                failed.append(sym)
                parts.append(f"[فشل] {e}")
            done += 1
            print(f"[{done}/{total}] {sym}: " + " | ".join(parts))

    print("[3] بدء التحميل...")
    await asyncio.gather(*[worker(s, m) for s, m in symbols.items()])

    n_reg = update_registry(registry_path, out_dir, interval, interval_ms, all_symbols)
    print(f"\n[✓] انتهى. الشموع في: {out_dir} | شموع جديدة: {total_candles:,}")
    print(f"    سجلّ الأصول: {registry_path} ({n_reg} عملة، بإحصاءات الجودة)")
    if failed:
        print(f"    فشلت ({len(failed)}): {', '.join(failed)} — أعد تشغيل نفس الأمر ليُكمل.")
    rel = lambda p: p.relative_to(root).as_posix() if root in p.parents else str(p)
    if not args.drive_root:
        print(f"\n    ارفع محتوى {root} إلى جذر MyDrive (المجلد {out_dir.name} والمجلد crypto_data)،"
              " أو شغّل بـ --drive-root لتُكتب في Drive مباشرة.")
    print("\n    في خط الأنابيب (crypto_data_pipeline_v6):")
    print(f"      update_config({{'drive_raw_dir': '{rel(out_dir)}', "
          f"'asset_registry_path': '{rel(registry_path)}'}})")
    if interval == "1h":
        print(f"      apply_hourly_preset({{'drive_raw_dir': '{rel(out_dir)}'}})   # لإعداد الساعة (20-ب)")


if __name__ == "__main__":
    asyncio.run(main())

"""
fetch_history_vision_colab.py
تحميل شموع كل عملات Binance Futures (USDT الدائمة) من تاريخ محدد حتى الآن، بالشكل الذي
يقرؤه خط الأنابيب (crypto_data_pipeline_v6) مباشرة — بلا أي تحويل يدوي بعده.

هذه نسخة مدمجة: بنية التشغيل داخل Colab/Jupyter (خلية ملصوقة أو %run، كشف notebook، تشغيل
asyncio بخيط منفصل عند وجود حلقة أحداث تعمل مسبقاً، الكتابة التلقائية في Drive عند تركيبه) +
مصدر بيانات فعلي يتجاوز حظر Binance الجغرافي (HTTP 451) الذي تواجهه أغلب خوادم Colab:

    الشموع تُجلب من data.binance.vision — أرشيف Binance العلني (ملفات ZIP شهرية/يومية جاهزة)،
    وهو خدمة تخزين منفصلة عن واجهة التداول fapi.binance.com، غير خاضعة لنفس الحظر الجغرافي.
    الفجوة الوحيدة: بيانات اليوم الحالي (لم تُرفع كملف يومي بعد) — تُجلب عبر fapi.binance.com
    المباشر إن كان الوصول متاحاً من شبكتك (يُفحص تلقائياً مرة واحدة في البداية)، وإلا تبقى ناقصة
    حتى تُعاد المحاولة لاحقاً (أعد تشغيل نفس الأمر، فمنطق الاستكمال resume يكمل من حيث توقف).

    التمويل (--funding) والفائدة المفتوحة (--open-interest) يُجلبان أيضاً من الأرشيف العلني نفسه:
        معدّل التمويل  ← data/futures/um/monthly/fundingRate/<SYMBOL>/  (تاريخ كامل، ملف شهري)
        الفائدة المفتوحة ← data/futures/um/daily/metrics/<SYMBOL>/       (كل 5 دقائق منذ 2020-09، تُجمَّع ساعياً)
    فلا يعتمدان على fapi (المحجوب بـ 451 على كولاب)، ولا يقتصر OI على آخر 30 يوماً. ملفات metrics تحمل
    أيضاً نسب المراكز (long/short) ونسبة حجم المشترين/البائعين (taker) — تُحفظ في futures_metrics/.
    إن كان API المباشر متاحاً يُكمَّل به الشهر/اليوم الحالي فقط (الأرشيف يتأخر يوماً).

    إن كان النطاق data.binance.vision نفسه محجوباً (بعض الشبكات المؤسسية/السحابية)، يُستخدم تلقائياً
    نفس الأرشيف عبر مضيف التخزين s3-ap-northeast-1.amazonaws.com/data.binance.vision.

ما يُنتجه (كلها داخل مجلد Drive واحد عند تمرير --drive-root، أو تلقائياً على Colab):
    <drive-root>/history_<interval>/<SYMBOL>.csv.gz    ← CONFIG['drive_raw_dir'] = "history_<interval>"
                                                          CONFIG['drive_raw_pattern'] = "{name}.csv.gz"
    <drive-root>/crypto_data/asset_registry.csv         ← CONFIG['asset_registry_path']
    <drive-root>/crypto_data/gaps_report.csv            ← تقرير مفصّل بكل ثغرة موجودة
    <drive-root>/funding_rate/<SYMBOL>.csv.gz   (--funding)
    <drive-root>/open_interest/<SYMBOL>.csv.gz  (--open-interest: timestamp, open_interest — ساعي من الأرشيف)
    <drive-root>/futures_metrics/<SYMBOL>.csv.gz (--open-interest: كل أعمدة metrics، للاستخدام لاحقاً)
    <drive-root>/premium_index_<interval>/<SYMBOL>.csv.gz (--premium: مؤشر العلاوة بنفس فريم الشموع)

كل عملة = ملف واحد متصل (لا ملف لكل شهر/يوم): تُحمَّل ملفات ZIP الشهرية/اليومية لها بالتوازي، تُفك في
الذاكرة، تُدمج بلا تكرار وبترتيب زمني، تُضغط gzip وتُحفظ ذرّياً، ثم تُحرَّر الذاكرة قبل العملة التالية
(--workers عملات معاً × --downloads تحميل متزامن). pandas يقرأ .csv.gz مباشرة: pd.read_csv(path).
ملفات csv عادية من تشغيل سابق تُحوَّل إلى csv.gz تلقائياً (بلا إعادة تحميل). --no-gzip يُبقي csv عادياً.

أعمدة ملف الشموع:
    timestamp (ms, وقت فتح الشمعة), datetime_utc, open, high, low, close, volume,
    quote_volume, trades, taker_buy_volume, taker_buy_quote_volume

الاستخدام على Google Colab (خلية واحدة — tools/fetch_history_colab_cell.py فيها النسخة الكاملة):
    from google.colab import drive
    drive.mount('/content/drive')
    !wget -q -O fetch_history_vision_colab.py https://raw.githubusercontent.com/yuosef772424/crypto-signal-prediction/claude/charming-sagan-kswo2r/tools/fetch_history_vision_colab.py
    !python fetch_history_vision_colab.py --drive-root /content/drive/MyDrive --interval 15m \
        --start 2020-01-01 --funding --open-interest --oi-start 2021-01-01 --vision-only

    ملف symbols.txt (اختياري لكن أضمن من الاكتشاف التلقائي): رمز واحد بكل سطر، وأي سطر يبدأ
    بـ # يُتجاهل. بدونه يحاول السكربت الاكتشاف تلقائياً (exchangeInfo إن توفر الوصول، وإلا فهرسة
    data.binance.vision نفسها — وهذه تشمل العملات المشطوبة أيضاً تلقائياً لأن الأرشيف لا يحذف
    ملفاتها القديمة، فتُخفَّف مشكلة انحياز البقاء التي كانت تحتاج --include-delisted سابقاً).

إضافات اختيارية عبر سطر الأوامر:
    --interval 15m --funding              # الفريم الافتراضي 15m + أرشيف معدّل التمويل (شهري)
    --open-interest --oi-start 2021-01-01 # الفائدة المفتوحة + metrics من الأرشيف (منذ 2020-09)
    --premium                             # مؤشر العلاوة بنفس الفريم (يغطي الشهر الحالي الذي ينقص التمويل)
    --workers 4 --downloads 24            # التوازي: عملات معاً × ملفات ZIP متزامنة
    --include-delisted                    # مع اكتشاف عبر exchangeInfo فقط؛ فهرسة vision تشملها دائماً
    --drive-root "G:/My Drive"            # اكتب البنية مباشرة في Drive
    --registry-only                       # أعد بناء السجلّ + تقرير الثغرات من الملفات الموجودة فقط
    --fix-gaps                            # حاول إصلاح الثغرات المكتشفة عبر API المباشر (يحتاج شبكة غير محظورة)
    --vision-only                         # تخطَّ حتى فحص الاتصال المباشر (بدء أسرع إن كنت متأكداً من الحظر)

كل التواريخ UTC. إعادة تشغيل نفس الأمر تُكمل من آخر شمعة محفوظة لكل عملة، ويُعاد بناء
asset_registry.csv وgaps_report.csv من الملفات الفعلية في كل تشغيل.
"""

import argparse
import asyncio
import csv
import gc
import gzip
import io
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


def _in_notebook() -> bool:
    """داخل Jupyter/Colab (خلية ملصوقة أو %run)؟"""
    return "ipykernel" in sys.modules


def _script_dir() -> Path:
    """مجلد السكربت — أو مجلد العمل الحالي حين يُلصق الكود في خلية (لا __file__ هناك)."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


# ═══════════════ الإعدادات ═══════════════
START_DATE = "2017-01-01"
END_DATE = ""
INTERVAL = "15m"
OUT_ROOT = str(_script_dir().parent / "data")
# ── لتشغيله بلصقه في خلية Colab بلا وسائط: عدّل هذه القيم (سطر الأوامر يتجاوزها) ──
DRIVE_ROOT = ""                 # فارغ في Colab مع Drive مُركَّب → /content/drive/MyDrive تلقائياً
SYMBOLS = ""                    # "BTCUSDT,ETHUSDT" للتجربة؛ فارغ = اكتشاف/فهرسة تلقائية
FUNDING = False                 # True: حمّل أيضاً تاريخ معدّل التمويل (عبر API المباشر)
FIX_GAPS = False                # True: حاول إصلاح الثغرات المكتشفة تلقائياً (يحتاج شبكة غير محظورة)

MAX_PER_REQUEST = 1500          # أقصى شموع في طلب REST واحد (لإكمال اليوم الحالي فقط)
MAX_CONCURRENT = 20             # طلبات REST متوازية (تُستخدم فقط إن كان الوصول المباشر متاحاً)
VISION_CONCURRENT = 24          # تحميلات ZIP متوازية من data.binance.vision (لا حدّ معدّل معلن)
SYMBOLS_IN_PARALLEL = 4         # عملات تُعالَج معاً (كل عملة: تحميل ← فك ← دمج ← ضغط ← حفظ ← تحرير الذاكرة)
USE_GZIP = True                 # حفظ <SYMBOL>.csv.gz (أصغر ~3-4 مرات؛ pandas يقرؤه مباشرة) — False: csv عادي
GZIP_LEVEL = 6
WEIGHT_BUDGET_PER_MIN = 2000
FUNDING_REQ_PER_MIN = 90
MAX_RETRIES = 5
VISION_RETRIES = 3
REQUEST_TIMEOUT = 25
REST_PROBE_TIMEOUT = 8          # مهلة فحص توفر API المباشر مرة واحدة في البداية
RESUME = True
ONLY_USDT_PERPETUAL = True

CSV_HEADER = ["timestamp", "datetime_utc", "open", "high", "low", "close", "volume",
              "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]
LEGACY_HEADER = CSV_HEADER[:7]
PREMIUM_HEADER = ["timestamp", "datetime_utc", "open", "high", "low", "close"]   # مؤشر العلاوة (أساس التمويل)

MAINNET = "https://fapi.binance.com"
TESTNET = "https://testnet.binancefuture.com"
VISION_BASE = "https://data.binance.vision"
VISION_LIST_BASE = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
# نفس الملفات عبر مضيف التخزين مباشرة — بديل تلقائي حين يُحجب النطاق data.binance.vision نفسه
VISION_HOSTS = ("https://data.binance.vision", VISION_LIST_BASE)
VISION_PROBE_KEY = "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01.zip"
OI_START_DATE = "2024-01-01"    # بداية أرشيف الفائدة المفتوحة (ملف يومي لكل عملة — أبكر = تحميل أطول)


# ═══════════════ دوال زمنية مساعدة ═══════════════
def interval_to_ms(interval_str: str) -> int:
    mapping = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    try:
        return int(interval_str[:-1]) * mapping[interval_str[-1]]
    except (ValueError, KeyError):
        raise ValueError(f"فريم زمني غير مدعوم: {interval_str}")


def parse_date_ms(text: str) -> int:
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
    return fmt_dt(ms) + "+00:00"


def kline_weight(limit: int) -> int:
    if limit < 100:
        return 1
    if limit < 500:
        return 2
    if limit <= 1000:
        return 5
    return 10


def _ym(ms: int) -> Tuple[int, int]:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.year, dt.month


def month_bounds(year: int, month: int) -> Tuple[int, int]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def month_list(start_ms: int, end_ms: int) -> List[Tuple[int, int, int, int]]:
    """كل الأشهر التقويمية الكاملة (قبل بداية الشهر الحالي) اللازمة لتغطية [start_ms, end_ms)."""
    cur_month_start_ms, _ = month_bounds(*_ym(int(time.time() * 1000)))
    ceiling = min(end_ms, cur_month_start_ms)
    out = []
    y, m = _ym(start_ms)
    while True:
        ms_start, ms_end = month_bounds(y, m)
        if ms_start >= ceiling:
            break
        out.append((y, m, ms_start, ms_end))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def days_between(lo_ms: int, hi_ms: int) -> List[Tuple[int, int, int, int]]:
    """كل الأيام (UTC) بين lo_ms وhi_ms (حصري)، كل عنصر (سنة، شهر، يوم، بداية اليوم بالمللي ثانية)."""
    lo_day = (lo_ms // 86_400_000) * 86_400_000
    out = []
    d = lo_day
    while d < hi_ms:
        dt = datetime.fromtimestamp(d / 1000, tz=timezone.utc)
        out.append((dt.year, dt.month, dt.day, d))
        d += 86_400_000
    return out


# ═══════════════ طلبات HTTP عامة ═══════════════
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
            return __import__("json").loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        ra = e.headers.get("Retry-After") if e.headers else None
        raise BinanceHTTPError(e.code, e.read().decode("utf-8", "ignore"), float(ra) if ra else None)


def resolve_base_url(testnet: bool) -> str:
    """العنوان من adapters.binance_live إن وُجد في مجلد المشروع، وإلا mainnet/testnet."""
    try:
        sys.path.insert(0, str(_script_dir().parent))
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


async def probe_rest_access(base_url: str) -> Tuple[bool, str]:
    """فحص مرة واحدة فقط: هل fapi.binance.com متاحة من هذه الشبكة؟ يميّز حظر 451 الجغرافي عن أي
    عطل شبكة آخر، لرسالة واضحة بدل تكرار المحاولة الفاشلة لكل عملة على شبكة محظورة (كولاب مثلاً)."""
    try:
        await asyncio.wait_for(asyncio.to_thread(http_get_json, base_url, "/fapi/v1/ping"),
                               timeout=REST_PROBE_TIMEOUT)
        return True, ""
    except BinanceHTTPError as e:
        if e.code == 451:
            return False, "HTTP 451 — Binance يحجب هذه الشبكة جغرافياً (عناوين IP أمريكية غالباً، كخوادم كولاب)"
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ═══════════════ WeightLimiter + طلبات REST (لإكمال اليوم الحالي/التمويل/الفائدة المفتوحة) ═══════════════
class WeightLimiter:
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
                raise
            await asyncio.sleep(min(2 ** attempt, 30))
        except Exception as e:
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"فشل {path} {params.get('symbol', '')} بعد {MAX_RETRIES} محاولات: {last_err}")


def list_symbols_rest(base_url: str, include_delisted: bool, only_usdt_perp: bool) -> Dict[str, dict]:
    data = http_get_json(base_url, "/fapi/v1/exchangeInfo")
    out: Dict[str, dict] = {}
    for s in data.get("symbols", []):
        if only_usdt_perp and (s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT"):
            continue
        if s.get("status") != "TRADING" and not include_delisted:
            continue
        out[s["symbol"]] = {"onboard_ms": int(s.get("onboardDate", 0) or 0), "status": s.get("status", "")}
    return out


async def fetch_rest_klines(symbol: str, lo_ms: int, hi_ms: int, interval: str, interval_ms: int,
                            rest_ctx: dict) -> List[list]:
    if lo_ms >= hi_ms:
        return []
    span = MAX_PER_REQUEST * interval_ms
    tasks = [asyncio.create_task(request_with_retry(
        rest_ctx["base_url"], "/fapi/v1/klines",
        {"symbol": symbol, "interval": interval, "limit": MAX_PER_REQUEST,
         "startTime": s, "endTime": min(s + span - 1, hi_ms)},
        rest_ctx["limiter"], kline_weight(MAX_PER_REQUEST), rest_ctx["req_sem"]))
        for s in range(lo_ms, hi_ms, span)]
    out: List[list] = []
    for r in await asyncio.gather(*tasks):
        out.extend(r or [])
    return out


# ═══════════════ data.binance.vision (المصدر الأساسي — يتجاوز حظر 451) ═══════════════
def choose_vision_base() -> str:
    """أول مضيف أرشيف يعمل من هذه الشبكة (data.binance.vision ثم مضيف S3 نفسه)."""
    global VISION_BASE
    for host in VISION_HOSTS:
        try:
            req = urllib.request.Request(f"{host}/{VISION_PROBE_KEY}", headers={"User-Agent": "candle-history-fetcher"})
            with urllib.request.urlopen(req, timeout=REST_PROBE_TIMEOUT) as resp:
                resp.read(64)
            VISION_BASE = host
            return host
        except Exception:
            continue
    return VISION_BASE


def list_vision_keys(prefix: str) -> List[str]:
    """كل مفاتيح الأرشيف تحت prefix (فهرسة S3 مُقسَّمة بصفحات 1000) — لمعرفة الملفات الموجودة فعلاً بدل
    تجربة كل يوم/شهر (طلبات 404 لأيام قبل إدراج العملة)."""
    keys: List[str] = []
    marker = ""
    for _ in range(200):
        params = {"prefix": prefix}
        if marker:
            params["marker"] = marker
        url = VISION_LIST_BASE + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "candle-history-fetcher"})
        raw = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT).read()
        root = ET.fromstring(raw)
        truncated, page = False, []
        for child in root:
            tag = child.tag.split("}")[-1]
            if tag == "Contents":
                for gc in child:
                    if gc.tag.split("}")[-1] == "Key" and gc.text:
                        page.append(gc.text)
            elif tag == "IsTruncated":
                truncated = (child.text or "").strip().lower() == "true"
        keys += page
        if not truncated or not page:
            break
        marker = page[-1]
    return [k for k in keys if k.endswith(".zip")]

def vision_klines_url(symbol: str, interval: str, year: int, month: int, day: Optional[int] = None) -> str:
    if day is None:
        return (f"{VISION_BASE}/data/futures/um/monthly/klines/{symbol}/{interval}/"
                f"{symbol}-{interval}-{year:04d}-{month:02d}.zip")
    return (f"{VISION_BASE}/data/futures/um/daily/klines/{symbol}/{interval}/"
            f"{symbol}-{interval}-{year:04d}-{month:02d}-{day:02d}.zip")


def download_zip_rows(url: str, transform=None):
    """يُرجع None إن لم يوجد الملف بعد (شهر/يوم لم يُرفع، أو عملة غير موجودة في تلك الفترة) —
    حالة طبيعية متوقعة. يرفع استثناء فقط عند مشاكل شبكة حقيقية (لتُعاد المحاولة).
    transform(rows) يُطبَّق داخل خيط التحميل نفسه: الصفوف تُحوَّل فوراً لشكلها النهائي المضغوط
    (سطر نصي واحد لكل شمعة) فلا تبقى قوائم الحقول الخام في الذاكرة."""
    req = urllib.request.Request(url, headers={"User-Agent": "candle-history-fetcher"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            names = zf.namelist()
            if not names:
                return None
            with zf.open(names[0]) as f:
                text = f.read().decode("utf-8", errors="ignore")
    except zipfile.BadZipFile:
        return None
    del raw
    rows = list(csv.reader(text.splitlines()))
    del text
    if not rows:
        return None
    first_cell = (rows[0][0] if rows[0] else "").strip()
    if not first_cell.lstrip("-").isdigit():          # صف رأس نصي في الإصدارات الأحدث من الأرشيف
        rows = rows[1:]
    return transform(rows) if transform is not None else rows


async def fetch_zip_rows_async(url: str, sem: asyncio.Semaphore, transform=None):
    last_err: Optional[Exception] = None
    for attempt in range(VISION_RETRIES):
        try:
            async with sem:
                return await asyncio.to_thread(download_zip_rows, url, transform)
        except Exception as e:
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 15))
    print(f"    [تحذير] فشل تحميل {url}: {last_err}")
    return None


def kline_line(k: list) -> str:
    """صف أرشيف/REST ← سطر CSV_HEADER (بدون close_time وignore)."""
    ts = int(k[0])
    return f"{ts},{fmt_dt(ts)},{k[1]},{k[2]},{k[3]},{k[4]},{k[5]},{k[7]},{k[8]},{k[9]},{k[10]}"


def premium_line(k: list) -> str:
    ts = int(k[0])
    return f"{ts},{fmt_dt(ts)},{k[1]},{k[2]},{k[3]},{k[4]}"


def kline_row(k: list) -> list:
    return kline_line(k).split(",")


async def list_symbols_from_vision(sem: asyncio.Semaphore) -> Dict[str, dict]:
    """اكتشاف قائمة العملات من فهرسة أرشيف data.binance.vision نفسه (بلا حاجة لـ exchangeInfo).
    يشمل هذا تلقائياً العملات المشطوبة أيضاً لأن الأرشيف لا يحذف ملفاتها القديمة."""
    prefix = "data/futures/um/monthly/klines/"
    names: List[str] = []
    marker = ""
    for _ in range(20):
        params = {"prefix": prefix, "delimiter": "/"}
        if marker:
            params["marker"] = marker
        url = VISION_LIST_BASE + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "candle-history-fetcher"})
        async with sem:
            raw = await asyncio.to_thread(lambda r=req: urllib.request.urlopen(r, timeout=REQUEST_TIMEOUT).read())
        root = ET.fromstring(raw)
        truncated, next_marker = False, None
        for child in root:
            tag = child.tag.split("}")[-1]
            if tag == "CommonPrefixes":
                for gc in child:
                    if gc.tag.split("}")[-1] == "Prefix" and gc.text:
                        names.append(gc.text.rstrip("/").rsplit("/", 1)[-1])
            elif tag == "IsTruncated":
                truncated = (child.text or "").strip().lower() == "true"
            elif tag == "NextMarker":
                next_marker = child.text
        if not truncated:
            break
        marker = next_marker or (names[-1] + "/" if names else "")
        if not marker:
            break
    out: Dict[str, dict] = {}
    for s in names:
        if s.endswith("USDT") and "_" not in s:      # استبعاد عقود التسليم الفصلية BTCUSDT_250328 مثلاً
            out[s] = {"onboard_ms": 0, "status": ""}
    return out


async def list_series_keys(symbol: str, kind: str, interval: str, first_ts: int, end_ms: int) -> List[str]:
    """مفاتيح ZIP اللازمة لتغطية [first_ts, end_ms) لسلسلة واحدة (klines أو premiumIndexKlines):
    ملف شهري لكل شهر مكتمل منشور، وملفات يومية للأشهر غير المنشورة شهرياً (الشهر الحالي، أو شهر
    انتهى للتو ولم يُرفع بعد). الفهرسة تعرف الموجود فعلاً ⇒ لا طلبات 404 لأشهر قبل إدراج العملة."""
    now_ms = int(time.time() * 1000)
    today = fmt_dt((now_ms // 86_400_000) * 86_400_000)[:10]
    lo_ym, hi_ym = _ym(first_ts), _ym(min(end_ms, now_ms))
    monthly = await asyncio.to_thread(
        list_vision_keys, f"data/futures/um/monthly/{kind}/{symbol}/{interval}/")
    by_month = {_month_of_key(k): k for k in monthly if lo_ym <= _month_of_key(k) <= hi_ym}
    if monthly:
        first_listed = min(_month_of_key(k) for k in monthly)
        need = [(y, m) for (y, m, _a, _b) in month_list(first_ts, end_ms)
                if (y, m) >= first_listed and (y, m) not in by_month]
        if hi_ym not in by_month and hi_ym not in need:
            need.append(hi_ym)                                     # الشهر الحالي (لا ملف شهري له بعد)
        daily_prefixes = [f"data/futures/um/daily/{kind}/{symbol}/{interval}/{symbol}-{interval}-{y:04d}-{m:02d}"
                          for (y, m) in need]
    else:                                                          # عملة جديدة: ملفات يومية فقط
        daily_prefixes = [f"data/futures/um/daily/{kind}/{symbol}/{interval}/"]
    lo_day, hi_day = fmt_dt(first_ts)[:10], fmt_dt(end_ms)[:10]
    daily: List[str] = []
    for pref in daily_prefixes:
        for k in await asyncio.to_thread(list_vision_keys, pref):
            if lo_day <= k[-14:-4] <= hi_day and k[-14:-4] < today:
                daily.append(k)
    return sorted(by_month.values()) + sorted(daily)


def _migrate_variant(path: Path, header: List[str]) -> str:
    """ملف من تشغيل سابق بالصيغة الأخرى (csv ↔ csv.gz) بنفس الرأس ⇒ يُحوَّل كما هو (استكمال بلا إعادة تحميل)
    ثم يُحذف الأصل. برأس مختلف ⇒ يُحذف ويُعاد التحميل كاملاً."""
    old = other_variant(path)
    if path.exists() or not old.exists():
        return ""
    if read_header(old) != header:
        old.unlink()
        return " | أُعيد تحميله كاملاً (صيغة قديمة)"
    tmp = path.with_name(path.name + ".tmp")
    with _open_bin(old, "rb") as src, _open_bin(tmp, "wb", gz=path.name.endswith(".gz")) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(tmp, path)
    old.unlink()
    return " | حُوِّل من " + old.name


async def download_series_vision(symbol: str, kind: str, header: List[str], line_fn, onboard_ms: int,
                                 start_ms: int, end_ms: int, interval: str, interval_ms: int, out_dir: Path,
                                 vision_sem: asyncio.Semaphore,
                                 rest_ctx: Optional[dict]) -> Tuple[str, int, str]:
    """سلسلة واحدة لعملة واحدة: تحميل كل ملفات ZIP المطلوبة **بالتوازي** ← فك الضغط وتحويل كل صف لسطر نهائي
    (داخل خيوط التحميل) ← دمج بلا تكرار وترتيب زمني ← كتابة ملف واحد متصل <SYMBOL>.csv.gz ← تحرير الذاكرة."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = data_path(out_dir, symbol)
    first_ts = max(start_ms, onboard_ms)
    resumed, note = False, ""

    if RESUME:
        note = _migrate_variant(path, header)
        if path.exists():
            if read_header(path) == header:
                last_ts = read_last_timestamp(path)
                if last_ts is not None:
                    first_ts = max(first_ts, last_ts + interval_ms)
                    resumed = True
            else:
                note += " | أُعيد تحميله كاملاً (رأس غير معروف)"

    if first_ts >= end_ms:
        return "محدّث مسبقاً", 0, note

    now_ms = int(time.time() * 1000)
    today_start_ms = (now_ms // 86_400_000) * 86_400_000

    def transform(rows: List[list]) -> List[Tuple[int, str]]:
        out = []
        for k in rows:
            try:
                ts, close_time = int(k[0]), int(k[6])
                if first_ts <= ts <= end_ms and close_time < now_ms:
                    out.append((ts, line_fn(k)))
            except (ValueError, IndexError):
                continue
        return out

    # 1) كل الأشهر/الأيام المنشورة — تحميل متوازٍ (سقفه vision_sem المشترك بين كل العملات)
    keys = await list_series_keys(symbol, kind, interval, first_ts, end_ms)
    results = await asyncio.gather(*[fetch_zip_rows_async(f"{VISION_BASE}/{k}", vision_sem, transform)
                                     for k in keys])
    unique: Dict[int, str] = {}
    for part in results:
        unique.update(part or ())
    del results

    # 2) اليوم الحالي (الشموع فقط) — عبر API المباشر إن كان متاحاً
    tail_note = ""
    if kind == "klines" and end_ms > today_start_ms:
        if rest_ctx is not None:
            try:
                tail = await fetch_rest_klines(symbol, max(first_ts, today_start_ms), end_ms,
                                               interval, interval_ms, rest_ctx)
                unique.update(transform(tail))
            except Exception as e:
                tail_note = f" | تعذّر إكمال اليوم الحالي عبر API المباشر: {e}"
        else:
            tail_note = " | اليوم الحالي غير مكتمل (API المباشر غير متاح من هذه الشبكة)"

    n = len(unique)
    if not n:
        return "لا توجد بيانات جديدة", 0, note + tail_note
    lines = [unique[t] for t in sorted(unique)]
    del unique
    await asyncio.to_thread(write_lines, path, lines, resumed, header)   # الضغط خارج حلقة الأحداث
    del lines
    gc.collect()
    return ("تم (استكمال)" if resumed else "تم"), n, note + tail_note


async def download_symbol_vision(symbol: str, onboard_ms: int, start_ms: int, end_ms: int,
                                 interval: str, interval_ms: int, out_dir: Path,
                                 vision_sem: asyncio.Semaphore,
                                 rest_ctx: Optional[dict]) -> Tuple[str, int, str]:
    return await download_series_vision(symbol, "klines", CSV_HEADER, kline_line, onboard_ms, start_ms, end_ms,
                                        interval, interval_ms, out_dir, vision_sem, rest_ctx)


async def download_premium_vision(symbol: str, start_ms: int, end_ms: int, interval: str, interval_ms: int,
                                  premium_dir: Path, vision_sem: asyncio.Semaphore) -> int:
    """مؤشر العلاوة (premiumIndexKlines: (سعر العقد − المؤشر)/المؤشر) بنفس فريم الشموع — أساس معدّل التمويل
    لحظياً، ويغطي الشهر الحالي الذي لا يوفّر أرشيف التمويل الشهري بياناته بعد."""
    _s, n, _n = await download_series_vision(symbol, "premiumIndexKlines", PREMIUM_HEADER, premium_line, 0,
                                             start_ms, end_ms, interval, interval_ms, premium_dir, vision_sem, None)
    return n


# ═══════════════ التمويل والفائدة المفتوحة من الأرشيف العلني (بلا fapi) ═══════════════
def _month_of_key(key: str) -> Tuple[int, int]:
    y, m = key.rsplit("-", 2)[-2:]
    return int(y), int(m[:2])


async def download_funding_vision(symbol: str, start_ms: int, end_ms: int, funding_dir: Path,
                                  vision_sem: asyncio.Semaphore) -> int:
    """ملفات fundingRate الشهرية (calc_time, funding_interval_hours, last_funding_rate) ← أرشيف
    timestamp,funding_rate بنفس صيغة download_funding_rest (يقرؤه خط الأنابيب كما هو)."""
    last = _archive_last_ms(funding_dir, symbol)
    lo = max(start_ms, (last + 1) if last is not None else 0)
    keys = await asyncio.to_thread(list_vision_keys, f"data/futures/um/monthly/fundingRate/{symbol}/")
    lo_ym, hi_ym = _ym(lo), _ym(end_ms)
    keys = [k for k in keys if lo_ym <= _month_of_key(k) <= hi_ym]
    batches = await asyncio.gather(*[fetch_zip_rows_async(f"{VISION_BASE}/{k}", vision_sem) for k in keys])
    new: Dict[str, str] = {}
    for rows in batches:
        for r in rows or []:
            try:
                ts = int(r[0])
            except (ValueError, IndexError):
                continue
            if lo <= ts <= end_ms and len(r) >= 3:
                new[fmt_pandas_utc(ts)] = r[2]
    return _merge_archive(funding_dir, symbol, "funding_rate", new) if new else 0


METRICS_COLS = ["sum_open_interest", "sum_open_interest_value", "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio", "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]


def _read_table(d: Path, symbol: str, cols: List[str]) -> Dict[str, List[str]]:
    """أرشيف (timestamp + cols) بأيّ الصيغتين (csv أو csv.gz)."""
    rows: Dict[str, List[str]] = {}
    for p in (data_path(d, symbol), other_variant(data_path(d, symbol))):
        if p.exists():
            with open_text(p) as f:
                for r in csv.DictReader(f):
                    if r.get("timestamp"):
                        rows[r["timestamp"]] = [r.get(c, "") for c in cols]
    return rows


def _merge_multi(d: Path, symbol: str, cols: List[str], new_rows: Dict[str, List[str]]) -> int:
    """دمج بلا تكرار على timestamp + ترتيب + كتابة ملف واحد (csv.gz افتراضياً)؛ نسخة الصيغة الأخرى تُحذف."""
    rows = _read_table(d, symbol, cols)
    before = len(rows)
    rows.update(new_rows)
    path = data_path(d, symbol)
    write_lines(path, [",".join([ts] + rows[ts]) for ts in sorted(rows)], False, ["timestamp"] + cols)
    if other_variant(path).exists():
        other_variant(path).unlink()
    n = len(rows) - before
    del rows
    return n


async def download_metrics_vision(symbol: str, start_ms: int, end_ms: int, oi_dir: Path, metrics_dir: Path,
                                  vision_sem: asyncio.Semaphore, period_ms: int = 3_600_000) -> int:
    """ملفات metrics اليومية (كل 5 دقائق) ← آخر قيمة في كل فترة period_ms:
    open_interest/<S>.csv (timestamp, open_interest = sum_open_interest — ما يقرؤه خط الأنابيب) و
    futures_metrics/<S>.csv (كل أعمدة METRICS_COLS)."""
    last = _archive_last_ms(oi_dir, symbol)
    lo = max(start_ms, ((last // 86_400_000) + 1) * 86_400_000 if last is not None else 0)
    keys = await asyncio.to_thread(list_vision_keys, f"data/futures/um/daily/metrics/{symbol}/")
    lo_day, hi_day = fmt_dt(lo)[:10], fmt_dt(end_ms)[:10]
    keys = [k for k in keys if lo_day <= k[-14:-4] <= hi_day]
    if not keys:
        return 0
    batches = await asyncio.gather(*[fetch_zip_rows_async(f"{VISION_BASE}/{k}", vision_sem) for k in keys])
    agg: Dict[int, Tuple[int, List[str]]] = {}          # فترة ← (آخر طابع داخلها، القيم)
    for rows in batches:
        for r in rows or []:
            if len(r) < 8:
                continue
            try:
                ts = int(datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
            except ValueError:
                continue
            bucket = (ts // period_ms) * period_ms
            if bucket not in agg or ts >= agg[bucket][0]:
                agg[bucket] = (ts, r[2:8])
    oi_new = {fmt_pandas_utc(b): v[1][0] for b, v in agg.items() if v[1][0]}
    met_new = {fmt_pandas_utc(b): v[1] for b, v in agg.items()}
    del batches, agg
    _merge_multi(metrics_dir, symbol, METRICS_COLS, met_new)
    return _merge_archive(oi_dir, symbol, "open_interest", oi_new) if oi_new else 0


# ═══════════════ التمويل والفائدة المفتوحة (عبر API المباشر — لإكمال الأحدث فقط) ═══════════════
def _archive_last_ms(d: Path, symbol: str) -> Optional[int]:
    ts = [t for t in _read_table(d, symbol, [])]
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(max(ts).replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _merge_archive(d: Path, symbol: str, value_col: str, new_rows: Dict[str, str]) -> int:
    return _merge_multi(d, symbol, [value_col], {k: [v] for k, v in new_rows.items()})


async def download_funding_rest(symbol, onboard_ms, start_ms, end_ms, funding_dir: Path, rest_ctx: dict) -> int:
    last = _archive_last_ms(funding_dir, symbol)
    cursor = max(start_ms, onboard_ms, (last + 1) if last is not None else 0)
    new: Dict[str, str] = {}
    while cursor < end_ms:
        batch = await request_with_retry(rest_ctx["base_url"], "/fapi/v1/fundingRate",
                                         {"symbol": symbol, "startTime": cursor, "endTime": end_ms,
                                          "limit": 1000}, rest_ctx["fr_limiter"], 1, rest_ctx["req_sem"])
        if not batch:
            break
        for r in batch:
            new[fmt_pandas_utc(int(r["fundingTime"]))] = r["fundingRate"]
        nxt = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000 or nxt <= cursor:
            break
        cursor = nxt
    return _merge_archive(funding_dir, symbol, "funding_rate", new) if new else 0


async def download_open_interest_rest(symbol, oi_dir: Path, period: str, rest_ctx: dict) -> int:
    new: Dict[str, str] = {}
    end = int(time.time() * 1000)
    begin = end - 30 * 86_400_000
    cursor = begin
    while cursor < end:
        batch = await request_with_retry(rest_ctx["base_url"], "/futures/data/openInterestHist",
                                         {"symbol": symbol, "period": period, "limit": 500,
                                          "startTime": cursor, "endTime": end},
                                         rest_ctx["fr_limiter"], 1, rest_ctx["req_sem"])
        if not batch:
            break
        for r in batch:
            new[fmt_pandas_utc(int(r["timestamp"]))] = r["sumOpenInterest"]
        nxt = int(batch[-1]["timestamp"]) + 1
        if len(batch) < 500 or nxt <= cursor:
            break
        cursor = nxt
    return _merge_archive(oi_dir, symbol, "open_interest", new) if new else 0


# ═══════════════ ملفات CSV / CSV.GZ ═══════════════
def data_ext() -> str:
    return ".csv.gz" if USE_GZIP else ".csv"


def data_path(d: Path, symbol: str) -> Path:
    return d / f"{symbol}{data_ext()}"


def other_variant(path: Path) -> Path:
    """<S>.csv.gz ↔ <S>.csv"""
    return path.with_name(path.name[:-3]) if path.name.endswith(".gz") else path.with_name(path.name + ".gz")


def symbol_of(path: Path) -> str:
    n = path.name
    return n[:-7] if n.endswith(".csv.gz") else n[:-4] if n.endswith(".csv") else path.stem


def list_data_files(d: Path) -> Dict[str, Path]:
    """رمز ← ملفه (csv.gz أو csv؛ عند وجود الاثنين تُفضَّل صيغة التشغيل الحالي)."""
    out: Dict[str, Path] = {}
    if not d.exists():
        return out
    for p in sorted(d.iterdir()):
        if p.name.endswith((".csv", ".csv.gz")):
            sym = symbol_of(p)
            if sym not in out or p.name.endswith(data_ext()):
                out[sym] = p
    return out


def _open_bin(path: Path, mode: str, gz: Optional[bool] = None):
    """gz=None: من امتداد path نفسه (الملف المؤقت <S>.csv.gz.tmp يمرّر gz صراحةً)."""
    gz = path.name.endswith(".gz") if gz is None else gz
    return gzip.open(path, mode, compresslevel=GZIP_LEVEL) if gz else open(path, mode)


def open_text(path: Path):
    return (gzip.open(path, "rt", newline="", encoding="utf-8") if path.name.endswith(".gz")
            else open(path, newline="", encoding="utf-8"))


def read_header(path: Path) -> Optional[List[str]]:
    try:
        with open_text(path) as f:
            return next(csv.reader(f), None)
    except (OSError, EOFError):
        return None


def read_last_timestamp(path: Path) -> Optional[int]:
    try:
        if path.name.endswith(".gz"):                    # لا قراءة من الذيل في gzip: مرور تدفقي واحد
            last = ""
            with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        last = line
            return int(last.split(",")[0])
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return None
            f.seek(max(0, size - 4096))
            lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
        return int(lines[-1].split(",")[0])
    except (ValueError, IndexError, OSError, EOFError):
        return None


def write_lines(path: Path, lines: List[str], append: bool, header: List[str] = CSV_HEADER):
    """كتابة ذرّية: ملف مؤقت ثم os.replace (انقطاع أثناء الكتابة لا يُفسد الملف القديم).
    append مع gzip = عضو gzip جديد في نهاية نسخة من الملف (صيغة gzip قياسية يقرؤها pandas/gzip)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if append and path.exists():
        shutil.copyfile(path, tmp)
        mode = "ab"
    else:
        mode = "wb"
    with _open_bin(tmp, mode, gz=path.name.endswith(".gz")) as f:
        if mode == "wb":
            f.write((",".join(header) + "\n").encode("utf-8"))
        for i in range(0, len(lines), 20_000):
            f.write(("\n".join(lines[i:i + 20_000]) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def write_rows(path: Path, rows: List[list], append: bool):
    write_lines(path, [",".join(str(x) for x in r) for r in rows], append)


# ═══════════════ فحص الثغرات (جديد) ═══════════════
def detect_gaps(path: Path, interval_ms: int) -> List[dict]:
    """يمسح ملف عملة واحدة ويُرجع كل ثغرة فعلية فيه (فرق أكبر من فريم واحد بين شمعتين متتاليتين)."""
    gaps: List[dict] = []
    prev = None
    with open_text(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                ts = int(row[0])
            except (ValueError, IndexError):
                continue
            if prev is not None and ts - prev > interval_ms:
                gaps.append({
                    "gap_start_utc": fmt_dt(prev),
                    "gap_end_utc": fmt_dt(ts),
                    "missing_candles": (ts - prev) // interval_ms - 1,
                    "gap_hours": round((ts - prev) / 3_600_000, 2),
                })
            prev = ts
    return gaps


def write_gaps_report(report_path: Path, out_dir: Path, interval_ms: int) -> Tuple[int, int]:
    """يُعاد بناؤه بالكامل من الملفات الفعلية في كل تشغيل. يُرجع (عدد الثغرات، عدد العملات المتأثرة)."""
    all_rows = []
    affected = 0
    for sym, p in list_data_files(out_dir).items():
        gaps = detect_gaps(p, interval_ms)
        if gaps:
            affected += 1
            for g in gaps:
                all_rows.append({"symbol": sym, **g})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "gap_start_utc", "gap_end_utc",
                                          "missing_candles", "gap_hours"])
        w.writeheader()
        w.writerows(all_rows)
    os.replace(tmp, report_path)
    return len(all_rows), affected


async def fix_gaps_for_symbol(symbol: str, path: Path, gaps: List[dict], interval: str,
                              interval_ms: int, rest_ctx: dict) -> int:
    """محاولة إصلاح الثغرات عبر API المباشر (best-effort). يعمل فقط إن كان rest_ctx متاحاً."""
    if not gaps:
        return 0
    existing: Dict[int, list] = {}
    with open_text(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                existing[int(row[0])] = row
            except (ValueError, IndexError):
                continue
    fixed = 0
    for g in gaps:
        lo = int(datetime.strptime(g["gap_start_utc"], "%Y-%m-%d %H:%M:%S")
                 .replace(tzinfo=timezone.utc).timestamp() * 1000) + interval_ms
        hi = int(datetime.strptime(g["gap_end_utc"], "%Y-%m-%d %H:%M:%S")
                 .replace(tzinfo=timezone.utc).timestamp() * 1000)
        try:
            rows = await fetch_rest_klines(symbol, lo, hi, interval, interval_ms, rest_ctx)
        except Exception:
            continue
        for k in rows:
            try:
                ts = int(k[0])
            except (ValueError, IndexError):
                continue
            if lo <= ts < hi and ts not in existing:
                existing[ts] = kline_row(k)
                fixed += 1
    if fixed:
        write_rows(path, [existing[t] for t in sorted(existing)], append=False)
    return fixed


# ═══════════════ تقرير اكتمال التمويل/الفائدة المفتوحة ═══════════════
def write_archive_report(report_path: Path, dirs: Dict[str, Path]) -> Tuple[int, int]:
    """لكل ملف أرشيف (funding_rate/open_interest/futures_metrics): أول/آخر طابع، عدد الصفوف، الفاصل المعتاد،
    عدد الفجوات (فاصل > 1.5 × المعتاد) وأكبرها، وكم يتأخر آخر صف عن الآن. يُعاد بناؤه كاملاً كل تشغيل.
    يُرجع (عدد الملفات، عدد الملفات ذات فجوات)."""
    rows, with_gaps = [], 0
    now = time.time()
    for kind, d in dirs.items():
        for sym, p in list_data_files(d).items():
            ts = []
            with open_text(p) as f:
                for r in csv.DictReader(f):
                    try:
                        ts.append(datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp())
                    except (KeyError, ValueError):
                        continue
            if not ts:
                continue
            ts.sort()
            diffs = [b - a for a, b in zip(ts, ts[1:])]
            step = sorted(diffs)[len(diffs) // 2] if diffs else 0
            gaps = [x for x in diffs if step and x > 1.5 * step]
            with_gaps += bool(gaps)
            rows.append({"kind": kind, "symbol": sym, "first_utc": fmt_dt(int(ts[0] * 1000)),
                         "last_utc": fmt_dt(int(ts[-1] * 1000)), "rows": len(ts),
                         "step_hours": round(step / 3600, 2), "n_gaps": len(gaps),
                         "max_gap_hours": round(max(gaps) / 3600, 1) if gaps else 0,
                         "stale_hours": round((now - ts[-1]) / 3600, 1)})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "symbol", "first_utc", "last_utc", "rows", "step_hours",
                                          "n_gaps", "max_gap_hours", "stale_hours"])
        w.writeheader()
        w.writerows(rows)
    return len(rows), with_gaps


# ═══════════════ سجلّ الأصول ═══════════════
def file_stats(path: Path, interval_ms: int) -> dict:
    n = gaps = missing = zero_vol = bad = 0
    first = last = prev = None
    max_gap = 0
    with open_text(path) as f:
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
    existing: Dict[str, dict] = {}
    extra_cols: List[str] = []
    if registry_path.exists():
        with open(registry_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            extra_cols = [c for c in (reader.fieldnames or []) if c]
            for r in reader:
                if r.get("name"):
                    existing[r["name"]] = r
    for name, p in list_data_files(out_dir).items():
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
    p = argparse.ArgumentParser(description="تحميل شموع Binance Futures من أرشيف data.binance.vision "
                                             "(يعمل من شبكات محظورة مثل كولاب) + فحص/إصلاح الثغرات")
    p.add_argument("--start", default=START_DATE, help="تاريخ البداية UTC، مثال: 2025-01-01")
    p.add_argument("--end", default=END_DATE, help="تاريخ النهاية UTC (افتراضي: الآن)")
    p.add_argument("--interval", default=INTERVAL, help="الفريم الزمني، مثال: 1m 5m 1h")
    p.add_argument("--symbols", default=SYMBOLS, help="عملات محددة مفصولة بفاصلة (افتراضي: اكتشاف تلقائي)")
    p.add_argument("--symbols-file", default="", help="ملف نصي: عملة في كل سطر (الأضمن على شبكات محظورة)")
    p.add_argument("--out-dir", default="", help="مجلد المخرجات (افتراضي: تلقائي داخل data/)")
    p.add_argument("--drive-root", default=DRIVE_ROOT, help="جذر Drive لبنية خط الأنابيب")
    p.add_argument("--registry", default="", help="مسار asset_registry.csv (افتراضي: <root>/crypto_data/)")
    p.add_argument("--funding", action="store_true", default=FUNDING, help="حمّل تاريخ معدّل التمويل (API مباشر)")
    p.add_argument("--open-interest", action="store_true",
                   help="الفائدة المفتوحة + metrics من الأرشيف (ساعياً) منذ --oi-start")
    p.add_argument("--oi-start", default=OI_START_DATE, help="بداية أرشيف الفائدة المفتوحة، مثال 2024-01-01")
    p.add_argument("--skip-klines", action="store_true", help="التمويل/الفائدة المفتوحة فقط (الشموع موجودة)")
    p.add_argument("--oi-period", default="1h", help="دقة الفائدة المفتوحة وmetrics (5m..1d؛ 15m = فريم الشموع)")
    p.add_argument("--premium", action="store_true",
                   help="مؤشر العلاوة premiumIndexKlines بنفس الفريم ← premium_index/<S>.csv.gz (أساس التمويل لحظياً)")
    p.add_argument("--workers", type=int, default=SYMBOLS_IN_PARALLEL,
                   help="عملات تُعالَج بالتوازي (كل واحدة: تحميل←فك←دمج←ضغط←حفظ←تحرير الذاكرة)")
    p.add_argument("--downloads", type=int, default=VISION_CONCURRENT, help="ملفات ZIP تُحمَّل بالتوازي (إجمالاً)")
    p.add_argument("--no-gzip", action="store_true", help="احفظ csv عادياً بدل csv.gz")
    p.add_argument("--include-delisted", action="store_true",
                   help="مع اكتشاف exchangeInfo فقط؛ فهرسة data.binance.vision تشمل المشطوبة دائماً")
    p.add_argument("--registry-only", action="store_true", help="أعد بناء السجلّ وتقرير الثغرات فقط")
    p.add_argument("--fix-gaps", action="store_true", default=FIX_GAPS,
                   help="حاول إصلاح الثغرات المكتشفة عبر API المباشر (يحتاج شبكة غير محظورة)")
    p.add_argument("--vision-only", action="store_true",
                   help="تخطَّ حتى فحص الاتصال المباشر — بدء أسرع إن كنت متأكداً أن الشبكة محظورة")
    p.add_argument("--testnet", action="store_true", default=os.environ.get("BINANCE_TESTNET", "").lower() == "true")
    p.add_argument("--base-url", default="", help="تجاوز عنوان الـ API (للاختبار)")
    if _in_notebook():
        # خلية ملصوقة: sys.argv هنا وسائط نواة Jupyter (-f kernel.json) لا وسائطنا — تُتجاهل.
        # مع %run تصل وسائطك الحقيقية وتُقرأ عادياً.
        args, _unknown = p.parse_known_args()
        return args
    return p.parse_args()


async def main():
    args = parse_args()
    colab_drive = Path("/content/drive/MyDrive")
    if _in_notebook() and not args.drive_root and not args.out_dir and colab_drive.exists():
        # في Colab: القرص المحلي يُمسح عند انقطاع الجلسة — اكتب في Drive مباشرة.
        args.drive_root = str(colab_drive)
        print(f"Colab: الكتابة في {colab_drive} مباشرة (history_<interval>/ و crypto_data/).")

    global USE_GZIP
    USE_GZIP = not args.no_gzip
    interval, interval_ms = args.interval, interval_to_ms(args.interval)
    start_ms = parse_date_ms(args.start)
    end_ms = parse_date_ms(args.end) if args.end else int(time.time() * 1000)
    if start_ms >= end_ms:
        print("[خطأ] تاريخ البداية يجب أن يكون قبل تاريخ النهاية.")
        return

    if args.drive_root:
        root = Path(args.drive_root)
        default_out = root / f"history_{interval}"
    else:
        root = Path(OUT_ROOT)
        default_out = root / f"history_{interval}_from_{date_label(start_ms)}"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    registry_path = Path(args.registry) if args.registry else root / "crypto_data" / "asset_registry.csv"
    gaps_report_path = registry_path.parent / "gaps_report.csv"
    funding_dir, oi_dir, metrics_dir = root / "funding_rate", root / "open_interest", root / "futures_metrics"
    premium_dir = root / f"premium_index_{interval}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url or resolve_base_url(args.testnet)

    host = await asyncio.to_thread(choose_vision_base)
    print(f"[0] مضيف الأرشيف: {host}")
    print("[1] فحص إمكانية الوصول لواجهة Binance المباشرة (fapi.binance.com)...")
    if args.vision_only:
        rest_available, rest_reason = False, "تم تخطي الفحص (--vision-only)"
    else:
        rest_available, rest_reason = await probe_rest_access(base_url)
    if rest_available:
        print("    ✓ متاحة — سيُستخدم الأرشيف لمعظم التاريخ + API المباشر لإكمال اليوم الحالي"
              " والتمويل/الفائدة المفتوحة.")
    else:
        print(f"    ⚠ غير متاحة من هذه الشبكة ({rest_reason}) — الاعتماد الكامل على data.binance.vision."
              " اليوم الحالي سيبقى ناقصاً حتى تُعاد المحاولة (نفس الأمر يُكمل تلقائياً بفضل resume)."
              " التمويل/الفائدة المفتوحة/العلاوة تُجلب من الأرشيف نفسه.")

    kl_limiter, fr_limiter = WeightLimiter(WEIGHT_BUDGET_PER_MIN), WeightLimiter(FUNDING_REQ_PER_MIN)
    req_sem = asyncio.Semaphore(MAX_CONCURRENT)
    rest_ctx = ({"base_url": base_url, "limiter": kl_limiter, "fr_limiter": fr_limiter, "req_sem": req_sem}
                if rest_available else None)
    vision_sem = asyncio.Semaphore(max(1, args.downloads))

    print("[2] جلب قائمة العملات...")
    has_wanted = bool(args.symbols.strip() or args.symbols_file)
    all_symbols: Dict[str, dict] = {}
    if has_wanted:
        wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols_file:
            wanted += [ln.strip().upper() for ln in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
                       if ln.strip() and not ln.startswith("#")]
        all_symbols = {s: {"onboard_ms": 0, "status": ""} for s in dict.fromkeys(wanted)}
        print(f"    استُخدمت قائمة مزوَّدة يدوياً: {len(all_symbols)} عملة.")
    elif rest_available:
        try:
            all_symbols = await asyncio.to_thread(list_symbols_rest, base_url, args.include_delisted,
                                                  ONLY_USDT_PERPETUAL)
            print(f"    اكتُشفت {len(all_symbols)} عملة عبر exchangeInfo.")
        except BinanceHTTPError as e:
            if e.code == 451:
                print("    [تنبيه] exchangeInfo محجوب أيضاً (451) رغم اجتياز فحص ping —"
                      " التحويل لفهرسة data.binance.vision...")
            else:
                print(f"    [تنبيه] فشل exchangeInfo ({e})، التحويل لفهرسة data.binance.vision...")
        except Exception as e:
            print(f"    [تنبيه] فشل exchangeInfo ({e})، التحويل لفهرسة data.binance.vision...")
    if not all_symbols:
        try:
            all_symbols = await list_symbols_from_vision(vision_sem)
            print(f"    اكتُشفت {len(all_symbols)} عملة من فهرسة data.binance.vision "
                  "(يشمل هذا العملات المشطوبة تلقائياً لأن الأرشيف لا يحذف ملفاتها).")
        except Exception as e:
            print(f"[خطأ] تعذّر اكتشاف قائمة العملات من أي مصدر: {e}")
            print("    مرّر العملات يدوياً عبر --symbols BTCUSDT,ETHUSDT أو --symbols-file مسار_الملف")
            return

    if args.registry_only:
        n = update_registry(registry_path, out_dir, interval, interval_ms, all_symbols)
        ng, na = write_gaps_report(gaps_report_path, out_dir, interval_ms)
        print(f"[✓] سجلّ الأصول: {registry_path} ({n} عملة)")
        print(f"[✓] تقرير الثغرات: {gaps_report_path} ({ng} ثغرة في {na} عملة)")
        return

    symbols = dict(sorted(all_symbols.items()))
    if not symbols:
        print("[خطأ] لا توجد عملات للتحميل.")
        return

    print(f"[3] {len(symbols)} عملة | {fmt_dt(start_ms)} → {fmt_dt(end_ms)} UTC | الفريم: {interval}")
    print(f"    الشموع: {out_dir}")
    print(f"    السجلّ: {registry_path}")
    print(f"    تقرير الثغرات: {gaps_report_path}")

    sym_sem = asyncio.Semaphore(max(1, args.workers))
    print(f"    التوازي: {args.workers} عملات × {args.downloads} تحميل متزامن | الحفظ: {data_ext()}")
    total, done, total_candles = len(symbols), 0, 0
    failed: List[str] = []

    async def worker(sym: str, meta: dict):
        nonlocal done, total_candles
        async with sym_sem:
            parts = []
            if not args.skip_klines:
                try:
                    status, n, note = await download_symbol_vision(sym, meta.get("onboard_ms", 0), start_ms, end_ms,
                                                                    interval, interval_ms, out_dir,
                                                                    vision_sem, rest_ctx)
                    total_candles += n
                    parts.append(f"{status} — {n:,} شمعة{note}")
                except Exception as e:
                    failed.append(sym)
                    parts.append(f"[فشل الشموع] {e}")

            if args.premium:
                try:
                    k = await download_premium_vision(sym, start_ms, end_ms, interval, interval_ms,
                                                      premium_dir, vision_sem)
                    parts.append(f"علاوة +{k:,}")
                except Exception as e:
                    parts.append(f"[تعذّر مؤشر العلاوة] {e}")

            if args.funding:
                try:
                    k = await download_funding_vision(sym, start_ms, end_ms, funding_dir, vision_sem)
                    if rest_ctx is not None:     # الشهر الحالي (لا ملف شهري له بعد) عبر API إن كان متاحاً
                        k += await download_funding_rest(sym, meta.get("onboard_ms", 0), start_ms, end_ms,
                                                         funding_dir, rest_ctx)
                    parts.append(f"تمويل +{k:,}")
                except Exception as e:
                    parts.append(f"[تعذّر التمويل] {e}")

            if args.open_interest:
                try:
                    k = await download_metrics_vision(sym, max(start_ms, parse_date_ms(args.oi_start)), end_ms,
                                                      oi_dir, metrics_dir, vision_sem,
                                                      interval_to_ms(args.oi_period))
                    parts.append(f"فائدة مفتوحة +{k:,}")
                except Exception as e:
                    parts.append(f"[تعذّرت الفائدة المفتوحة] {e}")

            done += 1
            gc.collect()                                   # كل ما حُمّل لهذه العملة كُتب — تحرير الذاكرة
            print(f"[{done}/{total}] {sym}: " + " | ".join(parts), flush=True)

    print("[4] بدء التحميل من data.binance.vision" + (" + استكمال بـ API المباشر" if rest_available else "")
          + " ...")
    await asyncio.gather(*[worker(s, m) for s, m in symbols.items()])

    if args.funding or args.open_interest:
        rep_path = registry_path.parent / "funding_oi_report.csv"
        n_files, n_gap = write_archive_report(rep_path, {"funding_rate": funding_dir, "open_interest": oi_dir,
                                                         "futures_metrics": metrics_dir})
        print(f"\n[✓] تقرير اكتمال التمويل/الفائدة المفتوحة: {rep_path} — {n_files} ملف، {n_gap} بها فجوات"
              + ("" if rest_available else " (التمويل يتأخر حتى نهاية الشهر السابق: أرشيفه شهري وAPI المباشر غير متاح)"))
    if args.skip_klines:
        print(f"[✓] انتهى (تمويل/فائدة مفتوحة فقط): {funding_dir} | {oi_dir} | {metrics_dir}")
        return
    n_reg = update_registry(registry_path, out_dir, interval, interval_ms, all_symbols)
    ng, na = write_gaps_report(gaps_report_path, out_dir, interval_ms)

    print(f"\n[✓] انتهى. شموع جديدة: {total_candles:,}")
    print(f"    سجلّ الأصول: {registry_path} ({n_reg} عملة)")
    if ng:
        hint = (" (مرّر --fix-gaps لمحاولة إصلاحها تلقائياً)" if rest_available else
                " (API المباشر غير متاح الآن لإصلاحها؛ أعد التشغيل بـ --fix-gaps من شبكة غير محظورة)")
        print(f"    ⚠ تقرير الثغرات: {gaps_report_path} — {ng} ثغرة في {na} عملة{hint}")
    else:
        print("    ✓ لا توجد ثغرات في أي عملة.")

    if args.fix_gaps and ng and rest_available:
        print("[5] محاولة إصلاح الثغرات المكتشفة عبر API المباشر...")
        total_fixed = 0
        for sym, p in list_data_files(out_dir).items():
            gaps = detect_gaps(p, interval_ms)
            if not gaps:
                continue
            fixed = await fix_gaps_for_symbol(sym, p, gaps, interval, interval_ms, rest_ctx)
            if fixed:
                total_fixed += fixed
                print(f"    {sym}: أُصلحت {fixed} شمعة")
        if total_fixed:
            ng2, na2 = write_gaps_report(gaps_report_path, out_dir, interval_ms)
            print(f"[✓] بعد الإصلاح: {ng2} ثغرة متبقية في {na2} عملة (كان {ng} في {na})")
        else:
            print("    لم يتم إصلاح أي ثغرة (قد تكون في فترات لا يوفرها API المباشر، مثل بيانات قديمة جداً).")
    elif args.fix_gaps and ng and not rest_available:
        print("    [تنبيه] --fix-gaps مُفعّل لكن API المباشر غير متاح الآن؛ أعد المحاولة من شبكة غير محظورة.")

    if failed:
        print(f"\n    فشلت الشموع بالكامل لـ ({len(failed)}): {', '.join(failed)} — أعد تشغيل نفس الأمر ليُكمل.")

    rel = lambda p: p.relative_to(root).as_posix() if root in p.parents else str(p)
    if not args.drive_root:
        print(f"\n    ارفع محتوى {root} إلى جذر MyDrive، أو شغّل بـ --drive-root لتُكتب في Drive مباشرة.")
    print("\n    في خط الأنابيب (crypto_data_pipeline_v6):")
    print(f"      update_config({{'drive_raw_dir': '{rel(out_dir)}', 'drive_raw_pattern': '{{name}}{data_ext()}', "
          f"'asset_registry_path': '{rel(registry_path)}'}})")
    if interval == "1h":
        print(f"      apply_hourly_preset({{'drive_raw_dir': '{rel(out_dir)}'}})")


def run():
    """سطر الأوامر/PyCharm: asyncio.run مباشرة. داخل Jupyter/Colab (%run) توجد حلقة أحداث تعمل مسبقاً
    فيفشل asyncio.run — عندها يُشغَّل main في خيط مستقل بحلقته الخاصة."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(main())
        return
    import threading
    err: List[BaseException] = []

    def _target():
        try:
            asyncio.run(main())
        except BaseException as e:                     # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if err:
        raise err[0]


if __name__ == "__main__":
    run()

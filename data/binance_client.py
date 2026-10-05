"""
PURPOSE:  Binance client and live candle fetching (fetch_data, futures symbols, request rate limiter) plus funding-rate / open-interest fetch and their Drive archive.
TAGS:     binance, live data, fetch_data, RateLimiter, futures symbols, funding rate, open interest, save_funding_open_interest, load_funding_open_interest, API
PITFALLS: API_KEY/API_SECRET stay empty here (public market data only); never write keys into the repo. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 17) عميل Binance وجلب شموع حيّة

ثلاث دوال — كما وردت — لتوفير مصدر بيانات حيّ:

* `_create_client` / `client` — عميل Binance (يُنشأ **مرة واحدة** عند تشغيل
  الخلية، بإعادة محاولة عند انقطاع الشبكة المؤقت).
* `get_normalized_futures_symbols` — سرد رموز العقود الآجلة الخطّية النشطة
  المنتهية بـ USDT (عبر `ccxt`).
* `fetch_data` — جلب شموع OHLCV لرمز وفريم معيّنين (عبر `python-binance`).

القسم التالي (18) يستخدم `fetch_data` كمصدر بديل عن `load_asset` (Drive)
داخل **نفس** `build_dataset_from_loader`/`prepare_single_asset` تماماً —
بلا أي نسخة موازية من منطق التجهيز.
"""
# @title
"""
عميل Binance وجلب الشموع الحيّة.

**مفتاحا API:** اتركهما فارغتين (``""``) لبيانات السوق العامة فقط (شموع
OHLCV) — لا تحتاج مصادقة. عيّنهما فقط إن احتجت نقاط نهاية خاصة بالحساب
(الرصيد، الصفقات...) لاحقاً. لا تكتبهما هنا مباشرة في دفتر تُشاركه — استخدم
متغيرات بيئة أو أسرار Colab (``google.colab.userdata.get(...)``).
"""

# 🔧 عدّل هذا (أو اتركه فارغاً لبيانات السوق العامة فقط):
API_KEY = ""
API_SECRET = ""

#: الحدّ الأقصى لعدد الشموع في طلب futures_klines واحد على Binance.
BINANCE_MAX_LIMIT = 1500


import threading
from collections import deque


class RateLimiter:
    """حدّ معدّل بنافذة منزلقة، آمن للخيوط: لا يتجاوز ``max_calls`` طلباً في أي
    فترة طولها ``period`` ثانية، مهما بلغ عدد الخيوط.

    ``acquire()`` **يحجب** الخيط حتى يُتاح له طلب ثم يسجّله (لا يرفض ولا يُسقط).
    النافذة المنزلقة بدل «عدّاد يُصفَّر كل دقيقة»: العدّاد الدوري يسمح بضعف الحدّ
    عند حدّ الدقيقتين (٢٠٠٠ في آخر ثانية + ٢٠٠٠ في أول ثانية من التالية).
    ``clock``/``sleep`` قابلان للحقن (للاختبار بلا انتظار حقيقي).
    """

    def __init__(self, max_calls: int, period: float = 60.0,
                 clock=time.monotonic, sleep=time.sleep):
        if int(max_calls) <= 0 or float(period) <= 0:
            raise ValueError("max_calls و period يجب أن يكونا موجبين.")
        self.max_calls, self.period = int(max_calls), float(period)
        self._clock, self._sleep = clock, sleep
        self._calls: deque = deque()
        self._lock = threading.Lock()
        self.total_waited = 0.0          # مجموع ثواني الانتظار (للمراقبة)

    def acquire(self) -> float:
        """يحجب إن لزم ثم يحجز طلباً. يُرجع ثواني الانتظار في هذا الاستدعاء."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()
                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    self.total_waited += waited
                    return waited
                wait = self.period - (now - self._calls[0])
            self._sleep(wait)            # خارج القفل: لا نحجب بقية الخيوط أثناء النوم
            waited += wait


_REQUEST_LIMITERS: Dict[int, RateLimiter] = {}
_REQUEST_LIMITERS_LOCK = threading.Lock()


def get_request_limiter(config: Optional[dict] = None) -> Optional[RateLimiter]:
    """محدِّد الطلبات المشترك بين كل الخيوط، من ``CONFIG['live_max_requests_per_minute']``.

    نفس القيمة ⇒ نفس المحدِّد (ليُحصى كل شيء في نافذة واحدة). ``0``/``None`` ⇒
    بلا حدّ. تغيير القيمة أثناء التشغيل يبدأ محدِّداً جديداً بنافذة فارغة.
    """
    config = CONFIG if config is None else config
    n = config.get("live_max_requests_per_minute", 2000)
    if not n:
        return None
    n = int(n)
    with _REQUEST_LIMITERS_LOCK:
        limiter = _REQUEST_LIMITERS.get(n)
        if limiter is None:
            limiter = _REQUEST_LIMITERS[n] = RateLimiter(n, 60.0)
        return limiter

#: هامش إحماء عام (بالشموع) تحتاجه أطول نافذة تدحرج في الميزات المخصّصة
#: (custom.py: range_windows حتى 50، vol_windows حتى 48) والمؤشرات الفنية —
#: يُستخدَم في build_live_dataset عند حساب limit تلقائياً، فلا تُستنزَف كل
#: الصفوف بعد dropna() على الفريمات الأعلى ذات العدد الأقل من الشموع.
LIVE_FEATURE_WARMUP = 60


def _create_client(retries: int = 5, delay: float = 5.0):
    """يُنشئ عميل Binance مع إعادة محاولة عند انقطاع الشبكة المؤقت.

    ``Client(...)`` يستدعي ping داخليًا عند الإنشاء، فأي انقطاع DNS/شبكة
    عابر (شائع في بوت يعمل ساعات طويلة) كان يُسقِط البرنامج بالكامل فورًا.
    """
    from binance.client import Client

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return Client(API_KEY, API_SECRET)
        except Exception as e:                             # noqa: BLE001
            last_err = e
            print(f"⚠️ فشل الاتصال بـ Binance (محاولة {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(
        f"لا يوجد اتصال بالإنترنت أو فشل تهيئة العميل بعد {retries} محاولات: {last_err}"
    ) from last_err


#: عميل Binance الحيّ — يُنشأ مرة واحدة عند تشغيل هذه الخلية. يتطلب اتصال شبكة
#: فعلياً؛ أعد تشغيل الخلية لإعادة إنشائه إن انقطع الاتصال طويلاً.
# client = _create_client()


def get_normalized_futures_symbols() -> List[str]:
    """أزواج عقود Binance الآجلة الخطّية النشطة المنتهية بـ USDT."""
    import ccxt

    markets = ccxt.binance({"options": {"defaultType": "future"}}).load_markets()
    return sorted(
        s.replace("/", "").split(":")[0] for s in markets
        if markets[s]["linear"] and markets[s]["active"] and s.endswith("USDT")
    )


_KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]
_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _klines_to_frame(klines) -> pd.DataFrame:
    """يحوّل ردّ ``futures_klines`` الخام إلى DataFrame (timestamp UTC + OHLCV رقمية)."""
    df = pd.DataFrame(klines, columns=_KLINE_COLUMNS)
    df = df[["timestamp"] + _OHLCV_COLUMNS]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in _OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=_OHLCV_COLUMNS).reset_index(drop=True)


def fetch_data(symbol: str, interval: str, limit: int,
               retries: int = 3, delay: float = 2.0,
               page_delay: float = 0.0,
               rate_limiter: Optional[Any] = None) -> Optional[pd.DataFrame]:
    """جلب شموع OHLCV من Binance Futures، مفهرسة زمنيًا (UTC) ومرتّبة.

    إن زاد ``limit`` على ``BINANCE_MAX_LIMIT`` (الحدّ الأقصى للطلب الواحد) يُقسَّم
    الجلب إلى عدة طلبات **من الأحدث إلى الأقدم** (كل طلب يبدأ حيث انتهى السابق عبر
    ``endTime``)، ثم تُدمج الصفحات وتُرتَّب وتُزال تكراراتها قبل التسليم — فيصل
    للمستدعي ``DataFrame`` واحد مرتّب بأحدث ``limit`` شمعة، كأن الطلب كان واحداً.

    * إن كان التاريخ المتاح أقصر من ``limit`` تُرجَع الشموع المتوفرة فقط (لا خطأ).
    * إن فشلت أي صفحة بعد ``retries`` محاولات تُرجَع ``None`` للطلب كله: نتيجة
      مبتورة من الأقدم تبدو سليمة وتمرّ بصمت إلى التدريب/الاستدلال، والفشل الصريح
      أفضل منها (المستدعي يتعامل أصلاً مع ``None``).
    * ``page_delay``: ثوانٍ انتظار بين الصفحات (لتخفيف ضغط حدّ الوزن على Binance).
    * **حدّ المعدّل:** كل طلب (صفحة أو إعادة محاولة) يمرّ عبر محدِّد مشترك بين الخيوط
      قيمته ``CONFIG['live_max_requests_per_minute']`` (2000 افتراضياً) — يحجب الخيط
      حتى يتوفّر طلب بدل رفضه. ``rate_limiter`` يتجاوزه لهذا الاستدعاء (للاختبار).
    """
    from binance.exceptions import BinanceAPIException

    binance_interval = "1d" if interval == "1D" else interval
    limit = int(limit)
    if limit <= 0:
        raise ValueError(f"limit يجب أن يكون موجباً (وصل {limit}).")
    limiter = rate_limiter if rate_limiter is not None else get_request_limiter()

    def _fetch_page(page_limit: int, end_time: Optional[int], page_no: int):
        """صفحة واحدة بإعادة محاولة → ``(عدد الشموع الخام، أول open_time، DataFrame)`` أو None."""
        params = dict(symbol=symbol, interval=binance_interval, limit=page_limit)
        if end_time is not None:
            params["endTime"] = end_time
        tag = f"{symbol}/{interval}" + (f" ص{page_no}" if page_no > 1 else "")
        for attempt in range(retries):
            try:
                if limiter is not None:
                    limiter.acquire()
                klines = client.futures_klines(**params)
                first_ms = int(klines[0][0]) if len(klines) else None
                return len(klines), first_ms, _klines_to_frame(klines)
            except BinanceAPIException as e:
                print(f"⚠️ [{tag}] خطأ API (محاولة {attempt + 1}): {e}")
            except Exception as e:                         # noqa: BLE001
                print(f"⚠️ [{tag}] خطأ غير متوقع (محاولة {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        print(f"❌ [{tag}] فشلت كل محاولات الجلب")
        return None

    frames: List[pd.DataFrame] = []      # الأحدث أولاً (ترتيب الجلب)
    remaining, end_time, page_no = limit, None, 0
    while remaining > 0:
        page_no += 1
        page_limit = min(remaining, BINANCE_MAX_LIMIT)
        page = _fetch_page(page_limit, end_time, page_no)
        if page is None:
            return None
        raw_n, first_ms, frame = page
        if raw_n == 0:
            break                        # لا تاريخ أقدم
        frames.append(frame)
        remaining -= raw_n
        if raw_n < page_limit:
            break                        # الصفحة ناقصة ⇒ استُنفد التاريخ المتاح
        next_end = first_ms - 1
        if end_time is not None and next_end >= end_time:
            break                        # شبكة أمان: لا تقدّم ⇒ لا حلقة لا نهائية
        end_time = next_end
        if remaining > 0 and page_delay:
            time.sleep(page_delay)

    if not frames:
        return _klines_to_frame([])
    df = pd.concat(frames[::-1], ignore_index=True)      # الأقدم أولاً
    df = (df.drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp").reset_index(drop=True))
    return df.tail(limit).reset_index(drop=True)[["timestamp"] + _OHLCV_COLUMNS]


# ══════════════════════════════════════════════════════════════════════════
# معدّل التمويل والفائدة المفتوحة (عقود آجلة) — للاستخدام لاحقاً
# ══════════════════════════════════════════════════════════════════════════
#: الحدّ الأقصى لعدد السجلات في طلب futures_funding_rate واحد.
FUNDING_MAX_LIMIT = 1000
#: الحدّ الأقصى لعدد السجلات في طلب futures_open_interest_hist واحد.
OPEN_INTEREST_MAX_LIMIT = 500
#: ⚠️ Binance نفسها لا تخدم openInterestHist أبعد من هذه المدة، مهما طُلب أو
#: قُسِّم الطلب — قيد من طرف المنصة، لا هذا الكود. معدّل التمويل لا يحمل هذا
#: القيد (تاريخه الكامل متاح عبر futures_funding_rate كأي شمعة).
OPEN_INTEREST_MAX_LOOKBACK = pd.Timedelta(days=30)


def fetch_funding_rate(symbol: str, limit: int, retries: int = 3, delay: float = 2.0,
                       page_delay: float = 0.0,
                       rate_limiter: Optional[Any] = None) -> Optional[pd.DataFrame]:
    """جلب تاريخ معدّل التمويل (funding rate) لعقد آجل — تاريخه الكامل متاح
    (لا حدّ ٣٠ يوماً، خلافاً للفائدة المفتوحة أدناه)، فتقسيم الطلبات الكبيرة
    يتبع **نفس** منطق :func:`fetch_data` (تراجع بـ endTime، إعادة محاولة،
    حدّ معدّل مشترك) — منسوخ لا مُشترَك معه عمداً، تفادياً لأي مخاطرة بكسر
    اختباراته القائمة عبر إعادة هيكلة مشتركة.

    يُرجع ``DataFrame`` بعمودي ``timestamp`` (UTC) و``funding_rate``، مرتّباً
    تصاعدياً بأحدث ``limit`` سجل. أحداث التمويل كل ٨ ساعات (٣ في اليوم)، فـ
    ``limit=1095`` يغطي سنة تقريباً.
    """
    from binance.exceptions import BinanceAPIException

    limit = int(limit)
    if limit <= 0:
        raise ValueError(f"limit يجب أن يكون موجباً (وصل {limit}).")
    limiter = rate_limiter if rate_limiter is not None else get_request_limiter()

    def _fetch_page(page_limit: int, end_time: Optional[int], page_no: int):
        params = dict(symbol=symbol, limit=page_limit)
        if end_time is not None:
            params["endTime"] = end_time
        tag = f"{symbol}/funding" + (f" ص{page_no}" if page_no > 1 else "")
        for attempt in range(retries):
            try:
                if limiter is not None:
                    limiter.acquire()
                rows = client.futures_funding_rate(**params)
                first_ms = int(rows[0]["fundingTime"]) if len(rows) else None
                return len(rows), first_ms, rows
            except BinanceAPIException as e:
                print(f"⚠️ [{tag}] خطأ API (محاولة {attempt + 1}): {e}")
            except Exception as e:                             # noqa: BLE001
                print(f"⚠️ [{tag}] خطأ غير متوقع (محاولة {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        print(f"❌ [{tag}] فشلت كل محاولات الجلب")
        return None

    frames: List[list] = []
    remaining, end_time, page_no = limit, None, 0
    while remaining > 0:
        page_no += 1
        page_limit = min(remaining, FUNDING_MAX_LIMIT)
        page = _fetch_page(page_limit, end_time, page_no)
        if page is None:
            return None
        raw_n, first_ms, rows = page
        if raw_n == 0:
            break
        frames.append(rows)
        remaining -= raw_n
        if raw_n < page_limit:
            break
        next_end = first_ms - 1
        if end_time is not None and next_end >= end_time:
            break
        end_time = next_end
        if remaining > 0 and page_delay:
            time.sleep(page_delay)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    rows = [r for page in reversed(frames) for r in page]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = (df[["timestamp", "funding_rate"]].dropna(subset=["funding_rate"])
          .drop_duplicates("timestamp", keep="last")
          .sort_values("timestamp").reset_index(drop=True))
    return df.tail(limit).reset_index(drop=True)


def fetch_open_interest_hist(symbol: str, limit: int, period: str = "1h",
                             retries: int = 3, delay: float = 2.0,
                             rate_limiter: Optional[Any] = None) -> Optional[pd.DataFrame]:
    """جلب تاريخ الفائدة المفتوحة (open interest).

    ⚠️ Binance نفسها لا تخدم أبعد من :data:`OPEN_INTEREST_MAX_LOOKBACK`
    (٣٠ يوماً) عبر هذا المسار، مهما كان ``limit`` أو عدد الصفحات — قيد من
    طرف المنصة لا هذا الكود، فلا فائدة من طلب تاريخ أبعد. لأرشيف أطول: اجمعها
    دورياً (يومياً مثلاً) بـ :func:`save_funding_open_interest`؛ كل تشغيل
    يُضيف آخر ٣٠ يوماً المتاحة فيتراكم الأرشيف مع الزمن.

    ``period``: دقة العيّنات (\"5m\".. \"1d\"). عند ``period=\"1h\"`` تحتاج ٣٠
    يوماً صفحتين على الأكثر (٧٢٠ نقطة > الحدّ الأقصى ٥٠٠ للطلب الواحد)،
    تُجلَبان تلقائياً بنفس منطق :func:`fetch_funding_rate`.
    """
    from binance.exceptions import BinanceAPIException

    limit = int(limit)
    if limit <= 0:
        raise ValueError(f"limit يجب أن يكون موجباً (وصل {limit}).")
    limiter = rate_limiter if rate_limiter is not None else get_request_limiter()

    def _fetch_page(page_limit: int, end_time: Optional[int], page_no: int):
        params = dict(symbol=symbol, period=period, limit=page_limit)
        if end_time is not None:
            params["endTime"] = end_time
        tag = f"{symbol}/OI" + (f" ص{page_no}" if page_no > 1 else "")
        for attempt in range(retries):
            try:
                if limiter is not None:
                    limiter.acquire()
                rows = client.futures_open_interest_hist(**params)
                first_ms = int(rows[0]["timestamp"]) if len(rows) else None
                return len(rows), first_ms, rows
            except BinanceAPIException as e:
                print(f"⚠️ [{tag}] خطأ API (محاولة {attempt + 1}): {e}")
            except Exception as e:                             # noqa: BLE001
                print(f"⚠️ [{tag}] خطأ غير متوقع (محاولة {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        print(f"❌ [{tag}] فشلت كل محاولات الجلب")
        return None

    frames: List[list] = []
    remaining, end_time, page_no = limit, None, 0
    while remaining > 0:
        page_no += 1
        page_limit = min(remaining, OPEN_INTEREST_MAX_LIMIT)
        page = _fetch_page(page_limit, end_time, page_no)
        if page is None:
            return None
        raw_n, first_ms, rows = page
        if raw_n == 0:
            break
        frames.append(rows)
        remaining -= raw_n
        if raw_n < page_limit:
            break
        next_end = first_ms - 1
        if end_time is not None and next_end >= end_time:
            break
        end_time = next_end

    if not frames:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
    rows = [r for page in reversed(frames) for r in page]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    df = (df[["timestamp", "open_interest"]].dropna(subset=["open_interest"])
          .drop_duplicates("timestamp", keep="last")
          .sort_values("timestamp").reset_index(drop=True))
    return df.tail(limit).reset_index(drop=True)


def _funding_oi_drive_path(root: Path, kind: str, symbol: str, config: dict) -> Path:
    sub = (config.get(kind) or {}).get('drive_dir', kind)
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}.csv"


def save_funding_open_interest(symbols: List[str], funding_limit: int = 3000,
                               oi_limit: int = 720, oi_period: str = "1h",
                               config: Optional[dict] = None,
                               verbose: bool = True) -> None:
    """يجلب معدّل التمويل والفائدة المفتوحة لكل رمز ويُلحقها بأرشيف CSV على
    Drive (مسار كلٍّ منهما في ``CONFIG['funding_rate']['drive_dir']`` /
    ``['open_interest']['drive_dir']``) — **للاستخدام لاحقاً**، غير مدموجة
    تلقائياً في ``add_features`` بعد.

    كل تشغيل: يقرأ الأرشيف الحالي إن وُجد، يجلب الأحدث، يدمج بلا تكرار على
    ``timestamp``، ويحفظ. شغّلها **دورياً** (يدوياً، أو خلية Colab مجدولة) —
    هذا هو الحلّ الوحيد لتجاوز قيد الـ٣٠ يوماً على الفائدة المفتوحة من
    Binance نفسها: كل تشغيل يُضيف نافذته المتاحة، فيتراكم أرشيف أطول من
    السجلّ الفعلي لتشغيلاتك، لا من طلب واحد بأثر رجعي.
    """
    config = CONFIG if config is None else config
    root = mount_drive(config=config)
    fr_cfg, oi_cfg = config.get('funding_rate') or {}, config.get('open_interest') or {}

    for sym in symbols:
        if fr_cfg.get('enabled', True):
            new = fetch_funding_rate(sym, funding_limit)
            _merge_and_save_archive(root, 'funding_rate', sym, new, config, verbose)
        if oi_cfg.get('enabled', True):
            new = fetch_open_interest_hist(sym, oi_limit, period=oi_cfg.get('period', oi_period))
            _merge_and_save_archive(root, 'open_interest', sym, new, config, verbose)


def _merge_and_save_archive(root: Path, kind: str, symbol: str,
                            new_df: Optional[pd.DataFrame], config: dict,
                            verbose: bool) -> None:
    if new_df is None:
        print(f"   ❌ [{symbol}/{kind}] فشل الجلب — لم يُحفَظ شيء لهذا الرمز.")
        return
    path = _funding_oi_drive_path(root, kind, symbol, config)
    if path.exists():
        old = pd.read_csv(path, parse_dates=['timestamp'])
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    merged = (merged.drop_duplicates('timestamp', keep='last')
             .sort_values('timestamp').reset_index(drop=True))
    merged.to_csv(path, index=False)
    if verbose:
        print(f"   💾 [{symbol}/{kind}] +{len(new_df):,} سجل جديد | "
              f"{len(merged):,} إجمالي محفوظ ({path.name})")


def load_funding_open_interest(symbol: str, kind: str = 'funding_rate',
                               config: Optional[dict] = None) -> Optional[pd.DataFrame]:
    """يقرأ الأرشيف المحفوظ (``kind`` = ``'funding_rate'`` أو ``'open_interest'``)
    لرمز من Drive. ``None`` إن لم يُجمَع شيء بعد له (شغّل
    :func:`save_funding_open_interest` أولاً)."""
    config = CONFIG if config is None else config
    root = mount_drive(config=config)
    path = _funding_oi_drive_path(root, kind, symbol, config)
    if not path.exists():
        path = path.with_name(path.name + ".gz")     # أرشيف fetch_history_vision_colab (مضغوط)
        if not path.exists():
            return None
    return pd.read_csv(path, parse_dates=['timestamp'])

"""
PURPOSE:  run_pipeline_selftests(): the notebook's self-tests on small synthetic data (no Drive, no network).
TAGS:     self-tests, run_pipeline_selftests, synthetic data, regression tests
PITFALLS: The tests patch names through globals() of the shared namespace; docs/research/audit/_nbload.load_pipeline() leaves this module out (it takes minutes). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 19-ب) اختبارات ذاتية

تتحقّق من تعديلات هذه النسخة (استبعاد العملات، ميزات الساعة، التقسيم) على بيانات
تركيبية صغيرة — **بلا Drive ولا شبكة**، فتعمل قبل أي تحميل حقيقي. شغّلها بعد أي
تعديل على الدوال أعلاه؛ الفشل يرفع `AssertionError` باسم الاختبار الفاشل.
"""
# @title
"""
اختبارات ذاتية لتعديلات هذا الدفتر — بلا Drive ولا شبكة (بيانات تركيبية صغيرة).

كل اختبار يُثبت سلوكاً محدَّداً كان مكسوراً أو غير موجود:
  * استبعاد العملات (``exclude_coins`` / ``CONFIG['excluded_coins']``).
  * غياب ``TIME_hour_*`` على فريم يومي (ثابتتان بلا معلومة) وبقاؤهما داخل اليوم.
  * ``fetch_data``: تقسيم الجلب على عدة طلبات ودمجها (عميل Binance وهمي).
  * حدّ تقسيم بديل (``compute_global_cutoff``/``build_leak_free_split``) ونسبته الخام.
  * فصل فريم التحميل الحيّ (download_interval) عن فريم النموذج.
  * ذروة رام أقلّ في align_multi_timeframes_time_based/process_windows، وسقف
    خيوط احترازي بالرام، ونقاط استئناف لكل أصل (checkpoint/resume).
  * سياق سوقي عابر للأصول (add_market_context) ومحاذاته وتصفيره للمرجع.
  * إعادة تدريب دورية (rolling_split_schedule/rolling_splits) بلا تسرّب بين النوافذ.
  * معدّل التمويل والفائدة المفتوحة: الجلب المُقسَّم وتراكم أرشيف Drive.
  * هدف الانحدار كعائد مباشر (``reg_target_mode``) وعكسه، وتسميات ``_class`` بترميز
    1/0، ومركز ``'window_scale'`` = آخر سعر من نفس النوع (لا وسيط النافذة)، والتطبيع المقطعي
    (``cross_sectional_normalize``) وعكسه — أفكار من Qlib.
  * ``exclude_features`` وحدّ الطلبات (``RateLimiter``) وتمريره في ``fetch_data``.
  * ``split_data``: val و test غير فارغين على توزيع مكدَّس زمنياً، مع احترام
    فجوة العزل، وخطأ صريح (لا قسم فارغ صامت) حين يستحيل التقسيم.
  * الفريمات الأعلى بوضع 'closed' (1h+4h): شمعة 4h الأخيرة مغلقة عند t، اختبار عدم نظر للمستقبل بتغيير كل ما بعد t (مع
    اختبار أسنان)، وشكل/محاذاة العملات وfloat16، وتطبيع أسعار 4h بنافذتها، وأن فريماً واحداً لا يتأثّر بالمفتاحين.

شغّلها بعد أي تعديل على الدوال أعلاه: ``run_pipeline_selftests()``.
"""
import contextlib
import io
import threading


def run_pipeline_selftests(verbose: bool = True) -> bool:
    """يشغّل كل الاختبارات؛ يرفع ``AssertionError`` بملخّص الفاشل إن فشل أي منها."""

    def _cfg(**over) -> dict:
        cfg = deepcopy(CONFIG)
        cfg.update(tf_order=['1D'], base_tf='1D', window_sizes={'1D': 32}, stride=1,
                   feature_order=None, split_dates=None, excluded_coins=[],
                   min_split_samples=64, split_mode='global_time',
                   train_pct=0.90, val_pct=0.05, keep_asset_test_separate=False,
                   forecast_horizon=1, embargo_candles=None,
                   market_context={'enabled': False}, download_interval='1h',
                   funding_rate={'enabled': False}, open_interest={'enabled': False},
                   momentum_orth_natr={'enabled': False}, market_breadth={'enabled': False},
                   phase2_data={'use_intraday_15m': False, 'use_futures_metrics': False})
        cfg.update(over)
        return cfg

    def _ohlcv(days: int, seed: int = 0, start: str = '2025-01-01') -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = days * 24
        idx = pd.date_range(start, periods=n, freq='h', tz='UTC')
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        open_ = np.r_[close[0], close[:-1]]
        return pd.DataFrame({
            'open': open_,
            'high': np.maximum(open_, close) * (1 + rng.random(n) * 0.003),
            'low': np.minimum(open_, close) * (1 - rng.random(n) * 0.003),
            'close': close, 'volume': rng.random(n) * 1000 + 50}, index=idx)

    def _fake_dataset(lengths, stride: int = 16) -> dict:
        """مجموعة بيانات هيكلية بأطوال عيّنات متفاوتة تنتهي كلها في نفس اليوم —
        يحاكي سجلاً فيه قليل من العملات القديمة وكثير من الحديثة (تكدّس زمني)."""
        day = 86_400 * 10 ** 9
        end = pd.Timestamp('2026-09-15', tz='UTC').value
        ts, bounds, off = [], [], 0
        for k, w in enumerate(lengths):
            ts.append(end - (w - 1 - np.arange(w)) * stride * day)
            bounds.append({'name': f'A{k}', 'start': off, 'end': off + w})
            off += w
        ts = np.concatenate(ts).astype('float64')
        N = len(ts)
        last = np.zeros((N, len(LAST_COLUMNS)), dtype='float64')
        last[:, TS_COL] = ts
        return {'base_params': np.zeros((N, 2), 'float32'), 'last_candles': last,
                'X_1D': np.zeros((N, 2, 1), 'float32'),
                'y_close_class': np.ones(N, 'float32'),
                'timeframes': ['1D'], 'targets': ['close_class'], 'base_tf': '1D',
                'window_sizes': {'1D': 32}, 'forecast_horizon': 1, 'stride': stride,
                'asset_bounds': bounds}

    def _quiet(fn, *a, **k):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a, **k)

    # ── استبعاد العملات ────────────────────────────────────────────────────
    def t_exclude_matching():
        configs = [{'name': 'BTCUSDT'}, {'name': 'TSLAUSDT'},
                   {'name': ' ethusdt '}, {'name': 'XAUUSDT'}]
        kept = exclude_coins(configs, ['tslausdt', ' XAUUSDT ', 'GHOSTUSDT'], verbose=False)
        assert [c['name'] for c in kept] == ['BTCUSDT', ' ethusdt '], kept
        assert len(configs) == 4, "configs الأصلية عُدِّلت"
        _, removed, missing = split_excluded_coins(
            configs, ['tslausdt', 'XAUUSDT', 'GHOSTUSDT'])
        assert removed == ['TSLAUSDT', 'XAUUSDT'], removed
        assert missing == ['GHOSTUSDT'], missing           # إملاء خاطئ لا يمرّ بصمت

    def t_exclude_edge_cases():
        configs = [{'name': 'BTCUSDT'}, {'name': 'BTCDOMUSDT'}]
        assert len(exclude_coins(configs, [], verbose=False)) == 2
        # نصّ واحد = اسم واحد (لا يُفكَّك إلى أحرف)
        assert [c['name'] for c in exclude_coins(configs, 'BTCUSDT', verbose=False)] == ['BTCDOMUSDT']
        # الجذر لا يطابق: 'BTC' لا تستبعد BTCUSDT ولا BTCDOMUSDT
        assert len(exclude_coins(configs, ['BTC'], verbose=False)) == 2
        # عناصر نصّية بدل قواميس
        assert exclude_coins(['A', 'B'], ['a'], verbose=False) == ['B']
        # None → CONFIG['excluded_coins']؛ قائمة صريحة فارغة تتجاوزه
        cfg = {'excluded_coins': ['BTCUSDT']}
        assert [c['name'] for c in exclude_coins(configs, None, config=cfg, verbose=False)] == ['BTCDOMUSDT']
        assert len(exclude_coins(configs, [], config=cfg, verbose=False)) == 2

    def t_exclude_in_pipeline():
        cfg = _cfg()
        names = ['AAAUSDT', 'BBBUSDT', 'CCCUSDT']
        data = {n: _quiet(resample_timeframes, _ohlcv(200, seed=i), ['1D'], config=cfg)
                for i, n in enumerate(names)}
        ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                    config=cfg, max_workers=1, exclude=['bbbusdt'])
        got = [b['name'] for b in ds['asset_bounds']]
        assert got == ['AAAUSDT', 'CCCUSDT'], got
        assert ds['skipped_assets'].get('excluded') == ['BBBUSDT'], ds['skipped_assets']
        # وعبر CONFIG بلا وسيط
        cfg2 = _cfg(excluded_coins=['AAAUSDT'])
        ds2 = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                     config=cfg2, max_workers=1)
        assert [b['name'] for b in ds2['asset_bounds']] == ['BBBUSDT', 'CCCUSDT']

    # ── ميزات الساعة ───────────────────────────────────────────────────────
    def t_hour_features_daily():
        # time_features مُعطَّلة افتراضياً منذ فرز الميزات — تُفعَّل هنا صراحةً
        cfg = _cfg(custom_settings={**CONFIG['custom_settings'], 'time_features': True})
        names = custom_feature_names(config=cfg)
        assert 'TIME_hour_sin' not in names and 'TIME_hour_cos' not in names, names
        assert 'TIME_dow_sin' in names and 'TIME_dow_cos' in names
        daily = _ohlcv(120).resample('1D').agg(OHLCV_AGG)
        out = add_custom_features(daily, config=cfg)
        assert not any(c.startswith('TIME_hour') for c in out.columns), list(out.columns)
        assert set(names) <= set(out.columns), set(names) - set(out.columns)
        feats = _quiet(infer_feature_columns, cfg)
        assert not any(f.startswith('TIME_hour') for f in feats), feats

    def t_hour_features_intraday():
        cfg = _cfg(tf_order=['1h'], base_tf='1h', window_sizes={'1h': 32}, custom_settings={**CONFIG['custom_settings'], 'time_features': True})
        names = custom_feature_names(config=cfg)
        assert {'TIME_hour_sin', 'TIME_hour_cos'} <= set(names), names
        out = add_custom_features(_ohlcv(10), config=cfg)
        assert set(names) <= set(out.columns)
        assert out['TIME_hour_sin'].nunique() > 1          # فعلاً تتغيّر داخل اليوم
        # فريم مختلط (داخل اليوم + يومي): تبقى لأن أصغر فريم داخل اليوم
        mixed = _cfg(tf_order=['4h', '1D'], base_tf='4h', window_sizes={'4h': 32, '1D': 32}, custom_settings={**CONFIG['custom_settings'], 'time_features': True})
        assert 'TIME_hour_sin' in custom_feature_names(config=mixed)

    # ── التقسيم ────────────────────────────────────────────────────────────
    #: أطوال عيّنات تحاكي سجلاً حقيقياً: قليل قديم وكثير حديث، كلها تنتهي معاً.
    CROWDED = [140] * 5 + [120] * 20 + [45] * 80 + [18] * 150 + [5] * 120

    def t_split_crowded_timeline():
        cfg = _cfg()
        ds = _fake_dataset(CROWDED)
        train, val, test = _quiet(split_data, ds, config=cfg)
        n_tr, n_va, n_te = (len(s['base_params']) for s in (train, val, test))
        assert min(n_tr, n_va, n_te) >= 64, (n_tr, n_va, n_te)     # كان val=test=0
        kept = n_tr + n_va + n_te
        assert 0.80 <= n_tr / kept <= 0.95, n_tr / kept
        gap = embargo_duration(ds, cfg)
        t_tr, t_va, t_te = (sample_timestamps(s) for s in (train, val, test))
        assert t_va.min() - t_tr.max() >= gap, (t_va.min() - t_tr.max(), gap)
        assert t_te.min() - t_va.max() >= gap, (t_te.min() - t_va.max(), gap)

    def t_split_uniform_matches_requested():
        # عيّنات موزَّعة بانتظام: النسب المُبقاة تقارب 90/5/5 (بعد اقتطاع العزل)
        cfg = _cfg()
        ds = _fake_dataset([3000] * 3, stride=1)
        train, val, test = _quiet(split_data, ds, config=cfg)
        kept = sum(len(s['base_params']) for s in (train, val, test))
        assert abs(len(train['base_params']) / kept - 0.90) < 0.03
        assert abs(len(val['base_params']) / kept - 0.05) < 0.03

    def t_split_raises_when_impossible():
        cfg = _cfg()
        ds = _fake_dataset([10] * 5)                       # 50 عيّنة فقط < الحدّ الأدنى
        try:
            _quiet(split_data, ds, config=cfg)
        except ValueError as exc:
            assert 'split_dates' in str(exc)
        else:
            raise AssertionError("كان يجب أن يرفع ValueError بدل قسم فارغ")

    def t_split_explicit_dates_guarded():
        cfg = _cfg(split_dates={'train_end': '2025-06-01', 'val_end': '2030-01-01'})
        ds = _fake_dataset(CROWDED)                         # test سيخرج فارغاً
        try:
            _quiet(split_data, ds, config=cfg)
        except ValueError as exc:
            assert 'test' in str(exc)
        else:
            raise AssertionError("تواريخ صريحة تُنتج test فارغاً يجب أن ترفع ValueError")

    # ── fetch_data: الجلب المُقسَّم على عدة طلبات ───────────────────────────
    class _FakeFuturesClient:
        """عميل وهمي يحاكي futures_klines: حدّ 1500، endTime، وتاريخ محدود الطول."""
        H = 3_600_000

        def __init__(self, n_total, end_ms=1_800_000_000_000, fail=None):
            self.n, self.end_ms, self.calls = n_total, end_ms, []
            self.ok = 0                     # عدد الصفحات التي نجحت حتى الآن
            self._fails_left = dict(fail or {})   # {رقم الصفحة: مرات الفشل قبل النجاح (99 = دائم)}

        def ts(self, i):                    # طابع الشمعة i (0 = الأقدم)
            return self.end_ms - (self.n - 1 - i) * self.H

        def futures_klines(self, symbol, interval, limit, endTime=None):
            self.calls.append({'limit': limit, 'endTime': endTime, 'interval': interval})
            page = self.ok + 1
            assert limit <= BINANCE_MAX_LIMIT, f"طلب {limit} > الحدّ الأقصى"
            if self._fails_left.get(page, 0) > 0:
                self._fails_left[page] -= 1
                raise RuntimeError("simulated network error")
            self.ok += 1
            idx = [i for i in range(self.n) if endTime is None or self.ts(i) <= endTime]
            return [[self.ts(i), "1", "2", "0.5", "1.5", "10", self.ts(i) + self.H - 1,
                     "0", 0, "0", "0", "0"] for i in idx[-limit:]]

    def _with_client(fake, fn):
        old_client, old_sleep = globals().get('client'), time.sleep
        globals()['client'], time.sleep = fake, (lambda *_: None)
        try:
            return _quiet(fn)
        finally:
            globals()['client'], time.sleep = old_client, old_sleep

    def t_fetch_single_page_unchanged():
        fake = _FakeFuturesClient(5000)
        df = _with_client(fake, lambda: fetch_data('BTCUSDT', '1h', 300))
        assert len(fake.calls) == 1 and fake.calls[0]['endTime'] is None, fake.calls
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert len(df) == 300 and str(df['timestamp'].dt.tz) == 'UTC'

    def t_fetch_paginates_and_merges():
        fake = _FakeFuturesClient(10_000)
        df = _with_client(fake, lambda: fetch_data('BTCUSDT', '1h', 4000))
        assert [c['limit'] for c in fake.calls] == [1500, 1500, 1000], fake.calls
        assert len(df) == 4000, len(df)
        assert df['timestamp'].is_monotonic_increasing and not df['timestamp'].duplicated().any()
        step = df['timestamp'].diff().dropna().unique()
        assert len(step) == 1 and step[0] == pd.Timedelta(hours=1), step   # بلا ثقوب ولا تداخل
        assert df['timestamp'].iloc[-1].value // 10 ** 6 == fake.ts(9_999)   # آخر شمعة = الأحدث
        assert df['timestamp'].iloc[0].value // 10 ** 6 == fake.ts(6_000)    # وأقدمها ٤٠٠٠ شمعة للخلف

    def t_fetch_exact_multiple_of_max():
        fake = _FakeFuturesClient(10_000)
        df = _with_client(fake, lambda: fetch_data('BTCUSDT', '1h', 3000))
        assert [c['limit'] for c in fake.calls] == [1500, 1500] and len(df) == 3000

    def t_fetch_history_shorter_than_limit():
        fake = _FakeFuturesClient(2000)              # عملة حديثة: 2000 شمعة فقط
        df = _with_client(fake, lambda: fetch_data('NEWUSDT', '1h', 5000))
        assert len(df) == 2000 and len(fake.calls) == 2, (len(df), fake.calls)

    def t_fetch_retries_transient_page_error():
        fake = _FakeFuturesClient(10_000, fail={2: 1})   # فشل عابر واحد في الصفحة الثانية
        df = _with_client(fake, lambda: fetch_data('BTCUSDT', '1h', 3000, retries=3))
        assert df is not None and len(df) == 3000 and len(fake.calls) == 3, len(fake.calls)

    def t_fetch_returns_none_on_persistent_page_failure():
        fake = _FakeFuturesClient(10_000, fail={2: 99})
        df = _with_client(fake, lambda: fetch_data('BTCUSDT', '1h', 3000, retries=3))
        assert df is None                                # لا نتيجة مبتورة صامتة

    def t_fetch_rejects_nonpositive_limit():
        try:
            _with_client(_FakeFuturesClient(10), lambda: fetch_data('BTCUSDT', '1h', 0))
        except ValueError:
            return
        raise AssertionError("limit=0 يجب أن يرفع ValueError")

    # ── استبعاد الميزات ────────────────────────────────────────────────────
    def t_exclude_features_names_and_patterns():
        cfg = _cfg()
        base = _quiet(feature_order, cfg)
        assert 'RSI_14' in base and any(f.startswith('BB') for f in base), base
        added = _quiet(exclude_features, ['rsi_14', 'BB*', 'GHOST_1'], config=cfg)
        assert 'RSI_14' in added and all(a == 'RSI_14' or a.startswith('BB') for a in added), added
        after = cfg['feature_order']
        assert 'RSI_14' not in after and not any(f.startswith('BB') for f in after)
        assert len(after) == len(base) - len(added)           # feature_order حُدِّثت
        assert 'GHOST_1' not in cfg['exclude_from_features']  # الاسم المجهول لا يُضاف بصمت

    def t_exclude_features_idempotent_and_guard():
        cfg = _cfg()
        _quiet(exclude_features, ['EMA_9'], config=cfg)
        n = len(cfg['exclude_from_features'])
        assert _quiet(exclude_features, ['ema_9'], config=cfg) == []       # لا تكرار
        assert len(cfg['exclude_from_features']) == n
        try:
            _quiet(exclude_features, ['*'], config=cfg)                     # سيُفرغ كل شيء
        except ValueError:
            assert len(cfg['exclude_from_features']) == n                  # بلا تعديل
        else:
            raise AssertionError("استبعاد كل الميزات يجب أن يرفع ValueError")

    def t_exclude_features_reaches_dataset():
        cfg = _cfg()
        _quiet(exclude_features, ['EMA_26', 'STOCH*'], config=cfg)
        names = ['AAAUSDT', 'BBBUSDT']
        data = {n: _quiet(resample_timeframes, _ohlcv(200, seed=i), ['1D'], config=cfg)
                for i, n in enumerate(names)}
        ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                    config=cfg, max_workers=1)
        assert 'EMA_26' not in ds['feature_order']
        assert not any(f.startswith('STOCH') for f in ds['feature_order'])
        assert ds['X_1D'].shape[-1] == len(cfg['feature_order']) == len(ds['feature_order'])

    # ── محدِّد معدّل الطلبات ───────────────────────────────────────────────
    def t_ratelimiter_sliding_window():
        clock, slept = [0.0], []

        def _sleep(s):
            slept.append(s)
            clock[0] += s

        rl = RateLimiter(3, 10.0, clock=lambda: clock[0], sleep=_sleep)
        assert [rl.acquire() for _ in range(3)] == [0.0, 0.0, 0.0] and not slept
        assert abs(rl.acquire() - 10.0) < 1e-9                  # الرابع ينتظر خروج أقدمها
        for t in (12.0, 14.0):
            clock[0] = t
            assert rl.acquire() == 0.0
        clock[0] = 15.0                                         # نافذة [10,12,14] ممتلئة
        assert abs(rl.acquire() - 5.0) < 1e-9 and clock[0] == 20.0

    def t_ratelimiter_threadsafe_never_exceeds():
        rl, stamps, lock = RateLimiter(8, 0.4), [], threading.Lock()

        def worker():
            for _ in range(8):
                rl.acquire()
                with lock:
                    stamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(5)]   # 40 طلباً
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stamps.sort()
        assert len(stamps) == 40
        # أي 9 طلبات متتالية لا تقع داخل فترة أقصر من النافذة (هامش لتأخّر تسجيل الطابع)
        worst = min(stamps[i + 8] - stamps[i] for i in range(len(stamps) - 8))
        assert worst >= 0.4 - 0.1, worst
        assert stamps[-1] - stamps[0] >= 4 * 0.4 - 0.15          # 5 دفعات ⇒ ≥ 4 نوافذ

    def t_default_request_limit_is_2000_per_minute():
        assert PIPELINE_DEFAULT_CONFIG['live_max_requests_per_minute'] == 2000
        cfg = {'live_max_requests_per_minute': 2000}
        a = get_request_limiter(cfg)
        assert a.max_calls == 2000 and a.period == 60.0
        assert get_request_limiter(cfg) is a                    # نافذة واحدة مشتركة
        assert get_request_limiter({'live_max_requests_per_minute': 500}).max_calls == 500
        assert get_request_limiter({'live_max_requests_per_minute': 0}) is None

    class _SpyLimiter:
        def __init__(self):
            self.n = 0

        def acquire(self):
            self.n += 1
            return 0.0

    def t_fetch_acquires_per_request_including_retries():
        spy, fake = _SpyLimiter(), _FakeFuturesClient(10_000)
        _with_client(fake, lambda: fetch_data('BTCUSDT', '1h', 4000, rate_limiter=spy))
        assert spy.n == len(fake.calls) == 3, (spy.n, len(fake.calls))     # كل صفحة طلب
        spy2, fake2 = _SpyLimiter(), _FakeFuturesClient(10_000, fail={2: 1})
        _with_client(fake2, lambda: fetch_data('BTCUSDT', '1h', 3000, retries=3,
                                               rate_limiter=spy2))
        assert spy2.n == len(fake2.calls) == 3, (spy2.n, len(fake2.calls))  # وكل إعادة محاولة

    def t_fetch_is_actually_throttled():
        fake = _FakeFuturesClient(10_000)
        t0 = time.monotonic()
        df = _with_client(fake, lambda: fetch_data('BTCUSDT', '1h', 4500,
                                                   rate_limiter=RateLimiter(2, 0.3)))
        assert len(fake.calls) == 3 and len(df) == 4500
        assert time.monotonic() - t0 >= 0.25        # الطلب الثالث انتظر نافذة الحدّ (2 لكل 0.3ث)

    # ── الهدف كعائد مباشر (Qlib idea 1) ─────────────────────────────────────
    def _price_asset(n=60, seed=0):
        """سلسلة يومية واقعية (مشي عشوائي هندسي) — كافية لإحماء كل المؤشرات."""
        rng = np.random.default_rng(seed)
        idx = pd.date_range('2025-01-01', periods=n, freq='D', tz='UTC')
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) * (1 + rng.random(n) * 0.006)
        low = np.minimum(open_, close) * (1 - rng.random(n) * 0.006)
        return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close,
                             'volume': rng.random(n) * 1000 + 50}, index=idx)

    def _reg_cfg(**over):
        cfg = deepcopy(CONFIG)
        cfg.update(tf_order=['1D'], base_tf='1D', window_sizes={'1D': 20}, stride=1,
                   feature_order=None, targets=['close'],
                   enabled_heads={'close_class': True, 'close_reg': True},
                   forecast_horizon=1, reg_target_mode='return', reg_target_clip=None,
                   scaler_type='robust', abstention={'enabled': False},
                   market_context={'enabled': False},
                   funding_rate={'enabled': False}, open_interest={'enabled': False},
                   momentum_orth_natr={'enabled': False}, market_breadth={'enabled': False},
                   phase2_data={'use_intraday_15m': False, 'use_futures_metrics': False})
        cfg.update(over)
        return cfg

    def t_reg_target_matches_direct_return():
        cfg = _reg_cfg()
        df = _price_asset(n=70, seed=1)
        dfs = _quiet(resample_timeframes, df, ['1D'], base_tf='1D', config=cfg)
        X, y, bases, last = _quiet(prepare_single_asset, dfs, config=cfg)
        assert 'close_reg' in y and len(y['close_reg']) > 0
        close = dfs['1D']['close'].values      # بعد add_features، لم يُمسّ close نفسه
        win = cfg['window_sizes']['1D']
        expected = [close[i + win] / close[i + win - 1] - 1.0
                   for i in range(len(y['close_reg']))]
        np.testing.assert_allclose(y['close_reg'], expected, atol=1e-5)
        # يتّفق في الاتجاه مع رأس التصنيف لنفس الهدف (نفس المرجع الآن)؛ الصعود = y > 0
        # في الترميزين (1/0 الحالي أو ±1 القديم)
        agree = (y['close_reg'] > 0) == (y['close_class'] > 0)
        assert agree.all(), agree.mean()

    def t_reg_target_uses_own_kind_not_last_close():
        cfg = _reg_cfg(targets=['high'], window_sizes={'1D': 15},
                       enabled_heads={'high_class': True, 'high_reg': True})
        df = _price_asset(n=60, seed=2)
        # نتحكّم بالشمعتين الأخيرتين فقط: قمة الشمعة الأخيرة (نهاية النافذة)
        # تساوي قمة الشمعة المستقبلية (120=120) بينما إغلاقاهما مختلفان تماماً
        # (100 مقابل 110) — لو استُخدم last_close خطأً كمرجع لكانت النتيجة
        # (120-100)/100=0.20 لا 0؛ الصحيح مرجعه last_high=120 فالنتيجة 0.
        df.iloc[-2, df.columns.get_loc('high')] = 120.0
        df.iloc[-2, df.columns.get_loc('close')] = 100.0
        df.iloc[-1, df.columns.get_loc('high')] = 120.0
        df.iloc[-1, df.columns.get_loc('close')] = 110.0
        dfs = _quiet(resample_timeframes, df, ['1D'], base_tf='1D', config=cfg)
        X, y, bases, last = _quiet(prepare_single_asset, dfs, config=cfg)
        assert len(y['high_reg'])
        assert abs(y['high_reg'][-1]) < 1e-6, y['high_reg'][-1]

    def t_reg_target_clip_default_depends_on_mode():
        df = _price_asset(n=90, seed=4)
        df.iloc[-1, df.columns.get_loc('close')] = 100_000.0   # قفزة ضخمة في آخر شمعة
        df.iloc[-1, df.columns.get_loc('high')] = 100_000.0
        df.iloc[-1, df.columns.get_loc('low')] = 100_000.0
        cfg_ret = _reg_cfg(window_sizes={'1D': 10}, reg_target_clip=None)
        dfs_ret = _quiet(resample_timeframes, df, ['1D'], base_tf='1D', config=cfg_ret)
        _, y_ret, _, _ = _quiet(prepare_single_asset, dfs_ret, config=cfg_ret)
        assert np.all(np.abs(y_ret['close_reg']) <= 1.0 + 1e-6), y_ret['close_reg']
        cfg_win = _reg_cfg(window_sizes={'1D': 10}, reg_target_mode='window_scale',
                           reg_target_clip=None)
        dfs_win = _quiet(resample_timeframes, df, ['1D'], base_tf='1D', config=cfg_win)
        _, y_win, _, _ = _quiet(prepare_single_asset, dfs_win, config=cfg_win)
        assert np.any(np.abs(y_win['close_reg']) > 1.0), \
            "الوضع القديم يجب ألّا يتأثر بحدّ 1.0 الجديد"
        assert np.all(np.abs(y_win['close_reg']) <= 10.0 + 1e-6)

    def t_invert_reg_predictions_return_mode():
        last = np.zeros((3, len(LAST_COLUMNS)))
        last[:, LAST_COLUMNS.index('last_high')] = [100.0, 50.0, 10.0]
        last[:, LAST_COLUMNS.index('last_close')] = [90.0, 45.0, 9.0]
        preds = np.array([0.10, -0.20, 0.0])
        cfg = _reg_cfg()
        got = invert_reg_predictions(preds, 'high_reg', last_candles=last, config=cfg)
        np.testing.assert_allclose(got, [110.0, 40.0, 10.0])
        got_close = invert_reg_predictions(preds, 'close_reg', last_candles=last, config=cfg)
        np.testing.assert_allclose(got_close, [99.0, 36.0, 9.0])

    def t_reg_target_scale_applied_after_clip():
        """y(scale=100) / 100 == y(scale=1)، والقصّ بوحدة العائد قبل الضرب (القفزة تُقصّ عند 1.0 ثم تصبح 100)."""
        df = _price_asset(n=90, seed=4)
        df.iloc[-1, df.columns.get_loc('close')] = 100_000.0   # قفزة ضخمة: عائد ≫ 1
        ys = {}
        for s in (1.0, 100.0):
            cfg = _reg_cfg(window_sizes={'1D': 10}, reg_target_scale=s)
            dfs = _quiet(resample_timeframes, df, ['1D'], base_tf='1D', config=cfg)
            _, ys[s], _, _ = _quiet(prepare_single_asset, dfs, config=cfg)
        np.testing.assert_allclose(ys[100.0]['close_reg'] / 100.0, ys[1.0]['close_reg'], rtol=1e-5, atol=1e-7)
        assert abs(ys[1.0]['close_reg'][-1] - 1.0) < 1e-6 and abs(ys[100.0]['close_reg'][-1] - 100.0) < 1e-4
        np.testing.assert_array_equal(ys[100.0]['close_class'], ys[1.0]['close_class'])   # التصنيف لا يتأثّر

    def t_invert_reg_roundtrip_with_scale():
        """سعر ← هدف ← عكس == السعر المستقبلي، بمقياس 1 و100 (scale من البيانات أو من config)."""
        for s in (1.0, 100.0):
            cfg = _reg_cfg(targets=['high', 'close'], reg_target_scale=s,
                           enabled_heads={'high_class': True, 'high_reg': True,
                                          'close_class': True, 'close_reg': True})
            dfs = _quiet(resample_timeframes, _price_asset(n=70, seed=5), ['1D'], base_tf='1D', config=cfg)
            _, y, _, last = _quiet(prepare_single_asset, dfs, config=cfg)
            for t, fut in (('high', 'future_high_max'), ('close', 'future_close')):
                want = last[:, LAST_COLUMNS.index(fut)]
                np.testing.assert_allclose(invert_reg_predictions(y[f'{t}_reg'], f'{t}_reg', last_candles=last,
                                                                  config=cfg), want, rtol=1e-5)
                np.testing.assert_allclose(invert_reg_predictions(y[f'{t}_reg'], f'{t}_reg', last_candles=last,
                                                                  config=_reg_cfg(), scale=s), want, rtol=1e-5)

    def t_dataset_records_reg_target_scale():
        """المقياس يُحفَظ في البيانات؛ ملف قديم بلا المفتاح يُعكس بـ 1.0 كما كان."""
        names = ['AAAUSDT', 'BBBUSDT']
        for s in (1.0, 100.0):
            cfg = _cfg(reg_target_scale=s)
            data = {n: _quiet(resample_timeframes, _ohlcv(200, seed=i), ['1D'], config=cfg)
                    for i, n in enumerate(names)}
            ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                        config=cfg, max_workers=1)
            assert ds['reg_target_scale'] == s and ds['reg_target_mode'] == 'return'
            y = ds['y_close_reg'] if 'y_close_reg' in ds else ds['y_high_reg']
            head = 'close_reg' if 'y_close_reg' in ds else 'high_reg'
            fut = LAST_COLUMNS.index('future_close' if head == 'close_reg' else 'future_high_max')
            got = invert_reg_predictions(y, head, last_candles=ds['last_candles'], config=_reg_cfg(),
                                         scale=ds.get('reg_target_scale', 1.0))
            np.testing.assert_allclose(got, ds['last_candles'][:, fut], rtol=1e-5)
        old = {k: v for k, v in ds.items() if k != 'reg_target_scale'}
        assert old.get('reg_target_scale', 1.0) == 1.0     # ملف قديم ← 1.0
        got = invert_reg_predictions(np.array([0.01]), 'close_reg', last_candles=ds['last_candles'][:1],
                                     config=_reg_cfg(), scale=old.get('reg_target_scale', 1.0))
        np.testing.assert_allclose(got, ds['last_candles'][:1, LAST_COLUMNS.index('last_close')] * 1.01)

    def t_invert_reg_predictions_window_scale_uses_own_last():
        # المركز آخر سعر من نفس النوع (كما بُني الهدف)، لا bases[:, 0] (وسيط النافذة)؛
        # IQR وحده من bases
        bases = np.array([[100.0, 5.0], [50.0, 2.0]])
        last = np.zeros((2, len(LAST_COLUMNS)))
        last[:, LAST_COLUMNS.index('last_high')] = [104.0, 51.0]
        last[:, LAST_COLUMNS.index('last_close')] = [103.0, 49.0]
        preds = np.array([1.0, -1.5])
        cfg = _reg_cfg(reg_target_mode='window_scale')
        got = invert_reg_predictions(preds, 'high_reg', last_candles=last, bases=bases, config=cfg)
        np.testing.assert_allclose(got, [109.0, 48.0])
        got_c = invert_reg_predictions(preds, 'close_reg', last_candles=last, bases=bases, config=cfg)
        np.testing.assert_allclose(got_c, [108.0, 46.0])
        for kw in ({'bases': bases}, {'last_candles': last}):       # كلاهما مطلوب
            try:
                invert_reg_predictions(preds, 'close_reg', config=cfg, **kw)
            except ValueError:
                continue
            raise AssertionError(f"window_scale بلا {set(('bases', 'last_candles')) - set(kw)} يجب أن يُرفض")

    _ALL_HEADS = {f'{t}_{k}': True for t in ('high', 'low', 'close') for k in ('class', 'reg')}
    _OWN_COLS = (('high', 'future_high_max', 'last_high'), ('low', 'future_low_min', 'last_low'),
                 ('close', 'future_close', 'last_close'))

    def t_class_labels_are_unit_encoded():
        # 1/0 لا ±1: 1 ⇔ المستقبلي > آخر سعر من نفس النوع؛ التعادل = 0
        cfg = _reg_cfg(targets=['high', 'low', 'close'], enabled_heads=_ALL_HEADS)
        df = _price_asset(n=80, seed=5)
        dfs = _quiet(resample_timeframes, df, ['1D'], base_tf='1D', config=cfg)
        X, y, bases, last = _quiet(prepare_single_asset, dfs, config=cfg)
        for t, fut, own in _OWN_COLS:
            yc = y[f'{t}_class']
            assert set(np.unique(yc).tolist()) == {0.0, 1.0}, (t, np.unique(yc))
            expected = last[:, LAST_COLUMNS.index(fut)] > last[:, LAST_COLUMNS.index(own)]
            np.testing.assert_array_equal(yc, expected.astype('float32'))
        # حالة تعادل صريحة: قمة الشمعة المستقبلية = قمة الأخيرة ⇒ 0 (لا -1)
        df.iloc[-2, df.columns.get_loc('high')] = 120.0
        df.iloc[-1, df.columns.get_loc('high')] = 120.0
        dfs = _quiet(resample_timeframes, df, ['1D'], base_tf='1D', config=cfg)
        _, y_tie, _, _ = _quiet(prepare_single_asset, dfs, config=cfg)
        assert y_tie['high_class'][-1] == 0.0, y_tie['high_class'][-1]

    def t_window_scale_reg_centered_on_own_last():
        # هدف window_scale = scale_data(مستقبلي، آخر سعر من نفس النوع، IQR) — لا وسيط
        # النافذة، الذي يكشف جزءاً من الاتجاه وقت الدخول (تسرّب)
        win = 20
        cfg = _reg_cfg(targets=['high', 'low', 'close'], enabled_heads=_ALL_HEADS,
                       window_sizes={'1D': win}, reg_target_mode='window_scale',
                       reg_target_clip=1e9)
        df = _price_asset(n=90, seed=6)
        dfs = _quiet(resample_timeframes, df, ['1D'], base_tf='1D', config=cfg)
        X, y, bases, last = _quiet(prepare_single_asset, dfs, config=cfg)
        assert len(bases) > 0
        for t, fut, own in _OWN_COLS:
            expected = scale_data(last[:, LAST_COLUMNS.index(fut)],
                                  last[:, LAST_COLUMNS.index(own)], bases[:, 1])
            np.testing.assert_allclose(y[f'{t}_reg'], expected, rtol=1e-5, atol=1e-5)
            # نفس مرجع رأس التصنيف ⇒ اتجاهان متّفقان دائماً
            assert np.all((y[f'{t}_reg'] > 0) == (y[f'{t}_class'] > 0)), t
            # والعكس يعيد السعر المستقبلي الفعلي
            inv = invert_reg_predictions(y[f'{t}_reg'], f'{t}_reg', last_candles=last,
                                         bases=bases, config=cfg)
            np.testing.assert_allclose(inv, last[:, LAST_COLUMNS.index(fut)], rtol=1e-4)
        # bases يبقى [وسيط نافذة الإغلاق، IQR] لتطبيع الميزات — لم يُمسّ
        close = dfs['1D']['close'].values
        med = [calc_scale_params(close[i:i + win], 'robust')[0] for i in range(len(bases))]
        np.testing.assert_allclose(bases[:, 0], med, rtol=1e-5)
        assert not np.allclose(bases[:, 0], last[:, LAST_COLUMNS.index('last_close')])

    # ── التطبيع المقطعي عبر الأصول (Qlib idea 2) ────────────────────────────
    def _fake_reg_dataset(groups):
        """``groups``: قائمة ``[{asset_name: عائد}, ...]`` — كل عنصر لحظة زمنية
        واحدة تشارك فيها كل الأصول المذكورة فيه."""
        day = 86_400 * 10 ** 9
        base = pd.Timestamp('2026-01-01', tz='UTC').value
        ts, y = [], []
        for k, grp in enumerate(groups):
            for ret in grp.values():
                ts.append(base + k * day)
                y.append(ret)
        n = len(ts)
        last = np.zeros((n, len(LAST_COLUMNS)))
        last[:, TS_COL] = ts
        return {'close_class': (np.asarray(y) > 0).astype('float32'),
                'close_reg': np.asarray(y, dtype='float32'),
                'last_candles': last}

    def t_cs_normalize_matches_manual_zscore():
        groups = [{'A': 0.05, 'B': -0.02, 'C': 0.01, 'D': 0.10, 'E': -0.05},
                  {'A': 0.02, 'B': 0.03, 'C': -0.01, 'D': 0.00, 'E': 0.04}]
        ds = _fake_reg_dataset(groups)
        cfg = _reg_cfg()
        out = _quiet(cross_sectional_normalize, ds, heads=['close_reg'],
                    method='zscore', min_assets=3, clip=10.0, config=cfg)
        for k, grp in enumerate(groups):
            vals = np.array(list(grp.values()))
            mu, sd = vals.mean(), vals.std()
            expected = (vals - mu) / sd
            got = out['close_reg'][k * len(grp):(k + 1) * len(grp)]
            np.testing.assert_allclose(sorted(got), sorted(expected), atol=1e-5)

    def t_cs_normalize_respects_min_assets():
        groups = [{'A': 0.05, 'B': -0.02}, {'A': 0.02, 'B': 0.03, 'C': -0.01, 'D': 0.00}]
        ds = _fake_reg_dataset(groups)
        cfg = _reg_cfg()
        out = _quiet(cross_sectional_normalize, ds, heads=['close_reg'],
                    method='zscore', min_assets=3, config=cfg)
        # اليوم الأول (أصلان فقط) بقي عائداً خاماً كما هو
        np.testing.assert_allclose(out['close_reg'][:2], ds['close_reg'][:2])
        # اليوم الثاني (٤ أصول ≥ 3) تغيّر فعلاً
        assert not np.allclose(out['close_reg'][2:], ds['close_reg'][2:])

    def t_cs_normalize_does_not_mutate_input():
        groups = [{'A': 0.05, 'B': -0.02, 'C': 0.01, 'D': 0.02}]
        ds = _fake_reg_dataset(groups)
        original = ds['close_reg'].copy()
        _quiet(cross_sectional_normalize, ds, heads=['close_reg'], min_assets=2,
              config=_reg_cfg())
        np.testing.assert_allclose(ds['close_reg'], original)

    def t_cs_normalize_rejects_window_scale_mode():
        ds = _fake_reg_dataset([{'A': 0.05, 'B': -0.02, 'C': 0.01}])
        try:
            _quiet(cross_sectional_normalize, ds, heads=['close_reg'],
                  config=_reg_cfg(reg_target_mode='window_scale'))
        except ValueError:
            return
        raise AssertionError("cs_norm على 'window_scale' يجب أن يُرفض")

    def t_cs_normalize_rank_bounded():
        groups = [{'A': 0.05, 'B': -0.02, 'C': 0.01, 'D': 9.0, 'E': -9.0}]  # شواذ متعمَّدة
        ds = _fake_reg_dataset(groups)
        out = _quiet(cross_sectional_normalize, ds, heads=['close_reg'],
                    method='rank', min_assets=3, config=_reg_cfg())
        assert np.all(out['close_reg'] >= -1.0 - 1e-6) and np.all(out['close_reg'] <= 1.0 + 1e-6)

    def t_invert_cross_sectional_roundtrips_zscore():
        groups = [{'A': 0.05, 'B': -0.02, 'C': 0.01, 'D': 0.02, 'E': -0.01}]
        ds = _fake_reg_dataset(groups)
        cfg = _reg_cfg()
        out = _quiet(cross_sectional_normalize, ds, heads=['close_reg'],
                    method='zscore', min_assets=3, clip=10.0, config=cfg)
        ts = out['last_candles'][:, TS_COL]
        back = invert_cross_sectional(out['close_reg'], ts, 'close_reg', out)
        np.testing.assert_allclose(back, ds['close_reg'], atol=1e-4)

    def t_invert_cross_sectional_rejects_rank():
        ds = _fake_reg_dataset([{'A': 0.05, 'B': -0.02, 'C': 0.01}])
        out = _quiet(cross_sectional_normalize, ds, heads=['close_reg'],
                    method='rank', min_assets=3, config=_reg_cfg())
        try:
            invert_cross_sectional(out['close_reg'], out['last_candles'][:, TS_COL],
                                   'close_reg', out)
        except ValueError:
            return
        raise AssertionError("عكس 'rank' يجب أن يُرفض")

    # ── حدّ فاصل بديل: نسبة العيّنات الخام (compute_global_cutoff) ──────────
    def t_compute_global_cutoff_matches_target_fraction():
        ds = _fake_dataset([400] * 30, stride=1)      # موزَّع بانتظام، بلا تكدّس
        ts = sample_timestamps(ds)
        for pct in (0.10, 0.30, 0.50):
            cutoff = compute_global_cutoff(ds, pct)
            frac = (ts >= cutoff).mean()
            assert abs(frac - pct) < 0.02, (pct, frac)

    def t_compute_global_cutoff_rejects_bad_pct():
        ds = _fake_dataset([100] * 5, stride=1)
        for bad in (0.0, 1.0, -0.1, 1.5):
            try:
                compute_global_cutoff(ds, bad)
            except ValueError:
                continue
            raise AssertionError(f"pct={bad} يجب أن يُرفض")

    def t_extract_test_dates_handles_both_shapes():
        ds = _fake_dataset([300] * 10, stride=1)
        cfg = _cfg()
        _, _, test_flat = _quiet(split_data, ds, config=cfg,
                                 keep_asset_test_separate=False)
        _, _, test_sep = _quiet(split_data, ds, config=cfg,
                                keep_asset_test_separate=True)
        d1 = extract_test_dates(test_flat)
        d2 = extract_test_dates(test_sep)
        assert len(d1) == len(test_flat['last_candles'])
        assert len(d2) == sum(len(v['last_candles']) for v in test_sep.values())

    def t_raw_quantile_converges_to_kept_share_as_data_grows():
        # الفرق بين الطريقتين يتناسب مع (عيّنات فجوة العزل الضائعة / حجم
        # القسم المستهدف) — يصغر كلما كبر val_pct×الإجمالي عن خسارة العزل،
        # حتى على بيانات موزَّعة بانتظام تماماً (5 أصول بنفس المدى الزمني).
        ds = _fake_dataset([15000] * 5, stride=1)
        cfg = _cfg()
        tr1, va1, te1 = _quiet(split_data, ds, config={**cfg, 'split_cutoff_method': 'kept_share'})
        tr2, va2, te2 = _quiet(split_data, ds, config={**cfg, 'split_cutoff_method': 'raw_quantile'})
        for a, b, name in ((tr1, tr2, 'train'), (va1, va2, 'val'), (te1, te2, 'test')):
            n1, n2 = len(a['base_params']), len(b['base_params'])
            assert abs(n1 - n2) / max(n1, n2, 1) < 0.15, (name, n1, n2)

    def t_raw_quantile_fails_loudly_on_crowded_data():
        ds = _fake_dataset(CROWDED)                    # نفس بيانات "التكدّس" الحقيقية
        cfg = _cfg()
        _quiet(split_data, ds, config={**cfg, 'split_cutoff_method': 'kept_share'})  # ينجح
        try:
            _quiet(split_data, ds, config={**cfg, 'split_cutoff_method': 'raw_quantile'})
        except ValueError as exc:
            assert 'min_split_samples' in str(exc) or 'أصغر من الحدّ الأدنى' in str(exc)
            return
        raise AssertionError(
            "raw_quantile على بيانات مكدَّسة كان يجب أن يفشل بوضوح لا أن ينجح صامتاً بقسم شبه فارغ")

    def t_build_leak_free_split_no_cross_asset_overlap():
        ds = _fake_dataset([3000] * 4, stride=1)
        cfg = _cfg()
        train, val, test = _quiet(build_leak_free_split, ds, config=cfg)
        gap = embargo_duration(ds, cfg)
        assert len(train['base_params']) and len(val['base_params'])
        t_tr = sample_timestamps(train)
        t_te = extract_test_dates(test)
        assert len(t_te)
        assert t_te.min() - t_tr.max() >= gap         # بلا أي تداخل بين أي عملتين

    def t_build_leak_free_split_matches_split_data_raw_quantile():
        ds = _fake_dataset([1500] * 4, stride=1)
        cfg = _cfg()
        tr_a, va_a, te_a = _quiet(build_leak_free_split, ds, config=cfg)
        tr_b, va_b, te_b = _quiet(split_data, ds, config={**cfg, 'split_cutoff_method': 'raw_quantile'})
        np.testing.assert_array_equal(tr_a['base_params'], tr_b['base_params'])
        np.testing.assert_array_equal(va_a['base_params'], va_b['base_params'])
        np.testing.assert_array_equal(te_a['base_params'], te_b['base_params'])

    # ── فريم التحميل مستقلّ عن فريم النموذج ─────────────────────────────────
    def t_fetch_data_uses_requested_interval_not_hardcoded():
        fake = _FakeFuturesClient(500)
        _with_client(fake, lambda: fetch_data('BTCUSDT', '4h', 100))
        assert fake.calls[0]['interval'] == '4h', fake.calls[0]

    def t_live_pad_length_scales_with_download_interval():
        cfg1 = _cfg(tf_order=['1D'], window_sizes={'1D': 32}, forecast_horizon=1)
        p_1h = _live_pad_length(['1D'], {'1D': 32}, 1, '1h')
        p_1d = _live_pad_length(['1D'], {'1D': 32}, 1, '1D')
        assert p_1h == 24 * p_1d, (p_1h, p_1d)      # وحدة أدقّ 24× ⇒ padding أكبر 24×

    def t_auto_live_limit_scales_with_download_interval():
        lim_1h = _auto_live_limit(['1D'], {'1D': 32}, 1, 25, '1h')
        lim_1d = _auto_live_limit(['1D'], {'1D': 32}, 1, 25, '1D')
        assert lim_1h > lim_1d, (lim_1h, lim_1d)   # بوحدة أدقّ يلزم شموع أكثر (قد يُقصّ عند BINANCE_MAX_LIMIT)
        lim_1h_small = _auto_live_limit(['1D'], {'1D': 5}, 1, 5, '1h')
        lim_1d_small = _auto_live_limit(['1D'], {'1D': 5}, 1, 5, '1D')
        assert lim_1h_small > lim_1d_small * 10, (lim_1h_small, lim_1d_small)   # بلا قصّ: الفارق واضح

    def t_build_dataset_live_decoupled_intervals_end_to_end():
        class _FakeMultiTF:
            H = {'1h': 3_600_000, '1d': 86_400_000}

            def __init__(self, days=120, end='2026-09-19 20:00'):
                self.end = pd.Timestamp(end, tz='UTC')
                self.days = days
                self.calls = []

            def futures_klines(self, symbol, interval, limit, endTime=None):
                self.calls.append(interval)
                step = self.H[interval]
                n = self.days * (24 if interval == '1h' else 1)
                end_ms = int(self.end.timestamp() * 1000) // step * step
                ts = end_ms - (n - 1 - np.arange(n)) * step
                if endTime is not None:
                    ts = ts[ts <= endTime]
                ts = ts[-limit:]
                rng = np.random.default_rng(1)
                c = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(ts))))
                return [[int(t), str(c[i]), str(c[i] * 1.01), str(c[i] * .99), str(c[i]),
                        "100", int(t) + step - 1, "0", 0, "0", "0", "0"]
                       for i, t in enumerate(ts)]

        cfg = _cfg(tf_order=['1D'], base_tf='1D', window_sizes={'1D': 20},
                  forecast_horizon=1, download_interval='1h')
        fake = _FakeMultiTF()
        old_default_workers = globals().get('default_workers')
        globals()['default_workers'] = lambda n=0: 2
        try:
            ds = _with_client(fake, lambda: build_dataset_live(
                ['AAAUSDT', 'BBBUSDT'], max_workers=1, config=cfg))
        finally:
            globals()['default_workers'] = old_default_workers
        assert set(fake.calls) == {'1h'}, fake.calls    # طُلبت الساعية فقط، لا اليومية إطلاقاً
        assert len(ds['base_params']) > 0
        assert ds['X_1D'].shape[1] == 20                # نافذة يومية سليمة رغم التحميل الساعي

    # ── سياق سوقي عابر للأصول (BTC) ─────────────────────────────────────────
    def t_market_context_columns_names():
        cfg = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                   'return_periods': [1, 3]})
        assert market_context_columns(cfg) == ['MKT_ret_1', 'MKT_ret_3']
        cfg_off = _cfg(market_context={'enabled': False})
        assert market_context_columns(cfg_off) == []

    def t_add_market_context_aligns_and_zeroes_reference():
        cfg = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                   'return_periods': [1]})
        idx = pd.date_range('2025-01-01', periods=10, freq='D', tz='UTC')
        btc_close = pd.Series(np.linspace(100, 109, 10), index=idx)
        market_dfs = {'1D': pd.DataFrame({'MKT_ret_1': btc_close.pct_change(1)}, index=idx)}
        alt_df = pd.DataFrame({'close': np.linspace(1, 2, 10)}, index=idx)

        out_alt = add_market_context({'1D': alt_df}, market_dfs, is_reference=False, config=cfg)
        np.testing.assert_allclose(out_alt['1D']['MKT_ret_1'].iloc[1:].values,
                                   btc_close.pct_change(1).iloc[1:].values)

        out_btc = add_market_context({'1D': alt_df}, market_dfs, is_reference=True, config=cfg)
        assert (out_btc['1D']['MKT_ret_1'] == 0.0).all()

    def t_market_context_reaches_dataset_via_build_dataset():
        # MKT_ret_1 مستبعد من المدخلات افتراضياً منذ فرز الميزات — يُعاد هنا ليُختبَر وصوله
        cfg = _cfg(market_context={'enabled': True, 'reference_symbol': 'AAAUSDT',
                                   'return_periods': [1]},
                   exclude_from_features=['open', 'high', 'low'])
        names = ['AAAUSDT', 'BBBUSDT']
        data = {n: _quiet(resample_timeframes, _ohlcv(200, seed=i), ['1D'], config=cfg)
               for i, n in enumerate(names)}
        ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                   config=cfg, max_workers=1)
        assert 'MKT_ret_1' in ds['feature_order']
        i = ds['feature_order'].index('MKT_ret_1')
        b_bbb = next(b for b in ds['asset_bounds'] if b['name'] == 'BBBUSDT')
        b_aaa = next(b for b in ds['asset_bounds'] if b['name'] == 'AAAUSDT')
        # BBBUSDT (ليست المرجع) لها قيم سياق غير صفرية في مكان ما على الأقل
        assert np.any(ds['X_1D'][b_bbb['start']:b_bbb['end'], :, i] != 0.0)
        # AAAUSDT هي المرجع نفسه: صفر دائماً
        assert np.all(ds['X_1D'][b_aaa['start']:b_aaa['end'], :, i] == 0.0)

    def t_market_corr_regime_columns_and_values():
        # (أ) corr_windows=[] (افتراضي) لا يُغيّر السلوك القديم
        cfg_off = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                       'return_periods': [1]})
        assert market_context_columns(cfg_off) == ['MKT_ret_1']

        cfg = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                   'return_periods': [1], 'corr_windows': [5]})
        assert market_context_columns(cfg) == ['MKT_ret_1', 'MKT_CORR_5']

        # corr_windows بلا return_periods=[1] يُعطَّل بصمت (شرط موثَّق)
        cfg_no1 = _cfg(market_context={'enabled': True, 'return_periods': [3],
                                       'corr_windows': [5]})
        assert market_context_columns(cfg_no1) == ['MKT_ret_3']

        # (ب) القيم صحيحة رياضياً: أصل مطابق تماماً لعائد BTC → ارتباط=1.0،
        # أصل معاكس تماماً → ارتباط=-1.0، أصل عشوائي مستقلّ → قريب من 0
        idx = pd.date_range('2025-01-01', periods=40, freq='D', tz='UTC')
        rng = np.random.default_rng(0)
        btc_ret = rng.normal(0, 0.02, 40)
        btc_close = pd.Series(100 * np.cumprod(1 + btc_ret), index=idx)
        market_dfs = {'1D': pd.DataFrame({'MKT_ret_1': btc_close.pct_change(1)}, index=idx)}

        same_close = pd.Series(50 * np.cumprod(1 + btc_ret), index=idx)  # نفس عوائد BTC بالضبط
        opp_close = pd.Series(50 * np.cumprod(1 - btc_ret), index=idx)   # معاكسة تماماً
        rand_close = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.02, 40)), index=idx)

        for close_series, expected in ((same_close, 1.0), (opp_close, -1.0)):
            df = pd.DataFrame({'close': close_series}, index=idx)
            out = add_market_context({'1D': df}, market_dfs, is_reference=False, config=cfg)
            val = out['1D']['MKT_CORR_5'].iloc[-1]
            assert np.isclose(val, expected, atol=1e-9), f'توقّعت {expected}، حصلت على {val}'

        df_rand = pd.DataFrame({'close': rand_close}, index=idx)
        out_rand = add_market_context({'1D': df_rand}, market_dfs, is_reference=False, config=cfg)
        assert abs(out_rand['1D']['MKT_CORR_5'].iloc[-1]) < 0.9, 'أصل عشوائي مستقلّ يجب ألا يُعطي ارتباطاً قوياً ثابتاً'

        # (ج) العملة المرجعية نفسها: صفر دائماً حتى لأعمدة MKT_CORR
        out_ref = add_market_context({'1D': df}, market_dfs, is_reference=True, config=cfg)
        assert (out_ref['1D']['MKT_CORR_5'] == 0.0).all()

        # (د) سببية: تقطيع السلسلة عند k لا يُغيّر MKT_CORR عند تلك النقطة
        df_same = pd.DataFrame({'close': same_close}, index=idx)
        out_full = add_market_context({'1D': df_same}, market_dfs, is_reference=False, config=cfg)
        for k in (20, 30, 39):
            market_dfs_k = {'1D': market_dfs['1D'].iloc[:k]}
            df_k = df_same.iloc[:k]
            out_k = add_market_context({'1D': df_k}, market_dfs_k, is_reference=False, config=cfg)
            full_val = out_full['1D']['MKT_CORR_5'].iloc[k - 1]
            trunc_val = out_k['1D']['MKT_CORR_5'].iloc[-1]
            assert np.isclose(full_val, trunc_val, equal_nan=True), \
                f'تسرّب معلومة مستقبلية عند k={k}: {full_val} != {trunc_val}'

    def t_market_lag_ret_columns_and_values():
        # (أ) lag_windows=[] (افتراضي) لا يُغيّر السلوك القديم
        cfg_off = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                       'return_periods': [1]})
        assert market_context_columns(cfg_off) == ['MKT_ret_1']

        cfg = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                   'return_periods': [1], 'lag_windows': [1, 3]})
        assert market_context_columns(cfg) == ['MKT_ret_1', 'MKT_LAG_RET_1', 'MKT_LAG_RET_3']

        # (ب) القيمة الصحيحة: MKT_LAG_RET_k عند اللحظة t تساوي عائد BTC
        # (MKT_ret_1) عند اللحظة t-k بالضبط — لا اليوم نفسه.
        idx = pd.date_range('2025-01-01', periods=30, freq='D', tz='UTC')
        rng = np.random.default_rng(1)
        btc_ret = rng.normal(0, 0.02, 30)
        btc_close = pd.Series(100 * np.cumprod(1 + btc_ret), index=idx)
        market_dfs = build_market_context(data={'BTCUSDT': {'1D': pd.DataFrame({'close': btc_close}, index=idx)}},
                                          config={**CONFIG, 'tf_order': ['1D'], 'market_context': cfg['market_context']})
        ref_ret_1 = btc_close.pct_change(1)
        for k in (1, 3):
            expected = ref_ret_1.shift(k)
            actual = market_dfs['1D'][f'MKT_LAG_RET_{k}']
            pd.testing.assert_series_equal(actual, expected, check_names=False,
                                           check_freq=False)

        # (ج) سببية: MKT_LAG_RET_k عند t يعتمد فقط على بيانات حتى t-k — تقطيع
        # السلسلة عند أي نقطة لاحقة لا يُغيّر القيمة عند نقطة سابقة.
        for cut in (15, 20, 29):
            market_dfs_cut = build_market_context(
                data={'BTCUSDT': {'1D': pd.DataFrame({'close': btc_close.iloc[:cut]}, index=idx[:cut])}},
                config={**CONFIG, 'tf_order': ['1D'], 'market_context': cfg['market_context']})
            for k in (1, 3):
                full_val = market_dfs['1D'][f'MKT_LAG_RET_{k}'].iloc[cut - 1]
                cut_val = market_dfs_cut['1D'][f'MKT_LAG_RET_{k}'].iloc[-1]
                assert np.isclose(full_val, cut_val, equal_nan=True), \
                    f'تسرّب معلومة مستقبلية عند k={k}, cut={cut}: {full_val} != {cut_val}'

    def t_market_crash_rebound_regime_columns_and_values():
        # (أ) crash_rebound_regime=None (افتراضي) لا يُغيّر السلوك القديم
        cfg_off = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                       'return_periods': [1]})
        assert market_context_columns(cfg_off) == ['MKT_ret_1']

        crr_settings = {'high_lookback': 60, 'recent_window': 30, 'drawdown_threshold': -0.30}
        cfg = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                   'return_periods': [1], 'crash_rebound_regime': crr_settings})
        assert market_context_columns(cfg) == ['MKT_ret_1', 'MKT_CRASH_REBOUND_REGIME']

        # (ب) هبوط صناعي حادّ (50%) في سلسلة العملة المرجعية يُفعِّل العلم
        # لكل الأصول الأخرى (نظام سوقي مشترك، لا خاصّ بأصل بعينه)
        n = 150
        idx = pd.date_range('2024-01-01', periods=n, freq='D', tz='UTC')
        rng = np.random.default_rng(13)
        ref_close = np.concatenate([
            100 + np.arange(60) * 0.5,
            np.linspace(130, 65, 20),
            65 + np.cumsum(rng.normal(0, 0.2, n - 80)),
        ])
        market_dfs = build_market_context(
            data={'BTCUSDT': {'1D': pd.DataFrame({'close': ref_close}, index=idx)}},
            config={**CONFIG, 'tf_order': ['1D'], 'market_context': cfg['market_context']})
        regime = market_dfs['1D']['MKT_CRASH_REBOUND_REGIME']
        assert regime.iloc[85:100].mean() == 1.0, 'يجب أن يُفعَّل العلم بعد هبوط ≥30% للمرجع مباشرة'
        assert regime.iloc[30] == 0.0, 'يجب أن يبقى العلم صفراً قبل حدوث أي هبوط'

        # هذا العلم مشترك بين كل الأصول (يُقرَأ من المرجع لا من الأصل نفسه):
        other_close = pd.Series(50 + np.cumsum(rng.normal(0, 0.1, n)), index=idx)  # أصل آخر مستقرّ تماماً
        df_other = pd.DataFrame({'close': other_close}, index=idx)
        out_other = add_market_context({'1D': df_other}, market_dfs, is_reference=False, config=cfg)
        assert out_other['1D']['MKT_CRASH_REBOUND_REGIME'].iloc[85:100].mean() == 1.0, \
            'العلم مشترك من المرجع — يجب أن يُفعَّل حتى لأصل آخر مستقرّ تماماً'

        # (ج) العملة المرجعية نفسها: صفر دائماً (نفس سلوك كل أعمدة MKT_)
        out_ref = add_market_context({'1D': pd.DataFrame({'close': ref_close}, index=idx)},
                                     market_dfs, is_reference=True, config=cfg)
        assert (out_ref['1D']['MKT_CRASH_REBOUND_REGIME'] == 0.0).all()

    def t_momentum_orth_natr():
        _test_momentum_orth_natr()

    def t_phase2_toggles_off_leave_pipeline_unchanged():
        # المفتاحان مُطفآن (أو "auto" بلا مجلدات) ⇒ لا عمود جديد، والخطافات تُعيد dfs نفسها
        cfg = _cfg()
        assert intraday_15m_columns(cfg) == [] and futures_metrics_columns(cfg) == []
        assert intraday_efficiency_columns(cfg) == []
        idx = pd.date_range('2025-01-01', periods=5, freq='D', tz='UTC')
        dfs = {'1D': pd.DataFrame({'close': np.linspace(1, 2, 5)}, index=idx)}
        assert add_intraday_15m_features(dfs, 'AAAUSDT', cfg) is dfs
        import tempfile
        empty = tempfile.mkdtemp()
        cfg_auto = _cfg(phase2_data={'use_intraday_15m': 'auto', 'use_futures_metrics': 'auto',
                                     'data_root': empty})
        assert resolve_phase2_toggles(cfg_auto, verbose=False) == {
            'use_intraday_15m': False, 'use_futures_metrics': False}
        assert infer_feature_columns(cfg_auto) == infer_feature_columns(cfg)

    def t_phase2_hooks_from_drive_files():
        # ملفات بصيغة fetch_history_vision_colab الفعلية في جذر مؤقت ← الخطافات ← build_dataset
        if _phase2_module() is None:
            print("   (تخطٍّ: tools/intraday_features.py غير متاح في هذا المجلد)")
            return
        import gzip
        import tempfile
        root = Path(tempfile.mkdtemp())
        for sub in ('history_15m', 'funding_rate', 'futures_metrics'):
            (root / sub).mkdir()
        rng = np.random.default_rng(5)
        n = 200 * 96
        t15 = pd.date_range('2025-01-01', periods=n, freq='15min', tz='UTC')
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        open_ = np.r_[100.0, close[:-1]]
        vol = rng.random(n) * 10 + 1
        k = pd.DataFrame({'timestamp': t15.asi8 // 10 ** 6, 'datetime_utc': t15.strftime('%Y-%m-%d %H:%M:%S'),
                          'open': open_, 'high': np.maximum(open_, close) * 1.001,
                          'low': np.minimum(open_, close) * 0.999, 'close': close, 'volume': vol,
                          'quote_volume': vol * close, 'trades': rng.integers(50, 150, n),
                          'taker_buy_volume': vol * rng.random(n), 'taker_buy_quote_volume': 0.0})
        with gzip.open(root / 'history_15m' / 'AAAUSDT.csv.gz', 'wt') as f:
            k.to_csv(f, index=False)
        ft = pd.date_range('2025-01-01', periods=600, freq='8h', tz='UTC')
        pd.DataFrame({'timestamp': ft.strftime('%Y-%m-%d %H:%M:%S+00:00'),
                      'funding_rate': 0.0005 * np.sin(np.linspace(0, 30, 600))}).to_csv(
            root / 'funding_rate' / 'AAAUSDT.csv', index=False)
        mt = pd.date_range('2025-03-01', '2025-07-20', freq='1h', tz='UTC')     # metrics تبدأ لاحقاً
        m = len(mt)
        with gzip.open(root / 'futures_metrics' / 'AAAUSDT.csv.gz', 'wt') as f:
            pd.DataFrame({'timestamp': mt.strftime('%Y-%m-%d %H:%M:%S+00:00'),
                          'sum_open_interest': 1000 + np.cumsum(rng.normal(0, 5, m)),
                          'sum_open_interest_value': 1.0,
                          'count_toptrader_long_short_ratio': 1 + rng.random(m),
                          'sum_toptrader_long_short_ratio': 1 + rng.random(m),
                          'count_long_short_ratio': 1 + rng.random(m),
                          'sum_taker_long_short_vol_ratio': 0.5 + rng.random(m)}).to_csv(f, index=False)

        cfg = _cfg(phase2_data={'use_intraday_15m': 'auto', 'use_futures_metrics': 'auto',
                                'data_root': str(root)},
                   funding_rate={'enabled': True, 'as_feature': True, 'drive_dir': 'funding_rate',
                                 'zscore_window': 30},
                   open_interest={'enabled': True, 'as_feature': True, 'drive_dir': 'open_interest',
                                  'change_period': 1})
        assert resolve_phase2_toggles(cfg, verbose=False) == {
            'use_intraday_15m': True, 'use_futures_metrics': True}
        new_cols = (intraday_15m_columns(cfg) + futures_metrics_columns(cfg)
                    + ['FUND_sum_1d', 'FUND_sum_3d', 'EFF_RATIO_24H', 'VWAP_DEVIATION', 'VOL_CONC_HHI'])
        for c in new_cols:
            assert classify_feature(c) != DEFAULT_KIND, c
        feats = infer_feature_columns(cfg)
        assert all(c in feats for c in new_cols), [c for c in new_cols if c not in feats]

        # OI بلا مجلد open_interest ← يُؤخذ من futures_metrics.sum_open_interest
        idx = pd.date_range('2025-01-01', periods=200, freq='D', tz='UTC')
        dfs = {'1D': pd.DataFrame({'close': np.ones(200)}, index=idx)}
        for hook in (add_funding_oi_features, add_intraday_efficiency_features, add_intraday_vwap_features,
                     add_intraday_volume_concentration_features, add_intraday_15m_features):
            dfs = hook(dfs, 'AAAUSDT', cfg)
        d = dfs['1D']
        assert not d[new_cols].isna().any().any()
        pre = d.index < pd.Timestamp('2025-03-01', tz='UTC')
        assert (d.loc[pre, 'MET_available'] == 0).all() and (d.loc[pre, 'LSR_GLOBAL'] == 0).all()
        assert (d.loc[~pre & (d.index < '2025-07-19'), 'MET_available'] == 1).all()
        assert (d.loc[d.index >= '2025-07-22', 'MET_available'] == 0).all()     # بعد نهاية الأرشيف: لا ffill
        assert (d['ITD_available'] == 1).all() and (d['FUND_available'] == 1).all()
        # شمعة D (إغلاقها D+1 00:00) ترى تسويات D 00/08/16 فقط
        fr = pd.read_csv(root / 'funding_rate' / 'AAAUSDT.csv')['funding_rate'].to_numpy()
        assert np.isclose(d['FUND_sum_1d'].iloc[10], fr[30:33].sum())

        data = {'AAAUSDT': _quiet(resample_timeframes, _ohlcv(200, seed=0), ['1D'], config=cfg)}
        ds = _quiet(build_dataset_from_preloaded, [{'name': 'AAAUSDT'}], data, config=cfg, max_workers=1)
        for c in ('ITD_RVOL', 'LSR_GLOBAL', 'FUND_sum_1d', 'VWAP_DEVIATION'):
            i = ds['feature_order'].index(c)
            assert np.any(ds['X_1D'][:, :, i] != 0.0), c

    def t_optional_feature_kinds_explicit():
        # كل ميزة اختيارية تُصنَّف بالنوع المقصود صراحةً — لا بالتقاط بادئة
        # عامّة خاطئة (VWAP/SUPERT/PSAR → مستوى سعري، MOM → وحدة سعرية،
        # VOL_ → حجم لوغاريتمي). أيّ عودة لهذا الخلل تُفسد القيمة المستخرَجة
        # بصمت (لا خطأ ظاهر) فتُفسد كل اختبار IC عليها.
        expected = {
            'VWAP_DEVIATION': SIGN_ROBUST, 'VWAP_DEVIATION_available': BINARY_FLAG,
            'MKT_LAG_RET_3': SIGN_ROBUST,
            'SUPERT_DIR_10': UNIT_SYM, 'SUPERT_STRETCH_10': PERCENT, 'PSAR_DIR': UNIT_SYM,
            'VOL_ASYMMETRY_20': UNIT_SYM, 'MKT_CORR_20': UNIT_SYM, 'PCT_FROM_ATH': UNIT_SYM,
            'MOM_RANK_24': UNIT_0_1, 'MKT_BREADTH_24': UNIT_0_1, 'DIR_CONSISTENCY_20': UNIT_0_1,
            'EFF_RATIO_24H': UNIT_0_1, 'EFF_RATIO_available': BINARY_FLAG,
            'VOL_CONC_HHI': UNIT_0_1, 'VOL_CONC_HHI_available': BINARY_FLAG,
            'ITD_TRADES_LOG': LOG_CENTERED, 'ITD_available': BINARY_FLAG, 'MET_available': BINARY_FLAG,
            'CRASH_REBOUND_REGIME': UNIT_0_1, 'MKT_CRASH_REBOUND_REGIME': UNIT_0_1,
            'VOL_TERM_3_30': ZSCORE, 'RISK_ADJ_MOM_20': ZSCORE, 'MKT_BETA_20': ZSCORE,
            'MOM_ORTH_NATR': UNIT_SYM,
        }
        for name, kind in expected.items():
            got = classify_feature(name)
            assert got == kind, f'{name}: توقّعت {kind}، حصلت على {got}'
        # البادئات العامّة الأصلية لم تتغيّر لأسمائها المقصودة
        assert classify_feature('VWAP_D_20') == PRICE_LEVEL
        assert classify_feature('SUPERT_7_3.0') == PRICE_LEVEL
        assert classify_feature('MOM_10') == PRICE_SCALE
        assert classify_feature('VOL_SMA_20') == LOG_VOLUME
        assert classify_feature('volume') == LOG_VOLUME
        # والقيمة المستخرَجة الآن تحفظ ترتيب VWAP_DEVIATION الخام عبر العيّنات
        # (خلافاً للمستوى السعري الذي كان يُشبعها عند -5 بمعزل عن قيمتها)
        rng = np.random.default_rng(3)
        N, T = 50, 32
        close = 100.0 + np.cumsum(rng.normal(0, 1, (N, T)), axis=1)
        dev = rng.normal(0, 0.02, (N, T))
        X = np.stack([close, dev], axis=2).astype('float32')
        centers = np.median(close, axis=1)
        scales = np.subtract(*np.percentile(close, [75, 25], axis=1))
        out = process_windows(X, ['close', 'VWAP_DEVIATION'], centers, scales,
                              config={**CONFIG, 'clip_abs': 5.0})
        last = out[:, -1, 1]
        assert np.mean(np.abs(last) >= 4.999) < 0.1, 'VWAP_DEVIATION ما زالت مشبعة عند حدّ القصّ'
        assert np.all(np.sign(last[dev[:, -1] != 0]) == np.sign(dev[dev[:, -1] != 0, -1])), \
            'يجب أن تُحفَظ إشارة الانحراف الخام'

    def t_market_beta_columns_and_values():
        # (أ) beta_windows=[] (افتراضي) لا يُغيّر السلوك القديم
        cfg_off = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                       'return_periods': [1]})
        assert market_context_columns(cfg_off) == ['MKT_ret_1']

        cfg = _cfg(market_context={'enabled': True, 'reference_symbol': 'BTCUSDT',
                                   'return_periods': [1], 'beta_windows': [30]})
        assert market_context_columns(cfg) == ['MKT_ret_1', 'MKT_BETA_30']

        # beta_windows بلا return_periods=[1] يُعطَّل بصمت (نفس شرط corr_windows)
        cfg_no1 = _cfg(market_context={'enabled': True, 'return_periods': [3],
                                       'beta_windows': [30]})
        assert market_context_columns(cfg_no1) == ['MKT_ret_3']

        # (ب) القيم صحيحة رياضياً: أصل عائده = k×عائد المرجع بالضبط (+ضجيج
        # مستقلّ صغير كسر التطابق التام) يجب أن يُعطي بيتا ≈ k
        idx = pd.date_range('2025-01-01', periods=80, freq='D', tz='UTC')
        rng = np.random.default_rng(2)
        btc_ret = rng.normal(0, 0.02, 80)
        btc_close = pd.Series(100 * np.cumprod(1 + btc_ret), index=idx)
        market_dfs = {'1D': pd.DataFrame({'MKT_ret_1': btc_close.pct_change(1)}, index=idx)}

        for k in (0.5, 2.0):
            asset_ret = k * btc_ret + rng.normal(0, 1e-6, 80)  # ضجيج ضئيل جداً لكسر var()=nan
            asset_close = pd.Series(50 * np.cumprod(1 + asset_ret), index=idx)
            df = pd.DataFrame({'close': asset_close}, index=idx)
            out = add_market_context({'1D': df}, market_dfs, is_reference=False, config=cfg)
            beta_val = out['1D']['MKT_BETA_30'].iloc[-1]
            assert np.isclose(beta_val, k, atol=0.05), f'توقّعت بيتا≈{k}، حصلت على {beta_val}'

        # أصل عشوائي مستقلّ تماماً عن المرجع: بيتا قريبة من صفر
        rand_close = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.02, 80)), index=idx)
        df_rand = pd.DataFrame({'close': rand_close}, index=idx)
        out_rand = add_market_context({'1D': df_rand}, market_dfs, is_reference=False, config=cfg)
        assert abs(out_rand['1D']['MKT_BETA_30'].iloc[-1]) < 0.5, 'أصل مستقلّ يجب ألا يُعطي بيتا كبيرة ثابتة'

        # (ج) العملة المرجعية نفسها: صفر دائماً
        out_ref = add_market_context({'1D': df}, market_dfs, is_reference=True, config=cfg)
        assert (out_ref['1D']['MKT_BETA_30'] == 0.0).all()

        # (د) سببية: تقطيع السلسلة عند k لا يُغيّر MKT_BETA عند تلك النقطة
        for cut in (50, 65, 79):
            market_dfs_k = {'1D': market_dfs['1D'].iloc[:cut]}
            df_k = df.iloc[:cut]
            out_k = add_market_context({'1D': df_k}, market_dfs_k, is_reference=False, config=cfg)
            full_val = out['1D']['MKT_BETA_30'].iloc[cut - 1]
            trunc_val = out_k['1D']['MKT_BETA_30'].iloc[-1]
            assert np.isclose(full_val, trunc_val, equal_nan=True), \
                f'تسرّب معلومة مستقبلية عند cut={cut}: {full_val} != {trunc_val}'

    # ── إعادة تدريب دورية (rolling_splits) ──────────────────────────────────
    def t_rolling_schedule_expanding_basic():
        ds = _fake_dataset([3000] * 3, stride=1)
        cfg = _cfg()
        sched = rolling_split_schedule(ds, test_span='20D', val_span='10D',
                                       initial_train_span='60D', config=cfg)
        assert len(sched) >= 2
        for w in sched:
            assert w['train_end'] < w['val_end'] < w['test_end']
        # النافذة الأولى تبدأ من أقدم عيّنة، والثانية train_end أبعد من الأولى (تمدّد)
        assert sched[0]['train_start'] == sample_timestamps(ds).min()
        assert sched[1]['train_end'] > sched[0]['train_end']
        assert sched[1]['train_start'] == sched[0]['train_start']   # نفس البداية دائماً (متمدّدة)

    def t_rolling_schedule_sliding_basic():
        ds = _fake_dataset([3000] * 3, stride=1)
        cfg = _cfg()
        sched = rolling_split_schedule(ds, test_span='20D', val_span='10D',
                                       train_span='60D', config=cfg)
        assert len(sched) >= 2
        for w in sched:
            assert (w['train_end'] - w['train_start']) == pd.Timedelta('60D')
        assert sched[1]['train_start'] > sched[0]['train_start']    # ينزلق فعلاً

    def t_rolling_schedule_requires_initial_train_span():
        ds = _fake_dataset([1000] * 2, stride=1)
        try:
            rolling_split_schedule(ds, test_span='20D', val_span='10D', config=_cfg())
        except ValueError as exc:
            assert 'initial_train_span' in str(exc)
            return
        raise AssertionError("نافذة متمدّدة بلا initial_train_span يجب أن تُرفض")

    def t_rolling_splits_no_leak_between_train_and_test_per_window():
        ds = _fake_dataset([4000] * 3, stride=1)
        cfg = _cfg(min_split_samples=20)
        windows = _quiet(rolling_splits, ds, test_span='20D', val_span='10D',
                         initial_train_span='60D', config=cfg)
        assert len(windows) >= 2
        gap = embargo_duration(ds, cfg)
        for train, val, test in windows:
            assert len(train['base_params']) and len(val['base_params'])
            t_tr, t_va = sample_timestamps(train), sample_timestamps(val)
            t_te = extract_test_dates(test)
            assert len(t_te)
            assert t_va.min() - t_tr.max() >= gap
            assert t_te.min() - t_va.max() >= gap

    def t_rolling_splits_max_windows_respected():
        ds = _fake_dataset([6000] * 3, stride=1)
        cfg = _cfg(min_split_samples=10)
        windows = _quiet(rolling_splits, ds, test_span='10D', val_span='5D',
                         initial_train_span='30D', max_windows=2, config=cfg)
        assert len(windows) == 2

    # ── معدّل التمويل والفائدة المفتوحة ──────────────────────────────────────
    def t_fetch_funding_rate_paginates_and_merges():
        class _FakeFunding:
            def __init__(self, n=2500, end_ms=1_800_000_000_000):
                self.n, self.end_ms, self.calls = n, end_ms, []
                self.step = 8 * 3_600_000

            def futures_funding_rate(self, symbol, limit, endTime=None):
                self.calls.append(limit)
                assert limit <= FUNDING_MAX_LIMIT
                idx = [i for i in range(self.n)
                      if endTime is None or self._ts(i) <= endTime]
                return [{"symbol": symbol, "fundingTime": self._ts(i),
                        "fundingRate": str(0.0001 * (i % 7 - 3))} for i in idx[-limit:]]

            def _ts(self, i):
                return self.end_ms - (self.n - 1 - i) * self.step

        fake = _FakeFunding()
        df = _with_client(fake, lambda: fetch_funding_rate('BTCUSDT', 1800))
        assert len(fake.calls) == 2 and fake.calls == [1000, 800]
        assert len(df) == 1800
        assert df['timestamp'].is_monotonic_increasing and not df['timestamp'].duplicated().any()
        step = df['timestamp'].diff().dropna().unique()
        assert len(step) == 1 and step[0] == pd.Timedelta(hours=8)

    def t_fetch_open_interest_hist_capped_by_platform_lookback():
        class _FakeOI:
            """يحاكي قيد Binance: لا يُرجع شيئاً أبعد من ٣٠ يوماً مهما طُلب."""
            def __init__(self, end_ms=1_800_000_000_000):
                self.end_ms, self.calls = end_ms, []
                self.step = 3_600_000
                self.available = 30 * 24     # ٣٠ يوماً بدقة ساعة

            def futures_open_interest_hist(self, symbol, period, limit, endTime=None):
                self.calls.append(limit)
                cutoff = self.end_ms - self.available * self.step
                lo = endTime if endTime is not None else self.end_ms
                idx = [t for t in range(cutoff, lo + 1, self.step)]
                idx = idx[-limit:]
                return [{"symbol": symbol, "timestamp": t,
                        "sumOpenInterest": "1000.0"} for t in idx]

        fake = _FakeOI()
        df = _with_client(fake, lambda: fetch_open_interest_hist(
            'BTCUSDT', 5000, period='1h'))
        assert len(df) <= 30 * 24 + 1          # لم يتجاوز قيد المنصّة رغم limit=5000
        assert len(fake.calls) >= 2            # احتاج أكثر من صفحة (٥٠٠ الحدّ الأقصى للطلب)

    def t_save_and_load_funding_open_interest_accumulates():
        import tempfile
        class _FakeFR:
            def __init__(self, rows):
                self.rows = rows

            def futures_funding_rate(self, symbol, limit, endTime=None):
                return self.rows[-limit:]

            def futures_open_interest_hist(self, symbol, period, limit, endTime=None):
                return []

        cfg = _cfg(funding_rate={'enabled': True, 'drive_dir': 'funding_rate'},
                  open_interest={'enabled': False})
        tmp = Path(tempfile.mkdtemp())
        old_mount = globals()['mount_drive']
        globals()['mount_drive'] = lambda mount_point=None, config=None: tmp
        try:
            batch1 = [{"symbol": "BTCUSDT", "fundingTime": 1_700_000_000_000 + i * 28_800_000,
                      "fundingRate": "0.0001"} for i in range(5)]
            fake1 = _FakeFR(batch1)
            _with_client(fake1, lambda: save_funding_open_interest(
                ['BTCUSDT'], funding_limit=5, config=cfg, verbose=False))
            first = load_funding_open_interest('BTCUSDT', 'funding_rate', config=cfg)
            assert len(first) == 5

            batch2 = batch1 + [{"symbol": "BTCUSDT",
                                "fundingTime": 1_700_000_000_000 + 5 * 28_800_000,
                                "fundingRate": "0.0002"}]
            fake2 = _FakeFR(batch2)
            _with_client(fake2, lambda: save_funding_open_interest(
                ['BTCUSDT'], funding_limit=6, config=cfg, verbose=False))
            second = load_funding_open_interest('BTCUSDT', 'funding_rate', config=cfg)
            assert len(second) == 6            # تراكم بلا تكرار، لا استبدال بالكامل
        finally:
            globals()['mount_drive'] = old_mount

    # ── دمج معدّل التمويل/الفائدة المفتوحة كميزات ────────────────────────────
    def t_funding_oi_columns_names():
        cfg = _cfg(funding_rate={'enabled': True, 'as_feature': True, 'zscore_window': 90},
                  open_interest={'enabled': True, 'as_feature': True, 'change_period': 1})
        assert funding_oi_feature_columns(cfg) == [
            'FUND_rate', 'FUND_rate_z', 'FUND_available', 'OI_chg_1', 'OI_available']

        cfg_off = _cfg(funding_rate={'enabled': False}, open_interest={'enabled': False})
        assert funding_oi_feature_columns(cfg_off) == []

        # enabled (جلب/أرشفة) بلا as_feature (إدماج) لا يُنتج أي عمود مدخل
        cfg_fetch_only = _cfg(funding_rate={'enabled': True, 'as_feature': False},
                              open_interest={'enabled': True, 'as_feature': False})
        assert funding_oi_feature_columns(cfg_fetch_only) == []

    def t_add_funding_oi_features_aligns_and_flags_availability():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        old_mount = globals()['mount_drive']
        globals()['mount_drive'] = lambda mount_point=None, config=None: tmp
        try:
            cfg = _cfg(funding_rate={'enabled': True, 'drive_dir': 'funding_rate',
                                     'as_feature': True, 'zscore_window': 5},
                      open_interest={'enabled': True, 'drive_dir': 'open_interest',
                                    'as_feature': True, 'change_period': 1})
            fr = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=10, freq='8h', tz='UTC'),
                'funding_rate': np.linspace(0.0001, 0.001, 10)})
            fr.to_csv(_funding_oi_drive_path(tmp, 'funding_rate', 'AAAUSDT', cfg), index=False)
            oi = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-03', periods=5, freq='1h', tz='UTC'),
                'open_interest': np.linspace(1000.0, 1100.0, 5)})
            oi.to_csv(_funding_oi_drive_path(tmp, 'open_interest', 'AAAUSDT', cfg), index=False)

            idx = pd.date_range('2025-01-01', periods=5, freq='D', tz='UTC')
            df = pd.DataFrame({'close': np.linspace(1, 2, 5)}, index=idx)
            out = add_funding_oi_features({'1D': df}, 'AAAUSDT', config=cfg)['1D']

            assert list(out['FUND_available']) == [1.0, 1.0, 1.0, 1.0, 1.0]
            # الفائدة المفتوحة تبدأ 2025-01-03، لكن أول نقطة فيها NaN بالبناء
            # (diff() بلا سابقة) — reindex(ffill) لا يستبدل قيمة موجودة
            # فعلاً بالفهرس (حتى لو NaN)، فتبقى 2025-01-03 نفسها غير متاحة
            assert list(out['OI_available']) == [0.0, 0.0, 0.0, 1.0, 1.0]
            assert out.loc[idx[0], 'OI_chg_1'] == 0.0    # محايد صفر لا NaN حين لا تغطية
            assert not out['FUND_rate'].isna().any() and not out['OI_chg_1'].isna().any()
        finally:
            globals()['mount_drive'] = old_mount

    def t_add_funding_oi_features_neutral_when_no_archive():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        old_mount = globals()['mount_drive']
        globals()['mount_drive'] = lambda mount_point=None, config=None: tmp
        try:
            cfg = _cfg(funding_rate={'enabled': True, 'as_feature': True},
                      open_interest={'enabled': True, 'as_feature': True})
            idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
            df = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=idx)
            out = add_funding_oi_features({'1D': df}, 'GHOSTUSDT', config=cfg)['1D']
            assert (out['FUND_rate'] == 0.0).all() and (out['FUND_available'] == 0.0).all()
            assert (out['OI_chg_1'] == 0.0).all() and (out['OI_available'] == 0.0).all()
        finally:
            globals()['mount_drive'] = old_mount

    def t_funding_oi_disabled_returns_dfs_unchanged():
        cfg = _cfg(funding_rate={'enabled': False}, open_interest={'enabled': False})
        idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=idx)
        out = add_funding_oi_features({'1D': df}, 'AAAUSDT', config=cfg)
        assert out['1D'] is df               # بلا نسخ إضافي حين لا شيء مُفعَّل

    def t_funding_oi_reaches_dataset_via_build_dataset():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        old_mount = globals()['mount_drive']
        globals()['mount_drive'] = lambda mount_point=None, config=None: tmp
        try:
            cfg = _cfg(funding_rate={'enabled': True, 'drive_dir': 'funding_rate',
                                     'as_feature': True, 'zscore_window': 5},
                      open_interest={'enabled': False}, market_context={'enabled': False})
            # يغطّي كامل مدى الأصل (~200 يوماً) بقيم متغيّرة باستمرار — أرشيف
            # قصير (يغطّي جزءاً من المدى فقط) يُصبح ثابتاً بعد ffill داخل أغلب
            # النوافذ، فيُصفّره process_windows كعمود بلا تباين (بنفس منطق
            # t_process_windows_degenerate_column_zeroed)، فيفشل فحص != 0.0 هنا
            # لا لخلل في الدمج بل لأن القيمة صفر عن حقّ في تلك النوافذ تحديداً.
            n_fr = 700
            fr = pd.DataFrame({
                'timestamp': pd.date_range('2025-01-01', periods=n_fr, freq='8h', tz='UTC'),
                'funding_rate': 0.0005 * np.sin(np.linspace(0, 40 * np.pi, n_fr))})
            fr.to_csv(_funding_oi_drive_path(tmp, 'funding_rate', 'AAAUSDT', cfg), index=False)

            names = ['AAAUSDT']
            data = {n: _quiet(resample_timeframes, _ohlcv(200, seed=0), ['1D'], config=cfg)
                   for n in names}
            ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                       config=cfg, max_workers=1)
            assert 'FUND_rate' in ds['feature_order']
            i = ds['feature_order'].index('FUND_rate')
            assert np.any(ds['X_1D'][:, :, i] != 0.0)   # قيم حقيقية وصلت للمصفوفة النهائية
        finally:
            globals()['mount_drive'] = old_mount

    def t_intraday_efficiency_columns_names():
        cfg = _cfg(intraday_efficiency={'enabled': True, 'as_feature': True})
        assert intraday_efficiency_columns(cfg) == ['EFF_RATIO_24H', 'EFF_RATIO_available']

        cfg_off = _cfg(intraday_efficiency={'enabled': False})
        assert intraday_efficiency_columns(cfg_off) == []

        # enabled بلا as_feature لا يُنتج أي عمود مدخل (نفس منطق funding_rate)
        cfg_no_feat = _cfg(intraday_efficiency={'enabled': True, 'as_feature': False})
        assert intraday_efficiency_columns(cfg_no_feat) == []

    def t_add_intraday_efficiency_features_aligns_and_flags_availability():
        cfg = _cfg(intraday_efficiency={
            'enabled': True, 'as_feature': True,
            'data': {'AAAUSDT': pd.DataFrame({
                'date': pd.date_range('2025-01-03', periods=5, freq='D', tz='UTC'),
                'efficiency_ratio_24h': [0.1, 0.2, 0.3, 0.4, 0.5]})}})

        idx = pd.date_range('2025-01-01', periods=7, freq='D', tz='UTC')
        df = pd.DataFrame({'close': np.linspace(1, 2, 7)}, index=idx)
        out = add_intraday_efficiency_features({'1D': df}, 'AAAUSDT', config=cfg)['1D']

        # الأرشيف يبدأ 2025-01-03 — أول يومين بلا تغطية (محايد صفر + علم غياب)
        assert list(out['EFF_RATIO_available']) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert list(out['EFF_RATIO_24H']) == [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        assert not out['EFF_RATIO_24H'].isna().any()

    def t_add_intraday_efficiency_features_neutral_when_no_archive():
        cfg = _cfg(intraday_efficiency={'enabled': True, 'as_feature': True, 'data': {}})
        idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=idx)
        out = add_intraday_efficiency_features({'1D': df}, 'GHOSTUSDT', config=cfg)['1D']
        assert (out['EFF_RATIO_24H'] == 0.0).all() and (out['EFF_RATIO_available'] == 0.0).all()

    def t_intraday_efficiency_disabled_returns_dfs_unchanged():
        cfg = _cfg(intraday_efficiency={'enabled': False})
        idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=idx)
        out = add_intraday_efficiency_features({'1D': df}, 'AAAUSDT', config=cfg)
        assert out['1D'] is df               # بلا نسخ إضافي حين لا شيء مُفعَّل

    def t_intraday_efficiency_reaches_dataset_via_build_dataset():
        names = ['AAAUSDT']
        rng = np.random.default_rng(3)
        eff_dates = pd.date_range('2025-01-01', periods=200, freq='D', tz='UTC')
        eff_df = pd.DataFrame({'date': eff_dates,
                               'efficiency_ratio_24h': rng.uniform(0, 1, 200)})
        cfg = _cfg(intraday_efficiency={'enabled': True, 'as_feature': True,
                                        'data': {'AAAUSDT': eff_df}},
                  market_context={'enabled': False})
        data = {n: _quiet(resample_timeframes, _ohlcv(200, seed=0), ['1D'], config=cfg)
               for n in names}
        ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                   config=cfg, max_workers=1)
        assert 'EFF_RATIO_24H' in ds['feature_order']
        i = ds['feature_order'].index('EFF_RATIO_24H')
        assert np.any(ds['X_1D'][:, :, i] != 0.0)   # قيم حقيقية وصلت للمصفوفة النهائية

    def t_intraday_vwap_columns_names():
        cfg = _cfg(intraday_vwap={'enabled': True, 'as_feature': True})
        assert intraday_vwap_columns(cfg) == ['VWAP_DEVIATION', 'VWAP_DEVIATION_available']

        cfg_off = _cfg(intraday_vwap={'enabled': False})
        assert intraday_vwap_columns(cfg_off) == []

        cfg_no_feat = _cfg(intraday_vwap={'enabled': True, 'as_feature': False})
        assert intraday_vwap_columns(cfg_no_feat) == []

    def t_add_intraday_vwap_features_aligns_and_flags_availability():
        cfg = _cfg(intraday_vwap={
            'enabled': True, 'as_feature': True,
            'data': {'AAAUSDT': pd.DataFrame({
                'date': pd.date_range('2025-01-03', periods=5, freq='D', tz='UTC'),
                'vwap_deviation': [0.01, -0.02, 0.03, -0.04, 0.05]})}})

        idx = pd.date_range('2025-01-01', periods=7, freq='D', tz='UTC')
        df = pd.DataFrame({'close': np.linspace(1, 2, 7)}, index=idx)
        out = add_intraday_vwap_features({'1D': df}, 'AAAUSDT', config=cfg)['1D']

        assert list(out['VWAP_DEVIATION_available']) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert list(out['VWAP_DEVIATION']) == [0.0, 0.0, 0.01, -0.02, 0.03, -0.04, 0.05]
        assert not out['VWAP_DEVIATION'].isna().any()

    def t_add_intraday_vwap_features_neutral_when_no_archive():
        cfg = _cfg(intraday_vwap={'enabled': True, 'as_feature': True, 'data': {}})
        idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=idx)
        out = add_intraday_vwap_features({'1D': df}, 'GHOSTUSDT', config=cfg)['1D']
        assert (out['VWAP_DEVIATION'] == 0.0).all() and (out['VWAP_DEVIATION_available'] == 0.0).all()

    def t_intraday_vwap_disabled_returns_dfs_unchanged():
        cfg = _cfg(intraday_vwap={'enabled': False})
        idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=idx)
        out = add_intraday_vwap_features({'1D': df}, 'AAAUSDT', config=cfg)
        assert out['1D'] is df

    def t_intraday_vwap_reaches_dataset_via_build_dataset():
        names = ['AAAUSDT']
        rng = np.random.default_rng(4)
        vwap_dates = pd.date_range('2025-01-01', periods=200, freq='D', tz='UTC')
        vwap_df = pd.DataFrame({'date': vwap_dates,
                                'vwap_deviation': rng.uniform(-0.05, 0.05, 200)})
        cfg = _cfg(intraday_vwap={'enabled': True, 'as_feature': True,
                                  'data': {'AAAUSDT': vwap_df}},
                  market_context={'enabled': False})
        data = {n: _quiet(resample_timeframes, _ohlcv(200, seed=0), ['1D'], config=cfg)
               for n in names}
        ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                   config=cfg, max_workers=1)
        assert 'VWAP_DEVIATION' in ds['feature_order']
        i = ds['feature_order'].index('VWAP_DEVIATION')
        assert np.any(ds['X_1D'][:, :, i] != 0.0)

    def t_intraday_volume_concentration_columns_names():
        cfg = _cfg(intraday_volume_concentration={'enabled': True, 'as_feature': True})
        assert intraday_volume_concentration_columns(cfg) == ['VOL_CONC_HHI', 'VOL_CONC_HHI_available']

        cfg_off = _cfg(intraday_volume_concentration={'enabled': False})
        assert intraday_volume_concentration_columns(cfg_off) == []

        cfg_no_feat = _cfg(intraday_volume_concentration={'enabled': True, 'as_feature': False})
        assert intraday_volume_concentration_columns(cfg_no_feat) == []

    def t_add_intraday_volume_concentration_features_aligns_and_flags_availability():
        cfg = _cfg(intraday_volume_concentration={
            'enabled': True, 'as_feature': True,
            'data': {'AAAUSDT': pd.DataFrame({
                'date': pd.date_range('2025-01-03', periods=5, freq='D', tz='UTC'),
                'volume_concentration_hhi': [0.05, 0.06, 0.07, 0.08, 0.09]})}})

        idx = pd.date_range('2025-01-01', periods=7, freq='D', tz='UTC')
        df = pd.DataFrame({'close': np.linspace(1, 2, 7)}, index=idx)
        out = add_intraday_volume_concentration_features({'1D': df}, 'AAAUSDT', config=cfg)['1D']

        assert list(out['VOL_CONC_HHI_available']) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        assert list(out['VOL_CONC_HHI']) == [0.0, 0.0, 0.05, 0.06, 0.07, 0.08, 0.09]
        assert not out['VOL_CONC_HHI'].isna().any()

    def t_add_intraday_volume_concentration_features_neutral_when_no_archive():
        cfg = _cfg(intraday_volume_concentration={'enabled': True, 'as_feature': True, 'data': {}})
        idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=idx)
        out = add_intraday_volume_concentration_features({'1D': df}, 'GHOSTUSDT', config=cfg)['1D']
        assert (out['VOL_CONC_HHI'] == 0.0).all() and (out['VOL_CONC_HHI_available'] == 0.0).all()

    def t_intraday_volume_concentration_disabled_returns_dfs_unchanged():
        cfg = _cfg(intraday_volume_concentration={'enabled': False})
        idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=idx)
        out = add_intraday_volume_concentration_features({'1D': df}, 'AAAUSDT', config=cfg)
        assert out['1D'] is df

    def t_intraday_volume_concentration_reaches_dataset_via_build_dataset():
        names = ['AAAUSDT']
        rng = np.random.default_rng(5)
        vc_dates = pd.date_range('2025-01-01', periods=200, freq='D', tz='UTC')
        vc_df = pd.DataFrame({'date': vc_dates,
                              'volume_concentration_hhi': rng.uniform(0.04, 0.2, 200)})
        cfg = _cfg(intraday_volume_concentration={'enabled': True, 'as_feature': True,
                                                   'data': {'AAAUSDT': vc_df}},
                  market_context={'enabled': False})
        data = {n: _quiet(resample_timeframes, _ohlcv(200, seed=0), ['1D'], config=cfg)
               for n in names}
        ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                   config=cfg, max_workers=1)
        assert 'VOL_CONC_HHI' in ds['feature_order']
        i = ds['feature_order'].index('VOL_CONC_HHI')
        assert np.any(ds['X_1D'][:, :, i] != 0.0)

    def t_checkpoint_fingerprint_changes_with_funding_oi_config():
        cfg1 = _cfg(funding_rate={'enabled': True, 'as_feature': False})
        cfg2 = _cfg(funding_rate={'enabled': True, 'as_feature': True})
        fp1 = _checkpoint_fingerprint(cfg1, ['a'], 0)
        fp2 = _checkpoint_fingerprint(cfg2, ['a'], 0)
        assert fp1 != fp2
        assert fp1 == _checkpoint_fingerprint(cfg1, ['a'], 0)


    # ── تسريع النوافذ: مسار فريم واحد بلا تكرار نسخ (align) ─────────────────
    def t_align_single_tf_fast_path_matches_manual_windows():
        idx = pd.date_range('2025-01-01', periods=10, freq='D', tz='UTC')
        df = pd.DataFrame({'a': np.arange(10.0), 'b': np.arange(10.0) * 10}, index=idx)
        cfg = _cfg(tf_order=['1D'], window_sizes={'1D': 3}, stride=1)
        windows, end_times = align_multi_timeframes_time_based({'1D': df}, config=cfg)
        assert windows['1D'].shape == (8, 3, 2)
        np.testing.assert_array_equal(windows['1D'][0], df.iloc[0:3].values.astype('float32'))
        np.testing.assert_array_equal(windows['1D'][-1], df.iloc[7:10].values.astype('float32'))
        assert list(end_times) == list(df.index[2:])

    def t_align_single_tf_fast_path_respects_stride():
        idx = pd.date_range('2025-01-01', periods=20, freq='D', tz='UTC')
        df = pd.DataFrame({'a': np.arange(20.0)}, index=idx)
        # العدّ من أول شمعة (بلا شبكة): يختبر آلية الشريحة نفسها؛ الشبكة لها اختباراتها أدناه.
        cfg = _cfg(tf_order=['1D'], window_sizes={'1D': 5}, stride=3, align_windows_to_grid=False)
        windows, end_times = align_multi_timeframes_time_based({'1D': df}, config=cfg)
        n_expected = len(range(4, 20, 3))
        assert windows['1D'].shape[0] == n_expected
        np.testing.assert_array_equal(windows['1D'][1][:, 0], df.iloc[3:8]['a'].values)

    def t_align_single_tf_fast_path_empty_when_too_short():
        idx = pd.date_range('2025-01-01', periods=3, freq='D', tz='UTC')
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0]}, index=idx)
        cfg = _cfg(tf_order=['1D'], window_sizes={'1D': 5}, stride=1)
        windows, end_times = align_multi_timeframes_time_based({'1D': df}, config=cfg)
        assert windows['1D'].shape == (0, 5, 1)
        assert end_times == []

    # ── نهايات النوافذ على شبكة زمنية مشتركة بين العملات (stride > 1) ─────────
    #    خلل مُقاس على 1h (stride=32): كل عملة تبدأ نوافذها من أول شمعة لها فتقع على
    #    طور مختلف — وسيط العملات في الطابع الواحد 2 من 83.
    def t_window_ends_aligned_across_coins_with_stride():
        cfg = _cfg(tf_order=['1h'], base_tf='1h', window_sizes={'1h': 16}, stride=16)
        starts = {'AAAUSDT': ('2025-01-01 00:00', 40), 'BBBUSDT': ('2025-01-01 05:00', 36),
                  'CCCUSDT': ('2025-01-02 13:00', 30)}
        data = {n: _quiet(resample_timeframes, _ohlcv(d, seed=i, start=s), ['1h'], config=cfg)
                for i, (n, (s, d)) in enumerate(starts.items())}
        ds = _quiet(build_dataset_from_preloaded, [{'name': n} for n in starts], data,
                    config=cfg, max_workers=1)
        last = ds['last_candles']
        ts = last[:, TS_COL].astype('int64')
        period = 16 * 3600 * 10 ** 9
        assert (ts % period == 0).all(), 'نهاية نافذة خارج الشبكة (ts − 1970) % (stride × 1h)'
        per = {b['name']: ts[b['start']:b['end']] for b in ds['asset_bounds']}
        lo, hi = max(v.min() for v in per.values()), min(v.max() for v in per.values())
        common = [set(v[(v >= lo) & (v <= hi)].tolist()) for v in per.values()]
        assert len(common[0]) >= 10 and all(c == common[0] for c in common), \
            f'أزمنة النهاية تختلف بين العملات في مداها المشترك: {[len(c) for c in common]}'
        i_close = LAST_COLUMNS.index('future_close')
        for b in ds['asset_bounds']:
            v = per[b['name']]
            assert (np.diff(v) == period).all(), 'stride = window يجب أن يُبقي النوافذ متلاصقة بلا تراكب'
            close = data[b['name']]['1h']['close']
            for k in range(b['start'], b['end']):
                pos = close.index.get_loc(pd.Timestamp(int(ts[k]), tz='UTC'))
                assert np.isclose(last[k, i_close], close.iloc[pos + 1]), 'الهدف ليس الشمعة التالية للنهاية'

    def t_window_ends_aligned_multi_tf_path():
        cfg = _cfg(tf_order=['1h', '4h'], base_tf='1h', window_sizes={'1h': 8, '4h': 3}, stride=6)
        ends = []
        for s in ('2025-01-01 00:00', '2025-01-01 07:00', '2025-01-02 03:00'):
            df = _ohlcv(10, seed=1, start=s)
            dfs = {'1h': df, '4h': df.resample('4h').agg(OHLCV_AGG).dropna()}
            _, et = align_multi_timeframes_time_based(dfs, config=cfg)
            ends.append(set(pd.DatetimeIndex(et).as_unit('ns').asi8.tolist()))
        lo, hi = max(min(e) for e in ends), min(max(e) for e in ends)
        common = [{t for t in e if lo <= t <= hi} for e in ends]
        assert len(common[0]) >= 10 and all(c == common[0] for c in common)
        assert all(t % (6 * 3600 * 10 ** 9) == 0 for e in ends for t in e)

    def t_window_grid_noop_for_stride_1_and_when_disabled():
        idx = pd.date_range('2024-03-05', periods=90, freq='D', tz='UTC')
        df = pd.DataFrame({'a': np.arange(90.0), 'b': np.sin(np.arange(90.0))}, index=idx)
        on = _cfg(window_sizes={'1D': 7}, stride=1, align_windows_to_grid=True)
        off = _cfg(window_sizes={'1D': 7}, stride=1, align_windows_to_grid=False)
        w_on, e_on = align_multi_timeframes_time_based({'1D': df}, config=on)
        w_off, e_off = align_multi_timeframes_time_based({'1D': df}, config=off)
        np.testing.assert_array_equal(w_on['1D'], w_off['1D'])
        assert list(e_on) == list(e_off) == list(idx[6:])
        # stride > 1 مع التعطيل = العدّ القديم من أول شمعة بالضبط
        off3 = _cfg(window_sizes={'1D': 7}, stride=3, align_windows_to_grid=False)
        _, e3 = align_multi_timeframes_time_based({'1D': df}, config=off3)
        assert list(e3) == list(idx[6::3])

    def t_window_grid_gap_keeps_min_spacing():
        idx = pd.date_range('2025-01-01', periods=400, freq='h', tz='UTC')
        idx = idx.delete(np.r_[100:103, 250:251])          # فجوتان: 3 شموع ثم شمعة
        ends = window_end_indices(idx, 12, 8, '1h', _cfg())
        assert len(ends) > 30
        assert (np.diff(ends) >= 8).all(), 'فجوة قرّبت نهايتين إلى أقل من stride شمعة'
        assert all(idx[e].value % (8 * 3600 * 10 ** 9) == 0 for e in ends)

    def t_hourly_w32_s8_grid_and_purge():
        # إعداد اللوحة على 1h (نافذة 32، stride 8 < النافذة، أفق 1): كل العملات على شبكة 8h نفسها، وفجوة العزل
        # تُبعد أول نافذة val/test عن آخر هدف قبلها — زمنياً في global_time، وبعدد العيّنات في per_asset.
        cfg = _cfg(tf_order=['1h'], base_tf='1h', window_sizes={'1h': 32}, stride=8, forecast_horizon=1,
                   min_split_samples=8, train_pct=0.7, val_pct=0.15)
        names = [f'C{i}USDT' for i in range(5)]
        starts = ['2025-01-01', '2025-01-01 03:00', '2025-01-01 05:00', '2025-01-01 11:00', '2025-01-01 17:00']

        def _loader(file_id, name):
            return _ohlcv(30, seed=20 + int(name[1]), start=starts[int(name[1])])

        def resample_fn(df, tf_order):
            return _quiet(resample_timeframes, df, tf_order, config=cfg)

        ds = _quiet(build_dataset_from_loader, [{'name': n} for n in names], _loader,
                    resample_fn, config=cfg, max_workers=2)
        H = 3600 * 10 ** 9
        ts = ds['last_candles'][:, TS_COL].astype('int64')
        assert (ts % (8 * H) == 0).all(), 'نهايات خارج شبكة 8h'
        assert pd.Series(ts).value_counts().median() == len(names), 'العملات لا تتشارك الطوابع'
        W, h = 32, 1
        train, val, test = _quiet(split_data, ds, config=cfg)
        t_tr, t_va, t_te = (np.asarray(s['last_candles'])[:, TS_COL].astype('int64') for s in (train, val, test))
        # أول شمعة في أول نافذة val/test بعد آخر شمعة هدف في القسم السابق (لا مدخل يرى هدفاً سابقاً ولا العكس)
        assert t_va.min() - (W - 1) * H > t_tr.max() + h * H, (t_va.min() - t_tr.max()) / H
        assert t_te.min() - (W - 1) * H > t_va.max() + h * H, (t_te.min() - t_va.max()) / H
        # per_asset: الفجوة بعدد العيّنات × stride تغطّي النافذة والأفق
        assert embargo_size(ds, cfg) * cfg['stride'] >= W + h, embargo_size(ds, cfg)

    def t_label_target_candle_is_ts_plus_horizon_across_gap():
        # فجوة بعد نهاية نافذة على الشبكة: هدف العيّنة كان يُؤخذ من «الصف التالي» (شمعة أبعد بساعات). المطلوب: كل عيّنة
        # باقية هدفها الشمعة التي تبدأ عند ts + 1h بالضبط، وعيّنة الفجوة تُحذف.
        cfg = _cfg(tf_order=['1h'], base_tf='1h', window_sizes={'1h': 32}, stride=8, forecast_horizon=1,
                   min_split_samples=8)
        raw = _ohlcv(30, seed=31)
        gap_after = pd.Timestamp('2025-01-20 08:00', tz='UTC')
        raw = raw.drop(pd.date_range(gap_after + pd.Timedelta('1h'), periods=5, freq='h', tz='UTC'))

        def resample_fn(df, tf_order):
            return _quiet(resample_timeframes, df, tf_order, config=cfg)

        ds = _quiet(build_dataset_from_loader, [{'name': 'GAPUSDT'}], lambda f, n: raw.copy(),
                    resample_fn, config=cfg, max_workers=1)
        lc = ds['last_candles']
        ts = pd.to_datetime(lc[:, TS_COL].astype('int64'), utc=True)
        assert len(ts) > 20
        assert gap_after not in set(ts), 'عيّنة الفجوة لم تُحذف'
        want = raw['close'].reindex(ts + pd.Timedelta('1h')).to_numpy()
        fc = lc[:, LAST_COLUMNS.index('future_close')]
        assert np.isfinite(want).all() and np.allclose(fc, want), 'هدف من شمعة غير ts + 1h'
        # ونافذة الإدخال 32 شمعة متتالية بالضبط لكل عيّنة باقية — لا تعبر الفجوة (r2_02)
        idx = resample_fn(raw.copy(), ['1h'])['1h'].index
        starts = idx[[idx.get_loc(t) - 31 for t in ts]]
        assert ((ts - starts) == pd.Timedelta('31h')).all(), 'نافذة إدخال تعبر فجوة'

    def t_checkpoint_fingerprint_tracks_grid_and_cross_asset():
        base = _cfg(stride=8)
        fp = _checkpoint_fingerprint(base, ['a'], 0)
        assert fp != _checkpoint_fingerprint(_cfg(stride=8, align_windows_to_grid=False), ['a'], 0)
        d1 = _cfg(stride=1)                                    # stride=1: الشبكة لا تُغيّر شيئاً
        assert _checkpoint_fingerprint(d1, ['a'], 0) == \
            _checkpoint_fingerprint(_cfg(stride=1, align_windows_to_grid=False), ['a'], 0)
        cross = _cfg(market_breadth={'enabled': True, 'horizons': [24]})
        assert _checkpoint_fingerprint(cross, ['a'], 0, universe=['X', 'Y']) != \
            _checkpoint_fingerprint(cross, ['a'], 0, universe=['X', 'Y', 'Z'])

    # ── الميزات العابرة للأصول في مسار التحميل (build_dataset_from_loader) ────
    #    خلل مُقاس على 1h: MKT_BREADTH_24 وMOM_ORTH_NATR صفر في كل الصفوف، لأن المسار
    #    كان يمرّر None فتأخذ القيمة المحايدة الثابتة.
    def t_cross_asset_features_nonconstant_via_loader():
        cfg = _cfg(tf_order=['1h'], base_tf='1h', window_sizes={'1h': 32}, stride=32,
                   market_breadth={'enabled': True, 'horizons': [24]},
                   momentum_orth_natr={'enabled': True, 'horizons': [1, 3, 6, 12, 24],
                                       'natr_length': 14, 'min_assets': 5})
        names = [f'C{i}USDT' for i in range(6)]
        starts = ['2025-01-01', '2025-01-01 03:00', '2025-01-02', '2025-01-01 11:00',
                  '2025-01-03', '2025-01-01']

        def _loader(file_id, name):
            i = int(name[1])
            return _ohlcv(40, seed=10 + i, start=starts[i])

        def resample_fn(df, tf_order):
            return _quiet(resample_timeframes, df, tf_order, config=cfg)

        ds = _quiet(build_dataset_from_loader, [{'name': n} for n in names], _loader,
                    resample_fn, config=cfg, max_workers=2)
        fo = ds['feature_order']
        for col in ('MKT_BREADTH_24', 'MOM_ORTH_NATR'):
            assert col in fo, f'{col} غائبة عن feature_order'
            v = ds['X_1h'][:, :, fo.index(col)]
            assert np.isfinite(v).all() and float(np.std(v)) > 1e-3, \
                f'{col} ثابتة (std={float(np.std(v)):.2e}) — ميّتة في مسار التحميل'
        # نفس القيم التي يحسبها المسار المُحمَّل مسبقاً في النوافذ التي يغطّيها تاريخ
        # كافٍ في المسارين (المُحمَّل مسبقاً يرى الإطارات بعد إسقاط شموع الإحماء).
        data = {n: resample_fn(_loader(None, n), ['1h']) for n in names}
        pre = _quiet(build_dataset_from_preloaded, [{'name': n} for n in names], data,
                     config=cfg, max_workers=1)
        assert [b['name'] for b in pre['asset_bounds']] == [b['name'] for b in ds['asset_bounds']]
        np.testing.assert_array_equal(pre['last_candles'][:, TS_COL], ds['last_candles'][:, TS_COL])
        late = ds['last_candles'][:, TS_COL] >= pd.Timestamp('2025-01-08', tz='UTC').value
        assert late.sum() > 20
        for col in ('MKT_BREADTH_24', 'MOM_ORTH_NATR'):
            j = fo.index(col)
            np.testing.assert_allclose(ds['X_1h'][late, :, j], pre['X_1h'][late, :, j], atol=1e-5)

    def t_market_breadth_ignores_unlisted_coins():
        idx = pd.date_range('2024-01-01', periods=60, freq='D', tz='UTC')
        rng = np.random.default_rng(5)
        early = pd.DataFrame({'close': 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))}, index=idx)
        late = pd.DataFrame({'close': 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 30)))}, index=idx[30:])
        cfg = _cfg(market_breadth={'enabled': True, 'horizons': [1, 3]})
        br = build_market_breadth(data={'E': {'1D': early}, 'L': {'1D': late}}, config=cfg)['1D']
        for h in (1, 3):
            r = early['close'].pct_change()
            r = r.rolling(h).sum() if h > 1 else r
            exp = (r > 0).astype(float).where(r.notna())
            pre = idx[h:30]                               # قبل إدراج L: حصّة E وحدها (0 أو 1)
            np.testing.assert_array_equal(br.loc[pre, f'MKT_BREADTH_{h}'].to_numpy(), exp.loc[pre].to_numpy())
        out = add_market_breadth({'1D': early}, {'1D': br}, cfg)['1D']
        assert out['MKT_BREADTH_1'].iloc[0] == 0.5          # لا عملة ببيانات ⇒ القيمة المحايدة

    def t_align_single_tf_peak_memory_bounded():
        import tracemalloc
        n = 2000
        idx = pd.date_range('2025-01-01', periods=n, freq='D', tz='UTC')
        rng = np.random.default_rng(0)
        df = pd.DataFrame({f'f{i}': rng.normal(0, 1, n) for i in range(20)}, index=idx)
        cfg = _cfg(tf_order=['1D'], window_sizes={'1D': 32}, stride=1)
        tracemalloc.start()
        windows, _ = align_multi_timeframes_time_based({'1D': df}, config=cfg)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < windows['1D'].nbytes * 2.5, (peak, windows['1D'].nbytes)   # قِيس فعلياً ~1.13×

    # ── تطبيع النوافذ: float32 من البداية للنهاية بلا مصفوفة out موازية ─────
    def t_process_windows_price_level_matches_manual_formula():
        cols = ['close']
        windows = np.zeros((2, 4, 1), dtype='float32')
        windows[0, :, 0] = [10, 11, 12, 13]
        windows[1, :, 0] = [20, 19, 18, 17]
        centers = np.array([12.0, 18.0], dtype='float32')
        scales = np.array([2.0, 2.0], dtype='float32')
        out = process_windows(windows.copy(), cols, centers, scales, 'robust', config=_cfg())
        expected = (windows - centers.reshape(2, 1, 1)) / scales.reshape(2, 1, 1)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def t_process_windows_degenerate_column_zeroed():
        cols = ['RSI_14']
        windows = np.full((2, 5, 1), 50.0, dtype='float32')
        out = process_windows(windows.copy(), cols, np.zeros(2, 'float32'),
                              np.ones(2, 'float32'), config=_cfg())
        assert np.all(out == 0.0)

    def t_process_windows_keeps_constant_flags_and_funding():
        # علم توفّر ثابت (1 أو 0) وتمويل ثابت يجب أن يصلا للنموذج — لا أن يُصفَّرا كعمود بلا تباين —
        # والنسختان (المتّجهة والمفردة) متطابقتان عليهما.
        cols = ['FUND_available', 'OI_available', 'FUND_rate', 'FUND_rate_z', 'RSI_14']
        w = np.zeros((2, 6, 5), dtype='float32')
        w[0, :, 0], w[1, :, 0] = 1.0, 0.0              # متوفّر / غير متوفّر، ثابتان
        w[:, :, 1] = 1.0
        w[0, :, 2], w[1, :, 2] = 0.0005, 0.0001        # تمويل ثابت بمستويين مختلفين
        w[:, :, 3] = 1.5                               # z ثابت
        w[:, :, 4] = 70.0                              # RSI ثابت — يبقى يُصفَّر كالسابق
        out = process_windows(w.copy(), cols, np.zeros(2, 'float32'), np.ones(2, 'float32'), config=_cfg())
        assert np.allclose(out[0, :, 0], 1.0) and np.allclose(out[1, :, 0], -1.0), out[:, 0, 0]
        assert np.allclose(out[:, :, 1], 1.0)
        assert np.allclose(out[0, :, 2], 0.5) and np.allclose(out[1, :, 2], 0.1), out[:, 0, 2]
        assert np.all(out[:, :, 3] == 0.0) and np.all(out[:, :, 4] == 0.0)
        for i in range(2):
            single = process_window(w[i].copy(), cols, 0.0, 1.0, config=_cfg())
            assert np.allclose(single, out[i], atol=1e-6), (single, out[i])
        assert classify_feature('FUND_rate_z') == ZSCORE and classify_feature('OI_chg_1') == SIGN_ROBUST

    def t_process_windows_clip_applied():
        cols = ['RET_1']
        windows = np.zeros((1, 5, 1), dtype='float32')
        windows[0, :, 0] = [0, 0, 0, 0, 1000.0]
        out = process_windows(windows.copy(), cols, np.zeros(1, 'float32'),
                              np.ones(1, 'float32'), config=_cfg(clip_abs=5.0))
        assert np.all(np.abs(out) <= 5.0 + 1e-6)

    def t_process_windows_peak_memory_bounded():
        # مزيج واقعي من الأنواع الدلالية (كما في مجموعة ميزات حقيقية) —
        # الذروة أفضل من حالة نوع واحد يغطّي كل الأعمدة (يُنسخ كل X دفعة واحدة).
        import tracemalloc
        N, T, F = 500, 32, 30
        rng = np.random.default_rng(0)
        windows = rng.normal(0, 1, (N, T, F)).astype('float32')
        kinds = ['close', 'RSI_14', 'RET_1', 'MACDh_12_26_9', 'BBP_20_2.0', 'OBV']
        cols = [kinds[i % len(kinds)] + f'_{i}' if i % len(kinds) not in (0,) else 'close'
               for i in range(F)]
        tracemalloc.start()
        out = process_windows(windows.copy(), cols, np.full(N, 100.0, 'float32'),
                              np.full(N, 5.0, 'float32'), config=_cfg())
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < out.nbytes * 6, (peak, out.nbytes)          # هامش أمان (قِيس فعلياً ~2.6-4.6×)

    # ── عدد الخيوط الافتراضي: سقف رام احترازي ────────────────────────────────
    def t_default_workers_capped_by_ram():
        old_fn = globals()['_system_ram_mb']
        globals()['_system_ram_mb'] = lambda: 4096.0
        try:
            w = default_workers(1000)
        finally:
            globals()['_system_ram_mb'] = old_fn
        assert w <= 4, w

    def t_default_workers_unaffected_when_ram_unknown():
        old_fn = globals()['_system_ram_mb']
        globals()['_system_ram_mb'] = lambda: None
        try:
            w = default_workers(1000)
        finally:
            globals()['_system_ram_mb'] = old_fn
        assert w >= 4

    # ── نقاط استئناف لكل أصل (checkpoint/resume) ────────────────────────────
    def t_checkpoint_fingerprint_changes_with_relevant_config():
        cfg1 = _cfg()
        cfg2 = _cfg(window_sizes={'1D': 64})
        fp1 = _checkpoint_fingerprint(cfg1, ['a', 'b'], 0)
        fp2 = _checkpoint_fingerprint(cfg2, ['a', 'b'], 0)
        assert fp1 != fp2
        assert fp1 == _checkpoint_fingerprint(cfg1, ['a', 'b'], 0)

    def t_save_load_checkpoint_roundtrip():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        X_tf = {'1D': rng.normal(0, 1, (5, 32, 3)).astype('float32')}
        y_t = {'close_reg': rng.normal(0, 1, 5).astype('float32')}
        bases = rng.normal(0, 1, (5, 2)).astype('float32')
        last = rng.normal(0, 1, (5, 7)).astype('float64')
        _save_asset_checkpoint(tmp, 'AAAUSDT', X_tf, y_t, bases, last, 'fp123')
        loaded = _load_asset_checkpoint(tmp, 'AAAUSDT', 'fp123')
        assert loaded is not None
        lX, ly, lb, ll = loaded
        np.testing.assert_array_equal(lX['1D'], X_tf['1D'])
        np.testing.assert_array_equal(ly['close_reg'], y_t['close_reg'])
        np.testing.assert_array_equal(lb, bases)
        np.testing.assert_array_equal(ll, last)

    def t_load_checkpoint_rejects_mismatched_fingerprint():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        X_tf = {'1D': np.zeros((2, 3, 1), 'float32')}
        y_t = {'close_reg': np.zeros(2, 'float32')}
        bases = np.zeros((2, 2), 'float32')
        last = np.zeros((2, 7), 'float64')
        _save_asset_checkpoint(tmp, 'X', X_tf, y_t, bases, last, 'fp_old')
        assert _load_asset_checkpoint(tmp, 'X', 'fp_new') is None

    def t_load_checkpoint_missing_returns_none():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        assert _load_asset_checkpoint(tmp, 'GHOST', 'anyfp') is None

    def t_clear_checkpoint_removes_files():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        X_tf = {'1D': np.zeros((1, 2, 1), 'float32')}
        y_t = {'close_reg': np.zeros(1, 'float32')}
        bases = np.zeros((1, 2), 'float32')
        last = np.zeros((1, 7), 'float64')
        _save_asset_checkpoint(tmp, 'A', X_tf, y_t, bases, last, 'fp')
        _save_asset_checkpoint(tmp, 'B', X_tf, y_t, bases, last, 'fp')
        n = clear_checkpoint(tmp, names=['A'])
        assert n == 1
        assert _load_asset_checkpoint(tmp, 'A', 'fp') is None
        assert _load_asset_checkpoint(tmp, 'B', 'fp') is not None
        n2 = clear_checkpoint(tmp)
        assert n2 == 1
        assert _load_asset_checkpoint(tmp, 'B', 'fp') is None

    def t_build_dataset_from_loader_checkpoint_resume_end_to_end():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        cfg = _cfg()
        seeds = {'AAAUSDT': 1, 'BBBUSDT': 2, 'CCCUSDT': 3}
        days = {'n': 200}
        call_count = {'n': 0}

        def _loader(file_id, name):
            return _ohlcv(days['n'], seed=seeds[name])

        def resample_fn(df, tf_order):
            call_count['n'] += 1                       # معالجة فعلية (المؤشرات) — ما يوفّره الاستئناف
            return _quiet(resample_timeframes, df, tf_order, config=cfg)

        names = list(seeds)
        ds1 = _quiet(build_dataset_from_loader, [{'name': n} for n in names], _loader,
                    resample_fn, config=cfg, max_workers=1, checkpoint_dir=tmp)
        n_calls_first = call_count['n']
        assert n_calls_first == 3

        ds2 = _quiet(build_dataset_from_loader, [{'name': n} for n in names], _loader,
                    resample_fn, config=cfg, max_workers=1, checkpoint_dir=tmp)
        assert call_count['n'] == n_calls_first            # لا إعادة معالجة (الملف الخام يُقرأ للبصمة فقط)
        np.testing.assert_array_equal(ds1['base_params'], ds2['base_params'])
        assert [b['name'] for b in ds1['asset_bounds']] == [b['name'] for b in ds2['asset_bounds']]

        # بيانات خام أطول (إعادة جلب إلى المجلد نفسه) ⇒ تُعاد المعالجة وتظهر العيّنات الأحدث (r2_01)
        days['n'] = 260
        ds3 = _quiet(build_dataset_from_loader, [{'name': n} for n in names], _loader,
                    resample_fn, config=cfg, max_workers=1, checkpoint_dir=tmp)
        ref = _quiet(build_dataset_from_loader, [{'name': n} for n in names], _loader,
                     resample_fn, config=cfg, max_workers=1)
        assert call_count['n'] == n_calls_first + 6
        np.testing.assert_array_equal(ds3['last_candles'], ref['last_candles'])
        assert len(ds3['base_params']) > len(ds1['base_params'])

    def t_build_dataset_checkpoint_invalidated_by_config_change():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        call_count = {'n': 0}

        def _loader(file_id, name):
            call_count['n'] += 1
            return _ohlcv(200, seed=1)

        cfg1 = _cfg(window_sizes={'1D': 20})

        def resample_fn1(df, tf_order):
            return _quiet(resample_timeframes, df, tf_order, config=cfg1)

        _quiet(build_dataset_from_loader, [{'name': 'AAAUSDT'}], _loader, resample_fn1,
              config=cfg1, max_workers=1, checkpoint_dir=tmp)
        assert call_count['n'] == 1

        cfg2 = _cfg(window_sizes={'1D': 25})

        def resample_fn2(df, tf_order):
            return _quiet(resample_timeframes, df, tf_order, config=cfg2)

        _quiet(build_dataset_from_loader, [{'name': 'AAAUSDT'}], _loader, resample_fn2,
              config=cfg2, max_workers=1, checkpoint_dir=tmp)
        assert call_count['n'] == 2                        # أُعيدت المعالجة (البصمة اختلفت)

    # ── بناء مدعوم بالقرص (disk_backed): مطابق بايتاً للرام، وحفظ/تحميل المجلد memmap ─────────────────────────────────────
    def t_disk_backed_build_identical_to_ram_and_dir_roundtrip():
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        seeds = {'AAAUSDT': 1, 'BBBUSDT': 2, 'CCCUSDT': 3, 'DDDUSDT': 4}

        def _loader(file_id, name):
            return _ohlcv(200, seed=seeds[name])

        def _build(disk, workers):
            cfg = _cfg(disk_backed=disk, scratch_dir=str(tmp / f'scratch_{disk}_{workers}'))

            def resample_fn(df, tf_order):
                return _quiet(resample_timeframes, df, tf_order, config=cfg)
            return _quiet(build_dataset_from_loader, [{'name': n} for n in seeds], _loader, resample_fn,
                          config=cfg, max_workers=workers)

        global SMALL_ARRAY_BYTES
        old_small, SMALL_ARRAY_BYTES = SMALL_ARRAY_BYTES, 0          # كل المصفوفات memmap
        try:
            ram = _build(False, 1)
            for workers in (1, 3):
                disk = _build(True, workers)
                assert list(ram) == list(disk)
                for k, v in ram.items():
                    if isinstance(v, np.ndarray):
                        assert isinstance(disk[k], np.memmap) and disk[k].dtype == v.dtype, k
                        assert np.array_equal(v, disk[k]), k
                    else:
                        assert repr(v) == repr(disk[k]), k
            d = tmp / 'rt.dataset'
            _quiet(save_dataset_dir, disk, d)
            back = _quiet(load_dataset_dir, d)
            assert list(back) == list(ram)
            for k, v in ram.items():
                if isinstance(v, np.ndarray):
                    assert isinstance(back[k], np.memmap) and np.array_equal(v, back[k]), k
                else:
                    assert repr(v) == repr(back[k]), k
            run_dir = Path(disk['base_params'].filename).parent.parent
            assert not (run_dir / 'parts').exists()
        finally:
            SMALL_ARRAY_BYTES = old_small

    # ── الفريمات الأعلى بوضع 'closed' (1h + 4h): بلا نظر للمستقبل ───────────────────────────────────────────────
    #    القاعدة: نافذة 4h لعيّنة نهايتها t (فتح آخر شمعة 1h) لا تحوي إلا شموعاً مغلقة عند t، كاملة، متصلة، وأحدثها هي
    #    الآخر المغلقة فعلاً. الاختبارات: قاعدة المحاذاة بمُقارنة مع أوراكل، الفجوات، ثم الدليل الحاسم — تغيير كل ما بعد
    #    زمن القطع لا يغيّر أي نافذة (1h ولا 4h) لعيّنة قبله، ومعه اختبار «أسنان»: قاعدة تسرّب عمداً يجب أن يكشفها.
    H_NS = 3600 * 10 ** 9

    def _hours_frame(index) -> pd.DataFrame:
        """عمود واحد = رقم ساعة epoch لفتح الشمعة (float64؛ يبقى دقيقاً في float32 حتى 1.6e7 ساعة) ⇒ نفك زمن كل صف من النافذة."""
        return pd.DataFrame({'h': index.as_unit('ns').asi8 / H_NS}, index=index)

    def _closed_cfg(**over):
        return _cfg(tf_order=['1h', '4h'], base_tf='1h', window_sizes={'1h': 8, '4h': 5}, stride=1,
                    higher_tf_mode='closed', **over)

    def t_closed_align_last_4h_bar_closes_before_t():
        raw = _ohlcv(12, seed=5, start='2025-03-01')
        f1 = raw.resample('1h').agg(OHLCV_AGG).dropna(how='all')
        f4 = raw.resample('4h').agg(OHLCV_AGG).dropna(how='all')
        dfs = {'1h': _hours_frame(f1.index), '4h': _hours_frame(f4.index)}
        w, ends = align_multi_timeframes_time_based(dfs, config=_closed_cfg())
        n = len(ends)
        assert n > 200 and w['1h'].shape == (n, 8, 1) and w['4h'].shape == (n, 5, 1), (n, w['1h'].shape, w['4h'].shape)
        t = w['1h'][:, -1, 0].astype('float64')                 # فتح آخر شمعة 1h (ساعات)
        assert np.array_equal(t, np.asarray([e.value / H_NS for e in ends])), 'أزمنة النهاية لا تطابق آخر خطوة 1h'
        last4 = w['4h'][:, -1, 0].astype('float64')            # فتح آخر شمعة 4h في النافذة
        assert (last4 + 4 <= t).all(), 'شمعة 4h إغلاقها بعد t (فتحها + 4h > t)'
        assert (t - (last4 + 4) < 4).all(), 'الشمعة الأخيرة ليست أحدث شمعة مغلقة'
        assert (np.diff(w['4h'][:, :, 0].astype('float64'), axis=1) == 4).all(), 'نافذة 4h غير متصلة'
        assert (np.diff(w['1h'][:, :, 0].astype('float64'), axis=1) == 1).all(), 'نافذة 1h غير متصلة'
        assert set(np.unique(t % 4).tolist()) == {0.0, 1.0, 2.0, 3.0}, 'يجب أن تغطي العيّنات كل أطوار t داخل شمعة 4h'
        # طور t داخل شمعة 4h قيد التكوّن: الشمعة الجارية (فتحها = t − t%4) لا تظهر أبداً
        forming = t - (t % 4)
        assert not (w['4h'][:, :, 0].astype('float64') >= forming[:, None]).any(), 'شمعة 4h قيد التكوّن دخلت النافذة'
        # على بيانات بلا فجوات: القاعدة تطابق القصّ الموضعي القديم (higher_tf_offset=2) حرفياً
        wl, el = align_multi_timeframes_time_based(dfs, config=_cfg(
            tf_order=['1h', '4h'], base_tf='1h', window_sizes={'1h': 8, '4h': 5}, stride=1, higher_tf_mode='legacy'))
        assert list(el) == list(ends)
        np.testing.assert_array_equal(wl['4h'], w['4h'])
        np.testing.assert_array_equal(wl['1h'], w['1h'])

    def t_closed_align_gaps_match_oracle_and_resample_drops_partial_bars():
        raw = _ohlcv(12, seed=6, start='2025-03-01')
        gap = pd.Timestamp('2025-03-05 06:00', tz='UTC')       # ساعة ناقصة داخل شمعة 4h [04:00, 08:00)
        raw = raw.drop(gap)
        cfg = _closed_cfg()
        dfs_c = _quiet(resample_timeframes, raw, ['1h', '4h'], with_features=False, config=cfg)
        dfs_l = _quiet(resample_timeframes, raw, ['1h', '4h'], with_features=False,
                       config=_cfg(tf_order=['1h', '4h'], base_tf='1h', higher_tf_mode='legacy'))
        # add_features يُسقط صفّ الساعة الناقصة (أسعاره NaN) كما هنا، فتصير فجوة حقيقية في فهرس 1h
        dfs_c['1h'], dfs_l['1h'] = (dfs_c['1h'].dropna(subset=['close']), dfs_l['1h'].dropna(subset=['close']))
        bad = pd.Timestamp('2025-03-05 04:00', tz='UTC')
        assert bad in dfs_l['4h'].index, 'legacy يجب أن يُبقي الشمعة الناقصة كما كان'
        assert bad not in dfs_c['4h'].index, "closed يجب أن يُسقط شمعة 4h الناقصة"
        assert len(dfs_c['4h']) == len(dfs_l['4h']) - 1 and dfs_c['1h'].equals(dfs_l['1h'])
        assert gap not in dfs_c['1h'].index
        dfs = {'1h': _hours_frame(dfs_c['1h'].index), '4h': _hours_frame(dfs_c['4h'].index)}
        w, ends = align_multi_timeframes_time_based(dfs, config=cfg)
        # أوراكل: t (بعد أول 8 شموع 1h) صالح إن كانت شموع 4h الخمس المتصلة المنتهية بأحدث شمعة مغلقة موجودة كلها.
        # اتصال نافذة 1h نفسها يفحصه prepare_single_asset (فترة النافذة) لا المحاذاة.
        have4 = set(dfs['4h'].index)
        want = []
        for t in dfs['1h'].index[7:]:
            last = t.floor('4h') - pd.Timedelta('4h')
            if all(last - pd.Timedelta('4h') * k in have4 for k in range(5)):
                want.append(t)
        assert list(ends) == want, (len(ends), len(want))
        # الفجوة أزالت نوافذ فعلاً: t ∈ [08:00، 12:00) (الأحدث المغلقة مفقودة) ثم كل t تمتدّ نافذتها فوق الشمعة المفقودة
        assert not any(pd.Timestamp('2025-03-05 08:00', tz='UTC') <= e < pd.Timestamp('2025-03-05 12:00', tz='UTC')
                       for e in ends)
        assert not any(bad in [pd.Timestamp(int(h * H_NS), tz='UTC') for h in row[:, 0]] for row in w['4h'])

    def _closed_dataset(perturb_at=None, offset=None, mode='closed', dtype=None, n_coins=6, days=45):
        """مجموعة بيانات 1h+4h كاملة عبر build_dataset_from_loader (سياق سوقي + اتساع + زخم متعامد فعّالة) بعملات ببدايات
        مختلفة؛ perturb_at: كل شمعة 1h فتحها ≥ perturb_at لكل العملات تُستبدل بقيم عشوائية (حجم وأسعار)."""
        cfg = _cfg(tf_order=['1h', '4h'], base_tf='1h', window_sizes={'1h': 32, '4h': 32}, stride=8,
                   higher_tf_mode=mode, x_storage_dtype=dtype, align_windows_to_grid=True, reg_target_scale=100.0,
                   market_context={'enabled': True, 'reference_symbol': 'BTCUSDT', 'return_periods': [1],
                                   'corr_windows': [20], 'beta_windows': [20]},
                   market_breadth={'enabled': True, 'horizons': [6]},
                   momentum_orth_natr={'enabled': True, 'horizons': [1, 3, 6], 'natr_length': 14, 'min_assets': 5})
        if offset is not None:
            cfg['higher_tf_offset'] = offset
        names = ['BTCUSDT'] + [f'C{i}USDT' for i in range(1, n_coins)]
        starts = ['2025-01-01', '2025-01-01 03:00', '2025-01-01 05:00', '2025-01-01 11:00', '2025-01-02', '2025-01-01']

        def _loader(file_id, name):
            i = names.index(name)
            df = _ohlcv(days, seed=60 + i, start=starts[i % len(starts)])
            if perturb_at is not None:
                m = df.index >= perturb_at
                rng, k = np.random.default_rng(900 + i), int(m.sum())
                close = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, k)))
                op = close * (1 + rng.normal(0, 0.01, k))
                df.loc[m, 'close'], df.loc[m, 'open'] = close, op
                df.loc[m, 'high'], df.loc[m, 'low'] = np.maximum(op, close) * 1.02, np.minimum(op, close) * 0.98
                df.loc[m, 'volume'] = rng.random(k) * 9000 + 1
            return df

        def resample_fn(df, tf_order):
            return _quiet(resample_timeframes, df, tf_order, config=cfg)

        return _quiet(build_dataset_from_loader, [{'name': n} for n in names], _loader, resample_fn,
                      config=cfg, max_workers=1)

    def _samples_changed_before(a, b, cut):
        """(عدد العيّنات المقارَنة قبل cut، عدد ما تغيّرت نافذة 1h، عدد ما تغيّرت نافذة 4h) — العيّنة تُعرَّف بـ(عملة، t)."""
        def keyed(ds):
            ts = ds['last_candles'][:, TS_COL].astype('int64')
            return {(bd['name'], int(ts[i])): i for bd in ds['asset_bounds'] for i in range(bd['start'], bd['end'])}
        ka, kb = keyed(a), keyed(b)
        n = c1 = c4 = 0
        for k, i in ka.items():
            if k[1] >= cut.value:
                continue
            n += 1
            j = kb.get(k)
            c1 += j is None or not np.array_equal(a['X_1h'][i], b['X_1h'][j])
            c4 += j is None or not np.array_equal(a['X_4h'][i], b['X_4h'][j])
        return n, c1, c4

    def t_closed_4h_no_lookahead_end_to_end():
        # القطع في منتصف شمعة 4h (10:00) وعلى شبكة 8h: عيّنة t=08:00 (آخر شمعة 1h فتحها 08:00 < 10:00) تقع شمعة 4h
        # الجارية [08:00، 12:00) فوق القطع — فأي تسرّب من الشمعة الجارية يظهر هنا.
        cut = pd.Timestamp('2025-02-05 10:00', tz='UTC')
        a, b = _closed_dataset(), _closed_dataset(perturb_at=cut)
        n, c1, c4 = _samples_changed_before(a, b, cut)
        assert n > 300, n
        assert (c1, c4) == (0, 0), f'نوافذ تغيّرت بتغيير ما بعد القطع: 1h={c1}، 4h={c4} من {n}'
        # وأن ما بعد القطع يتغيّر فعلاً (الاختبار ليس فارغاً)
        assert not np.array_equal(a['X_1h'][-50:], b['X_1h'][-50:])
        # «أسنان»: قاعدة legacy بإزاحة 1 تأخذ الشمعة الجارية — يجب أن يكشفها الاختبار نفسه
        la, lb = _closed_dataset(mode='legacy', offset=1), _closed_dataset(perturb_at=cut, mode='legacy', offset=1)
        _, l1, l4 = _samples_changed_before(la, lb, cut)
        assert l1 == 0 and l4 > 0, f'الاختبار لم يكشف التسرّب المتعمَّد (1h={l1}، 4h={l4})'

    def t_closed_dataset_shapes_alignment_and_float16():
        ds = _closed_dataset(dtype='float16')
        n = len(ds['base_params'])
        f = len(ds['feature_order'])
        assert ds['X_1h'].shape == (n, 32, f) and ds['X_4h'].shape == (n, 32, f), (ds['X_1h'].shape, ds['X_4h'].shape)
        assert ds['X_1h'].dtype == ds['X_4h'].dtype == np.float16 and ds['x_storage_dtype'] == 'float16'
        assert ds['timeframes'] == ['1h', '4h'] and ds['base_tf'] == '1h' and ds['higher_tf_mode'] == 'closed'
        assert ds['window_sizes']['1h'] == ds['window_sizes']['4h'] == 32
        assert np.isfinite(ds['X_1h'].astype('float32')).all() and np.isfinite(ds['X_4h'].astype('float32')).all()
        ts = ds['last_candles'][:, TS_COL].astype('int64')
        assert (ts % (8 * H_NS) == 0).all(), 'نهايات خارج شبكة 8h'
        per = {bd['name']: ts[bd['start']:bd['end']] for bd in ds['asset_bounds']}
        assert len(per) == 6
        lo, hi = max(v.min() for v in per.values()), min(v.max() for v in per.values())
        common = [set(v[(v >= lo) & (v <= hi)].tolist()) for v in per.values()]
        assert len(common[0]) >= 20 and all(c == common[0] for c in common), 'أزمنة النهاية تختلف بين العملات'
        # الهدف: شمعة 1h عند t + 1h (الفريم الأساسي وحده)، لا من 4h
        # نفس البيانات بـfloat32: القيم تطابق float16 ضمن خطأ التقريب، والذاكرة النصف
        ref = _closed_dataset()
        for tf in ('1h', '4h'):
            assert ref[f'X_{tf}'].dtype == np.float32 and ref[f'X_{tf}'].nbytes == 2 * ds[f'X_{tf}'].nbytes
            err = np.abs(ref[f'X_{tf}'] - ds[f'X_{tf}'].astype('float32')).max()
            assert err <= 2.5e-3, err
        np.testing.assert_array_equal(ref['last_candles'], ds['last_candles'])
        # التقسيم: كل الفريمات تُقسَم وdtype يبقى، وفجوة العزل تغطّي أوسع نافذة (4h × 32 = 128 ساعة) لا 32 ساعة فقط —
        # وإلا رأى مُدخل val (بسياق 4h) أهدافاً من train. فريم واحد: window + horizon كما كانت.
        cfg_s = _cfg(tf_order=['1h', '4h'], base_tf='1h', window_sizes={'1h': 32, '4h': 32}, stride=8,
                     forecast_horizon=1, higher_tf_mode='closed', min_split_samples=8, train_pct=0.7, val_pct=0.15)
        assert embargo_candles(ds, cfg_s) == 32 * 4 + 1 == embargo_candles({**ds, 'higher_tf_mode': 'closed'}, cfg_s)
        assert embargo_candles({**ds, 'higher_tf_mode': 'legacy'}, cfg_s) == 129      # legacy أيضاً: نافذة 4h تمتدّ 128h (R3-02)
        assert embargo_candles({**ds, 'timeframes': ['1h'], 'window_sizes': {'1h': 32}}, cfg_s) == 33
        train, val, test = _quiet(split_data, ds, config=cfg_s)
        for part in (train, val):
            assert part['X_4h'].dtype == np.float16 and len(part['X_4h']) == len(part['X_1h']) > 0
        t_tr, t_va, t_te = (np.asarray(p['last_candles'])[:, TS_COL].astype('int64') for p in (train, val, test))
        assert t_va.min() - 127 * H_NS > t_tr.max() + H_NS, (t_va.min() - t_tr.max()) / H_NS
        assert t_te.min() - 127 * H_NS > t_va.max() + H_NS, (t_te.min() - t_va.max()) / H_NS

    def t_closed_4h_price_normalised_by_own_window():
        # نافذة 4h تمتدّ 128 ساعة: بمركز/مقياس نافذة 1h (32 ساعة) تُقصّ أسعارها عند CLIP_ABS؛ 'closed' يطبّعها بنافذتها هي
        a, b = _closed_dataset(mode='legacy'), _closed_dataset()
        j = a['feature_order'].index('close')
        clipped = lambda ds: float((np.abs(ds['X_4h'][:, :, j]) >= CLIP_ABS - 1e-6).mean())
        assert clipped(a) > 0.02 and clipped(b) < 0.005, (clipped(a), clipped(b))

    def t_window_scale_params_batch_matches_calc_scale_params():
        rng = np.random.default_rng(3)
        x = (100 * np.exp(np.cumsum(rng.normal(0, 0.01, (40, 32)), axis=1))).astype('float32')
        x[7] = 5.0                                   # نافذة ثابتة: يحكمها حدّ المقياس المطلق/النسبي
        for method in ('robust', 'minmax'):
            c, s = _window_scale_params_batch(x, method)
            ref = np.array([calc_scale_params(row, method) for row in x], dtype='float64')
            np.testing.assert_allclose(c, ref[:, 0], rtol=1e-6)
            np.testing.assert_allclose(s, ref[:, 1], rtol=1e-6)

    def t_closed_mode_inert_for_single_tf_and_validated():
        # فريم واحد: 'closed' و x_storage_dtype=None لا يغيّران شيئاً — نفس المصفوفات ونفس مفاتيح مجموعة البيانات حرفياً
        def single(mode):
            cfg = _cfg(tf_order=['1h'], base_tf='1h', window_sizes={'1h': 32}, stride=8, higher_tf_mode=mode,
                       min_split_samples=8)

            def resample_fn(df, tf_order):
                return _quiet(resample_timeframes, df, tf_order, config=cfg)

            return _quiet(build_dataset_from_loader, [{'name': 'AUSDT'}, {'name': 'BUSDT'}],
                          lambda fid, n: _ohlcv(30, seed=70 + len(n)), resample_fn, config=cfg, max_workers=1)
        a, b = single('legacy'), single('closed')
        assert list(a) == list(b), (list(a), list(b))
        for k in a:
            if isinstance(a[k], np.ndarray):
                assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
        assert 'higher_tf_mode' not in b and 'x_storage_dtype' not in b
        try:
            _higher_tf_closed(_cfg(tf_order=['1h', '4h'], higher_tf_mode='strict'))
        except ValueError:
            pass
        else:
            raise AssertionError("higher_tf_mode غير معروف يجب أن يُرفض")

    def t_checkpoint_fingerprint_tracks_closed_mode_and_dtype():
        multi = dict(tf_order=['1h', '4h'], window_sizes={'1h': 32, '4h': 32}, stride=8)
        base = _checkpoint_fingerprint(_cfg(**multi), ['a'], 0)
        assert base != _checkpoint_fingerprint(_cfg(higher_tf_mode='closed', **multi), ['a'], 0)
        assert _checkpoint_fingerprint(_cfg(**multi, x_storage_dtype='float16'), ['a'], 0) != base
        assert _checkpoint_fingerprint(_cfg(**multi, x_storage_dtype=None), ['a'], 0) == base
        one = _cfg(tf_order=['1h'], window_sizes={'1h': 32}, stride=8)
        assert _checkpoint_fingerprint(one, ['a'], 0) == \
            _checkpoint_fingerprint(_cfg(tf_order=['1h'], window_sizes={'1h': 32}, stride=8,
                                         higher_tf_mode='closed'), ['a'], 0)

    tests = [t_exclude_matching, t_exclude_edge_cases, t_exclude_in_pipeline,
             t_hour_features_daily, t_hour_features_intraday,
             t_split_crowded_timeline, t_split_uniform_matches_requested,
             t_split_raises_when_impossible, t_split_explicit_dates_guarded,
             t_fetch_single_page_unchanged, t_fetch_paginates_and_merges,
             t_fetch_exact_multiple_of_max, t_fetch_history_shorter_than_limit,
             t_fetch_retries_transient_page_error,
             t_fetch_returns_none_on_persistent_page_failure,
             t_fetch_rejects_nonpositive_limit,
             t_exclude_features_names_and_patterns, t_exclude_features_idempotent_and_guard,
             t_exclude_features_reaches_dataset,
             t_ratelimiter_sliding_window, t_ratelimiter_threadsafe_never_exceeds,
             t_default_request_limit_is_2000_per_minute,
             t_fetch_acquires_per_request_including_retries, t_fetch_is_actually_throttled,
             t_reg_target_matches_direct_return, t_reg_target_uses_own_kind_not_last_close,
             t_reg_target_clip_default_depends_on_mode,
             t_invert_reg_predictions_return_mode, t_invert_reg_predictions_window_scale_uses_own_last,
             t_reg_target_scale_applied_after_clip, t_invert_reg_roundtrip_with_scale,
             t_dataset_records_reg_target_scale,
             t_class_labels_are_unit_encoded, t_window_scale_reg_centered_on_own_last,
             t_cs_normalize_matches_manual_zscore, t_cs_normalize_respects_min_assets,
             t_cs_normalize_does_not_mutate_input, t_cs_normalize_rejects_window_scale_mode,
             t_cs_normalize_rank_bounded, t_invert_cross_sectional_roundtrips_zscore,
             t_invert_cross_sectional_rejects_rank,
             t_compute_global_cutoff_matches_target_fraction, t_compute_global_cutoff_rejects_bad_pct,
             t_extract_test_dates_handles_both_shapes,
             t_raw_quantile_converges_to_kept_share_as_data_grows, t_raw_quantile_fails_loudly_on_crowded_data,
             t_build_leak_free_split_no_cross_asset_overlap,
             t_build_leak_free_split_matches_split_data_raw_quantile,
             t_compute_global_cutoff_matches_target_fraction, t_compute_global_cutoff_rejects_bad_pct,
             t_extract_test_dates_handles_both_shapes,
             t_raw_quantile_converges_to_kept_share_as_data_grows, t_raw_quantile_fails_loudly_on_crowded_data,
             t_build_leak_free_split_no_cross_asset_overlap,
             t_build_leak_free_split_matches_split_data_raw_quantile,
             t_fetch_data_uses_requested_interval_not_hardcoded,
             t_live_pad_length_scales_with_download_interval, t_auto_live_limit_scales_with_download_interval,
             t_build_dataset_live_decoupled_intervals_end_to_end,
             t_market_context_columns_names, t_add_market_context_aligns_and_zeroes_reference,
             t_market_context_reaches_dataset_via_build_dataset,
             t_market_corr_regime_columns_and_values, t_market_lag_ret_columns_and_values,
             t_market_crash_rebound_regime_columns_and_values,
             t_market_beta_columns_and_values,
             t_optional_feature_kinds_explicit, t_momentum_orth_natr,
             t_phase2_toggles_off_leave_pipeline_unchanged, t_phase2_hooks_from_drive_files,
             t_rolling_schedule_expanding_basic, t_rolling_schedule_sliding_basic,
             t_rolling_schedule_requires_initial_train_span,
             t_rolling_splits_no_leak_between_train_and_test_per_window,
             t_rolling_splits_max_windows_respected,
             t_fetch_funding_rate_paginates_and_merges,
             t_fetch_open_interest_hist_capped_by_platform_lookback,
             t_save_and_load_funding_open_interest_accumulates,
             t_funding_oi_columns_names, t_add_funding_oi_features_aligns_and_flags_availability,
             t_add_funding_oi_features_neutral_when_no_archive, t_funding_oi_disabled_returns_dfs_unchanged,
             t_funding_oi_reaches_dataset_via_build_dataset,
             t_checkpoint_fingerprint_changes_with_funding_oi_config,
             t_intraday_volume_concentration_columns_names,
             t_add_intraday_volume_concentration_features_aligns_and_flags_availability,
             t_add_intraday_volume_concentration_features_neutral_when_no_archive,
             t_intraday_volume_concentration_disabled_returns_dfs_unchanged,
             t_intraday_volume_concentration_reaches_dataset_via_build_dataset,
             t_intraday_vwap_columns_names,
             t_add_intraday_vwap_features_aligns_and_flags_availability,
             t_add_intraday_vwap_features_neutral_when_no_archive,
             t_intraday_vwap_disabled_returns_dfs_unchanged,
             t_intraday_vwap_reaches_dataset_via_build_dataset,
             t_intraday_efficiency_columns_names,
             t_add_intraday_efficiency_features_aligns_and_flags_availability,
             t_add_intraday_efficiency_features_neutral_when_no_archive,
             t_intraday_efficiency_disabled_returns_dfs_unchanged,
             t_intraday_efficiency_reaches_dataset_via_build_dataset,
             t_align_single_tf_fast_path_matches_manual_windows,
             t_align_single_tf_fast_path_respects_stride,
             t_align_single_tf_fast_path_empty_when_too_short,
             t_align_single_tf_peak_memory_bounded,
             t_window_ends_aligned_across_coins_with_stride, t_window_ends_aligned_multi_tf_path,
             t_window_grid_noop_for_stride_1_and_when_disabled, t_window_grid_gap_keeps_min_spacing,
             t_hourly_w32_s8_grid_and_purge, t_label_target_candle_is_ts_plus_horizon_across_gap,
             t_checkpoint_fingerprint_tracks_grid_and_cross_asset,
             t_cross_asset_features_nonconstant_via_loader, t_market_breadth_ignores_unlisted_coins,
             t_process_windows_price_level_matches_manual_formula,
             t_process_windows_degenerate_column_zeroed, t_process_windows_clip_applied,
             t_process_windows_keeps_constant_flags_and_funding,
             t_process_windows_peak_memory_bounded,
             t_default_workers_capped_by_ram, t_default_workers_unaffected_when_ram_unknown,
             t_checkpoint_fingerprint_changes_with_relevant_config,
             t_save_load_checkpoint_roundtrip, t_load_checkpoint_rejects_mismatched_fingerprint,
             t_load_checkpoint_missing_returns_none, t_clear_checkpoint_removes_files,
             t_build_dataset_from_loader_checkpoint_resume_end_to_end,
             t_build_dataset_checkpoint_invalidated_by_config_change,
             t_closed_align_last_4h_bar_closes_before_t,
             t_closed_align_gaps_match_oracle_and_resample_drops_partial_bars,
             t_closed_4h_no_lookahead_end_to_end,
             t_closed_dataset_shapes_alignment_and_float16,
             t_closed_4h_price_normalised_by_own_window,
             t_window_scale_params_batch_matches_calc_scale_params,
             t_closed_mode_inert_for_single_tf_and_validated,
             t_checkpoint_fingerprint_tracks_closed_mode_and_dtype,
             t_disk_backed_build_identical_to_ram_and_dir_roundtrip]

    failed = []
    for t in tests:
        try:
            t()
            if verbose:
                print(f"  ✅ {t.__name__}")
        except Exception as exc:                            # noqa: BLE001
            failed.append((t.__name__, f"{type(exc).__name__}: {exc}"))
            if verbose:
                print(f"  ❌ {t.__name__}: {type(exc).__name__}: {str(exc)[:160]}")

    if failed:
        raise AssertionError(f"فشل {len(failed)} من {len(tests)} اختباراً: "
                             f"{[n for n, _ in failed]}")
    if verbose:
        print(f"✅ نجحت كل الاختبارات الذاتية ({len(tests)}).")
    return True


run_pipeline_selftests()

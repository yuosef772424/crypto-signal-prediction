"""
PURPOSE:  Dataset assembly across all assets: build_dataset (loader or preloaded), exclusions, cross-asset first pass, accumulation; both sources delegate to prepare_single_asset.
TAGS:     pipeline, build_dataset, build_dataset_from_loader, build_dataset_from_preloaded, accumulator, asset_bounds, SKIP_LABELS
PITFALLS: Golden digests (tests/test_multi_tf.py::test_old_presets_byte_identical) cover this path: any output change must be deliberate. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 15) خط أنابيب تجميع البيانات عبر كل الأصول (`pipeline.py` سابقاً)
"""
# @title
"""
خط أنابيب تجميع البيانات عبر كل الأصول.

مصدران للبيانات، ومنطق تجميع واحد مشترك:

  * :func:`build_dataset_from_loader`    — تحميل كل عملة من ملف خام ثم resample.
  * :func:`build_dataset_from_preloaded` — بيانات مُجهَّزة مسبقاً ``{asset: {tf: df}}``.

كلاهما يُفوّض المنطق الفعلي إلى :func:`prepare_single_asset` فلا يوجد أي
تكرار أو اختلاف في السلوك بينهما.
"""

SKIP_LABELS = {
    "name_not_in_data": "اسم غير مطابق",
    "missing_tf": "فريم مفقود",
    "insufficient_length": "بيانات غير كافية",
    "no_valid_samples": "لا عينات بعد المحاذاة",
    "load_failed": "فشل التحميل",
    "excluded": "مستبعدة يدوياً",
}

#: كل كم عملة نُجري تنظيف ذاكرة (gc مكلف، فلا يُستدعى لكل عملة).
GC_EVERY = 10


class _AssetResult:
    """نتيجة معالجة عملة واحدة — تُمرَّر من الخيط العامل إلى الخيط الرئيسي."""

    __slots__ = ('name', 'status', 'payload', 'note', 'n_candles')

    def __init__(self, name: str, status: str, payload=None,
                 note: str = '', n_candles: int = 0):
        self.name = name
        self.status = status
        self.payload = payload
        self.note = note
        self.n_candles = n_candles


def _consume(acc: '_Accumulator', process_fn, configs: List[Dict],
             workers: int, prefetch: Optional[int], tf_order: List[str]) -> None:
    """يُشغّل ``process_fn`` على كل العملات (متوازياً) ويُجمّع النتائج بالترتيب.

    التجميع يبقى في الخيط الرئيسي وحده، فلا يحتاج ``_Accumulator`` أي أقفال،
    ويظل ترتيب الصفوف مطابقاً لترتيب ``configs`` في كل تشغيل.
    """
    total = len(configs)
    # disk_backed: التسريب داخل خيط العامل نفسه (قبل أن تصل الحمولة إلى الطابور) فلا تتراكم مصفوفات العملات المنجَزة في الرام.
    process_fn = acc.spilling(process_fn)
    for i, res in enumerate(
            imap_ordered(process_fn, configs, workers, prefetch), start=1):
        if res.status != "ok":
            acc.skip(res.status, res.name)
            print(f"   [{i}/{total}] ⚠️ {res.name}: "
                  f"{SKIP_LABELS.get(res.status, res.status)}"
                  f"{f' — {res.note}' if res.note else ''}")
            continue

        n = acc.add(res.name, *res.payload)
        if n:
            print(f"   [{i}/{total}] ✅ {res.name}: {res.n_candles:,} شمعة "
                  f"→ {n:,} نافذة")
        else:
            print(f"   [{i}/{total}] ⚠️ {res.name}: لا عيّنات صالحة بعد المحاذاة")

        res.payload = None                 # حرّر مراجع المصفوفات فوراً
        if acc.spill_dir is not None:
            _trim_memory()                 # عملة واحدة في الرام: أعد المحرَّر للنظام بعد كل عملة (غير مكلف)
        elif i % GC_EVERY == 0:
            gc.collect()
    _trim_memory() if acc.spill_dir is not None else gc.collect()


class _Accumulator:
    """يُجمِّع مخرجات كل الأصول ثم يدمجها في قاموس مجموعة بيانات واحد."""

    def __init__(self, tf_order: List[str], target_heads: List[str], spill_dir=None):
        #: مجلد scratch لهذا البناء (CONFIG['disk_backed']) أو None = تجميع في الرام كما كان.
        self.spill_dir = spill_dir
        self.tf_order = tf_order
        self.target_heads = target_heads
        self.X = {tf: [] for tf in tf_order}
        self.y = {h: [] for h in target_heads}
        self.bases: list = []
        self.last: list = []
        self.bounds: List[Dict] = []
        self.offset = 0
        self.skips = {k: 0 for k in SKIP_LABELS}
        #: أسماء العملات المتخطّاة لكل سبب — تُتيح نسخها مباشرة إلى excluded_coins.
        self.skipped: Dict[str, List[str]] = {k: [] for k in SKIP_LABELS}

    def spilling(self, process_fn):
        """يلفّ ``process_fn`` بحيث تُكتب حمولة كل عملة ناجحة إلى القرص وتُستبدل بمراجع :class:`_SpilledArray` (نفس البنية
        ``(X_tf, y_t, bases, last)``، فيعمل :meth:`add` كما هو). بلا ``spill_dir`` تُرجع الدالة نفسها."""
        if self.spill_dir is None:
            return process_fn

        def _wrapped(cfg):
            res = process_fn(cfg)
            if res.status == 'ok' and res.payload is not None and res.payload[2].shape[0] > 0:
                res.payload = _spill_payload(self.spill_dir, res.name, res.payload)
            return res
        return _wrapped

    def skip(self, reason: str, name: Optional[str] = None) -> None:
        self.skips[reason] = self.skips.get(reason, 0) + 1
        if name is not None:
            self.skipped.setdefault(reason, []).append(name)

    def add(self, name: str, X_tf, y_t, bases, last) -> int:
        n = bases.shape[0]
        if n == 0:
            self.skip("no_valid_samples", name)
            return 0
        self.bounds.append({'name': name, 'start': self.offset, 'end': self.offset + n})
        self.offset += n
        for tf in self.tf_order:
            self.X[tf].append(X_tf[tf])
        for h in self.target_heads:
            self.y[h].append(y_t[h])
        self.bases.append(bases)
        self.last.append(last)
        return n

    def report(self, requested: int) -> None:
        print(f"\n\n📋 ملخص: {len(self.bounds)} عملة صالحة من أصل {requested} مطلوبة")
        if sum(self.skips.values()):
            parts = [f"{SKIP_LABELS[k]}={v}" for k, v in self.skips.items() if v]
            print(f"   أسباب التخطي: {' | '.join(parts)}")
        failed = self.skipped.get('load_failed') or []
        if failed:
            # ملف غير موجود عادةً (أسهم/سلع/عملات لم تُنزَّل) — لا تُجدي إعادة المحاولة.
            print(f"   💡 تعذّر تحميل {len(failed)} عملة. لتخطّيها مباشرة في التشغيل القادم:\n"
                  f"      update_config(excluded_coins={failed})")

    def finalize(self, config: dict, scaler_type: str, base_tf: str,
                 features: List[str]) -> Dict:
        if not self.bases:
            raise ValueError(
                "❌ لم تُجمَّع أي عملة بنجاح — راجع 'أسباب التخطي' أعلاه.\n"
                "   الاحتمال الأكثر شيوعاً: أسماء configs لا تطابق مفاتيح data "
                "(استخدم diagnose_data_vs_configs للمقارنة)."
            )

        print("\n🔗 دمج" + (f" على القرص ({self.spill_dir})..." if self.spill_dir is not None else "..."), end=" ")
        if self.spill_dir is not None:
            # أجزاء العملات على القرص → ملفات مدموجة (open_memmap) بنفس الترتيب؛ الرام = دفعة نسخ واحدة فقط.
            merged = Path(self.spill_dir) / 'merged'
            merged.mkdir(parents=True, exist_ok=True)
            X_final = {tf: _merge_spilled(self.X[tf], merged / f'X_{tf}.npy')
                       for tf in self.tf_order if self.X[tf]}
            y_final = {h: _merge_spilled(self.y[h], merged / f'y_{h}.npy')
                       for h in self.target_heads if self.y[h]}
            bases_final = _merge_spilled(self.bases, merged / 'base_params.npy')
            last_final = _merge_spilled(self.last, merged / 'last_candles.npy')
            shutil.rmtree(Path(self.spill_dir) / 'parts', ignore_errors=True)
        else:
            X_final = {tf: np.concatenate(self.X[tf], axis=0)
                       for tf in self.tf_order if self.X[tf]}
            y_final = {h: np.concatenate(self.y[h], axis=0)
                       for h in self.target_heads if self.y[h]}
            bases_final = np.concatenate(self.bases, axis=0)
            last_final = np.concatenate(self.last, axis=0)
        self.X, self.y, self.bases, self.last = {}, {}, [], []
        gc.collect()

        print(f"✅ {len(bases_final):,} sequences")
        for tf in self.tf_order:
            print(f"   X_{tf}: {X_final[tf].shape}")
        for h in self.target_heads:
            print(f"   y_{h}: {y_final[h].shape}")

        out: Dict = {
            'base_params': bases_final,
            'last_candles': last_final,
            'mode': f"{config.get('target_mode', 'direction')}_{scaler_type}",
            'scaler_type': scaler_type,
            'feature_order': features,
            'timeframes': self.tf_order,
            'targets': self.target_heads,      # أسماء الرؤوس المُفعَّلة
            'price_targets': list(config['targets']),
            'asset_bounds': self.bounds,
            'base_tf': base_tf,
            'window_sizes': dict(config['window_sizes']),
            'stride': config['stride'],
            'forecast_horizon': config['forecast_horizon'],
            # وحدة أهداف {t}_reg: كل مستهلك يقرؤها من هنا (غيابها في ملف قديم = 'return' و1.0)
            'reg_target_mode': config.get('reg_target_mode', 'return'),
            'reg_target_scale': float(config.get('reg_target_scale', 1.0)),
            # حدود التقسيم وبداية الـholdout المختوم كما بُنيت البيانات (إعداد 1h_s8 يحملها): main يقرؤها منها ما لم
            # يحدّدها صراحةً، فلا تبقى في التوثيق وحده. None = غير محدّدة (ملف قديم: المفتاح غائب ← None أيضاً).
            'split_dates': deepcopy(config.get('split_dates')),
            'holdout_start': config.get('holdout_start'),
            # أسماء المتخطّاة لكل سبب (فارغة الأسباب محذوفة) — للتدقيق وللنسخ.
            'skipped_assets': {k: list(v) for k, v in self.skipped.items() if v},
        }
        # مفاتيح إضافية فقط حين تُفعَّل الميزتان: مجموعات البيانات القديمة تبقى بمفاتيحها حرفياً.
        if _higher_tf_closed(dict(config, tf_order=list(self.tf_order))):
            out['higher_tf_mode'] = 'closed'
        if config.get('x_storage_dtype'):
            out['x_storage_dtype'] = str(config['x_storage_dtype'])
        # وحدة الأعمدة السعرية في X (غيابها = 'window_scale'): decode_price_window يقرؤها من هنا.
        if price_norm_mode(config) != PRICE_NORM_WINDOW:
            out['price_norm_mode'] = price_norm_mode(config)
            out['price_pct_clip'] = price_pct_clip(config)
        # الشروط التي اختيرت بها العيّنات (غيابها = كل العيّنات).
        if config.get('sample_filters'):
            out['sample_filters'] = deepcopy(list(config['sample_filters']))
            print(f"   🎯 فلاتر العيّنات: {out['sample_filters']}")
        for tf in self.tf_order:
            out[f'X_{tf}'] = X_final[tf]
        for h in self.target_heads:
            out[f'y_{h}'] = y_final[h]
        return out


def _apply_exclusions(configs: List[Dict], exclude, config: dict,
                     acc: '_Accumulator') -> List[Dict]:
    """يُطبّق الاستبعاد اليدوي ويُسجّله في المُجمِّع، ويُرجع configs المتبقّية."""
    kept, removed, not_found = split_excluded_coins(configs, exclude, config)
    for name in removed:
        acc.skip("excluded", name)
    if removed:
        print(f"🚫 مستبعدة يدوياً ({len(removed)}): {removed[:8]}"
              f"{'...' if len(removed) > 8 else ''}")
    if not_found:
        print(f"   ⚠️ أسماء استبعاد غير موجودة في configs — تحقّق من الإملاء: "
              f"{not_found[:8]}{'...' if len(not_found) > 8 else ''}")
    return kept


#: تجميع high/low/close عند إعادة أخذ العيّنات — نفس قواعد OHLCV_AGG.
_HLC_AGG = {k: OHLCV_AGG[k] for k in ('high', 'low', 'close')}


def _load_cross_asset_frames(configs: List[Dict], load_asset_fn: Callable,
                             tf_order: List[str], tail: int, min_len: int,
                             workers: int, prefetch: Optional[int],
                             config: Optional[dict] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """مرور تحميل أول لمسار ``build_dataset_from_loader``: ``{أصل: {فريم: high/low/close}}``.

    **لماذا؟** MKT_BREADTH_ وMOM_ORTH_NATR وMOM_RANK_ تُحسب عبر كل الأصول في نفس اللحظة،
    بينما يعالج مسار التحميل كل عملة وحدها في خيط. سابقاً كانت تُمرَّر ``None`` فتأخذ
    قيمة محايدة ثابتة (0.5 أو 0) في كل الصفوف، وتصير صفراً بعد التطبيع — ميزة ميّتة
    لم يلحظها أحد لأن الشكل صحيح (قيس على بيانات 1h: صفر في 113,754 عيّنة).

    يُحتفَظ بثلاثة أعمدة float64 فقط لكل عملة ثم تُحمَّل العملة كاملة مرة ثانية أثناء
    المعالجة: ضعف القراءة من Drive أرخص من إبقاء كل البيانات الخام في الذاكرة، وهو ما
    صُمّم هذا المسار لتجنّبه. عملة يفشل تحميلها أو تقصر عن ``min_len * 2`` تُستبعد من
    المقطع هنا كما تُستبعد لاحقاً من المعالجة.

    ``higher_tf_mode='closed'``: شموع الفريمات الأعلى الناقصة (فجوة في ملف العملة) تُستبعد هنا كما في
    :func:`resample_timeframes`، فلا تدخل شمعة 4h ناقصة في ترتيب/اتساع عملة عند طابعها.
    """
    closed = _higher_tf_closed(dict(config if config is not None else CONFIG, tf_order=list(tf_order)))
    # مرور خفيف: الإطار الخام يُحرَّر فور أخذ الأعمدة الثلاثة ولا يُحتفَظ إلا بالنتائج (high/low/close لكل عملة × فريم).
    # ``cross_asset_dtype='float32'`` يُنصّف حتى هذا (≈ 44 MB لـ 83 عملة 1h) لكنه يغيّر الميزات العابرة للأصول في الخانة
    # السابعة فتنقلب بعض تقريبات float16 — لذا الافتراضي float64 (نتائج مطابقة بايتاً لمسار الرام).
    hlc_dtype = str((config if config is not None else CONFIG).get('cross_asset_dtype') or 'float64')

    def _one(cfg):
        name = cfg['name'] if isinstance(cfg, dict) else str(cfg)
        try:
            df = load_asset_fn(cfg.get('file_id') if isinstance(cfg, dict) else None, name)
        except Exception:                               # noqa: BLE001 — يُبلَّغ عنه في المرور الثاني
            return name, None
        if tail:
            df = df[-tail:]
        if len(df) < min_len * 2 or not {'high', 'low', 'close'} <= set(df.columns):
            return name, None
        hlc = df[['high', 'low', 'close']].astype(hlc_dtype).sort_index()
        del df
        frames = {tf: hlc.resample(tf).agg(_HLC_AGG).dropna(how='all') for tf in tf_order}
        if closed:
            frames = {tf: (fr if tf == tf_order[0] else _keep_full_bars(fr, hlc.index[hlc['close'].notna()], tf, tf_order[0]))
                      for tf, fr in frames.items()}
        return name, frames

    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for name, frames in imap_ordered(_one, configs, workers, prefetch):
        if frames is not None:
            out[name] = frames
    return out


def _resolve(config: Optional[dict], tf_order, window_sizes, targets,
             forecast_horizon, scaler_type, stride, base_tf):
    config = CONFIG if config is None else config
    tf_order = config["tf_order"] if tf_order is None else tf_order
    window_sizes = config["window_sizes"] if window_sizes is None else window_sizes
    targets = config["targets"] if targets is None else targets
    forecast_horizon = (config["forecast_horizon"]
                        if forecast_horizon is None else forecast_horizon)
    scaler_type = config["scaler_type"] if scaler_type is None else scaler_type
    stride = config["stride"] if stride is None else stride
    base_tf = base_tf or config.get("base_tf") or tf_order[0]
    return (config, tf_order, window_sizes, targets, forecast_horizon,
            scaler_type, stride, base_tf)


# ══════════════════════════════════════════════════════════════════════════
# المصدر 1: ملفات خام + resample
# ══════════════════════════════════════════════════════════════════════════
def build_dataset_from_loader(
    configs: List[Dict],
    load_asset_fn: Callable,
    resample_fn: Callable,
    tf_order: Optional[List[str]] = None,
    window_sizes: Optional[Dict[str, int]] = None,
    targets: Optional[List[str]] = None,
    forecast_horizon: Optional[int] = None,
    scaler_type: Optional[str] = None,
    stride: Optional[int] = None,
    base_tf: Optional[str] = None,
    tail: int = 0,
    max_workers: Optional[int] = None,
    prefetch: Optional[int] = None,
    exclude: Optional[Iterable[str]] = None,
    checkpoint_dir: Optional[str] = None,
    config: Optional[dict] = None,
) -> Dict:
    """بناء مجموعة البيانات بتحميل كل عملة من مصدرها الخام — **متوازياً**.

    التنزيل والمؤشرات وبناء النوافذ تجري كلها داخل خيط العامل، فيتداخل حساب
    عملة مع تنزيل أخرى بدل الانتظار عاطلين على الشبكة.

    Args:
        configs: ``[{'name': 'BTCUSDT', 'file_id': '...'}, ...]``
        load_asset_fn: ``callable(file_id, name) -> DataFrame`` بأعمدة OHLCV.
        resample_fn: ``callable(df, tf_order) -> {tf: DataFrame}`` — يجب أن
            تستدعي ``add_features`` على كل فريم.
        tail: إن كان > 0، يُقتصر على آخر ``tail`` شمعة من كل عملة (للتجارب السريعة).
        max_workers: عدد الخيوط. ``None`` يختار تلقائياً، و``1`` يُعيد السلوك
            التسلسلي القديم بالضبط.
        prefetch: أقصى عدد عملات قيد المعالجة معاً — يحدّ الذاكرة على Colab
            (افتراضياً ``max_workers * 2``).
        exclude: أسماء عملات تُتخطّى قبل التحميل (``None`` → ``CONFIG['excluded_coins']``).
            تظهر في ``dataset['skipped_assets']['excluded']``.
        checkpoint_dir: مسار (على Drive عادة) لحفظ نتيجة كل عملة فور معالجتها،
            واستئناف ما سبق حفظه بدل إعادة معالجته — يحمي من فقدان عمل ساعات
            عند انقطاع الجلسة. ``None`` (الافتراضي) يُعطّل هذا كلياً (لا تغيير
            في السلوك). راجع :func:`clear_checkpoint` للتحكّم اليدوي في نقاط
            الاستئناف، وتحذير التقادم في رأس هذا القسم.

    ``CONFIG['disk_backed']=True``: رام محدودة — كل عملة تُسرَّب إلى ``CONFIG['scratch_dir']`` (.npy) فور معالجتها ثم تُحرَّر،
    ويُدمج الناتج على القرص فتكون مصفوفات X memmap (مطابقة بايتاً للوضع العادي). راجع :func:`_scratch_run_dir`.

    الناتج **مطابق** للوضع التسلسلي مهما تغيّر عدد الخيوط: الترتيب محفوظ —
    بما في ذلك عند استئناف جزئي (أصل مُستأنَف من القرص أو مُعالَج لتوّه، كلاهما
    يمرّ عبر نفس ``_process``/``imap_ordered`` بترتيب ``configs`` نفسه).
    """
    (config, tf_order, window_sizes, targets, forecast_horizon,
     scaler_type, stride, base_tf) = _resolve(
        config, tf_order, window_sizes, targets, forecast_horizon,
        scaler_type, stride, base_tf)

    # ✅ تُشتق مرة واحدة **قبل** إطلاق الخيوط: لو تُركت لأول عامل يطلبها لتسابقت
    #    الخيوط على تعبئة config['feature_order'] معاً. مفاتيح المرحلة ٢ ("auto") تُثبَّت قبلها.
    resolve_phase2_toggles(config)
    features = feature_order(config)
    target_heads = get_target_heads(targets, config)
    acc = _Accumulator(tf_order, target_heads)
    requested = len(configs)
    configs = _apply_exclusions(configs, exclude, config, acc)
    min_len = window_sizes[tf_order[0]]
    workers = default_workers(len(configs)) if max_workers is None else max_workers
    ref_symbol = (config.get('market_context') or {}).get('reference_symbol', 'BTCUSDT')
    market_dfs = build_market_context(load_asset_fn=load_asset_fn, resample_fn=resample_fn,
                                      config=config)
    # ✅ الميزات العابرة للأصول تحتاج كل العملات معاً قبل معالجة أيٍّ منها.
    momentum_rank_dfs = market_breadth_dfs = momentum_orth_dfs = cross_sig = None
    universe = [c['name'] if isinstance(c, dict) else str(c) for c in configs]
    if config.get('disk_backed'):
        acc.spill_dir = _scratch_run_dir(config, _checkpoint_fingerprint(config, features, tail, universe=universe))
        print(f"💾 بناء مدعوم بالقرص: كل عملة تُكتب إلى {acc.spill_dir} ثم تُحرَّر من الرام ويُدمج على القرص")
    if (momentum_rank_columns(config) or market_breadth_columns(config)
            or momentum_orth_natr_columns(config)):
        print(f"🌐 مرور أول: high/low/close لـ {len(configs)} عملة للميزات العابرة للأصول...")
        cross_data = _load_cross_asset_frames(configs, load_asset_fn, tf_order, tail, min_len,
                                              workers, prefetch, config=config)
        print(f"   {len(cross_data)} عملة في المقطع العرضي"
              + (f" | الرام بعد المرور: {_rss_gb():.2f} GB" if _rss_gb() is not None else ""))
        momentum_rank_dfs = build_momentum_rank(data=cross_data, config=config)
        market_breadth_dfs = build_market_breadth(data=cross_data, config=config)
        momentum_orth_dfs = build_momentum_orth_natr(data=cross_data, config=config)
        if checkpoint_dir is not None:
            cross_sig = _frame_signature(cross_data)
        del cross_data
        _trim_memory()
    ckpt_fp = shared_sig = None
    if checkpoint_dir is not None:
        ckpt_fp = _checkpoint_fingerprint(config, features, tail, universe=universe)
        # بيانات الأقران تدخل ميزات كل عملة (السياق السوقي، الاتساع، الزخم العابر): تغيّرها يُبطل نقاط الجميع.
        shared_sig = _frame_signature({'market': market_dfs, 'cross': cross_sig})

    def _process(cfg: Dict) -> _AssetResult:
        """السلسلة الكاملة لعملة واحدة — تعمل داخل خيط عامل.

        نضع التنزيل والمؤشرات وبناء النوافذ معاً في العامل، فيتداخل حساب عملة
        مع تنزيل أخرى بدل انتظار الشبكة عاطلين.
        """
        name = cfg['name']
        try:
            df = load_asset_fn(cfg.get('file_id'), name)
        except Exception as exc:                        # noqa: BLE001
            return _AssetResult(name, "load_failed", note=str(exc)[:80])

        if tail:
            df = df[-tail:]

        # ✅ مسار الاستئناف: نقطة حفظ صالحة (نفس بصمة الإعدادات **والبيانات**) قبل أي حساب — إن وُجدت، هذه العملة
        # "تنجح" فوراً. الملف الخام يُقرأ أولاً لأنه جزء من البصمة: بيانات أُعيد جلبها تُعاد معالجتها.
        asset_fp = None
        if checkpoint_dir is not None:
            asset_fp = _asset_checkpoint_fingerprint(ckpt_fp, df, name, config, shared_sig)
            loaded = _load_asset_checkpoint(checkpoint_dir, name, asset_fp)
            if loaded is not None:
                n = int(loaded[2].shape[0])          # bases.shape[0] = عدد النوافذ
                return _AssetResult(name, "ok", payload=loaded, n_candles=n)
        if len(df) < min_len * 2:
            return _AssetResult(name, "insufficient_length",
                                note=f"{len(df):,} صف فقط")

        dfs = resample_fn(df, tf_order)
        missing = [tf for tf in tf_order if tf not in dfs]
        if missing:
            return _AssetResult(name, "missing_tf", note=f"مفقود: {missing}")
        dfs = add_market_context(dfs, market_dfs, name == ref_symbol, config)
        dfs = add_momentum_rank(dfs, (momentum_rank_dfs or {}).get(name), config)
        dfs = add_funding_oi_features(dfs, name, config)
        dfs = add_intraday_efficiency_features(dfs, name, config)
        dfs = add_intraday_vwap_features(dfs, name, config)
        dfs = add_intraday_volume_concentration_features(dfs, name, config)
        dfs = add_intraday_15m_features(dfs, name, config)
        dfs = add_market_breadth(dfs, market_breadth_dfs, config)
        dfs = add_momentum_orth_natr(dfs, (momentum_orth_dfs or {}).get(name), config)
        n_candles = len(dfs[tf_order[0]])
        if n_candles < min_len * 1.5:
            return _AssetResult(name, "insufficient_length",
                                note=f"{n_candles:,} شمعة فقط")

        payload = prepare_single_asset(
            dfs=dfs, tf_order=tf_order, window_sizes=window_sizes, targets=targets,
            forecast_horizon=forecast_horizon, scaler_type=scaler_type,
            base_tf=base_tf, stride=stride, config=config)

        if checkpoint_dir is not None:
            try:
                _save_asset_checkpoint(checkpoint_dir, name, *payload, fingerprint=asset_fp)
            except Exception as exc:                    # noqa: BLE001
                print(f"   ⚠️ [{name}] تعذّر حفظ نقطة استئناف ({exc}) — "
                      f"ستُعاد معالجتها عند أي استئناف لاحق.")
        return _AssetResult(name, "ok", payload=payload, n_candles=n_candles)

    print(f"⚙️ معالجة {len(configs)} عملة "
          f"({'تسلسلياً' if workers <= 1 else f'بـ {workers} خيطاً متوازياً'})...")
    _consume(acc, _process, configs, workers, prefetch, tf_order)

    acc.report(requested)
    return acc.finalize(config, scaler_type, base_tf, features)


# ══════════════════════════════════════════════════════════════════════════
# المصدر 2: بيانات مُجهَّزة مسبقاً {asset: {tf: DataFrame}}
# ══════════════════════════════════════════════════════════════════════════
def build_dataset_from_preloaded(
    configs: List[Dict],
    data: Dict[str, Dict[str, pd.DataFrame]],
    tf_order: Optional[List[str]] = None,
    window_sizes: Optional[Dict[str, int]] = None,
    targets: Optional[List[str]] = None,
    forecast_horizon: Optional[int] = None,
    scaler_type: Optional[str] = None,
    stride: Optional[int] = None,
    base_tf: Optional[str] = None,
    max_workers: Optional[int] = None,
    prefetch: Optional[int] = None,
    exclude: Optional[Iterable[str]] = None,
    config: Optional[dict] = None,
) -> Dict:
    """بناء مجموعة البيانات من قاموس مُحمَّل مسبقاً — **متوازياً**.

    لا شبكة هنا، لكن ``prepare_single_asset`` حسابات numpy ثقيلة تُحرِّر GIL،
    فالتوازي يفيد أيضاً. ``max_workers=1`` يُعيد السلوك التسلسلي.
    """
    if isinstance(data, dict) and 'base_params' in data:
        # خطأ شائع: تمرير مخرج build_dataset (مجموعة بيانات جاهزة) هنا بدل
        # {asset: {tf: DataFrame}} — يفشل لاحقاً بـ AttributeError غامض على
        # tf_dict.keys() لأن القيمة تكون مصفوفة numpy لا قاموس فريمات.
        raise TypeError(
            "data تحمل 'base_params' — تبدو مجموعة بيانات جاهزة (مخرج "
            "build_dataset) لا بيانات خام {asset: {tf: DataFrame}}. "
            "استخدمها مباشرة مع split_data() بدل تمريرها هنا.")

    (config, tf_order, window_sizes, targets, forecast_horizon,
     scaler_type, stride, base_tf) = _resolve(
        config, tf_order, window_sizes, targets, forecast_horizon,
        scaler_type, stride, base_tf)

    resolve_phase2_toggles(config)
    features = feature_order(config)
    target_heads = get_target_heads(targets, config)
    acc = _Accumulator(tf_order, target_heads)
    requested = len(configs)
    configs = _apply_exclusions(configs, exclude, config, acc)
    min_len = window_sizes[tf_order[0]]
    if config.get('disk_backed'):
        acc.spill_dir = _scratch_run_dir(config, _checkpoint_fingerprint(
            config, features, 0, universe=[c['name'] if isinstance(c, dict) else str(c) for c in configs]))
        print(f"💾 بناء مدعوم بالقرص: كل عملة تُكتب إلى {acc.spill_dir} ثم تُحرَّر من الرام ويُدمج على القرص")

    workers = default_workers(len(configs)) if max_workers is None else max_workers
    # ✅ نُبقي أعمدة الأهداف مع الميزات: prepare_single_asset يحتاج
    #    high/low/close الخام حتى لو كانت مستبعَدة من مدخلات النموذج.
    keep = list(features) + [c for c in TARGET_COLUMNS if c not in features]
    ref_symbol = (config.get('market_context') or {}).get('reference_symbol', 'BTCUSDT')
    market_dfs = build_market_context(data=data, config=config)
    momentum_rank_dfs = build_momentum_rank(data=data, config=config)
    market_breadth_dfs = build_market_breadth(data=data, config=config)
    momentum_orth_dfs = build_momentum_orth_natr(data=data, config=config)

    def _process(cfg: Dict) -> _AssetResult:
        name = cfg['name'] if isinstance(cfg, dict) else str(cfg)
        if name not in data:
            return _AssetResult(name, "name_not_in_data",
                                note="راجع diagnose_data_vs_configs")

        tf_dict = data[name]
        missing = [tf for tf in tf_order if tf not in tf_dict]
        if missing:
            return _AssetResult(name, "missing_tf",
                                note=f"مفقود {missing}، متوفر {list(tf_dict)}")

        raw_dfs = {tf: tf_dict[tf] for tf in tf_order}
        raw_dfs = add_market_context(raw_dfs, market_dfs, name == ref_symbol, config)
        raw_dfs = add_momentum_rank(raw_dfs, (momentum_rank_dfs or {}).get(name), config)
        raw_dfs = add_funding_oi_features(raw_dfs, name, config)
        raw_dfs = add_intraday_efficiency_features(raw_dfs, name, config)
        raw_dfs = add_intraday_vwap_features(raw_dfs, name, config)
        raw_dfs = add_intraday_volume_concentration_features(raw_dfs, name, config)
        raw_dfs = add_intraday_15m_features(raw_dfs, name, config)
        raw_dfs = add_market_breadth(raw_dfs, market_breadth_dfs, config)
        raw_dfs = add_momentum_orth_natr(raw_dfs, (momentum_orth_dfs or {}).get(name), config)
        dfs = {tf: extract_features(raw_dfs[tf], keep) for tf in tf_order}
        n_candles = len(dfs[tf_order[0]])
        if n_candles < min_len * 1.5:
            return _AssetResult(name, "insufficient_length",
                                note=f"{n_candles:,} شمعة فقط")

        payload = prepare_single_asset(
            dfs=dfs, tf_order=tf_order, window_sizes=window_sizes, targets=targets,
            forecast_horizon=forecast_horizon, scaler_type=scaler_type,
            base_tf=base_tf, stride=stride, config=config)
        return _AssetResult(name, "ok", payload=payload, n_candles=n_candles)

    print(f"⚙️ معالجة {len(configs)} عملة "
          f"({'تسلسلياً' if workers <= 1 else f'بـ {workers} خيطاً متوازياً'})...")
    _consume(acc, _process, configs, workers, prefetch, tf_order)

    acc.report(requested)
    return acc.finalize(config, scaler_type, base_tf, features)


def build_dataset(configs: List[Dict], *,
                  data: Optional[Dict] = None,
                  load_asset_fn: Optional[Callable] = None,
                  resample_fn: Optional[Callable] = None,
                  **kwargs) -> Dict:
    """واجهة موحّدة: تختار المصدر تلقائياً حسب ما يُمرَّر.

    مرّر ``data`` للبيانات المُجهَّزة مسبقاً، أو ``load_asset_fn`` + ``resample_fn``
    للتحميل من الملفات الخام (هنا: ``load_asset`` + ``make_resample_fn(CONFIG)``).
    """
    if data is not None:
        return build_dataset_from_preloaded(configs, data, **kwargs)
    if load_asset_fn is not None and resample_fn is not None:
        return build_dataset_from_loader(configs, load_asset_fn, resample_fn, **kwargs)
    raise ValueError(
        "حدّد مصدر البيانات: إمّا data={asset: {tf: df}} أو "
        "load_asset_fn + resample_fn.")


#: اسم متوافق مع الدفاتر القديمة.
prepare_regression_multiframe_multiasset = build_dataset_from_loader

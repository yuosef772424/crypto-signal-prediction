"""
PURPOSE:  Hourly presets: 20-b (1h w168 s32 h4, and 1h w32 s8 h1), 20-c (1h + 4h context, float16) with its contract self-test, 20-d (pct_change price normalisation) — overrides, names and one-call builders.
TAGS:     presets, hourly, apply_hourly_preset, HOURLY_W32_S8_OVERRIDES, HOURLY_4H_OVERRIDES, PCT_CHANGE_OVERRIDES, build_hourly_w32_s8_dataset, build_hourly_4h_dataset, build_hourly_pct_dataset, run_hourly_4h_selftests
PITFALLS: Presets inherit HOURLY_W32_S8_OVERRIDES; changing it changes every preset (and the golden digests). Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
# @title 20-ب) بيانات فريم الساعة: نافذة 168 شمعة، stride=32، أفق 4 ساعات — عملات COINS_BY_CATEGORY فقط
# ═══════════════════════════════════════════════════════════════════════════
# تعريفات فقط (لا تنفيذ تلقائي — آمن مع %run من main). الاستخدام في آخر الخلية.
#
# ⚠️ ثلاث حقائق تحكم جدوى هذا الإعداد (تطبعها estimate_hourly_feasibility بأرقام ملفاتك الفعلية):
#  1. stride=32 على فريم الساعة = 0.75 عيّنة/يوم لكل عملة — **أقل** من الفريم اليومي بـstride=1
#     (عيّنة/يوم). العيّنات لا تزيد بهذا الإعداد، بل تقلّ (مع عملات أقل من السجل الكامل أيضاً).
#     مكسبه الحقيقي: 168 ساعة من التفاصيل داخل كل نافذة، وأهداف لا تتداخل (32 ≥ 4).
#  2. أفق 4 ساعات: الحركة المتوقَّعة ≈ ضعف حركة ساعة واحدة (√4)، فتأكل التكلفة نسبة أقل من
#     الوقف — لكن الأفق الأطول لا يجعل الاتجاه أسهل تنبؤاً بذاته.
#  3. مع أفق > 1، هدف high/low يقارن **قمة 4 ساعات قادمة** بقمة الشمعة الأخيرة وحدها، فيميل
#     بشدّة لفئة واحدة (القمة القادمة أعلى في أغلب الحالات). دقة high/low_class العالية
#     عندها قد تكون مجرّد نسبة الفئة الأكبر — label_balance_report يُظهر خط الأساس الفعلي.
# ═══════════════════════════════════════════════════════════════════════════

HOURLY_PRESET = {
    "tf_order": ["1h"],
    "base_tf": "1h",
    "model_tf": "1h",
    "window_sizes": {"1h": 168},      # أسبوع كامل من الشموع
    "stride": 32,                     # نافذة كل 32 ساعة لكل عملة
    "forecast_horizon": 4,            # الهدف: إغلاق/قمة/قاع الساعات الأربع التالية
    "embargo_candles": None,          # None → 168 + 4 = 172 ساعة بين train/val/test
    # ⚠️ اسم مجلد ملفات الساعة على Drive (داخل MyDrive) — عدّله إن كان مختلفاً.
    #    load_asset يقبل عمود datetime_utc أو timestamp.
    "drive_raw_dir": "history_1h",
    "disk_backed": True,              # بناء مدعوم بالقرص: رام محدودة (CONFIG['disk_backed']) — الافتراضي لكل إعدادات 1h
}

#: اسم ملف الحفظ — مختلف عن الافتراضي كي لا يكتب فوق بيانات اليومي
#: (preprocessing_output_latest). main.ipynb يقرؤه عبر DATA_FILENAME_BASE.
HOURLY_DATA_NAME = "preprocessing_output_1h_w168_s32_h4"

# ── إعداد بيانات نموذج اللوحة على 1h: نافذة 32، stride 8، أفق شمعة واحدة، نفس 83 عملة ──────────
# لماذا هذا الإعداد: على بيانات 1h (نافذة 32، stride 32) كان A_ic وحده يحمل معلومة اتجاه ضعيفة لكن ثابتة عبر
# البذور، ولم يتشبّع تدريبه (docs/research/hourly_1h.md). stride 8 يعطي ~4× العيّنات. الأهداف تبقى غير متراكبة
# (8 ≥ أفق 1)؛ المتراكب هو المدخلات فقط (24 من 32 شمعة مشتركة بين عيّنتين متتاليتين لنفس العملة).
# فجوة العزل الافتراضية (embargo_candles=None ← 32 + 1 = 33 ساعة) زمنية لا بعدد العيّنات، فتكفي مع أي stride:
# أول نافذة val تبدأ بعد نهاية آخر هدف train (قيس: بهامش 8 ساعات على شبكة 8h؛ اختبار ذاتي في 19-ب).
# الذاكرة: X النهائي ≈ عيّنات × 32 × 23 × 4 بايت (~1.3 GB لـ ~450k). ذروة البناء ≈ 2.1× X (قوائم العملات + الدمج
# في _Accumulator.finalize؛ قيس على بيانات تركيبية)، أي ~3 GB، ثم في main: X المحمّل + نسخ التقسيم ≈ 2× X.
HOURLY_W32_S8_OVERRIDES = {
    "window_sizes": {"1h": 32},
    "stride": 8,
    "forecast_horizon": 1,
    "align_windows_to_grid": True,    # نهايات النوافذ عند 00/08/16 UTC لكل العملات ⇒ مجموعات مقطعية نقيّة
    "embargo_candles": None,          # None → 32 + 1 = 33 ساعة بين train/val/test
    "reg_target_scale": 100.0,        # عوائد 1h (انحراف ≈ 0.012) بنقاط مئوية — أرضيات NIG لا تهيمن (راجع CONFIG)
    # حدود التقسيم في الإعداد نفسه (تُحفظ في البيانات ويقرؤها main): حدود بيانات stride 32 نفسها فتبقى المقارنة ممكنة.
    "split_dates": {"train_end": "2025-06-24", "val_end": "2025-11-21"},
    # holdout مختوم (PROTOCOL.md): تاريخ ثابت لا «آخر 60 يوماً» — حدّ نسبي ينزلق مع كل إعادة جلب فتدخل أيام مختومة
    # في test. البيانات تنتهي 2026-09-26 ⇒ ~73 يوماً (~2.4 شهر) مختومة؛ test = 2025-11-22 ← 2026-07-13.
    "holdout_start": "2026-07-15",
    "disk_backed": True,              # صراحةً (يرثه HOURLY_4H_OVERRIDES): رام محدودة بغضّ النظر عن عدد العملات
}
#: مفاتيح المرحلة ٢ التي يتطلّبها build_hourly_w32_s8_dataset (43 ميزة): غيابها خطأ صريح لا رجوع صامت إلى 23 ميزة.
HOURLY_W32_S8_PHASE2 = ("use_intraday_15m", "use_futures_metrics")
HOURLY_W32_S8_NAME = "preprocessing_output_1h_w32_s8_h1"
#: عملات COINS_BY_CATEGORY كلها = نفس الـ83 في بيانات 1h السابقة (stride 32). رقم للتحقّق فقط لا للتصفية.
HOURLY_EXPECTED_COINS = 83


def apply_hourly_preset(overrides: Optional[dict] = None) -> Dict[str, Any]:
    """يطبّق HOURLY_PRESET (وأي تعديلات إضافية) على CONFIG الحيّ ويُعيد اشتقاق الميزات
    (ميزات الساعة TIME_hour_* تُضاف تلقائياً لأي فريم أصغر من يوم)."""
    update_config(deepcopy(HOURLY_PRESET))
    if overrides:
        update_config(deepcopy(overrides))
    refresh_features()
    return CONFIG


def hourly_coin_configs(registry: Optional[List[Dict]] = None, categories='all') -> List[Dict]:
    """عملات COINS_BY_CATEGORY فقط. ``'all'`` = كل فئات القاموس — وليس ALL_GLOBAL (السجل كاملاً).
    ⚠️ الخلية ٥٥ أعلاه تُسند DEFAULT_ENABLED_CATEGORIES=ALL_GLOBAL خارج التعليق، فالاستدعاء بلا
    categories صريحة يأخذ السجل كاملاً — لذلك تُمرَّر هنا صراحةً دائماً."""
    desired = flatten_coins_by_category(categories=categories)
    registry = load_asset_registry() if registry is None else registry
    return filter_desired_coins(registry, desired)


def x_memory_gb(n_samples: int, windows: List[int], n_features: int, dtype: str = "float32") -> float:
    """حجم X بالجيجابايت: ``n_samples`` عيّنة × مجموع نوافذ الفريمات ``windows`` × الميزات × بايتات الخانة (1e9 بايت = 1 GB)."""
    return n_samples * sum(int(w) for w in windows) * n_features * np.dtype(dtype).itemsize / 1e9


def estimate_hourly_feasibility(configs: List[Dict], load_asset_fn: Callable = None,
                                window: Optional[int] = None, stride: Optional[int] = None,
                                horizon: Optional[int] = None, warmup: int = 50,
                                config: Optional[dict] = None, verbose: bool = True) -> pd.DataFrame:
    """قبل البناء: يقرأ ملف كل عملة ويقدّر عدد العيّنات وحجم الذاكرة — بأرقام ملفاتك الفعلية.

    warmup: شموع أولى تسقط لأن المؤشرات تحتاج تاريخاً (أطول نافذة مؤشر = 50 في الإعداد الحالي).
    يتحقّق أيضاً أن الملفات فعلاً بفريم ساعة (وسيط الفرق بين الصفوف)."""
    config = CONFIG if config is None else config
    load_asset_fn = load_asset_fn or (lambda fid, name: load_asset(fid, name, config=config))
    window = window or config["window_sizes"][config["base_tf"]]
    stride = stride or config["stride"]
    horizon = horizon or config["forecast_horizon"]
    n_feat = len(feature_order(config))
    # ذاكرة X: نوافذ كل فريم في tf_order (الفريم الأساسي بـwindow المُمرَّر) × بايتات الخانة (float16 إن طُلب). فريم واحد
    # بـfloat32 ⇒ window × 4 كما كانت الصيغة حرفياً.
    x_windows = [window] + [int(config["window_sizes"][tf]) for tf in config["tf_order"][1:]]
    win_total = sum(x_windows)
    x_dtype = config.get("x_storage_dtype") or "float32"
    item = np.dtype(x_dtype).itemsize

    rows = []
    for cfg in configs:
        name = cfg["name"] if isinstance(cfg, dict) else str(cfg)
        try:
            df = load_asset_fn(cfg.get("file_id") if isinstance(cfg, dict) else None, name)
        except Exception as exc:                                   # noqa: BLE001
            rows.append({"asset": name, "error": str(exc)[:80]})
            continue
        idx = df.index
        step_h = float(pd.Series(idx).diff().median() / pd.Timedelta("1h")) if len(idx) > 1 else np.nan
        n = len(df)
        usable = n - warmup - window - horizon
        rows.append({"asset": name, "candles": n, "step_hours": step_h,
                     "start": idx.min(), "end": idx.max(),
                     "samples": max(0, usable // stride + 1) if usable >= 0 else 0,
                     "daily_samples_same_coin": max(0, n // 24 - 32 - 1)})
    table = pd.DataFrame(rows)
    ok = table[table["error"].isna()] if "error" in table else table
    if "samples" not in ok or ok.empty:
        # ✅ لا عملة قُرئت: اعرض السبب بدل الانهيار عند الجمع (كان KeyError: 'samples').
        print("═" * 70)
        print(f"🚨 تعذّرت قراءة كل العملات ({len(table)}) — لا تقدير ممكن.")
        root = mount_drive(config=config)
        raw_dir = config.get("drive_raw_dir")
        print(f"   المجلد المطلوب: {(root / raw_dir) if root is not None else raw_dir} "
              f"| النمط: {config.get('drive_raw_pattern', '{name}.csv')}")
        if root is not None and not (root / raw_dir).exists():
            print(f"   ❌ المجلد غير موجود — عدّل HOURLY_PRESET['drive_raw_dir'] (أو apply_hourly_preset"
                  f"({{'drive_raw_dir': 'اسم_مجلدك'}})) لاسم مجلد ملفات الساعة الفعلي داخل MyDrive.")
        if "error" in table:
            print("   أكثر الأسباب تكراراً:")
            for msg, cnt in table["error"].value_counts().head(5).items():
                print(f"     • ({cnt}×) {msg}")
        print("═" * 70)
        return table

    if verbose:
        total = int(ok["samples"].sum())
        daily = int(ok["daily_samples_same_coin"].sum())
        gb = x_memory_gb(total, x_windows, n_feat, x_dtype)
        print("═" * 70)
        print(f"📐 جدوى فريم الساعة: نافذة={window} | stride={stride} | أفق={horizon} | ميزات={n_feat}"
              + (f" | فريمات={config['tf_order']} (مجموع النوافذ {win_total}، {item * 8}-بت)"
                 if win_total != window or item != 4 else ""))
        print("═" * 70)
        print(f"  عملات مقروءة       : {len(ok)} من {len(configs)}")
        bad = ok[(ok["step_hours"] - 1.0).abs() > 0.01]
        if len(bad):
            print(f"  🚨 ملفات ليست بفريم ساعة (وسيط الفرق ≠ 1h): {bad['asset'].tolist()[:10]}"
                  " — تحقّق من drive_raw_dir")
        if "error" in table and table["error"].notna().any():
            print(f"  ⚠️ تعذّر قراءة: {table.loc[table['error'].notna(), 'asset'].tolist()}")
        print(f"  عيّنات متوقَّعة      : {total:,}  (≈ {total / max(len(ok), 1):,.0f} لكل عملة)")
        print(f"  للمقارنة — يومي stride=1 لنفس العملات: {daily:,} عيّنة "
              f"(نسبة الساعة/اليومي = {total / max(daily, 1):.2f})")
        if config.get("disk_backed"):
            print(f"  حجم X النهائي       : ≈ {gb:.1f} GB ({x_dtype}) على القرص — الرام أثناء البناء ≈ عملة واحدة "
                  f"(لا تتبع الحجم)، والقرص المحلي تحت scratch ≈ {gb:.1f} GB (+ {gb:.1f} GB لنسخة Drive المحلية في main)")
        else:
            print(f"  حجم X النهائي       : ≈ {gb:.1f} GB ({x_dtype}) — ذروة البناء ≈ 2.1× ({2.1 * gb:.1f} GB)، "
                  f"وفي main بعد التقسيم ≈ 2× ({2 * gb:.1f} GB) + TensorFlow")
        print(f"  خطوات/حقبة (batch 64، train 80%): ≈ {int(total * 0.8 / 64):,}")
        overlap = "لا تتداخل ✅" if stride >= horizon else f"تتداخل ({horizon - stride} شمعة مشتركة) ⚠️"
        print(f"  أهداف العيّنات المتتالية لنفس العملة: {overlap}")
        print(f"  نوافذ الإدخال المتتالية تشترك في {max(window - stride, 0)} من {window} شمعة")
        for s in (8, 16, 32):
            t = int(sum(max(0, (c - warmup - window - horizon) // s + 1) for c in ok["candles"]))
            print(f"     stride={s:<3} → {t:>10,} عيّنة ≈ {x_memory_gb(t, x_windows, n_feat, x_dtype):6.1f} GB")
        print("═" * 70)
    return table


def label_balance_report(dataset: Dict, verbose: bool = True) -> pd.DataFrame:
    """نسبة الصعود لكل رأس تصنيف = خط الأساس الذي يجب أن تتجاوزه الدقة (max(p, 1-p))."""
    rows = []
    for h in dataset["targets"]:
        if not h.endswith("_class"):
            continue
        y = np.asarray(dataset[f"y_{h}"]).ravel()
        p_up = float((y > 0).mean())
        rows.append({"head": h, "n": len(y), "p_up": p_up, "majority_baseline": max(p_up, 1 - p_up)})
    table = pd.DataFrame(rows)
    if verbose:
        print("\n⚖️ توازن فئات التصنيف (أي دقة ≤ majority_baseline لا تعني شيئاً):")
        with pd.option_context("display.float_format", "{:.3f}".format):
            print(table.to_string(index=False))
        skewed = table[(table["p_up"] - 0.5).abs() > 0.05]
        if len(skewed):
            print(f"   ⚠️ رؤوس غير متوازنة: {skewed['head'].tolist()} — قيّمها بـ AUC أو balanced accuracy")
    return table


def build_hourly_dataset(configs: List[Dict], save: bool = True, checkpoint_dir: Optional[str] = None,
                         max_workers: Optional[int] = None, filename_base: Optional[str] = None) -> Dict:
    """يبني بيانات الساعة بإعداد CONFIG الحالي (استدعِ apply_hourly_preset أولاً)، ثم يفحصها
    ويحفظها باسم filename_base (افتراضياً HOURLY_DATA_NAME). checkpoint_dir على Drive يحمي من انقطاع الجلسة."""
    assert CONFIG["base_tf"] == "1h", "استدعِ apply_hourly_preset() أولاً"
    dataset = build_dataset(configs, load_asset_fn=load_asset, resample_fn=make_resample_fn(CONFIG),
                            checkpoint_dir=checkpoint_dir, max_workers=max_workers, config=CONFIG)
    summarize_dataset(dataset)
    label_balance_report(dataset)
    if save:
        save_data_to_drive(dataset, filename_base=filename_base or HOURLY_DATA_NAME)
    return dataset


def _build_hourly_grid_dataset(overrides: dict, filename_base: str, checkpoint_dir: Optional[str] = None,
                               save: bool = True, max_workers: Optional[int] = None, estimate: bool = True,
                               require_phase2: bool = True) -> Dict:
    """الجسم المشترك لإعدادات الشبكة على 1h (HOURLY_W32_S8_OVERRIDES وامتداده 4h): ``overrides`` ← فحص بيانات المرحلة ٢ ←
    عملات COINS_BY_CATEGORY (83) ← تقدير الجدوى ← البناء ← فحص محاذاة الطوابع ← الحفظ باسم ``filename_base``.

    الفحص بعد البناء يطبع وسيط عدد العملات في الطابع الواحد: على الشبكة المشتركة يجب أن يقارب عدد العملات
    النشطة (لا 2 كما كان قبل align_windows_to_grid)."""
    apply_hourly_preset(overrides)
    if require_phase2:
        require_phase2_data(HOURLY_W32_S8_PHASE2, config=CONFIG)
        refresh_features()
        print(f"🧩 المرحلة ٢ مطلوبة ومتاحة: {len(CONFIG['feature_order'])} ميزة")
    configs = hourly_coin_configs()
    if len(configs) != HOURLY_EXPECTED_COINS:
        print(f"⚠️ {len(configs)} عملة بدل {HOURLY_EXPECTED_COINS} المتوقَّعة (بيانات 1h السابقة) — "
              f"النتائج لن تُقارَن بها مباشرة")
    if estimate:
        estimate_hourly_feasibility(configs)
    dataset = build_hourly_dataset(configs, save=False, checkpoint_dir=checkpoint_dir, max_workers=max_workers)
    ts = np.asarray(dataset["last_candles"])[:, TS_COL].astype("int64")
    step_ns = CONFIG["stride"] * int(pd.Timedelta("1h").value)
    on_grid = float(np.mean(ts % step_ns == 0))
    per_ts = pd.Series(ts).value_counts()
    shapes = " | ".join(f"X_{tf} {dataset[f'X_{tf}'].shape} ≈ {dataset[f'X_{tf}'].nbytes / 1e9:.2f} GB"
                        for tf in dataset["timeframes"])
    print(f"🧭 طوابع على شبكة {CONFIG['stride']}h: {on_grid:.2%} | طوابع فريدة {len(per_ts):,} | "
          f"وسيط العملات في الطابع {per_ts.median():.0f} (الأقصى {per_ts.max()}) | {shapes}")
    if on_grid < 1.0:
        print("   ⚠️ طوابع خارج الشبكة: شموع مُزاحة عن رأس الساعة في ملفات بعض العملات")
    if save:
        save_data_to_drive(dataset, filename_base=filename_base)
    return dataset


def build_hourly_w32_s8_dataset(checkpoint_dir: Optional[str] = None, save: bool = True,
                                max_workers: Optional[int] = None, estimate: bool = True,
                                require_phase2: bool = True) -> Dict:
    """بيانات اللوحة على 1h باستدعاء واحد: HOURLY_W32_S8_OVERRIDES ← فحص بيانات المرحلة ٢ ← عملات
    COINS_BY_CATEGORY (83) ← تقدير الجدوى ← البناء ← فحص محاذاة الطوابع ← الحفظ باسم HOURLY_W32_S8_NAME.

    ``require_phase2`` (افتراضياً True): مجلدات 15m/funding/OI/metrics والوحدة tools/intraday_features.py إلزامية —
    غيابها يرفع ``RuntimeError`` قبل أي تحميل (require_phase2_data) بدل بناء صامت بـ23 ميزة. False = السلوك 'auto'.
    الجسم في :func:`_build_hourly_grid_dataset` (مشترك مع إعداد 1h+4h)."""
    return _build_hourly_grid_dataset(HOURLY_W32_S8_OVERRIDES, HOURLY_W32_S8_NAME, checkpoint_dir, save,
                                      max_workers, estimate, require_phase2)


# ── الاستخدام (أزل التعليق وشغّل بالترتيب) ─────────────────────────────────
# apply_hourly_preset()
# configs = hourly_coin_configs()                  # كل فئات COINS_BY_CATEGORY فقط
# feas = estimate_hourly_feasibility(configs)      # راجع الأرقام قبل البناء
# dataset = build_hourly_dataset(configs, checkpoint_dir="/content/drive/MyDrive/crypto_model/ckpt_1h")
# ثم في main.ipynb: DATA_FILENAME_BASE = HOURLY_DATA_NAME
#
# ── بيانات اللوحة على 1h (نافذة 32، stride 8، أفق 1، 83 عملة) — سطر واحد يكفي ──
# dataset = build_hourly_w32_s8_dataset(checkpoint_dir="/content/drive/MyDrive/crypto_model/ckpt_1h_w32_s8")
# ثم في main.ipynb: DATA_FILENAME_BASE = "preprocessing_output_1h_w32_s8_h1" (docs/research/hourly_1h.md «تشغيل Colab»)
#
# ── بيانات 1h + سياق 4h: الخلية 20-ج التالية (build_hourly_4h_dataset) ──


# @title 20-ج) بيانات 1h + سياق 4h: نافذة 32 لكل فريم، stride 8، أفق 1 — عملات COINS_BY_CATEGORY
# ═══════════════════════════════════════════════════════════════════════════
# تعريفات فقط (لا تنفيذ تلقائي — آمن مع %run من main). الاستخدام في آخر الخلية.
# لماذا: بيانات 1h_s8 تعطي النموذج 32 ساعة فقط؛ سياق 4h يضيف ≈ 5 أيام (نظام التقلّب والاتجاه الأبطأ) دون أن يتغيّر ما
# يُتنبَّأ به: الأهداف والشبكة والتقسيم وholdout وreg_target_scale كلها من 1h_s8 حرفياً (الإعداد يرث HOURLY_W32_S8_OVERRIDES).
# ذاكرة: ~450k عيّنة × (32×43 + 32×43) × 4 بايت = 4.95 GB (float32)، ذروة البناء 2.1× = 10.4 GB > ~10 GB المتاحة عملياً على
# Colab العادية (12.7 GB)، ثم في main نسخ التقسيم ≈ 2× X. التخفيف المختار: float16 لـ X (نصف الذاكرة، خطأ ≤ 2.5e-3 على قيم مقصوصة
# ±5) مع رفع إلى float32 عند تكوين الدفعة — يُبقي stride 8 ونافذة 4h كما هما بدل تقليل العيّنات (stride 16) أو السياق.
# ═══════════════════════════════════════════════════════════════════════════

HOURLY_4H_OVERRIDES = {
    **HOURLY_W32_S8_OVERRIDES,        # نافذة 1h=32، stride 8، أفق 1، شبكة، reg_target_scale، split_dates، holdout_start
    "tf_order": ["1h", "4h"],
    "base_tf": "1h",
    "window_sizes": {"1h": 32, "4h": 32},   # 4h × 32 = 128 ساعة ≈ 5.3 يوماً
    "higher_tf_mode": "closed",       # شموع 4h المغلقة عند t فقط + تطبيع أسعار كل فريم بنافذته (انظر رأس CONFIG)
    "x_storage_dtype": "float16",     # نصف ذاكرة X؛ تُرفَع إلى float32 عند الدفعة
}
HOURLY_4H_NAME = "preprocessing_output_1h_4h_w32_s8_h1"


def build_hourly_4h_dataset(checkpoint_dir: Optional[str] = None, save: bool = True,
                            max_workers: Optional[int] = None, estimate: bool = True,
                            require_phase2: bool = True) -> Dict:
    """بيانات 1h + سياق 4h باستدعاء واحد (نفس خطوات :func:`build_hourly_w32_s8_dataset`) مع HOURLY_4H_OVERRIDES، ثم فحص أن
    البيانات المبنيّة بوضع 'closed' وفريميها وdtype المتوقَّع قبل الحفظ باسم HOURLY_4H_NAME."""
    dataset = _build_hourly_grid_dataset(HOURLY_4H_OVERRIDES, HOURLY_4H_NAME, checkpoint_dir, False,
                                         max_workers, estimate, require_phase2)
    assert dataset["timeframes"] == ["1h", "4h"] and dataset.get("higher_tf_mode") == "closed", \
        f"بيانات لا تطابق الإعداد: {dataset['timeframes']}, {dataset.get('higher_tf_mode')}"
    assert all(dataset[f"X_{tf}"].dtype == np.float16 for tf in dataset["timeframes"]), "X ليست float16"
    total_gb = sum(dataset[f"X_{tf}"].nbytes for tf in dataset["timeframes"]) / 1e9
    print(f"🕓 1h+4h: {len(dataset['base_params']):,} عيّنة | X الكلي ≈ {total_gb:.2f} GB (float16) | "
          f"holdout {dataset.get('holdout_start')} | reg_target_scale {dataset.get('reg_target_scale')}")
    if save:
        save_data_to_drive(dataset, filename_base=HOURLY_4H_NAME)
    return dataset


# ── الاستخدام ──
# dataset = build_hourly_4h_dataset(checkpoint_dir="/content/drive/MyDrive/crypto_model/ckpt_1h_4h_w32_s8")
# ثم في main.ipynb: DATA_FILENAME_BASE = HOURLY_4H_NAME ؛ MODEL_TFS = ["1h", "4h"]


# @title 20-ج) اختبار ذاتي لعقد إعداد 1h+4h
def run_hourly_4h_selftests(verbose: bool = True) -> bool:
    cfg = deepcopy(CONFIG)
    _deep_update(cfg, deepcopy(HOURLY_PRESET))
    _deep_update(cfg, deepcopy(HOURLY_4H_OVERRIDES))          # نسخة محلية: CONFIG الحيّ لا يُمَسّ

    def t_inherits_1h_s8_and_changes_only_tf_mode_dtype():
        base = deepcopy(CONFIG)
        _deep_update(base, deepcopy(HOURLY_PRESET))
        _deep_update(base, deepcopy(HOURLY_W32_S8_OVERRIDES))
        for k in ("stride", "forecast_horizon", "align_windows_to_grid", "reg_target_scale", "split_dates",
                  "holdout_start", "embargo_candles", "base_tf", "drive_raw_dir"):
            assert cfg[k] == base[k], (k, cfg[k], base[k])
        assert (cfg["stride"], cfg["forecast_horizon"], cfg["reg_target_scale"], cfg["holdout_start"]) == \
            (8, 1, 100.0, "2026-07-15")
        assert cfg["tf_order"] == ["1h", "4h"] and cfg["base_tf"] == "1h"
        assert cfg["window_sizes"]["1h"] == cfg["window_sizes"]["4h"] == 32
        assert cfg["higher_tf_mode"] == "closed" and cfg["x_storage_dtype"] == "float16"
        assert _higher_tf_closed(cfg)
        # فجوة العزل تغطّي سياق 4h (32 × 4h = 128 ساعة) + الأفق = 129 ساعة (لا 33 كما في 1h_s8): وإلا رأى مُدخل val أهدافاً من train
        ds = {"base_tf": "1h", "timeframes": cfg["tf_order"], "window_sizes": cfg["window_sizes"],
              "forecast_horizon": cfg["forecast_horizon"], "higher_tf_mode": cfg["higher_tf_mode"]}
        assert embargo_candles(ds, cfg) == 129 and embargo_candles({**ds, "timeframes": ["1h"]}, cfg) == 33
        # الإعدادات القائمة لم تُلمَس: 1h_s8 فريم واحد بلا الوضعين
        assert "tf_order" not in HOURLY_W32_S8_OVERRIDES and "higher_tf_mode" not in HOURLY_W32_S8_OVERRIDES
        assert HOURLY_W32_S8_OVERRIDES["window_sizes"] == {"1h": 32}
        assert DEFAULT_CONFIG["higher_tf_mode"] == "legacy" and DEFAULT_CONFIG["x_storage_dtype"] is None

    def t_memory_estimate_matches_documented_numbers():
        # ~450k عيّنة، فريمان بنافذة 32 و43 ميزة: 4.95 GB بـfloat32 (ذروة 2.1× = 10.4 GB)، ونصفها بـfloat16
        f32 = x_memory_gb(450_000, [32, 32], 43, "float32")
        f16 = x_memory_gb(450_000, [32, 32], 43, "float16")
        assert abs(f32 - 4.9536) < 1e-3 and abs(f16 - f32 / 2) < 1e-9, (f32, f16)
        assert 2.1 * f32 > 10.0 > 2.1 * f16                     # سبب اختيار float16: الذروة تتجاوز ~10 GB بدونه
        assert abs(x_memory_gb(450_000, [32], 23, "float32") - 1.3248) < 1e-3   # رقم 1h_s8 الموثَّق (1.3 GB)

    tests = [t_inherits_1h_s8_and_changes_only_tf_mode_dtype, t_memory_estimate_matches_documented_numbers]
    for t in tests:
        t()
        if verbose:
            print(f"  ✅ {t.__name__}")
    return True


# تُشغَّل تلقائياً كبقية الاختبارات الذاتية؛ أدوات التدقيق (docs/research/audit/_nbload.py) تُعطّلها بهذا العَلَم.
if globals().get("RUN_HOURLY_4H_SELFTESTS", True):
    run_hourly_4h_selftests()


# @title 20-د) تجربة التطبيع النسبي الهندسي: بيانات 1h_s8 بأسعار = تغيّر % عن الشمعة السابقة
# ═══════════════════════════════════════════════════════════════════════════
# تعريفات فقط (لا تنفيذ تلقائي — آمن مع %run من main). الاستخدام في آخر الخلية.
# التجربة: تغيير واحد فقط عن 1h_s8 — أعمدة price_level (close/high/low/open/المتوسطات/النطاقات) تُطبَّع كتغيّر نسبي هندسي
# عن الشمعة السابقة لنفس العمود بالنقاط المئوية (−1، +5، −0.4) بدل (x − وسيط النافذة)/IQR. العملات والنافذة والـstride والأفق
# والتقسيم والـholdout وreg_target_scale (=100: الأهداف عوائد بالنقاط المئوية أيضاً، فالمدخل والهدف بوحدة واحدة) كلها من
# HOURLY_W32_S8_OVERRIDES حرفياً، فتُقارَن النتائج ببيانات 1h_s8 مباشرة.
# ═══════════════════════════════════════════════════════════════════════════

PCT_CHANGE_OVERRIDES = {
    **HOURLY_W32_S8_OVERRIDES,        # نافذة 32، stride 8، أفق 1، شبكة، reg_target_scale، split_dates، holdout_start
    "price_norm_mode": "pct_change",  # 100 × (x_t / x_{t−1} − 1) لأعمدة price_level؛ بقية الأنواع كما هي
    "price_pct_clip": 100.0,          # قصّ ±100 نقطة مئوية: لا يمسّ حركة ساعة حقيقية فيبقى الفكّ دقيقاً
}
HOURLY_PCT_NAME = "preprocessing_output_1h_w32_s8_h1_pct"


def pct_decode_consistency(dataset: Dict, column: str = 'close') -> Dict[str, float]:
    """فحص فكّ حقيقي بلا أسعار خام: لعيّنتين متتاليتين لنفس العملة بينهما ``stride`` شمعة بالضبط، السعر المفكوك من نافذة
    الثانية (مرساتها ``last_<column>`` الخاص بها) عند الموضع ``T − 1 − stride`` يجب أن يساوي ``last_<column>`` للأولى — مرساة
    مستقلّة لم تدخل الفكّ. يُرجع عدد الأزواج وأقصى/p99 الخطأ النسبي ونسبة القيم المقصوصة (غير قابلة للفكّ الدقيق)."""
    tf = dataset['base_tf']
    T, s = int(dataset['window_sizes'][tf]), int(dataset['stride'])
    j = list(dataset['feature_order']).index(column)
    r = np.asarray(dataset[f'X_{tf}'][:, :, j], dtype='float64')
    clipped = float(np.mean(np.abs(r) >= float(dataset.get('price_pct_clip', PRICE_PCT_CLIP))))
    if s >= T:
        return {'pairs': 0, 'max_rel_err': float('nan'), 'p99_rel_err': float('nan'), 'clipped_frac': clipped}
    decoded = decode_price_window(dataset, column)
    lc = np.asarray(dataset['last_candles'])
    ts = lc[:, TS_COL].astype('int64')
    last = lc[:, LAST_COLUMNS.index(f'last_{column}')].astype('float64')
    step = s * int(pd.Timedelta(str(tf)).value)
    errs = []
    for b in dataset['asset_bounds']:
        i = b['start'] + np.flatnonzero(np.diff(ts[b['start']:b['end']]) == step)
        errs.append(np.abs(decoded[i + 1, T - 1 - s] / last[i] - 1.0))
    e = np.concatenate(errs) if errs else np.empty(0)
    return {'pairs': int(e.size), 'max_rel_err': float(e.max()) if e.size else float('nan'),
            'p99_rel_err': float(np.percentile(e, 99)) if e.size else float('nan'), 'clipped_frac': clipped}


def build_hourly_pct_dataset(checkpoint_dir: Optional[str] = None, save: bool = True,
                             max_workers: Optional[int] = None, estimate: bool = True,
                             require_phase2: bool = True) -> Dict:
    """بيانات تجربة 'pct_change' باستدعاء واحد (نفس خطوات :func:`build_hourly_w32_s8_dataset`) مع PCT_CHANGE_OVERRIDES، ثم
    تدقيق التطبيع وفحص الفكّ (:func:`pct_decode_consistency`) قبل الحفظ باسم HOURLY_PCT_NAME. ``checkpoint_dir`` مجلد مستقلّ
    عن 1h_s8 (البصمة تختلف على أي حال فلا تُخلط نقاط الاستئناف)."""
    dataset = _build_hourly_grid_dataset(PCT_CHANGE_OVERRIDES, HOURLY_PCT_NAME, checkpoint_dir, False,
                                         max_workers, estimate, require_phase2)
    assert dataset.get("price_norm_mode") == "pct_change", f"بيانات لا تطابق الإعداد: {dataset.get('price_norm_mode')}"
    X = dataset[f"X_{dataset['base_tf']}"]
    audit_normalization(X[_pct_sample_rows(len(X))], dataset["feature_order"], config=CONFIG)
    chk = pct_decode_consistency(dataset)
    print(f"🔁 فكّ close: {chk['pairs']:,} زوج | أقصى خطأ نسبي {chk['max_rel_err']:.2e} (p99 {chk['p99_rel_err']:.2e}) | "
          f"قيم مقصوصة {chk['clipped_frac']:.2e}")
    if save:
        save_data_to_drive(dataset, filename_base=HOURLY_PCT_NAME)
    return dataset


def _pct_sample_rows(n: int, max_n: int = 20000) -> np.ndarray:
    """عيّنة صفوف مرتّبة ثابتة البذرة — التدقيق على كل X (مئات آلاف النوافذ) يستهلك رام بلا داعٍ."""
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(0).choice(n, max_n, replace=False))


# ── الاستخدام ──
# dataset = build_hourly_pct_dataset(checkpoint_dir="/content/drive/MyDrive/crypto_model/ckpt_1h_w32_s8_pct")
# ثم في main.ipynb: DATA_FILENAME_BASE = HOURLY_PCT_NAME   (main يقرأ price_norm_mode من البيانات نفسها)
# فكّ نافذة الإغلاق إلى أسعار حقيقية: prices = decode_price_window(dataset, "close")      # (N, 32)

"""
PURPOSE:  Pre-training diagnostics: missing values, feature availability, registry-vs-data name mismatches, summarize_dataset.
TAGS:     diagnostics, summarize_dataset, check_missing_values, diagnose_feature_availability, diagnose_data_vs_configs
PITFALLS: Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 12) أدوات التشخيص (`diagnostics.py` سابقاً)
"""
# @title
"""
أدوات تشخيص البيانات قبل التدريب.

تكشف الأعطال الصامتة مبكراً: أعمدة ميزات مفقودة، أسماء أصول غير متطابقة بين
سجل الأصول والبيانات المُحمَّلة، وقيم مفقودة داخل الفريمات.
"""

def check_missing_values(data: Dict[str, Dict], tf_order: Optional[List[str]] = None,
                         config: Optional[dict] = None) -> Dict[str, dict]:
    """يبحث عن قيم مفقودة داخل كل (أصل × فريم) ويطبع تقريراً."""
    config = CONFIG if config is None else config
    tf_order = config["tf_order"] if tf_order is None else tf_order
    problems: Dict[str, dict] = {}

    for asset_name, tf_dict in data.items():
        if not isinstance(tf_dict, dict):
            continue
        for tf, df in tf_dict.items():
            if tf_order and tf not in tf_order:
                continue
            missing = df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing):
                problems[f"{asset_name}/{tf}"] = dict(missing)
                print(f"⚠️ {asset_name}/{tf}: أعمدة فيها قيم مفقودة → {dict(missing)}")

    if not problems:
        print("✅ لا توجد قيم مفقودة في البيانات المُحمَّلة")
    return problems


def diagnose_feature_availability(data: Dict[str, Dict],
                                  tf_order: Optional[List[str]] = None,
                                  desired_features: Optional[List[str]] = None,
                                  config: Optional[dict] = None) -> List[str]:
    """يتحقق من توفر كل عمود في ``feature_order`` داخل البيانات الفعلية.

    يُرجع قائمة الأعمدة المشتركة فعلياً بين كل الأصول/الفريمات، لضمان شكل
    مصفوفات موحّد عند الدمج. أسنِد الناتج إلى ``CONFIG['feature_order']``.
    """
    config = CONFIG if config is None else config
    tf_order = config["tf_order"] if tf_order is None else tf_order
    if desired_features is None:
        desired_features = feature_order(config)

    common = set(desired_features)
    per_asset_missing: Dict[str, set] = {}
    checked = 0

    for asset, tf_dict in data.items():
        if not isinstance(tf_dict, dict):
            continue
        for tf in tf_order:
            if tf not in tf_dict:
                continue
            checked += 1
            cols = set(tf_dict[tf].columns)
            missing = set(desired_features) - cols
            if missing:
                per_asset_missing[f"{asset}/{tf}"] = missing
            common &= cols

    missing_from_all = set(desired_features) - common
    print(f"\n🔎 فحص توفر الميزات: تحقّقت من {checked} (أصل × فريم)")
    if missing_from_all:
        print(f"⚠️ {len(missing_from_all)} عمود من أصل {len(desired_features)} غير متوفر "
              f"إطلاقاً في البيانات (سيُستبعد تلقائياً):")
        print("   ", sorted(missing_from_all))
        print("   → على الأرجح: البيانات المخزَّنة أُنشئت بإعدادات add_features أو "
              "بنسخة pandas_ta مختلفة عن الحالية.")

    inconsistent = {k: v for k, v in per_asset_missing.items() if v - missing_from_all}
    if inconsistent:
        print(f"🚨 تحذير أخطر: {len(inconsistent)} (أصل/فريم) ناقصة أعمدة *إضافية* غير "
              f"مفقودة من البقية — سيُنتج شكل بيانات غير متّسق عند الدمج!")
        for k, v in list(inconsistent.items())[:5]:
            print(f"   {k}: مفقود فقط هنا → {sorted(v - missing_from_all)}")

    available = [c for c in desired_features if c in common]
    print(f"✅ سيُستخدَم فعلياً {len(available)} عمود من أصل {len(desired_features)}\n")
    return available


def _strip_quote_suffix(s) -> str:
    s = str(s).strip().upper()
    for suf in ("USDT", "USDC", "BUSD", "-USDT", "/USDT", "_USDT"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def diagnose_data_vs_configs(data: dict, configs: list, n_preview: int = 15) -> dict:
    """مقارنة أسماء الأصول بين ``configs`` و``data`` بعدة استراتيجيات مطابقة.

    السبب الأشيع لخروج مجموعة بيانات فارغة هو اختلاف التسمية (حالة أحرف،
    مسافات، لاحقة USDT). هذه الدالة تكشفه فوراً.
    """
    print("=" * 90)
    print("1) بنية data")
    print("=" * 90)
    print(f"النوع: {type(data).__name__} | عدد المفاتيح: {len(data)}")

    data_keys = list(data.keys())
    print(f"\nأول {n_preview} مفتاحاً (repr لكشف المسافات والرموز المخفية):")
    for k in data_keys[:n_preview]:
        print(f"   {repr(k)}")

    if data_keys:
        k0 = data_keys[0]
        v0 = data[k0]
        print(f"\nنوع القيمة لأول مفتاح {repr(k0)}: {type(v0).__name__}")
        if isinstance(v0, dict):
            print(f"مفاتيح المستوى الثاني (الفريمات): {list(v0.keys())}")
            if v0:
                k1 = list(v0.keys())[0]
                v1 = v0[k1]
                if hasattr(v1, "shape"):
                    print(f"شكل الفريم {repr(k1)}: {v1.shape}")
                if hasattr(v1, "columns"):
                    print(f"أول 10 أعمدة: {list(v1.columns)[:10]}")
        else:
            print("⚠️ القيمة ليست dict — هل البنية data[tf][asset] بدل data[asset][tf]؟")
            print(f"   محتوى مختصر: {str(v0)[:300]}")

    print("\n" + "=" * 90)
    print("2) أسماء configs")
    print("=" * 90)
    print(f"عدد configs: {len(configs)}")
    config_names = [c.get("name") if isinstance(c, dict) else c for c in configs]
    for n in config_names[:n_preview]:
        print(f"   {repr(n)}")

    print("\n" + "=" * 90)
    print("3) استراتيجيات المطابقة")
    print("=" * 90)
    cfg_set = {n for n in config_names if n is not None}
    data_set = set(data_keys)

    exact = cfg_set & data_set
    print(f"✅ تطابق تام: {len(exact)} / {len(cfg_set)}")

    cfg_stripped = {str(n).strip(): n for n in cfg_set}
    data_stripped = {str(k).strip(): k for k in data_keys}
    stripped_match = set(cfg_stripped) & set(data_stripped)
    print(f"🔤 بعد strip(): {len(stripped_match)} / {len(cfg_stripped)}")

    cfg_lower = {str(n).strip().lower(): n for n in cfg_set}
    data_lower = {str(k).strip().lower(): k for k in data_keys}
    lower_match = set(cfg_lower) & set(data_lower)
    print(f"🔡 بتجاهل حالة الأحرف: {len(lower_match)} / {len(cfg_lower)}")

    cfg_base = {_strip_quote_suffix(n): n for n in cfg_set}
    data_base = {_strip_quote_suffix(k): k for k in data_keys}
    base_match = set(cfg_base) & set(data_base)
    print(f"💱 بعد إزالة لاحقة USDT/USDC/BUSD: {len(base_match)} / {len(cfg_base)}")
    if base_match and not exact:
        print("   أمثلة:")
        for b in list(base_match)[:8]:
            print(f"      config: {repr(cfg_base[b])}  ↔  data: {repr(data_base[b])}")

    if not exact and not base_match:
        print("\n   ⚠️ لا يوجد تطابق يُذكر — راجع مصدر data أو بنيتها (القسم 1).")

    return {
        "data_keys": data_keys,
        "config_names": config_names,
        "exact_match": exact,
        "stripped_match": stripped_match,
        "lower_match": lower_match,
        "base_match": base_match,
    }


def summarize_dataset(dataset: dict) -> dict:
    """ملخّص رقمي لمجموعة بيانات مُجهَّزة — يُستخدم كفحص أخير قبل الحفظ."""
    print("═" * 66)
    print("📦 ملخّص مجموعة البيانات")
    print("═" * 66)
    n = len(dataset['base_params'])
    print(f"  العيّنات      : {n:,}")
    print(f"  الفريمات      : {dataset['timeframes']}")
    print(f"  الرؤوس        : {dataset['targets']}")
    print(f"  الميزات       : {len(dataset['feature_order'])}")
    print(f"  عدد الأصول    : {len(dataset.get('asset_bounds', []))}")

    stats = {}
    for tf in dataset['timeframes']:
        X = dataset[f'X_{tf}']
        nan, inf = _count_nonfinite(X)        # بدفعات: بلا مصفوفة منطقية مؤقتة بحجم X (memmap بجيجابايتات)
        stats[f'X_{tf}'] = {'shape': X.shape, 'nan': nan, 'inf': inf}
        flag = "✅" if (nan == 0 and inf == 0) else "🚨"
        print(f"  {flag} X_{tf}: {X.shape} | NaN={nan} | Inf={inf}")

    for h in dataset['targets']:
        y = dataset[f'y_{h}']
        uniq = np.unique(y)
        if len(uniq) <= 3:
            counts = {float(u): int((y == u).sum()) for u in uniq}
            balance = ", ".join(f"{k:g}: {v / len(y):.1%}" for k, v in counts.items())
            print(f"  📊 y_{h}: توازن الفئات → {balance}")
            stats[f'y_{h}'] = counts
        else:
            print(f"  📊 y_{h}: متوسط={y.mean():+.4f} | انحراف={y.std():.4f} "
                  f"| المدى=[{y.min():+.3f}, {y.max():+.3f}]")
            stats[f'y_{h}'] = {'mean': float(y.mean()), 'std': float(y.std())}
    print("═" * 66)
    return stats

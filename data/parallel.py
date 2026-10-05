"""
PURPOSE:  Thread-parallel asset loading with preserved order: imap_ordered, load_assets, default_workers (capped by CPU and system RAM).
TAGS:     parallel, threads, imap_ordered, load_assets, default_workers, max_workers, RAM cap
PITFALLS: Results come back in input order whatever the completion order, so asset_bounds stay reproducible. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 13) التحميل والمعالجة المتوازية (`parallel.py` سابقاً)
"""
# @title
"""
تحميل ومعالجة متوازية للأصول.

**لماذا يفيد التوازي هنا؟** المسار التسلسلي يقضي أغلب وقته منتظراً الشبكة:
كل عملة تعني طلب ``read_csv`` مستقلاً من Google Drive. مع 77 عملة يصبح ذلك
77 انتظاراً متتابعاً. والخيوط (threads) كافية تماماً — لا حاجة لعمليات
منفصلة — لأن كلاً من انتظار الشبكة، وتحليل CSV في pandas، وحسابات numpy
داخل ``prepare_single_asset`` **تُحرِّر قفل المفسّر (GIL)** أثناء عملها.

**الترتيب محفوظ.** :func:`imap_ordered` تُرجع النتائج بترتيب الإدخال مهما
اختلف ترتيب اكتمالها، فيبقى ``asset_bounds`` وترتيب الصفوف في مجموعة البيانات
**قابلاً لإعادة الإنتاج بالضبط** — وهذا شرط لمقارنة تجربتين بأمان.

**الذاكرة مضبوطة.** ``prefetch`` يحدّ عدد المهام الطائرة في وقت واحد، فلا
تُحمَّل كل العملات دفعةً في الذاكرة إن لم تُرِد ذلك (مهم على Colab).
"""

def _system_ram_mb() -> Optional[float]:
    """رام النظام الكلي بالميجابايت من ``/proc/meminfo`` (لينكس — بيئة Colab
    القياسية). ``None`` إن تعذّرت القراءة (نظام غير لينكس مثلاً) — لا يُعطَّل
    شيء عندها، فقط يُفقَد هذا السقف الاحترازي الإضافي."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:                                          # noqa: BLE001
        pass
    return None


def default_workers(n_items: int = 0) -> int:
    """عدد خيوط افتراضي معقول: مقيَّد بالمهام المتاحة، وعدد الأنوية، ورام
    النظام الكلي.

    ✅ سقف رام إضافي (جيجا واحد متاح لكل خيط تقريباً، تقدير محافظ): على آلة
    بأنوية كثيرة لكن رام محدودة (شائع في Colab المجانية، ~12 جيجا) كان عدد
    الخيوط يُشتقّ من الأنوية وحدها فيُفرط في التوازي، وكل خيط يحمل ذروته
    الخاصة من `align_multi_timeframes_time_based`/`process_windows` —
    تراكمها معاً هو ما سبّب استهلاك رام أكبر من حجم البيانات النهائي بأضعاف.
    هذا السقف احترازي إضافي فوق إصلاح الذروة نفسها في الدالتين أعلاه، لا بديل
    عنه (لا يعرف حجم كل أصل مسبقاً، فهو تقدير خشن لا مضبوط بدقة).
    """
    cpu = os.cpu_count() or 4
    workers = min(100, max(4, cpu * 2))
    ram_mb = _system_ram_mb()
    if ram_mb is not None:
        ram_cap = max(2, int(ram_mb // 1024))
        workers = min(workers, ram_cap)
    return max(1, min(workers, n_items)) if n_items else workers


def imap_ordered(fn: Callable[[T], R],
                 items: Iterable[T],
                 max_workers: int = 8,
                 prefetch: Optional[int] = None) -> Iterator[R]:
    """``map`` متوازٍ يُرجع النتائج **بترتيب الإدخال** مع تقييد المهام الطائرة.

    Args:
        max_workers: عدد الخيوط. ``<= 1`` ينفّذ تسلسلياً بلا أي تجمّع خيوط.
        prefetch: أقصى عدد مهام طائرة معاً (افتراضياً ``max_workers * 2``).
            يحدّ الذاكرة: لا تُنجَز نتائج أكثر من هذا قبل استهلاكها.
    """
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        for it in items:
            yield fn(it)
        return

    prefetch = prefetch or max_workers * 2
    prefetch = max(prefetch, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix="cm_asset") as pool:
        pending: Dict[int, object] = {}
        next_in = 0
        next_out = 0

        while next_out < len(items):
            # املأ نافذة المهام الطائرة
            while next_in < len(items) and len(pending) < prefetch:
                pending[next_in] = pool.submit(fn, items[next_in])
                next_in += 1

            # سلّم كل ما جهز **بالترتيب** من مقدّمة الطابور
            if next_out in pending and pending[next_out].done():
                yield pending.pop(next_out).result()
                next_out += 1
                continue

            # انتظر اكتمال أي مهمة ثم أعد المحاولة
            wait([pending[next_out]] if next_out in pending
                 else list(pending.values()),
                 return_when=FIRST_COMPLETED)
            if next_out in pending:
                yield pending.pop(next_out).result()
                next_out += 1


def load_assets(configs: List[Dict],
                load_asset_fn: Callable,
                max_workers: Optional[int] = None,
                prefetch: Optional[int] = None,
                tail: int = 0,
                verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """يُحمّل **كل** العملات دفعةً واحدة بالتوازي ويُرجع ``{name: DataFrame}``.

    مفيد حين تريد البيانات الخام في الذاكرة لفحصها أو لإعادة استخدامها في عدة
    تجارب بلا إعادة تنزيل. للبناء المباشر لمجموعة بيانات فضّل
    :func:`build_dataset_from_loader` بمعامل ``max_workers`` — فهي تُعالج
    أثناء التنزيل ولا تحتفظ بكل شيء في الذاكرة.

    الأصول التي يفشل تحميلها تُتخطّى مع تقرير، ولا تُوقف البقية.
    """
    max_workers =    max_workers or default_workers(len(configs))

    def _one(cfg: Dict) -> Tuple[str, Optional[pd.DataFrame], str]:
        name = cfg['name'] if isinstance(cfg, dict) else str(cfg)
        try:
            df = load_asset_fn(cfg.get('file_id'), name)
            if tail:
                df = df[-tail:]
            return name, df, ''
        except Exception as exc:                       # noqa: BLE001
            return name, None, str(exc)

    out: Dict[str, pd.DataFrame] = {}
    failed: List[Tuple[str, str]] = []

    if verbose:
        print(f"⏬ تحميل {len(configs)} عملة بـ {max_workers} خيطاً...")

    for i, (name, df, err) in enumerate(
            imap_ordered(_one, configs, max_workers, prefetch), start=1):
        if df is None:
            failed.append((name, err))
            if verbose:
                print(f"   [{i}/{len(configs)}] ⚠️ {name}: {err[:70]}")
            continue
        out[name] = df
        if verbose:
            print(f"   [{i}/{len(configs)}] ✅ {name}: {len(df):,} صف")

    if verbose:
        print(f"✅ حُمّلت {len(out)} من {len(configs)} عملة")
        if failed:
            print(f"   ⚠️ فشل {len(failed)}: {[n for n, _ in failed][:8]}")
    return out

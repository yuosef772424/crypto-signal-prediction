"""
PURPOSE:  Output-head helpers: {target}_class / {target}_reg names, enabled heads, abstention/uncertainty switches, head config validation and loss weights.
TAGS:     heads, get_target_heads, enabled_heads, class head, reg head, output names, abstention, loss weights, validate_head_config
PITFALLS: enabled_heads is a dict {head: bool}, not a list of names. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 6) أدوات رؤوس المخرجات (`heads.py` سابقاً)

كل هدف سعري (high / low / close) له رأسان: `{target}_class` (تصنيف اتجاه 1/0)
و`{target}_reg` (قيمة انحدار). كل مراحل النظام تستدعي `get_target_heads` بدل
تثبيت الأسماء يدوياً، فإطفاء أي رأس في `CONFIG['enabled_heads']` يسري تلقائياً
على النظام كله — **هذا يستبدل** `get_target_heads` المبنية تخمينياً في النسخة
السابقة من هذا الدفتر (كانت تفترض `enabled_heads` قائمة أسماء، بينما هي فعلياً
قاموس `{اسم_الرأس: True/False}`).
"""
"""
أدوات التعامل مع رؤوس المخرجات (heads).

كل هدف سعري (high / low / close) له رأسان:
  * ``{target}_class`` — تصنيف اتجاه 1/0 (1 = صعود)
  * ``{target}_reg``   — قيمة انحدار (سعر مُطبَّع)

كل مراحل النظام (تحضير البيانات، بناء النموذج، الخسارة، التنبؤ، التقييم) تستدعي
:func:`get_target_heads` بدل تثبيت الأسماء يدوياً، لذا إطفاء أي رأس في
``CONFIG['enabled_heads']`` يسري تلقائياً على النظام كله.
"""

HEAD_KINDS: Tuple[str, str] = ('class', 'reg')

#: بادئة اسم طبقة/مفتاح المخرج المقابل لكل رأس.
#: اسم الرأس ``'close_class'`` ⇄ اسم المخرج ``'y_close_class'``.
OUTPUT_PREFIX = 'y_'


def output_name(head: str) -> str:
    """اسم مخرج النموذج المقابل لرأس: ``'close_class'`` → ``'y_close_class'``."""
    return head if head.startswith(OUTPUT_PREFIX) else f'{OUTPUT_PREFIX}{head}'


def head_name(output: str) -> str:
    """عكس :func:`output_name`: ``'y_close_class'`` → ``'close_class'``."""
    return output[len(OUTPUT_PREFIX):] if output.startswith(OUTPUT_PREFIX) else output


def get_output_names(targets: Optional[List[str]] = None,
                     config: Optional[dict] = None) -> List[str]:
    """أسماء مخرجات النموذج للرؤوس المُفعَّلة، بالترتيب نفسه."""
    return [output_name(h) for h in get_target_heads(targets, config)]


def get_target_heads(targets: Optional[List[str]] = None,
                     config: Optional[dict] = None,
                     include_disabled: bool = False) -> List[str]:
    """يبني قائمة أسماء رؤوس المخرجات المُفعَّلة من قائمة الأهداف السعرية.

    مثال: ``['high','low','close']`` →
    ``['high_class','high_reg','low_class','low_reg','close_class','close_reg']``

    Args:
        targets: الأهداف السعرية (افتراضياً ``config['targets']``).
        config: قاموس الإعدادات (افتراضياً ``CONFIG``).
        include_disabled: True يُرجع كل الرؤوس متجاهلاً مفاتيح التعطيل (للتشخيص).
    """
    config = CONFIG if config is None else config
    targets = list(targets) if targets is not None else list(config['targets'])
    enabled = config.get('enabled_heads', {}) or {}
    heads: List[str] = []
    for t in targets:
        for kind in HEAD_KINDS:
            head = f'{t}_{kind}'
            if include_disabled or enabled.get(head, True):
                heads.append(head)
    return heads


def split_head_name(head: str) -> Tuple[str, str]:
    """``'low_reg'`` → ``('low', 'reg')``."""
    for kind in HEAD_KINDS:
        suffix = f'_{kind}'
        if head.endswith(suffix):
            return head[: -len(suffix)], kind
    raise ValueError(f"اسم رأس غير معروف (لا ينتهي بـ _class أو _reg): {head}")


def is_regression_head(head: str) -> bool:
    return split_head_name(head)[1] == 'reg'


def is_class_head(head: str) -> bool:
    return split_head_name(head)[1] == 'class'


def get_class_heads(config: Optional[dict] = None) -> List[str]:
    """رؤوس التصنيف المُفعَّلة فقط."""
    return [h for h in get_target_heads(config=config) if is_class_head(h)]


def get_reg_heads(config: Optional[dict] = None) -> List[str]:
    """رؤوس الانحدار المُفعَّلة فقط."""
    return [h for h in get_target_heads(config=config) if is_regression_head(h)]


def paired_targets(config: Optional[dict] = None) -> List[str]:
    """الأهداف التي رأساها (class و reg) مُفعَّلان معاً — يلزمها head_agreement."""
    config = CONFIG if config is None else config
    heads = get_target_heads(config=config)
    return [t for t in config['targets']
            if f'{t}_class' in heads and f'{t}_reg' in heads]


# ══════════════════════════════════════════════════════════════════════════
# طبقة الامتناع
# ══════════════════════════════════════════════════════════════════════════
def abstention_cfg(config: Optional[dict] = None) -> dict:
    """إعدادات الامتناع مع قيم افتراضية آمنة."""
    config = CONFIG if config is None else config
    return config.get('abstention', {}) or {'enabled': False}


def abstention_enabled(config: Optional[dict] = None) -> bool:
    return bool(abstention_cfg(config).get('enabled', False))


def uncertainty_head_enabled(config: Optional[dict] = None) -> bool:
    """هل تُبنى رؤوس التصنيف بطبقة عدم اليقين (MC-dropout)؟

    يتطلب أن يكون الامتناع مُفعَّلاً — تعطيله يُعيد الرؤوس البسيطة.
    """
    cfg = abstention_cfg(config)
    return bool(cfg.get('enabled', False) and cfg.get('use_uncertainty_head', False))


def signal_enabled(name: str, config: Optional[dict] = None) -> bool:
    """هل إشارة ثقة معيّنة (margin / uncertainty / head_agreement) مُفعَّلة؟"""
    cfg = abstention_cfg(config)
    if not cfg.get('enabled', False):
        return False
    return bool((cfg.get('signals', {}) or {}).get(name, False))


def decision_head(config: Optional[dict] = None) -> Optional[str]:
    """رأس القرار المستخدم في التقييم الانتقائي، مع تراجع آمن لأول رأس تصنيف متاح."""
    cfg = abstention_cfg(config)
    heads = get_target_heads(config=config)
    wanted = cfg.get('decision_head')
    if wanted and wanted in heads:
        return wanted
    class_heads = [h for h in heads if is_class_head(h)]
    return class_heads[0] if class_heads else None


# ══════════════════════════════════════════════════════════════════════════
# فحص التناسق
# ══════════════════════════════════════════════════════════════════════════
def validate_head_config(config: Optional[dict] = None,
                         verbose: bool = True) -> List[str]:
    """فحص تناسق إعدادات الرؤوس قبل التدريب — يكشف الأخطاء الصامتة مبكراً.

    يُرجع قائمة التحذيرات (فارغة = كل شيء متناسق).
    """
    config = CONFIG if config is None else config
    warnings_list: List[str] = []
    heads = get_target_heads(config=config)

    if not heads:
        warnings_list.append(
            "❌ كل الرؤوس مُعطَّلة في enabled_heads — لا يمكن بناء نموذج بلا مخرجات.")

    unknown = set(config.get('enabled_heads', {})) - set(
        get_target_heads(config=config, include_disabled=True))
    if unknown:
        warnings_list.append(
            f"⚠️ مفاتيح في enabled_heads لا تطابق أي هدف في targets: {sorted(unknown)}")

    model_tf = config.get('model_tf')
    if model_tf and model_tf not in config.get('tf_order', []):
        warnings_list.append(
            f"❌ model_tf='{model_tf}' غير موجود في tf_order={config.get('tf_order')} — "
            "لن يجد النموذج مصفوفة الإدخال الخاصة به.")
    if model_tf and model_tf not in config.get('window_sizes', {}):
        warnings_list.append(f"❌ لا يوجد window_size للفريم '{model_tf}'.")

    cfg = abstention_cfg(config)
    if cfg.get('enabled', False):
        wanted = cfg.get('decision_head')
        if wanted and wanted not in heads:
            warnings_list.append(
                f"⚠️ رأس القرار '{wanted}' مُعطَّل أو غير موجود — "
                "سيُختار أول رأس تصنيف متاح بدلاً منه.")
        if signal_enabled('uncertainty', config) and not uncertainty_head_enabled(config):
            warnings_list.append(
                "⚠️ إشارة 'uncertainty' مُفعَّلة لكن use_uncertainty_head=False — "
                "الشكوك ستكون أصفاراً فلن تُقصي أي عينة (الإشارة بلا أثر فعلي).")
        if signal_enabled('head_agreement', config) and not paired_targets(config):
            warnings_list.append(
                "⚠️ إشارة 'head_agreement' مُفعَّلة لكن لا يوجد هدف لديه رأسا "
                "class و reg مُفعَّلان معاً — الإشارة ستُتجاهل تلقائياً.")

    if verbose:
        if warnings_list:
            print("🔍 فحص إعدادات الرؤوس:")
            for w in warnings_list:
                print(f"   {w}")
        else:
            print(f"✅ إعدادات الرؤوس متناسقة | الرؤوس المُفعَّلة ({len(heads)}): {heads}")
            print(f"   الامتناع: {'مُفعَّل' if abstention_enabled(config) else 'مُعطَّل'}"
                  f" | رأس عدم اليقين: "
                  f"{'مُفعَّل' if uncertainty_head_enabled(config) else 'مُعطَّل'}")
    return warnings_list


def head_loss_weights(config: Optional[dict] = None,
                      as_outputs: bool = True) -> Dict[str, float]:
    """وزن الخسارة لكل رأس مُفعَّل، مشتقاً من ``target_loss_weights``.

    Args:
        as_outputs: True يُرجع المفاتيح بأسماء المخرجات (``y_...``) كما يتوقّعها
            ``model.compile``؛ False يُرجعها بأسماء الرؤوس المجرّدة.
    """
    config = CONFIG if config is None else config
    weights = config.get('target_loss_weights', {}) or {}
    key = output_name if as_outputs else (lambda h: h)
    return {key(h): float(weights.get(split_head_name(h)[0], 1.0))
            for h in get_target_heads(config=config)}

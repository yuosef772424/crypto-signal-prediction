"""
PURPOSE:  The live CONFIG dict and its helpers: update_config/reset_config (in place), save/load_config, refresh_features, feature_order, seq_len, describe.
TAGS:     CONFIG, update_config, reset_config, refresh_features, feature_order, save_config, load_config, describe, deep update
PITFALLS: CONFIG is mutated in place and never rebound: every function (and every %run caller) holds the same dict object. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own): names from other modules resolve at call time.

## 5) الإعدادات الحيّة `CONFIG` (`runtime.py` سابقاً)

``CONFIG`` هو **نفس القاموس** أينما استُخدم في هذا الدفتر — تعديله عبر
`update_config()` يسري فوراً على كل الدوال (لأنها كلها تقرأ نفس المرجع).
لذلك يُعدَّل دائماً **في المكان** ولا يُعاد إسناده أبداً بـ `CONFIG = {...}`.
"""
"""
كائن الإعدادات الحيّ ``CONFIG`` وأدوات تعديله وحفظه وتحميله.

``CONFIG`` هو **نفس القاموس** في كل مكان من الدفتر: أي دالة تستخدمه تحصل على
المرجع ذاته، فتعديله مرة واحدة يسري فوراً على كل المراحل.
لذلك نُعدّله دائماً **في المكان** (in-place) ولا نُعيد إسناده أبداً.
"""

#: قاموس الإعدادات الحيّ — المرجع المشترك بين كل دوال هذا الدفتر.
CONFIG: Dict[str, Any] = deepcopy(DEFAULT_CONFIG)


def _deep_update(base: dict, new: dict) -> dict:
    """دمج متداخل: القواميس تُدمج، وأي قيمة أخرى تُستبدل."""
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def get_config() -> Dict[str, Any]:
    """يُرجع مرجع الإعدادات الحيّ (وليس نسخة)."""
    return CONFIG


def update_config(overrides: Optional[dict] = None, **kwargs) -> Dict[str, Any]:
    """تحديث ``CONFIG`` في المكان بدمج متداخل.

    >>> update_config({"abstention": {"min_margin": 0.3}}, epochs=50)
    """
    if overrides:
        _deep_update(CONFIG, overrides)
    if kwargs:
        _deep_update(CONFIG, kwargs)
    return CONFIG


def reset_config() -> Dict[str, Any]:
    """إعادة ``CONFIG`` إلى القيم الافتراضية (في المكان)."""
    CONFIG.clear()
    CONFIG.update(deepcopy(DEFAULT_CONFIG))
    return CONFIG


def _jsonable(obj: Any) -> Any:
    """تحويل القيم غير القابلة للتسلسل (tuples داخل indicator_settings) إلى قوائم."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _retuple_indicators(cfg: dict) -> dict:
    """يُعيد القوائم المتداخلة داخل ``indicator_settings`` إلى tuples.

    ``add_features`` تفكّ ``(fast, slow, signal)`` بالتفريغ، وJSON لا يحفظ الـ tuple.
    """
    settings = cfg.get("indicator_settings")
    if isinstance(settings, dict):
        for ind, values in settings.items():
            if isinstance(values, list):
                settings[ind] = [tuple(v) if isinstance(v, list) else v for v in values]
    return cfg


def save_config(path, config: Optional[dict] = None) -> Path:
    """حفظ الإعدادات كملف JSON (يشمل ``feature_order`` المُشتقة)."""
    config = CONFIG if config is None else config
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"💾 حُفظت الإعدادات: {path}")
    return path


def load_config(path, merge: bool = True) -> Dict[str, Any]:
    """تحميل إعدادات من ملف JSON إلى ``CONFIG`` الحيّ.

    Args:
        merge: True يدمج فوق الحالي، False يعيد التعيين للافتراضي أولاً.
    """
    path = Path(path)
    loaded = _retuple_indicators(json.loads(path.read_text(encoding="utf-8")))
    if not merge:
        reset_config()
    _deep_update(CONFIG, loaded)
    print(f"📂 حُمّلت الإعدادات: {path}")
    return CONFIG


def refresh_features(config: Optional[dict] = None, force: bool = True) -> List[str]:
    """إعادة اشتقاق ``feature_order`` من ``indicator_settings`` الحالية.

    استدعِها بعد أي تعديل على المؤشرات أو ``exclude_from_features``.
    """
    config = CONFIG if config is None else config
    if not force and config.get("feature_order"):
        return config["feature_order"]
    features = infer_feature_columns(config)
    config["feature_order"] = features
    print(f"🧬 عدد الميزات المُشتقة: {len(features)}")
    return features


def feature_order(config: Optional[dict] = None) -> List[str]:
    """قائمة الميزات الحالية، تُشتق تلقائياً عند أول طلب إن لم تكن موجودة."""
    config = CONFIG if config is None else config
    if not config.get("feature_order"):
        refresh_features(config)
    return config["feature_order"]


def n_features(config: Optional[dict] = None) -> int:
    return len(feature_order(config))


def seq_len(config: Optional[dict] = None) -> int:
    """طول التسلسل المُغذّى للنموذج (نافذة الفريم ``model_tf``)."""
    config = CONFIG if config is None else config
    return config["window_sizes"][config["model_tf"]]


def describe(config: Optional[dict] = None) -> None:
    """ملخّص مقروء للإعدادات الفعّالة — يُستخدم في رأس كل صفحة."""
    config = CONFIG if config is None else config
    heads = get_target_heads(config=config)
    print("═" * 66)
    print("⚙️  الإعدادات الفعّالة")
    print("═" * 66)
    print(f"  الفريمات       : {config['tf_order']}  (أساسي={config['base_tf']}, "
          f"نموذج={config['model_tf']})")
    print(f"  النوافذ        : {config['window_sizes']}  | stride={config['stride']}")
    print(f"  الأهداف        : {config['targets']}  | أفق={config['forecast_horizon']}")
    print(f"  الرؤوس المُفعَّلة : {heads}")
    print(f"  أوزان الأهداف  : {config.get('target_loss_weights')}")
    print(f"  الامتناع       : {'مُفعَّل' if abstention_enabled(config) else 'مُعطَّل'}"
          f" | رأس عدم اليقين: "
          f"{'مُفعَّل' if uncertainty_head_enabled(config) else 'مُعطَّل'}")
    feats = config.get("feature_order")
    print(f"  الميزات        : {len(feats) if feats else 'لم تُشتق بعد'}")
    print(f"  النموذج        : d_model={config['d_model']}, layers={config['num_layers']}, "
          f"heads={config['num_heads']}/{config['num_kv_heads']}")
    print(f"  التدريب        : lr={config['learning_rate']}, epochs={config['epochs']}, "
          f"batch={config['batch_size']}")
    print("═" * 66)

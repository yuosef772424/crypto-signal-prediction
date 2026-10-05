"""
PURPOSE:  Package holding ALL data-preparation code of the project (formerly the cells of
          crypto_data_pipeline_v6.ipynb): `import data` gives the whole pipeline (data.CONFIG, data.build_dataset,
          data.split_data, ...) in this package's namespace, loaded lazily on first use.
TAGS:     data pipeline, package entry, import data, lazy load, shared namespace, CONFIG, build_dataset, split_data,
          crypto_data_pipeline_v6, runner notebook
PITFALLS: The modules run in ONE shared namespace (see data/_loader.py), so never `import data.<module>`; use
          `import data` (namespace = this package) or `data.load_into(ns)` (namespace = your dict, what the runner
          notebook does with globals()). Patch names on the namespace that loaded them (data.mount_drive = ... or
          ns["mount_drive"] = ...). data/crypto_data/asset_registry.csv is data, not code.

خريطة الوحدات (بترتيب التحميل) — كل وحدة كانت خلية في crypto_data_pipeline_v6.ipynb:
  common            الاستيرادات المشتركة            drive             تركيب Google Drive
  defaults          DEFAULT_CONFIG                  runtime           CONFIG الحيّ وupdate_config
  heads             رؤوس المخرجات                   custom            الميزات المخصّصة
  features          المؤشرات الفنية                  normalize         التطبيع
  align             محاذاة الفريمات                  windows           النوافذ والأهداف لأصل واحد
  diagnostics       التشخيص                          parallel          التحميل المتوازي
  sources           سجل الأصول والتحميل من Drive     market_context    السياق السوقي (MKT_)
  cross_sectional_features  ميزات مقطعية (رتبة الزخم، الاتساع)
  checkpoints       نقاط الاستئناف                   disk_backed       البناء المدعوم بالقرص
  pipeline          build_dataset                    cross_sectional_norm  تطبيع الأهداف المقطعي
  split             التقسيم الزمني                   binance_client    عميل Binance والجلب الحيّ
  funding_oi        التمويل والفائدة المفتوحة         phase2            ربط المرحلة ٢ (tools/intraday_features.py)
  live              بيانات حيّة                      storage           الحفظ/التحميل من Drive
  selftests         run_pipeline_selftests           presets           إعدادات 1h الجاهزة (20-ب/ج/د)
"""
import threading as _threading

from data._loader import MODULES, PACKAGE_DIR, REPO_ROOT, load_into, module_path  # noqa: F401

_DATA_PKG_LOCK = _threading.RLock()
_DATA_PKG_STATE = {"loaded": False}


def _data_pkg_ensure_loaded() -> None:
    """Load every pipeline module into this package's namespace once (first attribute access)."""
    if _DATA_PKG_STATE["loaded"]:
        return
    with _DATA_PKG_LOCK:
        if not _DATA_PKG_STATE["loaded"]:
            load_into(globals())
            _DATA_PKG_STATE["loaded"] = True


def __getattr__(name):
    """PEP 562: data.<pipeline name> loads the pipeline on first use (so `import data` itself stays cheap and the
    runner can load into its own globals without loading twice)."""
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    _data_pkg_ensure_loaded()
    if name == "__all__":
        return sorted(k for k in globals() if not k.startswith("_"))
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'data' has no attribute {name!r}") from None


def __dir__():
    _data_pkg_ensure_loaded()
    return sorted(globals())

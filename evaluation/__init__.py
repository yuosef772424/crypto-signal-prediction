"""
PURPOSE:  Package holding ALL model-evaluation code of the project (formerly the cells of chicks_v4_5_input_output_patterns.ipynb): `import evaluation` gives predict_with_evaluation_v4, test_all_assets_v4, run_full_analysis, ... in this package's namespace, loaded lazily on first use.
TAGS:     model evaluation, package entry, import evaluation, lazy load, shared namespace, predict_with_evaluation_v4, test_all_assets_v4, run_full_analysis, chicks_v4_5, runner notebook
PITFALLS: The modules run in ONE shared namespace (see evaluation/_loader.py), so never `import evaluation.<module>`; use `import evaluation` (namespace = this package) or `evaluation.load_into(ns)` (namespace = your dict, what the runner notebook does with globals()). Patch names on the namespace that loaded them.

خريطة الوحدات (بترتيب التحميل) — كل وحدة كانت خلية (أو خلايا) في chicks_v4_5_input_output_patterns.ipynb (الرقم = فهرس الخلية القديم، 0-based):
  targets (2)           TargetSpec وresolve_targets والاستيرادات    outputs (4)           save_or_print / save_figure
  predict (6)           المرحلة 1: التنبؤ على الدفعات              decode (8)            المرحلة 2: فك التشفير
  verify (10)           المرحلة 2.5: التحقق من فك التشفير          metrics (12)          المرحلة 3: المقاييس
  latest_table (14)     جدول آخر العينات                          unified (16)          predict_with_evaluation_v4
  all_assets (18)       test_all_assets_v4                        live (20)             predict_latest_v4 (التداول الحي)
  flat (22)             build_flat_dataframe                      trust_calibration (24) الثقة والمعايرة
  output_conditioned (26) تحليل موجَّه للمخرجات                    patterns (28)         اكتشاف الأنماط
  tearsheet (30)        مقاييس التداول                            plots (32)            الرسوم
  integrity (34)        تشخيص نزاهة النموذج                       full_analysis (36)    run_full_analysis
  legacy_uncertainty (38) تقرير قديم                              legacy_assets (40)    تقارير محاكاة قديمة
  trade_selection (42-44) اختيار الصفقات                          io_patterns (46-49)   أنماط المدخلات ↔ المخرجات
"""
import threading as _threading

from evaluation._loader import MODULES, PACKAGE_DIR, REPO_ROOT, exec_module, load_into, module_path  # noqa: F401

_EVAL_PKG_LOCK = _threading.RLock()
_EVAL_PKG_STATE = {"loaded": False}


def _eval_pkg_ensure_loaded() -> None:
    """Load every evaluation module into this package's namespace once (first attribute access)."""
    if _EVAL_PKG_STATE["loaded"]:
        return
    with _EVAL_PKG_LOCK:
        if not _EVAL_PKG_STATE["loaded"]:
            load_into(globals())
            _EVAL_PKG_STATE["loaded"] = True


def __getattr__(name):
    """PEP 562: evaluation.<name> loads the modules on first use (so `import evaluation` itself stays cheap and the
    runner can load into its own globals without loading twice)."""
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    _eval_pkg_ensure_loaded()
    if name == "__all__":
        return sorted(k for k in globals() if not k.startswith("_"))
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'evaluation' has no attribute {name!r}") from None


def __dir__():
    _eval_pkg_ensure_loaded()
    return sorted(globals())

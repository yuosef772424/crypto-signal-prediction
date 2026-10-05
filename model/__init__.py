"""
PURPOSE:  Package holding the NIG-TimeNet v2 model code (formerly the cells of model_v2 (1).ipynb): `import model`
          gives the whole model (model.build_model_fn, model.MODEL_CONFIG, model.diagnose_model, ...) in this
          package's namespace, loaded lazily on first use.
TAGS:     NIG-TimeNet v2, model, build_model_fn, MODEL_CONFIG, model_v2, package entry, import model, lazy load,
          shared namespace, runner notebook
PITFALLS: The modules run in ONE shared namespace (see model/_loader.py), so never `import model.<module>`; use
          `import model` (namespace = this package) or `model.load_into(ns)` (namespace = your dict, what the runner
          notebook does with globals()). Patch names on the namespace that loaded them (model.name = ... or
          ns["name"] = ...). The selftests module runs at load (use model.load_into(ns, exclude=("selftests",)) to skip it).

خريطة الوحدات (بترتيب التحميل) — كل وحدة كانت خلية أو أكثر في model_v2 (1).ipynb:
  common                الاستيرادات و register
  input_norm            InstanceNorm وSymLog
  decomposition         تفكيك المقاييس الزمنية
  patches               الرُّقَع والمواضع
  transformer           كتلة المحوّل
  readout               مساعدات القراءة
  nig_layers            طبقات رؤوس NIG
  heads                 HEAD_REGISTRY
  builder               build_nig_timenet_v2
  config                MODEL_CONFIG وbuild_model_fn
  diagnostics           diagnose_model / model_health_verdicts
  layer_report          تقرير الطبقات
  selftests             الاختبار الذاتي (يعمل عند التحميل)
"""
import threading as _threading

from model._loader import MODULES, PACKAGE_DIR, REPO_ROOT, load_into, module_path  # noqa: F401

_MODEL_PKG_LOCK = _threading.RLock()
_MODEL_PKG_STATE = {"loaded": False}


def _model_pkg_ensure_loaded() -> None:
    """Load every module into this package's namespace once (first attribute access)."""
    if _MODEL_PKG_STATE["loaded"]:
        return
    with _MODEL_PKG_LOCK:
        if not _MODEL_PKG_STATE["loaded"]:
            load_into(globals())
            _MODEL_PKG_STATE["loaded"] = True


def __getattr__(name):
    """PEP 562: model.<name> loads the modules on first use (so `import model` itself stays cheap and the runner can
    load into its own globals without loading twice)."""
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    _model_pkg_ensure_loaded()
    if name == "__all__":
        return sorted(k for k in globals() if not k.startswith("_"))
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'model' has no attribute {name!r}") from None


def __dir__():
    _model_pkg_ensure_loaded()
    return sorted(globals())

"""
PURPOSE:  Package holding ALL code of the signal evaluation axis, phase 0 of the signal-discovery project (formerly the cells of "signal_evaluation_axis (3).ipynb"): `import signal_eval` gives compute_ic, decile_spread, permutation_baseline, evaluate_windows, evaluate_hypothesis_over_rolling_windows, register_hypothesis, ... in this package's namespace, loaded lazily on first use.
TAGS:     signal evaluation axis, phase 0, package entry, import signal_eval, lazy load, shared namespace, compute_ic, evaluate_windows, register_hypothesis, experiment registry, runner notebook
PITFALLS: The modules run in ONE shared namespace (see signal_eval/_loader.py), so never `import signal_eval.<module>`; use `import signal_eval` (namespace = this package) or `signal_eval.load_into(ns)` (namespace = your dict, what the runner notebook does with globals()). Loading only DEFINES the axis: the pipeline names (rolling_splits, CONFIG, load_preprocessed_data_from_drive, from the data/ package), the self-tests and the Drive dataset load are run by the runner notebook.

خريطة الوحدات (بترتيب التحميل) — كل وحدة كانت خلية (أو خلايا) في signal_evaluation_axis (3).ipynb (الرقم = فهرس الخلية القديم، 0-based):
  common (2)              الاستيرادات                              core (4, 6, 8)        compute_ic وdecile_spread وpermutation_baseline
  windows (10)            evaluate_windows                         integration (12)      طبقة التكامل مع rolling_splits
  selftests (14)          الاختبارات الذاتية (تعريفات فقط)           bootstrap (15)        download_notebook_from_drive (كان dataprocess.ipynb)
  registry (20)           سجلّ التجارب                              registry_selftests (22) اختبارات السجلّ (تعريف فقط)
خلايا 16 و18 (تحميل dataset من Drive ومثال بوّابة المرحلة ٠) وتشغيل الاختبارات الذاتية صارت خلايا في الدفتر المُشغِّل نفسه.
"""
import threading as _threading

from signal_eval._loader import MODULES, PACKAGE_DIR, REPO_ROOT, exec_module, load_into, module_path  # noqa: F401

_SIGEVAL_PKG_LOCK = _threading.RLock()
_SIGEVAL_PKG_STATE = {"loaded": False}


def _sigeval_pkg_ensure_loaded() -> None:
    """Load every signal_eval module into this package's namespace once (first attribute access)."""
    if _SIGEVAL_PKG_STATE["loaded"]:
        return
    with _SIGEVAL_PKG_LOCK:
        if not _SIGEVAL_PKG_STATE["loaded"]:
            load_into(globals())
            _SIGEVAL_PKG_STATE["loaded"] = True


def __getattr__(name):
    """PEP 562: signal_eval.<name> loads the modules on first use (so `import signal_eval` itself stays cheap and the
    runner can load into its own globals without loading twice)."""
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    _sigeval_pkg_ensure_loaded()
    if name == "__all__":
        return sorted(k for k in globals() if not k.startswith("_"))
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'signal_eval' has no attribute {name!r}") from None


def __dir__():
    _sigeval_pkg_ensure_loaded()
    return sorted(globals())

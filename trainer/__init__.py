"""
PURPOSE:  Package holding the generic training framework (formerly the cells of trainer_framework_v2.ipynb): `import
          trainer` gives the whole framework (trainer.build_training_system, trainer.build_config,
          trainer.GenericTrainer, ...) in this package's namespace, loaded lazily on first use.
TAGS:     generic trainer, trainer framework, build_training_system, GenericTrainer, build_config,
          trainer_framework_v2, package entry, import trainer, lazy load, shared namespace, runner notebook
PITFALLS: The modules run in ONE shared namespace (see trainer/_loader.py), so never `import trainer.<module>`; use
          `import trainer` (namespace = this package) or `trainer.load_into(ns)` (namespace = your dict, what the runner
          notebook does with globals()). Patch names on the namespace that loaded them (trainer.name = ... or
          ns["name"] = ...). The smoke_test module trains for a few epochs at load (use trainer.load_into(ns, exclude=("smoke_test",)) to skip it).

خريطة الوحدات (بترتيب التحميل) — كل وحدة كانت خلية أو أكثر في trainer_framework_v2.ipynb:
  env                   الاستيرادات والبيئة
  perf                  الأداء وGPU
  config                DEFAULT_CONFIG ودمجها
  tasks                 سجلّ الخسائر
  task_weighting        الموازنة التلقائية بين المهام
  trainer               GenericTrainer
  schedules             الجدولة
  best_tracker          BestModelTracker
  metrics               MetricsLogger وغيرها
  snapshot              SnapshotEnsemble
  training_diagnostics  TrainingDiagnostics
  checkpoints           الحفظ والاستئناف
  epoch_callbacks       DriveMirror وEpochGuard
  system                build_training_system
  smoke_test            Smoke Test (يعمل عند التحميل)
  example               قالب الاستخدام
  kfold                 K-Fold
  ensemble              استدلال Ensemble
"""
import threading as _threading

from trainer._loader import MODULES, PACKAGE_DIR, REPO_ROOT, load_into, module_path  # noqa: F401

_TRAINER_PKG_LOCK = _threading.RLock()
_TRAINER_PKG_STATE = {"loaded": False}


def _trainer_pkg_ensure_loaded() -> None:
    """Load every module into this package's namespace once (first attribute access)."""
    if _TRAINER_PKG_STATE["loaded"]:
        return
    with _TRAINER_PKG_LOCK:
        if not _TRAINER_PKG_STATE["loaded"]:
            load_into(globals())
            _TRAINER_PKG_STATE["loaded"] = True


def __getattr__(name):
    """PEP 562: trainer.<name> loads the modules on first use (so `import trainer` itself stays cheap and the runner can
    load into its own globals without loading twice)."""
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    _trainer_pkg_ensure_loaded()
    if name == "__all__":
        return sorted(k for k in globals() if not k.startswith("_"))
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'trainer' has no attribute {name!r}") from None


def __dir__():
    _trainer_pkg_ensure_loaded()
    return sorted(globals())

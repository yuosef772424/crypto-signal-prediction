"""
PURPOSE:  Package holding the project-assembly code of main.ipynb (target-mode switching, split helpers, training batches, model/trainer bridges, section-7 evaluation reports, diagnostics, capacity controls, wiring self-test); main.ipynb stays the step-by-step runner.
TAGS:     workflow, main notebook, assembly, retarget_splits, make_shuffled_dataset, model_health_report, capacity_report, section 7 reports, shared namespace, lazy load
PITFALLS: The modules run in ONE shared namespace (see workflow/_loader.py), so never `import workflow.<module>`; use `import workflow`
          (namespace = this package, loaded lazily on first attribute access) or `workflow.load_into(ns, only=...)` (namespace =
          your dict, what the runner notebook does with globals()). Patch names on the namespace that loaded them. The package-level
          namespace is seeded with the data pipeline's names only; names of the model_v2 / trainer / chicks notebooks (build_model_fn,
          build_training_system, TargetSpec...) exist only in the runner's namespace.

Modules (load order) and the old main.ipynb cell each was extracted from:
  dataset_io          cell 7 (section 3)
  splits              cell 8 (section 3)
  retarget            cell 10 (section 3-b)
  model_build         cell 13 (section 4)
  training_config     cells 15 and 17 (section 5)
  batches             cell 17 (section 5)
  chicks_bridge       cell 19 (section 6)
  reports             cell 23 (section 7)
  pooling             cell 24 (section 7)
  selective_eval      cell 26 (section 7-b)
  candle_baseline     cell 28 (section 7-c)
  verification        cell 30 (section 7-d)
  permutation_control cell 32 (section 7-e)
  market_neutral      cell 34 (section 7-f)
  generalization      cell 36 (section 7-g)
  panel_bridge        cell 39 (section 7-h; lifted from nested defs)
  diagnostics         cell 40 (section 7-i)
  capacity            cell 42 (section 7-j)
  wiring_selftest     cell 44 (section 8)
"""
import threading as _threading

from workflow._loader import MODULES, PACKAGE_DIR, REPO_ROOT, load_into, module_path  # noqa: F401

_LOCK = _threading.RLock()
_STATE = {"loaded": False}


def _ensure_loaded() -> None:
    """Load every module into this package's namespace once (first attribute access)."""
    if _STATE["loaded"]:
        return
    with _LOCK:
        if not _STATE["loaded"]:
            # the modules use pipeline names (CONFIG, LAST_COLUMNS, split_data...) at load or call time: seed them first, without
            # the pipeline's self-test module (`import data` would run it at load; its fetch tests need the Binance client mocks)
            import data
            seed = data.load_into({"__name__": "data"}, exclude=("selftests",))
            globals().update({k: v for k, v in seed.items() if not k.startswith("__") and k not in globals()})
            load_into(globals())
            _STATE["loaded"] = True


def __getattr__(name):
    """PEP 562: workflow.<name> loads the modules on first use (so `import workflow` itself stays cheap and a runner can load
    into its own globals without loading twice)."""
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    _ensure_loaded()
    if name == "__all__":
        return sorted(k for k in globals() if not k.startswith("_"))
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'workflow' has no attribute {name!r}") from None


def __dir__():
    _ensure_loaded()
    return sorted(globals())

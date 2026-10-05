"""
PURPOSE:  Package holding the code of the signal discovery lab (candidate predictors, evaluation core, scanner, batch runner, phase-3 tools, self-test) and of the pandas_ta full survey; the notebooks stay the runners of the lab steps.
TAGS:     discovery, signal discovery lab, candidate, clean_reg_target, evaluate_candidate, scan_candidates, run_batch_and_register, pandas_ta survey, run_survey, shared namespace, lazy load
PITFALLS: The modules run in ONE shared namespace (see discovery/_loader.py), so never `import discovery.<module>`; use `import discovery`
          (namespace = this package, loaded lazily on first attribute access) or `discovery.load_into(ns, only=...)` (namespace =
          your dict, what the runner notebook does with globals()). Patch names on the namespace that loaded them.

Modules (load order) and the notebook cells each was extracted from:
  axis_loader           signal_discovery_lab.ipynb cell 2 (section 1)
  evaluation            signal_discovery_lab.ipynb cell 6 (section 3)
  predictors            signal_discovery_lab.ipynb cell 8 (section 4)
  hypothesis_predictors signal_discovery_lab.ipynb cells 14, 16, 22 and 45 (sections 5, 7, 12)
  scanner               signal_discovery_lab.ipynb cell 19 (section 6)
  batch_runner          signal_discovery_lab.ipynb cell 25 (section 8)
  phase3_tools          signal_discovery_lab.ipynb cells 28, 31, 34, 38 and 41 (sections 9, 10)
  selftest              signal_discovery_lab.ipynb cell 92 (section 29)
  survey                pandas_ta_full_survey.ipynb cells 4, 6 and 14 (sections 2, 3, 6)
"""
import threading as _threading

from discovery._loader import MODULES, PACKAGE_DIR, REPO_ROOT, load_into, module_path  # noqa: F401

_LOCK = _threading.RLock()
_STATE = {"loaded": False}


def _ensure_loaded() -> None:
    """Load every module into this package's namespace once (first attribute access)."""
    if _STATE["loaded"]:
        return
    with _LOCK:
        if not _STATE["loaded"]:
            load_into(globals())
            _STATE["loaded"] = True


def __getattr__(name):
    """PEP 562: discovery.<name> loads the modules on first use (so `import discovery` itself stays cheap and a runner can load
    into its own globals without loading twice)."""
    if name.startswith("__") and name != "__all__":
        raise AttributeError(name)
    _ensure_loaded()
    if name == "__all__":
        return sorted(k for k in globals() if not k.startswith("_"))
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'discovery' has no attribute {name!r}") from None


def __dir__():
    _ensure_loaded()
    return sorted(globals())

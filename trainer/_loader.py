"""
PURPOSE:  Execute the trainer modules, in their fixed order, into ONE shared namespace dict (exactly what the notebook's
          cells did): load_into(ns) is the single code path behind `import trainer`, the runner notebook trainer_framework_v2.ipynb and
          docs/research/audit/_nbload.load_trainer().
TAGS:     loader, shared namespace, load_into, MODULES, module order, late binding, monkeypatch, %run, REPO_ROOT,
          trainer package
PITFALLS: The modules are not standalone Python modules: they reference each other's names (also names defined by
          LATER modules, resolved at call time) and share one dict, so they must run in one dict. Patching that dict
          (ns["name"] = ...) changes what every trainer function sees. MODULES order is load order: a module's
          top-level code may only use names of earlier modules. The smoke_test module RUNS A REAL TINY TRAINING AT LOAD (and metrics/system run small self-tests): pass exclude=("smoke_test",) to skip it.
"""
from __future__ import annotations

from pathlib import Path

#: Folder of this package, and the repo root that holds it (tools/, docs/, ... live next to trainer/).
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

#: Load order = the order of the cells these modules were extracted from (trainer_framework_v2.ipynb).
MODULES = (
    "env",
    "perf",
    "config",
    "tasks",
    "task_weighting",
    "trainer",
    "schedules",
    "best_tracker",
    "metrics",
    "snapshot",
    "training_diagnostics",
    "checkpoints",
    "epoch_callbacks",
    "system",
    "smoke_test",
    "example",
    "kfold",
    "ensemble",
)


def module_path(name: str) -> Path:
    """Path of trainer module ``name`` (must be one of :data:`MODULES`)."""
    if name not in MODULES:
        raise ValueError(f"unknown trainer module {name!r}; known: {MODULES}")
    return PACKAGE_DIR / f"{name}.py"


def load_into(ns: dict, exclude=()) -> dict:
    """Execute every module of :data:`MODULES` (minus ``exclude``) in order into ``ns`` and return ``ns``.

    Each file is compiled under its own path (tracebacks and ``inspect`` point at trainer/<module>.py) and with
    ``dont_inherit=True``, so a ``__future__`` import never leaks from the caller or between modules — the same as
    ``%run`` of the old notebook. ``ns['__doc__']`` is restored afterwards (module docstrings would overwrite it).
    """
    unknown = set(exclude) - set(MODULES)
    if unknown:
        raise ValueError(f"unknown trainer module(s) in exclude: {sorted(unknown)}; known: {MODULES}")
    had_doc, doc = "__doc__" in ns, ns.get("__doc__")
    try:
        for name in MODULES:
            if name in exclude:
                continue
            path = module_path(name)
            code = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
            exec(code, ns)
    finally:
        if had_doc:
            ns["__doc__"] = doc
        else:
            ns.pop("__doc__", None)
    return ns

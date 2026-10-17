"""
PURPOSE:  Execute the workflow modules, in their fixed order, into ONE shared namespace dict (the notebook's globals): load_into(ns)
          is the single code path behind `import workflow`, the runner notebook main.ipynb and docs/research/audit/_nbload. ``only=`` / ``exclude=`` select
          modules (a runner loads the modules of a section at the cell where the old notebook defined them).
TAGS:     loader, shared namespace, load_into, MODULES, module order, late binding, monkeypatch, %run, REPO_ROOT, workflow package
PITFALLS: The modules are not standalone Python modules: they reference names of the notebook namespace (data pipeline, model,
          trainer, chicks and each other, resolved at call time) and tests patch names through the dict, so they must run in
          one dict. MODULES order is load order: a module's top-level code may only use names of earlier modules or of the
          notebooks %run before it. A name defined here shadows a same-named %run notebook name loaded EARLIER (as before).
"""
from __future__ import annotations

from pathlib import Path

#: Folder of this package, and the repo root that holds it (tools/, docs/, ... live next to workflow/).
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

#: Load order = the order of the notebook cells these modules were extracted from (main.ipynb).
MODULES = (
    "settings",
    "dataset_io",
    "splits",
    "retarget",
    "model_build",
    "training_config",
    "batches",
    "chicks_bridge",
    "reports",
    "pooling",
    "selective_eval",
    "candle_baseline",
    "verification",
    "permutation_control",
    "market_neutral",
    "generalization",
    "panel_bridge",
    "diagnostics",
    "capacity",
    "wiring_selftest",
    "run",
)


def module_path(name: str) -> Path:
    """Path of workflow module ``name`` (must be one of :data:`MODULES`)."""
    if name not in MODULES:
        raise ValueError(f"unknown workflow module {name!r}; known: {MODULES}")
    return PACKAGE_DIR / f"{name}.py"


def load_into(ns: dict, only=None, exclude=()) -> dict:
    """Execute the modules of :data:`MODULES` (restricted to ``only`` when given, minus ``exclude``) in order into ``ns``.

    Each file is compiled under its own path (tracebacks and ``inspect`` point at workflow/<module>.py) and with
    ``dont_inherit=True``, so a ``__future__`` import never leaks from the caller or between modules. ``ns['__doc__']`` is
    restored afterwards (module docstrings would overwrite it). Returns ``ns``.
    """
    unknown = (set(exclude) | set(only or ())) - set(MODULES)
    if unknown:
        raise ValueError(f"unknown workflow module(s): {sorted(unknown)}; known: {MODULES}")
    had_doc, doc = "__doc__" in ns, ns.get("__doc__")
    try:
        for name in MODULES:
            if name in exclude or (only is not None and name not in only):
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

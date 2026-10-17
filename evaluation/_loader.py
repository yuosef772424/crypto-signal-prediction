"""
PURPOSE:  Execute the model-evaluation modules, in their fixed order, into ONE shared namespace dict (exactly what the notebook's cells did): load_into(ns) is the single code path behind `import evaluation`, the runner notebook chicks_v4_5_input_output_patterns.ipynb and docs/research/audit/_nbload.load_evaluation().
TAGS:     loader, shared namespace, load_into, exec_module, MODULES, module order, late binding, monkeypatch, %run, REPO_ROOT, evaluation package
PITFALLS: The modules are not standalone Python modules: they reference each other's names (resolved at call time) and the tests patch names through the dict, so they must run in one dict. MODULES order is load order: a module's top-level code may only use names of earlier modules. Modules all_assets/live/full_analysis are skipped by the tests' general load (exclude=...) and exec'd explicitly with exec_module.
"""
from __future__ import annotations

from pathlib import Path

#: Folder of this package, and the repo root that holds it (tools/, docs/, ... live next to evaluation/).
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

#: Load order = the order of the cells these modules were extracted from (chicks_v4_5_input_output_patterns.ipynb).
MODULES = (
    "targets", "outputs", "predict", "decode", "verify", "metrics", "latest_table", "unified", "all_assets", "live",
    "flat", "trust_calibration", "output_conditioned", "patterns", "tearsheet", "plots", "integrity", "full_analysis",
    "legacy_uncertainty", "legacy_assets", "trade_selection", "io_patterns",
)


def module_path(name: str) -> Path:
    """Path of evaluation module ``name`` (must be one of :data:`MODULES`)."""
    if name not in MODULES:
        raise ValueError(f"unknown evaluation module {name!r}; known: {MODULES}")
    return PACKAGE_DIR / f"{name}.py"


def exec_module(name: str, ns: dict) -> dict:
    """Execute one module into ``ns`` (compiled under its own path, ``dont_inherit=True``: no ``__future__`` leaks)."""
    path = module_path(name)
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
    exec(code, ns)
    return ns


def load_into(ns: dict, exclude=()) -> dict:
    """Execute every module of :data:`MODULES` (minus ``exclude``) in order into ``ns`` and return ``ns``.

    Same as ``%run`` of the old notebook: one shared namespace, each file compiled under its own path (tracebacks and
    ``inspect`` point at evaluation/<module>.py). ``ns['__doc__']`` is restored afterwards (module docstrings would
    overwrite it).
    """
    unknown = set(exclude) - set(MODULES)
    if unknown:
        raise ValueError(f"unknown evaluation module(s) in exclude: {sorted(unknown)}; known: {MODULES}")
    had_doc, doc = "__doc__" in ns, ns.get("__doc__")
    try:
        for name in MODULES:
            if name not in exclude:
                exec_module(name, ns)
    finally:
        if had_doc:
            ns["__doc__"] = doc
        else:
            ns.pop("__doc__", None)
    return ns

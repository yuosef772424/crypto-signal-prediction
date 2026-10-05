"""
PURPOSE:  Execute the signal-evaluation-axis modules, in their fixed order, into ONE shared namespace dict (exactly what the notebook's cells did): load_into(ns) is the single code path behind `import signal_eval`, the runner notebook "signal_evaluation_axis (3).ipynb" and docs/research/audit/_nbload.load_signal_eval().
TAGS:     loader, shared namespace, load_into, exec_module, MODULES, module order, late binding, monkeypatch, %run, REPO_ROOT, signal_eval package
PITFALLS: The modules are not standalone Python modules: they reference each other's names (resolved at call time), so they must run in one dict. MODULES order is load order: a module's top-level code may only use names of earlier modules. Loading defines the axis only: the self-tests (selftests.ALL_TESTS), the registry self-tests and the data bootstrap are run by the runner notebook, not by load_into.
"""
from __future__ import annotations

from pathlib import Path

#: Folder of this package, and the repo root that holds it (tools/, docs/, ... live next to signal_eval/).
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

#: Load order = the order of the cells these modules were extracted from (signal_evaluation_axis (3).ipynb).
MODULES = (
    "common", "core", "windows", "integration", "selftests", "bootstrap", "registry", "registry_selftests",
)


def module_path(name: str) -> Path:
    """Path of signal_eval module ``name`` (must be one of :data:`MODULES`)."""
    if name not in MODULES:
        raise ValueError(f"unknown signal_eval module {name!r}; known: {MODULES}")
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
    ``inspect`` point at signal_eval/<module>.py). ``ns['__doc__']`` is restored afterwards (module docstrings would
    overwrite it).
    """
    unknown = set(exclude) - set(MODULES)
    if unknown:
        raise ValueError(f"unknown signal_eval module(s) in exclude: {sorted(unknown)}; known: {MODULES}")
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

"""Audit helper: execute the code cells of a notebook into a namespace, without Colab.

Why: repro scripts must exercise the exact functions in the notebooks (not copies), so a
fix in the notebook makes the repro pass. Shell/magic lines ('!', '%') are dropped, and
cells whose names are passed in `skip` (e.g. self-test cells that take minutes) are not run.
"""
import contextlib
import io
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_notebook(name, skip_contains=(), quiet=True, ns=None):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        nb = json.load(f)
    ns = {} if ns is None else ns
    ns.setdefault("__name__", "audit_nb")
    buf = io.StringIO()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(s in src for s in skip_contains):
            continue
        lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%"))]
        code = "\n".join(lines)
        if not code.strip():
            continue
        with (contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()):
            exec(compile(code, name, "exec"), ns)
    return ns


def _data_package():
    """The repo's `data` package (all pipeline code, formerly the notebook's cells), imported from ROOT."""
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import data
    if os.path.dirname(os.path.abspath(data.__file__)) != os.path.join(ROOT, "data"):
        raise ImportError(f"'data' resolves to {data.__file__}, not this repo's package ({ROOT}/data)")
    return data


def load_pipeline(quiet=True):
    """The pipeline (package data/, ex crypto_data_pipeline_v6.ipynb) in a fresh shared namespace, without its big
    self-test module. Same code path as the runner notebook: data.load_into(namespace)."""
    # RUN_HOURLY_4H_SELFTESTS=False: presets.py (خلية 20-ج سابقاً) تشغّل اختبارها السريع تلقائياً في Colab؛ هنا تُعرَّف دالته فقط.
    ns = {"__name__": "audit_nb", "RUN_HOURLY_4H_SELFTESTS": False}
    with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
        _data_package().load_into(ns, exclude=("selftests",))
    return ns


def _repo_package(name):
    """The repo's package ``name`` ('model' | 'trainer'), imported from ROOT (same guard as _data_package)."""
    import importlib
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    pkg = importlib.import_module(name)
    if os.path.dirname(os.path.abspath(pkg.__file__)) != os.path.join(ROOT, name):
        raise ImportError(f"'{name}' resolves to {pkg.__file__}, not this repo's package ({ROOT}/{name})")
    return pkg


def load_model(ns=None, quiet=True, selftests=False):
    """The model (package model/, ex model_v2 (1).ipynb) in a shared namespace; its import-time self-test module
    (``selftests``: defines and runs run_model_selftests) is left out unless ``selftests=True``.
    Same code path as the runner notebook: model.load_into(namespace)."""
    ns = {"__name__": "audit_nb"} if ns is None else ns
    ns.setdefault("__name__", "audit_nb")
    with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
        _repo_package("model").load_into(ns, exclude=() if selftests else ("selftests",))
    return ns


def load_trainer(ns=None, quiet=True, smoke_test=False, kfold=False):
    """The generic trainer (package trainer/, ex trainer_framework_v2.ipynb) in a shared namespace, without its
    load-time smoke test (a real tiny training) and without the optional K-Fold module unless asked for. Same code path
    as the runner notebook: trainer.load_into(namespace)."""
    ns = {"__name__": "audit_nb"} if ns is None else ns
    ns.setdefault("__name__", "audit_nb")
    exclude = tuple(m for m, keep in (("smoke_test", smoke_test), ("kfold", kfold)) if not keep)
    with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
        _repo_package("trainer").load_into(ns, exclude=exclude)
    return ns

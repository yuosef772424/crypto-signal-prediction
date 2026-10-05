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
    """The repo's code package ``name`` (evaluation, signal_eval, ...), imported from ROOT (never a same-named installed one)."""
    import importlib
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    pkg = importlib.import_module(name)
    if os.path.dirname(os.path.abspath(pkg.__file__)) != os.path.join(ROOT, name):
        raise ImportError(f"'{name}' resolves to {pkg.__file__}, not this repo's package ({ROOT}/{name})")
    return pkg


def load_evaluation(ns=None, exclude=(), quiet=True):
    """Package evaluation/ (ex chicks_v4_5_input_output_patterns.ipynb) executed into ``ns`` (a fresh dict by default):
    same code path as the runner notebook, evaluation.load_into(ns). ``exclude`` = module names to skip."""
    ns = {"__name__": "audit_nb"} if ns is None else ns
    with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
        _repo_package("evaluation").load_into(ns, exclude=exclude)
    return ns


def exec_evaluation_module(name, ns, quiet=True):
    """Execute ONE evaluation/ module into ``ns`` (what exec'ing a single chicks cell used to do)."""
    with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
        _repo_package("evaluation").exec_module(name, ns)
    return ns

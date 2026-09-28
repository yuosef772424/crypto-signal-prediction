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
    nb = json.load(open(os.path.join(ROOT, name)))
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


def load_pipeline(quiet=True):
    """crypto_data_pipeline_v6.ipynb without its big self-test cell (cell with run_all_selftests-style suite)."""
    return load_notebook("crypto_data_pipeline_v6.ipynb",
                         skip_contains=("اختبارات ذاتية لتعديلات هذا الدفتر",), quiet=quiet)

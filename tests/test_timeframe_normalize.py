"""S02 (pandas 4): ``core.timeframe.normalize_tf`` is the ONE place that fixes lowercase day/week timeframe strings; the data
package modules that build a ``pd.Timedelta`` / ``resample`` from a timeframe go through it and emit no Pandas4Warning."""
import os
import sys
import warnings

import pandas as pd
import pytest
from pandas.errors import Pandas4Warning

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "docs", "research", "audit"))

import _nbload  # noqa: E402
from core.timeframe import normalize_tf  # noqa: E402


@pytest.mark.parametrize("raw,want", [
    ("1d", "1D"), ("7d", "7D"), ("1w", "1W"), ("1D", "1D"), ("4h", "4h"), ("1h", "1h"), ("15min", "15min"),
    ("1M", "1M"), ("30D", "30D"), ("1d12h", "1d12h"), ("", ""), ("abc", "abc"),
])
def test_normalize_tf_table(raw, want):
    assert normalize_tf(raw) == want


@pytest.mark.parametrize("bad", [None, 1, 1.5, pd.Timedelta("1h")])
def test_normalize_tf_non_string_raises(bad):
    with pytest.raises(TypeError):
        normalize_tf(bad)


@pytest.mark.parametrize("tf", ["1d", "1D", "4h", "1h", "1w", "15min"])
def test_normalized_tf_builds_timedelta_and_resample_without_warning(tf):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Pandas4Warning)
        td = pd.Timedelta(normalize_tf(tf))
        pd.Series(1, index=pd.date_range("2025-01-01", periods=5, freq="h")).resample(normalize_tf(tf))
    assert td == pd.Timedelta(tf.upper() if tf[-1] in "dw" else tf)


def test_lowercase_day_unit_is_what_pandas_warns_about():
    """Guards the premise: if pandas stops warning, this helper is no longer needed."""
    with pytest.warns(Pandas4Warning):
        pd.Timedelta("1d")


def test_data_package_tf_helpers_accept_lowercase_day_without_warning():
    ns = _nbload.load_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("error", Pandas4Warning)
        assert ns["_tf_to_timedelta"]("1d") == pd.Timedelta("1D") == ns["_tf_to_timedelta"]("1D")
        assert ns["_tf_to_timedelta"]("4h") == pd.Timedelta(hours=4)


# ───────────────────────── guard: no un-normalised timeframe reaches pandas ─────────────────────────
import ast  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CODE_PACKAGES = ("core", "data", "model", "trainer", "evaluation", "signal_eval", "workflow", "discovery",
                  "cross_asset", "tools")
#: the ONLY exemption: Binance's own interval vocabulary ('1d') is not a pandas timeframe string.
_GUARD_EXEMPT_FILES = {"data/binance_client.py"}


def _calls_normalize_tf(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "normalize_tf"
               for n in ast.walk(node))


def _is_literal(node: ast.AST) -> bool:
    """Literal arguments (``'4h'``, ``1``, f-strings like ``f'{m}min'``) are out of scope: the author wrote the unit."""
    return isinstance(node, (ast.Constant, ast.JoinedStr))


def _unnormalised_tf_calls(source: str):
    """(line, what) for every ``pd.Timedelta(<expr>)`` (one positional arg, no keywords: ``unit=``/``hours=`` forms are numeric),
    ``.resample(<expr>)`` / ``resample(rule=<expr>)`` and ``freq=<expr>`` whose expression is not a literal and does not
    pass through ``normalize_tf``."""
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        exprs = []
        if name == "Timedelta" and len(node.args) == 1 and not node.keywords:
            exprs.append(("Timedelta", node.args[0]))
        elif name == "resample":
            if node.args:
                exprs.append(("resample", node.args[0]))
            exprs += [("resample", k.value) for k in node.keywords if k.arg == "rule"]
        exprs += [("freq=", k.value) for k in node.keywords if k.arg == "freq"]
        for what, e in exprs:
            if not _is_literal(e) and not _calls_normalize_tf(e):
                yield node.lineno, (f"{what}{ast.unparse(e)}" if what.endswith("=") else f"{what}({ast.unparse(e)})")


def _code_files():
    for pkg in _CODE_PACKAGES:
        for dirpath, _dirs, files in os.walk(os.path.join(_ROOT, pkg)):
            for fn in files:
                if fn.endswith(".py"):
                    path = os.path.join(dirpath, fn)
                    rel = os.path.relpath(path, _ROOT).replace(os.sep, "/")
                    if rel not in _GUARD_EXEMPT_FILES:
                        yield rel, path


def test_guard_detects_unnormalised_and_accepts_normalised():
    bad = "import pandas as pd\npd.Timedelta(tf)\nx.resample(tf)\npd.date_range(a, freq=tf)\nx.resample(rule=r)\n"
    good = ("pd.Timedelta(normalize_tf(tf))\nx.resample(normalize_tf(tf))\npd.Timedelta(1, 'ns')\npd.Timedelta('4h')\n"
            "pd.Timedelta(hours=1)\npd.date_range(a, freq='h')\npd.Timedelta(normalize_tf(s) if isinstance(s, str) else s)\n"
            "pd.date_range(a, freq=f'{m}min')\npd.Timedelta(x, unit='ns')\n")
    assert [w for _, w in _unnormalised_tf_calls(bad)] == ["Timedelta(tf)", "resample(tf)", "freq=tf", "resample(r)"]
    assert list(_unnormalised_tf_calls(good)) == []


def test_every_timeframe_reaching_pandas_goes_through_normalize_tf():
    """Guard (S02 review): a variable timeframe given to ``pd.Timedelta`` / ``.resample`` / ``freq=`` must pass
    ``normalize_tf`` (lowercase ``1d`` raises Pandas4Warning, an error in pandas 4). Only ``data/binance_client.py`` is exempt."""
    offenders = []
    for rel, path in _code_files():
        with open(path, encoding="utf-8") as fh:
            offenders += [f"{rel}:{ln}: {what}" for ln, what in _unnormalised_tf_calls(fh.read())]
    assert not offenders, "timeframe not normalised:\n  " + "\n  ".join(offenders)

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

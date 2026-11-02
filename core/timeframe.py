"""
PURPOSE:  The ONE normaliser of timeframe strings: `normalize_tf('1d') -> '1D'`, `normalize_tf('1w') -> '1W'`; every other string (`'4h'`, `'1D'`, `'15min'`, `'1M'`, ...) is returned unchanged. Every `pd.Timedelta(tf)` / `.resample(tf)` / `pd.date_range(freq=tf)` on a user- or config-supplied timeframe goes through it (pandas 3 emits `Pandas4Warning` for the lowercase day/week units and pandas 4 will reject them).
TAGS:     timeframe, tf, normalize_tf, pandas 4, Pandas4Warning, deprecation, Timedelta, resample, tier 0
PITFALLS: core imports no third-party package (tier 0): this is a pure string function, the caller builds the `pd.Timedelta` (`pd.Timedelta(normalize_tf(tf))`). Only the day and week units are mapped (the two lowercase spellings pandas 3 deprecates and whose replacement is uppercase); an unknown or compound string (`'1d12h'`, `'1M'`) is returned unchanged so the caller's own ValueError/TypeError handling (monthly frames -> "no fixed duration") behaves exactly as before. A non-string raises TypeError (no silent coercion). Binance interval strings (`'1d'` in `data/binance_client.py`) are a different vocabulary and are NOT normalised here.
"""
from __future__ import annotations

import re

__all__ = ["normalize_tf"]

_TF_RE = re.compile(r"^(\d+)([A-Za-z]+)$")
_UNIT_FIX = {"d": "D", "w": "W"}


def normalize_tf(tf: str) -> str:
    """``'1d'`` -> ``'1D'``, ``'2w'`` -> ``'2W'``; anything else (incl. ``'4h'``, ``'1D'``, ``'1M'``) unchanged."""
    if not isinstance(tf, str):
        raise TypeError(f"timeframe must be a string, got {type(tf).__name__}: {tf!r}")
    m = _TF_RE.match(tf)
    if m is None:
        return tf
    count, unit = m.groups()
    return count + _UNIT_FIX.get(unit, unit)

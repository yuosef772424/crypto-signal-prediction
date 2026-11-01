"""
PURPOSE:  The PredictionTable schema as pure constants (tier 0, stdlib + core only): the required / optional columns of a table of model predictions, the time columns, the key that must be unique, the head-output column prefix and the sidecar file name. The writer and the validator live in stats/prediction_table.py.
TAGS:     prediction table, prediction schema, prediction_made_at, label_available_at, train_end, fold, split_role, identity_hash, REQUIRED_COLUMNS, KEY_COLUMNS, HEAD_OUTPUT_PREFIX, SCHEMA_VERSION, point in time, tier 0
PITFALLS: Time semantics (one meaning each, never re-derived by a consumer): `timestamp` = the window end bar (lc[:, TS_COL]); `prediction_made_at` = when the prediction could be used (a historical row: the same instant as `timestamp`; a live row: the wall clock); `label_available_at` = `timestamp` + horizon bars of the timeframe (the last bar the label reads, same convention as data/split.py `label_end`); `train_end` = the last timestamp the PRODUCING model could have seen (its train split). A non-train row made before `train_end` is a leak by construction. Head outputs are the open part of the schema: columns named `out_<raw output key>`; everything else not listed here is an unknown column and raises. Appending a column = bump SCHEMA_VERSION and list it here (never edit a written table).
"""
from __future__ import annotations

from typing import Tuple

from core.roles import ROLES

__all__ = ["SCHEMA_VERSION", "REQUIRED_COLUMNS", "OPTIONAL_COLUMNS", "TIME_COLUMNS", "KEY_COLUMNS", "HEAD_OUTPUT_PREFIX", "META_SUFFIX",
           "SPLIT_ROLES", "NON_TRAIN_ROLES", "head_output_column"]

SCHEMA_VERSION = 1

#: columns every table carries, in this order at the front of a written table (head outputs follow).
REQUIRED_COLUMNS: Tuple[str, ...] = ("asset", "timestamp", "prediction_made_at", "label_available_at", "train_end", "fold",
                                     "split_role", "identity_hash")

#: columns a table may carry besides the required ones and the head outputs (none today: the feedback loop adds `decision_*` / `outcome_*` later).
OPTIONAL_COLUMNS: Tuple[str, ...] = ()

#: the columns holding instants (stored as UTC datetimes).
TIME_COLUMNS: Tuple[str, ...] = ("timestamp", "prediction_made_at", "label_available_at", "train_end")

#: one row per (asset, timestamp, fold): the key that must be unique and never missing.
KEY_COLUMNS: Tuple[str, ...] = ("asset", "timestamp", "fold")

#: a head-output column is `out_<raw output key>` (e.g. `out_high`, `out_close_p_up`, `out_high_epistemic`).
HEAD_OUTPUT_PREFIX = "out_"

#: the sidecar of a written table: `<table path>.meta.json`.
META_SUFFIX = ".meta.json"

#: roles a row may carry (core/roles.py).
SPLIT_ROLES: Tuple[str, ...] = ROLES
#: roles whose predictions must be out of sample: made at or after the producing model's `train_end`.
NON_TRAIN_ROLES: Tuple[str, ...] = tuple(r for r in ROLES if r != "train")


def head_output_column(raw_key: str) -> str:
    """``head_output_column("high")`` -> ``"out_high"``: the ONLY place that spells a head-output column."""
    if not isinstance(raw_key, str) or not raw_key:
        raise ValueError(f"raw output key must be a non-empty str, got {raw_key!r}")
    return f"{HEAD_OUTPUT_PREFIX}{raw_key}"

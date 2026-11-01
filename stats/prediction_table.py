"""
PURPOSE:  The ONE PredictionTable: build_table (arrays -> table with the point-in-time columns), validate_table (the contract), write_table / read_table (csv.gz + `<path>.meta.json`, both validate). A table row = one prediction of one model for one (asset, timestamp), with prediction_made_at, label_available_at, train_end, fold, split_role, identity_hash and its head outputs.
TAGS:     prediction table, PredictionTable, build_table, validate_table, write_table, read_table, label_available_at, prediction_made_at, train_end, fold, split_role, identity_hash, leakage check, point in time, stage contract, hardening C5
PITFALLS: The schema names live in core/prediction_schema.py (constants), the semantics of each time column are documented there. validate_table raises PredictionTableError on: a missing / unknown column, a NaN key (asset, timestamp, fold) or NaN time, a duplicate (asset, timestamp, fold), a role that is not in core.roles.ROLES, label_available_at <= prediction_made_at, and prediction_made_at < train_end on any non-train role (the leak check: an out-of-sample prediction made before the model's train end). A historical table sets prediction_made_at = timestamp (default of build_table); do not backdate it. Times are stored as UTC datetimes; numeric inputs are nanoseconds since the epoch (last_candles timestamps are float64 ns: non-finite ones raise, never become NaT). Head outputs are `out_<raw output key>` columns, one number per row (a wider output raises). The table does not decode anything: raw model outputs only (decoding stays in evaluation). Tier 1: imports only core + numpy/pandas (tools/check_deps.py), so evaluation, workflow and tools can all use it.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from core.prediction_schema import (HEAD_OUTPUT_PREFIX, KEY_COLUMNS, META_SUFFIX, NON_TRAIN_ROLES, OPTIONAL_COLUMNS, REQUIRED_COLUMNS,
                                    SCHEMA_VERSION, SPLIT_ROLES, TIME_COLUMNS, head_output_column)

__all__ = ["PredictionTableError", "PredictionTableSpec", "to_utc", "label_available_at", "build_table", "validate_table", "write_table", "read_table",
           "meta_path"]


class PredictionTableError(ValueError):
    """A prediction table violates its contract (schema, keys, time semantics)."""


def to_utc(values: Any, *, what: str = "timestamp") -> pd.DatetimeIndex:
    """``values`` as a UTC DatetimeIndex. Datetime-like input is localized/converted to UTC; numeric input is nanoseconds since the epoch.
    A missing / non-finite value raises (a NaN key is never turned into NaT silently)."""
    arr = np.asarray(values)
    if arr.dtype.kind in "iuf":
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise PredictionTableError(f"{what}: {int((~np.isfinite(arr)).sum())} non-finite value(s)")
        idx = pd.to_datetime(arr.astype("int64"), unit="ns", utc=True)
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(values, utc=True))
    if idx.isna().any():
        raise PredictionTableError(f"{what}: {int(idx.isna().sum())} missing value(s)")
    return idx.as_unit("ns")


def label_available_at(timestamps: Any, horizon_bars: int, timeframe: str) -> pd.DatetimeIndex:
    """``timestamp + horizon_bars`` bars of ``timeframe`` (e.g. ``"1h"``): the last bar the label reads (data/split.py ``label_end``)."""
    if isinstance(horizon_bars, bool) or not isinstance(horizon_bars, (int, np.integer)) or horizon_bars < 1:
        raise PredictionTableError(f"horizon_bars must be an int >= 1, got {horizon_bars!r}")
    return to_utc(timestamps) + horizon_bars * pd.Timedelta(timeframe)


@dataclasses.dataclass(frozen=True)
class PredictionTableSpec:
    """What an evaluation call needs to write its predictions as a PredictionTable (opt-in: no spec = nothing is written).

    ``path``: ``.csv`` / ``.csv.gz`` (the sidecar ``<path>.meta.json`` is written next to it). ``split_role``: the role of the data evaluated.
    ``train_end``: the last timestamp the producing model's train split holds. ``fold``: the fold index (0 = the single global split).
    ``identity_hash``: the run's ``config_fingerprint`` (trainer identity.json). ``horizon_bars`` / ``timeframe``: the label horizon and the
    base timeframe (``label_available_at = timestamp + horizon_bars * timeframe``). ``meta``: extra sidecar entries (commit, settings...)."""
    path: str
    split_role: str
    train_end: Any
    identity_hash: str
    horizon_bars: int
    timeframe: str
    fold: int = 0
    meta: Optional[Mapping[str, Any]] = None

    def __post_init__(self):
        if not (isinstance(self.path, str) and (self.path.endswith(".csv") or self.path.endswith(".csv.gz"))):
            raise PredictionTableError(f"PredictionTableSpec.path must be a .csv / .csv.gz path, got {self.path!r}")
        if self.split_role not in SPLIT_ROLES:
            raise PredictionTableError(f"PredictionTableSpec.split_role {self.split_role!r} unknown; roles: {list(SPLIT_ROLES)}")
        if not (isinstance(self.identity_hash, str) and self.identity_hash):
            raise PredictionTableError("PredictionTableSpec.identity_hash must be a non-empty str (the run's config_fingerprint)")
        if isinstance(self.fold, bool) or not isinstance(self.fold, (int, np.integer)) or self.fold < 0:
            raise PredictionTableError(f"PredictionTableSpec.fold must be an int >= 0, got {self.fold!r}")
        label_available_at([0], self.horizon_bars, self.timeframe)        # horizon and timeframe parse, or raise
        to_utc([self.train_end], what="train_end")


def _column_1d(raw_key: str, values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape[0] != n:
        raise PredictionTableError(f"head output {raw_key!r}: {arr.shape[0]} rows, expected {n}")
    flat = arr.reshape(n, -1)
    if flat.shape[1] != 1:
        raise PredictionTableError(f"head output {raw_key!r} has {flat.shape[1]} values per row; a table column holds one (split it into named outputs)")
    return flat[:, 0]


def build_table(*, asset: Any, timestamp: Any, outputs: Mapping[str, Any], split_role: str, train_end: Any, fold: int,
                identity_hash: str, horizon_bars: int, timeframe: str, prediction_made_at: Any = None) -> pd.DataFrame:
    """A validated table from per-row arrays.

    ``asset``: per-row names (a str broadcasts). ``timestamp``: window-end instants (datetimes or ns numbers). ``outputs``: ``{raw output key:
    per-row array}`` -> columns ``out_<key>``. ``split_role`` / ``train_end`` / ``fold`` / ``identity_hash``: one value for the table.
    ``prediction_made_at``: None = the historical default (equal to ``timestamp``)."""
    ts = to_utc(timestamp)
    n = len(ts)
    assets = np.full(n, asset, dtype=object) if isinstance(asset, str) else np.asarray(asset, dtype=object)
    if len(assets) != n:
        raise PredictionTableError(f"asset: {len(assets)} rows, expected {n}")
    made = ts if prediction_made_at is None else to_utc(prediction_made_at, what="prediction_made_at")
    if len(made) != n:
        raise PredictionTableError(f"prediction_made_at: {len(made)} rows, expected {n}")
    if np.ndim(train_end) != 0:
        raise PredictionTableError("train_end must be a single instant")
    train_end_ts = to_utc([train_end], what="train_end")[0]
    cols: Dict[str, Any] = {
        "asset": assets, "timestamp": ts, "prediction_made_at": made,
        "label_available_at": label_available_at(ts, horizon_bars, timeframe),
        "train_end": pd.DatetimeIndex([train_end_ts] * n), "fold": np.full(n, fold, dtype="int64"),
        "split_role": np.full(n, split_role, dtype=object), "identity_hash": np.full(n, identity_hash, dtype=object),
    }
    for key, values in outputs.items():
        cols[head_output_column(key)] = _column_1d(key, values, n)
    df = pd.DataFrame(cols)
    return validate_table(df)


def _is_str_column(col: pd.Series) -> bool:
    return bool(col.map(lambda v: isinstance(v, str)).all())


def _utc_datetime_problem(col: pd.Series) -> Optional[str]:
    if not pd.api.types.is_datetime64_any_dtype(col):
        return f"dtype {col.dtype} is not a datetime"
    if getattr(col.dt, "tz", None) is None or str(col.dt.tz) != "UTC":
        return f"timezone {getattr(col.dt, 'tz', None)} is not UTC"
    return None


def validate_table(df: pd.DataFrame, *, head_outputs: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Return ``df`` unchanged if it honours the PredictionTable contract, else raise PredictionTableError.

    ``head_outputs``: optional raw output keys the table may carry (None = any ``out_*`` column); an ``out_*`` column outside it is unknown."""
    if not isinstance(df, pd.DataFrame):
        raise PredictionTableError(f"a PredictionTable is a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise PredictionTableError(f"missing required column(s) {missing}; required: {list(REQUIRED_COLUMNS)}")
    allowed_outputs = None if head_outputs is None else {head_output_column(k) for k in head_outputs}
    unknown = []
    for c in df.columns:
        if c in REQUIRED_COLUMNS or c in OPTIONAL_COLUMNS:
            continue
        if isinstance(c, str) and c.startswith(HEAD_OUTPUT_PREFIX) and len(c) > len(HEAD_OUTPUT_PREFIX) \
                and (allowed_outputs is None or c in allowed_outputs):
            continue
        unknown.append(c)
    if unknown:
        raise PredictionTableError(f"unknown column(s) {unknown}; allowed: {list(REQUIRED_COLUMNS)}, {list(OPTIONAL_COLUMNS)} and "
                                   f"head outputs named '{HEAD_OUTPUT_PREFIX}<raw output key>'"
                                   + (f" restricted to {sorted(allowed_outputs)}" if allowed_outputs is not None else ""))
    if df.columns.duplicated().any():
        raise PredictionTableError(f"duplicate column name(s) {sorted(set(df.columns[df.columns.duplicated()]))}")
    if len(df) == 0:
        raise PredictionTableError("empty table: a prediction table has at least one row")

    for c in REQUIRED_COLUMNS:                                   # missing keys / times / labels first: every later check assumes none
        n_na = int(df[c].isna().sum())
        if n_na:
            raise PredictionTableError(f"column {c!r} has {n_na} missing value(s); keys and time columns are never NaN")
    for c in TIME_COLUMNS:
        problem = _utc_datetime_problem(df[c])
        if problem:
            raise PredictionTableError(f"column {c!r}: {problem}")
    if not pd.api.types.is_integer_dtype(df["fold"]):
        raise PredictionTableError(f"column 'fold' must be integer, got {df['fold'].dtype}")
    for c in ("asset", "split_role", "identity_hash"):
        if not _is_str_column(df[c]):
            raise PredictionTableError(f"column {c!r} must hold strings")
    if (df["identity_hash"] == "").any():
        raise PredictionTableError("identity_hash is empty on some rows: every table is tied to the run that produced it")
    bad_roles = sorted(set(df["split_role"]) - set(SPLIT_ROLES))
    if bad_roles:
        raise PredictionTableError(f"unknown split_role value(s) {bad_roles}; roles: {list(SPLIT_ROLES)}")

    dup = df.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if dup.any():
        first = df.loc[dup, list(KEY_COLUMNS)].iloc[0].tolist()
        raise PredictionTableError(f"{int(dup.sum())} row(s) share a key {list(KEY_COLUMNS)}, e.g. {first}")

    late = df["label_available_at"] <= df["prediction_made_at"]
    if late.any():
        i = int(np.flatnonzero(late.to_numpy())[0])
        raise PredictionTableError(f"{int(late.sum())} row(s) have label_available_at <= prediction_made_at (the label would be known when "
                                   f"the prediction is made), e.g. row {i}: {df['label_available_at'].iloc[i]} <= {df['prediction_made_at'].iloc[i]}")
    leak = df["split_role"].isin(NON_TRAIN_ROLES) & (df["prediction_made_at"] < df["train_end"])
    if leak.any():
        i = int(np.flatnonzero(leak.to_numpy())[0])
        raise PredictionTableError(
            f"{int(leak.sum())} non-train row(s) were made before the producing model's train_end (leak), e.g. row {i} "
            f"role {df['split_role'].iloc[i]!r}: prediction_made_at {df['prediction_made_at'].iloc[i]} < train_end {df['train_end'].iloc[i]}")
    return df


def meta_path(path: str) -> str:
    """The sidecar path of the table at ``path``."""
    return str(path) + META_SUFFIX


def write_table(path: str, df: pd.DataFrame, meta: Optional[Mapping[str, Any]] = None) -> str:
    """Validate ``df`` and write it to ``path`` (``.csv`` / ``.csv.gz``) with the sidecar ``<path>.meta.json`` (schema_version, columns,
    row count, roles, folds, identity hashes + the caller's ``meta``: commit, settings, data fingerprint ...). Returns ``path``."""
    path = str(path)
    if not (path.endswith(".csv") or path.endswith(".csv.gz")):
        raise PredictionTableError(f"unsupported table file {path!r}: use .csv or .csv.gz")
    validate_table(df)
    head = [c for c in REQUIRED_COLUMNS] + [c for c in df.columns if c not in REQUIRED_COLUMNS]
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    df[head].to_csv(path, index=False)
    side = {"schema_version": SCHEMA_VERSION, "columns": head, "n_rows": int(len(df)),
            "split_roles": sorted(set(df["split_role"])), "folds": sorted(int(f) for f in set(df["fold"])),
            "identity_hashes": sorted(set(df["identity_hash"])), **dict(meta or {})}
    with open(meta_path(path), "w", encoding="utf-8") as f:
        json.dump(side, f, indent=2, sort_keys=True, default=str)
    return path


def read_table(path: str, *, head_outputs: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read a table written by :func:`write_table` and validate it (a hand-edited file that breaks the contract raises). The sidecar must
    exist and carry this SCHEMA_VERSION."""
    path = str(path)
    side_path = meta_path(path)
    if not os.path.exists(side_path):
        raise PredictionTableError(f"no sidecar {side_path}: a table without its meta is not read")
    with open(side_path, encoding="utf-8") as f:
        side = json.load(f)
    if side["schema_version"] != SCHEMA_VERSION:
        raise PredictionTableError(f"table schema_version {side['schema_version']} != {SCHEMA_VERSION}")
    df = pd.read_csv(path, parse_dates=list(TIME_COLUMNS), date_format="ISO8601")
    for c in TIME_COLUMNS:
        df[c] = pd.to_datetime(df[c], utc=True).dt.as_unit("ns")
    return validate_table(df, head_outputs=head_outputs)

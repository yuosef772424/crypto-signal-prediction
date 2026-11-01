"""
PURPOSE:  SplitPlan: the time boundaries of ONE split (train / optional calibration / val / test / optional sealed holdout) and the embargo between them as one frozen object, with the masks it implies over sample timestamps and a validity check (roles in time order, every gap >= the embargo). The default plan (SplitPlan.from_dates) reproduces data.split_data's and rolling_splits' masks exactly; CalibrationCarve + with_calibration opt a calibration split in.
TAGS:     SplitPlan, CalibrationCarve, split plan, plan masks, from_dates, with_calibration, calibration split, calibration carve, train_tail, val_head, embargo, role intervals, leakage control, hardening C6
PITFALLS: Intervals are (lo exclusive, hi inclusive) over the sample TIMESTAMP (window end); the embargo between two consecutive roles is lo_next - hi_prev >= embargo (checked by `check`). `with_calibration` carves a calibration slice by TIME from the kept rows of its source ("train_tail": the last `share` of train, so train shrinks and an embargo opens between train and calibration; "val_head": the first `share` of val, so val starts later); the embargo is on BOTH sides and the plan raises if the carve leaves a role empty. The default plan (no calibration) is exactly today's rule: train ts <= train_end | val train_end+embargo < ts <= val_end | test ts > val_end+embargo (and ts < holdout_start - embargo) | holdout ts >= holdout_start. Executed into the one shared pipeline namespace by data/_loader.py (never imported on its own); self-contained on purpose (uses no name of data/split.py: it is loaded before it).
"""
import dataclasses as _dc
from typing import Dict, Optional, Tuple

import numpy as _np
import pandas as _pd

from core.roles import CALIBRATION_CARVES

#: roles of a plan in time order (calibration only when the plan has one; holdout only when a holdout_start is set).
PLAN_ROLE_ORDER = ('train', 'calibration', 'val', 'test', 'holdout')


@_dc.dataclass(frozen=True)
class CalibrationCarve:
    """Opt-in recipe of the calibration split: ``source`` in CALIBRATION_CARVES, ``share`` in (0, 0.5] of the source's kept rows."""
    source: str = 'train_tail'
    share: float = 0.1

    def __post_init__(self):
        if self.source not in CALIBRATION_CARVES:
            raise ValueError(f"CalibrationCarve.source {self.source!r} unknown; known: {CALIBRATION_CARVES}")
        if isinstance(self.share, bool) or not isinstance(self.share, (int, float)) or not 0.0 < self.share <= 0.5:
            raise ValueError(f"CalibrationCarve.share must be a number in (0, 0.5], got {self.share!r}")


@_dc.dataclass(frozen=True)
class SplitPlan:
    """Time boundaries of one split. ``train``: ``train_start < ts <= train_end`` (train_start None = from the beginning).
    ``calibration``: optional ``(lo, hi)``. ``val``: ``(lo, hi)``. ``test``: ``ts > test_lo`` (and ``ts <= test_end`` when given, and
    ``ts < holdout_start - embargo`` when a holdout is sealed). ``holdout``: ``ts >= holdout_start``. All ``Timestamp`` (UTC)."""
    embargo: object                      # pd.Timedelta
    train_end: object                    # pd.Timestamp
    val: Tuple[object, object]
    test_lo: object
    train_start: Optional[object] = None
    calibration: Optional[Tuple[object, object]] = None
    test_end: Optional[object] = None
    holdout_start: Optional[object] = None

    @classmethod
    def from_dates(cls, train_end, val_end, embargo, train_start=None, test_end=None, holdout_start=None):
        """The default plan: today's rule (``split_data`` global_time and every ``rolling_splits`` window)."""
        return cls(embargo=embargo, train_end=train_end, val=(train_end + embargo, val_end), test_lo=val_end + embargo,
                   train_start=train_start, test_end=test_end, holdout_start=holdout_start)

    @property
    def val_end(self):
        return self.val[1]

    def intervals(self) -> Dict[str, Tuple[Optional[object], Optional[object]]]:
        """``{role: (lo exclusive | None, hi inclusive | None)}`` in time order; test's hi is its test_end (None = open); holdout's lo is
        ``holdout_start`` (inclusive, listed as lo - 1ns exclusive is NOT used: see :meth:`masks`)."""
        out = {'train': (self.train_start, self.train_end)}
        if self.calibration is not None:
            out['calibration'] = self.calibration
        out['val'] = self.val
        out['test'] = (self.test_lo, self.test_end)
        return out

    def check(self, min_embargo=None):
        """Raise ValueError unless the roles are in time order, none is empty as an interval, and every gap between consecutive roles is at
        least the embargo (the leakage rule, stated once). ``min_embargo`` (a Timedelta, e.g. the data's embargo_duration): the gap must also
        reach it, so a plan built with a smaller embargo than the data needs is refused. Returns self."""
        need = self.embargo if min_embargo is None else max(self.embargo, min_embargo)
        iv = self.intervals()
        names = list(iv)
        for name in names:
            lo, hi = iv[name]
            if lo is not None and hi is not None and not lo < hi:
                raise ValueError(f"SplitPlan: role {name!r} has an empty interval ({lo} , {hi}]")
        for a, b in zip(names, names[1:]):
            hi_a, lo_b = iv[a][1], iv[b][0]
            if hi_a is None or lo_b is None:
                continue
            if lo_b - hi_a < need:
                raise ValueError(f"SplitPlan: {a!r} ends {hi_a} and {b!r} starts after {lo_b}: the gap {lo_b - hi_a} is shorter than the "
                                 f"embargo {need} (labels / windows would overlap)")
        if self.holdout_start is not None:
            last_hi = self.test_end if self.test_end is not None else self.test_lo
            if self.holdout_start - self.embargo <= last_hi and self.test_end is not None:
                raise ValueError(f"SplitPlan: holdout_start {self.holdout_start} is not an embargo after test_end {self.test_end}")
            if self.holdout_start <= self.val_end + self.embargo:
                raise ValueError(f"SplitPlan: holdout_start {self.holdout_start} is not after val_end + embargo ({self.val_end + self.embargo})")
        return self

    def masks(self, ts) -> Dict[str, object]:
        """``{role: boolean mask over the DatetimeIndex ts}`` for the roles of the plan ('holdout' only when sealed). Same comparisons, in
        the same order, as the masks `_split_global_time` / `rolling_splits` built by hand."""
        out = {}
        train = ts <= self.train_end
        if self.train_start is not None:
            train = (ts > self.train_start) & train
        out['train'] = train
        if self.calibration is not None:
            out['calibration'] = (ts > self.calibration[0]) & (ts <= self.calibration[1])
        out['val'] = (ts > self.val[0]) & (ts <= self.val[1])
        test = ts > self.test_lo
        if self.test_end is not None:
            test = test & (ts <= self.test_end)
        if self.holdout_start is not None:
            test = test & (ts < self.holdout_start - self.embargo)        # the same embargo before the sealed holdout
            out['holdout'] = ts >= self.holdout_start
        out['test'] = test
        return {role: out[role] for role in PLAN_ROLE_ORDER if role in out}

    def with_calibration(self, carve: CalibrationCarve, ts) -> 'SplitPlan':
        """A plan with a calibration split carved from the kept rows of ``carve.source`` by time (``ts`` = every sample timestamp, a
        DatetimeIndex). ``train_tail``: the last ``share`` of the train rows become calibration, train ends one embargo earlier.
        ``val_head``: the first ``share`` of the val rows become calibration, val starts one embargo after it. An embargo is on both sides
        of the calibration slice; a plan that cannot hold the slice (an empty role) raises."""
        if self.calibration is not None:
            raise ValueError("SplitPlan already has a calibration split")
        e = self.embargo
        if carve.source == 'train_tail':
            rows = ts[(ts <= self.train_end) if self.train_start is None else ((ts > self.train_start) & (ts <= self.train_end))]
            if len(rows) < 2:
                raise ValueError("SplitPlan.with_calibration: train has fewer than 2 rows")
            ordered = _np.sort(rows.as_unit('ns').asi8)
            first_cal = ordered[min(len(ordered) - 1, int(len(ordered) * (1.0 - carve.share)))]      # first calibration row (ns)
            first_cal = _pd.Timestamp(int(first_cal), tz='UTC')
            new_train_end = first_cal - e - _pd.Timedelta(1, 'ns')
            plan = _dc.replace(self, train_end=new_train_end, calibration=(new_train_end + e, self.train_end))
        else:
            rows = ts[(ts > self.val[0]) & (ts <= self.val[1])]
            if len(rows) < 2:
                raise ValueError("SplitPlan.with_calibration: val has fewer than 2 rows")
            ordered = _np.sort(rows.as_unit('ns').asi8)
            last_cal = ordered[max(0, min(len(ordered) - 1, int(len(ordered) * carve.share) - 1))]   # last calibration row (ns)
            last_cal = _pd.Timestamp(int(last_cal), tz='UTC')
            plan = _dc.replace(self, calibration=(self.val[0], last_cal), val=(last_cal + e, self.val[1]))
        return plan.check()

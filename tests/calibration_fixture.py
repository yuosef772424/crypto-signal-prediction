"""Deterministic synthetic flat tables for the calibrator tests (tests/test_calibrator_role.py, golden_calibrator_outputs.json).

The golden file was produced from the branch head BEFORE the Calibrator component existed (the old
`fit_confidence_calibrators` / `apply_confidence_calibration` code path) on exactly these tables: do not change a number here
without regenerating the golden from an old checkout (`git show 6c30e22:evaluation/trust_calibration.py` has the old math).
`np.random.RandomState` is the frozen legacy generator (same stream on every numpy version).
"""
import numpy as np
import pandas as pd

GRID = np.linspace(0.0, 1.0, 101)


def _target(rs, name, n, conf_kind, with_class):
    if conf_kind == "informative":
        conf = rs.uniform(0.3, 0.95, n)
        correct = (rs.random_sample(n) < 0.3 + 0.35 * conf).astype(int)
    elif conf_kind == "uninformative":
        conf = rs.uniform(0.5, 0.95, n)
        correct = (rs.random_sample(n) < 0.55).astype(int)
    else:                                                     # constant
        conf = np.full(n, 0.72)
        correct = (rs.random_sample(n) < 0.55).astype(int)
    d = {"target": name, "confidence": conf, "correct": correct}
    if with_class:
        p_up = rs.uniform(0.05, 0.95, n)
        ok = rs.random_sample(n) < 0.5 + 0.6 * np.abs(p_up - 0.5)
        up = np.where(ok, p_up >= 0.5, p_up < 0.5)
        d.update(p_up=p_up, price_change=np.where(up, 1.0, -1.0))
    return pd.DataFrame(d)


def multi_target_table(seed, n=600):
    """high: informative confidence; low: uninformative confidence + class head (-> class_margin); close: constant confidence,
    no class head (-> base_rate); tiny: 30 rows < min_n (-> base_rate)."""
    rs = np.random.RandomState(seed)
    parts = [_target(rs, "high", n, "informative", True), _target(rs, "low", n, "uninformative", True),
             _target(rs, "close", n, "constant", False), _target(rs, "tiny", 30, "informative", False)]
    return pd.concat(parts, ignore_index=True)


VAL_SEED, TEST_SEED = 101, 202

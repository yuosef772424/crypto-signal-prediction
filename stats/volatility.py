"""
PURPOSE:  Past-only volatility estimators shared by the target code and the evaluation: ewma_std (RiskMetrics EWMA standard deviation of a window of returns, newest return weighted most).
TAGS:     ewma_std, volatility, ewma, past-only, volret, target normalisation
PITFALLS: Zero-mean second moment (RiskMetrics convention): sqrt(sum w_k r_k^2 / sum w_k) with w = lam ** age, age 0 = the LAST column. Rows with a non-finite return are NaN (the caller floors / drops them); the floor is the caller's (core.schema.VOLRET_FLOOR). Past-only means the caller passes only returns known at entry.
"""
import numpy as np


def ewma_std(returns, lam: float = 0.94) -> np.ndarray:
    """EWMA standard deviation along the last axis of ``returns`` (n, T), oldest column first: ``sqrt(sum_k w_k r_k^2 / sum_k w_k)``
    with ``w_k = lam ** age`` (age 0 = the last column). Returns an (n,) float64 array (NaN where any return of the row is non-finite)."""
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam!r}")
    r = np.asarray(returns, dtype="float64")
    if r.ndim != 2 or r.shape[1] < 1:
        raise ValueError(f"returns must be (n, T>=1), got shape {r.shape}")
    w = lam ** np.arange(r.shape[1] - 1, -1, -1, dtype="float64")
    return np.sqrt((r * r) @ w / w.sum())

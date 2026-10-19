"""Shared fixture of tests/test_normalizer.py: builds dataset + split for each price_norm mode on tiny synthetic data
and fingerprints them (sha256 over the arrays and the stamped meta). The pinned values were measured on the N1 head
BEFORE the Normalizer refactor."""
import hashlib
import json

import numpy as np

from tests import test_audit_round2 as r2
from tests.test_price_norm_mode import _build, _cfg, _frames, _split

NO_REG = {f"{t}_reg": False for t in ("high", "low", "close")}


def mode_cfg(mode):
    """Config for a registered normalizer name (also 'pct_change' for the legacy alias)."""
    if mode == "window_scale":
        return _cfg(None, with_ohlc=False)
    if mode == "logret_sigma":
        return _cfg(mode, enabled_heads=NO_REG)
    return _cfg(mode)


def _digest(h, d):
    for k in sorted(d):
        v = d[k]
        if k == "y":
            for kk in sorted(v):
                h.update(("y." + kk).encode())
                h.update(np.ascontiguousarray(v[kk]).tobytes())
        elif k.startswith("X_") or k in ("base_params", "last_candles"):
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif k == "price_norm_meta":
            h.update(json.dumps(v, sort_keys=True).encode())


def fingerprint_dataset(ds):
    h = hashlib.sha256()
    _digest(h, ds)
    for k in ("price_norm_mode", "price_norm_state", "price_norm_meta"):
        h.update(f"{k}={json.dumps(ds.get(k), sort_keys=True)}".encode())
    return h.hexdigest()


def fingerprint_splits(tr, va, te):
    h = hashlib.sha256()
    for sp in (tr, va, te):
        for part in ([sp] if "y" in sp else [sp[k] for k in sorted(sp)]):
            _digest(h, part)
    return h.hexdigest()


def run_mode(mode):
    cfg = mode_cfg(mode)
    ds = _build(cfg)
    tr, va, te = _split(ds, cfg)
    return cfg, ds, (tr, va, te)


if __name__ == "__main__":
    for m in ("window_scale", "pct_change", "logret_std", "logret_sigma"):
        cfg, ds, sp = run_mode(m) if m != "pct_change" else (None, None, None)
        if m == "pct_change":
            cfg = _cfg("pct_change", with_ohlc=False)
            ds = _build(cfg)
            sp = _split(ds, cfg)
        print(m, fingerprint_dataset(ds), fingerprint_splits(*sp))

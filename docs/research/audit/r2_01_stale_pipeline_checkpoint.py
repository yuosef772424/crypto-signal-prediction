"""Repro R2-01: the per-asset pipeline checkpoint (build_dataset_from_loader(checkpoint_dir=...)) is reused after the raw
data grew, so the rebuilt dataset silently lacks every newer sample.

_checkpoint_fingerprint hashes configuration only (feature_order, windows, stride, ...). It says nothing about the raw
file (row count / last timestamp) nor about the phase-2 archives (funding / OI / metrics / 15m), so re-fetching history
into the same Drive folder and rebuilding with the same checkpoint_dir returns the OLD arrays. Round-1 R1-02 fixed the
same class of defect for panel-training resume (content_hash) but not for this cache.

Expected (after fix): rebuilding on longer raw data with the same checkpoint_dir yields samples up to the new last day
(a stale entry is detected and recomputed).
Actual (b14b9bb): identical to the old build; newest samples missing.

Run:  python docs/research/audit/r2_01_stale_pipeline_checkpoint.py   (~10 s)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r2_synth import *  # noqa: E402,F401,F403

rng = np.random.default_rng(3)
coins = ["BTCUSDT", "AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]
h_full = {c: to_1h(make_coin_15m(rng, "2025-01-01", "2025-04-01")) for c in coins}
h_old = {c: h[h.index < "2025-03-01"] for c, h in h_full.items()}


def setup():
    ns = fresh_ns()
    C = ns["CONFIG"]
    C["funding_rate"]["enabled"] = C["open_interest"]["enabled"] = False
    C["feature_order"] = None
    ns["refresh_features"]()
    return ns


ck = tempfile.mkdtemp()
mx = lambda d: pd.to_datetime(d["last_candles"][:, 3].astype("int64"), utc=True).max()  # noqa: E731
old = build(setup(), ck, h_old, coins, phase2=False, checkpoint_dir=ck)
new = build(setup(), ck, h_full, coins, phase2=False, checkpoint_dir=ck)            # same checkpoint_dir, longer raw data
ref = build(setup(), ck, h_full, coins, phase2=False, checkpoint_dir=None)         # ground truth: no cache
print(f"samples  old build {len(old['base_params'])} | rebuild with same checkpoint_dir {len(new['base_params'])} | no-cache {len(ref['base_params'])}")
print(f"last ts  old build {mx(old)} | rebuild {mx(new)} | no-cache {mx(ref)}")
assert len(new["base_params"]) == len(ref["base_params"]) and mx(new) == mx(ref), \
    "DEFECT: stale per-asset checkpoint reused although the raw data changed"
print("PASS")

"""Repro 02: a panel run_dir trained on dataset A is silently reused for a different dataset B.

run_panel_variant's resume fingerprint holds only (variant, model cfg, train cfg, sample counts, first/last group).
A rebuilt dataset with the same shape and date range but different features/labels (e.g. the same
1h_s8 build after a pipeline fix) matches that fingerprint, so the finished run is "resumed": no
training happens and B's signals are produced with weights fitted on A.

Expected (after fix): run_panel_variant on B in A's run_dir raises (fingerprint mismatch) or retrains.
Actual (commit c62b4f0): returns normally with 0 new epochs.

Run:  python docs/research/audit/repro_02_resume_fingerprint_ignores_data.py   (~30 s, tiny encoder, CPU)
"""
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import numpy as np  # noqa: E402

from cross_asset.data import PanelSplit  # noqa: E402
from cross_asset.experiment import run_panel_variant  # noqa: E402
from cross_asset.selftest import _TinyBase, synthetic_split  # noqa: E402

SEQ, NF = 8, 5


def as_ps(ps, X, flip):
    y = {}
    for i, t in enumerate(ps.targets):
        c = ps.ycls[:, i]
        y[f"y_{t}_class"] = (1 - c) if flip else c
        y[f"y_{t}_reg"] = -ps.yreg[:, i] if flip else ps.yreg[:, i]
    return PanelSplit(X, y, ps.lc, ps.assets, ps.name, ps.targets)


tr, va, te = (synthetic_split(seed=0, start_day=18000, name="train"), synthetic_split(seed=1, start_day=18100, name="val"),
              synthetic_split(seed=2, start_day=18200, name="test"))
A = [as_ps(p, p.X, False) for p in (tr, va, te)]
rng = np.random.default_rng(9)
B = [as_ps(p, rng.normal(size=p.X.shape).astype("float32"), True) for p in (tr, va, te)]   # new X, flipped labels

cfg = dict(epochs=1, patience=5, batch_samples=150, max_days=8, lr_warmup_epochs=0, lr_schedule={"type": "constant"})
mcfg = dict(d_model=16, num_heads=4)
root = tempfile.mkdtemp(prefix="audit02_")
try:
    run_dir = os.path.join(root, "panel_A_ic_s0")
    run_panel_variant(*A, lambda: _TinyBase(SEQ, NF), SEQ, NF, run_dir, "A_ic", 0, mcfg, cfg, verbose=False)
    print("dataset A trained; now calling run_panel_variant with dataset B (same shape/dates, different X and y)")
    try:
        _, _, st, _ = run_panel_variant(*B, lambda: _TinyBase(SEQ, NF), SEQ, NF, run_dir, "A_ic", 0, mcfg, cfg,
                                        verbose=False)
    except RuntimeError as e:
        print("PASS: mismatch detected:", str(e)[:120])
    else:
        print(f"DEFECT: resumed silently, epochs in state={st['epoch']}, history rows={len(st['history'])} "
              f"(no epoch trained on B; B signals come from A's weights)")
        raise AssertionError("fingerprint does not cover the data content")
finally:
    shutil.rmtree(root, ignore_errors=True)

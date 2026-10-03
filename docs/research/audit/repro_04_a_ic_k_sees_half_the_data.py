"""Repro 04: variant A_ic_k trains on ~56% of the training samples per epoch; A_ic on 100%.

PANEL_PRESET="1h_s8" trains A_ic and A_ic_k with the same epochs (50) and patience (25), and the
pre-registered criterion compares them ("A_ic_k at k=all not below A_ic by more than 0.01").
make_batch(min_coins=5) keeps k ~ U[5, n] coins of each group and DROPS the rest for that epoch, while
the batch plan (steps per epoch) is unchanged. So A_ic_k gets about half the gradient signal of A_ic, and
the comparison mixes "variable context" with "less training data".

Expected (after fix): per epoch, A_ic_k's batches cover every training sample once (e.g. by splitting
each group into random chunks of size k instead of discarding the remainder), same as A_ic.
Actual (commit c62b4f0): coverage ~0.56 with 83 coins per group.

Run:  python docs/research/audit/repro_04_a_ic_k_sees_half_the_data.py   (~2 s, no TensorFlow)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from cross_asset.experiment import VARIANTS  # noqa: E402
from cross_asset.selftest import synthetic_split  # noqa: E402

ps = synthetic_split(n_assets=83, n_days=200, seed=0)      # 83 coins, like the 1h_s8 universe
cov = {}
for name in ("A_ic", "A_ic_k"):
    mc = VARIANTS[name]["train"].get("min_coins")
    seen = np.zeros(ps.n, int)
    for b in ps.iter_batches(1024, 16, shuffle=True, seed=0, epoch=0, min_coins=mc):
        np.add.at(seen, b["idx"], 1)
    cov[name] = float((seen > 0).mean())
    print(f"{name:7s} min_coins={mc}: fraction of training samples used in one epoch = {cov[name]:.3f}")
assert cov["A_ic_k"] > 0.99, "DEFECT: A_ic_k discards ~half the training samples every epoch"
print("PASS")

"""Round-2 check (expected: PASS): after PanelTrainer.fit + load_best(), the live weights reproduce the best recorded
val_loss (best.weights.h5 is written with the EMA weights swapped in, and BatchNorm statistics of that epoch), and that best
epoch is not simply the last one; predict() then uses those weights (no second EMA swap); the best epoch is chosen by
val only. Random data: what is tested is bookkeeping, not skill.

Run:  python docs/research/audit/r2_09_best_epoch_restore.py   (~20 s)
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from cross_asset.data import PanelSplit  # noqa: E402
from cross_asset.model import build_panel_model, tiny_encoder  # noqa: E402
from cross_asset.train import PanelTrainer, robust_scales  # noqa: E402

T, F, K = 8, 5, 12


def make(n_groups, seed, shift=0.0):
    r = np.random.default_rng(seed)
    n = n_groups * K
    X = r.normal(size=(n, T, F)).astype("float32")
    ts = np.repeat(1_750_003_200 * 10**9 + np.arange(n_groups) * 8 * 3600 * 10**9, K)
    lc = np.zeros((n, 7))
    lc[:, :3], lc[:, 3], lc[:, 4:] = 100.0, ts, 100.0
    signal = X[:, -1, 0] * 0.003 * (1 - shift)                  # weak learnable signal that fades on later data
    ret = signal + r.normal(0, 0.01, n)
    y = {"y_high_reg": (ret * 100).astype("float32"), "y_low_reg": (ret * 100).astype("float32"),
         "y_high_class": (ret > 0).astype("float32"), "y_low_class": (ret > 0).astype("float32")}
    return PanelSplit(X, y, lc, name="s", targets=("high", "low"), day_ns=8 * 3600 * 10**9, target_scale=100.0)


tr, va = make(120, 0), make(40, 1, shift=0.6)
tf.keras.utils.set_random_seed(0)
model = build_panel_model(tiny_encoder(T, F), T, F, targets=("high", "low"), d_model=16, num_heads=2, n_cross_layers=1)
cfg = dict(epochs=6, patience=99, batch_samples=256, max_days=8, lambda_ic=0.5, ic_min_coins=5, lr_initial=3e-3,
           lr_warmup_epochs=0, lr_schedule={"type": "constant"}, verbose=0)
tr_ = PanelTrainer(model, cfg, tempfile.mkdtemp(), robust_scales(tr.yreg), T, F, verbose=False)
state = tr_.fit(tr, va)
hist = [h["val_loss"] for h in state["history"]]
best_ep = int(np.argmin(hist)) + 1
print("val_loss per epoch:", np.round(hist, 4).tolist(), "| recorded best epoch", state["best_epoch"], "value", round(state["best"], 4))
assert state["best_epoch"] == best_ep and abs(state["best"] - min(hist)) < 1e-9
tr_.load_best()
after = tr_.evaluate(va, use_ema=False)["val_loss"]
print(f"val_loss after load_best (live weights, no EMA swap): {after:.4f}")
assert abs(after - state["best"]) < 2e-3, "best.weights.h5 does not reproduce the recorded best val_loss"
assert best_ep != len(hist) or len(hist) == 1, "best epoch is the last epoch: restore not exercised - change seed/data"
print("PASS")

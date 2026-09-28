"""Repro 05: floor-bucket groups over unaligned timestamps put a coin's future inside its peers' windows,
and the panel accepts such groups (check()["ok"] stays True).

This is the grouping used for the earlier 1h stride-32 panel runs (day_ns = 32h floor, coins spread over
~30 phases, docs/research/hourly_1h.md sec. 1/4). In a 32h bucket, a coin whose window ends at T is grouped
with coins whose windows end at T+1h ... T+31h; their inputs contain the candles of the first coin's target
horizon. Cross-coin attention can read them; the no-attention variant cannot.

Part 1 (structural, asserted): the fraction of samples whose target candles lie inside some group peer's
input window, and PanelSplit.check()["ok"].
Part 2 (behavioural, printed): tiny A (attention) vs B (no attention) trained on synthetic data where
the only signal is the future market move. Unaligned floor groups: A >> B. Same data, aligned
timestamps: A ~ B ~ 0.5.

Expected (after fix): check()["ok"] is False (or build_panel_splits raises) whenever a group mixes
timestamps such that a peer's window covers another member's target.
Actual (commit c62b4f0): ok=True.

The 1h_s8 preset itself builds grid-aligned data (misaligned_timestamps = 0), so it does not hit this path
as long as the aligned file is loaded; nothing stops an unaligned file or off-grid candles from using it.

Run:  python docs/research/audit/repro_05_floor_groups_leak_future_via_attention.py   (~2-3 min CPU)
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from cross_asset.data import PanelSplit  # noqa: E402

H = 3600 * 10**9
W, HOR, G = 32, 4, 32                      # window 32 h, horizon 4 h, stride = group width = 32 h (old 1h data)
N_COINS, N_GROUPS = 24, 360
rng = np.random.default_rng(0)
T_TOTAL = (N_GROUPS + 3) * G + 64
mkt = rng.normal(0, 0.01, T_TOTAL)          # common hourly market return


def make_split(g0, g1, aligned, name):
    rows = []
    for c in range(N_COINS):
        phase = 0 if aligned else c                     # unaligned: each coin its own phase (hours)
        for g in range(g0, g1):
            end = 64 + g * G + phase                    # open time (hour index) of the window's last candle
            rows.append((c, end))
    rows = np.array(rows)
    n = len(rows)
    X = np.empty((n, W, 1), "float32")
    fut = np.empty(n)
    for i, (c, end) in enumerate(rows):
        X[i, :, 0] = (mkt[end - W + 1:end + 1] + rng.normal(0, 0.002, W)) * 100   # coin sees its own past only
        fut[i] = mkt[end + 1:end + 1 + HOR].sum() + rng.normal(0, 0.005)          # target: next HOR hours
    ts = (rows[:, 1] * H).astype("float64")
    close = np.full(n, 100.0)
    lc = np.c_[close, close, close, ts, close * (1 + fut), close * (1 + fut), close * (1 + fut)]
    y = {}
    for t in ("high", "low", "close"):
        y[f"y_{t}_class"] = (fut > 0).astype("float32")
        y[f"y_{t}_reg"] = fut.astype("float32")
    assets = np.array([f"C{c:02d}" for c in rows[:, 0]], dtype=object)
    return PanelSplit(X, y, lc, assets, name, day_ns=G * H)


def peers_cover_target(ps):
    """share of samples whose target candles [ts+1h, ts+HOR h] fall inside a peer's window (peer ts > own ts)."""
    hit = 0
    for d in range(ps.n_days):
        t = ps.ts[ps.day_indices(d)]
        hit += int((t < t.max()).sum())
    return hit / ps.n


# ── Part 1 ──
un = make_split(0, 50, False, "unaligned")
al = make_split(0, 50, True, "aligned")
cu, ca = un.check(), al.check()
print(f"unaligned floor groups: ok={cu['ok']} misaligned={cu['misaligned_timestamps']} "
      f"samples whose target lies in a peer's window={peers_cover_target(un):.2%}")
print(f"aligned groups        : ok={ca['ok']} misaligned={ca['misaligned_timestamps']} "
      f"samples whose target lies in a peer's window={peers_cover_target(al):.2%}")

# ── Part 2 ──
def auc_pair(aligned):
    from cross_asset.experiment import run_panel_variant
    from cross_asset.selftest import _TinyBase
    tr = make_split(0, 240, aligned, "train")
    va = make_split(242, 290, aligned, "val")
    te = make_split(292, N_GROUPS, aligned, "test")
    out = {}
    root = tempfile.mkdtemp(prefix="audit05_")
    try:
        for v in ("A", "B"):
            cfg = dict(epochs=8, patience=8, batch_samples=24 * 8, max_days=8, lr_warmup_epochs=0,
                       lr_schedule={"type": "constant"}, lr_initial=3e-3)
            _, t_df, _, _ = run_panel_variant(tr, va, te, lambda: _TinyBase(W, 1), W, 1, os.path.join(root, v), v, 0,
                                              dict(d_model=16, num_heads=4), cfg, verbose=False)
            out[v] = roc_auc_score(te.ycls[:, 2], t_df["p_up_close"].to_numpy())
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return out


for aligned in (False, True):
    r = auc_pair(aligned)
    print(f"{'aligned  ' if aligned else 'unaligned'} test AUC (label = next {HOR}h market move): "
          f"A (attention) {r['A']:.3f} | B (no attention) {r['B']:.3f}")

assert not cu["ok"], "DEFECT: groups whose peers' windows contain a member's target pass the panel checks"
print("PASS")

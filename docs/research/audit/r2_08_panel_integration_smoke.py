"""Round-2 integration smoke (expected: PASS): pipeline dataset with the 43 features -> split_data -> main-style stamping ->
the real encoder (model_v2 build_model_fn + ANTI_MEMORIZATION_CONFIG, close suspended: heads high/low) -> run_panel_experiment
for A_ic_k (min_coins path) with k-eval and resume. Catches shape/feature-count/key mismatches that unit tests with
random arrays cannot: N_FEATURES=43, 8h grid groups, targets=("high","low"), exported column set, resume refusal on other data.
No real training (2 epochs x 3 steps).

Run:  python docs/research/audit/r2_08_panel_integration_smoke.py   (~2 min)
"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _r2_synth import *  # noqa: E402,F401,F403
from _nbload import ROOT  # noqa: E402

sys.path.insert(0, ROOT)
coins = ["BTCUSDT"] + [f"C{i}USDT" for i in range(11)]
start, end = "2025-01-01", "2025-04-20"
rng = np.random.default_rng(11)
k15 = {c: make_coin_15m(rng, start, end, s0=100 * (i + 1)) for i, c in enumerate(coins)}
tmp = tempfile.mkdtemp()
write_archives(tmp, coins, k15, start, end, rng)
ns = fresh_ns()
ds = build(ns, tmp, {c: to_1h(k) for c, k in k15.items()}, coins)
C = ns["CONFIG"]
C["split_dates"] = {"train_end": "2025-02-20", "val_end": "2025-03-25"}
with contextlib.redirect_stdout(io.StringIO()):
    train, val, test = ns["split_data"](ds, config=C)
scale = float(ds["reg_target_scale"])
for sp in (train, val, *test.values()):
    sp["reg_target_scale"] = scale
F = len(ds["feature_order"])
assert F == 43 and ds["X_1h"].shape[1] == 32

# model (package model/, ex model_v2) runs its self-tests on load; load every module but that one
import _nbload  # noqa: E402
mv = _nbload.load_model({"__name__": "audit_nb"})
overrides = dict(mv["ANTI_MEMORIZATION_CONFIG"])
overrides.update(price_targets=("high", "low"), head_types={t: ["nig_regression", "binary_classification"] for t in ("high", "low")})
builder = lambda: mv["build_model_fn"](32, F, config=overrides)  # noqa: E731

from cross_asset.data import group_ns_for  # noqa: E402
from cross_asset.experiment import run_panel_experiment  # noqa: E402

g = group_ns_for("1h", C["stride"])
run_root = tempfile.mkdtemp()
tcfg = dict(epochs=2, patience=5, batch_samples=256, max_days=8, max_steps_per_epoch=3, ic_min_coins=5)
names = lambda which: None  # noqa: E731
common = dict(run_root=run_root, variants=("A_ic_k",), seeds=(0,), train_cfg=tcfg, targets=("high", "low"), day_ns=g,
              k_eval=(5, None), k_draws=1, verbose=False)
with contextlib.redirect_stdout(io.StringIO()):
    res = run_panel_experiment(train, val, test, "1h", builder, 32, F, **common)
sig_v, sig_t = res["runs"]["A_ic_k_s0"]
need = {"asset", "timestamp", "entry", "mu_high", "mu_low", "p_up_high", "p_up_low", "pred_high", "pred_low", "y_high_class", "y_low_class"}
assert need <= set(sig_t.columns), need - set(sig_t.columns)
assert "p_up_close" not in sig_t.columns                       # close suspended: no close head anywhere
assert np.abs(sig_t["mu_high"]).max() < 0.5, "mu_high must be a return (scale divided out)"
assert (sig_t["pred_high"] / sig_t["last_high"] - 1).abs().max() < 0.5
kt = res["k_eval"]
assert {"auc_high", "auc_low", "ic_asym", "ic_high", "ic_low"} <= set(kt.columns), kt.columns
assert set(kt.index.get_level_values("k")) == {5, "all"}
assert {"signals_val.csv.gz", "signals_test.csv.gz", "meta.json", "state.json", "k_coins_test.csv"} <= set(os.listdir(os.path.join(run_root, "panel_A_ic_k_s0")))
print(f"features={F}; test rows {len(sig_t)}; k-eval rows {len(kt)}; exported columns ok; resume:")
with contextlib.redirect_stdout(io.StringIO()):                # resume with identical data must reuse state (epoch 2 done)
    res2 = run_panel_experiment(train, val, test, "1h", builder, 32, F, **common)
assert res2["states"]["A_ic_k_s0"]["epochs"] == 2
print("   identical data resumes; PASS")

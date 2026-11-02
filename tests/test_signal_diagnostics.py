"""tools/signal_diagnostics.py: is a confidence / uncertainty signal informative about the error or noise? Synthetic tables with a known truth:
informative (right sign), inverted (opposite sign), pure noise (|z| small), and a SCALE CONFOUND (assets differ in volatility: pooled rho is high,
within-asset rho ~ 0), plus the entry_range price columns and the CLI.
    python -m pytest tests/test_signal_diagnostics.py -q
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import signal_diagnostics as sd  # noqa: E402


def table(kind, n_assets=20, n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_assets):
        vol = 0.01 * (1 + a)                                    # assets differ in volatility
        entry = np.full(n, 100.0)
        mu = np.zeros(n)
        local = rng.lognormal(0, 0.5, n)                        # the within-asset uncertainty level
        err = vol * local * np.abs(rng.normal(size=n))
        sign = rng.choice([-1, 1], n)
        fut = entry * (1 + mu + sign * err)
        if kind == "informative":
            wst, conf = vol * local, 1 / (1 + local)
        elif kind == "inverted":
            wst, conf = vol / local, local / (1 + local)
        elif kind == "noise":
            wst, conf = rng.lognormal(0, 0.5, n), rng.random(n)
        elif kind == "scale_only":                              # signal = the asset's volatility only, no within-asset information
            wst, conf = np.full(n, vol) * (1 + 1e-3 * rng.random(n)), 1 / (1 + np.full(n, vol * 10)) + 1e-6 * rng.random(n)
            err = vol * np.abs(rng.normal(size=n))
            fut = entry * (1 + sign * err)
        rows.append(pd.DataFrame({"asset": f"A{a}", "entry": entry, "fut_high": fut, "mu_high": mu, "wst_high": wst, "conf_high": conf}))
    return pd.concat(rows, ignore_index=True)


def test_informative_signals_have_the_expected_sign_and_a_large_z():
    r = sd.check_all(table("informative"), targets=("high",), n_perm=50)["high"]
    assert r["uncertainty"]["within_asset"]["mean_rho"] > 0.3 and r["uncertainty"]["within_asset"]["z"] > 10
    assert r["confidence"]["within_asset"]["mean_rho"] < -0.3 and r["confidence"]["within_asset"]["z"] < -10
    assert r["uncertainty"]["deciles"]["decile_trend_rho"] > 0.9 and r["confidence"]["deciles"]["decile_trend_rho"] < -0.9
    assert "informative" in " ".join(sd.verdict({"high": r}))


def test_an_inverted_signal_is_reported_as_inverted_not_as_noise():
    r = sd.check_all(table("inverted"), targets=("high",), n_perm=50)["high"]
    assert r["uncertainty"]["within_asset"]["mean_rho"] < -0.3 and r["confidence"]["within_asset"]["mean_rho"] > 0.3
    assert all("INVERTED" in v for v in sd.verdict({"high": r}))


def test_pure_noise_is_noise():
    r = sd.check_all(table("noise"), targets=("high",), n_perm=100)["high"]
    for name in ("confidence", "uncertainty"):
        assert abs(r[name]["within_asset"]["mean_rho"]) < 0.05           # sd of the mean over 20 assets x 400 rows is ~0.011
        assert abs(r[name]["within_asset"]["z"]) < 3.5
    assert all("noise-like" in v for v in sd.verdict({"high": r}))


def test_a_scale_confound_is_high_pooled_and_zero_within_asset():
    r = sd.check_all(table("scale_only"), targets=("high",), n_perm=100)["high"]["uncertainty"]
    assert r["pooled_rho"] > 0.4                                # asset volatility alone makes the pooled correlation look strong
    assert abs(r["within_asset"]["mean_rho"]) < 0.05            # and the within-asset check removes it


def test_entry_range_tables_use_price_columns_and_skip_what_they_cannot_define():
    df = table("informative")
    df["pred_high"] = df["entry"] * (1 + df["mu_high"])
    df["entry_close_reg"] = "abs_return"
    assert sd.check_head(df, "high", n_perm=10) is not None
    assert sd.check_head(df, "close", n_perm=10) is None                        # no fut_close / pred_close in the table
    df2 = df.drop(columns=["pred_high"])                                         # entry_range without prices: no definition of the error
    assert sd.check_head(df2, "high", n_perm=10) is None


def test_cli_prints_a_verdict_line_per_signal():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "signals_test.csv.gz")
    table("informative", n_assets=6, n=300).to_csv(p, index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "signal_diagnostics.py"), p, "--targets", "high", "--perm", "20"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("within-asset rho") == 2
    j = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "signal_diagnostics.py"), p, "--targets", "high", "--perm", "20", "--json"],
                       capture_output=True, text=True, timeout=300)
    assert "decile_trend_rho" in json.loads(j.stdout)["signals_test.csv.gz"]["high"]["confidence"]["deciles"]

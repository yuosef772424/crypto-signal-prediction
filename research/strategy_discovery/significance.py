"""
PURPOSE: Regenerates the per-trade logs of the unseen-asset tests (E-disc-003: S1, S2, S3; E-disc-006: volume_donchian) on BCH, TRX
         and ZEC for 2024-01-01 to 2026-09-30 and computes the significance statistics the original cards did not report: n, mean R,
         SD, naive t, day-clustered t, mean R at twice the cost, mean R after removing the best 5 % of trades, per-coin means.
         Before writing anything it checks that the regenerated n and mean equal the committed result CSVs.
TAGS:    significance, day-clustered t, trade log, unseen assets, E-disc-003, E-disc-006, cost x2, trimmed mean, review correction
PITFALLS: Post-hoc by construction: the cards were committed together with their results (PREREGISTRATION_NOTE.md), and this script
         was written after the verdict was read. The day-clustered t groups trades by entry day (trades on different coins on the
         same day are correlated); it uses G/(G-1) small-sample scaling and a normal reference, so it is still approximate at
         about 100 trades. The three coins are a consumed period for the original seven coins only; they are not a fresh sample.

Usage (repo root):
    python research/strategy_discovery/significance.py --data /home/user/research/ohlc_full --out research/strategy_discovery/results
"""
import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trade_engine as E  # noqa: E402
import grid  # noqa: E402
import grid2  # noqa: E402
from survivor_checks import pooled, UNSEEN, UNSEEN_WIN  # noqa: E402

CONFIGS = {  # label -> (family, gate, card)
    "S1": ("donchian20", "adx25", "E-disc-003"),
    "S2": ("bb_break", "adx25", "E-disc-003"),
    "S3": ("donchian20", "none", "E-disc-003"),
    "volume_donchian": ("volume_donchian", "none", "E-disc-006"),
}
TRIM = 0.05


def load_unseen(data_dir):
    full = {}
    for c in UNSEEN:
        d = grid2.extra(E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), "1d")))
        full[c] = grid.extra_columns(d)
    return full


def naive_t(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def clustered_t(x, groups):
    """t of the mean with a cluster-robust SE; clusters = groups. SE = sqrt(G/(G-1) * sum_g S_g^2) / n, S_g = sum of (x - mean)."""
    x = np.asarray(x, float)
    n = len(x)
    s = pd.Series(x - x.mean()).groupby(np.asarray(groups)).sum()
    g = len(s)
    se = math.sqrt(g / (g - 1) * float((s ** 2).sum())) / n
    return float(x.mean() / se), g


def trimmed_mean_without_best(x, frac=TRIM):
    x = np.sort(np.asarray(x, float))
    k = int(math.ceil(frac * len(x)))
    return float(x[:-k].mean()), k


def stats_row(label, fam, gate, card, t, t2):
    x = t.net_r.values
    day = t.entry_time.dt.strftime("%Y-%m-%d").values
    tc, g = clustered_t(x, day)
    tm, k = trimmed_mean_without_best(x)
    return dict(config=label, family=fam, gate=gate, card=card, n=len(x), mean_R=x.mean(), sd_R=x.std(ddof=1),
                naive_t=naive_t(x), day_clusters=g, clustered_t=tc, mean_R_cost_x2=float(t2.net_r.mean()),
                trimmed_dropped=k, mean_R_without_best_5pct=tm,
                per_coin="; ".join(f"{c}={t[t.coin == c].net_r.mean():+.3f} (n={int((t.coin == c).sum())})" for c in UNSEEN))


def committed_check(out_dir, label, n, mean):
    """Regenerated n and mean must equal the committed card results (E-disc-003 CSV, E-disc-006 CSV)."""
    if label == "volume_donchian":
        df = pd.read_csv(os.path.join(out_dir, "E-disc-006_volume_donchian_none.csv"))
        m = re.search(r"n=(\d+) mean=([+-]\d+\.\d+)", df[df.test == "U_unseen_assets"].metric.iloc[0])
        cn, cm = int(m.group(1)), float(m.group(2))
    else:
        df = pd.read_csv(os.path.join(out_dir, "E-disc-003_results.csv"))
        r = df[(df.survivor == label) & (df.test == "A_unseen_assets_2024_2026_09")].iloc[0]
        cn, cm = int(r.n), float(r.mean_net_r)
    if cn != n or abs(cm - mean) > 5e-4:
        raise SystemExit(f"MISMATCH {label}: committed n={cn} mean={cm:+.4f}; regenerated n={n} mean={mean:+.4f}. Stop.")


def run(data_dir, out_dir):
    full = load_unseen(data_dir)
    rows, logs = [], {}
    for label, (fam, gate, card) in CONFIGS.items():
        t = pooled(full, UNSEEN, fam, gate, *UNSEEN_WIN)
        t2 = pooled(full, UNSEEN, fam, gate, *UNSEEN_WIN, cost=2 * E.COST)
        committed_check(out_dir, label, len(t), float(t.net_r.mean()))
        rows.append(stats_row(label, fam, gate, card, t, t2))
        logs[label] = t
    # per-trade logs (compact). cost_R is the round-trip cost in R, so the cost x2 mean is reproducible from the CSV alone.
    for label, t in logs.items():
        fam, gate, card = CONFIGS[label]
        name = f"{card}_trades_{label}.csv"
        o = pd.DataFrame(dict(symbol=t.coin, entry_time=t.entry_time.dt.strftime("%Y-%m-%d %H:%M"), side=t.side.astype(int),
                              R_net=t.net_r.round(6), cost_R=(t.gross_r - t.net_r).round(6)))
        o.to_csv(os.path.join(out_dir, name), index=False)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(out_dir, "E-disc-003_significance.csv"), index=False)
    write_md(st, os.path.join(out_dir, "E-disc-003_significance.md"))
    print(st.drop(columns=["per_coin"]).round(3).to_string(index=False))


def write_md(st, path):
    f = lambda v: f"{v:+.3f}"
    lines = [
        "# E-disc-003 / E-disc-006 significance table (unseen assets BCH, TRX, ZEC, 2024-01-01 to 2026-09-30)",
        "",
        "Generated by `research/strategy_discovery/significance.py` (do not edit by hand). Post-hoc: it was added after the verdicts",
        "of E-disc-003 and E-disc-006 were read, in review of PR 15. Before writing, the script checked that the regenerated trade",
        "counts and means equal the committed `E-disc-003_results.csv` and `E-disc-006_volume_donchian_none.csv`.",
        "",
        "Definitions: R = net R per trade after 0.12 % round-trip cost; naive t = mean / (SD / sqrt(n)); day-clustered t uses a",
        "cluster-robust SE with trades grouped by entry day (G/(G-1) scaling); cost x2 re-simulates with 0.24 % round-trip cost;",
        "\"without best 5 %\" drops the ceil(0.05 n) highest-R trades and recomputes the mean.",
        "",
        "| Config | Card | n | Mean R | SD R | Naive t | Day-clustered t (days) | Mean R at 2x cost | Mean R without best 5 % (dropped) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in st.itertuples():
        lines.append(f"| {r.config} (`{r.family}`, gate `{r.gate}`) | {r.card} | {r.n} | {f(r.mean_R)} | {r.sd_R:.3f} | {r.naive_t:.2f} | "
                     f"{r.clustered_t:.2f} ({r.day_clusters}) | {f(r.mean_R_cost_x2)} | {f(r.mean_R_without_best_5pct)} ({r.trimmed_dropped}) |")
    lines += ["", "## Per-coin mean R (n)", "", "| Config | BCHUSDT | TRXUSDT | ZECUSDT |", "|---|---|---|---|"]
    for r in st.itertuples():
        d = dict(p.split("=", 1) for p in r.per_coin.split("; "))
        lines.append(f"| {r.config} | {d['BCHUSDT']} | {d['TRXUSDT']} | {d['ZECUSDT']} |")
    lines += ["", "## Mechanical reading", ""]
    lines.append("- Largest absolute day-clustered t across the four configurations: "
                 f"{st.clustered_t.abs().max():.2f}" + (" (below 2.0, so none reaches the usual evidence bar)." if st.clustered_t.abs().max() < 2.0
                                                        else " (at least one reaches 2.0)."))
    pos = st[st.mean_R > 0]
    lines.append("- Mean R after removing the best 5 % of trades, positive configurations: " +
                 "; ".join(f"{r.config} {f(r.mean_R)} -> {f(r.mean_R_without_best_5pct)}" for r in pos.itertuples()) + ".")
    lines.append("- Mean R is positive at 2x cost for: " + (", ".join(st[st.mean_R_cost_x2 > 0].config) or "none") + ".")
    lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="research/strategy_discovery/results")
    a = ap.parse_args()
    run(a.data, a.out)

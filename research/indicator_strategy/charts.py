"""
PURPOSE: Renders the analysis of the chosen configuration as images so that success and failure regions can be read visually:
         cumulative net R over DEV, VAL and TEST; mean net R by year, ADX bucket, trend and volatility quartile; and candlestick
         samples of winning and losing TEST trades with entry, stop and target drawn from the same engine.
TAGS:    charts, figures, candlestick sample, cumulative R, region bars, visual inspection, E-ind-003
PITFALLS: Stop and target levels are recomputed from the engine (2×ATR stop, 3×ATR target, ATR at the bar before entry), so the
         drawn levels match the simulation. Charts only read existing outputs and the 1d data; they do not change any verdict.

Usage (repo root):
    python research/indicator_strategy/charts.py --data /home/user/research/ohlc --out research/indicator_strategy/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))


def equity_curves(trades, out):
    x = trades.sort_values("exit_time").copy()
    x["cum"] = x.net_r.cumsum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x.exit_time, x.cum, lw=1.4, color="#1f4e79")
    for bound, label in [("2022-01-01", "DEV | VAL"), ("2024-01-01", "VAL | TEST")]:
        ax.axvline(pd.Timestamp(bound, tz="UTC"), color="grey", ls="--", lw=1)
        ax.text(pd.Timestamp(bound, tz="UTC"), ax.get_ylim()[1] * 0.95, label, fontsize=8, color="grey")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("Chosen configuration: donchian 1d, long only — cumulative net R (all coins, DEV → TEST)")
    ax.set_ylabel("cumulative net R (sum over coins)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def region_bars(trades, out):
    x = trades.copy()
    x["year"] = x.entry_time.dt.year
    x["adx_b"] = pd.cut(x.adx, [0, 20, 25, 30, 40, 100], labels=["<20", "20-25", "25-30", "30-40", ">40"])
    x["natr_b"] = pd.qcut(x.natr, 4, labels=["Q1 calm", "Q2", "Q3", "Q4 volatile"])
    x["trend"] = np.where(x.trend_up, "close > EMA200", "close < EMA200")
    panels = [("year", "by year"), ("adx_b", "by ADX bucket"), ("trend", "by trend filter"), ("natr_b", "by volatility quartile")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, title) in zip(axes.ravel(), panels):
        g = x.groupby(col, observed=True).net_r.agg(["mean", "count"])
        colors = ["#2e7d32" if v > 0 else "#c62828" for v in g["mean"]]
        ax.bar(g.index.astype(str), g["mean"], color=colors)
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(i, m + (0.02 if m >= 0 else -0.05), f"n={n}", ha="center", fontsize=8)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"TEST: mean net R {title}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def candle_sample(data_dir, row, out):
    d = E.indicators(E.load(os.path.join(data_dir, f"ohlc_{row.coin}_1h.parquet"), "1d"))
    entry_idx = d.index.get_loc(pd.Timestamp(row.entry_time))
    atr_prev = d.atr.values[entry_idx - 1]
    e = d.open.values[entry_idx]
    risk = E.ATR_STOP * atr_prev
    side = int(row.side)
    stop = e - side * risk
    tgt = e + side * E.ATR_TGT * atr_prev
    lo, hi = max(0, entry_idx - 40), min(len(d), entry_idx + 35)
    w = d.iloc[lo:hi]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, (ts, r) in enumerate(w.iterrows()):
        color = "#2e7d32" if r.close >= r.open else "#c62828"
        ax.vlines(k, r.low, r.high, color=color, lw=1)
        ax.add_patch(plt.Rectangle((k - 0.3, min(r.open, r.close)), 0.6, abs(r.close - r.open) or 1e-9, color=color))
    ei = entry_idx - lo
    ax.axvline(ei, color="#1f4e79", ls=":", lw=1)
    ax.plot([ei, ei + 30], [e, e], color="#1f4e79", lw=1.2, label="entry")
    ax.plot([ei, ei + 30], [stop, stop], color="#c62828", lw=1.2, ls="--", label="stop (2×ATR)")
    ax.plot([ei, ei + 30], [tgt, tgt], color="#2e7d32", lw=1.2, ls="--", label="target (3×ATR)")
    ax.set_title(f"{row.coin} 1d  {'LONG' if side == 1 else 'SHORT'}  entry {str(row.entry_time)[:10]}  "
                 f"net {row.net_r:+.2f}R  exit={row.reason}  ADX={row.adx:.0f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([])
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def run(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tr = pd.read_csv(os.path.join(HERE, "E-ind-003_test_chosen_trades_annotated.csv"), parse_dates=["entry_time", "exit_time"])
    # the chosen configuration itself (long only), recomputed over the full history so that one-position-per-coin state is right
    from screen import COINS
    full_hist = {c: E.indicators(E.load(os.path.join(data_dir, f"ohlc_{c}_1h.parquet"), "1d")) for c in COINS}
    parts = []
    for c in COINS:
        lg, sh = E.build_signal(full_hist[c], "donchian", ("long_only",))
        t = E.simulate(full_hist[c], lg, sh, "1d")
        t["coin"] = c
        parts.append(t)
    chosen_all = pd.concat(parts, ignore_index=True)
    full = chosen_all[["exit_time", "net_r"]].sort_values("exit_time")
    fig_path = os.path.join(out_dir, "equity_curve_R.png")
    equity_curves(full, fig_path)
    region_bars(tr.assign(trend_up=tr.trend_up.astype(bool)), os.path.join(out_dir, "test_regions.png"))
    samples = {
        "win_big": tr[tr.reason == "target"].sort_values("net_r").tail(1),
        "win_typical": tr[tr.reason == "target"].iloc[[len(tr[tr.reason == "target"]) // 2]],
        "loss_stop": tr[tr.reason == "stop"].sort_values("net_r").head(1),
        "loss_stop_typical": tr[tr.reason == "stop"].iloc[[len(tr[tr.reason == "stop"]) // 2]],
    }
    for name, df in samples.items():
        for _, row in df.iterrows():
            candle_sample(data_dir, row, os.path.join(out_dir, f"sample_{name}_{row.coin}.png"))
    print("figures written to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.data, a.out)

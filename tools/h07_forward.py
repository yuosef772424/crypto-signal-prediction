"""
PURPOSE:  Forward paper trading of the frozen H07 rule (TSMOM30 long-only BTC+ETH, weight 0.02/std30 capped at 3, split
          equally, 0.06%/side) on data after 2026-09-30, the only data no study has touched. Rebuilds the ledger
          deterministically from daily Binance spot closes, refuses to rewrite past rows if the data changed, appends new
          days, and prints paper P&L vs equal-weight buy & hold. Shadow variants from research/h07_combos (V3 = the same
          rule on 10 coins, V4 = multi-speed ensemble on 10 coins) are tracked in their own ledgers, never adopted until
          the pre-registered forward comparison (registry F-0065 R1) says so.
TAGS:     H07, TSMOM30, paper trading, forward test, out of sample, ledger, BTC, ETH, trend filter, daily
PITFALLS: The rule is FROZEN (research/edge_discovery 16_h07_oos.py, re-confirmed in research/h07_volsizing): never tune
          it on the forward ledger. Weights decided at day d's close earn d -> d+1; the first decision is 2026-09-30, so
          P&L starts on 2026-10-01. Data: github Speirsy11/crypto-dataset (updated daily ~23:30 UTC through the previous
          day) or a CSV with date,BTC,ETH closes (e.g. exported from Binance on Colab).

Usage:
    python tools/h07_forward.py                         # fetch 1d closes from GitHub, update the ledger, print summary
    python tools/h07_forward.py --csv closes.csv        # use your own closes (date,BTC,ETH)
    python tools/h07_forward.py --dry-run               # print without writing the ledger
    python tools/h07_forward.py --variant all           # H07 + shadow variants v3_10coins, v4_ens_10coins
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, "research", "h07_forward", "ledger.csv")
FIRST_DECISION = pd.Timestamp("2026-09-30")
N, TARGET, CAP, COST_SIDE = 30, 0.02, 3.0, 0.0006
ALL_COINS = ["BTC", "ETH", "SOL", "ADA", "BCH", "BNB", "DOGE", "TRX", "XRP", "ZEC"]
#: variant -> (coins, lookbacks averaged into the signal). "h07" is the frozen rule; the others are shadows (F-0065).
VARIANTS = {"h07": (["BTC", "ETH"], (30,)),
            "v3_10coins": (ALL_COINS, (30,)),
            "v4_ens_10coins": (ALL_COINS, (10, 20, 30, 60))}
COINS = {c: f"{c}USDT" for c in ALL_COINS}


def ledger_path(variant: str, base: str = LEDGER) -> str:
    return base if variant == "h07" else base.replace("ledger.csv", f"ledger_{variant}.csv")


def columns_for(coins) -> list:
    return ["date"] + [f"{c}_close" for c in coins] + [f"{c}_w" for c in coins] + ["turnover", "pnl", "bh_pnl"]


def h07_weights(px: pd.DataFrame, lookbacks=(30,)) -> pd.DataFrame:
    """Frozen H07 (lookbacks=(30,)): long when the N-day log return > 0 (several N: the fraction that is positive);
    weight = signal * 0.02 / std30(daily log returns), capped at 3, / number of coins with >= 60 days of data."""
    lp = np.log(px)
    vol = lp.diff().rolling(N, min_periods=20).std()
    sig = sum((lp.diff(n) > 0).astype(float) for n in lookbacks) / len(lookbacks)
    live = px.notna() & (px.notna().cumsum() >= 60)
    n = live.sum(axis=1).replace(0, np.nan)
    return (sig.where(live) * (TARGET / vol).clip(upper=CAP)).div(n, axis=0).fillna(0.0)


def build_ledger(px: pd.DataFrame, variant: str = "h07") -> pd.DataFrame:
    """One row per decision day >= FIRST_DECISION. `pnl` / `bh_pnl` = return earned by that day's weights over the NEXT day
    (NaN until the next close exists), net of 0.06%/side turnover cost; buy & hold = equal weight over the coins."""
    coins, lookbacks = VARIANTS[variant]
    px = px.sort_index()[coins].dropna(how="all")
    w = h07_weights(px, lookbacks)
    nxt = px.pct_change().shift(-1)
    turn = (w - w.shift(1).fillna(0)).abs().sum(axis=1)
    complete = nxt.notna().all(axis=1)
    pnl = ((w * nxt).sum(axis=1) - turn * COST_SIDE).where(complete)
    bh = nxt.mean(axis=1).where(complete)
    out = pd.DataFrame({"date": px.index.strftime("%Y-%m-%d")})
    for c in coins:
        out[f"{c}_close"] = px[c].values
    for c in coins:
        out[f"{c}_w"] = w[c].values
    out["turnover"], out["pnl"], out["bh_pnl"] = turn.values, pnl.values, bh.values
    return out[px.index >= FIRST_DECISION].reset_index(drop=True)


def merge_ledger(old: pd.DataFrame, new: pd.DataFrame, atol=1e-9) -> pd.DataFrame:
    """Past decisions must not change: rows present in both must match (closes and weights); a pnl that was NaN may be
    filled. Returns the updated ledger (old rows + filled pnl + new rows)."""
    if old.empty:
        return new
    if list(old.columns) != list(new.columns):
        raise ValueError(f"ledger columns differ: {list(old.columns)} vs {list(new.columns)}")
    m = old.merge(new, on="date", how="left", suffixes=("_old", ""), validate="one_to_one")
    for c in [c for c in new.columns if c.endswith(("_close", "_w"))]:
        bad = ~np.isclose(m[f"{c}_old"], m[c], atol=atol, rtol=1e-9) & m[c].notna()
        if bad.any():
            raise ValueError(f"ledger rows changed for {c} on {m.loc[bad, 'date'].tolist()} — data revision; "
                             "investigate before appending (the past is never rewritten)")
    for c in ("pnl", "bh_pnl"):
        filled = m[f"{c}_old"].notna()
        if not np.allclose(m.loc[filled, f"{c}_old"], m.loc[filled, c], atol=atol, equal_nan=True):
            raise ValueError(f"recorded {c} changed — data revision")
    keep = old.copy()
    keep[["pnl", "bh_pnl"]] = m[["pnl", "bh_pnl"]].values
    return pd.concat([keep, new[~new.date.isin(old.date)]], ignore_index=True)[list(new.columns)]


def summary(ledger: pd.DataFrame) -> dict:
    done = ledger.dropna(subset=["pnl"])
    eq, bh = (1 + done.pnl).cumprod(), (1 + done.bh_pnl).cumprod()
    last = ledger.iloc[-1]
    return dict(days=len(done), total=float(eq.iloc[-1] - 1) if len(done) else 0.0,
                bh_total=float(bh.iloc[-1] - 1) if len(done) else 0.0,
                maxdd=float((eq / eq.cummax() - 1).min()) if len(done) else 0.0,
                bh_maxdd=float((bh / bh.cummax() - 1).min()) if len(done) else 0.0,
                position_date=last.date,
                **{k: float(last[k]) for k in ledger.columns if k.endswith("_w") and float(last[k]) != 0.0})


def fetch_closes_github(start="2026-06-01", coins=("BTC", "ETH")) -> pd.DataFrame:
    """Daily closes from the GitHub dataset (1d files for the months >= start; a fresh tree listing every run)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fetch_crypto_dataset as fcd
    paths = fcd.list_files(tempfile.mkdtemp(prefix="crypto-dataset-"))
    months = pd.period_range(start, pd.Timestamp.now(tz="UTC").tz_localize(None), freq="M")
    keys = {f"year={p.year}/month={p.month:02d}" for p in months}
    out = {}
    for k in coins:
        sym = COINS[k]
        ps = sorted(p for p in paths if f"interval_id=1d/symbol_id={sym}/" in p and any(x in p for x in keys))
        d = pd.concat([fcd._get(p) for p in ps]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
        out[k] = d["close"]
    px = pd.DataFrame(out)
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    return px


def main(argv=None):
    ap = argparse.ArgumentParser(description="Forward paper trading of the frozen H07 rule (+ shadow variants).")
    ap.add_argument("--csv", help="date,<COIN>... daily closes (overrides the GitHub fetch)")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--variant", default="h07", choices=list(VARIANTS) + ["all"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    names = list(VARIANTS) if a.variant == "all" else [a.variant]
    coins = sorted({c for v in names for c in VARIANTS[v][0]}, key=ALL_COINS.index)
    px = (pd.read_csv(a.csv, parse_dates=["date"]).set_index("date")[coins] if a.csv
          else fetch_closes_github(coins=coins))
    pd.set_option("display.width", 200)
    for v in names:
        path = ledger_path(v, a.ledger)
        new = build_ledger(px, v)
        old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=new.columns)
        led = merge_ledger(old, new)
        if not a.dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            led.to_csv(path, index=False, float_format="%.10g")
        print(f"== {v} ({'frozen' if v == 'h07' else 'shadow'}) -> {os.path.relpath(path, ROOT)}")
        print(led[["date", "turnover", "pnl", "bh_pnl"]].tail(5).to_string(index=False))
        print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in summary(led).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

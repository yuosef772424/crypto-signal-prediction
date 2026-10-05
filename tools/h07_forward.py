"""
PURPOSE:  Forward paper trading of the frozen H07 rule (TSMOM30 long-only BTC+ETH, weight 0.02/std30 capped at 3, split
          equally, 0.06%/side) on data after 2026-09-30, the only data no study has touched. Rebuilds the ledger
          deterministically from daily Binance spot closes, refuses to rewrite past rows if the data changed, appends new
          days, and prints paper P&L vs 50/50 buy & hold.
TAGS:     H07, TSMOM30, paper trading, forward test, out of sample, ledger, BTC, ETH, trend filter, daily
PITFALLS: The rule is FROZEN (research/edge_discovery 16_h07_oos.py, re-confirmed in research/h07_volsizing): never tune
          it on the forward ledger. Weights decided at day d's close earn d -> d+1; the first decision is 2026-09-30, so
          P&L starts on 2026-10-01. Data: github Speirsy11/crypto-dataset (updated daily ~23:30 UTC through the previous
          day) or a CSV with date,BTC,ETH closes (e.g. exported from Binance on Colab).

Usage:
    python tools/h07_forward.py                         # fetch 1d closes from GitHub, update the ledger, print summary
    python tools/h07_forward.py --csv closes.csv        # use your own closes (date,BTC,ETH)
    python tools/h07_forward.py --dry-run               # print without writing the ledger
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
COINS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
COLUMNS = ["date", "BTC_close", "ETH_close", "BTC_w", "ETH_w", "turnover", "pnl", "bh_pnl"]


def h07_weights(px: pd.DataFrame) -> pd.DataFrame:
    """Frozen H07: long when the 30-day log return > 0; weight = 0.02 / std30(daily log returns), capped at 3, / n."""
    lp = np.log(px)
    vol = lp.diff().rolling(N, min_periods=20).std()
    sig = (lp.diff(N) > 0).astype(float)
    return ((sig * (TARGET / vol).clip(upper=CAP)) / px.shape[1]).fillna(0.0)


def build_ledger(px: pd.DataFrame) -> pd.DataFrame:
    """One row per decision day >= FIRST_DECISION. `pnl` / `bh_pnl` = return earned by that day's weights over the NEXT day
    (NaN until the next close exists), net of 0.06%/side turnover cost."""
    px = px.sort_index()[["BTC", "ETH"]].dropna()
    w = h07_weights(px)
    nxt = px.pct_change().shift(-1)
    turn = (w - w.shift(1).fillna(0)).abs().sum(axis=1)
    pnl = (w * nxt).sum(axis=1, min_count=2) - turn * COST_SIDE
    bh = (nxt * 0.5).sum(axis=1, min_count=2)
    out = pd.DataFrame({"date": px.index.strftime("%Y-%m-%d"), "BTC_close": px.BTC.values, "ETH_close": px.ETH.values,
                        "BTC_w": w.BTC.values, "ETH_w": w.ETH.values, "turnover": turn.values, "pnl": pnl.values,
                        "bh_pnl": bh.values})
    return out[px.index >= FIRST_DECISION].reset_index(drop=True)


def merge_ledger(old: pd.DataFrame, new: pd.DataFrame, atol=1e-9) -> pd.DataFrame:
    """Past decisions must not change: rows present in both must match (closes and weights); a pnl that was NaN may be
    filled. Returns the updated ledger (old rows + filled pnl + new rows)."""
    if old.empty:
        return new
    m = old.merge(new, on="date", how="left", suffixes=("_old", ""), validate="one_to_one")
    for c in ("BTC_close", "ETH_close", "BTC_w", "ETH_w"):
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
    return pd.concat([keep, new[~new.date.isin(old.date)]], ignore_index=True)[COLUMNS]


def summary(ledger: pd.DataFrame) -> dict:
    done = ledger.dropna(subset=["pnl"])
    eq, bh = (1 + done.pnl).cumprod(), (1 + done.bh_pnl).cumprod()
    last = ledger.iloc[-1]
    return dict(days=len(done), h07_total=float(eq.iloc[-1] - 1) if len(done) else 0.0,
                bh_total=float(bh.iloc[-1] - 1) if len(done) else 0.0,
                h07_maxdd=float((eq / eq.cummax() - 1).min()) if len(done) else 0.0,
                bh_maxdd=float((bh / bh.cummax() - 1).min()) if len(done) else 0.0,
                position_date=last.date, BTC_w=float(last.BTC_w), ETH_w=float(last.ETH_w))


def fetch_closes_github(start="2026-06-01") -> pd.DataFrame:
    """Daily closes from the GitHub dataset (1d files for the months >= start; a fresh tree listing every run)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import fetch_crypto_dataset as fcd
    paths = fcd.list_files(tempfile.mkdtemp(prefix="crypto-dataset-"))
    months = pd.period_range(start, pd.Timestamp.now(tz="UTC").tz_localize(None), freq="M")
    keys = {f"year={p.year}/month={p.month:02d}" for p in months}
    out = {}
    for k, sym in COINS.items():
        ps = sorted(p for p in paths if f"interval_id=1d/symbol_id={sym}/" in p and any(x in p for x in keys))
        d = pd.concat([fcd._get(p) for p in ps]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
        out[k] = d["close"]
    px = pd.DataFrame(out)
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    return px


def main(argv=None):
    ap = argparse.ArgumentParser(description="Forward paper trading of the frozen H07 rule.")
    ap.add_argument("--csv", help="date,BTC,ETH daily closes (overrides the GitHub fetch)")
    ap.add_argument("--ledger", default=LEDGER)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.csv:
        px = pd.read_csv(a.csv, parse_dates=["date"]).set_index("date")[["BTC", "ETH"]]
    else:
        px = fetch_closes_github()
    new = build_ledger(px)
    old = pd.read_csv(a.ledger) if os.path.exists(a.ledger) else pd.DataFrame(columns=COLUMNS)
    led = merge_ledger(old, new)
    if not a.dry_run:
        os.makedirs(os.path.dirname(a.ledger), exist_ok=True)
        led.to_csv(a.ledger, index=False, float_format="%.10g")
    pd.set_option("display.width", 200)
    print(led.tail(10).to_string(index=False))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary(led).items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

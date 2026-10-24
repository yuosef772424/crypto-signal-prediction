"""
PURPOSE:  Descriptive report on a TRADE LOG of the stop-order grid (E-grid1m-001): concentration of profit (top 1/10/50/100 trades,
          top 1%/5%, best days, leave-one-coin-out, biggest coin), distribution, breakdowns (exit reason, hour, weekday, day, hold
          time, entry-minute quote volume, bar range), equity curve and drawdown, and sanity flags (PASS/WARN) that expose
          simulator-convention optimism (profit from thin / jumpy minutes, gap fills, a random-walk null that earns money).
          Pure functions: build_report(trades_df, ...) -> dict (JSON-able), render_markdown(dict) -> str, save_plots(...) optional.
TAGS:     stop grid report, trade log, concentration, leave one coin out, profit factor, drawdown, sanity flags, null comparison,
          quote volume bucket, gap fill, E-grid1m-001, diagnostics
PITFALLS: Descriptive only: it never tests a hypothesis and never changes a verdict. Reads the trade table only (plus optional
          per-coin bar stats and null draws passed in); imports nothing from the repo. Shares of total are NaN when the total is
          0 and meaningless when the total is negative (read the absolute numbers then). Thresholds of the flags are heuristics
          (THRESHOLDS), not tuned on data. Trade-level means ignore position sizing (every trade = 1 notional unit).
Columns used (tools/stop_grid_sim.annotate_trades): coin, ordering, cost_mult (optional, default 1), net_bp, gross_bp, entry_ts,
exit_ts, exit_reason, hold_min, entry_quote_volume, entry_range_bp, gap_fill, day.
"""
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

THRESHOLDS = {
    "gross_vs_range": 1.5,        # WARN if a coin's mean gross > this x its median 1m bar range
    "thin_entry_qv": 20_000.0,    # entry-minute quote volume below this = thin
    "thin_share_warn": 0.25,      # WARN if more than this share of trades enter on thin minutes
    "gap_share_warn": 0.05,       # WARN if more than this share of trades are gap fills
    "fills_per_day_per_k_qv": 5.0,  # WARN if fills/day per $1k of median minute quote volume exceeds this
    "null_mean_vs_cost": -0.5,    # WARN if a null's mean net > this x round-trip cost (a martingale must lose ~ the cost)
}
TOPS = (1, 10, 50, 100)
HIST_EDGES = [-math.inf, -100, -50, -25, -10, 0, 10, 25, 50, 100, math.inf]
HOLD_EDGES = [0, 5, 15, 60, 240, math.inf]
QV_EDGES = [-math.inf, 1_000, 5_000, 20_000, 100_000, math.inf]
RANGE_EDGES = [-math.inf, 5, 10, 20, 50, math.inf]


def _f(x):
    """float or None for NaN/inf (JSON-safe)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _share(part, total):
    return _f(part / total) if total not in (0, 0.0) and math.isfinite(total) else None


def _bucket_labels(edges, unit=""):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if a == -math.inf:
            out.append(f"<{b:g}{unit}")
        elif b == math.inf:
            out.append(f">={a:g}{unit}")
        else:
            out.append(f"{a:g}-{b:g}{unit}")
    return out


def _group(df: pd.DataFrame, key, total_net: float, order=None) -> list:
    rows = []
    g = df.groupby(key, observed=True)["net_bp"]
    for k, s in g:
        rows.append({"key": str(k), "n": int(len(s)), "net_total": _f(s.sum()), "net_mean": _f(s.mean()),
                     "share_of_net": _share(float(s.sum()), total_net), "win_rate": _f((s > 0).mean())})
    if order is not None:
        pos = {str(k): i for i, k in enumerate(order)}
        rows.sort(key=lambda r: pos.get(r["key"], 10 ** 9))
    return rows


def _bucketed(df, col, edges, unit=""):
    labels = _bucket_labels(edges, unit)
    b = pd.cut(df[col], bins=edges, labels=labels, right=False)
    return df.assign(_b=b), labels


def _streaks(win: np.ndarray):
    best_w = best_l = cur_w = cur_l = 0
    for w in win:
        if w:
            cur_w, cur_l = cur_w + 1, 0
        else:
            cur_l, cur_w = cur_l + 1, 0
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return best_w, best_l


def _top_stats(vals_sorted_desc: np.ndarray, total: float) -> dict:
    out = {}
    n = len(vals_sorted_desc)
    for k in TOPS:
        kk = min(k, n)
        out[f"top{k}"] = {"k": int(kk), "sum": _f(vals_sorted_desc[:kk].sum()), "share": _share(float(vals_sorted_desc[:kk].sum()), total)}
    for pct in (1, 5):
        kk = min(n, max(1, math.ceil(n * pct / 100.0))) if n else 0
        out[f"top{pct}pct"] = {"k": int(kk), "sum": _f(vals_sorted_desc[:kk].sum()), "share": _share(float(vals_sorted_desc[:kk].sum()), total)}
    return out


def analyze(df: pd.DataFrame) -> dict:
    """All blocks for ONE scope (a trade table already filtered)."""
    n = len(df)
    if n == 0:
        return {"n": 0}
    net = df["net_bp"].to_numpy(float)
    gross = df["gross_bp"].to_numpy(float)
    tot_net, tot_gross = float(net.sum()), float(gross.sum())
    res = {"n": int(n), "net_total": _f(tot_net), "gross_total": _f(tot_gross), "net_mean": _f(net.mean()), "gross_mean": _f(gross.mean()),
           "fee_mean": _f(df["fee_bp"].mean()) if "fee_bp" in df else None, "slip_mean": _f(df["slip_bp"].mean()) if "slip_bp" in df else None}

    # concentration
    ns = np.sort(net)[::-1]
    gs = np.sort(gross)[::-1]
    conc = {"by_net": _top_stats(ns, tot_net), "by_gross": _top_stats(gs, tot_gross)}
    rem = {}
    for label, k in (("top100", min(100, n)), ("top1pct", min(n, max(1, math.ceil(n * 0.01))))):
        rest = ns[k:]
        rem[label] = {"removed": int(k), "net_total_after": _f(rest.sum()), "net_mean_after": _f(rest.mean()) if len(rest) else None}
    conc["after_removing"] = rem
    day_net = df.groupby("day")["net_bp"].sum().sort_values(ascending=False)
    conc["best5_days_share_of_net"] = _share(float(day_net.iloc[:5].sum()), tot_net)
    conc["best5_days_net"] = _f(day_net.iloc[:5].sum())
    coin_net = df.groupby("coin")["net_bp"].agg(["sum", "count", "mean"])
    loco = {}
    for coin in coin_net.index:
        rest = df[df["coin"] != coin]["net_bp"]
        loco[str(coin)] = {"coin_net_total": _f(coin_net.loc[coin, "sum"]), "coin_trades": int(coin_net.loc[coin, "count"]),
                           "coin_net_mean": _f(coin_net.loc[coin, "mean"]), "rest_net_total": _f(rest.sum()),
                           "rest_net_mean": _f(rest.mean()) if len(rest) else None, "rest_trades": int(len(rest))}
    conc["leave_one_coin_out"] = loco
    top_coin = str(coin_net["sum"].idxmax())
    conc["biggest_coin"] = {"coin": top_coin, "share_of_net": _share(float(coin_net.loc[top_coin, "sum"]), tot_net),
                            "share_of_trades": _f(coin_net.loc[top_coin, "count"] / n), **{k: v for k, v in loco[top_coin].items() if k.startswith("rest")}}
    res["concentration"] = conc

    # distribution
    wins, losses = net[net > 0], net[net <= 0]
    avg_w = float(wins.mean()) if len(wins) else float("nan")
    avg_l = float(losses.mean()) if len(losses) else float("nan")
    order = np.argsort(df["exit_ts"].to_numpy(), kind="stable")
    sw, sl = _streaks(net[order] > 0)
    q = np.quantile(net, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    hist = pd.cut(pd.Series(net), bins=HIST_EDGES, right=False).value_counts(sort=False)
    labels = _bucket_labels(HIST_EDGES, "bp")
    res["distribution"] = {
        "win_rate": _f((net > 0).mean()), "avg_win": _f(avg_w), "avg_loss": _f(avg_l),
        "payoff_ratio": _f(avg_w / abs(avg_l)) if len(wins) and len(losses) and avg_l != 0 else None,
        "profit_factor": _f(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else None,
        "median": _f(np.median(net)), "p1": _f(q[0]), "p5": _f(q[1]), "p25": _f(q[2]), "p75": _f(q[3]), "p95": _f(q[4]), "p99": _f(q[5]),
        "skew": _f(pd.Series(net).skew()) if n >= 3 else None, "largest_win": _f(net.max()), "largest_loss": _f(net.min()),
        "max_win_streak": int(sw), "max_loss_streak": int(sl),
        "histogram": [{"bucket": lb, "n": int(v)} for lb, v in zip(labels, hist.to_numpy())]}

    # breakdowns
    res["by_exit_reason"] = _group(df, "exit_reason", tot_net)
    ents = pd.to_datetime(df["entry_ts"], unit="ms", utc=True)
    d2 = df.assign(_hour=ents.dt.hour.to_numpy(), _wd=ents.dt.day_name().to_numpy())
    res["by_hour_utc"] = _group(d2, "_hour", tot_net, order=range(24))
    res["by_weekday"] = _group(d2, "_wd", tot_net, order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    days = day_net.sort_index()
    res["by_day"] = {"n_days": int(len(days)), "best_day": {"day": str(day_net.index[0]), "net": _f(day_net.iloc[0])},
                     "worst_day": {"day": str(day_net.index[-1]), "net": _f(day_net.iloc[-1])},
                     "pct_positive_days": _f((days > 0).mean()), "daily_net": {str(k): _f(v) for k, v in days.items()}}
    for name, col, edges, unit in (("by_hold_min", "hold_min", HOLD_EDGES, "min"), ("by_entry_quote_volume", "entry_quote_volume", QV_EDGES, "$"),
                                   ("by_entry_range_bp", "entry_range_bp", RANGE_EDGES, "bp")):
        if col in df and df[col].notna().any():
            b, labels = _bucketed(d2.dropna(subset=[col]), col, edges, unit)
            res[name] = _group(b, "_b", tot_net, order=labels)
        else:
            res[name] = []

    # equity (notional units: cumulative net_bp ordered by exit time)
    d3 = df.assign(_x=df["exit_ts"].to_numpy()).sort_values("_x", kind="stable")
    cum = np.cumsum(d3["net_bp"].to_numpy(float))
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    dd = peak - cum
    xt = d3["_x"].to_numpy(dtype="int64")
    span = float(xt[-1] - xt[0]) if n > 1 else 0.0
    under = 0.0
    if span > 0:
        under = float(np.diff(xt)[(dd[:-1] > 0)].sum()) / span
    res["equity"] = {"final": _f(cum[-1]), "max_drawdown_bp": _f(dd.max()), "time_underwater_share": _f(under) if span > 0 else None,
                     "curve_points": [[int(t), _f(v)] for t, v in zip(xt[:: max(1, n // 400)], cum[:: max(1, n // 400)])]}
    return res


def _scope_frames(df: pd.DataFrame):
    """(scope_key, frame) for: cost x ordering x (pooled | coin)."""
    if "cost_mult" not in df:
        df = df.assign(cost_mult=1.0)
    for cm in sorted(df["cost_mult"].unique()):
        d1 = df[df["cost_mult"] == cm]
        for od in ["all"] + sorted(d1["ordering"].unique()):
            d2 = d1 if od == "all" else d1[d1["ordering"] == od]
            yield f"x{cm:g}|{od}|pooled", d2
            for coin in sorted(d2["coin"].unique()):
                yield f"x{cm:g}|{od}|{coin}", d2[d2["coin"] == coin]


def sanity_flags(df: pd.DataFrame, bar_stats: Optional[Dict[str, dict]], nulls: Optional[pd.DataFrame], cost_bp: float,
                 real_net_lb: Optional[float] = None) -> list:
    """List of {flag, status PASS|WARN|NA, detail}. ``df``: trades of the 1x-cost LB ordering (or all if no ordering split)."""
    T = THRESHOLDS
    flags = []
    if len(df) == 0:
        return [{"flag": "no_trades", "status": "NA", "detail": "empty trade log"}]
    # 1 gross vs median range
    if bar_stats:
        bad, rows = [], []
        for coin, g in df.groupby("coin"):
            mr = (bar_stats.get(coin) or {}).get("median_range_bp")
            if mr:
                r = float(g["gross_bp"].mean()) / mr
                rows.append(f"{coin}: gross {g['gross_bp'].mean():.2f}bp / median range {mr:.1f}bp = {r:.2f}x")
                if g["gross_bp"].mean() > T["gross_vs_range"] * mr:
                    bad.append(coin)
        flags.append({"flag": "mean_gross_vs_median_bar_range", "status": "WARN" if bad else "PASS",
                      "detail": f"WARN if mean gross > {T['gross_vs_range']}x median range. " + "; ".join(rows)})
        bad2, rows2 = [], []
        for coin, g in df.groupby("coin"):
            st = bar_stats.get(coin) or {}
            qv, nd = st.get("median_qv"), st.get("n_days")
            if qv and nd:
                fpd = len(g) / nd
                ratio = fpd / (qv / 1000.0)
                rows2.append(f"{coin}: {fpd:.1f} fills/day, median qv ${qv:,.0f}/min, {ratio:.2f} fills/day per $1k")
                if ratio > T["fills_per_day_per_k_qv"]:
                    bad2.append(coin)
        flags.append({"flag": "fills_per_day_vs_median_quote_volume", "status": "WARN" if bad2 else "PASS",
                      "detail": f"WARN if fills/day per $1k of median minute volume > {T['fills_per_day_per_k_qv']}. " + "; ".join(rows2)})
    else:
        for nm in ("mean_gross_vs_median_bar_range", "fills_per_day_vs_median_quote_volume"):
            flags.append({"flag": nm, "status": "NA", "detail": "no per-coin bar stats supplied"})
    # 3 thin entry minutes
    if "entry_quote_volume" in df and df["entry_quote_volume"].notna().any():
        sh = float((df["entry_quote_volume"] < T["thin_entry_qv"]).mean())
        flags.append({"flag": "share_entry_quote_volume_below_20k", "status": "WARN" if sh > T["thin_share_warn"] else "PASS",
                      "detail": f"{sh:.1%} of trades enter on a minute with quote volume < ${T['thin_entry_qv']:,.0f} (WARN above {T['thin_share_warn']:.0%})"})
    else:
        flags.append({"flag": "share_entry_quote_volume_below_20k", "status": "NA", "detail": "no quote volume in the log"})
    # 4 gap fills
    if "gap_fill" in df:
        sh = float(df["gap_fill"].astype(bool).mean())
        flags.append({"flag": "share_gap_fills", "status": "WARN" if sh > T["gap_share_warn"] else "PASS",
                      "detail": f"{sh:.1%} of trades are gap fills at the open (WARN above {T['gap_share_warn']:.0%})"})
    # 5 nulls
    if nulls is not None and len(nulls):
        rows, bad = [], []
        for kind, g in nulls.groupby("null"):
            mean, p95 = float(g["net_lb"].mean()), float(g["net_lb"].quantile(0.95))
            thr = T["null_mean_vs_cost"] * cost_bp
            rows.append({"null": str(kind), "mean_net": _f(mean), "p95_net": _f(p95), "real_net": _f(real_net_lb), "threshold": _f(thr),
                         "status": "WARN" if mean > thr else "PASS"})
            if mean > thr:
                bad.append(kind)
        flags.append({"flag": "null_vs_real", "status": "WARN" if bad else "PASS", "table": rows,
                      "detail": f"a martingale null must lose about the cost ({cost_bp:g}bp); WARN when its mean net > {T['null_mean_vs_cost']} x cost"
                                + (f"; WARN: {', '.join(bad)}" if bad else "")})
    else:
        flags.append({"flag": "null_vs_real", "status": "NA", "detail": "no null draws supplied"})
    return flags


def build_report(trades_df: pd.DataFrame, nulls: Optional[pd.DataFrame] = None, cost_bp: float = 12.0,
                 bar_stats: Optional[Dict[str, dict]] = None, header: Optional[dict] = None, real_net_lb: Optional[float] = None) -> dict:
    """Report dict (JSON-able) from a trade log. Required columns: coin, ordering, net_bp, gross_bp, entry_ts, exit_ts, exit_reason, day
    (+ optional cost_mult, hold_min, entry_quote_volume, entry_range_bp, gap_fill, fee_bp, slip_bp).
    bar_stats: {coin: {median_range_bp, median_qv, n_days}} for the range / liquidity flags. nulls: DataFrame(null, net_lb, ...)."""
    df = trades_df.copy()
    if len(df) == 0:
        return {"header": header or {}, "cost_bp": cost_bp, "n_trades_total": 0, "scopes": {},
                "flags": sanity_flags(df, bar_stats, nulls, cost_bp, real_net_lb), "flags_basis": "empty"}
    if "cost_mult" not in df:
        df["cost_mult"] = 1.0
    rep = {"header": header or {}, "cost_bp": cost_bp, "n_trades_total": int(len(df)), "scopes": {}}
    for key, frame in _scope_frames(df):
        rep["scopes"][key] = analyze(frame)
    one = df[df["cost_mult"] == 1.0] if (df["cost_mult"] == 1.0).any() else df
    lb = one[one["ordering"] == "lb"] if "lb" in set(one["ordering"]) else one
    rep["flags"] = sanity_flags(lb, bar_stats, nulls, cost_bp, real_net_lb)
    rep["flags_basis"] = "x1 cost, LB ordering" if lb is not one else "x1 cost, all trades"
    return rep


# --------------------------------------------------------------------------- rendering
def _fmt(x, nd=2, pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if pct else (f"{x:,.{nd}f}" if isinstance(x, float) else str(x))


def _table(rows, cols, pcts=()):
    if not rows:
        return "_(no rows)_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c), pct=c in pcts) if c != "key" else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _scope_md(key: str, s: dict, full: bool) -> str:
    if s.get("n", 0) == 0:
        return f"### {key}\nلا صفقات (no trades).\n"
    c, d = s["concentration"], s["distribution"]
    o = [f"### {key}  (n={s['n']}, net total {_fmt(s['net_total'])} bp, net/trade {_fmt(s['net_mean'])} bp, gross/trade {_fmt(s['gross_mean'])} bp)\n"]
    rows = []
    for lab in ("top1", "top10", "top50", "top100", "top1pct", "top5pct"):
        rows.append({"key": lab, "k": c["by_net"][lab]["k"], "net_sum": c["by_net"][lab]["sum"], "share_of_net": c["by_net"][lab]["share"],
                     "gross_sum": c["by_gross"][lab]["sum"], "share_of_gross": c["by_gross"][lab]["share"]})
    o.append("**التركّز (Concentration)**\n\n" + _table(rows, ["key", "k", "net_sum", "share_of_net", "gross_sum", "share_of_gross"], ("share_of_net", "share_of_gross")))
    a = c["after_removing"]
    o.append(f"- after removing top100: net total {_fmt(a['top100']['net_total_after'])}, net/trade {_fmt(a['top100']['net_mean_after'])} bp; "
             f"after removing top 1%: net total {_fmt(a['top1pct']['net_total_after'])}, net/trade {_fmt(a['top1pct']['net_mean_after'])} bp\n"
             f"- best 5 days share of net: {_fmt(c['best5_days_share_of_net'], pct=True)} ({_fmt(c['best5_days_net'])} bp)\n"
             f"- biggest coin: {c['biggest_coin']['coin']} = {_fmt(c['biggest_coin']['share_of_net'], pct=True)} of net, "
             f"{_fmt(c['biggest_coin']['share_of_trades'], pct=True)} of trades; everything else: net {_fmt(c['biggest_coin']['rest_net_total'])} bp, "
             f"{_fmt(c['biggest_coin']['rest_net_mean'])} bp/trade\n")
    loco = [{"key": k, **v} for k, v in c["leave_one_coin_out"].items()]
    o.append("leave-one-coin-out:\n\n" + _table(loco, ["key", "coin_trades", "coin_net_total", "rest_trades", "rest_net_total", "rest_net_mean"]))
    if not full:
        return "\n".join(o)
    o.append("**التوزيع (Distribution)**\n\n" + "\n".join(f"- {k}: {_fmt(v, pct=k == 'win_rate')}" for k, v in d.items() if k != "histogram") + "\n")
    o.append(_table(d["histogram"], ["bucket", "n"]))
    cols = ["key", "n", "net_total", "net_mean", "share_of_net", "win_rate"]
    for title, k in (("حسب سبب الخروج (by exit_reason)", "by_exit_reason"), ("حسب الساعة UTC (by_hour_utc)", "by_hour_utc"),
                     ("حسب يوم الأسبوع (by_weekday)", "by_weekday"), ("حسب مدة الاحتفاظ (by_hold_min)", "by_hold_min"),
                     ("حسب حجم تداول دقيقة الدخول (by_entry_quote_volume)", "by_entry_quote_volume"),
                     ("حسب مدى شمعة الدخول (by_entry_range_bp)", "by_entry_range_bp")):
        o.append(f"**{title}**\n\n" + _table(s[k], cols, ("share_of_net", "win_rate")))
    bd = s["by_day"]
    o.append(f"**حسب اليوم (by_day)**: {bd['n_days']} days, positive {_fmt(bd['pct_positive_days'], pct=True)}, best {bd['best_day']['day']} "
             f"{_fmt(bd['best_day']['net'])} bp, worst {bd['worst_day']['day']} {_fmt(bd['worst_day']['net'])} bp\n")
    e = s["equity"]
    o.append(f"**منحنى الأسهم (equity, notional units)**: final {_fmt(e['final'])} bp, max drawdown {_fmt(e['max_drawdown_bp'])} bp, "
             f"time underwater {_fmt(e['time_underwater_share'], pct=True)}\n")
    return "\n".join(o)


def render_markdown(rep: dict) -> str:
    h = rep.get("header", {})
    o = ["# تقرير شبكة أوامر الإيقاف (stop_grid_1m) - trade-log report\n"]
    for k, v in h.items():
        o.append(f"- **{k}**: {v}")
    o.append("")
    o.append(f"## علامات السلامة (Sanity flags) - basis: {rep.get('flags_basis')}\n")
    for f in rep["flags"]:
        o.append(f"- **{f['status']}** `{f['flag']}`: {f['detail']}")
        if f.get("table"):
            o.append("\n" + _table(f["table"], ["null", "mean_net", "p95_net", "real_net", "threshold", "status"]))
    o.append(f"\n## الملخّص (Scopes)  - {rep['n_trades_total']} trades in the log\n")
    keys = list(rep["scopes"])
    main = [k for k in keys if k.endswith("|pooled") and "|all|" not in k]  # "all" (lb+ub mixed) stays in the JSON only
    for k in main:
        o.append(_scope_md(k, rep["scopes"][k], full=k.startswith("x1|")))
    o.append("\n## لكل عملة (per coin) - net/trade bp, x1 cost\n")
    rows = []
    for k in keys:
        cm, od, coin = k.split("|")
        if cm == "x1" and coin != "pooled" and od != "all" and rep["scopes"][k].get("n"):
            s = rep["scopes"][k]
            rows.append({"key": f"{coin} [{od}]", "n": s["n"], "net_total": s["net_total"], "net_mean": s["net_mean"], "gross_mean": s["gross_mean"],
                         "share_top10": s["concentration"]["by_net"]["top10"]["share"]})
    o.append(_table(rows, ["key", "n", "net_total", "net_mean", "gross_mean", "share_top10"], ("share_top10",)))
    return "\n".join(o) + "\n"


def save_plots(trades_df: pd.DataFrame, outdir: str) -> list:
    """PNGs (equity curve, net histogram, net by hour) when matplotlib imports; [] otherwise."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    import os
    os.makedirs(outdir, exist_ok=True)
    made = []
    d = trades_df[trades_df.get("cost_mult", 1.0) == 1.0] if "cost_mult" in trades_df else trades_df
    if not len(d):
        return made
    fig, ax = plt.subplots(figsize=(9, 4))
    for od, g in d.groupby("ordering"):
        g = g.sort_values("exit_ts")
        ax.plot(pd.to_datetime(g["exit_ts"], unit="ms"), g["net_bp"].cumsum(), label=str(od))
    ax.set_title("equity (cumulative net_bp)")
    ax.legend()
    p = os.path.join(outdir, "equity.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    made.append(p)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(d["net_bp"].clip(-200, 200), bins=80)
    ax.set_title("net_bp per trade (clipped +-200)")
    p = os.path.join(outdir, "net_hist.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    made.append(p)
    fig, ax = plt.subplots(figsize=(9, 4))
    h = pd.to_datetime(d["entry_ts"], unit="ms").dt.hour
    d.groupby(h)["net_bp"].sum().plot.bar(ax=ax)
    ax.set_title("net_bp total by entry hour UTC")
    p = os.path.join(outdir, "net_by_hour.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    made.append(p)
    return made

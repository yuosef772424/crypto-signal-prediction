"""
PURPOSE:  E-grid1m-001 runner: the owner's stop-order grid on 1-minute bars, tested honestly. Selects 5 coins on the SELECT
          window (first 40 percent of the common date range) by median 1m range / round-trip cost, scans a SMALL pre-declared
          parameter grid on DEV (next 30 percent) -> results/dev_grid.csv, freezes the best config by lower-bound net bp per
          trade (with a Bonferroni / Benjamini-Hochberg note on the day-clustered t), then runs TEST (last 30 percent) ONCE for
          that config -> results/test_frozen.csv, the sign-flip and random-walk nulls, a 2x-cost run and the accept rule of
          cards/E-grid1m-001.md (results/verdict.json).
TAGS:     stop grid 1m, E-grid1m-001, dev grid, frozen config, test once, sign-flip null, random walk null, Benjamini-Hochberg,
          Bonferroni, accept rule, coin selection, synthetic smoke
PITFALLS: SPEED knobs never change a number: signal caches (always on, bit-identical), `--engine numba` (opt-in JIT of the same
          event loop, default python; raises if numba is missing; verify with tools/stop_grid_bench.py --parity), `--null-draws N`
          (default 20 per kind; fewer = faster but a noisier null p95 for the accept rule: use the default for the verdict run) and
          `--jobs` (DEV scan and null draws run in forked workers, results in the same order). `--smoke` uses SYNTHETIC martingale data: its numbers are a pipeline check, never evidence. TEST is opened once: a
          second run refuses to overwrite results/test_frozen.csv (use --force-test only for a documented new config). The
          DEV pick is the best of many configs: judge it by the TEST rule, not by its DEV numbers. Funding ignored; LB = the
          worse of two fixed intrabar orderings (see tools/stop_grid_sim.py PITFALLS). Run from the repo root.

Usage:
    python research/stop_grid_1m/run_stop_grid_1m.py --data /content/drive/MyDrive/history_1m [--jobs 2] [--out research/stop_grid_1m/results]
    python research/stop_grid_1m/run_stop_grid_1m.py --data <dir> --signal-tf 5m --out <NEW dir>   # gates on 5m bars, execution 1m (new config family)
    python research/stop_grid_1m/run_stop_grid_1m.py --smoke          # synthetic, 2 weeks, 5+ coins, writes to a temp dir
    python tools/stop_grid_bench.py --parity                           # once per runtime BEFORE using --engine numba (exit code != 0 on mismatch)
    python research/stop_grid_1m/run_stop_grid_1m.py --data <dir> --jobs 2 --engine numba   # same numbers, faster (needs numba); --null-draws 5 for exploration
    # trade log + event log + report of an ALREADY FROZEN config (no grid re-run; never touches test_frozen.csv / verdict.json):
    python research/stop_grid_1m/run_stop_grid_1m.py --data <dir> --out <earlier out> --report-only [--report-scope dev|test|both] [--log-events]
    python research/stop_grid_1m/run_stop_grid_1m.py --data <dir> --out <earlier out> --report-only --config-json cfg.json   # explicit config: not frozen, not evidence
"""
import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stop_grid_sim as sg  # noqa: E402
import stop_grid_report as rp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED = dict(n_levels=10, max_positions=10, daily_loss_cap_bp=1000.0, taker_bp=5.0, maker_bp=2.0, slip_bps=1.0, slip_range_frac=0.0)

# --- the pre-declared DEV grid (192 configs). Nothing is added after seeing a result.
EXIT_VARIANTS = {
    "tp": dict(exit_mode="tp", tp=0.005),
    "trail_a05_g01": dict(exit_mode="trail", act=0.005, gap=0.001),
    "trail_a10_g03": dict(exit_mode="trail", act=0.010, gap=0.003),
    "split_a05_g01": dict(exit_mode="split", tp=0.005, act=0.005, gap=0.001),
}
REBUILD_VARIANTS = {
    "breakout_x3": dict(rebuild="breakout", breakout_x=3.0),
    "flat_c5": dict(rebuild="flat", flat_cooldown=5),
    "time_t60": dict(rebuild="time", time_every=60),
}
GATE_VARIANTS = {
    "none": dict(gates=()),
    "adx5m_25": dict(gates=("adx",), adx_tf="5m", adx_thr=25.0),
    "volexp_1.2": dict(gates=("vol",), vol_ratio=1.2),
    "costvia_3": dict(gates=("cost",), cost_k=3.0),
}
SPACINGS = (0.001, 0.002)
SLS = (0.005, 0.010)


def build_grid(smoke=False, signal_tf="1m"):
    """The pre-declared grid. ``signal_tf="5m"`` sets signal_tf (and adx_tf="5m") on EVERY config: same grid size, same names."""
    out = []
    spacings, sls = SPACINGS, SLS
    exits, rebs, gates = EXIT_VARIANTS, REBUILD_VARIANTS, GATE_VARIANTS
    if smoke:
        spacings, sls = SPACINGS[:1], SLS[:1]
        exits = {k: EXIT_VARIANTS[k] for k in ("tp", "trail_a05_g01")}
        rebs = {k: REBUILD_VARIANTS[k] for k in ("breakout_x3", "time_t60")}
        gates = {k: GATE_VARIANTS[k] for k in ("none", "volexp_1.2")}
    for s, sl, (en, e), (rn, r), (gn, g) in itertools.product(spacings, sls, exits.items(), rebs.items(), gates.items()):
        name = f"s{s}_sl{sl}_{en}_{rn}_{gn}"
        kw = dict(FIXED, spacing=s, sl=sl, **e, **r, **g)
        if signal_tf != "1m":
            kw.update(signal_tf=signal_tf, adx_tf=signal_tf)
        out.append((name, sg.GridConfig(**kw)))
    return out


# --------------------------------------------------------------------------- data
def slice_window(arrs, window):
    ts = arrs["ts"]
    a, b = np.searchsorted(ts, window[0], "left"), np.searchsorted(ts, window[1], "left")
    return {k: v[a:b] for k, v in arrs.items()}


def load_coins(files_by_symbol, coins, splits):
    data = {}
    for s in coins:
        d = sg.read_symbol(files_by_symbol[s], ["open", "high", "low", "close", "quote_volume"], splits.start, splits.end)
        data[s] = {"ts": d["timestamp"].to_numpy(dtype="int64"), "o": d["open"].to_numpy(float), "h": d["high"].to_numpy(float),
                   "l": d["low"].to_numpy(float), "c": d["close"].to_numpy(float), "qv": d["quote_volume"].to_numpy(float)}
    return data


_CACHES = {}


def coin_cache(key, arrs):
    """The SignalCache of one coin's window (gate series, 5m resample, null relative bars computed once and shared by every config,
    both orderings, the 1x/2x cost runs and the trade-log re-simulation). ``key`` only names the slot: a slot whose bars differ from
    ``arrs`` is rebuilt, so a stale name can never serve another window's series."""
    cc = _CACHES.get(key)
    if cc is None or not cc.matches(arrs["ts"], arrs["o"], arrs["h"], arrs["l"], arrs["c"]):
        cc = sg.SignalCache(arrs["ts"], arrs["o"], arrs["h"], arrs["l"], arrs["c"])
        _CACHES[key] = cc
    return cc


def gates_for(key, arrs, cfg):
    return sg.gate_array(arrs["ts"], arrs["o"], arrs["h"], arrs["l"], arrs["c"], cfg, cache=coin_cache(key, arrs))


def prewarm(windows, key_prefix, grid=None, nulls=False):
    """Fill the caches in the parent process BEFORE forking workers (they inherit the entries): every gate series of the grid,
    and optionally the null relative bars. Pure speed: nothing here changes a result."""
    for coin, a in windows.items():
        if len(a["ts"]) < 200:
            continue
        cc = coin_cache((key_prefix, coin), a)
        for _, cfg in (grid or []):
            sg.gate_array(a["ts"], a["o"], a["h"], a["l"], a["c"], cfg, cache=cc)
        if nulls:
            sg._null_base(a["o"], a["h"], a["l"], a["c"], cc)


# --------------------------------------------------------------------------- evaluation of one config on one window set
def evaluate(cfg, windows, key_prefix, cache_gates=True, engine="python"):
    """windows: {coin: arrays}. Per coin both orderings; pooled by ordering; LB = the ordering with the lower pooled mean net.
    ``cache_gates``: reuse the per-coin SignalCache named by ``key_prefix`` (off for the nulls: their prices differ per draw).
    ``engine``: 'python' (default) or 'numba' (same numbers, see tools/stop_grid_numba.py)."""
    per = {od: [] for od in sg.ORDERINGS}
    coin_stats = {od: {} for od in sg.ORDERINGS}
    tim = {od: [] for od in sg.ORDERINGS}
    for coin, a in windows.items():
        if len(a["ts"]) < 200:
            continue
        cc = coin_cache((key_prefix, coin), a) if cache_gates else None
        gate = sg.gate_array(a["ts"], a["o"], a["h"], a["l"], a["c"], cfg, cache=cc)
        n_days = (a["ts"][-1] - a["ts"][0]) / sg.MS_DAY
        for od in sg.ORDERINGS:
            d = sg.simulate(a["ts"], a["o"], a["h"], a["l"], a["c"], cfg, od, gate, engine=engine)
            d = d.assign(coin=coin, exit_ts=a["ts"][d["exit_bar"].to_numpy(dtype=int)] if len(d) else np.array([], dtype="int64"))
            per[od].append(d)
            s = sg.summarize(d, len(a["ts"]), n_days)
            coin_stats[od][coin] = s["net_bp"]
            tim[od].append((s["time_in_market"], s["fills_per_day"]))
    pooled = {}
    for od in sg.ORDERINGS:
        d = pd.concat(per[od], ignore_index=True) if per[od] else pd.DataFrame()
        pooled[od] = d
    mean_net = {od: (pooled[od]["net_bp"].mean() if len(pooled[od]) else 0.0) for od in sg.ORDERINGS}
    lb, ub = sorted(sg.ORDERINGS, key=lambda od: mean_net[od])
    row = {}
    for tag, od in (("lb", lb), ("ub", ub)):
        d = pooled[od]
        n = len(d)
        row[f"trades_{tag}"] = n
        row[f"gross_{tag}"] = float(d["gross_bp"].mean()) if n else float("nan")
        row[f"gross_noslip_{tag}"] = float(d["gross_noslip_bp"].mean()) if n else float("nan")
        row[f"net_{tag}"] = float(d["net_bp"].mean()) if n else float("nan")
        row[f"win_{tag}"] = float((d["net_bp"] > 0).mean()) if n else float("nan")
        row[f"t_{tag}"] = sg.day_clustered_t(d["net_bp"].to_numpy(), d["entry_ts"].to_numpy()) if n else float("nan")
        row[f"maxdd_bp_{tag}"] = sg.max_drawdown_bp(d["net_bp"].to_numpy(), d["exit_ts"].to_numpy()) if n else 0.0
        pos, tot = sg.sign_consistency(coin_stats[od])
        row[f"coins_pos_{tag}"], row[f"coins_{tag}"] = pos, tot
        row[f"time_in_market_{tag}"] = float(np.mean([x[0] for x in tim[od]])) if tim[od] else 0.0
        row[f"fills_per_day_{tag}"] = float(np.mean([x[1] for x in tim[od]])) if tim[od] else 0.0
    row["lb_order"], row["ub_order"] = lb, ub
    row["net_by_coin_lb"] = json.dumps({k: (None if v != v else round(v, 3)) for k, v in coin_stats[lb].items()})
    return row, pooled[lb]


def cfg_fields(cfg):
    d = dataclasses.asdict(cfg)
    d["gates"] = "+".join(d["gates"]) or "none"
    return d


def normal_sf(t):
    return 0.5 * math.erfc(t / math.sqrt(2.0)) if t == t else float("nan")


def add_multiple_testing(df, alpha=0.05):
    """One-sided p of the lower-bound day-clustered t (normal approximation), Bonferroni and BH over ALL configs in the grid."""
    m = len(df)
    p = df["t_lb"].map(normal_sf).fillna(1.0).to_numpy()
    df = df.copy()
    df["p_lb"] = p
    df["p_bonferroni"] = np.minimum(1.0, p * m)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        prev = min(prev, p[idx] * m / rank)
        q[idx] = prev
    df["q_bh"] = q
    df["bonferroni_pass"] = df["p_bonferroni"] < alpha
    df["bh_pass"] = df["q_bh"] < alpha
    return df


# --------------------------------------------------------------------------- parallel DEV scan
_DEV = {}


def _dev_task(i):
    name, cfg = _DEV["grid"][i]
    row, _ = evaluate(cfg, _DEV["windows"], "dev", engine=_DEV["engine"])
    return {"name": name, **cfg_fields(cfg), **row}


def scan_dev(grid, dev, jobs, engine="python"):
    _DEV["grid"], _DEV["windows"], _DEV["engine"] = grid, dev, engine
    t0 = time.time()
    prewarm(dev, "dev", grid)                              # gate series once in the parent: forked workers inherit them
    rows = []
    if jobs > 1 and "fork" in mp.get_all_start_methods():
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
            for k, r in enumerate(ex.map(_dev_task, range(len(grid)), chunksize=1)):
                rows.append(r)
                if (k + 1) % 10 == 0:
                    print(f"  dev {k + 1}/{len(grid)}  {time.time() - t0:.0f}s", flush=True)
    else:
        for k in range(len(grid)):
            rows.append(_dev_task(k))
            if (k + 1) % 10 == 0:
                print(f"  dev {k + 1}/{len(grid)}  {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- null runs
_NULL = {}
NULL_KINDS = (("sign_flip", sg.sign_flip_null), ("random_walk", sg.random_walk_null))


def _null_task(task):
    """One null draw (all coins) -> one result row. Seeds depend on (draw, coin) only, so the row is the same in any order/process."""
    kind, k = task
    fn = dict(NULL_KINDS)[kind]
    w = {}
    for j, (coin, a) in enumerate(_NULL["test"].items()):
        o, h, l, c = fn(a["ts"], a["o"], a["h"], a["l"], a["c"], 1000 + 100 * k + j, cache=coin_cache(("test", coin), a))
        w[coin] = {"ts": a["ts"], "o": o, "h": h, "l": l, "c": c}
    r, _ = evaluate(_NULL["cfg"], w, None, cache_gates=False, engine=_NULL["engine"])
    return {"null": kind, "draw": k, **{x: r[x] for x in ("trades_lb", "gross_lb", "gross_noslip_lb", "net_lb", "net_ub", "t_lb")}}


def null_runs(cfg, test, draws, jobs=1, engine="python"):
    """``draws`` sign-flip + ``draws`` random-walk nulls on the TEST window. Draw k uses the same seeds whatever ``draws`` is (a smaller
    ``--null-draws`` is a prefix of the default one). ``jobs`` > 1: draws run in forked workers (rows come back in the serial order)."""
    _NULL["cfg"], _NULL["test"], _NULL["engine"] = cfg, test, engine
    prewarm(test, "test", nulls=True)                      # relative bars of the real coins, once, before any fork
    tasks = [(kind, k) for kind, _ in NULL_KINDS for k in range(draws)]
    if jobs > 1 and "fork" in mp.get_all_start_methods():
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
            rows = list(ex.map(_null_task, tasks, chunksize=1))
    else:
        rows = [_null_task(t) for t in tasks]
    return pd.DataFrame(rows)


def accept_rule(real_1x, real_2x, null_df):
    sf = null_df[null_df["null"] == "sign_flip"]["net_lb"]
    p95 = float(np.nanquantile(sf, 0.95)) if len(sf) else float("nan")
    conds = {
        "1_pooled_LB_net_gt_0": bool(real_1x["net_lb"] > 0),
        "2_day_clustered_t_gt_3": bool(real_1x["t_lb"] > 3),
        "3_at_least_3_of_5_coins_positive": bool(real_1x["coins_pos_lb"] >= 3),
        "4_LB_net_above_sign_flip_null_p95": bool(real_1x["net_lb"] > p95),
        "5_LB_net_gt_0_at_2x_cost": bool(real_2x["net_lb"] > 0),
    }
    return {"conditions": conds, "sign_flip_null_p95_net_lb": p95, "verdict": "ACCEPT" if all(conds.values()) else "REJECT"}


# --------------------------------------------------------------------------- trade log / event log / report (descriptive, additive)
def trade_log(cfg, windows, split, want_events=False, events_max=2_000_000, cost_mults=(1.0, 2.0), engine="python"):
    """Re-simulate ``cfg`` on ``windows`` with recording on. Returns (trades_df, events_df, bar_stats). ordering = lb/ub by the pooled
    mean net per cost multiple (same rule as evaluate()); ordering_raw = hl/lh. Events are logged at 1x cost only (the numba engine has
    no event log: a run that logs events uses the python engine for that run, same trades either way)."""
    trades, events, left = [], [], int(events_max)
    caches = {coin: sg.SignalCache(a["ts"], a["o"], a["h"], a["l"], a["c"]) for coin, a in windows.items() if len(a["ts"]) >= 200}
    for cm in cost_mults:
        c = cfg.scaled_costs(cm)
        runs = {od: [] for od in sg.ORDERINGS}
        for coin, a in windows.items():
            if len(a["ts"]) < 200:
                continue
            gate = sg.gate_array(a["ts"], a["o"], a["h"], a["l"], a["c"], c, cache=caches[coin])
            for od in sg.ORDERINGS:
                ev = [] if (want_events and cm == 1.0 and left > 0) else None
                d = sg.simulate(a["ts"], a["o"], a["h"], a["l"], a["c"], c, od, gate, record=True, events=ev, event_max=max(left, 1),
                                engine="python" if ev is not None else engine)
                runs[od].append((coin, a, d))
                if ev:
                    left -= len(ev)
                    events.append(sg.events_frame(ev, a["ts"], coin, od))
        mean_net = {}
        for od in sg.ORDERINGS:
            allnet = np.concatenate([d["net_bp"].to_numpy() for _, _, d in runs[od]]) if runs[od] else np.array([])
            mean_net[od] = float(allnet.mean()) if len(allnet) else 0.0
        lb, ub = sorted(sg.ORDERINGS, key=lambda od: mean_net[od])
        label = {lb: "lb", ub: "ub"}
        for od in sg.ORDERINGS:
            for coin, a, d in runs[od]:
                t = sg.annotate_trades(d, a, c, coin, label[od], cache=caches[coin])
                t.insert(2, "ordering_raw", od)
                t.insert(0, "split", split)
                t.insert(1, "cost_mult", cm)
                trades.append(t)
    tdf = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    edf = pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=["ts", "coin", "ordering", *sg.EVENT_COLUMNS])
    if len(edf):
        edf["ordering_raw"] = edf["ordering"]
        edf["ordering"] = edf["ordering_raw"].map({od: "lb" if od == lb else "ub" for od in sg.ORDERINGS})
        edf.insert(0, "split", split)
    bar_stats = {}
    for coin, a in windows.items():
        if len(a["ts"]) > 1:
            bar_stats[coin] = {"median_range_bp": float(np.median((a["h"] - a["l"]) / a["c"] * 1e4)),
                               "median_qv": float(np.median(a["qv"])) if "qv" in a else None,
                               "n_days": float((a["ts"][-1] - a["ts"][0]) / sg.MS_DAY)}
    return tdf, edf, bar_stats


def write_report(rdir, cfg, windows_by_split, header, args, nulls=None, real_net_lb=None):
    """Writes <rdir>/trades.csv.gz, events.csv.gz (opt-in), report.md, report.json (+ PNGs if matplotlib). Never touches anything else."""
    os.makedirs(rdir, exist_ok=True)
    all_tr, all_ev, reps, md = [], [], {}, []
    for split, windows in windows_by_split.items():
        tdf, edf, bar_stats = trade_log(cfg, windows, split, args.log_events, args.log_events_max, engine=args.engine)
        all_tr.append(tdf)
        if len(edf):
            all_ev.append(edf)
        hdr = {**header, "split": split}
        n_ = nulls if split == "test" else None
        rep = rp.build_report(tdf, nulls=n_, cost_bp=cfg.round_trip_cost_bp, bar_stats=bar_stats, header=hdr,
                              real_net_lb=real_net_lb if split == "test" else None) if len(tdf) else rp.build_report(tdf, header=hdr)
        reps[split] = rep
        md.append(f"# ===== split: {split} =====\n\n" + rp.render_markdown(rep))
        print(f"report[{split}]: {len(tdf)} trade rows ({len(edf)} events)", flush=True)
    trades = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()
    trades.to_csv(os.path.join(rdir, "trades.csv.gz"), index=False, compression="gzip")
    if args.log_events:
        ev = pd.concat(all_ev, ignore_index=True) if all_ev else pd.DataFrame()
        ev.to_csv(os.path.join(rdir, "events.csv.gz"), index=False, compression="gzip")
    with open(os.path.join(rdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(md))
    jdump(reps, os.path.join(rdir, "report.json"))
    pngs = rp.save_plots(trades, rdir) if len(trades) else []
    print(f"report written to {rdir} (PNGs: {len(pngs) or 'skipped'})", flush=True)


def cfg_from_dict(d):
    d = dict(d)
    d["gates"] = tuple(d.get("gates", ()))
    return sg.GridConfig(**d)


def report_only(args, out):
    """Re-simulate ONLY the frozen (or explicit) config on the chosen split(s); no grid, no selection, no test_frozen/verdict writes."""
    need = ["splits.json", "selection.csv"] + ([] if args.config_json else ["frozen_config.json"])
    miss = [f for f in need if not os.path.exists(os.path.join(out, f))]
    if miss:
        sys.exit(f"--report-only needs an earlier run in {out}: missing {miss}")
    sp = json.load(open(os.path.join(out, "splits.json")))
    frozen = json.load(open(os.path.join(out, "frozen_config.json"))) if os.path.exists(os.path.join(out, "frozen_config.json")) else None
    if args.config_json:
        raw = json.load(open(args.config_json))
        cfg = cfg_from_dict(raw.get("config", raw))
        name = os.path.basename(args.config_json)
        label = (f"EXPLICIT config from {args.config_json}: NOT FROZEN, NOT EVIDENCE (report for diagnosis only; no accept rule applies)")
    else:
        cfg = cfg_from_dict(frozen["config"])
        name = frozen["name"]
        label = (f"descriptive re-run of an already-frozen config ({frozen['name']}, sha256 {frozen['sha256'][:12]}): no grid was re-run, "
                 f"test_frozen.csv and verdict.json were NOT touched; this report is not a new test and changes no verdict")
    coins = frozen["coins"] if frozen else pd.read_csv(os.path.join(out, "selection.csv")).query("eligible")["symbol"].head(args.top).tolist()
    syms = sg.list_symbols(args.data)
    window = types_ns(start=sp["select"]["start_ms"], end=sp["test"]["end_ms"])
    full = load_coins(syms, coins, window)
    scopes = ["dev", "test"] if args.report_scope == "both" else [args.report_scope]
    wins = {k: {c: slice_window(full[c], (sp[k]["start_ms"], sp[k]["end_ms"])) for c in coins} for k in scopes}
    header = {"mode": "report-only", "note": label, "config": name, "signal_tf": cfg.signal_tf, "config_json": json.dumps(dataclasses.asdict(cfg), default=list, sort_keys=True),
              "coins": ",".join(coins), "splits": json.dumps({k: sp[k] for k in scopes}), "data": args.data}
    nulls = real = None
    if not args.config_json and os.path.exists(os.path.join(out, "nulls.csv")):
        nulls = pd.read_csv(os.path.join(out, "nulls.csv"))
        tf = os.path.join(out, "test_frozen.csv")
        if os.path.exists(tf):
            t = pd.read_csv(tf)
            real = float(t[(t["scope"] == "pooled") & (t["cost_mult"] == 1.0)]["net_lb"].iloc[0])
    write_report(os.path.join(out, "report"), cfg, wins, header, args, nulls, real)
    return 0


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


# --------------------------------------------------------------------------- smoke data
def make_smoke_dir(tmp, days=14):
    n = days * 1440
    vols = {"AAA": 0.02, "BBB": 0.05, "CCC": 0.04, "DDD": 0.03, "EEE": 0.07, "FFF": 0.06, "GGG": 0.01, "ILLIQ": 0.09, "SHORT": 0.08}
    for i, (s, v) in enumerate(vols.items()):
        f = sg.synth_frame(n, 500 + i, daily_vol=v, qv_per_min=3_000.0 if s == "ILLIQ" else 80_000.0,
                           start_ms=1_700_000_000_000 // sg.MS_DAY * sg.MS_DAY)
        if s == "SHORT":
            f = f.iloc[: n // 2]
        f.to_csv(os.path.join(tmp, f"{s}.csv.gz"), index=False, compression="gzip")


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0])
    ap.add_argument("--data", help="dir with <SYMBOL>.csv.gz / .csv / .parquet or per-symbol sub-folders")
    ap.add_argument("--out", help="results dir (default research/stop_grid_1m/results; smoke: a temp dir)")
    ap.add_argument("--smoke", action="store_true", help="synthetic data, 2 weeks, tiny grid: pipeline check, NOT evidence")
    ap.add_argument("--signal-tf", choices=("1m", "5m"), default="1m",
                    help="timeframe of ALL gate indicators (execution stays 1m). 5m = a NEW configuration family: use a NEW --out")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1, help="forked workers for the DEV scan and the null draws (results identical for any value)")
    ap.add_argument("--engine", choices=sg.ENGINES, default="python",
                    help="simulation engine: python (default, reference) or numba (opt-in JIT, same trade table, needs numba; run tools/stop_grid_bench.py --parity once per runtime)")
    ap.add_argument("--null-draws", type=int, default=20, help="draws per null kind (default 20 = 40 null runs); fewer is a prefix of the same draws but a noisier p95 for the accept rule")
    ap.add_argument("--min-trades", type=int, default=300, help="DEV: a config needs this many pooled trades to be freezable")
    ap.add_argument("--min-quote-volume", type=float, default=20_000.0)
    ap.add_argument("--report-only", action="store_true",
                    help="re-simulate ONLY the frozen config of an earlier run in --out (no grid) and write <out>/report/; never touches test_frozen.csv / verdict.json")
    ap.add_argument("--report-scope", choices=("dev", "test", "both"), default="test", help="split(s) for the trade log / report (default test)")
    ap.add_argument("--log-events", action="store_true", help="also write report/events.csv.gz (orders placed/filled/cancelled, rebuilds, gate on/off, daily cap)")
    ap.add_argument("--log-events-max", type=int, default=2_000_000, help="max event rows (default 2,000,000)")
    ap.add_argument("--config-json", help="with --report-only: explicit GridConfig JSON for a report only (labelled NOT FROZEN, NOT EVIDENCE)")
    ap.add_argument("--no-report", action="store_true", help="skip the trade-log report after the normal flow")
    ap.add_argument("--force-test", action="store_true", help="overwrite an existing test_frozen.csv (only for a documented new config)")
    args = ap.parse_args(argv)
    try:
        sg.require_engine(args.engine)
    except RuntimeError as e:
        ap.error(str(e))
    if args.engine == "numba":
        print(f"engine numba: JIT ready in {sg.warmup_numba():.1f}s (compile or on-disk cache)", flush=True)

    t0 = time.time()
    tmp = None
    if args.config_json and not args.report_only:
        ap.error("--config-json is only for --report-only")
    if args.report_only:
        if args.signal_tf != "1m":
            ap.error("--report-only takes signal_tf from the frozen config (or --config-json); do not pass --signal-tf")
        if args.smoke or not args.data or not args.out:
            ap.error("--report-only needs --data and --out of an earlier run (not --smoke)")
        return report_only(args, args.out)
    if args.smoke:
        tmp = tempfile.mkdtemp(prefix="stop_grid_smoke_")
        make_smoke_dir(tmp)
        args.data = tmp
        args.null_draws = min(args.null_draws, 4)
        args.min_trades = min(args.min_trades, 30)
        args.min_quote_volume = 20_000.0
        print(f"SMOKE DATA DIR: {tmp}")
        print("SMOKE RUN: synthetic martingale data (9 fake coins, 14 days). Pipeline check only - NO evidence of any kind.")
    if not args.data:
        ap.error("--data is required (or --smoke)")
    if args.signal_tf != "1m" and not args.out and not args.smoke:
        ap.error("--signal-tf 5m is a NEW configuration family: pass a NEW --out (never the default results dir of the 1m run)")
    out = args.out or (os.path.join(tmp, "results") if args.smoke else os.path.join(HERE, "results"))
    os.makedirs(out, exist_ok=True)
    test_path = os.path.join(out, "test_frozen.csv")
    if os.path.exists(test_path) and not args.force_test and not args.smoke:
        sys.exit(f"{test_path} exists: TEST is used ONCE per frozen config (use --force-test for a documented new config).")

    # 1. selection on SELECT only
    syms = sg.list_symbols(args.data)
    print(f"{len(syms)} symbols in {args.data}", flush=True)
    sel, table, splits = sg.select_coins(args.data, top=args.top, taker_bp=FIXED["taker_bp"], slip_bps=FIXED["slip_bps"],
                                         min_quote_volume=args.min_quote_volume, symbols=syms)
    jdump(splits.as_dict(), os.path.join(out, "splits.json"))
    table.to_csv(os.path.join(out, "selection.csv"), index=False)
    print("splits (UTC):")
    for k, v in splits.as_dict().items():
        print(f"  {k:6s} {v['start']} -> {v['end']}")
    print("selection table (top 12 by score, eligible flag):")
    print(table.head(12)[["symbol", "median_range_bp", "median_qv", "coverage", "score", "eligible", "why"]].round(2).to_string(index=False))
    if len(sel) < min(args.top, 3):
        sys.exit(f"only {len(sel)} eligible coins: cannot run")
    print("SELECTED:", sel, flush=True)

    # 2. data per window
    full = load_coins(syms, sel, splits)
    dev = {s: slice_window(full[s], splits.dev) for s in sel}
    test = {s: slice_window(full[s], splits.test) for s in sel}
    for name, w, win in (("DEV", dev, splits.dev), ("TEST", test, splits.test)):
        exp = (win[1] - win[0]) / sg.MS_MIN
        cov = {s: round(len(a["ts"]) / exp, 3) for s, a in w.items()}
        print(f"{name} bars per coin: { {s: len(a['ts']) for s, a in w.items()} }  coverage {cov}")
        if min(cov.values()) < 0.9:
            print(f"  WARNING: {name} coverage below 0.9 for some selected coin (holes are NOT forward-filled)")

    # 3. DEV grid
    grid = build_grid(args.smoke, args.signal_tf)
    print(f"DEV grid: {len(grid)} configs x {len(sel)} coins x 2 orderings", flush=True)
    dev_df = scan_dev(grid, dev, args.jobs, args.engine)
    dev_df = add_multiple_testing(dev_df)
    dev_df.to_csv(os.path.join(out, "dev_grid.csv"), index=False)
    ok = dev_df[dev_df["trades_lb"] >= args.min_trades]
    if ok.empty:
        sys.exit("no config has enough DEV trades to freeze")
    best = ok.sort_values("net_lb", ascending=False).iloc[0]
    m = len(dev_df)
    print(f"\nDEV: {m} configs; best by LB net: {best['name']}  net_lb={best['net_lb']:.2f} bp  t_lb={best['t_lb']:.2f}  "
          f"p={best['p_lb']:.3g}  Bonferroni p={best['p_bonferroni']:.3g}  BH q={best['q_bh']:.3g}")
    print(f"     configs with LB net > 0: {int((dev_df['net_lb'] > 0).sum())}/{m}; Bonferroni passes: {int(dev_df['bonferroni_pass'].sum())}; "
          f"BH passes: {int(dev_df['bh_pass'].sum())}  (best-of-{m} by construction: DEV numbers are not evidence)")
    cfg = next(c for n_, c in grid if n_ == best["name"])
    frozen_cfg = dataclasses.asdict(cfg)
    if cfg.signal_tf == "1m":
        frozen_cfg.pop("signal_tf")                       # default: the config dict (and its sha256) is byte-identical to earlier runs
    frozen = {"name": best["name"], "signal_tf": cfg.signal_tf, "config": {**frozen_cfg, "gates": list(frozen_cfg["gates"])}, "n_configs_scanned": m,
              "dev": {k: (None if (isinstance(best[k], float) and best[k] != best[k]) else best[k])
                      for k in ("net_lb", "net_ub", "t_lb", "p_lb", "p_bonferroni", "q_bh", "bonferroni_pass", "bh_pass", "trades_lb", "coins_pos_lb")},
              "coins": sel, "splits": splits.as_dict()}
    frozen["sha256"] = hashlib.sha256(json.dumps(frozen["config"], sort_keys=True).encode()).hexdigest()
    jdump(frozen, os.path.join(out, "frozen_config.json"))

    # 4. TEST, once: real data at 1x and 2x cost, then the nulls
    print("\nTEST (once) for the frozen config ...", flush=True)
    rows, trades_1x = [], None
    res = {}
    for mult in (1.0, 2.0):
        r, tr = evaluate(cfg.scaled_costs(mult), test, "test", engine=args.engine)
        res[mult] = r
        rows.append({"scope": "pooled", "cost_mult": mult, "config": best["name"], **r})
        if mult == 1.0:
            trades_1x = tr
    # per-coin rows (1x), same global LB ordering
    for coin in sel:
        r, _ = evaluate(cfg, {coin: test[coin]}, "test", engine=args.engine)
        rows.append({"scope": coin, "cost_mult": 1.0, "config": best["name"], **r})
    test_df = pd.DataFrame(rows)
    test_df.to_csv(test_path, index=False)
    if trades_1x is not None and len(trades_1x):
        trades_1x.to_csv(os.path.join(out, "test_trades_lb.csv.gz"), index=False, compression="gzip")
    print(f"nulls: {args.null_draws} sign-flip + {args.null_draws} random-walk draws x {len(sel)} coins", flush=True)
    nul = null_runs(cfg, test, args.null_draws, args.jobs, args.engine)
    nul.to_csv(os.path.join(out, "nulls.csv"), index=False)
    verdict = accept_rule(res[1.0], res[2.0], nul)
    verdict.update({"synthetic": bool(args.smoke), "config": best["name"], "signal_tf": cfg.signal_tf, "frozen_sha256": frozen["sha256"],
                    "test_1x": {k: res[1.0][k] for k in ("trades_lb", "gross_lb", "gross_noslip_lb", "net_lb", "net_ub", "t_lb", "coins_pos_lb")},
                    "test_2x_net_lb": res[2.0]["net_lb"],
                    "null_summary": {k: {"mean_gross_lb": float(g["gross_lb"].mean()), "mean_net_lb": float(g["net_lb"].mean()),
                                         "p95_net_lb": float(g["net_lb"].quantile(0.95))} for k, g in nul.groupby("null")},
                    "elapsed_s": round(time.time() - t0, 1)})
    jdump(verdict, os.path.join(out, "verdict.json"))
    r1 = res[1.0]
    print(f"\nTEST pooled ({r1['trades_lb']} trades, LB ordering {r1['lb_order']}): gross {r1['gross_lb']:.2f} bp (no-slip {r1['gross_noslip_lb']:.2f}), "
          f"net LB {r1['net_lb']:.2f} / UB {r1['net_ub']:.2f} bp, t {r1['t_lb']:.2f}, coins>0 {r1['coins_pos_lb']}/{r1['coins_lb']}, "
          f"2x-cost net LB {res[2.0]['net_lb']:.2f}")
    for k, v in verdict["null_summary"].items():
        print(f"  null {k:12s}: mean gross {v['mean_gross_lb']:.2f} bp, mean net LB {v['mean_net_lb']:.2f} bp, p95 net LB {v['p95_net_lb']:.2f}")
    print("accept rule:", json.dumps(verdict["conditions"]))
    print(f"VERDICT: {verdict['verdict']}" + ("  (SYNTHETIC smoke data: not evidence)" if args.smoke else ""))
    if not args.no_report:
        try:
            scopes = ["dev", "test"] if args.report_scope == "both" else [args.report_scope]
            wins = {"dev": dev, "test": test}
            hdr = {"mode": "after the normal run", "note": "descriptive trade-log report of the frozen config; not a new test, changes no verdict"
                   + ("; SYNTHETIC smoke data: not evidence" if args.smoke else ""), "config": best["name"], "signal_tf": cfg.signal_tf, "coins": ",".join(sel),
                   "splits": json.dumps({k: splits.as_dict()[k] for k in scopes}), "data": args.data}
            write_report(os.path.join(out, "report"), cfg, {k: wins[k] for k in scopes}, hdr, args, nul, res[1.0]["net_lb"])
        except Exception as e:  # the report is additive: its failure must not hide a finished run, but it is printed loudly
            import traceback
            traceback.print_exc()
            print(f"WARNING: report failed ({e!r}); the run itself (test_frozen.csv, verdict.json) is complete")
    print(f"results in {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
PURPOSE:  Benchmark and parity check of the stop-grid simulator engines (tools/stop_grid_sim.py): wall times of the python engine
          without cache, the python engine with the per-coin SignalCache, and the optional numba JIT engine (its JIT warm-up
          reported separately) on seeded SYNTHETIC 1m data, plus a clearly labelled PROJECTION for a full grid run; and
          ``--parity``: the numba engine against the python engine over many configs x both orderings (+ the diagnostic 'rand')
          x seeds, exit code != 0 on any mismatch. Run ``--parity`` once per Colab runtime before ``--engine numba``.
TAGS:     stop grid, benchmark, numba, parity, speed, engine, E-grid1m-001
PITFALLS: Synthetic data only: timings depend on the machine and on how busy the grid is (trades per bar), so the projection
          is arithmetic on the measured per-run time (same bars per coin, no data loading, no null draws, no report), NOT a
          measurement of a real run. Parity needs numba: without it the command exits 2 (cannot verify), never 0. The
          python path is the reference: parity compares ints/strings exactly and floats within rtol 1e-9.

Usage:
    python tools/stop_grid_bench.py                          # 1 year, 1 coin, 8 configs
    python tools/stop_grid_bench.py --days 90 --coins 2 --configs 4 --repeat 2
    python tools/stop_grid_bench.py --parity                 # exit 0 only if the numba engine reproduces the python engine
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stop_grid_sim as sg  # noqa: E402

# ---- configs: BENCH_CFGS = a mix like the study's DEV grid; PARITY_CFGS = every code path of the simulator
BENCH_CFGS = [
    dict(spacing=0.001, sl=0.005, exit_mode="tp", rebuild="breakout"),
    dict(spacing=0.001, sl=0.005, exit_mode="trail", rebuild="flat", gates=("adx",), adx_tf="5m", adx_thr=25.0),
    dict(spacing=0.001, sl=0.005, exit_mode="split", rebuild="time", time_every=60, gates=("vol",), vol_ratio=1.2),
    dict(spacing=0.002, sl=0.010, exit_mode="trail", act=0.010, gap=0.003, rebuild="breakout", gates=("cost",), cost_k=0.5),
    dict(spacing=0.001, sl=0.005, exit_mode="tp", rebuild="time", time_every=60, gates=("adx",), adx_tf="5m", adx_thr=15.0),
    dict(spacing=0.002, sl=0.005, exit_mode="split", rebuild="breakout"),
    dict(spacing=0.002, sl=0.010, exit_mode="trail", rebuild="flat", gates=("vol",), vol_ratio=1.0),
    dict(spacing=0.001, sl=0.010, exit_mode="tp", rebuild="flat", gates=("cost",), cost_k=0.3),
]
PARITY_CFGS = {
    "trail": dict(spacing=0.001, sl=0.005, exit_mode="trail", rebuild="breakout"),
    "tp_time": dict(spacing=0.002, sl=0.01, exit_mode="tp", rebuild="time", time_every=30),
    "split_flat": dict(spacing=0.001, sl=0.005, exit_mode="split", rebuild="flat", slip_range_frac=0.1),
    "gated_cap": dict(spacing=0.001, sl=0.005, exit_mode="trail", rebuild="breakout", gates=("vol",), vol_ratio=1.0,
                      daily_loss_cap_bp=150.0, max_positions=3),
    "adx_cancel": dict(spacing=0.0015, sl=0.007, exit_mode="split", rebuild="once", gates=("adx",), adx_tf="5m", adx_thr=15.0,
                       cancel_opposite_on_fill=True),
    "tp_once": dict(spacing=0.001, sl=0.004, exit_mode="tp", tp=0.003, rebuild="once"),
    "cancel_opp_small": dict(spacing=0.0005, sl=0.003, exit_mode="trail", act=0.002, gap=0.0005, rebuild="breakout", n_levels=3,
                             cancel_opposite_on_fill=True, max_positions=2),
    "split_time_cap": dict(spacing=0.001, sl=0.006, exit_mode="split", rebuild="time", time_every=45, daily_loss_cap_bp=100.0),
    "all_gates_5m": dict(spacing=0.001, sl=0.005, exit_mode="trail", rebuild="flat", gates=("adx", "vol", "cost"), signal_tf="5m",
                         adx_tf="5m", adx_thr=10.0, vol_ratio=0.8, cost_k=0.3, adx_n=7, atr_fast=6, atr_slow=30, cost_window=30),
    "cost_2x": dict(spacing=0.001, sl=0.005, exit_mode="split", rebuild="breakout", taker_bp=10.0, maker_bp=4.0, slip_bps=2.0),
    "slip_range_nocap": dict(spacing=0.0008, sl=0.005, exit_mode="trail", rebuild="breakout", slip_range_frac=0.3, daily_loss_cap_bp=None),
    "wide_levels": dict(spacing=0.0006, sl=0.004, exit_mode="tp", tp=0.004, rebuild="flat", flat_cooldown=2, n_levels=20,
                        max_positions=25, breakout_x=1.5),
}


def make_data(n: int, seed: int, gaps: bool = True, daily_vol: float = 0.04):
    """Seeded synthetic 1m bars. ``gaps=True`` perturbs every open (so gap fills at the open happen; high/low widened to stay valid)."""
    ts, o, h, l, c = sg.synth_ohlc(n, seed, daily_vol=daily_vol)
    if gaps:
        rng = np.random.default_rng(seed + 99)
        o = o * np.exp(rng.normal(0.0, 0.0004, n))
        h, l = np.maximum(h, o), np.minimum(l, o)
    return ts, o, h, l, c


# --------------------------------------------------------------------------- parity
def frames_match(a: pd.DataFrame, b: pd.DataFrame, rtol: float = 1e-9):
    """(ok, why): same columns/rows; ints, bools and strings identical; floats within rtol."""
    if list(a.columns) != list(b.columns):
        return False, f"columns differ: {list(a.columns)} vs {list(b.columns)}"
    if len(a) != len(b):
        return False, f"trade count {len(a)} vs {len(b)}"
    for col in a.columns:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            if not np.allclose(x.astype(float), y.astype(float), rtol=rtol, atol=0.0, equal_nan=True):
                i = int(np.flatnonzero(~np.isclose(x.astype(float), y.astype(float), rtol=rtol, atol=0.0, equal_nan=True))[0])
                return False, f"column {col} differs at trade {i}: {x[i]!r} vs {y[i]!r}"
        elif not (x == y).all():
            i = int(np.flatnonzero(x != y)[0])
            return False, f"column {col} differs at trade {i}: {x[i]!r} vs {y[i]!r}"
    return True, ""


def summaries_match(a: dict, b: dict, rtol: float = 1e-9):
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) or isinstance(y, float):
            if not (np.isclose(x, y, rtol=rtol, atol=0.0, equal_nan=True)):
                return False, f"summary {k}: {x!r} vs {y!r}"
        elif x != y:
            return False, f"summary {k}: {x!r} vs {y!r}"
    return True, ""


def check_pair(ts, o, h, l, c, cfg, order, record, label, rtol=1e-9):
    """One python-vs-numba comparison. Returns a list of mismatch strings (empty = identical within tolerance)."""
    pa, sa = sg.simulate(ts, o, h, l, c, cfg, order, record=record, return_stats=True)
    nb, sb = sg.simulate(ts, o, h, l, c, cfg, order, record=record, return_stats=True, engine="numba")
    bad = []
    ok, why = frames_match(pa, nb, rtol)
    if not ok:
        bad.append(f"{label}: trades: {why}")
    if sa != sb:
        bad.append(f"{label}: stats {sa} vs {sb}")
    n_days = (ts[-1] - ts[0]) / sg.MS_DAY
    ok, why = summaries_match(sg.summarize(pa, len(ts), n_days), sg.summarize(nb, len(ts), n_days), rtol)
    if not ok:
        bad.append(f"{label}: {why}")
    return bad, len(pa)


def run_parity(days: int = 8, seeds=(11, 12, 13), rtol: float = 1e-9, verbose: bool = True) -> int:
    if not sg.numba_available():
        print("PARITY CANNOT RUN: numba is not installed (pip install numba). Exit 2 = not verified.")
        return 2
    t0 = time.time()
    n_runs = n_trades = 0
    bad = []
    for seed in seeds:
        for gaps in (True, False):
            ts, o, h, l, c = make_data(days * 1440, seed, gaps=gaps, daily_vol=0.04 + 0.01 * (seed % 3))
            for name, kw in PARITY_CFGS.items():
                cfg = sg.GridConfig(**kw)
                for order in sg.ORDERINGS + sg.DIAG_ORDERINGS:
                    for record in (False, True):
                        b, n = check_pair(ts, o, h, l, c, cfg, order, record, f"seed{seed}|gaps={gaps}|{name}|{order}|rec={record}", rtol)
                        bad += b
                        n_runs += 1
                        n_trades += n
        if verbose:
            print(f"  seed {seed}: {n_runs} runs, {n_trades} trades compared, {len(bad)} mismatches so far", flush=True)
    for b in bad[:20]:
        print("MISMATCH", b)
    print(f"PARITY {'OK' if not bad else 'FAILED'}: {n_runs} runs ({len(PARITY_CFGS)} configs x {len(sg.ORDERINGS + sg.DIAG_ORDERINGS)} orderings "
          f"x record on/off x {len(seeds)} seeds x gaps on/off), {n_trades} trades, {len(bad)} mismatches, {time.time() - t0:.0f}s")
    return 0 if not bad else 1


# --------------------------------------------------------------------------- benchmark
def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def run_bench(days: int, coins: int, n_cfg: int, repeat: int, seed: int, grid_cfgs: int, grid_coins: int) -> int:
    cfgs = [sg.GridConfig(**BENCH_CFGS[i % len(BENCH_CFGS)]) for i in range(n_cfg)]
    if n_cfg > len(BENCH_CFGS):
        cfgs = [c.replace(spacing=c.spacing * (1 + 0.1 * (i // len(BENCH_CFGS)))) for i, c in enumerate(cfgs)]
    data = [make_data(days * 1440, seed + k, gaps=True) for k in range(coins)]
    n_bars = days * 1440
    print(f"benchmark: {days} days ({n_bars} 1m bars) x {coins} coin(s) x {n_cfg} configs x 2 orderings, seeded synthetic, best of {repeat}")
    have_nb = sg.numba_available()

    def pass_(engine, use_cache):
        out = []
        caches = [sg.SignalCache(*d) if use_cache else None for d in data]     # cache build is part of the timed work
        for cfg in cfgs:
            for d, cc in zip(data, caches):
                r = sg.run_both(*d, cfg, engine=engine, cache=cc)
                out.append((r["LB"], r["UB"], len(r["trades"]["hl"]), len(r["trades"]["lh"])))
        return out

    results = {}
    rows = []
    paths = [("python, no cache", "python", False), ("python, with cache", "python", True)]
    for label, eng, use_cache in paths:
        best = None
        for _ in range(repeat):
            res, dt = timed(lambda: pass_(eng, use_cache))
            best = dt if best is None else min(best, dt)
        results[label] = res
        rows.append((label, best))
    jit_s = None
    if have_nb:
        jit_s = sg.warmup_numba()
        best = None
        for _ in range(repeat):
            res, dt = timed(lambda: pass_("numba", True))
            best = dt if best is None else min(best, dt)
        results["numba, with cache"] = res
        rows.append(("numba, with cache", best))
    base = rows[0][1]
    print(f"\n{'path':24s} {'wall s':>9s} {'per run s':>10s} {'speed-up vs python/no cache':>28s}")
    n_pairs = n_cfg * coins
    for label, dt in rows:
        print(f"{label:24s} {dt:9.2f} {dt / n_pairs:10.3f} {base / dt:27.2f}x")
    if have_nb:
        print(f"numba JIT warm-up (compile, or load from the on-disk cache; NOT in the numba row above): {jit_s:.1f}s")
    else:
        print("numba: not installed here -> not measured (pip install numba; then re-run)")
    # the paths must agree
    ref = results["python, no cache"]

    def agree(x, y, rtol):
        return all(a[2] == b[2] and a[3] == b[3] and summaries_match(a[0], b[0], rtol)[0] and summaries_match(a[1], b[1], rtol)[0]
                   for a, b in zip(x, y))

    same_py = agree(ref, results["python, with cache"], 0.0)
    print(f"\nresults identical, python cache vs no cache (exact): {'yes' if same_py else 'NO'}")
    if have_nb:
        ok = agree(ref, results["numba, with cache"], 1e-9)
        print(f"results identical, numba vs python (trades exact, metrics rtol 1e-9): {'yes' if ok else 'NO'}")
        same_py = same_py and ok
    # projection
    print(f"\nPROJECTION for {grid_cfgs} configs x {grid_coins} coins (projection from the benchmark, not a measurement; same bars per coin as above,")
    print("no data loading, no TEST/null/report runs):")
    for label, dt in rows:
        per = dt / n_pairs
        print(f"  {label:24s} ~ {per * grid_cfgs * grid_coins / 60:7.1f} min  ({per:.3f} s per config x coin)")
    return 0 if same_py else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0])
    ap.add_argument("--parity", action="store_true", help="numba vs python parity check on synthetic data; exit 0 only if identical (2 = numba missing)")
    ap.add_argument("--parity-days", type=int, default=8, help="days of 1m bars per parity series (default 8)")
    ap.add_argument("--parity-seeds", type=int, default=3, help="number of seeds (default 3)")
    ap.add_argument("--days", type=int, default=365, help="benchmark: days of 1m bars per coin (default 365)")
    ap.add_argument("--coins", type=int, default=1)
    ap.add_argument("--configs", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=1, help="benchmark: repeats per path, best kept")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--proj-configs", type=int, default=192, help="projection: configs of the full grid (default 192)")
    ap.add_argument("--proj-coins", type=int, default=5, help="projection: coins (default 5)")
    a = ap.parse_args(argv)
    if a.parity:
        return run_parity(a.parity_days, tuple(range(11, 11 + a.parity_seeds)))
    return run_bench(a.days, a.coins, a.configs, a.repeat, a.seed, a.proj_configs, a.proj_coins)


if __name__ == "__main__":
    sys.exit(main())

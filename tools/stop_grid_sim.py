"""
PURPOSE:  Event-driven simulator of the owner's stop-order grid on 1-minute OHLC bars: N buy-stop-market orders above and N
          sell-stop-market orders below a reference price, every fill with an initial stop-loss and a fixed take-profit
          (limit), a trailing exit that activates only after a favourable move, or half/half; the grid is rebuilt by one
          trigger at a time (breakout, after-flat, time) and optionally gated by regime information known one bar earlier
          (ADX on 1m/5m, fast/slow ATR ratio, cost-viability). Runs BOTH intrabar orderings (O-H-L-C and O-L-H-C): LB = the
          worse, UB = the better (not a worst-case bound: see PITFALLS). Also holds the null baselines (sign-flip of whole days, pure random walk), the metrics
          (day-clustered t, drawdown, time in market), the cost-to-volatility coin selection with the SELECT/DEV/TEST split,
          the 1m file loaders and a synthetic 1m generator (tests and `--smoke` only).
TAGS:     stop grid, buy stop, sell stop, trailing stop activation, take profit, 1m bars, intrabar ordering, lower bound,
          upper bound, rebuild trigger, breakout, regime gate, ADX, ATR ratio, cost viability, sign-flip null, random walk null,
          day-clustered t, coin selection, high volatility coins, slippage, taker fee, maker fee, E-grid1m-001, simulation
PITFALLS: Only OHLC is known: the order of the high and the low inside a bar is not, so trust LB only (same convention as
          tools/bracket_eval.py). LB is the worse of TWO FIXED global orderings, not a per-bar worst case: both can be
          optimistic (a bar visited O-H-L always lets the up-fills AND the down-fills profit from their own extreme), so the
          honest comparison is against the null baselines run with the same orderings; the 'rand' ordering (random per bar)
          is a diagnostic (on martingale bridge data LB, UB and rand agree within ~0.5 bp gross); a touch of a trigger counts as a fill (also for the TP limit: optimistic, it is part of UB).
          A bar that opens beyond a stop trigger fills at the OPEN plus slippage; a TP limit that is gapped through fills at the
          limit price (no gap bonus). Funding is ignored. Gates are shifted one bar: the value used at bar i only uses bars
          <= i-1 (test: no look-ahead), the grid reference price is close[i-1]. Positions are 1 unit of notional per fill
          (split = 2 legs of 0.5), PnL in bp of notional, no compounding. Missing minutes are NOT forward-filled: the jump
          is treated as a gap, and 'time'/'flat' counters count bars, not minutes. Trades open at the end of the data are
          closed at the last close (taker). A DAILY LOSS CAP biases the per-trade MEAN of a martingale negative (the number of
          trades depends on the outcomes; the per-day total stays unbiased): measured gross -1 to -2 bp/trade at cap 300 bp, ~0
          at 1000 bp, so the default is 1000 and the nulls must use the same config. Related closed failures: F-0038, F-0061, F-0062 in
          docs/research/failure_registry.csv - the prior is gross ~ 0 before costs; the sim gives no edge by construction
          (tests assert gross ~ 0 and net ~ -cost on nulls). Pure numpy/pandas/python (no numba needed).

Usage:
    import sys; sys.path.insert(0, "tools"); import stop_grid_sim as sg
    cfg = sg.GridConfig(spacing=0.001, n_levels=10, exit_mode="trail", act=0.005, gap=0.001, rebuild="breakout", gates=("adx",))
    res = sg.run_both(ts_ms, o, h, l, c, cfg)          # {'LB': summary, 'UB': summary, 'trades': {'hl': df, 'lh': df}}
    sel = sg.select_coins("/content/drive/MyDrive/history_1m")   # SELECT window only (first 40 percent of the common range)
"""
from __future__ import annotations

import dataclasses
import glob
import gzip
import io
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MS_MIN = 60_000
MS_DAY = 86_400_000
EXIT_MODES = ("tp", "trail", "split")
REBUILD_MODES = ("once", "breakout", "flat", "time")
GATES = ("adx", "vol", "cost")
ORDERINGS = ("hl", "lh")                      # O-H-L-C and O-L-H-C
DIAG_ORDERINGS = ("rand",)                    # diagnostic only: a random ordering per bar (fixed seed)


# --------------------------------------------------------------------------- configuration
@dataclass(frozen=True)
class GridConfig:
    """One strict frozen config (unknown names raise: no silent defaults). Distances are fractions (0.001 = 0.1 percent)."""
    n_levels: int = 10                    # N orders per side
    spacing: float = 0.001                # s: level k at P0 * (1 +/- k*s)
    sl: float = 0.005                     # S: initial stop-loss distance from the entry
    exit_mode: str = "trail"              # 'tp' | 'trail' | 'split' (half TP limit, half trailing)
    tp: float = 0.005                     # fixed take-profit distance (tp and split)
    act: float = 0.005                    # A: trailing activates after this favourable move from the order's entry
    gap: float = 0.001                    # G: trailing gap from the extreme
    cancel_opposite_on_fill: bool = False
    rebuild: str = "breakout"             # 'once' | 'breakout' | 'flat' | 'time' (exactly one trigger)
    breakout_x: float = 3.0               # breakout: close beyond the outermost level by more than X*s
    flat_cooldown: int = 5                # flat: bars after the last close before the rebuild (needs >= 1 fill in the grid)
    time_every: int = 60                  # time: rebuild every T bars
    max_positions: int = 10               # concurrent parent positions; a trigger at the cap cancels that order
    daily_loss_cap_bp: Optional[float] = 1000.0  # sum of closed net bp in a UTC day <= -cap: cancel pending, no new grid today
    gates: Tuple[str, ...] = ()           # subset of GATES, all must be true to (re)build
    adx_thr: float = 25.0
    adx_tf: str = "1m"                    # '1m' or '5m' (5m = resampled, last COMPLETED 5m bar)
    signal_tf: str = "1m"                 # timeframe of ALL gate indicators: '1m' (default) or '5m' (needs adx_tf='5m'); execution stays 1m
    adx_n: int = 14
    atr_fast: int = 14
    atr_slow: int = 100
    vol_ratio: float = 1.2                # fast/slow ATR ratio >= r
    cost_k: float = 3.0                   # median recent 1m range >= k * round-trip cost
    cost_window: int = 60
    taker_bp: float = 5.0                 # per side, stop entries and stop exits
    maker_bp: float = 2.0                 # TP limit
    slip_bps: float = 1.0                 # fixed adverse slippage on stop fills (per fill)
    slip_range_frac: float = 0.0          # plus this fraction of the bar's (high-low), in price

    def __post_init__(self):
        if self.exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode {self.exit_mode!r} not in {EXIT_MODES}")
        if self.rebuild not in REBUILD_MODES:
            raise ValueError(f"rebuild {self.rebuild!r} not in {REBUILD_MODES}")
        if self.adx_tf not in ("1m", "5m"):
            raise ValueError("adx_tf must be '1m' or '5m'")
        if self.signal_tf not in ("1m", "5m"):
            raise ValueError("signal_tf must be '1m' or '5m'")
        if self.signal_tf == "5m" and self.adx_tf != "5m":
            raise ValueError("signal_tf='5m' conflicts with adx_tf='1m': all gates run on 5m bars, set adx_tf='5m' too")
        bad = [g for g in self.gates if g not in GATES]
        if bad:
            raise ValueError(f"unknown gates {bad}; known {GATES}")
        if self.n_levels < 1 or self.spacing <= 0 or self.sl <= 0 or self.tp <= 0 or self.act <= 0 or self.gap <= 0:
            raise ValueError("n_levels >= 1 and spacing, sl, tp, act, gap > 0 required")
        if self.max_positions < 1 or self.flat_cooldown < 1 or self.time_every < 1:
            raise ValueError("max_positions, flat_cooldown, time_every >= 1 required")

    def replace(self, **kw) -> "GridConfig":
        return dataclasses.replace(self, **kw)

    def scaled_costs(self, mult: float) -> "GridConfig":
        """Fees and slippage times ``mult`` (e.g. 2.0 for the 'still positive at 2x cost' check)."""
        return self.replace(taker_bp=self.taker_bp * mult, maker_bp=self.maker_bp * mult, slip_bps=self.slip_bps * mult,
                            slip_range_frac=self.slip_range_frac * mult)

    @property
    def round_trip_cost_bp(self) -> float:
        """Taker both sides plus slippage both sides (the cost a stop-in / stop-out trade pays)."""
        return 2.0 * self.taker_bp + 2.0 * self.slip_bps


# --------------------------------------------------------------------------- gates (no look-ahead: value at i uses bars <= i-1)
def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _adx_values(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    """Wilder ADX at each bar close (same construction as research/edge_discovery/33_regime_filters.py)."""
    up = np.r_[np.nan, np.diff(h)]
    dn = np.r_[np.nan, -np.diff(l)]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    atr = _wilder(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * _wilder(pdm, n) / atr
        ndi = 100.0 * _wilder(ndm, n) / atr
        dx = 100.0 * np.abs(pdi - ndi) / (pdi + ndi)
    dx = np.nan_to_num(dx, nan=0.0)
    out = np.array(_wilder(dx, n))
    out[: n] = np.nan                                   # warm-up
    return out


def resample_5m(ts, h, l, c):
    """1m -> 5m with UTC-aligned buckets (ts // 5min). Returns (h5, l5, c5, inv): ``inv[i]`` = index of the 5m bar holding 1m bar i.
    Only buckets that contain at least one minute exist (a bucket with missing minutes is built from the minutes present: max high,
    min low, last close; a fully missing bucket is simply absent from the 5m series, never forward-filled)."""
    h, l, c = (np.asarray(a, dtype="float64") for a in (h, l, c))
    bucket = np.asarray(ts, dtype="int64") // (5 * MS_MIN)
    _, inv = np.unique(bucket, return_inverse=True)
    g = pd.DataFrame({"h": h, "l": l, "c": c, "b": inv}).groupby("b", sort=True)
    return g["h"].max().to_numpy(), g["l"].min().to_numpy(), g["c"].last().to_numpy(), inv


def adx_signal(ts, h, l, c, n: int = 14, tf: str = "1m") -> np.ndarray:
    """ADX known at the START of each 1m bar. '1m': ADX of bar i-1. '5m': ADX of the last COMPLETED 5m bar before bar i's bucket."""
    h, l, c = (np.asarray(a, dtype="float64") for a in (h, l, c))
    if tf == "1m":
        a = _adx_values(h, l, c, n)
        return np.r_[np.nan, a[:-1]]
    h5, l5, c5, inv = resample_5m(ts, h, l, c)
    a5 = _adx_values(h5, l5, c5, n)
    prev = np.r_[np.nan, a5[:-1]]                       # the previous bucket is complete when this bucket's bars trade
    return prev[inv]


def _true_range(h, l, c) -> np.ndarray:
    pc = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr_ratio_signal(h, l, c, fast: int = 14, slow: int = 100) -> np.ndarray:
    """Fast/slow ATR (simple means of the true range) known at the start of each bar (shifted one bar)."""
    tr = pd.Series(_true_range(np.asarray(h, float), np.asarray(l, float), np.asarray(c, float)))
    r = (tr.rolling(fast).mean() / tr.rolling(slow).mean()).to_numpy()
    return np.r_[np.nan, r[:-1]]


def median_range_bp_signal(h, l, c, window: int = 60) -> np.ndarray:
    """Median of the last ``window`` 1m (high-low)/close in bp, known at the start of each bar (shifted one bar)."""
    rng = pd.Series((np.asarray(h, float) - np.asarray(l, float)) / np.asarray(c, float) * 1e4)
    m = rng.rolling(window).median().to_numpy()
    return np.r_[np.nan, m[:-1]]


def signal_values(ts, h, l, c, cfg: GridConfig, which=("adx", "vol", "cost")) -> dict:
    """Per-1m-bar gate indicator values known at the bar's open: 'adx', 'vol' (fast/slow ATR ratio), 'cost' (median range in bp).
    ``cfg.signal_tf='5m'``: vol and cost are computed on 5m bars (periods counted in 5m bars; cost_window // 5 bars, min 3) and a 1m bar
    sees the last COMPLETED 5m bar only (piecewise constant over each 5-minute block). adx uses ``cfg.adx_tf``."""
    out = {}
    if "adx" in which:
        out["adx"] = adx_signal(ts, h, l, c, cfg.adx_n, cfg.adx_tf)
    if cfg.signal_tf == "1m":
        if "vol" in which:
            out["vol"] = atr_ratio_signal(h, l, c, cfg.atr_fast, cfg.atr_slow)
        if "cost" in which:
            out["cost"] = median_range_bp_signal(h, l, c, cfg.cost_window)
        return out
    h5, l5, c5, inv = resample_5m(ts, h, l, c)
    if "vol" in which:
        out["vol"] = atr_ratio_signal(h5, l5, c5, cfg.atr_fast, cfg.atr_slow)[inv]
    if "cost" in which:
        out["cost"] = median_range_bp_signal(h5, l5, c5, max(3, cfg.cost_window // 5))[inv]
    return out


def gate_array(ts, o, h, l, c, cfg: GridConfig) -> Optional[np.ndarray]:
    """Boolean per bar: all gates in ``cfg.gates`` true (NaN -> False). None when there are no gates."""
    if not cfg.gates:
        return None
    ok = np.ones(len(c), dtype=bool)
    v = signal_values(ts, h, l, c, cfg, which=cfg.gates)
    with np.errstate(invalid="ignore"):
        if "adx" in cfg.gates:
            ok &= v["adx"] >= cfg.adx_thr
        if "vol" in cfg.gates:
            ok &= v["vol"] >= cfg.vol_ratio
        if "cost" in cfg.gates:
            ok &= v["cost"] >= cfg.cost_k * cfg.round_trip_cost_bp
    return ok


# --------------------------------------------------------------------------- the simulator
class _Leg:
    __slots__ = ("par", "w", "kind", "sl", "tp", "act", "activated", "ext", "trail")


class _Par:
    __slots__ = ("side", "entry_ns", "entry", "bar", "k", "open_legs", "gross", "gross_ns", "net", "exit_bar", "kinds",
                 "xval", "fee", "gapfill", "rid", "conc", "mfe", "mae")


class _Grid:
    __slots__ = ("p0", "bar", "fills", "top", "bot")


class _Sim:
    def __init__(self, ts, o, h, l, c, cfg: GridConfig, order: str, gate: Optional[np.ndarray],
                 record: bool = False, events: Optional[list] = None, event_max: int = 2_000_000):
        if order not in ORDERINGS + DIAG_ORDERINGS:
            raise ValueError(f"order must be one of {ORDERINGS + DIAG_ORDERINGS}")
        self.rand = np.random.default_rng(12345).random(len(c)).tolist() if order == "rand" else None
        self.cfg, self.order = cfg, order
        self.ts = np.asarray(ts, dtype="int64")
        self.o, self.h, self.l, self.c = (np.asarray(a, dtype="float64").tolist() for a in (o, h, l, c))
        self.days = (self.ts // MS_DAY).tolist()
        self.gate = None if gate is None else np.asarray(gate, dtype=bool).tolist()
        self.buys: List[list] = []           # [trigger, k], nearest (lowest) first
        self.sells: List[list] = []          # [trigger, k], nearest (highest) first
        self.legs: List[_Leg] = []
        self.grid: Optional[_Grid] = None
        self.n_open_par = 0
        self.last_close_bar = -10 ** 9
        self.day = None
        self.day_pnl = 0.0
        self.halted = False
        self.rows: List[tuple] = []
        self.stats = {"skipped_cap": 0, "halted_days": 0, "builds": 0}
        self.i = 0
        self.rng_i = 0.0
        # additive recording (off by default; never read by the simulation logic)
        self.rec = bool(record)
        self.ev, self.ev_max = events, int(event_max)
        self.extra: List[tuple] = []
        self.rid = 0
        self.why = "initial"
        self.gate_state = None

    def _ev(self, i: int, kind: str, side: int, level: int, price: float, qty: float, reason: str):
        if self.ev is not None and len(self.ev) < self.ev_max:
            self.ev.append((i, kind, side, level, price, qty, reason))

    # ----- grid
    def _build(self, i: int):
        cfg = self.cfg
        p0 = self.c[i - 1]
        s, n = cfg.spacing, cfg.n_levels
        self.buys = [[p0 * (1.0 + k * s), k] for k in range(1, n + 1)]
        self.sells = [[p0 * (1.0 - k * s), k] for k in range(1, n + 1)]
        g = _Grid()
        g.p0, g.bar, g.fills = p0, i, 0
        g.top = p0 * (1.0 + (n + cfg.breakout_x) * s)
        g.bot = p0 * (1.0 - (n + cfg.breakout_x) * s)
        self.grid = g
        self.stats["builds"] += 1
        if self.rec or self.ev is not None:
            self.rid += 1
            self._ev(i, "rebuilt", 0, 0, p0, float(2 * n), f"{self.why}#{self.rid}")
            if self.ev is not None:
                for k in range(1, n + 1):
                    self._ev(i, "placed", 1, k, self.buys[k - 1][0], 1.0, f"buy_stop#{self.rid}")
                    self._ev(i, "placed", -1, k, self.sells[k - 1][0], 1.0, f"sell_stop#{self.rid}")

    def _cancel_pending(self, why: str = "rebuild"):
        if self.ev is not None and (self.buys or self.sells):
            self._ev(self.i, "cancelled", 0, 0, float("nan"), float(len(self.buys) + len(self.sells)), why)
        self.buys, self.sells, self.grid = [], [], None

    def _decide(self, i: int):
        cfg = self.cfg
        if self.halted:
            if self.buys or self.sells or self.grid is not None:
                self._cancel_pending("halted")
            return
        g = self.grid
        need = g is None
        if not need:
            m = cfg.rebuild
            self.why = m
            if m == "breakout":
                pc = self.c[i - 1]
                need = pc > g.top or pc < g.bot
            elif m == "flat":
                need = g.fills > 0 and self.n_open_par == 0 and i - self.last_close_bar >= cfg.flat_cooldown
            elif m == "time":
                need = i - g.bar >= cfg.time_every
        if not need:
            return
        if g is None:
            self.why = "initial"
        self._cancel_pending("rebuild")
        if self.gate is None or self.gate[i]:
            self._build(i)

    # ----- fills
    def _slip(self, base: float) -> float:
        return base * self.cfg.slip_bps / 1e4 + self.cfg.slip_range_frac * self.rng_i

    def _track(self, p: _Par, px: float):
        m = p.side * (px / p.entry - 1.0) * 1e4
        if m > p.mfe:
            p.mfe = m
        if m < p.mae:
            p.mae = m

    def _enter(self, side: int, trig: float, k: int, cf: float, i: int, gap: bool = False):
        cfg = self.cfg
        base = max(trig, cf) if side > 0 else min(trig, cf)
        fill = base + side * self._slip(base)
        p = _Par()
        p.side, p.entry_ns, p.entry, p.bar, p.k = side, base, fill, i, k
        p.gross = p.gross_ns = p.net = 0.0
        p.exit_bar = i
        p.kinds = []
        p.xval = p.fee = p.mfe = p.mae = 0.0
        p.gapfill, p.rid = gap, self.rid
        specs = {"tp": [("tp", 1.0)], "trail": [("trail", 1.0)], "split": [("tp", 0.5), ("trail", 0.5)]}[cfg.exit_mode]
        p.open_legs = len(specs)
        for kind, w in specs:
            lg = _Leg()
            lg.par, lg.w, lg.kind = p, w, kind
            lg.sl = base * (1.0 - side * cfg.sl)
            lg.tp = base * (1.0 + side * cfg.tp)
            lg.act = base * (1.0 + side * cfg.act)
            lg.activated, lg.ext, lg.trail = False, base, 0.0
            self.legs.append(lg)
        self.n_open_par += 1
        p.conc = self.n_open_par
        if self.grid is not None:
            self.grid.fills += 1
        if self.ev is not None:
            self._ev(i, "filled", side, k, fill, 1.0, "entry_gap" if gap else "entry")

    def _exit(self, leg: _Leg, base: float, limit: bool, i: int, kind: str):
        cfg, p = self.cfg, leg.par
        side, w = p.side, leg.w
        if limit:
            fill, fee = base, cfg.maker_bp
        else:
            fill, fee = base - side * self._slip(base), cfg.taker_bp
        gross = side * (fill / p.entry - 1.0) * 1e4 * w
        gross_ns = side * (base / p.entry_ns - 1.0) * 1e4 * w
        p.gross += gross
        p.gross_ns += gross_ns
        p.net += gross - (cfg.taker_bp + fee) * w
        p.kinds.append(kind)
        p.exit_bar = i
        if self.rec or self.ev is not None:
            p.xval += fill * w
            p.fee += (cfg.taker_bp + fee) * w
            self._track(p, fill)
            self._ev(i, "filled", -side, p.k, fill, w, "exit_" + kind)
        self.legs.remove(leg)
        p.open_legs -= 1
        if p.open_legs == 0:
            self.n_open_par -= 1
            self.last_close_bar = i
            self.rows.append((p.bar, i, p.side, p.k, p.gross, p.gross_ns, p.net, "+".join(p.kinds)))
            if self.rec:
                self.extra.append((p.entry_ns, p.entry, p.xval, p.fee, p.gross_ns - p.gross, p.mfe, p.mae, p.gapfill, p.rid, p.conc))
            self.day_pnl += p.net
            cap = cfg.daily_loss_cap_bp
            if cap is not None and not self.halted and self.day_pnl <= -cap:
                self.halted = True
                self.stats["halted_days"] += 1
                self._ev(i, "daily_cap_hit", 0, 0, float("nan"), 0.0, f"day_pnl_bp={self.day_pnl:.1f}")
                self._cancel_pending("daily_cap")

    # ----- one monotone price segment p -> q (gap=True: the opening jump, stops fill at the open q)
    def _seg(self, p: float, q: float, gap: bool, i: int):
        if q == p:
            return
        up = q > p
        cur = p
        cfg = self.cfg
        while True:
            best = None                                    # (sortkey, prio, trig, kind, obj)
            if up:
                if self.buys and self.buys[0][0] <= q:
                    t = self.buys[0][0]
                    best = ((t, 1), t, "entry", self.buys[0])
                for lg in self.legs:
                    if lg.par.side > 0:
                        if lg.kind == "tp" and lg.tp <= q:
                            cand = ((lg.tp, 2), lg.tp, "tp", lg)
                        else:
                            continue
                    else:
                        t = min(lg.sl, lg.trail) if lg.activated else lg.sl
                        if t <= q:
                            cand = ((t, 0), t, "stop", lg)
                        else:
                            continue
                    if best is None or cand[0] < best[0]:
                        best = cand
            else:
                if self.sells and self.sells[0][0] >= q:
                    t = self.sells[0][0]
                    best = ((-t, 1), t, "entry", self.sells[0])
                for lg in self.legs:
                    if lg.par.side < 0:
                        if lg.kind == "tp" and lg.tp >= q:
                            cand = ((-lg.tp, 2), lg.tp, "tp", lg)
                        else:
                            continue
                    else:
                        t = max(lg.sl, lg.trail) if lg.activated else lg.sl
                        if t >= q:
                            cand = ((-t, 0), t, "stop", lg)
                        else:
                            continue
                    if best is None or cand[0] < best[0]:
                        best = cand
            if best is None:
                break
            _, trig, kind, obj = best
            cf = q if gap else cur
            if kind == "entry":
                (self.buys if up else self.sells).pop(0)
                side = 1 if up else -1
                if self.n_open_par >= cfg.max_positions:
                    self.stats["skipped_cap"] += 1
                    self._ev(i, "cancelled", side, obj[1], trig, 1.0, "position_cap")
                else:
                    self._enter(side, trig, obj[1], cf, i, gap)
                    if cfg.cancel_opposite_on_fill:
                        if self.ev is not None and (self.sells if up else self.buys):
                            self._ev(i, "cancelled", -side, 0, float("nan"), float(len(self.sells if up else self.buys)), "opposite_on_fill")
                        if up:
                            self.sells = []
                        else:
                            self.buys = []
            elif kind == "tp":
                self._exit(obj, trig, True, i, "tp")
            else:
                base = max(trig, cf) if up else min(trig, cf)
                self._exit(obj, base, False, i, "trail" if obj.activated else "sl")
            if not gap:
                cur = trig if (trig > cur) == up else cur
        # end of segment: trailing legs see the new extreme
        gp = cfg.gap
        if self.rec:
            for lg in self.legs:
                self._track(lg.par, q)
        for lg in self.legs:
            if lg.kind != "trail":
                continue
            side = lg.par.side
            if side > 0:
                if q >= lg.act and (not lg.activated or q > lg.ext):
                    lg.activated, lg.ext = True, q
                    lg.trail = q * (1.0 - gp)
            else:
                if q <= lg.act and (not lg.activated or q < lg.ext):
                    lg.activated, lg.ext = True, q
                    lg.trail = q * (1.0 + gp)

    def run(self) -> pd.DataFrame:
        n = len(self.c)
        h, l, o, c = self.h, self.l, self.o, self.c
        for i in range(1, n):
            d = self.days[i]
            if d != self.day:
                self.day, self.day_pnl, self.halted = d, 0.0, False
            self.i = i
            if self.ev is not None and self.gate is not None and self.gate[i] != self.gate_state:
                self.gate_state = self.gate[i]
                self._ev(i, "gate_on" if self.gate_state else "gate_off", 0, 0, self.c[i - 1], 0.0, "gate")
            self._decide(i)
            if not (self.legs or self.buys or self.sells):
                continue
            if not self.legs:
                if not ((self.buys and self.buys[0][0] <= h[i]) or (self.sells and self.sells[0][0] >= l[i])):
                    continue
            self.rng_i = h[i] - l[i]
            pc = c[i - 1]
            if o[i] != pc:
                self._seg(pc, o[i], True, i)
            od = self.order
            if od == "rand":
                od = "hl" if self.rand[i] < 0.5 else "lh"
            pts = (o[i], h[i], l[i], c[i]) if od == "hl" else (o[i], l[i], h[i], c[i])
            for a, b in zip(pts[:-1], pts[1:]):
                self._seg(a, b, False, i)
        last = n - 1
        self.rng_i = 0.0
        for lg in list(self.legs):
            self._exit(lg, c[last], False, last, "eod")
        cols = ["entry_bar", "exit_bar", "side", "level", "gross_bp", "gross_noslip_bp", "net_bp", "exit"]
        df = pd.DataFrame(self.rows, columns=cols)
        df.insert(0, "entry_ts", self.ts[df["entry_bar"].to_numpy(dtype=int)] if len(df) else np.array([], dtype="int64"))
        if self.rec:
            xc = ["trigger_px", "entry_px", "exit_px", "fee_bp", "slip_bp", "mfe_bp", "mae_bp", "gap_fill", "rebuild_id", "concurrent"]
            ex = pd.DataFrame(self.extra, columns=xc)
            df = pd.concat([df, ex], axis=1)
            df["exit_ts"] = self.ts[df["exit_bar"].to_numpy(dtype=int)] if len(df) else np.array([], dtype="int64")
        return df


EVENT_COLUMNS = ["bar", "event", "side", "level", "price", "qty", "reason"]


def simulate(ts, o, h, l, c, cfg: GridConfig, order: str = "hl", gate: Optional[np.ndarray] = None,
             return_stats: bool = False, record: bool = False, events: Optional[list] = None, event_max: int = 2_000_000):
    """One run with ONE intrabar ordering ('hl' = O-H-L-C, 'lh' = O-L-H-C). ``gate``: bool per bar, default computed from cfg.
    ``record=True`` appends trade columns (trigger_px, entry_px, exit_px, fee_bp, slip_bp, mfe_bp, mae_bp, gap_fill, rebuild_id,
    concurrent, exit_ts) AFTER the usual ones; ``events`` (a list) collects EVENT_COLUMNS tuples up to ``event_max``. Pure
    observation: the trades and every summary metric are identical with or without them."""
    if gate is None:
        gate = gate_array(ts, o, h, l, c, cfg)
    sim = _Sim(ts, o, h, l, c, cfg, order, gate, record, events, event_max)
    df = sim.run()
    return (df, sim.stats) if return_stats else df


def events_frame(events: list, ts, coin: str = "", ordering: str = "") -> pd.DataFrame:
    """EVENT_COLUMNS tuples -> DataFrame with ts (ms), coin and ordering columns (bar index kept)."""
    df = pd.DataFrame(events, columns=EVENT_COLUMNS)
    ts = np.asarray(ts, dtype="int64")
    df.insert(0, "ts", ts[df["bar"].to_numpy(dtype=int)] if len(df) else np.array([], dtype="int64"))
    df.insert(1, "coin", coin)
    df.insert(2, "ordering", ordering)
    return df


def annotate_trades(trades: pd.DataFrame, arrs: dict, cfg: GridConfig, coin: str = "", ordering: str = "") -> pd.DataFrame:
    """Trade-log view of a ``simulate(..., record=True)`` result: English column names, entry/exit minute range_bp and
    quote_volume (needs ``arrs['qv']``; NaN when absent), gate values at the entry (adx, vol ratio, cost ratio; computed with the
    same no-look-ahead signals as the gates, whatever gates the config uses), hold_min, UTC day. Reads only; changes nothing."""
    ts = np.asarray(arrs["ts"], dtype="int64")
    o, h, l, c = (np.asarray(arrs[k], dtype="float64") for k in ("o", "h", "l", "c"))
    qv = np.asarray(arrs["qv"], dtype="float64") if "qv" in arrs else np.full(len(ts), np.nan)
    eb = trades["entry_bar"].to_numpy(dtype=int)
    xb = trades["exit_bar"].to_numpy(dtype=int)
    rng_bp = (h - l) / c * 1e4
    with np.errstate(invalid="ignore", divide="ignore"):
        sv = signal_values(ts, h, l, c, cfg)
        adx, vr, cr = sv["adx"], sv["vol"], sv["cost"] / cfg.round_trip_cost_bp
    out = pd.DataFrame({
        "coin": coin, "ordering": ordering, "side": trades["side"].to_numpy(), "level": trades["level"].to_numpy(),
        "entry_ts": trades["entry_ts"].to_numpy(), "entry_px": trades["entry_px"].to_numpy(), "trigger_px": trades["trigger_px"].to_numpy(),
        "exit_ts": trades["exit_ts"].to_numpy(), "exit_px": trades["exit_px"].to_numpy(), "exit_reason": [("+".join(dict.fromkeys(x.split("+")))) for x in trades["exit"]],
        "hold_min": (ts[xb] - ts[eb]) / MS_MIN if len(trades) else np.array([]),
        "gross_bp": trades["gross_bp"].to_numpy(), "slip_bp": trades["slip_bp"].to_numpy(), "fee_bp": trades["fee_bp"].to_numpy(),
        "net_bp": trades["net_bp"].to_numpy(), "mfe_bp": trades["mfe_bp"].to_numpy(), "mae_bp": trades["mae_bp"].to_numpy(),
        "gap_fill": trades["gap_fill"].to_numpy(dtype=bool),
        "entry_range_bp": rng_bp[eb], "entry_quote_volume": qv[eb], "exit_range_bp": rng_bp[xb], "exit_quote_volume": qv[xb],
        "gate_adx": adx[eb], "gate_vol_ratio": vr[eb], "gate_cost_ratio": cr[eb],
        "rebuild_id": trades["rebuild_id"].to_numpy(), "concurrent": trades["concurrent"].to_numpy(),
        "day": pd.to_datetime(ts[eb] // MS_DAY * MS_DAY, unit="ms", utc=True).strftime("%Y-%m-%d") if len(trades) else [],
    })
    return out


# --------------------------------------------------------------------------- metrics
def day_clustered_t(x, t_ms) -> float:
    """t of the mean with standard errors clustered by UTC day (as ``ctstat`` in research/edge_discovery/33_regime_filters.py)."""
    x = np.asarray(x, dtype="float64")
    if len(x) < 2:
        return float("nan")
    d = (np.asarray(t_ms, dtype="int64") // MS_DAY)
    g = pd.DataFrame({"r": x, "d": d}).groupby("d")["r"].agg(["sum", "count"])
    mu = x.mean()
    se = math.sqrt(float(((g["sum"] - g["count"] * mu) ** 2).sum())) / len(x)
    return float(mu / se) if se > 0 else float("nan")


def max_drawdown_bp(net_bp, exit_order) -> float:
    """Largest peak-to-trough fall of the cumulative net bp (trades ordered by exit time), as a positive number."""
    x = np.asarray(net_bp, dtype="float64")[np.argsort(np.asarray(exit_order), kind="stable")]
    if len(x) == 0:
        return 0.0
    cum = np.cumsum(x)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return float(np.max(peak - cum).clip(min=0.0))


def summarize(trades: pd.DataFrame, n_bars: int, n_days: float) -> dict:
    """Metrics of one run. net_bp / gross_bp are means per trade (parent position)."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": float("nan"), "gross_bp": float("nan"), "gross_noslip_bp": float("nan"),
                "net_bp": float("nan"), "t_net": float("nan"), "max_dd_bp": 0.0, "time_in_market": 0.0,
                "fills_per_day": 0.0}
    cover = np.zeros(n_bars + 2, dtype=np.int64)
    np.add.at(cover, trades["entry_bar"].to_numpy(dtype=int), 1)
    np.add.at(cover, trades["exit_bar"].to_numpy(dtype=int) + 1, -1)
    return {"trades": int(n), "win_rate": float((trades["net_bp"] > 0).mean()),
            "gross_bp": float(trades["gross_bp"].mean()), "gross_noslip_bp": float(trades["gross_noslip_bp"].mean()),
            "net_bp": float(trades["net_bp"].mean()),
            "t_net": day_clustered_t(trades["net_bp"].to_numpy(), trades["entry_ts"].to_numpy()),
            "max_dd_bp": max_drawdown_bp(trades["net_bp"].to_numpy(), trades["exit_bar"].to_numpy()),
            "time_in_market": float((np.cumsum(cover[:n_bars]) > 0).mean()),
            "fills_per_day": float(n / max(n_days, 1e-9))}


def run_both(ts, o, h, l, c, cfg: GridConfig, gate: Optional[np.ndarray] = None) -> dict:
    """Both intrabar orderings. LB = the ordering with the lower mean net bp per trade, UB = the higher.
    Returns {'LB': summary, 'UB': summary, 'LB_order', 'UB_order', 'trades': {'hl': df, 'lh': df}, 'n_bars', 'n_days'}."""
    ts = np.asarray(ts, dtype="int64")
    if gate is None:
        gate = gate_array(ts, o, h, l, c, cfg)
    n_days = (ts[-1] - ts[0]) / MS_DAY if len(ts) > 1 else 0.0
    out, summ = {}, {}
    for od in ORDERINGS:
        out[od] = simulate(ts, o, h, l, c, cfg, od, gate)
        summ[od] = summarize(out[od], len(ts), n_days)
    key = lambda od: (summ[od]["net_bp"] if summ[od]["trades"] else 0.0)  # noqa: E731
    lb, ub = sorted(ORDERINGS, key=key)
    return {"LB": summ[lb], "UB": summ[ub], "LB_order": lb, "UB_order": ub, "trades": out, "n_bars": len(ts), "n_days": n_days}


def sign_consistency(net_by_coin: Dict[str, float]) -> Tuple[int, int]:
    """(coins with mean net > 0, coins with a defined mean)."""
    vals = [v for v in net_by_coin.values() if v == v]
    return int(sum(v > 0 for v in vals)), len(vals)


# --------------------------------------------------------------------------- null baselines
def _valid_px(a):
    return np.isfinite(a) & (a > 0)


def _first_valid(c) -> float:
    ok = _valid_px(c)
    if not ok.any():
        raise ValueError("null baselines need at least one finite positive close")
    return float(c[np.argmax(ok)])


def _rel(o, h, l, c):
    """Relative log OHLC vs the previous close. Real 1m files can hold NaN / zero / non-positive prices (dead coins, bad bars):
    one such value used to make log() non-finite and the cumulative sum of the rebuilt walk NaN from there to the end (every
    draw then traded nothing). Invalid values are treated as a FLAT bar: the previous valid close is carried (reference only),
    an invalid price becomes 'no move' (o, c -> 0; h, l -> the bar's own o/c extreme). Clean data is untouched (byte-identical)."""
    okc = _valid_px(c)
    cc = pd.Series(np.where(okc, c, np.nan)).ffill().bfill().to_numpy()
    pc = np.r_[cc[0], cc[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ro, rh, rl, rc = [np.where(_valid_px(a), np.log(np.where(_valid_px(a), a, 1.0) / pc), np.nan) for a in (o, h, l, c)]
    rc = np.where(np.isnan(rc), 0.0, rc)
    ro = np.where(np.isnan(ro), 0.0, ro)
    hi, lo = np.maximum(ro, rc), np.minimum(ro, rc)
    rh = np.where(np.isnan(rh), hi, rh)
    rl = np.where(np.isnan(rl), lo, rl)
    bad = ~(_valid_px(o) & _valid_px(h) & _valid_px(l) & okc)
    rh = np.where(bad, np.maximum(rh, hi), rh)
    rl = np.where(bad, np.minimum(rl, lo), rl)
    return [ro, rh, rl, rc]


def _rebuild_ohlc(ro, rh, rl, rc, sgn, c0):
    """Rebuild a price path from relative OHLC (log vs the previous close); sgn=-1 mirrors the bar (high and low swap)."""
    flip = sgn < 0
    ro_, rc_ = ro * sgn, rc * sgn
    rh_ = np.where(flip, -rl, rh)
    rl_ = np.where(flip, -rh, rl)
    cl = np.log(c0) + np.cumsum(rc_)
    pcl = np.r_[np.log(c0), cl[:-1]]
    out = tuple(np.exp(pcl + r) for r in (ro_, rh_, rl_)) + (np.exp(cl),)
    if not all(np.isfinite(x).all() and (x > 0).all() for x in out):
        raise ValueError("null baseline produced a non-finite or non-positive price (overflow of the cumulative walk?)")
    return out


def sign_flip_null(ts, o, h, l, c, seed: int, flip_prob: float = 0.5):
    """Flip the sign of the log-returns of randomly chosen WHOLE UTC days (the bar's O/H/L mirror with them): the volatility
    structure and the intraday shape are kept, the direction across days is randomised. Returns (o, h, l, c)."""
    o, h, l, c = (np.asarray(a, dtype="float64") for a in (o, h, l, c))
    rng = np.random.default_rng(seed)
    day = np.asarray(ts, dtype="int64") // MS_DAY
    ud, inv = np.unique(day, return_inverse=True)
    sgn = np.where(rng.random(len(ud)) < flip_prob, -1.0, 1.0)[inv]
    ro, rh, rl, rc = _rel(o, h, l, c)
    sgn = sgn.copy()
    return _rebuild_ohlc(ro, rh, rl, rc, sgn, _first_valid(c))


def random_walk_null(ts, o, h, l, c, seed: int):
    """Pure random walk with the same marginal bar shapes: the bars (relative O/H/L/C) are shuffled i.i.d. and each gets a
    random sign. No volatility clustering, no intraday autocorrelation: a martingale. Returns (o, h, l, c)."""
    o, h, l, c = (np.asarray(a, dtype="float64") for a in (o, h, l, c))
    rng = np.random.default_rng(seed)
    ro, rh, rl, rc = _rel(o, h, l, c)
    perm = rng.permutation(len(c))
    sgn = np.where(rng.random(len(c)) < 0.5, -1.0, 1.0)
    return _rebuild_ohlc(ro[perm], rh[perm], rl[perm], rc[perm], sgn, _first_valid(c))


# --------------------------------------------------------------------------- synthetic 1m data (tests / --smoke only)
def synth_ohlc(n: int, seed: int, daily_vol: float = 0.04, vol_persist: float = 0.98, vol_of_vol: float = 0.15,
               p0: float = 100.0, start_ms: int = 1_700_000_000_000 // MS_DAY * MS_DAY):
    """Martingale 1m bars with volatility clustering (log-vol AR(1) per minute). Each bar is a continuous Brownian bridge
    summarised by its open (= previous close), close and the extremes drawn from the bridge-extreme laws, so a trigger
    crossing has NO discretisation overshoot (a coarse sub-step path would hand every stop fill a free ~1-2 bp: measured).
    Max and min are drawn independently given (open, close): approximate, but always consistent (h >= max(o,c), l <= min(o,c)).
    Returns (ts_ms, o, h, l, c)."""
    rng = np.random.default_rng(seed)
    sig0 = daily_vol / math.sqrt(1440.0)
    eps = rng.normal(0.0, vol_of_vol * math.sqrt(1.0 - vol_persist ** 2), n)
    lv = np.empty(n)
    acc = 0.0
    for i in range(n):                                   # AR(1), n is small (<= ~600k): plain loop is fine
        acc = vol_persist * acc + eps[i]
        lv[i] = acc
    sig = sig0 * np.exp(lv - 0.5 * np.var(lv))
    x = rng.normal(0.0, 1.0, n) * sig                    # log(close/open)
    u1 = rng.random(n)
    u2 = rng.random(n)
    mx = 0.5 * (x + np.sqrt(x * x - 2.0 * sig * sig * np.log(1.0 - u1)))
    mn = 0.5 * (x - np.sqrt(x * x - 2.0 * sig * sig * np.log(1.0 - u2)))
    cl = np.log(p0) + np.cumsum(x)
    pcl = np.r_[np.log(p0), cl[:-1]]
    o, h, l, c = np.exp(pcl), np.exp(pcl + mx), np.exp(pcl + mn), np.exp(cl)
    ts = start_ms + np.arange(n, dtype="int64") * MS_MIN
    return ts, o, h, l, c


COLUMNS = ["timestamp", "datetime_utc", "open", "high", "low", "close", "volume", "quote_volume", "trades",
           "taker_buy_volume", "taker_buy_quote_volume"]


def synth_frame(n: int, seed: int, daily_vol: float = 0.04, qv_per_min: float = 50_000.0, start_ms: Optional[int] = None) -> pd.DataFrame:
    """A DataFrame with the owner's 1m file columns, built from ``synth_ohlc``."""
    kw = {} if start_ms is None else {"start_ms": start_ms}
    ts, o, h, l, c = synth_ohlc(n, seed, daily_vol=daily_vol, **kw)
    rng = np.random.default_rng(seed + 7)
    qv = qv_per_min * rng.lognormal(0.0, 0.5, n)
    vol = qv / c
    return pd.DataFrame({"timestamp": ts, "datetime_utc": pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S"),
                         "open": o, "high": h, "low": l, "close": c, "volume": vol, "quote_volume": qv,
                         "trades": (qv / 200.0).astype("int64"), "taker_buy_volume": vol * 0.5,
                         "taker_buy_quote_volume": qv * 0.5})


# --------------------------------------------------------------------------- loading the owner's 1m files
_EXTS = (".csv.gz", ".csv", ".parquet")


def list_symbols(data_dir: str) -> Dict[str, List[str]]:
    """{SYMBOL: [files]} from ``<SYMBOL>.csv.gz|.csv|.parquet`` in data_dir, or per-symbol sub-folders holding csv/parquet files."""
    out: Dict[str, List[str]] = {}
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p):
            files = sorted(f for e in _EXTS for f in glob.glob(os.path.join(p, "*" + e)))
            if files:
                out[name] = files
        else:
            for e in _EXTS:
                if name.endswith(e):
                    out.setdefault(name[: -len(e)], []).append(p)
                    break
    return out


def _to_ms(ts: pd.Series) -> pd.Series:
    ts = pd.to_numeric(ts, errors="coerce")
    ts = ts.where(ts < 1e14, ts // 1000)                 # microseconds -> ms
    return ts.astype("int64")


def read_symbol(files: List[str], columns: Optional[List[str]] = None, start_ms: Optional[int] = None,
                end_ms: Optional[int] = None) -> pd.DataFrame:
    """Read one symbol (one or several files), sorted by timestamp with duplicates dropped, window [start_ms, end_ms)."""
    need = None if columns is None else sorted(set(columns) | {"timestamp"})
    parts = []
    for f in files:
        if f.endswith(".parquet"):
            d = pd.read_parquet(f, columns=need)
        else:
            d = pd.read_csv(f, usecols=need)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    d["timestamp"] = _to_ms(d["timestamp"])
    d = d.sort_values("timestamp", kind="stable").drop_duplicates("timestamp", keep="last")
    if start_ms is not None:
        d = d[d["timestamp"] >= start_ms]
    if end_ms is not None:
        d = d[d["timestamp"] < end_ms]
    return d.reset_index(drop=True)


def date_range(files: List[str]) -> Tuple[int, int, int]:
    """(first_ms, last_ms, n_bars) reading the timestamp column only."""
    d = read_symbol(files, columns=["timestamp"])
    if d.empty:
        return 0, 0, 0
    return int(d["timestamp"].iloc[0]), int(d["timestamp"].iloc[-1]), int(len(d))


@dataclass(frozen=True)
class Splits:
    """Time splits (ms, half-open) of the common date range: SELECT first 40 percent, DEV next 30, TEST last 30."""
    start: int
    end: int
    select: Tuple[int, int]
    dev: Tuple[int, int]
    test: Tuple[int, int]

    def as_dict(self) -> dict:
        iso = lambda ms: pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")  # noqa: E731
        return {k: {"start_ms": v[0], "end_ms": v[1], "start": iso(v[0]), "end": iso(v[1])}
                for k, v in (("select", self.select), ("dev", self.dev), ("test", self.test))}


def make_splits(start_ms: int, end_ms: int, fracs=(0.4, 0.3, 0.3)) -> Splits:
    if not abs(sum(fracs) - 1.0) < 1e-9 or end_ms <= start_ms:
        raise ValueError("fracs must sum to 1 and end_ms > start_ms")
    span = end_ms - start_ms + MS_MIN
    b1 = start_ms + int(round(span * fracs[0]))
    b2 = start_ms + int(round(span * (fracs[0] + fracs[1])))
    return Splits(start_ms, end_ms + MS_MIN, (start_ms, b1), (b1, b2), (b2, end_ms + MS_MIN))


def common_range(ranges: Dict[str, Tuple[int, int, int]], quantile: float = 0.8) -> Tuple[int, int]:
    """Common date range of the universe: start = the ``quantile`` of the first timestamps, end = the (1-quantile) of the last ones,
    so a handful of very late listings or very early delistings does not shrink the range for everyone (they fail the
    coverage rule instead)."""
    live = {k: v for k, v in ranges.items() if v[2] > 0}
    if not live:
        raise ValueError("no symbol has data")
    starts = np.array([v[0] for v in live.values()])
    ends = np.array([v[1] for v in live.values()])
    s = int(np.quantile(starts, quantile, method="higher"))
    e = int(np.quantile(ends, 1.0 - quantile, method="lower"))
    if e <= s:
        raise ValueError("common date range is empty")
    return s, e


def select_coins(data_dir: str, top: int = 5, taker_bp: float = 5.0, slip_bps: float = 1.0, min_quote_volume: float = 20_000.0,
                 min_coverage: float = 0.90, select_frac: float = 0.4, range_quantile: float = 0.8, edge_tolerance: float = 0.02,
                 verbose: bool = True,
                 symbols: Optional[Dict[str, List[str]]] = None):
    """Coin selection on the SELECT window ONLY (first ``select_frac`` of the common date range).

    score = median 1m (high-low)/close in bp / round-trip cost in bp (2*taker + 2*slip), subject to the median quote volume
    per minute >= ``min_quote_volume`` and the bar coverage of the window >= ``min_coverage``. A coin must also span the whole
    common range (first bar within ``edge_tolerance`` of its start, last bar within it of its end) because DEV and TEST need
    it: this uses FUTURE EXISTENCE (never prices) and tilts the pool to survivors - a stated caveat, not a leak of returns. Returns (selected, table, splits):
    ``selected`` = top symbols, ``table`` = every symbol with its score and the reason it was excluded, ``splits`` = Splits."""
    syms = symbols if symbols is not None else list_symbols(data_dir)
    if not syms:
        raise ValueError(f"no symbol files in {data_dir}")
    ranges = {}
    for i, (s, files) in enumerate(syms.items()):
        ranges[s] = date_range(files)
        if verbose and (i + 1) % 25 == 0:
            print(f"  scanned {i + 1}/{len(syms)} symbols", flush=True)
    s0, e0 = common_range(ranges, range_quantile)
    fr = (select_frac, (1 - select_frac) / 2, (1 - select_frac) / 2)
    splits = make_splits(s0, e0, fr)
    a, b = splits.select
    expected = (b - a) / MS_MIN
    rt = 2.0 * taker_bp + 2.0 * slip_bps
    rows = []
    for s, files in syms.items():
        if ranges[s][2] == 0:
            rows.append({"symbol": s, "bars": 0, "coverage": 0.0, "median_range_bp": np.nan, "median_qv": np.nan, "score": np.nan,
                         "eligible": False, "why": "no data"})
            continue
        d = read_symbol(files, ["high", "low", "close", "quote_volume"], a, b)
        cov = len(d) / expected
        mr = float(((d["high"] - d["low"]) / d["close"] * 1e4).median()) if len(d) else np.nan
        mq = float(d["quote_volume"].median()) if len(d) else np.nan
        why = "" if len(d) else "no bars in SELECT"
        tol = int(edge_tolerance * (e0 - s0))
        if ranges[s][0] > s0 + tol or ranges[s][1] < e0 - tol:
            why = "does not span the common date range (DEV/TEST need it)"
        elif cov < min_coverage:
            why = f"coverage {cov:.2f} < {min_coverage}"
        elif mq < min_quote_volume:
            why = f"median quote volume {mq:.0f} < {min_quote_volume:.0f}"
        rows.append({"symbol": s, "bars": int(len(d)), "coverage": cov, "median_range_bp": mr, "median_qv": mq,
                     "score": mr / rt if mr == mr else np.nan, "eligible": why == "", "why": why})
    table = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
    sel = table[table["eligible"]].head(top)["symbol"].tolist()
    return sel, table, splits

"""
PURPOSE:  Optional numba (JIT) implementation of the event loop of tools/stop_grid_sim.py (class _Sim): the SAME simulation on
          plain numpy arrays, selected with ``engine="numba"`` in stop_grid_sim.simulate / run_both / the runner. Produces the
          same trade table (columns, exit reasons, trade order; the optional ``record`` columns too). Only for speed: the python
          engine stays the default and the reference, and every rule (fills, gaps, slippage, trailing, caps, rebuild triggers,
          daily loss cap, gates, end-of-data close) is a line-by-line port of it; same floating-point operations in the same order.
TAGS:     numba, JIT, stop grid, engine, simulation speed, parity, E-grid1m-001
PITFALLS: Importing this module imports numba (ImportError when missing: stop_grid_sim turns that into a clear RuntimeError, no
          silent fallback). No ``events`` log (python engine only). ``fastmath=False`` always (bit-for-bit float semantics);
          ``cache=True`` writes compiled code next to this file (falls back to no cache when that is not writable). Keep it in sync
          with stop_grid_sim._Sim: any change of the rules there needs the same change here, and ``python tools/stop_grid_bench.py
          --parity`` (tests/test_stop_grid_numba.py) must pass. State lives in small arrays because numba helpers cannot share
          Python objects: legs and parents are fixed-capacity slot arrays (legs keep insertion order, which decides ties).
"""
import numpy as np
from numba import njit

MS_DAY = 86_400_000


def _jit(f):
    try:
        return njit(cache=True, fastmath=False)(f)
    except Exception:                                   # no writable cache location: still JIT, just not cached
        return njit(cache=False, fastmath=False)(f)


# ---- config vectors (CF float64, CI int64)
C_SPACING, C_SL, C_TP, C_ACT, C_GAP, C_BREAKX, C_CAP, C_TAKER, C_MAKER, C_SLIPBPS, C_SLIPFRAC = range(11)
K_NLEV, K_EXIT, K_REBUILD, K_CANCELOPP, K_FLATCD, K_TIMEEVERY, K_MAXPOS, K_HASCAP, K_REC, K_ORDER = range(10)
EXIT_CODE = {"tp": 0, "trail": 1, "split": 2}
REBUILD_CODE = {"once": 0, "breakout": 1, "flat": 2, "time": 3}
ORDER_CODE = {"hl": 0, "lh": 1, "rand": 2}
# ---- integer state I
(I_NOPEN, I_LASTCLOSE, I_HALTED, I_HDAYS, I_SKIPCAP, I_BUILDS, I_RID, I_NLEGS, I_NROWS,
 I_BH, I_BN, I_SH, I_SN, I_HASGRID, I_GFILLS, I_GBAR) = range(16)
S_DAYPNL, S_RNG = 0, 1
G_P0, G_TOP, G_BOT = 0, 1, 2
# ---- legs: LF float columns, LI int columns
L_W, L_SL, L_TP, L_ACT, L_EXT, L_TRAIL = range(6)
L_PAR, L_KIND, L_ACTIVATED, L_SIDE = range(4)          # kind: 0 tp leg, 1 trailing leg; side = the parent's side (copied: one load less)
# ---- parents: PF float columns, PIN int columns
PF_ENTRYNS, PF_ENTRY, PF_GROSS, PF_GROSSNS, PF_NET, PF_XVAL, PF_FEE, PF_MFE, PF_MAE = range(9)
P_SIDE, P_BAR, P_K, P_OPEN, P_NK, P_K1, P_K2, P_GAPFILL, P_RID, P_CONC, P_ACTIVE = range(11)
# ---- output rows: OF float columns, OI int columns
O_GROSS, O_GROSSNS, O_NET, O_ENTRYNS, O_ENTRY, O_XVAL, O_FEE, O_MFE, O_MAE = range(9)
OI_ENTRYBAR, OI_EXITBAR, OI_SIDE, OI_LEVEL, OI_K1, OI_K2, OI_RID, OI_CONC, OI_GAPFILL = range(9)
KIND_NAMES = ("tp", "trail", "sl", "eod")              # exit kind codes
KC_TP, KC_TRAIL, KC_SL, KC_EOD = 0, 1, 2, 3


@_jit
def _slip(base, rng_i, CF):
    return base * CF[C_SLIPBPS] / 1e4 + CF[C_SLIPFRAC] * rng_i


@_jit
def _cancel(I):
    I[I_BH] = 0
    I[I_BN] = 0
    I[I_SH] = 0
    I[I_SN] = 0
    I[I_HASGRID] = 0


@_jit
def _build(i, c, CF, CI, I, G, bt, st):
    n = CI[K_NLEV]
    s = CF[C_SPACING]
    p0 = c[i - 1]
    for k in range(1, n + 1):
        bt[k - 1] = p0 * (1.0 + k * s)
        st[k - 1] = p0 * (1.0 - k * s)
    I[I_BH] = 0
    I[I_BN] = n
    I[I_SH] = 0
    I[I_SN] = n
    G[G_P0] = p0
    G[G_TOP] = p0 * (1.0 + (n + CF[C_BREAKX]) * s)
    G[G_BOT] = p0 * (1.0 - (n + CF[C_BREAKX]) * s)
    I[I_HASGRID] = 1
    I[I_GFILLS] = 0
    I[I_GBAR] = i
    I[I_BUILDS] += 1
    I[I_RID] += 1


@_jit
def _decide(i, c, gate, CF, CI, I, G, bt, st):
    if I[I_HALTED] != 0:
        if I[I_BH] < I[I_BN] or I[I_SH] < I[I_SN] or I[I_HASGRID] != 0:
            _cancel(I)
        return
    need = I[I_HASGRID] == 0
    if not need:
        m = CI[K_REBUILD]
        if m == 1:
            pc = c[i - 1]
            need = pc > G[G_TOP] or pc < G[G_BOT]
        elif m == 2:
            need = I[I_GFILLS] > 0 and I[I_NOPEN] == 0 and (i - I[I_LASTCLOSE]) >= CI[K_FLATCD]
        elif m == 3:
            need = (i - I[I_GBAR]) >= CI[K_TIMEEVERY]
    if not need:
        return
    _cancel(I)
    if gate[i]:
        _build(i, c, CF, CI, I, G, bt, st)


@_jit
def _track(ps, px, PF, PIN):
    m = PIN[ps, P_SIDE] * (px / PF[ps, PF_ENTRY] - 1.0) * 1e4
    if m > PF[ps, PF_MFE]:
        PF[ps, PF_MFE] = m
    if m < PF[ps, PF_MAE]:
        PF[ps, PF_MAE] = m


@_jit
def _enter(side, trig, k, cf, i, gap, CF, CI, S, I, LF, LI, PF, PIN):
    if side > 0:
        base = cf if cf > trig else trig               # python max(trig, cf)
    else:
        base = cf if cf < trig else trig               # python min(trig, cf)
    fill = base + side * _slip(base, S[S_RNG], CF)
    ps = 0
    while PIN[ps, P_ACTIVE] != 0:
        ps += 1
    PIN[ps, P_ACTIVE] = 1
    PIN[ps, P_SIDE] = side
    PIN[ps, P_BAR] = i
    PIN[ps, P_K] = k
    PIN[ps, P_NK] = 0
    PIN[ps, P_K1] = -1
    PIN[ps, P_K2] = -1
    PIN[ps, P_GAPFILL] = 1 if gap else 0
    PIN[ps, P_RID] = I[I_RID]
    PF[ps, PF_ENTRYNS] = base
    PF[ps, PF_ENTRY] = fill
    PF[ps, PF_GROSS] = 0.0
    PF[ps, PF_GROSSNS] = 0.0
    PF[ps, PF_NET] = 0.0
    PF[ps, PF_XVAL] = 0.0
    PF[ps, PF_FEE] = 0.0
    PF[ps, PF_MFE] = 0.0
    PF[ps, PF_MAE] = 0.0
    mode = CI[K_EXIT]
    nspec = 2 if mode == 2 else 1
    PIN[ps, P_OPEN] = nspec
    for j in range(nspec):
        if mode == 0:
            kind, w = 0, 1.0
        elif mode == 1:
            kind, w = 1, 1.0
        else:
            kind, w = j, 0.5                           # split: leg 0 = tp, leg 1 = trailing
        x = I[I_NLEGS]
        LI[x, L_PAR] = ps
        LI[x, L_KIND] = kind
        LI[x, L_ACTIVATED] = 0
        LI[x, L_SIDE] = side
        LF[x, L_W] = w
        LF[x, L_SL] = base * (1.0 - side * CF[C_SL])
        LF[x, L_TP] = base * (1.0 + side * CF[C_TP])
        LF[x, L_ACT] = base * (1.0 + side * CF[C_ACT])
        LF[x, L_EXT] = base
        LF[x, L_TRAIL] = 0.0
        I[I_NLEGS] = x + 1
    I[I_NOPEN] += 1
    PIN[ps, P_CONC] = I[I_NOPEN]
    if I[I_HASGRID] != 0:
        I[I_GFILLS] += 1


@_jit
def _exit(x, base, limit, i, kindcode, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI):
    ps = LI[x, L_PAR]
    side = PIN[ps, P_SIDE]
    w = LF[x, L_W]
    if limit:
        fill = base
        fee = CF[C_MAKER]
    else:
        fill = base - side * _slip(base, S[S_RNG], CF)
        fee = CF[C_TAKER]
    gross = side * (fill / PF[ps, PF_ENTRY] - 1.0) * 1e4 * w
    gross_ns = side * (base / PF[ps, PF_ENTRYNS] - 1.0) * 1e4 * w
    PF[ps, PF_GROSS] += gross
    PF[ps, PF_GROSSNS] += gross_ns
    PF[ps, PF_NET] += gross - (CF[C_TAKER] + fee) * w
    if PIN[ps, P_NK] == 0:
        PIN[ps, P_K1] = kindcode
    else:
        PIN[ps, P_K2] = kindcode
    PIN[ps, P_NK] += 1
    rec = CI[K_REC] != 0
    if rec:
        PF[ps, PF_XVAL] += fill * w
        PF[ps, PF_FEE] += (CF[C_TAKER] + fee) * w
        _track(ps, fill, PF, PIN)
    nl = I[I_NLEGS]
    for j in range(x, nl - 1):                          # remove the leg, keep the order of the others
        for q in range(6):
            LF[j, q] = LF[j + 1, q]
        for q in range(4):
            LI[j, q] = LI[j + 1, q]
    I[I_NLEGS] = nl - 1
    PIN[ps, P_OPEN] -= 1
    if PIN[ps, P_OPEN] == 0:
        I[I_NOPEN] -= 1
        I[I_LASTCLOSE] = i
        r = I[I_NROWS]
        OI[r, OI_ENTRYBAR] = PIN[ps, P_BAR]
        OI[r, OI_EXITBAR] = i
        OI[r, OI_SIDE] = side
        OI[r, OI_LEVEL] = PIN[ps, P_K]
        OI[r, OI_K1] = PIN[ps, P_K1]
        OI[r, OI_K2] = PIN[ps, P_K2]
        OI[r, OI_RID] = PIN[ps, P_RID]
        OI[r, OI_CONC] = PIN[ps, P_CONC]
        OI[r, OI_GAPFILL] = PIN[ps, P_GAPFILL]
        OF[r, O_GROSS] = PF[ps, PF_GROSS]
        OF[r, O_GROSSNS] = PF[ps, PF_GROSSNS]
        OF[r, O_NET] = PF[ps, PF_NET]
        OF[r, O_ENTRYNS] = PF[ps, PF_ENTRYNS]
        OF[r, O_ENTRY] = PF[ps, PF_ENTRY]
        OF[r, O_XVAL] = PF[ps, PF_XVAL]
        OF[r, O_FEE] = PF[ps, PF_FEE]
        OF[r, O_MFE] = PF[ps, PF_MFE]
        OF[r, O_MAE] = PF[ps, PF_MAE]
        I[I_NROWS] = r + 1
        PIN[ps, P_ACTIVE] = 0
        S[S_DAYPNL] += PF[ps, PF_NET]
        if CI[K_HASCAP] != 0 and I[I_HALTED] == 0 and S[S_DAYPNL] <= -CF[C_CAP]:
            I[I_HALTED] = 1
            I[I_HDAYS] += 1
            _cancel(I)


@_jit
def _seg(p, q, gap, i, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI):
    if q == p:
        return
    up = q > p
    cur = p
    while True:
        found = False
        bkey = 0.0
        bprio = 0
        btrig = 0.0
        bkind = 0                                       # 0 entry, 1 tp, 2 stop
        bidx = -1
        nl = I[I_NLEGS]
        if up:
            if I[I_BH] < I[I_BN] and bt[I[I_BH]] <= q:
                t = bt[I[I_BH]]
                found, bkey, bprio, btrig, bkind, bidx = True, t, 1, t, 0, -1
            for j in range(nl):
                if LI[j, L_SIDE] > 0:
                    if LI[j, L_KIND] == 0 and LF[j, L_TP] <= q:
                        ckey = LF[j, L_TP]
                        cprio = 2
                        ckind = 1
                    else:
                        continue
                else:
                    sl = LF[j, L_SL]
                    if LI[j, L_ACTIVATED] != 0:
                        tr = LF[j, L_TRAIL]
                        ckey = tr if tr < sl else sl    # python min(sl, trail)
                    else:
                        ckey = sl
                    if ckey <= q:
                        cprio = 0
                        ckind = 2
                    else:
                        continue
                if (not found) or (ckey < bkey or (ckey == bkey and cprio < bprio)):
                    found, bkey, bprio, btrig, bkind, bidx = True, ckey, cprio, ckey, ckind, j
        else:
            if I[I_SH] < I[I_SN] and st[I[I_SH]] >= q:
                t = st[I[I_SH]]
                found, bkey, bprio, btrig, bkind, bidx = True, -t, 1, t, 0, -1
            for j in range(nl):
                if LI[j, L_SIDE] < 0:
                    if LI[j, L_KIND] == 0 and LF[j, L_TP] >= q:
                        t = LF[j, L_TP]
                        ckind = 1
                        cprio = 2
                    else:
                        continue
                else:
                    sl = LF[j, L_SL]
                    if LI[j, L_ACTIVATED] != 0:
                        tr = LF[j, L_TRAIL]
                        t = tr if tr > sl else sl       # python max(sl, trail)
                    else:
                        t = sl
                    if t >= q:
                        cprio = 0
                        ckind = 2
                    else:
                        continue
                ckey = -t
                if (not found) or (ckey < bkey or (ckey == bkey and cprio < bprio)):
                    found, bkey, bprio, btrig, bkind, bidx = True, ckey, cprio, t, ckind, j
        if not found:
            break
        trig = btrig
        cf = q if gap else cur
        if bkind == 0:
            if up:
                k = I[I_BH] + 1
                I[I_BH] += 1
                side = 1
            else:
                k = I[I_SH] + 1
                I[I_SH] += 1
                side = -1
            if I[I_NOPEN] >= CI[K_MAXPOS]:
                I[I_SKIPCAP] += 1
            else:
                _enter(side, trig, k, cf, i, gap, CF, CI, S, I, LF, LI, PF, PIN)
                if CI[K_CANCELOPP] != 0:
                    if up:
                        I[I_SH] = 0
                        I[I_SN] = 0
                    else:
                        I[I_BH] = 0
                        I[I_BN] = 0
        elif bkind == 1:
            _exit(bidx, trig, True, i, KC_TP, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI)
        else:
            if up:
                base = cf if cf > trig else trig        # python max(trig, cf)
            else:
                base = cf if cf < trig else trig        # python min(trig, cf)
            kc = KC_TRAIL if LI[bidx, L_ACTIVATED] != 0 else KC_SL
            _exit(bidx, base, False, i, kc, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI)
        if not gap:
            if (trig > cur) == up:
                cur = trig
    gp = CF[C_GAP]
    nl = I[I_NLEGS]
    if CI[K_REC] != 0:
        for j in range(nl):
            _track(LI[j, L_PAR], q, PF, PIN)
    for j in range(nl):
        if LI[j, L_KIND] != 1:
            continue
        if LI[j, L_SIDE] > 0:
            if q >= LF[j, L_ACT] and (LI[j, L_ACTIVATED] == 0 or q > LF[j, L_EXT]):
                LI[j, L_ACTIVATED] = 1
                LF[j, L_EXT] = q
                LF[j, L_TRAIL] = q * (1.0 - gp)
        else:
            if q <= LF[j, L_ACT] and (LI[j, L_ACTIVATED] == 0 or q < LF[j, L_EXT]):
                LI[j, L_ACTIVATED] = 1
                LF[j, L_EXT] = q
                LF[j, L_TRAIL] = q * (1.0 + gp)


@_jit
def _grow(OF, OI):
    cap = OF.shape[0]
    NF = np.zeros((cap * 2, OF.shape[1]))
    NI = np.zeros((cap * 2, OI.shape[1]), dtype=np.int64)
    for r in range(cap):
        for q in range(OF.shape[1]):
            NF[r, q] = OF[r, q]
        for q in range(OI.shape[1]):
            NI[r, q] = OI[r, q]
    return NF, NI


@_jit
def _run(o, h, l, c, days, gate, rand, CF, CI):
    n = len(c)
    nlev = CI[K_NLEV]
    maxpos = CI[K_MAXPOS]
    S = np.zeros(2)
    I = np.zeros(16, dtype=np.int64)
    I[I_LASTCLOSE] = -1000000000
    G = np.zeros(3)
    bt = np.zeros(nlev)
    st = np.zeros(nlev)
    LF = np.zeros((2 * maxpos + 4, 6))
    LI = np.zeros((2 * maxpos + 4, 4), dtype=np.int64)
    PF = np.zeros((maxpos + 2, 9))
    PIN = np.zeros((maxpos + 2, 11), dtype=np.int64)
    OF = np.zeros((1024, 9))
    OI = np.zeros((1024, 9), dtype=np.int64)
    margin = maxpos + 2 * nlev + 4                      # parents that can close inside one bar
    day = 0
    have_day = False
    for i in range(1, n):
        d = days[i]
        if (not have_day) or d != day:
            have_day = True
            day = d
            S[S_DAYPNL] = 0.0
            I[I_HALTED] = 0
        if I[I_NROWS] + margin > OF.shape[0]:
            OF, OI = _grow(OF, OI)
        _decide(i, c, gate, CF, CI, I, G, bt, st)
        has_b = I[I_BH] < I[I_BN]
        has_s = I[I_SH] < I[I_SN]
        if I[I_NLEGS] == 0:
            if not (has_b or has_s):
                continue
            if not ((has_b and bt[I[I_BH]] <= h[i]) or (has_s and st[I[I_SH]] >= l[i])):
                continue
        S[S_RNG] = h[i] - l[i]
        pc = c[i - 1]
        if o[i] != pc:
            _seg(pc, o[i], True, i, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI)
        od = CI[K_ORDER]
        hl = od == 0 or (od == 2 and rand[i] < 0.5)
        if hl:
            p1, p2 = h[i], l[i]
        else:
            p1, p2 = l[i], h[i]
        _seg(o[i], p1, False, i, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI)
        _seg(p1, p2, False, i, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI)
        _seg(p2, c[i], False, i, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI)
    if n > 0:
        last = n - 1
        S[S_RNG] = 0.0
        if I[I_NROWS] + margin > OF.shape[0]:
            OF, OI = _grow(OF, OI)
        while I[I_NLEGS] > 0:
            _exit(0, c[last], False, last, KC_EOD, CF, CI, S, I, G, bt, st, LF, LI, PF, PIN, OF, OI)
    return OF[: I[I_NROWS]].copy(), OI[: I[I_NROWS]].copy(), I[I_SKIPCAP], I[I_HDAYS], I[I_BUILDS]


def _cfg_vectors(cfg, order: str, record: bool):
    CF = np.array([cfg.spacing, cfg.sl, cfg.tp, cfg.act, cfg.gap, cfg.breakout_x,
                   0.0 if cfg.daily_loss_cap_bp is None else cfg.daily_loss_cap_bp,
                   cfg.taker_bp, cfg.maker_bp, cfg.slip_bps, cfg.slip_range_frac], dtype=np.float64)
    CI = np.array([cfg.n_levels, EXIT_CODE[cfg.exit_mode], REBUILD_CODE[cfg.rebuild], 1 if cfg.cancel_opposite_on_fill else 0,
                   cfg.flat_cooldown, cfg.time_every, cfg.max_positions, 0 if cfg.daily_loss_cap_bp is None else 1,
                   1 if record else 0, ORDER_CODE[order]], dtype=np.int64)
    return CF, CI


def simulate_raw(ts, o, h, l, c, cfg, order: str, gate, record: bool):
    """Run the JIT loop. Returns (rows, extra, stats) exactly as the python ``_Sim`` holds them: ``rows`` = tuples
    (entry_bar, exit_bar, side, level, gross, gross_ns, net, 'a+b'), ``extra`` = record tuples (empty list when not recording)."""
    ts = np.asarray(ts, dtype=np.int64)
    o, h, l, c = (np.ascontiguousarray(a, dtype=np.float64) for a in (o, h, l, c))
    n = len(c)
    days = ts // MS_DAY
    g = np.ones(n, dtype=np.bool_) if gate is None else np.ascontiguousarray(gate, dtype=np.bool_)
    rand = np.random.default_rng(12345).random(n) if order == "rand" else np.zeros(1)
    CF, CI = _cfg_vectors(cfg, order, record)
    OF, OI, skipped, hdays, builds = _run(o, h, l, c, days, g, rand, CF, CI)
    kn = np.array(KIND_NAMES + ("",), dtype=object)
    k1 = kn[OI[:, OI_K1]]                              # -1 -> the last entry ''
    k2 = kn[OI[:, OI_K2]]
    kinds = [a if not b else a + "+" + b for a, b in zip(k1.tolist(), k2.tolist())]
    rows = list(zip(OI[:, OI_ENTRYBAR].tolist(), OI[:, OI_EXITBAR].tolist(), OI[:, OI_SIDE].tolist(), OI[:, OI_LEVEL].tolist(),
                    OF[:, O_GROSS].tolist(), OF[:, O_GROSSNS].tolist(), OF[:, O_NET].tolist(), kinds))
    extra = []
    if record:
        extra = list(zip(OF[:, O_ENTRYNS].tolist(), OF[:, O_ENTRY].tolist(), OF[:, O_XVAL].tolist(), OF[:, O_FEE].tolist(),
                         (OF[:, O_GROSSNS] - OF[:, O_GROSS]).tolist(), OF[:, O_MFE].tolist(), OF[:, O_MAE].tolist(),
                         [bool(x) for x in OI[:, OI_GAPFILL].tolist()], OI[:, OI_RID].tolist(), OI[:, OI_CONC].tolist()))
    return rows, extra, {"skipped_cap": int(skipped), "halted_days": int(hdays), "builds": int(builds)}

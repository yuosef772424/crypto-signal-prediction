"""
PURPOSE:  H16: owner's previous-candle range fade (sell limit at prev high, buy limit at prev low, TP at opposite
          end), event-driven numba sim on the 5m path; variants V0-V4.
TAGS:     H16, range fade, previous candle, limit orders, V0-V4, grid, 5m path, numba, range_fade csv
PITFALLS: V0/V1 (no stop) show a 100% win rate only because losers stay open (huge floating loss): judge V3/V4; no
          edge even in the optimistic upper bound; heavy runtime; needs ohlc_<SYMBOL>_5m.parquet in
          /home/user/research (built from github Speirsy11/crypto-dataset via LFS; not in repo)

H16: owner's previous-candle range fade (Addendum E). Event-driven on the 5-minute path (numba)."""
import numpy as np, pandas as pd, sys
from numba import njit
D = '/home/user/research'
EPS, MAKER, TAKER = 0.0002, 0.0002, 0.0007          # trade-through 2bp; maker 0.02%; taker 0.05% + 0.02% slip
MAXP = 60000
FILL = (EPS, MAKER, TAKER, False)          # conservative (pre-registered)

@njit
def sim(h, l, o, cid, refH, refL, allow, newc, variant, cap, tstop, day, nday, EPS, MAKER, TAKER, sameTP):
    # position slots
    side = np.zeros(MAXP, np.int8); ent = np.zeros(MAXP); tp = np.zeros(MAXP); sl = np.zeros(MAXP)
    ek = np.zeros(MAXP, np.int64); eb = np.zeros(MAXP, np.int64); act = np.zeros(MAXP, np.bool_)
    tr_pnl = np.zeros(MAXP * 4); tr_side = np.zeros(MAXP * 4, np.int8); tr_typ = np.zeros(MAXP * 4, np.int8)
    tr_ein = np.zeros(MAXP * 4, np.int64); tr_eout = np.zeros(MAXP * 4, np.int64); nt = 0
    real = np.zeros(nday); mtm = np.zeros(nday); maxopen = 0
    nL = 0; nS = 0; top = 0; sold = False; bought = False
    for i in range(len(h)):
        k = cid[i]
        if newc[i]:
            sold = False; bought = False
        # ---------- exits of existing positions ----------
        imb = nS - nL
        for p in range(top):
            if not act[p] or (eb[p] == i and not sameTP): continue
            s = side[p]; ex = 0.0; typ = 0
            if variant >= 3 and newc[i] and k - ek[p] >= tstop:          # time exit at bar open
                ex = o[i]; typ = 3
            elif variant >= 3 and ((s == 1 and l[i] <= sl[p]) or (s == -1 and h[i] >= sl[p])):
                ex = sl[p]; typ = 2                                        # stop first (conservative)
            else:
                susp = variant == 2 and ((s == 1 and imb >= 2) or (s == -1 and -imb >= 2))
                if not susp:
                    if s == 1 and h[i] >= tp[p] * (1 + EPS): ex = tp[p]; typ = 1
                    elif s == -1 and l[i] <= tp[p] * (1 - EPS): ex = tp[p]; typ = 1
                    elif variant == 2 and ((s == 1 and o[i] > tp[p]) or (s == -1 and o[i] < tp[p])):
                        ex = o[i]; typ = 4                                 # re-armed TP already passed
            if typ > 0:
                cost = MAKER + (MAKER if typ == 1 else TAKER)
                pnl = s * (ex / ent[p] - 1) - cost
                act[p] = False
                if s == 1: nL -= 1
                else: nS -= 1
                real[day[i]] += pnl
                tr_pnl[nt] = pnl; tr_side[nt] = s; tr_typ[nt] = typ; tr_ein[nt] = eb[p]; tr_eout[nt] = i; nt += 1
        # ---------- entries ----------
        if allow[i]:
            H = refH[i]; L = refL[i]; R = H - L
            for s in (-1, 1):
                if s == -1 and (sold or nS >= cap or not (h[i] >= H * (1 + EPS))): continue
                if s == 1 and (bought or nL >= cap or not (l[i] <= L * (1 - EPS))): continue
                p = top; top += 1
                side[p] = s; ek[p] = k; eb[p] = i; act[p] = True
                if s == -1:
                    ent[p] = H; tp[p] = L; sl[p] = H + R; sold = True; nS += 1
                else:
                    ent[p] = L; tp[p] = H; sl[p] = L - R; bought = True; nL += 1
                if variant >= 3 and ((s == 1 and l[i] <= sl[p]) or (s == -1 and h[i] >= sl[p])):   # stop inside entry bar
                    pnl = s * (sl[p] / ent[p] - 1) - MAKER - TAKER
                    act[p] = False
                    if s == 1: nL -= 1
                    else: nS -= 1
                    real[day[i]] += pnl
                    tr_pnl[nt] = pnl; tr_side[nt] = s; tr_typ[nt] = 2; tr_ein[nt] = i; tr_eout[nt] = i; nt += 1
        # compact slots occasionally
        if top > MAXP - 10:
            j = 0
            for p in range(top):
                if act[p]:
                    side[j] = side[p]; ent[j] = ent[p]; tp[j] = tp[p]; sl[j] = sl[p]; ek[j] = ek[p]; eb[j] = eb[p]; act[j] = True; j += 1
            for p in range(j, top): act[p] = False
            top = j
        if nL + nS > maxopen: maxopen = nL + nS
        # end-of-day mark to market
        if i == len(h) - 1 or day[i + 1] != day[i]:
            u = 0.0
            for p in range(top):
                if act[p]: u += side[p] * (o[i] / ent[p] - 1)
            mtm[day[i]] = u
    return tr_pnl[:nt], tr_side[:nt], tr_typ[:nt], tr_ein[:nt], tr_eout[:nt], real, mtm, maxopen, nL + nS

TFMAP = {'1h': '1h', '4h': '4h', '1d': '1D'}
def run(sym, tf, variant):
    b = pd.read_parquet(f'{D}/ohlc_{sym}_5m.parquet')
    ts = b.index
    cnd = ts.floor(TFMAP[tf])
    C = b.groupby(cnd).agg(high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
    er = (C.close - C.close.shift(20)).abs() / C.close.diff().abs().rolling(20).sum()
    ref = pd.DataFrame({'H': C.high.shift(1), 'L': C.low.shift(1), 'er': er.shift(1)}, index=C.index)
    ref = ref.reindex(cnd)
    cid = pd.Index(C.index).get_indexer(cnd).astype(np.int64)
    allow = ref.H.notna().values & (ref.H.values > ref.L.values)
    if variant == 4: allow &= (ref.er.values < 0.3)
    newc = np.r_[True, cid[1:] != cid[:-1]]
    days = ts.floor('1D'); du = pd.Index(days.unique()); day = du.get_indexer(days).astype(np.int64)
    cap = {0: 10 ** 7, 1: 3, 2: 5, 3: 3, 4: 3}[variant]
    out = sim(b.high.values, b.low.values, b.open.values, cid, ref.H.fillna(0).values, ref.L.fillna(0).values,
              allow, newc, variant, cap, 3, day, len(du), *FILL)
    pnl, sd, typ, ein, eout, real, mtm, maxopen, open_end = out
    T = pd.DataFrame({'pnl': pnl, 'side': sd, 'type': typ, 'entry': ts[ein], 'exit': ts[eout]})
    E = pd.Series(np.cumsum(real) + mtm, index=du)             # equity in units of one position notional
    return T, E, maxopen, open_end

def summarize(T, E, a, b):
    t = T[(T.entry >= a) & (T.entry <= b)]; e = E.loc[a:b]; d = e.diff().dropna()
    if len(t) == 0: return {}
    w = t.pnl[t.pnl > 0]; lo = t.pnl[t.pnl <= 0]
    return dict(trades=len(t), win=(t.pnl > 0).mean(), avg_win_bp=w.mean() * 1e4, avg_loss_bp=lo.mean() * 1e4,
                exp_bp=t.pnl.mean() * 1e4, pf=w.sum() / -lo.sum() if len(lo) else np.inf,
                total_units=d.sum(), maxdd_units=(e - e.cummax()).min(), t_daily=d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else np.nan,
                worst_bp=t.pnl.min() * 1e4, tp_share=(t.type == 1).mean())
SEGS = {'DISC': ('2017-08-01', '2021-12-31'), 'VAL': ('2022-01-01', '2023-12-31'), 'HOLD': ('2024-01-01', '2026-09-30')}
if __name__ == '__main__':
    segs = sys.argv[1].split(',') if len(sys.argv) > 1 else ['DISC']
    if len(sys.argv) > 2 and sys.argv[2] == 'optimistic':
        FILL = (0.0, 0.0, 0.0, True)                 # upper bound: touch = fill, zero costs, TP allowed in entry bar
    VARS = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 1, 2, 3, 4]
    rows = []
    for v in VARS:
        for tf in ['1h', '4h', '1d']:
            Es = []; Ts = []; mo = 0; oe = 0
            for s in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
                T, E, maxopen, open_end = run(s, tf, v); Ts.append(T); Es.append(E); mo = max(mo, maxopen); oe += open_end
            T = pd.concat(Ts); E = pd.concat(Es, axis=1).ffill().fillna(0).sum(axis=1)   # pooled 3 coins, 1 unit/position
            for sg in segs:
                r = summarize(T, E, *SEGS[sg]); r.update(variant=f'V{v}', tf=tf, seg=sg, max_open=mo); rows.append(r)
    R = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    print(R[['variant', 'tf', 'seg', 'trades', 'win', 'avg_win_bp', 'avg_loss_bp', 'exp_bp', 'pf', 'total_units', 'maxdd_units', 't_daily', 'worst_bp', 'tp_share', 'max_open']].round(2).to_string())
    R.to_csv(f'range_fade_{"_".join(segs)}{"_opt" if FILL[1] == 0 else ""}.csv', index=False)

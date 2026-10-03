"""Range grid / two-sided hedge on the next 1h bar. Path = P -> X1 -> X2 -> C with (X1,X2) = (H,L) or (L,H);
without 15m bars the order is unknown, so LB = worse of the two orderings (decisive), UB = better.
Grid: N buy levels P(1-span*pdn*i/N) and N sell levels P(1+span*pup*i/N), i=1..N, notional 1/N each; each fill's TP is
one level back (level 0 = P); levels re-arm after their TP fills. Stop: price beyond P(1-pdn(1+m)) / P(1+pup(1+m))
flattens the whole book (taker + slippage) and ends trading. End of bar: flatten at C (taker). Maker 0.02% per filled limit."""
import itertools, numpy as np, pandas as pd
import bracket_sim as B
from cross_asset.report import _t
e, P, H, L, C = B.e, B.P, B.H, B.L, B.C
MK, TK, SLIP = B.MK, B.TK, B.SLIP

def run_path(pu, pd_, N, span, m, active, first_up):
    n = len(P); i = np.arange(1, N + 1)[None, :]
    BL = P[:, None] * (1 - span * pd_[:, None] * i / N); BTP = P[:, None] * (1 - span * pd_[:, None] * (i - 1) / N)
    SLv = P[:, None] * (1 + span * pu[:, None] * i / N); STP = P[:, None] * (1 + span * pu[:, None] * (i - 1) / N)
    stop_lo, stop_hi = P * (1 - pd_ * (1 + m)), P * (1 + pu * (1 + m))
    hold_b = np.zeros((n, N), bool); hold_s = np.zeros((n, N), bool)
    pnl = np.zeros(n); fills = np.zeros(n); tk = np.zeros(n); alive = active.copy()
    way = [H, L, C] if first_up else [L, H, C]
    cur = P.copy()
    for nxt in way:
        up = nxt > cur; dn = nxt < cur
        lo, hi = np.minimum(cur, nxt)[:, None], np.maximum(cur, nxt)[:, None]
        a = alive[:, None]
        # down move: buy entries and short take-profits inside (lo, hi); up move: sell entries and long TPs
        eb = a & dn[:, None] & ~hold_b & (BL > lo) & (BL < hi)
        tps = a & dn[:, None] & hold_s & (STP > lo) & (STP < hi)
        es = a & up[:, None] & ~hold_s & (SLv > lo) & (SLv < hi)
        tpb = a & up[:, None] & hold_b & (BTP > lo) & (BTP < hi)
        # a level whose TP fills in the same monotone move it was entered cannot happen (TP is on the other side)
        pnl += ((tpb * (BTP / BL - 1)).sum(1) + (tps * (1 - STP / SLv)).sum(1)) / N
        fills += eb.sum(1) + es.sum(1) + tpb.sum(1) + tps.sum(1)
        hold_b = (hold_b & ~tpb) | eb; hold_s = (hold_s & ~tps) | es
        # stop at the end of the move (stop is beyond every level)
        st_lo = alive & dn & (nxt < stop_lo); st_hi = alive & up & (nxt > stop_hi)
        for mask, px in ((st_lo, stop_lo * (1 - SLIP)), (st_hi, stop_hi * (1 + SLIP))):
            if mask.any():
                pnl[mask] += ((hold_b[mask] * (px[mask, None] / BL[mask] - 1)).sum(1) + (hold_s[mask] * (1 - px[mask, None] / SLv[mask])).sum(1)) / N
                tk[mask] += (hold_b[mask].sum(1) + hold_s[mask].sum(1)) / N
                hold_b[mask] = False; hold_s[mask] = False; alive = alive & ~mask
        cur = nxt
    # flatten at close
    pnl += ((hold_b * (C[:, None] / BL - 1)).sum(1) + (hold_s * (1 - C[:, None] / SLv)).sum(1)) / N
    tk += (hold_b.sum(1) + hold_s.sum(1)) / N
    gross = pnl.copy()                       # before all fees
    pnl -= MK * fills / N + TK * tk
    return pnl, gross, fills

def sim(src, N, span, m, gate):
    pu, pd_ = (B.PU, B.PD) if src == 'model' else (VOLB, VOLB)
    rk = pd.Series(pu + pd_).groupby(e.bucket.to_numpy()).rank(pct=True).to_numpy()
    active = rk <= gate
    a = run_path(pu, pd_, N, span, m, active, True); b = run_path(pu, pd_, N, span, m, active, False)
    lb = np.where(a[0] <= b[0], 0, 1)
    pick = lambda k, w: np.where(w == 0, a[k], b[k])
    return (pick(0, lb), pick(1, lb), pick(2, lb)), (pick(0, 1 - lb), pick(1, 1 - lb)), active

vm = (e.split == 'val').to_numpy()
c = np.median((B.PU + B.PD)[vm] / 2) / np.median(e.pvol.to_numpy()[vm])
VOLB = c * e.pvol.to_numpy()          # past-vol bounds, scale-matched to the model's median predicted half-range
def per_stats(net, gross, fills, active, sub):
    per = pd.Series(net[sub]).groupby(e.bucket.to_numpy()[sub]).mean() * 100
    g = pd.Series(gross[sub]).groupby(e.bucket.to_numpy()[sub]).mean() * 100
    tr = active[sub] & (fills[sub] > 0)
    mon = pd.Series(net[sub]).groupby(e.month.to_numpy()[sub]).sum()
    return dict(per=per.mean(), t=_t(per)[1], gross_t=_t(g)[1], traded=tr.mean(), net_trade=net[sub][tr].mean() * 100,
                win=(net[sub][tr] > 0).mean(), months_pos=f'{(mon > 0).sum()}/{len(mon)}')
if __name__ == '__main__':
    rows = []
    cfgs = [('grid', N, 1.0, m) for N in (3, 5) for m in (0.25, 0.5, 1.0)] + [('hedge', 1, a, m) for a in (0.5, 0.75, 1.0) for m in (0.25, 0.5, 1.0)]
    for (kind, N, span, m), src, gate in itertools.product(cfgs, ('model', 'pastvol'), (1.0, 0.5, 0.33)):
        (net, gross, fills), (ub_net, ub_g), active = sim(src, N, span, m, gate)
        for sp in ('val', 'test'):
            sub = (e.split == sp).to_numpy()
            r = per_stats(net, gross, fills, active, sub)
            ubp = pd.Series(ub_net[sub]).groupby(e.bucket.to_numpy()[sub]).mean() * 100
            rows.append(dict(kind=kind, N=N, span=span, m=m, src=src, gate=gate, split=sp, **r, ub_per=ubp.mean(), ub_t=_t(ubp)[1]))
    R = pd.DataFrame(rows); R.to_pickle('grid_res.pkl')
    key = ['kind', 'N', 'span', 'm', 'src', 'gate']
    V = R[R.split == 'val'].set_index(key); T = R[R.split == 'test'].set_index(key)
    print('configs:', len(V), '| val t>2 (LB):', (V.t > 2).sum())
    best = V.t.idxmax(); print('VAL-best (LB):', best, 'val per %.4f t %.2f' % (V.loc[best].per, V.loc[best].t))
    pd.set_option('display.width', 250)
    cols = ['per', 't', 'gross_t', 'traded', 'net_trade', 'win', 'months_pos', 'ub_per', 'ub_t']
    print('TEST at val-best, model vs past-vol bounds, gates:')
    print(T.xs(best[:4], level=[0, 1, 2, 3], drop_level=False)[cols].round(4).to_string())
    for k in ('grid', 'hedge'):
        b = V.loc[k].t.idxmax(); print(k, 'val-best', b, '| val t %.2f | test per %.4f t %.2f gross_t %.2f ub_t %.2f months+ %s' % (
            V.loc[(k,) + b].t, T.loc[(k,) + b].per, T.loc[(k,) + b].t, T.loc[(k,) + b].gross_t, T.loc[(k,) + b].ub_t, T.loc[(k,) + b].months_pos))

"""صفقات يومية بقوس ثابت: جني ربح +B / وقف خسارة -B (من سعر الدخول)، وإلا الإغلاق في نهاية اليوم.
ترتيب اللمس عند لمس الطرفين في اليوم نفسه: من شموع الساعة (bracket_fetch.py)؛ ما بقي غامضاً (نفس الساعة/بلا بيانات)
يُحسب بحدّين: متشائم (الوقف أولاً) ومتفائل (الهدف أولاً).
المعلومة الحقيقية = النموذج أفضل من اختيار عشوائي لنفس عدد العملات في نفس الأيام (اختبار تبديل)."""
import sys, numpy as np, pandas as pd
S = sys.argv[1]; B = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
COST = 0.08 / 100; N_PERM = 300
first = pd.read_pickle(f"{S}/bracket_first_{int(B*100)}.pkl").drop_duplicates(["asset", "day"])
rng = np.random.default_rng(0)

def load(run, sp):
    d = pd.read_csv(f"{S}/{run}/signals_{sp}.csv.gz", usecols=["asset", "timestamp", "entry", "fut_close", "fut_high", "fut_low", "p_up_close"])
    d["day"] = pd.to_datetime(d["timestamp"]) + pd.Timedelta(days=1)
    d["r"] = d.fut_close / d.entry - 1
    up, dn = d.fut_high >= d.entry * (1 + B), d.fut_low <= d.entry * (1 - B)
    d["ev"] = np.select([up & ~dn, dn & ~up, ~up & ~dn], ["up", "down", "none"], "both")
    d = d.merge(first, on=["asset", "day"], how="left")
    both = d.ev == "both"
    d.loc[both, "ev"] = d.loc[both, "first"].replace({"none": "amb", "same_hour": "amb", "nodata": "amb"}).fillna("amb")
    return d.drop(columns="first").sort_values(["timestamp", "asset"]).reset_index(drop=True)

def trade(ev, r, side, amb, slip=0.0):
    """side +1 شراء / -1 بيع. amb: 'pess' أو 'opt' لما بقي غامضاً."""
    tp = (ev == "up") if side > 0 else (ev == "down")
    sl = (ev == "down") if side > 0 else (ev == "up")
    a = ev == "amb"
    tp = tp | (a & (amb == "opt")); sl = sl | (a & (amb == "pess"))
    return np.where(tp, B, np.where(sl, -B - slip, side * r)) - COST

def strat_days(d, q, kind, amb, score, slip=0.0):
    """متوسط عائد الصفقة لكل يوم (وزن متساوٍ). kind: long / short / ls."""
    out = []
    for _, g in d.groupby("timestamp", sort=False):
        n = len(g); k = max(1, int(q * n))
        if n < 20: continue
        o = np.argsort(g[score].to_numpy(), kind="stable")
        ev, r = g.ev.to_numpy(), g.r.to_numpy()
        L = trade(ev[o[-k:]], r[o[-k:]], +1, amb, slip).mean()
        Sh = trade(ev[o[:k]], r[o[:k]], -1, amb, slip).mean()
        out.append(L if kind == "long" else Sh if kind == "short" else 0.5 * (L + Sh))
    return np.array(out)

def stats(x):
    m, s = x.mean(), x.std(ddof=1)
    return m * 100, m / s * np.sqrt(len(x))

def perm_p(d, q, kind, amb, model_mean):
    """نفس الاستراتيجية بدرجات عشوائية (تبديل الدرجة داخل كل يوم)."""
    ms = []
    for i in range(N_PERM):
        d2 = d.assign(rnd=rng.permutation(len(d)))
        ms.append(strat_days(d2, q, kind, amb, "rnd").mean() * 100)
    ms = np.array(ms)
    return ms.mean(), (ms >= model_mean).mean()

for run, label in (("eval_timesplit_sig", "relative"), ("eval_volnorm", "volnorm+relative")):
    V, T = load(run, "val"), load(run, "test")
    print(f"\n################ {label} | قوس ±{B:.0%} | تكلفة 0.08%/صفقة")
    print("توزيع الأحداث (test):", (T.ev.value_counts(normalize=True) * 100).round(2).to_dict())
    # معدّل الإصابة حسب عُشر الدرجة: هل الأعلى يلمس +5% أولاً أكثر؟
    dec = T.groupby("timestamp")["p_up_close"].transform(lambda x: pd.qcut(x.rank(method="first"), 10, labels=False)) + 1
    h = T.assign(dec=dec).groupby("dec")["ev"].value_counts(normalize=True).unstack().fillna(0) * 100
    h["up-down"] = h.get("up", 0) - h.get("down", 0)
    h["up_share"] = 100 * h.get("up", 0) / (h.get("up", 0) + h.get("down", 0))   # من الحسومة: كم صعد أولاً؟
    print("نسب الأحداث % لكل عُشر (1 = أدنى p_up، 10 = أعلى):"); print(h.round(2).T.to_string())
    rows = []
    for kind in ("long", "short", "ls"):
        for q in (0.05, 0.1, 0.2):
            rv = strat_days(V, q, kind, "pess", "p_up_close")
            rt_p = strat_days(T, q, kind, "pess", "p_up_close"); rt_o = strat_days(T, q, kind, "opt", "p_up_close")
            rows.append((kind, q, *stats(rv), *stats(rt_p), stats(rt_o)[0]))
    df = pd.DataFrame(rows, columns=["kind", "q", "val_mean%", "val_t", "test_mean%(pess)", "test_t(pess)", "test_mean%(opt)"])
    print(df.round(3).to_string(index=False))
    best = df.loc[df["val_t"].idxmax()]
    kind, q = best["kind"], best["q"]
    rt = strat_days(T, q, kind, "pess", "p_up_close")
    rnd_mean, p = perm_p(T, q, kind, "pess", rt.mean() * 100)
    print(f"المختار على val: {kind} q={q} → test (متشائم) {rt.mean()*100:+.3f}%/صفقة، t={stats(rt)[1]:+.2f}، "
          f"إجمالي تراكمي {np.prod(1 + rt) - 1:+.1%} على {len(rt)} يوماً")
    print(f"   مقابل عشوائي ({N_PERM} تبديلاً): متوسط العشوائي {rnd_mean:+.3f}%/صفقة | p(عشوائي ≥ النموذج) = {p:.3f}")
    for k2 in ("long", "short", "ls"):
        rt2 = strat_days(T, 0.1, k2, "pess", "p_up_close")
        rm, p2 = perm_p(T, 0.1, k2, "pess", rt2.mean() * 100)
        print(f"   [q=0.1 {k2:5s}] النموذج {rt2.mean()*100:+.3f} | عشوائي {rm:+.3f} | الفرق {rt2.mean()*100-rm:+.3f} | p={p2:.3f}")

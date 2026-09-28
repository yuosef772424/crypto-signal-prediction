"""تقييم ملفات الإشارات (النموذج الحالي أو نموذج اللوحة) بنفس المقاييس، من الأعمدة المُصدَّرة وحدها.

الأهداف النسبية تُعاد من الأسعار: عائد كل نوع (close من الدخول، high/low من آخر high/low) ناقص وسيط كل عملات القسم
في نفس الطابع الزمني — مطابق لـ retarget_splits(mode="relative") لأن ملف الإشارات يحوي كل عيّنات القسم.

المقاييس (test للحكم، val للمرجع فقط — لا شيء يُختار على test):
  auc_*            : AUC لكل هدف (p_up_* مقابل «تتفوّق على وسيط اليوم»)، مُجمَّع على كل الصفوف.
  ic_*             : IC يومي = متوسط Spearman(الدرجة، العائد) لكل يوم (≥ min_coins عملة) و t = متوسط/انحراف·√أيام.
  reg_ic_close     : IC يومي لـ mu_close مقابل العائد النسبي (IC الانحدار في تقرير الفجوة).
  volq_ic          : IC داخل خُمسيات التقلّب السابق (انحراف عوائد 20 يوماً سابقة لكل عملة)، متوسط الخُمسيات ثم الأيام.
  partial_ic       : IC بعد إزالة رتبة التقلّب السابق خطّياً من رتبتي الدرجة والعائد (لكل يوم).
  score_vs_vol_ic  : Spearman(الدرجة، −التقلّب) — كم من الدرجة مجرّد «هدوء».
  mn_*             : محفظة أوزان رتب محايدة للسوق على كل العملات (cmp_models.py): إجمالي وصافٍ بتكلفة 0.08% على الدوران.
  bracket          : قوس ±B: لكل عُشر من الدرجة داخل اليوم نسب «+B أولاً/−B أولاً/لا لمس/غامض» و up_share = +B أولاً ÷ المحسومة.
                     اللمس المزدوج في اليوم نفسه يُحسم من جدول first_touch (bracket_fetch.py، شموع الساعة) إن أُعطي، وإلا «غامض».
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="An input array is constant")
DAY = pd.Timedelta(days=1)


def prepare(df, group_ns=None):
    """يضيف r (عائد close الخام)، rel_{هدف} (نسبي لوسيط اليوم)، cls_{هدف}، وpvol (تقلّب سابق بلا نظر للمستقبل).
    group_ns: None = التجميع بالطابع الدقيق. غير ذلك: timestamp يُستبدَل بأرضيته (من epoch) — لطوابع غير متطابقة
    بين العملات (فريم الساعة)، بنفس group_freq في retarget_splits؛ الطابع الأصلي يبقى في timestamp_raw."""
    d = df.copy()
    if group_ns:
        d["timestamp_raw"] = d["timestamp"]
        d["timestamp"] = (d["timestamp"].astype("int64") // int(group_ns)) * int(group_ns)
    d["r"] = d["fut_close"] / d["entry"] - 1.0
    raw = {"close": d["r"], "high": d["fut_high"] / d["last_high"] - 1.0, "low": d["fut_low"] / d["last_low"] - 1.0}
    for t, v in raw.items():
        d[f"rel_{t}"] = (v - v.groupby(d["timestamp"]).transform("median")).clip(-1, 1)
        d[f"cls_{t}"] = (v > v.groupby(d["timestamp"]).transform("median")).astype(int)
    d = d.sort_values(["asset", "timestamp"])
    # r عند اليوم t-1 = عائد (t-1 → t) معروف عند إغلاق t ⇒ shift(1) بلا تسرّب
    d["pvol"] = d.groupby("asset")["r"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).std())
    d["pvol"] = d["pvol"].fillna(d.groupby("timestamp")["pvol"].transform("median")).fillna(d["pvol"].median())
    return d.sort_values(["timestamp", "asset"]).reset_index(drop=True)


def _t(x):
    x = pd.Series(x).dropna()
    return (float(x.mean()), float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan,
            int(len(x)))


def daily_ic(d, score, target="r", min_coins=10):
    g = d.groupby("timestamp")
    ic = g.apply(lambda x: x[score].corr(x[target], method="spearman") if len(x) >= min_coins else np.nan)
    return _t(ic)


def vol_quintile_ic(d, score, target="r", n_q=5, min_cell=5):
    def per_day(x):
        if len(x) < n_q * min_cell:
            return np.nan
        q = pd.qcut(x["pvol"].rank(method="first"), n_q, labels=False)
        v = [c[score].corr(c[target], method="spearman") for _, c in x.groupby(q) if len(c) >= min_cell]
        return np.nanmean(v) if v else np.nan
    return _t(d.groupby("timestamp").apply(per_day))


def partial_ic(d, score, target="r", min_coins=10):
    def per_day(x):
        if len(x) < min_coins:
            return np.nan
        z = np.c_[np.ones(len(x)), x["pvol"].rank().to_numpy()]
        res = [a - z @ np.linalg.lstsq(z, a, rcond=None)[0] for a in (x[score].rank().to_numpy(), x[target].rank().to_numpy())]
        return np.corrcoef(res[0], res[1])[0, 1]
    return _t(d.groupby("timestamp").apply(per_day))


def mn_portfolio(d, score, q=0.5, cost_pct=0.0, min_coins=20):
    """محفظة أوزان رتب (نسخة port في cmp_models.py): %/يوم وقيمة t."""
    rows, prev = [], pd.Series(dtype=float)
    for _, g in d.groupby("timestamp"):
        if len(g) < min_coins:
            continue
        rk = g[score].rank(method="first").to_numpy()
        c = rk - (len(g) + 1) / 2
        k = int(np.floor(q * len(g)))
        o = np.argsort(rk)
        sel = np.zeros(len(g), bool)
        sel[o[:k]] = sel[o[-k:]] = True
        w = np.where(sel, c, 0.0)
        pos, neg = w.clip(min=0), (-w).clip(min=0)
        w = 0.5 * pos / pos.sum() - 0.5 * neg / neg.sum()
        ws = pd.Series(w, index=g["asset"].to_numpy())
        turn = 1.0 if prev.empty else 0.5 * ws.sub(prev, fill_value=0).abs().sum()
        prev = ws
        rows.append(np.dot(w, g["r"].to_numpy()) - cost_pct / 100 * turn)
    x = np.array(rows)
    return float(x.mean() * 100), float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def _decile(d, score, n=10):
    return d.groupby("timestamp")[score].transform(lambda x: pd.qcut(x.rank(method="first"), n, labels=False)) + 1


def bracket_table(d, score="p_up_close", B=0.05, first_touch=None, n_dec=10):
    """نسب الأحداث % لكل عُشر من الدرجة (1 = الأدنى) + up_share و touch. first_touch: DataFrame(asset, day, first)."""
    up, dn = d["fut_high"] >= d["entry"] * (1 + B), d["fut_low"] <= d["entry"] * (1 - B)
    ev = pd.Series(np.select([up & ~dn, dn & ~up, ~up & ~dn], ["up", "down", "none"], "both"), index=d.index)
    if first_touch is not None:
        ft = first_touch.drop_duplicates(["asset", "day"])
        key = pd.DataFrame({"asset": d["asset"].to_numpy(), "day": pd.to_datetime(d["timestamp"]) + DAY})
        m = key.merge(ft, on=["asset", "day"], how="left")["first"].to_numpy()
        both = (ev == "both").to_numpy()
        res = pd.Series(m[both]).replace({"none": "amb", "same_hour": "amb", "nodata": "amb"}).fillna("amb").to_numpy()
        ev.loc[both] = res
    else:
        ev = ev.replace({"both": "amb"})
    t = pd.crosstab(_decile(d, score, n_dec), ev, normalize="index") * 100
    for c in ("up", "down", "none", "amb"):
        if c not in t:
            t[c] = 0.0
    counts = pd.crosstab(_decile(d, score, n_dec), ev)
    t["touch"] = 100 - t["none"]
    t["up_share"] = 100 * t["up"] / (t["up"] + t["down"])
    t["n_resolved"] = counts.get("up", 0) + counts.get("down", 0)
    t.index.name = "decile"
    return t[["up", "down", "amb", "none", "touch", "up_share", "n_resolved"]]


def summarize(val_df, test_df, first_touch=None, B=0.05, min_coins=10, group_ns=None):
    """قاموس المقاييس لنموذج واحد + جدول القوس (test)."""
    V, T = prepare(val_df, group_ns), prepare(test_df, group_ns)
    m = {"test_days": T["timestamp"].nunique(), "test_rows": len(T)}
    from sklearn.metrics import roc_auc_score
    for t in ("high", "low", "close"):
        if f"p_up_{t}" in T:
            m[f"auc_{t}"] = roc_auc_score(T[f"cls_{t}"], T[f"p_up_{t}"])
    if "p_up_close" not in T:     # close معلّق: بقية المقاييس معرّفة على درجة close
        return m, pd.DataFrame(columns=["up", "down", "amb", "none", "touch", "up_share", "n_resolved"])
    m["ic_p_up_close"], m["ic_p_up_close_t"], _ = daily_ic(T, "p_up_close", "r", min_coins)
    m["val_ic_p_up_close"], m["val_ic_p_up_close_t"], _ = daily_ic(V, "p_up_close", "r", min_coins)
    if "mu_close" in T:
        m["reg_ic_close"], m["reg_ic_close_t"], _ = daily_ic(T, "mu_close", "rel_close", min_coins)
        m["ic_mu_close"], m["ic_mu_close_t"], _ = daily_ic(T, "mu_close", "r", min_coins)
    m["volq_ic"], m["volq_ic_t"], _ = vol_quintile_ic(T, "p_up_close")
    m["partial_ic"], m["partial_ic_t"], _ = partial_ic(T, "p_up_close", "r", min_coins)
    m["score_vs_lowvol_ic"], _, _ = daily_ic(T.assign(neg_vol=-T["pvol"]), "p_up_close", "neg_vol", min_coins)
    m["mn_gross"], m["mn_gross_t"] = mn_portfolio(T, "p_up_close")
    m["mn_net"], m["mn_net_t"] = mn_portfolio(T, "p_up_close", cost_pct=0.08)
    dec = T.assign(dec=_decile(T, "p_up_close")).groupby("dec")["r"]
    m["dec1_mean_r%"], m["dec10_mean_r%"] = dec.mean().iloc[0] * 100, dec.mean().iloc[-1] * 100
    m["dec1_pump%"] = (T.assign(dec=_decile(T, "p_up_close")).query("dec == 1")["r"] > 0.2).mean() * 100
    br = bracket_table(T, "p_up_close", B, first_touch)
    m["upshare_dec1"], m["upshare_dec10"] = br["up_share"].iloc[0], br["up_share"].iloc[-1]
    m["upshare_spearman"] = pd.Series(br.index).corr(br["up_share"].reset_index(drop=True), method="spearman")
    m["touch_dec1"], m["touch_dec10"] = br["touch"].iloc[0], br["touch"].iloc[-1]
    return m, br


ROWS = [("auc_high", "AUC high"), ("auc_low", "AUC low"), ("auc_close", "AUC close"),
        ("ic_p_up_close", "IC يومي p_up_close"), ("ic_p_up_close_t", "  t (IC يومي)"),
        ("val_ic_p_up_close", "IC يومي p_up_close (val)"),
        ("reg_ic_close", "IC الانحدار (mu_close مقابل النسبي)"),
        ("volq_ic", "IC داخل خُمسيات التقلّب"), ("volq_ic_t", "  t (خُمسيات التقلّب)"),
        ("partial_ic", "IC جزئي (بعد إزالة التقلّب)"), ("partial_ic_t", "  t (جزئي)"),
        ("score_vs_lowvol_ic", "Spearman(الدرجة، −التقلّب)"),
        ("mn_gross", "محفظة رتب إجمالي %/يوم"), ("mn_net", "محفظة رتب صافٍ 0.08% %/يوم"), ("mn_net_t", "  t (صافٍ)"),
        ("upshare_dec1", "قوس ±5%: up_share أدنى عُشر"), ("upshare_dec10", "قوس ±5%: up_share أعلى عُشر"),
        ("upshare_spearman", "قوس ±5%: رتابة up_share (Spearman)"),
        ("touch_dec1", "قوس ±5%: لمس أدنى عُشر %"), ("touch_dec10", "قوس ±5%: لمس أعلى عُشر %"),
        ("dec1_mean_r%", "متوسط العائد أدنى عُشر %"), ("dec10_mean_r%", "متوسط العائد أعلى عُشر %"),
        ("test_days", "أيام test")]


def compare(runs, first_touch=None, B=0.05, verbose=True, group_ns=None):
    """runs: {اسم: (val_df, test_df)}. يُرجع (جدول المقارنة، {اسم: جدول القوس}). group_ns: انظر prepare."""
    metrics, brackets = {}, {}
    for name, (v, t) in runs.items():
        metrics[name], brackets[name] = summarize(v, t, first_touch, B, group_ns=group_ns)
    table = pd.DataFrame({name: {label: m.get(k, np.nan) for k, label in ROWS} for name, m in metrics.items()})
    if verbose:
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(table.to_string())
            for name, br in brackets.items():
                print(f"\nقوس ±{B:.0%} — {name} (test، %؛ 1 = أدنى p_up_close):")
                print(br.round(2).T.to_string())
    return table, brackets


def up_share_table(brackets):
    """up_share لكل عُشر × نموذج (للتوثيق)."""
    return pd.DataFrame({n: b["up_share"] for n, b in brackets.items()}).round(1)

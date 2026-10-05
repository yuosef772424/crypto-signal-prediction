"""
PURPOSE:  Section 7-f: market-neutral portfolio report (long_short, short_only, long_only, rank_weighted, rank_weighted_vol) with net-of-cost bootstrap and a last-step GBM baseline.
TAGS:     market_neutral_report, split_asset_names, rank IC, decile table, rank_weighted, long_short, portfolio
PITFALLS: (side, q) are picked on val only; the full test grid is shown for transparency, not for selection. split_asset_names is also used by the panel cell and tools/evaluate_trained_model.py. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 34 (section 7-f).
"""
import numpy as np
import pandas as pd

_MN_SIDES = ("long_short", "short_only", "long_only", "rank_weighted", "rank_weighted_vol")


def _norm_symbol(s):
    s = str(s).upper()
    return s[:-4] if s.endswith("USDT") else s


def _resolve_universe(universe):
    """None = كل العملات؛ "categories" = COINS_BY_CATEGORY من خط الأنابيب؛ أو أي قائمة رموز (BTC أو BTCUSDT)."""
    if universe is None:
        return None
    if isinstance(universe, str) and universe == "categories":
        cats = globals().get("COINS_BY_CATEGORY")
        if not cats:
            raise ValueError("COINS_BY_CATEGORY غير معرَّف — شغّل خط الأنابيب أو مرّر قائمة رموز")
        universe = [c for v in (cats.values() if isinstance(cats, dict) else cats) for c in v]
    return {_norm_symbol(s) for s in universe}


def split_asset_names(dataset, split="val", config=None):
    """أسماء العملات لصفوف train أو val المدمجة، بنفس أقنعة split_data (global_time) وترتيبها. None إن تعذّر."""
    config = CONFIG if config is None else config
    if config.get("split_mode", "global_time") != "global_time" or not dataset.get("asset_bounds"):
        return None
    ts = sample_timestamps(dataset)
    train_end, val_end = resolve_split_dates(dataset, config["train_pct"], config["val_pct"], config)
    gap = embargo_duration(dataset, config)
    mask = np.asarray({"train": ts <= train_end, "val": (ts > train_end + gap) & (ts <= val_end)}[split])
    names = np.empty(len(ts), dtype=object)
    for b in dataset["asset_bounds"]:
        names[b["start"]:b["end"]] = b.get("name", b.get("asset", f"asset_{b['start']}_{b['end']}"))
    return names[mask]


def _mn_frame(df, score_col, universe):
    out = pd.DataFrame({"asset": df["asset"].values, "ts": df["timestamp"].values,
                        "score": df[score_col].values.astype("float64"),
                        "r": (df["fut_close"].values / df["entry"].values - 1.0).astype("float64")})
    out = out[np.isfinite(out["score"]) & np.isfinite(out["r"])]
    if universe is not None and (out["asset"] != "all").any():   # val بلا أسماء عملات (بلا dataset) لا يُفلتَر
        out = out[out["asset"].map(_norm_symbol).isin(universe)]
    # تقلّب سابق لكل عملة (انحراف عوائد الفترات السابقة فقط — r عند ts-1 معروف عند ts) لأوزان rank_weighted_vol
    out = out.sort_values(["asset", "ts"])
    past = out.groupby("asset")["r"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    out["vol"] = past.fillna(out.groupby("ts")["r"].transform(lambda s: s.abs().median())).clip(lower=1e-4)
    return out.sort_values(["ts", "asset"])


def _holding_stride(ts, horizon_ns):
    """كل كم طابعاً زمنياً تُؤخذ فترة كي لا تتداخل فترات الاحتفاظ (أفق > خطوة العيّنات)."""
    uniq = np.unique(ts)
    step = float(np.median(np.diff(uniq))) if len(uniq) > 1 else float(horizon_ns)
    k = max(1, int(np.ceil(horizon_ns / step))) if step > 0 else 1
    return k, step


def _mn_period_returns(frame, q, side, cost_pct, min_assets, stride=1, min_per_leg=5, cost_model="turnover"):
    """(سلسلة عائد المحفظة لكل فترة كسراً، متوسط الدوران). فترة بأقل من min_per_leg عملة في الطرف تُتخطّى —
    طرف من 2-3 عملات ضجيج خالص (عملة واحدة تقفز 50% تحكم الفترة كلها).
    cost_model="turnover": التكلفة على نسبة العملات التي تغيّرت في كل طرف عن الفترة السابقة (الترتيب يستمر غالباً)؛
    "full": كل المراكز تُغلق وتُفتح كل فترة. سلة السوق دورانها ≈ 0. بلا أسماء عملات يُستخدم "full"."""
    if side in ("rank_weighted", "rank_weighted_vol"):
        return _mn_rank_weighted(frame, q, side, cost_pct, min_assets, stride, min_per_leg)
    keep = set(np.unique(frame["ts"])[::stride])
    use_turnover = cost_model == "turnover" and frame["asset"].nunique() > 1
    prev = {"long": None, "short": None}
    rows, turns = {}, []

    def _turn(key, names):
        old = prev[key]
        prev[key] = names
        return 1.0 if (old is None or not names) else 1.0 - len(names & old) / len(names)
    for ts, g in frame.groupby("ts", sort=True):
        if ts not in keep or len(g) < min_assets:
            continue
        r = g["r"].to_numpy()
        order = np.argsort(g["score"].to_numpy(), kind="stable")
        k = int(np.floor(q * len(g)))
        if k < min_per_leg:
            continue
        top, bottom, mkt = r[order[-k:]].mean(), r[order[:k]].mean(), r.mean()
        long_leg, short_leg = {"long_short": (top, bottom), "short_only": (mkt, bottom),
                               "long_only": (top, mkt)}[side]
        if use_turnover:
            a = g["asset"].to_numpy()
            t_long = _turn("long", set(a[order[-k:]])) if side != "short_only" else 0.0
            t_short = _turn("short", set(a[order[:k]])) if side != "long_only" else 0.0
            turn = 0.5 * (t_long + t_short)
        else:
            turn = 1.0 if side == "long_short" else 0.5 if cost_model == "turnover" else 1.0
        turns.append(turn)
        rows[ts] = 0.5 * (long_leg - short_leg) - cost_pct / 100.0 * turn
    return pd.Series(rows, dtype="float64").sort_index(), (float(np.mean(turns)) if turns else np.nan)


def _mn_rank_weighted(frame, q, side, cost_pct, min_assets, stride=1, min_per_leg=5):
    """أوزان متناسبة مع الرتبة المركزية لكل عملة في الطرفين (q=0.5 = الكون كله) — الطريقة القياسية لتحويل IC إلى
    محفظة، بدل أطراف 5% الأكثر ضجيجاً. rank_weighted_vol: الوزن ÷ التقلّب السابق. كل طرف = نصف رأس المال (محايد).
    الدوران = نصف مجموع |تغيّر الأوزان| بين فترتين (استبدال كامل = 1، كتكلفة الأطراف في _mn_period_returns)."""
    keep = set(np.unique(frame["ts"])[::stride])
    rows, turns, prev = {}, [], pd.Series(dtype="float64")
    for ts, g in frame.groupby("ts", sort=True):
        if ts not in keep or len(g) < min_assets:
            continue
        k = int(np.floor(q * len(g)))
        if k < min_per_leg:
            continue
        rk = g["score"].rank(method="first").to_numpy()
        c = rk - (len(g) + 1) / 2.0                         # رتبة مركزية: سالبة للأضعف، موجبة للأقوى
        order = np.argsort(rk)
        sel = np.zeros(len(g), dtype=bool)
        sel[order[:k]] = sel[order[-k:]] = True
        w = np.where(sel, c, 0.0)
        if side == "rank_weighted_vol":
            w = w / g["vol"].to_numpy()
        pos, neg = w.clip(min=0), (-w).clip(min=0)
        if pos.sum() <= 0 or neg.sum() <= 0:
            continue
        w = 0.5 * pos / pos.sum() - 0.5 * neg / neg.sum()
        ws = pd.Series(w, index=g["asset"].to_numpy())
        turn = 1.0 if prev.empty else 0.5 * ws.sub(prev, fill_value=0.0).abs().sum()
        prev = ws[ws != 0]
        turns.append(turn)
        rows[ts] = float(np.dot(w, g["r"].to_numpy())) - cost_pct / 100.0 * turn
    return pd.Series(rows, dtype="float64").sort_index(), (float(np.mean(turns)) if turns else np.nan)


def _mn_stats(ret, periods_per_year, n_boot, seed):
    ret = np.asarray(ret, dtype="float64")
    n = len(ret)
    if n < 3 or np.std(ret) < 1e-12:
        return {"n_periods": n, "mean_pct": np.nan, "mean_lo": np.nan, "mean_hi": np.nan, "sharpe": np.nan,
                "sharpe_lo": np.nan, "sharpe_hi": np.nan, "pos_periods": np.nan, "total_pct": np.nan}
    ann = np.sqrt(periods_per_year) if periods_per_year else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = ret[idx]
    b_mean = boot.mean(1)
    b_sharpe = b_mean / np.maximum(boot.std(1, ddof=1), 1e-12) * ann
    return {"n_periods": n, "mean_pct": ret.mean() * 100,
            "mean_lo": np.percentile(b_mean, 2.5) * 100, "mean_hi": np.percentile(b_mean, 97.5) * 100,
            "sharpe": ret.mean() / ret.std(ddof=1) * ann,
            "sharpe_lo": np.percentile(b_sharpe, 2.5), "sharpe_hi": np.percentile(b_sharpe, 97.5),
            "pos_periods": float(np.mean(ret > 0)), "total_pct": (np.prod(1 + ret) - 1) * 100}


def _rank_ic(frame, min_assets):
    ics = [g["score"].corr(g["r"], method="spearman") for _, g in frame.groupby("ts") if len(g) >= min_assets]
    ics = np.asarray([x for x in ics if np.isfinite(x)])
    if len(ics) < 3:
        return {"ic_mean": np.nan, "ic_t": np.nan, "n": len(ics)}
    return {"ic_mean": ics.mean(), "ic_t": ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics)), "n": len(ics)}


def _decile_table(frame, min_assets, n_bins=10):
    """متوسط العائد النسبي (ناقص متوسط السوق في نفس الفترة) لكل عُشر من الدرجة — هل هو رتيب؟"""
    parts = []
    for ts, g in frame.groupby("ts"):
        if len(g) < min_assets:
            continue
        rel = g["r"] - g["r"].mean()
        b = pd.qcut(g["score"].rank(method="first"), n_bins, labels=False)
        parts.append(pd.DataFrame({"bin": b.values + 1, "rel": rel.values}))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts)
    return (d.groupby("bin")["rel"].agg(["mean", "count"]).assign(mean=lambda x: x["mean"] * 100)
             .rename(columns={"mean": "rel_ret_pct", "count": "n"}))


def _mn_evaluate(val_df, test_df, score_col, universe, quantiles, sides, cost_pct, min_assets, n_boot, seed,
                 horizon_ns, min_per_leg, cost_model):
    fr = {"val": _mn_frame(val_df, score_col, universe), "test": _mn_frame(test_df, score_col, universe)}
    rows, series = [], {}
    for part, f in fr.items():
        k, step = _holding_stride(f["ts"].to_numpy(), horizon_ns)
        ppy = 365.25 * 86400e9 / (step * k) if step >= 60e9 else None   # طوابع حقيقية بالنانوثانية فقط
        for side in sides:
            for q in quantiles:
                s, turn = _mn_period_returns(f, q, side, cost_pct, min_assets, stride=k, min_per_leg=min_per_leg,
                                             cost_model=cost_model)
                series[(part, side, q)] = s
                rows.append({"split": part, "side": side, "q": q, "turnover": turn,
                             **_mn_stats(s.values, ppy, n_boot, seed)})
    grid = pd.DataFrame(rows)
    # قابلية التقييم على test (عدد الفترات بطرفين كافيين) تُعرف من أحجام المقاطع وحدها، بلا أي عائد من test —
    # فتقييد الاختيار بها لا يُدخل تحيّزاً، ويمنع اختيار إعداد لا يمكن قياسه على الكون المطلوب.
    t_n = grid[grid["split"] == "test"].set_index(["side", "q"])["n_periods"]
    feasible = {key for key, n in t_n.items() if n >= 0.5 * max(t_n.max(), 1)}
    val = grid[(grid["split"] == "val") & (grid["mean_pct"] > 0)
               & np.array([(sd, qq) in feasible for sd, qq in zip(grid["side"], grid["q"])])]
    chosen = None
    if len(val):
        best = val.loc[val["sharpe"].idxmax()]
        chosen = {"side": best["side"], "q": float(best["q"])}
        t = grid[(grid["split"] == "test") & (grid["side"] == chosen["side"]) & (grid["q"] == chosen["q"])].iloc[0]
        chosen.update({"val": best.to_dict(), "test": t.to_dict()})
        lo, hi = t["mean_lo"], t["mean_hi"]
        chosen["verdict"] = ("✅ ربح صافٍ دالّ على test بالإعداد المختار على val" if lo > 0 else
                             "❌ خسارة صافية دالّة على test" if hi < 0 else
                             "➖ لا فرق دالّ عن الصفر على test بعد التكلفة")
    else:
        chosen = {"verdict": "❌ لا يوجد إعداد (قابل للتقييم على test) بمتوسط صافٍ موجب على val — لا شيء يُختبر"}
    extra = {part: {"ic": _rank_ic(f, min_assets), "deciles": _decile_table(f, min_assets)} for part, f in fr.items()}
    return grid, chosen, extra, series


def _gbm_scores(train_split, val_split, test_split, model_tf, max_train, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr, _ = _concat_splits(train_split, model_tf)
    base_tf = _tfs_of(model_tf)[0]            # خط أساس آخر خطوة على الفريم الأساسي وحده
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(tr["last_candles"]), size=min(max_train, len(tr["last_candles"])), replace=False))
    X = np.asarray(tr[f"X_{base_tf}"][idx, -1, :], dtype="float32")
    y = (np.asarray(tr["y"]["y_close_class"]).ravel()[idx] > 0).astype(int)
    gbm = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=seed).fit(X, y)
    out = {}
    for name, sp in (("val", val_split), ("test", test_split)):
        s, _ = _concat_splits(sp, model_tf)
        out[name] = gbm.predict_proba(np.asarray(s[f"X_{base_tf}"][:, -1, :], dtype="float32"))[:, 1]
    return out


def market_neutral_report(model, train_split, val_split, test_split, model_tf=None, score="p_up_close",
                          quantiles=(0.05, 0.1, 0.2, 0.3), sides=_MN_SIDES, cost_pct=0.08, universe=None,
                          min_assets=20, min_per_leg=5, cost_model="turnover", n_boot=1000, seed=0,
                          baseline_gbm=True, max_train=300_000, val_assets=None, val_df=None, test_df=None,
                          verbose=True):
    """يُرجع {"model": نتائج النموذج، "gbm": نتائج خط الأساس أو None}؛ كلٌّ فيه grid وchosen وic وdeciles.

    universe  : None (كل العملات) | "categories" (COINS_BY_CATEGORY) | قائمة رموز — على val وtest معاً. val مدمج في
                خط الأنابيب بلا أسماء عملات؛ تُستعاد من `dataset` (split_asset_names) أو تُمرَّر في val_assets.
    cost_pct  : تكلفة ذهاب وإياب لكل وحدة رأس مال إجمالي؛ cost_model="turnover" يدفعها على الدوران الفعلي فقط.
    min_assets: أقل عدد عملات في الفترة ليُعتدّ بالترتيب؛ min_per_leg: أقل عدد عملات في كل طرف."""
    model_tf = model_tf or _tfs_of()
    uni = _resolve_universe(universe)
    val_df = val_df if val_df is not None else collect_signals(model, val_split, model_tf)
    if (val_df["asset"] == "all").all():
        if val_assets is None and isinstance(globals().get("dataset"), dict):
            try:
                val_assets = split_asset_names(globals()["dataset"], "val")
            except Exception as e:                       # بيانات بلا asset_bounds أو وضع تقسيم آخر
                print(f"ℹ️ تعذّرت استعادة أسماء عملات val ({e})")
        if val_assets is not None and len(val_assets) == len(val_df):
            val_df = val_df.assign(asset=np.asarray(val_assets))
    val_named = not (val_df["asset"] == "all").all()
    test_df = test_df if test_df is not None else collect_signals(model, test_split, model_tf)
    if score not in val_df:
        raise KeyError(f"الدرجة {score!r} غير موجودة — المتاح: {[c for c in val_df if c.startswith(('p_up_', 'mu_', 'conf_'))]}")
    horizon_ns = pd.Timedelta(_tfs_of(model_tf)[0]).value * int(CONFIG.get("forecast_horizon", 1))
    args = (uni, quantiles, sides, cost_pct, min_assets, n_boot, seed, horizon_ns, min_per_leg, cost_model)
    results = {}
    grid, chosen, extra, series = _mn_evaluate(val_df, test_df, score, *args)
    results["model"] = {"grid": grid, "chosen": chosen, "by_split": extra, "series": series}
    results["gbm"] = None
    if baseline_gbm:
        sc = _gbm_scores(train_split, val_split, test_split, model_tf, max_train, seed)
        vd, td = val_df.assign(gbm_score=sc["val"]), test_df.assign(gbm_score=sc["test"])
        g2, c2, e2, s2 = _mn_evaluate(vd, td, "gbm_score", *args)
        results["gbm"] = {"grid": g2, "chosen": c2, "by_split": e2, "series": s2}
        # دمج بلا معاملات مُدرَّبة: متوسط الرتبة المئوية للدرجتين داخل كل فترة — لا شيء يُضبط على test، والجانب/q
        # يُختاران على val كالبقية. التقييم الأول (79 يوماً): النموذج أعلى IC والشجرة أفضل في العُشير الأدنى.
        rk = lambda d, c: d.groupby("timestamp")[c].rank(pct=True)
        vd = vd.assign(combo_score=(rk(vd, score) + rk(vd, "gbm_score")) / 2)
        td = td.assign(combo_score=(rk(td, score) + rk(td, "gbm_score")) / 2)
        g3, c3, e3, s3 = _mn_evaluate(vd, td, "combo_score", *args)
        results["combo"] = {"grid": g3, "chosen": c3, "by_split": e3, "series": s3}

    if verbose:
        n_days = results["model"]["by_split"]["test"]["ic"]["n"]
        print("\n" + "═" * 110 + f"\n💼 محفظة محايدة للسوق — الدرجة: {score} | تكلفة {cost_pct}% ذهاباً وإياباً "
              + ("على الدوران الفعلي" if cost_model == "turnover" else "على كل المراكز كل فترة")
              + (f" | الكون: {len(uni)} رمزاً" + ("" if val_named else " (test فقط — val بلا أسماء عملات)") if uni
                 else " | كل العملات") + "\n" + "═" * 110)
        for name, res in (("النموذج", results["model"]), ("laststep_gbm", results["gbm"]),
                          ("النموذج+الشجرة (متوسط الرتب)", results.get("combo"))):
            if res is None:
                continue
            ic_v, ic_t = res["by_split"]["val"]["ic"], res["by_split"]["test"]["ic"]
            print(f"\n[{name}] IC اليومي (Spearman بين الدرجة والعائد): val {ic_v['ic_mean']:+.4f} (t={ic_v['ic_t']:+.2f}) | "
                  f"test {ic_t['ic_mean']:+.4f} (t={ic_t['ic_t']:+.2f}, {ic_t['n']} فترة)")
            c = res["chosen"]
            if "side" in c:
                v, t = c["val"], c["test"]
                print(f"[{name}] المختار على val: {c['side']} q={c['q']:g} → val: متوسط {v['mean_pct']:+.3f}%/فترة، "
                      f"Sharpe {v['sharpe']:+.2f}")
                print(f"[{name}] على test: متوسط {t['mean_pct']:+.3f}%/فترة [{t['mean_lo']:+.3f}, {t['mean_hi']:+.3f}] | "
                      f"Sharpe {t['sharpe']:+.2f} [{t['sharpe_lo']:+.2f}, {t['sharpe_hi']:+.2f}] | "
                      f"إجمالي {t['total_pct']:+.1f}% | فترات رابحة {t['pos_periods']:.0%} | {t['n_periods']} فترة | "
                      f"دوران {t['turnover']:.0%}/فترة")
            print(f"[{name}] {c['verdict']}")
        dec = results["model"]["by_split"]["test"]["deciles"]
        if len(dec):
            print("\n   العائد النسبي لكل عُشر من درجة النموذج على test (%/فترة، 1 = الأدنى، 10 = الأعلى) — المطلوب تصاعد:")
            print("   " + " | ".join(f"{int(b)}: {r:+.3f}" for b, r in dec["rel_ret_pct"].items()))
        print("\n   الشبكة كاملة (الاختيار على val فقط — أفضل خلية على test ليست نتيجة):")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
            g = results["model"]["grid"].pivot_table(index=["side", "q"], columns="split",
                                                     values=["mean_pct", "sharpe"]).round(3)
            print(g.to_string())
        if n_days < 250:
            print(f"\n   ⚠️ {n_days} فترة فقط على test — فاصل Sharpe السنوي عريض (±~{2 * np.sqrt(365 / max(n_days, 1)):.1f}): "
                  "غياب الدلالة لا يعني غياب الإشارة، ودلالتها تحتاج فترات أطول")
    return results


# الاستخدام (بعد تحميل نموذج؛ الأنسب نموذج مدرَّب على TARGET_MODE="relative"):
#   mn = market_neutral_report(model, train, val, test)
#   mn = market_neutral_report(model, train, val, test, universe="categories")   # عملات COINS_BY_CATEGORY فقط
#   mn = market_neutral_report(model, train, val, test, cost_pct=0.15)           # تكلفة أعلى

"""
PURPOSE:  Section 7-b: selective evaluation (accuracy at confident slices, 2:1 trades): collect_signals, thresholds picked on val only, Wilson intervals, rr_trading_report.
TAGS:     collect_signals, selective_evaluation, selective_direction_report, rr_trading_report, simulate_rr_trades, wilson_interval, pick_threshold, confidence
PITFALLS: Every threshold is chosen on val and judged on test. Uses LAST_COLUMNS from the pipeline. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 26 (section 7-b).
"""
import numpy as np
import pandas as pd

_LC = {name: i for i, name in enumerate(LAST_COLUMNS)}


def _concat_splits(split_or_dict, model_tf):
    """split واحد (train/val) أو قاموس {أصل: split} (test) ← split واحد + عمود asset."""
    if "y" in split_or_dict:
        n = len(split_or_dict["last_candles"])
        return split_or_dict, np.array(["all"] * n)
    parts = list(split_or_dict.items())
    merged = {
        **{f"X_{tf}": np.concatenate([s[f"X_{tf}"] for _, s in parts]) for tf in _tfs_of(model_tf)},
        "last_candles": np.concatenate([s["last_candles"] for _, s in parts]),
        "y": {k: np.concatenate([s["y"][k] for _, s in parts]) for k in parts[0][1]["y"]},
    }
    assets = np.concatenate([np.array([a] * len(s["last_candles"])) for a, s in parts])
    return merged, assets


def collect_signals(model, split_or_dict, model_tf, batch_size=1024, outputs=None, price_targets=None):
    """مخرجات النموذج + الحقيقة الفعلية للشمعة التالية في DataFrame واحد (صف لكل عيّنة).
    ``outputs``: مخرجات جاهزة (dict) بدل استدعاء النموذج — للاختبار فقط. ``price_targets``: الأهداف المفعّلة (ModelPlan.price_targets)."""
    price_targets = _required(price_targets, "price_targets")
    split, assets = _concat_splits(split_or_dict, model_tf)
    if outputs is None:
        outputs = model.predict(model_x(split, model_tf), batch_size=batch_size, verbose=0)
    o = {k: np.asarray(v, dtype="float64").reshape(len(assets), -1)[:, 0] for k, v in outputs.items()}
    mode = target_mode_of(split_or_dict)
    lc = np.asarray(split["last_candles"], dtype="float64")
    df = pd.DataFrame({
        "asset": assets,
        "timestamp": lc[:, _LC["timestamp"]],
        "entry": lc[:, _LC["last_close"]],
        "last_high": lc[:, _LC["last_high"]], "last_low": lc[:, _LC["last_low"]],
        "fut_close": lc[:, _LC["future_close"]],
        "fut_high": lc[:, _LC["future_high_max"]], "fut_low": lc[:, _LC["future_low_min"]],
    })
    for t in price_targets:
        if f"y_{t}" not in o:
            continue
        nu, alpha, beta = o[f"y_{t}_nu"], o[f"y_{t}_alpha"], o[f"y_{t}_beta"]
        # مخرجات النموذج بوحدة الهدف (عائد × المقياس) — mu_/wst_ أدناه عائد؛ مقياس كل هدف مختوم (close في entry_range = 1)
        scale = reg_scale_of(split_or_dict, t)
        df[f"mu_{t}"] = o[f"y_{t}"] / scale
        df[f"wst_{t}"] = np.sqrt(beta * (1.0 + nu) / (nu * alpha)) / scale   # عرض Student-t (بوحدات العائد)
        if f"y_{t}_confidence" in o:
            df[f"conf_{t}"] = o[f"y_{t}_confidence"]
        if f"y_{t}_class_logits" in o:
            df[f"p_up_{t}"] = o[f"y_{t}_class_logits"]           # احتمال بعد sigmoid
    if mode == "entry_range":
        # mu_* مقادير غير سالبة من سعر الدخول (mu_close موقع [0,1] مع range_pos)؛ اتجاه close من p_up_close (القسم ٣-ب)
        close_reg = entry_close_reg_of(split_or_dict)
        px = entry_range_to_prices(df["entry"].to_numpy(), close_reg=close_reg,
                                   p_close_up=df["p_up_close"].to_numpy() if "p_up_close" in df else None,
                                   **{t: df[f"mu_{t}"].to_numpy() for t in ("high", "low", "close") if f"mu_{t}" in df})
        df["pred_high"], df["pred_low"] = px.get("high", np.nan), px.get("low", np.nan)
        for k in ("close", "close_up", "close_down"):
            if k in px:
                df[f"pred_{k}"] = px[k]
        df["entry_close_reg"] = close_reg   # ختم التعريف: المستهلك لا يخمّن وحدة mu_close ولا طريقة اتجاهه
    else:
        df["pred_high"] = df["last_high"] * (1.0 + df["mu_high"]) if "mu_high" in df else np.nan
        df["pred_low"] = df["last_low"] * (1.0 + df["mu_low"]) if "mu_low" in df else np.nan
    df["up"] = (df["fut_close"] > df["entry"]).astype(int)
    df["target_mode"] = mode or "return"   # ختم الوضع: مستهلكو ملف الإشارات (report.train_labels) لا يخمّنونه
    if mode == "entry_range":
        # تسميات التدريب نفسها — report.summarize يقيس AUC عليها
        for t in price_targets:
            if f"y_{t}_class" in split["y"]:
                df[f"y_{t}_class"] = (np.asarray(split["y"][f"y_{t}_class"]).ravel() > 0).astype(int)
    return df


def direction_scores(df, target="close"):
    """مرشّحات (اتجاه، درجة ثقة) — الأعلى درجة = الأكثر ثقة. كلها تتنبأ باتجاه الإغلاق بعد الأفق."""
    out = {}
    if f"p_up_{target}" in df:
        p = df[f"p_up_{target}"].to_numpy()
        out["class_margin"] = (np.where(p >= 0.5, 1, -1), np.abs(p - 0.5))
    if f"mu_{target}" in df:
        mu, wst = df[f"mu_{target}"].to_numpy(), df[f"wst_{target}"].to_numpy()
        side = np.where(mu >= 0, 1, -1)
        out["nig_edge"] = (side, np.abs(mu) / (wst + 1e-12))       # |تحرك متوقع| / عدم اليقين
        if f"conf_{target}" in df:
            out["conf_head"] = (side, df[f"conf_{target}"].to_numpy())
        if "class_margin" in out:
            agree = out["class_margin"][0] == side
            out["agree_margin"] = (side, np.where(agree, out["class_margin"][1], -np.inf))
    return out


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def selective_curve(score, correct, n_bins=10):
    """دقة كل شريحة ثقة (عُشر) + ارتباط سبيرمان بين رتبة الشريحة ودقتها (1.0 = اتساق تام)."""
    ok = np.isfinite(score)
    s, c = score[ok], correct[ok].astype(float)
    if len(s) < n_bins:
        return pd.DataFrame(), np.nan
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    rows = [{"bin": i + 1, "n": int((b == i).sum()), "accuracy": c[b == i].mean()}
            for i in range(len(edges) - 1) if (b == i).any()]
    table = pd.DataFrame(rows)
    rho = table["bin"].corr(table["accuracy"], method="spearman") if len(table) > 2 else np.nan
    return table, rho


def pick_threshold(score, correct, target_acc=0.65, min_n=200):
    """أكبر تغطية على val تبلغ فيها دقة الشريحة الأعلى ثقة target_acc (مع ≥ min_n عيّنة).
    يُرجع العتبة أو None إن لم تبلغ أي شريحة الهدف."""
    ok = np.isfinite(score)
    s, c = score[ok], correct[ok].astype(float)
    order = np.argsort(-s, kind="stable")
    cum_acc = np.cumsum(c[order]) / np.arange(1, len(s) + 1)
    k_ok = np.where((cum_acc >= target_acc) & (np.arange(1, len(s) + 1) >= min_n))[0]
    return None if len(k_ok) == 0 else float(s[order][k_ok[-1]])


def _subset_stats(df, mask, correct):
    n, k = int(mask.sum()), int(correct[mask].sum())
    lo, hi = wilson_interval(k, n)
    return {"n": n, "n_dates": int(df.loc[mask, "timestamp"].nunique()),
            "coverage": n / max(len(df), 1), "accuracy": k / n if n else np.nan,
            "wilson_lo": lo, "wilson_hi": hi}


def selective_direction_report(val_df, test_df, target_acc=0.65, min_n=200, target="close", verbose=True):
    """لكل مرشّح ثقة: اتساق (دقة ترتفع مع الثقة؟) على val وtest، ثم عتبة من val تُقاس على test."""
    v_scores, t_scores = direction_scores(val_df, target), direction_scores(test_df, target)
    rows = []
    for name in v_scores:
        v_side, v_s = v_scores[name]
        t_side, t_s = t_scores[name]
        v_ok = (v_side == np.where(val_df["up"] == 1, 1, -1)).astype(int)
        t_ok = (t_side == np.where(test_df["up"] == 1, 1, -1)).astype(int)
        _, rho_v = selective_curve(v_s, v_ok)
        _, rho_t = selective_curve(t_s, t_ok)
        thr = pick_threshold(v_s, v_ok, target_acc, min_n)
        row = {"score": name, "acc_all_test": t_ok.mean(), "monotonic_val": rho_v, "monotonic_test": rho_t,
               "threshold": thr}
        if thr is not None:
            vs = _subset_stats(val_df, v_s >= thr, v_ok)
            ts = _subset_stats(test_df, t_s >= thr, t_ok)
            row.update({"val_acc": vs["accuracy"], "val_n": vs["n"],
                        "test_acc": ts["accuracy"], "test_n": ts["n"], "test_dates": ts["n_dates"],
                        "test_coverage": ts["coverage"], "test_wilson_lo": ts["wilson_lo"],
                        "test_wilson_hi": ts["wilson_hi"]})
            passed = ts["n"] >= min_n and ts["accuracy"] >= target_acc and ts["wilson_lo"] > 0.5
            row["verdict"] = "✅ صمد على test" if passed else "❌ لم يصمد على test"
        else:
            row["verdict"] = f"❌ لا شريحة على val بلغت {target_acc:.0%}"
        rows.append(row)
    report = pd.DataFrame(rows)
    if verbose:
        print(f"\n🎯 دقة الاتجاه الانتقائية (الهدف ≥ {target_acc:.0%} على test، العتبة من val فقط)")
        print("   monotonic_* = سبيرمان بين شريحة الثقة ودقتها: قرب +1 اتساق، قرب 0 الثقة لا تعني شيئاً")
        print("   ⚠️ فاصل Wilson يفترض عيّنات مستقلة؛ العملات المترابطة في نفس اليوم ليست كذلك —"
              " انظر test_dates (عدد الأيام الفعلية) لا test_n وحده")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print(report.to_string(index=False))
    return report


def simulate_rr_trades(df, side, rr=2.0, stop_mode="nig", stop_k=1.0, cost_pct=0.08):
    """صفقة لكل صف: دخول عند last_close، وقف على مسافة d، هدف على rr·d. يُقيَّم بقمة/قاع/إغلاق
    الأفق الفعلي (forecast_horizon شمعة؛ يوم واحد في الإعداد اليومي).

    stop_mode:
      'nig'          → d = stop_k × عرض Student-t لـ close (عدم يقين النموذج نفسه)
      'model_levels' → d = المسافة للقاع المتوقَّع (شراء) / للقمة المتوقَّعة (بيع)
    ⚠️ شمعة يومية لا تخبر أيّهما لُمس أولاً: لمس الهدف والوقف معاً يُحسب خسارة (افتراض محافظ).
    بلا لمس لأيّهما: خروج عند إغلاق الغد. cost_pct: تكلفة الذهاب والإياب % (افتراضي عمولة+انزلاق
    خط الأنابيب 0.06+0.02). يُرجع R لكل صفقة (بعد التكلفة) ونوع الخروج."""
    entry = df["entry"].to_numpy()
    if stop_mode == "nig":
        d = stop_k * df["wst_close"].to_numpy()
    elif stop_mode == "model_levels":
        d = np.where(side > 0, (entry - df["pred_low"].to_numpy()) / entry,
                     (df["pred_high"].to_numpy() - entry) / entry)
    else:
        raise ValueError("stop_mode: 'nig' | 'model_levels'")
    valid = np.isfinite(d) & (d > 1e-5)
    d = np.where(valid, d, np.nan)
    tp = entry * (1 + side * rr * d)
    sl = entry * (1 - side * d)
    hi, lo, cl = df["fut_high"].to_numpy(), df["fut_low"].to_numpy(), df["fut_close"].to_numpy()
    hit_tp = np.where(side > 0, hi >= tp, lo <= tp)
    hit_sl = np.where(side > 0, lo <= sl, hi >= sl)
    r = np.where(hit_sl, -1.0, np.where(hit_tp, rr, side * (cl - entry) / entry / d))
    r = r - (cost_pct / 100.0) / d
    exit_kind = np.where(hit_sl & hit_tp, "both", np.where(hit_sl, "sl", np.where(hit_tp, "tp", "close")))
    return pd.DataFrame({"r": np.where(valid, r, np.nan), "exit": np.where(valid, exit_kind, "invalid"),
                         "timestamp": df["timestamp"].to_numpy()})


def _trade_stats(tr):
    t = tr[tr["exit"] != "invalid"]
    n = len(t)
    if n == 0:
        return {"n": 0}
    # t-stat على مستوى الأيام: صفقات نفس اليوم مترابطة (العملات تتحرك معاً)، فالعيّنة المستقلة هي
    # اليوم لا الصفقة — متوسط R لكل يوم، ثم متوسط/خطأ معياري عبر الأيام.
    daily = t.groupby("timestamp")["r"].mean()
    t_days = float(daily.mean() / (daily.std(ddof=1) / np.sqrt(len(daily)))) if len(daily) > 2 and daily.std() > 0 else 0.0
    return {"n": n, "n_dates": int(t["timestamp"].nunique()),
            "win_rate": (t["exit"] == "tp").mean(), "both_hit": (t["exit"] == "both").mean(),
            "expectancy_R": t["r"].mean(), "total_R": t["r"].sum(), "t_days": t_days}


def rr_trading_report(val_df, test_df, rr=2.0, stop_mode="nig", stop_k=1.0, cost_pct=0.08,
                      min_trades=200, quantiles=(0.5, 0.7, 0.8, 0.9, 0.95), target="close",
                      seed=0, verbose=True):
    """لكل مرشّح ثقة: عتبة تُختار على val (أعلى متوسط R صافٍ بين عدّة شرائح)، ثم تُقاس على test.
    خط الأساس: نفس المستويات والعيّنات باتجاه عشوائي — الفرق عنه هو الميزة الفعلية.
    النجاح يتطلّب: ربحاً على val (وإلا فالعتبة المختارة لم تجد ميزة)، وربحاً على test بدلالة t ≥ 2
    محسوبة على متوسطات الأيام (test_t_days) — لا على عدد الصفقات، لأن صفقات اليوم الواحد مترابطة.
    (نسبة الربح النظرية لمشي عشوائي عند rr=2 ≈ 1/3، فالربح 40% لا يكفي وحده دليلاً.)"""
    rng = np.random.default_rng(seed)
    v_scores, t_scores = direction_scores(val_df, target), direction_scores(test_df, target)
    rows = []
    for name in v_scores:
        v_side, v_s = v_scores[name]
        t_side, t_s = t_scores[name]
        v_tr = simulate_rr_trades(val_df, v_side, rr, stop_mode, stop_k, cost_pct)
        best = None
        for q in quantiles:
            thr = float(np.nanquantile(v_s[np.isfinite(v_s)], q))
            st = _trade_stats(v_tr[v_s >= thr])
            if st["n"] >= min_trades and (best is None or st["expectancy_R"] > best[1]["expectancy_R"]):
                best = (thr, st)
        if best is None:
            rows.append({"score": name, "verdict": f"❌ أقل من {min_trades} صفقة على val"})
            continue
        thr, vst = best
        sel = t_s >= thr
        tst = _trade_stats(simulate_rr_trades(test_df[sel], t_side[sel], rr, stop_mode, stop_k, cost_pct))
        rand_side = rng.choice([-1, 1], size=int(sel.sum()))
        base = _trade_stats(simulate_rr_trades(test_df[sel], rand_side, rr, stop_mode, stop_k, cost_pct))
        # ✅ يتطلّب ربحاً على val أيضاً (العتبة اختيرت منه — إن كانت أفضل شريحة خاسرة على val فلا ميزة
        # اكتُشفت، وأي ربح على test صدفة) + دلالة على مستوى الأيام (t ≥ 2) لا على عدد الصفقات.
        passed = tst.get("n", 0) >= min_trades and vst["expectancy_R"] > 0 and tst["expectancy_R"] > 0 \
            and tst.get("t_days", 0.0) >= 2.0 and tst["expectancy_R"] > base.get("expectancy_R", np.inf)
        rows.append({"score": name, "threshold": thr,
                     "val_n": vst["n"], "val_win": vst["win_rate"], "val_exp_R": vst["expectancy_R"],
                     "test_n": tst.get("n", 0), "test_dates": tst.get("n_dates", 0),
                     "test_win": tst.get("win_rate"), "test_both_hit": tst.get("both_hit"),
                     "test_exp_R": tst.get("expectancy_R"), "test_t_days": tst.get("t_days"),
                     "random_dir_win": base.get("win_rate"), "random_dir_exp_R": base.get("expectancy_R"),
                     "verdict": ("✅ ربح على val وtest، دالّ على مستوى الأيام، وأفضل من اتجاه عشوائي" if passed
                                 else "❌ لم يصمد (يلزم: val_exp_R>0 و test_exp_R>0 و test_t_days≥2)")})
    report = pd.DataFrame(rows)
    if verbose:
        print(f"\n💹 صفقات {rr:g}:1 (وقف={stop_mode}, تكلفة {cost_pct}% ذهاباً وإياباً، العتبة من val فقط)")
        print("   win = نسبة بلوغ الهدف فعلاً؛ exp_R = متوسط الربح بوحدات المخاطرة بعد التكلفة (> 0 = مربح)؛ both_hit = لمس الهدف والوقف معاً، حُسب خسارة")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
            print(report.to_string(index=False))
    return report


def selective_evaluation(model=None, val_split=None, test_split=None, model_tf=None, target_acc=0.65,
                         rr=2.0, stop_mode="nig", min_n=200, val_df=None, test_df=None, verbose=True, price_targets=None):
    """نقطة دخول واحدة: يجمع إشارات val/test ثم يُخرج التقريرين. مرّر val_df/test_df جاهزين لتفادي
    إعادة التنبؤ عند تجربة إعدادات صفقات مختلفة. وضع الهدف من ختم test (أو من عمود target_mode في test_df)."""
    mode = (target_mode_of(test_split) if test_split is not None
            else (test_df["target_mode"].iloc[0] if test_df is not None and len(test_df) else None))
    if mode not in (None, "return"):   # القسم ٣-ب
        print(f"⏭️ ٧-ب يفترض أهداف 'return' (اتجاه وصفقات بأسعار حقيقية) — تخطٍّ مع TARGET_MODE={mode!r}")
        return None
    model_tf = _required(model_tf, "model_tf")
    val_df = val_df if val_df is not None else collect_signals(model, val_split, model_tf, price_targets=price_targets)
    test_df = test_df if test_df is not None else collect_signals(model, test_split, model_tf, price_targets=price_targets)
    return {
        "val_df": val_df, "test_df": test_df,
        "direction": selective_direction_report(val_df, test_df, target_acc=target_acc, min_n=min_n,
                                                verbose=verbose),
        "trades": rr_trading_report(val_df, test_df, rr=rr, stop_mode=stop_mode, min_trades=min_n,
                                    verbose=verbose),
    }


# الاستخدام (بعد التدريب):
#   sel = selective_evaluation(model, val, test)
#   rr_trading_report(sel["val_df"], sel["test_df"], rr=2.0, stop_mode="model_levels")   # بلا إعادة تنبؤ

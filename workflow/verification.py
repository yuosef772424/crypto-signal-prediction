"""
PURPOSE:  Section 7-d: one-call integrated verification (data integrity, naive baselines, confidence/trading, candle shape) with a single verdict table.
TAGS:     run_full_verification, verification table, label checks, finite checks, split order, day bootstrap
PITFALLS: All thresholds chosen on val/train only; verdict from test. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 30 (section 7-d).
"""
import json
import os
import numpy as np
import pandas as pd


def _day_bootstrap_mean(values, days, n_boot=500, seed=0):
    """متوسط قيمة لكل عيّنة، بفاصل 95% بإعادة سحب الأيام (كل يوم بكل عيّناته)."""
    values = np.asarray(values, dtype="float64")
    uniq, inv = np.unique(days, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv).astype("float64")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _all_finite(X, chunk=20000):
    return all(np.isfinite(X[i:i + chunk]).all() for i in range(0, len(X), chunk))


def _split_ts(split_or_dict):
    if "y" in split_or_dict:
        return np.asarray(split_or_dict["last_candles"])[:, _LC["timestamp"]]
    return np.concatenate([np.asarray(s["last_candles"])[:, _LC["timestamp"]] for s in split_or_dict.values()])


def _label_checks(split_or_dict, model_tf, price_targets):
    if target_mode_of(split_or_dict) not in (None, "return"):   # أهداف القسم ٣-ب لا تُقارَن بعائد خام
        return {}
    split, _ = _concat_splits(split_or_dict, model_tf)
    lc = np.asarray(split["last_candles"], dtype="float64")
    last = {"high": lc[:, _LC["last_high"]], "low": lc[:, _LC["last_low"]], "close": lc[:, _LC["last_close"]]}
    fut = {"high": lc[:, _LC["future_high_max"]], "low": lc[:, _LC["future_low_min"]],
           "close": lc[:, _LC["future_close"]]}
    out = {}
    for t in price_targets:
        if f"y_{t}_class" in split["y"]:
            y = np.asarray(split["y"][f"y_{t}_class"]).ravel() > 0
            out[f"{t}_class"] = float(np.mean(y == (fut[t] > last[t])))
        if f"y_{t}_reg" in split["y"]:
            y = np.asarray(split["y"][f"y_{t}_reg"], dtype="float64").ravel()
            raw = np.clip(fut[t] / last[t] - 1.0, -1.0, 1.0) * reg_scale_of(split_or_dict)   # قصّ ثم ضرب كخط الأنابيب
            out[f"{t}_reg"] = float(np.mean(np.abs(y - raw) <= 1e-4 + 1e-3 * np.abs(raw)))
    return out


def run_full_verification(model, train_split, val_split, test_split, model_tf=None, target_acc=0.65,
                          rr=2.0, stop_mode="nig", n_boot=500, out_dir=None, run_candle=True,
                          verbose=True, price_targets=None):
    """يشغّل كل الفحوص ويُرجع {"summary": جدول الحكم، "details": كل التقارير الفرعية}.
    model_tf (فريم أو قائمة فريمات = DatasetInfo.model_tfs) وprice_targets (= ModelPlan.price_targets) صريحان — لا قيمة افتراضية."""
    model_tf = _required(model_tf, "model_tf")
    price_targets = _required(price_targets, "price_targets")
    tfs = _tfs_of(model_tf)
    base_tf = tfs[0]
    rows, details = [], {}

    def add(section, check, value, verdict, note=""):
        rows.append({"section": section, "check": check, "value": value, "verdict": verdict, "note": note})

    # ── أ) سلامة البيانات ─────────────────────────────────────────────────
    for name, sp in (("train", train_split), ("val", val_split), ("test", test_split)):
        ok = all(_all_finite(sp[f"X_{tf}"]) if "y" in sp else all(_all_finite(s[f"X_{tf}"]) for s in sp.values())
                 for tf in tfs)
        add("أ) البيانات", f"X_{name} بلا NaN/Inf", ok, "✅" if ok else "❌")
    for name, sp in (("val", val_split), ("test", test_split)):
        chk = _label_checks(sp, model_tf, price_targets)
        if not chk:
            add("أ) البيانات", f"الأهداف تطابق الأسعار الخام ({name})", "—", "ℹ️",
                f"تخطٍّ: TARGET_MODE={target_mode_of(sp)!r} (فحصه الذاتي في القسم ٣-ب)")
            continue
        worst = min(chk.values())
        add("أ) البيانات", f"الأهداف تطابق الأسعار الخام ({name})", round(worst, 4),
            "✅" if worst >= 0.99 else "❌", json.dumps({k: round(v, 4) for k, v in chk.items()}))
    ts = {n: _split_ts(sp) for n, sp in (("train", train_split), ("val", val_split), ("test", test_split))}
    step = pd.Timedelta(base_tf).value
    # أوسع نافذة بشموع الفريم الأساسي (4h × 32 = 128 شمعة 1h): مُدخل val/test بأي فريم لا يجوز أن يرى هدفاً من القسم السابق
    window = max(int(train_split[f"X_{tf}"].shape[1]) * int(pd.Timedelta(tf) // pd.Timedelta(base_tf)) for tf in tfs)
    horizon = int(CONFIG.get("forecast_horizon", 1))
    for a, b in (("train", "val"), ("val", "test")):
        gap = (ts[b].min() - ts[a].max()) / step
        add("أ) البيانات", f"{a} قبل {b} زمنياً (فجوة بالشموع)", round(float(gap), 1),
            "✅" if gap >= window + horizon else ("⚠️" if gap > 0 else "❌"),
            f"المطلوب ≥ نافذة+أفق = {window + horizon}")
    for t in price_targets:
        k = f"y_{t}_class"
        if k in train_split["y"]:
            p = float(np.mean(np.asarray(train_split["y"][k]) > 0))
            add("أ) البيانات", f"نسبة الصعود في train ({t})", round(p, 4), "ℹ️", "خط أساس الفئة الأكبر = max(p, 1−p)")

    # ── ب) خطوط الأساس على test ────────────────────────────────────────────
    val_df = collect_signals(model, val_split, model_tf, price_targets=price_targets)
    test_df = collect_signals(model, test_split, model_tf, price_targets=price_targets)
    te_split, _ = _concat_splits(test_split, model_tf)
    days = test_df["timestamp"].values
    add("ب) خطوط الأساس", "أيام test المستقلة", int(len(np.unique(days))), "ℹ️",
        "كل الأحكام أدناه على هذه الأيام — أقل من ~250 يوماً يعني أحكاماً ضعيفة إحصائياً")
    for t in price_targets:
        if f"p_up_{t}" not in test_df or f"y_{t}_class" not in train_split["y"]:
            continue
        y = (np.asarray(te_split["y"][f"y_{t}_class"]).ravel() > 0).astype(int)   # نفس هدف التدريب أياً كان وضعه
        maj = int(np.mean(np.asarray(train_split["y"][f"y_{t}_class"]) > 0) >= 0.5)   # فئة train الأكبر
        pred = (test_df[f"p_up_{t}"].values >= 0.5).astype(int)
        m, lo, hi = _day_bootstrap_mean((pred == y).astype(float) - (maj == y).astype(float), days, n_boot)
        add("ب) خطوط الأساس", f"دقة اتجاه {t} − الفئة الأكبر (test)", round(m, 4),
            "✅" if lo > 0 else ("❌" if hi < 0 else "➖"),
            f"دقة النموذج {np.mean(pred == y):.4f} | الفئة الأكبر {np.mean(maj == y):.4f} | فاصل [{lo:+.4f}, {hi:+.4f}]")
    for t in price_targets:
        if f"y_{t}_reg" not in te_split["y"] or f"mu_{t}" not in test_df:
            continue
        y = np.asarray(te_split["y"][f"y_{t}_reg"], dtype="float64").ravel() / reg_scale_of(test_split, t)   # وحدة mu_
        # ثابت خط الأساس = وسيط train (أفضل ثابت لـ MAE): ≈ 0 لأهداف العائد، وموجب لـ magnitude حيث الصفر أساس تافه
        const = float(np.median(np.asarray(train_split["y"][f"y_{t}_reg"], dtype="float64"))) / reg_scale_of(train_split, t)
        gain = np.abs(y - const) - np.abs(y - test_df[f"mu_{t}"].values)     # > 0 = النموذج أقرب من الثابت
        m, lo, hi = _day_bootstrap_mean(gain, days, n_boot)
        add("ب) خطوط الأساس", f"MAE الثابت − MAE النموذج ({t}_reg، test)", round(m, 5),
            "✅" if lo > 0 else ("❌" if hi < 0 else "➖"),
            f"MAE النموذج {np.mean(np.abs(y - test_df[f'mu_{t}'].values)):.5f} | الثابت ({const:+.5f})"
            f" {np.mean(np.abs(y - const)):.5f} | فاصل [{lo:+.5f}, {hi:+.5f}]")

    # ── ج) الثقة والتداول ─────────────────────────────────────────────────
    if target_mode_of(test_split) not in (None, "return"):
        add("ج) الثقة والتداول", "تخطٍّ", "—", "ℹ️", f"يفترض أهداف 'return' — TARGET_MODE={target_mode_of(test_split)!r}")
        sel = trd = pd.DataFrame()
    else:
        sel = selective_direction_report(val_df, test_df, target_acc=target_acc, verbose=verbose)
        trd = rr_trading_report(val_df, test_df, rr=rr, stop_mode=stop_mode, verbose=verbose)
    details.update({"selective": sel, "trades": trd, "val_df": val_df, "test_df": test_df})
    for _, r in sel.iterrows():
        cons = (r["monotonic_val"] > 0.5) and (r["monotonic_test"] > 0.5)
        add("ج) الثقة", f"اتساق الثقة ({r['score']})", f"val {r['monotonic_val']:+.2f} / test {r['monotonic_test']:+.2f}",
            "✅" if cons else "❌", "المطلوب > +0.5 على الفترتين معاً")
        add("ج) الثقة", f"شريحة ≥ {target_acc:.0%} ({r['score']})", r.get("test_acc", np.nan),
            "✅" if str(r["verdict"]).startswith("✅") else "❌", str(r["verdict"]))
    for _, r in trd.iterrows():
        add("ج) التداول", f"صفقات {rr:g}:1 ({r['score']})", r.get("test_exp_R", np.nan),
            "✅" if str(r["verdict"]).startswith("✅") else "❌",
            f"val_exp_R={r.get('val_exp_R', np.nan):.3f}, t_days={r.get('test_t_days', np.nan):.2f}")

    # ── د) شكل الشمعة ─────────────────────────────────────────────────────
    if run_candle:
        cb = candle_baseline_report(model, train_split, val_split, test_split, model_tf, verbose=verbose)
        details["candle"] = cb
        for t, v in cb["verdicts"].items():
            add("د) شكل الشمعة", f"{t}_class فوق {v['best_baseline']}",
                round(v["auc_gain_when_combined"][0], 4), v["verdict"][:1], v["verdict"])

    summary = pd.DataFrame(rows)
    if verbose:
        print("\n" + "═" * 100 + "\n🧾 ملخص التحقق المتكامل\n" + "═" * 100)
        with pd.option_context("display.max_colwidth", 90, "display.width", 250):
            print(summary.to_string(index=False))
        n_ok = int((summary["verdict"] == "✅").sum())
        n_bad = int((summary["verdict"] == "❌").sum())
        print(f"\n   ✅ {n_ok} | ❌ {n_bad} | ➖/⚠️/ℹ️ {len(summary) - n_ok - n_bad}")
        trade_ok = summary[summary["section"].isin(["ج) الثقة", "ج) التداول"])]["verdict"].eq("✅").any()
        print("   الخلاصة:", "يوجد على الأقل مصدر ثقة/صفقات اجتاز test — افحصه على نوافذ زمنية أخرى قبل أي استخدام"
              if trade_ok else "لا شيء قابل للتداول اجتاز test بهذا النموذج وهذه البيانات")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(os.path.join(out_dir, "verification_summary.csv"), index=False, encoding="utf-8-sig")
        with open(os.path.join(out_dir, "verification_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary.astype(str).to_dict("records"), f, ensure_ascii=False, indent=1)
        if verbose:
            print(f"   💾 حُفظ في {out_dir}/verification_summary.csv و.json")
    return {"summary": summary, "details": details}


# الاستخدام (بعد تحميل أفضل أوزان):
#   ver = run_full_verification(model, train, val, test, out_dir="analysis_outputs")

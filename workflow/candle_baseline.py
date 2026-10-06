"""
PURPOSE:  Section 7-c: does the model add anything over the last candle's shape? Candle/last-step baselines and the day-bootstrap AUC difference.
TAGS:     candle_baseline_report, rule_cpos, candle_gbm, laststep_gbm, high_class, low_class, day bootstrap
PITFALLS: Baselines train on train only; the verdict is on test AUC difference with a day-clustered bootstrap. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 28 (section 7-c).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


def _candle_frame(split_or_dict, model_tf):
    split, assets = _concat_splits(split_or_dict, model_tf)
    lc = np.asarray(split["last_candles"], dtype="float64")
    h, l, c = lc[:, _LC["last_high"]], lc[:, _LC["last_low"]], lc[:, _LC["last_close"]]
    rng = np.maximum(h - l, 1e-12)
    df = pd.DataFrame({
        "asset": assets, "timestamp": lc[:, _LC["timestamp"]],
        "cpos": (c - l) / rng,                        # 1 = أغلق عند القمة، 0 = عند القاع
        "upper_wick": (h - c) / c, "lower_wick": (c - l) / c, "range_rel": (h - l) / c,
    })
    for t in ("high", "low"):
        y = np.asarray(split["y"][f"y_{t}_class"]).ravel()
        df[f"y_{t}"] = (y > 0).astype(int)             # ترميز خط الأنابيب ±1 → {0,1}
    # فحص تعريف الهدف من الأسعار الخام نفسها (يجب أن يطابق ~100%)
    df.attrs["label_check"] = {
        "high": float(np.mean(df["y_high"].values == (lc[:, _LC["future_high_max"]] > h))),
        "low": float(np.mean(df["y_low"].values == (lc[:, _LC["future_low_min"]] > l))),
    }
    return df, split


_CANDLE_COLS = ["cpos", "upper_wick", "lower_wick", "range_rel"]


def _day_bootstrap_auc_diff(y, s_a, s_b, days, n_boot=300, seed=0):
    """فرق AUC (a − b) مع فاصل 95% بإعادة سحب **الأيام** لا العيّنات."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(days, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(uniq))]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    point = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (np.nan, np.nan))
    return point, lo, hi


def _logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def candle_baseline_report(model, train_split, val_split, test_split, model_tf=None,
                           targets=("high", "low"), max_train=300_000, n_boot=300, seed=0,
                           batch_size=1024, min_auc_gain=0.005, verbose=True):
    """يقارن رأسَي التصنيف high/low في النموذج بخطوط أساس من شكل الشمعة. يُرجع جدولاً وحكماً لكل هدف."""
    model_tf = _required(model_tf, "model_tf")
    base_tf = _tfs_of(model_tf)[0]        # خط أساس آخر خطوة (laststep_gbm) على الفريم الأساسي وحده
    tr, tr_split = _candle_frame(train_split, model_tf)
    va, va_split = _candle_frame(val_split, model_tf)
    te, te_split = _candle_frame(test_split, model_tf)

    rng = np.random.default_rng(seed)
    tr_idx = np.sort(rng.choice(len(tr), size=min(max_train, len(tr)), replace=False))   # سقف للسرعة/الذاكرة
    X_last = {"train": np.asarray(tr_split[f"X_{base_tf}"][tr_idx, -1, :], dtype="float32"),
              "val": np.asarray(va_split[f"X_{base_tf}"][:, -1, :], dtype="float32"),
              "test": np.asarray(te_split[f"X_{base_tf}"][:, -1, :], dtype="float32")}
    out_val = model.predict(model_x(va_split, model_tf), batch_size=batch_size, verbose=0)
    out_test = model.predict(model_x(te_split, model_tf), batch_size=batch_size, verbose=0)
    frames = {"val": va, "test": te}

    rows, verdicts = [], {}
    for t in targets:
        y_tr = tr[f"y_{t}"].values[tr_idx]
        scores = {"val": {}, "test": {}}

        cpos_lr = LogisticRegression().fit(tr[["cpos"]].values[tr_idx], y_tr)
        candle = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1,
                                                random_state=seed).fit(tr[_CANDLE_COLS].values[tr_idx], y_tr)
        last = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1,
                                              random_state=seed).fit(X_last["train"], y_tr)
        for part, df in frames.items():
            scores[part]["rule_cpos"] = cpos_lr.predict_proba(df[["cpos"]].values)[:, 1]
            scores[part]["candle_gbm"] = candle.predict_proba(df[_CANDLE_COLS].values)[:, 1]
            scores[part]["laststep_gbm"] = last.predict_proba(X_last[part])[:, 1]
            out = out_val if part == "val" else out_test
            scores[part]["model_class"] = np.asarray(out[f"y_{t}_class_logits"]).ravel()
            mu = np.asarray(out[f"y_{t}"]).ravel()
            scores[part]["model_reg_mu"] = 1.0 / (1.0 + np.exp(-mu / (np.std(mu) + 1e-12)))  # sign(mu) عند 0.5

        for part, df in frames.items():
            y = df[f"y_{t}"].values
            base = max(y.mean(), 1 - y.mean())
            for name, s in scores[part].items():
                pred = (s >= 0.5).astype(int)
                rows.append({"target": t, "split": part, "score": name, "n": len(y),
                             "auc": roc_auc_score(y, s), "accuracy": float((pred == y).mean()),
                             "balanced_acc": balanced_accuracy_score(y, pred), "majority_baseline": base})

        # ── الحكم على test ──
        y_te, days = te[f"y_{t}"].values, te["timestamp"].values
        base_names = ["rule_cpos", "candle_gbm", "laststep_gbm"]
        best = max(base_names, key=lambda k: roc_auc_score(va[f"y_{t}"].values, scores["val"][k]))  # يُختار على val
        d, lo, hi = _day_bootstrap_auc_diff(y_te, scores["test"]["model_class"], scores["test"][best], days,
                                            n_boot=n_boot, seed=seed)
        # قيمة مضافة: هل تحسّن درجة النموذج أفضل خط أساس حين تُدمَجان؟ (الدمج يُتعلَّم على val فقط)
        y_va = va[f"y_{t}"].values
        Z = lambda part, with_model: np.column_stack(
            [_logit(scores[part][best])] + ([_logit(scores[part]["model_class"])] if with_model else []))
        only_base = LogisticRegression().fit(Z("val", False), y_va)
        combined = LogisticRegression().fit(Z("val", True), y_va)
        s_base = only_base.predict_proba(Z("test", False))[:, 1]
        s_comb = combined.predict_proba(Z("test", True))[:, 1]
        di, loi, hii = _day_bootstrap_auc_diff(y_te, s_comb, s_base, days, n_boot=n_boot, seed=seed + 1)

        # ✅ يتطلّب مكسباً له قيمة عملية (≥ min_auc_gain) لا مجرّد فاصل فوق الصفر: مع مئات آلاف العيّنات
        # يصبح +0.0002 AUC «دالاً» إحصائياً وهو لا شيء فعلياً.
        if loi >= min_auc_gain:
            verdict = f"✅ النموذج يضيف معلومة فوق {best} (ΔAUC عند الدمج {di:+.4f}، فاصل [{loi:+.4f}, {hii:+.4f}])"
        elif d < 0 or hi < 0:
            verdict = (f"❌ {best} يعادل النموذج أو يتفوّق عليه — النموذج لا يضيف شيئاً فوق شكل الشمعة"
                       f" (ΔAUC عند الدمج {di:+.4f} < {min_auc_gain})")
        else:
            verdict = f"➖ لا فرق دالّ عن {best} — لا دليل على قيمة مضافة"
        verdicts[t] = {"best_baseline": best, "auc_diff_vs_best": (d, lo, hi),
                       "auc_gain_when_combined": (di, loi, hii), "verdict": verdict,
                       "n_test_days": int(len(np.unique(days)))}

    table = pd.DataFrame(rows)
    if verbose:
        print("\n🕯️ رأسا high/low مقابل خطوط أساس من شكل الشمعة (كل خطوط الأساس دُرِّبت على train فقط)")
        if target_mode_of(test_split) in (None, "return"):
            print(f"   فحص تعريف الهدف من الأسعار الخام — val: {va.attrs['label_check']} | test: {te.attrs['label_check']}")
        else:   # أهداف القسم ٣-ب لا تساوي «السعر المستقبلي > آخر سعر» بالتعريف — المطابقة الجزئية متوقَّعة
            print(f"   (فحص تعريف الهدف من الأسعار الخام متخطّى: TARGET_MODE={target_mode_of(test_split)!r})")
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
            print(table.pivot_table(index=["target", "score"], columns="split",
                                    values=["auc", "accuracy"]).round(4).to_string())
            print("\n   majority_baseline:", table.groupby(["target", "split"])["majority_baseline"].first().round(4).to_dict())
        for t, v in verdicts.items():
            d, lo, hi = v["auc_diff_vs_best"]
            print(f"\n   [{t}] أفضل خط أساس (مختار على val): {v['best_baseline']} | "
                  f"AUC النموذج − الأساس على test: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | أيام test: {v['n_test_days']}")
            print(f"   [{t}] {v['verdict']}")
    return {"table": table, "verdicts": verdicts}


# الاستخدام (بعد تحميل أفضل أوزان):
#   cb = candle_baseline_report(model, train, val, test)

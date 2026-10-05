"""
PURPOSE:  Section 7-g: normalization audit before training and generalization-gap report after it (anti-memorization PR #7).
TAGS:     normalization_audit, generalization_gap_report, gap, linear reference, memorization
PITFALLS: _auc here shadows model_v2's _auc of the same name (as in the old notebook): load this module after the model_v2 %run. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 36 (section 7-g).
"""
import numpy as np
import pandas as pd


def _x_of(split_or_dict, model_tf=None):
    # فريم واحد بالاسم؛ قائمة فريمات ← أولها (الأساسي). None ← MODEL_TF: التدقيق على الفريم الأساسي، و"4h" صراحةً لسياقه.
    model_tf = _tfs_of(model_tf)[0] if model_tf else MODEL_TF
    return np.concatenate([np.asarray(s[f"X_{model_tf}"]) for _, s in _split_parts(split_or_dict)])


def _sample_rows(n, max_n, seed=0):
    if n <= max_n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))


def normalization_audit(train_split, val_split=None, test_split=None, feature_names=None, model_tf=None,
                        max_windows=20000, level_ratio_max=20.0, near_const_frac_max=0.2,
                        extreme_max=50.0, shift_ks_max=0.25, verbose=True):
    """يفحص كل ميزة في X (بعد تطبيع خط الأنابيب، أي ما يراه النموذج فعلاً) بحثاً عن أربعة أنماط تجعل الحفظ
    أسهل من التعلّم — بلا أي تدريب:

    level_ratio : تشتّت متوسّطات النوافذ ÷ وسيط التشتّت داخل النافذة. ميزة مستقرّة (عائد، مذبذب) قيمتها
                  ~1–5؛ ميزة **مستوى** غير مُطبَّع (سعر، حجم بالدولار، عمر العملة، عدّاد) قيمتها بالمئات —
                  InstanceNorm يمحو المستوى من z، لكنه يصل عبر مسار stats كرقم فريد لكل عملة/فترة = معرِّف
                  تحفظ به الشبكة «أيّ عملة في أيّ يوم» (رأس النموذج: stats_mode).
    near_const  : نسبة النوافذ التي تكاد الميزة تثبت فيها. z عندها (x−m)/√(var+eps) يُضخّم ضجيجاً ضئيلاً.
    extreme     : أقصى |x| ÷ مقياس متين (IQR). ذيل ثقيل = عيّنات قليلة تهيمن على التدرّج.
    shift_ks    : مسافة KS بين train وtest لقيمة آخر خطوة — تطبيع غير ساكن يجعل ما حُفظ في train لا يتكرّر.

    يُرجع DataFrame بعلم لكل مشكلة وتوصية. لا يعدّل البيانات."""
    from scipy.stats import ks_2samp
    Xtr = _x_of(train_split, model_tf)
    Xtr = Xtr[_sample_rows(len(Xtr), max_windows)]
    Xte = None
    if test_split is not None:
        Xte = _x_of(test_split, model_tf)
        Xte = Xte[_sample_rows(len(Xte), max_windows, 1)]
    n_f = Xtr.shape[2]
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_f)]
    rows = []
    for j in range(n_f):
        x = Xtr[:, :, j].astype("float64")
        finite = np.isfinite(x)
        xv = x[finite]
        w_mean = np.nanmean(np.where(finite, x, np.nan), axis=1)
        w_std = np.nanstd(np.where(finite, x, np.nan), axis=1)
        q25, q75 = np.nanpercentile(xv, [25, 75]) if xv.size else (0.0, 0.0)
        scale = max(q75 - q25, np.nanstd(xv) if xv.size else 0.0, 1e-12)
        med_w_std = float(np.nanmedian(w_std))
        level_ratio = float(np.nanstd(w_mean) / max(med_w_std, 1e-12 * scale, 1e-12))
        near_const = float(np.nanmean(w_std < 1e-3 * scale))
        extreme = float(np.nanmax(np.abs(xv - np.nanmedian(xv))) / scale) if xv.size else np.nan
        ks = np.nan
        if Xte is not None:
            a, b = Xtr[:, -1, j], Xte[:, -1, j]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) > 50 and len(b) > 50:
                ks = float(ks_2samp(a, b).statistic)
        flags = []
        if level_ratio > level_ratio_max:
            flags.append("مستوى غير مُطبَّع (معرِّف عملة/فترة)")
        if near_const > near_const_frac_max:
            flags.append("شبه ثابتة داخل النوافذ")
        if extreme > extreme_max:
            flags.append("ذيل متطرّف")
        if np.isfinite(ks) and ks > shift_ks_max:
            flags.append("انزياح train→test")
        rows.append({"feature": names[j], "level_ratio": level_ratio, "near_const": near_const,
                     "extreme": extreme, "shift_ks": ks, "nan_frac": float(1 - finite.mean()),
                     "flags": " | ".join(flags) or "✅"})
    df = pd.DataFrame(rows)
    if verbose:
        bad = df[df["flags"] != "✅"]
        print(f"🔎 تدقيق التطبيع: {len(df)} ميزة، {len(bad)} عليها ملاحظة")
        with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format,
                               "display.max_colwidth", 60):
            print((bad if len(bad) else df.head(0)).to_string(index=False))
        if (df["flags"].str.contains("مستوى")).any():
            print("   ⚠️ ميزات مستوى: طبّعها في خط الأنابيب (نسبة/عائد/رتبة)، أو على الأقل ANTI_MEMORIZATION=True "
                  "(stats_mode='symlog' يضغط مستواها لبضع وحدات بدل تمريره خاماً).")
        if (df["flags"].str.contains("انزياح")).any():
            print("   ⚠️ انزياح train→test: ما يتعلّمه النموذج من مستوى هذه الميزة في train لن يتكرّر على test.")
    return df


def _heads_eval(model, split_or_dict, max_n, seed, model_tf, batch_size=2048):
    parts = _split_parts(split_or_dict)
    tfs = _tfs_of(model_tf)
    X = _x_of(split_or_dict, tfs[0])
    y = {k: np.concatenate([np.asarray(s["y"][k]).ravel() for _, s in parts]) for k in parts[0][1]["y"]}
    ts = _price_columns(parts)["timestamp"]
    idx = _sample_rows(len(X), max_n, seed)
    x_in = X[idx] if len(tfs) == 1 else {tf: _x_of(split_or_dict, tf)[idx] for tf in tfs}   # قاموس لنموذج متعدّد الفريمات
    out = model.predict(x_in, batch_size=batch_size, verbose=0)
    res = {}
    for t in PRICE_TARGETS:
        if f"y_{t}_class_logits" in out and f"y_{t}_class" in y:
            yt = y[f"y_{t}_class"][idx] > 0
            p = np.asarray(out[f"y_{t}_class_logits"]).ravel()
            res[(t, "class_auc")] = _auc(yt, p)
            res[(t, "class_acc")] = float(np.mean((p >= 0.5) == yt))
        if f"y_{t}" in out and f"y_{t}_reg" in y:
            df = pd.DataFrame({"y": y[f"y_{t}_reg"][idx], "s": np.asarray(out[f"y_{t}"]).ravel(), "d": ts[idx]})
            ic = df.groupby("d").filter(lambda g: len(g) >= 5).groupby("d")[["y", "s"]].apply(
                lambda g: g["y"].rank().corr(g["s"].rank()))
            res[(t, "reg_daily_ic")] = float(ic.mean()) if len(ic) else float(
                df["y"].rank().corr(df["s"].rank()))
    return res


def _auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _linear_reference(train_split, eval_splits, model_tf, max_n, seed=0):
    """انحدار لوجستي على [آخر خطوة، متوسط النافذة] لكل ميزة — «أبسط نموذج معقول». لا يستطيع الحفظ عملياً
    (معاملات قليلة، C صغير)، فما يحقّقه على val/test هو معلومة متاحة فعلاً في الميزات بهذا الهدف."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feats = lambda X: np.concatenate([X[:, -1, :], X.mean(axis=1)], axis=1).astype("float64")
    parts = _split_parts(train_split)
    X = _x_of(train_split, model_tf)
    idx = _sample_rows(len(X), max_n * 3, seed)
    F = np.nan_to_num(feats(X[idx]))
    sc = StandardScaler().fit(F)
    out = {}
    for t in PRICE_TARGETS:
        k = f"y_{t}_class"
        if k not in parts[0][1]["y"]:
            continue
        yt = np.concatenate([np.asarray(s["y"][k]).ravel() for _, s in parts])[idx] > 0
        if yt.all() or not yt.any():
            continue
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(F), yt)
        for name, sp in eval_splits.items():
            ps = _split_parts(sp)
            Xe = _x_of(sp, model_tf)
            ye = np.concatenate([np.asarray(s["y"][k]).ravel() for _, s in ps]) > 0
            j = _sample_rows(len(Xe), max_n, 1)
            out[(t, name)] = _auc(ye[j], clf.predict_proba(sc.transform(np.nan_to_num(feats(Xe[j]))))[:, 1])
    return out


def generalization_gap_report(model, train_split, val_split, test_split, model_tf=None, max_n=20000,
                              auc_gap_max=0.03, linear_reference=True, verbose=True):
    """يقيس نفس المقاييس على train (عيّنة، بوضع الاستدلال: بلا dropout/ضجيج) وval وtest:
    AUC ودقّة رؤوس التصنيف، وIC يومي (سبيرمان داخل كل طابع زمني ثم المتوسط) لرؤوس الانحدار.

    دقّة التدريب التي يطبعها Keras أثناء fit تُحسب **بوضع التدريب** (dropout/ضجيج مفعَّلان) وكمتوسط متحرّك عبر
    الحقبة — تُخفي الحفظ. هنا تُقاس الأوزان النهائية نفسها على الأقسام الثلاثة بنفس الطريقة.

    الحكم لكل رأس تصنيف:
      train − val > auc_gap_max وval ≈ 0.5  → يحفظ: تعلّم train ولم ينقل شيئاً.
      train − val > auc_gap_max وval > 0.5  → يتعلّم ويحفظ معاً: التنظيم (ANTI_MEMORIZATION) يُرجَّح أن يرفع val.
      الفجوة صغيرة وval > 0.5              → تعلّم قابل للتعميم.
      الفجوة صغيرة وval ≈ 0.5              → لا يحفظ ولا يتعلّم: فشل صادق (لا معلومة بهذا الإعداد) — لا خلل.

    linear_reference: يضيف عمود linear_ref_{val,test} = AUC انحدار لوجستي على [آخر خطوة، متوسط النافذة].
      النموذج العميق **تحت** المرجع الخطّي على val/test = لا يلتقط معلومة متاحة فعلاً (قيس في PR #7: السبب
      المعماري أن InstanceNorm يمحو مستوى الميزة — level_passthrough). مع فجوة كبيرة = الحفظ بديل ما لم يستطع رؤيته."""
    rows = {}
    for name, sp, seed in (("train", train_split, 0), ("val", val_split, 1), ("test", test_split, 2)):
        rows[name] = _heads_eval(model, sp, max_n, seed, model_tf)
    df = pd.DataFrame(rows)
    df.index = pd.MultiIndex.from_tuples(df.index, names=["target", "metric"])
    df["gap_train_val"] = df["train"] - df["val"]
    df["gap_train_test"] = df["train"] - df["test"]
    lin = _linear_reference(train_split, {"val": val_split, "test": test_split}, model_tf, max_n) \
        if linear_reference else {}
    for name in ("val", "test"):
        df[f"linear_ref_{name}"] = [lin.get((t, name), np.nan) if m == "class_auc" else np.nan for t, m in df.index]
    verdicts = {}
    for t in PRICE_TARGETS:
        if (t, "class_auc") not in df.index:
            continue
        g, v = df.loc[(t, "class_auc"), "gap_train_val"], df.loc[(t, "class_auc"), "val"]
        learns = v > 0.5 + auc_gap_max / 3
        if g > auc_gap_max:
            verdicts[t] = "⚠️ يتعلّم ويحفظ معاً" if learns else "❌ يحفظ (train فقط)"
        else:
            verdicts[t] = "✅ تعلّم قابل للتعميم" if learns else "➖ فشل صادق: لا يحفظ ولا يجد معلومة"
        ref = lin.get((t, "val"))
        if ref is not None and np.isfinite(ref) and v < ref - 0.01:
            verdicts[t] += f" | تحت المرجع الخطّي على val ({v:.3f} < {ref:.3f}): معلومة متاحة لا يلتقطها"
    if verbose:
        print("📏 فجوة التعميم — كل الأقسام بوضع الاستدلال (بلا dropout/ضجيج)، نفس الأوزان")
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print(df.to_string())
        for t, v in verdicts.items():
            print(f"   {t}_class: {v}")
    return {"table": df, "verdicts": verdicts}

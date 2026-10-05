"""
PURPOSE:  Section 7-e: label-permutation control (same model trained on shuffled labels; global or within-day) against the real run.
TAGS:     run_label_permutation_control, permutation control, shuffled labels, within_day, leakage check
PITFALLS: A shuffled-label model beating the baseline on test globally means leakage; within_day shuffles keep market-level information. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 32 (section 7-e).
"""
import copy
import hashlib
import json
import os
import numpy as np
import pandas as pd


def _permuted_rows(n, days, mode, rng):
    if mode == "global":
        return rng.permutation(n)
    if mode == "within_day":
        by_day = np.argsort(days, kind="stable")
        shuffled = np.lexsort((rng.random(n), days))          # نفس الأيام بنفس الأحجام، ترتيب عشوائي داخلها
        perm = np.empty(n, dtype=np.int64)
        perm[by_day] = shuffled
        return perm
    raise ValueError("mode: 'global' | 'within_day'")


def _train_control_run(run_dir, X, y, val_split, epochs, seed, verbose):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)                      # نفس الأوزان الابتدائية للتشغيلين
    cfg = copy.deepcopy(main_config)
    # train_mode="auto": يستأنف من آخر حقبة محفوظة في run_dir على Drive إن وُجدت، وإلا يبدأ من جديد.
    # بلا mirror_dir كي لا تُستعاد نسخة التدريب الأساسي في هذا المجلد.
    cfg["run"].update({"run_dir": run_dir, "mirror_dir": None, "epochs": epochs, "train_mode": "auto"})
    cfg["callbacks"]["early_stopping"]["patience"] = 10 ** 6  # حقب ثابتة؛ أفضل حقبة (val_raw_loss) تُسترجع في النهاية
    cfg = build_config(cfg)
    TRAINER_REGISTRY.pop(run_dir, None)                       # الحالة من Drive لا من كائن قديم في الجلسة
    bs = cfg["run"]["batch_size"]
    train_ds = make_shuffled_dataset(X, y, bs, seed=seed)
    trainer, callbacks, initial_epoch = build_training_system(model_builder, cfg, next(iter(train_ds)))

    # دقة التدريب لكل حقبة تُحفظ بجانب الـ checkpoint، فتبقى متاحة بعد الاستئناف
    hist_path = os.path.join(run_dir, "control_history.json")
    history = {}
    if initial_epoch > 0 and os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as f:
            history = {k: v[:initial_epoch] for k, v in json.load(f).items()}
    if initial_epoch >= epochs:
        print(f"✅ مكتمل من قبل ({initial_epoch} حقب) — استُرجع من Drive بلا تدريب: {run_dir}")
        return trainer.model, history

    def _log_epoch(epoch, logs):
        for k, v in (logs or {}).items():
            history.setdefault(k, []).append(float(v))
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f)

    val_ds = make_eval_dataset(model_x(val_split), _y_for(val_split), bs)
    trainer.fit(train_ds, validation_data=val_ds, initial_epoch=initial_epoch, epochs=epochs, verbose=verbose,
                callbacks=callbacks + [tf.keras.callbacks.LambdaCallback(on_epoch_end=_log_epoch)])
    return trainer.model, history


def run_label_permutation_control(train_split, val_split, test_split, epochs=6, mode="global",
                                  max_train_samples=None, seed=0, n_boot=500,
                                  run_root="/content/drive/MyDrive/training_runs/permutation_control",
                                  run_tag=None, leak_min_acc=0.01, verbose=1):
    """يدرّب real وshuffled بنفس الإعداد ويُرجع {"summary", "history_real", "history_shuffled", "model_real",
    "model_shuffled"} — model_real يصلح مباشرة لـ candle_baseline_report / run_full_verification.

    max_train_samples: عيّنة فرعية عشوائية من train للسرعة (نفسها للتشغيلين، فالمقارنة تبقى عادلة).
    run_tag: اسم المجلد تحت run_root. الافتراضي يُشتقّ من الحقب والعيّنة والبذرة وبصمة البيانات، فإعادة الاستدعاء
             بنفس الإعداد تستأنف من Drive: ما اكتمل يُحمَّل بلا تدريب، وما انقطع يكمل من آخر حقبة محفوظة.
             real مشترك بين global وwithin_day (نفس النموذج تماماً)، فلا يُدرَّب مرتين.
    الزمن ≈ 2 × epochs × زمن الحقبة الواحدة في تدريبك العادي."""
    missing = [n for n in ("_concat_splits", "_LC", "_day_bootstrap_mean") if n not in globals()]
    if missing:   # يُفحص قبل أي تدريب، لا بعد انتهائه
        raise NameError(f"شغّل أولاً خليتي تعريف القسمين ٧-ب و٧-د — ينقص: {missing}")
    rng = np.random.default_rng(seed)
    X_full = model_x(train_split)                 # مصفوفة، أو قاموس {فريم: مصفوفة} لنموذج متعدّد الفريمات
    n_full = len(y_full_first := next(iter(_y_for(train_split).values())))
    y_full = _y_for(train_split)
    days_full = np.asarray(train_split["last_candles"])[:, _LC["timestamp"]]
    _rows = lambda X_, idx: {k: v[idx] for k, v in X_.items()} if isinstance(X_, dict) else X_[idx]
    if max_train_samples and max_train_samples < n_full:
        sub = np.sort(rng.choice(n_full, size=max_train_samples, replace=False))
        X, y, days = _rows(X_full, sub), {k: v[sub] for k, v in y_full.items()}, days_full[sub]
    else:
        X, y, days = X_full, y_full, days_full
    step = max(1, n_full // 1000)
    data_sig = repr((n_full, float(sum(np.asarray(a).sum() for a in _rows(X_full, slice(None, None, step)).values())
                                   if isinstance(X_full, dict) else np.asarray(X_full[::step]).sum()),
                     tuple(float(np.asarray(v).sum()) for v in y_full.values())))
    run_tag = run_tag or (f"e{epochs}_n{max_train_samples or 'all'}_s{seed}_"
                          f"{hashlib.md5(data_sig.encode()).hexdigest()[:8]}")
    print(f"📁 مجلد التجربة: {run_root}/{run_tag}")
    perm = _permuted_rows(len(days), days, mode, rng)
    y_shuf = {k: np.asarray(v)[perm] for k, v in y.items()}   # كل الأهداف بنفس الترتيب: يبقى ترابطها ببعضها

    runs = {}
    for tag, yy in (("real", y), (f"shuffled_{mode}", y_shuf)):
        print(f"\n{'═' * 80}\n🔀 تشغيل {tag}: {len(days):,} عيّنة، {epochs} حقب\n{'═' * 80}")
        runs[tag] = _train_control_run(f"{run_root}/{run_tag}/{tag}", X, yy, val_split, epochs, seed, verbose)
    (m_real, h_real), (m_shuf, h_shuf) = runs["real"], runs[f"shuffled_{mode}"]

    # ── المقارنة على test عيّنة بعيّنة ──
    te, _ = _concat_splits(test_split, _tfs_of())
    Xte = model_x(te)
    out_r = m_real.predict(Xte, batch_size=1024, verbose=0)
    out_s = m_shuf.predict(Xte, batch_size=1024, verbose=0)
    days_te = np.asarray(te["last_candles"])[:, _LC["timestamp"]]
    rows = []
    for t in PRICE_TARGETS:
        if f"y_{t}_class" in te["y"] and f"y_{t}_class_logits" in out_r:
            yt = (np.asarray(te["y"][f"y_{t}_class"]).ravel() > 0).astype(int)
            maj = int(np.mean(np.asarray(y[f"y_{t}_class"]).ravel() >= 0.5) >= 0.5)   # فئة train الأكبر
            cr = ((np.asarray(out_r[f"y_{t}_class_logits"]).ravel() >= 0.5).astype(int) == yt).astype(float)
            cs = ((np.asarray(out_s[f"y_{t}_class_logits"]).ravel() >= 0.5).astype(int) == yt).astype(float)
            cm = (maj == yt).astype(float)
            d_info = _day_bootstrap_mean(cr - cs, days_te, n_boot, seed)
            d_leak = _day_bootstrap_mean(cs - cm, days_te, n_boot, seed + 1)
            d_base = _day_bootstrap_mean(cr - cm, days_te, n_boot, seed + 2)
            tr_r = h_real.get(f"{t}_class_accuracy", [np.nan])[-1]
            tr_s = h_shuf.get(f"{t}_class_accuracy", [np.nan])[-1]
            rows.append({"target": f"{t}_class", "metric": "accuracy",
                         "train_real": tr_r, "train_shuffled": tr_s,
                         "test_real": cr.mean(), "test_shuffled": cs.mean(), "test_baseline": cm.mean(),
                         "real−shuffled [95%]": f"{d_info[0]:+.4f} [{d_info[1]:+.4f}, {d_info[2]:+.4f}]",
                         "real−baseline [95%]": f"{d_base[0]:+.4f} [{d_base[1]:+.4f}, {d_base[2]:+.4f}]",
                         "shuffled−baseline [95%]": f"{d_leak[0]:+.4f} [{d_leak[1]:+.4f}, {d_leak[2]:+.4f}]",
                         "_info_lo": d_info[1], "_base_lo": d_base[1], "_leak_lo": d_leak[1], "_leak_m": d_leak[0], "_mem": tr_s - max(np.mean(y[f'y_{t}_class']), 1 - np.mean(y[f'y_{t}_class']))})
        if f"y_{t}_reg" in te["y"] and f"y_{t}" in out_r:
            yt = np.asarray(te["y"][f"y_{t}_reg"], dtype="float64").ravel()
            const = float(np.median(np.asarray(y[f"y_{t}_reg"], dtype="float64")))   # وسيط train: ≈0 للعوائد
            gr = np.abs(yt - const) - np.abs(yt - np.asarray(out_r[f"y_{t}"]).ravel())  # > 0 = أقرب من الثابت
            gs = np.abs(yt - const) - np.abs(yt - np.asarray(out_s[f"y_{t}"]).ravel())
            d_info = _day_bootstrap_mean(gr - gs, days_te, n_boot, seed)
            d_leak = _day_bootstrap_mean(gs, days_te, n_boot, seed + 1)
            d_base = _day_bootstrap_mean(gr, days_te, n_boot, seed + 2)
            rows.append({"target": f"{t}_reg", "metric": "MAE(ثابت) − MAE",
                         "train_real": h_real.get(f"{t}_reg_mae", [np.nan])[-1],
                         "train_shuffled": h_shuf.get(f"{t}_reg_mae", [np.nan])[-1],
                         "test_real": gr.mean(), "test_shuffled": gs.mean(), "test_baseline": 0.0,
                         "real−shuffled [95%]": f"{d_info[0]:+.5f} [{d_info[1]:+.5f}, {d_info[2]:+.5f}]",
                         "real−baseline [95%]": f"{d_base[0]:+.5f} [{d_base[1]:+.5f}, {d_base[2]:+.5f}]",
                         "shuffled−baseline [95%]": f"{d_leak[0]:+.5f} [{d_leak[1]:+.5f}, {d_leak[2]:+.5f}]",
                         "_info_lo": d_info[1], "_base_lo": d_base[1], "_leak_lo": d_leak[1], "_leak_m": d_leak[0], "_mem": np.nan})

    # global: التسميات المخلوطة بلا أي معلومة، فتفوّقها على الأساس = تسرّب.
    # within_day: تُبقي توزيع تسميات كل يوم عمداً (اتجاه السوق)، فتفوّقها على الأساس = معلومة على مستوى السوق،
    # وreal − shuffled = ما يخصّ كل عملة فوق حركة السوق في يومها.
    day = mode == "within_day"
    for r in rows:
        v = []
        # نموذج بلا معلومة لا تساوي دقته الأساس تماماً (تنبؤاته ليست فئة واحدة): انحراف < leak_min_acc
        # يقع ضمن هذا التذبذب — فاصل الأيام لا يلتقط تباين النموذج نفسه — فلا يُحكَم عليه بالتسرّب.
        small = r["metric"] == "accuracy" and abs(r["_leak_m"]) < leak_min_acc
        if r["_leak_lo"] > 0 and not small:
            v.append("📈 معلومة على مستوى السوق (اليوم)" if day else "⚠️ تسرّب: المخلوط تفوّق على خط الأساس")
        elif r["_leak_lo"] > 0:
            v.append(f"ℹ️ المخلوط فوق الأساس بأقل من {leak_min_acc:.0%} — ضمن تذبذب نموذج بلا معلومة")
        # المعلومة تتطلّب الشرطين: real فوق shuffled (نفس الإعداد) وفوق خط الأساس الساذج. real > shuffled وحده
        # لا يكفي: قد يكون shuffled أسوأ من الأساس (يحفظ ضجيجاً فيتنبأ بالفئة الأصغر) فيبدو real أفضل منه بلا معلومة.
        if r["_info_lo"] > 0 and r["_base_lo"] > 0:
            v.append("✅ معلومة خاصة بالعملة فوق حركة السوق اليومية" if day
                     else "✅ معلومة حقيقية (real > shuffled وreal > خط الأساس على test)")
        elif r["_info_lo"] > 0:
            v.append("➖ real > shuffled لكن ليس فوق خط الأساس — shuffled أسوأ من الأساس، لا دليل على معلومة")
        else:
            v.append("➖ لا شيء خاص بالعملة فوق حركة السوق" if day else "➖ لا دليل على معلومة بهذا الإعداد")
        if r["metric"] == "accuracy" and r["_mem"] > 0.05:
            v.append("النموذج يحفظ تسميات اليوم (اتجاه السوق)" if day else "النموذج يحفظ (دقة تدريب المخلوط فوق الأساس)")
        r["verdict"] = " | ".join(v)
    summary = pd.DataFrame(rows).drop(columns=["_info_lo", "_base_lo", "_leak_lo", "_leak_m", "_mem"])
    print(f"\n{'═' * 110}\n🔀 نتيجة تجربة التحكّم (mode={mode}) — الحكم من test، وفواصل الثقة بإعادة سحب الأيام "
          f"({len(np.unique(days_te))} يوماً)\n{'═' * 110}")
    print("   للتصنيف: train_* دقة التدريب في آخر حقبة، test_* دقة test. للانحدار: train_* MAE تدريب، "
          "test_* = MAE(ثابت) − MAE (موجب = أفضل من التنبؤ بوسيط train)")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 260, "display.max_colwidth", 80):
        print(summary.to_string(index=False))
    return {"summary": summary, "history_real": h_real, "history_shuffled": h_shuf,
            "model_real": m_real, "model_shuffled": m_shuf}


# الاستخدام: استدعاء واحد في كل خلية جديدة (لا تُزِل التعليق عن الأسطر الثلاثة معاً — كل سطر تجربة كاملة).
# كل تدريب ≈ epochs × زمن حقبة. أوقفته؟ أعد نفس الاستدعاء فيكمل من Drive. real يُدرَّب مرة واحدة لكل إعداد.
#   ctrl = run_label_permutation_control(train, val, test, epochs=6)                  # كامل البيانات
#   ctrl = run_label_permutation_control(train, val, test, epochs=6, max_train_samples=100_000)  # أسرع
#   ctrl_day = run_label_permutation_control(train, val, test, epochs=6, mode="within_day")

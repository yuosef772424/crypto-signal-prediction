"""
PURPOSE:  Section 7-i: on-demand model diagnostics: training recorder, model health report, per-layer report.
TAGS:     make_training_diagnostics, model_health_report, model_layer_report, diagnostics, recorder, layer probe
PITFALLS: Optional, nothing automatic; see docs/research/model_diagnostics.md. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 40 (section 7-i).
"""
import numpy as np
import pandas as pd


def _diag_xy(split_or_dict, model_tf=None, max_n=2000, seed=0):
    """(X, y) من قسم أو قاموس عملات، مأخوذَين بعيّنة max_n **قبل** الضمّ (لا نسخ كامل لـ X): X مصفوفة (فريم واحد) أو {فريم: مصفوفة}؛
    y بمفاتيح خط الأنابيب كما هي (y_{t}_class بترميزه الخام: الدوال تقرأ >0)."""
    parts = _split_parts(split_or_dict)
    tfs = _tfs_of(model_tf)
    lens = [len(np.asarray(s["y"][next(iter(s["y"]))])) for _, s in parts]
    total = int(sum(lens))
    pick = np.arange(total) if total <= max_n else np.sort(np.random.default_rng(seed).choice(total, max_n, replace=False))
    off = np.cumsum([0] + lens)
    take = lambda getter: np.concatenate([np.asarray(getter(s))[pick[(pick >= a) & (pick < b)] - a]   # noqa: E731
                                          for (_, s), a, b in zip(parts, off[:-1], off[1:])])
    X = {tf: take(lambda s, tf=tf: s[f"X_{tf}"]) for tf in tfs}
    y = {k: take(lambda s, k=k: np.asarray(s["y"][k]).ravel()) for k in parts[0][1]["y"]}
    return (X[tfs[0]] if len(tfs) == 1 else X), y


def _diag_feature_names(model_tf=None):
    ds = globals().get("dataset")
    names = list(ds["feature_order"]) if isinstance(ds, dict) and "feature_order" in ds else None
    return {tf: names for tf in _tfs_of(model_tf)} if names and len(_tfs_of(model_tf)) > 1 else names


def make_training_diagnostics(train_split, val_split, batch_size=None, seed=0, probe_size=256, cartography_size=2000,
                              with_timestamps=True, **kw):
    """يبني (diag, train_ds): مسجّل TrainingDiagnostics (trainer_framework_v2 §6.5) + dataset التدريب نفسه مغلَّفاً بـ diag.tap وبمفتاح
    فهرس العيّنة. probe = عيّنة ثابتة من val، وخريطة البيانات على عيّنة ثابتة من train. kw: بقية خيارات TrainingDiagnostics
    (grad_every, influence_every, influence_scope, report_fn, …)؛ اسم خاطئ ← TypeError.
    train_ds يطابق dataset القسم ٥ تماماً (نفس الدفعات والترتيب)، فلا يتغيّر التدريب."""
    bs = batch_size or main_config["run"]["batch_size"]
    Xtr, ytr = model_x(train_split), with_sample_index(_y_for(train_split))
    meta = None
    if with_timestamps and "last_candles" in train_split:
        meta = pd.DataFrame({"timestamp": np.asarray(train_split["last_candles"])[:, LAST_COLUMNS.index("timestamp")]})
    diag = TrainingDiagnostics(probe=(model_x(val_split), _y_for(val_split)), probe_size=probe_size, cartography=(Xtr, ytr),
                               cartography_size=cartography_size, sample_meta=meta, seed=seed, **kw)
    return diag, diag.tap(make_shuffled_dataset(Xtr, ytr, bs, seed=seed))


def model_health_report(model, train_split, val_split, recorder=None, model_tf=None, max_n=512, sections=None,
                        sensitivity="grad_x_input", thresholds=None, capacity=None, with_capacity=True, verbose=True):
    """تقرير صحّة النموذج في استدعاء واحد: diagnose_model على عيّنة من val (+ من train للمقارنة بالتنشيطات) → جدول حكم ✅/⚠️/🚨
    بسبب وإجراء مقترح لكل بند. recorder (TrainingDiagnostics) يُضيف بنود التدريب: هيمنة الرؤوس، تضارب التدرّجات، تلاشي/انفجار الطبقات،
    نسبة التحديث، قفزات الخسارة، تأثير الدفعات، وخريطة البيانات.
    with_capacity: يضيف بند «سعة مقابل عيّنات فعّالة» (العيّنات الفعّالة + المرجع الخطّي بلا تدريب) يفصل «إخفاق إعداد (سعة)» عن «لا إشارة»؛
    capacity={'sweep','curve',...} من capacity_report يُدخل نتائج المسح/المنحنى في الحكم. يُرجع {'verdicts', 'diagnosis', 'recorder', 'capacity'}."""
    Xv, yv = _diag_xy(val_split, model_tf, max_n)
    rep = diagnose_model(model, Xv, yv, sections=sections or DIAG_SECTIONS, max_n=max_n, sensitivity=sensitivity,
                         feature_names=_diag_feature_names(model_tf))
    v = model_health_verdicts(rep, recorder.stats() if recorder is not None else None, thresholds)
    cap_lines = []
    if with_capacity or capacity:
        cl = rep["classification"]
        ms = float(cl["auc"].mean()) if "auc" in cl and cl["auc"].notna().any() else None
        cap_lines = _capacity_lines(model, train_split, val_split, model_tf, capacity, ms)
        for line in cap_lines:
            status = "⚠️" if line.startswith("⚠️") else "✅"
            v.loc[len(v)] = {"check": "سعة مقابل عيّنات فعّالة", "status": status, "detail": line.split(": ", 1)[-1],
                             "action": "خفّض الميزات/السعة أو زِد البيانات الفعّالة (RESEARCH_RULES §2.1)" if status == "⚠️" else ""}
    if verbose:
        print_verdicts(v)
    return {"verdicts": v, "diagnosis": rep, "recorder": recorder.stats() if recorder is not None else None, "capacity": cap_lines}


def model_layer_report(model, train_split, val_split, test_split=None, recorder=None, model_tf=None, max_train=2000,
                       max_val=2000, max_eval=1500, n_rand=3, n_perm=30, ridge=10.0, seed=0, capacity=None, with_capacity=True,
                       verbose=True):
    """تقرير أداء كل طبقة في استدعاء واحد (يعمل على نموذج مُدرَّب دون إعادة تدريب؛ مع recorder يضيف أعمدة التدرّج):
      ١) تنشيطات: انحراف/وحدات ميتة/تشبّع/الرتبة الفعّالة (diagnose_model 'activations').
      ٢) مسبار خطّي لكل طبقة (train → val): أين تظهر المعلومة وأين تضيع (layer_probe_report) — مع تسميات مخلوطة ونسخة عشوائية كأساس.
      ٣) صائب مقابل خاطئ لكل طبقة على test (أو val إن لم يُمرَّر) + فرق إسناد الميزات (layer_compare_report) — بنفس الأدلة.
      ٤) recorder: اتجاه نظيم تدرّج كل طبقة (أخير ÷ أول) ونسبة التحديث/الوزن.
    يُرجع {'table' (صفّ لكل طبقة)، 'probe', 'compare', 'verdict' (نصوص، ومنها سطر السعة 📐)}. الأعمدة probe:<رأس> = skill المسبار، sep:<رأس> = AUC فاصل صائب/خاطئ، '⚑' = فرق معنوي."""
    Xt, yt = _diag_xy(train_split, model_tf, max_train, seed)
    Xv, yv = _diag_xy(val_split, model_tf, max_val, seed + 1)
    Xe, ye = (Xv, yv) if test_split is None else _diag_xy(test_split, model_tf, max_eval, seed + 2)
    act = diagnose_model(model, Xv, yv, sections=("activations",), max_n=min(max_val, 512))["activations"]
    probe = layer_probe_report(model, Xt, yt, Xv, yv, max_train=max_train, max_val=max_val, ridge=ridge, n_rand=1, seed=seed)
    cmp_ = layer_compare_report(model, Xe, ye, max_n=max_eval, n_perm=n_perm, n_rand=n_rand, seed=seed)
    tab = act[["layer", "stage", "dim", "std", "dead_unit_frac", "eff_rank_ratio", "saturated_frac"]].copy()
    sk = probe.pivot_table(index="layer", columns="head", values="skill", sort=False)
    tab = tab.merge(sk.add_prefix("probe:").reset_index(), on="layer", how="left")
    ct = cmp_["table"]
    ct = ct[ct["kind"] == "activation"]
    if len(ct):
        sep = ct.pivot_table(index="layer", columns="head", values="sep_auc", sort=False).add_prefix("sep:").reset_index()
        flg = ct.groupby("layer", sort=False)["flag"].any().rename("⚑").reset_index()
        tab = tab.merge(sep, on="layer", how="left").merge(flg, on="layer", how="left")
    if recorder is not None:
        tr = recorder.group_trend().rename(columns={"group": "layer"})
        base = tab["layer"].str.split(".").str[0]
        tab = tab.assign(_g=base).merge(tr.rename(columns={"layer": "_g"}), on="_g", how="left").drop(columns="_g")
    verdict = layer_probe_verdict(probe) + cmp_["verdict"]
    if with_capacity or capacity:
        cl = diagnose_model(model, Xv, yv, sections=("classification",), max_n=min(max_val, 512))["classification"]
        ms = float(cl["auc"].mean()) if "auc" in cl and cl["auc"].notna().any() else None
        verdict += ["📐 " + line for line in _capacity_lines(model, train_split, val_split, model_tf, capacity, ms)]
    if verbose:
        print("🧅 تقرير الطبقات — لكل طبقة: تنشيطات | مسبار خطّي (val) | صائب/خاطئ (sep، ⚑ = فوق الخلط والنسخة العشوائية) | تدرّج")
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
            print(tab.to_string(index=False))
        for line in verdict:
            print("  •", line)
    return {"table": tab, "probe": probe, "compare": cmp_, "verdict": verdict}

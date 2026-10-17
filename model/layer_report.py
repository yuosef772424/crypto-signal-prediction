r"""
PURPOSE:  Per-layer reports on a trained model: layer_probe_report, layer_compare_report, random_init_copy.
TAGS:     layer_probe_report, layer_probe_verdict, layer_compare_report, random_init_copy, per-layer probe
PITFALLS: Needs a TRAINED model plus its random-init twin (random_init_copy); depends on diagnostics.py helpers.
          Executed into the one shared model namespace by model/_loader.py (never imported on its own): names from
          other modules resolve at call time.

## 9-ج) تقرير الطبقات — `layer_probe_report` / `layer_compare_report`

يعملان على **نموذج مُدرَّب دون إعادة تدريب**. كل إحصاءة تأتي بحجم الأثر **وخط أساس**: (أ) تسميات مخلوطة، (ب) نسخة عشوائية التهيئة
(`random_init_copy`، لا تمسّ عشوائية بايثون)، ولا يُرفع علمٌ إلا إن تجاوزها.

```python
probe = layer_probe_report(model, X_tr, y_tr, X_va, y_va)   # مسبار ridge لكل طبقة: أين تظهر المعلومة وأين تضيع (AUC/IC على val)
print("\n".join(layer_probe_verdict(probe)))
cmp_  = layer_compare_report(model, X_te, y_te)             # صائب مقابل خاطئ لكل رأس وطبقة + فرق إسناد الميزات
cmp_["table"]; cmp_["features"]; print("\n".join(cmp_["verdict"]))
```
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🧅 تقرير الطبقات — أين تظهر المعلومة وأين تضيع، وأيّ طبقة تفرّق بين التنبؤ الصائب والخاطئ؟
# ═══════════════════════════════════════════════════════════════════════════
#   probe = layer_probe_report(model, X_train, y_train, X_val, y_val)         # مسبار خطّي لكل طبقة (AUC/IC على val)
#   cmp   = layer_compare_report(model, X_test, y_test)                       # صائب مقابل خاطئ لكل طبقة (بلا إعادة تدريب)
# كل إحصاءة تأتي مع **أثر الحجم + عدم/خط أساس**: (أ) نفس الإحصاءة على تسميات/تجميع مخلوط، (ب) نسخة عشوائية التهيئة من
# النموذج نفسه. «فرق» لا يُرفع علماً إلا إن تجاوز الاثنين — فنموذج عشوائي التهيئة لا يُعلَّم (اختبار في tests/).


def random_init_copy(model, seed=0):
    """نسخة بنفس البنية وأوزان عشوائية جديدة (clone_model). تهيئة Keras 3 تسحب بذورها من random العامة في بايثون، فتُحفظ
    حالتها وتُعاد بعد الاستنساخ: لا أثر على عشوائية التدريب اللاحق، والنسخة حتمية بالبذرة."""
    import random
    st = random.getstate()
    try:
        random.seed(10_000 + int(seed))
        return tf.keras.models.clone_model(_unwrap_model(model))
    finally:
        random.setstate(st)


def _pooled(model, X, batch_size):
    taps = _make_taps(model, "groups")
    accs = _collect(model, X, taps, batch_size, keep_pooled=True)
    return [(n, s) for n, s, _ in taps], {n: np.concatenate(a.pooled) for n, a in accs.items()}


def _layer_applies(layer_name, stage, head):
    """طبقات «رأس» تخصّ رأسها فقط: class_fc_{t} لـ {t}_class، وnig_{t}(.fc) لـ {t}_reg. الباقي (المدخل…الجذع) مشترك بين الرؤوس."""
    if stage != "head":
        return True
    t, kind = head.rsplit("_", 1)
    return layer_name == f"class_fc_{t}" if kind == "class" else layer_name.split(".")[0] == f"nig_{t}"


def _std_fit(Z):
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    return mu, sd


def _head_specs(model, y):
    """[(اسم الرأس، مفتاح y، 'class'|'reg')] للرؤوس التي للنموذج مخرج لها ولـ y تسمية."""
    outs = list(model.output) if isinstance(model.output, dict) else []
    specs = []
    for k in outs:
        if k.endswith("_class_logits"):
            t = k[2:-len("_class_logits")]
            if f"y_{t}_class" in y:
                specs.append((f"{t}_class", f"y_{t}_class", "class"))
        elif k.startswith("y_") and f"{k}_nu" in outs and f"{k}_reg" in y:
            specs.append((f"{k[2:]}_reg", f"{k}_reg", "reg"))
    return specs


def _metric(kind, y, s):
    return _auc(y > 0, s) if kind == "class" else _spearman(s, y)


def layer_probe_report(model, X_train, y_train, X_val, y_val, max_train=2000, max_val=2000, ridge=10.0, n_null=5,
                       n_rand=1, seed=0, batch_size=256, z_min=3.0, rand_margin=0.02):
    """مسبار خطّي (ridge مغلق الصيغة على تنشيطات الطبقة المُجمَّعة: متوسط الزمن + آخر خطوة) لكل طبقة/رأس، يُدرَّب على عيّنة من
    train ويُقاس على val: AUC لرؤوس التصنيف وIC سبيرمان لرؤوس الانحدار ← **عند أي عمق تظهر المعلومة التنبؤية وأين تضيع**.

    الأدلة (لا تخمين): لكل صفّ skill مع
      null_mean/null_max : نفس المسبار مُدرَّباً على تسميات train **مخلوطة** (n_null مرّات) ثم المُقاس على val الحقيقي؛
      z                  : (skill − 0.5 أو 0) ÷ الخطأ المعياري تحت الصفر (Hanley–McNeil للـAUC، 1/√(n−1) للـIC)؛
      rand_skill         : نفس المسبار على نسخة عشوائية التهيئة (n_rand) — ميزات عشوائية من المدخل قد تحمل معلومة أصلاً.
      significant        : z ≥ z_min وskill > null_max.   beyond_random : skill > rand_skill + rand_margin (تعلّمت الطبقة فعلاً شيئاً).
    يُرجع DataFrame طويلاً (layer, stage, order, head, metric, skill, ...). الدالة لا تغيّر النموذج ولا تمسّ عشوائية بايثون/numpy."""
    model = _unwrap_model(model)
    sel_t = _subsample(_n_of(X_train), max_train, seed)
    sel_v = _subsample(_n_of(X_val), max_val, seed + 1)
    Xt, Xv = _sub(X_train, sel_t), _sub(X_val, sel_v)
    yt = {k: np.asarray(v)[sel_t] for k, v in y_train.items()}
    yv = {k: np.asarray(v)[sel_v] for k, v in y_val.items()}
    specs = [s for s in _head_specs(model, y_train) if s[1] in y_val]
    if not specs:
        raise ValueError("لا رأس مشترك بين مخارج النموذج وتسميات y")
    order, pt = _pooled(model, Xt, batch_size)
    _, pv = _pooled(model, Xv, batch_size)
    rands = []
    for r in range(n_rand):
        cp = random_init_copy(model, seed + r)
        _, a = _pooled(cp, Xt, batch_size)
        _, b = _pooled(cp, Xv, batch_size)
        rands.append((a, b))

    Y = np.stack([(yt[k] > 0).astype(np.float64) * 2 - 1 if kd == "class" else (yt[k] - yt[k].mean()) / (yt[k].std() + 1e-12)
                  for _, k, kd in specs], axis=1)
    Y = Y - Y.mean(0)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(Y)) for _ in range(n_null)]

    def fit_eval(Ztr, Zva):
        mu, sd = _std_fit(Ztr)
        A, B = (Ztr - mu) / sd, (Zva - mu) / sd
        M = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T)
        score = lambda Yp: B @ (M @ Yp)  # noqa: E731
        real = score(Y)
        nulls = [score(Y[p]) for p in perms]
        return real, nulls

    rows = []
    for li, (name, stage) in enumerate(order):
        real, nulls = fit_eval(pt[name], pv[name])
        rr = [fit_eval(a[name], b[name])[0] for a, b in rands] if all(name in a for a, _ in rands) else []
        for hi, (h, k, kd) in enumerate(specs):
            if not _layer_applies(name, stage, h):
                continue
            yvv = yv[k]
            sk = _metric(kd, yvv, real[:, hi])
            nl = np.array([_metric(kd, yvv, s[:, hi]) for s in nulls])
            rs = np.array([_metric(kd, yvv, r[:, hi]) for r in rr]) if rr else np.array([np.nan])
            n1 = int((yvv > 0).sum()) if kd == "class" else 0
            n0 = len(yvv) - n1
            se = (np.sqrt((n1 + n0 + 1) / (12.0 * n1 * n0)) if (kd == "class" and n1 and n0) else 1.0 / np.sqrt(max(len(yvv) - 1, 1)))
            z = (sk - (0.5 if kd == "class" else 0.0)) / se
            rows.append({"layer": name, "stage": stage, "order": li, "head": h, "metric": "auc" if kd == "class" else "ic",
                         "skill": sk, "null_mean": float(nl.mean()), "null_max": float(nl.max()), "se": float(se), "z": float(z),
                         "rand_skill": float(np.nanmean(rs)), "significant": bool(z >= z_min and sk > nl.max()),
                         "beyond_random": bool(sk > np.nanmean(rs) + rand_margin)})
    return pd.DataFrame(rows)


def layer_probe_verdict(df):
    """ملخّص نصّي لكل رأس: أول عمق تظهر فيه معلومة معنوية، أفضل طبقة، وهل ضاعت المعلومة لاحقاً (أفضل − الأخيرة > 3 se)."""
    lines = []
    for h, g in df.groupby("head", sort=False):
        g = g.sort_values("order")
        sig = g[g["significant"]]
        if sig.empty:
            lines.append(f"{h}: لا طبقة فيها معلومة معنوية فوق الصفر (أفضل skill {g['skill'].max():.3f}) — لا إشارة في أي عمق")
            continue
        shared = g[g["stage"] != "head"]
        best = shared.loc[shared["skill"].idxmax()]
        last = shared.iloc[-1]
        msg = f"{h}: تظهر المعلومة عند {sig.iloc[0]['layer']} (skill {sig.iloc[0]['skill']:.3f}) وذروتها عند {best['layer']} ({best['skill']:.3f})"
        if best["skill"] - last["skill"] > 3 * best["se"] and best["order"] < last["order"]:
            msg += f" ثم تضيع: {last['layer']} = {last['skill']:.3f}"
        hd = g[g["stage"] == "head"]
        if len(hd):
            msg += f" | خفيّ الرأس {hd.iloc[0]['layer']} = {hd.iloc[0]['skill']:.3f}"
        if not g.loc[g["significant"], "beyond_random"].any():
            msg += " — لا تتجاوز مسبار النسخة العشوائية (قد تكون معلومة المدخل لا أثراً للتعلّم)"
        lines.append(msg)
    return lines


# ── صائب مقابل خاطئ ──────────────────────────────────────────────────────────────────────────────────────
def _correct_flags(out, ys, specs):
    flags = {}
    for h, k, kd in specs:
        t = h.rsplit("_", 1)[0]
        if kd == "class":
            lab = np.asarray(ys[k]).reshape(-1) > 0
            flags[h] = (out[f"y_{t}_class_logits"].reshape(-1) >= 0.5) == lab
        else:
            err = np.abs(np.asarray(ys[k], dtype=np.float64).reshape(-1) - out[f"y_{t}"].reshape(-1))
            flags[h] = err <= np.median(err)
    return flags


def _cohen_cols(V, G):
    """d كوهين لكل عمود من V [n,F] لكل تقسيم في G [P,n] (bool) — مُتجَّه (بلا حلقات)."""
    Gf = G.astype(np.float64)
    ng, nw = Gf.sum(1, keepdims=True), (1 - Gf).sum(1, keepdims=True)
    s1g, s1 = Gf @ V, V.sum(0)[None]
    s2g, s2 = Gf @ (V ** 2), (V ** 2).sum(0)[None]
    mg, mw = s1g / ng, (s1 - s1g) / nw
    vg, vw = s2g / ng - mg ** 2, (s2 - s2g) / nw - mw ** 2
    return (mg - mw) / (np.sqrt((vg + vw) / 2) + 1e-12)


def _group_stats(Zs, good, folds):
    """(effect_d, sep_auc): d = جذر متوسط مربعات الفرق المعياري بين متوسّطي المجموعتين لكل وحدة؛ sep_auc = AUC فاصل بمتجه واحد
    (اتجاه الفرق بين المتوسّطين) مُدرَّب على نصف ومُقاس على النصف الآخر (تقاطعي) فلا تفاؤل من التدريب على نفس العيّنات."""
    c, w = Zs[good], Zs[~good]
    d = float(np.sqrt(np.mean(((c.mean(0) - w.mean(0)) / (Zs.std(0) + 1e-9)) ** 2)))
    aucs = []
    for a, b in ((folds[0], folds[1]), (folds[1], folds[0])):
        ga, gb = good[a], good[b]
        if ga.all() or (~ga).all() or gb.all() or (~gb).all():
            continue
        wv = Zs[a][ga].mean(0) - Zs[a][~ga].mean(0)
        aucs.append(_auc(gb, Zs[b] @ wv))
    return d, (float(np.mean(aucs)) if aucs else float("nan"))


def layer_compare_report(model, X, y, heads=None, max_n=1500, n_perm=30, n_rand=3, min_group=30, margin_auc=0.03,
                         margin_d=0.0, margin_feat=0.25, rand_mult=1.5, seed=0, batch_size=256, verbose=False):
    """يقارن تنشيطات كل طبقة بين العيّنات التي تنبّأ بها النموذج **صائباً** وتلك **الخاطئة** (لكل رأس)، على نموذج مُدرَّب مسبقاً
    وبلا إعادة تدريب. صائب: تصنيف = الفئة المتوقَّعة تطابق التسمية؛ انحدار = |الخطأ| ≤ وسيط الأخطاء.

    لكل (رأس، طبقة): effect_d (الفرق المعياري بين المتوسّطين)، sep_auc (فاصل بمتجه واحد تقاطعي). الأدلة المرفقة:
      null_q95_*   : المئين 95 لنفس الإحصاءة حين **تُخلَط تسميات y** (n_perm مرّة) ويُعاد حساب صائب/خاطئ بنفس تنبؤات النموذج — فيبقى
                     تلازم المجموعة بسلوك النموذج نفسه (ثقته/حساسيته) ويسقط وحده التلازم بالتسمية الحقيقية؛
      rand_*       : نفس الإحصاءة على نسخة عشوائية التهيئة (n_rand) بنفس التقسيم صائب/خاطئ — الصواب والخطأ دالّتان في المدخل،
                     فميزات عشوائية منه قد «تفرّقهما» قليلاً بلا أيّ تعلّم؛ هذا هو خط الأساس الحقيقي.
      flag         : sep_auc > max(null_q95_sep, 0.5 + rand_mult·(rand_sep−0.5) + margin_auc) و effect_d > max(null_q95_d, rand_mult·rand_d + margin_d). (الانتباه والميزات: d كوهين أكبر مقياساً فهامشها margin_feat.)
    ويُضاف صفّ لكل كتلة انتباه (kind='attention': d كوهين لمتوسط إنتروبيا الانتباه بين المجموعتين، بنفس الأدلة).
    + جدول `features`: فرق إسناد الميزات (تدرّج×مدخل) بين المجموعتين (d كوهين لكل ميزة، مقابل المئين 95 لأقصى |d| تحت الخلط وخط أساس النسخة العشوائية).
    يُرجع {'table', 'features', 'verdict' (قائمة نصوص), 'groups' (حجم المجموعتين لكل رأس)}."""
    model = _unwrap_model(model)
    sel = _subsample(_n_of(X), max_n, seed)
    Xs = _sub(X, sel)
    ys = {k: np.asarray(v)[sel] for k, v in y.items()}
    specs = _head_specs(model, ys)
    if heads is not None:
        unknown = set(heads) - {s[0] for s in specs}
        if unknown:
            raise ValueError(f"رؤوس غير معروفة {sorted(unknown)} — المتاح: {[s[0] for s in specs]}")
        specs = [s for s in specs if s[0] in heads]
    out = _predict(model, Xs, batch_size)
    flags = _correct_flags(out, ys, specs)
    order, pm = _pooled(model, Xs, batch_size)
    copies = [random_init_copy(model, seed + r) for r in range(n_rand)]
    pr = [_pooled(c, Xs, batch_size)[1] for c in copies]
    att_m = _attention_per_sample(model, Xs, batch_size)
    att_r = [_attention_per_sample(c, Xs, batch_size) for c in copies]
    obj = _objectives(set(out), _targets_of(out))
    attr_m = _input_attribution(model, Xs, {h: obj[h] for h, _, _ in specs if h in obj}, max(batch_size // 2, 32))
    attr_r = [_input_attribution(c, Xs, {h: obj[h] for h, _, _ in specs if h in obj}, max(batch_size // 2, 32)) for c in copies]
    rng = np.random.default_rng(seed)
    n = len(sel)
    perm_idx = np.argsort(rng.random((n_perm, n)), axis=1)          # خلط صفوف التسميات (الخلط يُعيد حساب صائب/خاطئ بنفس تنبؤات النموذج)
    good_perms = {}
    for sp in specs:
        good_perms[sp[0]] = np.stack([_correct_flags(out, {sp[1]: ys[sp[1]][p]}, [sp])[sp[0]] for p in perm_idx])
    perm_folds = np.argsort(rng.random(n))
    folds = (np.sort(perm_folds[: n // 2]), np.sort(perm_folds[n // 2:]))
    cohen = lambda v, g: float(_cohen_cols(v[:, None], g[None])[0, 0])  # noqa: E731
    rows, frows, groups = [], [], {}
    for h, k, kd in specs:
        good = flags[h]
        groups[h] = (int(good.sum()), int((~good).sum()))
        if min(groups[h]) < min_group:
            continue
        for li, (name, stage) in enumerate(order):
            if not _layer_applies(name, stage, h):
                continue
            Z = (pm[name] - pm[name].mean(0)) / (pm[name].std(0) + 1e-9)
            d, a = _group_stats(Z, good, folds)
            nulls = np.array([_group_stats(Z, gp, folds) for gp in good_perms[h]])
            rand = []
            for pz in pr:
                if name in pz:
                    Zr = (pz[name] - pz[name].mean(0)) / (pz[name].std(0) + 1e-9)
                    rand.append(_group_stats(Zr, good, folds))
            rand = np.array(rand) if rand else np.full((1, 2), np.nan)
            q95d, q95a = np.nanquantile(nulls[:, 0], 0.95), np.nanquantile(nulls[:, 1], 0.95)
            rd, ra = np.nanmax(rand[:, 0]), np.nanmax(rand[:, 1])
            thr_a, thr_d = max(q95a, rand_mult * (ra - 0.5) + 0.5 + margin_auc), max(q95d, rand_mult * rd + margin_d)
            rows.append({"head": h, "layer": name, "stage": stage, "order": li, "kind": "activation", "effect_d": d, "sep_auc": a,
                         "null_q95_d": float(q95d), "null_q95_sep": float(q95a), "rand_d": float(rd), "rand_sep": float(ra),
                         "excess_auc": float(a - thr_a), "flag": bool(a > thr_a and d > thr_d)})
        for bname, (ent, _, _) in att_m.items():
            v = ent.mean(1)
            dd = abs(cohen(v, good))
            nl = np.abs(_cohen_cols(v[:, None], good_perms[h])[:, 0])
            rr = [abs(cohen(ar[bname][0].mean(1), good)) for ar in att_r if bname in ar]
            q95, rmax = float(np.quantile(nl, 0.95)), (max(rr) if rr else float("nan"))
            thr = max(q95, rand_mult * rmax + margin_feat)
            rows.append({"head": h, "layer": bname, "stage": "attention", "order": 10_000, "kind": "attention", "effect_d": dd,
                         "sep_auc": float("nan"), "null_q95_d": q95, "null_q95_sep": float("nan"), "rand_d": rmax,
                         "rand_sep": float("nan"), "excess_auc": float(dd - thr), "flag": bool(dd > thr)})
        if h in attr_m:
            for key, imp in attr_m[h].items():
                dvec = _cohen_cols(imp, good[None])[0]
                nl = np.abs(_cohen_cols(imp, good_perms[h])).max(1)
                rmax = max([np.abs(_cohen_cols(ar[h][key], good[None])).max()
                            for ar in attr_r if h in ar and key in ar[h]] or [float("nan")])
                thr = max(float(np.quantile(nl, 0.95)), rand_mult * rmax + margin_feat)
                names = _feature_names_for(None, key, imp.shape[1])
                for j, dj in enumerate(dvec):
                    frows.append({"head": h, "input": key, "feature": names[j], "d": float(dj), "null_q95_maxabs": float(np.quantile(nl, 0.95)),
                                  "rand_maxabs": float(rmax), "flag": bool(abs(dj) > thr)})
    table = pd.DataFrame(rows)
    features = pd.DataFrame(frows, columns=["head", "input", "feature", "d", "null_q95_maxabs", "rand_maxabs", "flag"])
    verdict = []
    for h in [s[0] for s in specs]:
        if min(groups[h]) < min_group:
            verdict.append(f"{h}: مجموعة صغيرة (صائب {groups[h][0]}، خاطئ {groups[h][1]}) — لا مقارنة")
            continue
        g = table[(table["head"] == h) & (table["kind"] == "activation")]
        fl = g[g["flag"]].sort_values("excess_auc", ascending=False)
        if fl.empty:
            verdict.append(f"{h}: لا طبقة تفرّق الصائب من الخاطئ فوق الخلط والنسخة العشوائية — الخطأ لا يترك أثراً قابلاً للقراءة في التنشيطات")
        else:
            top = fl.iloc[0]
            weak = g[~g["flag"]]
            verdict.append(f"{h}: أكثر طبقة تفرّق بين الصائب والخاطئ = {top['layer']} (sep_auc {top['sep_auc']:.3f} مقابل خلط {top['null_q95_sep']:.3f} / عشوائي {top['rand_sep']:.3f}؛ d={top['effect_d']:.2f})"
                           + (f" | لا تفرّق: {', '.join(weak.sort_values('order')['layer'].head(4))}" if len(weak) else ""))
        ff = features[(features["head"] == h) & features["flag"]].sort_values("d", key=np.abs, ascending=False)
        if len(ff):
            verdict.append(f"   إسناد الميزات: {', '.join(f'{r.feature} (d={r.d:+.2f})' for r in ff.head(3).itertuples())} تختلف بين المجموعتين فوق الخلط")
    if verbose:
        for line in verdict:
            print(line)
    return {"table": table, "features": features, "verdict": verdict, "groups": groups}

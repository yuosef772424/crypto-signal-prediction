"""
PURPOSE:  Section 7-j: capacity vs effective sample size: effective_sample_size, simple_baseline, feature_count_sweep, learning_curve, capacity_verdict, capacity_report.
TAGS:     effective_sample_size, feature_count_sweep, learning_curve, capacity_verdict, capacity_report, CAPACITY_THRESHOLDS, capacity, ridge baseline
PITFALLS: RESEARCH_RULES section 2.1: a setup failure (capacity) is different from no signal; sweeps/curves train models. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 42 (section 7-j).
"""
import copy
import shutil
import tempfile


def _split_asset_ids(split):
    """رقم الأصل لكل صف: split['asset_ids'] إن وُجد، وإلا يُستنتج من انخفاض الطابع الزمني (split_data يرصّ عملةً عملة زمنياً)."""
    if "asset_ids" in split:
        return np.asarray(split["asset_ids"])
    ts = np.asarray(split["last_candles"])[:, LAST_COLUMNS.index("timestamp")]
    return np.concatenate([[0], np.cumsum(np.diff(ts) < 0)])


def effective_sample_size(split, window=None, stride=None, horizon=1, target=None, n_features=None, n_params=None, model=None,
                          max_assets=80, seed=0, model_tf=None):
    """العيّنات **الفعّالة** لا الخام: (١) خطوات زمنية فريدة غير متداخلة = مدى الزمن ÷ max(النافذة، أفق الهدف) — عيّنتان متجاورتان
    تتشاركان نافذة المدخل و/أو أفق الهدف؛ (٢) أصول مستقلّة في كل خطوة = N ÷ (1 + (N−1)·ρ̄) (أثر التصميم)، ρ̄ = متوسّط الارتباط
    المقطعي للهدف بين الأصول (العملات تتحرّك معاً فلا تُحسب N مستقلّة). الفعّالة = (١) × (٢).
    window: خطوات النافذة (الافتراضي = بُعد X الزمني)، stride: خطوات بين عيّنتين (للتقرير)، horizon: أفق الهدف بالخطوات.
    n_params: عدد معاملات النموذج (أو model=)؛ n_features: الافتراضي = عدد ميزات X. target: مفتاح y المستعمل للارتباط (الافتراضي أول y_*_reg).
    يُرجع dict: n_rows، n_assets، n_time_unique، n_time_indep، rho_bar، n_eff_assets، n_effective، ونسب eff_per_feature/eff_per_param
    وحكم نصّي: نسبة < 1 لكل معامل = السعة تفوق ما تدعمه البيانات (خطر حفظ)."""
    lc = np.asarray(split["last_candles"], dtype="float64")
    ts = lc[:, LAST_COLUMNS.index("timestamp")]
    a = _split_asset_ids(split)
    X = split[f"X_{_tfs_of(_required(model_tf, 'model_tf'))[0]}"]
    window = int(window or X.shape[1])
    n_features = int(n_features or X.shape[-1])
    if model is not None and n_params is None:
        n_params = int(_unwrap_model(model).count_params())
    uts = np.unique(ts)
    step = float(np.median(np.diff(uts))) if len(uts) > 1 else 1.0
    span = (uts[-1] - uts[0]) / step + 1.0
    n_time_indep = float(min(len(uts), span / max(window, int(horizon))))
    key = target or next((k for k in split["y"] if k.endswith("_reg")), next(iter(split["y"])))
    y = np.asarray(split["y"][key], dtype="float64").ravel()
    n_assets = int(a.max()) + 1
    panel = pd.DataFrame({"t": ts, "a": a, "y": y}).pivot_table(index="t", columns="a", values="y", aggfunc="mean")
    per_step = float(panel.notna().sum(axis=1).mean())
    cols = panel.columns.to_numpy()
    if len(cols) > max_assets:
        cols = np.sort(np.random.default_rng(seed).choice(cols, max_assets, replace=False))
    c = panel[cols].corr(min_periods=20).to_numpy()
    rho = float(np.nanmean(c[~np.eye(len(c), dtype=bool)])) if len(cols) > 1 else 0.0
    rho = float(np.clip(rho if np.isfinite(rho) else 0.0, 0.0, 0.999))
    n_eff_assets = per_step / (1.0 + (per_step - 1.0) * rho)
    n_eff = n_time_indep * n_eff_assets
    out = {"n_rows": int(len(ts)), "n_assets": n_assets, "assets_per_step": per_step, "n_time_unique": int(len(uts)),
           "n_time_indep": n_time_indep, "rho_bar": rho, "n_eff_assets": float(n_eff_assets), "n_effective": float(n_eff),
           "n_features": n_features, "n_params": n_params, "window": window, "stride": stride, "horizon": int(horizon),
           "eff_per_feature": float(n_eff / n_features), "eff_per_param": (float(n_eff / n_params) if n_params else None),
           "raw_over_effective": float(len(ts) / max(n_eff, 1e-9))}
    ratio = out["eff_per_param"]
    out["verdict"] = ("🚨 معاملات أكثر من العيّنات الفعّالة" if ratio is not None and ratio < 1 else
                      "⚠️ أقل من 10 عيّنات فعّالة لكل ميزة" if out["eff_per_feature"] < 10 else
                      "⚠️ أقل من 10 عيّنات فعّالة لكل معامل" if ratio is not None and ratio < 10 else "✅ كفاية معقولة")
    return out


# ── ridge سريع على ميزات مُجمَّعة [آخر خطوة، متوسط النافذة] — أبسط نموذج معقول ───────────────────────────────
def _pool_feats(X, cols=None):
    X = X if isinstance(X, dict) else {"x": X}
    parts = []
    for v in X.values():
        v = np.asarray(v)
        v = v if cols is None else v[:, :, cols]
        parts += [v[:, -1, :], v.mean(1)]
    return np.nan_to_num(np.concatenate(parts, axis=1).astype("float64"))


def _head_targets(y):
    return [(k[2:-6] + "_class", k, "class") for k in y if k.startswith("y_") and k.endswith("_class")] + \
           [(k[2:-4] + "_reg", k, "reg") for k in y if k.startswith("y_") and k.endswith("_reg")]


def _skill(kind, yv, s):
    return _auc(np.asarray(yv) > 0, s) if kind == "class" else _spearman(s, np.asarray(yv, dtype="float64"))


def _se_null(kind, yv):
    yv = np.asarray(yv)
    if kind == "class":
        n1, n0 = int((yv > 0).sum()), int((yv <= 0).sum())
        return float(np.sqrt((n1 + n0 + 1) / (12.0 * max(n1, 1) * max(n0, 1))))
    return float(1.0 / np.sqrt(max(len(yv) - 1, 1)))


def _ridge_skills(Ftr, ytr, Fva, yva, heads, ridge=10.0, n_null=3, seed=0):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    A, B = (Ftr - mu) / sd, (Fva - mu) / sd
    Y = np.stack([(np.asarray(ytr[k]).ravel() > 0) * 2.0 - 1 if kd == "class" else
                  (np.asarray(ytr[k], dtype="float64").ravel() - np.mean(ytr[k])) / (np.std(ytr[k]) + 1e-12) for _, k, kd in heads], 1)
    Y = Y - Y.mean(0)
    M = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T)
    real = B @ (M @ Y)
    rng = np.random.default_rng(seed)
    nulls = [B @ (M @ Y[rng.permutation(len(Y))]) for _ in range(n_null)]
    rows = []
    for i, (h, k, kd) in enumerate(heads):
        yv = np.asarray(yva[k]).ravel()
        sk = _skill(kd, yv, real[:, i])
        nl = np.array([_skill(kd, yv, s[:, i]) for s in nulls])
        z = (sk - (0.5 if kd == "class" else 0.0)) / _se_null(kd, yv)
        rows.append({"head": h, "metric": "auc" if kd == "class" else "ic", "skill": sk, "null_mean": float(nl.mean()),
                     "null_max": float(nl.max()), "z": float(z)})
    return rows


def simple_baseline(train_split, val_split, model_tf=None, max_train=20000, max_val=20000, ridge=10.0, seed=0):
    """أبسط نموذج معقول: ridge مغلق الصيغة على [آخر خطوة، متوسط النافذة] لكل ميزة، مُدرَّب على train ومُقاس على val (AUC للتصنيف،
    IC للانحدار) مع تسميات مخلوطة كأساس وz فوق الصفر. ما يحقّقه هو معلومة متاحة فعلاً بميزات خطّية؛ نموذج عميق تحته ← إخفاق إعداد."""
    Xt, yt = _diag_xy(train_split, model_tf, max_train, seed)
    Xv, yv = _diag_xy(val_split, model_tf, max_val, seed + 1)
    heads = [h for h in _head_targets(yt) if h[1] in yv]
    return pd.DataFrame(_ridge_skills(_pool_feats(Xt), yt, _pool_feats(Xv), yv, heads, ridge, seed=seed))


def _univariate_scores(X, y, heads):
    """درجة كل ميزة = أعلى |AUC−0.5| (أو |IC|) لآخر خطوة ومتوسط النافذة مع كل رؤوس التسميات — بلا نموذج."""
    X0 = X[next(iter(X))] if isinstance(X, dict) else X
    F = X0.shape[-1]
    sc = np.zeros(F)
    last, mean = np.nan_to_num(X0[:, -1, :].astype("float64")), np.nan_to_num(X0.mean(1).astype("float64"))
    for _, k, kd in heads:
        yy = np.asarray(y[k]).ravel()
        for M in (last, mean):
            for j in range(F):
                s = _skill(kd, yy, M[:, j])
                if np.isfinite(s):
                    sc[j] = max(sc[j], abs(s - 0.5) if kd == "class" else abs(s))
    return sc


def _select_cols(X, cols):
    return {k: np.asarray(v)[:, :, cols] for k, v in X.items()} if isinstance(X, dict) else np.asarray(X)[:, :, cols]


def _capacity_fit(builder, X, y, Xv, yv, epochs, seed, config=None, batch_size=None):
    """تدريب سريع بنفس مسار main (build_training_system) في مجلد مؤقّت: حقب ثابتة بلا إيقاف مبكر ولا استعادة «الأفضل» على val
    (فلا تدخل val في اختيار الحقبة — مقياس val نظيف)، ثم يُحذف المجلد."""
    cfg = copy.deepcopy(_required(config, "config"))
    d = tempfile.mkdtemp(prefix="capacity_")
    cfg["run"].update({"run_dir": d, "mirror_dir": None, "epochs": epochs, "train_mode": "new", "seed": seed, "verbose": 0})
    if batch_size:
        cfg["run"]["batch_size"] = batch_size
    cfg.setdefault("callbacks", {}).setdefault("early_stopping", {}).update(patience=10 ** 6, restore_best_weights=False)
    cfg = build_config(cfg)
    TRAINER_REGISTRY.pop(d, None)
    bs = cfg["run"]["batch_size"]
    ds = make_shuffled_dataset(X, y, bs, seed=seed)
    trainer, callbacks, _ = build_training_system(builder, cfg, next(iter(ds)))
    trainer.fit(ds, validation_data=make_eval_dataset(Xv, yv, bs), epochs=epochs, callbacks=callbacks, verbose=0)
    TRAINER_REGISTRY.pop(d, None)
    shutil.rmtree(d, ignore_errors=True)
    return trainer.model


def _model_skills(model, X, y, heads, batch_size=512):
    out = _predict(model, X, batch_size)
    res = {}
    for h, k, kd in heads:
        t = h.rsplit("_", 1)[0]
        key = f"y_{t}_class_logits" if kd == "class" else f"y_{t}"
        if key in out:
            res[h] = _skill(kd, y[k], out[key].reshape(len(y[k]), -1)[:, 0])
    return res


def _mean_skill(sk):
    cls = [v for h, v in sk.items() if h.endswith("_class") and np.isfinite(v)]
    if cls:
        return float(np.mean(cls)), "auc"
    reg = [v for v in sk.values() if np.isfinite(v)]
    return (float(np.mean(reg)) if reg else float("nan")), "ic"


def feature_count_sweep(build_fn, train_split, val_split, feature_names, ks=(1, 3, 5, 10), rank_by="train", epochs=6, seed=0,
                        max_train=4096, max_val=4096, config=None, model_tf=None, ridge=10.0, with_null=True, verbose=True):
    """يدرّب النموذج نفسه على أفضل k ميزة لكل k ويُرجع DataFrame: k، الميزات، skill على train (وضع الاستدلال) وعلى val، الفجوة، وخط الأساس
    (ridge على نفس الميزات)، و**الأدلة**: null_val = نفس التدريب على تسميات train **مخلوطة** (يجب ≈ 0.5؛ فوقه = تسرّب، وnull_train فوق 0.5 =
    قدرة على الحفظ عند هذا k)، وz = (val − 0.5) ÷ SE.
    rank_by: 'train' | 'val' | 'train_val' — يُرتَّب بإحصاءة أحادية المتغيّر (|AUC−0.5|/|IC| لآخر خطوة ومتوسط النافذة)؛ **test لا يُمرَّر إلى هنا أصلاً**.
    ('val' يستعمل val للترتيب فيتفاءل skill على val قليلاً — استعمله للاستكشاف لا للحكم، وفضّل 'train'.)
    build_fn(k) ← نموذج جديد بـ k ميزة. ks أكبر من عدد الميزات تُقصّ إلى عددها."""
    if rank_by not in ("train", "val", "train_val"):
        raise ValueError("rank_by: 'train' | 'val' | 'train_val'")
    Xt, yt = _diag_xy(train_split, model_tf, max_train, seed)
    Xv, yv = _diag_xy(val_split, model_tf, max_val, seed + 1)
    heads = [h for h in _head_targets(yt) if h[1] in yv]
    n_f = (Xt[next(iter(Xt))] if isinstance(Xt, dict) else Xt).shape[-1]
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_f)]
    st = _univariate_scores(Xt, yt, heads) if rank_by in ("train", "train_val") else 0
    sv = _univariate_scores(Xv, yv, heads) if rank_by in ("val", "train_val") else 0
    order = np.argsort(-(st + sv), kind="stable")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(next(iter(yt.values()))))
    y_null = {k: np.asarray(v)[perm] for k, v in yt.items()}
    rows = []
    for k in sorted({min(int(k), n_f) for k in ks}):
        cols = np.sort(order[:k])
        Xtk, Xvk = _select_cols(Xt, cols), _select_cols(Xv, cols)
        builder = lambda k=k: build_fn(k)  # noqa: E731
        m = _capacity_fit(builder, Xtk, yt, Xvk, yv, epochs, seed, config)
        tr, va = _model_skills(m, Xtk, yt, heads), _model_skills(m, Xvk, yv, heads)
        r = {"k": k, "features": [names[j] for j in cols]}
        (r["train_skill"], metric), (r["val_skill"], _) = _mean_skill(tr), _mean_skill(va)
        r["gap"] = r["train_skill"] - r["val_skill"]
        base = _ridge_skills(_pool_feats(Xtk), yt, _pool_feats(Xvk), yv, heads, ridge, seed=seed)
        bc = [b["skill"] for b in base if b["metric"] == metric]
        r["baseline_val"] = float(np.mean(bc)) if bc else float("nan")
        if with_null:
            mn = _capacity_fit(builder, Xtk, y_null, Xvk, yv, epochs, seed, config)
            (r["null_train"], _), (r["null_val"], _) = _mean_skill(_model_skills(mn, Xtk, y_null, heads)), _mean_skill(_model_skills(mn, Xvk, yv, heads))
        same = [h for h in heads if (h[2] == "class") == (metric == "auc")]
        se = _se_null(same[0][2], yv[same[0][1]]) / np.sqrt(len(same))
        r.update(metric=metric, se=float(se), z=float((r["val_skill"] - (0.5 if metric == "auc" else 0.0)) / se))
        rows.append(r)
    df = pd.DataFrame(rows)
    if verbose:
        with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format, "display.max_colwidth", 50):
            print("📐 مسح عدد الميزات (ترتيب بـ " + rank_by + " فقط؛ null = تسميات train مخلوطة)")
            print(df.drop(columns=["features"]).to_string(index=False))
    return df


def learning_curve(build_fn, train_split, val_split, fractions=(0.25, 0.5, 1.0), anchor="recent", epochs=6, seed=0,
                   max_train=4096, max_val=4096, config=None, model_tf=None, verbose=True):
    """skill على val مقابل نسبة فترة التدريب — **زمنياً لا عشوائياً**: anchor='recent' أحدث f من الصفوف حسب الطابع الزمني (الأقرب
    لـ val)، 'oldest' الأقدم. DataFrame: fraction، n، val_skill، train_skill. و`slope` (في .attrs) = ميل skill لكل مضاعفة بيانات
    بين آخر نسبتين (عند 100٪): موجب واضح = البيانات تنقص (السعة/الحفظ)، ≈0 = هضبة. build_fn(n_features) بكل ميزات X."""
    if anchor not in ("recent", "oldest"):
        raise ValueError("anchor: 'recent' | 'oldest'")
    fr = sorted(float(f) for f in fractions)
    if not fr or fr[0] <= 0 or fr[-1] > 1:
        raise ValueError("fractions في (0، 1]")
    Xt, yt = _diag_xy(train_split, model_tf, 10 ** 9, seed)
    ts = np.concatenate([np.asarray(s["last_candles"])[:, LAST_COLUMNS.index("timestamp")] for _, s in _split_parts(train_split)])
    Xv, yv = _diag_xy(val_split, model_tf, max_val, seed + 1)
    heads = [h for h in _head_targets(yt) if h[1] in yv]
    order = np.argsort(ts, kind="stable")
    n = len(order)
    n_f = (Xt[next(iter(Xt))] if isinstance(Xt, dict) else Xt).shape[-1]
    rows = []
    for f in fr:
        m_n = max(int(round(f * n)), 8)
        sel = np.sort(order[-m_n:] if anchor == "recent" else order[:m_n])
        if m_n > max_train:                                  # سقف التكلفة: عيّنة متباعدة زمنياً من النافذة المختارة (ليست عشوائية)
            sel = sel[np.linspace(0, len(sel) - 1, max_train).astype(int)]
        sub = lambda A: {k: np.asarray(v)[sel] for k, v in A.items()} if isinstance(A, dict) else np.asarray(A)[sel]  # noqa: E731
        Xs, ys = sub(Xt), {k: np.asarray(v)[sel] for k, v in yt.items()}
        m = _capacity_fit(lambda: build_fn(n_f), Xs, ys, Xv, yv, epochs, seed, config)
        (tr, metric), (va, _) = _mean_skill(_model_skills(m, Xs, ys, heads)), _mean_skill(_model_skills(m, Xv, yv, heads))
        rows.append({"fraction": f, "n": int(len(sel)), "val_skill": va, "train_skill": tr, "metric": metric})
    df = pd.DataFrame(rows)
    slope = float("nan")
    if len(df) >= 2:
        a, b = df.iloc[-2], df.iloc[-1]
        slope = float((b["val_skill"] - a["val_skill"]) / np.log2(b["n"] / a["n"])) if b["n"] > a["n"] else float("nan")
    df.attrs["slope_per_doubling"] = slope
    if verbose:
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print(f"📈 منحنى التعلّم ({anchor})\n{df.to_string(index=False)}\n   الميل عند 100٪: {slope:+.4f} لكل مضاعفة بيانات")
    return df


CAPACITY_THRESHOLDS = dict(z_signal=3.0, baseline_gap=0.01, slope_more_data=0.01, sweep_gain_z=2.0, gap_overfit=0.1)


def capacity_verdict(ess=None, baseline=None, sweep=None, curve=None, model_skill=None, model_metric="auc", thresholds=None):
    """سطر/أسطر حكم يفصل: «إخفاق إعداد (سعة)» ← ميزات/معاملات تفوق العيّنات الفعّالة أو نموذج تحت أبسط مرجع أو تتحسّن val بخفض k/بزيادة
    البيانات؛ «لا إشارة» ← لا النموذج ولا المرجع الخطّي ولا أي k يتجاوز التسميات المخلوطة (z<3) ومنحنى التعلّم مسطّح؛ «إشارة» ← يتجاوزها.
    كل المدخلات اختيارية (ess: effective_sample_size، baseline: simple_baseline، sweep: feature_count_sweep، curve: learning_curve،
    model_skill: skill النموذج الكامل على val). يُرجع قائمة نصوص (الأخطر أولاً)."""
    th = dict(CAPACITY_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(th)
        if unknown:
            raise ValueError(f"عتبات غير معروفة: {sorted(unknown)} — المتاح: {sorted(th)}")
        th.update(thresholds)
    lines, fail, sig = [], [], []
    mid = 0.5 if model_metric == "auc" else 0.0
    base_best, base_z = None, None
    if baseline is not None and len(baseline):
        b = baseline[baseline["metric"] == model_metric]
        if len(b):
            base_best, base_z = float(b["skill"].mean()), float(b["z"].mean())
            if base_z >= th["z_signal"]:
                sig.append(f"المرجع الخطّي {base_best:.3f} (z={base_z:.1f})")
    if model_skill is not None and base_best is not None and model_skill < base_best - th["baseline_gap"] and base_z >= th["z_signal"]:
        fail.append(f"النموذج ({model_skill:.3f}) تحت المرجع الخطّي ({base_best:.3f}) رغم وجود معلومة خطّية — لا يلتقط ما هو متاح")
    if ess is not None:
        r = ess.get("eff_per_param")
        if r is not None and r < 1:
            fail.append(f"{ess['n_params']:,} معامل لـ {ess['n_effective']:.0f} عيّنة فعّالة (خام {ess['n_rows']:,}، ρ̄={ess['rho_bar']:.2f}) — السعة تفوق البيانات")
        elif ess["eff_per_feature"] < 10:
            fail.append(f"{ess['n_features']} ميزة لـ {ess['n_effective']:.0f} عيّنة فعّالة (<10 لكل ميزة)")
    if sweep is not None and len(sweep):
        s = sweep.sort_values("k")
        full = s.iloc[-1]
        best = s.loc[s["val_skill"].idxmax()]
        if best["k"] < full["k"] and (best["val_skill"] - full["val_skill"]) > th["sweep_gain_z"] * full["se"]:
            fail.append(f"خفض الميزات يحسّن val: k={int(best['k'])} → {best['val_skill']:.3f} مقابل k={int(full['k'])} → {full['val_skill']:.3f} (فجوة train−val {full['gap']:+.2f})")
        if (s["z"] >= th["z_signal"]).any():
            sig.append(f"k={int(best['k'])}: val {best['val_skill']:.3f} (z={best['z']:.1f})")
        if "null_val" in s and ((s["null_val"] - mid) > th["sweep_gain_z"] * s["se"]).any():
            fail.append("تسميات مخلوطة تعطي val فوق الصفر في بعض k — تسرّب/مقياس غير صالح، لا تثق بأي skill هنا")
        if "null_train" in s and ((s["null_train"] - mid) > th["gap_overfit"]).any():
            kk = int(s.loc[(s["null_train"] - mid).idxmax(), "k"])
            fail.append(f"النموذج يحفظ تسميات مخلوطة على train عند k={kk} (skill {s['null_train'].max():.2f}) — سعة زائدة")
    if curve is not None and len(curve):
        sl = curve.attrs.get("slope_per_doubling", float("nan"))
        if np.isfinite(sl) and sl > th["slope_more_data"]:
            fail.append(f"منحنى التعلّم ما زال صاعداً عند 100٪ ({sl:+.3f} لكل مضاعفة) — البيانات الفعّالة تنقص")
        flat = np.isfinite(sl) and abs(sl) <= th["slope_more_data"]
    else:
        flat = False
    if fail:
        lines.append("⚠️ إخفاق إعداد (سعة): " + " | ".join(fail))
    if sig and not fail:
        lines.append("✅ إشارة: " + " | ".join(sig))
    elif sig:
        lines.append("ℹ️ توجد إشارة: " + " | ".join(sig))
    if not sig:
        parts = [x for x in (("المرجع الخطّي z=%.1f" % base_z) if base_z is not None else None,
                             ("val النموذج %.3f" % model_skill) if model_skill is not None else None,
                             "منحنى التعلّم مسطّح" if flat else None) if x]
        lines.append("➖ لا إشارة: لا نموذج ولا مرجع خطّي ولا أي k يتجاوز التسميات المخلوطة (z<%.0f)" % th["z_signal"]
                     + (" — " + "، ".join(parts) if parts else "")
                     + (" — لكن راجع إخفاق السعة أعلاه: قد تكون العيّنات الفعّالة أقل من أن تُظهر إشارة." if fail
                        else " — لا خلل ظاهر في الإعداد؛ لا معلومة بهذه الميزات/الهدف."))
    return lines


def capacity_report(build_fn, train_split, val_split, feature_names=None, ks=(1, 3, 5, 10), fractions=(0.25, 0.5, 1.0), window=None,
                    model=None, rank_by="train", epochs=6, seed=0, config=None, model_tf=None, verbose=True):
    """كل ضوابط السعة في استدعاء واحد (تدريب لكل k ولكل نسبة — الأغلى): effective_sample_size + simple_baseline + feature_count_sweep +
    learning_curve ثم capacity_verdict. يُرجع {'ess','baseline','sweep','curve','verdict'}."""
    ess = effective_sample_size(train_split, window=window, model=model, model_tf=model_tf)
    base = simple_baseline(train_split, val_split, model_tf)
    sw = feature_count_sweep(build_fn, train_split, val_split, feature_names, ks, rank_by, epochs, seed, config=config,
                             model_tf=model_tf, verbose=verbose)
    lc = learning_curve(build_fn, train_split, val_split, fractions, epochs=epochs, seed=seed, config=config, model_tf=model_tf, verbose=verbose)
    v = capacity_verdict(ess=ess, baseline=base, sweep=sw, curve=lc)
    if verbose:
        print(f"📐 عيّنات فعّالة: {ess['n_effective']:.0f} (خام {ess['n_rows']:,}، ρ̄={ess['rho_bar']:.2f}) — {ess['verdict']}")
        for line in v:
            print(" ", line)
    return {"ess": ess, "baseline": base, "sweep": sw, "curve": lc, "verdict": v}


def _capacity_lines(model, train_split, val_split, model_tf=None, capacity=None, model_skill=None, model_metric="auc"):
    """سطور حكم السعة لتقارير الصحّة/الطبقات: تستعمل نتائج جاهزة (capacity={'ess','baseline','sweep','curve'}) إن مُرِّرت، وإلا تحسب الرخيصة
    فقط (العيّنات الفعّالة + المرجع الخطّي، بلا تدريب). المسح والمنحنى يحتاجان تدريباً فلا يُشغَّلان تلقائياً (capacity_report)."""
    cap = dict(capacity or {})
    if "ess" not in cap:
        cap["ess"] = effective_sample_size(train_split, model=model, model_tf=model_tf)
    if "baseline" not in cap:
        cap["baseline"] = simple_baseline(train_split, val_split, model_tf)
    return capacity_verdict(ess=cap["ess"], baseline=cap["baseline"], sweep=cap.get("sweep"), curve=cap.get("curve"),
                            model_skill=model_skill, model_metric=model_metric)

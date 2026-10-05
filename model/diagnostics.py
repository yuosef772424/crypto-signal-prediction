"""
PURPOSE:  On-demand model diagnostics: diagnose_model, model_health_verdicts, print_verdicts (+ DIAG_SECTIONS,
          HEALTH_THRESHOLDS).
TAGS:     diagnose_model, model_health_verdicts, print_verdicts, DIAG_SECTIONS, HEALTH_THRESHOLDS, model health,
          activations, attention
PITFALLS: Runs nothing at load; only on explicit call. Verdict thresholds live in HEALTH_THRESHOLDS. Executed into the
          one shared model namespace by model/_loader.py (never imported on its own): names from other modules resolve
          at call time.

## 9-ب) تشخيص النموذج عند الطلب — `diagnose_model` / `model_health_verdicts`

**لا يعمل شيء هنا إلا إن استدعيتَه**: النموذج وأسماء طبقاته ومخارجه ونقاط حفظه لا تتغيّر (المسابر نماذج فرعية تقرأ الطبقات القائمة نفسها).

```python
rep = diagnose_model(model, X_val, y_val)               # y اختياري؛ X مصفوفة أو {فريم: مصفوفة}
v   = model_health_verdicts(rep); print_verdicts(v)     # ✅/⚠️/🚨 + سبب + إجراء
rep["classification"]; rep["nig"]; rep["activations"]; rep["attention"]; rep["sensitivity"]; rep["samples"]
```

| القسم | ما يقيسه |
|---|---|
| `heads` | وسط/انحراف/انهيار كل مخرج، وارتباط التنبؤ بالهدف ونسبة تباينه |
| `nig` | اصطدام ν/α/β بأرضياتها، سقف عدم اليقين، aleatoric مقابل epistemic، ومعايرة (الخطأ المطلق مقابل عدم اليقين) |
| `classification` | حصّة الفئة المتوقَّعة مقابل حصّة التسميات، التشبّع، ECE، AUC مقابل الأغلبية |
| `activations` | وحدات ميتة/نادرة/ثابتة، تشبّع sigmoid/tanh، الرتبة الفعّالة لكل طبقة |
| `attention` | إنتروبيا كل رأس انتباه ومسافته (+ فحص تطابق إعادة الحساب مع الطبقة) |
| `sensitivity` | حصّة كل ميزة (تدرّج×مدخل أو `sensitivity="permutation"`) وتركّز الاعتماد |
| `samples` | العيّنات الواثقة والخاطئة، والبقايا القياسية الكبيرة، وepistemic العالي |

أقسام خاطئة الاسم ← `ValueError`. شرح كل إشارة وطريقة الاستدعاء من `main`: `docs/research/model_diagnostics.md`.
"""
# ═══════════════════════════════════════════════════════════════════════════
# 🩺 تشخيص النموذج عند الطلب — لا يعمل شيء منه إلا إن استُدعي (النموذج وأسماء طبقاته ومخارجه لا تتغيّر)
# ═══════════════════════════════════════════════════════════════════════════
#   rep = diagnose_model(model, X, y)             # جداول: رؤوس، NIG، تصنيف، تنشيطات، انتباه، حساسية، عيّنات
#   v   = model_health_verdicts(rep)              # جدول حكم ✅/⚠️/🚨 (+ recorder.stats() إن وُجد مسجّل تدريب)
# المسابر (probes) نماذج فرعية تقرأ طبقات النموذج القائمة نفسها؛ لا طبقة تُضاف ولا تُغيَّر.
import pandas as pd

DIAG_SECTIONS = ("heads", "nig", "classification", "activations", "attention", "sensitivity", "samples")
_SKIP_TAP_TYPES = {"InputLayer", "Dropout", "SpatialDropout1D", "GaussianNoise", "Flatten", "Reshape", "Lambda"}
_STAGE_RULES = (  # (بادئة اسم الطبقة، مرحلة) — ترتيب الأولوية
    ("instance_norm", "input_norm"), ("trend_seasonal", "decomp"), ("shape_and_level", "level"),
    ("patch_embed", "embed"), ("step_proj", "embed"), ("block_", "encoder_block"), ("gru_res_", "encoder_block"),
    ("tcn_res_", "encoder_block"), ("final_norm", "encoder_out"), ("last_token", "readout_pool"),
    ("attn_pool", "readout_pool"), ("readout_fc", "readout"), ("add_linear_path", "readout"),
    ("branch_concat", "trunk"), ("trunk_norm", "trunk"), ("class_fc_", "head"), ("nig_", "head"),
)


def _unwrap_model(m):
    """GenericTrainer (له _compute_losses) ← نموذجه الداخلي؛ أي نموذج آخر كما هو."""
    return m.model if hasattr(m, "_compute_losses") else m


def _stage_of(layer_name):
    base = layer_name.split(".")[0]
    for pre, stage in _STAGE_RULES:
        if base.startswith(pre):
            return stage
    return "other"


def _n_of(X):
    return len(next(iter(X.values())) if isinstance(X, dict) else X)


def _sub(X, sel, dtype="float32"):
    cast = lambda a: np.asarray(a)[sel].astype(dtype) if np.issubdtype(np.asarray(a).dtype, np.floating) else np.asarray(a)[sel]
    return {k: cast(v) for k, v in X.items()} if isinstance(X, dict) else cast(X)


def _predict(model, X, batch_size=256):
    n = _n_of(X)
    outs = [{k: np.asarray(v) for k, v in model(_sub(X, slice(a, a + batch_size)), training=False).items()}
            for a in range(0, n, batch_size)]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _targets_of(out):
    ts = []
    for k in out:
        if k.endswith("_class_logits"):
            ts.append(k[2:-len("_class_logits")])
        elif k.startswith("y_") and f"{k}_nu" in out:
            ts.append(k[2:])
    return sorted(set(ts))


def _rank(a):
    return pd.Series(np.asarray(a, dtype=np.float64)).rank().to_numpy()


def _auc(y, s):
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    return float((_rank(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _label_bool(y, t):
    return np.asarray(y[f"y_{t}_class"]).reshape(-1) > 0


def _subsample(n, max_n, seed):
    return np.arange(n) if n <= max_n else np.sort(np.random.default_rng(seed).choice(n, max_n, replace=False))


def _head_layer(model, prefix, t):
    try:
        return model.get_layer(f"{prefix}_{t}")
    except ValueError:
        return None


# ── مسابر التنشيط: نموذج فرعي واحد يقرأ مخارج طبقات قائمة (بلا تعديل للرسم الأصلي) ──────────────────────────
def _make_taps(model, mode="all"):
    """قائمة (اسم، مرحلة، موتّر) لطبقات النموذج. mode='all': كل الطبقات ذات المخرج العددي؛ 'groups': كتل الأداء المسمّاة
    (مدخل، تفكيك، تضمين، كل كتلة، قراءة، جذع، وخفيّ كل رأس) — ما يُعرَض في تقرير الطبقات."""
    taps = []
    for layer in model.layers:
        cls = type(layer).__name__
        if cls in _SKIP_TAP_TYPES:
            continue
        name = layer.name
        if mode == "groups" and _stage_of(name) == "other":
            continue
        if mode == "groups" and name.startswith(("conf_", "unc_", "ordered_means")):
            continue
        try:
            if cls == "NIGHead":   # خفيّ الرأس = fc(المدخل): يقرأ الطبقة الفرعية القائمة، لا مخرج الرأس الرباعي
                taps.append((f"{name}.fc", _stage_of(name), layer.fc(layer.input)))
                continue
            out = layer.output
            out = out[0] if isinstance(out, (list, tuple)) else out
            if getattr(out, "shape", None) is None or len(out.shape) < 2 or "float" not in str(out.dtype):
                continue
            if mode == "all" and name.startswith(("y_", "unc_", "ordered_means")):
                continue
            if mode == "groups" and name.startswith("y_"):
                continue
            taps.append((name, _stage_of(name), out))
        except Exception:
            continue
    return taps


class _Acc:
    """مجمِّع بثّي لإحصاءات تنشيط طبقة (لا نخزّن التنشيطات كاملةً): لكل وحدة المتوسّط/التباين/أقصى |a|/نسبة التنشيط، وتغاير الوحدات
    لرتبة التمثيل الفعّالة، وتشبّع الحدود. + متجهات مُجمَّعة لكل عيّنة (متوسط الزمن وآخر خطوة) لمسابر الطبقات."""

    def __init__(self):
        self.n = 0
        self.s1 = self.s2 = self.mx = self.act = self.cov = None
        self.nonfinite = 0
        self.sat_hi = self.sat_lo = 0
        self.pooled = []

    def add(self, a, keep_pooled):
        a = np.asarray(a, dtype=np.float64)
        d = a.shape[-1]
        flat = a.reshape(-1, d)
        fin = np.isfinite(flat)
        self.nonfinite += int((~fin).sum())
        flat = np.where(fin, flat, 0.0)
        if self.s1 is None:
            self.s1, self.s2, self.mx, self.act = np.zeros(d), np.zeros(d), np.zeros(d), np.zeros(d)
            self.cov = np.zeros((d, d)) if d <= 512 else None
        self.n += len(flat)
        self.s1 += flat.sum(0)
        self.s2 += (flat ** 2).sum(0)
        self.mx = np.maximum(self.mx, np.abs(flat).max(0))
        self.act += (np.abs(flat) > 1e-6).sum(0)
        if self.cov is not None:
            self.cov += flat.T @ flat
        self.sat_hi += int((flat >= 0.98).sum())
        self.sat_lo += int((flat <= -0.98).sum() + (np.abs(flat) <= 0.02).sum())
        if keep_pooled:
            a3 = a.reshape(len(a), -1, d)
            self.pooled.append(np.concatenate([a3.mean(1), a3[:, -1, :]], axis=1) if a3.shape[1] > 1 else a3[:, 0, :])

    def stats(self, activation=None):
        mean = self.s1 / self.n
        var = np.maximum(self.s2 / self.n - mean ** 2, 0.0)
        d = len(mean)
        row = {"dim": d, "mean": float(mean.mean()), "std": float(np.sqrt(var).mean()), "max_abs": float(self.mx.max()),
               "nonfinite_frac": self.nonfinite / max(self.n * d, 1),
               "dead_unit_frac": float(np.mean(self.mx < 1e-6)),
               "rarely_active_frac": float(np.mean(self.act / self.n < 0.01)),
               "const_unit_frac": float(np.mean(np.sqrt(var) < 1e-6 * (np.abs(mean) + 1e-3)))}
        if self.cov is not None and d > 1:
            c = self.cov / self.n - np.outer(mean, mean)
            ev = np.maximum(np.linalg.eigvalsh(c), 0.0)
            row["eff_rank_ratio"] = float((ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30)) / d)
        else:
            row["eff_rank_ratio"] = float("nan")
        bounded = activation in ("sigmoid", "tanh")
        row["saturated_frac"] = float((self.sat_hi + (self.sat_lo if activation == "tanh" else 0)) / (self.n * d)) if bounded else float("nan")
        return row


def _collect(model, X, taps, batch_size=256, keep_pooled=False):
    """تمريرة واحدة: {اسم_الطبقة: _Acc}. الطبقة تُقرأ بوضع الاستدلال (training=False)."""
    probe = tf.keras.Model(model.input, [t[2] for t in taps])
    accs = {t[0]: _Acc() for t in taps}
    n = _n_of(X)
    for a in range(0, n, batch_size):
        outs = probe(_sub(X, slice(a, a + batch_size)), training=False)
        outs = outs if isinstance(outs, (list, tuple)) else [outs]
        for (name, _, _), o in zip(taps, outs):
            accs[name].add(o.numpy(), keep_pooled)
    return accs


def _layer_activation_fn(model, name):
    try:
        layer = model.get_layer(name.split(".")[0])
        act = getattr(layer, "activation", None)
        return getattr(act, "__name__", None)
    except ValueError:
        return None


# ── الانتباه: احتمالات softmax تُعاد حسابها من أوزان RelativeGQAttention القائمة (بلا تغيير للرسم) ───────────────
def _attn_probs(blk, x):
    """نفس خطوات RelativeGQAttention.call حتى softmax (ويُرجع أيضاً ناتج الطبقة المعاد حسابه للتحقّق من التطابق)."""
    a = blk.attn
    xn = blk.norm1(x)
    b, t = tf.shape(xn)[0], tf.shape(xn)[1]
    q = tf.reshape(a.wq(xn), (b, t, a.h, a.dh))
    k = tf.reshape(a.wk(xn), (b, t, a.hkv, a.dh))
    v = tf.reshape(a.wv(xn), (b, t, a.hkv, a.dh))
    if a.hkv != a.h:
        rep = a.h // a.hkv
        k, v = tf.repeat(k, rep, axis=2), tf.repeat(v, rep, axis=2)
    q, k, v = [tf.transpose(z, [0, 2, 1, 3]) for z in (q, k, v)]
    scores = tf.matmul(q, k, transpose_b=True) * (float(a.dh) ** -0.5)
    pos = tf.range(t)
    rel = pos[:, None] - pos[None, :]
    idx = tf.clip_by_value(rel, -a.max_rel_pos, a.max_rel_pos) + a.max_rel_pos
    scores = scores + tf.cast(tf.transpose(tf.gather(a.rel_bias, idx), [2, 0, 1])[None], scores.dtype)
    if a.causal or a.window is not None:
        allowed = tf.ones_like(rel, dtype=tf.bool)
        if a.causal:
            allowed = tf.logical_and(allowed, rel >= 0)
        if a.window is not None:
            allowed = tf.logical_and(allowed, tf.abs(rel) <= a.window)
        scores = tf.where(allowed[None, None], scores, tf.cast(scores.dtype.min, scores.dtype))
    probs = tf.nn.softmax(scores, axis=-1)
    ctx = tf.transpose(tf.matmul(probs, v), [0, 2, 1, 3])
    return probs, a.wo(tf.reshape(ctx, (b, t, a.d_model))), a(xn, training=False)


def _attention_per_sample(model, X, batch_size=256):
    """{اسم الكتلة: (entropy_ratio [n, heads] طبيعيّة إلى [0,1]، mean_distance [n, heads] نسبة إلى T، sync_err)} لكل TransformerBlock."""
    blocks = [l for l in model.layers if type(l).__name__ == "TransformerBlock"]
    if not blocks:
        return {}
    probe = tf.keras.Model(model.input, [b.input for b in blocks])
    ent_l, dist_l = {b.name: [] for b in blocks}, {b.name: [] for b in blocks}
    sync = {b.name: 0.0 for b in blocks}
    for a in range(0, _n_of(X), batch_size):
        xs = probe(_sub(X, slice(a, a + batch_size)), training=False)
        xs = xs if isinstance(xs, (list, tuple)) else [xs]
        for blk, x in zip(blocks, xs):
            p, mine, real = _attn_probs(blk, x)
            p = p.numpy()
            t = p.shape[-1]
            ent = -(p * np.log(p + 1e-12)).sum(-1) / np.log(max(t, 2))            # [b, h, T_q]
            dist = (p * np.abs(np.arange(t)[:, None] - np.arange(t)[None, :])[None, None]).sum(-1) / max(t - 1, 1)
            ent_l[blk.name].append(ent.mean(-1))
            dist_l[blk.name].append(dist.mean(-1))
            sync[blk.name] = max(sync[blk.name], float(np.abs(mine.numpy() - real.numpy()).max()))
    return {n: (np.concatenate(ent_l[n]), np.concatenate(dist_l[n]), sync[n]) for n in ent_l}


# ── حساسية المدخل: تدرّج × مدخل لكل عيّنة ──────────────────────────────────────────────────────────────────
def _objectives(out_keys, targets):
    """{اسم الرأس: مفتاح المخرج المشتقّ}: mu لرؤوس الانحدار، احتمال الفئة لرؤوس التصنيف."""
    obj = {}
    for t in targets:
        if f"y_{t}_nu" in out_keys:
            obj[f"{t}_reg"] = f"y_{t}"
        if f"y_{t}_class_logits" in out_keys:
            obj[f"{t}_class"] = f"y_{t}_class_logits"
    return obj


def _input_attribution(model, X, objectives, batch_size=128):
    """{رأس: {مدخل: [n, F]}} = متوسّط |∂المخرج/∂x · x| على الزمن لكل عيّنة وميزة. مدخلات غير عددية (coin_id) تُتجاهل."""
    keys = list(X) if isinstance(X, dict) else None
    res = {h: {} for h in objectives}
    for a in range(0, _n_of(X), batch_size):
        xb = _sub(X, slice(a, a + batch_size))
        xt = {k: tf.constant(v) for k, v in xb.items()} if keys else tf.constant(xb)
        watch = [k for k in keys if np.issubdtype(xb[k].dtype, np.floating)] if keys else [None]
        for h, key in objectives.items():
            with tf.GradientTape() as tape:
                for k in watch:
                    tape.watch(xt if k is None else xt[k])
                o = model(xt, training=False)[key]
                if o.shape[-1] > 1:
                    o = tf.reduce_max(o, axis=-1)
                obj = tf.reduce_sum(o)
            gs = tape.gradient(obj, [xt if k is None else xt[k] for k in watch])
            for k, g in zip(watch, gs):
                if g is None:
                    continue
                imp = tf.reduce_mean(tf.abs(g * (xt if k is None else xt[k])), axis=1).numpy()
                res[h].setdefault("input" if k is None else k, []).append(imp)
    return {h: {k: np.concatenate(v) for k, v in d.items()} for h, d in res.items()}


def _feature_names_for(feature_names, key, f):
    if isinstance(feature_names, dict):
        names = feature_names.get(key)
    else:
        names = feature_names
    return list(names) if names is not None and len(names) == f else [f"f{i}" for i in range(f)]


# ── الدالة الرئيسية ────────────────────────────────────────────────────────────────────────────────────────
def diagnose_model(model, X, y=None, sections=DIAG_SECTIONS, max_n=512, batch_size=256, seed=0, feature_names=None,
                   sensitivity="grad_x_input", top_k=10, floor_tol=0.1, verbose=False):
    """تشخيص عند الطلب لنموذج NIG-TimeNet v2 (أو مشتقّ بنفس تسمية المخارج) على عيّنة X.

    X: مصفوفة [n,T,F] أو قاموس {فريم: مصفوفة}. y (اختياري): قاموس بتسمية خط الأنابيب (y_{t}_class بترميز >0، y_{t}_reg).
    sections: أي مجموعة جزئية من DIAG_SECTIONS؛ اسم غير معروف ← ValueError. sensitivity: 'grad_x_input' (بلا تسميات) |
    'permutation' (يتطلّب y). max_n: حدّ العيّنة (عشوائية ثابتة بـ seed). يُرجع قاموس DataFrames:
      heads          إحصاءات كل مخرج (وسط/انحراف/انهيار/غير منتهٍ) + ارتباط ونسبة تباين التنبؤ إلى الهدف إن وُجد y.
      nig            لكل هدف: نسبة اصطدام ν/α/β بأرضياتها، سقف عدم اليقين، aleatoric مقابل epistemic، ومعايرة (ارتباط |الخطأ| بعدم اليقين).
      classification لكل رأس تصنيف: حصّة الفئة المتوقَّعة مقابل حصّة التسميات، تشبّع الاحتمال، ECE، AUC، الدقّة مقابل الأغلبية.
      activations    لكل طبقة: وحدات ميتة/نادرة النشاط/ثابتة، تشبّع sigmoid/tanh، الرتبة الفعّالة للتمثيل، قيم غير منتهية.
      attention      لكل كتلة انتباه ورأس: إنتروبيا طبيعية (0 مركّز، 1 منتظم)، متوسط المسافة، ومطابقة إعادة الحساب مع الطبقة.
      sensitivity    حصّة كل ميزة من حساسية كل رأس (+ تركّز: أعلى حصّة، عدد الميزات الفعّال).
      samples        أصعب/أشبه العيّنات بالمشبوهة: واثقة وخاطئة، بقايا قياسية كبيرة، epistemic عالٍ."""
    bad = [s for s in sections if s not in DIAG_SECTIONS]
    if bad:
        raise ValueError(f"أقسام غير معروفة {bad} — المتاح: {DIAG_SECTIONS}")
    if sensitivity not in ("grad_x_input", "permutation"):
        raise ValueError("sensitivity: 'grad_x_input' | 'permutation'")
    if sensitivity == "permutation" and y is None and "sensitivity" in sections:
        raise ValueError("sensitivity='permutation' يحتاج y")
    model = _unwrap_model(model)
    sel = _subsample(_n_of(X), max_n, seed)
    Xs = _sub(X, sel)
    ys = None if y is None else {k: np.asarray(v)[sel] for k, v in y.items()}
    out = _predict(model, Xs, batch_size)
    targets = _targets_of(out)
    rep = {"meta": {"n": len(sel), "targets": targets}}
    if "heads" in sections:
        rep["heads"] = _diag_heads(out, ys)
    if "nig" in sections:
        rep["nig"] = _diag_nig(model, out, ys, targets, floor_tol)
    if "classification" in sections:
        rep["classification"] = _diag_classification(out, ys, targets)
    if "activations" in sections:
        taps = _make_taps(model, "all")
        accs = _collect(model, Xs, taps, batch_size)
        rows = []
        for name, stage, _ in taps:
            r = accs[name].stats(_layer_activation_fn(model, name))
            rows.append({"layer": name, "stage": stage, **r})
        rep["activations"] = pd.DataFrame(rows)
    if "attention" in sections:
        att = _attention_per_sample(model, Xs, batch_size)
        rows = []
        for name, (ent, dist, sync) in att.items():
            for h in range(ent.shape[1]):
                rows.append({"layer": name, "head": h, "entropy_ratio": float(ent[:, h].mean()),
                             "mean_distance": float(dist[:, h].mean()), "sync_err": sync})
        rep["attention"] = pd.DataFrame(rows, columns=["layer", "head", "entropy_ratio", "mean_distance", "sync_err"])
    if "sensitivity" in sections:
        rep["sensitivity"] = _diag_sensitivity(model, Xs, ys, targets, out, sensitivity, feature_names, batch_size, seed)
    if "samples" in sections:
        rep["samples"] = _diag_samples(out, ys, targets, top_k) if ys is not None else pd.DataFrame()
        if len(rep["samples"]):
            rep["samples"]["row"] = sel[rep["samples"]["row"].to_numpy()]     # فهرس الصف في X الأصلي
    rep["sel"] = sel
    if verbose:
        for k, v in rep.items():
            if isinstance(v, pd.DataFrame):
                print(f"── {k}\n{v.to_string(max_rows=30)}")
    return rep


def _diag_heads(out, ys):
    rows = []
    for k, v in out.items():
        a = np.asarray(v, dtype=np.float64).reshape(len(v), -1)
        fin = np.isfinite(a)
        f = a[:, 0][fin[:, 0]]
        r = {"output": k, "mean": float(f.mean()) if f.size else np.nan, "std": float(f.std()) if f.size else np.nan,
             "min": float(f.min()) if f.size else np.nan, "max": float(f.max()) if f.size else np.nan,
             "nonfinite_frac": float((~fin).mean()), "unique_frac": float(len(np.unique(np.round(f, 6))) / max(len(f), 1)),
             "collapsed": bool(f.size and f.std() < 1e-5 * (abs(f.mean()) + 1e-3))}
        if ys is not None and k.startswith("y_") and f"{k}_reg" in ys:
            yt = np.asarray(ys[f"{k}_reg"], dtype=np.float64).reshape(-1)
            r["corr_with_target"] = _spearman(a[:, 0], yt)
            r["std_ratio_to_target"] = float(f.std() / (yt.std() + 1e-12)) if f.size else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _diag_nig(model, out, ys, targets, tol):
    rows = []
    for t in targets:
        if f"y_{t}_nu" not in out:
            continue
        nu, al, be = (out[f"y_{t}_{k}"].reshape(-1).astype(np.float64) for k in ("nu", "alpha", "beta"))
        head, unc = _head_layer(model, "nig", t), _head_layer(model, "unc", t)
        cfg = getattr(head, "cfg", {}) or {}
        ale, epi = out[f"y_{t}_aleatoric"].reshape(-1), out[f"y_{t}_epistemic"].reshape(-1)
        near = lambda v, fl: float(np.mean(v - fl <= tol * fl)) if fl else float("nan")  # noqa: E731
        cap = getattr(unc, "unc_max", None)
        r = {"target": t, "nu_floor_frac": near(nu, cfg.get("nu_min")), "alpha_floor_frac": near(al, cfg.get("alpha_min")),
             "beta_floor_frac": near(be, cfg.get("beta_min")), "aleatoric_mean": float(ale.mean()),
             "epistemic_mean": float(epi.mean()), "epi_over_ale": float(epi.mean() / (ale.mean() + 1e-12)),
             "unc_cap_frac": float(np.mean((ale >= 0.999 * cap) | (epi >= 0.999 * cap))) if cap else float("nan"),
             "nonfinite_frac": float(1 - np.isfinite(np.stack([nu, al, be])).mean())}
        if f"y_{t}_confidence" in out:
            c = out[f"y_{t}_confidence"].reshape(-1)
            r["confidence_std"], r["confidence_mean"] = float(c.std()), float(c.mean())
        if ys is not None and f"y_{t}_reg" in ys:
            yt = np.asarray(ys[f"y_{t}_reg"], dtype=np.float64).reshape(-1)
            err = np.abs(yt - out[f"y_{t}"].reshape(-1))
            tot = np.sqrt(ale ** 2 + epi ** 2)
            r.update(unc_err_spearman=_spearman(err, tot), coverage_1sigma=float(np.mean(err <= tot)),
                     aleatoric_over_target_std=float(np.median(ale) / (yt.std() + 1e-12)))
        rows.append(r)
    return pd.DataFrame(rows)


def _diag_classification(out, ys, targets, bins=10):
    rows = []
    for t in targets:
        k = f"y_{t}_class_logits"
        if k not in out:
            continue
        p = out[k].astype(np.float64)
        r = {"head": f"{t}_class", "n_classes": p.shape[-1] if p.ndim > 1 and p.shape[-1] > 1 else 2}
        if r["n_classes"] == 2:
            q = p.reshape(-1)
            pred = q >= 0.5
            r.update(pred_pos_share=float(pred.mean()), prob_mean=float(q.mean()), prob_std=float(q.std()),
                     saturated_frac=float(np.mean((q < 0.02) | (q > 0.98))))
            if ys is not None and f"y_{t}_class" in ys:
                lab = _label_bool(ys, t)
                conf = np.where(pred, q, 1 - q)
                acc = (pred == lab)
                edges = np.linspace(0.5, 1.0, bins + 1)
                ece = 0.0
                for lo, hi in zip(edges[:-1], edges[1:]):
                    m = (conf >= lo) & (conf <= hi if hi == 1.0 else conf < hi)
                    if m.any():
                        ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
                r.update(label_pos_share=float(lab.mean()), share_gap=float(abs(pred.mean() - lab.mean())),
                         accuracy=float(acc.mean()), majority_baseline=float(max(lab.mean(), 1 - lab.mean())),
                         auc=_auc(lab, q), ece=float(ece), brier=float(np.mean((q - lab) ** 2)))
        else:
            pred = p.argmax(-1)
            share = np.bincount(pred, minlength=p.shape[-1]) / len(pred)
            r.update(pred_share=np.round(share, 3).tolist(), prob_std=float(p.max(-1).std()),
                     saturated_frac=float(np.mean(p.max(-1) > 0.98)))
            if ys is not None and f"y_{t}_class" in ys:
                lab = np.asarray(ys[f"y_{t}_class"]).reshape(-1).astype(int)
                ls = np.bincount(lab, minlength=p.shape[-1]) / len(lab)
                r.update(label_share=np.round(ls, 3).tolist(), share_gap=float(np.abs(share - ls).max()),
                         accuracy=float(np.mean(pred == lab)), majority_baseline=float(ls.max()))
        rows.append(r)
    return pd.DataFrame(rows)


def _diag_sensitivity(model, X, ys, targets, out, method, feature_names, batch_size, seed):
    obj = _objectives(set(out), targets)
    rows = []
    if method == "grad_x_input":
        attr = _input_attribution(model, X, obj, batch_size)
        for h, d in attr.items():
            for key, imp in d.items():
                m = imp.mean(0)
                share = m / (m.sum() + 1e-30)
                names = _feature_names_for(feature_names, key, len(m))
                for j, (nm, s, v) in enumerate(zip(names, share, m)):
                    rows.append({"head": h, "input": key, "feature": nm, "importance": float(v), "share": float(s)})
    else:
        rng = np.random.default_rng(seed)
        n = _n_of(X)
        base = _perm_metric(out, ys, obj)
        for key in (list(X) if isinstance(X, dict) else [None]):
            arr = X[key] if key else X
            if not np.issubdtype(np.asarray(arr).dtype, np.floating):
                continue
            names = _feature_names_for(feature_names, key, arr.shape[-1])
            drops = {h: [] for h in obj}
            for j in range(arr.shape[-1]):
                Xp = {k: v.copy() for k, v in X.items()} if isinstance(X, dict) else X.copy()
                tgt = Xp[key] if key else Xp
                tgt[:, :, j] = tgt[rng.permutation(n), :, j]
                met = _perm_metric(_predict(model, Xp, batch_size), ys, obj)
                for h in obj:
                    drops[h].append(max(base[h] - met[h], 0.0))
            for h, dv in drops.items():
                dv = np.asarray(dv)
                for nm, v in zip(names, dv):
                    rows.append({"head": h, "input": key or "input", "feature": nm, "importance": float(v),
                                 "share": float(v / (dv.sum() + 1e-30))})
    return pd.DataFrame(rows, columns=["head", "input", "feature", "importance", "share"])


def _perm_metric(out, ys, obj):
    """مقياس «الأعلى أفضل» لكل رأس: AUC للتصنيف، ارتباط سبيرمان (IC) للانحدار."""
    res = {}
    for h, key in obj.items():
        t = h.rsplit("_", 1)[0]
        if h.endswith("_class") and f"y_{t}_class" in ys:
            res[h] = _auc(_label_bool(ys, t), out[key].reshape(-1))
        elif h.endswith("_reg") and f"y_{t}_reg" in ys:
            res[h] = _spearman(out[key].reshape(-1), np.asarray(ys[f"y_{t}_reg"], dtype=np.float64).reshape(-1))
        else:
            res[h] = float("nan")
    return res


def _diag_samples(out, ys, targets, top_k):
    """عيّنات مشبوهة لكل رأس: (واثقة وخاطئة) للتصنيف، (|الخطأ|/عدم اليقين) الأكبر للانحدار، مع epistemic."""
    rows = []
    for t in targets:
        if f"y_{t}_class_logits" in out and f"y_{t}_class" in ys and out[f"y_{t}_class_logits"].shape[-1] == 1:
            q = out[f"y_{t}_class_logits"].reshape(-1)
            lab = _label_bool(ys, t)
            wrong_conf = np.where((q >= 0.5) != lab, np.abs(q - 0.5) * 2, 0.0)
            for i in np.argsort(-wrong_conf)[:top_k]:
                if wrong_conf[i] > 0:
                    rows.append({"head": f"{t}_class", "row": int(i), "kind": "confident_wrong", "score": float(wrong_conf[i])})
        if f"y_{t}_nu" in out and f"y_{t}_reg" in ys:
            yt = np.asarray(ys[f"y_{t}_reg"], dtype=np.float64).reshape(-1)
            z = np.abs(yt - out[f"y_{t}"].reshape(-1)) / (np.sqrt(out[f"y_{t}_aleatoric"].reshape(-1) ** 2
                                                               + out[f"y_{t}_epistemic"].reshape(-1) ** 2) + 1e-12)
            for i in np.argsort(-z)[:top_k]:
                rows.append({"head": f"{t}_reg", "row": int(i), "kind": "large_standardized_residual", "score": float(z[i])})
            epi = out[f"y_{t}_epistemic"].reshape(-1)
            for i in np.argsort(-epi)[:top_k]:
                rows.append({"head": f"{t}_reg", "row": int(i), "kind": "high_epistemic", "score": float(epi[i])})
    return pd.DataFrame(rows, columns=["head", "row", "kind", "score"])


# ── حكم موحّد ✅/⚠️/🚨 ───────────────────────────────────────────────────────────────────────────────────
HEALTH_THRESHOLDS = dict(
    floor_warn=0.3, floor_bad=0.7, cap_warn=0.05, share_gap_warn=0.2, share_gap_bad=0.35, saturated_warn=0.5, ece_warn=0.1,
    dead_warn=0.2, dead_bad=0.5, rank_warn=0.05, entropy_low=0.1, entropy_high=0.98, sens_top_warn=0.5, sens_top_bad=0.8,
    dominance_warn=10.0, dominance_bad=100.0, conflict_warn=0.5, group_ratio_warn=1e-3, update_low=1e-5, update_high=1e-1,
    spike_z_warn=30.0, harmful_frac_warn=0.5, hard_frac_warn=0.5, noise_warn=0.2, std_ratio_low=0.02, std_ratio_high=20.0)


def model_health_verdicts(rep, recorder_stats=None, thresholds=None):
    """يحوّل مخرج diagnose_model (+ stats() لمسجّل التدريب اختياري) إلى جدول حكم: check | status | detail | action.
    thresholds: تجاوز جزئي لـ HEALTH_THRESHOLDS؛ مفتاح غير معروف ← ValueError."""
    th = dict(HEALTH_THRESHOLDS)
    if thresholds:
        unknown = set(thresholds) - set(th)
        if unknown:
            raise ValueError(f"عتبات غير معروفة: {sorted(unknown)} — المتاح: {sorted(th)}")
        th.update(thresholds)
    rows = []

    def add(check, status, detail, action=""):
        rows.append({"check": check, "status": status, "detail": detail, "action": action if status != "✅" else ""})

    h = rep.get("heads")
    if h is not None and len(h):
        nf = h[h["nonfinite_frac"] > 0]
        add("مخارج منتهية", "🚨" if len(nf) else "✅", f"غير منتهية في {list(nf['output'])[:4]}" if len(nf) else "كل المخارج منتهية",
            "افحص التطبيع/القيم الشاذة في X وتدرّجات الانفجار")
        col = h[h["collapsed"] & ~h["output"].str.endswith(("_nu", "_alpha", "_beta", "_confidence", "_aleatoric", "_epistemic"))]
        add("انهيار مخرج (ثابت)", "🚨" if len(col) else "✅", f"ثابت: {list(col['output'])[:4]}" if len(col) else "المخارج الأساسية متغيّرة",
            "الرأس يتنبّأ بثابت: افحص المعدّل التعلّمي/وزن المهمة/توازن التسميات")
        if "std_ratio_to_target" in h:
            r = h.dropna(subset=["std_ratio_to_target"])
            r = r[~r["output"].str.contains("_nu|_alpha|_beta|_conf|_alea|_epis")]
            lo = r[r["std_ratio_to_target"] < th["std_ratio_low"]]
            hi = r[r["std_ratio_to_target"] > th["std_ratio_high"]]
            add("مقياس تنبؤ الانحدار", "⚠️" if len(lo) or len(hi) else "✅",
                (f"تنبؤ أضيق بكثير من الهدف: {list(lo['output'])}" if len(lo) else "")
                + (f" أوسع بكثير: {list(hi['output'])}" if len(hi) else "") or "نسبة الانحراف المعياري معقولة",
                "راجع reg_target_scale ومقياس الأهداف، أو الإفراط/نقص التكيّف")
    n = rep.get("nig")
    if n is not None and len(n):
        fl = n[["nu_floor_frac", "alpha_floor_frac", "beta_floor_frac"]].max(axis=1)
        w = n.assign(fl=fl)
        worst = w.loc[w["fl"].idxmax()]
        st = "🚨" if worst["fl"] > th["floor_bad"] else "⚠️" if worst["fl"] > th["floor_warn"] else "✅"
        add("اصطدام NIG بالأرضيات", st, f"{worst['target']}: حتى {worst['fl']:.0%} من العيّنات عند أرضية ν/α/β",
            "اخفض alpha_min/beta_min أو افحص مقياس الهدف (الأرضية أكبر من مقياس الإشارة؟)")
        cap = n["unc_cap_frac"].max()
        add("سقف عدم اليقين", "⚠️" if cap > th["cap_warn"] else "✅", f"{cap:.1%} من العيّنات عند unc_max", "عيّنات خارج التوزيع أو β منفجر")
        if "unc_err_spearman" in n:
            c = n.dropna(subset=["unc_err_spearman"])
            if len(c):
                worst = c.loc[c["unc_err_spearman"].idxmin()]
                add("معايرة عدم اليقين", "⚠️" if worst["unc_err_spearman"] <= 0.0 else "✅",
                    f"{worst['target']}: ارتباط |الخطأ| بعدم اليقين = {worst['unc_err_spearman']:.2f}، تغطية 1σ = {worst['coverage_1sigma']:.0%}",
                    "عدم اليقين لا يتتبّع الخطأ: لا تستعمله للتصفية قبل المعايرة")
    c = rep.get("classification")
    if c is not None and len(c):
        if "share_gap" in c:
            cc = c.dropna(subset=["share_gap"])
            if len(cc):
                worst = cc.loc[cc["share_gap"].idxmax()]
                st = "🚨" if worst["share_gap"] > th["share_gap_bad"] else "⚠️" if worst["share_gap"] > th["share_gap_warn"] else "✅"
                add("توازن الفئات المتوقَّعة", st, f"{worst['head']}: حصّة الإيجابي المتوقَّعة {worst.get('pred_pos_share', np.nan):.0%} "
                    f"مقابل التسميات {worst.get('label_pos_share', np.nan):.0%}", "انحياز لفئة: افحص pos_weight/تنعيم التسميات/انزياح التوزيع")
        sat = c["saturated_frac"].max()
        add("تشبّع احتمالات التصنيف", "⚠️" if sat > th["saturated_warn"] else "✅", f"حتى {sat:.0%} من الاحتمالات < 0.02 أو > 0.98",
            "ثقة مفرطة: زِد التنعيم/التنظيم")
        if "ece" in c:
            e = c.dropna(subset=["ece"])
            if len(e):
                worst = e.loc[e["ece"].idxmax()]
                add("معايرة احتمالات التصنيف", "⚠️" if worst["ece"] > th["ece_warn"] else "✅", f"{worst['head']}: ECE = {worst['ece']:.3f}",
                    "طبّق معايرة post-hoc (Platt/isotonic) على val")
    a = rep.get("activations")
    if a is not None and len(a):
        nf = a[a["nonfinite_frac"] > 0]
        add("تنشيطات غير منتهية", "🚨" if len(nf) else "✅", f"في {list(nf['layer'])[:3]}" if len(nf) else "لا قيم غير منتهية", "أول طبقة تظهر فيها هي المصدر")
        d = a[a["stage"] != "input_norm"]
        worst = d.loc[d["dead_unit_frac"].idxmax()]
        st = "🚨" if worst["dead_unit_frac"] > th["dead_bad"] else "⚠️" if worst["dead_unit_frac"] > th["dead_warn"] else "✅"
        add("وحدات ميتة", st, f"{worst['layer']}: {worst['dead_unit_frac']:.0%} من الوحدات لا تنشط أبداً", "اخفض LR/غيّر التهيئة/أزل الطبقة")
        rk = a.dropna(subset=["eff_rank_ratio"])
        rk = rk[(rk["dim"] >= 8) & (rk["stage"].isin(["encoder_block", "encoder_out", "readout", "trunk"]))]
        if len(rk):
            worst = rk.loc[rk["eff_rank_ratio"].idxmin()]
            add("انهيار بُعد التمثيل", "⚠️" if worst["eff_rank_ratio"] < th["rank_warn"] else "✅",
                f"{worst['layer']}: الرتبة الفعّالة = {worst['eff_rank_ratio']:.1%} من البُعد", "تمثيل ضيّق: خفّض الانتظام أو افحص تطبيع المدخل")
    at = rep.get("attention")
    if at is not None and len(at):
        bad = at[(at["entropy_ratio"] < th["entropy_low"]) | (at["entropy_ratio"] > th["entropy_high"])]
        add("انتباه منهار/منتظم", "⚠️" if len(bad) > 0.5 * len(at) else "✅", f"{len(bad)}/{len(at)} رأس انتباه إنتروبياه متطرّفة",
            "الانتباه لا يميّز مواضع: افحص انحياز الموضع النسبي والتطبيع")
        sy = at["sync_err"].max()
        add("تطابق مسبار الانتباه", "⚠️" if sy > 1e-3 else "✅", f"أقصى فرق مع الطبقة {sy:.1e}", "تغيّرت RelativeGQAttention: حدّث _attn_probs")
    s = rep.get("sensitivity")
    if s is not None and len(s):
        top = s.groupby("head")["share"].max()
        h_, v_ = top.idxmax(), float(top.max())
        row = s[(s["head"] == h_)].sort_values("share", ascending=False).iloc[0]
        st = "🚨" if v_ > th["sens_top_bad"] else "⚠️" if v_ > th["sens_top_warn"] else "✅"
        add("اعتماد على ميزة واحدة", st, f"{h_}: {row['feature']} ({row['input']}) = {v_:.0%} من الحساسية", "ميزة هيمنت: افحص تسرّب/قناة حفظ")
    if recorder_stats:
        r = recorder_stats
        if "head_dominance_median" in r:
            d = r["head_dominance_median"]
            add("هيمنة رأس على الجذع", "🚨" if d > th["dominance_bad"] else "⚠️" if d > th["dominance_warn"] else "✅",
                f"وسيط نسبة أكبر/أصغر نظيم تدرّج رأس = {d:.1f} (المهيمن: {r.get('dominant_head', '?')})", "غيّر loss_weight أو فعّل Kendall/GradNorm")
        if "conflict_frac" in r:
            f = r["conflict_frac"]
            add("تضارب تدرّجات الرؤوس", "⚠️" if f > th["conflict_warn"] else "✅", f"{f:.0%} من أزواج القياسات cos<0 (أسوأ زوج: {r.get('conflict_pair', '?')})",
                "مهام تتصارع على الجذع: افصلها أو استعمل PCGrad")
        if r.get("nonfinite_grad", 0) or r.get("nonfinite_loss_batches", 0):
            add("تدرّجات/خسائر غير منتهية", "🚨", f"{r.get('nonfinite_grad', 0)} قياس تدرّج و{r.get('nonfinite_loss_batches', 0)} دفعة", "قصّ التدرّج/خفض LR")
        if "group_grad_ratio" in r:
            g = r["group_grad_ratio"]
            add("تلاشي/انفجار بين الطبقات", "⚠️" if g < th["group_ratio_warn"] else "✅",
                f"أضعف مجموعة ({r.get('weakest_group')}) = {g:.1e} من أقواها ({r.get('strongest_group')})", "تدرّجات غير متوازنة: افحص التهيئة/التطبيع")
        if "update_ratio_median" in r:
            lo, hi = r["update_ratio_min"], r["update_ratio_max"]
            st = "⚠️" if (hi > th["update_high"] or r["update_ratio_median"] < th["update_low"]) else "✅"
            add("نسبة التحديث/الوزن", st, f"وسيط {r['update_ratio_median']:.1e} (أدنى {r.get('update_ratio_min_group')} {lo:.1e}، أعلى {r.get('update_ratio_max_group')} {hi:.1e})",
                "≈1e-3 سليمة؛ أعلى بكثير = غير مستقر، أدنى بكثير = لا تتعلّم")
        if "loss_spike_z" in r:
            z = r["loss_spike_z"]
            add("قفزات الخسارة", "⚠️" if z > th["spike_z_warn"] else "✅", f"أكبر قفزة = {z:.0f}×MAD", "دفعات شاذة: راجع top_batches")
        if "influence_harmful_frac" in r:
            hf = r["influence_harmful_frac"]
            add("تأثير الدفعات على val", "⚠️" if hf > th["harmful_frac_warn"] else "✅",
                f"{hf:.0%} من الدفعات المقاسة تضرّ بخسارة probe (حصّة أكبر {r.get('n_influence_measures')} قياس: top-k = {r.get('influence_topk_abs_share', float('nan')):.0%} من |التأثير|)",
                "التدريب لا يتوافق مع val: افحص التسريب/انزياح التوزيع/تضارب الأهداف")
        if "cartography_hard_frac" in r:
            nz = r.get("noise_suspect_frac", 0.0)
            st = "⚠️" if (nz > th["noise_warn"] or r["cartography_hard_frac"] > th["hard_frac_warn"]) else "✅"
            add("خريطة البيانات", st, f"hard={r['cartography_hard_frac']:.0%}، ambiguous={r.get('cartography_ambiguous_frac', 0):.0%}، تسمية مشبوهة={nz:.0%}",
                "افحص hardest_samples (تسميات خاطئة محتملة) قبل الحكم على النموذج")
    out = pd.DataFrame(rows, columns=["check", "status", "detail", "action"])
    return out


def print_verdicts(v, title="🩺 تقرير صحّة النموذج"):
    """يطبع جدول الحكم بأسلوب دوال التدقيق: الأسوأ أولاً."""
    order = {"🚨": 0, "⚠️": 1, "✅": 2}
    v = v.assign(_o=v["status"].map(order)).sort_values("_o", kind="stable").drop(columns="_o")
    print(f"{title}: 🚨 {int((v['status'] == '🚨').sum())} | ⚠️ {int((v['status'] == '⚠️').sum())} | ✅ {int((v['status'] == '✅').sum())}")
    for _, r in v.iterrows():
        print(f" {r['status']} {r['check']}: {r['detail']}" + (f"  ← {r['action']}" if r["action"] else ""))

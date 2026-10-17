"""
PURPOSE:  TASK_REGISTRY and the loss functions (evidential NIG, regression, classification, confidence calibration).
TAGS:     TASK_REGISTRY, register_task_type, evidential_task_loss, regression_task_loss, classification_task_loss,
          nig_regularizer, loss
PITFALLS: An unregistered task_type raises. NIG loss is numerically fragile (log/lgamma): losses cast to float32.
          Executed into the one shared trainer namespace by trainer/_loader.py (never imported on its own): names from
          other modules resolve at call time.

## 3) سجلّ دوال الخسارة (Loss Registry) — قابل للتوسعة بالكامل

كل "نوع مهمة" (`task_type`) هو ببساطة:

```
دالة بتوقيع  (y_true, outputs, target_cfg, scheduled_vars) -> (loss_scalar, stats_dict)
+ قائمة بأسماء الإحصائيات (stat_keys) التي تُرجعها الدالة دائمًا (نفس المفاتيح
  في كل استدعاء، بلا شرط) — تُستخدم لإنشاء metrics التتبّع تلقائيًا عند بناء المدرّب.
```

**لإضافة نوع مهمة جديد بالكامل** (مثلاً quantile regression):
```python
def quantile_task_loss(y_true, outputs, target_cfg, scheduled_vars):
    ...
    return loss, {"pinball": ...}

register_task_type("quantile", quantile_task_loss, ["pinball"])
```
ثم استخدم `"task_type": "quantile"` في `config['targets'][...]` — لن تحتاج
لتعديل `GenericTrainer` إطلاقًا.

## 3) سجلّ دوال الخسارة (Loss Registry) — قابل للتوسعة بالكامل

كل "نوع مهمة" (`task_type`) هو ببساطة:

```
دالة بتوقيع  (y_true, outputs, target_cfg, scheduled_vars) -> (loss_scalar, stats_dict)
+ قائمة بأسماء الإحصائيات (stat_keys) التي تُرجعها الدالة دائمًا (نفس المفاتيح
  في كل استدعاء، بلا شرط) — تُستخدم لإنشاء metrics التتبّع تلقائيًا عند بناء المدرّب.
```

**لإضافة نوع مهمة جديد بالكامل** (مثلاً quantile regression):
```python
def quantile_task_loss(y_true, outputs, target_cfg, scheduled_vars):
    ...
    return loss, {"pinball": ...}

register_task_type("quantile", quantile_task_loss, ["pinball"])
```
ثم استخدم `"task_type": "quantile"` في `config['targets'][...]` — لن تحتاج
لتعديل `GenericTrainer` إطلاقًا.

### 🆕 خيارات evidential إضافية (اختيارية، الافتراضي = السلوك القديم)
تُضبط داخل `config['targets']['<اسم الهدف>']`:
- `normalize_reg: True` — يقسم الباقي في منظِّم الأدلة على عرض توزيع Student-t (`w_St`) بدل استخدامه خامًا
  (Meinert et al. 2023، معادلة 11). يقلّل تسرّب عدم اليقين الإحصائي (aleatoric) إلى تقدير عدم اليقين
  المعرفي (epistemic) عندما تتفاوت شدة الضجيج عبر العينات — وهذا بالضبط نمط بيانات الأسعار.
- `beta_nll: 0.5` — يرجّح NLL بـ `w_St^(2·beta_nll)` (امتداد لفكرة Seitzer et al. 2022 المصمَّمة أصلًا
  لـ Gaussian NLL؛ تطبيقها على NIG هنا اجتهاد غير موثَّق في الأدبيات — قيّمها تجريبيًا).
- `huber_weight: 0.1` — يضيف خسارة Huber مباشرة على `mu` فلا يستطيع النموذج «تفسير» خطأ التوقع بتضخيم
  عدم اليقين بدل تحسين الدقة.
- `alpha_floor_straight_through: True` — القصّ القديم `max(alpha, 1.0)` يقتل التدرّج تمامًا حين تكون
  `alpha<1` (شائع مع رأس `softplus` وحده). هذا الخيار يُبقي القيمة الأمامية كما هي لكنه يُمرّر التدرّج
  (straight-through estimator) بدل قتله — بديل أنظف هو تغيير رأس النموذج نفسه إلى `1+softplus(x)`.
- دالة مستقلة `nig_uncertainties_meinert(nu, alpha, beta)` تُعيد تعريف Meinert لعدم اليقين
  (aleatoric=عرض Student-t، epistemic=1/√ν) — **للاستدلال والتحليل فقط**، لم تُدمَج في الخسارة أو في
  إحصاءات التدريب المحفوظة، حتى تبقى checkpoints ونماذج قديمة صالحة دون تغيير في معناها.
"""
# @title 3) سجلّ دوال الخسارة (Loss Registry)
TASK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_task_type(name: str, fn: Callable, stat_keys: List[str], supports_sample_weight: bool = False):
    """سجّل نوع مهمة جديد.
    fn: (y_true, outputs, target_cfg, scheduled_vars) -> (loss, stats_dict)
    إن أعلنتَ supports_sample_weight=True فيجب أن تقبل fn وسيطًا إضافيًا sample_weight=None
    (وإلا فإعطاء sample_weight لهدف من هذا النوع يرفع خطأً صريحًا بدل أن يُتجاهل بصمت)."""
    TASK_REGISTRY[name] = {"fn": fn, "stat_keys": list(stat_keys),
                           "supports_sample_weight": bool(supports_sample_weight)}


# ───────────────────────────── مساعدات الأوزان ─────────────────────────────

def _flat(x):
    return tf.reshape(tf.cast(x, tf.float32), [-1])


def _wmean(x, w=None):
    """متوسط مرجَّح Σ(w·x)/Σw — لا يعتمد على مقياس الأوزان، ويُرجع 0 (لا NaN) إن كانت كلها أصفارًا.
    مع w=None يساوي reduce_mean تمامًا."""
    x = tf.cast(x, tf.float32)
    if w is None:
        return tf.reduce_mean(x)
    w = tf.cast(w, tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(x * w), tf.reduce_sum(w))


def _wstd(x, w=None):
    if w is None:
        return tf.math.reduce_std(x)
    m = _wmean(x, w)
    return tf.sqrt(_wmean(tf.square(x - m), w) + 1e-12)


# ───────────────────────────── evidential (NIG) ─────────────────────────────

def compute_confidence_calibration_loss(y_true, y_pred, confidence, sample_weight=None):
    y_true = _flat(y_true)
    y_pred = _flat(y_pred)
    confidence = _flat(confidence)
    w = None if sample_weight is None else _flat(sample_weight)

    # قطع تدرّج y_pred يمنع calibration loss من التأثير مباشرة على mu (Bug #2 الموثّق سابقًا)
    y_pred_detached = tf.stop_gradient(y_pred)
    abs_error = tf.abs(y_true - y_pred_detached)
    batch_mean = _wmean(abs_error, w)
    batch_std = _wstd(abs_error, w)
    max_expected_error = tf.stop_gradient(batch_mean + 1.5 * batch_std)
    normalized_error = tf.clip_by_value(abs_error / (max_expected_error + 1e-8), 0.0, 1.0)
    target_confidence = 1.0 - normalized_error

    diff = confidence - target_confidence
    focal_weight = tf.pow(tf.abs(diff), 2.0)
    focal_calibration_loss = _wmean(focal_weight * tf.square(diff), w)

    conf_std = _wstd(confidence, w)
    diversity_penalty = tf.maximum(0.0, 0.15 - conf_std)
    saturation_penalty = _wmean(
        tf.maximum(0.0, confidence - 0.95) + tf.maximum(0.0, 0.05 - confidence), w
    )
    return focal_calibration_loss + 0.5 * diversity_penalty + 0.3 * saturation_penalty


def nig_w_st(nu, alpha, beta):
    """عرض توزيع Student-t الناتج عن NIG: w_St = sqrt(beta(1+nu)/(nu*alpha))  (Meinert et al. 2023, Eq. 9)"""
    return tf.sqrt(beta * (1.0 + nu) / (nu * alpha))


def nig_regularizer(error, nu, alpha, beta, normalize=False, stop_grad=True):
    """منظِّم الأدلة:
       normalize=False → |y-γ|·(2ν+α)          (Amini et al. 2020 — السلوك القديم)
       normalize=True  → |y-γ|/w_St ·(2ν+α)    (Meinert et al. 2023, Eq. 11 عند p=1)
    stop_grad: قطع التدرج عن w_St. الورقة لا تحدد ذلك — هذا اختيار تصميم كي لا يستطيع النموذج
    تصغير المنظِّم بتضخيم w_St وحده."""
    resid = tf.abs(error)
    if normalize:
        w_st = nig_w_st(nu, alpha, beta)
        if stop_grad:
            w_st = tf.stop_gradient(w_st)
        resid = resid / (w_st + 1e-8)
    return resid * (2.0 * nu + alpha)


def nig_uncertainties_meinert(nu, alpha, beta) -> Dict[str, tf.Tensor]:
    """إعادة تعريف Meinert et al. 2023 (Eq. 10): aleatoric = w_St (بوحدات y)، epistemic = 1/sqrt(nu) (بلا وحدات).
    للاستدلال/التحليل فقط — لا تدخل في الخسارة ولا في metrics التدريب (كي تبقى checkpoints القديمة صالحة)."""
    nu = tf.maximum(tf.cast(nu, tf.float32), 1e-6)
    alpha = tf.maximum(tf.cast(alpha, tf.float32), 1e-6)
    beta = tf.cast(beta, tf.float32)
    return {"aleatoric_wst": nig_w_st(nu, alpha, beta), "epistemic_inv_sqrt_nu": 1.0 / tf.sqrt(nu)}


def _floor_alpha(alpha, target_cfg):
    """alpha<1 مستحيل في NIG السليم (E[σ²]=β/(α-1)). القصّ عند 1.0 هو السلوك القديم لكنه يقتل التدرج
    عن alpha عندما يكون <1 (يحدث كثيرًا إن كان رأس alpha هو softplus وحده). البديل الأفضل هو 1+softplus داخل
    النموذج (Meinert Eq. 6). خيار straight-through هنا يُبقي القيمة الأمامية كما هي ويمرّر التدرج."""
    clamped = tf.maximum(alpha, 1.0)
    if target_cfg.get("alpha_floor_straight_through", False):
        return alpha + tf.stop_gradient(clamped - alpha)
    return clamped


def evidential_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    keys = target_cfg["output_keys"]
    w = None if sample_weight is None else _flat(sample_weight)
    y_true = _flat(y_true)
    gamma = _flat(outputs[keys["mu"]])
    nu = _flat(outputs[keys["nu"]])
    alpha = _flat(outputs[keys["alpha"]])
    beta = _flat(outputs[keys["beta"]])

    error = y_true - gamma
    omega = tf.maximum(2.0 * beta * (1.0 + nu), 1e-6)
    nu = tf.maximum(nu, 1e-6)
    alpha = _floor_alpha(alpha, target_cfg)

    nig_base = (
        0.5 * tf.math.log(np.pi / nu)
        - alpha * tf.math.log(omega)
        + (alpha + 0.5) * tf.math.log(nu * tf.square(error) + omega)
        + tf.math.lgamma(alpha) - tf.math.lgamma(alpha + 0.5)
    )
    nig_pen = nig_regularizer(
        error, nu, alpha, beta,
        normalize=bool(target_cfg.get("normalize_reg", False)),
        stop_grad=bool(target_cfg.get("normalize_reg_stop_grad", True)),
    )

    # β-NLL (Seitzer et al. 2022 — للـ Gaussian؛ تطبيقه على NIG امتداد يحتاج مقارنة عملية):
    # نستخدم w_St² كتباين وليس β/(α-1): الأخير ينفجر عند α→1 (وقيمة α بعد القصّ هي 1 بالضبط)،
    # بينما w_St² منتهٍ لكل α>0 وهو نفسه بديل الـ aleatoric عند Meinert.
    beta_nll = float(target_cfg.get("beta_nll", 0.0))
    nll_term = nig_base
    if beta_nll > 0.0:
        var = tf.stop_gradient(tf.square(nig_w_st(nu, alpha, beta)))
        nll_term = nig_base * tf.pow(var, beta_nll)

    lambda_reg_name = target_cfg.get("lambda_reg_var", "lambda_reg")
    lambda_reg = scheduled_vars[lambda_reg_name] if lambda_reg_name in scheduled_vars else tf.constant(0.0)

    per_sample = nll_term + tf.cast(lambda_reg, tf.float32) * nig_pen

    # Huber مباشر على mu: يعطي المتوسط إشارة لا تمرّ عبر عدم اليقين (لا يستطيع "تفسير" الخطأ بتضخيم β)
    huber_weight = float(target_cfg.get("huber_weight", 0.0))
    if huber_weight > 0.0:
        delta = float(target_cfg.get("huber_delta", 1.0))
        a = tf.abs(error)
        q = tf.minimum(a, delta)
        per_sample = per_sample + huber_weight * (0.5 * tf.square(q) + delta * (a - q))

    total = _wmean(per_sample, w)

    aleatoric = beta / (alpha - 1.0 + 1e-6)
    epistemic = beta / (nu * (alpha - 1.0) + 1e-6)
    total_unc = aleatoric + epistemic
    confidence_est = 1.0 / (1.0 + total_unc)

    stats = {
        "nig_base": _wmean(nig_base, w),
        "nig_pen": _wmean(nig_pen, w),
        "mae": _wmean(tf.abs(error), w),
        "aleatoric": _wmean(aleatoric, w),
        "epistemic": _wmean(epistemic, w),
        "confidence": _wmean(confidence_est, w),
        "calib_loss": tf.constant(0.0),
    }

    if target_cfg.get("use_calibration_loss", False) and "confidence" in keys:
        calib_loss = compute_confidence_calibration_loss(y_true, gamma, outputs[keys["confidence"]], sample_weight=w)
        lambda_calib_name = target_cfg.get("lambda_calib_var", "lambda_calib")
        lambda_calib = scheduled_vars[lambda_calib_name] if lambda_calib_name in scheduled_vars else tf.constant(0.0)
        total = total + calib_loss * tf.cast(lambda_calib, tf.float32)
        stats["calib_loss"] = calib_loss

    return total, stats


register_task_type(
    "evidential", evidential_task_loss,
    ["nig_base", "nig_pen", "mae", "aleatoric", "epistemic", "confidence", "calib_loss"],
    supports_sample_weight=True,
)


# ───────────────────────────── regression بسيطة ─────────────────────────────

def regression_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    keys = target_cfg["output_keys"]
    w = None if sample_weight is None else _flat(sample_weight)
    y_true = _flat(y_true)
    y_pred = _flat(outputs[keys["pred"]])
    loss_kind = target_cfg.get("loss_fn", "mse")

    error = y_true - y_pred
    if loss_kind == "mae":
        per = tf.abs(error)
    elif loss_kind == "huber":
        delta = float(target_cfg.get("huber_delta", 1.0))
        a = tf.abs(error)
        q = tf.minimum(a, delta)
        per = 0.5 * tf.square(q) + delta * (a - q)
    else:  # mse (افتراضي)
        per = tf.square(error)

    return _wmean(per, w), {"mae": _wmean(tf.abs(error), w)}


register_task_type("regression", regression_task_loss, ["mae"], supports_sample_weight=True)


# ───────────────────────────── classification ─────────────────────────────

def classification_task_loss(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    keys = target_cfg["output_keys"]
    logits = outputs[keys["logits"]]
    w = None if sample_weight is None else _flat(sample_weight)

    ls = float(target_cfg.get("label_smoothing", 0.0))
    if target_cfg.get("binary", True):
        y_true_f = _flat(y_true)
        logits_f = _flat(logits)
        # [B,1] لنحصل على خسارة لكل عيّنة (على [B] تُختزل Keras المحور الأخير فتعطي رقمًا واحدًا)
        y_loss = y_true_f * (1.0 - ls) + 0.5 * ls if ls > 0 else y_true_f
        per = tf.keras.losses.binary_crossentropy(y_loss[:, None], logits_f[:, None])
        pred_label = tf.cast(logits_f >= 0.5, tf.float32)
        acc = _wmean(tf.cast(tf.equal(pred_label, y_true_f), tf.float32), w)   # الدقة على التسمية الصلبة دائمًا
    else:
        y_true_i = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        if ls > 0:
            k = tf.shape(logits)[-1]
            soft = tf.one_hot(y_true_i, k) * (1.0 - ls) + ls / tf.cast(k, tf.float32)
            per = tf.keras.losses.categorical_crossentropy(soft, logits)
        else:
            per = tf.keras.losses.sparse_categorical_crossentropy(y_true_i, logits)
        pred_label = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        acc = _wmean(tf.cast(tf.equal(pred_label, y_true_i), tf.float32), w)

    return _wmean(per, w), {"accuracy": acc}


register_task_type("classification", classification_task_loss, ["accuracy"], supports_sample_weight=True)

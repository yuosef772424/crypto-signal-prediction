"""
PURPOSE:  Raw NIG head layers: NIGHead, OrderedMeans, NIGUncertainty, ConfidenceHead (+ NIG_ALPHA_DEN_MIN).
TAGS:     NIGHead, OrderedMeans, NIGUncertainty, ConfidenceHead, NIG_ALPHA_DEN_MIN, nu, alpha, beta, evidential
PITFALLS: NIG outputs must stay float32 under mixed precision (lgamma/log are numerically fragile in float16).
          Executed into the one shared model namespace by model/_loader.py (never imported on its own): names from
          other modules resolve at call time.

## 6) رؤوس NIG (بلا انتباه طول-1 منحطّ، أوساط مُرتَّبة، ثقة صادقة)

هذه الطبقات الخام كما هي — القسم التالي (٧) هو ما تغيّر: تحويلها إلى
"نوع رأس" مُسجَّل بدل استدعاء مباشر مُثبَّت بالكود.
"""
@register
class NIGHead(layers.Layer):
    def __init__(self, hidden=128, dropout=0.1, nu_min=0.1, alpha_min=2.0, beta_min=0.01,
                 l2=1e-5, **kw):
        super().__init__(**kw)
        self.cfg = dict(hidden=hidden, dropout=dropout, nu_min=nu_min, alpha_min=alpha_min,
                         beta_min=beta_min, l2=l2)
        self.fc = layers.Dense(hidden, activation="swish", kernel_regularizer=regularizers.l2(l2))
        self.drop = layers.Dropout(dropout)
        self.out = layers.Dense(4)

    def build(self, input_shape):
        self.fc.build(input_shape)
        self.out.build(tuple(input_shape[:-1]) + (self.cfg["hidden"],))
        super().build(input_shape)

    def call(self, h, training=None):
        p = self.out(self.drop(self.fc(h), training=training))
        mu = p[:, 0:1]
        nu = tf.nn.softplus(p[:, 1:2]) + self.cfg["nu_min"]
        alpha = tf.nn.softplus(p[:, 2:3]) + self.cfg["alpha_min"]
        beta = tf.nn.softplus(p[:, 3:4]) + self.cfg["beta_min"]
        return mu, nu, alpha, beta

    def get_config(self):
        return {**super().get_config(), **self.cfg}


@register
class OrderedMeans(layers.Layer):
    """mu_high = mu_close + softplus(a);  mu_low = mu_close - softplus(b)."""

    def call(self, inputs):
        mu_c, raw_h, raw_l = inputs
        return mu_c + tf.nn.softplus(raw_h), mu_c, mu_c - tf.nn.softplus(raw_l)


# أرضية مقام (alpha − 1) في عدم اليقين: لا أثر لها إطلاقاً حين alpha ≥ 1.01 (رأس NIG يضمن alpha ≥ alpha_min = 2 افتراضياً)،
# وتمنع الانفجار β/(α−1) → ∞ إن خُفِّض alpha_min نحو 1 أو جاءت alpha من نموذج آخر.
NIG_ALPHA_DEN_MIN = 1e-2


@register
class NIGUncertainty(layers.Layer):
    """تعريفات Amini et al.: aleatoric = √(β/(α−1))، epistemic = √(β/(ν(α−1))) — بحدّ أعلى مُعلَن.

    لماذا الحدّ: β = softplus(logit) + beta_min بلا سقف (softplus خطي للوجيتات الكبيرة)، وν ≥ nu_min = 0.1 تُكبِّر
    epistemic ×√10 على الأكثر؛ عيّنة خارج التوزيع بوجيت β كبير تُنتج عدم يقين بمئات الوحدات. ``unc_max`` (بوحدة الهدف
    المخرَج y_*_reg = عائد × reg_target_scale) يقصّ القيمتين عنده بدل أن تبلغا ∞/NaN: القيم الطبيعية (< ~0.5) لا تتغيّر
    إطلاقاً، وغير المنتهي (NaN/±∞) يصير unc_max. ``unc_max=None`` = بلا قصّ (السلوك القديم، مع أرضية المقام وحدها).
    الطبقة بلا أوزان وخارج الخسارة (الخسارة تقرأ ν/α/β مباشرة) فالقصّ لا يمسّ التدريب ولا حفظ النقاط."""

    def __init__(self, unc_max=None, **kw):
        super().__init__(**kw)
        self.unc_max = None if unc_max is None else float(unc_max)

    def call(self, inputs):
        nu, alpha, beta = inputs
        a1 = tf.maximum(alpha - 1.0, NIG_ALPHA_DEN_MIN)
        aleatoric = tf.sqrt(beta / a1)
        epistemic = tf.sqrt(beta / (nu * a1))
        if self.unc_max is not None:
            cap = tf.constant(self.unc_max, aleatoric.dtype)
            aleatoric = tf.minimum(tf.where(tf.math.is_finite(aleatoric), aleatoric, cap), cap)
            epistemic = tf.minimum(tf.where(tf.math.is_finite(epistemic), epistemic, cap), cap)
        return epistemic, aleatoric

    def get_config(self):
        return {**super().get_config(), "unc_max": self.unc_max}


@register
class ConfidenceHead(layers.Layer):
    """سلسة (بلا قصّ صلب -> بلا تدرّجات ميّتة)، مشروطة بمعاملات evidential،
    ومقطوعة التدرّج بالكامل (stop_gradient) لتضمن أنها لا تقود جذع التنبؤ أبداً."""

    def __init__(self, hidden=32, **kw):
        super().__init__(**kw)
        self.hidden = hidden
        self.d1 = layers.Dense(hidden, activation="relu")
        self.d2 = layers.Dense(1, activation="sigmoid")

    def build(self, input_shape):
        h_shape, nu_shape, alpha_shape, beta_shape = input_shape
        concat_dim = h_shape[-1] + nu_shape[-1] + alpha_shape[-1] + beta_shape[-1]
        self.d1.build(tuple(h_shape[:-1]) + (concat_dim,))
        self.d2.build(tuple(h_shape[:-1]) + (self.hidden,))
        super().build(input_shape)

    def call(self, inputs):
        h, nu, alpha, beta = inputs
        u = tf.concat([h, tf.math.log(nu), tf.math.log(alpha - 1.0), tf.math.log(beta)], axis=-1)
        return 0.05 + 0.93 * self.d2(self.d1(tf.stop_gradient(u)))

    def get_config(self):
        return {**super().get_config(), "hidden": self.hidden}

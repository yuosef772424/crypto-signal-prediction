"""
PURPOSE:  InstanceNorm (RevIN-style per-window normalisation with stats path) and SymLog level passthrough.
TAGS:     InstanceNorm, SymLog, RevIN, stats_mode, input_clip, level_passthrough, anti-memorization, input
          normalisation
PITFALLS: stats_mode 'full' is the old behaviour (default); symlog/none exist because the stats path is the only way
          raw feature levels reach the model. Executed into the one shared model namespace by model/_loader.py (never
          imported on its own): names from other modules resolve at call time.

## 1) تطبيع المدخل (RevIN-style, نصف المدخل فقط)
"""
@register
class InstanceNorm(layers.Layer):
    """تقييس لكل نافذة ولكل ميزة على حدة. تُرجع (z, stats) حيث stats =
    [mean, log std] لكل ميزة — بحيث لا يُفقَد مستوى التقلّب.
    بلا نصف إعادة تسوية: الأهداف أصلاً بلا وحدة قياس (نسبية للسعر الحالي).

    حارسان ضد الحفظ حين يكون تطبيع البيانات سيئاً (كلاهما معطَّل افتراضياً = السلوك القديم تماماً):

    * clip: قصّ ناعم z → clip·tanh(z/clip). z داخل النافذة محدود أصلاً بـ√(T−1)، لكن ميزة شبه ثابتة
      داخل النافذة تقفز مرّة واحدة تُعطي ±√(T−1) في خطوة واحدة (بصمة فريدة لتلك العيّنة).
    * stats_mode: مسار الإحصاءات هو **الطريق الوحيد** الذي يصل منه المستوى المطلق للميزة إلى النموذج
      (z يُلغيه). ميزة غير مُطبَّعة (سعر خام، حجم بالدولار، عمر العملة) تجعل mean فيه رقماً فريداً لكل
      عملة/فترة — معرِّفاً يحفظ به النموذج «أيّ عملة في أيّ يوم» بدل نمط قابل للتعميم (قيس فعلياً،
      docs/research/anti_memorization_pr7.md).
        "full"   : [mean, log std] كما هي (القديم).
        "symlog" : mean → sign·log1p|mean| و log std مقصوص في [-stats_clip, stats_clip]: يبقى مستوى
                   التقلّب ويُضغط أيّ مستوى خام لعدّة وحدات فقط بدل 10^5.
        "none"   : بلا مسار إحصاءات إطلاقاً (stats=None)."""

    STATS_MODES = ("full", "symlog", "none")

    def __init__(self, eps=1e-4, affine=True, clip=None, stats_mode="full", stats_clip=8.0, **kw):
        super().__init__(**kw)
        if stats_mode not in self.STATS_MODES:
            raise ValueError(f"stats_mode يجب أن يكون واحداً من {self.STATS_MODES}، لا {stats_mode!r}")
        self.eps, self.affine = eps, affine
        self.clip, self.stats_mode, self.stats_clip = clip, stats_mode, stats_clip

    def build(self, input_shape):
        f = int(input_shape[-1])
        if self.affine:
            self.gamma = self.add_weight(name="gamma", shape=(f,), initializer="ones")
            self.beta = self.add_weight(name="beta", shape=(f,), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=1, keepdims=True)
        std = tf.sqrt(var + self.eps)
        z = (x - mean) / std
        if self.clip:
            z = self.clip * tf.tanh(z / self.clip)
        if self.affine:
            z = z * self.gamma + self.beta
        m, log_s = tf.squeeze(mean, 1), tf.math.log(tf.squeeze(std, 1))
        if self.stats_mode == "symlog":
            m = tf.sign(m) * tf.math.log1p(tf.abs(m))
            log_s = tf.clip_by_value(log_s, -self.stats_clip, self.stats_clip)
        stats = tf.concat([m, log_s], axis=-1)
        return z, stats

    def compute_output_shape(self, input_shape):
        b, t, f = input_shape
        return (b, t, f), (b, 2 * f)

    def get_config(self):
        return {**super().get_config(), "eps": self.eps, "affine": self.affine, "clip": self.clip,
                "stats_mode": self.stats_mode, "stats_clip": self.stats_clip}


@register
class SymLog(layers.Layer):
    """sign(x)·log1p(|x|) ثم قصّ في [-clip, clip]: تمرير **المستوى** الخام للميزة (لا z) بمقياس محصور.
    ≈ الهوية للقيم الصغيرة (|x|<1، أي ميزات خط الأنابيب المُطبَّعة أصلاً)، ويضغط أي مستوى غير مُطبَّع
    (سعر خام 6.5e4 → 11) بدل أن يُغرق الطبقة التالية."""

    def __init__(self, clip=8.0, **kw):
        super().__init__(**kw)
        self.clip = clip

    def call(self, x):
        y = tf.sign(x) * tf.math.log1p(tf.abs(x))
        return tf.clip_by_value(y, -self.clip, self.clip) if self.clip else y

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "clip": self.clip}

"""
PURPOSE:  Readout layers: LastToken, AttentionPool, FiLM, GatedFusion.
TAGS:     LastToken, AttentionPool, FiLM, GatedFusion, readout, pooling, fusion
PITFALLS: Layer names matter to model_health diagnostics and saved weights; do not rename. Executed into the one
          shared model namespace by model/_loader.py (never imported on its own): names from other modules resolve at
          call time.

## 5) مساعدات القراءة (readout)
"""
@register
class LastToken(layers.Layer):
    def call(self, x):
        return x[:, -1, :]

    def compute_output_shape(self, s):
        return (s[0], s[2])


@register
class AttentionPool(layers.Layer):
    """تجميع بانتباه بمفتاح استعلام مُتعلَّم (يستخدم كل الرموز لا آخرها فقط)."""

    def build(self, input_shape):
        self.q = self.add_weight(name="query", shape=(int(input_shape[-1]),),
                                  initializer=initializers.RandomNormal(stddev=0.02))
        super().build(input_shape)

    def call(self, x):
        d = tf.cast(tf.shape(x)[-1], x.dtype)
        w = tf.nn.softmax(tf.einsum("btd,d->bt", x, self.q) / tf.sqrt(d), axis=-1)
        return tf.einsum("bt,btd->bd", w, x)

    def compute_output_shape(self, s):
        return (s[0], s[2])


@register
class FiLM(layers.Layer):
    """تكييف FiLM: x·(1+γ(e)) + β(e) لكل رمز، حيث e متجه مستوى العيّنة (مقطعي). γ وβ يبدآن صفراً (هوية تامة
    عند البدء) فلا يُفسد التكييف ما يتعلّمه المسار الزمني قبل أن يجد في المستوى ما يستحق."""

    def __init__(self, d_model, **kw):
        super().__init__(**kw)
        self.d_model = d_model
        self.to_gamma = layers.Dense(d_model, kernel_initializer="zeros", name="film_gamma")
        self.to_beta = layers.Dense(d_model, kernel_initializer="zeros", name="film_beta")

    def build(self, input_shape):
        x_shape, e_shape = input_shape
        self.to_gamma.build(e_shape)
        self.to_beta.build(e_shape)
        super().build(input_shape)

    def call(self, inputs):
        x, e = inputs
        return x * (1.0 + self.to_gamma(e)[:, None, :]) + self.to_beta(e)[:, None, :]

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}


@register
class GatedFusion(layers.Layer):
    """دمج متأخّر لبرجين: g = σ(W[a; b]) ثم g·a + (1−g)·b — العيّنة تختار كم تعتمد على كل برج."""

    def __init__(self, d_model, **kw):
        super().__init__(**kw)
        self.d_model = d_model
        self.gate = layers.Dense(d_model, activation="sigmoid", name="fusion_gate")

    def build(self, input_shape):
        a_shape, b_shape = input_shape
        self.gate.build(tuple(a_shape[:-1]) + (int(a_shape[-1]) + int(b_shape[-1]),))
        super().build(input_shape)

    def call(self, inputs):
        a, b = inputs
        g = self.gate(tf.concat([a, b], axis=-1))
        return g * a + (1.0 - g) * b

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model}

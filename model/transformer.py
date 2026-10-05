"""
PURPOSE:  Transformer block parts: RMSNorm, RelativeGQAttention (GQA + relative bias), SwiGLU, TransformerBlock.
TAGS:     RMSNorm, RelativeGQAttention, GQA, num_kv_heads, SwiGLU, TransformerBlock, attention, relative bias
PITFALLS: TransformerBlock is looked up by class name in the diagnostics (type(layer).__name__ == 'TransformerBlock').
          Executed into the one shared model namespace by model/_loader.py (never imported on its own): names from
          other modules resolve at call time.

## 4) كتلة محوّل (RMSNorm ما قبل الطبقة، GQA + انحياز نسبي، SwiGLU)
"""
@register
class RMSNorm(layers.Layer):
    def __init__(self, epsilon=1e-6, **kw):
        super().__init__(**kw)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name="scale", shape=(int(input_shape[-1]),), initializer="ones")
        super().build(input_shape)

    def call(self, x):
        ms = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
        return self.scale * x * tf.math.rsqrt(ms + self.epsilon)

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon}


@register
class RelativeGQAttention(layers.Layer):
    def __init__(self, d_model, num_heads, num_kv_heads=None, max_rel_pos=16,
                 causal=False, window=None, dropout=0.0, out_std=0.02, **kw):
        super().__init__(**kw)
        num_kv_heads = num_kv_heads or num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.d_model, self.h, self.hkv = d_model, num_heads, num_kv_heads
        self.dh = d_model // num_heads
        self.max_rel_pos, self.causal, self.window = max_rel_pos, causal, window
        self.dropout_rate, self.out_std = dropout, out_std
        self.wq = layers.Dense(self.h * self.dh, use_bias=False)
        self.wk = layers.Dense(self.hkv * self.dh, use_bias=False)
        self.wv = layers.Dense(self.hkv * self.dh, use_bias=False)
        self.wo = layers.Dense(d_model, use_bias=False,
                                kernel_initializer=initializers.TruncatedNormal(stddev=out_std))
        self.drop = layers.Dropout(dropout)

    def build(self, input_shape):
        self.rel_bias = self.add_weight(name="rel_bias", shape=(2 * self.max_rel_pos + 1, self.h),
                                         initializer="zeros")
        # wq/wk/wv كلها تُبنى من نفس بُعد الإدخال (d_model)؛ wo يستقبل واقعياً
        # ctx مُعاد تشكيله لنفس d_model أيضاً — نفس input_shape يبنيها جميعاً.
        self.wq.build(input_shape)
        self.wk.build(input_shape)
        self.wv.build(input_shape)
        self.wo.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        b, t = tf.shape(x)[0], tf.shape(x)[1]
        q = tf.reshape(self.wq(x), (b, t, self.h, self.dh))
        k = tf.reshape(self.wk(x), (b, t, self.hkv, self.dh))
        v = tf.reshape(self.wv(x), (b, t, self.hkv, self.dh))
        if self.hkv != self.h:
            rep = self.h // self.hkv
            k, v = tf.repeat(k, rep, axis=2), tf.repeat(v, rep, axis=2)
        q, k, v = [tf.transpose(a, [0, 2, 1, 3]) for a in (q, k, v)]

        scores = tf.matmul(q, k, transpose_b=True) * (float(self.dh) ** -0.5)
        pos = tf.range(t)
        rel = pos[:, None] - pos[None, :]
        idx = tf.clip_by_value(rel, -self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        bias = tf.transpose(tf.gather(self.rel_bias, idx), [2, 0, 1])[None]
        scores = scores + tf.cast(bias, scores.dtype)

        if self.causal or self.window is not None:
            allowed = tf.ones_like(rel, dtype=tf.bool)
            if self.causal:
                allowed = tf.logical_and(allowed, rel >= 0)
            if self.window is not None:
                allowed = tf.logical_and(allowed, tf.abs(rel) <= self.window)
            neg = tf.cast(scores.dtype.min, scores.dtype)
            scores = tf.where(allowed[None, None], scores, neg)

        attn = self.drop(tf.nn.softmax(scores, axis=-1), training=training)
        ctx = tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3])
        return self.wo(tf.reshape(ctx, (b, t, self.d_model)))

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "num_heads": self.h,
                "num_kv_heads": self.hkv, "max_rel_pos": self.max_rel_pos, "causal": self.causal,
                "window": self.window, "dropout": self.dropout_rate, "out_std": self.out_std}


@register
class SwiGLU(layers.Layer):
    """بلا dropout داخلي: الكتلة تُطبِّق مرّة واحدة فقط dropout على المسار المتبقّي."""

    def __init__(self, d_model, dff=None, out_std=0.02, **kw):
        super().__init__(**kw)
        self.d_model = d_model
        self.dff = dff or int(round(8 * d_model / 3 / 8) * 8)
        self.out_std = out_std
        self.w1 = layers.Dense(self.dff, use_bias=False)
        self.w3 = layers.Dense(self.dff, use_bias=False)
        self.w2 = layers.Dense(d_model, use_bias=False,
                                kernel_initializer=initializers.TruncatedNormal(stddev=out_std))

    def build(self, input_shape):
        self.w1.build(input_shape)
        self.w3.build(input_shape)
        self.w2.build(tuple(input_shape[:-1]) + (self.dff,))  # مدخل w2 هو ناتج w1*w3، ببُعد dff
        super().build(input_shape)

    def call(self, x):
        return self.w2(tf.nn.silu(self.w1(x)) * self.w3(x))

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "dff": self.dff, "out_std": self.out_std}


@register
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, num_kv_heads=None, max_rel_pos=16, causal=False,
                 window=None, dropout=0.1, attn_dropout=0.0, num_layers_for_init=4, **kw):
        super().__init__(**kw)
        self.cfg = dict(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads,
                         max_rel_pos=max_rel_pos, causal=causal, window=window, dropout=dropout,
                         attn_dropout=attn_dropout, num_layers_for_init=num_layers_for_init)
        out_std = 0.02 / np.sqrt(2.0 * num_layers_for_init)
        self.norm1, self.norm2 = RMSNorm(), RMSNorm()
        self.attn = RelativeGQAttention(d_model, num_heads, num_kv_heads, max_rel_pos,
                                         causal, window, attn_dropout, out_std)
        self.ffn = SwiGLU(d_model, out_std=out_std)
        self.drop1, self.drop2 = layers.Dropout(dropout), layers.Dropout(dropout)

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.attn.build(input_shape)
        self.norm2.build(input_shape)
        self.ffn.build(input_shape)
        super().build(input_shape)

    def call(self, x, training=None):
        x = x + self.drop1(self.attn(self.norm1(x), training=training), training=training)
        x = x + self.drop2(self.ffn(self.norm2(x)), training=training)
        return x

    def get_config(self):
        return {**super().get_config(), **self.cfg}

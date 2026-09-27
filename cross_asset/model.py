"""نموذج اللوحة: مُرمِّز زمني مشترك لكل عملة ← انتباه عبر عملات اليوم نفسه (بقناع) ← رؤوس لكل عملة.

المُرمِّز هو **نفس** مُرمِّز النموذج الحالي حرفياً: يُبنى build_model_fn(SEQ_LEN, N_FEATURES, config=MODEL_OVERRIDES)
كما في دفتر main (مقاومة الحفظ: تمرير المستوى + BatchNorm، ضجيج/إسقاط قنوات، GRU، d_model=64)، ثم يُقتطَع حتى
طبقة الجذع ("trunk_drop") — نفس المدخلات الـ18 ونفس المعالجة، والفرق الوحيد ما يأتي بعد الجذع.

الانتباه عبر العملات: متجهات (M, d) المسطّحة تُبعثَر إلى (B, N_max, d) حسب (day, pos)، وقناع المفاتيح يمنع أي عملة
من رؤية الحشو، ولا يرى يومٌ يوماً آخر (كل يوم صف مستقل في محور B). ثم تُجمَع ثانيةً إلى (M, d) للرؤوس.

cross_attention=False (المتغيّر B): نفس الكتل تماماً لكن بلا طبقة الانتباه — يبقى الـ FFN لكل عملة على حدة، فالفرق
بين A وB هو تدفّق المعلومة بين العملات وحده (نفس الدفعات والخسائر وعدد مقارب من المعاملات).
"""
import tensorflow as tf

keras = tf.keras
layers = keras.layers

TARGETS = ("high", "low", "close")


def extract_encoder(base_model, layer_name="trunk_drop"):
    """يقتطع نموذج build_nig_timenet_v2 حتى الجذع: (N, T, F) ← (N, d). الرؤوس غير المستخدمة لا تدخل النموذج الناتج."""
    inp = base_model.inputs[0] if len(base_model.inputs) == 1 else base_model.inputs
    return keras.Model(inp, base_model.get_layer(layer_name).output, name="coin_encoder")


def tiny_encoder(seq_len, n_features, d_model=16):
    """مُرمِّز صغير للاختبارات الذاتية فقط (فيه BatchNorm مثل المُرمِّز الحقيقي كي يُختبَر أثره)."""
    inp = layers.Input((seq_len, n_features))
    h = layers.BatchNormalization()(inp)
    h = layers.GRU(d_model)(h)
    h = layers.Dropout(0.1)(h)
    return keras.Model(inp, h, name="tiny_encoder")


class CrossAssetBlock(layers.Layer):
    """كتلة Transformer (pre-norm) على محور العملات: x + MHA(LN(x)) ثم x + FFN(LN(x)). attention=False يُسقط الانتباه."""

    def __init__(self, d_model=64, num_heads=4, ff_mult=2, dropout=0.25, attn_dropout=0.1, attention=True, **kw):
        super().__init__(**kw)
        self.d_model, self.num_heads, self.ff_mult = d_model, num_heads, ff_mult
        self.dropout, self.attn_dropout, self.attention = dropout, attn_dropout, attention
        if attention:
            self.ln1 = layers.LayerNormalization(epsilon=1e-5, name="ln_attn")
            self.mha = layers.MultiHeadAttention(num_heads, key_dim=d_model // num_heads, dropout=attn_dropout, name="mha")
            self.drop1 = layers.Dropout(dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-5, name="ln_ffn")
        self.ff1 = layers.Dense(ff_mult * d_model, activation="gelu", name="ffn_in")
        self.ff2 = layers.Dense(d_model, name="ffn_out")
        self.drop2 = layers.Dropout(dropout)

    def build(self, input_shape):
        shp = (None, None, self.d_model)
        if self.attention:
            self.ln1.build(shp)
            self.mha.build(shp, shp)
        self.ln2.build(shp)
        self.ff1.build(shp)
        self.ff2.build((None, None, self.ff_mult * self.d_model))
        super().build(input_shape)

    def call(self, x, key_mask, training=None):
        """x: (B, N, d)، key_mask: (B, N) منطقي (True = عملة حقيقية)."""
        if self.attention:
            # كل استعلام (حتى الحشو) يرى المفاتيح الحقيقية فقط؛ كل يوم فيه عملة حقيقية واحدة على الأقل ⇒ لا صفوف فارغة.
            q_ones = tf.ones_like(key_mask)[:, :, None]
            am = tf.logical_and(q_ones, key_mask[:, None, :])
            h = self.ln1(x)
            x = x + self.drop1(self.mha(h, h, attention_mask=am, training=training), training=training)
        return x + self.drop2(self.ff2(self.ff1(self.ln2(x))), training=training)

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model, "num_heads": self.num_heads, "ff_mult": self.ff_mult,
                "dropout": self.dropout, "attn_dropout": self.attn_dropout, "attention": self.attention}


class PanelModel(keras.Model):
    """المدخل: {"x": (M,T,F), "day": (M,), "pos": (M,)}. المخرج: {"logit": (M,3), "mu": (M,3)} بترتيب TARGETS.
    logit = لوجِت «تتفوّق على وسيط اليوم» (sigmoid عند التصدير)، mu = العائد النسبي المُقيَّس (× المقياس عند التصدير)."""

    def __init__(self, encoder, d_model=64, n_cross_layers=1, num_heads=4, cross_attention=True, dropout=0.25,
                 attn_dropout=0.1, ff_mult=2, class_head_hidden=32, head_hidden=64, targets=TARGETS, **kw):
        super().__init__(**kw)
        self.encoder, self.targets, self.d_model = encoder, tuple(targets), d_model
        self.cross_attention, self.n_cross_layers = cross_attention, n_cross_layers
        enc_dim = int(encoder.output.shape[-1])
        self.proj = layers.Dense(d_model, name="enc_proj") if enc_dim != d_model else None
        self.blocks = [CrossAssetBlock(d_model, num_heads, ff_mult, dropout, attn_dropout, cross_attention,
                                       name=f"cross_block_{i + 1}") for i in range(n_cross_layers)]
        self.out_norm = layers.LayerNormalization(epsilon=1e-5, name="panel_out_norm")
        self.cls_fc = [layers.Dense(class_head_hidden, activation="relu", name=f"cls_fc_{t}") for t in self.targets]
        self.cls_out = [layers.Dense(1, name=f"cls_logit_{t}") for t in self.targets]
        self.reg_fc = [layers.Dense(head_hidden, activation="swish", name=f"reg_fc_{t}") for t in self.targets]
        self.reg_out = [layers.Dense(1, name=f"reg_mu_{t}") for t in self.targets]
        self.head_drop = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, day, pos = inputs["x"], tf.cast(inputs["day"], tf.int32), tf.cast(inputs["pos"], tf.int32)
        h = self.encoder(x, training=training)
        if self.proj is not None:
            h = self.proj(h)
        if self.blocks:
            idx = tf.stack([day, pos], axis=1)
            shape = tf.stack([tf.reduce_max(day) + 1, tf.reduce_max(pos) + 1, tf.constant(self.d_model)])
            P = tf.scatter_nd(idx, h, shape)                                           # (B, N_max, d)
            key_mask = tf.scatter_nd(idx, tf.ones_like(day, dtype=h.dtype), shape[:2]) > 0
            for blk in self.blocks:
                P = blk(P, key_mask, training=training)
            h = tf.gather_nd(P, idx)                                                   # (M, d)
        h = self.out_norm(h)
        logit = tf.concat([o(self.head_drop(f(h), training=training)) for f, o in zip(self.cls_fc, self.cls_out)], 1)
        mu = tf.concat([o(self.head_drop(f(h), training=training)) for f, o in zip(self.reg_fc, self.reg_out)], 1)
        return {"logit": logit, "mu": mu}


def build_panel_model(encoder, seq_len, n_features, **cfg):
    """يبني PanelModel ويستدعيه مرّة على دفعة وهمية (يومان) كي تُنشأ كل الأوزان قبل التدريب/التحميل."""
    m = PanelModel(encoder, **cfg)
    dummy = {"x": tf.zeros((3, seq_len, n_features)), "day": tf.constant([0, 0, 1], tf.int32),
             "pos": tf.constant([0, 1, 0], tf.int32)}
    m(dummy, training=False)
    return m

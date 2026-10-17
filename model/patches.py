"""
PURPOSE:  num_patches helper and PatchEmbedding (patching + learned absolute positions).
TAGS:     num_patches, PatchEmbedding, patch_len, stride, positional embedding
PITFALLS: num_patches pads the sequence so the last patch ends at the last step; keep it in sync with PatchEmbedding.
          Executed into the one shared model namespace by model/_loader.py (never imported on its own): names from
          other modules resolve at call time.

## 3) تضمين رُقَع (patches) + مواضع مطلقة مُتعلَّمة
"""
def num_patches(seq_len, patch_len, stride):
    pad = (stride - (seq_len - patch_len) % stride) % stride
    return (seq_len + pad - patch_len) // stride + 1, pad


@register
class PatchEmbedding(layers.Layer):
    """رُقَع مُتراكبة (خلط قنوات خطّي) مُحاذاة بحيث تنتهي آخر رقعة عند آخر
    خطوة زمنية تماماً. patch_len=1, stride=1 يُعيد نفس سلوك رمز لكل خطوة."""

    def __init__(self, d_model, patch_len=4, stride=2, **kw):
        super().__init__(**kw)
        self.d_model, self.patch_len, self.stride = d_model, patch_len, stride
        self.proj = layers.Conv1D(d_model, patch_len, strides=stride, padding="valid", name="patch_proj")

    def build(self, input_shape):
        t = int(input_shape[1])
        assert t >= self.patch_len, "seq_len must be >= patch_len"
        self.n_tokens, self.pad = num_patches(t, self.patch_len, self.stride)
        self.pos = self.add_weight(name="pos_emb", shape=(1, self.n_tokens, self.d_model),
                                    initializer=initializers.RandomNormal(stddev=0.02))
        self.proj.build(input_shape)  # لنفس سبب CausalMultiScaleDecomp.build أعلاه
        super().build(input_shape)

    def call(self, x):
        if self.pad > 0:
            x = tf.concat([tf.repeat(x[:, :1, :], self.pad, axis=1), x], axis=1)
        return self.proj(x) + self.pos

    def get_config(self):
        return {**super().get_config(), "d_model": self.d_model,
                "patch_len": self.patch_len, "stride": self.stride}

"""
PURPOSE:  CausalMultiScaleDecomp: causal multi-scale trend/residual split (no future leakage).
TAGS:     CausalMultiScaleDecomp, trend, residual, causal, moving average, decomposition
PITFALLS: Causal padding only: a centred window would leak the future into the trend. Executed into the one shared
          model namespace by model/_loader.py (never imported on its own): names from other modules resolve at call
          time.

## 2) تفكيك مقاييس زمنية سببي (يمنع تسرّب المستقبل + بقايا وهمية)
"""
@register
class CausalMultiScaleDecomp(layers.Layer):
    """trend = مزيج محدَّب مُتعلَّم من متوسطات متحرّكة سابقة (حشو بتكرار الطرف
    الأيسر). seasonal = x - trend -> trend + seasonal == x تماماً (بلا مسار
    بقايا وهمي). النوافذ اللاحقة تستخدم الماضي فقط، وموثوقة بنفس القدر عند
    آخر خطوة زمنية — وهي الخطوة التي يقرؤها الرأس."""

    def __init__(self, kernel_sizes=(3, 5, 9, 17), **kw):
        super().__init__(**kw)
        self.kernel_sizes = tuple(kernel_sizes)
        self.pools = [layers.AveragePooling1D(pool_size=k, strides=1, padding="valid")
                      for k in self.kernel_sizes]
        self.gate = layers.Dense(len(self.kernel_sizes), name="scale_gate")

    def build(self, input_shape):
        # يبني الأبناء صراحةً (لا انتظار أول call) — إلزامي لحفظ/تحميل
        # النموذج كاملاً (model.save/load_model) بلا إعادة بناء عبر
        # build_model_fn: بلا هذا، Keras لا يعرف شكل أوزان self.gate وقت
        # استعادتها من ملف الحفظ فيرفض التحميل بخطأ "never built".
        f = int(input_shape[-1])
        self.gate.build(tuple(input_shape[:-1]) + (2 * f,))
        super().build(input_shape)

    def call(self, x):
        trends = []
        for k, pool in zip(self.kernel_sizes, self.pools):
            pad = tf.repeat(x[:, :1, :], repeats=k - 1, axis=1)
            trends.append(pool(tf.concat([pad, x], axis=1)))
        trends = tf.stack(trends, axis=1)
        rough = tf.reduce_mean(tf.abs(x[:, 1:, :] - x[:, :-1, :]), axis=1)
        g = tf.concat([x[:, -1, :], rough], axis=-1)
        w = tf.nn.softmax(self.gate(g), axis=-1)
        trend = tf.einsum("bktf,bk->btf", trends, w)
        return trend, x - trend

    def compute_output_shape(self, input_shape):
        return input_shape, input_shape

    def get_config(self):
        return {**super().get_config(), "kernel_sizes": list(self.kernel_sizes)}

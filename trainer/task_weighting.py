"""
PURPOSE:  UncertaintyWeightedLoss: automatic task balancing (Kendall et al., CVPR 2018) with dynamic names.
TAGS:     UncertaintyWeightedLoss, task weighting, homoscedastic uncertainty, log_vars
PITFALLS: Weights are trainable variables: they are part of the checkpointed state. Executed into the one shared
          trainer namespace by trainer/_loader.py (never imported on its own): names from other modules resolve at
          call time.

## 4) الموازنة التلقائية بين المهام (Kendall et al., CVPR 2018)

نسخة معمَّمة تعمل بأي عدد وأي أسماء مهام (بدلًا من `log_var_high/low/close`
المكتوبة يدويًا في الكود القديم). الاسم يُشتق تلقائيًا من `trainer.target_names`
(أي من مفاتيح `config['targets']`).

> ⚠️ **شرط رياضي لهذه الموازنة: خسارة المهمة `L` يجب أن تكون موجبة.** الحدّ
> `0.5·exp(-s)·L + 0.5·s` له أدنى نقطة عند `exp(-s) = 1/L` فقط إن كانت `L > 0`.
> إن كانت `L < 0` (وهذا طبيعي لـ NLL توزيع NIG حين يصبح عدم اليقين صغيراً — ظهر
> فعلياً `nig_base ≈ -2` في تدريب حقيقي) فمشتقّة الحدّ بالنسبة لـ `s` موجبة دائماً:
> `s` تنحدر بلا توقف، ووزن المهمة `exp(-s)` يتضاعف كل حقبة، والخسارة الكلية تنهار
> نحو `-∞` (رُصد: `loss` من `+0.9` إلى `-42` خلال 18 حقبة بينما `val_mae` ثابت
> تماماً). ثلاثة آثار صامتة لذلك: (1) `val_loss` يتحسّن كل حقبة بسبب انجراف `s`
> وحده، فـ`BestModelTracker` يعلن «أفضل جديد» كل حقبة ولا يتوقف مبكراً أبداً؛
> (2) تدرّج رؤوس NIG يطغى على الجذع المشترك تحت `clip_norm` العام، فتُحرَم رؤوس
> التصنيف تدريجياً (أوزانها الفعلية 0.15 ← 0.08)؛ (3) الجذع يُدفَع لتضخيم الدليل
> (`nu`/`alpha`) على عيّنات التدريب — إفراط ثقة لا يُعمَّم (`val_nig_base` يسوء
> منذ الحقبة ~4 بينما `nig_base` للتدريب يتحسّن).
>
> لذلك تُستثنى أنواع المهام المذكورة في
> `config['loss']['uncertainty_weighting_exclude_task_types']` (افتراضياً
> `["evidential"]`) من هذه الطبقة وتُجمَع بوزنها الثابت. هذا لا يُفقِد شيئاً:
> NIG تتعلّم مقياس ضجيجها بنفسها عبر `beta`، فوزن Kendall فوقها زائد رياضياً.
> ويُسجَّل أيضاً `raw_loss` (مجموع خسائر المهام بأوزانها الثابتة، بلا حدود Kendall)
> كمقياس مستقرّ لاختيار أفضل نموذج — `val_loss` مع Kendall يتغيّر حتى بلا أي
> تحسّن حقيقي لأن `s` نفسها تتعلّم.
"""
# @title 4) طبقة الموازنة التلقائية بين المهام (بأسماء ديناميكية)
class UncertaintyWeightedLoss(tf.keras.layers.Layer):
    """L_total = Σ_i (L_i / (2σ²_i) + log(σ_i)) — σ²_i متعلَّمة لكل مهمة"""

    def __init__(self, task_names: List[str], name="uncertainty_weighted_loss"):
        super().__init__(name=name)
        self.task_names = list(task_names)
        self._log_vars = {}
        for t in self.task_names:
            self._log_vars[t] = self.add_weight(
                name=f"log_var_{t}", shape=(), dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(0.0), trainable=True,
            )

    def call(self, losses: Dict[str, tf.Tensor]):
        total = 0.0
        for t in self.task_names:
            precision = tf.exp(-self._log_vars[t])
            total = total + 0.5 * precision * losses[t] + 0.5 * self._log_vars[t]
        return total

    def get_task_variances(self) -> Dict[str, float]:
        return {t: float(tf.exp(v).numpy()) for t, v in self._log_vars.items()}

    def get_effective_weights(self) -> Dict[str, float]:
        variances = self.get_task_variances()
        total_precision = sum(1.0 / v for v in variances.values())
        return {t: (1.0 / variances[t]) / total_precision for t in self.task_names}

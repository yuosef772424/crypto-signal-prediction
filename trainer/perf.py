"""
PURPOSE:  Performance/GPU setup: memory growth and the mixed-precision switch (USE_MIXED_PRECISION).
TAGS:     GPU, memory growth, mixed precision, USE_MIXED_PRECISION, XLA, performance
PITFALLS: Runs at load and changes global TensorFlow state (GPU memory growth, global dtype policy). Executed into the
          one shared trainer namespace by trainer/_loader.py (never imported on its own): names from other modules
          resolve at call time.

## 1.1) إعداد الأداء وGPU — استغلال كامل الطاقة الحاسوبية المتاحة

هذه الخلية **جديدة** (لم تكن في النسخة السابقة). أهم نقطة توضيح أولاً:
بشكل افتراضي فإن TensorFlow يحجز عمليًا كل ذاكرة الـ GPU الحرة عند أول عملية
تخصيص — أي أن "الذاكرة الفارغة" التي تراها في `nvidia-smi` أثناء التدريب
غالبًا ليست ذاكرة غير محجوزة، بل ذاكرة محجوزة لكن **غير مُستغَلة حسابيًا**
(GPU utilization % منخفض) بسبب: حجم batch صغير نسبيًا، أو عدم استخدام
Tensor Cores (تحتاج mixed precision)، أو عنق زجاجة في تغذية البيانات
(`tf.data`). لذلك الحل الفعلي لتسريع التدريب هو:

1. **Mixed Precision (float16)** — يضاعف الإنتاجية على GPUs الحديثة
   (T4/V100/A100/L4...) عبر Tensor Cores، ويقلّل حجم الـ activations للنصف،
   مما يسمح فعليًا بزيادة `batch_size` لاستغلال المساحة المتبقية.
2. **XLA (`jit_compile`)** — يدمج (fuse) العمليات الحسابية في نواة GPU واحدة
   بدل عشرات النداءات المنفصلة، مفعّل عبر `config['optimizer']['use_xla']`.
3. **زيادة `batch_size`** — إن كانت الذاكرة فعليًا فيها متسع (جرّب مضاعفتها
   تدريجيًا حتى تقترب من حد الذاكرة، ثم تراجع خطوة واحدة للأمان). ملاحظة:
   مضاعفة `batch_size` كثيرًا قد تحتاج رفع `lr_initial` بنفس النسبة تقريبًا
   (Linear Scaling Rule) للحفاظ على جودة التقارب.
4. **تحسين `tf.data`**: `.cache()` + `.prefetch(tf.data.AUTOTUNE)` +
   `drop_remainder=True` (الأخيرة ضرورية أيضًا لضمان أشكال ثابتة مع XLA).

### ⚠️ تحذير خاص بنموذج NIG (evidential) تحديدًا
دالة الخسارة تحتوي على `tf.math.log` و`tf.math.lgamma` وقيم `nu/alpha/beta`
حسّاسة رقميًا جدًا — وهذا بالضبط نوع الحسابات التي تُصاب بـ `NaN`/`overflow`
بسهولة تحت `float16`. لذلك:
- مخرجات النموذج (`mu`, `nu`, `alpha`, `beta`, `confidence`) يجب أن تبقى
  `float32` صراحة حتى مع تفعيل mixed precision (أضف `dtype='float32'` على
  طبقات `Dense` الأخيرة في `model_builder_fn` — انظر المثال المُحدَّث في
  القسمين 9 و10 أدناه). دوال الخسارة نفسها في القسم 3 تُجري `tf.cast(..., tf.float32)`
  بالفعل، فهي آمنة بغض النظر عن سياسة الدقّة العامة.
- `USE_MIXED_PRECISION` أدناه **معطّلة افتراضيًا (`False`)** لهذا السبب —
  فعّلها فقط بعد تأكيد أن مخرجات نموذجك محمية بـ `float32` كما في الملاحظة أعلاه،
  وراقب `nig_base`/`nig_pen` بحثًا عن `NaN` في أول بضع حقب.
"""
# @title 1.1) إعداد الأداء وGPU (جديد)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            # يسمح بالتخصيص التدريجي بدل حجز كل الذاكرة دفعة واحدة —
            # يمنع تعطّل الجلسة عند مشاركة الـ GPU مع عمليات أخرى (شائع على Colab).
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"⚠️ تعذر ضبط memory_growth لـ {gpu.name}: {e}")
    print(f"✅ تم العثور على {len(gpus)} GPU: {[g.name for g in gpus]}")
else:
    print("⚠️ لا يوجد GPU متاح في هذه الجلسة — التدريب سيعمل على CPU (أبطأ بكثير)")

# ── Mixed Precision (float16) ──────────────────────────────────────────────
# معطّلة افتراضيًا لحساسية دالة NIG الرقمية (انظر التحذير في الخلية الشارحة أعلاه).
# فعّلها (True) بعد تطبيق dtype='float32' على مخرجات نموذجك الأخيرة.
USE_MIXED_PRECISION = False
if USE_MIXED_PRECISION and gpus:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ Mixed precision (float16) مفعّل — build_optimizer سيغلّف الـ optimizer تلقائيًا بـ LossScaleOptimizer")
else:
    tf.keras.mixed_precision.set_global_policy('float32')
    print("ℹ️ Mixed precision معطّل — الحساب بالكامل float32 (الوضع الآمن الافتراضي)")

# ── XLA ─────────────────────────────────────────────────────────────────────
# لا حاجة لأي إعداد هنا: XLA يُفعَّل عبر config['optimizer']['use_xla'] الذي
# يمرَّر إلى trainer.compile(jit_compile=...) داخل build_training_system (القسم 8) —
# هذا أضمن من التفعيل العام tf.config.optimizer.set_jit(True) لأنه محصور بخطوات
# التدريب/التقييم فقط، ويسهل تعطيله لهذا المدرّب وحده إن سبّب مشاكل توافق.

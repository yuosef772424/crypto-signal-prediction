"""
PURPOSE:  TaskWeightUpdater, MetricsLogger (class baselines) and DerivedMetrics callbacks (+ a load-time self-test).
TAGS:     TaskWeightUpdater, MetricsLogger, DerivedMetrics, class_baselines, callbacks, metrics
PITFALLS: RUNS _test_metrics_logger_class_baselines() AT LOAD. Executed into the one shared trainer namespace by
          trainer/_loader.py (never imported on its own): names from other modules resolve at call time.
"""
# @title 6.3) TaskWeightUpdater + MetricsLogger
class TaskWeightUpdater(tf.keras.callbacks.Callback):
    """تحديث وزن كل هدف تلقائيًا نسبةً لصعوبته الحالية — يعمل مع أي عدد/أسماء أهداف.
    استخدمها فقط إذا عطّلت use_uncertainty_weighting (الطريقتان تتعارضان منطقيًا)."""

    def __init__(self, trainer: "GenericTrainer", update_frequency=1, verbose=1):
        super().__init__()
        self.trainer = trainer
        self.target_names = trainer.target_names
        self.update_frequency = max(int(update_frequency), 0)
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if self.update_frequency == 0 or (epoch + 1) % self.update_frequency != 0:
            return
        logs = logs or {}
        losses = {t: float(logs.get(f"loss_{t}", 0.0)) for t in self.target_names}
        total = sum(losses.values())
        if total <= 0:
            return
        n = len(self.target_names)
        for t, l in losses.items():
            self.trainer.loss_weights[t].assign(n * l / total)
        if self.verbose:
            s = ", ".join(f"{t}={self.trainer.loss_weights[t].numpy():.2f}" for t in self.target_names)
            print(f"📊 [TaskWeights] {s}")


class MetricsLogger(tf.keras.callbacks.Callback):
    """ملخص تدريب دوري + history قابل للاستئناف عبر get_state/load_state.

    class_baselines: {مفتاح هدف تصنيف: نسبة الفئة الأغلب في val} اختياري —
    عام تماماً (المفتاح أي اسم هدف يظهر بمفاتيح `{مفتاح}_accuracy`/
    `val_{مفتاح}_accuracy` في logs، لا شيء خاص بمشروع بعينه). إن مُرِّر، يُطبَع
    مقابل val_accuracy المقابلة في كل ملخّص — نفس مبدأ `tree_naive_baseline_accuracy`
    في `detect_success_failure_patterns` (دقّة خام قد تبدو جيدة وهي فعلياً لا
    تتجاوز تخمين الفئة الأغلب بلا أي مهارة حقيقية، خصوصاً مع أهداف غير متوازنة)."""

    def __init__(self, log_every=5, verbose=1, initial_state=None, class_baselines=None):
        super().__init__()
        self.log_every = log_every
        self.verbose = verbose
        self.class_baselines = class_baselines or {}
        self.history: Dict[str, list] = {"epoch": [], "loss": [], "val_loss": []}
        if initial_state:
            self.load_state(initial_state)

    def get_state(self) -> dict:
        return self.history

    def load_state(self, state: dict):
        if state:
            self.history = state

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["epoch"].append(epoch + 1)
        self.history["loss"].append(float(logs.get("loss", 0)))
        self.history["val_loss"].append(float(logs.get("val_loss", 0)))

        if not self.verbose or (epoch + 1) % self.log_every != 0:
            return

        print(f"\n{'=' * 70}\n📊 ملخص Epoch {epoch + 1}\n{'=' * 70}")
        for k, v in logs.items():
            try:
                print(f"   {k:<22}: {float(v):.5f}")
            except (TypeError, ValueError):
                print(f"   {k:<22}: {v}")

        if self.class_baselines:
            print("   — مقارنة بخطّ أساس الفئة الأغلب (val، لا مهارة حقيقية إن لم يُتجاوَز) —")
            for t, baseline in self.class_baselines.items():
                val_acc = logs.get(f"val_{t}_accuracy")
                if val_acc is None:
                    continue
                beats = float(val_acc) > baseline + 0.01
                verdict = "✅ فوق خطّ الأساس" if beats else "⚠️ عند/تحت خطّ الأساس"
                print(f"   {t:<22}: val_accuracy={float(val_acc):.3f} مقابل خطّ أساس={baseline:.3f}  {verdict}")

        trainer = self.model  # داخل fit()، self.model هو الـ GenericTrainer نفسه
        if getattr(trainer, "uncertainty_layer", None) is not None:
            weights = trainer.uncertainty_layer.get_effective_weights()
            print("   أوزان المهام الفعلية : " + ", ".join(f"{t}={w:.2f}" for t, w in weights.items()))
            if trainer.fixed_task_names:
                print("   مهام بوزن ثابت (خارج Kendall): " + ", ".join(trainer.fixed_task_names))
        if getattr(trainer, "scheduled_vars", None):
            sched_str = ", ".join(f"{k}={v.numpy():.5f}" for k, v in trainer.scheduled_vars.items())
            print(f"   المعاملات المجدولة    : {sched_str}")
        print(f"{'=' * 70}\n")


def _test_metrics_logger_class_baselines():
    """يتحقّق من إضافة class_baselines إلى MetricsLogger (24 سبتمبر 2026):
    (أ) بلا class_baselines، السلوك القديم تماماً — لا سطر مقارنة يُطبَع،
    (ب) بها، يُطبَع سطر مقارنة صريح لكل هدف تصنيف موجود في logs، يُصنِّف
    val_accuracy كـ"فوق"/"عند أو تحت" خطّ الأساس بمقارنة رقمية مباشرة، لا
    نصّاً ثابتاً. يلتقط stdout بدل تشغيل تدريب حقيقي (لا حاجة لبيانات)."""
    import io, contextlib

    logs = {
        'loss': 1.0, 'val_loss': -1.0,
        'close_class_accuracy': 0.77, 'val_close_class_accuracy': 0.498,
        'high_class_accuracy': 0.80, 'val_high_class_accuracy': 0.75,
    }

    logger_no_baseline = MetricsLogger(log_every=1, verbose=1)
    logger_no_baseline.set_model(None)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        logger_no_baseline.on_epoch_end(0, dict(logs))
    assert 'خطّ الأساس' not in buf.getvalue(), (
        'بلا class_baselines يجب ألا يظهر أي سطر مقارنة — تغيير سلوك قديم غير مقصود.')

    logger_with_baseline = MetricsLogger(
        log_every=1, verbose=1,
        class_baselines={'close_class': 0.50, 'high_class': 0.60})
    logger_with_baseline.set_model(None)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        logger_with_baseline.on_epoch_end(0, dict(logs))
    out = buf2.getvalue()
    assert 'val_accuracy=0.498' in out and '⚠️' in out, (
        'close_class (val=0.498 مقابل خطّ أساس 0.50) يجب أن يُصنَّف "عند/تحت خطّ الأساس".')
    assert 'val_accuracy=0.750' in out and '✅' in out, (
        'high_class (val=0.75 مقابل خطّ أساس 0.60) يجب أن يُصنَّف "فوق خطّ الأساس".')
    print("✅ _test_metrics_logger_class_baselines: التوافق الخلفي محفوظ، والمقارنة صحيحة عددياً.")
    return True


_test_metrics_logger_class_baselines()


class DerivedMetrics(tf.keras.callbacks.Callback):
    """مقاييس مشتقّة من السجلّات الموجودة أصلاً — **بلا أي متغيّر** (فلا تمسّ checkpoint قائماً ولا استئنافه).

    class_loss / val_class_loss = مجموع خسائر رؤوس التصنيف (loss_{هدف}) — معيار لاختيار «الأفضل» والإيقاف المبكر
    يستبعد NLL رؤوس NIG التي تهيمن على raw_loss. الفائز الأول في Jane Street 2021: «Only monitor the BCE loss of
    MLP instead of the overall loss for early stopping». يُضاف قبل BestModelTracker فيرى المفتاح في نفس الحقبة."""

    def __init__(self, class_targets: List[str]):
        super().__init__()
        self.class_targets = list(class_targets)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or not self.class_targets:
            return
        for prefix in ("", "val_"):
            keys = [f"{prefix}loss_{t}" for t in self.class_targets]
            if all(k in logs for k in keys):
                logs[f"{prefix}class_loss"] = float(sum(float(logs[k]) for k in keys))

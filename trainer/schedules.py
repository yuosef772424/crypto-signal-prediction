"""
PURPOSE:  ParamScheduler, schedule_value, apply_schedules and the learning-rate schedules (cosine warm restarts).
TAGS:     ParamScheduler, schedule_value, apply_schedules, cosine_warm_restarts, build_lr_schedule_fn, lr schedule
PITFALLS: cross_asset/train.py re-states build_lr_schedule_fn: keep both in sync. Executed into the one shared trainer
          namespace by trainer/_loader.py (never imported on its own): names from other modules resolve at call time.

## 6) الكولباكس العامة (Callbacks)

كل كولباك هنا **عامة** بالمعنى الحرفي: لا تعرف شيئًا عن أسماء أهدافك.

- `ParamScheduler`: كولباك **واحدة تكفي كل المعاملات القابلة للجدولة** —
  تُغني عن كتابة كلاس منفصل لكل معامل (`lambda_reg`, `lambda_calib`,
  `penalty_weight`, ...). الجدولة دالة محضة في رقم الحقبة → **آمنة تمامًا عند
  الاستئناف**، لأن Keras يستدعي `on_epoch_begin` بنفس رقم الحقبة الصحيح حتى
  بعد `resume` (لا حاجة لحفظ حالتها في الـ checkpoint إطلاقًا).
- جدولة معدل التعلّم (`build_lr_schedule_fn`) لنفس السبب: دالة محضة في الحقبة.
- `BestModelTracker`: مصدر واحد لتعريف «الأفضل» + التوقف المبكر معًا (يحلّ محل
  `SmoothedEarlyStopping` و`BestWeightsSaver` القديمتين اللتين كانتا تحسبان «الأفضل» بطريقتين
  مختلفتين ويفقدان حالتهما عند الاستئناف). **قابلة للاستئناف الكامل** عبر `get_state()`/`load_state()`
  + ملف `best_meta.json` مستقل يُقرأ عند البناء.
- `TaskWeightUpdater`: تحديث وزن كل هدف تلقائيًا نسبةً لصعوبته الحالية — بديل
  يدوي اختياري (فقط إذا عطّلت الموازنة التلقائية Kendall).
- `MetricsLogger`: ملخص دوري + `history` قابل للاستئناف.
- `SnapshotEnsemble` / `BestWeightsSaver`: حفظ لقطات دورية / أفضل أوزان
  للنشر النهائي (اختياريتان تمامًا).
"""
# @title 6.1) ParamScheduler + جدولة معدل التعلّم
def _progress_curve(p: float, schedule: str) -> float:
    if schedule == "cosine":
        return 0.5 * (1 - np.cos(np.pi * p))
    elif schedule == "exponential":
        return p ** 2
    elif schedule == "sqrt":
        return np.sqrt(p)
    return p  # linear


def schedule_value(start: float, end: float, warmup_epochs: int, schedule: str, epoch: int) -> float:
    """قيمة معامل مجدول عند حقبة معيّنة — دالة محضة في رقم الحقبة (هذا ما يجعل الاستئناف آمنًا)."""
    warmup_epochs = max(int(warmup_epochs), 0)
    if warmup_epochs == 0 or epoch >= warmup_epochs:
        return end
    return start + (end - start) * _progress_curve(epoch / warmup_epochs, schedule)


class ParamScheduler(tf.keras.callbacks.Callback):
    """جدولة عامة لأي tf.Variable — تُستخدم لأي معامل مجدول عرّفته في config['loss']['schedules']"""

    def __init__(self, var: tf.Variable, start: float, end: float, warmup_epochs: int,
                 schedule: str = "linear", label: Optional[str] = None, log_every: int = 5, verbose: int = 1):
        super().__init__()
        self.var = var
        self.start = start
        self.end = end
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.schedule = schedule
        self.label = label or var.name
        self.log_every = log_every
        self.verbose = verbose

    def value_at(self, epoch: int) -> float:
        return schedule_value(self.start, self.end, self.warmup_epochs, self.schedule, epoch)

    def on_epoch_begin(self, epoch, logs=None):
        new_value = self.value_at(epoch)
        self.var.assign(new_value)
        if self.verbose and (epoch % self.log_every == 0):
            print(f"🔄 [{self.label}] Epoch {epoch + 1}: {new_value:.6f}")


def apply_schedules(trainer, config: dict, epoch: int):
    """يضبط كل المعاملات المجدولة فورًا على قيمتها عند الحقبة `epoch` (قبل أول fit) —
    كي يرى أي evaluate() أو استدلال قبل التدريب القيم الصحيحة، لا قيم البداية."""
    for pname, pcfg in config["loss"].get("schedules", {}).items():
        trainer.scheduled_vars[pname].assign(schedule_value(
            pcfg["start"], pcfg["end"], pcfg.get("warmup_epochs", 0), pcfg.get("schedule", "linear"), epoch))


def cosine_warm_restarts(epoch, lr_initial, lr_min, cycle_length=10, cycle_mult=1.5):
    cycle_epoch = epoch
    current_len = cycle_length
    while cycle_epoch >= current_len:
        cycle_epoch -= current_len
        current_len = int(current_len * cycle_mult)
    progress = cycle_epoch / current_len
    return lr_min + 0.5 * (lr_initial - lr_min) * (1 + np.cos(np.pi * progress))


def build_lr_schedule_fn(opt_cfg: dict, rewarm_from_epoch: Optional[int] = None,
                         rewarm_epochs: int = 0) -> Callable[[int], float]:
    """يبني دالة lr(epoch) بالكامل من config['optimizer'] — دالة محضة، آمنة عند الاستئناف.
    rewarm_*: للبدء الدافئ (warm_start) فقط: optimizer جديد ⇒ عزوم Adam صفرية، فنرفع lr تدريجيًا
    خلال rewarm_epochs حقبة بدل القفز مباشرة إلى القيمة الكاملة فوق أوزان متقاربة."""
    lr_initial = opt_cfg["lr_initial"]
    lr_min = opt_cfg["lr_min"]
    warmup = int(opt_cfg.get("lr_warmup_epochs", 0))
    sched = opt_cfg.get("lr_schedule", {"type": "constant"})

    def base(epoch):
        if warmup and epoch < warmup:
            return lr_initial * (epoch + 1) / warmup
        e = epoch - warmup
        if sched["type"] == "cosine_restarts":
            return cosine_warm_restarts(e, lr_initial, lr_min, sched.get("cycle_length", 10), sched.get("cycle_mult", 1.5))
        elif sched["type"] == "cosine":
            total = sched.get("total_epochs", 100)
            progress = min(e / max(total, 1), 1.0)
            return lr_min + 0.5 * (lr_initial - lr_min) * (1 + np.cos(np.pi * progress))
        else:  # constant
            return lr_initial

    def fn(epoch):
        lr = base(epoch)
        if rewarm_from_epoch is not None and rewarm_epochs > 0:
            k = epoch - rewarm_from_epoch + 1          # 1 = أول حقبة بعد البدء الدافئ
            if 1 <= k <= rewarm_epochs:
                lr = lr * k / (rewarm_epochs + 1)
        return lr

    return fn

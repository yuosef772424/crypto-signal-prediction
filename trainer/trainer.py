"""
PURPOSE:  GenericTrainer: the training-step core (any number of targets, evidential/regression/classification).
TAGS:     GenericTrainer, train_step, test_step, compute_loss, tf.keras.Model subclass, multi-task
PITFALLS: The model is passed to self.model(x) as is (the framework never opens x); keep train_step and test_step
          symmetric. Executed into the one shared trainer namespace by trainer/_loader.py (never imported on its own):
          names from other modules resolve at call time.

## 5) `GenericTrainer` — قلب الإطار

مدرّب واحد يخدم كل الحالات: هدف واحد أو عشرات الأهداف، evidential أو
regression أو classification أو خليط منها، مع موازنة تلقائية أو بدونها، مع
قيود منطقية أو بدونها. **لا يحتوي على أي اسم هدف صريح** — كل شيء يُقرأ من
`self.config` في وقت البناء (`__init__`)، وتُنشأ عليه الـ metrics والمتغيرات
القابلة للجدولة ديناميكيًا.

كل متغيرات هذا الكائن (`self.model`, `self.optimizer` بعد `compile`,
`self.scheduled_vars`, `self.loss_weights`, `self.uncertainty_layer`,
`self.ckpt_epoch`) هي attributes عادية على كائن `tf.keras.Model`، وبالتالي
تُتتبَّع **تلقائيًا** بواسطة `tf.train.Checkpoint` — هذا بالضبط ما يجعل
استئناف التدريب صحيحًا 100% (القسم 7).

> ✅ **تصحيح مقارنةً بالكود القديم**: في `train_step` كانت التدرّجات تُحسب فقط
> على `self.model.trainable_variables`، فتتجاهل متغيرات طبقة الموازنة التلقائية
> (`log_var` لكل مهمة) — أي أن "الموازنة المتعلَّمة" (Kendall) لم تكن تتحدّث
> بالفعل عبر التدريب! هنا نستخدم `self.trainable_variables` التي تضمّ النموذج
> وطبقة الموازنة معًا.
"""
# @title 5) المدرّب العام (GenericTrainer)
class GenericTrainer(tf.keras.Model):
    def __init__(self, model: tf.keras.Model, config: dict, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self.target_names = list(config["targets"].keys())

        # ── معاملات قابلة للجدولة (بأي اسم عرّفه المستخدم في config['loss']['schedules']) ──
        self.scheduled_vars: Dict[str, tf.Variable] = {}
        for pname, pcfg in config["loss"].get("schedules", {}).items():
            self.scheduled_vars[pname] = tf.Variable(
                float(pcfg.get("start", 0.0)), trainable=False, dtype=tf.float32, name=pname
            )

        # ── وزن كل هدف (قابل للتحديث الديناميكي بواسطة TaskWeightUpdater) ──
        self.loss_weights: Dict[str, tf.Variable] = {
            t: tf.Variable(float(tcfg.get("loss_weight", 1.0)), trainable=False,
                            dtype=tf.float32, name=f"loss_weight_{t}")
            for t, tcfg in config["targets"].items()
        }

        # ── عدد الحقب المنجزة — جزء من حالة الـ checkpoint نفسها (وليس فقط ملف JSON خارجي) ──
        # يُحدَّث في نهاية كل حقبة (EpochCheckpointCallback) وهو المرجع الوحيد لـ initial_epoch.
        self.ckpt_epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name="ckpt_epoch")

        # ── موازنة تلقائية بين المهام (اختيارية) ──
        # تُطبَّق فقط على المهام ذات الخسارة الموجبة؛ المستثناة (evidential افتراضياً) تُجمَع بوزنها الثابت
        # (راجع ملاحظة القسم 4: Kendall مع خسارة سالبة تنهار نحو -∞).
        self.use_uncertainty_weighting = bool(config["loss"].get("use_uncertainty_weighting", False))
        excluded_types = set(config["loss"].get("uncertainty_weighting_exclude_task_types", ["evidential"]))
        self.kendall_task_names = [
            t for t, tcfg in config["targets"].items()
            if self.use_uncertainty_weighting and tcfg["task_type"] not in excluded_types
        ]
        self.fixed_task_names = [t for t in self.target_names if t not in self.kendall_task_names]
        self.uncertainty_layer = (
            UncertaintyWeightedLoss(self.kendall_task_names) if self.kendall_task_names else None
        )

        # ── قيود منطقية/فيزيائية اختيارية بين المخرجات ──
        self.constraints_cfg = config["loss"].get("constraints", [])

        self._create_metrics()

    # ─────────────────────────────────────────────────────────────
    def _create_metrics(self):
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # مجموع خسائر المهام بأوزانها الثابتة (بلا حدود Kendall) — مقياس مستقرّ لاختيار أفضل نموذج
        self.raw_loss_tracker = tf.keras.metrics.Mean(name="raw_loss")
        self._target_loss_trackers = {t: tf.keras.metrics.Mean(name=f"loss_{t}") for t in self.target_names}

        self._stat_trackers: Dict[str, tf.keras.metrics.Mean] = {}
        for t, tcfg in self.config["targets"].items():
            task_type = tcfg["task_type"]
            for key in TASK_REGISTRY[task_type]["stat_keys"]:
                self._stat_trackers[f"{t}_{key}"] = tf.keras.metrics.Mean(name=f"{t}_{key}")

        self._constraint_trackers = {
            c["name"]: tf.keras.metrics.Mean(name=f"constraint_{c['name']}")
            for c in self.constraints_cfg
        }

    @property
    def metrics(self):
        return (
            [self.loss_tracker, self.raw_loss_tracker]
            + list(self._target_loss_trackers.values())
            + list(self._stat_trackers.values())
            + list(self._constraint_trackers.values())
        )

    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _weight_for_target(sample_weight, target_name: str, tcfg: dict):
        """sample_weight إما موتّر واحد (يسري على كل الأهداف) أو dict بمفتاح اسم الهدف أو true_key."""
        if sample_weight is None:
            return None
        if isinstance(sample_weight, collections.abc.Mapping):
            for key in (target_name, tcfg["true_key"]):
                if key in sample_weight:
                    return sample_weight[key]
            return None
        return sample_weight

    def _compute_losses(self, x, y, sample_weight, training):
        outputs = self.model(x, training=training)

        per_target_losses = {}
        per_stat_values: Dict[str, tf.Tensor] = {}

        for t, tcfg in self.config["targets"].items():
            y_true_t = y[tcfg["true_key"]]
            entry = TASK_REGISTRY[tcfg["task_type"]]
            w_t = self._weight_for_target(sample_weight, t, tcfg)
            if w_t is not None:
                if not entry["supports_sample_weight"]:
                    raise ValueError(
                        f"❌ الهدف '{t}' من النوع '{tcfg['task_type']}' لا يدعم sample_weight "
                        f"(سجّله بـ supports_sample_weight=True وأضف وسيط sample_weight لدالته) — "
                        f"رفضنا تجاهله بصمت لأن هذا يُفسد التدريب دون أي إنذار."
                    )
                raw_loss, stats = entry["fn"](y_true_t, outputs, tcfg, self.scheduled_vars, sample_weight=w_t)
            else:
                raw_loss, stats = entry["fn"](y_true_t, outputs, tcfg, self.scheduled_vars)

            raw_loss = tf.cast(raw_loss, tf.float32)
            flood = tcfg.get("flood_level")
            if training and flood is not None:
                # Flooding: القيمة الأمامية تبقى ≥ b، والتدرّج يعكس اتجاهه تحت b (صعود بدل نزول)
                raw_loss = tf.abs(raw_loss - float(flood)) + float(flood)
            w = self.loss_weights[t]
            per_target_losses[t] = raw_loss * w
            for k, v in stats.items():
                per_stat_values[f"{t}_{k}"] = tf.cast(v, tf.float32)

        # القيود المنطقية لا تُرجَّح بـ sample_weight (دالتها تأخذ outputs فقط)
        per_constraint_values = {}
        constraint_total = tf.constant(0.0, tf.float32)
        for c in self.constraints_cfg:
            penalty = tf.cast(c["fn"](outputs), tf.float32)
            weight = self.scheduled_vars.get(c.get("weight_var"), tf.constant(1.0))
            constraint_total = constraint_total + penalty * tf.cast(weight, tf.float32)
            per_constraint_values[c["name"]] = penalty

        raw_loss = tf.add_n(list(per_target_losses.values()))
        if self.uncertainty_layer is not None:
            total_loss = self.uncertainty_layer({t: per_target_losses[t] for t in self.kendall_task_names})
            if self.fixed_task_names:
                total_loss = total_loss + tf.add_n([per_target_losses[t] for t in self.fixed_task_names])
        else:
            total_loss = raw_loss

        extra = constraint_total
        if self.model.losses:
            extra = extra + tf.add_n([tf.cast(l, tf.float32) for l in self.model.losses])
        total_loss = total_loss + extra
        raw_loss = raw_loss + extra

        return total_loss, raw_loss, per_target_losses, per_stat_values, per_constraint_values

    # ─────────────────────────────────────────────────────────────
    def _update_trackers(self, total_loss, raw_loss, per_target_losses, per_stat_values, per_constraint_values):
        self.loss_tracker.update_state(total_loss)
        self.raw_loss_tracker.update_state(raw_loss)
        for t, v in per_target_losses.items():
            self._target_loss_trackers[t].update_state(v)
        for k, v in per_stat_values.items():
            if k in self._stat_trackers:
                self._stat_trackers[k].update_state(v)
        for name, v in per_constraint_values.items():
            self._constraint_trackers[name].update_state(v)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        # self.trainable_variables تشمل النموذج + متغيرات طبقة الموازنة (log_var لكل مهمة)
        train_vars = self.trainable_variables
        with tf.GradientTape() as tape:
            total_loss, raw_loss, per_target, stats, constraints = self._compute_losses(x, y, sample_weight, training=True)

        grads = tape.gradient(total_loss, train_vars)
        # القص (global_clipnorm) يتم داخل الـ optimizer نفسه — لا قص يدوي مكرر هنا.
        self.optimizer.apply_gradients(zip(grads, train_vars))

        self._update_trackers(total_loss, raw_loss, per_target, stats, constraints)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        total_loss, raw_loss, per_target, stats, constraints = self._compute_losses(x, y, sample_weight, training=False)
        self._update_trackers(total_loss, raw_loss, per_target, stats, constraints)
        return {m.name: m.result() for m in self.metrics}

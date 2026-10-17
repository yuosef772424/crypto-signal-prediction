"""
PURPOSE:  build_optimizer, build_training_system (the single entry point) and EMA warm-up (+ a load-time self-test).
TAGS:     build_training_system, build_optimizer, enable_ema_warmup, EmaWindow, _resolve_train_mode, train_mode, entry
          point
PITFALLS: RUNS _test_ema_warmup() AT LOAD. Always call build_training_system the same way for new, resumed and warm-
          start runs. Executed into the one shared trainer namespace by trainer/_loader.py (never imported on its
          own): names from other modules resolve at call time.

## 8) `build_training_system` — نقطة الدخول الوحيدة

استدعِها دائمًا **بنفس الطريقة**، سواء كانت هذه أول مرة تدريب أو استئنافًا
بعد انقطاع — الدالة تكتشف ذلك تلقائيًا وتتصرف بالشكل الصحيح دون أي تدخّل منك.

```python
trainer, callbacks, initial_epoch = build_training_system(model_builder_fn, config, sample_batch)
trainer.fit(train_ds, validation_data=val_ds,
            initial_epoch=initial_epoch, epochs=config["run"]["epochs"], callbacks=callbacks)
```

- `model_builder_fn`: دالة **بلا مدخلات** تُرجع نموذج Keras جديد (نفس البنية
  دائمًا في كل استدعاء — هذا شرط أساسي لصحّة الاستئناف).
- `config`: ناتج `build_config(...)`.
- `sample_batch`: دفعة واحدة حقيقية `(x, y)` من بيانات التدريب (لإجبار بناء
  كل المتغيرات قبل أي استرجاع — انظر القسم 7).
"""
# @title 8) build_optimizer + build_training_system
def enable_ema_warmup(opt):
    """EMA بإحماء: الزخم الفعلي min(سقف, (1+t)/(10+t)) (نفس num_updates في tf.train.ExponentialMovingAverage
    و timm ModelEmaV3). بدونه يبقى 0.999^N من وزن المتوسط على أوزان أول الخطوات: 60 حقبة × 16 خطوة (بيانات قليلة
    جداً) ⇒ 0.999^960 ≈ 38% من المتوسط أوزان شبه عشوائية، فتخسر أوزان EMA المحفوظة كـ«أفضل» أمام الخام
    (Drive: 0.555 مقابل 0.611). السقف = ema_momentum، ويضبطه EmaWindow بالحقب بدل الخطوات.
    السقف tf.Variable في إغلاق (closure) لا خاصية على optimizer ⇒ لا متغيّر متتبَّع جديد والاستئناف سليم."""
    from keras import ops
    cap = tf.Variable(float(opt.ema_momentum), trainable=False, dtype="float32")

    def _update(trainable_variables):
        if not opt.use_ema:
            return
        t = ops.cast(opt.iterations, "float32")
        m = ops.minimum(cap, (1.0 + t) / (10.0 + t))
        for var, avg in zip(trainable_variables, opt._model_variables_moving_average):
            if avg is not None:
                mm = ops.cast(m, var.dtype)
                avg.assign(mm * avg + (1.0 - mm) * var)

    opt._update_model_variables_moving_average = _update   # يُستدعى عبر self في BaseOptimizer.apply
    opt._set_ema_cap = lambda v: cap.assign(float(v))
    opt._get_ema_cap = lambda: float(cap.numpy())
    return opt


class EmaWindow(tf.keras.callbacks.Callback):
    """نافذة EMA بالحقب لا بالخطوات: السقف = min(ema_momentum, 1 - 1/(window_epochs × خطوات الحقبة)).
    0.999 الثابت = ~1000 خطوة: ~3 حقب على Drive الكامل لكن ~19 حقبة على 13 ألف عيّنة و~60 على 4 آلاف — فتتأخّر
    أوزان EMA عن نموذج ما زال يتحسّن. يتطلب ema_warmup (السقف متغيّر في إغلاق enable_ema_warmup). بلا متغيّرات."""

    def __init__(self, window_epochs: float):
        super().__init__()
        self.window_epochs = float(window_epochs)

    def on_train_begin(self, logs=None):
        opt = self.model.optimizer
        opt = getattr(opt, "inner_optimizer", opt)
        steps = (self.params or {}).get("steps")
        if not steps or not hasattr(opt, "_set_ema_cap"):
            return
        cap = min(float(opt.ema_momentum), 1.0 - 1.0 / max(self.window_epochs * steps, 2.0))
        opt._set_ema_cap(cap)
        print(f"   [EmaWindow] {steps} خطوة/حقبة × {self.window_epochs:g} ⇒ زخم EMA الأقصى {cap:.5f}")


def _test_ema_warmup():
    """60 خطوة فقط: EMA بإحماء يتبع الأوزان المدرَّبة، وEMA كيراس الافتراضي يبقى قرب التهيئة."""
    import io, contextlib
    x = np.random.default_rng(0).normal(size=(64, 3)).astype("float32")
    y = x @ np.array([[1.0], [2.0], [3.0]], "float32")
    gaps = {}
    for warm in (False, True):
        tf.keras.utils.set_random_seed(0)
        m = tf.keras.Sequential([tf.keras.Input((3,)), tf.keras.layers.Dense(1, use_bias=False)])
        opt = build_optimizer({"name": "adam", "lr_initial": 0.1, "use_ema": True, "ema_momentum": 0.999,
                               "ema_warmup": warm})
        m.compile(opt, "mse")
        m.fit(x, y, batch_size=64, epochs=60, verbose=0)
        # بُعد متوسط EMA عن الحل الحقيقي [1,2,3] (fit يستبدل الأوزان بمتوسط EMA في نهايته)
        gaps[warm] = float(np.abs(opt._model_variables_moving_average[0].numpy().ravel() - [1.0, 2.0, 3.0]).max())
    assert gaps[True] < 0.5 < gaps[False], gaps
    # EmaWindow: 64 عيّنة / 16 = 4 خطوات/حقبة، نافذة حقبتين ⇒ سقف 1 - 1/8
    m = tf.keras.Sequential([tf.keras.Input((3,)), tf.keras.layers.Dense(1, use_bias=False)])
    opt = build_optimizer({"name": "adam", "lr_initial": 0.1, "use_ema": True, "ema_momentum": 0.999, "ema_warmup": True})
    m.compile(opt, "mse")
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(tf.data.Dataset.from_tensor_slices((x, y)).batch(16), epochs=1, verbose=0, callbacks=[EmaWindow(2.0)])
    assert abs(opt._get_ema_cap() - 0.875) < 1e-6, opt._get_ema_cap()
    return True


def build_optimizer(opt_cfg: dict) -> tf.keras.optimizers.Optimizer:
    common = dict(
        learning_rate=opt_cfg["lr_initial"],
        global_clipnorm=opt_cfg.get("clip_norm"),
        use_ema=opt_cfg.get("use_ema", False),
        ema_momentum=opt_cfg.get("ema_momentum", 0.999),
    )
    if opt_cfg.get("name", "adamw") == "adamw":
        opt = tf.keras.optimizers.AdamW(weight_decay=opt_cfg.get("weight_decay", 0.0), **common)
        exclude = list(opt_cfg.get("weight_decay_exclude") or [])
        if exclude:
            opt.exclude_from_weight_decay(var_names=exclude)   # قبل build: يُقرأ عند بناء المتغيرات
    else:
        opt = tf.keras.optimizers.Adam(**common)
    if common["use_ema"] and opt_cfg.get("ema_warmup", False):
        enable_ema_warmup(opt)

    # ⚙️ Mixed precision: إن كانت السياسة العامة لـ Keras تستخدم float16 للحساب، نغلّف الـ optimizer
    # تلقائيًا بـ LossScaleOptimizer: إلزامي مع mixed_float16 لمنع اختفاء (underflow) التدرجات الصغيرة جدًا.
    policy = tf.keras.mixed_precision.global_policy()
    if policy.compute_dtype == "float16":
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    return opt


def _resolve_train_mode(train_mode: str, run_dir: str) -> str:
    if train_mode != "auto":
        return train_mode
    return "resume" if has_saved_state(run_dir) else "new"


def build_training_system(
    model_builder_fn: Callable[[], tf.keras.Model],
    config: dict,
    sample_batch: Tuple[Any, Any],
):
    """نقطة الدخول الوحيدة. تُستدعى بنفس الطريقة دائمًا؛ السلوك يُحدَّده config['run']['train_mode']:

      "auto" (افتراضي)  : resume إن وُجدت حالة سابقة في run_dir، وإلا new — هذا هو السلوك القديم بالضبط.
      "new"              : تجاهل أي حالة سابقة (تُؤرشَف تلقائيًا في run_dir/_archive/<timestamp>/، لا تُحذف).
      "resume"           : يفشل بوضوح إن لم توجد حالة — بدل أن يبدأ صامتًا من جديد بغيابها بالخطأ.
      "warm_start"       : يحمّل أوزان model فقط من config['run']['warm_start']['weights_path'] (optimizer/
                           الجداول جديدة)، ويستأنف جداول العقوبات من epochs_done بدل الصفر — هذا يحل مباشرة
                           مشكلة «عقوبات الثقة تبدأ من صفر عند إكمال التدريب فتفسد الموثوقية».
    """
    run = config["run"]
    run_dir = run["run_dir"]
    mirror_dir = run.get("mirror_dir")
    train_mode = run["train_mode"]
    mount_drive_if_needed(run_dir)

    fingerprint = config_fingerprint(config)

    # ── إعادة استخدام نفس الكائنات إن كانت موجودة بالفعل في هذه الجلسة، وبنفس البصمة ──
    cached = TRAINER_REGISTRY.get(run_dir)
    if cached is not None:
        if cached["fingerprint"] != fingerprint:
            print("♻️ config تغيّر منذ آخر بناء لهذا run_dir في هذه الجلسة — تجاهل الكائن المخزَّن وإعادة البناء")
        else:
            print("♻️ تم العثور على مدرّب قائم لهذا الـ run_dir (نفس config) في نفس الجلسة — إعادة استخدامه")
            trainer = cached["trainer"]
            return trainer, cached["callbacks"], int(trainer.ckpt_epoch.numpy())

    # ── استعادة نسخة احتياطية كاملة إن كان run_dir فارغًا وmirror_dir يحوي بيانات ──
    if mirror_dir:
        mount_drive_if_needed(mirror_dir)
        if not has_saved_state(run_dir) and has_saved_state(mirror_dir):
            print(f"📥 استعادة نسخة احتياطية كاملة: {mirror_dir} → {run_dir}")
            shutil.copytree(mirror_dir, run_dir, dirs_exist_ok=True)

    train_mode = _resolve_train_mode(train_mode, run_dir)
    is_resuming = (train_mode == "resume")

    if train_mode == "resume" and not has_saved_state(run_dir):
        raise FileNotFoundError(
            f"❌ run.train_mode='resume' لكن لا يوجد checkpoint في {run_dir}. "
            f"استخدم train_mode='new' للبدء من الصفر عمدًا، أو 'auto' ليختار الإطار تلقائيًا.")

    warm_start_staged = warm_start_epochs_done = warm_rewarm_epochs = None
    if train_mode == "warm_start":
        warm_start_staged, warm_start_epochs_done, warm_rewarm_epochs = stage_warm_start_weights(config)

    if train_mode in ("new", "warm_start") and has_saved_state(run_dir):
        if run["on_existing"] == "error":
            raise FileExistsError(
                f"❌ توجد حالة تدريب سابقة في {run_dir} وrun.on_existing='error'. "
                f"غيّرها إلى 'archive' للسماح بنقلها تلقائيًا، أو استخدم run_dir جديدًا.")
        tag = time.strftime("%Y%m%d_%H%M%S")
        dest = archive_state(run_dir, tag)
        print(f"🗄️ أُرشِفت حالة التشغيل السابقة إلى: {dest} (لم تُحذف)")

    tf.keras.utils.set_random_seed(run.get("seed", 42))

    base_model = model_builder_fn()
    if train_mode == "warm_start":
        base_model.load_weights(warm_start_staged)
        shutil.rmtree(os.path.dirname(warm_start_staged), ignore_errors=True)
        print(f"🌱 [warm_start] أوزان النموذج حُمِّلت من نسخة أُنجزت {warm_start_epochs_done} حقبة — "
              f"optimizer جديد بالكامل، والجداول ستستأنف من هذه الحقبة (لا من الصفر)")

    trainer = GenericTrainer(base_model, config)
    lr_fn = build_lr_schedule_fn(
        config["optimizer"],
        rewarm_from_epoch=warm_start_epochs_done if train_mode == "warm_start" else None,
        rewarm_epochs=warm_rewarm_epochs or 0,
    )
    trainer.compile(
        optimizer=build_optimizer(config["optimizer"]),
        jit_compile=bool(config["optimizer"].get("use_xla", False)),
    )

    ckpt_mgr = CheckpointManager(trainer, run_dir=run_dir, max_to_keep=config["checkpoint"].get("max_to_keep", 3))

    # ── بناء كل متغيرات trainer (نموذج + optimizer) بلا أي خطوة تدريب (انظر القسم 7.1) ──
    build_trainer_variables(trainer, sample_batch)

    if is_resuming:
        initial_epoch, callback_states = ckpt_mgr.restore()
    else:
        initial_epoch, callback_states = 0, {}
        if train_mode == "warm_start":
            initial_epoch = warm_start_epochs_done
            trainer.ckpt_epoch.assign(warm_start_epochs_done)
            apply_schedules(trainer, config, warm_start_epochs_done)   # الجداول تبدأ من هنا لا من الصفر
        print(f"🆕 بدء تدريب {'دافئ (warm_start)' if train_mode == 'warm_start' else 'جديد'} — run_dir: {run_dir}")

    # ═══════════════════════════════ بناء الكولباكس من config بالكامل ═══════════════════════════════
    callbacks = []
    verbose = run.get("verbose", 1)

    for pname, pcfg in config["loss"].get("schedules", {}).items():
        callbacks.append(ParamScheduler(
            trainer.scheduled_vars[pname], start=pcfg["start"], end=pcfg["end"],
            warmup_epochs=pcfg.get("warmup_epochs", 0), schedule=pcfg.get("schedule", "linear"),
            label=pname, verbose=verbose,
        ))

    callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_fn, verbose=0))

    if run.get("strict_epoch_guard", True):
        callbacks.append(EpochGuard())

    ema_window = config["optimizer"].get("ema_window_epochs")
    if ema_window and config["optimizer"].get("use_ema") and config["optimizer"].get("ema_warmup"):
        callbacks.append(EmaWindow(ema_window))

    class_targets = [t for t, tcfg in config["targets"].items() if tcfg["task_type"] == "classification"]
    if class_targets:   # قبل BestModelTracker: يضيف class_loss/val_class_loss لسجلّ الحقبة (monitor ممكن)
        callbacks.append(DerivedMetrics(class_targets))

    es_cfg = config["callbacks"]["early_stopping"]
    best_path = os.path.join(run_dir, "best.weights.h5") if config["checkpoint"].get("save_best_weights", True) else None
    best_tracker = BestModelTracker(
        base_model=base_model, best_path=best_path,
        monitor=es_cfg["monitor"], mode=es_cfg.get("mode", "min"),
        patience=es_cfg["patience"], min_delta=es_cfg.get("min_delta", 1e-4),
        smoothing=es_cfg.get("smoothing", "window"), window=es_cfg.get("smoothing_window", 3),
        ema_beta=es_cfg.get("ema_beta", 0.7), restore_best_weights=es_cfg.get("restore_best_weights", True),
        verbose=verbose, initial_state=callback_states.get("BestModelTracker"),
        trainer=trainer, weights_snapshot=es_cfg.get("weights_snapshot", "raw"),
    )
    callbacks.append(best_tracker)
    stateful_callbacks = [best_tracker]

    tw_freq = config["callbacks"].get("task_weight_update_frequency", 0)
    if tw_freq and not trainer.use_uncertainty_weighting:
        callbacks.append(TaskWeightUpdater(trainer, update_frequency=tw_freq, verbose=verbose))

    metrics_logger = MetricsLogger(
        log_every=config["callbacks"].get("metrics_log_every", 5), verbose=verbose,
        initial_state=callback_states.get("MetricsLogger"),
        class_baselines=config["callbacks"].get("class_baselines"),
    )
    callbacks.append(metrics_logger)
    stateful_callbacks.append(metrics_logger)

    snap_cfg = config["callbacks"].get("snapshots", {})
    if snap_cfg.get("epochs"):
        callbacks.append(SnapshotEnsemble(
            save_epochs=snap_cfg["epochs"], save_dir=os.path.join(run_dir, snap_cfg.get("dir", "snapshots")),
            base_model=base_model, verbose=verbose,
        ))

    # الحفظ الكامل للحالة (آخر كولباك يضيف حالة، وقبل DriveMirror حتى تُنسخ أحدث نسخة)
    callbacks.append(EpochCheckpointCallback(
        ckpt_mgr, stateful_callbacks=stateful_callbacks,
        save_every=config["checkpoint"].get("save_every", 1), verbose=verbose,
    ))

    if mirror_dir:
        callbacks.append(DriveMirror(run_dir, mirror_dir, every=run.get("mirror_every", 1), verbose=verbose))

    TRAINER_REGISTRY[run_dir] = {
        "trainer": trainer, "ckpt_mgr": ckpt_mgr, "callbacks": callbacks,
        "fingerprint": fingerprint, "model_sig": model_signature(base_model),
    }

    print(f"\n{'=' * 70}\n🚀 نظام التدريب جاهز")
    print(f"   وضع البدء: {train_mode}")
    print(f"   الأهداف: {trainer.target_names}")
    print(f"   الاستئناف: {'نعم، من Epoch ' + str(initial_epoch + 1) if is_resuming else ('لا، تدريب جديد' if train_mode != 'warm_start' else f'لا — بدء دافئ من حقبة {initial_epoch}')}")
    print(f"   موازنة المهام التلقائية: {trainer.use_uncertainty_weighting}"
          + (f" (Kendall على: {trainer.kendall_task_names} | وزن ثابت: {trainer.fixed_task_names})"
             if trainer.use_uncertainty_weighting else ""))
    print(f"   المعاملات المجدولة: {list(trainer.scheduled_vars.keys())}")
    print(f"   اختيار الأفضل: monitor={es_cfg['monitor']} mode={es_cfg.get('mode','min')} smoothing={es_cfg.get('smoothing','window')}")
    print(f"{'=' * 70}\n")

    return trainer, callbacks, initial_epoch


_test_ema_warmup()

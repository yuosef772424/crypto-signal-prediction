"""
PURPOSE:  Smoke test: proves optimizer state survives an interruption and warm_start works (runs real tiny training at
          load).
TAGS:     smoke test, optimizer state, resume, warm_start, TRAINER_REGISTRY, interrupt
PITFALLS: RUNS AT LOAD (trains a few epochs on random data into a temp dir); tests exclude this module: load_into(ns,
          exclude=('smoke_test',)). Executed into the one shared trainer namespace by trainer/_loader.py (never
          imported on its own): names from other modules resolve at call time.

## 9) اختبار تحقّق فعلي (Smoke Test)

شغّل هذه الخلية للتأكد **بنفسك** أن مشكلة "فقدان زخم الـ optimizer عند
الاستئناف" قد حُلّت فعليًا، قبل استخدام الإطار على بياناتك الحقيقية. الاختبار:

1. يبني نموذجًا صغيرًا وهميًا بهدفين مختلفي النوع (`evidential` + `classification`)
   لإظهار مرونة الإطار مع مهام متعددة ومختلطة.
2. يدرّبه حقبتين، ثم يأخذ لقطة من قيم كل متغيرات الـ optimizer (بما فيها
   *slots* الزخم).
3. **يحذف المدرّب بالكامل من الذاكرة** (محاكاة انقطاع Colab).
4. يعيد بناء مدرّب جديد الكائن من الصفر لنفس `run_dir`، ويتأكد أن:
   - رقم الحقبة المستأنفة صحيح (`initial_epoch == 2`).
   - **كل** قيم متغيرات الـ optimizer مطابقة تمامًا لما كانت قبل "الانقطاع".
5. يستكمل التدريب حقبتين إضافيتين للتأكد أن كل شيء يعمل بسلاسة بعد الاستئناف.
6. **🆕 يتحقق أيضًا من `warm_start`**: يُدرِّب نموذجًا «منتهيًا» حتى تصل جداول العقوبات لقيمتها
   النهائية، ثم يستكمله بـ `train_mode='warm_start'` ويتأكد أن الجداول تُستأنف من تلك القيمة
   **فورًا** لا من الصفر — عكس ما يحدث مع `load_weights` بسيطة.
"""
# @title 9) Smoke Test — يثبت الحفاظ على حالة الـ optimizer + إصلاحات هذه النسخة
def _dummy_model_builder():
    inp = tf.keras.Input(shape=(8,), name="features")
    h = tf.keras.layers.Dense(16, activation="relu")(inp)

    mu = tf.keras.layers.Dense(1, name="y_a", dtype='float32')(h)
    nu = tf.keras.layers.Dense(1, activation="softplus", name="y_a_nu", dtype='float32')(h)
    alpha = tf.keras.layers.Dense(1, activation="softplus", name="y_a_alpha", dtype='float32')(h)
    beta = tf.keras.layers.Dense(1, activation="softplus", name="y_a_beta", dtype='float32')(h)
    conf = tf.keras.layers.Dense(1, activation="sigmoid", name="y_a_confidence", dtype='float32')(h)
    logits_b = tf.keras.layers.Dense(1, activation="sigmoid", name="y_b_logits", dtype='float32')(h)

    outputs = {
        "y_a": mu, "y_a_nu": nu, "y_a_alpha": alpha, "y_a_beta": beta, "y_a_confidence": conf,
        "y_b_logits": logits_b,
    }
    return tf.keras.Model(inputs=inp, outputs=outputs)


# Colab: /content (as before); elsewhere (CI runners, local non-root) the temp dir, like data/disk_backed.py's scratch root.
_SMOKE_ROOT = "/content" if os.path.isdir("/content") else tempfile.gettempdir()

smoke_config = build_config({
    "run": {"run_dir": os.path.join(_SMOKE_ROOT, "_smoke_test_run"), "epochs": 4, "batch_size": 32, "verbose": 1},
    "targets": {
        "a": {
            "true_key": "y_a", "task_type": "evidential",
            "output_keys": {"mu": "y_a", "nu": "y_a_nu", "alpha": "y_a_alpha",
                             "beta": "y_a_beta", "confidence": "y_a_confidence"},
            "use_calibration_loss": True, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib",
        },
        "b": {"true_key": "y_b", "task_type": "classification", "output_keys": {"logits": "y_b_logits"}, "binary": True},
    },
    "loss": {
        "use_uncertainty_weighting": True,
        "schedules": {
            "lambda_reg": {"start": 0.0, "end": 0.05, "warmup_epochs": 2, "schedule": "linear"},
            "lambda_calib": {"start": 0.0, "end": 0.1, "warmup_epochs": 2, "schedule": "cosine"},
        },
    },
    "optimizer": {"lr_initial": 1e-3, "lr_warmup_epochs": 0, "lr_schedule": {"type": "constant"}},
    "callbacks": {"early_stopping": {"patience": 100}, "metrics_log_every": 1},
})

shutil.rmtree(smoke_config["run"]["run_dir"], ignore_errors=True)

_rng = np.random.default_rng(0)
_N = 256
_X = _rng.normal(size=(_N, 8)).astype("float32")
_y_a = _rng.normal(size=(_N, 1)).astype("float32")
_y_b = _rng.integers(0, 2, size=(_N, 1)).astype("float32")
smoke_train_ds = tf.data.Dataset.from_tensor_slices((_X, {"y_a": _y_a, "y_b": _y_b})).batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
smoke_sample_batch = next(iter(smoke_train_ds))

print("── المرحلة 1: تدريب من الصفر لحقبتين ──")
trainer1, callbacks1, initial_epoch1 = build_training_system(_dummy_model_builder, smoke_config, smoke_sample_batch)
trainer1.fit(smoke_train_ds, initial_epoch=initial_epoch1, epochs=2, callbacks=callbacks1, verbose=1)

opt_values_before = [v.numpy().copy() for v in opt_variables(trainer1.optimizer)]

print("\n── محاكاة انقطاع Colab: مسح كل شيء من الذاكرة ──")
del trainer1, callbacks1
TRAINER_REGISTRY.pop(smoke_config["run"]["run_dir"], None)
tf.keras.backend.clear_session()

print("\n── المرحلة 2: إعادة بناء المدرّب من الصفر (Python) + استئناف من الـ checkpoint ──")
trainer2, callbacks2, initial_epoch2 = build_training_system(_dummy_model_builder, smoke_config, smoke_sample_batch)
assert initial_epoch2 == 2, f"❌ initial_epoch متوقع=2 لكن الفعلي={initial_epoch2}"

opt_values_after = [v.numpy() for v in opt_variables(trainer2.optimizer)]
assert len(opt_values_before) == len(opt_values_after), "❌ عدد متغيرات الـ optimizer غير متطابق!"
for a, b in zip(opt_values_before, opt_values_after):
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
print(f"✅ تم التحقق: كل متغيرات الـ optimizer ({len(opt_values_after)} متغيرًا، بما فيها الزخم) استُرجعت بدقة كاملة")

print("\n── المرحلة 3: استكمال التدريب حقبتين إضافيتين (3 و 4) ──")
trainer2.fit(smoke_train_ds, initial_epoch=initial_epoch2, epochs=4, callbacks=callbacks2, verbose=1)

print("\n🎉 الاختبار الأصلي نجح — الإطار يحافظ على حالة التدريب الكاملة عبر الاستئناف")

# ── 🆕 اختبار warm_start: عقوبات الثقة تستأنف من حقبتها، لا من الصفر ──
print(f"\n{'=' * 70}\n── اختبار warm_start ──\n{'=' * 70}")
warm_run_dir = os.path.join(_SMOKE_ROOT, "_smoke_test_warm_start")
shutil.rmtree(warm_run_dir, ignore_errors=True)
finished_cfg = deep_update(smoke_config, {"run": {"run_dir": warm_run_dir, "epochs": 6}})
finished_trainer, finished_cbs, ie = build_training_system(_dummy_model_builder, finished_cfg, smoke_sample_batch)
finished_trainer.fit(smoke_train_ds, initial_epoch=ie, epochs=6, callbacks=finished_cbs, verbose=0)
lambda_reg_at_end = float(finished_trainer.scheduled_vars["lambda_reg"].numpy())
assert abs(lambda_reg_at_end - 0.05) < 1e-6, f"❌ توقعنا lambda_reg=0.05 بعد اكتمال warmup لكن الفعلي={lambda_reg_at_end}"
weights_path = os.path.join(_SMOKE_ROOT, "_smoke_finished.weights.h5")
finished_trainer.model.save_weights(weights_path)

TRAINER_REGISTRY.pop(warm_run_dir, None)
tf.keras.backend.clear_session()

warm_cfg = deep_update(smoke_config, {
    "run": {"run_dir": os.path.join(_SMOKE_ROOT, "_smoke_test_warm_start_2"), "epochs": 8, "train_mode": "warm_start",
            "warm_start": {"weights_path": weights_path, "epochs_done": 6, "lr_rewarmup_epochs": 1}},
})
shutil.rmtree(warm_cfg["run"]["run_dir"], ignore_errors=True)
warm_trainer, warm_cbs, warm_ie = build_training_system(_dummy_model_builder, warm_cfg, smoke_sample_batch)
lambda_reg_after_warm_start = float(warm_trainer.scheduled_vars["lambda_reg"].numpy())
assert warm_ie == 6, f"❌ توقعنا initial_epoch=6 لكن الفعلي={warm_ie}"
assert abs(lambda_reg_after_warm_start - 0.05) < 1e-6, (
    f"❌ warm_start فشل في استئناف الجدول: lambda_reg={lambda_reg_after_warm_start} (توقعنا 0.05، لا 0.0)")
print(f"✅ تم التحقق: warm_start استأنف lambda_reg={lambda_reg_after_warm_start:.4f} من الحقبة {warm_ie} "
      f"(لا 0.0000 كما يحدث مع load_weights بسيطة)")
warm_trainer.fit(smoke_train_ds, initial_epoch=warm_ie, epochs=8, callbacks=warm_cbs, verbose=0)
print("🎉 اختبار warm_start نجح — التدريب استكمل بلا أي انهيار في عقوبات الثقة")

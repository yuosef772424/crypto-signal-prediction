"""
PURPOSE:  Usage template: real_model_builder, high_low_close_order_penalty and real_config (an example config).
TAGS:     example, real_model_builder, real_config, high_low_close_order_penalty, template
PITFALLS: Example only; real_config is built and printed at load. The commented training cell (10.3) is in the runner
          notebook. Executed into the one shared trainer namespace by trainer/_loader.py (never imported on its own):
          names from other modules resolve at call time.

## 10) قالب استخدام كامل على مشروعك الحقيقي

هذا القالب **مثال فقط** لإظهار المرونة (أهداف `high`/`low`/`close` من نوع
`evidential` + قيد منطقي بينها + هدف `direction` من نوع `classification`).
**أسماء الأهداف هنا ليست جزءًا من الإطار** — احذفها وضع أسماء أهدافك الحقيقية
(قد يكون هدفًا واحدًا فقط، أو خمسين هدفًا، بأي أسماء تريد).

عدّل الخلية التالية بالكامل حسب مشروعك: `real_model_builder`، `real_config`،
والبيانات (`real_train_ds` / `real_val_ds`).
"""
# @title 10.1) مثال: بناء النموذج (بدّله ببنيتك الحقيقية)
def real_model_builder():
    """
    مثال بمدخل واحد فقط لغرض التبسيط. الإطار يدعم مدخلات متعددة بالتساوي —
    فقط مرّر tuple/dict من tf.keras.Input إلى inputs=[...] كما يفعل Keras عادة،
    فالإطار لا يفتح x إطلاقًا (يمرره لـ self.model(x) كما هو).
    """
    inp = tf.keras.Input(shape=(32,), name="features")
    h = tf.keras.layers.Dense(64, activation="relu")(inp)
    h = tf.keras.layers.Dense(64, activation="relu")(h)

    def evidential_head(prefix):
        # dtype='float32' صريح لكل مخرجات NIG — أمان إلزامي إن فعّلت mixed precision لاحقًا
        # (انظر التحذير في القسم 1.1)؛ بلا ضرر إطلاقًا إن كانت السياسة العامة float32 أصلًا.
        mu = tf.keras.layers.Dense(1, name=f"y_{prefix}", dtype='float32')(h)
        nu = tf.keras.layers.Dense(1, activation="softplus", name=f"y_{prefix}_nu", dtype='float32')(h)
        alpha = tf.keras.layers.Dense(1, activation="softplus", name=f"y_{prefix}_alpha", dtype='float32')(h)
        beta = tf.keras.layers.Dense(1, activation="softplus", name=f"y_{prefix}_beta", dtype='float32')(h)
        conf = tf.keras.layers.Dense(1, activation="sigmoid", name=f"y_{prefix}_confidence", dtype='float32')(h)
        return mu, nu, alpha, beta, conf

    outputs = {}
    for t in ["high", "low", "close"]:
        mu, nu, alpha, beta, conf = evidential_head(t)
        outputs.update({f"y_{t}": mu, f"y_{t}_nu": nu, f"y_{t}_alpha": alpha,
                         f"y_{t}_beta": beta, f"y_{t}_confidence": conf})

    outputs["y_direction_logits"] = tf.keras.layers.Dense(1, activation="sigmoid", name="y_direction_logits", dtype='float32')(h)

    return tf.keras.Model(inputs=inp, outputs=outputs)


# ── مثال على قيد منطقي اختياري بين المخرجات (high >= low, close بين الاثنين) ──
def high_low_close_order_penalty(outputs):
    h, l, c = outputs["y_high"], outputs["y_low"], outputs["y_close"]
    viol_hl = tf.nn.relu(l - h)
    viol_hc = tf.nn.relu(c - h)
    viol_lc = tf.nn.relu(l - c)
    return tf.reduce_mean(viol_hl + viol_hc + viol_lc)


# @title 10.2) مثال: قاموس الإعدادات الكامل لمشروعك
real_config = build_config({
    "run": {
        "run_dir": "/content/drive/MyDrive/training_runs/my_project_v1",  # ⚠️ غيّره لمسارك
        "mirror_dir": None,
        "epochs": 60,
        "batch_size": 64,
        "seed": 42,
        "verbose": 1,
        # 🆕 وضع البدء — اتركه "auto" في الاستخدام العادي (نفس السلوك القديم: يستأنف إن وُجدت حالة).
        # لإكمال تدريب نموذج *لم يُدرَّب عبر هذا الإطار* (مثل حالتك — نموذج منتهٍ محفوظ بـ save_weights
        # بسيطة) استخدم warm_start كما في القسم 11 أدناه، حتى لا تبدأ عقوبات الثقة من الصفر مجددًا.
        "train_mode": "auto",   # auto | new | resume | warm_start
    },
    "targets": {
        "high": {
            "true_key": "y_high", "task_type": "evidential",
            "output_keys": {"mu": "y_high", "nu": "y_high_nu", "alpha": "y_high_alpha",
                             "beta": "y_high_beta", "confidence": "y_high_confidence"},
            "use_calibration_loss": True, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib",
            # 🆕 خيارات اختيارية (معلَّقة) — فعّلها فقط بعد تجربتها على High أولًا ومقارنة val_mae:
            # "normalize_reg": True,      # Meinert et al. 2023 — يقلل تسرّب aleatoric إلى epistemic
            # "beta_nll": 0.5,            # امتداد تجريبي غير موثَّق رسميًا لـ NIG — قيّمه بنفسك
            # "huber_weight": 0.05,
        },
        "low": {
            "true_key": "y_low", "task_type": "evidential",
            "output_keys": {"mu": "y_low", "nu": "y_low_nu", "alpha": "y_low_alpha",
                             "beta": "y_low_beta", "confidence": "y_low_confidence"},
            "use_calibration_loss": True, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib",
        },
        "close": {
            "true_key": "y_close", "task_type": "evidential",
            "output_keys": {"mu": "y_close", "nu": "y_close_nu", "alpha": "y_close_alpha",
                             "beta": "y_close_beta", "confidence": "y_close_confidence"},
            "use_calibration_loss": True, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib",
        },
        "direction": {
            "true_key": "y_direction", "task_type": "classification",
            "output_keys": {"logits": "y_direction_logits"}, "binary": True, "loss_weight": 0.5,
        },
    },
    "loss": {
        "use_uncertainty_weighting": True,
        "schedules": {
            "lambda_reg":   {"start": 0.0, "end": 0.05, "warmup_epochs": 20, "schedule": "linear"},
            "lambda_calib": {"start": 0.0, "end": 0.15, "warmup_epochs": 20, "schedule": "cosine"},
            "penalty_weight": {"start": 0.0, "end": 0.3, "warmup_epochs": 25, "schedule": "linear"},
        },
        "constraints": [
            {"name": "ohlc_order", "fn": high_low_close_order_penalty, "weight_var": "penalty_weight"},
        ],
    },
    "optimizer": {
        "name": "adamw", "lr_initial": 5e-5, "lr_min": 5e-7, "lr_warmup_epochs": 3,
        "lr_schedule": {"type": "cosine_restarts", "cycle_length": 10, "cycle_mult": 1.5},
        "weight_decay": 1e-4, "clip_norm": 1.0, "use_ema": True, "ema_momentum": 0.999,
    },
    "callbacks": {
        "early_stopping": {
            "monitor": "val_loss", "patience": 15, "min_delta": 1e-4,
            "smoothing": "window", "smoothing_window": 3,   # أو "ema" لتنعيم أسّي بدل نافذة ثابتة
            # 🆕 "ema_weights" يحفظ متوسط EMA (يتطلب optimizer.use_ema=True أعلاه، وهو مفعّل هنا) بدل
            # الأوزان الخام عند كل تحسّن — أكثر استقرارًا للنشر النهائي. اتركها "raw" إن كنت تفضّل
            # الأوزان الخام كما كانت في النسخة السابقة من الإطار.
            "weights_snapshot": "raw",
        },
        "metrics_log_every": 5,
    },
    "checkpoint": {"save_every": 1, "max_to_keep": 3, "save_best_weights": True},
})

print(json.dumps({"targets": list(real_config["targets"].keys()),
                   "schedules": list(real_config["loss"]["schedules"].keys()),
                   "constraints": [c["name"] for c in real_config["loss"]["constraints"]]}, indent=2, ensure_ascii=False))

"""
PURPOSE:  Optional ensemble inference for evidential targets: ensemble_predict_evidential(_meinert).
TAGS:     ensemble_predict_evidential, ensemble_predict_evidential_meinert, ensemble, inference, NIG mixture
PITFALLS: Optional; models must share the same output_keys format. Executed into the one shared trainer namespace by
          trainer/_loader.py (never imported on its own): names from other modules resolve at call time.

## 13) (اختياري) استدلال Ensemble معمَّم لأي هدف evidential

`output_keys` بنفس صيغة `target_cfg['output_keys']` المستخدمة في `config` —
بلا أي افتراض لاسم الهدف. استدعها بـ
`ensemble_predict_evidential(models, x, real_config['targets']['close']['output_keys'])`
مثلًا.
"""
# @title 13) Ensemble Inference (اختياري)
def ensemble_predict_evidential(models: List[tf.keras.Model], x, output_keys: dict) -> Dict[str, np.ndarray]:
    mus, nus, alphas, betas = [], [], [], []
    for m in models:
        out = m(x, training=False)
        mus.append(out[output_keys["mu"]].numpy())
        nus.append(out[output_keys["nu"]].numpy())
        alphas.append(out[output_keys["alpha"]].numpy())
        betas.append(out[output_keys["beta"]].numpy())

    mus, nus, alphas, betas = (np.stack(a, axis=0) for a in (mus, nus, alphas, betas))

    mu_ensemble = mus.mean(axis=0)
    aleatoric_within = (betas / (alphas - 1.0 + 1e-6)).mean(axis=0)
    epistemic_within = (betas / (nus * (alphas - 1.0) + 1e-6)).mean(axis=0)
    epistemic_between = mus.var(axis=0)  # تباين آراء النماذج المختلفة = عدم يقين حقيقي إضافي

    epistemic_total = epistemic_within + epistemic_between
    total_uncertainty = aleatoric_within + epistemic_total
    confidence = np.clip(1.0 / (1.0 + total_uncertainty), 0.01, 0.99)

    return {
        "mu": mu_ensemble, "aleatoric": aleatoric_within,
        "epistemic": epistemic_total, "confidence": confidence,
    }


def ensemble_predict_evidential_meinert(models: List[tf.keras.Model], x, output_keys: dict) -> Dict[str, np.ndarray]:
    """نفس ensemble_predict_evidential أعلاه، لكن بعدم يقين كل نموذج مُعرَّفًا بطريقة Meinert et al. 2023
    (aleatoric=عرض Student-t w_St، epistemic=1/√ν) بدل تعريف Amini et al. 2020 الأصلي. جرّب كلا الدالتين
    وقارن المعايرة الفعلية (calibration) على بيانات تحقق حقيقية — لا يوجد تعريف "صحيح" مطلق للفصل بين
    aleatoric/epistemic، والورقتان تختلفان في أيّهما أفضل تجريبيًا حسب طبيعة البيانات."""
    mus, alea, epi_inv_sqrt_nu = [], [], []
    for m in models:
        out = m(x, training=False)
        nu = out[output_keys["nu"]].numpy()
        alpha = out[output_keys["alpha"]].numpy()
        beta = out[output_keys["beta"]].numpy()
        u = nig_uncertainties_meinert(tf.constant(nu), tf.constant(alpha), tf.constant(beta))
        mus.append(out[output_keys["mu"]].numpy())
        alea.append(u["aleatoric_wst"].numpy())
        epi_inv_sqrt_nu.append(u["epistemic_inv_sqrt_nu"].numpy())

    mus, alea, epi_inv_sqrt_nu = (np.stack(a, axis=0) for a in (mus, alea, epi_inv_sqrt_nu))
    mu_ensemble = mus.mean(axis=0)
    aleatoric_within = alea.mean(axis=0)                 # بوحدات y مباشرة (عكس aleatoric الأصلي)
    epistemic_within = epi_inv_sqrt_nu.mean(axis=0)       # بلا وحدات — مؤشر نسبي فقط، لا تقارنه رقميًا بـ aleatoric
    epistemic_between = mus.var(axis=0)

    return {
        "mu": mu_ensemble, "aleatoric": aleatoric_within,
        "epistemic_relative": epistemic_within + epistemic_between,  # لا نجمعه مع aleatoric (وحدات مختلفة)
    }

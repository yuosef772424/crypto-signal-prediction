"""
PURPOSE:  Section 7 reports on the trained model: latest trading report, real-price predictions, classification accuracy report.
TAGS:     latest_trading_report, real_price_predictions, classification_accuracy_report, invert_reg_predictions
PITFALLS: Read model, test_dict, EVAL_TARGET_SPECS from the notebook namespace. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted verbatim from main.ipynb cell 23 (section 7).
"""
def latest_trading_report(n_display=5):
    """تقرير 'آخر N عيّنات' لكل أصل — للتداول الحيّ، يعمل بلا أهداف حقيقية."""
    return predict_latest_all_assets(
        model, test_dict, timeframes=_tfs_of(), target_specs=EVAL_TARGET_SPECS,
        n_display=n_display,
    )


def real_price_predictions(asset, target):
    """يحوّل مخرَج النموذج (عائد مباشر) لسعر حقيقي عبر invert_reg_predictions
    نفسها المستخدَمة في خط الأنابيب — مفيد حين تحتاج سعراً لا عائداً.
    ``invert_reg_predictions`` تتوقّع اسم *رأس* (مثلاً 'close_reg')، لا اسم
    الهدف المجرَّد — راجع split_head_name في خط الأنابيب.
    entry_range (القسم ٣-ب): الأهداف مقادير من آخر إغلاق لا عوائد من سعر نوعها — يُعكس بـ entry_range_to_prices
    (اتجاه close من رأس تصنيفه؛ target='close_up'/'close_down' يعيدان الاتجاهين معاً)؛ invert_reg_predictions لا يعرف
    هذا الوضع."""
    split = test[asset]
    x = model_x(split)
    if target_mode_of(split) == "entry_range":
        out = model(x, training=False)
        mu = {t: out[f"y_{t}"].numpy().ravel() / reg_scale_of(split, t) for t in ("high", "low", "close")
              if f"y_{t}" in out}
        p_up = out["y_close_class_logits"].numpy().ravel() if "y_close_class_logits" in out else None
        prices = entry_range_to_prices(split["last_candles"][:, LAST_COLUMNS.index("last_close")],
                                       close_reg=entry_close_reg_of(split), p_close_up=p_up, **mu)
        if target not in prices:
            raise ValueError(f"entry_range: سعر {target!r} غير متاح — رؤوس النموذج: {sorted(mu)}"
                             + (" (close بـ range_pos يحتاج high وlow؛ وبـ abs_return يحتاج رأس تصنيف close)"
                                if target.startswith("close") else ""))
        return prices[target]
    preds = model(x, training=False)[f"y_{target}"].numpy().ravel()
    mode = target_mode_of(split)
    if mode == "scaled":
        # (السعر_المستقبلي − آخر_سعر_من_نوعه) / IQR (القسم ٣-ب) ← آخر_سعر + التوقّع × IQR، بنفس حماية القسمة على صفر
        # في retarget_splits. (فكرة PR #8 بتعريف scaled الحالي: مركزه آخر سعر من النوع نفسه لا مركز base_params.)
        iqr = np.asarray(split["base_params"], dtype="float64")[:, 1]
        iqr = np.where(np.abs(iqr) > 1e-12, iqr, 1e-8)
        last = split["last_candles"][:, LAST_COLUMNS.index(f"last_{target}")].astype("float64")
        return last + preds.astype("float64") / reg_scale_of(split, target) * iqr
    if mode not in (None, "return"):
        # invert_reg_predictions يفترض «عائداً نسبة لآخر سعر من نوع الهدف»؛ غيره يعطي سعراً خاطئاً بلا خطأ.
        raise ValueError(f"real_price_predictions: وضع الهدف {mode!r} لا يُعكس لسعر هنا (المدعوم: return وscaled "
                         "وentry_range) — عكسه بعائد نفس النوع يعطي سعراً خاطئاً بصمت")
    return invert_reg_predictions(preds, f"{target}_reg", last_candles=split["last_candles"], config=CONFIG,
                                  scale=reg_scale_of(split))


def classification_accuracy_report():
    """دقة/AUC رؤوس التصنيف الثنائي (صعود/هبوط) لكل هدف وأصل — منفصل عن
    تقرير chicks (مخصَّص للأهداف المستمرة فقط، انظر ملاحظة القسم ٦).
    y_true من خط الأنابيب بترميز +1.0/-1.0 — يُحوَّل هنا بنفس _to_unit_label
    المستخدَمة في التدريب (القسم ٥) قبل المقارنة."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    rows = []
    for asset, split in test.items():
        x = model_x(split)
        out = model(x, training=False)
        for t in PRICE_TARGETS:
            y_true = _to_unit_label(np.asarray(split["y"][f"y_{t}_class"]).ravel())
            y_prob = out[f"y_{t}_class_logits"].numpy().ravel()
            y_pred = (y_prob >= 0.5).astype("float32")
            auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan")
            rows.append({"asset": asset, "target": t, "n": len(y_true),
                         "accuracy": accuracy_score(y_true, y_pred), "auc": auc})
    return pd.DataFrame(rows)

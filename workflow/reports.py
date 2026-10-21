"""
PURPOSE:  Section 7 reports on the trained model: latest trading report, real-price predictions, classification accuracy report, per-sample prediction table (decoded truth beside prediction, confidence, uncertainty, class).
TAGS:     latest_trading_report, real_price_predictions, classification_accuracy_report, invert_reg_predictions, prediction_table, prediction_table_summary
PITFALLS: model, test / test_dict, eval_target_specs, price_targets and model_tf are explicit arguments (RunResult / ChicksInputs / ModelPlan / DatasetInfo of workflow/run.py): no notebook global is read. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own): names from other modules and the %run notebooks resolve at call time. Extracted from main.ipynb cell 23 (section 7).
"""
def latest_trading_report(model, test_dict, eval_target_specs, model_tf, n_display=5):
    """تقرير 'آخر N عيّنات' لكل أصل — للتداول الحيّ، يعمل بلا أهداف حقيقية.
    test_dict وeval_target_specs من ChicksInputs (prepare_chicks)؛ model_tf من DatasetInfo."""
    return predict_latest_all_assets(
        model, test_dict, timeframes=_tfs_of(model_tf), target_specs=eval_target_specs,
        n_display=n_display,
    )


def real_price_predictions(model, test, asset, target, model_tf):
    """يحوّل مخرَج النموذج (عائد مباشر) لسعر حقيقي عبر invert_reg_predictions
    نفسها المستخدَمة في خط الأنابيب — مفيد حين تحتاج سعراً لا عائداً.
    ``invert_reg_predictions`` تتوقّع اسم *رأس* (مثلاً 'close_reg')، لا اسم
    الهدف المجرَّد — راجع split_head_name في خط الأنابيب.
    entry_range (القسم ٣-ب): الأهداف مقادير من آخر إغلاق لا عوائد من سعر نوعها — يُعكس بـ entry_range_to_prices
    (اتجاه close من رأس تصنيفه؛ target='close_up'/'close_down' يعيدان الاتجاهين معاً)؛ invert_reg_predictions لا يعرف
    هذا الوضع."""
    split = test[asset]
    x = model_x(split, model_tf)
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
    # price_norm_mode='logret_std': price_norm_meta (إحصاءات train المختومة على القسم) ← العكس last_close × exp(pred·std + mean)
    return invert_reg_predictions(preds, f"{target}_reg", last_candles=split["last_candles"], config=CONFIG,
                                  scale=reg_scale_of(split), price_norm_meta=price_norm_meta_of(split))


def classification_accuracy_report(model, test, price_targets, model_tf):
    """دقة/AUC رؤوس التصنيف الثنائي (صعود/هبوط) لكل هدف وأصل — منفصل عن
    تقرير chicks (مخصَّص للأهداف المستمرة فقط، انظر ملاحظة القسم ٦).
    y_true من خط الأنابيب بترميز +1.0/-1.0 — يُحوَّل هنا بنفس _to_unit_label
    المستخدَمة في التدريب (القسم ٥) قبل المقارنة."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    rows = []
    for asset, split in test.items():
        x = model_x(split, model_tf)
        out = model(x, training=False)
        for t in price_targets:
            y_true = _to_unit_label(np.asarray(split["y"][f"y_{t}_class"]).ravel())
            y_prob = out[f"y_{t}_class_logits"].numpy().ravel()
            y_pred = (y_prob >= 0.5).astype("float32")
            auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan")
            rows.append({"asset": asset, "target": t, "n": len(y_true),
                         "accuracy": accuracy_score(y_true, y_pred), "auc": auc})
    return pd.DataFrame(rows)


def _prediction_frame(asset, split, out, price_targets):
    """صفّ لكل عيّنة لقسم واحد: الحقيقة المفكوكة بجانب التنبؤ وبقية المخرجات (انظر prediction_table)."""
    lc = np.asarray(split["last_candles"], dtype="float64")
    n = len(lc)
    o = {k: np.asarray(v, dtype="float64").reshape(n, -1)[:, 0] for k, v in out.items()}
    last_close = lc[:, LAST_COLUMNS.index("last_close")]
    meta = price_norm_meta_of(split)

    def decode(v, t):
        # y_*_reg = (عائد مقصوص) × مقياس الهدف ← قسمة المقياس (affine). logret_*: العكس الوحيد invert_reg_predictions ← سعر ← log(سعر/آخر_إغلاق)
        v = np.asarray(v, dtype="float64")
        if meta is None:
            return v / reg_scale_of(split, t)
        price = invert_reg_predictions(v, f"{t}_reg", last_candles=split["last_candles"], config=CONFIG,
                                       scale=reg_scale_of(split), price_norm_meta=meta)
        return np.log(price / last_close)

    df = pd.DataFrame({"asset": asset,
                       "timestamp": pd.to_datetime(lc[:, LAST_COLUMNS.index("timestamp")].astype("int64")),
                       "entry": last_close})
    for t in price_targets:
        y = split["y"]
        if f"y_{t}" in o and f"y_{t}_reg" in y:
            true, pred = decode(np.asarray(y[f"y_{t}_reg"]).ravel(), t), decode(o[f"y_{t}"], t)
            df[f"true_{t}"], df[f"pred_{t}"], df[f"err_{t}"] = true, pred, pred - true
            for short, key in (("epi", "epistemic"), ("ale", "aleatoric")):
                if f"y_{t}_{key}" in o:   # عدم اليقين بوحدة الحقيقة المفكوكة: affine ⇒ الفرق عن فكّ الصفر يزيل الإزاحة
                    df[f"{short}_{t}"] = decode(o[f"y_{t}_{key}"], t) - decode(np.zeros(n), t)
            if f"ale_{t}" in df:
                ale = df[f"ale_{t}"].to_numpy()
                df[f"z_{t}"] = np.where(ale > 0, (pred - true) / np.where(ale > 0, ale, 1.0), np.nan)
            if f"y_{t}_confidence" in o:
                df[f"conf_{t}"] = o[f"y_{t}_confidence"]
        if f"y_{t}_class_logits" in o and f"y_{t}_class" in y:
            p = o[f"y_{t}_class_logits"]              # احتمال بعد sigmoid (لا logit خام)
            cls = np.asarray(y[f"y_{t}_class"]).ravel() > 0
            df[f"p_{t}"], df[f"cls_{t}"], df[f"hit_{t}"] = p, cls.astype(int), ((p >= 0.5) == cls).astype(int)
    return df


def prediction_table(model, split_or_dict, model_tf, price_targets, batch_size=1024, outputs=None):
    """جدول لكل عيّنة: الحقيقة المفكوكة بجانب التنبؤ وبقية مخرجات النموذج، لرؤية كيف تتفاعل النتائج مع القيم الحقيقية.

    split_or_dict : قسم واحد (train/val) أو قاموس {أصل: قسم} (test). price_targets من ModelPlan.price_targets.
    الأعمدة (لكل هدف t، وتُحذف التي لا مخرج لها — مثلاً class_only بلا رؤوس انحدار):
      true_t / pred_t / err_t = pred − true : بوحدة الهدف بعد الفكّ — عائد (return، magnitude، entry_range، relative)
                      أو عائد لوغاريتمي (price_norm_mode=logret_*) أو IQR (scaled)؛ الحقيقة هي y_*_reg كما دُرِّب عليها (بعد القصّ).
      epi_t / ale_t : عدم اليقين المعرفي (epistemic) والعشوائي (aleatoric) بوحدة الحقيقة؛ z_t = err/ale (بقيم معيارية).
      conf_t        : رأس الثقة (0.05–0.98، مقطوع التدرّج عن الجذع).
      p_t / cls_t / hit_t : احتمال الفئة 1 (sigmoid)، التسمية الحقيقية 0/1، وهل أصاب التصنيف عند 0.5.
    ويضاف asset وtimestamp وentry (آخر إغلاق). outputs: مخرجات جاهزة (قسم واحد فقط) بدل model.predict — للاختبار."""
    price_targets = _required(price_targets, "price_targets")
    parts = _split_parts(split_or_dict)
    if outputs is not None and len(parts) != 1:
        raise ValueError("prediction_table: outputs الجاهزة تصحّ لقسم واحد فقط، لا لقاموس أصول")
    frames = []
    for asset, split in parts:
        out = outputs if outputs is not None else model.predict(model_x(split, model_tf), batch_size=batch_size, verbose=0)
        frames.append(_prediction_frame("all" if asset is None else asset, split, out, price_targets))
    return pd.concat(frames, ignore_index=True)


def prediction_table_summary(df, price_targets, n_bins=5, verbose=True):
    """ملخّص تفاعل المخرجات مع الحقيقة من جدول prediction_table. يُرجع {"overview": جدول لكل هدف، "<t>_by_conf": ...،
    "<t>_by_margin": ...}:
      overview : spearman(pred, true)، MAE، ارتباط عدم اليقين (ale/epi) والثقة بالخطأ المطلق، دقّة التصنيف وAUC
                 مقابل نسبة الفئة الأغلب. عدم يقين صادق = ارتباط موجب مع |الخطأ|؛ ثقة صادقة = ارتباط سالب.
      by_conf  : متوسط |الخطأ| ودقّة التصنيف لكل خُمس من الثقة (conf_t). by_margin: لكل خُمس من |p − 0.5|."""
    from sklearn.metrics import roc_auc_score

    def sp(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:   # ثابت ⇒ الارتباط غير معرَّف (لا تحذير)
            return np.nan
        return float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman"))

    rows, out = [], {}
    for t in price_targets:
        row = {"target": t}
        if f"true_{t}" in df:
            true, pred = df[f"true_{t}"].to_numpy(), df[f"pred_{t}"].to_numpy()
            abs_err = np.abs(df[f"err_{t}"].to_numpy())
            row.update(n=len(df), spearman_pred_true=sp(pred, true), mae=float(np.nanmean(abs_err)),
                       spearman_ale_abs_err=sp(df[f"ale_{t}"].to_numpy(), abs_err) if f"ale_{t}" in df else np.nan,
                       spearman_epi_abs_err=sp(df[f"epi_{t}"].to_numpy(), abs_err) if f"epi_{t}" in df else np.nan,
                       spearman_conf_abs_err=sp(df[f"conf_{t}"].to_numpy(), abs_err) if f"conf_{t}" in df else np.nan)
        if f"p_{t}" in df:
            cls, hit = df[f"cls_{t}"].to_numpy(), df[f"hit_{t}"].to_numpy()
            row.update(class_acc=float(hit.mean()), class_base=float(max(cls.mean(), 1 - cls.mean())),
                       class_auc=float(roc_auc_score(cls, df[f"p_{t}"])) if len(set(cls.tolist())) > 1 else np.nan)
            frame = pd.DataFrame({"hit": hit, "margin": np.abs(df[f"p_{t}"].to_numpy() - 0.5)})
            if f"conf_{t}" in df:
                frame["conf"] = df[f"conf_{t}"].to_numpy()
            if f"err_{t}" in df:
                frame["abs_err"] = np.abs(df[f"err_{t}"].to_numpy())
            for key in ("conf", "margin"):
                if key in frame and frame[key].nunique() > 1:
                    q = pd.qcut(frame[key], n_bins, duplicates="drop")
                    agg = {"n": ("hit", "size"), "hit": ("hit", "mean"), key: (key, "mean")}
                    if "abs_err" in frame:
                        agg["abs_err"] = ("abs_err", "mean")
                    out[f"{t}_by_{key}"] = frame.groupby(q, observed=True).agg(**agg).reset_index(drop=True)
        rows.append(row)
    out["overview"] = pd.DataFrame(rows).set_index("target")
    if verbose:
        with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
            print("📋 تفاعل المخرجات مع الحقيقة — spearman_*_abs_err: موجب = عدم اليقين يتتبّع الخطأ؛ سالب للثقة = صادقة")
            print(out["overview"].to_string())
            for k, v in out.items():
                if k != "overview":
                    print(f"\n{k}:")
                    print(v.to_string())
    return out

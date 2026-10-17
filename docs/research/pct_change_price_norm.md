# تجربة التطبيع النسبي الهندسي للأسعار (`price_norm_mode='pct_change'`)

> **ملاحظة (S7 — `main.ipynb` مُوجَّه بالإعدادات):** المتغيّرات المذكورة بأسمائها القديمة في خطوات Colab أدناه (`TARGET_MODE`، `ENTRY_CLOSE_REG`، `MODEL_TFS`، `DATA_FILENAME_BASE`، `RUN_MAIN_TRAINING`، `PANEL_MODE`، `PANEL_PRESET`، `PANEL_*`...) صارت حقولاً في `settings` (خلية «٢) الإعدادات»):
> `target.target_mode`، `target.entry_close_reg`، `data.model_tfs`، `data.filename_base`، `train.run_main_training`، `panel.enabled`، `panel.preset`، `panel.*`. الجدول الكامل: [`main_lab_code_layout.md`](main_lab_code_layout.md) («كان / صار»). الخطوات نفسها والمجلدات نفسها (`run_dir_for`)، والنتائج المسجَّلة أدناه لم تتغيّر.

## الفكرة

أعمدة `price_level` في X (`close`، و`open`/`high`/`low` والمتوسطات ونطاقات بولنجر حين تُفعَّل) كانت تُطبَّع بمرجع النافذة:
`(x − وسيط إغلاق النافذة) / IQR`، مقصوصة عند ±5. الوضع الجديد يُطبّعها كتغيّر نسبي **هندسي** عن الشمعة السابقة لنفس العمود
بالنقاط المئوية:

```
r_t = 100 × (x_t / x_{t−1} − 1)          سعر 100 ← 99 ← 103.95 ← 103.53   →   r = [0, −1, +5, −0.4]
```

- أول صف في كل نافذة `0` (لا سابق له داخلها).
- القصّ عند ±`price_pct_clip` (افتراضياً 100) لا عند ±`clip_abs` (5): حركة +7% تبقى +7.
- بقية أنواع الميزات (ATR/MACD بوحدة السعر، المذبذبات، الحجم...) لا تتغيّر.
- الأهداف لا تتغيّر: في إعداد 1h_s8 هي عوائد × `reg_target_scale=100`، أي نقاط مئوية أيضاً — المدخل والهدف بوحدة واحدة.

الافتراضي `price_norm_mode='window_scale'` هو السلوك السابق حرفياً: البيانات المحفوظة ونقاط الاستئناف والنتائج المسجّلة تبقى صالحة،
ومجموعة بيانات مبنيّة بالوضع القديم لا تحمل المفتاح أصلاً.

## الفكّ (من القيم النسبية إلى الأسعار)

الترميز قابل للعكس بالضرب التراكمي من مرساة واحدة — آخر سعر حقيقي في النافذة:

```
x_{t−1} = x_t / (1 + r_t / 100)          (للخلف من المرساة؛ أو للأمام من أول سعر بـ anchor_at='first')
```

| الدالة | الاستخدام |
|---|---|
| `pct_change_encode(x, axis)` | الترميز لأي مصفوفة. |
| `pct_change_decode(r, anchor, anchor_at='last', axis)` | الفكّ لأي مصفوفة. |
| `decode_price_window(dataset, "close")` | أسعار نوافذ X الحقيقية `(N, T)`؛ المرساة `last_close`/`last_high`/`last_low` من `last_candles`. لغير هذه الأعمدة أو لفريم أعلى مرّر `anchor=` صراحةً. لقسم من `split_data` مرّر `feature_order=` و`mode=` (الأقسام لا تحملهما). |
| `pct_decode_consistency(dataset)` | فحص على البيانات المبنيّة: السعر المفكوك من نافذة عيّنة يطابق `last_close` للعيّنة السابقة لنفس العملة (مرساة مستقلّة). |

الفكّ دقيق ما دامت القيم غير مقصوصة ولم تُخزَّن X بـ float16 (`x_storage_dtype`). على بيانات تركيبية: خطأ نسبي < 1e-5
(`tests/test_pct_change_norm.py`).

## تشغيل التجربة (Colab)

تغيير واحد فقط عن 1h_s8 (`PCT_CHANGE_OVERRIDES = {**HOURLY_W32_S8_OVERRIDES, "price_norm_mode": "pct_change"}`): نفس العملات
والنافذة والـstride والأفق والتقسيم والـholdout، فتُقارَن النتائج ببيانات 1h_s8 مباشرة.

1. خط الأنابيب (`crypto_data_pipeline_v6`، القسم 20-د؛ الكود في `data/presets.py`):
   `dataset = build_hourly_pct_dataset(checkpoint_dir="/content/drive/MyDrive/crypto_model/ckpt_1h_w32_s8_pct")`
   يطبع تدقيق التطبيع وفحص الفكّ، ويحفظ باسم `HOURLY_PCT_NAME = "preprocessing_output_1h_w32_s8_h1_pct"`.
2. `main.ipynb`: `DATA_FILENAME_BASE = HOURLY_PCT_NAME`. يقرأ `price_norm_mode` من البيانات نفسها ويطبعه.
3. قارن بتشغيل 1h_s8 بنفس البذور ومقاييس التقييم.

## ما يجب مراقبته

- **مستوى السعر يختفي من المدخل**: لا يرى النموذج موقع السعر داخل نافذته (قرب القمة/القاع) إلا بتراكم التغيّرات؛ `POS_50` و`RANGE_rel`
  ما زالا يحملان شيئاً منه.
- **ذيل ثقيل**: عوائد الساعة متطرّفة أحياناً (±20%)، فقد يعلّم `normalization_audit` في main عمود `close` بـ«ذيل متطرّف» — معلومة
  حقيقية لا خلل تطبيع. `audit_normalization` في خط الأنابيب يقيس أعمدة `pct_change` بحدّ `price_pct_clip` لا `CLIP_ABS`.
- مقياس المدخل (نقاط مئوية، انحراف ≈ 1 على 1h) قريب من بقية الميزات؛ على فريم يومي يكبر (انحراف ≈ 3–5).

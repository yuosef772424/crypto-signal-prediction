# تدقيق مستقل — الجولة ١ (commit c62b4f0)

**النطاق:** المسار الذي يُنتج نتائج Colab القادمة: `crypto_data_pipeline_v6.ipynb` بإعداد `HOURLY_W32_S8_OVERRIDES`،
و`split_data`، و`main.ipynb` القسم ٧-ح (`PANEL_PRESET="1h_s8"`)، وحزمة `cross_asset/`.

**الطريقة:** كل سكربت يشغّل دوال الدفتر أو الحزمة نفسها، لا نسخاً منها. `_nbload.py` ينفّذ خلايا الدفتر بلا Colab.
البيانات تركيبية فقط، إذ لا بيانات 1h حقيقية في هذه البيئة. لم يُشغَّل أي تدريب كامل؛ repro_02 وrepro_05 يدرّبان
مُرمِّزاً صغيراً بضع حقب على CPU. كل سكربت `repro_0[1-5]` يفشل الآن بـ `AssertionError`، ويجب أن ينجح بعد الإصلاح.

## الملخّص

| # | الخلل | الخطورة | السكربت |
|---|---|---|---|
| F1 | مجموعات «أرضية» فوق طوابع غير متحاذية تضع مستقبل العملة داخل نوافذ أقرانها، والانتباه يقرؤه. فحوص اللوحة تمرّرها (`ok=True`) | يحرّف النتائج (نتائج stride 32 السابقة ومعيار 0.05 المبني عليها) | `repro_05_floor_groups_leak_future_via_attention.py` |
| F2 | AUC في `panel_compare.csv` يُحسب مقابل تسمية غير التي دُرِّب عليها النموذج | يحرّف النتائج (التقرير) | `repro_01_report_auc_label_mismatch.py` |
| F3 | المتغيّر `A_ic_k` يتدرّب على ~56% من عيّنات train في كل حقبة، و`A_ic` على 100%، بنفس الحقب والصبر | يحرّف النتائج (معيار A_ic_k مقابل A_ic) | `repro_04_a_ic_k_sees_half_the_data.py` |
| F4 | بعد ثغرة في صفوف العملة، يُؤخذ الهدف من شمعة لاحقة لا من ts+1h | ثانوي | `repro_03_label_across_data_gap.py` |
| F5 | بصمة الاستئناف لا تغطّي محتوى البيانات، فبيانات مختلفة بنفس الشكل تُستأنف بأوزان قديمة | ثانوي | `repro_02_resume_fingerprint_ignores_data.py` |

لم يثبت خلل يوقف تشغيل Colab. التشغيل التجريبي للقسم ٧-ح كاملاً (3 متغيّرات، تقييم k، بيانات بناها خط الأنابيب) انتهى بلا
أخطاء وكتب كل الملفات.

---

## F1 — تسرّب المستقبل عبر الانتباه في مجموعات الأرضية غير المتحاذية

**الأمر:** `python docs/research/audit/repro_05_floor_groups_leak_future_via_attention.py` (~1 دقيقة CPU)

**الإعداد:** نافذة 32 ساعة، أفق 4 ساعات، stride 32، مجموعات أرضية بعرض 32 ساعة، و24 عملة كل منها بطور مختلف.
هذا هو شكل بيانات 1h stride 32 الموصوف في `hourly_1h.md` §١. الإشارة الوحيدة في البيانات هي حركة السوق في الساعات
الأربع القادمة، وكل عملة ترى ماضيها فقط.

**المخرَج:**
```
unaligned floor groups: ok=True misaligned=1150 samples whose target lies in a peer's window=95.83%
aligned groups        : ok=True misaligned=0 samples whose target lies in a peer's window=0.00%
unaligned test AUC (label = next 4h market move): A (attention) 0.720 | B (no attention) 0.498
aligned   test AUC (label = next 4h market move): A (attention) 0.522 | B (no attention) 0.476
AssertionError: DEFECT: groups whose peers' windows contain a member's target pass the panel checks
```

| | المتوقَّع | الفعلي |
|---|---|---|
| مجموعة تغطي فيها نافذةُ عملةٍ هدفَ عملةٍ أخرى | ترفضها `check()`، أو ترفع `build_panel_splits` خطأً | `ok=True` (عمود `misaligned_timestamps` معلوماتي فقط) |
| AUC لنموذج بلا معلومة حقيقية | ~0.5 للمتغيّرين | A=0.720 مقابل B=0.498 |

**الأثر على المسار الحالي:** بيانات `1h_s8` مبنيّة على الشبكة (`misaligned=0` في `repro_00`)، فهي لا تمرّ بهذا المسار
ما دام الملف المحمَّل هو الملف المحاذى. أرقام A_ic/B_ic على stride 32 في `hourly_1h.md` §٤ بُنيت بهذا التجميع: أرضية
32h فوق ~30 طوراً. ومعيار `1h_s8` الأول يقارن بقيمتها (`ic_asym` = 0.05).

**الإصلاح المقترح:** اجعل `ok=False` حين `misaligned_timestamps > 0`. بديلاً عن ذلك، ارفض أي مجموعة يزيد فيها الفرق
بين أكبر طابع وأصغره على 0. ثم أعد تقييم نتائج stride 32 أو أسقطها كمرجع للمعيار.

## F2 — AUC في `panel_compare.csv` مقابل تسمية مختلفة

**الأمر:** `python docs/research/audit/repro_01_report_auc_label_mismatch.py`

**الإعداد:** أهداف `return` (`TARGET_MODE=None` كما في `1h_s8`)، وclose معلّق، ودرجة `p_up_high` تطابق تسمية التدريب
`fut_high > last_high` تماماً.

**المخرَج:**
```
AUC high vs trained label (evaluate_k_coins / panel_k_coins.csv): 1.0000
AUC high reported by report.summarize (panel_compare.csv):         0.7592
AUC low  reported by report.summarize (panel_compare.csv):         0.7583
```

| | المتوقَّع | الفعلي |
|---|---|---|
| `auc_high` في `summarize` لدرجة مثالية | 1.0 | 0.7592. `report.prepare` يعيد اشتقاق `cls_*` كـ«أعلى من وسيط المجموعة» مهما كان وضع الهدف |

نفس التشغيل يُخرج رقمين مختلفين باسم `auc_high` للإشارات نفسها. في التشغيل التجريبي لـ A_ic_k:
`panel_compare` = 0.5609، و`panel_k_coins` (k=all) = 0.5730.

**الإصلاح المقترح:** مرّر وضع الهدف إلى `summarize`/`prepare`. في `return` تكون التسمية `raw > 0`، وفي `relative`
تكون `raw > median`. بديلاً عن ذلك، أعد تسمية العمود `auc_*_vs_median`.

## F3 — `A_ic_k` يرى نصف بيانات التدريب في كل حقبة

**الأمر:** `python docs/research/audit/repro_04_a_ic_k_sees_half_the_data.py`

**المخرَج:**
```
A_ic    min_coins=None: fraction of training samples used in one epoch = 1.000
A_ic_k  min_coins=5: fraction of training samples used in one epoch = 0.558
```

| | المتوقَّع | الفعلي |
|---|---|---|
| تغطية train في الحقبة لـ `A_ic_k` (83 عملة/مجموعة) | ≈ 1.0 مثل `A_ic`، ليقيس المعيار «أثر السياق المتغيّر» وحده | 0.558. `make_batch(min_coins)` يُسقط العملات غير المختارة، ويبقى عدد الحقب والصبر نفسه |

**الإصلاح المقترح:** قسّم كل مجموعة تدريب إلى أجزاء عشوائية بحجم k تغطي كل العملات، على غرار `chunk_groups`، بدل
إسقاط الباقي. بديلاً عن ذلك، عوّض بعدد الحقب ووثّق ذلك في المعيار.

## F4 — الهدف يُؤخذ من شمعة لاحقة بعد ثغرة

**الأمر:** `python docs/research/audit/repro_03_label_across_data_gap.py`

**المخرَج:**
```
AAAUSDT sample ts=2024-02-01 08:00 : target candle should open 2024-02-01 09:00, label taken from candle opening 2024-02-02 09:00
BBBUSDT sample ts=2024-01-14 00:00 : target candle should open 2024-01-14 01:00, label taken from candle opening 2024-01-14 08:00
```

- **AAA:** في الملف الخام 5 ساعات ناقصة. `resample('1h')` يعيد إدراجها صفوفاً بأسعار NaN وحجم 0، ثم يحذف `dropna` في
  `add_features` تلك الصفوف و~20 صفاً بعدها. النتيجة ثغرة 25 ساعة، والهدف من شمعة بعد 25 ساعة.
- **BBB:** 20 ساعة بسعر ثابت وحجم صفر. `dropna` يحذف 7 صفوف في منتصف السلسلة بصمت، والهدف من شمعة بعد 7 ساعات.

| | المتوقَّع | الفعلي |
|---|---|---|
| شمعة الهدف | تُفتح عند ts + 1h، أو تُسقط العيّنة | الصف التالي (`end_idx + 1`) أيّاً كان زمنه |

الأثر الجانبي: نافذة الـ32 صفاً بعد الثغرة تمتدّ أكثر من 32 ساعة. وإن وقعت الثغرة قبل `train_end` مباشرة فقد يمتدّ هدف
train إلى داخل نوافذ val.

**الإصلاح المقترح:** في `prepare_single_asset` تحقّق من `price_df.index[fe-1] == ts + horizon·tf`، ومن أن النافذة
تغطي `win` ساعة بالضبط، وإلا أسقط العيّنة. وأسقط صفوف NaN الناتجة عن `resample` قبل المؤشرات.

## F5 — الاستئناف لا يتحقّق من محتوى البيانات

**الأمر:** `python docs/research/audit/repro_02_resume_fingerprint_ignores_data.py`

**المخرَج:**
```
DEFECT: resumed silently, epochs in state=1, history rows=1 (no epoch trained on B; B signals come from A's weights)
```

| | المتوقَّع | الفعلي |
|---|---|---|
| `run_panel_variant` على بيانات B (نفس الشكل والتواريخ، X وy مختلفان) في مجلد A | خطأ بصمة أو إعادة تدريب | استئناف بلا تدريب، وتصدير إشارات B بأوزان A |

الحالة العكسية تُوقف التشغيل: إن كان في `…/crypto_model_v1_am_panel/panel_*` تشغيل سابق بعدد عيّنات مختلف، يرفع
التشغيل `RuntimeError`. هذا المسار مشترك لكل بيانات `TARGET_MODE=None`.

**الإصلاح المقترح:** أضف إلى البصمة تجزئة لـ `last_candles` (الطوابع والأسعار) ولعيّنة من X، ومجلد تشغيل مشتقاً من
`DATA_FILENAME_BASE`.

---

## شكوك غير مثبتة (بلا repro)

| # | الشكّ | ما يلزم لإثباته |
|---|---|---|
| S1 | الميزات `FUND_rate` و`FUND_rate_z` و`FUND_available` و`OI_chg_1` و`OI_available` (5 من 23) تصبح ثابتة إن غاب أرشيف التمويل/OI لعملة ما، بلا تحذير عند البناء. على البيانات التركيبية بلا أرشيف: std=0 للخمس | انحراف كل عمود على `X_1h` الحقيقي |
| S2 | انحياز البقاء: الكون هو 83 عملة من `COINS_BY_CATEGORY` الحالية، والعملات المشطوبة غائبة | قائمة العملات المدرجة تاريخياً |
| S3 | فترة test (2025-11-22 ← 2026-09-26) استُخدمت مراراً في تحليلات 1h السابقة (`hourly_1h.md` §٣–٦)، فليست holdout جديدة | — |
| S4 | لا مقياس عدم يقين للفرق A_ic−B_ic: بذرتان، بلا اختبار مزدوج على مستوى الطابع. `hourly_1h.md` §٩ يذكر t بطريقة Newey-West، ولا كود يحسبها (`grep -ri newey` فارغ) | حساب ذلك على `signals_test.csv.gz` |
| S5 | أوزان EMA تُحفظ مع إحصاءات BatchNorm المتحرّكة للأوزان الحيّة لا لأوزان EMA | مقارنة val بإعادة حساب إحصاءات BN |
| S6 | بصمة نقاط الاستئناف في خط الأنابيب لا تشمل محتوى الملفات الخام ولا نسخة الكود | — |

## ما فُحص ووُجد سليماً

**الأمر:** `python docs/research/audit/repro_00_clean_checks.py` (ينجح)
```
1. truncation: 881 rows compared | features that changed: [] | y equal: True
2. on 8h grid: 100% | coins per timestamp median 10 of 10
3. purge: free hours train->val 7, val->test 7
4. last-step std: {'MKT_BREADTH_24': 0.315, 'MOM_ORTH_NATR': 0.526}
```

فُحص أيضاً بقراءة الكود أو بالتشغيل التجريبي:
- **التطبيع:** لكل نافذة وحدها، و`robust_scales` من train فقط.
- **التسميات:** `{t}_class` بترميز 1/0، و`PanelSplit` يقرأ `y > 0`، و`_to_unit_label` يقبل الترميزين.
- **التجميع:** المجموعات بالطابع الدقيق 8h، نقية، وكل عملة مرّة واحدة.
- **`chunk_groups`:** كل عيّنة تُتنبّأ مرّة بسياق k بالضبط. في التشغيل التجريبي `context_mean` = 5 عند k=5، وB ثابت
  عبر k.
- **المحاذاة:** IC وAUC في `evaluate` و`evaluate_k_coins` محاذيان بالصفوف عبر `idx`، والبعثرة (day, pos) فريدة.
- **الإيقاف المبكر:** الأفضل على val، و`load_best` قبل التصدير. test لا يدخل أي اختيار.
- **`MKT_BREADTH`:** يستبعد العملات غير المدرجة؛ فحص مستقل أعطى 1.0 قبل الإدراج و2/3 بعده.
- **ادّعاء الـ50 حقبة:** صحيح. التسخين 3 ثم دورات 10 و15 و22، والحقبة 50 هي آخر الدورة الثالثة.
- **الحفظ:** 1.3 GB مضغوطة مرّتين تأخذ ~2 دقيقة (قيس 100 MB في 4.6 ث).
- **التشغيل التجريبي:** القسم ٧-ح كاملاً على `split_data` حقيقي، بأهداف (high, low) و`group_ns_for("1h", 8)` و`k_eval`،
  انتهى بلا أخطاء في 23 ث.

## سجلّ الفرضيات

| id | الادّعاء | الحالة | الدليل |
|---|---|---|---|
| H1 | لا نظر للمستقبل في الميزات الـ23، بما فيها العابرة للأصول | مثبت (تركيبي) | repro_00 §1 |
| H2 | نهايات النوافذ على شبكة 8h مشتركة | مثبت (تركيبي) | repro_00 §2 |
| H3 | فجوة العزل 33h كافية مع stride 8 | مثبت | repro_00 §3 |
| H4 | ميزتا MKT_BREADTH وMOM_ORTH لم تعودا ميّتتين (مسار التحميل) | مثبت (تركيبي) | repro_00 §4 |
| H5 | الاتساع لا يحسب العملات غير المدرجة | مثبت | فحص مستقل، القسم أعلاه |
| H6 | تسرّب الهدف المُقاس (`scaled`) مُصلَح | مثبت بالقراءة (`_raw_target` لا يقرأ المركز) | main، الخلية 10 |
| H7 | مجموعات اللوحة نقية ولا تمزج طوابع | مرفوض جزئياً: صحيح للبيانات المحاذاة، والفحص لا يمنع غير المحاذاة | repro_05 |
| H8 | «A_ic وحده يحمل معلومة اتجاه» (stride 32) | غير موثوق: التجميع المستخدم يسمح بتسرّب عبر الانتباه | repro_05 |
| H9 | AUC في `panel_compare` = AUC تسميات التدريب | مرفوض | repro_01 |
| H10 | A_ic_k يختلف عن A_ic في السياق فقط | مرفوض | repro_04 |
| H11 | الهدف = الشمعة التالية مباشرة | مرفوض عند الثغرات | repro_03 |
| H12 | بصمة الاستئناف تمنع خلط تشغيلات مختلفة | مرفوض جزئياً | repro_02 |
| H13 | t بطريقة Newey-West تُحسب للمقاييس الأهم | غير منفَّذ | `grep -ri newey` فارغ |
| H14 | ذروة رام البناء ≈ 2.1× X | لم يُفحص | — |
| H15 | الميزات الـ23 كلها غير ثابتة على البيانات الحقيقية | لم يُفحص | S1 |

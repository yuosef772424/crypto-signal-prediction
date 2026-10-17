# المشروع البحثي — فهرس التوثيق

توثيق مسار اكتشاف الإشارات، مقسَّم بحسب الموضوع بدل ملف واحد متضخّم — كل ملف يقابل مرحلة أو فرضية محدَّدة، وأي واحد منها قد يعتمد على الذي قبله. راجع أيضاً [خطة المشروع البحثي الكاملة](../../خطة%20نظام%20اكتشاف%20الإشارة%20—%20مشروع%20توقع%20اتجاه%20العملات.md) للمنهجية والدوافع.

## أين كود خط الأنابيب (`crypto_data_pipeline_v6`)

كل كود تجهيز البيانات نُقل حرفياً من خلايا `crypto_data_pipeline_v6.ipynb` إلى الحزمة [`data/`](../../data/) (الدفتر صار مُشغِّلاً رفيعاً
يحمّلها في نطاقه بـ`data.load_into(globals())`؛ الأسماء المتاحة بعد `%run` هي نفسها). الوثائق المسجَّلة قبل النقل تشير إلى «الخلية N» أو
«القسم N» من الدفتر القديم؛ هذا جدول التحويل (أرقام الخلايا في الدفتر قبل النقل → الوحدة؛ الخريطة الحالية في [`maps/data.md`](../../maps/data.md)):

| الخلية (قبل النقل) | الوحدة | الخلية | الوحدة |
|---|---|---|---|
| 4 (استيرادات) | `data/common.py` | 36 (القسم 15، `build_dataset`) | `data/pipeline.py` |
| 7 (Drive) | `data/drive.py` | 38 (15-ب، تطبيع مقطعي) | `data/cross_sectional_norm.py` |
| 9 (`DEFAULT_CONFIG`) | `data/defaults.py` | 40 (التقسيم) | `data/split.py` |
| 11 (`CONFIG`) | `data/runtime.py` | 42 (Binance) | `data/binance_client.py` |
| 14 (الرؤوس) | `data/heads.py` | 44 (التمويل/OI + المرحلة ٢) | `data/funding_oi.py`، `data/phase2.py` |
| 16 (مخصّصة) | `data/custom.py` | 46 (الحيّ) | `data/live.py` |
| 18 (الميزات) | `data/features.py` | 48 (الحفظ/التحميل) | `data/storage.py` |
| 20 (التطبيع، `FEATURE_KINDS`) | `data/normalize.py` | 51 (19-ب، الاختبارات الذاتية) | `data/selftests.py` |
| 22 (المحاذاة) | `data/align.py` | 54 (`default_workers` مكرَّرة) | `data/parallel.py` |
| 24 (النوافذ) | `data/windows.py` | 59 / 61 / 63 / 65 (20-ب / 20-ج / اختبارها / 20-د) | `data/presets.py` |
| 26 (التشخيص) | `data/diagnostics.py` | 33 (السياق السوقي) | `data/market_context.py` |
| 28 (التوازي) | `data/parallel.py` | 34 (رتبة الزخم المقطعية) | `data/cross_sectional_features.py` |
| 30 (المصادر) | `data/sources.py` | 35 (نقاط الاستئناف + بناء مدعوم بالقرص) | `data/checkpoints.py`، `data/disk_backed.py` |

للتعديل: غيّر ملف الوحدة في `data/` (لا الدفتر)، وراجع `CLAUDE.md` (تعديل لا يغيّر السلوك الافتراضي، واختبارات `tests/`).

## أين كود النموذج والمدرّب (`model_v2` و`trainer_framework_v2`)

كود النموذج نُقل حرفياً من خلايا `model_v2 (1).ipynb` إلى الحزمة [`model/`](../../model/)، وكود المدرّب العام من خلايا `trainer_framework_v2.ipynb`
إلى الحزمة [`trainer/`](../../trainer/) (الدفتران صارا مُشغِّلَين رفيعين يحمّلان حزمتيهما في نطاقهما بـ`model.load_into(globals())` /
`trainer.load_into(globals())`؛ الأسماء المتاحة بعد `%run` هي نفسها، وكل ما كان يعمل عند التحميل — الاختبار الذاتي للنموذج، واختبارات المدرّب
الذاتية واختبار الدخان (Smoke Test) — يعمل كما كان). الوثائق المسجَّلة قبل النقل تشير إلى «الخلية N» أو «§N» من الدفتر القديم؛ هذا جدول
التحويل (أرقام الخلايا 0-based في الدفتر قبل النقل → الوحدة؛ الخريطة الحالية في [`maps/model.md`](../../maps/model.md) و[`maps/trainer.md`](../../maps/trainer.md)):

| `model_v2 (1).ipynb` الخلية | الوحدة | `trainer_framework_v2.ipynb` الخلية | الوحدة |
|---|---|---|---|
| 1 (استيرادات، `register`) | `model/common.py` | 3 (§1 استيرادات، `IN_COLAB`، `atomic_write_json`) | `trainer/env.py` |
| 3 (§1 `InstanceNorm`، `SymLog`) | `model/input_norm.py` | 5 (§1.1 GPU وmixed precision) | `trainer/perf.py` |
| 5 (§2 تفكيك سببي) | `model/decomposition.py` | 7 (§2 `DEFAULT_CONFIG`، `build_config`) | `trainer/config.py` |
| 7 (§3 رُقَع) | `model/patches.py` | 10 (§3 سجلّ الخسائر) | `trainer/tasks.py` |
| 9 (§4 كتلة المحوّل) | `model/transformer.py` | 12 (§4 `UncertaintyWeightedLoss`) | `trainer/task_weighting.py` |
| 11 (§5 القراءة) | `model/readout.py` | 14 (§5 `GenericTrainer`) | `trainer/trainer.py` |
| 13 (§6 طبقات NIG) | `model/nig_layers.py` | 16 (§6.1 جدولة) | `trainer/schedules.py` |
| 15 (§7 `HEAD_REGISTRY`) | `model/heads.py` | 17 (§6.2 `BestModelTracker`) | `trainer/best_tracker.py` |
| 17 (§8 `build_nig_timenet_v2`) | `model/builder.py` | 18 (§6.3 `MetricsLogger`...) | `trainer/metrics.py` |
| 19 (§9 `MODEL_CONFIG`، `build_model_fn`) | `model/config.py` | 19 (§6.4 `SnapshotEnsemble`) | `trainer/snapshot.py` |
| 21 (§9-ب `diagnose_model`...) | `model/diagnostics.py` | 21 (§6.5 `TrainingDiagnostics`) | `trainer/training_diagnostics.py` |
| 23 (§9-ج تقرير الطبقات) | `model/layer_report.py` | 23 (§7.1 `CheckpointManager`) | `trainer/checkpoints.py` |
| 25 (§10 الاختبار الذاتي) | `model/selftests.py` | 24 (§7.2 `DriveMirror`، `EpochGuard`) | `trainer/epoch_callbacks.py` |
| | | 26 (§8 `build_training_system`) | `trainer/system.py` |
| | | 28 (§9 Smoke Test) | `trainer/smoke_test.py` |
| | | 30 + 31 (§10.1/10.2 مثال النموذج والإعداد) | `trainer/example.py` |
| | | 32 (§10.3 مثال التدريب، أسطر مُعلَّقة) | يبقى في دفتر `trainer_framework_v2.ipynb` |
| | | 35 (§12 K-Fold) | `trainer/kfold.py` |
| | | 37 (§13 Ensemble) | `trainer/ensemble.py` |

للتعديل: غيّر ملف الوحدة (لا الدفتر)، وراجع `CLAUDE.md`. الاختبارات تحمّل الحزمتين عبر `docs/research/audit/_nbload.load_model()` و`load_trainer()`
(بلا الاختبار الذاتي ولا Smoke Test ولا K-Fold)، وتفحص `tests/test_model_trainer_packages.py` البنية.
## أين كود تقييم النموذج (`chicks_v4_5_input_output_patterns`) ومحور الإشارة (`signal_evaluation_axis`)

كودا الدفترين نُقلا حرفياً إلى حزمتين: [`evaluation/`](../../evaluation/) (تقييم النموذج بعد التدريب؛ خريطتها [`maps/evaluation.md`](../../maps/evaluation.md))
و[`signal_eval/`](../../signal_eval/) (محور الإشارة، المرحلة ٠؛ [`maps/signal_eval.md`](../../maps/signal_eval.md)). الدفتران صارا مُشغِّلَين رفيعَين
(بلا أي `def`) يحمّلان حزمتيهما في نطاقهما بـ`evaluation.load_into(globals())` / `signal_eval.load_into(globals())`، فتبقى الأسماء نفسها لمن يستدعيهما بـ`%run`
(`main.ipynb`، ودفاتر الفرضيات `hypothesis_h00*.ipynb`). كل حزمة وحداتها تعمل في **نطاق واحد مشترك** (`_loader.py`)، فلا `import evaluation.<module>`.
الوثائق المسجَّلة قبل النقل تشير إلى «الخلية N» أو «القسم N»؛ هذان جدولا التحويل (أرقام الخلايا قبل النقل، 0-based):

**`chicks_v4_5_input_output_patterns.ipynb` ← `evaluation/`**

| الخلية (القسم) | الوحدة | الخلية (القسم) | الوحدة |
|---|---|---|---|
| 2 (١، `TargetSpec` + الاستيرادات) | `evaluation/targets.py` | 24 (١٠، الثقة والمعايرة) | `evaluation/trust_calibration.py` |
| 4 (٢، `save_or_print`) | `evaluation/outputs.py` | 26 (١١، تحليل موجَّه للمخرجات) | `evaluation/output_conditioned.py` |
| 6 (٣، التنبؤ) | `evaluation/predict.py` | 28 (١٢، اكتشاف الأنماط) | `evaluation/patterns.py` |
| 8 (٤، فك التشفير) | `evaluation/decode.py` | 30 (١٣، مقاييس التداول) | `evaluation/tearsheet.py` |
| 10 (٥، التحقق من فك التشفير) | `evaluation/verify.py` | 32 (١٤، الرسوم) | `evaluation/plots.py` |
| 12 (٦، المقاييس) | `evaluation/metrics.py` | 34 (١٥، نزاهة النموذج) | `evaluation/integrity.py` |
| 14 (٦.٥، جدول آخر العينات) | `evaluation/latest_table.py` | 36 (١٦، `run_full_analysis`) | `evaluation/full_analysis.py` |
| 16 (٧، `predict_with_evaluation_v4`) | `evaluation/unified.py` | 38 (١٧، تقرير قديم) | `evaluation/legacy_uncertainty.py` |
| 18 (٨، `test_all_assets_v4`) | `evaluation/all_assets.py` | 40 (١٨، محاكاة قديمة) | `evaluation/legacy_assets.py` |
| 20 (التداول الحي) | `evaluation/live.py` | 42–44 (١٩، اختيار الصفقات) | `evaluation/trade_selection.py` |
| 22 (٩، `build_flat_dataframe`) | `evaluation/flat.py` | 46–49 (٢٠، أنماط المدخلات ↔ المخرجات) | `evaluation/io_patterns.py` |

**`signal_evaluation_axis (3).ipynb` ← `signal_eval/`**

| الخلية (القسم) | الوحدة / الموضع الجديد |
|---|---|
| 2 (١، الاستيرادات) | `signal_eval/common.py` |
| 4 + 6 + 8 (٢–٤، `compute_ic`/`decile_spread`/`permutation_baseline`) | `signal_eval/core.py` |
| 10 (٥، `evaluate_windows`) | `signal_eval/windows.py` |
| 12 (٦، طبقة التكامل مع `rolling_splits`) | `signal_eval/integration.py` |
| 14 (٧، الاختبارات الذاتية) | التعريفات في `signal_eval/selftests.py`؛ **تشغيلها** (`PASS.clear()...`) خلية في الدفتر المُشغِّل |
| 15 (تنزيل `dataprocess.ipynb` + `%run` + تحميل `dataset`) | `download_notebook_from_drive` في `signal_eval/bootstrap.py`؛ خط الأنابيب يُحمَّل من `data/` (خلية في الدفتر المُشغِّل)؛ تحميل `dataset` خلية في الدفتر المُشغِّل |
| 16 (تحميل `dataset` من Drive) و18 (٨، بوّابة المرحلة ٠) | تبقيان خليتين في الدفتر المُشغِّل (دفاتر الفرضيات تعتمد على `dataset`/`windows`/`report` الناتجة عنهما) |
| 20 (٩، سجلّ التجارب) | `signal_eval/registry.py` |
| 22 (اختبارات السجلّ) | التعريف في `signal_eval/registry_selftests.py`؛ استدعاؤه خلية في الدفتر المُشغِّل |

**قرار `dataprocess.ipynb`:** كان الدفتر الأصلي ينزّل `dataprocess.ipynb` (نسخة من دفتر خط الأنابيب على Drive، غير موجودة في المستودع) ثم يعمل `%run` له ليحصل على
`rolling_splits` وCONFIG و`load_preprocessed_data_from_drive`. الدفتر المُشغِّل يحمّل الآن الحزمة `data/` (المصدر الحالي لخط الأنابيب نفسه) في نطاقه
**فقط إن لم تكن هذه الأسماء موجودة** (مثلاً بعد `%run "crypto_data_pipeline_v6.ipynb"`)؛ فإن وُجدت لا يتغيّر شيء. لا تنزيل من Drive بعد الآن. اختبارات `dataprocess` الذاتية
التي كانت تعمل عند ذلك `%run` (ونتائجها المحفوظة في مخرجات الدفتر القديم) لا تُشغَّل هنا؛ مكانها `run_pipeline_selftests` في `data/selftests.py`.

للتعديل: غيّر ملف الوحدة (لا الدفتر)، وراجع `CLAUDE.md` (تعديل لا يغيّر السلوك الافتراضي، واختبارات `tests/`: `tests/test_evaluation_packages.py` يثبّت البنية).
## أين كود `main` ومختبر الإشارات ومسح pandas_ta

نُقل حرفياً إلى الحزمتين [`workflow/`](../../workflow/) و[`discovery/`](../../discovery/) (الدفاتر مُشغِّلات)؛ جدول «الخلية القديمة ← الوحدة» في
[`main_lab_code_layout.md`](main_lab_code_layout.md).

## بنية دفاتر محور التقييم (`signal_evaluation_axis`)

كان `signal_evaluation_axis (3).ipynb` يضمّ المحور القياسي وكل الفرضيات المُختبَرة عبره في ملف واحد كبير. أُفرِد الآن إلى دفتر أساسي + دفتر مستقلّ لكل فرضية، كلٌّ منها يعتمد على الأساسي عبر `%run` (نفس أسلوب `main.ipynb` في تجميع الدفاتر):

| الدفتر | الدور |
|---|---|
| [`signal_evaluation_axis (3).ipynb`](../../signal_evaluation_axis%20(3).ipynb) | **المحور الأساسي فقط** (مُشغِّل رفيع؛ الكود في الحزمة `signal_eval/`): `compute_ic`/`decile_spread`/`permutation_baseline`/`evaluate_windows`، طبقة ربط بخط الأنابيب، وسجلّ التجارب (`register_hypothesis`/`list_registry`) — بلا أي فرضية مُختبَرة. |
| [`hypothesis_h001_short_term_reversal.ipynb`](../../hypothesis_h001_short_term_reversal.ipynb) | تسجيل واستقصاء [H001](h001_short_term_reversal.md) (مقبولة). |
| [`hypothesis_h002_classification_head.ipynb`](../../hypothesis_h002_classification_head.ipynb) | تسجيل [H002](h002_classification_head.md) (مرفوضة). |
| [`hypothesis_h003_volatility_reversal.ipynb`](../../hypothesis_h003_volatility_reversal.ipynb) | تسجيل [H003](h003_volatility_reversal.md) (مقبولة). |

كل دفتر فرضية مستقلّ التشغيل (خليته الأولى `%run "signal_evaluation_axis (3).ipynb"`)، فيمكن فتح أيٍّ منها منفرداً بلا الحاجة لقراءة البقية أولاً.

## الفرضيات المُختبَرة (بالترتيب الزمني)

| الفرضية | الحالة | الملخّص |
|---|---|---|
| [H001 — الانعكاس قصير المدى](h001_short_term_reversal.md) | ✅ مقبولة | آخر تغيّر سعر يتنبأ بعكس اتجاهه، لا استمراره. |
| [H002 — رأس التصنيف الثنائي](h002_classification_head.md) | ❌ مرفوضة | لا إشارة موثوقة على 5 أصول مترابطة، 8 حقب تدريب فقط. |
| [مرحلة ١-٢ — مكتبة مرشّحين + باكتيست قديم](discovery_lab_phase1_2.md) | ❌ مرفوضة (75 تجربة) | يكشف أيضاً خلل `MKT_beta` الصامت. |
| [مرحلة ٣ — Matrix Profile/SHAP/عنقدة](discovery_lab_phase3.md) | ❌ مرفوضة (117 تجربة) | يشمل اختبار WIDEHIST (كامل التاريخ المتاح، لا عيّنة صغيرة). |
| [معمارية مشتركة بين الأصول](cross_asset_architecture.md) | ⏸️ مؤجَّل | اختبار IC رخيص خطي لا يدعم الفكرة بصيغتها البسيطة. |
| [H003 — انعكاس التقلّب (`NATR_14`)](h003_volatility_reversal.md) | ✅ مقبولة | أول فرضية مقبولة رسمياً في تاريخ المشروع؛ عيّنة موسّعة 50 أصلاً. `Fractal_reversal_w5` (سؤال مفتوح ٢) اختُبِر فعلياً — مرفوض، يدعم أن الأثر خاص بالتقلّب لا انعكاساً هندسياً عاماً. |
| [مراجعة الاختبارات المتعددة لـH003](h003_multiple_testing_review.md) | ✅ أُغلِق بنتيجة قاطعة | سؤال H003 المفتوح ١: اختبار تبديل مشترك حقيقي (500 تكرار) — صفر اكتشافات كاذبة على الإطلاق مقابل 4 حقيقية؛ احتمال الصدفة <0.6%. تصحيح فيشر التقليدي فشل منهجياً (درس موثَّق). |
| [مرشّحو pandas_ta المؤجَّلون سابقاً](pandas_ta_deferred_candidates.md) | ❌ مرفوضة (27 تجربة) | CCI/WILLR/Vortex/DX/UO/CTI/SKEW/FISHER/RVI — high/low متاحة الآن، ولا إشارة جديدة. |
| [الدفعة الثانية من مرشّحي pandas_ta](pandas_ta_batch2_candidates.md) | ❌ مرفوضة (39) نهائياً | `CVI_14`/high و`ENTROPY_14`/low عبرا معيار المحور على عيّنة صغيرة — فشلا لاحقاً في ترقية H003. |
| [الدفعة الثالثة من مرشّحي pandas_ta](pandas_ta_batch3_candidates.md) | ❌ مرفوضة (67) + ⚠️ غير قابل للاختبار (10) نهائياً | تصحيح منهجي (الميزات الخام مُطبَّعة محلياً أصلاً)؛ `AOBV`/high عبر المعيار أيضاً — فشل لاحقاً في ترقية H003. |
| [ترقية H003 النهائية للثلاثة "قيد الاختبار"](pandas_ta_h003_upgrade_final.md) | ❌ مرفوضة (3 من 3) | **نتيجة منهجية مهمّة**: `CVI_14`/`ENTROPY_14`/`AOBV` فشلوا جميعاً على 50 أصلاً/30 نافذة (`consistent_sign` انهار) — تأكيد تجريبي لفخّ العيّنة الصغيرة. |
| [الدفعتان الرابعة والخامسة (نافذة أطول + مولّدات إشارة)](pandas_ta_batch4_5_final.md) | ❌ مرفوضة (84) + ⚠️ غير قابل للاختبار (2) | 28 مرشّحاً إضافياً، صفر نتائج جديدة — يختم مسار `SURVEY_CANDIDATES_ROBUST` (154 مرشّحاً فريداً، ~450 محاولة إجمالاً). |
| [مراجعة أدبيات خارجية](external_literature_review.md) | 📚 بحث/توصيات | بعد استنفاد pandas_ta: `funding_rate`/`open_interest` الحقيقيان (بنية تحتية جاهزة، دعم أدبي قوي) أعلى أولوية موصى بها؛ order book/on-chain غير قابلين للتنفيذ بالبيانات الحالية. |
| [إضافة `RVI_14`/`FISHER_14` كميزتين فعليتين](rvi_fisher_features_added.md) | ✅ مُنفَّذة ومُتحقَّق منها | إضافة اختيارية سببية (`custom_settings`)، مُتحقَّق منها على بيانات BTC حقيقية (88/88 اختباراً ذاتياً ناجح)؛ بانتظار تدريب `NIG-TimeNet v2` فعلي (يحتاج GPU) لقياس الأثر الحقيقي. |
| [مقدّرات تقلّب/سيولة كلاسيكية من خارج pandas_ta](new_volatility_liquidity_estimators.md) | ❌ مرفوضة (18 من 18) | أوّل بحث ميزات جديدة كلياً خارج pandas_ta (Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang/Amihud/Roll)، مُختبَرة مباشرة على مقياس H003 الكامل (50 أصلاً/30 نافذة)؛ `consistent_sign=False` بلا استثناء رغم IC مُجمَّع مضلِّل يبدو قوياً. |
| [أسّ هيرست وVariance Ratio من خارج pandas_ta](hurst_variance_ratio_features.md) | ❌ مرفوضة (12 من 12) | ثاني بحث ميزات جديدة؛ اختبارا انعكاس/استمرار مباشران (لا قياس حجم تقلّب فقط كـ`NATR_14`)، لكن `mean_ic` شبه صفري على مقياس H003 الكامل — رفض أوضح من مقدّرات التقلّب، يدعم أن أثر H003 خاصّ بالتقلّب المُطبَّع لا انعكاساً هندسياً عاماً. |
| [نسبة التقلّب القفزي (Bipower Variation) من خارج pandas_ta](jump_ratio_feature.md) | ❌ مرفوضة (6 من 6) | ثالث بحث ميزات جديدة؛ يختبر بنية التقلّب (قفزي/مستمرّ) بدل حجمه أو اتجاهه — رفض حاسم أيضاً. ثالث دليل مستقلّ (بعد Fractal وHurst/VR) على أن أثر H003 خاصّ بصيغة `NATR_14` تحديداً لا بمعناها الأعمّ. |
| [الاستخدام "الصحيح" لمؤشّرات معروفة (RSI/MACD/ADX/Bollinger/SuperTrend)](proper_indicator_signals.md) | ❌ مرفوضة (45 من 45) | بحث مختلف: كيف تُستخدَم مؤشّرات معروفة بصيغتها الصحيحة (تباعد/تقاطع/دمج DI+ADX/%B) حسب مصادرها الأصلية، لا قيمتها الخام. `RSI_14` الخام أقوى مفاجئاً من صيغة التباعد "الصحيحة" — `frac_significant=77%` لكن `consistent_sign=False`. |
| [الجولة الأولى من الفرضيات الأعمق (9 عائلات)](deeper_hypotheses_round1.md) | ❌ مرفوضة (63 توليفة) | فرضيات مبنية على معرفة اقتصادية-إحصائية حقيقية (اتفاق الزخم متعدّد الآفاق، تفاعلات مُمركَزة، ترتيب الزخم المقطعي، ارتباط BTC الدوّار (مباشر+مشروط)، عمر الاتجاه، اتساق الاتجاه، عدم تناظر التقلّب، الانتشار المتأخّر لعائد BTC) لا مسح آلي للمكتبات. ~~إعادة اكتشاف صادقة واحدة~~ (مسحوبة: كانت `−NATR_14` بسبب تمركز مزدوج)، صفر اكتشافات جديدة. |
| [نسبة كفاءة كوفمان الداخل-يومية (`EFF_RATIO_24H`)](kaufman_efficiency_ratio.md) | ❌ مرفوضة (3 من 3) | **أول ميزة في المشروع من بيانات ساعية حقيقية** (50/50 أصلاً، Google Drive) لا يومية فقط — بنية أرشيف خارجي جديدة (`intraday_efficiency`) بروح `funding_rate`/`open_interest`. نظافة الاتجاه الداخل-يومي ليوم واحد لا تتنبأ بحركة الأصل لاحقاً؛ تفتح الطريق لميزات داخل-يومية أخرى بنفس البنية. |
| [انحراف VWAP وتركّز الحجم الساعي](vwap_deviation_and_volume_concentration.md) | ❌ مرفوضتان (6 من 6) + ثلاث متابعات تشخيصية | ثاني وثالث ميزة من بيانات ساعية حقيقية. `VWAP_DEVIATION`/high: **أقوى نتيجة بين كل الميزات الداخل-يومية** (`frac_significant=57%`) لكن `consistent_sign=False`. تشخيص: نفس نافذة انهيار FTX تكسر RSI وVWAP معاً — تأكيد مستقلّ لنظام سوق حقيقي. ثلاث محاولات عزل (نسبي بالزخم، أصل فردي، سوق ككل عبر BTC) فشلت جميعها في عزل الظاهرة بنظافة — تبقى موثَّقة وحقيقية لكن تعريفها التشغيلي مفتوح لمحاولة أضيق زمنياً مستقبلاً. `VOL_CONC_HHI` مرفوضة بوضوح. |
| [بيتا السوق المتدحرجة (`MKT_BETA`)](market_beta.md) | ❌ مرفوضة (9 من 9) | **أول اختبار فعلي حقيقي** لمفهوم ظهر في بدايات المشروع لكن ظلّ غير مُختبَر فعلياً (خلل صامت: `market_context` معطَّلة أنتجت `IC=0.0` حرفياً لا رفضاً حقيقياً). يقيس تضخيم الأصل لحركة BTC (مختلف عن `MKT_CORR`، قوّة الترابط) — لا معلومة تنبؤية مستقلّة. |
| [الزخم المُعدَّل بالمخاطرة (`RISK_ADJ_MOM`)](risk_adjusted_momentum.md) | ❌ مرفوضة (9 من 9) | نسبة شبيهة بشارب (Barroso & Santa-Clara 2015) — جودة الاتجاه (متوسط/انحراف معياري) لا اتجاهه الخام. أفضل توليفة (53%) دون `consistent_sign=True`. |
| [المسافة عن القمّة التاريخية (`PCT_FROM_ATH`)](ath_distance.md) | ❌ مرفوضة (6 من 6) | نافذة متوسّعة (لا فركتالية قصيرة المدى)؛ مقياس سلوكي شهير في الكريبتو لم يُختبَر من قبل. `PCT_FROM_ATH`/close: أعلى `frac_significant` على close شُهِد هذه الجلسة (63%)، لكن `consistent_sign=False`. |
| [بنية الشمعة (`WICK_upper`/`WICK_lower`/`BODY_ratio`)](candle_wick_body.md) | ❌ مرفوضة (9 من 9) | أعمدة أساسية مُستخدَمة دائماً كمدخلات لكن لم تُختبَر قط كفرضية IC قائمة بذاتها — نسخة مستمرّة من أنماط الشموع الكلاسيكية (hammer/shooting star). `WICK_lower`/low أقوى توليفة (60%) لكن `consistent_sign=False`. |
| [اتساق السوق (`MKT_BREADTH`)](market_breadth.md) | ❌ مرفوضة (9 من 9) | أول ميزة مقطعية جماعية حقيقية (نسبة الأصول الصاعدة عبر كل الـ50 معاً، لا مرجع واحد أو رتبة فردية). `MKT_BREADTH_24`/close تعادل أعلى `frac_significant` على close هذه الجلسة (63%)، لكن `consistent_sign=False`. |
| [البنية الزمنية للتقلّب (`VOL_TERM`)](volatility_term_structure.md) | ❌ مرفوضة (9 من 9) | نسبة NATR قصير/طويل المدى — أقوى نتيجة high/low هذه الجلسة (`VOL_TERM_3_30`/high: 70%)، لكن النمط (high موجبة/low سالبة بثبات، close شبه صفرية) متّسق مع تكتّل التقلّب الميكانيكي (نطاق أوسع) لا اكتشافاً اتجاهياً حقيقياً. |
| 🔬 [تدقيق التطبيع وإعادة الاختبار المصحَّحة + اختبار خارج العيّنة](normalization_audit_and_corrected_retest.md) | ✅ **H003 مؤكَّدة خارج العيّنة** + ✅ `MOM_ORTH_NATR`/low مقبولة | خلل تصنيف صامت شوّه 7 ميزات اختيارية — نتائجها **مسحوبة** وأُعيد اختبارها (مطلق + سوقي-محايد). ثمّ اختبار مسجَّل مسبقاً ([التسجيل](preregistration_momentum_orth_natr_holdout.md)) على 38 نافذة لم تُلمَس (2023-07→2026-09): `NATR_14` high/low 95–100%، 38/38 على الهدفين؛ الزخم المتعامد مع التقلّب داخل كل يوم يتنبّأ بالهبوط النسبي (−0.109، 84%، 38/38). `MKT_CORR` ظلّ لـ`NATR_14`. |
| 🧰 [اختيار ميزات خطّ الأنابيب بالفرز التجريبي](pipeline_feature_selection.md) | ✅ مُطبَّق في `crypto_data_pipeline_v6` | 94 ميزة مرشّحة قيست منفردة على 50 أصلاً × 68 نافذة (38 خارج العيّنة): الافتراضي صار 18 ميزة بدل 37 — 4 مقبولة (`NATR_14`, `RANGE_rel`, `BBB_20`, `MOM_ORTH_NATR`) + القريبة المؤكَّدة خارج العيّنة (منها ميزات رُفضت سابقاً لكنها تفوّقت: `MKT_BETA_20`, `volume`, `MKT_CORR`, `MKT_BREADTH_24`, `VOL_TERM`)؛ حُذف 22. يتطلّب إعادة التدريب. |
| 🧯 [مقاومة الحفظ في `NIG-TimeNet v2` (PR #7)](anti_memorization_pr7.md) | ✅ مُطبَّق (`ANTI_MEMORIZATION` في `main`) | السبب الجذري معماري: `InstanceNorm` يمحو **مستوى** الميزة (انحدار لوجستي على ما يراه المحوّل: test AUC 0.52، على المستويات 0.59) — فيصبح الحفظ الطريق الوحيد. الإعداد القديم يحفظ في كل السيناريوهات (تسميات عشوائية حتى 0.75)، وينهار صامتاً مع تطبيع سيئ. `weight_decay` كان بلا أثر (Keras يضربه في lr). `RL_gru` (مستويات مُطبَّعة عبر العملات + GRU) على بيانات Drive (222 عملة): test 0.645/0.555 مقابل 0.609/0.538 للمرجع الخطّي، بفجوة 0.014/0.016. خمسة اتجاهات معمارية مُقاسة (FiLM، تضمين العملة، برجان، سعة أصغر، TCN/GRU) + تجارب مطوّرين موثّقة بدرجة تحقّق. |
| 🩺 [تشخيص النموذج عند الطلب](model_diagnostics.md) | ✅ مُطبَّق (اختياري، معطَّل افتراضياً) | `TrainingDiagnostics` (خسارة/تدرّج كل رأس، هيمنة وتضارب الرؤوس، تأثير الدفعات TracIn، خريطة البيانات)، `diagnose_model`/`model_health_report` (جدول حكم ✅/⚠️/🚨)، `model_layer_report` (مسبار خطّي وصائب/خاطئ لكل طبقة مع أدلة: تسميات مخلوطة ونسخة عشوائية)، وضوابط السعة (`effective_sample_size`, `feature_count_sweep`, `learning_curve`) التي تفصل «إخفاق إعداد» عن «لا إشارة». |
| 🧩 [نموذج اللوحة عبر العملات — المرحلة ١](panel_phase1.md) | 🛠️ مبنيّ ومُختبَر، التدريب الكامل على Colab | عيّنة = يوم UTC كامل بكل عملاته؛ نفس المُرمِّز + انتباه عبر عملات اليوم (A) مقابل بدونه (B)، بنفس العيّنات والتقسيم والمقاييس. مرجع النموذج الحالي: IC 0.093، ويصمد داخل خُمسيات التقلّب (0.082) رغم up_share المسطّح في قوس ±5%. |
| 🗄️ [المرحلة ٢ — بيانات Binance الكاملة في خط الأنابيب](phase2_data.md) | 🛠️ مبنيّ ومُختبَر، البناء على Colab | شموع 15m + تمويل + OI + نسب long/short/taker ← ميزات يومية بلا تسرّب (`ITD_*`، `FUND_sum_*`، `LSR_*`) بمفتاحي `USE_INTRADAY_15M`/`USE_FUTURES_METRICS` (تلقائيان حسب وجود المجلدات). |
| 📐 [تجربة التطبيع النسبي الهندسي للأسعار (`pct_change`)](pct_change_price_norm.md) | 🛠️ مبنيّ ومُختبَر، البناء والتدريب على Colab | أعمدة الأسعار في X = تغيّر % عن الشمعة السابقة (`−1`، `+5`، `−0.4`) بدل `(x − وسيط النافذة)/IQR`، قابلة للفكّ إلى الأسعار الحقيقية (`decode_price_window`). تغيير واحد عن 1h_s8 (`build_hourly_pct_dataset`)؛ الافتراضي يبقى السلوك القديم. |
| 🚫 [قواعد البحث وسجلّ الفشل](RESEARCH_RULES.md) | 📏 قاعدة عمل (مفروضة في CI) | كل فشل يُغلق بصفّ في [`failure_registry.csv`](failure_registry.csv): المكان، والسبب المُتحقَّق منه، والثوابت، وشروط إعادة الفتح. أي بطاقة تجربة يغطّيها فشل مُغلق تُرفض ما لم تسمِّ شرط إعادة فتح تحقّق (`tools/experiment_registry.py check`). |
| 🧭 [بروتوكول تطوير النماذج](MODEL_DEV_PROTOCOL.md) | 📏 قاعدة عمل | كل تجربة: بطاقة مسبقة (تغيير واحد + إشارة التشخيص المتوقَّع تحرّكها + قاعدة القبول — [القالب](templates/EXPERIMENT_CARD.md))، وتقرير صحّة النموذج طبقةً طبقة بعد التدريب، وتحديد موضع الفشل بترتيب ثابت (بيانات ← تمثيل ← تحسين ← رأس ← تعميم ← اقتصاد). |
| ⚠️ [تجميع أدلّة: عدم استقرار هدف `close` نفسه](close_target_sign_instability_synthesis.md) | 📊 تجميع/توصية | **⚠️ مُفنَّد جزئياً (راجع التدقيق)**: الميزات "المستقلّة" كانت وكلاء لنفس زخم 24 يوماً، وتوقّع تحسّن close على الهدف النسبي لم يتحقّق. النصّ الأصلي: 6+ ميزات مستقلّة تماماً تتشارك نفس النمط (أعلى `frac_significant` على close، `mean_ic` سالبة دائماً، `consistent_sign=False`) — دليل إحصائي مستقلّ يدعم تشخيص الفرع الموازي أن الهدف المطلق نفسه، لا الميزات، هو المشكلة. يحوّل قائمة المرفوضات إلى قائمة أولوية لإعادة الاختبار عند توفّر هدف نسبي. |

## الحالة الحالية والخطوة التالية

المرحلة ٣ من خطة المشروع مكتملة بأدواتها الثلاث. التوسّع إلى 50 أصلاً (بدل 5) هو ما أنتج أول قبول فعلي (H003). مسار `SURVEY_CANDIDATES_ROBUST` (598 مرشّحاً، 192 مؤشّراً فريداً) مُغلَق الآن بالكامل بعد خمس دفعات (154 مرشّحاً فريداً، ~450 محاولة) وترقية H003 النهائية للثلاثة المرشّحين الذين عبروا المعيار على عيّنة صغيرة — **لم ينجُ أيّ منهم**، فبقي H003 (`NATR_14`) الفرضية الوحيدة المقبولة من عائلة pandas_ta التقليدية. جولة موسّعة من الفرضيات الأعمق هذه الجلسة (16 عائلة إضافية: [الجولة الأولى](deeper_hypotheses_round1.md) 63 توليفة، ثلاثية الميزات الساعية [كوفمان/VWAP/تركّز الحجم](vwap_deviation_and_volume_concentration.md)، بيتا السوق، الزخم المُعدَّل بالمخاطرة، [المسافة عن القمّة التاريخية](ath_distance.md)، [بنية الشمعة](candle_wick_body.md)، [اتساق السوق](market_breadth.md)) لم تُضِف اكتشافاً جديداً أيضاً — H003 لا يزال الاكتشاف الجوهري الوحيد. **لكن أهمّ نتيجة من هذه الجولة الموسّعة ليست فرضية واحدة بل نمط متكرّر عبرها جميعاً**: راجع [تجميع الأدلّة](close_target_sign_instability_synthesis.md) — دعم إحصائي مستقلّ قويّ لتشخيص الفرع الموازي بأن الهدف المطلق نفسه (لا الميزات) هو مصدر عدم الاستقرار، مع قائمة أولوية جاهزة لإعادة الاختبار فور توفّر هدف نسبي. البنية المعمارية (`intraday_*`، `market_breadth`) جاهزة الآن لأي ميزة إضافية بلا عمل معماري جديد.

**`funding_rate`/`open_interest` مؤجَّلان بقرار صاحب المشروع** (يبني حالياً قاعدة بيانات ضخمة بدقّة 15 دقيقة لأكثر من 500 عملة، لم تكتمل بعد) — لم يعودا الأولوية القادمة رغم توصية [مراجعة الأدبيات الخارجية](external_literature_review.md). **الاتجاه الحالي**: استغلال بيانات داخل-يومية (ساعية) حقيقية أصبحت متاحة حديثاً عبر Google Drive (`timeframes_1h`، 50/50 أصلاً مطابقة) لبناء ميزات لم تكن ممكنة سابقاً — أوّلها نسبة الكفاءة لـKaufman (`|إزاحة صافية| / إجمالي مسافة المسار` محسوبة من الحركة الساعية داخل كل يوم)، عمل جارٍ حالياً. الأسئلة المفتوحة الثلاثة الموثَّقة في [صفحة H003](h003_volatility_reversal.md#أسئلة-مفتوحة-قرار-صريح-من-صاحب-المشروع-مطلوب-قبل-أي-اعتماد-عملي) لا تزال بانتظار قرار صريح. (`RVI_14`/`FISHER_14` أُضيفا ومُتحقَّق منهما بالفعل — [التفاصيل](rvi_fisher_features_added.md) — بانتظار تدريب فعلي فقط لقياس الأثر.)

بعد ذلك يبقى مؤجَّلاً بقرار صريح: (أ) توسيع عدد الأصول أكثر لكسر الارتباط، (ب) إعادة صياغة [المعمارية المشتركة](cross_asset_architecture.md) كترتيب محفظة نسبي (long-short) بدل IC مباشر، (ج) إعادة اختبار H002 بتدريب كامل على مجموعة أصول أكبر (بحاجة GPU غير متوفّرة حالياً)، (د) إضافة عمود taker buy/sell عند أي إعادة سحب مستقبلية لبيانات الشموع.

**تحديث (تدقيق التطبيع)**: عبارة "H003 لا يزال الاكتشاف الجوهري الوحيد" أعلاه تبقى صحيحة داخل العيّنة، لكن H003 أصبح الآن **مؤكَّداً خارج العيّنة** على ثلاث سنوات لم تُستخدَم، وانضمّت إليه أول ميزة مقبولة بتسجيل مسبق: `MOM_ORTH_NATR`/low على الهدف السوقي-المحايد — راجع [التدقيق](normalization_audit_and_corrected_retest.md).

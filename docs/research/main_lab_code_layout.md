# أين كود `main` ومختبر الإشارات ومسح pandas_ta (الحزم `workflow/` و`discovery/`)

كل دوال `main.ipynb` و`signal_discovery_lab.ipynb` و`pandas_ta_full_survey.ipynb` نُقلت حرفياً إلى حزمتين (نمط `data/` نفسه: وحدة لكل مجموعة
خلايا، تُنفَّذ كلها في **نطاق الدفتر نفسه** بـ`load_into(globals(), only=...)` فتبقى الأسماء المتاحة بعد `%run` كما كانت، والتعريفات المتأخرة
`late binding` والقراءة من متغيّر `CONFIG` الحيّ للخط تعمل كما كانت). الدفاتر صارت مُشغِّلات: `main.ipynb` **مُوجَّه بالإعدادات**: خلية إعدادات
واحدة (`RunSettings`، `workflow/settings.py`) ثم خلية قصيرة لكل خطوة تستدعي دالتها من `workflow/run.py` (انظر «كان / صار» أدناه).

للتعديل: غيّر ملف الوحدة (لا الدفتر)، وراجع `CLAUDE.md`. الوثائق المسجَّلة قبل النقل تذكر «القسم §» أو «الخلية N» من الدفتر القديم؛ الجدولان
أدناه يحوّلان رقم الخلية القديم (قبل النقل) إلى الوحدة. الخريطة الحالية: [`maps/workflow.md`](../../maps/workflow.md) و[`maps/discovery.md`](../../maps/discovery.md).

## `main.ipynb` ← `workflow/`

الجدول يحوّل رقم الخلية **في `main.ipynb` القديم** (قبل S7) إلى الوحدة. بعد S7 يُحمَّل كل شيء بسطر واحد `workflow.load_into(globals())` بعد الدفاتر الأربعة
(`%run`: خط الأنابيب، `model_v2`، المدرّب، chicks)؛ ترتيب التحميل بالنسبة لها محفوظ فـ`_auc` في ٧-ز ما زال يحجب `_auc` من `model_v2` كما كان.

### خلايا `main.ipynb` الآن (45 خلية؛ الأقسام كما هي)

| الخلايا | المحتوى |
|---|---|
| 0–1 | الشرح، وجدول «كان / صار» |
| 2–7 | جلب المستودع (Drive/GitHub)، ثم `%run` للدفاتر الأربعة بترتيبها، ثم تحميل الحزمة `workflow/` كلها |
| 8–9 | **الإعدادات**: `settings = RunSettings(project=…, data=…, target=…, model=…, train=…, evaluation=…, panel=…)` — كل ما تعدّله |
| 10–12 | ٣: `kit = Toolkit.from_namespace(globals())`، `load_dataset`، `apply_dataset_config`، `make_splits` |
| 13–14 | ٣-ب: `retarget` |
| 15–16 | ٤: `plan_model`، `build_model` |
| 17–19 | ٥: `make_training_config`، `make_datasets`، `train_model` |
| 20–22 | ٦: `prepare_chicks`، `run_chicks` |
| 23–42 | ٧: أمثلة استدعاء (معلَّقة) لتقارير `workflow/` + خلية اللوحة `run_panel` (٧-ح) |
| 43–44 | ٨: `run_wiring_selftest(kit)` |

### `workflow/settings.py` و`workflow/run.py` (جديدتان)

| الوحدة | ما فيها |
|---|---|
| `workflow/settings.py` | `RunSettings` وأقسامها السبعة (`ProjectSettings`، `DataSettings`، `TargetSettings`، `ModelSettings`، `TrainSettings`، `EvalSettings`، `PanelSettings`): 53 إعداداً، frozen، اسم مجهول = خطأ، قيمة غير صالحة = خطأ عند البناء، افتراضياتها = قيم الدفتر القديم حرفياً (`tests/golden_run_settings.json`)؛ و`run_dir_for` (اسم مجلد التدريب) |
| `workflow/run.py` | `Toolkit` (نقاط الدخول إلى الدفاتر الأربعة كحقول صريحة)، `DatasetInfo`/`ModelPlan`/`ChicksInputs`/`RunResult`، والخطوات: `apply_project_config`، `load_dataset`، `apply_dataset_config`، `make_splits`، `retarget`، `plan_model`، `build_model`، `make_training_config`، `make_datasets`، `train_model`، `prepare_chicks`، `run_chicks`، `run_panel`، `run_reports`، و`run_main` (كلها بالترتيب) |

### كان / صار

| كان (متغيّر في خلايا `main`) | صار |
|---|---|
| `update_config({...})` (٢) | `settings.project.config_overrides` |
| `DATA_FILENAME_BASE`، `DATA_FORMAT`، `DATA_MMAP`، `DATA_LOCAL_DIR`، `DATA_PATH` | `settings.data.filename_base` / `.format` / `.mmap` / `.local_dir` / `.path` |
| `MODEL_TFS` | `settings.data.model_tfs` (والمشتقّ: `info.model_tf`، `info.model_tfs`) |
| `TARGET_MODE`، `ENTRY_CLOSE_REG` | `settings.target.target_mode` / `.entry_close_reg` (+ `.group_freq`) |
| `ANTI_MEMORIZATION`، `CLASS_ONLY` | `settings.model.anti_memorization` / `.class_only` |
| `RUN_MAIN_TRAINING`، `run_dir`، `epochs`، `batch_size`، `train_mode`، جداول `lambda_*`، `ANTI_MEMORIZATION_TRAINER` | `settings.train.*` |
| `CALIBRATE_CONFIDENCE`، `CALIBRATION_METHOD` | `settings.evaluation.calibrate_confidence` / `.calibration_method` |
| `PANEL_MODE`، `PANEL_*`، `PANEL_PRESET`/`PANEL_PRESETS` | `settings.panel.enabled`، `settings.panel.*`، `.preset` / `PANEL_PRESETS` |
| `REG_TARGET_SCALE` | `info.reg_target_scale` |
| `PRICE_TARGETS`، `SUSPENDED_TARGETS`، `MODEL_OVERRIDES`، `SEQ_LEN`، `MODEL_SEQ_LEN`، `MODEL_N_FEATURES` | `plan.price_targets`، `.suspended_targets`، `.model_overrides`، `.seq_len`، `.model_seq_len`، `.model_n_features` |
| `model_builder()` | `model_builder` تُرجعه `build_model(plan, kit)` (دالة بلا وسائط) |
| `main_config` | `main_config = make_training_config(...)` (المتغيّر نفسه) |
| `CHICKS_TARGETS`، `test_dict`، `EVAL_TARGET_SPECS`، `CHICKS_MARKET_NEUTRAL` | `chicks.chicks_targets`، `.test_dict`، `.eval_target_specs`، `.market_neutral` |
| تقارير ٧ بلا وسائط عن النموذج/الأهداف | `model_tf=info.model_tfs`، `price_targets=plan.price_targets`، `config=main_config`، `dataset=dataset` صريحة؛ و`real_price_predictions(model, test, asset, target, model_tf)` |
| `run_wiring_selftest()` | `run_wiring_selftest(kit)` |

القاعدة: دوال `workflow/` لا تقرأ أي متغيّر إعداد من نطاق الدفتر؛ تتلقّاه وسيطاً. `tools/evaluate_trained_model.py` يبني `RunSettings` من سطر الأوامر ويستدعي `run_main`/`run_panel`/`run_reports` (لا يقرأ `main.ipynb`)،
والاختبارات (`tests/test_run_steps.py`) تستدعي الخطوات وتقارن بما سجّلته خلايا الدفتر القديم (`1b1b14f`).

| الخلية (قبل النقل) | القسم | الوحدة | ما فيها |
|---|---|---|---|
| 7 | ٣ تحميل البيانات | `workflow/dataset_io.py` | `_find_on_drive` |
| 8 | ٣ التقسيم | `workflow/splits.py` | `model_x`، `_tfs_of`، `reg_scale_of`، `target_mode_of`، `entry_close_reg_of` |
| 10 | ٣-ب تغيير الهدف | `workflow/retarget.py` | `retarget_splits`، `entry_range_to_prices`، `TARGET_MODES`، `ENTRY_CLOSE_REGS` |
| 13 | ٤ النموذج | `workflow/model_build.py` | `model_builder` |
| 15، 17 | ٥ التدريب | `workflow/training_config.py` | `build_target_configs`، `_naive_class_baseline` |
| 17 | ٥ التدريب | `workflow/batches.py` | `make_shuffled_dataset`، `make_eval_dataset`، `_y_for`، `_to_unit_label` |
| 19، 20 | ٦ chicks | `workflow/chicks_bridge.py` | `build_chicks_test_dict`، `entry_range_target_spec` (كانت `_entry_range_spec` متداخلة في الخلية 20) |
| 23 | ٧ تقارير | `workflow/reports.py` | `latest_trading_report`، `real_price_predictions`، `classification_accuracy_report` |
| 24 | ٧ دمج الأصول | `workflow/pooling.py` | `pool_test_dict`، `predict_pooled_batch_by_asset` |
| 26 | ٧-ب | `workflow/selective_eval.py` | `collect_signals`، `selective_direction_report`، `rr_trading_report` |
| 28 | ٧-ج | `workflow/candle_baseline.py` | `candle_baseline_report` |
| 30 | ٧-د | `workflow/verification.py` | `run_full_verification` |
| 32 | ٧-هـ | `workflow/permutation_control.py` | `run_label_permutation_control` |
| 34 | ٧-و | `workflow/market_neutral.py` | `market_neutral_report`، `split_asset_names` |
| 36 | ٧-ز | `workflow/generalization.py` | `normalization_audit`، `generalization_gap_report` |
| 38 | ٧-ح اللوحة | `workflow/panel_bridge.py` | `panel_names`، `panel_baseline` (كانتا `_panel_names`/`_panel_baseline` متداخلتين في الخلية) |
| 40 | ٧-ط | `workflow/diagnostics.py` | `make_training_diagnostics`، `model_health_report`، `model_layer_report` |
| 42 | ٧-ي | `workflow/capacity.py` | `effective_sample_size`، `simple_baseline`، `feature_count_sweep`، `learning_curve`، `capacity_verdict`، `capacity_report` |
| 44 | ٨ | `workflow/wiring_selftest.py` | `run_wiring_selftest` |

بقيت في الدفتر (لا تُنقل): خلية ٢ (`github_token`/`git_auth`: تعمل قبل أن يوجد المستودع على Colab) وخلية الإعدادات واستدعاءات الخطوات. بعد S7 لا تُرقَّع نصوص الخلايا
في أي مكان: `tools/evaluate_trained_model.py` يبني `RunSettings` ويستدعي الخطوات.

## `signal_discovery_lab.ipynb` و`pandas_ta_full_survey.ipynb` ← `discovery/`

| الدفتر / الخلية (قبل النقل) | الوحدة | ما فيها |
|---|---|---|
| lab 2 | `discovery/axis_loader.py` | `load_notebook_defs`، `_notebook_code` (تحميل تعريفات دفتر المحور بـ`ast`) |
| lab 6 | `discovery/evaluation.py` | `clean_reg_target`، `evaluate_candidate` |
| lab 8 | `discovery/predictors.py` | `extract_feature_*`، `make_*_predict_fn`، `make_candidate_predict_fn` |
| lab 14، 16، 22، 45 | `discovery/hypothesis_predictors.py` | مرشّحو الفرضيات: عزل الغابة، DPO، المركَّب الخطي، فركتالات الانعكاس... |
| lab 19 | `discovery/scanner.py` | `scan_candidates` |
| lab 25 | `discovery/batch_runner.py` | `run_batch_and_register`، `classify_result` |
| lab 28، 31، 34، 38، 41 | `discovery/phase3_tools.py` | Matrix Profile، SHAP، أهمية مجمّعة، عنقدة، اقتران عابر للأصول |
| lab 92 | `discovery/selftest.py` | `run_discovery_lab_selftest` |
| survey 4، 6، 14، 23، 25 | `discovery/survey.py` | `make_dummy_ohlcv`، `run_survey`، `param_variants`، `build_candidate_dicts`، `robustness_agg` (كانت `_agg` متداخلة)، `build_robust_candidate_dicts` (كانت متداخلة) |

خلايا التشغيل والنتائج في دفتر المختبر (قوائم المرشّحين، تسجيل الدفعات في السجلّ) بقيت في الدفتر. سكربتات `docs/research/scripts/` التي كانت تستخرج
تعريفات المختبر بـ`ast` من الدفتر تحمّل الآن `discovery.load_into(g, exclude=("survey",))`.

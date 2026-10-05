# أين كود `main` ومختبر الإشارات ومسح pandas_ta (الحزم `workflow/` و`discovery/`)

كل دوال `main.ipynb` و`signal_discovery_lab.ipynb` و`pandas_ta_full_survey.ipynb` نُقلت حرفياً إلى حزمتين (نمط `data/` نفسه: وحدة لكل مجموعة
خلايا، تُنفَّذ كلها في **نطاق الدفتر نفسه** بـ`load_into(globals(), only=...)` فتبقى الأسماء المتاحة بعد `%run` كما كانت، والتعريفات المتأخرة
`late binding` والقراءة من المتغيرات العامة `CONFIG` / `TARGET_MODE` / `REG_TARGET_SCALE` تعمل كما كانت). الدفاتر صارت مُشغِّلات: إعدادات،
ثم استدعاءات بالترتيب.

للتعديل: غيّر ملف الوحدة (لا الدفتر)، وراجع `CLAUDE.md`. الوثائق المسجَّلة قبل النقل تذكر «القسم §» أو «الخلية N» من الدفتر القديم؛ الجدولان
أدناه يحوّلان رقم الخلية القديم (قبل النقل) إلى الوحدة. الخريطة الحالية: [`maps/workflow.md`](../../maps/workflow.md) و[`maps/discovery.md`](../../maps/discovery.md).

## `main.ipynb` ← `workflow/`

أرقام الخلايا الحالية في `main.ipynb` أكبر بواحد من القديمة بعد الخلية 3 (أُضيفت خلية «تحميل الحزمة workflow/» بعد `%run` خط الأنابيب).
كل خلية قسم كانت تحوي تعريفات صارت تبدأ بسطر `workflow.load_into(globals(), only=("<الوحدة>",))` عند الموضع نفسه الذي كانت تُعرَّف فيه
(فلا يتغيّر ترتيب التعريفات بالنسبة للدفاتر المُشغَّلة بـ`%run`؛ مثلاً `_auc` في ٧-ز يحجب `_auc` من `model_v2` كما كان).

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

بقيت في الدفتر (لا تُنقل): خلية ٢ (`github_token`/`git_auth`: تعمل قبل أن يوجد المستودع على Colab)، وكل الإعدادات (`TARGET_MODE`،
`ENTRY_CLOSE_REG`، `ANTI_MEMORIZATION`، `main_config`، `PANEL_*`...) وخلايا التدريب والتقييم نفسها (`trainer.fit`، `run_full_analysis`...). أداة
`tools/evaluate_trained_model.py` تشغّل خلايا الدفتر بترتيبها وتُرقّع نصوصها، فتبقى هذه النصوص ثابتة (يحرسها `tests/test_workflow_packages.py`).

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

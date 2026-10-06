"""
PURPOSE:  The steps of a main run as functions of (RunSettings, explicit inputs) -> outputs: load data, adapt CONFIG, split, retarget, plan and build the model, training config and datasets, train, prepare the chicks evaluation, run it, panel model, reports; run_main chains them. main.ipynb calls them one per cell, tools/evaluate_trained_model.py and the tests call them directly.
TAGS:     run_main, load_dataset, apply_dataset_config, make_splits, retarget, plan_model, build_model, make_training_config, make_datasets, train_model, prepare_chicks, run_chicks, run_panel, run_reports, Toolkit, DatasetInfo, ModelPlan, ChicksInputs, RunResult, RunSettings steps
PITFALLS: Steps read no notebook global: settings come as the first argument and everything else (the pipeline/model/trainer/chicks entry points = Toolkit.from_namespace(globals()), the loaded data, the splits, the derived facts DatasetInfo / ModelPlan / ChicksInputs) as explicit arguments. They still call the other workflow modules and the pipeline's own CONFIG-reading helpers late-bound in the shared namespace (this module is loaded last). apply_dataset_config mutates the pipeline's CONFIG exactly as the old cell did (update_config + split_dates / holdout_start from the data). Prints are the old cells' prints. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own).
"""
import dataclasses as _dc
import gzip
import os
import pickle
import time as _time
import typing as _t
from copy import deepcopy as _deepcopy

import numpy as np

#: The reports run_reports knows, in the order the evaluation tool runs them by default.
REPORTS = ("gap", "verification", "market_neutral", "candle", "chicks", "signals")


@_dc.dataclass(frozen=True)
class Toolkit:
    """The entry points of the other packages (data pipeline, model, trainer, chicks evaluation) the steps call, as explicit fields
    instead of names read from the notebook namespace. Build it once after the runner notebooks are loaded:
    ``kit = Toolkit.from_namespace(globals())`` (a missing name raises, naming it). Tests / tools swap one field with
    ``dataclasses.replace(kit, load_data_from_drive=...)``."""
    config: dict                          # the data pipeline's live CONFIG dict
    update_config: _t.Callable
    load_data_from_drive: _t.Callable
    load_dataset_dir: _t.Callable
    split_data: _t.Callable
    build_model_fn: _t.Callable           # model package
    anti_memorization_config: dict        # model package: the architecture overrides of ModelSettings.anti_memorization
    build_config: _t.Callable             # trainer package
    deep_update: _t.Callable
    build_training_system: _t.Callable
    run_full_analysis: _t.Callable        # chicks evaluation
    default_price_targets: _t.Sequence

    #: field -> the name it has in the notebook namespace
    NAMES = {"config": "CONFIG", "update_config": "update_config", "load_data_from_drive": "load_data_from_drive",
             "load_dataset_dir": "load_dataset_dir", "split_data": "split_data", "build_model_fn": "build_model_fn",
             "anti_memorization_config": "ANTI_MEMORIZATION_CONFIG", "build_config": "build_config",
             "deep_update": "deep_update", "build_training_system": "build_training_system",
             "run_full_analysis": "run_full_analysis", "default_price_targets": "DEFAULT_PRICE_TARGETS"}

    @classmethod
    def from_namespace(cls, ns):
        missing = sorted(name for name in cls.NAMES.values() if name not in ns)
        if missing:
            raise NameError(f"Toolkit: {missing} are not defined in the namespace — run the pipeline, model_v2, trainer and chicks "
                            "runner notebooks first (main.ipynb cells 3-6)")
        return cls(**{field: ns[name] for field, name in cls.NAMES.items()})


@_dc.dataclass(frozen=True)
class DatasetInfo:
    """What the loaded dataset fixes for the rest of the run (apply_dataset_config)."""
    model_tf: str             # the base timeframe (the first of model_tfs): targets, last_candles and the grid come from it
    model_tfs: tuple          # timeframes fed to the model; one = the old single-array model, more = one encoder branch each
    reg_target_scale: float   # y_*_reg = return x this (100 in 1h_s8); old files without the key = 1.0


@_dc.dataclass(frozen=True)
class ModelPlan:
    """The model-side facts derived from the settings, CONFIG and the dataset (plan_model)."""
    price_targets: tuple      # the targets the model is trained and evaluated on (close suspended outside entry_range)
    suspended_targets: tuple
    model_overrides: dict     # the build_model_fn config
    seq_len: int              # window length of the base timeframe
    n_features: int
    model_seq_len: _t.Any     # int for one timeframe, {tf: int} for more
    model_n_features: _t.Any


@_dc.dataclass(frozen=True)
class ChicksInputs:
    """What the chicks evaluation (section 6) needs from the test split (prepare_chicks)."""
    chicks_targets: tuple
    test_dict: _t.Optional[dict]          # None = the test split's target mode is not supported by chicks (section 6 is skipped)
    market_neutral: bool                  # relative targets: the tearsheet is market neutral
    eval_target_specs: list


@_dc.dataclass(frozen=True)
class RunResult:
    """Everything run_main produced (the visible variables of the notebook)."""
    dataset: dict
    info: DatasetInfo
    train: dict
    val: dict
    test: dict
    plan: ModelPlan
    model_builder: _t.Callable
    model: _t.Any
    main_config: dict
    train_ds: _t.Any
    val_ds: _t.Any
    trainer: _t.Any
    callbacks: _t.Any
    history: _t.Any
    chicks: ChicksInputs


# ═══════════════════════════════════════════════════════════════════════════
# القسم ٢-٣: الإعدادات والبيانات
# ═══════════════════════════════════════════════════════════════════════════
def apply_project_config(settings, kit):
    """القسم ٢: إعدادات المشروع فوق افتراضيات خط الأنابيب (ProjectSettings.config_overrides)."""
    kit.update_config(_deepcopy(settings.project.config_overrides))
    print("project_name:", kit.config["project_name"])


def load_dataset(settings, kit):
    """القسم ٣: يحمّل ملف البيانات الجاهز (من Drive أو DataSettings.path) ويُرجع ``dataset``."""
    d = settings.data
    try:
        if d.path:
            raise FileNotFoundError(d.path)
        dataset = kit.load_data_from_drive(filename_base=d.filename_base, fmt=d.format, mmap=d.mmap,
                                           local_dir=d.local_dir)  # يقرأ {filename_base}_latest.dataset أو .pkl.gz
    except FileNotFoundError:
        # حساب آخر: الملف ليس في MyDrive/<project>/preprocessed_data — ابحث عنه (اختصار لمجلد مُشارَك مثلاً)
        _path = (d.path or _find_on_drive(f"{d.filename_base}_latest.dataset")
                 or _find_on_drive(f"{d.filename_base}_latest.pkl.gz"))
        if not _path or not os.path.exists(_path):
            raise FileNotFoundError(
                f"❌ لم أجد {d.filename_base}_latest.pkl.gz في هذا الحساب. من حسابك الأول: شارك مجلد "
                f"crypto_model/preprocessed_data مع هذا الحساب، ثم هنا في Drive ← «مُشارَك معي» ← نقر يمين ← «إضافة اختصار "
                f"إلى Drive» ← «ملفاتي» وأعد التشغيل؛ أو ضع DataSettings.path صراحةً؛ أو ابنِ البيانات هنا بخط الأنابيب "
                f"(crypto_data_pipeline_v6: يحتاج مجلد history_1d في MyDrive — اختصار له يكفي).")
        if os.path.isdir(_path):
            dataset = kit.load_dataset_dir(_path, mmap=d.mmap, local_dir=d.local_dir)
        else:
            with gzip.open(_path, "rb") as _f:
                dataset = pickle.load(_f)
        print(f"✅ تم التحميل من {_path}")
    return dataset


def apply_dataset_config(settings, dataset, kit):
    """القسم ٣: يختار الفريم(ات) من البيانات نفسها لا من افتراضات هنا، ويكتب في CONFIG (حيّ، كما كانت الخلية) ما تحدّده البيانات:
    الفريمات والنوافذ والأفق وstride ومقياس الأهداف وتطبيع الأسعار وحدود التقسيم/الـholdout المختوم. يُرجع DatasetInfo."""
    CONFIG = kit.config
    # وحدة أهداف y_{هدف}_reg من البيانات نفسها لا من CONFIG: عائد × reg_target_scale (100 في 1h_s8). ملف قديم بلا
    # المفتاح = عائد خام (1.0). كل تحويل إلى سعر أو عائد يقسم عليه أولاً (reg_scale_of).
    reg_target_scale = float(dataset.get("reg_target_scale", 1.0))
    model_tf = CONFIG.get("model_tf") if CONFIG.get("model_tf") in dataset["timeframes"] else dataset["base_tf"]
    # متعدّد الفريمات (1h + سياق 4h): DataSettings.model_tfs = ("1h", "4h"). فرع مُرمِّز لكل فريم في النموذج، والنماذج تُغذَّى
    # بقاموس {فريم: X_فريم}؛ الأهداف والشبكة من الفريم الأول وهو الأساسي. الافتراضي [model_tf] = فريم واحد: السلوك القديم حرفياً.
    model_tfs = list(settings.data.model_tfs or [model_tf])
    _missing_tfs = [tf for tf in model_tfs if tf not in dataset["timeframes"]]
    if _missing_tfs:
        raise ValueError(f"model_tfs يحوي فريمات غير موجودة في البيانات: {_missing_tfs} — المتوفّر {dataset['timeframes']}")
    model_tf = model_tfs[0]
    if len(model_tfs) > 1:
        if model_tf != dataset["base_tf"]:
            raise ValueError(f"الفريم الأول في model_tfs ({model_tf}) يجب أن يكون الفريم الأساسي للبيانات ({dataset['base_tf']}): "
                             "منه الأهداف وlast_candles والشبكة")
        if dataset.get("higher_tf_mode") != "closed":
            print("⚠️ البيانات لا تحمل higher_tf_mode='closed': محاذاة الفريم الأعلى بالقاعدة القديمة (موضع الصف) لا بقاعدة الشمعة "
                  "المغلقة الموثَّقة بلا نظر للمستقبل — ابنِ البيانات بـ build_hourly_4h_dataset (خط الأنابيب 20-ج)")
    kit.update_config({"tf_order": model_tfs, "base_tf": model_tf,
                       # من البيانات نفسها لا من افتراضات CONFIG (اليومي: 32/1/1، الساعة: 168/32/4)
                       "window_sizes": dict(dataset["window_sizes"]),
                       "forecast_horizon": dataset.get("forecast_horizon", CONFIG["forecast_horizon"]),
                       "stride": dataset.get("stride", CONFIG["stride"]),
                       "reg_target_scale": reg_target_scale,
                       # وحدة الأعمدة السعرية في X من البيانات نفسها: 'pct_change' = تغيّر % عن الشمعة السابقة (يُفكّ بـ
                       # decode_price_window)؛ ملف بلا المفتاح = 'window_scale' (السلوك القديم).
                       "price_norm_mode": dataset.get("price_norm_mode", "window_scale"),
                       "price_pct_clip": float(dataset.get("price_pct_clip", CONFIG.get("price_pct_clip", 100.0)))})
    # حدود التقسيم وبداية الـholdout المختوم من البيانات نفسها (إعداد 1h_s8 يحملها) — ما لم يحدّدها القسم ٢ صراحةً.
    # ملف قديم بلا المفتاحين: لا تغيير (split_dates تلقائية، لا holdout).
    for _k in ("split_dates", "holdout_start"):
        if not CONFIG.get(_k) and dataset.get(_k):
            CONFIG[_k] = _deepcopy(dataset[_k])
    if CONFIG.get("holdout_start"):
        print(f"🔒 holdout مختوم من {CONFIG['holdout_start']}: خارج train/val/test وكل ملفات الإشارات "
              f"(docs/research/hourly_1h.md). حدود التقسيم: {CONFIG.get('split_dates')}")

    print(f"الأطر المتوفّرة في البيانات: {dataset['timeframes']} — الفريم المُستخدَم فعلياً: "
          f"{model_tf if len(model_tfs) == 1 else model_tfs}")
    print(f"عدد الميزات: {len(dataset['feature_order'])} — طول النافذة: "
          f"{dataset['window_sizes'][model_tf] if len(model_tfs) == 1 else {tf: dataset['window_sizes'][tf] for tf in model_tfs}}"
          f" — الأفق: {CONFIG['forecast_horizon']} — stride: {CONFIG['stride']}"
          f" — تطبيع الأسعار: {CONFIG['price_norm_mode']}")
    return DatasetInfo(model_tf=model_tf, model_tfs=tuple(model_tfs), reg_target_scale=reg_target_scale)


def make_splits(settings, dataset, info, kit):
    """القسم ٣: التقسيم الزمني train/val/test، ويُختم كل قسم بمقياس الأهداف (المستهلكون: chicks، collect_signals، اللوحة)."""
    if settings.data.split_dates:           # تقسيم زمني صريح فوق CONFIG/البيانات (وسيط --split-dates في أداة التقييم)
        sd = settings.data.split_dates
        kit.update_config({"split_dates": {"train_end": sd["train_end"], "val_end": sd["val_end"]}})
        print(f"📅 تقسيم زمني صريح: train ≤ {sd['train_end']} | val ≤ {sd['val_end']} | test بعده (مع فجوة العزل)", flush=True)
    train, val, test = kit.split_data(dataset, config=kit.config)
    # split_data لا ينقل مفاتيح البيانات الوصفية، فيُختم كل قسم بمقياسه — المستهلكون (chicks، collect_signals، اللوحة)
    # يقرؤونه من القسم نفسه.
    for _s in (train, val, test):
        for _m in _split_members(_s):
            _m["reg_target_scale"] = info.reg_target_scale
    for _tf in list(info.model_tfs)[1:]:
        print(f"سياق {_tf}: train {train[f'X_{_tf}'].shape} | val {val[f'X_{_tf}'].shape} — نوع X: {train[f'X_{_tf}'].dtype}")
    print("train:", train[f"X_{info.model_tf}"].shape, "| val:", val[f"X_{info.model_tf}"].shape,
          "| test assets:", list(test.keys()) if isinstance(test, dict) and all(isinstance(v, dict) and "y" in v for v in test.values()) else test[f"X_{info.model_tf}"].shape)
    return train, val, test


def retarget(settings, train, val, test, info):
    """القسم ٣-ب: أهداف TargetSettings.target_mode على نفس البيانات (None = أهداف خط الأنابيب؛ بعد تجربة وضع آخر في نفس الجلسة
    تُعاد أهداف 'return'). يُرجع (train, val, test) جديدة."""
    t = settings.target
    if t.target_mode:
        train, val, test = retarget_splits(train, val, test, mode=t.target_mode, reg_target_scale=info.reg_target_scale,
                                           close_reg=t.entry_close_reg, group_freq=t.group_freq)
    elif _split_parts(train)[0][1].get("target_mode") not in (None, "return"):
        # عودة إلى None بعد تجربة وضع آخر في نفس الجلسة: نعيد أهداف خط الأنابيب ('return')
        train, val, test = retarget_splits(train, val, test, mode="return", reg_target_scale=info.reg_target_scale,
                                           close_reg=t.entry_close_reg, group_freq=t.group_freq)
    return train, val, test


# ═══════════════════════════════════════════════════════════════════════════
# القسم ٤: النموذج
# ═══════════════════════════════════════════════════════════════════════════
def plan_model(settings, dataset, info, kit):
    """القسم ٤: ما يُشتقّ من الإعدادات وCONFIG والبيانات للنموذج: الأهداف المفعّلة، وإعداد المعمارية، وأطوال النوافذ."""
    CONFIG = kit.config
    seq_len = dataset["window_sizes"][info.model_tf]
    n_features = len(dataset["feature_order"])
    # لكل فريم (model_tfs): طول نافذته وعدد ميزاته (الميزات نفسها في كل الفريمات). seq_len/n_features للفريم الأساسي وحده،
    # وmodel_seq_len/model_n_features ما يُمرَّر لـ build_model_fn وللّوحة: عددان لفريم واحد (البناء القديم حرفياً)،
    # وقاموسان {فريم: قيمة} لأكثر (فرع مُرمِّز لكل فريم).
    model_tfs = list(info.model_tfs)
    seq_lens = {tf: dataset["window_sizes"][tf] for tf in model_tfs}
    n_features_by_tf = {tf: n_features for tf in model_tfs}
    model_seq_len = seq_lens if len(model_tfs) > 1 else seq_len
    model_n_features = n_features_by_tf if len(model_tfs) > 1 else n_features
    # ⏸️ close معلّق مؤقتاً: اتجاه الإغلاق بلا معلومة قابلة للتنبؤ (test AUC ≈ 0.544 ≈ خط الأساس)، فيُدرَّب النموذج
    # ويُقيَّم على high/low فقط. خط الأنابيب ما زال يحسب y_close_* — إعادته = إفراغ هذه القائمة، بلا إعادة بناء بيانات.
    # entry_range (القسم ٣-ب): close ليس اتجاهاً بل موقع الإغلاق داخل المدى المحقَّق [0,1]، فيُفعَّل (قرار المستخدم).
    entry_range = str(settings.target.target_mode or "").startswith("entry_range")
    suspended = () if entry_range else ("close",)
    price_targets = [t for t in CONFIG["targets"] if t not in suspended]
    if entry_range:
        print("ℹ️ entry_range: close = موقع الإغلاق داخل المدى المحقَّق [0,1] لا اتجاهه — مُفعَّل (suspended_targets = ()).")
    # ⚠️ OrderedMeans (enforce_order) يفرض عائد_high ≥ عائد_close ≥ عائد_low. هذا صحيح للأسعار لا للعوائد
    # حين يُقاس كل هدف من سعر نفس نوعه (reg_target_mode='return'): على شموع مشي عشوائي واقعية يخالف الهدف الحقيقي هذا القيد
    # في ~88% من الأيام — فالنموذج كان مُجبَراً بنيوياً على توقّعات خاطئة. يبقى مفعّلاً فقط للوضع القديم.
    # أهداف القسم ٣-ب كلها عوائد لا أسعار، فالقيد يُعطَّل معها أيضاً.
    overrides = {"enforce_order": settings.target.target_mode is None
                                  and CONFIG.get("reg_target_mode", "return") != "return"}
    # 🧯 مقاومة الحفظ (PR #7): نموذج أصغر + ضجيج/إسقاط قنوات على المدخل + حارس مسار stats، ومعه في القسم ٥ weight_decay فعّال
    # وتنعيم تسميات وأوزان EMA. القيم والتجارب: docs/research/anti_memorization_pr7.md. False = المعمارية والتدريب القديمان تماماً.
    if settings.model.anti_memorization:
        overrides.update(kit.anti_memorization_config)
    # رؤوس النموذج للأهداف المفعّلة فقط (بلا رأس close المعلّق). class_only: رؤوس التصنيف وحدها (بلا NIG/الثقة) — تجربة لمعرفة هل
    # تعدّد الخسائر يشتّت النموذج؛ خلايا التقييم التي تحتاج رؤوس الانحدار (chicks، الأسعار المتوقعة، الإشارات) تُتخطّى فيه.
    overrides.update(price_targets=tuple(price_targets),
                     head_types={t: (["binary_classification"] if settings.model.class_only
                                     else ["nig_regression", "binary_classification"]) for t in price_targets})
    return ModelPlan(price_targets=tuple(price_targets), suspended_targets=tuple(suspended), model_overrides=overrides,
                     seq_len=seq_len, n_features=n_features, model_seq_len=model_seq_len, model_n_features=model_n_features)


def build_model(plan, kit, summary=True):
    """القسم ٤: ``(model_builder, model)`` — model_builder دالة بلا وسائط تبني نفس المعمارية كل مرّة (ضروري لاستئناف الحالة المحفوظة)."""
    model_builder = make_model_builder(kit.build_model_fn, plan.model_seq_len, plan.model_n_features, plan.model_overrides)
    model = model_builder()
    if summary:
        model.summary()
    return model_builder, model


# ═══════════════════════════════════════════════════════════════════════════
# القسم ٥: التدريب
# ═══════════════════════════════════════════════════════════════════════════
def make_training_config(settings, plan, info, val, kit):
    """القسم ٥: إعداد المدرّب (trainer_framework) من الإعدادات: مجلد التشغيل، الأهداف (true_key ↔ output_keys)، الخسائر، الإيقاف
    المبكر، وخطوط الأساس الساذجة لكل رأس تصنيف على val. يُرجع main_config."""
    tr = settings.train
    _am = settings.model.anti_memorization
    class_only = settings.model.class_only
    main_config = kit.build_config(kit.deep_update({
        "run": {
            # لكل وضع هدف (القسم ٣-ب) مجلده، فلا يُستأنف نموذج دُرِّب على هدف آخر؛ ومقاومة الحفظ تغيّر المعمارية
            # فلها مجلدها أيضاً (استئناف checkpoint بمعمارية مختلفة يفشل)
            "run_dir": run_dir_for(settings, info.reg_target_scale, info.model_tfs),
            "epochs": tr.epochs,
            "batch_size": tr.batch_size,
            "train_mode": tr.train_mode,
        },
        "targets": {k: v for k, v in build_target_configs(
                        list(plan.price_targets),
                        label_smoothing=tr.anti_memorization_label_smoothing if _am else 0.0).items()
                    if not (class_only and v["task_type"] != "classification")},
        "loss": {
            # Kendall تبقى لرؤوس التصنيف فقط؛ رؤوس evidential (NLL قد تكون سالبة) تُستثنى افتراضياً في
            # trainer_framework — معها كانت الخسارة الكلية تنهار نحو -∞ (راجع ملاحظة القسم 4 هناك).
            "use_uncertainty_weighting": tr.use_uncertainty_weighting,
            "schedules": {
                # قيمتان ابتدائيتان من مثال trainer_framework نفسه — عدّلهما بعد مراقبة val_*_nig_pen.
                "lambda_reg": _deepcopy(tr.lambda_reg),
                "lambda_calib": _deepcopy(tr.lambda_calib),
            },
        },
        "callbacks": {
            # val_raw_loss = مجموع خسائر المهام بأوزانها الثابتة (بلا حدود Kendall المتعلَّمة)، فلا يتحسّن
            # إلا بتحسّن حقيقي على val. val_loss يتغيّر أيضاً لأن log_var نفسها تتعلّم.
            "early_stopping": {"monitor": tr.early_stopping_monitor, "mode": tr.early_stopping_mode},
        },
    }, _deepcopy(tr.anti_memorization_trainer) if _am else {}))

    if tr.early_stopping_patience:
        main_config["callbacks"]["early_stopping"]["patience"] = tr.early_stopping_patience

    main_config.setdefault("callbacks", {})["class_baselines"] = {
        t: _naive_class_baseline(val, cfg)
        for t, cfg in main_config["targets"].items()
        if cfg["task_type"] == "classification"
    }
    print("خطوط الأساس (نسبة الفئة الأغلب في val):",
          {t: f"{b:.3f}" for t, b in main_config["callbacks"]["class_baselines"].items()})
    return main_config


def make_datasets(main_config, train, val, info):
    """القسم ٥: ``(train_ds, val_ds, sample_batch)`` — دفعات خلط كامل للتدريب، وبالترتيب لـ val."""
    tfs = list(info.model_tfs)
    train_ds = make_shuffled_dataset(model_x(train, tfs), _y_for(train, main_config), main_config["run"]["batch_size"])
    val_ds = make_eval_dataset(model_x(val, tfs), _y_for(val, main_config), main_config["run"]["batch_size"])
    sample_batch = next(iter(train_ds))
    return train_ds, val_ds, sample_batch


def train_model(settings, kit, *, model, model_builder, main_config, train_ds, val_ds, sample_batch):
    """القسم ٥: يدرّب (أو يستأنف من run_dir) ويُرجع ``(model, trainer, callbacks, history)``. TrainSettings.run_main_training=False:
    لا تدريب، ويُرجع ``(model, None, None, None)`` بالنموذج الذي بُني في القسم ٤."""
    if not settings.train.run_main_training:
        print("⏭️ run_main_training=False: لا تدريب — الأقسام ٦ و٧ إلى ٧-د تحتاج model، أما ٧-هـ فتعمل مباشرة")
        return model, None, None, None
    trainer, callbacks, initial_epoch = kit.build_training_system(model_builder, main_config, sample_batch)
    history = trainer.fit(
        train_ds, validation_data=val_ds, initial_epoch=initial_epoch,
        epochs=main_config["run"]["epochs"], callbacks=callbacks, verbose=settings.train.fit_verbose,
    )
    return trainer.model, trainer, callbacks, history


# ═══════════════════════════════════════════════════════════════════════════
# القسم ٦: التقييم (chicks)
# ═══════════════════════════════════════════════════════════════════════════
def prepare_chicks(settings, plan, info, test, kit):
    """القسم ٦: يحوّل test إلى شكل chicks ويبني مواصفات الفكّ (TargetSpec). يُرجع ChicksInputs (``test_dict is None`` = وضع هدف
    غير مدعوم في chicks: القسم ٦ يُتخطّى)."""
    CONFIG = kit.config
    # entry_range: high/low يُفكّان من last_close. close مع range_pos موقع بين سعرَيهما المتوقَّعين (TargetSpec.range_of في
    # chicks) ويحتاج high وlow معاً، فيُستبعد بدونهما. close مع abs_return مقدار بلا اتجاه واتجاهه من رأس التصنيف: يقرؤه chicks
    # نفسه من مخرَج النموذج (TargetSpec.class_key/signed_by_class) فلا شيء يُمرَّر هنا في test_dict — يبقى y_close_reg مقداراً فقط،
    # والاتجاه الحقيقي من last_candles.
    chicks_targets = [t for t in plan.price_targets
                      if not (target_mode_of(test) == "entry_range" and t == "close"
                              and entry_close_reg_of(test) == "range_pos" and not {"high", "low"} <= set(plan.price_targets))]
    _test_modes = {s.get("target_mode") for s in test.values()}
    test_dict = (None if _test_modes - set(CHICKS_TARGET_MODES)
                 else build_chicks_test_dict(test, list(info.model_tfs), chicks_targets=chicks_targets))
    market_neutral = bool(_test_modes & set(_RELATIVE_MODES))   # Tearsheet محايد للسوق مع أهداف relative
    if test_dict is None:
        print(f"⏭️ test يحمل أهداف {sorted(_test_modes, key=str)} — القسم ٦ (chicks) يدعم {CHICKS_TARGET_MODES[1:]} فقط ويُتخطّى")
    elif chicks_targets != list(plan.price_targets):
        print(f"ℹ️ entry_range: chicks يقيّم {chicks_targets} فقط — close مع range_pos يحتاج رأسَي high وlow")
    elif market_neutral:
        print("ℹ️ أهداف relative: chicks يقيس التفوّق على السوق، وTearsheet محفظة محايدة للسوق (عائد ناقص متوسط اليوم)")

    # reg_target_mode='return': كل هدف عائد نسبة لآخر سعر من *نفس نوعه* (high لآخر high، low لآخر low).
    # base_params واحدة مشتركة بين الأهداف لا تستطيع تمثيل ذلك — كانت high/low تُفكّ حول آخر close
    # فتنزاح أسعارها وتنحرف كل مقاييسها (MAE، نسبة النجاح، جداول الحركة). relative_to_entry يفكّ كل هدف
    # بـ آخر_سعره × (1 + العائد) — مطابق لـ invert_reg_predictions في خط الأنابيب.
    # reg_scale: الهدف عائد × reg_target_scale (من البيانات، مختوم على test) — chicks يقسم عليه قبل الفكّ.
    return_specs = [_dc.replace(s, relative_to_entry=True, reg_scale=reg_scale_of(test)) for s in kit.default_price_targets]
    specs = return_specs if CONFIG.get("reg_target_mode", "return") == "return" else kit.default_price_targets
    # entry_range: high/low من سعر الدخول last_close (العمود 2): high = P·(1 + raw/s)، low = P·(1 − raw/s) ⇐ reg_scale سالب؛
    # close مع range_pos = pred_low + clip(raw, 0, 1)·(pred_high − pred_low) ⇐ range_of؛ close مع abs_return = P·(1 + s·raw/scale)،
    # s = +1 إن كان p_up_close ≥ 0.5 وإلا −1 ⇐ signed_by_class: chicks يقرأ P(صعود) من مخرَج رأس التصنيف.
    if target_mode_of(test) == "entry_range":
        specs = [entry_range_target_spec(s, test) for s in kit.default_price_targets]
    specs = [s for s in specs if s.name in chicks_targets]    # بلا الأهداف المعلّقة (suspended_targets)
    return ChicksInputs(chicks_targets=tuple(chicks_targets), test_dict=test_dict, market_neutral=market_neutral,
                        eval_target_specs=specs)


def run_chicks(settings, kit, *, model, val, chicks, info):
    """القسم ٦: تقرير chicks الكامل (tearsheet + الدلالة الإحصائية) على test. يُرجع ``full_results`` أو None إن تُخطّي القسم."""
    if chicks.test_dict is None:    # القسم ٦ يُتخطّى إن كانت أهداف test غير مدعومة (انظر build_chicks_test_dict)
        print(f"⏭️ القسم ٦ متخطّى — أهداف test غير مدعومة في chicks (target_mode={settings.target.target_mode!r})")
        return None
    e = settings.evaluation
    tfs = list(info.model_tfs)
    # معايرة الثقة post-hoc (بلا إعادة تدريب): تُلائَم على val وحده وتُطبَّق على test؛ التقرير يطبع ECE قبل/بعد ويُبقي الخام في
    # 'confidence'. calibrate_confidence=False أو calibration_method="platt" لتغييرها. eval_all_rows=True: كل عملات وصفوف test.
    val_dict = None
    if e.calibrate_confidence:
        try:
            val_dict = build_chicks_test_dict({"VAL": val}, tfs, chicks_targets=chicks.chicks_targets)
        except Exception as _e:   # noqa: BLE001 — المعايرة اختيارية، لا تُسقط التقييم
            print(f"⚠️ تعذّر بناء val للمعايرة ({type(_e).__name__}: {_e}) — يُكمَل بلا معايرة")
    return kit.run_full_analysis(
        model=model,
        test_dict=chicks.test_dict,
        timeframes=tfs,
        target_specs=chicks.eval_target_specs,
        out_dir=e.out_dir,
        pnl_market_neutral=chicks.market_neutral,
        eval_all_rows=e.eval_all_rows,
        val_dict=val_dict,
        calibration_method=e.calibration_method,
    )


# ═══════════════════════════════════════════════════════════════════════════
# القسم ٧-ح: نموذج اللوحة عبر العملات
# ═══════════════════════════════════════════════════════════════════════════
def run_panel(settings, kit, *, dataset, train, val, test, info, plan, model, model_builder, main_config, apply_preset=True):
    """القسم ٧-ح: نموذج اللوحة عبر العملات (المرحلة ١، docs/research/panel_phase1.md). PanelSettings.preset يكتب فوق الحقول
    (``apply_preset=False`` لمن طبّقه بنفسه ثم عدّل فوقه). يُرجع نتائج التجربة."""
    import pandas as pd
    from cross_asset.data import group_ns_for
    from cross_asset.experiment import run_panel_experiment, train_cfg_from_main

    p = settings.panel.applied() if apply_preset else settings.panel
    tfs = list(info.model_tfs)
    _panel_train_cfg = {**train_cfg_from_main(main_config), "batch_samples": p.batch_samples,
                        "max_days": p.max_days, "monitor": p.monitor, **p.overrides.get("train", {})}
    if p.epochs:
        _panel_train_cfg["epochs"] = p.epochs
    if p.patience:
        _panel_train_cfg["patience"] = p.patience
    for _k, _v in (("min_coins", p.min_coins), ("max_coins", p.max_coins)):
        if _v:
            _panel_train_cfg[_k] = _v
    _panel_day_ns = (group_ns_for(info.model_tf, kit.config.get("stride", 1)) if p.group == "auto"
                     else pd.Timedelta(p.group).value if p.group else None)
    print(f"🧩 المجموعة المقطعية: {pd.Timedelta(_panel_day_ns or 86_400 * 10**9)} | أهداف: {list(plan.price_targets)} | "
          f"حقب {_panel_train_cfg.get('epochs')} صبر {_panel_train_cfg.get('patience')} | مراقبة {p.monitor}", flush=True)

    _panel_root = p.run_root or (main_config["run"]["run_dir"] + "_panel")
    baseline = ("model" if settings.train.run_main_training else None) if p.baseline == "auto" else p.baseline
    panel_results = run_panel_experiment(
        train, val, test, tfs, model_builder, plan.model_seq_len, plan.model_n_features,
        run_root=_panel_root,
        variants=list(p.variants), seeds=list(p.seeds), model_cfg=p.overrides.get("model"),
        train_cfg=_panel_train_cfg, train_assets=panel_names(dataset, "train"), val_assets=panel_names(dataset, "val"),
        subset=p.subset, day_ns=_panel_day_ns, targets=tuple(plan.price_targets), target_mode=settings.target.target_mode,
        allow_mixed_timestamps=p.allow_mixed_timestamps,
        k_eval=p.k_eval, k_draws=p.k_draws, baseline=panel_baseline(baseline, model, model_builder, dataset, val, test, tfs),
        first_touch=pd.read_pickle(p.first_touch) if p.first_touch else None)
    # أمر الضغط للتقييم المحلي بالمسار الفعلي لهذا التشغيل (لا يُنسخ من التوثيق: اسم المجلد يتبع target_mode والمقياس)
    print(f'📦 للتقييم المحلي:\n!cd {_panel_root} && zip -r /content/panel_{p.preset or "_".join(tfs)}.zip . -x "*/last*"')
    return panel_results


# ═══════════════════════════════════════════════════════════════════════════
# تقارير القسم ٧ على نموذج مدرَّب (أداة التقييم)
# ═══════════════════════════════════════════════════════════════════════════
def run_reports(settings, which, kit, *, model, train, val, test, dataset, info, plan, chicks):
    """يشغّل تقارير القسم ٧ (REPORTS: gap، verification، market_neutral، candle، signals، chicks) المطلوبة في ``which`` بالترتيب
    الثابت أدناه، كلٌّ منها في try/except (تقرير فاشل لا يوقف الباقي). المسارات نسبة لمجلد العمل الحالي. يُرجع {اسم: ناتج}."""
    unknown = sorted(set(which) - set(REPORTS))
    if unknown:
        raise ValueError(f"تقارير غير معروفة: {unknown} — المتاح: {REPORTS}")
    e = settings.evaluation
    tfs = list(info.model_tfs)
    tf_ = info.model_tf if len(tfs) == 1 else tfs         # فريم واحد: الاسم كما كان؛ أكثر: قائمة الفريمات لمُدخل النموذج
    pt = list(plan.price_targets)
    universe = e.mn_universe if e.mn_universe in (None, "categories") else list(e.mn_universe)

    def _named(df, split):
        """val مدمج بلا أسماء عملات (asset="all"): تُستعاد من dataset بنفس أقنعة split_data، كما في market_neutral_report."""
        if (df["asset"] == "all").all():
            names = split_asset_names(dataset, split)
            if names is not None and len(names) == len(df):
                df = df.assign(asset=np.asarray(names))
        return df

    runs = [
        ("gap", "٧-ز فجوة التعميم",
         lambda: generalization_gap_report(model, train, val, test, model_tf=tf_, price_targets=pt)),
        ("verification", "٧-د التحقق المتكامل",
         lambda: run_full_verification(model, train, val, test, model_tf=tf_, out_dir="verification", price_targets=pt)),
        ("market_neutral", "٧-و المحفظة المحايدة للسوق",
         lambda: market_neutral_report(model, train, val, test, model_tf=tf_, quantiles=tuple(e.mn_quantiles),
                                       universe=universe, dataset=dataset)),
        ("candle", "٧-ج مقابل شكل الشمعة",
         lambda: candle_baseline_report(model, train, val, test, model_tf=tf_)),
        ("signals", "تصدير الإشارات (val/test) للتحليل خارج الدفتر",
         lambda: [_named(collect_signals(model, sp, tf_, price_targets=pt), name).assign(split=name)
                  .to_csv(f"signals_{name}.csv.gz", index=False) for name, sp in (("val", val), ("test", test))]),
        ("chicks", "٦ chicks (tearsheet + الدلالة الإحصائية)",
         lambda: run_chicks(settings, kit, model=model, val=val, chicks=chicks, info=info)
         if chicks.test_dict is not None else print("⏭️ test_dict غير مدعوم لهذا الوضع")),
    ]
    out = {}
    for key, title, fn in runs:
        if key not in which:
            continue
        print("\n" + "#" * 100 + f"\n# {title}\n" + "#" * 100, flush=True)
        t = _time.time()
        try:
            out[key] = fn()
        except Exception as ex:  # noqa: BLE001 — تقرير فاشل لا يوقف الباقي
            import traceback
            traceback.print_exc()
            print(f"❌ {title}: {ex}")
        print(f"   ({_time.time() - t:.0f}s)", flush=True)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# كل الخطوات معاً
# ═══════════════════════════════════════════════════════════════════════════
def run_main(settings, kit, dataset=None, summary=True):
    """الخطوات بالترتيب نفسه الذي تعمل به خلايا main.ipynb: إعدادات المشروع ← تحميل البيانات (أو ``dataset`` الجاهز) ← تقسيم ←
    أهداف ← نموذج ← إعداد المدرّب والدفعات ← تدريب ← تجهيز chicks. يُرجع RunResult."""
    apply_project_config(settings, kit)
    if dataset is None:
        dataset = load_dataset(settings, kit)
    info = apply_dataset_config(settings, dataset, kit)
    train, val, test = make_splits(settings, dataset, info, kit)
    train, val, test = retarget(settings, train, val, test, info)
    plan = plan_model(settings, dataset, info, kit)
    model_builder, model = build_model(plan, kit, summary=summary)
    main_config = make_training_config(settings, plan, info, val, kit)
    train_ds, val_ds, sample_batch = make_datasets(main_config, train, val, info)
    model, trainer, callbacks, history = train_model(settings, kit, model=model, model_builder=model_builder, main_config=main_config,
                                                     train_ds=train_ds, val_ds=val_ds, sample_batch=sample_batch)
    chicks = prepare_chicks(settings, plan, info, test, kit)
    return RunResult(dataset=dataset, info=info, train=train, val=val, test=test, plan=plan, model_builder=model_builder, model=model,
                     main_config=main_config, train_ds=train_ds, val_ds=val_ds, trainer=trainer, callbacks=callbacks, history=history,
                     chicks=chicks)

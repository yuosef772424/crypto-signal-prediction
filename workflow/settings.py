"""
PURPOSE:  RunSettings: every setting of a main run (dataset, target mode, model options, training, evaluation, panel model) in one strict, frozen object whose defaults are the values main.ipynb always held; main.ipynb, tools/evaluate_trained_model.py and the tests build it and hand it to the workflow/run.py steps.
TAGS:     RunSettings, settings, ProjectSettings, DataSettings, TargetSettings, ModelSettings, TrainSettings, EvalSettings, PanelSettings, TARGET_MODE, ENTRY_CLOSE_REG, ANTI_MEMORIZATION, CLASS_ONLY, RUN_MAIN_TRAINING, PANEL_PRESETS, run_dir_for, strict config, frozen dataclass
PITFALLS: Defaults must stay equal to the notebook values the golden test records (tests/golden_run_settings.json): change one only as a new experiment (its checkpoints live in a folder named after the settings, see run_dir_for). Unknown field names raise (constructor: TypeError; RunSettings.from_dict / .updated and every section's .updated: ValueError listing the known names), values are validated at construction (target mode, data format, positive epochs...). A panel preset overrides the panel fields only through PanelSettings.applied(), exactly as the old PANEL_PRESET cell line did after the other PANEL_ lines. Dict-valued fields are copied on construction; treat every settings object as read-only. Executed into the notebook's shared namespace by workflow/_loader.py (never imported on its own); only core/ names are used.
"""
import dataclasses as _dc
import typing as _t
from copy import deepcopy as _deepcopy

from core.schema import ENTRY_CLOSE_REGS as _ENTRY_CLOSE_REGS, NO_RELATIVE_BASES as _NO_RELATIVE_BASES, TARGET_MODES as _TARGET_MODES

#: Formats of the saved dataset file accepted by the pipeline's loader (data/storage.py).
DATA_FORMATS = ("auto", "pkl.gz", "npy_dir")
#: Post-hoc confidence calibration methods of the chicks report.
CALIBRATION_METHODS = ("isotonic", "platt")
#: The panel model's ready-made setups; applied by PanelSettings.applied() (they override the fields they name).
PANEL_PRESETS = {
    # أفضل حقبة على 1h (stride 32، 25 حقبة) كانت الأخيرة أو قبلها: التدريب لم يتشبّع. هنا الحقبة أطول 4× (stride 8)،
    # و50 = نهاية الدورة الثالثة لجدول cosine_restarts في main (تسخين 3 + 10 + 15 + 22)، فتنتهي عند أدنى معدّل تعلّم
    # (60 الافتراضية تقف وسط الدورة الرابعة بمعدّل مرتفع). صبر 25 > طول الدورة الثالثة (22): إعادة التشغيل عند الحقبة 28
    # ترفع خسارة val مؤقتاً، وصبر main (15) كان سيوقف التدريب عند ~42 قبل أن تبلغ الدورة قاعها.
    "1h_s8": dict(variants=("A_ic", "B_ic", "A_ic_k"), seeds=(0, 1), epochs=50, patience=25, group="auto",
                  k_eval=(5, 10, 20, None), k_draws=3, baseline=None),
}


def _settings_check(ok, message):
    if not ok:
        raise ValueError(message)


def _settings_is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _target_mode_problem(mode):
    """Why ``mode`` is not a valid target mode, or None. Same grammar as retarget._parse_mode (tests/test_run_settings.py pins the
    agreement): a base of core.schema.TARGET_MODES, optionally '+relative' (not on NO_RELATIVE_BASES), or the alias 'relative'."""
    base, _, suffix = mode.partition("+")
    if base == "relative" and not suffix:
        return None
    if base not in _TARGET_MODES or suffix not in ("", "relative"):
        return f"unknown mode {mode!r}; known: {_TARGET_MODES}, each optionally + '+relative', or 'relative'"
    if suffix and base in _NO_RELATIVE_BASES:
        return f"{mode!r} is not supported: the cross-sectional median is not applied to {base!r}; use {base!r} alone"
    return None


class _SettingsSection:
    """Base of the settings sections: strict construction from a dict and strict partial update (unknown name = error)."""

    @classmethod
    def field_names(cls):
        return tuple(f.name for f in _dc.fields(cls))

    @classmethod
    def from_dict(cls, d):
        _settings_check(isinstance(d, dict), f"{cls.__name__}.from_dict: a dict is required, got {type(d).__name__}")
        unknown = sorted(set(d) - set(cls.field_names()))
        _settings_check(not unknown, f"{cls.__name__}: unknown setting(s) {unknown}; known: {list(cls.field_names())}")
        return cls(**d)

    def updated(self, **changes):
        """A copy with ``changes`` applied (strict names, re-validated)."""
        unknown = sorted(set(changes) - set(self.field_names()))
        _settings_check(not unknown, f"{type(self).__name__}: unknown setting(s) {unknown}; known: {list(self.field_names())}")
        return _dc.replace(self, **changes)

    def to_dict(self):
        return _deepcopy(_dc.asdict(self))


def _settings_set(obj, **values):
    """Normalise fields of a frozen section inside __post_init__."""
    for k, v in values.items():
        object.__setattr__(obj, k, v)


@_dc.dataclass(frozen=True)
class ProjectSettings(_SettingsSection):
    """Overrides of the data pipeline's CONFIG applied before anything is loaded (``update_config`` — unknown keys raise there)."""
    config_overrides: dict = _dc.field(default_factory=lambda: {"project_name": "crypto_model"})

    def __post_init__(self):
        _settings_check(isinstance(self.config_overrides, dict), "ProjectSettings.config_overrides must be a dict")
        _settings_set(self, config_overrides=_deepcopy(self.config_overrides))


@_dc.dataclass(frozen=True)
class DataSettings(_SettingsSection):
    """Which dataset file to load and how (section 3)."""
    #: "preprocessing_output" (daily), HOURLY_DATA_NAME, HOURLY_W32_S8_NAME, HOURLY_4H_NAME, HOURLY_PCT_NAME...
    filename_base: str = "preprocessing_output"
    #: "auto" = the newest of {filename_base}_latest.dataset (a .npy folder, memmap) / _latest.pkl.gz; "pkl.gz" / "npy_dir" force one.
    format: str = "auto"
    #: True = open a .dataset folder with mmap_mode='r' (X stays on disk, batches are read lazily); False = load it all into RAM.
    mmap: bool = True
    #: None = copy a Drive folder to /content/pipeline_data first (memmap over FUSE is slow); False = read it in place; or a path.
    local_dir: _t.Union[str, bool, None] = None
    #: Explicit path of the dataset file or folder (e.g. a shared folder of another account); None = search automatically.
    path: _t.Optional[str] = None
    #: Timeframes fed to the model, the first being the base one (e.g. ("1h", "4h")); None = the data's own model timeframe only.
    model_tfs: _t.Optional[tuple] = None
    #: Explicit time split {"train_end": "YYYY-MM-DD", "val_end": "YYYY-MM-DD"} instead of the sample shares (None = CONFIG / dataset).
    split_dates: _t.Optional[dict] = None

    def __post_init__(self):
        _settings_check(isinstance(self.filename_base, str) and self.filename_base, "DataSettings.filename_base must be a non-empty str")
        _settings_check(self.format in DATA_FORMATS, f"DataSettings.format {self.format!r} unknown; known: {DATA_FORMATS}")
        _settings_check(isinstance(self.mmap, bool), "DataSettings.mmap must be a bool")
        _settings_check(self.local_dir is None or isinstance(self.local_dir, (str, bool)), "DataSettings.local_dir: None, False or a path")
        _settings_check(self.path is None or isinstance(self.path, str), "DataSettings.path must be a str or None")
        if self.model_tfs is not None:
            _settings_check(isinstance(self.model_tfs, (list, tuple)) and len(self.model_tfs) > 0
                   and all(isinstance(t, str) for t in self.model_tfs),
                   "DataSettings.model_tfs must be a non-empty list/tuple of timeframe names")
            _settings_set(self, model_tfs=tuple(self.model_tfs))
        if self.split_dates is not None:
            _settings_check(isinstance(self.split_dates, dict) and set(self.split_dates) == {"train_end", "val_end"},
                   "DataSettings.split_dates must be {'train_end': ..., 'val_end': ...}")
            _settings_set(self, split_dates=dict(self.split_dates))


@_dc.dataclass(frozen=True)
class TargetSettings(_SettingsSection):
    """Target mode on the same data (section 3-b); None = the pipeline's own targets."""
    #: "return" | "return_close" | "scaled" | "magnitude" | "volnorm" | "entry_range" | "relative" | "<mode>+relative"
    target_mode: _t.Optional[str] = None
    #: The close regression of entry_range: "abs_return" | "range_pos".
    entry_close_reg: str = "abs_return"
    #: Width of the cross-section group of the +relative modes (e.g. "1D", "4h", "32h"); None = the exact timestamp.
    group_freq: _t.Optional[str] = None

    def __post_init__(self):
        if self.target_mode is not None:
            _settings_check(isinstance(self.target_mode, str) and self.target_mode, "TargetSettings.target_mode must be a str or None")
            problem = _target_mode_problem(self.target_mode)           # unknown / refused modes raise here, not after loading data
            _settings_check(problem is None, f"TargetSettings.target_mode: {problem}")
        _settings_check(self.entry_close_reg in _ENTRY_CLOSE_REGS,
               f"TargetSettings.entry_close_reg {self.entry_close_reg!r} unknown; known: {_ENTRY_CLOSE_REGS}")
        _settings_check(self.group_freq is None or isinstance(self.group_freq, str), "TargetSettings.group_freq must be a str or None")


@_dc.dataclass(frozen=True)
class ModelSettings(_SettingsSection):
    """Architecture-level options (section 4); they change the architecture, hence the training folder name."""
    #: True = the smaller model + input noise/channel drop + the matching trainer settings (PR #7); False = the old architecture.
    anti_memorization: bool = True
    #: True = classification heads only (no NIG / confidence heads).
    class_only: bool = False

    def __post_init__(self):
        _settings_check(isinstance(self.anti_memorization, bool), "ModelSettings.anti_memorization must be a bool")
        _settings_check(isinstance(self.class_only, bool), "ModelSettings.class_only must be a bool")


@_dc.dataclass(frozen=True)
class TrainSettings(_SettingsSection):
    """Training of the main model (section 5)."""
    #: False = build everything but do not train (the permutation control trains its own models).
    run_main_training: bool = True
    #: Base of the training folder; run_dir_for appends the target mode / scale / architecture so a saved model is never resumed
    #: under another setup.
    run_dir: str = "/content/drive/MyDrive/training_runs/crypto_model_v1"
    epochs: int = 60
    batch_size: int = 64
    train_mode: str = "auto"
    #: None = the trainer's default patience.
    early_stopping_patience: _t.Optional[int] = None
    #: Keras fit verbosity (1 = progress bar, 2 = one line per epoch, 0 = silent).
    fit_verbose: int = 1
    #: Kendall weighting of the classification heads (evidential heads are excluded by the trainer).
    use_uncertainty_weighting: bool = True
    lambda_reg: dict = _dc.field(default_factory=lambda: {"start": 0.0, "end": 0.05, "warmup_epochs": 5, "schedule": "linear"})
    lambda_calib: dict = _dc.field(default_factory=lambda: {"start": 0.0, "end": 0.1, "warmup_epochs": 5, "schedule": "cosine"})
    #: Early stopping / best-model monitor: the raw loss (task losses with fixed weights), not the Kendall-weighted one.
    early_stopping_monitor: str = "val_raw_loss"
    early_stopping_mode: str = "min"
    #: Merged over the trainer settings when ModelSettings.anti_memorization is on (PR #7, docs/research/anti_memorization_pr7.md §10-c).
    anti_memorization_trainer: dict = _dc.field(default_factory=lambda: {
        "optimizer": {"lr_initial": 3e-4, "weight_decay": 0.05, "ema_warmup": True, "ema_window_epochs": 1.0},
        "callbacks": {"early_stopping": {"weights_snapshot": "ema_weights"}},
    })
    #: Label smoothing of the classification heads when anti_memorization is on.
    anti_memorization_label_smoothing: float = 0.1

    def __post_init__(self):
        _settings_check(isinstance(self.run_main_training, bool), "TrainSettings.run_main_training must be a bool")
        _settings_check(isinstance(self.run_dir, str) and self.run_dir, "TrainSettings.run_dir must be a non-empty str")
        _settings_check(_settings_is_int(self.epochs) and self.epochs > 0, "TrainSettings.epochs must be a positive int")
        _settings_check(_settings_is_int(self.batch_size) and self.batch_size > 0, "TrainSettings.batch_size must be a positive int")
        _settings_check(isinstance(self.train_mode, str), "TrainSettings.train_mode must be a str")
        _settings_check(self.early_stopping_patience is None or (_settings_is_int(self.early_stopping_patience) and self.early_stopping_patience > 0),
               "TrainSettings.early_stopping_patience must be a positive int or None")
        _settings_check(_settings_is_int(self.fit_verbose) and self.fit_verbose in (0, 1, 2), "TrainSettings.fit_verbose must be 0, 1 or 2")
        _settings_check(isinstance(self.use_uncertainty_weighting, bool), "TrainSettings.use_uncertainty_weighting must be a bool")
        for name in ("lambda_reg", "lambda_calib", "anti_memorization_trainer"):
            _settings_check(isinstance(getattr(self, name), dict), f"TrainSettings.{name} must be a dict")
            _settings_set(self, **{name: _deepcopy(getattr(self, name))})
        _settings_check(self.early_stopping_mode in ("min", "max"), "TrainSettings.early_stopping_mode must be 'min' or 'max'")
        _settings_check(isinstance(self.early_stopping_monitor, str), "TrainSettings.early_stopping_monitor must be a str")
        _settings_check(isinstance(self.anti_memorization_label_smoothing, (int, float))
               and 0.0 <= self.anti_memorization_label_smoothing < 1.0,
               "TrainSettings.anti_memorization_label_smoothing must be in [0, 1)")


@_dc.dataclass(frozen=True)
class EvalSettings(_SettingsSection):
    """The chicks evaluation (section 6) and the report runner (workflow.run.run_reports)."""
    #: Post-hoc confidence calibration (fit on val only, applied to test): the report prints ECE before/after.
    calibrate_confidence: bool = True
    calibration_method: str = "isotonic"
    #: True = every asset and row of test enters the metrics.
    eval_all_rows: bool = True
    out_dir: str = "analysis_outputs"
    #: Market-neutral portfolio report: the legs' quantiles (0.5 = the whole universe for rank weights) and the universe
    #: (None = all coins | "categories" | a tuple of symbols).
    mn_quantiles: tuple = (0.05, 0.1, 0.2, 0.3, 0.5)
    mn_universe: _t.Union[str, tuple, None] = None

    def __post_init__(self):
        _settings_check(isinstance(self.calibrate_confidence, bool), "EvalSettings.calibrate_confidence must be a bool")
        _settings_check(self.calibration_method in CALIBRATION_METHODS,
               f"EvalSettings.calibration_method {self.calibration_method!r} unknown; known: {CALIBRATION_METHODS}")
        _settings_check(isinstance(self.eval_all_rows, bool), "EvalSettings.eval_all_rows must be a bool")
        _settings_check(isinstance(self.out_dir, str) and self.out_dir, "EvalSettings.out_dir must be a non-empty str")
        _settings_check(isinstance(self.mn_quantiles, (list, tuple)) and len(self.mn_quantiles) > 0
               and all(isinstance(q, (int, float)) and 0 < q <= 0.5 for q in self.mn_quantiles),
               "EvalSettings.mn_quantiles must be a non-empty sequence of numbers in (0, 0.5]")
        _settings_set(self, mn_quantiles=tuple(float(q) for q in self.mn_quantiles))
        _settings_check(self.mn_universe is None or isinstance(self.mn_universe, (str, list, tuple)),
               "EvalSettings.mn_universe must be None, 'categories' or a sequence of symbols")
        if isinstance(self.mn_universe, list):
            _settings_set(self, mn_universe=tuple(self.mn_universe))


@_dc.dataclass(frozen=True)
class PanelSettings(_SettingsSection):
    """The cross-asset panel model (section 7-h, phase 1). ``enabled=False`` does nothing."""
    enabled: bool = False
    #: "A", "A_ic", "B", "B_ic", "A_ic_k" (A = attention, B = without, _ic = IC bound, _k = random coin count per group).
    variants: tuple = ("A_ic", "B_ic")
    seeds: tuple = (0,)
    #: None = the epochs of the current model's training settings (with the same early-stopping patience).
    epochs: _t.Optional[int] = None
    #: (coin, day) rows per batch, filled with whole days only; GPU out of memory: 512.
    batch_samples: int = 1024
    max_days: int = 16
    #: Comparison baseline: "auto" = "model" when the main model is trained here, else None | "model" | a best.weights.h5 path |
    #: a folder with signals_{val,test}.csv.gz | None. It must share split_dates and the target mode.
    baseline: _t.Optional[str] = "auto"
    #: bracket_first_5.pkl to settle double touches in the bracket (optional).
    first_touch: _t.Optional[str] = None
    #: Quick test only: {"last_days": 60, "coins": 40}.
    subset: _t.Optional[dict] = None
    #: None = the training folder + "_panel".
    run_root: _t.Optional[str] = None
    #: Advanced: {"train": {"lambda_ic": 1.0, ...}, "model": {"n_cross_layers": 2, ...}}.
    overrides: dict = _dc.field(default_factory=dict)
    #: None = the early-stopping patience of the main training settings.
    patience: _t.Optional[int] = None
    #: val metric for early stopping and best-epoch selection: val_loss | val_ic_asym | val_auc_low ...
    monitor: str = "val_loss"
    #: Cross-section group: "auto" = stride x the timeframe's duration | None = UTC day | "32h"... (a group mixing timestamps is refused).
    group: _t.Optional[str] = "auto"
    #: True = skip that refusal (to measure the leak itself only; the results are not valid for judging).
    allow_mixed_timestamps: bool = False
    min_coins: _t.Optional[int] = None
    max_coins: _t.Optional[int] = None
    #: After training: test metrics when the model sees only k coins, e.g. (5, 10, 20, None).
    k_eval: _t.Optional[tuple] = None
    k_draws: int = 3
    #: A ready-made setup overriding the fields above when applied (PANEL_PRESETS): None | "1h_s8".
    preset: _t.Optional[str] = None

    def __post_init__(self):
        _settings_check(isinstance(self.enabled, bool), "PanelSettings.enabled must be a bool")
        _settings_check(isinstance(self.variants, (list, tuple)) and len(self.variants) > 0 and all(isinstance(v, str) for v in self.variants),
               "PanelSettings.variants must be a non-empty sequence of variant names")
        _settings_set(self, variants=tuple(self.variants))
        _settings_check(isinstance(self.seeds, (list, tuple)) and len(self.seeds) > 0 and all(_settings_is_int(s) for s in self.seeds),
               "PanelSettings.seeds must be a non-empty sequence of ints")
        _settings_set(self, seeds=tuple(self.seeds))
        _settings_check(self.epochs is None or (_settings_is_int(self.epochs) and self.epochs > 0), "PanelSettings.epochs must be a positive int or None")
        _settings_check(_settings_is_int(self.batch_samples) and self.batch_samples > 0, "PanelSettings.batch_samples must be a positive int")
        _settings_check(_settings_is_int(self.max_days) and self.max_days > 0, "PanelSettings.max_days must be a positive int")
        _settings_check(self.baseline is None or isinstance(self.baseline, str), "PanelSettings.baseline must be a str or None")
        _settings_check(self.subset is None or (isinstance(self.subset, dict) and set(self.subset) == {"last_days", "coins"}),
               "PanelSettings.subset must be {'last_days': ..., 'coins': ...} or None")
        _settings_check(isinstance(self.overrides, dict) and set(self.overrides) <= {"train", "model"},
               "PanelSettings.overrides must be a dict with only 'train' / 'model' sections")
        _settings_set(self, overrides=_deepcopy(self.overrides), subset=None if self.subset is None else dict(self.subset))
        _settings_check(self.patience is None or (_settings_is_int(self.patience) and self.patience > 0), "PanelSettings.patience must be a positive int or None")
        _settings_check(isinstance(self.monitor, str) and self.monitor, "PanelSettings.monitor must be a non-empty str")
        _settings_check(self.group is None or isinstance(self.group, str), "PanelSettings.group must be 'auto', None or a duration string")
        _settings_check(isinstance(self.allow_mixed_timestamps, bool), "PanelSettings.allow_mixed_timestamps must be a bool")
        for name in ("min_coins", "max_coins"):
            v = getattr(self, name)
            _settings_check(v is None or (_settings_is_int(v) and v > 0), f"PanelSettings.{name} must be a positive int or None")
        if self.k_eval is not None:
            _settings_check(isinstance(self.k_eval, (list, tuple)) and all(k is None or _settings_is_int(k) for k in self.k_eval),
                   "PanelSettings.k_eval must be a sequence of ints / None")
            _settings_set(self, k_eval=tuple(self.k_eval))
        _settings_check(_settings_is_int(self.k_draws) and self.k_draws > 0, "PanelSettings.k_draws must be a positive int")
        _settings_check(self.preset is None or self.preset in PANEL_PRESETS,
               f"PanelSettings.preset {self.preset!r} unknown; known: {sorted(PANEL_PRESETS)} or None")

    def applied(self):
        """The settings with the preset's values written over the fields it names (the old ``globals().update(PANEL_PRESETS[...])``);
        ``preset`` is kept. Without a preset: this object."""
        return self if self.preset is None else _dc.replace(self, **_deepcopy(PANEL_PRESETS[self.preset]))


@_dc.dataclass(frozen=True)
class RunSettings:
    """All settings of a run, one section per concern. ``RunSettings()`` is what main.ipynb ran with before this object existed."""
    project: ProjectSettings = _dc.field(default_factory=ProjectSettings)
    data: DataSettings = _dc.field(default_factory=DataSettings)
    target: TargetSettings = _dc.field(default_factory=TargetSettings)
    model: ModelSettings = _dc.field(default_factory=ModelSettings)
    train: TrainSettings = _dc.field(default_factory=TrainSettings)
    evaluation: EvalSettings = _dc.field(default_factory=EvalSettings)
    panel: PanelSettings = _dc.field(default_factory=PanelSettings)

    SECTIONS = {"project": ProjectSettings, "data": DataSettings, "target": TargetSettings, "model": ModelSettings,
                "train": TrainSettings, "evaluation": EvalSettings, "panel": PanelSettings}

    def __post_init__(self):
        for name, cls in self.SECTIONS.items():
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"RunSettings.{name} must be a {cls.__name__}, got {type(getattr(self, name)).__name__}")

    @classmethod
    def from_dict(cls, d):
        """Strict: unknown section or setting names raise ValueError; missing ones keep their defaults."""
        _settings_check(isinstance(d, dict), f"RunSettings.from_dict: a dict is required, got {type(d).__name__}")
        unknown = sorted(set(d) - set(cls.SECTIONS))
        _settings_check(not unknown, f"RunSettings: unknown section(s) {unknown}; known: {list(cls.SECTIONS)}")
        return cls(**{name: cls.SECTIONS[name].from_dict(v) for name, v in d.items()})

    def updated(self, changes):
        """A copy with ``{"section": {"setting": value}}`` applied (strict names); e.g.
        ``settings.updated({"train": {"epochs": 5}, "target": {"target_mode": "relative"}})``."""
        _settings_check(isinstance(changes, dict), "RunSettings.updated: a {section: {setting: value}} dict is required")
        unknown = sorted(set(changes) - set(self.SECTIONS))
        _settings_check(not unknown, f"RunSettings: unknown section(s) {unknown}; known: {list(self.SECTIONS)}")
        return _dc.replace(self, **{name: getattr(self, name).updated(**vals) for name, vals in changes.items()})

    def to_dict(self):
        return {name: getattr(self, name).to_dict() for name in self.SECTIONS}


def run_dir_for(settings, reg_target_scale, model_tfs):
    """The training folder of a run: ``train.run_dir`` + a suffix per setup choice, so a saved model is never resumed under another
    target / architecture / scale. ``reg_target_scale`` and ``model_tfs`` come from the loaded data (workflow.run.DatasetInfo)."""
    tm = settings.target.target_mode
    return (settings.train.run_dir
            + (f"_{tm}" if tm else "")
            # تعريف close في entry_range جزء من الهدف: تعريف آخر = مجلد آخر
            + (f"_{settings.target.entry_close_reg}" if tm == "entry_range" else "")
            # هدف بمقياس آخر (reg_target_scale من البيانات) = مجلد آخر: لا يُستأنف نموذج دُرِّب على عوائد خام
            + (f"_s{reg_target_scale:g}" if reg_target_scale != 1.0 else "")
            + ("_am" if settings.model.anti_memorization else "")
            # فريمات متعدّدة = معمارية أخرى (فرع لكل فريم): مجلدها منفصل عن النموذج أحادي الفريم
            + (("_" + "_".join(model_tfs)) if len(model_tfs or ()) > 1 else "")
            # رؤوس تصنيف فقط = معمارية أخرى: مجلد منفصل
            + ("_classonly" if settings.model.class_only else ""))

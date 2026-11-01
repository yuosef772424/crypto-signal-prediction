"""
PURPOSE:  The vocabulary of the loss plan and of model selection (tier 0, stdlib + core only): LOSS_BALANCERS (task-balancing methods by registered name), SELECTION_METRICS (the validation metrics a run may pick its best epoch by, each declaring whether it depends on the training objective), the pure checks workflow/settings.py runs at construction (check_loss_weights, check_loss_schedules, resolve_selection_metric) and the trainer `loss` section a plan produces (trainer_loss_section).
TAGS:     LOSS_BALANCERS, SELECTION_METRICS, SelectionChoice, resolve_selection_metric, check_loss_weights, check_loss_schedules, trainer_loss_section, SCHEDULE_CURVES, SUB_TERM_SCHEDULES, loss plan, loss weights, task balancing, kendall, static, selection metric, val_raw_loss, objective independent, early stopping monitor, one component one interface, tier 0
PITFALLS: A loss weight reaches the trainer ONLY through `RunSettings.train.loss` (workflow/settings.py LossSettings) -> `HeadSet.with_loss_weights` -> `targets[head].loss_weight`; a balancing method is a REGISTERED name whose entry says which trainer `loss` keys it sets (today "static" = fixed weights, "kendall" = learned homoscedastic weights on the classification heads; the default is "kendall" = what `use_uncertainty_weighting=True` always did). TaskWeightUpdater (callbacks.task_weight_update_frequency) is NOT a registered balancer yet, and GradNorm-style methods are not implemented: a new method is a new `LOSS_BALANCERS.register(...)` (unknown names raise) plus the trainer code it needs, never an edit spread over the modules. Under "kendall" a static weight on a classification head is partly absorbed by the learned log-variance (the trainer applies Kendall on weight*loss), so use "static" to compare weights cleanly. A selection metric name is `val_<stat>` (global: raw_loss, loss, class_loss) or `val_<head>_<stat>` (per head: accuracy, mae, nig_base, ic); the stats are the ones the trainer already logs (trainer/tasks.py stat_keys, trainer/trainer.py trackers). `val_raw_loss` (the default) is the weighted sum of the task losses plus penalties, i.e. the training objective: changing a loss weight or a lambda changes which epoch is best, which `objective_independent=False` says out loud; accuracy / mae / nig_base / ic are computed from outputs and labels only (ic = stats.metrics.pearson over the whole val set, logged only when it is the monitor or the head has track_ic / LossSettings.report_ic), so they do not move with loss_weight or lambda schedules (tests/test_loss_plan.py proves it on a real trainer step). Choosing a metric whose `mode` differs from the monitor's mode raises (min/max is part of the definition). trainer/config.py validate_config still checks the final monitor against the metrics the trainer logs. SCHEDULE_CURVES is the one list of curve names (trainer/config.py imports it). Nothing here imports TensorFlow.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

from core.heads import HeadSet
from core.registry import Contract, Registry

__all__ = ["LOSS_BALANCERS", "HEAD_LOSS_TERMS", "check_head_loss_terms", "check_report_heads", "SAMPLE_WEIGHTINGS", "check_sample_weighting", "SELECTION_METRICS", "SelectionChoice", "SCHEDULE_CURVES", "SUB_TERM_SCHEDULES", "DEFAULT_BALANCING",
           "DEFAULT_SELECTION_METRIC", "check_loss_weights", "check_loss_schedules", "resolve_selection_metric", "trainer_loss_section"]

#: curve names of a scheduled loss sub-term (trainer/schedules.py `_progress_curve`; trainer/config.py validates against this tuple).
SCHEDULE_CURVES = ("linear", "cosine", "exponential", "sqrt")
#: the scheduled sub-terms of the loss today: the evidence regularizer and the confidence-head calibration weight (trainer/tasks.py
#: reads them through `lambda_reg_var` / `lambda_calib_var`; with no schedule of that name the term is OFF silently, so a plan must
#: name both - switching one off is `start=end=0`, said explicitly).
SUB_TERM_SCHEDULES = ("lambda_reg", "lambda_calib")
DEFAULT_BALANCING = "kendall"
DEFAULT_SELECTION_METRIC = "val_raw_loss"

_SCHEDULE_KEYS = ("start", "end", "warmup_epochs", "schedule")


# ───────────────────────────── task balancing ─────────────────────────────
def _check_balancer(name, fn, meta):
    if not callable(fn):
        return [f"must be a callable (heads -> trainer loss-section keys), got {type(fn).__name__}"]
    out = fn(None)
    if not isinstance(out, Mapping):
        return [f"must return a mapping of trainer `loss` keys, got {type(out).__name__}"]
    bad = sorted(set(out) - {"use_uncertainty_weighting", "uncertainty_weighting_exclude_task_types", "loss_normalization"})
    return [f"returns unknown trainer loss key(s) {bad}"] if bad else []


#: balancing method name -> ``fn(heads) -> {trainer loss-section key: value}``. A core.registry Registry (unknown name raises with
#: the known names). Registered: only what the trainer implements today.
LOSS_BALANCERS = Registry("loss balancer", Contract("loss balancer", required_meta=("description",), optional_meta=(),
                                                    validator=_check_balancer))
LOSS_BALANCERS.register("static", lambda heads: {"use_uncertainty_weighting": False}, version=1,
                        description="fixed per-head weights (loss_weight), no learned balancing")
LOSS_BALANCERS.register("kendall", lambda heads: {"use_uncertainty_weighting": True}, version=1,
                        description="Kendall et al. 2018 learned homoscedastic weights on the classification heads; evidential heads "
                                    "keep their static weight (trainer default exclude list)")


LOSS_BALANCERS.register("ema_norm", lambda heads: {"use_uncertainty_weighting": False, "loss_normalization": "ema"}, version=1,
                        description="each head's loss is divided by its own EMA of |loss| (stop-gradient, beta 0.98) and then multiplied by the "
                                    "FIXED loss_weight: heads of different scales (NIG nll vs BCE vs Huber) get comparable gradient shares "
                                    "without a learned weight (no Kendall on a variance head)")


def _finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ───────────────────────────── per-head auxiliary loss terms ─────────────────────────────
def _check_head_loss_term(name, term, meta):
    problems = []
    if term != name:
        problems.append(f"the registered object must be the term name {name!r}, got {term!r}")
    if meta["kind"] not in ("per_sample", "batch"):
        problems.append(f"kind must be 'per_sample' or 'batch', got {meta['kind']!r}")
    opts = meta["options"]
    if not isinstance(opts, dict) or not opts or not all(isinstance(k, str) and _finite_number(v) and v >= 0 for k, v in opts.items()):
        problems.append(f"options must be a non-empty dict {{trainer target key: default >= 0}}, got {opts!r}")
    elif meta["weight_key"] not in opts or opts[meta["weight_key"]] != 0.0:
        problems.append(f"weight_key {meta['weight_key']!r} must be one of the options with default 0.0 (term OFF = the old behavior)")
    if not isinstance(meta["task_types"], tuple) or not meta["task_types"]:
        problems.append("task_types must be a non-empty tuple of trainer task types")
    return problems


#: auxiliary loss term name -> the same name; metadata = kind ("per_sample": added to every sample's loss before the weighted mean;
#: "batch": one scalar over the whole batch), options ({trainer target key: default}, the weight key's default 0.0 = OFF = the old
#: behavior exactly), weight_key, task_types (the trainer task types whose heads accept it), description. This is the VOCABULARY the
#: plan (LossSettings.head_terms) validates against and the trainer config keys it writes; the code of a term lives in the trainer
#: (trainer/tasks.py: huber inline in evidential_task_loss, batch terms in BATCH_LOSS_TERMS - it fails at load if one is missing).
HEAD_LOSS_TERMS = Registry("head loss term", Contract(
    "head loss term", required_meta=("kind", "options", "weight_key", "task_types", "description"), optional_meta=(),
    validator=_check_head_loss_term))
HEAD_LOSS_TERMS.register("huber", "huber", version=1, kind="per_sample", options={"huber_weight": 0.0, "huber_delta": 1.0},
                         weight_key="huber_weight", task_types=("evidential",),
                         description="Huber loss directly on mu, added to the NIG loss: mu gets a gradient that does not pass through "
                                     "the uncertainty (the head cannot 'explain' its error by inflating beta)")
HEAD_LOSS_TERMS.register("ic", "ic", version=1, kind="batch", options={"ic_weight": 0.0}, weight_key="ic_weight",
                         task_types=("evidential", "regression"),
                         description="-pearson(mu, y) over the (pooled) batch, differentiable, 0 for a constant batch: aligns the point "
                                     "forecast with the information coefficient. Batches carry no timestamp key, so it is NOT "
                                     "cross-sectional (per-timestamp) today")
HEAD_LOSS_TERMS.register("var", "var", version=1, kind="batch", options={"var_weight": 0.0}, weight_key="var_weight",
                         task_types=("evidential", "regression"),
                         description="(std(mu) - std(y))^2 over the batch (weighted): the anti-shrink term against a 'cowardly' head that "
                                     "predicts a near-constant mu to be safe; pairs with the ic term, which is scale-free")


def check_head_loss_terms(terms: Any, heads: HeadSet, what: str = "loss.head_terms") -> Dict[str, Dict[str, float]]:
    """Validate ``{head name: {trainer target key: value}}`` (the options of HEAD_LOSS_TERMS) against ``heads`` and return a plain
    copy. An unknown head, an option that is no registered term's, an option of a term the head's task type does not accept, or a
    non-finite / negative value raises (every problem listed)."""
    if not isinstance(terms, Mapping):
        raise ValueError(f"{what} must be a dict {{head name: {{option: value}}}}, got {type(terms).__name__}")
    owner = {k: e for e in HEAD_LOSS_TERMS.entries() for k in e.meta["options"]}
    problems, out = [], {}
    for h, opts in terms.items():
        if not isinstance(h, str) or h not in heads:
            problems.append(f"unknown head {h!r}; known heads: {list(heads.names)}")
            continue
        if not isinstance(opts, Mapping):
            problems.append(f"{h!r}: options must be a dict, got {type(opts).__name__}")
            continue
        got = {}
        for k, v in opts.items():
            if k not in owner:
                problems.append(f"{h!r}: unknown option {k!r}; known: {sorted(owner)}")
            elif heads[h].task_type not in owner[k].meta["task_types"]:
                problems.append(f"{h!r}: option {k!r} (term {owner[k].name!r}) is for task type(s) {list(owner[k].meta['task_types'])}, "
                                f"head is {heads[h].task_type!r}")
            elif not _finite_number(v) or v < 0 or (k.endswith("_delta") and v <= 0):
                problems.append(f"{h!r}: {k} must be a finite number {'> 0' if k.endswith('_delta') else '>= 0'}, got {v!r}")
            else:
                got[k] = float(v)
        out[h] = got
    if problems:
        raise ValueError(f"{what}: " + "; ".join(problems))
    return out


# ───────────────────────────── training sample weights ─────────────────────────────
def _abs_target_clip(abs_y, *, clip, where, lo, hi, dead_zone):
    """w = clip(|y|, lo, hi), and w = lo inside the dead zone |y| < dead_zone (noise-sized moves down-weighted, never dropped).
    Array-agnostic: the caller passes ``clip`` / ``where`` (numpy.clip / numpy.where) and float64 |y|."""
    return where(abs_y < dead_zone, lo, clip(abs_y, lo, hi))


def _check_sample_weighting(name, fn, meta):
    problems = []
    if not callable(fn):
        problems.append(f"must be a callable (abs_y, *, clip, where, **params) -> weights, got {type(fn).__name__}")
    p = meta["params"]
    if not isinstance(p, dict) or not all(isinstance(k, str) and _finite_number(v) for k, v in p.items()):
        problems.append(f"params must be a dict of finite numbers, got {p!r}")
    for k in ("target_modes", "task_types"):
        if not isinstance(meta[k], tuple) or not meta[k]:
            problems.append(f"{k} must be a non-empty tuple")
    return problems


#: training sample-weighting method name -> ``fn(abs_y, *, clip, where, **params)``: a pointwise function of each training sample's OWN
#: regression label (no learned state, nothing fitted on val/test), applied to the training loss only (the validation loss and every
#: selection metric stay unweighted). params are part of the registered definition (another setting = another registered name);
#: target_modes = the target-mode bases whose labels are in the units the params assume (volnorm: |y| in window-volatility units).
#: The trainer applies weights through its existing sample_weight path (tasks with supports_sample_weight), per head by label key.
SAMPLE_WEIGHTINGS = Registry("sample weighting", Contract(
    "sample weighting", required_meta=("params", "target_modes", "task_types", "description"), optional_meta=(),
    validator=_check_sample_weighting))
SAMPLE_WEIGHTINGS.register("abs_target_clip", _abs_target_clip, version=1, params={"lo": 0.2, "hi": 2.0, "dead_zone": 0.3},
                           target_modes=("volnorm",), task_types=("evidential", "regression"),
                           description="w = clip(|z|, 0.2, 2) with w = 0.2 for |z| < 0.3: large moves weigh up to 10x a noise move")


def check_sample_weighting(name: Any, target_mode: Optional[str], what: str = "loss.sample_weighting") -> Optional[str]:
    """None, or a registered SAMPLE_WEIGHTINGS name whose target_modes contain the base of ``target_mode`` (``"volnorm+relative"``
    -> ``"volnorm"``); anything else raises ValueError."""
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError(f"{what} must be None or a registered name, got {name!r}")
    try:
        meta = SAMPLE_WEIGHTINGS.entry(name).meta
    except ValueError as e:
        raise ValueError(f"{what}: {e}") from None
    base = (target_mode or "").partition("+")[0]
    if base not in meta["target_modes"]:
        raise ValueError(f"{what} {name!r} assumes labels of target mode(s) {list(meta['target_modes'])} (its thresholds are in those "
                         f"units); target_mode is {target_mode!r}")
    return name


def check_report_heads(names: Any, heads: HeadSet, stat: str = "ic", what: str = "loss.report_ic") -> Tuple[str, ...]:
    """Validate the heads whose epoch-level selection-metric ``stat`` (SELECTION_METRICS, per-head, e.g. ``ic``) is logged although
    it is not the monitor: a list/tuple of head names that exist and whose task type the metric supports. Returns a tuple (order kept)."""
    if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)):
        raise ValueError(f"{what} must be a list/tuple of head names, got {type(names).__name__}")
    types = SELECTION_METRICS.entry(stat).meta["task_types"]
    problems = []
    for h in names:
        if not isinstance(h, str) or h not in heads:
            problems.append(f"unknown head {h!r}; known heads: {list(heads.names)}")
        elif heads[h].task_type not in types:
            problems.append(f"{h!r}: {stat!r} is defined for task type(s) {list(types)}, head is {heads[h].task_type!r}")
    if len(set(names)) != len(names):
        problems.append(f"duplicate head(s) in {list(names)}")
    if problems:
        raise ValueError(f"{what}: " + "; ".join(problems))
    return tuple(names)


def check_loss_weights(weights: Any, heads: HeadSet, what: str = "loss.weights") -> Dict[str, float]:
    """Validate ``{head name: weight}`` against ``heads`` (unknown head raises with the known names; a weight must be a finite
    number >= 0) and return a plain ``{name: float}`` copy."""
    if not isinstance(weights, Mapping):
        raise ValueError(f"{what} must be a dict {{head name: weight}}, got {type(weights).__name__}")
    problems = []
    for k, w in weights.items():
        if not isinstance(k, str) or k not in heads:
            problems.append(f"unknown head {k!r}; known heads: {list(heads.names)}")
        if not _finite_number(w) or w < 0:
            problems.append(f"weight of {k!r} must be a finite number >= 0, got {w!r}")
    if problems:
        raise ValueError(f"{what}: " + "; ".join(problems))
    return {k: float(w) for k, w in weights.items()}


def check_loss_schedules(schedules: Any, what: str = "loss.schedules") -> Dict[str, Dict[str, Any]]:
    """Validate the sub-term schedules of a plan: exactly the names of SUB_TERM_SCHEDULES, each ``{start, end[, warmup_epochs,
    schedule]}`` with finite numbers and a known curve. Returns a deep-enough copy."""
    if not isinstance(schedules, Mapping):
        raise ValueError(f"{what} must be a dict {{sub-term: schedule}}, got {type(schedules).__name__}")
    problems = []
    missing = [n for n in SUB_TERM_SCHEDULES if n not in schedules]
    extra = sorted(str(k) for k in schedules if k not in SUB_TERM_SCHEDULES)
    if missing:
        problems.append(f"missing sub-term(s) {missing} (a missing schedule would switch the term off silently; "
                        f"switch it off with start=end=0)")
    if extra:
        problems.append(f"unknown sub-term(s) {extra}; known: {list(SUB_TERM_SCHEDULES)}")
    out = {}
    for name in SUB_TERM_SCHEDULES:
        s = schedules.get(name)
        if s is None:
            continue
        if not isinstance(s, Mapping):
            problems.append(f"{name} must be a dict, got {type(s).__name__}")
            continue
        unknown = sorted(str(k) for k in s if k not in _SCHEDULE_KEYS)
        if unknown:
            problems.append(f"{name}: unknown key(s) {unknown}; known: {list(_SCHEDULE_KEYS)}")
        for req in ("start", "end"):
            if req not in s:
                problems.append(f"{name}: required key {req!r} is missing")
            elif not _finite_number(s[req]):
                problems.append(f"{name}.{req} must be a finite number, got {s[req]!r}")
        w = s.get("warmup_epochs", 0)
        if not (isinstance(w, int) and not isinstance(w, bool) and w >= 0):
            problems.append(f"{name}.warmup_epochs must be an int >= 0, got {w!r}")
        if s.get("schedule", "linear") not in SCHEDULE_CURVES:
            problems.append(f"{name}.schedule must be one of {list(SCHEDULE_CURVES)}, got {s.get('schedule')!r}")
        out[name] = dict(s)
    if problems:
        raise ValueError(f"{what}: " + "; ".join(problems))
    return out


def trainer_loss_section(balancing: str, schedules: Mapping[str, Mapping[str, Any]], heads: Optional[HeadSet] = None) -> dict:
    """The trainer config ``loss`` section of a plan: the balancer's keys + ``schedules`` (a copy). An unknown balancer raises.
    For the default plan this is exactly ``{"use_uncertainty_weighting": True, "schedules": {lambda_reg, lambda_calib}}``."""
    section = dict(LOSS_BALANCERS.get(balancing)(heads))
    section["schedules"] = {k: dict(v) for k, v in schedules.items()}
    return section


# ───────────────────────────── selection metrics ─────────────────────────────
def _check_selection_metric(name, stat, meta):
    problems = []
    if stat != name:
        problems.append(f"the registered object must be the stat name {name!r}, got {stat!r}")
    if meta["scope"] not in ("global", "head"):
        problems.append(f"scope must be 'global' or 'head', got {meta['scope']!r}")
    if meta["mode"] not in ("min", "max"):
        problems.append(f"mode must be 'min' or 'max', got {meta['mode']!r}")
    if not isinstance(meta["objective_independent"], bool):
        problems.append("objective_independent must be a bool")
    if not isinstance(meta["task_types"], tuple):
        problems.append("task_types must be a tuple of trainer task types (empty = any head)")
    if meta["scope"] == "global" and meta["task_types"] and name != "class_loss":
        problems.append("only a per-head metric names task types (class_loss is the one global metric that needs classification heads)")
    return problems


#: stat name (the log key without its `val_` prefix / head prefix) -> the same name; metadata = scope, mode, objective_independent,
#: task_types, description. A core.registry Registry: an unknown metric raises with the known names.
SELECTION_METRICS = Registry("selection metric", Contract(
    "selection metric", required_meta=("scope", "mode", "objective_independent", "task_types", "description"), optional_meta=(),
    validator=_check_selection_metric))


def _register_metric(stat, *, scope, mode, objective_independent, task_types=(), description):
    SELECTION_METRICS.register(stat, stat, version=1, scope=scope, mode=mode, objective_independent=objective_independent,
                               task_types=tuple(task_types), description=description)


_register_metric("raw_loss", scope="global", mode="min", objective_independent=False,
                 description="weighted sum of the task losses + scheduled penalties + model regularizers = the training objective "
                             "(the default; today's behavior)")
_register_metric("loss", scope="global", mode="min", objective_independent=False,
                 description="the total training loss including the learned Kendall terms (drifts even without real improvement)")
_register_metric("class_loss", scope="global", mode="min", objective_independent=False, task_types=("classification",),
                 description="sum of the (weighted) classification-head losses (DerivedMetrics)")
_register_metric("accuracy", scope="head", mode="max", objective_independent=True, task_types=("classification",),
                 description="val accuracy of one classification head (thresholded at 0.5); no loss weight enters it")
_register_metric("mae", scope="head", mode="min", objective_independent=True, task_types=("evidential", "regression"),
                 description="val mean absolute error of one regression head's point prediction; no loss weight enters it")
_register_metric("nig_base", scope="head", mode="min", objective_independent=True, task_types=("evidential",),
                 description="val Student-t negative log-likelihood of one evidential head (before the lambda penalties and the "
                             "loss weight)")


_register_metric("ic", scope="head", mode="max", objective_independent=True, task_types=("evidential", "regression"),
                 description="val Pearson correlation (stats.metrics.pearson, degenerate = 0.0) between one regression head's point "
                             "prediction (mu) and its label over the FULL validation set (collected per batch, not a mean of batch "
                             "correlations); no loss weight and no sample weight enters it; not computed on training batches")


class SelectionChoice(NamedTuple):
    """A resolved selection metric: the trainer log key to monitor and its definition."""
    monitor: str
    stat: str
    head: Optional[str]
    mode: str
    objective_independent: bool


def resolve_selection_metric(monitor: str, heads: HeadSet, mode: Optional[str] = None) -> SelectionChoice:
    """Resolve a monitor name (``val_raw_loss``, ``val_close_class_accuracy``, ...) against SELECTION_METRICS and ``heads``.

    Raises ValueError (listing the known metrics and heads) for a name that is not ``val_<global stat>`` or ``val_<head>_<head stat>``
    with a head in ``heads`` whose task type the stat supports, and, when ``mode`` is given, for a mode that is not the metric's.
    """
    if not isinstance(monitor, str) or not monitor.startswith("val_"):
        raise ValueError(f"selection metric {monitor!r}: a model is selected on validation data, the name must be 'val_<metric>' "
                         f"or 'val_<head>_<metric>'; registered metrics: {list(SELECTION_METRICS.names())}")
    rest = monitor[4:]
    choice = None
    if rest in SELECTION_METRICS and SELECTION_METRICS.entry(rest).meta["scope"] == "global":
        meta = SELECTION_METRICS.entry(rest).meta
        if meta["task_types"] and not any(h.task_type in meta["task_types"] for h in heads):
            raise ValueError(f"selection metric {monitor!r} needs a head of task type {list(meta['task_types'])}; heads: {list(heads.names)}")
        choice = SelectionChoice(monitor, rest, None, meta["mode"], meta["objective_independent"])
    else:
        per_head = sorted((e.name for e in SELECTION_METRICS.entries() if e.meta["scope"] == "head"), key=len, reverse=True)
        for stat in per_head:
            if rest.endswith("_" + stat):
                head = rest[: -len(stat) - 1]
                meta = SELECTION_METRICS.entry(stat).meta
                if head not in heads:
                    raise ValueError(f"selection metric {monitor!r}: unknown head {head!r}; known heads: {list(heads.names)}")
                if heads[head].task_type not in meta["task_types"]:
                    raise ValueError(f"selection metric {monitor!r}: {stat!r} is defined for task type(s) {list(meta['task_types'])}, "
                                     f"head {head!r} is {heads[head].task_type!r}")
                choice = SelectionChoice(monitor, stat, head, meta["mode"], meta["objective_independent"])
                break
    if choice is None:
        raise ValueError(f"unknown selection metric {monitor!r}; use val_<{[n for n in SELECTION_METRICS.names() if SELECTION_METRICS.entry(n).meta['scope'] == 'global']}> "
                         f"or val_<head>_<{[n for n in SELECTION_METRICS.names() if SELECTION_METRICS.entry(n).meta['scope'] == 'head']}> "
                         f"(heads: {list(heads.names)})")
    if mode is not None and mode != choice.mode:
        raise ValueError(f"selection metric {monitor!r} is a {choice.mode!r} metric, early_stopping_mode must be {choice.mode!r}, got {mode!r}")
    return choice

"""
PURPOSE:  The vocabulary of the loss plan and of model selection (tier 0, stdlib + core only): LOSS_BALANCERS (task-balancing methods by registered name), SELECTION_METRICS (the validation metrics a run may pick its best epoch by, each declaring whether it depends on the training objective), the pure checks workflow/settings.py runs at construction (check_loss_weights, check_loss_schedules, resolve_selection_metric) and the trainer `loss` section a plan produces (trainer_loss_section).
TAGS:     LOSS_BALANCERS, SELECTION_METRICS, SelectionChoice, resolve_selection_metric, check_loss_weights, check_loss_schedules, trainer_loss_section, SCHEDULE_CURVES, SUB_TERM_SCHEDULES, loss plan, loss weights, task balancing, kendall, static, selection metric, val_raw_loss, objective independent, early stopping monitor, one component one interface, tier 0
PITFALLS: A loss weight reaches the trainer ONLY through `RunSettings.train.loss` (workflow/settings.py LossSettings) -> `HeadSet.with_loss_weights` -> `targets[head].loss_weight`; a balancing method is a REGISTERED name whose entry says which trainer `loss` keys it sets (today "static" = fixed weights, "kendall" = learned homoscedastic weights on the classification heads; the default is "kendall" = what `use_uncertainty_weighting=True` always did). TaskWeightUpdater (callbacks.task_weight_update_frequency) is NOT a registered balancer yet, and GradNorm-style methods are not implemented: a new method is a new `LOSS_BALANCERS.register(...)` (unknown names raise) plus the trainer code it needs, never an edit spread over the modules. Under "kendall" a static weight on a classification head is partly absorbed by the learned log-variance (the trainer applies Kendall on weight*loss), so use "static" to compare weights cleanly. A selection metric name is `val_<stat>` (global: raw_loss, loss, class_loss) or `val_<head>_<stat>` (per head: accuracy, mae, nig_base); the stats are the ones the trainer already logs (trainer/tasks.py stat_keys, trainer/trainer.py trackers). `val_raw_loss` (the default) is the weighted sum of the task losses plus penalties, i.e. the training objective: changing a loss weight or a lambda changes which epoch is best, which `objective_independent=False` says out loud; accuracy / mae / nig_base are computed from outputs and labels only, so they do not move with loss_weight or lambda schedules (tests/test_loss_plan.py proves it on a real trainer step). Choosing a metric whose `mode` differs from the monitor's mode raises (min/max is part of the definition). trainer/config.py validate_config still checks the final monitor against the metrics the trainer logs. SCHEDULE_CURVES is the one list of curve names (trainer/config.py imports it). Nothing here imports TensorFlow.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

from core.heads import HeadSet
from core.registry import Contract, Registry

__all__ = ["LOSS_BALANCERS", "SELECTION_METRICS", "SelectionChoice", "SCHEDULE_CURVES", "SUB_TERM_SCHEDULES", "DEFAULT_BALANCING",
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
    bad = sorted(set(out) - {"use_uncertainty_weighting", "uncertainty_weighting_exclude_task_types"})
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


def _finite_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


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

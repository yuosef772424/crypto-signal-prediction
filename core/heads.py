"""
PURPOSE:  The ONE declaration of "a head / a target / a loss weight" (tier 0, stdlib + core.registry): `HeadSpec` (name, target, target columns, head type, task type, role, loss, loss weight, label key, loss-facing output keys, ALL model output names, trainer options, version) and `HeadSet` (an ordered, validated tuple of HeadSpec: unique names / output names / label keys). The trainer `targets` section (`HeadSet.target_configs`), the model `head_types` (`HeadSet.head_types`), the model output names (`HeadSet.output_names`), the pipeline head names (data/heads.py) and the label keys are DERIVED VIEWS of one HeadSet; `y_<target>_*` names are built by ONE function (`y_name`; `output_name(target, role)` for a model output by role) and parsed by ONE function (`parse_output_name`).
TAGS:     HeadSet, HeadSpec, HeadKind, HEAD_KINDS, register_head_kind, y_name, output_name, label_key, head_name, parse_head_name, parse_output_name, target_configs, from_target_configs, head_types, output_names, loss_weight, head vocabulary, output names, label key, role, confidence head, uncertainty head, cascaded stages, forecast input features, one component one interface, tier 0
PITFALLS: A head TYPE (what the model builds: nig_regression, binary_classification, ...) is a `HeadKind` registered in HEAD_KINDS with its output suffixes, trainer task type, role and fixed trainer options; a `HeadSpec` is one head of one target built ONLY by `HeadSpec.make(target, head_type, ...)` (direct construction re-derives every derived field and raises on a mismatch). `HeadSet.default(price_targets, ...)` is today's head set (nig_regression + binary_classification per target; class_only = classification only): it must keep reproducing the old targets / head_types / output names exactly (tests/golden_head_vocabulary.json, tests/golden_trainer_configs.json). A loss weight equal to the trainer default (1.0, `TRAINER_DEFAULT_LOSS_WEIGHT`) is NOT written to the target config, so a default HeadSet builds the same config (and the same run fingerprint) as before; a non-default weight is written as `loss_weight`. `HeadSet.from_target_configs` accepts only configs a HeadSet would itself produce (anything else raises: a target config with extra keys is not a head set; two kinds with the same task type AND the same fixed options are ambiguous and raise, so give a new kind its own task type or a distinguishing fixed option). Consumers migrated so far: workflow/training_config.py (build_target_configs), workflow/run.py (plan_model head_types, make_training_config targets), model/builder.py and model/heads.py (output names; build_head_outputs raises when a head builder and its HeadKind disagree), model/config.py (MODEL_CONFIG head_types / price_targets defaults), data/heads.py (HEAD_KINDS / output_name / split_head_name / enabled_head_set). The model builder emits outputs in its own order (every nig head first, then the other heads per target): `HeadSet.output_names` is per head, so compare as a set / sorted when checking a built model. The key named `y_<t>_class_logits` carries a PROBABILITY (sigmoid / softmax), kept for checkpoint compatibility. Output suffixes of different kinds must not collide (a `parse_output_name` ambiguity is a registration error). Not yet migrated (PR 8): model/diagnostics.py, model/layer_report.py, workflow/selective_eval.py, evaluation/targets.py, tools/bracket_eval.py and the other literal consumers listed in tools/pattern_baseline.json (rule output-name-literal).

HOW TO EXTEND (the point of this module): a new head is a new HeadSpec in a HeadSet, never an N-file edit. (1) A new head TYPE: register its model builder (model/heads.py HEAD_REGISTRY) and a `HeadKind` here with its output suffixes. (2) A confidence head: a kind with role "confidence" (and its own task type / loss) added to the default set with `HeadSet.default(...).extended(...)`. (3) "Forecast all ~40 input features" (docs/research/hardening/proposal_cascaded_stages.md): one HeadSpec per feature, `HeadSpec.make(feature_name, "feature_regression", columns=(feature_name,))`, i.e. a generated HeadSet; the builder, the trainer targets, the label keys and the output names then follow with no further edits. Neither is implemented here.

Usage:
    heads = HeadSet.default(("high", "low"), label_smoothing=0.1)
    heads.target_configs()          # {"high_reg": {...evidential...}, "high_class": {...classification...}, ...}
    heads.head_types()              # {"high": ["nig_regression", "binary_classification"], ...}
    heads.output_names()            # ("y_high", "y_high_nu", ..., "y_high_class_logits", ...)
    HeadSet.from_target_configs(heads.target_configs()) == heads
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple

from core.registry import Contract, Registry

__all__ = ["HeadKind", "HeadSpec", "HeadSet", "HEAD_KINDS", "register_head_kind", "ROLES", "OUTPUT_PREFIX",
           "TRAINER_DEFAULT_LOSS_WEIGHT", "DEFAULT_HEAD_TYPES", "CLASS_ONLY_HEAD_TYPES", "y_name", "label_key", "head_name",
           "parse_head_name", "parse_output_name", "output_name"]

#: what a head is for (evaluation / diagnostics read the role, never the name).
ROLES = ("regression", "classification", "uncertainty", "confidence")

#: every model output / label key is `y_<stem><suffix>`; `y_name` is the only place that spells it.
OUTPUT_PREFIX = "y_"

#: the loss weight the trainer uses when a target config has no `loss_weight` (trainer/trainer.py): not written when equal.
TRAINER_DEFAULT_LOSS_WEIGHT = 1.0

#: the head types per target of today's default model (NIG regression + binary direction classification) and of `class_only`.
DEFAULT_HEAD_TYPES = ("nig_regression", "binary_classification")
CLASS_ONLY_HEAD_TYPES = ("binary_classification",)


def y_name(stem: str, suffix: str = "") -> str:
    """``y_name("high")`` -> ``"y_high"``; ``y_name("high", "_nu")`` -> ``"y_high_nu"``; ``y_name("close_class")`` -> ``"y_close_class"``
    (a label key). The ONLY place that formats a ``y_...`` name."""
    return f"{OUTPUT_PREFIX}{stem}{suffix}"


def head_name(target: str, slot: str) -> str:
    """Trainer / pipeline name of a head: ``head_name("high", "reg")`` -> ``"high_reg"``."""
    return f"{target}_{slot}"


def label_key(target: str, slot: str) -> str:
    """Key of a head's label in the data splits: ``label_key("high", "class")`` -> ``"y_high_class"``."""
    return y_name(head_name(target, slot))


@dataclass(frozen=True)
class HeadKind:
    """A head TYPE: what its model builder emits and how the trainer sees it.

    head_type:    the name registered in the model's HEAD_REGISTRY (and used in MODEL_CONFIG['head_types']).
    slot:         suffix of the head name and of its label key (`<target>_<slot>`, `y_<target>_<slot>`): "reg" / "class".
    task_type:    the trainer task type (TASK_REGISTRY); `loss` is the name of the loss that task registers.
    role:         one of ROLES.
    outputs:      {output role: suffix of `y_<target><suffix>`} for EVERY output the model builder emits, in emission order.
    loss_outputs: the output roles the loss reads (the trainer `output_keys`); a subset of `outputs`.
    fixed:        trainer options every head of this kind carries (not changeable per head).
    tunable:      {trainer option: default} a head may override (written to the target config even at its default).
    """
    head_type: str
    slot: str
    task_type: str
    role: str
    loss: str
    outputs: Mapping[str, str]
    loss_outputs: Tuple[str, ...]
    fixed: Mapping[str, Any] = field(default_factory=dict)
    tunable: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        problems = []
        for f in ("head_type", "slot", "task_type", "loss"):
            if not isinstance(getattr(self, f), str) or not getattr(self, f).strip():
                problems.append(f"{f} must be a non-empty string")
        if self.role not in ROLES:
            problems.append(f"role {self.role!r} is not one of {list(ROLES)}")
        if not self.outputs:
            problems.append("outputs is empty")
        if len(set(self.outputs.values())) != len(self.outputs):
            problems.append(f"outputs {dict(self.outputs)} repeat a suffix")
        missing = [r for r in self.loss_outputs if r not in self.outputs]
        if missing:
            problems.append(f"loss_outputs {missing} are not in outputs {list(self.outputs)}")
        both = sorted(set(self.fixed) & set(self.tunable))
        if both:
            problems.append(f"options {both} are both fixed and tunable")
        if problems:
            raise ValueError(f"HeadKind {self.head_type!r}: " + "; ".join(problems))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "loss_outputs", tuple(self.loss_outputs))
        object.__setattr__(self, "fixed", MappingProxyType(dict(self.fixed)))
        object.__setattr__(self, "tunable", MappingProxyType(dict(self.tunable)))


def _check_kind(name, kind, meta):
    if not isinstance(kind, HeadKind):
        return [f"must be a HeadKind, got {type(kind).__name__}"]
    problems = []
    if kind.head_type != name:
        problems.append(f"registered as {name!r} but kind.head_type is {kind.head_type!r}")
    # a suffix must identify one output role across all kinds (parse_output_name): a clash is a registration error
    for other_name, other in HEAD_KINDS.items():
        for role, suffix in kind.outputs.items():
            for orole, osuffix in other.outputs.items():
                if suffix == osuffix and role != orole:
                    problems.append(f"output suffix {suffix!r} is role {role!r} here but {orole!r} in {other_name!r}")
    return problems


#: head type -> HeadKind. A core.registry Registry: a duplicate name raises, an unknown one raises with the known names.
HEAD_KINDS = Registry("head kind", Contract("head kind", optional_meta=(), validator=_check_kind))


def register_head_kind(kind: HeadKind, version: int = 1) -> HeadKind:
    """Register a head type's naming / trainer facts under ``kind.head_type`` (its model builder is registered separately in the
    model's HEAD_REGISTRY). Returns the kind."""
    return HEAD_KINDS.register(kind.head_type, kind, version=version)


register_head_kind(HeadKind(
    head_type="nig_regression", slot="reg", task_type="evidential", role="regression", loss="evidential",
    outputs={"mu": "", "nu": "_nu", "alpha": "_alpha", "beta": "_beta", "epistemic": "_epistemic",
             "aleatoric": "_aleatoric", "confidence": "_confidence"},
    loss_outputs=("mu", "nu", "alpha", "beta", "confidence"),
    # the evidence regularizer (lambda_reg) and the calibration of the confidence head (lambda_calib): their schedules live in
    # main_config loss.schedules; without them lambda_reg = 0 and the conf_* head receives no gradient (see workflow/training_config.py)
    fixed={"use_calibration_loss": True, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib"}))
register_head_kind(HeadKind(
    head_type="binary_classification", slot="class", task_type="classification", role="classification", loss="classification",
    outputs={"logits": "_class_logits"}, loss_outputs=("logits",),
    fixed={"binary": True}, tunable={"label_smoothing": 0.0}))
register_head_kind(HeadKind(
    head_type="multiclass_classification", slot="class", task_type="classification", role="classification", loss="classification",
    outputs={"logits": "_class_logits"}, loss_outputs=("logits",),
    fixed={"binary": False}, tunable={"label_smoothing": 0.0}))


def _derive(kind: HeadKind, target: str) -> dict:
    """Every field of a HeadSpec that is a function of (kind, target): the ONE place the names are built."""
    outputs = {role: y_name(target, suffix) for role, suffix in kind.outputs.items()}
    return dict(name=head_name(target, kind.slot), head_type=kind.head_type, task_type=kind.task_type, role=kind.role,
                loss=kind.loss, label_key=label_key(target, kind.slot),
                output_keys={r: outputs[r] for r in kind.loss_outputs}, output_names=tuple(outputs.values()))


@dataclass(frozen=True)
class HeadSpec:
    """One head of one target. Build it with :meth:`make`; direct construction is checked against the head kind.

    name:           trainer / pipeline name (`high_reg`).
    target:         the stem the names are built from (`high`).
    target_columns: the data columns the head predicts (default: just ``target``).
    head_type, task_type, role, loss: copied from the head kind.
    loss_weight:    static loss weight (default `TRAINER_DEFAULT_LOSS_WEIGHT`).
    label_key:      key of the label in the data splits (`y_high_reg`).
    output_keys:    {role: model output name} the loss reads (the trainer `output_keys`).
    output_names:   EVERY output the model builder emits for this head.
    options:        the trainer options (fixed + tunable values).
    version:        the head kind's registered version.
    """
    name: str
    target: str
    target_columns: Tuple[str, ...]
    head_type: str
    task_type: str
    role: str
    loss: str
    loss_weight: float
    label_key: str
    output_keys: Mapping[str, str]
    output_names: Tuple[str, ...]
    options: Mapping[str, Any]
    version: Any

    @classmethod
    def make(cls, target: str, head_type: str, *, loss_weight: float = TRAINER_DEFAULT_LOSS_WEIGHT,
             columns: Optional[Iterable[str]] = None, **options) -> "HeadSpec":
        """The head of ``head_type`` for ``target``; ``options`` may name only the kind's tunable options (an unknown one raises)."""
        entry = HEAD_KINDS.entry(head_type)
        kind = entry.obj
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"HeadSpec: target must be a non-empty string, got {target!r}")
        unknown = sorted(set(options) - set(kind.tunable))
        if unknown:
            raise ValueError(f"HeadSpec {head_name(target, kind.slot)!r}: unknown option(s) {unknown} for head type "
                             f"{head_type!r}; tunable: {sorted(kind.tunable)}")
        merged = {**kind.fixed, **kind.tunable, **options}
        return cls(target=target, target_columns=tuple(columns) if columns is not None else (target,),
                   loss_weight=loss_weight, options=merged, version=entry.version, **_derive(kind, target))

    def __post_init__(self):
        kind = HEAD_KINDS.get(self.head_type)       # unknown head type raises (with the known ones)
        problems = []
        want = _derive(kind, self.target) if isinstance(self.target, str) and self.target.strip() else None
        if want is None:
            problems.append(f"target must be a non-empty string, got {self.target!r}")
        else:
            for k, v in want.items():
                got = getattr(self, k)
                got = dict(got) if isinstance(v, dict) else tuple(got) if isinstance(v, tuple) else got
                if got != v:
                    problems.append(f"{k} is {got!r}, the {self.head_type!r} head of target {self.target!r} has {v!r}")
        cols = tuple(self.target_columns)
        if not cols or not all(isinstance(c, str) and c for c in cols):
            problems.append(f"target_columns must be non-empty strings, got {self.target_columns!r}")
        w = self.loss_weight
        if isinstance(w, bool) or not isinstance(w, (int, float)) or w != w or w in (float("inf"), float("-inf")) or w < 0:
            problems.append(f"loss_weight must be a finite number >= 0, got {w!r}")
        opts = dict(self.options)
        bad_fixed = sorted(k for k, v in kind.fixed.items() if opts.get(k, object()) != v)
        extra = sorted(set(opts) - set(kind.fixed) - set(kind.tunable))
        missing = sorted(set(kind.tunable) - set(opts))
        if bad_fixed:
            problems.append(f"fixed option(s) {bad_fixed} of {self.head_type!r} must equal {dict(kind.fixed)}")
        if extra:
            problems.append(f"unknown option(s) {extra}; tunable: {sorted(kind.tunable)}")
        if missing:
            problems.append(f"missing option(s) {missing}")
        if problems:
            raise ValueError(f"HeadSpec {self.name!r}: " + "; ".join(problems))
        object.__setattr__(self, "target_columns", cols)
        object.__setattr__(self, "output_keys", MappingProxyType(dict(self.output_keys)))
        object.__setattr__(self, "output_names", tuple(self.output_names))
        object.__setattr__(self, "options", MappingProxyType(opts))
        object.__setattr__(self, "loss_weight", float(w))

    def target_config(self) -> dict:
        """The trainer `targets[name]` entry of this head. `loss_weight` is written only when it differs from the trainer default."""
        cfg = {"true_key": self.label_key, "task_type": self.task_type, "output_keys": dict(self.output_keys)}
        cfg.update(self.options)
        if self.loss_weight != TRAINER_DEFAULT_LOSS_WEIGHT:
            cfg["loss_weight"] = self.loss_weight
        return cfg

    def __hash__(self):
        return hash((self.name, self.head_type, self.loss_weight))


@dataclass(frozen=True)
class HeadSet:
    """An ordered, validated set of heads: unique names, unique output names, unique label keys."""
    heads: Tuple[HeadSpec, ...]

    def __post_init__(self):
        heads = tuple(self.heads)
        problems = []
        if not heads:
            problems.append("a HeadSet needs at least one head (a model without outputs cannot be built)")
        if not all(isinstance(h, HeadSpec) for h in heads):
            raise ValueError("HeadSet: every element must be a HeadSpec (build them with HeadSpec.make)")
        for what, values in (("name", [h.name for h in heads]), ("label key", [h.label_key for h in heads]),
                             ("output name", [o for h in heads for o in h.output_names])):
            seen, dup = set(), set()
            for v in values:
                (dup if v in seen else seen).add(v)
            if dup:
                problems.append(f"duplicate {what}(s) {sorted(dup)}")
        if problems:
            raise ValueError("HeadSet: " + "; ".join(problems))
        object.__setattr__(self, "heads", heads)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_head_types(cls, head_types: Mapping[str, Iterable[str]], *, label_smoothing: float = 0.0,
                        loss_weights: Optional[Mapping[str, float]] = None) -> "HeadSet":
        """``{target: [head type, ...]}`` (the model's head_types) -> a HeadSet; heads in the dict's order. ``label_smoothing`` goes to
        every head type that has that option; ``loss_weights`` is ``{head name: weight}`` (unknown names raise)."""
        heads = []
        for target, types in head_types.items():
            for head_type in types:
                kind = HEAD_KINDS.get(head_type)
                opts = {"label_smoothing": label_smoothing} if "label_smoothing" in kind.tunable else {}
                heads.append(HeadSpec.make(target, head_type, **opts))
        hs = cls(tuple(heads))
        return hs.with_loss_weights(loss_weights) if loss_weights else hs

    @classmethod
    def default(cls, price_targets: Iterable[str], *, class_only: bool = False, label_smoothing: float = 0.0) -> "HeadSet":
        """Today's heads for ``price_targets``: NIG regression + binary classification per target (class_only: classification only)."""
        types = CLASS_ONLY_HEAD_TYPES if class_only else DEFAULT_HEAD_TYPES
        return cls.from_head_types({t: list(types) for t in price_targets}, label_smoothing=label_smoothing)

    @classmethod
    def from_target_configs(cls, targets: Mapping[str, Mapping[str, Any]]) -> "HeadSet":
        """The inverse of :meth:`target_configs`. A config a HeadSet would not itself produce raises (no silent drop of keys)."""
        heads = []
        for name, cfg in targets.items():
            kinds = [k for k in HEAD_KINDS.values()
                     if k.task_type == cfg.get("task_type") and all(cfg.get(f, object()) == v for f, v in k.fixed.items())]
            if len(kinds) != 1:
                raise ValueError(f"target {name!r}: {len(kinds)} head kinds match task_type={cfg.get('task_type')!r} and its fixed "
                                 f"options (known kinds: {HEAD_KINDS.names()})")
            kind = kinds[0]
            suffix = f"_{kind.slot}"
            if not name.endswith(suffix) or len(name) == len(suffix):
                raise ValueError(f"target {name!r}: a {kind.head_type!r} head is named '<target>{suffix}'")
            opts = {k: cfg[k] for k in kind.tunable if k in cfg}
            spec = HeadSpec.make(name[:-len(suffix)], kind.head_type,
                                 loss_weight=cfg.get("loss_weight", TRAINER_DEFAULT_LOSS_WEIGHT), **opts)
            if spec.target_config() != dict(cfg):
                raise ValueError(f"target {name!r}: the config {dict(cfg)!r} is not what a HeadSet produces "
                                 f"({spec.target_config()!r})")
            heads.append(spec)
        return cls(tuple(heads))

    # ------------------------------------------------------------------ container
    def __iter__(self) -> Iterator[HeadSpec]:
        return iter(self.heads)

    def __len__(self) -> int:
        return len(self.heads)

    def __getitem__(self, name: str) -> HeadSpec:
        for h in self.heads:
            if h.name == name:
                return h
        raise KeyError(f"unknown head {name!r}; known: {list(self.names)}")

    def __contains__(self, name: object) -> bool:
        return any(h.name == name for h in self.heads)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(h.name for h in self.heads)

    @property
    def targets(self) -> Tuple[str, ...]:
        """The distinct targets in order of first appearance."""
        return tuple(dict.fromkeys(h.target for h in self.heads))

    # ------------------------------------------------------------------ derived views
    def target_configs(self) -> dict:
        """The trainer `targets` section: ``{head name: target config}`` in head order."""
        return {h.name: h.target_config() for h in self.heads}

    def head_types(self) -> dict:
        """The model's ``head_types``: ``{target: [head type, ...]}`` in head order."""
        out: dict = {}
        for h in self.heads:
            out.setdefault(h.target, []).append(h.head_type)
        return out

    def output_names(self) -> Tuple[str, ...]:
        """Every model output name of every head, in head order (a built model lists its nig heads first: compare sorted)."""
        return tuple(o for h in self.heads for o in h.output_names)

    def label_keys(self) -> Tuple[str, ...]:
        return tuple(h.label_key for h in self.heads)

    def loss_weights(self) -> dict:
        return {h.name: h.loss_weight for h in self.heads}

    # ------------------------------------------------------------------ derived sets
    def select(self, *, role: Optional[str] = None, task_type: Optional[str] = None,
               targets: Optional[Iterable[str]] = None) -> "HeadSet":
        """The sub-set whose heads match every given criterion (an empty result raises)."""
        keep = set(targets) if targets is not None else None
        return HeadSet(tuple(h for h in self.heads if (role is None or h.role == role)
                             and (task_type is None or h.task_type == task_type)
                             and (keep is None or h.target in keep)))

    def select_names(self, names: Iterable[str]) -> "HeadSet":
        """The sub-set with exactly the heads named in ``names`` (head order kept; an unknown name raises, an empty result raises)."""
        names = set(names)
        unknown = sorted(names - set(self.names))
        if unknown:
            raise ValueError(f"select_names: unknown head(s) {unknown}; known: {list(self.names)}")
        return HeadSet(tuple(h for h in self.heads if h.name in names))

    def with_loss_weights(self, weights: Mapping[str, float]) -> "HeadSet":
        """A copy with the given ``{head name: weight}`` (an unknown head name raises)."""
        unknown = sorted(set(weights) - set(self.names))
        if unknown:
            raise ValueError(f"with_loss_weights: unknown head(s) {unknown}; known: {list(self.names)}")
        return HeadSet(tuple(HeadSpec.make(h.target, h.head_type, loss_weight=weights.get(h.name, h.loss_weight),
                                           columns=h.target_columns, **{k: h.options[k] for k in HEAD_KINDS.get(h.head_type).tunable})
                             for h in self.heads))

    def extended(self, *more: HeadSpec) -> "HeadSet":
        """A copy with more heads appended (a name / output-name clash raises)."""
        return HeadSet(self.heads + tuple(more))


def parse_head_name(name: str) -> Tuple[str, str]:
    """``"low_reg"`` -> ``("low", "reg")`` for the registered slots (an unknown slot raises)."""
    for slot in sorted({k.slot for k in HEAD_KINDS.values()}, key=len, reverse=True):
        suffix = f"_{slot}"
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)], slot
    raise ValueError(f"unknown head name {name!r}: it does not end with one of "
                     f"{['_' + s for s in sorted({k.slot for k in HEAD_KINDS.values()})]}")


def output_name(target: str, role: str) -> str:
    """The model output of ``role`` for ``target``: ``output_name("high", "nu")`` -> ``"y_high_nu"``, ``output_name("low", "mu")`` ->
    ``"y_low"``, ``output_name("close", "logits")`` -> ``"y_close_class_logits"`` (a probability, see PITFALLS). The inverse of
    `parse_output_name`; an unknown role raises with the known ones, a role with two different suffixes in the registered kinds
    raises (ambiguous). Consumers read outputs through this and never spell the key."""
    suffixes: dict = {}
    for kind in HEAD_KINDS.values():
        for r, suffix in kind.outputs.items():
            if suffixes.setdefault(r, suffix) != suffix:
                raise ValueError(f"output role {r!r} has two suffixes ({suffixes[r]!r}, {suffix!r}) in the registered head kinds")
    if role not in suffixes:
        raise ValueError(f"unknown output role {role!r}; registered roles: {sorted(suffixes)}")
    return y_name(target, suffixes[role])


def parse_output_name(key: str) -> Tuple[str, str]:
    """A model output name -> ``(target, output role)``: ``"y_high_nu"`` -> ``("high", "nu")``, ``"y_low"`` -> ``("low", "mu")``,
    ``"y_close_class_logits"`` -> ``("close", "logits")``. The inverse of `y_name` over the registered kinds; anything else raises."""
    if not key.startswith(OUTPUT_PREFIX):
        raise ValueError(f"not a model output name (no {OUTPUT_PREFIX!r} prefix): {key!r}")
    stem = key[len(OUTPUT_PREFIX):]
    roles = {}
    for kind in HEAD_KINDS.values():
        for role, suffix in kind.outputs.items():
            roles[suffix] = role
    for suffix in sorted((s for s in roles if s), key=len, reverse=True):
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[:-len(suffix)], roles[suffix]
    if "" in roles and stem:
        return stem, roles[""]
    raise ValueError(f"not an output name of a registered head kind: {key!r}")

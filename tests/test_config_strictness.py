"""trainer.build_config is strict at construction (audit 2026-10-31, class C2, PR 3), and strictness changed nothing that already worked.

Four guards:
1. MUTATION TABLE: every bad mutation of a valid config (typo'd key at any level, NaN / negative / string weights, unregistered
   task type, dangling schedule / weight_var reference, a monitor the trainer never logs, a knob that would be ignored...) must
   raise ValueError with the expected message fragment. A new validated property = a new row.
2. COVERAGE: every key of TRAINER_DEFAULT_CONFIG has a validation rule (a new default key without one fails), and the validator
   is the ONE place: build_config calls validate_config, trainer modules carry no second copy of the checks.
3. EQUIVALENCE: tests/golden_trainer_configs.json holds, recorded from the code BEFORE the validator existed (commit a248537),
   the merged config + config_fingerprint of (a) every user config the repo's tests/fixtures/self-tests pass to build_config,
   (b) the training config of 30+ RunSettings variants (workflow.run.make_training_config) incl. the six recorded scenarios of
   tests/golden_run_settings.json and the capacity / permutation-control re-build of a built config, (c) the trainer example
   config. Validation only rejects, so every one must still build the SAME config and the SAME fingerprint.
4. DEAD KEYS: the PIPELINE_DEFAULT_CONFIG keys read nowhere in the code packages are frozen (they may only shrink; a new dead
   key fails), and every trainer default key is read by trainer code.
    python -m pytest tests/test_config_strictness.py -q
"""
import ast
import contextlib
import copy
import glob
import hashlib
import io
import json
import math
import os
import re
import sys
import types
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

import _nbload  # noqa: E402

GOLDEN_PATH = os.path.join(ROOT, "tests", "golden_trainer_configs.json")
CODE_PACKAGES = ("core", "cross_asset", "tools", "data", "model", "trainer", "evaluation", "signal_eval", "workflow", "discovery")

#: PIPELINE_DEFAULT_CONFIG keys that no code package reads (quoted-string scan below). The audit listed 19 (+ target_loss_weights,
#: which IS read: data/heads.py); the scan also finds model_name, use_decomposition and top_k, whose names occur in the code only as
#: function parameters of the model, never as CONFIG keys. May only SHRINK: removing a key (separate `refactor` PR) shrinks it.
DEAD_PIPELINE_KEYS = frozenset("""
scaling_window preprocessed_data_file_id kv_latent expansion model_dropout model_name use_decomposition decomp_kernels moe_every
n_shared n_routed top_k gradient_clipnorm use_class_weights class_weight_smoothing reduce_lr_patience reduce_lr_factor
eval_batch_size eval_range_frac eval_n_display workspace_dir use_drive
""".split())


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


_NS = {}


def trainer_ns():
    """A fresh trainer namespace per call site is not needed: tests that register task types clean up after themselves."""
    if "ns" not in _NS:
        _NS["ns"] = _quiet(_nbload.load_trainer)
    return _NS["ns"]


def light_run_ns():
    """Trainer + the three workflow modules that build the training config (no pipeline, no model): ~3 s."""
    if "run" not in _NS:
        ns = _quiet(_nbload.load_trainer)
        _quiet(_nbload.workflow_package().load_into, ns, only=("settings", "training_config", "run"))
        _NS["run"] = ns
    return _NS["run"]


def canon_sha(ns, cfg):
    return hashlib.sha256(json.dumps(ns["_canonical"](cfg), sort_keys=True).encode("utf-8")).hexdigest()


# ───────────────────────────── 1) the mutation table ─────────────────────────────
def good_config():
    """A valid user config exercising every section (evidential + classification + regression, schedules, a constraint)."""
    return {
        "run": {"run_dir": "/tmp/strict_run", "epochs": 3, "batch_size": 16, "train_mode": "new", "verbose": 0, "warm_start": {}},
        "targets": {
            "a_reg": {"true_key": "y_a_reg", "task_type": "evidential",
                      "output_keys": {"mu": "y_a", "nu": "y_a_nu", "alpha": "y_a_alpha", "beta": "y_a_beta",
                                      "confidence": "y_a_confidence"},
                      "use_calibration_loss": True, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib"},
            "a_class": {"true_key": "y_a_class", "task_type": "classification", "binary": True,
                        "output_keys": {"logits": "y_a_class_logits"}, "label_smoothing": 0.1, "loss_weight": 0.5},
            "b": {"true_key": "y_b", "task_type": "regression", "output_keys": {"pred": "y_b"}, "loss_fn": "huber",
                  "huber_delta": 2.0},
        },
        "loss": {"use_uncertainty_weighting": True,
                 "schedules": {"lambda_reg": {"start": 0.0, "end": 0.05, "warmup_epochs": 5, "schedule": "linear"},
                               "lambda_calib": {"start": 0.0, "end": 0.1, "warmup_epochs": 5, "schedule": "cosine"},
                               "penalty": {"start": 0.0, "end": 0.3, "warmup_epochs": 2}},
                 "constraints": [{"name": "order", "fn": lambda outputs: 0.0, "weight_var": "penalty"}]},
        "optimizer": {"name": "adamw", "lr_initial": 1e-3, "clip_norm": 1.0, "use_ema": True, "ema_momentum": 0.999,
                      "lr_schedule": {"type": "cosine_restarts", "cycle_length": 10, "cycle_mult": 1.5}, "weight_decay_exclude": []},
        "callbacks": {"early_stopping": {"monitor": "val_raw_loss", "mode": "min"}, "snapshots": {"epochs": []}},
        "checkpoint": {"save_every": 1, "max_to_keep": 2, "save_best_weights": True},
    }


def _set(path, value):
    """Mutation: set config[path...] = value (path is a tuple of keys)."""
    def mutate(cfg):
        d = cfg
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return mutate


def _del(path):
    def mutate(cfg):
        d = cfg
        for k in path[:-1]:
            d = d[k]
        del d[path[-1]]
    return mutate


def _many(*ms):
    return lambda cfg: [m(cfg) for m in ms]


_A = ("targets", "a_reg")
_ES = ("callbacks", "early_stopping")
#: (id, mutation, message fragment that must appear in the ValueError)
MUTATIONS = [
    # unknown keys at every level, with a close-match suggestion
    ("top_level_typo", _set(("losss",), {}), "unknown key 'losss'"),
    ("top_level_typo_suggests", _set(("losss",), {}), "did you mean 'loss'"),
    ("run_typo", _set(("run", "epocs"), 3), "did you mean 'epochs'"),
    ("target_loss_weight_typo", _set(_A + ("loss_wieght",), 2.0), "did you mean 'loss_weight'"),
    ("loss_typo", _set(("loss", "use_uncertainity_weighting"), True), "did you mean 'use_uncertainty_weighting'"),
    ("optimizer_typo", _set(("optimizer", "lr_initail"), 1e-3), "did you mean 'lr_initial'"),
    ("early_stopping_typo", _set(_ES + ("paitence",), 3), "did you mean 'patience'"),
    ("checkpoint_typo", _set(("checkpoint", "max_to_kepp"), 3), "did you mean 'max_to_keep'"),
    ("callbacks_typo", _set(("callbacks", "metrics_log_evry"), 1), "did you mean 'metrics_log_every'"),
    ("warm_start_typo", _set(("run", "warm_start", "epoch_done"), 3), "did you mean 'epochs_done'"),
    ("lists_known_keys", _set(("losss",), {}), "known keys:"),
    ("key_of_another_task_type", _set(_A + ("label_smoothing",), 0.1), "unknown key 'label_smoothing' for task type 'evidential'"),
    ("classification_key_on_regression", _set(("targets", "b", "binary"), True), "unknown key 'binary' for task type 'regression'"),
    # loss weights: finite and >= 0
    ("loss_weight_negative", _set(("targets", "a_class", "loss_weight"), -0.5), "loss_weight: must be >= 0.0"),
    ("loss_weight_nan", _set(("targets", "a_class", "loss_weight"), float("nan")), "loss_weight: must be a finite number"),
    ("loss_weight_inf", _set(("targets", "a_class", "loss_weight"), float("inf")), "loss_weight: must be a finite number"),
    ("loss_weight_string", _set(("targets", "a_class", "loss_weight"), "1.0"), "loss_weight: must be a finite number"),
    ("loss_weight_bool", _set(("targets", "a_class", "loss_weight"), True), "loss_weight: must be a finite number"),
    ("loss_weight_none", _set(("targets", "a_class", "loss_weight"), None), "loss_weight: must be a finite number"),
    ("huber_weight_negative", _set(_A + ("huber_weight",), -1.0), "huber_weight: must be >= 0.0"),
    ("beta_nll_nan", _set(_A + ("beta_nll",), float("nan")), "beta_nll: must be a finite number"),
    ("huber_delta_zero", _set(("targets", "b", "huber_delta"), 0.0), "huber_delta: must be > 0.0"),
    ("label_smoothing_one", _set(("targets", "a_class", "label_smoothing"), 1.0), "label_smoothing: must be < 1.0"),
    ("flood_level_negative", _set(("targets", "a_class", "flood_level"), -0.1), "flood_level: must be >= 0.0"),
    ("regression_loss_fn_typo", _set(("targets", "b", "loss_fn"), "mse2"), "loss_fn: must be one of"),
    ("flag_not_bool", _set(_A + ("normalize_reg",), "yes"), "normalize_reg: must be a bool"),
    # task types
    ("unregistered_task_type", _set(("targets", "b", "task_type"), "quantile"), "unregistered task type 'quantile'"),
    ("task_type_not_a_string", _set(("targets", "b", "task_type"), 3), "unregistered task type 3"),
    ("target_missing_true_key", _del(("targets", "b", "true_key")), "المفتاح المطلوب 'true_key'"),
    ("target_missing_output_keys", _del(("targets", "b", "output_keys")), "المفتاح المطلوب 'output_keys'"),
    ("target_output_keys_empty", _set(("targets", "b", "output_keys"), {}), "output_keys: must be non-empty"),
    ("no_targets", _set(("targets",), {}), "هدف واحد على الأقل"),
    ("target_not_a_dict", _set(("targets", "b"), 5), "must be a dict"),
    ("exclude_types_unregistered", _set(("loss", "uncertainty_weighting_exclude_task_types"), ["evidentail"]),
     "unregistered task type 'evidentail'"),
    # references that must resolve (a dangling schedule name used to switch the regularizer OFF silently)
    ("lambda_reg_var_dangling", _set(_A + ("lambda_reg_var",), "lamda_reg"), "names the schedule 'lamda_reg', which is not defined in loss.schedules"),
    ("lambda_reg_var_dangling_suggests", _set(_A + ("lambda_reg_var",), "lamda_reg"), "did you mean 'lambda_reg'"),
    ("lambda_calib_var_dangling", _set(_A + ("lambda_calib_var",), "lambda_calibb"), "lambda_calib_var: names the schedule"),
    ("schedule_removed_but_still_referenced", _del(("loss", "schedules", "lambda_reg")), "lambda_reg_var: names the schedule 'lambda_reg'"),
    ("constraint_weight_var_dangling", _set(("loss", "constraints", 0, "weight_var"), "pnalty"), "weight_var: names the schedule 'pnalty'"),
    ("constraint_without_fn", _set(("loss", "constraints", 0, "fn"), None), "fn: must be callable"),
    ("constraint_without_name", _set(("loss", "constraints", 0, "name"), ""), "name: must be a non-empty str"),
    ("constraint_unknown_key", _set(("loss", "constraints", 0, "wieght_var"), "penalty"), "unknown key 'wieght_var'"),
    ("constraint_duplicate_name", lambda c: c["loss"]["constraints"].append(dict(c["loss"]["constraints"][0])), "duplicate constraint name 'order'"),
    # schedules
    ("schedule_curve_typo", _set(("loss", "schedules", "lambda_reg", "schedule"), "cosin"), "schedule: must be one of ['linear', 'cosine', 'exponential', 'sqrt']"),
    ("schedule_missing_end", _del(("loss", "schedules", "lambda_reg", "end")), "required key 'end' is missing"),
    ("schedule_nan_start", _set(("loss", "schedules", "lambda_reg", "start"), float("nan")), "start: must be a finite number"),
    ("schedule_negative_warmup", _set(("loss", "schedules", "lambda_reg", "warmup_epochs"), -1), "warmup_epochs: must be an int >= 0"),
    ("schedule_key_typo", _set(("loss", "schedules", "lambda_reg", "warmup_epoch"), 2), "did you mean 'warmup_epochs'"),
    # early stopping / monitor sanity
    ("monitor_typo", _set(_ES + ("monitor",), "val_raw_los"), "is not a metric this trainer logs"),
    ("monitor_typo_suggests", _set(_ES + ("monitor",), "val_raw_los"), "did you mean 'raw_loss'"),
    ("monitor_unknown_target", _set(_ES + ("monitor",), "val_zzz_mae"), "is not a metric this trainer logs"),
    ("monitor_stat_of_wrong_task_type", _set(_ES + ("monitor",), "val_a_class_mae"), "is not a metric this trainer logs"),
    ("monitor_loss_with_max", _set(_ES + ("mode",), "max"), "(smaller is better), mode must be 'min'"),
    ("monitor_accuracy_with_min", _set(_ES + ("monitor",), "val_a_class_accuracy"), "(larger is better), mode must be 'max'"),
    ("early_stopping_mode_bad", _set(_ES + ("mode",), "minimum"), "mode: must be one of ['min', 'max']"),
    ("early_stopping_smoothing_bad", _set(_ES + ("smoothing",), "ewma"), "smoothing: must be one of"),
    ("early_stopping_ema_beta_one", _set(_ES + ("ema_beta",), 1.0), "ema_beta: must be < 1.0"),
    ("early_stopping_patience_zero", _set(_ES + ("patience",), 0), "patience: must be an int >= 1"),
    ("early_stopping_weights_snapshot_bad", _set(_ES + ("weights_snapshot",), "ema"), "weights_snapshot: must be one of"),
    ("ema_weights_without_ema", _many(_set(_ES + ("weights_snapshot",), "ema_weights"), _set(("optimizer", "use_ema"), False)),
     "needs optimizer.use_ema=True"),
    ("task_weight_updater_ignored_under_kendall", _set(("callbacks", "task_weight_update_frequency"), 5), "has no effect while loss.use_uncertainty_weighting=True"),
    ("class_baselines_unknown_target", _set(("callbacks", "class_baselines"), {"zzz": 0.5}), "'zzz' is not a target"),
    ("class_baselines_not_number", _set(("callbacks", "class_baselines"), {"a_class": float("nan")}), "class_baselines['a_class']"),
    ("snapshot_epochs_bad", _set(("callbacks", "snapshots", "epochs"), [0]), "snapshots.epochs: must be a list of epoch numbers"),
    # optimizer
    ("optimizer_name_typo", _set(("optimizer", "name"), "adamx"), "name: must be one of ['adamw', 'adam']"),
    ("lr_negative", _set(("optimizer", "lr_initial"), -1e-3), "lr_initial: must be > 0.0"),
    ("lr_nan", _set(("optimizer", "lr_initial"), float("nan")), "lr_initial: must be a finite number"),
    ("lr_schedule_type_typo", _set(("optimizer", "lr_schedule", "type"), "cosine_restart"), "type: must be one of ['constant', 'cosine', 'cosine_restarts']"),
    ("lr_schedule_cycle_mult_below_one", _set(("optimizer", "lr_schedule", "cycle_mult"), 0.5), "cycle_mult: must be >= 1.0"),
    ("weight_decay_negative", _set(("optimizer", "weight_decay"), -1.0), "weight_decay: must be >= 0.0"),
    ("weight_decay_exclude_bad_regex", _set(("optimizer", "weight_decay_exclude"), ["(unclosed"]), "invalid regex"),
    ("clip_norm_zero", _set(("optimizer", "clip_norm"), 0.0), "clip_norm: must be > 0.0"),
    ("ema_momentum_one", _set(("optimizer", "ema_momentum"), 1.0), "ema_momentum: must be < 1.0"),
    # run / checkpoint
    ("epochs_zero", _set(("run", "epochs"), 0), "epochs: must be an int >= 1"),
    ("epochs_float", _set(("run", "epochs"), 2.5), "epochs: must be an int"),
    ("batch_size_negative", _set(("run", "batch_size"), -4), "batch_size: must be an int >= 1"),
    ("verbose_out_of_range", _set(("run", "verbose"), 5), "verbose: must be one of [0, 1, 2]"),
    ("train_mode_bad", _set(("run", "train_mode"), "restart"), "run.train_mode='restart'"),
    ("on_existing_bad", _set(("run", "on_existing"), "overwrite"), "on_existing: must be one of ['archive', 'error']"),
    ("run_dir_empty", _set(("run", "run_dir"), ""), "run_dir: must be a non-empty path"),
    ("warm_start_without_epochs_done", _many(_set(("run", "train_mode"), "warm_start"), _set(("run", "warm_start", "weights_path"), "/x.h5")),
     "warm_start.epochs_done: required"),
    ("warm_start_without_weights", _many(_set(("run", "train_mode"), "warm_start"), _set(("run", "warm_start", "epochs_done"), 3)),
     "warm_start.weights_path: required"),
    ("max_to_keep_zero", _set(("checkpoint", "max_to_keep"), 0), "max_to_keep: must be an int >= 1"),
    ("save_best_weights_not_bool", _set(("checkpoint", "save_best_weights"), "yes"), "save_best_weights: must be a bool"),
    # section shapes
    ("run_not_a_dict", _set(("run",), 5), "config.run: must be a dict"),
    ("loss_not_a_dict", _set(("loss",), "x"), "config.loss: must be a dict"),
    ("schedules_not_a_dict", _set(("loss", "schedules"), []), "loss.schedules: must be a dict"),
    ("constraints_not_a_list", _set(("loss", "constraints"), {}), "loss.constraints: must be a list"),
]


class MutationTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = trainer_ns()

    def build(self, user):
        return self.ns["build_config"](user)

    def test_the_good_config_builds_and_is_unchanged_by_validation(self):
        user = good_config()
        cfg = self.build(copy.deepcopy(user))
        self.assertEqual(cfg["targets"]["a_class"]["loss_weight"], 0.5)
        # validation rejects, it never rewrites: the result is exactly the deep merge of the defaults and the user config
        merged = self.ns["deep_update"](self.ns["TRAINER_DEFAULT_CONFIG"], user)
        self.assertEqual(self.ns["_canonical"](cfg), self.ns["_canonical"](merged))

    def test_every_mutation_raises_with_its_message(self):
        self.assertGreaterEqual(len(MUTATIONS), 80)
        ids = [m[0] for m in MUTATIONS]
        self.assertEqual(len(ids), len(set(ids)), "duplicate mutation ids")
        for mid, mutate, fragment in MUTATIONS:
            with self.subTest(mutation=mid):
                user = good_config()
                mutate(user)
                with self.assertRaises(ValueError) as cm:
                    self.build(user)
                self.assertIn(fragment, str(cm.exception), f"{mid}: {cm.exception}")

    def test_valid_edge_values_are_accepted(self):
        """Strictness is not over-reach: zero weights (a switched-off head), None-able knobs, numpy ints, a custom monitor of a
        derived metric, a regression head with every loss_fn are legitimate."""
        user = good_config()
        user["targets"]["a_class"]["loss_weight"] = 0                      # int zero = head switched off
        user["targets"]["a_reg"]["loss_weight"] = np.float32(0.25)
        user["targets"]["a_class"]["flood_level"] = None
        user["run"]["epochs"] = np.int64(2)
        user["run"]["seed"] = 0
        user["optimizer"]["clip_norm"] = None
        user["optimizer"]["ema_window_epochs"] = 1.0
        user["callbacks"]["early_stopping"].update(monitor="val_class_loss", mode="min", patience=10 ** 6)
        user["callbacks"]["class_baselines"] = {"a_class": 0.51}           # added after the build by workflow.make_training_config
        for fn in ("mse", "mae", "huber"):
            user["targets"]["b"]["loss_fn"] = fn
            self.build(copy.deepcopy(user))
        user["callbacks"]["early_stopping"].update(monitor="val_a_class_accuracy", mode="max")
        self.build(copy.deepcopy(user))
        user["callbacks"]["early_stopping"].update(monitor="constraint_order", mode="min")
        self.build(copy.deepcopy(user))
        user["callbacks"]["early_stopping"].update(monitor="loss", mode="min")  # a training metric (no val_ prefix)
        self.build(copy.deepcopy(user))

    def test_a_default_named_schedule_that_is_absent_is_not_an_explicit_reference(self):
        """A target that does not name lambda_reg_var keeps the tasks.py default (regularizer off without a schedule): only an
        EXPLICIT dangling name is a typo we can prove."""
        user = good_config()
        for k in ("lambda_reg_var", "lambda_calib_var"):
            del user["targets"]["a_reg"][k]
        user["loss"]["schedules"] = {}
        user["loss"]["constraints"] = []
        self.build(user)

    def test_all_problems_are_reported_together(self):
        user = good_config()
        user["losss"] = {}
        user["targets"]["a_class"]["loss_weight"] = float("nan")
        user["optimizer"]["lr_initial"] = -1.0
        with self.assertRaises(ValueError) as cm:
            self.build(user)
        msg = str(cm.exception)
        self.assertIn("3 problem(s)", msg)
        for frag in ("unknown key 'losss'", "loss_weight: must be a finite number", "lr_initial: must be > 0.0"):
            self.assertIn(frag, msg)

    def test_unknown_keys_of_a_registered_task_type_come_from_its_registration(self):
        ns = self.ns
        reg = ns["TASK_REGISTRY"]
        ns["register_task_type"]("quantile_x", lambda y, o, t, s: (0.0, {"pinball": 0.0}), ["pinball"], config_keys=["quantiles"])
        try:
            user = good_config()
            user["targets"]["q"] = {"true_key": "y_q", "task_type": "quantile_x", "output_keys": {"q": "y_q"}, "quantiles": [0.1, 0.9]}
            user["callbacks"]["early_stopping"].update(monitor="val_q_pinball")
            self.build(copy.deepcopy(user))
            user["targets"]["q"]["quantilez"] = [0.5]
            with self.assertRaises(ValueError) as cm:
                self.build(copy.deepcopy(user))
            self.assertIn("did you mean 'quantiles'", str(cm.exception))
        finally:
            reg.unregister("quantile_x")
        user = good_config()
        user["targets"]["q"] = {"true_key": "y_q", "task_type": "quantile_x", "output_keys": {"q": "y_q"}}
        with self.assertRaises(ValueError) as cm:
            self.build(user)
        self.assertIn("unregistered task type 'quantile_x'", str(cm.exception))

    def test_builtin_task_types_declare_the_keys_their_loss_reads(self):
        """tasks.py reads each key via target_cfg.get / target_cfg[...]: the declared config_keys must cover them (a missing one
        would make a legitimate key raise as unknown)."""
        src = open(os.path.join(ROOT, "trainer", "tasks.py"), encoding="utf-8").read()
        read = set(re.findall(r"target_cfg\.get\(\"(\w+)\"", src)) | set(re.findall(r"target_cfg\[\"(\w+)\"\]", src))
        read |= {"output_keys"}
        declared = set()
        for entry in self.ns["TASK_REGISTRY"].values():
            declared |= set(entry["config_keys"])
        declared |= {"true_key", "task_type", "output_keys", "loss_weight", "flood_level"}
        self.assertEqual(sorted(read - declared), [])


class ValidatorIsTheOnePlace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = trainer_ns()

    def test_every_default_key_has_a_validation_rule(self):
        schema = self.ns["_CONFIG_SCHEMA"]
        missing = []

        def walk(defaults, rules, path):
            for k, v in defaults.items():
                if k not in rules:
                    missing.append(f"{path}.{k}")
                elif isinstance(v, dict) and isinstance(rules[k], dict):
                    walk(v, rules[k], f"{path}.{k}")
        walk(self.ns["TRAINER_DEFAULT_CONFIG"], schema, "config")
        self.assertEqual(missing, [], "a TRAINER_DEFAULT_CONFIG key without a rule in _CONFIG_SCHEMA is an accepted-but-unchecked key")

    def test_every_rule_has_a_default(self):
        """No rule names a key the defaults do not have, except the declared late-bound ones (class_baselines is added from val)."""
        extra = []

        def walk(rules, defaults, path):
            for k, r in rules.items():
                if k not in defaults:
                    extra.append(f"{path}.{k}")
                elif isinstance(r, dict) and isinstance(defaults[k], dict):
                    walk(r, defaults[k], f"{path}.{k}")
        walk(self.ns["_CONFIG_SCHEMA"], self.ns["TRAINER_DEFAULT_CONFIG"], "config")
        self.assertEqual(sorted(extra), ["config.callbacks.class_baselines", "config.optimizer.lr_schedule.total_epochs"])

    def test_build_config_calls_validate_config_and_nothing_else_validates(self):
        calls = []
        ns = trainer_ns()
        orig = ns["validate_config"]
        ns["validate_config"] = lambda cfg: (calls.append(1), orig(cfg))[1]
        try:
            ns["build_config"](good_config())
        finally:
            ns["validate_config"] = orig
        self.assertEqual(calls, [1])
        body = ast.parse(open(os.path.join(ROOT, "trainer", "config.py"), encoding="utf-8").read())
        fn = next(n for n in body.body if isinstance(n, ast.FunctionDef) and n.name == "build_config")
        raises = [n for n in ast.walk(fn) if isinstance(n, ast.Raise)]
        self.assertEqual(raises, [], "build_config must not carry its own checks: validate_config is the one place")

    def test_the_card_no_longer_overclaims(self):
        doc = open(os.path.join(ROOT, "trainer", "config.py"), encoding="utf-8").read().split('"""')[1]
        pitfalls = doc.split("PITFALLS:")[1].split("\n\n")[0]
        self.assertIn("validate_config", pitfalls)
        self.assertNotIn("Unknown config keys raise in build_config;", pitfalls)

    def test_a_re_built_config_is_valid_and_identical(self):
        """capacity.py / permutation_control.py pass a built config (with class_baselines) through build_config again."""
        ns = trainer_ns()
        user = good_config()
        user["callbacks"]["class_baselines"] = {"a_class": 0.5}
        once = ns["build_config"](copy.deepcopy(user))
        twice = ns["build_config"](copy.deepcopy(once))
        self.assertEqual(ns["_canonical"](once), ns["_canonical"](twice))
        self.assertEqual(ns["config_fingerprint"](once), ns["config_fingerprint"](twice))


# ───────────────────────────── 3) equivalence with the code before validation ─────────────────────────────
def runsettings_user_configs(ns):
    """``[(name, user_config)]``: the dict ``workflow.run.make_training_config`` hands to ``build_config`` for each RunSettings variant
    (default, every training / anti-memorization / target-mode / timeframe knob, the six recorded scenarios of
    tests/golden_run_settings.json), on stub inputs (a val split with +-1 labels, a plan with price targets). Plus the re-built
    config of each (capacity / permutation control feed the built config back) and the capacity-style patch of it."""
    from test_run_steps import settings_for
    S = ns["RunSettings"]
    seen = []
    kit = types.SimpleNamespace(
        build_config=lambda uc: (seen.append(copy.deepcopy(uc)), ns["build_config"](uc))[1], deep_update=ns["deep_update"])
    rng = np.random.default_rng(0)
    P2, P3, P1 = ("high", "low"), ("high", "low", "close"), ("close",)

    def val_for(pts):
        return {"y": {f"y_{t}_class": np.where(rng.random(200) > 0.45, 1.0, -1.0) for t in pts}}

    variants = [
        ("default", {}, P2, 1.0, ("1h",)),
        ("default_three_targets", {}, P3, 1.0, ("1h",)),
        ("default_close_only", {}, P1, 1.0, ("1h",)),
        ("am_off", {"model": {"anti_memorization": False}}, P2, 1.0, ("1h",)),
        ("class_only", {"model": {"class_only": True}}, P2, 1.0, ("1h",)),
        ("class_only_am_off", {"model": {"class_only": True, "anti_memorization": False}}, P3, 1.0, ("1h",)),
        ("patience", {"train": {"early_stopping_patience": 7}}, P2, 1.0, ("1h",)),
        ("train_new", {"train": {"train_mode": "new", "epochs": 3, "batch_size": 32}}, P2, 1.0, ("1h",)),
        ("train_resume", {"train": {"train_mode": "resume"}}, P2, 1.0, ("1h",)),
        ("no_kendall", {"train": {"use_uncertainty_weighting": False}}, P2, 1.0, ("1h",)),
        ("no_kendall_am_off", {"train": {"use_uncertainty_weighting": False}, "model": {"anti_memorization": False}}, P2, 1.0, ("1h",)),
        ("monitor_class_loss", {"train": {"early_stopping_monitor": "val_class_loss"}}, P2, 1.0, ("1h",)),
        ("monitor_val_loss", {"train": {"early_stopping_monitor": "val_loss"}}, P2, 1.0, ("1h",)),
        ("monitor_accuracy_max", {"train": {"early_stopping_monitor": "val_high_class_accuracy", "early_stopping_mode": "max"}}, P2, 1.0, ("1h",)),
        ("lambda_schedules", {"train": {"lambda_reg": {"start": 0.0, "end": 0.2, "warmup_epochs": 9, "schedule": "sqrt"},
                                        "lambda_calib": {"start": 0.01, "end": 0.01, "warmup_epochs": 0, "schedule": "linear"}}}, P2, 1.0, ("1h",)),
        ("label_smoothing_zero", {"train": {"anti_memorization_label_smoothing": 0.0}}, P2, 1.0, ("1h",)),
        ("label_smoothing_03", {"train": {"anti_memorization_label_smoothing": 0.3}}, P2, 1.0, ("1h",)),
        ("anti_memorization_trainer_override", {"train": {"anti_memorization_trainer": {"optimizer": {"lr_initial": 1e-4}}}}, P2, 1.0, ("1h",)),
        ("target_return", {"target": {"target_mode": "return"}}, P2, 0.01, ("1h",)),
        ("target_relative", {"target": {"target_mode": "relative"}}, P2, 1.0, ("1h",)),
        ("target_entry_range", {"target": {"target_mode": "entry_range", "entry_close_reg": "range_pos"}}, P3, 1.0, ("1h",)),
        ("target_entry_range_abs", {"target": {"target_mode": "entry_range", "entry_close_reg": "abs_return"}}, P3, 2.5, ("1h", "4h")),
        ("target_magnitude_two_tf", {"target": {"target_mode": "magnitude"}}, P2, 1.0, ("1h", "4h")),
        ("two_timeframes", {"data": {"model_tfs": ("1h", "4h")}}, P2, 1.0, ("1h", "4h")),
        ("three_timeframes", {"data": {"model_tfs": ("1h", "4h", "1d")}}, P2, 1.0, ("1h", "4h", "1d")),
        ("run_dir_other", {"train": {"run_dir": "/tmp/other_run"}}, P2, 1.0, ("1h",)),
        ("config_overrides", {"model": {"config_overrides": {"d_model": 32}}}, P2, 1.0, ("1h",)),
        ("suspended_targets", {"target": {"suspended_targets": ()}}, P3, 1.0, ("1h",)),
    ]
    gold = json.load(open(os.path.join(ROOT, "tests", "golden_run_settings.json"), encoding="utf-8"))["scenarios"]
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        for name, upd, pts, scale, tfs in variants:
            settings = S().updated(upd)
            plan, info = types.SimpleNamespace(price_targets=pts), types.SimpleNamespace(reg_target_scale=scale, model_tfs=tfs)
            ns["make_training_config"](settings, plan, info, val_for(pts), kit)
            out.append((f"runsettings/{name}", seen[-1]))
        for name, sc in sorted(gold.items()):
            settings = settings_for(ns, sc["scenario"], "/tmp/golden_scenario_run")
            plan = types.SimpleNamespace(price_targets=P2)
            info = types.SimpleNamespace(reg_target_scale=1.0, model_tfs=("1h", "4h") if sc["scenario"].get("two_tf") else ("1h",))
            ns["make_training_config"](settings, plan, info, val_for(P2), kit)
            out.append((f"scenario/{name}", seen[-1]))
        # the built configs fed back through build_config (capacity.py / permutation_control.py): class_baselines is in the config now
        for name, uc in list(out):
            built = ns["build_config"](copy.deepcopy(uc))
            built["callbacks"]["class_baselines"] = {t: 0.5 for t, c in built["targets"].items() if c["task_type"] == "classification"}
            out.append((f"rebuild/{name}", built))
            cap = copy.deepcopy(built)
            cap["run"].update({"run_dir": "/tmp/capacity_x", "mirror_dir": None, "epochs": 4, "train_mode": "new", "seed": 3, "verbose": 0})
            cap["callbacks"]["early_stopping"].update(patience=10 ** 6, restore_best_weights=False)
            out.append((f"capacity_patch/{name}", cap))
    return out


class EquivalenceWithTheCodeBeforeValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = json.load(open(GOLDEN_PATH, encoding="utf-8"))
        cls.ns = light_run_ns()

    def test_the_golden_file_says_where_it_came_from(self):
        self.assertEqual(self.gold["meta"]["recorded_from_commit"], "a248537")
        self.assertGreaterEqual(len(self.gold["fixtures"]), 12)
        self.assertGreaterEqual(len(self.gold["runsettings"]), 100)

    def _assert_same(self, name, user, rec):
        cfg = self.ns["build_config"](copy.deepcopy(user))
        self.assertEqual(canon_sha(self.ns, cfg), rec["sha"], f"{name}: the built config changed")
        self.assertEqual(self.ns["config_fingerprint"](cfg), rec["fingerprint"], f"{name}: the config fingerprint changed")

    @staticmethod
    def _dangling(user):
        """The evidential targets of ``user`` that name a schedule loss.schedules does not define (the old code accepted them and
        silently switched the regularizer / calibration loss off)."""
        sched = user.get("loss", {}).get("schedules", {})
        return [(t, k) for t, tc in user["targets"].items() if tc["task_type"] == "evidential"
                for k, needed in (("lambda_reg_var", True), ("lambda_calib_var", tc.get("use_calibration_loss")))
                if needed and k in tc and tc[k] not in sched]

    def test_every_config_the_repo_builds_still_builds_identically(self):
        """Fixtures: the user configs the repo's tests, self-tests and tools hand to build_config (recorded from a run of the suite).
        A fixture that named a schedule it never defined (workflow/wiring_selftest.py, three test modules; fixed in the commit before
        the validator) must now raise, and its fix - declaring the schedules as constant 0, which is what the missing schedule meant -
        must build exactly the recorded config plus those schedules."""
        n_dangling = 0
        for name, rec in sorted(self.gold["fixtures"].items()):
            with self.subTest(fixture=name):
                user = rec["user"]
                merged = self.ns["deep_update"](self.ns["TRAINER_DEFAULT_CONFIG"], user)
                self.assertEqual(canon_sha(self.ns, merged), rec["sha"], f"{name}: the recorded config is the plain merge")
                if not self._dangling(user):
                    self._assert_same(name, user, rec)
                    continue
                n_dangling += 1
                with self.assertRaises(ValueError) as cm:
                    self.ns["build_config"](copy.deepcopy(user))
                self.assertIn("is not defined in loss.schedules", str(cm.exception))
                zero = {"lambda_reg": {"start": 0.0, "end": 0.0}, "lambda_calib": {"start": 0.0, "end": 0.0}}
                fixed = self.ns["deep_update"](user, {"loss": {"schedules": zero}})
                cfg = self.ns["build_config"](copy.deepcopy(fixed))
                self.assertEqual(self.ns["_canonical"](cfg), self.ns["_canonical"](self.ns["deep_update"](merged, {"loss": {"schedules": zero}})))
        self.assertGreaterEqual(n_dangling, 1)

    def test_every_runsettings_variant_still_builds_identical_configs_and_fingerprints(self):
        cases = dict(runsettings_user_configs(self.ns))
        self.assertEqual(sorted(cases), sorted(self.gold["runsettings"]), "the variant set changed: re-record on the OLD code only")
        for name, user in cases.items():
            with self.subTest(variant=name):
                rec = self.gold["runsettings"][name]
                self.assertEqual(canon_sha(self.ns, user), rec["user_sha"],
                                 f"{name}: the config make_training_config passes to build_config changed")
                self._assert_same(name, user, rec)

    def test_the_trainer_example_config_is_unchanged(self):
        """trainer/example.py builds `real_config` at load time (4 targets, a constraint, three schedules)."""
        ns = _quiet(_nbload.load_trainer)
        rec = self.gold["example"]
        self.assertEqual(canon_sha(ns, ns["real_config"]), rec["sha"])
        self.assertEqual(ns["config_fingerprint"](ns["real_config"]), rec["fingerprint"])


# ───────────────────────────── 4) dead keys ─────────────────────────────
def _pipeline_default_keys():
    tree = ast.parse(open(os.path.join(ROOT, "data", "defaults.py"), encoding="utf-8").read())
    for n in tree.body:
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "PIPELINE_DEFAULT_CONFIG":
            return [k.value for k in n.value.keys]
    raise AssertionError("PIPELINE_DEFAULT_CONFIG not found in data/defaults.py")


def _code_texts(exclude=()):
    texts = {}
    for pkg in CODE_PACKAGES:
        for f in glob.glob(os.path.join(ROOT, pkg, "**", "*.py"), recursive=True):
            rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
            if rel not in exclude:
                texts[rel] = open(f, encoding="utf-8").read()
    return texts


def _read_nowhere(keys, texts):
    return sorted(k for k in keys if not any(re.search(r"""['"]%s['"]""" % re.escape(k), t) for t in texts.values()))


class DeadKeys(unittest.TestCase):
    def test_pipeline_default_keys_read_nowhere_are_frozen(self):
        keys = _pipeline_default_keys()
        self.assertEqual(len(keys), len(set(keys)))
        dead = set(_read_nowhere(keys, _code_texts(exclude=("data/defaults.py",))))
        new = sorted(dead - DEAD_PIPELINE_KEYS)
        self.assertEqual(new, [], f"new dead PIPELINE_DEFAULT_CONFIG key(s) {new}: a config key nothing reads silently does nothing - "
                                  f"read it, or do not add it")
        gone = sorted(DEAD_PIPELINE_KEYS - dead)
        self.assertEqual(gone, [], f"{gone} are no longer dead (read, or removed from the defaults): shrink DEAD_PIPELINE_KEYS")

    def test_every_trainer_default_key_is_read_by_trainer_code(self):
        texts = {k: v for k, v in _code_texts(exclude=("trainer/config.py",)).items() if k.startswith(("trainer/", "workflow/"))}
        keys = set()

        def walk(d):
            for k, v in d.items():
                keys.add(k)
                if isinstance(v, dict) and k not in ("lr_schedule",):
                    walk(v)
        walk(trainer_ns()["TRAINER_DEFAULT_CONFIG"])
        keys -= {"targets", "schedules", "constraints", "epochs", "dir"}           # structural names / read as a different literal
        self.assertEqual(_read_nowhere(sorted(keys), texts), [])


if __name__ == "__main__":
    unittest.main()

"""The loss plan (RunSettings.train.loss) and the selection-metric registry (audit 2026-10-31, classes C1 loss-weight part and C7, PR 9).

Owner's example: "changing the loss weights of training is fragile". Before this PR a weight needed the free-form
`anti_memorization_trainer` dict (merged ONLY when anti_memorization was on) or an ad-hoc config edit, and the best-epoch metric
`val_raw_loss` was the training objective itself, so a weight change also changed which checkpoint is "best".

Guards:
1. VOCABULARY: LOSS_BALANCERS holds exactly what the trainer implements today ("static", "kendall"); SELECTION_METRICS holds the
   metrics the trainer already logs, each saying whether it depends on the objective; unknown names raise (with the known ones).
2. DEFAULTS REPRODUCE TODAY: RunSettings() has the old values (balancing kendall = use_uncertainty_weighting True, the two lambda
   schedules, monitor val_raw_loss / min) and builds a trainer config without any loss_weight (the byte-identical goldens are
   tests/golden_trainer_configs.json, golden_run_settings.json, golden_head_vocabulary.json, run by their own tests).
3. FAIL EARLY: unknown head / balancer / selection metric, wrong mode, bad weight, missing or bad schedule, the old setting names,
   and loss / monitor smuggled through anti_memorization_trainer all raise at settings construction (heads of a smaller plan, e.g. a
   suspended target, raise at make_training_config).
4. ONE ROUTE: changing ONE head's weight in the plan changes exactly that head's `loss_weight` in the trainer config and nothing
   else (the selection metric definition included), with anti_memorization on AND off (it used to be dropped silently off);
   balancing and schedules likewise change only their own keys.
5. SELECTION IS OBJECTIVE-INDEPENDENT WHERE DECLARED: on a real GenericTrainer step with fixed model outputs, changing loss weights or
   the lambda schedule moves `raw_loss` but not val accuracy / mae / nig_base (the metrics registered objective_independent=True).
6. IDENTITY: a resumed run whose loss plan changed raises RunIdentityMismatch naming the key (PR 4 machinery).
    python -m pytest tests/test_loss_plan.py -q
"""
import contextlib
import copy
import io
import os
import shutil
import sys
import tempfile
import types
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

import _nbload  # noqa: E402
from core.heads import HeadSet  # noqa: E402
from core.loss_plan import (LOSS_BALANCERS, SELECTION_METRICS, SCHEDULE_CURVES, check_loss_schedules,  # noqa: E402
                            resolve_selection_metric, trainer_loss_section)
from core.registry import UnknownNameError  # noqa: E402

P2, P3 = ("high", "low"), ("high", "low", "close")
_NS = {}


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def ns():
    """Trainer + the workflow modules that build the training config (no pipeline, no model)."""
    if "ns" not in _NS:
        t = _quiet(_nbload.load_trainer)
        _quiet(_nbload.workflow_package().load_into, t, only=("settings", "training_config", "run"))
        _NS["ns"] = t
    return _NS["ns"]


def make_config(settings, price_targets=P2):
    """The trainer config make_training_config builds for ``settings`` (stub val split with +-1 labels)."""
    n = ns()
    rng = np.random.default_rng(0)
    val = {"y": {f"y_{t}_class": np.where(rng.random(100) > 0.45, 1.0, -1.0) for t in price_targets}}
    kit = types.SimpleNamespace(build_config=n["build_config"], deep_update=n["deep_update"])
    plan = types.SimpleNamespace(price_targets=tuple(price_targets))
    info = types.SimpleNamespace(reg_target_scale=1.0, model_tfs=("1h",))
    return _quiet(n["make_training_config"], settings, plan, info, val, kit)


def flat(o, prefix=""):
    if isinstance(o, dict) and o:
        for k in sorted(o, key=str):
            yield from flat(o[k], f"{prefix}.{k}" if prefix else str(k))
    else:
        yield prefix, o


def diff(a, b):
    """``{dotted key: (a, b)}`` of the keys whose value differs (a missing key reads as None)."""
    fa, fb = dict(flat(a)), dict(flat(b))
    return {k: (fa.get(k), fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)}


def S(changes=None):
    return ns()["RunSettings"]().updated(changes or {})


def plan(**kw):
    """A train.loss change as ``RunSettings.updated`` input (replaces the plan, missing parts keep their defaults)."""
    return {"train": {"loss": kw}}


class Vocabulary(unittest.TestCase):
    def test_balancers_are_what_the_trainer_implements_today(self):
        self.assertEqual(LOSS_BALANCERS.names(), ("static", "kendall"))
        self.assertEqual(LOSS_BALANCERS.get("static")(None), {"use_uncertainty_weighting": False})
        self.assertEqual(LOSS_BALANCERS.get("kendall")(None), {"use_uncertainty_weighting": True})
        self.assertEqual(LOSS_BALANCERS.check_all(), [])
        with self.assertRaises(UnknownNameError) as cm:
            LOSS_BALANCERS.get("gradnorm")
        self.assertIn("kendall", str(cm.exception))

    def test_selection_metrics_default_and_independence_flags(self):
        self.assertEqual(SELECTION_METRICS.names(), ("raw_loss", "loss", "class_loss", "accuracy", "mae", "nig_base", "ic"))
        self.assertEqual(SELECTION_METRICS.check_all(), [])
        meta = {e.name: e.meta for e in SELECTION_METRICS.entries()}
        for objective in ("raw_loss", "loss", "class_loss"):
            self.assertFalse(meta[objective]["objective_independent"], objective)
        for free in ("accuracy", "mae", "nig_base", "ic"):
            self.assertTrue(meta[free]["objective_independent"], free)
        self.assertEqual((meta["accuracy"]["mode"], meta["mae"]["mode"], meta["nig_base"]["mode"], meta["raw_loss"]["mode"]),
                         ("max", "min", "min", "min"))

    def test_registered_balancer_must_honour_its_contract(self):
        with self.assertRaises(ValueError):
            LOSS_BALANCERS.register("bad", lambda heads: {"learning_rate": 1}, version=1, description="x")
        self.assertNotIn("bad", LOSS_BALANCERS)

    def test_curve_names_have_one_owner(self):
        self.assertEqual(tuple(ns()["_SCHEDULE_NAMES"]), SCHEDULE_CURVES)

    def test_resolve_selection_metric(self):
        heads = HeadSet.default(P3)
        c = resolve_selection_metric("val_raw_loss", heads, "min")
        self.assertEqual((c.monitor, c.stat, c.head, c.mode, c.objective_independent), ("val_raw_loss", "raw_loss", None, "min", False))
        c = resolve_selection_metric("val_close_class_accuracy", heads)
        self.assertEqual((c.stat, c.head, c.mode, c.objective_independent), ("accuracy", "close_class", "max", True))
        c = resolve_selection_metric("val_high_reg_nig_base", heads)
        self.assertEqual((c.stat, c.head), ("nig_base", "high_reg"))
        for bad, frag in [("raw_loss", "must be 'val_<metric>'"), ("val_rawloss", "unknown selection metric"),
                          ("val_nope_class_accuracy", "unknown head 'nope_class'"), ("val_high_reg_accuracy", "defined for task type"),
                          ("val_high_class_nig_base", "defined for task type")]:
            with self.assertRaises(ValueError, msg=bad) as cm:
                resolve_selection_metric(bad, heads)
            self.assertIn(frag, str(cm.exception), bad)
        with self.assertRaises(ValueError) as cm:
            resolve_selection_metric("val_close_class_accuracy", heads, "min")
        self.assertIn("must be 'max'", str(cm.exception))
        with self.assertRaises(ValueError):          # class_loss needs a classification head; a reg-only set has none
            resolve_selection_metric("val_class_loss", HeadSet.default(P2).select(task_type="evidential"))


class DefaultsReproduceToday(unittest.TestCase):
    def test_default_plan_values(self):
        s = S().train
        self.assertEqual(s.loss.weights, {})
        self.assertEqual(s.loss.balancing, "kendall")
        self.assertEqual(s.loss.schedules, {"lambda_reg": {"start": 0.0, "end": 0.05, "warmup_epochs": 5, "schedule": "linear"},
                                            "lambda_calib": {"start": 0.0, "end": 0.1, "warmup_epochs": 5, "schedule": "cosine"}})
        self.assertEqual((s.early_stopping_monitor, s.early_stopping_mode), ("val_raw_loss", "min"))

    def test_default_config_has_no_loss_weight_and_the_old_loss_and_monitor(self):
        for am in (True, False):
            cfg = make_config(S({"model": {"anti_memorization": am}}), P3)
            self.assertFalse([t for t, c in cfg["targets"].items() if "loss_weight" in c], am)
            self.assertEqual(cfg["loss"]["use_uncertainty_weighting"], True)
            self.assertEqual(cfg["loss"]["schedules"], S().train.loss.schedules)
            es = cfg["callbacks"]["early_stopping"]
            self.assertEqual((es["monitor"], es["mode"]), ("val_raw_loss", "min"))

    def test_trainer_loss_section_of_the_default_plan(self):
        sec = trainer_loss_section("kendall", S().train.loss.schedules)
        self.assertEqual(list(sec), ["use_uncertainty_weighting", "schedules"])
        self.assertEqual(sec["use_uncertainty_weighting"], True)

    def test_plan_is_frozen_copied_and_roundtrips(self):
        s = S(plan(weights={"high_reg": 2.0}))
        with self.assertRaises(Exception):
            s.train.loss.balancing = "static"
        d = s.to_dict()
        self.assertEqual(ns()["RunSettings"].from_dict(d), s)
        w = {"high_class": 1.5}
        loss = ns()["LossSettings"](weights=w)
        w["high_class"] = 9.0
        self.assertEqual(loss.weights, {"high_class": 1.5})              # copied on construction
        self.assertEqual(S().train.loss.updated(balancing="static").balancing, "static")


class FailEarly(unittest.TestCase):
    def assertRaisesWith(self, fragment, changes):
        with self.assertRaises(ValueError) as cm:
            S(changes)
        self.assertIn(fragment, str(cm.exception))

    def test_unknown_head_balancer_metric(self):
        self.assertRaisesWith("unknown head 'high_clas'", plan(weights={"high_clas": 2.0}))
        self.assertRaisesWith("known heads: ['high_reg', 'high_class'", plan(weights={"nope": 1.0}))
        self.assertRaisesWith("unknown loss balancer 'gradnorm'", plan(balancing="gradnorm"))
        self.assertRaisesWith("unknown selection metric 'val_aucc'", {"train": {"early_stopping_monitor": "val_aucc"}})
        self.assertRaisesWith("unknown head", {"train": {"early_stopping_monitor": "val_nope_class_accuracy", "early_stopping_mode": "max"}})

    def test_heads_of_class_only_are_checked_against_the_run(self):
        self.assertRaisesWith("unknown head 'high_reg'", {"model": {"class_only": True}, **plan(weights={"high_reg": 2.0})})
        S({"model": {"class_only": True}, **plan(weights={"high_class": 2.0})})              # fine
        self.assertRaisesWith("unknown head 'high_reg'",
                              {"model": {"class_only": True}, "train": {"early_stopping_monitor": "val_high_reg_nig_base"}})

    def test_wrong_mode_for_the_metric(self):
        self.assertRaisesWith("must be 'max'", {"train": {"early_stopping_monitor": "val_high_class_accuracy"}})
        self.assertRaisesWith("must be 'min'", {"train": {"early_stopping_mode": "max"}})
        S({"train": {"early_stopping_monitor": "val_high_class_accuracy", "early_stopping_mode": "max"}})

    def test_bad_weights_and_schedules(self):
        for bad in (-1.0, float("nan"), float("inf"), "2", True, None):
            self.assertRaisesWith("finite number >= 0", plan(weights={"high_reg": bad}))
        self.assertRaisesWith("must be a dict", plan(weights=[("high_reg", 2.0)]))
        self.assertRaisesWith("missing sub-term(s) ['lambda_calib']",
                              plan(schedules={"lambda_reg": {"start": 0.0, "end": 0.0}}))
        self.assertRaisesWith("unknown sub-term(s) ['lambda_other']",
                              plan(schedules={**S().train.loss.schedules, "lambda_other": {"start": 0.0, "end": 0.0}}))
        sch = copy.deepcopy(S().train.loss.schedules)
        sch["lambda_reg"]["schedule"] = "cosin"
        self.assertRaisesWith("schedule must be one of", plan(schedules=sch))
        sch["lambda_reg"] = {"start": 0.0, "end": float("nan")}
        self.assertRaisesWith("lambda_reg.end must be a finite number", plan(schedules=sch))
        sch["lambda_reg"] = {"start": 0.0}
        self.assertRaisesWith("required key 'end' is missing", plan(schedules=sch))
        sch["lambda_reg"] = {"start": 0.0, "end": 0.0, "warmup": 3}
        self.assertRaisesWith("unknown key(s) ['warmup']", plan(schedules=sch))
        with self.assertRaises(ValueError):
            check_loss_schedules({})

    def test_unknown_plan_field_and_old_names_say_where_they_moved(self):
        self.assertRaisesWith("unknown setting(s) ['wieghts']", plan(wieghts={}))
        for old, new in (("use_uncertainty_weighting", "train.loss.balancing"), ("lambda_reg", "train.loss.schedules['lambda_reg']"),
                         ("lambda_calib", "train.loss.schedules['lambda_calib']")):
            self.assertRaisesWith(f"{old!r} moved to {new}", {"train": {old: False}})

    def test_loss_and_monitor_cannot_travel_through_anti_memorization_trainer(self):
        self.assertRaisesWith("may not carry 'targets' / 'loss'", {"train": {"anti_memorization_trainer": {"loss": {"use_uncertainty_weighting": False}}}})
        self.assertRaisesWith("may not carry 'targets' / 'loss'", {"train": {"anti_memorization_trainer": {"targets": {"high_reg": {"loss_weight": 3}}}}})
        self.assertRaisesWith("callbacks.early_stopping.monitor / mode",
                              {"train": {"anti_memorization_trainer": {"callbacks": {"early_stopping": {"monitor": "val_loss"}}}}})
        S({"train": {"anti_memorization_trainer": {"callbacks": {"early_stopping": {"weights_snapshot": "raw"}}}}})   # other keys still fine

    def test_a_weight_for_a_head_the_run_does_not_train_raises_at_the_steps(self):
        s = S(plan(weights={"close_class": 2.0}))           # valid for the settings: close_class is a head of the vocabulary
        with self.assertRaises(ValueError) as cm:           # but the plan of this run trains high / low only
            make_config(s, P2)
        self.assertIn("unknown head 'close_class'", str(cm.exception))
        self.assertEqual(make_config(s, P3)["targets"]["close_class"]["loss_weight"], 2.0)


class OneRoute(unittest.TestCase):
    def test_changing_one_head_weight_changes_only_that_loss_weight(self):
        for am in (True, False):                              # the old route dropped an override silently when anti_memorization was off
            base = make_config(S({"model": {"anti_memorization": am}}), P3)
            for head in ("high_reg", "low_class", "close_class"):
                cfg = make_config(S({"model": {"anti_memorization": am}, **plan(weights={head: 3.0})}), P3)
                self.assertEqual(diff(base, cfg), {f"targets.{head}.loss_weight": (None, 3.0)}, (am, head))
                self.assertEqual(base["callbacks"]["early_stopping"], cfg["callbacks"]["early_stopping"])    # selection metric untouched
                self.assertEqual(base["loss"], cfg["loss"])

    def test_two_weights_and_a_weight_equal_to_the_default(self):
        base = make_config(S(), P2)
        cfg = make_config(S(plan(weights={"high_reg": 0.5, "low_class": 2.0, "high_class": 1.0})), P2)
        self.assertEqual(diff(base, cfg), {"targets.high_reg.loss_weight": (None, 0.5), "targets.low_class.loss_weight": (None, 2.0)})

    def test_balancing_and_schedules_change_only_their_own_keys(self):
        base = make_config(S(), P2)
        self.assertEqual(diff(base, make_config(S(plan(balancing="static")), P2)),
                         {"loss.use_uncertainty_weighting": (True, False)})
        sch = copy.deepcopy(S().train.loss.schedules)
        sch["lambda_reg"]["end"] = 0.2
        self.assertEqual(diff(base, make_config(S(plan(schedules=sch)), P2)), {"loss.schedules.lambda_reg.end": (0.05, 0.2)})

    def test_selection_metric_by_name_reaches_the_monitor_and_nothing_else(self):
        base = make_config(S(), P2)
        cfg = make_config(S({"train": {"early_stopping_monitor": "val_high_class_accuracy", "early_stopping_mode": "max"}}), P2)
        self.assertEqual(diff(base, cfg), {"callbacks.early_stopping.monitor": ("val_raw_loss", "val_high_class_accuracy"),
                                           "callbacks.early_stopping.mode": ("min", "max")})
        with self.assertRaises(ValueError):                  # a metric of a head the run does not train
            make_config(S({"train": {"early_stopping_monitor": "val_close_class_accuracy", "early_stopping_mode": "max"}}), P2)

    def test_every_registered_metric_is_a_monitor_the_trainer_accepts(self):
        """The registry and trainer.validate_config agree: every (metric, head) pair the registry resolves is a monitor build_config
        accepts with the metric's own mode (a new metric that the trainer does not log fails here)."""
        n = ns()
        for class_only in (False, True):
            heads = HeadSet.default(P3, class_only=class_only)
            names = []
            for e in SELECTION_METRICS.entries():
                if e.meta["scope"] == "global":
                    names.append(f"val_{e.name}")
                else:
                    names += [f"val_{h.name}_{e.name}" for h in heads if h.task_type in e.meta["task_types"]]
            self.assertTrue(any(x.endswith("accuracy") for x in names))
            for monitor in names:
                if monitor == "val_class_loss" and not any(h.task_type == "classification" for h in heads):
                    continue
                choice = resolve_selection_metric(monitor, heads)
                settings = S({"model": {"class_only": class_only},
                              "train": {"early_stopping_monitor": monitor, "early_stopping_mode": choice.mode}})
                cfg = make_config(settings, P3)
                self.assertEqual(cfg["callbacks"]["early_stopping"]["monitor"], monitor)
                self.assertEqual(n["_expected_metric_names"](cfg) >= {monitor[4:]}, True, monitor)


def _tracked(targets):
    """targets with the val-set IC tracker on every regression head (track_ic), so val_<head>_ic is logged too."""
    for cfg in targets.values():
        if cfg["task_type"] in SELECTION_METRICS.entry("ic").meta["task_types"]:
            cfg["track_ic"] = True
    return targets


class ObjectiveIndependence(unittest.TestCase):
    """Real GenericTrainer step: a loss weight / lambda moves raw_loss, never the registered objective-independent metrics."""

    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf
        n = ns()
        cls.heads = HeadSet.default(("high",))
        keys = {}
        for h in cls.heads:
            keys.update({k: None for k in h.output_names})
        inp = tf.keras.Input(shape=(6,), name="x")
        hidden = tf.keras.layers.Dense(8, activation="tanh")(inp)
        outs = {}
        for name in sorted(keys):
            if name.endswith(("_nu", "_alpha", "_beta")):
                outs[name] = tf.keras.layers.Dense(1, activation="softplus", name=name)(hidden) + 1.0
            elif name.endswith("_confidence") or name.endswith("_class_logits"):
                outs[name] = tf.keras.layers.Dense(1, activation="sigmoid", name=name)(hidden)
            else:
                outs[name] = tf.keras.layers.Dense(1, name=name)(hidden)
        tf.keras.utils.set_random_seed(7)
        cls.model = tf.keras.Model(inp, outs)
        rng = np.random.default_rng(3)
        cls.x = rng.normal(size=(64, 6)).astype("float32")
        cls.y = {"y_high_reg": rng.normal(size=(64, 1)).astype("float32"),
                 "y_high_class": rng.integers(0, 2, size=(64, 1)).astype("float32")}
        cls.n = n

    def metrics(self, weights, lambda_reg_end=0.0, balancing="static"):
        n = self.n
        heads = self.heads.with_loss_weights(weights) if weights else self.heads
        sched = {"lambda_reg": {"start": lambda_reg_end, "end": lambda_reg_end, "warmup_epochs": 0, "schedule": "linear"},
                 "lambda_calib": {"start": 0.0, "end": 0.0, "warmup_epochs": 0, "schedule": "linear"}}
        cfg = n["build_config"]({"run": {"run_dir": "/tmp/loss_plan_unused", "epochs": 1, "batch_size": 64, "verbose": 0},
                                 "targets": _tracked(heads.target_configs()), "loss": trainer_loss_section(balancing, sched, heads)})
        trainer = n["GenericTrainer"](self.model, cfg)
        trainer.scheduled_vars["lambda_reg"].assign(lambda_reg_end)
        res = trainer.test_step((self.x, self.y))
        return {k: float(v) for k, v in res.items()}

    def test_weights_and_lambda_move_raw_loss_but_not_the_independent_metrics(self):
        base = self.metrics({})
        for label, other in (("weights", self.metrics({"high_class": 4.0, "high_reg": 0.25})),
                             ("lambda", self.metrics({}, lambda_reg_end=0.5))):
            self.assertNotEqual(base["raw_loss"], other["raw_loss"], label)
            for stat in ("high_class_accuracy", "high_reg_mae", "high_reg_nig_base", "high_reg_ic"):
                self.assertEqual(base[stat], other[stat], f"{label}: {stat} must not depend on the objective")
        keys = set(base)
        for e in SELECTION_METRICS.entries():
            if e.meta["scope"] == "head":
                self.assertTrue(any(k.endswith("_" + e.name) for k in keys), f"the trainer logs no {e.name}")

    def test_kendall_moves_the_total_loss_not_raw_loss_nor_the_independent_metrics(self):
        base, kendall = self.metrics({}), self.metrics({}, balancing="kendall")
        self.assertNotEqual(base["loss"], kendall["loss"])
        for stat in ("raw_loss", "high_class_accuracy", "high_reg_mae", "high_reg_nig_base"):
            self.assertEqual(base[stat], kendall[stat], stat)

    def test_the_default_selection_metric_does_depend_on_the_objective(self):
        a, b = self.metrics({}), self.metrics({"high_reg": 3.0})
        self.assertNotEqual(a["raw_loss"], b["raw_loss"])
        self.assertFalse(SELECTION_METRICS.entry("raw_loss").meta["objective_independent"])


class RunIdentityTracksThePlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="loss_plan_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_resume_with_a_changed_plan_raises_and_names_the_key(self):
        n = ns()
        run_dir = os.path.join(self.tmp, "run")

        def identity(changes):
            cfg = make_config(S(changes), P2)
            cfg["run"]["run_dir"] = run_dir
            return n["collect_run_identity"](cfg)

        n["write_run_identity"](run_dir, identity({}))
        self.assertEqual(n["check_run_identity"](run_dir, identity({})), "match")
        sch = copy.deepcopy(S().train.loss.schedules)
        sch["lambda_reg"]["end"] = 0.2
        cases = {"weight": (plan(weights={"high_class": 2.0}), "targets.high_class.loss_weight: added (2.0)"),
                 "balancing": (plan(balancing="static"), "loss.use_uncertainty_weighting: True -> False"),
                 "schedule": (plan(schedules=sch), "loss.schedules.lambda_reg.end: 0.05 -> 0.2"),
                 "selection": ({"train": {"early_stopping_monitor": "val_high_class_accuracy", "early_stopping_mode": "max"}},
                               "callbacks.early_stopping.monitor: 'val_raw_loss' -> 'val_high_class_accuracy'")}
        for name, (changes, line) in cases.items():
            with self.assertRaises(n["RunIdentityMismatch"], msg=name) as cm:
                n["check_run_identity"](run_dir, identity(changes))
            self.assertIn(line, str(cm.exception), name)


if __name__ == "__main__":
    unittest.main()

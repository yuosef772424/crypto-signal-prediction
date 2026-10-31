"""The head vocabulary (audit C1, PR 7): one declaration (core/heads.py HeadSet) is the source of the trainer `targets`, the model
`head_types`, the model output names and the pipeline head names.

Oracles:
 * tests/golden_head_vocabulary.json holds, for every RunSettings variant, the sha of the canonical trainer `targets`, the model
   `head_types` plan_model hands the builder and the output names the built model really has. It was RECORDED FROM THE CODE BEFORE
   core/heads.py existed (commit 6c30e22, `python tests/test_head_vocabulary.py --record` on that code); never re-record on code that
   uses HeadSet. The one trained epoch of the `return_trained_holdout` scenario is recorded too (same seeds, same machine: the code
   after the refactor reproduces it exactly; the test allows 1% for other CPUs).
 * round trip: HeadSet -> target configs -> HeadSet is the identity; HeadSet -> model output names == the built model's.
 * the HeadSet validation (unique names / output names / labels, unknown kind, unknown role, bad weight) raises.
    python -m pytest tests/test_head_vocabulary.py -q
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "docs", "research", "audit")):
    sys.path.insert(0, _p)
import numpy as np  # noqa: E402

import run_fixture as rf  # noqa: E402
import _nbload  # noqa: E402

GOLDEN_PATH = os.path.join(ROOT, "tests", "golden_head_vocabulary.json")
RECORDED_FROM = "6c30e22"
ALL3 = ["high", "low", "close"]
_NS = {}


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def run_ns():
    if "run" not in _NS:
        _NS["run"] = rf.run_namespace()
    return _NS["run"]


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=list).encode("utf-8")).hexdigest()


def variants():
    """``(named, scenarios)``: every knob that changes the targets / heads, and the six recorded scenarios."""
    out = [
        ("default", {}),
        ("am_off", {"model": {"anti_memorization": False}}),
        ("class_only", {"model": {"class_only": True}}),
        ("class_only_am_off", {"model": {"class_only": True, "anti_memorization": False}}),
        ("label_smoothing_zero", {"train": {"anti_memorization_label_smoothing": 0.0}}),
        ("label_smoothing_03", {"train": {"anti_memorization_label_smoothing": 0.3}}),
        ("no_kendall", {"train": {"loss": {"balancing": "static"}}}),
        ("lambda_schedules", {"train": {"loss": {"schedules": {
            "lambda_reg": {"start": 0.0, "end": 0.2, "warmup_epochs": 9, "schedule": "sqrt"},
            "lambda_calib": {"start": 0.01, "end": 0.01, "warmup_epochs": 0, "schedule": "linear"}}}}}),
        ("anti_memorization_trainer_override", {"train": {"anti_memorization_trainer": {"optimizer": {"lr_initial": 1e-4}}}}),
        ("target_return", {"target": {"target_mode": "return"}}),
        ("target_relative", {"target": {"target_mode": "relative"}}),
        ("target_magnitude", {"target": {"target_mode": "magnitude"}}),
        ("target_entry_range", {"target": {"target_mode": "entry_range", "entry_close_reg": "range_pos"}}),
        ("target_entry_range_class_only", {"target": {"target_mode": "entry_range", "entry_close_reg": "range_pos"},
                                           "model": {"class_only": True}}),
        ("target_entry_range_abs", {"target": {"target_mode": "entry_range", "entry_close_reg": "abs_return"}}),
        ("suspended_nothing", {"target": {"suspended_targets": ()}}),
        ("suspended_nothing_class_only", {"target": {"suspended_targets": ()}, "model": {"class_only": True}}),
        ("suspended_high", {"target": {"suspended_targets": ("high",)}}),
        ("suspended_high_low", {"target": {"suspended_targets": ("high", "low")}}),
        ("two_timeframes", {"data": {"model_tfs": ("1h", "4h")}}),
        ("config_overrides", {"model": {"config_overrides": {"d_model": 32}}}),
    ]
    gold = json.load(open(os.path.join(ROOT, "tests", "golden_run_settings.json"), encoding="utf-8"))["scenarios"]
    return out, gold


def all_cases(ns):
    from test_run_steps import settings_for
    named, gold = variants()
    cases = {name: ns["RunSettings"]().updated(upd) for name, upd in named}
    for name, sc in sorted(gold.items()):
        cases[f"scenario/{name}"] = settings_for(ns, sc, "/tmp/golden_scenario_run")
    return cases


def _stub_inputs():
    """plan_model needs a dataset (window sizes, feature order) and a kit (CONFIG targets, model config names)."""
    ds = {"window_sizes": {"1h": 16, "4h": 8}, "feature_order": ["f0", "f1", "f2"]}
    kit = types.SimpleNamespace(config={"targets": list(ALL3)}, model_config={"d_model": 128, "encoder": "transformer"},
                                anti_memorization_config={"d_model": 64, "encoder": "gru"})
    return ds, kit


def head_vocabulary_of(ns, settings):
    """``(targets_sha, head_types, price_targets)``: what plan_model + make_training_config derive from ``settings``."""
    ds, kit = _stub_inputs()
    info = types.SimpleNamespace(model_tf="1h", model_tfs=tuple(settings.data.model_tfs or ("1h",)), reg_target_scale=1.0)
    rng = np.random.default_rng(0)
    plan = _quiet(ns["plan_model"], settings, ds, info, kit)
    val = {"y": {f"y_{t}_class": np.where(rng.random(200) > 0.45, 1.0, -1.0) for t in plan.price_targets}}
    seen = []
    kit2 = types.SimpleNamespace(build_config=lambda uc: (seen.append(copy.deepcopy(uc)), ns["build_config"](uc))[1],
                                 deep_update=ns["deep_update"])
    _quiet(ns["make_training_config"], settings, plan, info, val, kit2)
    targets = seen[-1]["targets"]
    return sha(ns["_canonical"](targets)), rf.jsonable(plan.model_overrides["head_types"]), list(plan.price_targets)


def built_output_names(ns, price_targets, head_types):
    """The output names of a real (tiny) model built for these heads, in the builder's order."""
    cfg = dict(price_targets=tuple(price_targets), head_types=head_types, d_model=16, num_layers=1, head_hidden=16,
               class_head_hidden=8, encoder="gru", num_heads=2, num_kv_heads=1, dropout=0.1)
    return list(ns["build_model_fn"](16, 8, config=cfg).output)


def combo_key(price_targets, head_types):
    return json.dumps([list(price_targets), head_types], sort_keys=True)


def record():
    """Record the golden from the CURRENT code (run it on the code before core/heads.py only)."""
    from test_run_steps import _dataset, settings_for
    ns = run_ns()
    rec, combos = {}, {}
    for name, settings in all_cases(ns).items():
        t_sha, head_types, pts = head_vocabulary_of(ns, settings)
        key = combo_key(pts, head_types)
        if key not in combos:
            combos[key] = built_output_names(ns, pts, head_types)
        rec[name] = {"targets_sha": t_sha, "head_types": head_types, "price_targets": pts, "combo": key}
    work = tempfile.mkdtemp(prefix="head_vocab_")
    gold = variants()[1]
    sc = gold["return_trained_holdout"]["scenario"]
    settings = settings_for(ns, sc, os.path.join(work, "run"))
    kit = ns["Toolkit"].from_namespace(ns)
    res = _quiet(ns["run_main"], settings, kit, dataset=_dataset(sc["two_tf"], sc["holdout"]), summary=False)
    hist = {k: [float(x) for x in v] for k, v in res.history.history.items()}
    out = {"meta": {"recorded_from_commit": RECORDED_FROM,
                    "note": "Recorded from the code BEFORE core/heads.py existed. Never re-record on code that uses HeadSet."},
           "variants": rec, "output_names_by_combo": combos, "one_epoch_history": hist}
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print("recorded", len(rec), "variants,", len(combos), "model combos")



# ───────────────────────────── HeadSet units (pure, no TensorFlow) ─────────────────────────────
class HeadSetUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import core.heads as ch
        cls.ch = ch

    def test_default_set_is_todays_heads_targets_and_outputs(self):
        ch = self.ch
        hs = ch.HeadSet.default(("high", "low"), label_smoothing=0.1)
        self.assertEqual(hs.names, ("high_reg", "high_class", "low_reg", "low_class"))
        self.assertEqual(hs.head_types(), {t: ["nig_regression", "binary_classification"] for t in ("high", "low")})
        self.assertEqual(hs.targets, ("high", "low"))
        cfg = hs.target_configs()
        self.assertEqual(cfg["high_reg"], {
            "true_key": "y_high_reg", "task_type": "evidential",
            "output_keys": {"mu": "y_high", "nu": "y_high_nu", "alpha": "y_high_alpha", "beta": "y_high_beta",
                            "confidence": "y_high_confidence"},
            "use_calibration_loss": True, "lambda_reg_var": "lambda_reg", "lambda_calib_var": "lambda_calib"})
        self.assertEqual(cfg["low_class"], {
            "true_key": "y_low_class", "task_type": "classification", "binary": True,
            "output_keys": {"logits": "y_low_class_logits"}, "label_smoothing": 0.1})
        self.assertEqual(sorted(hs.output_names()),
                         sorted(["y_%s%s" % (t, s) for t in ("high", "low")
                                 for s in ("", "_nu", "_alpha", "_beta", "_epistemic", "_aleatoric", "_confidence", "_class_logits")]))
        self.assertEqual(hs.label_keys(), ("y_high_reg", "y_high_class", "y_low_reg", "y_low_class"))

    def test_class_only_keeps_only_the_classification_heads(self):
        hs = self.ch.HeadSet.default(("high", "low", "close"), class_only=True)
        self.assertEqual(hs.names, ("high_class", "low_class", "close_class"))
        self.assertEqual(hs.head_types(), {t: ["binary_classification"] for t in ("high", "low", "close")})

    def test_round_trip_headset_to_target_configs_and_back(self):
        ch = self.ch
        for hs in (ch.HeadSet.default(("high", "low", "close"), label_smoothing=0.3),
                   ch.HeadSet.default(("close",), class_only=True),
                   ch.HeadSet.default(("high", "low")).with_loss_weights({"high_class": 0.5, "low_reg": 2.0}),
                   ch.HeadSet.from_head_types({"high": ["nig_regression", "multiclass_classification"]})):
            with self.subTest(names=hs.names):
                back = ch.HeadSet.from_target_configs(hs.target_configs())
                self.assertEqual(back, hs)
                self.assertEqual(back.target_configs(), hs.target_configs())
                self.assertEqual(back.head_types(), hs.head_types())
                self.assertEqual(back.output_names(), hs.output_names())

    def test_a_weight_equal_to_the_trainer_default_is_not_written(self):
        hs = self.ch.HeadSet.default(("high",))
        self.assertNotIn("loss_weight", hs.target_configs()["high_reg"])
        hw = hs.with_loss_weights({"high_reg": 0.25})
        self.assertEqual(hw.target_configs()["high_reg"]["loss_weight"], 0.25)
        self.assertEqual(hw.loss_weights(), {"high_reg": 0.25, "high_class": 1.0})

    def test_validation_raises(self):
        ch = self.ch
        mk = ch.HeadSpec.make
        a = mk("high", "nig_regression")
        with self.assertRaisesRegex(ValueError, "duplicate name"):
            ch.HeadSet((a, a))
        with self.assertRaisesRegex(ValueError, "at least one head"):
            ch.HeadSet(())
        # two classification heads of one target share the output `y_high_class_logits`
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ch.HeadSet((mk("high", "binary_classification"), mk("high", "multiclass_classification")))
        with self.assertRaises(KeyError):             # UnknownNameError: known kinds + did-you-mean
            mk("high", "nig_regresion")
        with self.assertRaisesRegex(ValueError, "unknown option"):
            mk("high", "nig_regression", label_smoothing=0.1)
        for bad in (-1.0, float("nan"), float("inf"), "1", True, None):
            with self.subTest(weight=bad), self.assertRaisesRegex(ValueError, "loss_weight"):
                mk("high", "binary_classification", loss_weight=bad)
        with self.assertRaisesRegex(ValueError, "output_keys"):          # a hand-built spec must match its head kind
            ch.HeadSpec(**{**a.__dict__, "output_keys": {"mu": "y_wrong"}})
        with self.assertRaisesRegex(ValueError, "unknown head"):
            ch.HeadSet.default(("high",)).with_loss_weights({"nope": 1.0})
        with self.assertRaisesRegex(ValueError, "not what a HeadSet produces"):
            ch.HeadSet.from_target_configs({"high_class": {**ch.HeadSet.default(("high",)).target_configs()["high_class"],
                                                           "flood_level": 0.1}})
        with self.assertRaisesRegex(ValueError, "head kinds match"):
            ch.HeadSet.from_target_configs({"high_x": {"true_key": "y_high_x", "task_type": "quantile", "output_keys": {}}})

    def test_kind_contract_and_registry(self):
        ch = self.ch
        with self.assertRaises(ValueError):           # duplicate registration
            ch.register_head_kind(ch.HEAD_KINDS["nig_regression"])
        with self.assertRaisesRegex(ValueError, "role"):
            ch.HeadKind("x", "x", "t", "oops", "t", {"mu": ""}, ("mu",))
        with self.assertRaisesRegex(ValueError, "loss_outputs"):
            ch.HeadKind("x", "x", "t", "regression", "t", {"mu": ""}, ("nu",))
        clash = ch.HeadKind("clash", "c", "t", "regression", "t", {"weird": "_nu"}, ("weird",))
        with self.assertRaisesRegex(ValueError, "suffix"):    # `_nu` already means role nu
            ch.register_head_kind(clash)
        self.assertNotIn("clash", ch.HEAD_KINDS)

    def test_a_new_head_is_one_registration_not_an_n_file_edit(self):
        ch = self.ch
        kind = ch.HeadKind("zz_demo_conf", "zzconf", "zz_confidence_task", "confidence", "zz_confidence_task",
                           {"conf": "_zz_conf"}, ("conf",))
        ch.register_head_kind(kind)
        try:
            hs = ch.HeadSet.default(("high",)).extended(ch.HeadSpec.make("high", "zz_demo_conf", loss_weight=0.2))
            self.assertEqual(hs.names[-1], "high_zzconf")
            self.assertIn("y_high_zz_conf", hs.output_names())
            self.assertEqual(hs.head_types(), {"high": ["nig_regression", "binary_classification", "zz_demo_conf"]})
            self.assertEqual(hs.target_configs()["high_zzconf"]["loss_weight"], 0.2)
            self.assertEqual(hs.select(role="confidence").names, ("high_zzconf",))
            self.assertEqual(self.ch.parse_output_name("y_low_zz_conf"), ("low", "conf"))
            self.assertEqual(ch.HeadSet.from_target_configs(hs.target_configs()), hs)
        finally:
            ch.HEAD_KINDS.unregister("zz_demo_conf")

    def test_one_function_builds_and_one_parses_the_names(self):
        ch = self.ch
        self.assertEqual(ch.y_name("high"), "y_high")
        self.assertEqual(ch.y_name("high", "_nu"), "y_high_nu")
        self.assertEqual(ch.label_key("high", "class"), "y_high_class")
        self.assertEqual(ch.parse_head_name("low_reg"), ("low", "reg"))
        self.assertEqual(ch.parse_head_name("close_class"), ("close", "class"))
        with self.assertRaisesRegex(ValueError, "unknown head name"):
            ch.parse_head_name("low_foo")
        for t in ("high", "low", "close"):
            hs = ch.HeadSet.default((t,))
            for spec in hs:
                for role, key in zip(HEAD_ROLES_OF[spec.head_type], spec.output_names):
                    self.assertEqual(ch.parse_output_name(key), (t, role))
        with self.assertRaises(ValueError):
            ch.parse_output_name("high_nu")
        with self.assertRaises(ValueError):
            ch.parse_output_name("y_")

    def test_output_name_is_the_inverse_of_parse_output_name(self):
        ch = self.ch
        self.assertEqual(ch.output_name("low", "mu"), "y_low")
        self.assertEqual(ch.output_name("high", "nu"), "y_high_nu")
        self.assertEqual(ch.output_name("close", "logits"), "y_close_class_logits")
        for t in ("high", "low", "close"):
            for spec in ch.HeadSet.default((t,)):
                for role, key in zip(HEAD_ROLES_OF[spec.head_type], spec.output_names):
                    self.assertEqual(ch.output_name(t, role), key)
        with self.assertRaisesRegex(ValueError, "unknown output role"):
            ch.output_name("high", "zz")

    def test_core_heads_imports_nothing_but_the_stdlib_and_core(self):
        import ast
        tree = ast.parse(open(os.path.join(ROOT, "core", "heads.py"), encoding="utf-8").read())
        mods = {n.module.split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [None]) if not (isinstance(n, ast.ImportFrom) and n.level)}
        self.assertLessEqual(mods, {"__future__", "dataclasses", "types", "typing", "core"})


HEAD_ROLES_OF = {"nig_regression": ("mu", "nu", "alpha", "beta", "epistemic", "aleatoric", "confidence"),
                 "binary_classification": ("logits",)}


# ───────────────────────────── the consumers are derived views (golden recorded before core/heads.py) ─────────────────────────────
class ConsumersAreDerivedViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import core.heads as ch
        cls.ch = ch
        cls.ns = run_ns()
        cls.gold = json.load(open(GOLDEN_PATH, encoding="utf-8"))
        cls.cases = all_cases(cls.ns)

    def test_the_golden_says_where_it_came_from(self):
        self.assertEqual(self.gold["meta"]["recorded_from_commit"], RECORDED_FROM)
        self.assertEqual(sorted(self.cases), sorted(self.gold["variants"]), "the variant set changed: re-record on the OLD code only")

    def test_every_variant_keeps_its_targets_and_head_types_and_is_a_headset(self):
        ns, ch = self.ns, self.ch
        for name, settings in self.cases.items():
            with self.subTest(variant=name):
                rec = self.gold["variants"][name]
                t_sha, head_types, pts = head_vocabulary_of(ns, settings)
                self.assertEqual((t_sha, head_types, pts), (rec["targets_sha"], rec["head_types"], rec["price_targets"]))
                # the three views of one HeadSet agree: head_types -> HeadSet -> target configs -> HeadSet
                hs = ch.HeadSet.from_head_types(head_types, label_smoothing=0.0)
                self.assertEqual(hs.head_types(), head_types)
                self.assertEqual(ch.HeadSet.from_target_configs(hs.target_configs()), hs)

    def test_the_built_models_have_the_recorded_outputs_and_exactly_the_headset_outputs(self):
        ns, ch = self.ns, self.ch
        for key, names in self.gold["output_names_by_combo"].items():
            pts, head_types = json.loads(key)
            with self.subTest(combo=key[:70]):
                built = built_output_names(ns, pts, head_types)
                self.assertEqual(built, names, "the builder's outputs (names and order) changed")
                hs = ch.HeadSet.from_head_types(head_types)
                self.assertEqual(sorted(built), sorted(hs.output_names()))
                self.assertEqual(sorted(hs.output_names()), sorted(set(hs.output_names())))
                # every output the trainer reads (output_keys) and every label the pipeline writes' model-side name exists
                loss_outs = {v for tc in hs.target_configs().values() for v in tc["output_keys"].values()}
                self.assertLessEqual(loss_outs, set(built))

    def test_variant_targets_round_trip_through_the_headset(self):
        ns, ch = self.ns, self.ch
        for name in ("default", "class_only", "target_entry_range", "label_smoothing_03", "suspended_nothing"):
            with self.subTest(variant=name):
                settings = self.cases[name]
                ds, kit = _stub_inputs()
                info = types.SimpleNamespace(model_tf="1h", model_tfs=("1h",), reg_target_scale=1.0)
                plan = _quiet(ns["plan_model"], settings, ds, info, kit)
                class_only = settings.model.class_only
                ls = settings.train.anti_memorization_label_smoothing if settings.model.anti_memorization else 0.0
                cfgs = ns["build_target_configs"](list(plan.price_targets), label_smoothing=ls, class_only=class_only)
                hs = ch.HeadSet.from_target_configs(cfgs)
                self.assertEqual(hs.head_types(), plan.model_overrides["head_types"])
                self.assertEqual(hs.target_configs(), cfgs)
                self.assertEqual(hs.targets, plan.price_targets)

    def test_a_non_default_loss_weight_reaches_the_trainer_and_the_default_is_silent(self):
        ns, ch = self.ns, self.ch
        sched = {"lambda_reg": {"start": 0.0, "end": 0.0}, "lambda_calib": {"start": 0.0, "end": 0.0}}

        def user(targets):
            return {"run": {"run_dir": "/tmp/head_vocab_cfg"}, "targets": targets, "loss": {"schedules": sched}}
        hs = ch.HeadSet.default(("high", "low")).with_loss_weights({"high_class": 0.5})
        cfg = ns["build_config"](user(hs.target_configs()))
        self.assertEqual(cfg["targets"]["high_class"]["loss_weight"], 0.5)
        self.assertNotIn("loss_weight", cfg["targets"]["low_class"])
        base = ns["build_config"](user(ch.HeadSet.default(("high", "low")).target_configs()))
        self.assertNotEqual(ns["config_fingerprint"](cfg), ns["config_fingerprint"](base))
        bad = hs.target_configs()
        bad["high_class"]["loss_weight"] = -1.0
        with self.assertRaisesRegex(ValueError, "loss_weight"):          # the trainer's own validation still guards the value
            ns["build_config"](user(bad))

    def test_one_epoch_loss_equals_the_one_recorded_before_the_refactor(self):
        from test_run_steps import _dataset, settings_for
        ns = self.ns
        gold = variants()[1]["return_trained_holdout"]["scenario"]
        work = tempfile.mkdtemp(prefix="head_vocab_epoch_")
        settings = settings_for(ns, gold, os.path.join(work, "run"))
        kit = ns["Toolkit"].from_namespace(ns)
        res = _quiet(ns["run_main"], settings, kit, dataset=_dataset(gold["two_tf"], gold["holdout"]), summary=False)
        got = {k: [float(x) for x in v] for k, v in res.history.history.items()}
        want = self.gold["one_epoch_history"]
        self.assertEqual(sorted(got), sorted(want))
        for k in want:
            np.testing.assert_allclose(got[k], want[k], rtol=1e-2, atol=1e-6, err_msg=k)


class PipelineHeadsAreAView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import core.heads as ch
        cls.ch = ch
        cls.ns = run_ns()

    def test_pipeline_head_names_are_the_default_headsets(self):
        ns, ch = self.ns, self.ch
        self.assertEqual(ns["HEAD_KINDS"], ("class", "reg"))
        self.assertEqual(ns["OUTPUT_PREFIX"], ch.OUTPUT_PREFIX)
        cfgs = [{"targets": ["high", "low", "close"], "enabled_heads": {}},
                {"targets": ["high", "low"], "enabled_heads": {"high_class": False}},
                {"targets": ["close"], "enabled_heads": {"close_reg": False}},
                {"targets": ["high", "low", "close"], "enabled_heads": {"low_class": False, "low_reg": False}}]
        for cfg in cfgs:
            with self.subTest(enabled=cfg["enabled_heads"]):
                heads = ns["get_target_heads"](config=cfg)
                hs = ns["enabled_head_set"](config=cfg)
                self.assertEqual(sorted(heads), sorted(hs.names))
                for h in heads:
                    target, slot = ns["split_head_name"](h)
                    self.assertEqual(ch.parse_head_name(h), (target, slot))
                    self.assertEqual(ns["output_name"](h), hs[h].label_key)
                    self.assertEqual(hs[h].target, target)
        self.assertEqual(ns["get_target_heads"](["high"], {"targets": ["high"]}), ["high_class", "high_reg"])
        with self.assertRaisesRegex(ValueError, "اسم رأس غير معروف"):
            ns["split_head_name"]("high_foo")


class ModelHeadsAgreeWithTheirKinds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import core.heads as ch
        cls.ch = ch
        cls.ns = run_ns()

    def test_every_kind_has_a_model_builder_and_the_builder_emits_the_kinds_outputs(self):
        ns, ch = self.ns, self.ch
        for head_type in ch.HEAD_KINDS:
            if head_type != "nig_regression":                 # built by build_nig_timenet_v2 (cross-target OrderedMeans)
                self.assertIn(head_type, ns["HEAD_REGISTRY"], f"{head_type}: HeadKind without a model head builder")
        # all three kinds at once (multiclass beside binary needs another target: the same output name would clash)
        heads = {"high": ["nig_regression", "binary_classification"], "low": ["nig_regression", "multiclass_classification"]}
        built = built_output_names(ns, ("high", "low"), heads)
        self.assertEqual(sorted(built), sorted(ch.HeadSet.from_head_types(heads).output_names()))

    def test_a_model_head_that_disagrees_with_its_kind_raises_at_build(self):
        ns = self.ns
        import tensorflow as tf
        ns["HEAD_REGISTRY"].register("binary_classification_renamed", lambda h, t, c: {f"y_{t}_oops": h}, version=1)
        self.ch.register_head_kind(self.ch.HeadKind(
            "binary_classification_renamed", "class2", "classification2", "classification", "classification2",
            {"logits": "_class_logits"}, ("logits",)))
        try:
            with self.assertRaisesRegex(ValueError, "HeadKind"):
                ns["build_head_outputs"]("binary_classification_renamed", tf.zeros((2, 4)), "high", {})
        finally:
            ns["HEAD_REGISTRY"].unregister("binary_classification_renamed")
            self.ch.HEAD_KINDS.unregister("binary_classification_renamed")

    def test_builder_defaults_are_the_old_values(self):
        ns = self.ns
        import inspect
        self.assertEqual(inspect.signature(ns["build_nig_timenet_v2"]).parameters["price_targets"].default, ("high", "low", "close"))
        self.assertEqual(ns["DEFAULT_HEAD_TYPES"], {t: ["nig_regression"] for t in ("high", "low", "close")})
        self.assertEqual(ns["MODEL_CONFIG"]["price_targets"], ("high", "low", "close"))
        self.assertEqual(ns["MODEL_CONFIG"]["head_types"],
                         {t: ["nig_regression", "binary_classification"] for t in ("high", "low", "close")})


if __name__ == "__main__" and "--record" in sys.argv:
    record()
    sys.exit(0)

"""Registered auxiliary loss terms of a regression / evidential head: `ic_weight` (-pearson(mu, y) over the batch) and `var_weight`
((std(mu) - std(y))^2, anti-shrink), reaching the trainer through RunSettings.train.loss.head_terms (experiment E-tgt-002).

Guards:
1. VOCABULARY: core.loss_plan.HEAD_LOSS_TERMS declares the terms and their trainer keys, trainer/tasks.py BATCH_LOSS_TERMS implements
   every "batch" term (and nothing else); each option is accepted by build_config for every task type the term names (derived from
   the term registry, so TASK_REGISTRY content and tests/golden_registry_contents.json do not change) and has a value rule.
2. OFF = OLD BEHAVIOR EXACTLY: on a real GenericTrainer step with a fixed model, a config without the keys and one with
   ic_weight = var_weight = 0.0 give the SAME floats, and both equal the numbers master (bc29f9d, before this change) produced for the
   same step (pinned below, taken from a checkout of that commit).
3. THE VALUE: the term equals numpy -corrcoef (also weighted), the loss is base + ic_weight * (-corr), a constant batch / zero weights /
   a size-1 batch give 0 and finite gradients (no NaN), and the term moves a toy regression toward a higher correlation.
4. ONE ROUTE: train.loss.head_terms is validated at settings construction (unknown head / option, option of the wrong task type, bad
   value raise), writes only those keys into the trainer config, and an empty plan leaves the config untouched.
    python -m pytest tests/test_ic_loss_term.py -q
"""
import contextlib
import io
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

from core.heads import HeadSet  # noqa: E402
from core.loss_plan import HEAD_LOSS_TERMS, check_head_loss_terms  # noqa: E402
from tests.test_loss_plan import S, diff, make_config, ns, plan, P2, P3  # noqa: E402

# test_step of the fixed step in `_fixed_step` on master bc29f9d (tests/test_ic_loss_term.py was produced from a checkout of it)
MASTER_STEP = {"loss": 2.594529867172241, "raw_loss": 2.594529867172241, "loss_high_reg": 1.8742964267730713,
               "high_reg_nig_base": 1.634416937828064, "high_reg_mae": 0.9223865270614624, "high_reg_nig_pen": 4.700644016265869}


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _model(tf):
    tf.keras.utils.set_random_seed(7)
    heads = HeadSet.default(("high",))
    inp = tf.keras.Input(shape=(6,), name="x")
    hid = tf.keras.layers.Dense(8, activation="tanh")(inp)
    outs = {}
    for h in heads:
        for name in sorted(h.output_names):
            if name.endswith(("_nu", "_alpha", "_beta")):
                outs[name] = tf.keras.layers.Dense(1, activation="softplus", name=name)(hid) + 1.0
            elif name.endswith("_confidence") or name.endswith("_class_logits"):
                outs[name] = tf.keras.layers.Dense(1, activation="sigmoid", name=name)(hid)
            else:
                outs[name] = tf.keras.layers.Dense(1, name=name)(hid)
    return heads, tf.keras.Model(inp, outs)


_SCHED = {"lambda_reg": {"start": 0.05, "end": 0.05, "warmup_epochs": 0, "schedule": "linear"},
          "lambda_calib": {"start": 0.1, "end": 0.1, "warmup_epochs": 0, "schedule": "linear"}}


def _fixed_step(extra=None):
    """test_step of the fixed model / batch (same recipe as the master run that produced MASTER_STEP); ``extra`` = keys of high_reg."""
    import tensorflow as tf
    n = ns()
    heads, model = _model(tf)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(64, 6)).astype("float32")
    y = {"y_high_reg": rng.normal(size=(64, 1)).astype("float32"), "y_high_class": rng.integers(0, 2, size=(64, 1)).astype("float32")}
    tg = heads.target_configs()
    tg["high_reg"].update(extra or {})
    cfg = n["build_config"]({"run": {"run_dir": "/tmp/ic_term_unused", "epochs": 1, "batch_size": 64, "verbose": 0}, "targets": tg,
                             "loss": {"use_uncertainty_weighting": False, "schedules": _SCHED}})
    tr = n["GenericTrainer"](model, cfg)
    for k in _SCHED:
        tr.scheduled_vars[k].assign(_SCHED[k]["end"])
    return {k: float(v) for k, v in tr.test_step((x, y)).items()}


class Vocabulary(unittest.TestCase):
    def test_terms_are_declared_in_core_and_implemented_in_the_trainer(self):
        n = ns()
        self.assertEqual(HEAD_LOSS_TERMS.check_all(), [])
        self.assertEqual(HEAD_LOSS_TERMS.names(), ("huber", "ic", "var"))
        batch = {e.name for e in HEAD_LOSS_TERMS.entries() if e.meta["kind"] == "batch"}
        self.assertEqual(batch, set(n["BATCH_LOSS_TERMS"]))
        for e in HEAD_LOSS_TERMS.entries():
            self.assertEqual(e.meta["options"][e.meta["weight_key"]], 0.0, e.name)        # OFF by default
            for ttype in e.meta["task_types"]:
                for key in e.meta["options"]:
                    self.assertIn(key, n["_head_term_keys"](ttype), (e.name, ttype, key))   # accepted by build_config
            for key in e.meta["options"]:
                self.assertIn(key, n["_TARGET_VALUE_RULES"], key)

    def test_option_names_have_one_owner(self):
        owners = [k for e in HEAD_LOSS_TERMS.entries() for k in e.meta["options"]]
        self.assertEqual(len(owners), len(set(owners)))


class OffIsTheOldBehavior(unittest.TestCase):
    def test_absent_and_zero_weight_are_the_same_floats_and_match_master(self):
        absent = _quiet(_fixed_step)
        zero = _quiet(_fixed_step, {"ic_weight": 0.0, "var_weight": 0.0})
        self.assertEqual(absent, zero)                                    # exactly the same floats, every logged metric
        for k, v in MASTER_STEP.items():
            self.assertAlmostEqual(absent[k], v, places=6, msg=k)
        self.assertEqual(set(absent), {"loss", "raw_loss", "loss_high_class", "loss_high_reg", "high_class_accuracy", "high_reg_aleatoric",
                                       "high_reg_calib_loss", "high_reg_confidence", "high_reg_epistemic", "high_reg_mae",
                                       "high_reg_nig_base", "high_reg_nig_pen"})            # no new log key

    def test_on_changes_the_loss_by_the_term_and_nothing_else_logged(self):
        base = _quiet(_fixed_step)
        on = _quiet(_fixed_step, {"ic_weight": 0.4})
        self.assertNotEqual(base["loss_high_reg"], on["loss_high_reg"])
        for k in ("high_reg_mae", "high_reg_nig_base", "high_reg_nig_pen", "high_class_accuracy"):
            self.assertEqual(base[k], on[k], k)                          # stats are the same predictions' stats
        self.assertLess(on["loss_high_reg"] - base["loss_high_reg"], 0.4 + 1e-6)     # -corr in [-1, 1] times the weight
        self.assertGreater(on["loss_high_reg"] - base["loss_high_reg"], -0.4 - 1e-6)


class TheValue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf
        cls.n = ns()
        rng = np.random.default_rng(11)
        cls.y = rng.normal(size=64).astype("float32")
        cls.mu = (0.3 * cls.y + rng.normal(size=64)).astype("float32")
        cls.w = rng.uniform(0.2, 2.0, size=64).astype("float32")

    def term(self, name, mu, y, w=None):
        return float(self.n["BATCH_LOSS_TERMS"][name](self.tf.constant(mu), self.tf.constant(y), None if w is None else self.tf.constant(w), {}))

    def test_ic_matches_numpy_negative_corrcoef(self):
        want = -np.corrcoef(self.mu.astype("float64"), self.y.astype("float64"))[0, 1]
        self.assertAlmostEqual(self.term("ic", self.mu, self.y), want, places=5)
        self.assertAlmostEqual(self.term("ic", self.mu.reshape(-1, 1), self.y.reshape(-1, 1)), want, places=5)   # [B, 1] as the heads give

    def test_weighted_ic_matches_numpy(self):
        m, y, w = (a.astype("float64") for a in (self.mu, self.y, self.w))
        mp, my = (w * m).sum() / w.sum(), (w * y).sum() / w.sum()
        want = -(w * (m - mp) * (y - my)).sum() / np.sqrt((w * (m - mp) ** 2).sum() * (w * (y - my) ** 2).sum())
        self.assertAlmostEqual(self.term("ic", self.mu, self.y, self.w), want, places=5)

    def test_var_matches_numpy(self):
        want = (self.mu.astype("float64").std() - self.y.astype("float64").std()) ** 2
        self.assertAlmostEqual(self.term("var", self.mu, self.y), want, places=5)

    def test_scale_free_ic_and_perfect_prediction(self):
        self.assertAlmostEqual(self.term("ic", 5.0 * self.y + 2.0, self.y), -1.0, places=5)
        self.assertAlmostEqual(self.term("ic", -self.y, self.y), 1.0, places=5)
        self.assertAlmostEqual(self.term("var", self.y, self.y), 0.0, places=5)

    def test_constant_batches_zero_weights_and_size_one_are_zero_not_nan(self):
        tf = self.tf
        const = np.full(64, 0.7, "float32")
        for name in ("ic",):
            self.assertEqual(self.term(name, const, self.y), 0.0)
            self.assertEqual(self.term(name, self.mu, const), 0.0)
            self.assertEqual(self.term(name, self.mu, self.y, np.zeros(64, "float32")), 0.0)
            self.assertEqual(self.term(name, self.mu[:1], self.y[:1]), 0.0)
        self.assertEqual(self.term("var", self.mu, self.y, np.zeros(64, "float32")), 0.0)
        for name in ("ic", "var"):                         # gradients are finite exactly where the value is guarded
            for mu, y, w in ((const, self.y, None), (self.mu, const, None), (self.mu, self.y, np.zeros(64, "float32")),
                             (self.mu[:1], self.y[:1], None), (self.mu, self.y, None)):
                v = tf.Variable(mu)
                with tf.GradientTape() as tape:
                    out = self.n["BATCH_LOSS_TERMS"][name](v, tf.constant(y), None if w is None else tf.constant(w), {})
                g = tape.gradient(out, v)
                self.assertTrue(np.isfinite(float(out)), name)
                self.assertTrue(np.all(np.isfinite(g.numpy())), name)

    def test_evidential_loss_is_base_plus_weighted_term(self):
        tf, n = self.tf, self.n
        heads = HeadSet.default(("high",))
        cfg = heads.target_configs()["high_reg"]
        keys = cfg["output_keys"]
        rng = np.random.default_rng(5)
        outs = {keys["mu"]: tf.constant(self.mu), keys["nu"]: tf.constant(rng.uniform(0.5, 2, 64).astype("float32")),
                keys["alpha"]: tf.constant(rng.uniform(1.2, 3, 64).astype("float32")), keys["beta"]: tf.constant(rng.uniform(0.5, 2, 64).astype("float32"))}
        sv = {"lambda_reg": tf.constant(0.05), "lambda_calib": tf.constant(0.1)}
        cfg_off = {k: v for k, v in cfg.items() if k != "use_calibration_loss"}
        fn = n["evidential_task_loss"]
        base, _ = fn(tf.constant(self.y), outs, cfg_off, sv)
        for w in (None, tf.constant(self.w)):
            kw = {} if w is None else {"sample_weight": w}
            b, _ = fn(tf.constant(self.y), outs, cfg_off, sv, **kw)
            on, _ = fn(tf.constant(self.y), outs, {**cfg_off, "ic_weight": 0.4, "var_weight": 0.1}, sv, **kw)
            want = self.term("ic", self.mu, self.y, None if w is None else self.w) * 0.4 + self.term("var", self.mu, self.y, None if w is None else self.w) * 0.1
            self.assertAlmostEqual(float(on - b), want, places=5)
        self.assertEqual(float(base), float(fn(tf.constant(self.y), outs, {**cfg_off, "ic_weight": 0.0, "var_weight": 0.0}, sv)[0]))

    def test_regression_task_also_takes_the_terms(self):
        tf, n = self.tf, self.n
        cfg = {"output_keys": {"pred": "p"}, "loss_fn": "mse"}
        outs = {"p": tf.constant(self.mu)}
        off = float(n["regression_task_loss"](tf.constant(self.y), outs, cfg, {})[0])
        on = float(n["regression_task_loss"](tf.constant(self.y), outs, {**cfg, "ic_weight": 1.0}, {})[0])
        self.assertAlmostEqual(on - off, self.term("ic", self.mu, self.y), places=5)

    def test_the_term_trains_a_head_toward_higher_correlation(self):
        """A regression head with mse + ic_weight on a toy problem ends with a higher batch correlation than the same head without it."""
        tf, n = self.tf, self.n
        rng = np.random.default_rng(2)
        x = rng.normal(size=(512, 4)).astype("float32")
        y = (x @ np.array([1.0, -0.5, 0.0, 0.2], "float32") + 3.0 * rng.normal(size=512)).astype("float32")

        def fit(ic_weight):
            tf.keras.utils.set_random_seed(0)
            inp = tf.keras.Input(shape=(4,), name="x")
            model = tf.keras.Model(inp, {"p": tf.keras.layers.Dense(1, name="p")(inp)})
            cfg = {"output_keys": {"pred": "p"}, "loss_fn": "mse"}
            if ic_weight:
                cfg["ic_weight"] = ic_weight
            opt = tf.keras.optimizers.Adam(0.05)
            for _ in range(150):
                with tf.GradientTape() as tape:
                    loss, _ = n["regression_task_loss"](tf.constant(y), model(tf.constant(x), training=True), cfg, {})
                opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
            return float(np.corrcoef(model(x)["p"].numpy().ravel(), y)[0, 1])

        self.assertGreater(fit(2.0), fit(0.0) - 1e-3)       # an extra pull toward corr never lowers it beyond noise
        self.assertGreater(fit(2.0), 0.3)


class OneRoute(unittest.TestCase):
    def test_empty_plan_leaves_the_config_untouched(self):
        a = make_config(S(), P3)
        self.assertEqual(diff(a, make_config(S(plan(head_terms={})), P3)), {})
        self.assertEqual(diff(a, make_config(S(plan(head_terms={"close_reg": {}})), P3)), {})

    def test_head_terms_write_exactly_their_keys(self):
        a = make_config(S(), P3)
        cfg = make_config(S(plan(head_terms={"close_reg": {"huber_weight": 0.1, "ic_weight": 0.4, "var_weight": 0.1}})), P3)
        self.assertEqual(diff(a, cfg), {"targets.close_reg.huber_weight": (None, 0.1), "targets.close_reg.ic_weight": (None, 0.4),
                                        "targets.close_reg.var_weight": (None, 0.1)})
        self.assertEqual(a["callbacks"], cfg["callbacks"])
        self.assertEqual(a["loss"], cfg["loss"])

    def test_roundtrip_and_frozen_copy(self):
        s = S(plan(head_terms={"high_reg": {"ic_weight": 0.4}}))
        self.assertEqual(ns()["RunSettings"].from_dict(s.to_dict()), s)

    def test_bad_plans_raise_at_construction(self):
        for bad, needle in (({"nope_reg": {"ic_weight": 0.4}}, "unknown head"),
                            ({"high_reg": {"icw": 0.4}}, "unknown option"),
                            ({"high_class": {"ic_weight": 0.4}}, "task type"),
                            ({"high_reg": {"ic_weight": -1}}, ">= 0"),
                            ({"high_reg": {"ic_weight": float("nan")}}, ">= 0"),
                            ({"high_reg": {"huber_delta": 0}}, "> 0"),
                            ({"high_reg": 0.4}, "must be a dict")):
            with self.assertRaises(ValueError, msg=str(bad)) as cm:
                S(plan(head_terms=bad))
            self.assertIn(needle, str(cm.exception), bad)
        with self.assertRaises(ValueError):                  # a head the run does not train (P2 has no close)
            make_config(S(plan(head_terms={"close_reg": {"ic_weight": 0.4}})), P2)

    def test_check_function_returns_floats(self):
        self.assertEqual(check_head_loss_terms({"high_reg": {"ic_weight": 1}}, HeadSet.default(P2)), {"high_reg": {"ic_weight": 1.0}})

    def test_trainer_config_rejects_the_keys_on_the_wrong_task_type(self):
        n = ns()
        tg = HeadSet.default(("high",)).target_configs()
        tg["high_class"]["ic_weight"] = 0.4
        with self.assertRaises(ValueError) as cm:
            n["build_config"]({"run": {"run_dir": "/tmp/x"}, "targets": tg})
        self.assertIn("ic_weight", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

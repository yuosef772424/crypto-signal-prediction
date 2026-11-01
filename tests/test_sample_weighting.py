"""Training sample weights as ONE registered component (core.loss_plan.SAMPLE_WEIGHTINGS, "abs_target_clip"), routed through
RunSettings.train.loss.sample_weighting (E-tgt-002 arm D).

Guards:
1. DEFINITION: w = clip(|z|, 0.2, 2) and w = 0.2 inside |z| < 0.3; unknown names raise; the method needs a volnorm target mode.
2. OFF = OLD BATCHES: sample_weighting=None gives the same (x, y) elements as the old make_shuffled_dataset call.
3. TRAIN ONLY: make_datasets adds weights computed from the TRAIN labels to the training batches only (val batches stay (x, y)),
   one weight per regression head by its label key, none for classification heads.
4. THE LOSS: a GenericTrainer train step on a weighted batch equals the weighted loss computed by hand (the existing sample_weight path),
   and the val loss (test_step without weights) is unchanged.
5. ROUTE: bad names / modes raise at construction; a weighting changes the run folder name.
    python -m pytest tests/test_sample_weighting.py -q
"""
import contextlib
import io
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from core.loss_plan import SAMPLE_WEIGHTINGS, check_sample_weighting  # noqa: E402
from tests import run_fixture as rf  # noqa: E402
from tests.test_run_steps import _dataset  # noqa: E402


def _formula(abs_y):
    return np.where(abs_y < 0.3, 0.2, np.clip(abs_y, 0.2, 2.0))


class Definition(unittest.TestCase):
    def test_values(self):
        fn = SAMPLE_WEIGHTINGS.get("abs_target_clip")
        meta = SAMPLE_WEIGHTINGS.entry("abs_target_clip").meta
        z = np.array([0.0, 0.1, 0.29, 0.3, 0.5, 1.0, 2.0, 5.0, 40.0])
        np.testing.assert_allclose(fn(z, clip=np.clip, where=np.where, **meta["params"]), [0.2, 0.2, 0.2, 0.3, 0.5, 1.0, 2.0, 2.0, 2.0])
        self.assertEqual(SAMPLE_WEIGHTINGS.check_all(), [])

    def test_check(self):
        self.assertIsNone(check_sample_weighting(None, None))
        self.assertEqual(check_sample_weighting("abs_target_clip", "volnorm"), "abs_target_clip")
        self.assertEqual(check_sample_weighting("abs_target_clip", "volnorm+relative"), "abs_target_clip")
        for name, mode in (("abs_target_clip", None), ("abs_target_clip", "return"), ("abs_target_clip", "scaled"), ("nope", "volnorm"), (3, "volnorm")):
            with self.assertRaises(ValueError, msg=(name, mode)):
                check_sample_weighting(name, mode)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = rf.run_namespace()
        cls.ns, cls.S = ns, ns["RunSettings"]
        cls.kit = ns["Toolkit"].from_namespace(ns)
        cls.ds = _dataset(False, None)
        cls.info = rf._quiet(ns["apply_dataset_config"], cls.S(), cls.ds, cls.kit)
        base = rf._quiet(ns["make_splits"], cls.S(), cls.ds, cls.info, cls.kit)
        cls.on = cls.S().updated({"target": {"target_mode": "volnorm", "clip": 4.0},
                                  "train": {"loss": {"sample_weighting": "abs_target_clip"}, "batch_size": 32}})
        cls.off = cls.on.updated({"train": {"loss": {}}})
        cls.train, cls.val, _ = rf._quiet(ns["retarget"], cls.on, *base, cls.info)
        plan = rf._quiet(ns["plan_model"], cls.on, cls.ds, cls.info, cls.kit)
        cls.cfg = rf._quiet(ns["make_training_config"], cls.on, plan, cls.info, cls.val, cls.kit)

    def test_off_gives_the_old_elements(self):
        ns, tfs = self.ns, list(self.info.model_tfs)
        X, y = ns["model_x"](self.train, tfs), ns["_y_for"](self.train, self.cfg)
        old = ns["make_shuffled_dataset"](X, y, 32, seed=3)
        new = ns["make_shuffled_dataset"](X, y, 32, seed=3, sample_weight=None)
        for a, b in zip(old.take(3), new.take(3)):
            self.assertEqual(len(b), 2)
            np.testing.assert_array_equal(a[0].numpy(), b[0].numpy())
            for k in a[1]:
                np.testing.assert_array_equal(a[1][k].numpy(), b[1][k].numpy())
        tr_off, val_off, _ = rf._quiet(ns["make_datasets"], self.cfg, self.train, self.val, self.info, settings=self.off)
        self.assertEqual(len(next(iter(tr_off))), 2)

    def test_weights_on_train_batches_only_from_train_labels(self):
        ns = self.ns
        tr, val, sample = rf._quiet(ns["make_datasets"], self.cfg, self.train, self.val, self.info, settings=self.on)
        self.assertEqual(len(sample), 3)
        self.assertEqual(len(next(iter(val))), 2)                                   # val stays unweighted
        reg_keys = {c["true_key"] for c in self.cfg["targets"].values() if c["task_type"] in ("evidential", "regression")}
        for x, y, w in tr.take(5):
            self.assertEqual(set(w), reg_keys)                                      # no weight for classification heads
            for k in reg_keys:
                np.testing.assert_allclose(w[k].numpy(), _formula(np.abs(y[k].numpy().astype("float64"))), rtol=1e-6)
        sw = ns["sample_weights_for"](self.train, self.cfg, "abs_target_clip")
        for k in reg_keys:
            np.testing.assert_allclose(sw[k], _formula(np.abs(np.asarray(self.train["y"][k], "float64"))), rtol=1e-6)
            self.assertLessEqual(float(sw[k].max()), 2.0)
            self.assertGreaterEqual(float(sw[k].min()), 0.2)

    def test_route(self):
        with self.assertRaises(ValueError):
            self.S().updated({"train": {"loss": {"sample_weighting": "abs_target_clip"}}})      # target_mode None
        with self.assertRaises(ValueError):
            self.S().updated({"target": {"target_mode": "volnorm"}, "train": {"loss": {"sample_weighting": "nope"}}})
        rd = self.ns["run_dir_for"]
        self.assertEqual(rd(self.on, 1.0, ("4h",)), rd(self.off, 1.0, ("4h",)).replace("_am", "_sw-abs_target_clip_am"))
        self.assertEqual(self.ns["RunSettings"].from_dict(self.on.to_dict()), self.on)


class WeightedLoss(unittest.TestCase):
    def test_train_step_uses_the_weights_and_test_step_does_not(self):
        import tensorflow as tf
        from tests.test_ic_selection import _cfg, _toy
        from tests.test_loss_plan import ns

        def run(weights):
            heads, model = _toy(tf)
            tr = ns()["GenericTrainer"](model, _cfg(heads))
            tr.compile(optimizer=tf.keras.optimizers.SGD(0.0))                     # lr 0: the step reports the loss of fixed weights
            rng = np.random.default_rng(0)
            x = rng.normal(size=(32, 6)).astype("float32")
            yr = rng.normal(size=(32, 1)).astype("float32")
            y = {"y_high_reg": yr, "y_high_class": (yr > 0).astype("float32")}
            data = (x, y) if weights is None else (x, y, {"y_high_reg": weights(yr)})
            with contextlib.redirect_stdout(io.StringIO()):
                train = {k: float(v) for k, v in tr.train_step(data).items()}
            for m in tr.metrics:
                m.reset_state()
            test = {k: float(v) for k, v in tr.test_step((x, y)).items()}
            outs = model(x)
            tcfg = tr.config["targets"]["high_reg"]
            sv = tr.scheduled_vars
            by_hand = float(ns()["evidential_task_loss"](tf.constant(yr), outs, tcfg, sv,
                                                         **({} if weights is None else {"sample_weight": tf.constant(weights(yr))}))[0])
            return train, test, by_hand

        w_fn = lambda yr: _formula(np.abs(yr.astype("float64"))).astype("float32")   # noqa: E731
        tr0, te0, hand0 = run(None)
        tr1, te1, hand1 = run(w_fn)
        self.assertAlmostEqual(tr0["loss_high_reg"], hand0, places=5)
        self.assertAlmostEqual(tr1["loss_high_reg"], hand1, places=5)
        self.assertNotAlmostEqual(tr0["loss_high_reg"], tr1["loss_high_reg"], places=4)
        self.assertEqual(tr0["loss_high_class"], tr1["loss_high_class"])          # classification head unweighted
        self.assertEqual(te0, te1)                                                 # validation loss never weighted


if __name__ == "__main__":
    unittest.main()

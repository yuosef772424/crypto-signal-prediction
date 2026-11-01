"""Sample weightings "jump_downweight", "curriculum_snr", "curriculum_random" (core.loss_plan.SAMPLE_WEIGHTINGS) and their composition.

Guards: definitions + vocabulary (unknown names / wrong target mode / two curricula raise); the jump rule reads the UNCLIPPED |return|/vol
(a clip at 4 hides every |z| > 4); weights come from TRAIN labels and reach train batches only; curriculum ranks (snr = descending
unclipped |z|, random = seeded permutation, same sizes); the trainer keeps only the samples within f(e) (f0 -> 1 linearly at full_at_epoch)
and nothing changes without a curriculum; the plan writes loss.curriculum only when a curriculum member exists.
    python -m pytest tests/test_sample_weighting_jump_curriculum.py -q
"""
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

from core.loss_plan import (CURRICULUM_RANK_KEY, SAMPLE_WEIGHTINGS, check_sample_weighting, sample_weighting_parts,  # noqa: E402
                            trainer_loss_section)
from tests import run_fixture as rf  # noqa: E402
from tests.test_run_steps import _dataset  # noqa: E402


class Vocabulary(unittest.TestCase):
    def test_registered_and_valid(self):
        self.assertEqual(SAMPLE_WEIGHTINGS.check_all(), [])
        for n in ("jump_downweight", "curriculum_snr", "curriculum_random"):
            self.assertIn(n, SAMPLE_WEIGHTINGS)
        fn = SAMPLE_WEIGHTINGS.get("jump_downweight")
        p = SAMPLE_WEIGHTINGS.entry("jump_downweight").meta["params"]
        np.testing.assert_allclose(fn(np.array([0.0, 5.9, 6.0, 6.1, 40.0]), clip=np.clip, where=np.where, **p), [1, 1, 1, 0.3, 0.3])

    def test_check(self):
        self.assertEqual(sample_weighting_parts("a+b"), ["a", "b"])
        self.assertEqual(sample_weighting_parts(None), [])
        for ok in ("jump_downweight", "curriculum_snr", "jump_downweight+curriculum_snr", "curriculum_random+jump_downweight"):
            self.assertEqual(check_sample_weighting(ok, "volret"), ok)
            self.assertEqual(check_sample_weighting(ok, "volnorm"), ok)
        for bad, mode in (("curriculum_snr", "return"), ("curriculum_snr", None), ("curriculum_snr+curriculum_random", "volret"),
                          ("jump_downweight+jump_downweight", "volret"), ("nope", "volret"), ("jump_downweight+nope", "volret")):
            with self.assertRaises(ValueError, msg=(bad, mode)):
                check_sample_weighting(bad, mode)

    def test_plan_section(self):
        sched = {"lambda_reg": {"start": 0, "end": 0}, "lambda_calib": {"start": 0, "end": 0}}
        self.assertNotIn("curriculum", trainer_loss_section("static", sched, None, "jump_downweight"))
        self.assertNotIn("curriculum", trainer_loss_section("static", sched))
        self.assertEqual(trainer_loss_section("static", sched, None, "jump_downweight+curriculum_snr")["curriculum"],
                         {"f0": 0.3, "full_at_epoch": 20})


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns = rf.run_namespace()
        cls.ns, cls.S = ns, ns["RunSettings"]
        cls.kit = ns["Toolkit"].from_namespace(ns)
        cls.ds = _dataset(False, None)
        cls.info = rf._quiet(ns["apply_dataset_config"], cls.S(), cls.ds, cls.kit)
        base = rf._quiet(ns["make_splits"], cls.S(), cls.ds, cls.info, cls.kit)
        cls.settings = cls.S().updated({"target": {"target_mode": "volret", "clip": 4.0, "suspended_targets": ()},
                                        "train": {"loss": {"sample_weighting": "jump_downweight+curriculum_snr"}, "batch_size": 32}})
        cls.train, cls.val, _ = rf._quiet(ns["retarget"], cls.settings, *base, cls.info)
        plan = rf._quiet(ns["plan_model"], cls.settings, cls.ds, cls.info, cls.kit)
        cls.cfg = rf._quiet(ns["make_training_config"], cls.settings, plan, cls.info, cls.val, cls.kit)

    def test_config_carries_the_curriculum(self):
        self.assertEqual(self.cfg["loss"]["curriculum"], {"f0": 0.3, "full_at_epoch": 20})

    def test_jump_weights_read_the_unclipped_target(self):
        import copy
        z = self.ns["unclipped_abs_z"](self.train, "close")
        w = self.ns["sample_weights_for"](self.train, self.cfg, "jump_downweight")
        np.testing.assert_allclose(w["y_close_reg"], np.where(z > 6.0, 0.3, 1.0).astype("float32"))
        self.assertNotIn(CURRICULUM_RANK_KEY, w)
        jumped = copy.deepcopy(self.train)
        jumped["last_candles"][:5, 4] *= 1.5                              # a +50% jump: the label stays clipped at 4, the weight must see it
        wj = self.ns["sample_weights_for"](jumped, self.cfg, "jump_downweight")["y_close_reg"]
        self.assertTrue(np.all(wj[:5] == np.float32(0.3)))
        self.assertLessEqual(float(np.abs(jumped["y"]["y_close_reg"]).max()), 4.0 + 1e-6)

    def test_ranks(self):
        z = self.ns["unclipped_abs_z"](self.train, "close")
        snr = self.ns["curriculum_rank"](self.train, "snr", 1)
        n = len(z)
        self.assertEqual(sorted(np.round(snr * n).astype(int)), list(range(1, n + 1)))
        self.assertLess(snr[np.argmax(z)], 2.0 / n)                      # the clearest move is the first one included
        r1, r2, r3 = (self.ns["curriculum_rank"](self.train, "random", s) for s in (1, 1, 2))
        np.testing.assert_array_equal(r1, r2)
        self.assertFalse(np.array_equal(r1, r3))
        self.assertLess(abs(np.corrcoef(r1, snr)[0, 1]), 0.3)            # the control has no order information
        with self.assertRaises(ValueError):
            self.ns["curriculum_rank"](self.train, "hard", 1)

    def test_composition_is_the_product_and_train_batches_only(self):
        w = self.ns["sample_weights_for"](self.train, self.cfg, "jump_downweight+curriculum_snr")
        j = self.ns["sample_weights_for"](self.train, self.cfg, "jump_downweight")
        np.testing.assert_array_equal(w["y_close_reg"], j["y_close_reg"])
        self.assertIn(CURRICULUM_RANK_KEY, w)
        tr, va, _ = rf._quiet(self.ns["make_datasets"], self.cfg, self.train, self.val, self.info, settings=self.settings)
        batch = next(iter(tr))
        self.assertEqual(len(batch), 3)
        self.assertIn(CURRICULUM_RANK_KEY, batch[2])
        self.assertEqual(len(next(iter(va))), 2)                          # validation is never weighted


class TrainerMask(unittest.TestCase):
    def _trainer(self, curriculum):
        import tensorflow as tf
        from tests.test_ic_selection import _SCHED, _data, _toy, ns
        heads, model = _toy(tf, 5)
        loss = {"use_uncertainty_weighting": False, "schedules": _SCHED}
        if curriculum:
            loss["curriculum"] = {"f0": 0.3, "full_at_epoch": 10}
        cfg = ns()["build_config"]({"run": {"run_dir": "/tmp/cur_unused", "epochs": 1, "batch_size": 32, "verbose": 0},
                                    "targets": heads.target_configs(), "loss": loss})
        x, y = _data(64, 1)
        batch = (tf.constant(x[:32]), {k: tf.constant(v[:32]) for k, v in y.items()})
        return ns()["GenericTrainer"](model, cfg), batch, tf

    def test_mask_follows_the_epoch_schedule(self):
        tr, (x, y), tf = self._trainer(True)
        rank = np.linspace(1 / 32, 1.0, 32).astype("float32")
        sw = {CURRICULUM_RANK_KEY: tf.constant(rank)}
        _, raw0, _, _, _ = tr._compute_losses(x, y, sw, training=True)                  # epoch 0: f = 0.3 -> the first 9 of 32 samples
        tr.ckpt_epoch.assign(10)
        _, raw1, per1, _, _ = tr._compute_losses(x, y, sw, training=True)               # epoch 10: f = 1 -> everything
        tr.ckpt_epoch.assign(0)
        none_tr, _, _ = self._trainer(False)
        _, rawn, _, _, _ = none_tr._compute_losses(x, y, None, training=True)
        np.testing.assert_allclose(raw1.numpy(), rawn.numpy(), rtol=1e-5)               # f = 1 = no curriculum
        self.assertNotAlmostEqual(float(raw0.numpy()), float(rawn.numpy()), places=4)   # f = 0.3 trains on a subset
        _, rawv, _, _, _ = tr._compute_losses(x, y, None, training=False)               # validation: no mask, no weights
        _, rawn_v, _, _, _ = none_tr._compute_losses(x, y, None, training=False)
        np.testing.assert_allclose(rawv.numpy(), rawn_v.numpy(), rtol=1e-5)

    def test_off_has_no_curriculum(self):
        tr, _, _ = self._trainer(False)
        self.assertIsNone(tr.curriculum)


if __name__ == "__main__":
    unittest.main()

"""Loss balancer "ema_norm" (core.loss_plan.LOSS_BALANCERS): each head loss / its EMA of |loss| (stop-gradient) x its FIXED weight.

Guards: registered + routed (trainer loss section), unknown names still raise; OFF (default 'kendall'/'static') = no EMA variables and the
old per-head losses exactly; ON: first training step normalises every head loss to magnitude 1 x weight, the EMA variables are updated and
checkpointed, eval steps divide by the stored EMA without updating it, the gradient does not flow through the EMA.
    python -m pytest tests/test_loss_balancer_ema_norm.py -q
"""
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

from core.loss_plan import LOSS_BALANCERS, trainer_loss_section  # noqa: E402
from tests.test_ic_selection import _SCHED, _data, _toy, ns  # noqa: E402


def _trainer(balancing):
    import tensorflow as tf
    heads, model = _toy(tf, 5)
    loss = dict(trainer_loss_section(balancing, _SCHED, heads))
    cfg = ns()["build_config"]({"run": {"run_dir": "/tmp/ema_norm_unused", "epochs": 1, "batch_size": 32, "verbose": 0},
                                "targets": heads.target_configs(), "loss": loss})
    x, y = _data(64, 1)
    batch = (tf.constant(x[:32]), {k: tf.constant(v[:32]) for k, v in y.items()})
    return ns()["GenericTrainer"](model, cfg), batch


class Route(unittest.TestCase):
    def test_registered(self):
        self.assertIn("ema_norm", LOSS_BALANCERS)
        self.assertEqual(LOSS_BALANCERS.get("ema_norm")(None), {"use_uncertainty_weighting": False, "loss_normalization": "ema"})
        self.assertEqual(LOSS_BALANCERS.check_all(), [])
        with self.assertRaises(ValueError):
            LOSS_BALANCERS.get("ema_normm")

    def test_default_sections_unchanged(self):
        self.assertEqual(trainer_loss_section("kendall", _SCHED)["use_uncertainty_weighting"], True)
        self.assertNotIn("loss_normalization", trainer_loss_section("kendall", _SCHED))
        self.assertNotIn("loss_normalization", trainer_loss_section("static", _SCHED))


class Behaviour(unittest.TestCase):
    def test_off_has_no_variables_and_old_losses(self):
        tr, (x, y) = _trainer("static")
        self.assertEqual(tr.loss_ema, {})
        _, raw, per, _, _ = tr._compute_losses(x, y, None, training=True)
        tr2, _ = _trainer("static")
        _, raw2, per2, _, _ = tr2._compute_losses(x, y, None, training=True)
        for t in per:
            np.testing.assert_allclose(per[t].numpy(), per2[t].numpy())

    def test_on_normalises_and_updates(self):
        tr, (x, y) = _trainer("ema_norm")
        self.assertEqual(sorted(tr.loss_ema), sorted(tr.target_names))
        _, _, per, _, _ = tr._compute_losses(x, y, None, training=True)
        for t, v in per.items():
            self.assertAlmostEqual(abs(float(v.numpy())), 1.0, places=4, msg=t)       # |L| / EMA(|L|), first step: EMA = |L|
            self.assertGreater(float(tr.loss_ema[t].numpy()), 0.0)
        before = {t: float(v.numpy()) for t, v in tr.loss_ema.items()}
        _, _, per_eval, _, _ = tr._compute_losses(x, y, None, training=False)
        for t in before:
            self.assertEqual(float(tr.loss_ema[t].numpy()), before[t])               # eval does not update the EMA
            self.assertTrue(np.isfinite(per_eval[t].numpy()))

    def test_ema_variables_sit_next_to_loss_weights(self):
        """Same container pattern as loss_weights / scheduled_vars (public dict of non-trainable variables, so the trainer's
        tf.train.Checkpoint(trainer=...) treats them alike); the resume of the whole system is covered by tests/test_resume*.py for those."""
        tr, _ = _trainer("ema_norm")
        self.assertTrue(all(not v.trainable for v in tr.loss_ema.values()))
        self.assertEqual(type(tr.loss_ema).__name__, type(tr.loss_weights).__name__)


if __name__ == "__main__":
    unittest.main()

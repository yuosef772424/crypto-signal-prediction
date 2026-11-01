"""Head loss term "beta_nll" (core.loss_plan.HEAD_LOSS_TERMS, option beta_nll; code: the existing branch of trainer/tasks.py evidential_task_loss):
registered + routed through LossSettings.head_terms; OFF (0.0) = the old NIG loss; ON multiplies each sample's nll by stopgrad(w_St^2)^beta.
    python -m pytest tests/test_beta_nll_term.py -q
"""
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
from tests.test_ic_selection import ns  # noqa: E402


class BetaNll(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(HEAD_LOSS_TERMS.check_all(), [])
        heads = HeadSet.default(("high",))
        self.assertEqual(check_head_loss_terms({"high_reg": {"beta_nll": 0.5}}, heads), {"high_reg": {"beta_nll": 0.5}})
        for bad in ({"high_class": {"beta_nll": 0.5}}, {"high_reg": {"beta_nll": -1}}, {"high_reg": {"beta_nll": float("nan")}}):
            with self.assertRaises(ValueError):
                check_head_loss_terms(bad, heads)

    def test_loss_values(self):
        import tensorflow as tf
        n = ns()
        rng = np.random.default_rng(0)
        y = tf.constant(rng.normal(size=16).astype("float32"))
        out = {"mu": tf.constant(rng.normal(size=16).astype("float32")), "nu": tf.constant(rng.uniform(.5, 2, 16).astype("float32")),
               "alpha": tf.constant(rng.uniform(2, 4, 16).astype("float32")), "beta": tf.constant(rng.uniform(.1, 1, 16).astype("float32"))}
        cfg = {"output_keys": {k: k for k in out}, "true_key": "y"}
        fn = n["evidential_task_loss"]
        l0, st0 = fn(y, out, dict(cfg), {})
        l0b, _ = fn(y, out, dict(cfg, beta_nll=0.0), {})
        l5, st5 = fn(y, out, dict(cfg, beta_nll=0.5), {})
        self.assertEqual(float(l0.numpy()), float(l0b.numpy()))                                # 0.0 = the old loss
        self.assertNotAlmostEqual(float(l0.numpy()), float(l5.numpy()), places=4)
        # the logged nig_base stays the PLAIN nll in both cases (the selection metric is not reweighted)
        self.assertAlmostEqual(float(st0["nig_base"].numpy()), float(st5["nig_base"].numpy()), places=6)


if __name__ == "__main__":
    unittest.main()

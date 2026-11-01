"""Optional NIG head variant mu_skip='unc_gated' (model/builder.py): mu = mu + sigmoid(Dense(stop_gradient(epistemic))) * skip, skip =
Dense(128, swish)(MHA(query = mean over time, key = value = inputs))[:, 0, 0:1].

Guards: None (the default) builds the SAME model as before (same weights, same outputs for the same seed); unknown values raise; the variant
keeps the output vocabulary (HeadSet.output_names), adds gate layers, gates are in (0, 1), the skip params get gradient from mu, the gate does NOT
send gradient into epistemic (stop_gradient), unc_gate_model returns the gates; the key is in MODEL_CONFIG (so ModelSettings.config_overrides accepts it).
    python -m pytest tests/test_mu_skip.py -q
"""
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

import _nbload  # noqa: E402
from core.heads import HeadSet  # noqa: E402

_NS = {}


def ns():
    if not _NS:
        _nbload.load_model(_NS)
    return _NS


def _model(**cfg):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(3)
    base = dict(encoder="gru", d_model=16, num_layers=1, head_hidden=16, class_head_hidden=8, price_targets=("high", "low", "close"),
                head_types=HeadSet.default(("high", "low", "close")).head_types(), enforce_order=False)
    return ns()["build_model_fn"](12, 5, {**base, **cfg})


class Variant(unittest.TestCase):
    def test_default_is_the_old_model(self):
        a, b = _model(), _model(mu_skip=None)
        self.assertEqual([tuple(w.shape) for w in a.weights], [tuple(w.shape) for w in b.weights])
        x = np.random.default_rng(0).normal(size=(4, 12, 5)).astype("float32")
        for k, v in a(x, training=False).items():
            np.testing.assert_allclose(v.numpy(), b(x, training=False)[k].numpy(), rtol=1e-6)
        self.assertFalse(any("skip" in w.path or "unc_gate" in w.path for w in a.weights))
        self.assertIsNone(ns()["MODEL_CONFIG"]["mu_skip"])

    def test_unknown_value_raises(self):
        with self.assertRaises(ValueError):
            _model(mu_skip="gated")

    def test_variant_keeps_the_vocabulary_and_exposes_the_gates(self):
        import tensorflow as tf
        m = _model(mu_skip="unc_gated")
        self.assertEqual(sorted(m(np.zeros((1, 12, 5), "float32"), training=False)), sorted(HeadSet.default(("high", "low", "close")).output_names()))
        x = np.random.default_rng(1).normal(size=(8, 12, 5)).astype("float32")
        gates = ns()["unc_gate_model"](m)(x, training=False)
        self.assertEqual(sorted(gates), ["close", "high", "low"])
        for g in gates.values():
            g = g.numpy()
            self.assertEqual(g.shape, (8, 1))
            self.assertTrue(np.all((g > 0) & (g < 1)))
        with self.assertRaises(ValueError):
            ns()["unc_gate_model"](_model())
        with tf.GradientTape() as tape:                                     # mu reaches the skip path
            out = m(x, training=True)
            loss = tf.reduce_sum(out["y_close"] ** 2)
        grads = dict(zip([v.path for v in m.trainable_variables], tape.gradient(loss, m.trainable_variables)))
        skip = [g for n, g in grads.items() if "skip_fc_close" in n or "unc_gate_close" in n]
        self.assertTrue(skip and all(g is not None and float(tf.reduce_sum(tf.abs(g))) > 0 for g in skip))

    def test_the_gate_cannot_push_epistemic(self):
        import tensorflow as tf
        m = _model(mu_skip="unc_gated")
        x = np.random.default_rng(2).normal(size=(8, 12, 5)).astype("float32")
        gate_m = ns()["unc_gate_model"](m)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(gate_m(x, training=True)["close"])
        names = [v.path for v in m.trainable_variables]
        grads = dict(zip(names, tape.gradient(loss, m.trainable_variables)))
        # the gate's own Dense gets gradient; the NIG head that PRODUCES nu/alpha/beta (nig_close) gets none through the gate (stop_gradient)
        self.assertTrue(any(g is not None for n, g in grads.items() if "unc_gate_close" in n))
        self.assertTrue(all(g is None for n, g in grads.items() if "nig_close" in n))


if __name__ == "__main__":
    unittest.main()

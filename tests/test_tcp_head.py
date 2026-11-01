"""TCP confidence head (model option tcp_head, LossSettings.tcp, GenericTrainer._tcp_loss, stats.confidence_quality): a small MLP on
stop_gradient(trunk features) trained with a Brier loss to be P(sign(mu_close) == sign(y_close)); weight 0 before warmup_epochs.

Guards: options validated; the schedule (0 until warmup_epochs, then the weight); the default model has no tcp weights and the SAME outputs;
the variant adds the output tcp_close in (0, 1) without a registered-head name; tcp gradient reaches the tcp MLP and nothing else; the loss
equals a numpy Brier; OFF = the old loss exactly and weight 0 adds nothing; settings require loss.tcp and the model option together;
confidence_quality matches hand values.
    python -m pytest tests/test_tcp_head.py -q
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
from core.loss_plan import TCP_OUTPUT_KEY, check_loss_schedules, check_tcp, tcp_plan  # noqa: E402
from stats.metrics import confidence_quality  # noqa: E402
from tests.test_ic_selection import _SCHED, _data, _toy, ns  # noqa: E402

_M = {}


def _build(**cfg):
    import tensorflow as tf
    if not _M:
        _nbload.load_model(_M)
    tf.keras.utils.set_random_seed(3)
    base = dict(encoder="gru", d_model=16, num_layers=1, head_hidden=16, class_head_hidden=8, price_targets=("high", "low", "close"),
                head_types=HeadSet.default(("high", "low", "close")).head_types(), enforce_order=False)
    return _M["build_model_fn"](12, 5, {**base, **cfg})


class Options(unittest.TestCase):
    def test_validation_and_schedule(self):
        self.assertEqual(check_tcp({}), {})
        self.assertEqual(check_tcp({"weight": 0.2}), {"weight": 0.2, "warmup_epochs": 10})
        for bad in ({"weight": 0}, {"weight": -1}, {"weight": 0.1, "nope": 1}, {"weight": 0.1, "warmup_epochs": 1.5}, {"weight": float("nan")}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check_tcp(bad)
        section, s = tcp_plan(check_tcp({"weight": 0.2, "warmup_epochs": 3}))
        self.assertEqual(section["output_key"], TCP_OUTPUT_KEY)
        sched = ns()["schedule_value"]
        vals = [sched(s["start"], s["end"], s["warmup_epochs"], s["schedule"], e, s["delay_epochs"]) for e in range(6)]
        np.testing.assert_allclose(vals, [0, 0, 0, 0.2, 0.2, 0.2])
        self.assertEqual(check_loss_schedules({**_SCHED, "lambda_tcp": s})["lambda_tcp"]["delay_epochs"], 3)


class Model(unittest.TestCase):
    def test_default_is_the_old_model(self):
        a, b = _build(), _build(tcp_head=False)
        self.assertEqual([tuple(w.shape) for w in a.weights], [tuple(w.shape) for w in b.weights])
        self.assertNotIn(TCP_OUTPUT_KEY, a.output)
        self.assertFalse(any("tcp" in w.path for w in a.weights))
        self.assertFalse(_M["MODEL_CONFIG"]["tcp_head"])

    def test_variant_adds_one_output_and_only_the_tcp_mlp_gets_its_gradient(self):
        import tensorflow as tf
        m = _build(tcp_head=True)
        x = np.random.default_rng(1).normal(size=(8, 12, 5)).astype("float32")
        out = m(x, training=False)
        self.assertEqual(sorted(set(out) - {TCP_OUTPUT_KEY}), sorted(HeadSet.default(("high", "low", "close")).output_names()))
        p = out[TCP_OUTPUT_KEY].numpy()
        self.assertEqual(p.shape, (8, 1))
        self.assertTrue(np.all((p > 0) & (p < 1)))
        with self.assertRaises(ValueError):
            from core.heads import parse_output_name
            parse_output_name(TCP_OUTPUT_KEY)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(m(x, training=True)[TCP_OUTPUT_KEY])
        grads = dict(zip([v.path for v in m.trainable_variables], tape.gradient(loss, m.trainable_variables)))
        self.assertTrue(any(g is not None for n, g in grads.items() if "tcp" in n))
        self.assertTrue(all(g is None for n, g in grads.items() if "tcp" not in n))


def _trainer(tcp):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(7)
    heads = HeadSet.default(("high",))
    _, base = _toy(tf, 7)
    hid = base.layers[1].output
    outs = dict(base.output)
    outs[TCP_OUTPUT_KEY] = tf.keras.layers.Dense(1, activation="sigmoid", name="tcp_prob")(tf.keras.layers.Lambda(tf.stop_gradient)(hid))
    model = tf.keras.Model(base.inputs, outs)
    loss = {"use_uncertainty_weighting": False, "schedules": dict(_SCHED)}
    if tcp:
        section, sched = tcp_plan(check_tcp({"weight": 0.5, "warmup_epochs": 2}))
        loss["tcp"] = {**section, "head": "high_reg"}
        loss["schedules"]["lambda_tcp"] = sched
    cfg = ns()["build_config"]({"run": {"run_dir": "/tmp/tcp_unused", "epochs": 1, "batch_size": 32, "verbose": 0},
                                "targets": heads.target_configs(), "loss": loss})
    x, y = _data(64, 1)
    return ns()["GenericTrainer"](model, cfg), (tf.constant(x[:32]), {k: tf.constant(v[:32]) for k, v in y.items()})


class Loss(unittest.TestCase):
    def test_brier_against_numpy_and_off_and_warmup(self):
        tr, (x, y) = _trainer(True)
        out = tr.model(x, training=False)
        mu = out["y_high"].numpy().ravel().astype("float64")
        yy = y["y_high_reg"].numpy().ravel().astype("float64")
        p = out[TCP_OUTPUT_KEY].numpy().ravel().astype("float64")
        expect = float(np.mean((p - (mu * yy > 0)) ** 2))
        self.assertAlmostEqual(float(tr._tcp_loss(y, out).numpy()), expect, places=5)
        off, _ = _trainer(False)
        self.assertIsNone(off.tcp)
        _, raw_off, _, _, _ = off._compute_losses(x, y, None, training=True)
        tr.scheduled_vars["lambda_tcp"].assign(0.0)                      # before warmup_epochs: nothing added
        _, raw0, _, _, _ = tr._compute_losses(x, y, None, training=True)
        np.testing.assert_allclose(raw0.numpy(), raw_off.numpy(), rtol=1e-6)
        tr.scheduled_vars["lambda_tcp"].assign(0.5)
        total, _, _, _, _ = tr._compute_losses(x, y, None, training=True)
        self.assertAlmostEqual(float(total.numpy()), float(raw_off.numpy()) + 0.5 * float(tr._tcp_loss(y, tr.model(x, training=True)).numpy()),
                               delta=1e-3)


class Route(unittest.TestCase):
    def test_settings_need_both_halves(self):
        from tests import run_fixture as rf
        S = rf.run_namespace()["RunSettings"]
        ok = S().updated({"model": {"config_overrides": {"tcp_head": True}}, "train": {"loss": {"tcp": {"weight": 0.1}}}})
        self.assertEqual(ok.train.loss.tcp, {"weight": 0.1, "warmup_epochs": 10})
        for bad in ({"train": {"loss": {"tcp": {"weight": 0.1}}}}, {"model": {"config_overrides": {"tcp_head": True}}}):
            with self.assertRaisesRegex(ValueError, "go together"):
                S().updated(bad)
        self.assertEqual(S().train.loss.tcp, {})


class Quality(unittest.TestCase):
    def test_confidence_quality(self):
        mu = np.array([1, 1, -1, -1, 1.0])
        y = np.array([1, -1, -1, 1, 1.0])                                # correct: 1 0 1 0 1
        good = np.array([0.9, 0.2, 0.8, 0.1, 0.7])                       # ranks the correct rows first
        r = confidence_quality({"good": good, "flat": np.full(5, 0.5)}, mu, y)
        self.assertAlmostEqual(r["accuracy"], 0.6)
        self.assertAlmostEqual(r["good"]["auroc"], 1.0)
        self.assertAlmostEqual(r["good"]["risk_coverage_auc"], 0.13)       # errors only at ranks 4, 5: mean(0,0,0,1/4,2/5)
        self.assertAlmostEqual(r["flat"]["auroc"], 0.5)
        with self.assertRaises(ValueError):
            confidence_quality({"x": np.zeros(3)}, mu, y)

    def test_evaluation_report_reads_both_outputs(self):
        ev = {}
        _nbload.load_evaluation(ev)
        rng = np.random.default_rng(0)
        y = rng.normal(size=50)
        preds = {"y_close": y + rng.normal(size=50), "y_close_confidence": rng.uniform(size=50), TCP_OUTPUT_KEY: rng.uniform(size=50)}
        r = ev["confidence_quality_report"](preds, y)
        self.assertEqual(sorted(r), ["accuracy", "confidence", "tcp"])
        del preds[TCP_OUTPUT_KEY]
        self.assertEqual(sorted(ev["confidence_quality_report"](preds, y)), ["accuracy", "confidence"])


if __name__ == "__main__":
    unittest.main()

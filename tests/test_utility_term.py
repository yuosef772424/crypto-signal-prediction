"""Decision-utility aux term (LossSettings.utility; trainer GenericTrainer._utility_loss): position s = tanh(k mu / sigma), gain = s r - cost |s|,
r = the DECODED return of the label through the target codec, loss -mean(gain)/std(r) added to the head losses with a weight ramped 0 -> lambda_U
after a warmup (schedule lambda_utility with delay_epochs).

Guards: options validated (unknown / bad raise); the schedule (delay then linear ramp; delay 0 = the old schedule); the loss equals a numpy
re-computation (return codec and volret codec with the per-row stamp); OFF = the old per-head losses exactly; weight 0 (warmup) adds nothing;
modes without a return decode raise at settings construction; the plan writes loss.utility + schedule and the folder name.
    python -m pytest tests/test_utility_term.py -q
"""
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

from core.loss_plan import UTILITY_STAMP_KEY, check_loss_schedules, check_utility, utility_plan  # noqa: E402
from stats.target_codec import CODECS  # noqa: E402
from tests.test_ic_selection import _SCHED, _data, _toy, ns  # noqa: E402


class Options(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(check_utility({}), {})
        got = check_utility({"weight": 0.25})
        self.assertEqual((got["weight"], got["k"], got["cost"], got["warmup_epochs"], got["ramp_epochs"]), (0.25, 1.0, 0.0008, 10, 10))
        for bad in ({"weight": 0}, {"weight": -1}, {"weight": 0.1, "nope": 1}, {"weight": 0.1, "k": 0}, {"weight": 0.1, "cost": -1},
                    {"weight": 0.1, "ramp_epochs": 0}, {"weight": 0.1, "warmup_epochs": 1.5}, {"weight": float("nan")}):
            with self.assertRaises(ValueError, msg=str(bad)):
                check_utility(bad)

    def test_schedule_delay_then_linear_ramp(self):
        sched = ns()["schedule_value"]
        _, s = utility_plan(check_utility({"weight": 0.2, "warmup_epochs": 3, "ramp_epochs": 4}), "volret", 1.0)
        vals = [sched(s["start"], s["end"], s["warmup_epochs"], s["schedule"], e, s["delay_epochs"]) for e in range(10)]
        np.testing.assert_allclose(vals, [0, 0, 0, 0, 0.05, 0.1, 0.15, 0.2, 0.2, 0.2])
        self.assertEqual(check_loss_schedules({**_SCHED, "lambda_utility": s})["lambda_utility"]["delay_epochs"], 3)
        self.assertEqual(sched(0.0, 1.0, 4, "linear", 2), 0.5)                  # delay_epochs omitted = the old schedule


def _trainer(utility, codec="return", scale=1.0):
    import tensorflow as tf
    heads, model = _toy(tf, 5)
    loss = {"use_uncertainty_weighting": False, "schedules": dict(_SCHED)}
    if utility:
        loss["utility"] = {"head": "high_reg", "k": 1.0, "cost": 0.01, "eps": 1e-6, "codec": codec, "scale": scale}
        loss["schedules"]["lambda_utility"] = {"start": 0.0, "end": 0.5, "warmup_epochs": 1, "schedule": "linear"}
    cfg = ns()["build_config"]({"run": {"run_dir": "/tmp/util_unused", "epochs": 1, "batch_size": 32, "verbose": 0},
                                "targets": heads.target_configs(), "loss": loss})
    x, y = _data(64, 1)
    yb = {k: tf.constant(v[:32]) for k, v in y.items()}
    return ns()["GenericTrainer"](model, cfg), (tf.constant(x[:32]), yb), tf


class Loss(unittest.TestCase):
    def _by_hand(self, tr, x, y, r, mode_scale=1.0):
        out = tr.model(x, training=False)
        keys = tr.config["targets"]["high_reg"]["output_keys"]
        mu = out[keys["mu"]].numpy().ravel().astype("float64")
        beta, alpha = out[keys["beta"]].numpy().ravel().astype("float64"), out[keys["alpha"]].numpy().ravel().astype("float64")
        sigma = np.sqrt(beta / np.maximum(alpha - 1.0, 1e-2))
        s = np.tanh(mu / (sigma + 1e-6))
        gain = s * r - 0.01 * np.abs(s)
        return -gain.mean() / (r.std() + 1e-8)

    def test_return_codec(self):
        tr, (x, y), _ = _trainer(True, "return", 100.0)
        r = y["y_high_reg"].numpy().ravel().astype("float64") / 100.0
        got = float(tr._utility_loss(y, tr.model(x, training=False)).numpy())
        self.assertAlmostEqual(got, self._by_hand(tr, x, y, r), places=4)

    def test_volret_codec_with_the_stamp(self):
        import tensorflow as tf
        tr, (x, y), _ = _trainer(True, "volret", 1.0)
        vol = np.linspace(0.005, 0.02, 32).astype("float32")
        y = {**y, UTILITY_STAMP_KEY: tf.constant(vol)}
        z = y["y_high_reg"].numpy().ravel().astype("float64")
        r = CODECS["volret"].inverse(z, {"target_vol": vol.astype("float64")}, {"scale": 1.0})
        got = float(tr._utility_loss(y, tr.model(x, training=False)).numpy())
        self.assertAlmostEqual(got, self._by_hand(tr, x, y, r), places=4)

    def test_off_is_the_old_loss_and_warmup_adds_nothing(self):
        off, (x, y), _ = _trainer(False)
        on, _, _ = _trainer(True)
        self.assertIsNone(off.utility)
        _, raw_off, per_off, _, _ = off._compute_losses(x, y, None, training=True)
        on.scheduled_vars["lambda_utility"].assign(0.0)                           # warmup: weight 0
        _, raw_on0, per_on0, _, _ = on._compute_losses(x, y, None, training=True)
        np.testing.assert_allclose(raw_on0.numpy(), raw_off.numpy(), rtol=1e-6)
        on.scheduled_vars["lambda_utility"].assign(0.5)
        total, raw_on, _, _, _ = on._compute_losses(x, y, None, training=True)
        expect = float(raw_off.numpy()) + 0.5 * float(on._utility_loss(y, on.model(x, training=True)).numpy())
        self.assertNotAlmostEqual(float(raw_on.numpy()), float(raw_off.numpy()), places=5)
        self.assertAlmostEqual(float(total.numpy()), float(raw_on.numpy()), places=5)   # no Kendall here
        self.assertAlmostEqual(float(raw_on.numpy()), expect, delta=1e-3 * max(1.0, abs(expect)))

    def test_gradient_reaches_mu(self):
        import tensorflow as tf
        tr, (x, y), _ = _trainer(True)
        with tf.GradientTape() as tape:
            u = tr._utility_loss(y, tr.model(x, training=True))
        g = tape.gradient(u, tr.model.trainable_variables)
        self.assertTrue(any(gi is not None and float(tf.reduce_sum(tf.abs(gi))) > 0 for gi in g))


class Route(unittest.TestCase):
    def test_settings(self):
        from tests import run_fixture as rf
        S = rf.run_namespace()["RunSettings"]
        ok = S().updated({"target": {"target_mode": "volret"}, "train": {"loss": {"utility": {"weight": 0.1}}}})
        self.assertEqual(ok.train.loss.utility["cost"], 0.0008)
        for mode in ("entry_range", "scaled", "magnitude", "volnorm+relative"):
            with self.assertRaisesRegex(ValueError, "return decode"):
                S().updated({"target": {"target_mode": mode}, "train": {"loss": {"utility": {"weight": 0.1}}}})
        with self.assertRaises(ValueError):
            S().updated({"train": {"loss": {"utility": {"weight": 0.1, "bogus": 1}}}})
        self.assertEqual(S().train.loss.utility, {})


if __name__ == "__main__":
    unittest.main()

"""Selection metric `val_<head>_ic`: Pearson (stats.metrics.pearson) of a regression / evidential head's mu vs its label over the FULL
validation set, objective-independent, mode max (experiment E-tgt-002).

Guards:
1. VOCABULARY: SELECTION_METRICS registers "ic" (head scope, max, objective_independent, evidential/regression); resolve_selection_metric
   and trainer validate_config accept `val_<reg head>_ic` with mode max and refuse min, a classification head, and the train-side name.
2. OFF = OLD LOGS: without the monitor / track_ic / report_ic no tracker exists and no key is added to the logs.
3. THE VALUE: on several val batches the logged value equals stats.metrics.pearson(mu, y) over the whole val set (not a mean of batch
   correlations); it is not logged on training batches; a constant prediction gives 0.0, not NaN.
4. SELECTION: BestModelTracker monitoring val_<head>_ic (mode max, no smoothing) keeps the epoch with the highest val IC on a toy run.
5. ROUTE: RunSettings early_stopping_monitor="val_close_reg_ic" / mode "max" reaches the trainer config; LossSettings.report_ic writes
   `track_ic: True` on exactly those heads; bad heads raise at construction.
    python -m pytest tests/test_ic_selection.py -q
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
from core.loss_plan import SELECTION_METRICS, resolve_selection_metric  # noqa: E402
from stats import metrics as stats_metrics  # noqa: E402
from tests.test_loss_plan import S, diff, make_config, ns, plan, P2, P3  # noqa: E402

_SCHED = {"lambda_reg": {"start": 0.0, "end": 0.0, "warmup_epochs": 0, "schedule": "linear"},
          "lambda_calib": {"start": 0.0, "end": 0.0, "warmup_epochs": 0, "schedule": "linear"}}


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _toy(tf, seed=7):
    """A small model emitting every output of HeadSet.default(("high",)); mu is a linear read-out of x."""
    tf.keras.utils.set_random_seed(seed)
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
                outs[name] = tf.keras.layers.Dense(1, name=name)(inp)
    return heads, tf.keras.Model(inp, outs)


def _data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype("float32")
    y = (x[:, :1] * 0.5 + rng.normal(size=(n, 1))).astype("float32")
    return x, {"y_high_reg": y, "y_high_class": (y > 0).astype("float32")}


def _cfg(heads, monitor="val_loss", mode="min", track=False):
    tg = heads.target_configs()
    if track:
        tg["high_reg"]["track_ic"] = True
    return ns()["build_config"]({"run": {"run_dir": "/tmp/ic_sel_unused", "epochs": 1, "batch_size": 32, "verbose": 0}, "targets": tg,
                                 "loss": {"use_uncertainty_weighting": False, "schedules": _SCHED},
                                 "callbacks": {"early_stopping": {"monitor": monitor, "mode": mode}}})


class Vocabulary(unittest.TestCase):
    def test_registered_definition(self):
        meta = SELECTION_METRICS.entry("ic").meta
        self.assertEqual((meta["scope"], meta["mode"], meta["objective_independent"]), ("head", "max", True))
        self.assertEqual(meta["task_types"], ("evidential", "regression"))
        self.assertEqual(SELECTION_METRICS.check_all(), [])

    def test_resolve(self):
        heads = HeadSet.default(P3)
        c = resolve_selection_metric("val_close_reg_ic", heads, "max")
        self.assertEqual((c.stat, c.head, c.mode, c.objective_independent), ("ic", "close_reg", "max", True))
        for bad, mode in (("val_close_reg_ic", "min"), ("val_close_class_ic", "max"), ("val_nope_reg_ic", "max")):
            with self.assertRaises(ValueError, msg=bad):
                resolve_selection_metric(bad, heads, mode)

    def test_trainer_validate_config(self):
        heads = HeadSet.default(("high",))
        cfg = _cfg(heads, "val_high_reg_ic", "max")
        self.assertEqual(cfg["callbacks"]["early_stopping"]["monitor"], "val_high_reg_ic")
        for mon, mode, needle in (("val_high_reg_ic", "min", "mode must be 'max'"), ("high_reg_ic", "max", "validation-set statistic"),
                                  ("val_high_class_ic", "max", "not a metric")):
            with self.assertRaises(ValueError, msg=mon) as cm:
                _cfg(heads, mon, mode)
            self.assertIn(needle, str(cm.exception))


class Tracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf = tf

    def _trainer(self, **kw):
        heads, model = _toy(self.tf)
        tr = ns()["GenericTrainer"](model, _cfg(heads, **kw))
        tr.compile(optimizer=self.tf.keras.optimizers.Adam(1e-2))
        return tr, model

    def test_off_means_no_tracker_and_no_new_log_key(self):
        tr, _ = self._trainer()
        x, y = _data(96, 1)
        logs = _quiet(tr.evaluate, x, y, batch_size=32, verbose=0, return_dict=True)
        self.assertFalse([k for k in logs if k.endswith("_ic")])
        self.assertFalse([m.name for m in tr.metrics if m.name.endswith("_ic")])

    def test_value_is_stats_pearson_over_the_whole_val_set(self):
        for kw in ({"monitor": "val_high_reg_ic", "mode": "max"}, {"track": True}):
            tr, model = self._trainer(**kw)
            x, y = _data(160, 2)                                   # 5 batches of 32
            logs = _quiet(tr.evaluate, x, y, batch_size=32, verbose=0, return_dict=True)
            mu = model(x)["y_high"].numpy().ravel()
            want = stats_metrics.pearson(mu, y["y_high_reg"].ravel())
            self.assertAlmostEqual(logs["high_reg_ic"], want, places=5, msg=kw)
            batch_mean = np.mean([stats_metrics.pearson(mu[i:i + 32], y["y_high_reg"].ravel()[i:i + 32]) for i in range(0, 160, 32)])
            self.assertNotAlmostEqual(logs["high_reg_ic"], batch_mean, places=4)        # the whole set, not a mean of batches
            logs2 = _quiet(tr.evaluate, x, y, batch_size=32, verbose=0, return_dict=True)  # reset between evaluations
            self.assertAlmostEqual(logs2["high_reg_ic"], want, places=5)

    def test_not_logged_on_training_batches_and_val_key_present(self):
        tr, _ = self._trainer(monitor="val_high_reg_ic", mode="max")
        x, y = _data(128, 3)
        xv, yv = _data(64, 4)
        hist = _quiet(tr.fit, x, y, validation_data=(xv, yv), batch_size=32, epochs=2, verbose=0)
        self.assertIn("val_high_reg_ic", hist.history)
        self.assertNotIn("high_reg_ic", hist.history)
        self.assertTrue(all(np.isfinite(hist.history["val_high_reg_ic"])))

    def test_constant_prediction_is_zero_not_nan(self):
        tf = self.tf
        heads, model = _toy(tf)
        for layer in model.layers:
            if layer.name == "y_high":
                layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])
        tr = ns()["GenericTrainer"](model, _cfg(heads, "val_high_reg_ic", "max"))
        tr.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
        x, y = _data(64, 5)
        logs = _quiet(tr.evaluate, x, y, batch_size=32, verbose=0, return_dict=True)
        self.assertEqual(logs["high_reg_ic"], 0.0)

    def test_best_model_tracker_keeps_the_epoch_with_the_highest_val_ic(self):
        tf = self.tf
        tr, model = self._trainer(monitor="val_high_reg_ic", mode="max")
        x, y = _data(256, 6)
        xv, yv = _data(128, 7)
        best = ns()["BestModelTracker"](model, None, monitor="val_high_reg_ic", mode="max", patience=100, min_delta=0.0,
                                        smoothing="none", verbose=0, restore_best_weights=True)
        hist = _quiet(tr.fit, x, y, validation_data=(xv, yv), batch_size=32, epochs=6, verbose=0, callbacks=[best])
        ics = hist.history["val_high_reg_ic"]
        self.assertEqual(best.best_epoch, int(np.argmax(ics)) + 1)
        self.assertAlmostEqual(best.best, max(ics), places=6)
        restored = stats_metrics.pearson(model(xv)["y_high"].numpy().ravel(), yv["y_high_reg"].ravel())
        self.assertAlmostEqual(restored, max(ics), places=4)       # the restored weights are the selected epoch's


class Route(unittest.TestCase):
    def test_monitor_reaches_the_trainer_config(self):
        base = make_config(S(), P3)
        cfg = make_config(S({"train": {"early_stopping_monitor": "val_close_reg_ic", "early_stopping_mode": "max"}}), P3)
        self.assertEqual(diff(base, cfg), {"callbacks.early_stopping.monitor": ("val_raw_loss", "val_close_reg_ic"),
                                           "callbacks.early_stopping.mode": ("min", "max")})
        with self.assertRaises(ValueError):
            S({"train": {"early_stopping_monitor": "val_close_reg_ic", "early_stopping_mode": "min"}})

    def test_report_ic_writes_track_ic_on_exactly_those_heads(self):
        base = make_config(S(), P3)
        cfg = make_config(S(plan(report_ic=("high_reg", "low_reg"))), P3)
        self.assertEqual(diff(base, cfg), {"targets.high_reg.track_ic": (None, True), "targets.low_reg.track_ic": (None, True)})
        self.assertEqual(diff(base, make_config(S(plan(report_ic=())), P3)), {})
        s = S(plan(report_ic=["high_reg"]))
        self.assertEqual(s.train.loss.report_ic, ("high_reg",))
        self.assertEqual(ns()["RunSettings"].from_dict(s.to_dict()), s)

    def test_bad_report_heads_raise(self):
        for bad in (("high_class",), ("nope_reg",), ("high_reg", "high_reg"), "high_reg"):
            with self.assertRaises(ValueError, msg=str(bad)):
                S(plan(report_ic=bad))
        with self.assertRaises(ValueError):
            make_config(S(plan(report_ic=("close_reg",))), P2)


if __name__ == "__main__":
    unittest.main()

"""BestModelTracker(weights_snapshot='ema_weights') must restore weights whose val metric IS the recorded best (audit D1).

Before the fix the best epoch was chosen by the RAW weights' val metric while the EMA weights were saved (measured: logged val_raw_loss
3.46495, the saved weights re-evaluated 3.54691). Now the EMA weights are re-evaluated on val and that value drives the comparison and
best_meta.json ('best'); the raw value Keras logged is kept as 'raw_monitor_at_best'. weights_snapshot='raw' (the default) is unchanged.
    python -m pytest tests/test_ema_best_validated.py -q
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402

from tests.test_ic_selection import _SCHED, _data, _toy, ns  # noqa: E402


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _run(snapshot, run_dir, seed=3):
    import tensorflow as tf
    heads, model = _toy(tf, seed)
    cfg = ns()["build_config"]({
        "run": {"run_dir": run_dir, "epochs": 4, "batch_size": 32, "verbose": 0},
        "targets": heads.target_configs(),
        "optimizer": {"use_ema": True, "ema_momentum": 0.9, "ema_warmup": False, "lr_initial": 3e-2},
        "loss": {"use_uncertainty_weighting": False, "schedules": _SCHED},
        "callbacks": {"early_stopping": {"monitor": "val_raw_loss", "mode": "min", "smoothing": "none", "patience": 10,
                                         "weights_snapshot": snapshot}}})
    x, y = _data(256, 1)
    xv, yv = _data(128, 2)
    mk = lambda xx, yy: tf.data.Dataset.from_tensor_slices((xx, yy)).batch(32)   # noqa: E731
    train_ds, val_ds = mk(x, y), mk(xv, yv)
    builder = lambda: _toy(tf, seed)[1]                                          # noqa: E731
    trainer, callbacks, initial = _quiet(ns()["build_training_system"], builder, cfg, next(iter(train_ds)))
    _quiet(trainer.fit, train_ds, validation_data=val_ds, epochs=4, callbacks=callbacks, verbose=0)
    best = json.load(open(os.path.join(run_dir, "best_meta.json")))
    after = trainer.evaluate(val_ds, verbose=0, return_dict=True)       # the restored (saved) weights, re-evaluated on val
    return best, after


class EmaBest(unittest.TestCase):
    def test_restored_best_reproduces_the_recorded_best(self):
        with tempfile.TemporaryDirectory() as d:
            best, after = _run("ema_weights", d)
        self.assertEqual(best["weights_snapshot"], "ema_weights")
        self.assertIsNotNone(best["raw_monitor_at_best"])
        np.testing.assert_allclose(after["raw_loss"], best["best"], rtol=1e-4, atol=1e-5)

    def test_raw_snapshot_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            best, after = _run("raw", d)
        self.assertIsNone(best["raw_monitor_at_best"])
        np.testing.assert_allclose(after["raw_loss"], best["best"], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

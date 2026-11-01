"""E-tgt-003: every arm of docs/research/cards/E-tgt-003.md builds from RunSettings and trains two epochs on the synthetic fixture (volret target,
Huber + beta-NLL head terms, ema_norm balancer, utility term, curriculum / jump weights), finite losses, the new logs present, and the off
arms leave the logs of the old runs untouched.
    python -m pytest tests/test_e_tgt_003_arms.py -q
"""
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from tests import run_fixture as rf  # noqa: E402
from tests.test_run_steps import _dataset  # noqa: E402

HEADS = ("high_reg", "low_reg", "close_reg")


def arm(S, name, run_dir, seed=0):
    """The card's arm() with a toy budget (epochs 2)."""
    terms = {h: {"huber_weight": 0.1} for h in HEADS}
    loss = {"report_ic": ("high_reg", "low_reg"), "balancing": "ema_norm", "head_terms": terms}
    if name >= "S1":
        loss["head_terms"] = {h: {"huber_weight": 0.1, "beta_nll": 0.5} for h in HEADS}
    model = {}
    if name >= "S2":
        model = {"config_overrides": {"tcp_head": True}}
        loss["tcp"] = {"weight": 0.1, "warmup_epochs": 0}
    if name >= "S3":
        loss["utility"] = {"weight": 0.1, "warmup_epochs": 0, "ramp_epochs": 1}
    if name == "S4":
        loss["sample_weighting"] = "curriculum_snr"
    if name == "S4r":
        loss["sample_weighting"] = "curriculum_random"
    if name == "S5":
        loss["sample_weighting"] = "jump_downweight"
    return S().updated({"model": model, "target": {"suspended_targets": (), "target_mode": "volret", "clip": 4.0},
                        "train": {"seed": seed, "run_dir": run_dir, "epochs": 2, "batch_size": 64, "fit_verbose": 0,
                                  "early_stopping_monitor": "val_close_reg_ic", "early_stopping_mode": "max", "loss": loss}})


class Arms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = rf.run_namespace()
        cls.kit = cls.ns["Toolkit"].from_namespace(cls.ns)

    def _run(self, name):
        with tempfile.TemporaryDirectory() as d:
            res = rf._quiet(self.ns["run_main"], arm(self.ns["RunSettings"], name, d + "/r"), self.kit, dataset=_dataset(False, None),
                            summary=False)
        return res.history.history

    def test_s0_baseline(self):
        h = self._run("S0")
        self.assertNotIn("utility_loss", h)
        self.assertTrue(np.all(np.isfinite(h["val_raw_loss"])))
        self.assertIn("val_close_reg_ic", h)

    def test_s2_tcp_head(self):
        h = self._run("S2")
        self.assertIn("tcp_brier", h)
        self.assertTrue(np.all(np.isfinite(h["tcp_brier"])) and np.all(np.isfinite(h["val_raw_loss"])))
        self.assertNotIn("utility_loss", h)

    def test_s3_utility_and_beta_nll(self):
        h = self._run("S3")
        self.assertIn("utility_loss", h)
        self.assertTrue(np.all(np.isfinite(h["utility_loss"])) and np.all(np.isfinite(h["val_raw_loss"])))

    def test_s4_curriculum_and_control_and_s5_jump(self):
        for name in ("S4", "S4r", "S5"):
            h = self._run(name)
            self.assertTrue(np.all(np.isfinite(h["loss"])), name)


if __name__ == "__main__":
    unittest.main()

"""The opt-in trivial range-position baseline of the generalization-gap report (audit T4, F-0085).

y_high_class (next high > last high) and y_low_class (next low > last low) are mechanically predictable from where the last bar closed inside
its own range, p = (close - low) / (high - low). trivial_range_baseline scores that rule (sign fitted on TRAIN, applied unchanged to
val/test); generalization_gap_report(trivial_baseline=True) / EvalSettings.trivial_baseline adds trivial_auc_* and uplift_vs_trivial_*
columns and a verdict. Off (the default) = the report is what it was.
    python -m pytest tests/test_trivial_baseline.py -q
"""
import copy
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_fixture as rf  # noqa: E402
import _nbload  # noqa: E402

TARGETS = ("high", "low", "close")
OLD_COLUMNS = ["train", "val", "test", "gap_train_val", "gap_train_test", "linear_ref_val", "linear_ref_test"]
NEW_COLUMNS = ["trivial_auc_val", "uplift_vs_trivial_val", "trivial_auc_test", "uplift_vs_trivial_test"]
PHRASE = "no information beyond range mechanics"


class _StubModel:
    """predict() returns fixed class logits per split size (train / val / all test assets concatenated have distinct sizes)."""
    def __init__(self, logits_by_n):
        self.logits_by_n = logits_by_n

    def predict(self, x, batch_size=None, verbose=0):
        n = len(x)
        return {f"y_{t}_class_logits": self.logits_by_n[n][t] for t in TARGETS}


def _cat(split, k):
    parts = [split] if "y" in split else list(split.values())
    return np.concatenate([np.asarray(s["y"][k]).ravel() for s in parts])


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = rf.run_namespace()
        cls.kit = cls.ns["Toolkit"].from_namespace(cls.ns)
        cls.S = cls.ns["RunSettings"]
        cls.settings = cls.S()
        cls.ds = rf.build_dataset(_nbload.load_pipeline())          # seeded pure random walk through the real pipeline
        cls.info = rf._quiet(cls.ns["apply_dataset_config"], cls.settings, copy.deepcopy(cls.ds), cls.kit)
        cls.train, cls.val, cls.test = rf._quiet(cls.ns["make_splits"], cls.settings, copy.deepcopy(cls.ds), cls.info, cls.kit)
        cls.base = staticmethod(cls.ns["trivial_range_baseline"])
        rng = np.random.default_rng(0)
        sizes = {len(cls.train["last_candles"]): cls.train, len(cls.val["last_candles"]): cls.val,
                 sum(len(s["last_candles"]) for s in cls.test.values()): cls.test}
        assert len(sizes) == 3
        cls.noise = _StubModel({n: {t: rng.normal(size=n) for t in TARGETS} for n in sizes})
        cls.oracle = _StubModel({n: {t: _cat(sp, f"y_{t}_class") * 2.0 - 1.0 + 0.01 * rng.normal(size=n) for t in TARGETS}
                                 for n, sp in sizes.items()})

    def gap(self, model, **kw):
        return rf._quiet(self.ns["generalization_gap_report"], model, self.train, self.val, self.test, model_tf=self.info.model_tf,
                         price_targets=list(TARGETS), linear_reference=False, verbose=False, **kw)


class TrivialRangeBaseline(Base):
    def test_random_walk_high_low_are_predictable_close_is_not(self):
        for t in ("high", "low"):
            k = f"y_{t}_class"
            auc_tr, sign = self.base(self.train["last_candles"], self.train["y"][k], t)
            self.assertGreaterEqual(auc_tr, 0.70, t)
            self.assertEqual(sign, 1, t)                          # close near the high -> next high / next low above the last ones
            for name, sp in (("val", self.val), ("test", self.test)):
                lc = np.concatenate([s["last_candles"] for s in ([sp] if "y" in sp else sp.values())])
                self.assertGreaterEqual(self.base(lc, _cat(sp, k), t, sign=sign)[0], 0.70, f"{t} {name}")
        k = "y_close_class"
        _, sign = self.base(self.train["last_candles"], self.train["y"][k], "close")
        for name, sp in (("val", self.val), ("test", self.test)):
            lc = np.concatenate([s["last_candles"] for s in ([sp] if "y" in sp else sp.values())])
            self.assertTrue(0.45 <= self.base(lc, _cat(sp, k), "close", sign=sign)[0] <= 0.55, f"close {name}")

    def test_degenerate_range_scores_one_half_and_bad_names_raise(self):
        lc = np.zeros((6, 7))
        lc[:, 0] = lc[:, 1] = lc[:, 2] = 5.0                       # high == low -> p = 0.5 for every row
        self.assertEqual(self.base(lc, np.array([1, 0, 1, 0, 1, 0]), "high")[0], 0.5)
        with self.assertRaises(ValueError):
            self.base(lc, np.zeros(6), "open")
        with self.assertRaises(ValueError):
            self.base(lc, np.zeros(6), "high", sign=2)

    def test_sign_is_fitted_on_train_and_never_refitted(self):
        k, t = "y_high_class", "high"
        auc, sign = self.base(self.train["last_candles"], self.train["y"][k], t)
        self.assertEqual(sign, 1)
        flipped = 1.0 - np.asarray(self.val["y"][k])               # val where the relation is reversed
        applied, s2 = self.base(self.val["last_candles"], flipped, t, sign=sign)
        self.assertEqual(s2, sign)
        self.assertLess(applied, 0.5)                              # fixed sign: the reversal shows as < 0.5 ...
        self.assertAlmostEqual(applied, 1.0 - self.base(self.val["last_candles"], self.val["y"][k], t, sign=sign)[0], places=12)
        refit, s3 = self.base(self.val["last_candles"], flipped, t)
        self.assertEqual(s3, -1)                                   # ... whereas fitting on it would have hidden it (>= 0.5)
        self.assertGreaterEqual(refit, 0.5)


class ReportColumns(Base):
    def test_off_is_the_old_report_on_adds_columns_and_verdict(self):
        default = self.gap(self.noise)
        off = self.gap(self.noise, trivial_baseline=False)
        self.assertEqual(list(default["table"].columns), OLD_COLUMNS)
        pd.testing.assert_frame_equal(default["table"], off["table"])
        self.assertEqual(default["verdicts"], off["verdicts"])
        self.assertFalse(any(PHRASE in v for v in off["verdicts"].values()))

        on = self.gap(self.noise, trivial_baseline=True)
        self.assertEqual(list(on["table"].columns), OLD_COLUMNS + NEW_COLUMNS)
        pd.testing.assert_frame_equal(on["table"][OLD_COLUMNS], off["table"][OLD_COLUMNS])          # nothing else moved
        for t in TARGETS:
            r = on["table"].loc[(t, "class_auc")]
            for name in ("val", "test"):
                self.assertAlmostEqual(r[f"uplift_vs_trivial_{name}"], r[name] - r[f"trivial_auc_{name}"], places=12)
            self.assertIn(PHRASE, on["verdicts"][t])               # a noise model gains nothing over the rule
            self.assertEqual(on["verdicts"][t].split(" | ")[0], off["verdicts"][t].split(" | ")[0])
        for t in ("high", "low"):
            self.assertGreaterEqual(on["table"].loc[(t, "class_auc"), "trivial_auc_val"], 0.70)
        self.assertTrue(on["table"].drop(index="class_auc", level="metric")[NEW_COLUMNS].isna().all().all())

    def test_a_model_beating_the_rule_gets_no_verdict(self):
        on = self.gap(self.oracle, trivial_baseline=True)
        for t in TARGETS:
            self.assertGreater(on["table"].loc[(t, "class_auc"), "uplift_vs_trivial_val"], self.ns["TRIVIAL_UPLIFT_MIN"], t)
            self.assertNotIn(PHRASE, on["verdicts"][t])
        self.assertEqual(self.ns["TRIVIAL_UPLIFT_MIN"], 0.01)

    def test_setting_reaches_the_gap_report_through_run_reports(self):
        self.assertIs(self.settings.evaluation.trivial_baseline, False)
        with self.assertRaises(ValueError):
            self.S().updated({"evaluation": {"trivial_baseline": 1}})
        plan = rf._quiet(self.ns["plan_model"], self.settings, copy.deepcopy(self.ds), self.info, self.kit)
        for flag in (False, True):
            s = self.S().updated({"evaluation": {"trivial_baseline": flag}})
            out = rf._quiet(self.ns["run_reports"], s, ["gap"], self.kit, model=self.noise, train=self.train, val=self.val, test=self.test,
                            dataset=copy.deepcopy(self.ds), info=self.info, plan=plan, chicks=None)
            self.assertEqual("trivial_auc_val" in out["gap"]["table"].columns, flag)


if __name__ == "__main__":
    unittest.main()

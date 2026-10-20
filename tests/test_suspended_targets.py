"""TargetSettings.suspended_targets: None = the old rule (close suspended outside entry_range) and the old folder name; () trains close
too and it reaches the model heads and the chicks evaluation inputs; unknown names / suspending everything raise; a non-None value
changes the folder name; it composes with ModelSettings.config_overrides.
    python -m pytest tests/test_suspended_targets.py -q
"""
import copy
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "docs", "research", "audit")):
    sys.path.insert(0, p)
import run_fixture as rf  # noqa: E402
import _nbload  # noqa: E402

_DS = {}


def _dataset():
    if "d" not in _DS:
        _DS["d"] = rf.build_dataset(_nbload.load_pipeline())
    return copy.deepcopy(_DS["d"])


class SuspendedTargets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = rf.run_namespace()
        cls.kit = cls.ns["Toolkit"].from_namespace(cls.ns)
        cls.S = cls.ns["RunSettings"]

    def _plan(self, **target_and_model):
        s = self.S().updated(target_and_model) if target_and_model else self.S()
        ds = _dataset()
        info = rf._quiet(self.ns["apply_dataset_config"], s, ds, self.kit)
        return s, ds, info, self.ns["plan_model"](s, ds, info, self.kit)

    def test_none_is_the_old_rule(self):
        s, _, _, p = self._plan()
        self.assertIsNone(s.target.suspended_targets)
        self.assertEqual((p.price_targets, p.suspended_targets), (("high", "low"), ("close",)))
        self.assertNotIn("close", p.model_overrides["head_types"])
        e = self._plan(target={"target_mode": "entry_range"})[3]
        self.assertEqual((e.price_targets, e.suspended_targets), (("high", "low", "close"), ()))

    def test_empty_trains_close_and_reaches_the_model_and_chicks(self):
        s, ds, info, p = self._plan(target={"suspended_targets": ()})
        self.assertEqual((p.price_targets, p.suspended_targets), (("high", "low", "close"), ()))
        self.assertEqual(p.model_overrides["head_types"]["close"], ["nig_regression", "binary_classification"])
        co = self._plan(target={"suspended_targets": ()}, model={"class_only": True})[3]
        self.assertEqual(co.model_overrides["head_types"]["close"], ["binary_classification"])
        res = self.ns["run_main"](s.updated({"train": {"run_main_training": False}}), self.kit, dataset=ds, summary=False)
        self.assertEqual(res.plan.price_targets, ("high", "low", "close"))
        self.assertEqual(res.chicks.chicks_targets, ("high", "low", "close"))
        self.assertEqual([x.name for x in res.chicks.eval_target_specs], ["high", "low", "close"])
        self.assertTrue(any("close" in n for n in res.model.output_names))

    def test_explicit_names_suspend_exactly_those(self):
        p = self._plan(target={"suspended_targets": ("low",)})[3]
        self.assertEqual((p.price_targets, p.suspended_targets), (("high", "close"), ("low",)))
        self.assertEqual(self._plan(target={"suspended_targets": ["close"]})[3].price_targets, ("high", "low"))

    def test_bad_values_raise(self):
        with self.assertRaises(ValueError) as cm:
            self._plan(target={"suspended_targets": ("clse",)})
        self.assertIn("clse", str(cm.exception))
        with self.assertRaises(ValueError):
            self._plan(target={"suspended_targets": ("high", "low", "close")})
        for bad in ("close", (1,), 3):
            with self.assertRaises(ValueError):
                self.S().updated({"target": {"suspended_targets": bad}})

    def test_folder_name(self):
        f = self.ns["run_dir_for"]
        base = f(self.S(), 1.0, ("1h",))
        self.assertEqual(f(self.S().updated({"target": {"suspended_targets": None}}), 1.0, ("1h",)), base)
        none = f(self.S().updated({"target": {"suspended_targets": ()}}), 1.0, ("1h",))
        self.assertEqual(none, base + "_susp-none")
        a = f(self.S().updated({"target": {"suspended_targets": ("low", "close")}}), 1.0, ("1h",))
        b = f(self.S().updated({"target": {"suspended_targets": ("close", "low")}}), 1.0, ("1h",))
        self.assertEqual(a, base + "_susp-close-low")
        self.assertEqual(a, b)

    def test_roundtrip(self):
        s = self.S().updated({"target": {"suspended_targets": ()}})
        import json
        self.assertEqual(self.S.from_dict(json.loads(json.dumps(s.to_dict()))), s)

    def test_with_config_overrides_both_apply(self):
        _, _, _, p = self._plan(target={"suspended_targets": ()}, model={"config_overrides": {"use_instance_norm": False}})
        self.assertIs(p.model_overrides["use_instance_norm"], False)
        self.assertEqual(p.price_targets, ("high", "low", "close"))
        self.assertIn("close", p.model_overrides["head_types"])


if __name__ == "__main__":
    unittest.main()

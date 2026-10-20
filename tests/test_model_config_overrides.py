"""ModelSettings.config_overrides: default {} = the old plan and folder name; an override wins over ANTI_MEMORIZATION_CONFIG;
run-owned and unknown keys raise; a non-empty override changes the training folder name (sorted keys, deterministic).
    python -m pytest tests/test_model_config_overrides.py -q
"""
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "tests"), os.path.join(ROOT, "docs", "research", "audit")):
    sys.path.insert(0, p)
import run_fixture as rf  # noqa: E402


class ConfigOverrides(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = rf.run_namespace()
        cls.kit = cls.ns["Toolkit"].from_namespace(cls.ns)
        cls.S = cls.ns["RunSettings"]

    def _plan(self, **model):
        s = self.S().updated({"model": model}) if model else self.S()
        window = {"1h": 24}
        dataset = {"window_sizes": window, "feature_order": ["a", "b", "c"]}
        info = self.ns["DatasetInfo"](model_tf="1h", model_tfs=("1h",), reg_target_scale=1.0)
        return s, self.ns["plan_model"](s, dataset, info, self.kit)

    def test_default_plan_is_unchanged(self):
        s, plan = self._plan()
        self.assertEqual(s.model.config_overrides, {})
        want = {"enforce_order": self.kit.config.get("reg_target_mode", "return") != "return"}
        want.update(self.kit.anti_memorization_config)
        for k, v in want.items():
            self.assertEqual(plan.model_overrides[k], v, k)
        self.assertEqual(set(plan.model_overrides), set(want) | {"price_targets", "head_types"})

    def test_override_reaches_plan_and_beats_anti_memorization(self):
        _, base = self._plan()
        s, plan = self._plan(config_overrides={"use_instance_norm": False, "d_model": 32})
        self.assertIs(plan.model_overrides["use_instance_norm"], False)
        self.assertEqual(plan.model_overrides["d_model"], 32)
        self.assertEqual({k: v for k, v in plan.model_overrides.items() if k not in ("use_instance_norm", "d_model")},
                         {k: v for k, v in base.model_overrides.items() if k not in ("use_instance_norm", "d_model")})

    def test_override_wins_over_an_anti_memorization_key(self):
        self.assertEqual(self.kit.anti_memorization_config["d_model"], 64)
        _, plan = self._plan(config_overrides={"d_model": 32})
        self.assertEqual(plan.model_overrides["d_model"], 32)
        _, off = self._plan(anti_memorization=False, config_overrides={"d_model": 32})
        self.assertEqual(off.model_overrides["d_model"], 32)

    def test_run_owned_keys_raise(self):
        for k in ("price_targets", "head_types", "enforce_order"):
            with self.assertRaises(ValueError):
                self.S().updated({"model": {"config_overrides": {k: 1}}})

    def test_bad_types_raise(self):
        for bad in (["d_model"], {1: 2}):
            with self.assertRaises(ValueError):
                self.S().updated({"model": {"config_overrides": bad}})

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError) as cm:
            self._plan(config_overrides={"no_such_option": 1})
        self.assertIn("no_such_option", str(cm.exception))

    def test_folder_name(self):
        run_dir_for = self.ns["run_dir_for"]
        base = run_dir_for(self.S(), 1.0, ("1h",))
        self.assertEqual(run_dir_for(self.S().updated({"model": {"config_overrides": {}}}), 1.0, ("1h",)), base)
        a = self.S().updated({"model": {"config_overrides": {"use_instance_norm": False, "d_model": 32}}})
        b = self.S().updated({"model": {"config_overrides": {"d_model": 32, "use_instance_norm": False}}})
        self.assertNotEqual(run_dir_for(a, 1.0, ("1h",)), base)
        self.assertTrue(run_dir_for(a, 1.0, ("1h",)).startswith(base))
        self.assertEqual(run_dir_for(a, 1.0, ("1h",)), run_dir_for(b, 1.0, ("1h",)))
        c = self.S().updated({"model": {"config_overrides": {"d_model": 64}}})
        self.assertNotEqual(run_dir_for(c, 1.0, ("1h",)), run_dir_for(a, 1.0, ("1h",)))

    def test_roundtrip(self):
        s = self.S().updated({"model": {"config_overrides": {"d_model": 32}}})
        self.assertEqual(self.S.from_dict(s.to_dict()), s)


if __name__ == "__main__":
    unittest.main()

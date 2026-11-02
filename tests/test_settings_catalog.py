"""tools/settings_catalog.py: the generated catalog (docs/research/settings_catalog.md / .json) lists every RunSettings field, every
data CONFIG key and every model config key that config_overrides can carry, the registered names, and the recommendation of the
preset; it is not stale; a RunSettings field added without a catalog line fails here.
    python -m pytest tests/test_settings_catalog.py -q
"""
import dataclasses
import json
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
from tools import settings_catalog as sc  # noqa: E402
import _nbload  # noqa: E402


class SettingsCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = sc.build()
        cls.ns = {}
        _nbload.workflow_package().load_into(cls.ns, only=("settings", "presets"))

    def test_every_runsettings_field_is_in_the_catalog(self):
        """The fail-when-a-field-is-added test: walk the dataclasses independently of the tool and compare with the COMMITTED json."""
        want = {f"{sec}.{f.name}" for sec, cls in self.ns["RunSettings"].SECTIONS.items() for f in dataclasses.fields(cls)}
        committed = json.load(open(sc.JSON_PATH, encoding="utf-8"))
        have = {r["path"] for r in committed["settings"]}
        self.assertEqual(want - have, set(), "RunSettings field(s) missing from docs/research/settings_catalog.json: run python tools/settings_catalog.py")
        self.assertEqual(have - want, set(), "the catalog lists fields RunSettings no longer has: regenerate it")

    def test_every_field_has_an_accepted_values_note(self):
        missing = [r["path"] for r in self.cat["settings"] if r["accepted"].startswith("MISSING NOTE")]
        self.assertEqual(missing, [], "add a line to tools/settings_catalog.py ACCEPTED_NOTES")
        stale = set(sc.ACCEPTED_NOTES) - {r["path"] for r in self.cat["settings"]}
        self.assertEqual(stale, set(), "ACCEPTED_NOTES names fields that do not exist")

    def test_committed_files_are_not_stale(self):
        self.assertEqual(open(sc.MD_PATH, encoding="utf-8").read(), sc.render_markdown(self.cat), "run python tools/settings_catalog.py")
        self.assertEqual(open(sc.JSON_PATH, encoding="utf-8").read(), sc.render_json(self.cat), "run python tools/settings_catalog.py")

    def test_config_keys_and_registries(self):
        paths = {r["path"] for r in self.cat["config_keys"]}
        for key in ("price_norm_mode", "stale_bars", "reg_target_mode", "sample_filters"):
            self.assertIn(f"project.config_overrides.{key}", paths)
        mpaths = {r["path"] for r in self.cat["model_keys"]}
        for key in ("use_instance_norm", "d_model", "level_passthrough"):
            self.assertIn(f"model.config_overrides.{key}", mpaths)
        regs = self.cat["registries"]
        self.assertIn("logret_std", regs["data.NORMALIZERS"]["names"])
        self.assertIn("binary_classification", regs["model.HEAD_REGISTRY"]["names"])
        self.assertIn("evidential", regs["trainer.TASK_REGISTRY"]["names"])
        self.assertIn("entry_range", regs["core.schema.TARGET_MODES"]["names"])

    def test_recommendations_match_the_preset_table(self):
        by_path = {r["path"]: r for r in self.cat["settings"] + self.cat["config_keys"] + self.cat["model_keys"]}
        for row in self.ns["RECOMMENDED_CHOICES"]:
            got = by_path[row["path"]]
            self.assertEqual(got["recommended"], repr(row["value"]), row["path"])
            self.assertEqual((got["evidence"], got["source"]), (row["evidence"], row["source"]), row["path"])
        self.assertEqual(sorted(self.cat["recommended_paths"]), sorted(r["path"] for r in self.ns["RECOMMENDED_CHOICES"]))

    def test_defaults_shown_are_the_live_defaults(self):
        d = self.ns["RunSettings"]()
        by_path = {r["path"]: r for r in self.cat["settings"]}
        self.assertEqual(by_path["train.epochs"]["default"], repr(d.train.epochs))
        self.assertEqual(by_path["target.suspended_targets"]["default"], repr(d.target.suspended_targets))
        self.assertEqual(by_path["evaluation.calibration_method"]["accepted"], "one of ('isotonic', 'platt')")

    def test_unchecked_fields_are_reported(self):
        unchecked = {r["path"] for r in self.cat["settings"] if not r["validated"]}
        self.assertTrue({"panel.first_touch", "panel.run_root"} <= unchecked)
        self.assertNotIn("train.epochs", unchecked)
        self.assertIn("panel.first_touch", open(sc.MD_PATH, encoding="utf-8").read().split("Settings without a value check")[1])

    def test_check_flag_passes_when_fresh(self):
        self.assertEqual(sc.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()

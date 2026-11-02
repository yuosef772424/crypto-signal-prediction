"""The "recommended_2026_11" preset (workflow/presets.py): it builds, leaves the RunSettings defaults alone, every row of its table carries
its evidence and source, it passes the strict config builders and trains one epoch on the synthetic fixture (CPU), a dataset built
without the options it names is reported, and the tool / notebook entry points choose it in one line.
    python -m pytest tests/test_presets.py -q
"""
import copy
import dataclasses
import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import run_fixture as rf  # noqa: E402
import _nbload  # noqa: E402

NAME = "recommended_2026_11"
#: What the preset must switch on (the owner's list): path -> value. Defaults are NOT asserted to stay equal here (golden test does that).
EXPECTED = {
    ("project", "config_overrides", "price_norm_mode"): "logret_std",
    ("project", "config_overrides", "stale_bars"): {"min_run_hours": 168},
    ("model", "config_overrides", "use_instance_norm"): False,
    ("target", "suspended_targets"): (),
    ("evaluation", "trivial_baseline"): True,
}


def _get(settings, path):
    cur = settings
    for p in path:
        cur = cur[p] if isinstance(cur, dict) else getattr(cur, p)
    return cur


class PresetTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = {}
        _nbload.workflow_package().load_into(cls.ns, only=("settings", "presets"))

    def test_builds_and_switches_on_the_owner_list(self):
        s = self.ns["recommended_settings"]()
        for path, want in EXPECTED.items():
            self.assertEqual(_get(s, path), want, ".".join(path))
        self.assertEqual(s.project.config_overrides["project_name"], "crypto_model")        # base keys survive

    def test_defaults_of_runsettings_are_untouched(self):
        d = self.ns["RunSettings"]()
        self.assertEqual(d.project.config_overrides, {"project_name": "crypto_model"})
        self.assertEqual(d.model.config_overrides, {})
        self.assertIsNone(d.target.suspended_targets)       # D-013's () arrives with PR #62; this PR does not touch the default
        self.assertFalse(d.evaluation.trivial_baseline)
        self.assertEqual(self.ns["recommended_settings"]().to_dict() == d.to_dict(), False)
        self.assertEqual(self.ns["RunSettings"]().to_dict(), d.to_dict())

    def test_every_row_has_evidence_and_a_source(self):
        rows = self.ns["RECOMMENDED_CHOICES"]
        self.assertGreaterEqual(len(rows), 8)
        for r in rows:
            self.assertIn(r["evidence"], ("proven", "new"), r["path"])
            self.assertGreater(len(r["source"]), 20, r["path"])
        paths = [r["path"] for r in rows]
        self.assertEqual(len(paths), len(set(paths)))
        for want in EXPECTED:
            self.assertIn(".".join(want), paths)
        self.assertTrue({r["evidence"] for r in rows} == {"proven", "new"})

    def test_every_row_path_is_a_real_field(self):
        fields = {sec: set(cls.field_names()) for sec, cls in self.ns["RunSettings"].SECTIONS.items()}
        for r in self.ns["RECOMMENDED_CHOICES"]:
            parts = r["path"].split(".")
            self.assertIn(parts[1], fields[parts[0]], r["path"])
            self.assertIn(len(parts), (2, 3))

    def test_run_dir_gets_the_preset_name_once(self):
        s = self.ns["recommended_settings"]()
        self.assertTrue(s.train.run_dir.endswith("_" + NAME))
        again = self.ns["recommended_settings"](s)
        self.assertEqual(again.train.run_dir, s.train.run_dir)
        custom = self.ns["recommended_settings"](self.ns["RunSettings"]().updated({"train": {"run_dir": "/x/run", "epochs": 7}}))
        self.assertEqual((custom.train.run_dir, custom.train.epochs), ("/x/run_" + NAME, 7))

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.ns["settings_with_preset"]("recommended_2099", None)
        self.assertIn(NAME, str(cm.exception))
        self.assertEqual(self.ns["settings_with_preset"](NAME).to_dict(), self.ns["recommended_settings"]().to_dict())

    def test_dataset_mismatches(self):
        s = self.ns["recommended_settings"]()
        m = self.ns["dataset_mismatches"]
        self.assertEqual(len(m(s, {})), 2)                                           # window_scale file without a stale stamp
        scaled = m(s, {"price_norm_mode": "logret_std", "stale_bars": {"min_run": 168}, "reg_target_scale": 100.0})
        self.assertEqual(len(scaled), 1)                                             # logret_std refuses reg_target_scale != 1 (1h_s8 files)
        self.assertIn("reg_target_scale", scaled[0])
        self.assertEqual(m(s, {"price_norm_mode": "logret_std", "stale_bars": {"min_run": 168}}), [])
        self.assertEqual(m(self.ns["RunSettings"](), {}), [])                        # defaults name nothing
        self.assertIn("price_norm_mode", m(s, {"price_norm_mode": "window_scale", "stale_bars": {"min_run": 3}})[0])


class PresetRuns(unittest.TestCase):
    """The preset through the strict builders and one trained epoch, on a fixture dataset built the way the preset says."""

    @classmethod
    def setUpClass(cls):
        cls.ns = rf.run_namespace()
        s = cls.ns["recommended_settings"]()
        cfg = dict(s.project.config_overrides)
        cfg.pop("project_name")
        cfg["reg_target_scale"] = 1.0                       # the fixture's 1h_s8 preset uses 100; logret_std refuses it
        cls.work = tempfile.mkdtemp(prefix="preset_run_")
        cls.dataset = rf.build_dataset(_nbload.load_pipeline(), config_overrides=cfg)
        cls.settings = s.updated({"train": {"run_dir": os.path.join(cls.work, "run"), "epochs": 1, "fit_verbose": 0}})

    def test_one_epoch_on_the_fixture(self):
        ns = self.ns
        self.assertEqual(ns["dataset_mismatches"](self.settings, self.dataset), [])
        kit = ns["Toolkit"].from_namespace(ns)
        res = ns["run_main"](self.settings, kit, dataset=copy.deepcopy(self.dataset), summary=False)
        self.assertEqual(res.plan.price_targets, ("high", "low", "close"))              # suspended_targets=()
        self.assertFalse(res.plan.model_overrides["use_instance_norm"])                 # config_overrides won over anti_memorization
        self.assertEqual(ns["CONFIG"]["price_norm_mode"], "logret_std")
        folder = ns["run_dir_for"](self.settings, res.info.reg_target_scale, res.info.model_tfs)         # never the old model's folder
        self.assertIn("_susp-none", folder)
        self.assertIn("use-instance-norm-False", folder)
        self.assertTrue(os.path.isdir(folder) and any(os.scandir(folder)), "one epoch wrote nothing into the run folder")
        self.assertIsNotNone(res.trainer)
        hist = res.history.history
        self.assertEqual(len(hist["loss"]), 1)
        self.assertTrue(all(v == v for v in hist["loss"]), "NaN loss")

    def test_build_config_and_plan_model_alone(self):
        ns = self.ns
        kit = ns["Toolkit"].from_namespace(ns)
        info = rf._quiet(ns["apply_dataset_config"], self.settings, copy.deepcopy(self.dataset), kit)
        plan = rf._quiet(ns["plan_model"], self.settings, copy.deepcopy(self.dataset), info, kit)
        self.assertEqual(plan.suspended_targets, ())
        self.assertIs(plan.model_overrides["use_instance_norm"], False)


class EntryPoints(unittest.TestCase):
    def test_evaluate_tool_takes_the_preset_flag(self):
        src = open(os.path.join(ROOT, "tools", "evaluate_trained_model.py"), encoding="utf-8").read()
        self.assertIn("--preset", src)
        self.assertIn("settings_with_preset", src)

    def test_evaluate_tool_builds_the_preset_settings(self):
        import evaluate_trained_model as etm
        ns = {}
        _nbload.workflow_package().load_into(ns, only=("settings", "presets"))
        a = etm.build_parser().parse_args(["--data", "d.pkl.gz", "--weights", "w.h5", "--preset", NAME])
        settings, _ = etm.build_settings(a, ns["RunSettings"], ns=ns)
        self.assertEqual(settings.project.config_overrides["price_norm_mode"], "logret_std")
        self.assertEqual(settings.model.config_overrides, {"use_instance_norm": False})
        self.assertEqual(settings.target.suspended_targets, ())
        self.assertTrue(settings.train.run_dir.endswith("_" + NAME))
        self.assertFalse(settings.evaluation.calibrate_confidence)         # the tool never calibrated: the flag set by the tool wins
        plain = etm.build_settings(etm.build_parser().parse_args(["--data", "d", "--weights", "w"]), ns["RunSettings"], ns=ns)[0]
        self.assertEqual(plain.project.config_overrides, {"project_name": "crypto_model"})      # no preset = the old behaviour
        b = etm.build_parser().parse_args(["--data", "d", "--weights", "w", "--preset", "nope"])
        with self.assertRaises(ValueError):
            etm.build_settings(b, ns["RunSettings"], ns=ns)

    def test_main_notebook_chooses_it_with_one_variable(self):
        nb = json.load(open(os.path.join(ROOT, "main.ipynb"), encoding="utf-8"))
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        hits = [c for c in cells if "USE_RECOMMENDED" in c]
        self.assertEqual(len(hits), 1)
        self.assertIn("USE_RECOMMENDED = False", hits[0])
        self.assertIn("recommended_settings(", hits[0])


if __name__ == "__main__":
    unittest.main()

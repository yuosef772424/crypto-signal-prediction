"""tools/evaluate_trained_model.py builds a RunSettings and calls the workflow/run.py steps: no main.ipynb cell text is read or patched.

What is pinned: the command line maps to the settings the old text patches expressed (target mode, close definition, group freq,
timeframes, split dates, anti-memorization, training overrides, the panel options and how a preset interacts with explicit panel
options), the tool's report list is the workflow's, the tool source holds none of the old anchors, and one real run (weights loaded
into the model main would have built, section-7 reports executed in the output folder) works end to end.
    python -m pytest tests/test_run_tool.py -q
"""
import os
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))

import evaluate_trained_model as tool  # noqa: E402
import run_fixture as rf  # noqa: E402
import _nbload  # noqa: E402


def _settings(*argv):
    ns = _nbload.workflow_package().load_into({"__name__": "t"}, only=("settings",))
    a = tool.build_parser().parse_args(["--data", "d.pkl.gz", *argv])
    return tool.build_settings(a, ns["RunSettings"], invoke_cwd="/invoke")


class CommandLineToSettings(unittest.TestCase):
    def test_plain_weights_run_is_the_default_settings_but_for_the_tools_own_choices(self):
        s, panel = _settings("--weights", "w.h5")
        self.assertEqual(panel, {})
        self.assertEqual((s.target.target_mode, s.target.entry_close_reg, s.target.group_freq), (None, "abs_return", None))
        self.assertEqual((s.model.anti_memorization, s.model.class_only, s.train.run_main_training), (True, False, False))
        self.assertEqual((s.data.model_tfs, s.data.split_dates), (None, None))
        self.assertFalse(s.panel.enabled)
        # the tool never calibrated confidence (no val_dict was passed to chicks): the one deliberate difference from main's cell
        self.assertFalse(s.evaluation.calibrate_confidence)
        self.assertEqual(s.evaluation.mn_quantiles, (0.05, 0.1, 0.2, 0.3, 0.5))

    def test_every_data_and_target_option(self):
        s, _ = _settings("--weights", "w.h5", "--target-mode", "entry_range", "--entry-close-reg", "range_pos", "--group-freq", "32h",
                         "--model-tfs", "1h, 4h", "--split-dates", "2025-06-24,2025-11-21", "--anti-memorization", "false",
                         "--mn-quantiles", "0.1,0.5", "--mn-universe", "BTC,ETH")
        self.assertEqual((s.target.target_mode, s.target.entry_close_reg, s.target.group_freq), ("entry_range", "range_pos", "32h"))
        self.assertEqual(s.data.model_tfs, ("1h", "4h"))
        self.assertEqual(s.data.split_dates, {"train_end": "2025-06-24", "val_end": "2025-11-21"})
        self.assertFalse(s.model.anti_memorization)
        self.assertEqual((s.evaluation.mn_quantiles, s.evaluation.mn_universe), ((0.1, 0.5), ("BTC", "ETH")))
        self.assertEqual(_settings("--weights", "w.h5", "--mn-universe", "categories")[0].evaluation.mn_universe, "categories")

    def test_train_options(self):
        s, _ = _settings("--train", "--epochs", "3", "--batch-size", "32", "--patience", "4", "--run-dir", "myrun")
        self.assertTrue(s.train.run_main_training)
        self.assertEqual((s.train.epochs, s.train.batch_size, s.train.early_stopping_patience, s.train.fit_verbose), (3, 32, 4, 2))
        self.assertEqual(s.train.run_dir, os.path.abspath("myrun"))
        plain, _ = _settings("--weights", "w.h5", "--epochs", "3")                 # training options mean nothing without --train
        self.assertEqual((plain.train.epochs, plain.train.run_main_training), (60, False))

    def test_invalid_values_fail_before_anything_loads(self):
        with self.assertRaises(SystemExit):
            _settings("--weights", "w.h5", "--target-mode", "nope")
        ns = _nbload.workflow_package().load_into({"__name__": "t"}, only=("settings",))
        bad = tool.build_parser().parse_args(["--data", "d", "--weights", "w", "--mn-quantiles", "0.9"])
        with self.assertRaises(ValueError):
            tool.build_settings(bad, ns["RunSettings"])

    def test_panel_options_and_preset_precedence(self):
        s, explicit = _settings("--weights", "w.h5", "--panel", "A,B", "--panel-seeds", "1,2", "--panel-epochs", "2",
                                "--panel-patience", "3", "--panel-subset", "60,40", "--panel-k-eval", "5,all",
                                "--panel-baseline", "sigs", "--panel-first-touch", "ft.pkl", "--panel-batch-samples", "256",
                                "--panel-overrides", '{"train": {"lambda_ic": 1.0}}', "--group-freq", "8h")
        p = s.panel
        self.assertTrue(p.enabled)
        self.assertEqual((p.batch_samples, p.subset, p.run_root), (256, {"last_days": 60, "coins": 40}, os.path.abspath("panel_runs")))
        self.assertEqual((p.first_touch, p.baseline), ("/invoke/ft.pkl", "/invoke/sigs"))
        self.assertEqual(p.overrides, {"train": {"lambda_ic": 1.0}})
        self.assertEqual(explicit, {"variants": ("A", "B"), "seeds": (1, 2), "epochs": 2, "patience": 3, "group": "8h",
                                    "k_eval": (5, None), "baseline": "/invoke/sigs"})
        # no explicit baseline: "model" when a trained/loaded model exists, None otherwise
        self.assertEqual(_settings("--weights", "w.h5", "--panel", "A")[0].panel.baseline, "model")
        self.assertIsNone(_settings("--panel", "A")[0].panel.baseline)
        self.assertEqual(_settings("--panel", "A", "--panel-baseline", "model")[0].panel.baseline, "model")
        # preset: the preset overrides what it names, then the explicit options override the preset (the old order)
        s, explicit = _settings("--panel", "preset", "--panel-preset", "1h_s8", "--panel-epochs", "7", "--weights", "w.h5")
        self.assertNotIn("variants", explicit)
        self.assertNotIn("seeds", explicit)                                        # the preset's seeds (0, 1) stay
        final = s.panel.applied().updated(**explicit)
        self.assertEqual((final.variants, final.seeds, final.epochs, final.patience, final.baseline),
                         (("A_ic", "B_ic", "A_ic_k"), (0, 1), 7, 25, None))
        self.assertEqual(final.preset, "1h_s8")
        s, explicit = _settings("--panel", "A_ic", "--panel-preset", "1h_s8")      # explicit variants beat the preset's
        self.assertEqual(s.panel.applied().updated(**explicit).variants, ("A_ic",))


class ToolStructure(unittest.TestCase):
    def test_reports_are_the_workflows(self):
        import workflow
        self.assertEqual(tool.REPORTS, tuple(workflow.REPORTS))

    def test_no_main_cell_text_is_read_or_patched(self):
        src = open(os.path.join(ROOT, "tools", "evaluate_trained_model.py"), encoding="utf-8").read()
        body = src[src.index("REPO = os.path"):]                                    # the code, not the docstring
        for anchor in ("TARGET_MODE = None", 'ENTRY_CLOSE_REG = "abs_return"', "RUN_MAIN_TRAINING = True", "PANEL_MODE = False",
                       "# ── نهاية الإعدادات ──", "retarget_splits(train, val, test, mode=TARGET_MODE)", "callbacks=callbacks, verbose=1,",
                       "model.summary()", "PANEL_PRESETS[PANEL_PRESET]", "/content/drive/MyDrive/training_runs", "run_wiring_selftest(",
                       "json.load(open(os.path.join(REPO, \"main.ipynb\")"):
            self.assertNotIn(anchor, body, anchor)
        self.assertNotIn('RUNNER_NOTEBOOKS + ("main.ipynb"', body)
        self.assertNotIn('os.path.join(REPO, "main.ipynb"), encoding', body)        # the notebook file is never opened

    def test_runner_notebooks_are_main_ipynbs_run_lines(self):
        import json
        import re
        with open(os.path.join(ROOT, "main.ipynb"), encoding="utf-8") as f:
            cells = json.load(f)["cells"]
        runs = [m for c in cells if c["cell_type"] == "code"
                for m in re.findall(r'^%run "([^"]+)"', c["source"] if isinstance(c["source"], str) else "".join(c["source"]), re.M)]
        self.assertEqual(tuple(runs), tool.RUNNER_NOTEBOOKS)


class EndToEnd(unittest.TestCase):
    def test_weights_run_executes_reports_in_the_output_folder(self):
        """A real run: synthetic dataset file + weights saved from the very model main builds, tool as a subprocess."""
        import gzip
        import pickle
        work = tempfile.mkdtemp(prefix="eval_tool_")
        ns = rf.run_namespace()
        ds = rf.build_dataset(_nbload.load_pipeline(), two_tf=False)
        data = os.path.join(work, "d.pkl.gz")
        with gzip.open(data, "wb") as f:
            pickle.dump(ds, f)
        kit = ns["Toolkit"].from_namespace(ns)
        settings = ns["RunSettings"]().updated({"train": {"run_main_training": False}})
        res = ns["run_main"](settings, kit, dataset=ds, summary=False)
        weights = os.path.join(work, "w.weights.h5")
        res.model.save_weights(weights)
        out = os.path.join(work, "out")
        proc = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "evaluate_trained_model.py"), "--data", data, "--weights", weights,
                               "--reports", "gap,signals", "--out", out], capture_output=True, text=True, cwd=work, timeout=1500)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        self.assertIn("أُحمّلت الأوزان", proc.stdout)
        self.assertIn("فجوة التعميم", proc.stdout)
        for name in ("signals_val.csv.gz", "signals_test.csv.gz"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        self.assertNotIn("❌", proc.stdout)


if __name__ == "__main__":
    unittest.main()

"""main.ipynb's own code cells, executed in order against a synthetic dataset (what a Colab session does after the Drive / %run / repo cells).

The Colab-only cells (repo fetch with drive.mount, the four %run cells, the workflow package loader) are replaced by loading the same
packages into the namespace (tests/run_fixture.run_namespace); every other code cell runs as written: the settings cell, the data / split /
retarget / model / training-config / training / chicks cells, the optional-report examples (comments) and the panel cell (disabled). Training
is shortened by overriding the settings object right after its cell, exactly what a user does by editing it.
    python -m pytest tests/test_main_notebook.py -q
"""
import copy
import json
import os
import re
import sys
import tempfile
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))

import run_fixture as rf  # noqa: E402


def _code_cells():
    with open(os.path.join(ROOT, "main.ipynb"), encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            yield i, c["source"] if isinstance(c["source"], str) else "".join(c["source"])


def _colab_only(src):
    return ("drive.mount(" in src or re.search(r"^%run ", src, re.M) or "_WF_REPO_CANDIDATES" in src
            or "run_wiring_selftest(" in src)           # the self-test is covered by tests/test_run_steps.py


class MainNotebookRuns(unittest.TestCase):
    def test_code_cells_run_in_order_and_fill_the_visible_variables(self):
        work = tempfile.mkdtemp(prefix="main_nb_")
        ds = rf.build_dataset(__import__("_nbload").load_pipeline(), two_tf=False)
        ns = rf.run_namespace()
        ns.update(display=print, load_data_from_drive=lambda **k: copy.deepcopy(ds))
        ran = []
        for i, src in _code_cells():
            if _colab_only(src):
                continue
            rf._quiet(exec, compile(src, f"main#cell{i}", "exec"), ns)
            ran.append(i)
            if "settings = RunSettings(" in src:                       # a user edits the settings cell: short training in a temp folder
                ns["settings"] = ns["settings"].updated({"train": {"run_dir": os.path.join(work, "run"), "epochs": 1, "batch_size": 32,
                                                                   "fit_verbose": 0}})
        self.assertGreaterEqual(len(ran), 15, ran)
        for name in ("settings", "kit", "dataset", "info", "train", "val", "test", "plan", "model_builder", "model", "main_config",
                     "train_ds", "val_ds", "trainer", "history", "chicks", "full_results"):
            self.assertIn(name, ns, name)
        self.assertEqual((ns["info"].model_tf, ns["info"].reg_target_scale), ("1h", 100.0))
        self.assertEqual(ns["plan"].price_targets, ("high", "low"))
        self.assertTrue(ns["main_config"]["run"]["run_dir"].startswith(os.path.join(work, "run")))
        self.assertIsNotNone(ns["history"])
        self.assertIsNotNone(ns["full_results"])                       # section 6 ran on the trained model
        self.assertNotIn("panel_results", ns)                          # panel.enabled=False
        for gone in ("TARGET_MODE", "PRICE_TARGETS", "MODEL_TFS", "REG_TARGET_SCALE", "main_config_global"):
            self.assertNotIn(gone, ns, gone)                           # no setting global is created by the notebook


if __name__ == "__main__":
    unittest.main()

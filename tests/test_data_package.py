"""The data/ package (all pipeline code, formerly the cells of crypto_data_pipeline_v6.ipynb) and its runner notebook.

Guards the structure, not the numbers (the numbers are covered by test_multi_tf.py's golden digests and the self-tests):
every data/*.py module is in the load order, the shared namespace exposes the public API and is patchable, and the runner
notebook defines no function of its own.
"""
import ast
import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))

import _nbload  # noqa: E402
from data import _loader  # noqa: E402

PUBLIC = ("CONFIG", "update_config", "build_dataset", "split_data", "load_data_from_drive", "decode_price_window",
          "save_data_to_drive", "build_hourly_4h_dataset", "mount_drive", "default_workers")


class DataPackageTests(unittest.TestCase):
    def test_every_module_file_is_loaded_and_vice_versa(self):
        files = {f[:-3] for f in os.listdir(os.path.join(ROOT, "data")) if f.endswith(".py") and not f.startswith("_")}
        self.assertEqual(files, set(_loader.MODULES))
        for m in _loader.MODULES:
            self.assertTrue(_loader.module_path(m).is_file(), m)

    def test_load_into_exposes_public_api_and_is_patchable(self):
        ns = _nbload.load_pipeline()
        for name in PUBLIC:
            self.assertIn(name, ns)
        self.assertNotIn("run_pipeline_selftests", ns)            # selftests module excluded by load_pipeline
        calls = []
        ns["_system_ram_mb"] = lambda: calls.append(1) or 4096.0    # late binding: callers see the patched name
        ns["default_workers"](100)
        self.assertEqual(calls, [1])

    def test_config_is_one_shared_dict(self):
        ns = _nbload.load_pipeline()
        cfg = ns["CONFIG"]
        ns["update_config"]({"stride": 7})
        self.assertIs(ns["CONFIG"], cfg)
        self.assertEqual(cfg["stride"], 7)

    def test_runner_notebook_defines_no_function(self):
        with open(os.path.join(ROOT, "crypto_data_pipeline_v6.ipynb"), encoding="utf-8") as f:
            nb = json.load(f)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "\n".join(ln for ln in "".join(cell["source"]).splitlines() if not ln.lstrip().startswith(("!", "%")))
            defs = [n.name for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            self.assertEqual(defs, [], f"runner cell {i} defines {defs}: move it to data/")


if __name__ == "__main__":
    unittest.main()

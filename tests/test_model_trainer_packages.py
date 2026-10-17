"""The model/ and trainer/ packages (formerly the cells of model_v2 (1).ipynb and trainer_framework_v2.ipynb) and their runner
notebooks.

Guards the structure, not the numbers (the numbers are covered by the model/trainer tests that load these packages and by
the load-time self-tests): every module file is in the load order and carries a card, the shared namespace exposes the
public API and is patchable, the import-time modules (selftests / smoke_test) can be excluded, and each runner notebook
defines no function or class of its own.
"""
import ast
import json
import os
import subprocess
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))

import _nbload  # noqa: E402
from model import _loader as model_loader  # noqa: E402
from trainer import _loader as trainer_loader  # noqa: E402

MODEL_PUBLIC = ("build_model_fn", "build_nig_timenet_v2", "MODEL_CONFIG", "ANTI_MEMORIZATION_CONFIG", "HEAD_REGISTRY",
                "register_head_type", "InstanceNorm", "TransformerBlock", "NIGHead", "diagnose_model",
                "model_health_verdicts", "layer_probe_report", "layer_compare_report", "random_init_copy")
TRAINER_PUBLIC = ("build_training_system", "build_config", "DEFAULT_CONFIG", "GenericTrainer", "TASK_REGISTRY",
                  "register_task_type", "CheckpointManager", "BestModelTracker", "TrainingDiagnostics",
                  "with_sample_index", "build_lr_schedule_fn", "ensemble_predict_evidential", "run_kfold_training")

PACKAGES = (("model", model_loader, "model_v2 (1).ipynb"), ("trainer", trainer_loader, "trainer_framework_v2.ipynb"))


def _top_level_names(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.append(n.name)
        elif isinstance(n, ast.Assign):
            names += [t.id for t in n.targets if isinstance(t, ast.Name)]
    return names


class StructureTests(unittest.TestCase):
    def test_every_module_file_is_loaded_and_vice_versa(self):
        for pkg, loader, _ in PACKAGES:
            files = {f[:-3] for f in os.listdir(os.path.join(ROOT, pkg)) if f.endswith(".py") and not f.startswith("_")}
            self.assertEqual(files, set(loader.MODULES), pkg)
            self.assertEqual(len(loader.MODULES), len(set(loader.MODULES)), pkg)
            for m in loader.MODULES:
                self.assertTrue(loader.module_path(m).is_file(), f"{pkg}.{m}")

    def test_every_module_has_a_card(self):
        for pkg, loader, _ in PACKAGES:
            for m in loader.MODULES:
                doc = ast.get_docstring(ast.parse(loader.module_path(m).read_text(encoding="utf-8"))) or ""
                for field in ("PURPOSE:", "TAGS:"):
                    self.assertIn(field, doc, f"{pkg}/{m}.py has no {field} card")

    def test_unknown_module_names_raise(self):
        for pkg, loader, _ in PACKAGES:
            with self.assertRaises(ValueError):
                loader.module_path("no_such_module")
            with self.assertRaises(ValueError):
                loader.load_into({}, exclude=("no_such_module",))

    def test_no_top_level_name_defined_in_two_modules(self):
        """One shared namespace: a name defined by two modules would silently shadow (the later one wins)."""
        for pkg, loader, _ in PACKAGES:
            seen = {}
            for m in loader.MODULES:
                for n in _top_level_names(loader.module_path(m)):
                    if n in ("__doc__",):
                        continue
                    self.assertNotIn(n, seen, f"{pkg}: {n!r} defined in {seen.get(n)} and {m}")
                    seen[n] = m

    def test_lazy_import_does_not_load(self):
        for pkg, _, _ in PACKAGES:
            code = (f"import sys; sys.path.insert(0, {ROOT!r}); import {pkg}; "
                    f"print({pkg}._{pkg.upper()}_PKG_STATE['loaded'], 'tensorflow' in sys.modules)")
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
            self.assertEqual(out, ["False", "False"], pkg)

    def test_runner_notebooks_define_no_function_and_load_their_package(self):
        for pkg, _, nbname in PACKAGES:
            with open(os.path.join(ROOT, nbname), encoding="utf-8") as f:
                nb = json.load(f)
            loads = 0
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] != "code":
                    continue
                src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
                loads += f"_{pkg}_pkg.load_into(globals())" in src
                code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%")))
                defs = [n.name for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
                self.assertEqual(defs, [], f"{nbname} runner cell {i} defines {defs}: move it to {pkg}/")
            self.assertEqual(loads, 1, nbname)


class ModelNamespaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _nbload.load_model()

    def test_public_api(self):
        for name in MODEL_PUBLIC:
            self.assertIn(name, self.ns)
        self.assertNotIn("run_model_selftests", self.ns)             # selftests module excluded by default

    def test_head_registry_is_one_shared_dict(self):
        ns = _nbload.load_model()
        registry = ns["HEAD_REGISTRY"]
        self.assertIn("binary_classification", registry)
        ns["register_head_type"]("zz_structure_test")(lambda *a, **k: {})
        self.assertIs(ns["HEAD_REGISTRY"], registry)
        self.assertIn("zz_structure_test", registry)
        self.assertNotIn("zz_structure_test", self.ns["HEAD_REGISTRY"])      # separate load = separate namespace

    def test_selftests_module_can_be_included(self):
        ns = _nbload.load_model(selftests=True)
        self.assertTrue(callable(ns["run_model_selftests"]))


class TrainerNamespaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _nbload.load_trainer(kfold=True)

    def test_public_api(self):
        for name in TRAINER_PUBLIC:
            self.assertIn(name, self.ns)
        for name in ("smoke_config", "_dummy_model_builder"):         # smoke_test module excluded
            self.assertNotIn(name, self.ns)

    def test_late_binding_patched_name_is_seen_by_callers(self):
        ns = _nbload.load_trainer()
        seen = []

        def fake(*a, **k):
            seen.append(1)
            raise RuntimeError("patched deep_update was called")
        ns["deep_update"] = fake
        with self.assertRaises(RuntimeError):
            ns["build_config"]({"targets": {}})
        self.assertEqual(seen, [1])

    def test_task_registry_is_one_shared_dict(self):
        ns = _nbload.load_trainer()
        for kind in ("evidential", "regression", "classification"):
            self.assertIn(kind, ns["TASK_REGISTRY"])

    def test_default_excludes_kfold_and_smoke_test(self):
        ns = _nbload.load_trainer()
        self.assertNotIn("run_kfold_training", ns)
        self.assertIn("ensemble_predict_evidential", ns)


if __name__ == "__main__":
    unittest.main()

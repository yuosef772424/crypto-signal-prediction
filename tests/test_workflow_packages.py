"""The workflow/ package (ex main.ipynb code) and the discovery/ package (ex signal_discovery_lab.ipynb + pandas_ta_full_survey.ipynb
code), and their runner notebooks.

Guards the structure, not the numbers (numbers are covered by the other suites that execute these modules): every module file is in
the load order and carries a card, `load_into(only=, exclude=)` works on one shared dict, the runner notebooks define no function of
their own (except main's pre-repo GitHub bootstrap) and load their modules in the order the old cells defined them, the contract
tools/evaluate_trained_model.py patches (cell texts) still holds, and the lab self-test runs from the package.
"""
import ast
import json
import os
import re
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))

import _nbload  # noqa: E402
from discovery import _loader as discovery_loader  # noqa: E402
from workflow import _loader as workflow_loader  # noqa: E402

PACKAGES = {"workflow": workflow_loader, "discovery": discovery_loader}
MAIN_BOOTSTRAP_DEFS = {"github_token", "git_auth"}     # run before the repo exists on Colab, so they cannot live in workflow/


def _cells(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)["cells"]


def _code(cell):
    src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
    return "\n".join(re.match(r"\s*", ln).group() + "pass" if ln.lstrip().startswith(("!", "%")) else ln for ln in src.splitlines())


def _load_lines(name):
    """[(cell index, tuple of module names)] for every `<pkg>.load_into(globals()[, only=(...)])` line of a notebook, in notebook order;
    a line without ``only`` loads every module of the package."""
    out = []
    for i, c in enumerate(_cells(name)):
        if c["cell_type"] != "code":
            continue
        for ln in _code(c).splitlines():
            m = re.match(r"\s*(workflow|discovery)\.load_into\(globals\(\)(?:, only=\(([^)]*)\))?(?:, exclude=\(([^)]*)\))?\)", ln)
            if m:
                pkg_modules = PACKAGES[m.group(1)].MODULES
                mods = (tuple(x.strip().strip('"') for x in m.group(2).split(",") if x.strip()) if m.group(2)
                        else tuple(x for x in pkg_modules if x not in (m.group(3) or "")))
                out.append((i, mods))
    return out


class PackageStructureTests(unittest.TestCase):
    def test_every_module_file_is_loaded_and_vice_versa(self):
        for pkg, loader in PACKAGES.items():
            files = {f[:-3] for f in os.listdir(os.path.join(ROOT, pkg)) if f.endswith(".py") and not f.startswith("_")}
            self.assertEqual(files, set(loader.MODULES), pkg)
            self.assertEqual(len(loader.MODULES), len(set(loader.MODULES)), pkg)
            for m in loader.MODULES:
                self.assertTrue(loader.module_path(m).is_file(), f"{pkg}/{m}")

    def test_every_module_has_a_card(self):
        for pkg, loader in PACKAGES.items():
            for m in loader.MODULES:
                doc = ast.get_docstring(ast.parse(loader.module_path(m).read_text(encoding="utf-8"))) or ""
                for field in ("PURPOSE:", "TAGS:", "PITFALLS:"):
                    self.assertIn(field, doc, f"{pkg}/{m}.py card lacks {field}")

    def test_load_into_only_exclude_and_unknown_names(self):
        ns = workflow_loader.load_into({"__name__": "t"}, only=("splits",))
        self.assertIn("model_x", ns)
        self.assertNotIn("retarget_splits", ns)
        ns = workflow_loader.load_into({"__name__": "t"}, only=("retarget",))
        self.assertIn("retarget_splits", ns)
        self.assertNotIn("model_x", ns)
        for loader in PACKAGES.values():
            with self.assertRaises(ValueError):
                loader.load_into({}, only=("nope",))
            with self.assertRaises(ValueError):
                loader.load_into({}, exclude=("nope",))
        marker = {"__doc__": "keep me"}
        workflow_loader.load_into(marker, only=("splits",))
        self.assertEqual(marker["__doc__"], "keep me")           # module docstrings never leak into the caller's __doc__

    def test_shared_namespace_late_binding_and_patching(self):
        ns = workflow_loader.load_into({"__name__": "t"}, only=("splits",))
        self.assertEqual(ns["_tfs_of"]("1h"), ["1h"])                      # the timeframes are explicit: no notebook global is read
        self.assertEqual(ns["_tfs_of"](["1h", "4h"]), ["1h", "4h"])
        with self.assertRaises(ValueError):
            ns["_tfs_of"]()                                                # nothing to fall back on
        ns["MODEL_TF"] = "1h"
        with self.assertRaises(ValueError):
            ns["_tfs_of"]()                                                # a notebook MODEL_TF / MODEL_TFS is ignored now
        ns["_tfs_of"] = lambda model_tf=None: ["patched"]                  # patching a name changes what every function of the dict sees
        self.assertEqual(ns["model_x"]({"X_patched": 7}, "1h"), 7)

    def test_lazy_package_attribute_access(self):
        import discovery
        import workflow
        self.assertTrue(callable(workflow.retarget_splits))
        self.assertTrue(callable(discovery.run_survey))
        self.assertTrue(callable(discovery.scan_candidates))
        self.assertIn("model_health_report", dir(workflow))
        with self.assertRaises(AttributeError):
            workflow.no_such_name

    def test_workflow_names_shadow_only_the_known_notebook_name(self):
        """Loading workflow/ after the %run notebooks must not silently replace a name they define (the old notebook shadowed
        `_auc` the same way: main's cell 7-g came after the model_v2 %run)."""
        defined = {}
        # the %run notebooks are runners now: their definitions live in these packages
        for pkg in ("data", "model", "trainer", "evaluation"):
            for f in sorted(os.listdir(os.path.join(ROOT, pkg))):
                if f.endswith(".py") and not f.startswith("_"):
                    for n in ast.parse(open(os.path.join(ROOT, pkg, f), encoding="utf-8").read()).body:
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                            defined.setdefault(n.name, f"{pkg}/{f}")
        mine = {}
        for m in workflow_loader.MODULES:
            for n in ast.parse(workflow_loader.module_path(m).read_text(encoding="utf-8")).body:
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                    mine.setdefault(n.name, m)
        self.assertEqual(sorted(set(mine) & set(defined)), ["_auc"], {k: (mine[k], defined[k]) for k in set(mine) & set(defined)})


class RunnerNotebookTests(unittest.TestCase):
    def test_runners_define_no_function_of_their_own(self):
        allowed = {"main.ipynb": MAIN_BOOTSTRAP_DEFS, "signal_discovery_lab.ipynb": set(), "pandas_ta_full_survey.ipynb": set()}
        for nb, ok in allowed.items():
            for i, c in enumerate(_cells(nb)):
                if c["cell_type"] != "code":
                    continue
                defs = {n.name for n in ast.walk(ast.parse(_code(c))) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                self.assertFalse(defs - ok, f"{nb} cell {i} defines {sorted(defs - ok)}: move it to its package")

    def test_main_loads_every_workflow_module_once_after_the_runner_notebooks(self):
        """One `workflow.load_into(globals())` after the four %run cells (the modules use the names of those notebooks; generalization
        shadows model_v2's `_auc` exactly as when it was loaded after model_v2's %run), and the settings cell comes after it."""
        loads = _load_lines("main.ipynb")
        self.assertEqual(len(loads), 1, loads)
        self.assertEqual(set(loads[0][1]), set(workflow_loader.MODULES))
        cells = _cells("main.ipynb")
        run_cells = {}
        for i, c in enumerate(cells):
            for m in re.findall(r'^%run "([^"]+)"', "".join(c["source"]), re.M):
                run_cells[m] = i
        self.assertEqual(list(run_cells), ["crypto_data_pipeline_v6.ipynb", "model_v2 (1).ipynb", "trainer_framework_v2.ipynb",
                                           "chicks_v4_5_input_output_patterns.ipynb"])      # main's order, as tools/evaluate_trained_model.py runs them
        self.assertEqual(sorted(run_cells.values()), list(run_cells.values()))
        self.assertGreater(loads[0][0], max(run_cells.values()))
        settings_cell = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and "RunSettings(" in "".join(c["source"]))
        self.assertGreater(settings_cell, loads[0][0])

    def test_main_is_settings_driven(self):
        """main.ipynb: one settings cell, then one short cell per step calling workflow/run.py; no setting global is read or defined."""
        cells = [c for c in _cells("main.ipynb") if c["cell_type"] == "code"]
        code = "\n".join(_code(c) for c in cells)
        self.assertEqual(code.count("settings = RunSettings("), 1)
        steps = ["apply_project_config(", "load_dataset(", "apply_dataset_config(", "make_splits(", "retarget(", "plan_model(", "build_model(",
                 "make_training_config(", "make_datasets(", "train_model(", "prepare_chicks(", "run_chicks(", "run_panel(", "run_wiring_selftest("]
        pos = [code.index(s_) for s_ in steps]
        self.assertEqual(pos, sorted(pos), "the steps must run in this order")
        used = {n.id for c in cells for n in ast.walk(ast.parse(_code(c))) if isinstance(n, ast.Name)}
        removed = {"TARGET_MODE", "ENTRY_CLOSE_REG", "ANTI_MEMORIZATION", "CLASS_ONLY", "RUN_MAIN_TRAINING", "PANEL_MODE", "PANEL_PRESET",
                   "MODEL_TFS", "MODEL_TF", "PRICE_TARGETS", "REG_TARGET_SCALE", "MODEL_OVERRIDES", "MODEL_SEQ_LEN", "MODEL_N_FEATURES",
                   "EVAL_TARGET_SPECS", "CHICKS_TARGETS", "DATA_FILENAME_BASE", "DATA_FORMAT", "CALIBRATE_CONFIDENCE"}
        self.assertEqual(used & removed, set(), "main.ipynb still reads/defines a notebook setting global")
        self.assertNotIn("update_config(", code)                              # the project config goes through settings.project
        settings_src = next("".join(c["source"]) for c in cells if "settings = RunSettings(" in "".join(c["source"]))
        for section in ("ProjectSettings(", "DataSettings(", "TargetSettings(", "ModelSettings(", "TrainSettings(", "EvalSettings(", "PanelSettings("):
            self.assertIn(section, settings_src)

    def test_discovery_runners_load_their_modules(self):
        lab = "\n".join("".join(c["source"]) for c in _cells("signal_discovery_lab.ipynb") if c["cell_type"] == "code")
        self.assertIn('discovery.load_into(globals(), only=("axis_loader",))', lab)
        self.assertIn('discovery.load_into(globals(), exclude=("axis_loader", "survey"))', lab)
        survey = "\n".join("".join(c["source"]) for c in _cells("pandas_ta_full_survey.ipynb") if c["cell_type"] == "code")
        self.assertIn('discovery.load_into(globals(), only=("survey",))', survey)
        self.assertIn("survey", discovery_loader.MODULES)          # lab = every module but survey (axis_loader first), survey = survey only
        self.assertEqual(discovery_loader.MODULES[0], "axis_loader")


class DiscoveryFromPackageTests(unittest.TestCase):
    def test_lab_selftest_runs_from_the_package(self):
        sys.modules.setdefault("gdown", types.ModuleType("gdown"))          # the axis notebook imports it at load (Colab/Drive only)
        ns = _nbload.load_pipeline()
        ns.update(np=__import__("numpy"), pd=__import__("pandas"), __name__="lab")
        disc = _nbload.discovery_package()
        disc.load_into(ns, only=("axis_loader",))
        _nbload._repo_package("signal_eval").load_into(ns, exclude=("bootstrap",))         # the axis (as the lab runner loads it)
        disc.load_into(ns, exclude=("axis_loader", "survey"))
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ns["run_discovery_lab_selftest"]()

    def test_survey_module_runs_on_dummy_data(self):
        import pandas_ta_classic as ta
        ns = {"ta": ta, "__name__": "survey"}
        _nbload.discovery_package().load_into(ns, only=("survey",))
        ns["DUMMY_DF"] = ns["make_dummy_ohlcv"]()
        res = ns["run_survey"](indicators=["rsi", "sma"])
        self.assertEqual(set(res.status), {"ok"})
        ns["CATEGORY_HYPOTHESES"] = {}
        self.assertEqual(len(ns["build_candidate_dicts"](res)), len(res))


if __name__ == "__main__":
    unittest.main()

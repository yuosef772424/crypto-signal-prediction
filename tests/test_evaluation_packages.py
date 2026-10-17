"""The evaluation/ (ex chicks_v4_5_input_output_patterns.ipynb) and signal_eval/ (ex signal_evaluation_axis (3).ipynb) packages
and their runner notebooks.

Guards the structure, not the numbers (the numbers are covered by the tests that use these functions: test_reg_target_scale,
test_nig_calibration, test_entry_range, and the axis self-tests below): every module file is in the load order and has a card,
the shared namespace exposes the public API and is patchable, the runner notebooks define no function of their own and load
the package into their own globals, and the signal-axis runner leaves already-present pipeline names alone.
"""
import ast
import contextlib
import io
import json
import os
import sys
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))

import _nbload  # noqa: E402
from evaluation import _loader as eval_loader  # noqa: E402
from signal_eval import _loader as axis_loader  # noqa: E402

EVAL_PUBLIC = ("TargetSpec", "resolve_targets", "save_or_print", "predict_batch_v4", "decode_predictions_v4", "verify_decoding",
               "evaluate_predictions_v4", "build_latest_table", "predict_with_evaluation_v4", "test_all_assets_v4",
               "predict_latest_v4", "build_flat_dataframe", "generate_trust_report", "fit_confidence_calibrators",
               "detect_success_failure_patterns", "compute_trading_performance_metrics", "run_integrity_diagnostics",
               "run_full_analysis", "select_and_rank_trades", "run_input_output_pattern_discovery", "wilson_ci")
AXIS_PUBLIC = ("compute_ic", "decile_spread", "permutation_baseline", "evaluate_windows", "concat_splits", "extract_actuals",
               "extract_feature_last_diff", "momentum_predict_fn", "evaluate_hypothesis_over_rolling_windows",
               "register_hypothesis", "list_registry", "get_hypothesis", "run_registry_selftests", "ALL_TESTS", "check",
               "download_notebook_from_drive")
RUNNERS = {"evaluation": "chicks_v4_5_input_output_patterns.ipynb", "signal_eval": "signal_evaluation_axis (3).ipynb"}


def _notebook(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)


def _code_cells(name):
    for i, c in enumerate(_notebook(name)["cells"]):
        if c["cell_type"] == "code":
            yield i, "".join(c["source"])


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _gdown_stub():
    """signal_eval/bootstrap.py imports gdown (a Colab dependency, not in requirements-ci): stub it only when missing."""
    class _Ctx:
        def __enter__(self):
            self.added = "gdown" not in sys.modules
            if self.added:
                try:
                    import gdown  # noqa: F401
                    self.added = False
                except ImportError:
                    sys.modules["gdown"] = types.ModuleType("gdown")
        def __exit__(self, *exc):
            if self.added:
                sys.modules.pop("gdown", None)
    return _Ctx()


class PackageStructureTests(unittest.TestCase):
    def test_every_module_file_is_loaded_and_has_a_card(self):
        for pkg, loader in (("evaluation", eval_loader), ("signal_eval", axis_loader)):
            files = {f[:-3] for f in os.listdir(os.path.join(ROOT, pkg)) if f.endswith(".py") and not f.startswith("_")}
            self.assertEqual(files, set(loader.MODULES), pkg)
            for m in loader.MODULES:
                doc = ast.get_docstring(ast.parse(loader.module_path(m).read_text(encoding="utf-8")))
                self.assertTrue(doc and "PURPOSE:" in doc and "TAGS:" in doc, f"{pkg}/{m}.py has no card")

    def test_package_docstring_maps_every_module(self):
        for pkg, loader in (("evaluation", eval_loader), ("signal_eval", axis_loader)):
            doc = ast.get_docstring(ast.parse(open(os.path.join(ROOT, pkg, "__init__.py"), encoding="utf-8").read()))
            for m in loader.MODULES:
                self.assertIn(m, doc, f"{pkg}/__init__.py module map lacks {m}")

    def test_loader_rejects_unknown_names(self):
        for loader in (eval_loader, axis_loader):
            with self.assertRaises(ValueError):
                loader.module_path("nope")
            with self.assertRaises(ValueError):
                loader.load_into({}, exclude=("nope",))

    def test_runner_notebooks_define_no_function(self):
        for pkg, nb in RUNNERS.items():
            for i, src in _code_cells(nb):
                text = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%")))
                defs = [n.name for n in ast.walk(ast.parse(text)) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
                self.assertEqual(defs, [], f"{nb} cell {i} defines {defs}: move it to {pkg}/")


class EvaluationPackageTests(unittest.TestCase):
    def test_load_exposes_public_api_and_is_patchable(self):
        ns = _nbload.load_evaluation()
        for name in EVAL_PUBLIC:
            self.assertIn(name, ns)
        ns["_fmt_num"] = lambda *a, **k: "patched"          # late binding: callers see the patched name
        self.assertEqual(ns["_fmt_num"](1.0), "patched")

    def test_exclude_and_explicit_module_exec(self):
        ns = _nbload.load_evaluation(exclude=("all_assets", "live", "full_analysis"))
        for name in ("test_all_assets_v4", "predict_latest_v4", "run_full_analysis"):
            self.assertNotIn(name, ns)
        self.assertIn("predict_with_evaluation_v4", ns)
        _nbload.exec_evaluation_module("all_assets", ns)     # what exec'ing the old chicks cell 18 did
        self.assertIn("test_all_assets_v4", ns)
        self.assertNotIn("run_full_analysis", ns)

    def test_package_namespace_is_lazy(self):
        import evaluation
        self.assertTrue(callable(evaluation.resolve_targets))
        self.assertIn("wilson_ci", dir(evaluation))
        with self.assertRaises(AttributeError):
            evaluation.no_such_name


class SignalAxisPackageTests(unittest.TestCase):
    def test_load_exposes_public_api_and_runs_nothing(self):
        with _gdown_stub():
            ns = _quiet(axis_loader.load_into, {"__name__": "audit_nb"})
        for name in AXIS_PUBLIC:
            self.assertIn(name, ns)
        self.assertEqual((ns["PASS"], ns["FAIL"]), ([], []))   # loading only defines; the runner runs the self-tests
        self.assertEqual(len(ns["ALL_TESTS"]), len(set(ns["ALL_TESTS"])))

    def test_axis_selftests_pass(self):
        """What the runner notebook's self-test cell does (the old notebook did it at load)."""
        with _gdown_stub():
            ns = _quiet(axis_loader.load_into, {"__name__": "audit_nb"})
        _quiet(exec, "".join(s for i, s in _code_cells(RUNNERS["signal_eval"]) if "PASS.clear()" in s), ns)
        self.assertEqual(ns["FAIL"], [])
        self.assertEqual(len(ns["PASS"]), len(ns["ALL_TESTS"]))
        _quiet(ns["run_registry_selftests"])

    def test_registry_rejects_unknown_status(self):
        with _gdown_stub():
            ns = _quiet(axis_loader.load_into, {"__name__": "audit_nb"})
        with self.assertRaises(ValueError):
            ns["register_hypothesis"]("H", "x", "hypothesis_driven", "maybe", registry_path=os.devnull)


class RunnerNotebookTests(unittest.TestCase):
    def _run_cells(self, nb, ns, stop_before=None):
        for i, src in _code_cells(nb):
            if stop_before and stop_before in src:
                break
            src = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("!", "%")))
            _quiet(exec, compile(src, f"{nb}#cell{i}", "exec"), ns)

    def test_chicks_runner_loads_package_into_its_globals(self):
        ns = {"__name__": "__main__", "__file__": os.path.join(ROOT, RUNNERS["evaluation"])}
        self._run_cells(RUNNERS["evaluation"], ns)
        for name in EVAL_PUBLIC:
            self.assertIn(name, ns)

    def test_axis_runner_loads_package_and_keeps_existing_pipeline_names(self):
        ns = {"__name__": "__main__", "__file__": os.path.join(ROOT, RUNNERS["signal_eval"])}
        sentinel = {n: object() for n in ("rolling_splits", "CONFIG", "load_preprocessed_data_from_drive")}
        ns.update(sentinel)                                    # e.g. after %run of the pipeline notebook
        with _gdown_stub():
            self._run_cells(RUNNERS["signal_eval"], ns, stop_before="_data_pkg.load_into")   # the cell after the pipeline cell is Drive-only
            # the pipeline cell itself: names exist -> data/ is not loaded and nothing is replaced
            for i, src in _code_cells(RUNNERS["signal_eval"]):
                if "_data_pkg.load_into" in src:
                    _quiet(exec, compile(src, f"axis#cell{i}", "exec"), ns)
        for n, v in sentinel.items():
            self.assertIs(ns[n], v)
        for name in AXIS_PUBLIC:
            self.assertIn(name, ns)

    def test_axis_runner_finds_repo_from_sys_path_without_file_or_cwd(self):
        """tools/evaluate_trained_model.py runs notebook cells with no __file__ and a foreign cwd: sys.path must be enough."""
        old = os.getcwd()
        os.chdir(os.path.dirname(ROOT) if os.path.dirname(ROOT) != ROOT else "/")
        try:
            ns = {"__name__": "__main__"}
            self._run_cells(RUNNERS["evaluation"], ns)
        finally:
            os.chdir(old)
        self.assertIn("resolve_targets", ns)


if __name__ == "__main__":
    unittest.main()

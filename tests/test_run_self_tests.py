"""tools/run_self_tests.py: loading data/model/trainer/evaluation only defines (no self-test runs at load); the one explicit
entry point runs them. Offline, CPU only.
    python -m pytest tests/test_run_self_tests.py -q
"""
import ast
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import run_self_tests as rst  # noqa: E402

PACKAGES = ("data", "model", "trainer", "evaluation")
EXPLICIT_CALLS = set(rst.ENTRY_POINTS)


def test_collect_orders_inline_tests_then_entry_points_and_skips_functions_with_required_arguments():
    ns = {"_test_b": lambda: None, "_test_needs_arg": lambda x: None, "_test_not_callable": 3, "other": lambda: None,
          "run_model_selftests": lambda: None, "run_pipeline_selftests": lambda: None, "_test_a": lambda: None}
    assert [n for n, _ in rst.collect_tests(ns)] == ["_test_b", "_test_a", "run_pipeline_selftests", "run_model_selftests"]


def test_run_calls_each_once_in_order_and_returns_names():
    calls = []
    ns = {"_test_x": lambda: calls.append("x"), "run_model_selftests": lambda: calls.append("model")}
    assert rst.run_self_tests(ns, echo=lambda *_: None) == ["_test_x", "run_model_selftests"]
    assert calls == ["x", "model"]


def test_first_failure_raises_its_own_exception_and_stops():
    calls = []

    def bad():
        raise AssertionError("boom")

    ns = {"_test_bad": bad, "run_model_selftests": lambda: calls.append("model")}
    with pytest.raises(AssertionError, match="boom"):
        rst.run_self_tests(ns, echo=lambda *_: None)
    assert calls == []


def test_empty_namespace_raises_instead_of_passing_silently():
    with pytest.raises(RuntimeError, match="no self-test"):
        rst.run_self_tests({"unrelated": 1}, echo=lambda *_: None)


def _top_level_test_calls(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            name = node.value.func.id
            if name.startswith(rst.TEST_PREFIX) or name in EXPLICIT_CALLS:
                yield name
        if isinstance(node, ast.If):                                   # `if flag: run_xxx()` at module level
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and (
                        sub.func.id.startswith(rst.TEST_PREFIX) or sub.func.id in EXPLICIT_CALLS):
                    yield sub.func.id


@pytest.mark.parametrize("package", PACKAGES)
def test_no_module_of_the_package_calls_a_self_test_at_load(package):
    offenders = []
    for fname in sorted(os.listdir(os.path.join(ROOT, package))):
        if fname.endswith(".py"):
            offenders += [f"{package}/{fname}: {n}" for n in _top_level_test_calls(os.path.join(ROOT, package, fname))]
    assert offenders == []


def test_loading_the_model_selftests_module_only_defines_and_the_entry_point_runs_it():
    import _nbload
    ns = _nbload.load_model(selftests=True)                            # full load, selftests module included
    assert callable(ns["run_model_selftests"])
    ran = []
    ns["run_model_selftests"] = lambda: ran.append(1)                  # the real one builds models: tested by the model suites
    assert ran == []
    assert rst.run_self_tests(ns, echo=lambda *_: None) == ["run_model_selftests"]
    assert ran == [1]


def test_loading_the_trainer_smoke_test_only_defines_and_collects_the_inline_checks():
    import _nbload
    ns = _nbload.load_trainer(smoke_test=True)
    assert callable(ns["run_trainer_smoke_test"]) and "smoke_config" not in ns          # nothing was executed
    names = [n for n, _ in rst.collect_tests(ns)]
    assert sorted(names[:-1]) == ["_test_ema_warmup", "_test_metrics_logger_class_baselines"] and names[-1] == "run_trainer_smoke_test"


def test_data_and_evaluation_loads_define_their_checks_without_running_them():
    import _nbload
    pipeline = _nbload._data_package().load_into({"__name__": "audit_nb"})                # selftests + presets included
    assert callable(pipeline["run_pipeline_selftests"]) and callable(pipeline["run_hourly_4h_selftests"])
    assert "_test_momentum_rank" in pipeline and "_test_fractal_features_causal" in pipeline
    evaluation = _nbload.load_evaluation()
    assert {"_test_drawdown_never_exceeds_100pct", "_test_signal_significance_report"} <= set(evaluation)


def test_main_notebook_has_the_single_explicit_cell_right_after_the_loading_cells():
    cells = json.load(open(os.path.join(ROOT, "main.ipynb"), encoding="utf-8"))["cells"]
    srcs = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in cells]
    hits = [i for i, s in enumerate(srcs) if "run_self_tests(globals())" in s]
    assert len(hits) == 1
    cell = srcs[hits[0]]
    assert cells[hits[0]]["cell_type"] == "code"
    assert "# comment out when the code has not changed since the last run (SYNCED_FROM.txt shows the commit)" in cell
    assert any("workflow.load_into(globals())" in s for s in srcs[:hits[0]])          # after the loading cells

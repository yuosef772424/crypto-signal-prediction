"""S3: every former copy of a core/ constant now points at core (same object / same value), the hard-coded last_candles indexes of
cross_asset are gone and give the old numbers, and no package re-defines a core name.
    python -m pytest tests/test_core_single_source.py -q
"""
import ast
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "docs" / "research" / "audit"))
import _nbload  # noqa: E402
from core import constants, schema  # noqa: E402

CODE_DIRS = ("core", "cross_asset", "tools", "data", "model", "trainer", "evaluation", "signal_eval", "workflow", "discovery")
CORE_NAMES = {"LAST_COLUMNS", "LAST_COLUMN_INDEX", "TS_COL", "LAST_DTYPE", "TARGET_COLUMNS", "TARGET_MODES",
              "NO_RELATIVE_BASES", "ENTRY_CLOSE_REGS", "NIG_ALPHA_DEN_MIN"}


class SharedNamespacePointsAtCoreTests(unittest.TestCase):
    def test_pipeline_namespace_names_are_core_objects(self):
        ns = _nbload.load_pipeline()
        for name in ("LAST_COLUMNS", "TS_COL", "LAST_DTYPE", "TARGET_COLUMNS"):
            self.assertIs(ns[name], getattr(schema, name), name)

    def test_nig_alpha_floor_is_one_value_in_model_and_evaluation(self):
        mv = _nbload.load_model()
        ev = _nbload.load_evaluation()
        self.assertIs(mv["NIG_ALPHA_DEN_MIN"], constants.NIG_ALPHA_DEN_MIN)
        self.assertIs(ev["NIG_ALPHA_DEN_MIN"], constants.NIG_ALPHA_DEN_MIN)
        self.assertEqual(ev["nig_uncertainty_bounded"].__defaults__[-1], constants.NIG_ALPHA_DEN_MIN)

    def test_retarget_target_mode_names_are_core_objects(self):
        ns = {"__name__": "t"}
        _nbload.workflow_package().load_into(ns, only=("retarget",))
        self.assertIs(ns["TARGET_MODES"], schema.TARGET_MODES)
        self.assertIs(ns["ENTRY_CLOSE_REGS"], schema.ENTRY_CLOSE_REGS)
        self.assertIs(ns["_NO_RELATIVE_BASES"], schema.NO_RELATIVE_BASES)
        for mode in schema.TARGET_MODES:                       # every base mode still parses
            self.assertEqual(ns["_parse_mode"](mode)[0], mode)

    def test_evaluate_trained_model_choices_derive_from_core_and_equal_the_old_list(self):
        import evaluate_trained_model as etm
        old = ("return", "return_close", "scaled", "magnitude", "volnorm", "relative", "return+relative",
               "return_close+relative", "scaled+relative", "magnitude+relative", "volnorm+relative", "entry_range")
        self.assertEqual(etm.TARGET_MODE_CHOICES, old)
        self.assertEqual(etm.ENTRY_CLOSE_REG_CHOICES, ("abs_return", "range_pos"))


class ToolsAndCrossAssetTests(unittest.TestCase):
    def test_bracket_eval_and_cross_asset_columns_are_core(self):
        import bracket_eval
        from cross_asset import data as cad
        self.assertIs(bracket_eval.LAST_COLUMNS, schema.LAST_COLUMNS)
        self.assertIs(cad.LC, schema.LAST_COLUMN_INDEX)
        self.assertIs(cad.TARGETS, schema.TARGET_COLUMNS)
        self.assertEqual(cad.LC, {"last_high": 0, "last_low": 1, "last_close": 2, "timestamp": 3,
                                  "future_close": 4, "future_low_min": 5, "future_high_max": 6})   # the old literal dict
        for name, i in cad.LC.items():
            self.assertEqual(schema.LAST_COLUMNS.index(name), i)
        ns = _nbload.load_pipeline()
        self.assertEqual({n: i for i, n in enumerate(ns["LAST_COLUMNS"])}, dict(cad.LC))

    def test_cross_asset_train_functions_give_the_numbers_of_the_old_hard_coded_indexes(self):
        from cross_asset.data import PanelSplit
        from cross_asset import train as ctr
        rng = np.random.default_rng(0)
        n = 24
        # distinct, positive values per column so a wrong column changes the result: schema order high, low, close, ts, f_close, f_low, f_high
        lc = np.c_[rng.uniform(100, 110, n), rng.uniform(90, 99, n), rng.uniform(99, 101, n),
                   np.repeat(np.arange(n // 6) * 86_400 * 10**9, 6).astype("float64"),
                   rng.uniform(95, 105, n), rng.uniform(80, 98, n), rng.uniform(102, 120, n)]
        y = {f"y_{t}_{k}": rng.normal(size=n).astype("float32") for t in ("high", "low", "close") for k in ("class", "reg")}
        for t in ("high", "low", "close"):
            y[f"y_{t}_class"] = (y[f"y_{t}_class"] > 0).astype("float32")
        ps = PanelSplit(np.zeros((n, 4, 2), "float32"), y, lc, np.array([f"C{i % 6}" for i in range(n)]), "t")
        # reference = the formulas exactly as they were written with literal indexes
        np.testing.assert_array_equal(ctr.realized_return(ps), lc[:, 4] / lc[:, 2] - 1.0)
        mu = rng.normal(scale=0.01, size=(n, 3))
        t = list(ps.targets)
        m = mu / getattr(ps, "target_scale", 1.0)
        up = np.maximum(lc[:, 0] * (1.0 + m[:, t.index("high")]) / lc[:, 2] - 1.0, 1e-5)
        dn = np.maximum(1.0 - lc[:, 1] * (1.0 + m[:, t.index("low")]) / lc[:, 2], 1e-5)
        np.testing.assert_array_equal(ctr.asym_score(ps, mu), np.log(up + 1e-3) - np.log(dn + 1e-3))
        df = ctr.export_signals(ps, rng.normal(size=(n, 3)), mu, "val")
        for col, i in (("timestamp", 3), ("entry", 2), ("last_high", 0), ("last_low", 1), ("fut_close", 4), ("fut_high", 6),
                       ("fut_low", 5)):
            np.testing.assert_array_equal(df[col].to_numpy(), lc[:, i], col)


class NoLiteralCopiesTests(unittest.TestCase):
    def _code_files(self):
        for d in CODE_DIRS:
            for p in sorted((ROOT / d).rglob("*.py")):
                yield p

    def test_core_names_are_defined_only_in_core(self):
        """(lint, PHILOSOPHY rule 4) a code package may import a core name but never assign it: one definition, in core/."""
        bad = []
        for p in self._code_files():
            if p.parts[-2] == "core":
                continue
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                targets = (node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
                bad += [f"{p.relative_to(ROOT)}:{node.lineno} {t.id}" for t in targets if isinstance(t, ast.Name) and t.id in CORE_NAMES]
        self.assertEqual(bad, [])

    def test_cross_asset_has_no_hard_coded_last_candles_column_index(self):
        bad = []
        for p in sorted((ROOT / "cross_asset").glob("*.py")):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                    base = ast.unparse(node.value)
                    col = node.slice.elts[1]
                    if base.split(".")[-1] in ("lc", "last_candles") and isinstance(col, ast.Constant) and isinstance(col.value, int):
                        bad.append(f"{p.name}:{node.lineno} {ast.unparse(node)}")
        self.assertEqual(bad, [])


class ColabRunnersTests(unittest.TestCase):
    def test_every_runner_notebook_puts_the_repo_root_on_sys_path_before_importing_its_package(self):
        """core/ sits next to each package, so the root they insert for `import data|model|...` is what makes `import core` work."""
        pairs = {"crypto_data_pipeline_v6.ipynb": "data", "model_v2 (1).ipynb": "model", "chicks_v4_5_input_output_patterns.ipynb": "evaluation",
                 "main.ipynb": "workflow", "signal_discovery_lab.ipynb": "discovery", "pandas_ta_full_survey.ipynb": "discovery"}
        for nb, pkg in pairs.items():
            cells = json.load(open(ROOT / nb, encoding="utf-8"))["cells"]
            boot = [("".join(c["source"])) for c in cells if c["cell_type"] == "code" and f'"{pkg}", "_loader.py"' in "".join(c["source"])]
            self.assertTrue(boot, nb)
            src = boot[0]
            self.assertIn("_sys.path.insert(0,", src, nb)
            self.assertTrue((ROOT / pkg / "_loader.py").is_file() and (ROOT / "core" / "schema.py").is_file())
            self.assertEqual(os.path.dirname(os.path.dirname(ROOT / pkg / "_loader.py")), str(ROOT))   # package and core share a parent

    def test_core_imports_from_a_directory_that_only_has_the_repo_root_on_the_path(self):
        import subprocess
        import tempfile
        code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from core.schema import LAST_COLUMNS, TS_COL; from core.constants import NIG_ALPHA_DEN_MIN; print(TS_COL)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "3"), out.stderr)


if __name__ == "__main__":
    unittest.main()

"""core/ (Tier 0): the values are frozen to the literals the packages used before S3, the package imports nothing from the repo
(checked by tools/check_deps.py), and importing it has no side effects.
    python -m pytest tests/test_core.py -q
"""
import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import check_deps  # noqa: E402
from core import constants, schema  # noqa: E402

# the literals as they stood in data/windows.py, cross_asset/data.py, tools/bracket_eval.py, workflow/retarget.py, evaluation/targets.py
_OLD_LAST_COLUMNS = ['last_high', 'last_low', 'last_close', 'timestamp', 'future_close', 'future_low_min', 'future_high_max']
_OLD_CROSS_ASSET_LC = {"last_high": 0, "last_low": 1, "last_close": 2, "timestamp": 3,
                       "future_close": 4, "future_low_min": 5, "future_high_max": 6}


class ValuesTests(unittest.TestCase):
    def test_schema_values_are_the_old_literals(self):
        self.assertEqual(list(schema.LAST_COLUMNS), _OLD_LAST_COLUMNS)
        self.assertEqual(dict(schema.LAST_COLUMN_INDEX), _OLD_CROSS_ASSET_LC)
        self.assertEqual(schema.LAST_COLUMN_INDEX, _OLD_CROSS_ASSET_LC)
        self.assertEqual(schema.TS_COL, 3)
        self.assertEqual(schema.LAST_DTYPE, "float64")
        self.assertEqual(schema.TARGET_COLUMNS, ("high", "low", "close"))
        self.assertEqual(schema.TARGET_MODES, ("return", "return_close", "scaled", "magnitude", "volnorm", "entry_range"))
        self.assertEqual(schema.NO_RELATIVE_BASES, ("entry_range",))
        self.assertEqual(schema.ENTRY_CLOSE_REGS, ("abs_return", "range_pos"))
        self.assertEqual(constants.NIG_ALPHA_DEN_MIN, 1e-2)

    def test_schema_is_immutable(self):
        self.assertIsInstance(schema.LAST_COLUMNS, tuple)
        with self.assertRaises(TypeError):
            schema.LAST_COLUMN_INDEX["extra"] = 7

    def test_index_mapping_is_consistent(self):
        for i, name in enumerate(schema.LAST_COLUMNS):
            self.assertEqual(schema.LAST_COLUMN_INDEX[name], i)
        self.assertEqual(len(schema.LAST_COLUMN_INDEX), len(schema.LAST_COLUMNS))


class TierZeroTests(unittest.TestCase):
    def test_core_modules_import_only_the_stdlib(self):
        stdlib = sys.stdlib_module_names
        for f in sorted((ROOT / "core").glob("*.py")):
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                mods = ([a.name for a in node.names] if isinstance(node, ast.Import)
                        else [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
                for m in mods:
                    self.assertIn(m.split(".")[0], stdlib, f"{f.name} imports {m}")

    def test_importing_core_has_no_side_effects_and_loads_no_repo_package(self):
        code = ("import sys; sys.path.insert(0, %r); before = set(sys.modules); import core.schema, core.constants; "
                "new = sorted(set(sys.modules) - before); "
                "bad = [m for m in new if m.split('.')[0] not in ('core', 'types')]; "
                "assert not bad, bad; print('ok')" % str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tempfile.gettempdir())
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "ok"), out.stderr)

    def test_check_deps_flags_a_repo_import_inside_core_and_accepts_the_real_core(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "core").mkdir()
            (root / "data").mkdir()
            (root / "core" / "bad.py").write_text('"""card"""\nimport json\nfrom data import windows\nimport tools.x\n', encoding="utf-8")
            (root / "core" / "good.py").write_text('"""card"""\nimport json\nfrom . import bad\n', encoding="utf-8")
            (root / "data" / "windows.py").write_text("x = 1\n", encoding="utf-8")
            repo = check_deps.Repo(root, files=["core/bad.py", "core/good.py", "data/windows.py"])
            bad = [f"{rel}:{ln}" for rel in repo.py_files for ln, sev, rule, _t in check_deps.check_file(repo, rel)
                   if rule == "core-must-not-import-repo" and sev == "error"]
            self.assertEqual(bad, ["core/bad.py:3", "core/bad.py:4"])
        lines, _n, errors, _w, _a = check_deps.run(check_deps.Repo(ROOT))
        self.assertEqual(errors, 0, lines)

    def test_every_package_may_import_core(self):
        for pkg in check_deps.CODE_PACKAGES:
            self.assertIsNone(check_deps.violation_for(pkg, "core") if pkg != "core" else None, pkg)
        self.assertIn("core", check_deps.CODE_PACKAGES)
        self.assertEqual(check_deps.zone_of("core/schema.py"), "core")


if __name__ == "__main__":
    unittest.main()

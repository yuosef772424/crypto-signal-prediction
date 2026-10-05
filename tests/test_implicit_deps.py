"""tools/implicit_deps.py: the free-name / definition analysis of the shared-namespace packages, its allowlist ratchet and
the USES / USED BY lines of the maps. Synthetic mini-repos in a temp dir (stdlib only) plus the live repo.
    python -m pytest tests/test_implicit_deps.py -q
"""
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import check_deps  # noqa: E402
import implicit_deps as idp  # noqa: E402


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def _mini_repo(root: Path) -> None:
    """data: common -> a -> b ; model: m (uses data names, a notebook global, and defines a duplicate of data's name)."""
    for pkg, mods in (("data", ("common", "a", "b")), ("model", ("m",))):
        _write(root, f"{pkg}/__init__.py", '"""pkg"""\n')
        _write(root, f"{pkg}/_loader.py", f"MODULES = {mods!r}\n")
    _write(root, "core/__init__.py", "")
    _write(root, "data/common.py", "import numpy as np\nimport os\n")
    _write(root, "data/a.py", """
        from core import SCHEMA
        def helper(x):
            return np.asarray(x) + later_value + helper2(x)   # np: shared import; later_value: defined in b (forward)
        def helper2(x):
            return [y for y in x if y > 0]
        DUP = 1
    """)
    _write(root, "data/b.py", """
        later_value = 3
        def uses_alias():
            return SCHEMA, helper(1), len("builtin ok"), os.sep
        DUP = 2
    """)
    _write(root, "model/m.py", """
        def build(cfg=None):
            return helper(main_config), later_value        # helper/later_value: another package; main_config: nobody
        class Local:
            def go(self):
                return self.go, [helper2(i) for i in range(3)]
        DUP = 3
    """)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _mini_repo(self.root)
        self.an = idp.analyse(self.root)
        self.kind = {(r.module, r.name): (r.kind, r.target) for r in self.an.refs}

    def tearDown(self):
        self._tmp.cleanup()

    def test_modules_follow_loader_order(self):
        self.assertEqual(list(self.an.mods), ["data/common.py", "data/a.py", "data/b.py", "model/m.py"])
        self.assertEqual(idp.package_modules(self.root, "data"), ["common", "a", "b"])
        self.assertEqual(idp.package_modules(self.root, "workflow"), [])  # no loader -> not a shared-namespace package

    def test_reference_kinds(self):
        k = self.kind
        self.assertEqual(k[("data/a.py", "np")], ("import", "data/common.py"))          # shared import of an EARLIER module
        self.assertEqual(k[("data/a.py", "later_value")], ("forward", "data/b.py"))      # LATER module
        self.assertEqual(k[("data/b.py", "helper")], ("same", "data/a.py"))
        self.assertEqual(k[("data/b.py", "os")], ("import", "data/common.py"))
        self.assertEqual(k[("data/b.py", "SCHEMA")], ("same", "data/a.py"))             # `from core import X` is a real definition
        self.assertEqual(k[("model/m.py", "helper")], ("cross", "data/a.py"))
        self.assertEqual(k[("model/m.py", "main_config")], ("notebook", ""))
        self.assertEqual(k[("model/m.py", "helper2")], ("cross", "data/a.py"))          # used inside a comprehension in a method
        self.assertNotIn(("data/b.py", "len"), k)                                       # builtins are not free names
        self.assertNotIn(("model/m.py", "range"), k)
        self.assertNotIn(("data/a.py", "y"), k)                                         # comprehension variable is local

    def test_defs_and_kinds(self):
        self.assertEqual(self.an.mods["data/a.py"].defs["SCHEMA"], "alias")
        self.assertEqual(self.an.mods["data/common.py"].defs["np"], "import")
        self.assertEqual(self.an.mods["model/m.py"].defs["Local"], "class")

    def test_duplicates_ignore_imports_and_aliases(self):
        self.assertEqual(idp.duplicates(self.an), {"DUP": ["data/a.py", "data/b.py", "model/m.py"]})

    def test_used_by_and_map_lines(self):
        self.assertEqual(list(self.an.used_by("data/a.py")), ["data/b.py", "model/m.py"])
        self.assertEqual(idp.map_lines(self.an, "data/a.py"), ["- USES: b*", "- USED BY: b, model/m"])
        self.assertEqual(idp.map_lines(self.an, "model/m.py"), ["- USES: data/a, data/b", "- NOTEBOOK-GLOBALS: main_config"])
        self.assertEqual(idp.map_lines(self.an, "data/common.py"), [])                   # only shared imports used

    def test_global_declared_in_function_is_a_definition(self):
        defs, free, _ = idp.analyse_source("def f():\n    global G\n    G = 1\ndef g():\n    return G + other\n", "x.py")
        self.assertEqual(defs["G"], "assign")
        self.assertEqual(sorted(free), ["other"])


class RatchetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        _mini_repo(self.root)
        self.an = idp.analyse(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def _allow(self, keys):
        (self.root / idp.ALLOWLIST_PATH).parent.mkdir(exist_ok=True)
        (self.root / idp.ALLOWLIST_PATH).write_text(idp.allowlist_text(self.an, keys), encoding="utf-8")

    def test_everything_is_a_new_problem_without_an_allowlist(self):
        problems, allowed = idp.check(self.root)
        self.assertEqual(allowed, 0)
        self.assertEqual(len(problems), len(idp.findings(self.an)))
        self.assertTrue(any("main_config" in p for p in problems))
        self.assertTrue(any("DUP" in p for p in problems))

    def test_allowlist_covers_today_and_a_new_reference_fails(self):
        self._allow(list(idp.findings(self.an)))
        self.assertEqual(idp.check(self.root)[0], [])
        _write(self.root, "model/m.py", (self.root / "model/m.py").read_text() + "\ndef extra():\n    return uses_alias, brand_new_global\n")
        problems, _ = idp.check(self.root)
        self.assertEqual(len(problems), 2, problems)  # a new notebook global AND a new cross-package reference
        self.assertTrue(any("brand_new_global" in p for p in problems))

    def test_stale_entry_fails_so_the_allowlist_only_shrinks(self):
        self._allow(list(idp.findings(self.an)) + ["cross model/m.py gone_name"])
        problems, _ = idp.check(self.root)
        self.assertEqual(len(problems), 1)
        self.assertIn("stale entry", problems[0])

    def test_allowlist_round_trip_and_target_note_is_ignored(self):
        keys = sorted(idp.findings(self.an))
        self._allow(keys)
        self.assertEqual(sorted(idp.read_allowlist(self.root)), keys)
        text = (self.root / idp.ALLOWLIST_PATH).read_text()
        self.assertIn("cross model/m.py helper -> data/a.py", text)

    def test_check_deps_reports_implicit_problems_as_errors(self):
        repo = check_deps.Repo(self.root, files=["data/a.py"])
        lines, _n, errors, _w, _allowed = check_deps.run(repo)
        self.assertGreater(errors, 0)
        self.assertTrue(any("error[implicit-dependency]" in ln for ln in lines))


class LiveRepoTests(unittest.TestCase):
    def test_live_repo_has_no_new_implicit_dependency_and_no_stale_allowlist_entry(self):
        problems, allowed = idp.check(ROOT)
        self.assertEqual(problems, [])
        self.assertGreater(allowed, 0)

    def test_every_allowlist_kind_is_gated_and_known(self):
        for key in idp.read_allowlist(ROOT):
            self.assertIn(key.split(" ", 1)[0], idp.GATED_KINDS, key)

    def test_loader_modules_all_exist(self):
        for pkg in idp.SHARED_PACKAGES:
            for name in idp.package_modules(ROOT, pkg):
                self.assertTrue((ROOT / pkg / f"{name}.py").is_file(), f"{pkg}/{name}.py")


if __name__ == "__main__":
    unittest.main()

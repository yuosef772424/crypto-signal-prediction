"""tools/check_patterns.py: the generic pattern ratchet. A planted violation per rule must fail, the baseline can only
shrink (growth is refused by --update-baseline), and the live repo is green against tools/pattern_baseline.json.
Synthetic mini-repos in a temp dir (no git: the tool walks the tree) plus the live repo.
    python -m pytest tests/test_check_patterns.py -q
"""
import contextlib
import io
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import check_deps  # noqa: E402
import check_patterns as cp  # noqa: E402

#: rule name -> (path, source with exactly the planted violation(s), expected number of matches)
PLANTED = {
    "output-name-literal": ("model/plant.py", 'def f(t):\n    return f"y_{t}_class_logits"\n', 1),
    "hlc-target-tuple": ("workflow/plant.py", 'COLS = ("high", "low", "close")\n', 1),
    "config-get-default": ("trainer/plant.py", 'def f(config):\n    return config.get("patience", 5)\n', 1),
    "target-mode-branch": ("workflow/plant.py", 'def f(mode):\n    if mode == "entry_range":\n        return 1\n    return 0\n', 1),
    "seed-literal": ("evaluation/plant.py", 'import numpy as np\nrng = np.random.default_rng(42)\n', 1),
    "metric-def-outside-stats": ("evaluation/plant.py", 'def my_auc(y, p):\n    return 0.5\n', 1),
    "per-asset-model-call-loop": ("evaluation/plant.py",
                                  'def f(model, test_dict):\n    for asset, d in test_dict.items():\n        model.predict(d)\n', 1),
}

#: rule name -> sources that must NOT match (the false-positive guards of each matcher)
CLEAN = {
    "output-name-literal": ['"""docstring mentioning y_{t}_class_logits"""\nx = "y_axis"\nk = f"{a}_reg"\n'],
    "hlc-target-tuple": ['A = ("high", "low")\nB = ("high", "low", "close", "open")\nC = (1, 2, 3)\n'],
    "config-get-default": ['def f(config, d):\n    a = config.get("k")\n    b = config.get("k", None)\n    c = d.get("k", 3)\n'
                           '    return a, b, c\n'],
    "target-mode-branch": ['def f(x, mode):\n    if x == "return":\n        pass\n    if mode == "unknown_mode":\n        pass\n'
                           '    return x == "scaled"\n'],
    "seed-literal": ['import numpy as np\ndef f(seed):\n    rng = np.random.default_rng(seed)\n    return rng, g(seed=seed), h(seed=None)\n'],
    "metric-def-outside-stats": ['def test_auc():\n    pass\ndef t_ic_check():\n    pass\ndef pauc2(): pass\ndef basic(): pass\n'],
    "per-asset-model-call-loop": ['def f(model, batches, assets):\n    for b in batches:\n        model.predict(b)\n'
                                  '    return [a for a in assets]\n'],
}


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cp.main(argv)
    return code, out.getvalue(), err.getvalue()


class RulesTests(unittest.TestCase):
    def test_every_rule_has_a_planted_and_a_clean_case(self):
        self.assertEqual({r.name for r in cp.RULES}, set(PLANTED))
        self.assertEqual({r.name for r in cp.RULES}, set(CLEAN))
        for r in cp.RULES:  # each rule documents its purpose and the audit class it guards
            self.assertRegex(r.guards, r"^C[1-8]$")
            self.assertGreater(len(r.purpose), 40, r.name)

    def test_planted_violation_matches_and_fails_the_check(self):
        for name, (rel, src, n) in PLANTED.items():
            with self.subTest(rule=name):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    _write(root, rel, src)
                    found = cp.scan(root)[name]
                    self.assertEqual(len(found.get(rel, [])), n, found)
                    # frozen at an empty baseline -> the planted violation is growth -> check fails
                    (root / "tools").mkdir(exist_ok=True)
                    (root / cp.BASELINE_PATH).write_text(cp.format_baseline({r.name: {} for r in cp.RULES}))
                    problems, _ = cp.check(root)
                    self.assertTrue(any(f"[{name}]" in p and rel in p for p in problems), problems)

    def test_clean_sources_do_not_match(self):
        for name, sources in CLEAN.items():
            for i, src in enumerate(sources):
                with self.subTest(rule=name, case=i):
                    with tempfile.TemporaryDirectory() as td:
                        root = Path(td)
                        _write(root, "workflow/clean.py", src)
                        self.assertEqual(cp.scan(root)[name], {})

    def test_more_forms(self):
        cases = [
            ("output-name-literal", 'k = "y_{}_nu".format(t)\nj = "y_%s_alpha" % t\nm = k.endswith("_class_logits")\n', 3),
            ("hlc-target-tuple", 'A = ["close", "high", "low"]\nB = {"high", "low", "close"}\n', 2),
            ("config-get-default", 'def f(self, cfg):\n    a = self.config.get("a", 1)\n    b = cfg["opt"].get("b", {})\n'
                                   '    c = cfg.get("c", default=2)\n    return a, b, c\n', 3),
            ("target-mode-branch", 'def f(base, m):\n    a = base == "return"\n    b = m.startswith("entry_range")\n'
                                   '    c = base in ("scaled", "volnorm")\n    return a, b, c\n', 3),
            ("seed-literal", 'import tensorflow as tf\ntf.keras.utils.set_random_seed(0)\nmake(seed=7)\nR = RandomState(3)\n', 3),
            ("metric-def-outside-stats", 'def a_roc_auc(): pass\nclass K:\n    def _rank_ic(self): pass\n'
                                         'def wilson_interval(): pass\ndef pinball(): pass\ndef _ece_quantile(): pass\n'
                                         'def spearman(): pass\ndef day_pearson(): pass\n', 7),
            ("per-asset-model-call-loop", 'def f(model_tf, coins):\n    return [model_tf(c) for c in coins]\n', 1),
        ]
        for name, src, n in cases:
            with self.subTest(rule=name):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    _write(root, "workflow/x.py", src)
                    self.assertEqual(sum(len(m) for m in cp.scan(root)[name].values()), n, cp.scan(root)[name])

    def test_scope_tests_research_docs_and_self_are_not_scanned(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in ("tests/t.py", "research/s/a.py", "docs/d.py", "tools/check_patterns.py"):
                _write(root, rel, PLANTED["hlc-target-tuple"][1] + PLANTED["seed-literal"][1] + PLANTED["output-name-literal"][1])
            _write(root, "data/selftests.py", PLANTED["seed-literal"][1])
            self.assertEqual({r: f for r, f in cp.scan(root).items() if f}, {})

    def test_config_get_rule_is_limited_to_config_consuming_packages(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "tools/x.py", PLANTED["config-get-default"][1])
            self.assertEqual(cp.scan(root)["config-get-default"], {})
            _write(root, "model/x.py", PLANTED["config-get-default"][1])
            self.assertEqual(list(cp.scan(root)["config-get-default"]), ["model/x.py"])

    def test_regex_rule_kind(self):
        rule = cp.Rule("todo", "C8", "regex rule used to test the regex matcher kind of the ratchet", globs=("*.py",),
                       regex=r"XXX")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "x = 1  # XXX\ny = 2\n# XXX again\n")
            self.assertEqual(cp.counts_of(cp.scan(root, [rule])), {"todo": {"a.py": 2}})
        with self.assertRaises(ValueError):
            cp.Rule("bad", "C8", "neither matcher nor regex given", globs=("*.py",))

    def test_glob_double_star_matches_top_level(self):
        self.assertTrue(cp._glob("model/x.py", "model/**/*.py"))
        self.assertTrue(cp._glob("model/a/b/x.py", "model/**/*.py"))
        self.assertFalse(cp._glob("modelx/x.py", "model/**/*.py"))


class BaselineTests(unittest.TestCase):
    def _repo(self, td: str) -> Path:
        root = Path(td)
        _write(root, "workflow/a.py", 'A = ("high", "low", "close")\nB = ("high", "low", "close")\n')
        _write(root, "workflow/b.py", 'C = ("high", "low", "close")\n')
        return root

    def test_freeze_then_green_then_growth_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            self.assertEqual(_run(["--root", td, "--update-baseline"])[0], 0)
            base = json.loads((root / cp.BASELINE_PATH).read_text())
            self.assertEqual(base["rules"]["hlc-target-tuple"], {"workflow/a.py": 2, "workflow/b.py": 1})
            self.assertEqual(cp.check(root)[0], [])
            self.assertEqual(_run(["--root", td])[0], 0)
            # growth in an existing file, and a violation in a new file
            _write(root, "workflow/a.py", 'A = ("high", "low", "close")\nB = ("high", "low", "close")\nZ = ("low", "high", "close")\n')
            _write(root, "workflow/new.py", 'N = ("high", "low", "close")\n')
            code, out, _ = _run(["--root", td])
            self.assertEqual(code, 1)
            self.assertIn("workflow/a.py", out)
            self.assertIn("workflow/new.py", out)
            self.assertIn("pattern-ratchet", out)

    def test_update_baseline_refuses_growth_and_leaves_file_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            _run(["--root", td, "--update-baseline"])
            before = (root / cp.BASELINE_PATH).read_text()
            _write(root, "workflow/new.py", 'N = ("high", "low", "close")\n')
            code, _out, err = _run(["--root", td, "--update-baseline"])
            self.assertEqual(code, 1)
            self.assertIn("refused (growth)", err)
            self.assertEqual((root / cp.BASELINE_PATH).read_text(), before)

    def test_update_baseline_refuses_even_when_something_else_shrank(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            _run(["--root", td, "--update-baseline"])
            before = (root / cp.BASELINE_PATH).read_text()
            _write(root, "workflow/b.py", "x = 1\n")                              # shrink
            _write(root, "workflow/new.py", 'N = ("high", "low", "close")\n')    # growth elsewhere
            self.assertEqual(_run(["--root", td, "--update-baseline"])[0], 1)
            self.assertEqual((root / cp.BASELINE_PATH).read_text(), before)

    def test_shrink_is_stale_until_recorded_and_then_green(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            _run(["--root", td, "--update-baseline"])
            _write(root, "workflow/a.py", 'A = ("high", "low", "close")\n')
            _write(root, "workflow/b.py", "x = 1\n")
            code, out, _ = _run(["--root", td])
            self.assertEqual(code, 1)
            self.assertIn("baseline records", out)
            self.assertEqual(_run(["--root", td, "--update-baseline"])[0], 0)
            base = json.loads((root / cp.BASELINE_PATH).read_text())["rules"]["hlc-target-tuple"]
            self.assertEqual(base, {"workflow/a.py": 1})  # b.py dropped at zero
            self.assertEqual(_run(["--root", td])[0], 0)

    def test_moving_a_finding_to_another_file_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            _run(["--root", td, "--update-baseline"])
            _write(root, "workflow/b.py", "x = 1\n")
            _write(root, "workflow/c.py", 'C = ("high", "low", "close")\n')
            problems, _ = cp.check(root)
            self.assertTrue(any("workflow/c.py" in p and "baseline allows 0" in p for p in problems), problems)

    def test_missing_baseline_fails_check(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "workflow/ok.py", "x = 1\n")
            problems, _ = cp.check(root)
            self.assertTrue(any("missing" in p for p in problems), problems)

    def test_shrink_only_helper_never_raises_a_count(self):
        new, refused = cp.shrink_only({"r": {"a": 5, "b": 1}}, {"r": {"a": 3, "b": 2, "c": 1}})
        self.assertEqual(new, {"r": {"a": 3, "b": 1}})
        self.assertEqual(len(refused), 1)
        self.assertIn("a", refused[0])

    def test_freeze_rule_adds_a_new_rule_only_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            (root / "tools").mkdir(exist_ok=True)
            partial = {r.name: {} for r in cp.RULES if r.name != "hlc-target-tuple"}
            (root / cp.BASELINE_PATH).write_text(cp.format_baseline(partial))
            self.assertEqual(_run(["--root", td, "--update-baseline"])[0], 1)  # a rule is missing from the baseline
            self.assertEqual(_run(["--root", td, "--freeze-rule", "hlc-target-tuple"])[0], 0)
            self.assertEqual(_run(["--root", td, "--freeze-rule", "hlc-target-tuple"])[0], 1)  # already frozen: refused
            self.assertEqual(_run(["--root", td, "--freeze-rule", "no-such-rule"])[0], 2)
            self.assertEqual(cp.check(root)[0], [])

    def test_wired_into_check_deps(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._repo(td)
            _run(["--root", td, "--update-baseline"])
            repo = check_deps.Repo(root, files=["workflow/a.py", "workflow/b.py"])
            lines, _n, errors, _w, allowed = check_deps.run(repo)
            self.assertFalse([ln for ln in lines if "pattern-ratchet" in ln], lines)
            self.assertEqual(allowed >= 3, True)
            _write(root, "workflow/new.py", 'N = ("high", "low", "close")\n')
            lines, _n, errors, _w, _a = check_deps.run(check_deps.Repo(root, files=["workflow/a.py", "workflow/b.py", "workflow/new.py"]))
            self.assertGreaterEqual(errors, 1)
            self.assertTrue(any("pattern-ratchet" in ln and "workflow/new.py" in ln for ln in lines), lines)


class LiveRepoTests(unittest.TestCase):
    def test_live_repo_is_green_against_the_frozen_baseline(self):
        problems, allowed = cp.check(ROOT)
        self.assertEqual(problems, [])
        self.assertGreater(allowed, 0)

    def test_baseline_file_covers_every_rule_and_is_canonical(self):
        baseline = cp.read_baseline(ROOT)
        self.assertIsNotNone(baseline)
        self.assertEqual(set(baseline), {r.name for r in cp.RULES})
        text = (ROOT / cp.BASELINE_PATH).read_text(encoding="utf-8")
        self.assertEqual(text, cp.format_baseline(baseline), "baseline must be in the tool's canonical format")
        for rule, files in baseline.items():
            for f, n in files.items():
                self.assertGreater(n, 0, (rule, f))
                self.assertTrue((ROOT / f).is_file(), f)

    def test_the_audit_sites_are_found(self):
        found = cp.scan(ROOT)
        loops = set(found["per-asset-model-call-loop"])
        self.assertTrue({"evaluation/all_assets.py", "evaluation/live.py", "workflow/reports.py"} <= loops, loops)
        self.assertIn("model/diagnostics.py", found["output-name-literal"])
        self.assertIn("workflow/retarget.py", found["target-mode-branch"])
        self.assertIn("trainer/system.py", found["seed-literal"])
        self.assertIn("signal_eval/core.py", found["metric-def-outside-stats"])
        self.assertIn("model/config.py", found["hlc-target-tuple"])


if __name__ == "__main__":
    unittest.main()

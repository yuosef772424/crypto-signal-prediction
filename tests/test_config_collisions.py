"""S1 of docs/architecture_review.md: the config collisions found by the architecture review (W2).

1. `reset_config()` must reset the pipeline CONFIG to the PIPELINE defaults even when the trainer's `DEFAULT_CONFIG` was
   loaded into the same namespace after (main.ipynb: %run pipeline at cell 3, trainer at cell 17), and the trainer's
   `build_config` must keep using the TRAINER defaults whichever package was loaded last.
2. `update_config` rejects unknown top-level keys (CLAUDE.md: "No silent defaults: unknown config keys ... raise").
3. The lazy `import workflow` package is standalone: its CONFIG is its own copy; `workflow.load_into(ns)` shares ns's CONFIG.
    python -m pytest tests/test_config_collisions.py -q
"""
import ast
import json
import os
import sys
import unittest
from copy import deepcopy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402

_TARGET = {"targets": {"t": {"true_key": "y_t", "task_type": "regression", "output_keys": ["p"]}}}


def _data_then_trainer():
    ns = _nbload.load_pipeline()                    # what main.ipynb does: %run pipeline (cell 3) ...
    return _nbload.load_trainer(ns)                 # ... then %run trainer (cell 17), same globals()


def _trainer_then_data():
    ns = _nbload.load_trainer()
    _nbload._data_package().load_into(ns, exclude=("selftests",))
    return ns


class ResetConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_only = _nbload.load_pipeline()
        cls.trainer_only = _nbload.load_trainer()
        cls.pipeline_keys = set(cls.pipeline_only["PIPELINE_DEFAULT_CONFIG"])
        cls.trainer_keys = set(cls.trainer_only["build_config"](deepcopy(_TARGET)))

    def test_the_two_default_dicts_really_differ(self):
        self.assertGreater(len(self.pipeline_keys), 90)
        self.assertEqual(len(self.trainer_keys), 6)                               # the trainer's 6 sections (trainer/config.py)
        self.assertLess(len(self.pipeline_keys & self.trainer_keys), 3)

    def test_reset_config_keeps_pipeline_defaults_when_trainer_is_loaded_after(self):
        ns = _data_then_trainer()
        cfg = ns["CONFIG"]
        ns["update_config"]({"stride": 7, "epochs": 3})
        out = ns["reset_config"]()
        self.assertIs(out, cfg)                                                  # still the one live dict
        self.assertEqual(set(cfg), self.pipeline_keys)                           # not the trainer's 6-7 keys
        self.assertEqual(cfg["higher_tf_mode"], "legacy")
        self.assertEqual(cfg["stride"], self.pipeline_only["PIPELINE_DEFAULT_CONFIG"]["stride"])

    def test_load_config_without_merge_resets_to_pipeline_defaults(self):
        import tempfile
        ns = _data_then_trainer()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"stride": 5}, f)
            ns["load_config"](path, merge=False)
        self.assertEqual(set(ns["CONFIG"]), self.pipeline_keys)
        self.assertEqual(ns["CONFIG"]["stride"], 5)

    def test_trainer_build_config_uses_trainer_defaults_whichever_loaded_last(self):
        for make in (_data_then_trainer, _trainer_then_data):
            ns = make()
            cfg = ns["build_config"](deepcopy(_TARGET))
            self.assertEqual(set(cfg), self.trainer_keys, make.__name__)

    def test_public_default_config_alias_is_kept_in_both_packages(self):
        self.assertIs(self.pipeline_only["DEFAULT_CONFIG"], self.pipeline_only["PIPELINE_DEFAULT_CONFIG"])
        self.assertIs(self.trainer_only["DEFAULT_CONFIG"], self.trainer_only["TRAINER_DEFAULT_CONFIG"])

    def test_defaults_are_not_mutated_by_reset_or_update(self):
        ns = _nbload.load_pipeline()
        before = deepcopy(ns["PIPELINE_DEFAULT_CONFIG"])
        ns["update_config"]({"stride": 99, "phase2_data": {"use_intraday_15m": False}})
        ns["reset_config"]()
        self.assertEqual(ns["PIPELINE_DEFAULT_CONFIG"], before)


class UpdateConfigStrictTests(unittest.TestCase):
    def setUp(self):
        self.ns = _nbload.load_pipeline()
        self.ns["reset_config"]()

    def test_unknown_top_level_key_raises_and_lists_the_valid_keys(self):
        for call in (lambda: self.ns["update_config"](seed=12345),
                     lambda: self.ns["update_config"]({"strde": 3}),
                     lambda: self.ns["update_config"]({"stride": 3}, no_such_key=1)):
            with self.assertRaises(ValueError) as cm:
                call()
            msg = str(cm.exception)
            self.assertIn("stride", msg)                                          # the valid keys are listed
            self.assertIn("higher_tf_mode", msg)
        self.assertNotIn("seed", self.ns["CONFIG"])

    def test_nothing_is_applied_when_any_key_is_unknown(self):
        before = deepcopy(self.ns["CONFIG"])
        with self.assertRaises(ValueError):
            self.ns["update_config"]({"stride": 11, "bogus": 1})
        self.assertEqual(self.ns["CONFIG"], before)

    def test_known_keys_work_as_before(self):
        cfg = self.ns["CONFIG"]
        out = self.ns["update_config"]({"phase2_data": {"use_intraday_15m": False}}, stride=7, epochs=2)
        self.assertIs(out, cfg)
        self.assertEqual((cfg["stride"], cfg["epochs"], cfg["phase2_data"]["use_intraday_15m"]), (7, 2, False))
        self.assertIs(self.ns["update_config"](), cfg)                            # no-op calls stay fine
        self.assertIs(self.ns["update_config"]({}), cfg)

    def test_every_default_key_is_accepted(self):
        for k, v in self.ns["PIPELINE_DEFAULT_CONFIG"].items():
            self.ns["update_config"]({k: deepcopy(v)})

    def test_every_literal_update_config_call_in_the_repo_uses_known_keys(self):
        """Notebooks, tests, tools, docs/research scripts: a literal key passed to update_config is a DEFAULT_CONFIG key."""
        known = set(self.ns["PIPELINE_DEFAULT_CONFIG"])
        bad = []
        for rel, tree in _all_trees():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else \
                    f.slice.value if isinstance(f, ast.Subscript) and isinstance(f.slice, ast.Constant) else ""
                if name != "update_config":
                    continue
                keys = [k.arg for k in node.keywords if k.arg]
                for a in node.args[:1]:
                    if isinstance(a, ast.Dict):
                        keys += [k.value for k in a.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
                bad += [(rel, k) for k in keys if k not in known]
        self.assertEqual(bad, [])


def _all_trees():
    skip = {".git", "__pycache__", ".claude"}
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in skip]
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, ROOT)
            if rel == os.path.join("tests", "test_config_collisions.py"):
                continue
            if fn.endswith(".py"):
                try:
                    yield rel, ast.parse(open(p, encoding="utf-8").read())
                except SyntaxError:
                    pass
            elif fn.endswith(".ipynb") and dp == ROOT:
                for i, c in enumerate(json.load(open(p, encoding="utf-8"))["cells"]):
                    if c["cell_type"] == "code":
                        src = "\n".join(ln for ln in "".join(c["source"]).splitlines() if not ln.lstrip().startswith(("%", "!")))
                        try:
                            yield f"{rel}#cell{i}", ast.parse(src)
                        except SyntaxError:
                            pass


class WorkflowConfigRelationTests(unittest.TestCase):
    """`workflow.CONFIG` (lazy `import workflow`) is a STANDALONE copy; main.ipynb never uses it: it calls
    `workflow.load_into(globals(), ...)`, whose modules read the pipeline CONFIG of the namespace they are loaded into."""

    def test_load_into_shares_the_namespace_config(self):
        workflow = _nbload.workflow_package()
        ns = _nbload.load_pipeline()
        cfg = ns["CONFIG"]
        workflow.load_into(ns, only=("retarget", "splits"))
        self.assertIs(ns["CONFIG"], cfg)                                          # workflow modules did not replace it
        ns["update_config"]({"stride": 9})
        self.assertEqual(ns["CONFIG"]["stride"], 9)                               # one dict for pipeline AND workflow code
        self.assertIn("retarget_splits", ns)

    def test_lazy_package_config_is_standalone_and_equal_to_the_pipeline_defaults(self):
        import workflow
        pipeline = _nbload.load_pipeline()
        pipeline["reset_config"]()
        self.assertEqual(workflow.CONFIG, pipeline["CONFIG"])
        self.assertIsNot(workflow.CONFIG, pipeline["CONFIG"])
        before = deepcopy(workflow.CONFIG)
        pipeline["update_config"]({"stride": 13})
        self.assertEqual(workflow.CONFIG, before)                                 # documented: not shared with another namespace
        try:
            workflow.update_config({"stride": 3})                                 # its own pipeline functions act on its own CONFIG
            self.assertEqual(workflow.CONFIG["stride"], 3)
        finally:
            workflow.reset_config()
        self.assertEqual(workflow.CONFIG, before)                                 # and reset_config resets to the pipeline defaults


if __name__ == "__main__":
    unittest.main()

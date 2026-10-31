"""core/registry.py: the generic Registry primitive (tier 0). Duplicate names raise, unknown names raise with the known
names and a did-you-mean hint, the contract (version, required metadata, validator) is checked AT REGISTRATION, and the
module imports nothing from the repo.
    python -m pytest tests/test_core_registry.py -q
"""
import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.registry import (Contract, ContractError, DuplicateNameError, Registry, RegistryError,  # noqa: E402
                           UnknownNameError)


def _loss(x):
    return x


def _check_stat_keys(name, obj, meta):
    return [] if callable(obj) and meta["stat_keys"] else ["stat_keys is empty or obj is not callable"]


def _task_registry():
    return Registry("task", Contract("task", required_meta=("stat_keys",), optional_meta=("supports_sample_weight",),
                                     validator=_check_stat_keys))


class RegisterTests(unittest.TestCase):
    def test_register_get_names_in_order(self):
        r = Registry("head")
        r.register("b", 1, version=1)
        r.register("a", 2, version="2026-10")
        self.assertEqual(r.names(), ("b", "a"))
        self.assertEqual(r.get("a"), 2)
        self.assertEqual(r.entry("a").version, "2026-10")
        self.assertIn("b", r)
        self.assertEqual(len(r), 2)
        self.assertEqual(list(r), ["b", "a"])

    def test_register_returns_object_and_decorator_returns_function(self):
        r = Registry("head")
        self.assertIs(r.register("f", _loss, version=1), _loss)
        deco = r.decorator("g", version=1)
        self.assertIs(deco(_loss), _loss)
        self.assertIs(r.get("g"), _loss)

    def test_duplicate_name_raises_and_keeps_the_first(self):
        r = Registry("head")
        r.register("x", 1, version=1)
        with self.assertRaises(DuplicateNameError) as cm:
            r.register("x", 2, version=2)
        self.assertIn("already registered", str(cm.exception))
        self.assertIsInstance(cm.exception, ValueError)
        self.assertEqual(r.get("x"), 1)
        self.assertEqual(r.entry("x").version, 1)

    def test_unknown_name_raises_with_known_names_and_hint(self):
        r = Registry("head")
        r.register("nig_regression", 1, version=1)
        r.register("binary_classification", 2, version=1)
        with self.assertRaises(UnknownNameError) as cm:
            r.get("nig_regresion")
        msg = str(cm.exception)
        self.assertIn("'nig_regression'", msg)
        self.assertIn("binary_classification", msg)
        self.assertIn("Did you mean 'nig_regression'", msg)
        self.assertFalse(msg.startswith('"'), "str() must be the plain message, not the KeyError repr")
        # both legacy exception types keep working for a later migration
        self.assertIsInstance(cm.exception, KeyError)
        self.assertIsInstance(cm.exception, ValueError)
        self.assertIsInstance(cm.exception, RegistryError)

    def test_unknown_name_in_empty_registry(self):
        with self.assertRaises(UnknownNameError) as cm:
            Registry("head").get("a")
        self.assertIn("[]", str(cm.exception))
        with self.assertRaises(UnknownNameError):
            Registry("head").unregister("a")

    def test_unregister_allows_reuse(self):
        r = Registry("head")
        r.register("x", 1, version=1)
        self.assertEqual(r.unregister("x").obj, 1)
        r.register("x", 3, version=1)
        self.assertEqual(r.get("x"), 3)


class ContractTests(unittest.TestCase):
    def test_version_is_required_and_validated(self):
        r = Registry("head")
        with self.assertRaises(TypeError):
            r.register("x", 1)  # the version keyword is mandatory
        for bad in (0, -1, True, 1.5, "", "  ", None):
            with self.assertRaises(ContractError, msg=repr(bad)):
                r.register("x", 1, version=bad)
        self.assertEqual(len(r), 0)

    def test_bad_names_rejected(self):
        r = Registry("head")
        for bad in ("", " ", " a", "a ", None, 3):
            with self.assertRaises(ContractError, msg=repr(bad)):
                r.register(bad, 1, version=1)

    def test_required_metadata_missing_lists_every_problem(self):
        r = Registry("loss", Contract("loss", required_meta=("stat_keys", "roles")))
        with self.assertRaises(ContractError) as cm:
            r.register("x", 1, version=0, roles=())
        msg = str(cm.exception)
        self.assertIn("version", msg)
        self.assertIn("missing required metadata ['stat_keys']", msg)
        self.assertEqual(len(r), 0)

    def test_unknown_metadata_rejected_only_when_optional_meta_declared(self):
        strict = Registry("t", Contract("t", required_meta=("a",), optional_meta=("b",)))
        strict.register("ok", 1, version=1, a=1, b=2)
        with self.assertRaises(ContractError) as cm:
            strict.register("bad", 1, version=1, a=1, c=3)
        self.assertIn("['c']", str(cm.exception))
        loose = Registry("t", Contract("t", required_meta=("a",)))
        loose.register("fine", 1, version=1, a=1, anything=3)
        self.assertEqual(loose.entry("fine").meta, {"a": 1, "anything": 3})

    def test_validator_runs_at_registration_and_blocks_storage(self):
        r = _task_registry()
        r.register("good", _loss, version=1, stat_keys=("mae",), supports_sample_weight=True)
        with self.assertRaises(ContractError) as cm:
            r.register("bad", _loss, version=1, stat_keys=())
        self.assertIn("stat_keys is empty", str(cm.exception))
        self.assertNotIn("bad", r)

    def test_validator_exception_becomes_contract_error_with_cause(self):
        def boom(name, obj, meta):
            raise RuntimeError("probe failed")
        r = Registry("loss", Contract("loss", validator=boom))
        with self.assertRaises(ContractError) as cm:
            r.register("x", _loss, version=1)
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        self.assertIn("probe failed", str(cm.exception))

    def test_validator_not_run_when_static_contract_already_fails(self):
        calls = []
        r = Registry("loss", Contract("loss", required_meta=("k",), validator=lambda n, o, m: calls.append(n)))
        with self.assertRaises(ContractError):
            r.register("x", _loss, version=1)
        self.assertEqual(calls, [])

    def test_check_all_and_describe(self):
        state = {"ok": True}
        r = Registry("t", Contract("t", validator=lambda n, o, m: [] if state["ok"] else ["now broken"]))
        r.register("a", 1, version=1, note="n")
        self.assertEqual(r.check_all(), [])
        self.assertEqual(r.describe(), [{"name": "a", "version": 1, "note": "n"}])
        state["ok"] = False
        self.assertEqual(len(r.check_all()), 1)

    def test_contract_declaration_errors(self):
        with self.assertRaises(ContractError):
            Contract("")
        with self.assertRaises(ContractError):
            Contract("t", required_meta=("name",))
        with self.assertRaises(ContractError):
            Registry("a", Contract("b"))


class TierZeroTests(unittest.TestCase):
    def test_registry_imports_only_stdlib(self):
        tree = ast.parse((ROOT / "core" / "registry.py").read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0)
                roots.add((node.module or "").split(".")[0])
        self.assertLessEqual(roots, set(sys.stdlib_module_names), roots)


if __name__ == "__main__":
    unittest.main()

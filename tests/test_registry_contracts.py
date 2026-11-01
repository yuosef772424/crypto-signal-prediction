"""PR 5 (audit C8): TASK_REGISTRY (trainer), HEAD_REGISTRY (model), NORMALIZERS and SAMPLE_FILTER_REGISTRY (data) are core.registry
Registry instances with declared contracts.

Proof of the migration: the registry CONTENT (names, per-name metadata, the function / class behind each name) fingerprinted by one
canonical function is identical to the content frozen from origin/master BEFORE the migration (tests/golden_registry_contents.json);
every entry has version 1; every entry still honours its contract (`check_all`). Duplicate registration raises, per registry; a
malformed registration fails AT REGISTRATION and leaves the registry unchanged; reads keep the old dict patterns.
    python -m pytest tests/test_registry_contracts.py -q
"""
import hashlib
import json
import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))

import _nbload  # noqa: E402
from core.registry import ContractError, DuplicateNameError, Registry, UnknownNameError  # noqa: E402

GOLDEN = os.path.join(ROOT, "tests", "golden_registry_contents.json")


def canon(model_ns, trainer_ns, data_ns):
    """Legacy-shaped content of the four registries; written against the dict read protocol, so it ran on the plain dicts of
    origin/master (frozen in GOLDEN) and runs unchanged on the Registry instances."""
    tr = trainer_ns["TASK_REGISTRY"]
    tasks = {n: {"stat_keys": list(tr[n]["stat_keys"]), "supports_sample_weight": tr[n]["supports_sample_weight"],
                 "config_keys": list(tr[n]["config_keys"]), "fn": tr[n]["fn"].__name__} for n in sorted(tr)}
    hr = model_ns["HEAD_REGISTRY"]
    heads = {n: {"fn": hr[n].__name__} for n in sorted(hr)}
    nr = data_ns["NORMALIZERS"]
    norms = {n: {"cls": type(nr[n]).__name__, "supports_cross_sectional": bool(nr[n].supports_cross_sectional)} for n in sorted(nr)}
    filters = sorted(data_ns["SAMPLE_FILTER_REGISTRY"])
    out = {"task": tasks, "head": heads, "normalizer": norms, "sample_filter": filters}
    out["fingerprint"] = {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()[:16] for k, v in list(out.items())}
    return out


def _task_fn(y_true, outputs, target_cfg, scheduled_vars, sample_weight=None):
    return 0.0, {"s": 0.0}


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _nbload.load_model()
        cls.trainer = _nbload.load_trainer()
        cls.data = _nbload.load_pipeline()


class ContentFingerprintTests(_Base):
    def test_content_identical_to_origin_master(self):
        with open(GOLDEN, encoding="utf-8") as f:
            golden = json.load(f)
        now = canon(self.model, self.trainer, self.data)
        for kind in ("task", "head", "normalizer", "sample_filter"):
            self.assertEqual(now[kind], golden[kind], kind)
            self.assertEqual(now["fingerprint"][kind], golden["fingerprint"][kind], kind)
        self.assertEqual(now["fingerprint"], golden["fingerprint"])

    def test_every_builtin_is_a_registry_with_version_one_and_honours_its_contract(self):
        for label, reg in (("TASK_REGISTRY", self.trainer["TASK_REGISTRY"]), ("HEAD_REGISTRY", self.model["HEAD_REGISTRY"]),
                           ("NORMALIZERS", self.data["NORMALIZERS"]), ("SAMPLE_FILTER_REGISTRY", self.data["SAMPLE_FILTER_REGISTRY"])):
            self.assertIsInstance(reg, Registry, label)
            self.assertEqual(reg.check_all(), [], label)
            self.assertEqual({e["version"] for e in reg.describe()}, {1} if len(reg) else set(), label)

    def test_dict_read_patterns_still_work(self):
        tr = self.trainer["TASK_REGISTRY"]
        self.assertIn("evidential", tr)
        self.assertEqual(tr["regression"]["stat_keys"], ["mae"])
        self.assertIsNone(tr.get("nope", None))
        self.assertEqual({n for n, _ in tr.items()}, set(tr.keys()))
        self.assertTrue(all("config_keys" in e for e in tr.values()))
        self.assertEqual(sorted(self.data["NORMALIZERS"]), ["logret_sigma", "logret_std", "window_scale"])


class DuplicateRegistrationTests(_Base):
    def test_task_duplicate_raises_and_keeps_the_original(self):
        reg, register = self.trainer["TASK_REGISTRY"], self.trainer["register_task_type"]
        before = reg["regression"]
        with self.assertRaises(DuplicateNameError):
            register("regression", _task_fn, ["s"])
        self.assertIs(reg["regression"], before)
        register("zz_dup", _task_fn, ["s"])
        try:
            with self.assertRaises(DuplicateNameError):
                register("zz_dup", _task_fn, ["s"])
        finally:
            reg.unregister("zz_dup")

    def test_head_duplicate_raises_and_keeps_the_original(self):
        reg, register = self.model["HEAD_REGISTRY"], self.model["register_head_type"]
        before = reg["binary_classification"]
        with self.assertRaises(DuplicateNameError):
            register("binary_classification")(lambda h, target, cfg: {})
        self.assertIs(reg["binary_classification"], before)
        register("zz_dup")(lambda h, target, cfg: {})
        try:
            with self.assertRaises(DuplicateNameError):
                register("zz_dup")(lambda h, target, cfg: {})
        finally:
            reg.unregister("zz_dup")

    def test_normalizer_duplicate_raises(self):
        d = self.data
        with self.assertRaises(DuplicateNameError):
            d["register_normalizer"](d["WindowScale"]())
        self.assertEqual(len(d["NORMALIZERS"]), 3)

    def test_sample_filter_duplicate_raises(self):
        d = self.data
        reg = d["SAMPLE_FILTER_REGISTRY"]
        d["register_sample_filter"]("zz_dup", lambda df: df.index == df.index)
        try:
            with self.assertRaises(DuplicateNameError):
                d["register_sample_filter"]("zz_dup", lambda df: df.index == df.index)
        finally:
            reg.unregister("zz_dup")
        self.assertNotIn("zz_dup", reg)


class ContractAtRegistrationTests(_Base):
    def test_task_contract_violations_fail_at_registration(self):
        reg, register = self.trainer["TASK_REGISTRY"], self.trainer["register_task_type"]
        n = len(reg)
        no_sw = lambda y, o, t, s: (0.0, {"s": 0.0})                                    # noqa: E731
        bad = [dict(fn=no_sw, stat_keys=["s"], supports_sample_weight=True),            # declares weights, cannot take them
               dict(fn=lambda a: 0, stat_keys=["s"]),                                   # wrong arity
               dict(fn="not callable", stat_keys=["s"]),
               dict(fn=_task_fn, stat_keys=[]),
               dict(fn=_task_fn, stat_keys=["s", "s"]),
               dict(fn=_task_fn, stat_keys=["s"], config_keys=("loss_weight",)),        # a common target key
               dict(fn=_task_fn, stat_keys=["s"], supports_sample_weight=1)]            # not a bool
        for i, kw in enumerate(bad):
            with self.assertRaises(ContractError, msg=str(kw)):
                register(f"zz_bad{i}", **kw)
        for label in ("stat_keys", "config_keys"):                                      # list("mae") would become ['m','a','e']
            with self.assertRaises(ContractError):
                register("zz_str", _task_fn, **{"stat_keys": ["s"], label: "mae"})
        self.assertEqual(len(reg), n)

    def test_head_contract_violation_fails_at_registration(self):
        reg, register = self.model["HEAD_REGISTRY"], self.model["register_head_type"]
        n = len(reg)
        with self.assertRaises(ContractError):
            register("zz_bad")(lambda h: {})                                            # wrong arity
        with self.assertRaises(ContractError):
            register("zz_bad")("not callable")
        with self.assertRaises(ContractError):
            register("zz_bad", version=0)(lambda h, t, c: {})
        self.assertEqual(len(reg), n)

    def test_normalizer_contract_violation_fails_at_registration(self):
        d = self.data
        n = len(d["NORMALIZERS"])
        with self.assertRaises(ContractError):
            d["NORMALIZERS"].register("window_scale_x", d["WindowScale"](), version=1)   # .name != registered name
        with self.assertRaises(ContractError):
            d["NORMALIZERS"].register("zz", object(), version=1)                         # not a Normalizer
        self.assertEqual(len(d["NORMALIZERS"]), n)

    def test_sample_filter_contract_violation_fails_at_registration(self):
        d = self.data
        with self.assertRaises(ContractError):
            d["register_sample_filter"]("zz_bad", lambda: True)                          # takes no frame
        with self.assertRaises(TypeError):
            d["register_sample_filter"]("zz_bad", "not callable")                        # the pre-migration error type is kept
        self.assertNotIn("zz_bad", d["SAMPLE_FILTER_REGISTRY"])

    def test_unknown_task_type_lookup_names_the_known_ones(self):
        with self.assertRaises(KeyError) as cm:
            self.trainer["TASK_REGISTRY"]["evidentail"]
        self.assertIsInstance(cm.exception, UnknownNameError)
        self.assertIn("Did you mean 'evidential'", str(cm.exception))

    def test_item_assignment_is_refused(self):
        for reg in (self.trainer["TASK_REGISTRY"], self.model["HEAD_REGISTRY"], self.data["NORMALIZERS"],
                    self.data["SAMPLE_FILTER_REGISTRY"]):
            with self.assertRaises(TypeError):
                reg["zz"] = object()


if __name__ == "__main__":
    unittest.main()

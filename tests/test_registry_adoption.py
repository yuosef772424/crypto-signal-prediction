"""S01 (hardening audit C8 / PR 5): TASK_REGISTRY, HEAD_REGISTRY, NORMALIZERS and SAMPLE_FILTER_REGISTRY are `core.registry.Registry`
instances with a contract each. The CONTENT hard-coded below (names in registration order + metadata) was computed BEFORE the
migration from the old plain dicts; the sha256 is that of the canonical dump (see `_fingerprint`). A duplicate name raises, an
unknown name raises with the known list, dict-style reads keep working, a contract violation raises AT REGISTRATION.
    python -m pytest tests/test_registry_adoption.py -q
"""
import hashlib
import json
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402
from core.registry import ContractError, DuplicateNameError, Registry, UnknownNameError  # noqa: E402

#: name -> (stat_keys, supports_sample_weight, config_keys, fn name), in registration order (pre-migration dict content)
TASKS = {
    "evidential": (["nig_base", "nig_pen", "mae", "aleatoric", "epistemic", "confidence", "calib_loss"], True,
                   ("use_calibration_loss", "lambda_reg_var", "lambda_calib_var", "normalize_reg", "normalize_reg_stop_grad",
                    "beta_nll", "huber_weight", "huber_delta", "alpha_floor_straight_through"), "evidential_task_loss"),
    "regression": (["mae"], True, ("loss_fn", "huber_delta"), "regression_task_loss"),
    "classification": (["accuracy"], True, ("binary", "label_smoothing"), "classification_task_loss"),
}
HEADS = {"binary_classification": "build_binary_classification_head",
         "multiclass_classification": "build_multiclass_classification_head"}
NORMS = {"window_scale": "WindowScale", "logret_std": "LogretStd", "logret_sigma": "LogretSigma"}
#: sha256 of the canonical dump of all four registries, computed on the pre-migration dicts
PRE_MIGRATION_SHA256 = "0af08a121152b307a4ee32a1ef29edbf0e4aa91c4cb604bc94e3faf7f77b0ae9"


def _norm(v):
    if callable(v) and not hasattr(v, "name"):
        return "callable:" + getattr(v, "__name__", "?")
    if hasattr(v, "name") and hasattr(v, "fit"):
        return "normalizer:" + v.name + ":" + type(v).__name__
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


def _fingerprint(t, m, d):
    out = {"TASK": {k: _norm(v) for k, v in sorted(t["TASK_REGISTRY"].items())},
           "HEAD": {k: _norm(v) for k, v in sorted(m["HEAD_REGISTRY"].items())},
           "NORM": {k: _norm(v) for k, v in sorted(d["NORMALIZERS"].items())},
           "FILT": {k: _norm(v) for k, v in sorted(d["SAMPLE_FILTER_REGISTRY"].items())}}
    return hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()


class RegistryAdoption(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t, cls.m, cls.d = _nbload.load_trainer(), _nbload.load_model(), _nbload.load_pipeline()
        cls.regs = {"task": cls.t["TASK_REGISTRY"], "head": cls.m["HEAD_REGISTRY"], "normalizer": cls.d["NORMALIZERS"],
                    "sample filter": cls.d["SAMPLE_FILTER_REGISTRY"]}

    def test_all_four_are_registry_instances_that_honour_their_contract(self):
        for kind, reg in self.regs.items():
            self.assertIsInstance(reg, Registry, kind)
            self.assertEqual(reg.kind, kind)
            self.assertEqual(reg.check_all(), [], kind)

    def test_content_equals_the_pre_migration_content(self):
        tasks = self.regs["task"]
        self.assertEqual(list(tasks), list(TASKS))
        for name, (stat_keys, sw, config_keys, fn_name) in TASKS.items():
            e = tasks[name]
            self.assertEqual(sorted(e), ["config_keys", "fn", "stat_keys", "supports_sample_weight"])
            self.assertEqual((e["stat_keys"], e["supports_sample_weight"], e["config_keys"], e["fn"].__name__),
                             (stat_keys, sw, config_keys, fn_name))
            self.assertIsInstance(e["stat_keys"], list)
            self.assertIsInstance(e["config_keys"], tuple)
        self.assertEqual({k: v.__name__ for k, v in self.regs["head"].items()}, HEADS)
        self.assertEqual(list(self.regs["head"]), list(HEADS))
        self.assertEqual({k: type(v).__name__ for k, v in self.regs["normalizer"].items()}, NORMS)
        self.assertEqual(list(self.regs["normalizer"]), list(NORMS))
        self.assertEqual(len(self.regs["sample filter"]), 0)

    def test_fingerprint_sha256_equals_the_pre_migration_one(self):
        self.assertEqual(_fingerprint(self.t, self.m, self.d), PRE_MIGRATION_SHA256)

    def test_dict_style_reads(self):
        for kind, reg in self.regs.items():
            names = list(reg)
            self.assertEqual(names, list(reg.keys()), kind)
            self.assertEqual(len(reg), len(names), kind)
            self.assertEqual([k for k, _ in reg.items()], names, kind)
            self.assertEqual(len(reg.values()), len(names), kind)
            self.assertNotIn("zz_nope", reg)
            self.assertIsNone(reg.get("zz_nope", None))
            for n in names:
                self.assertIn(n, reg)
                self.assertEqual(reg.get(n), reg[n])

    def test_unknown_name_raises_with_the_known_list(self):
        for kind, reg in self.regs.items():
            with self.assertRaises(UnknownNameError) as cm:
                reg["zz_nope"]
            self.assertIsInstance(cm.exception, KeyError)          # old `except KeyError` callers keep working
            self.assertIn(f"unknown {kind} 'zz_nope'", str(cm.exception))
            for n in reg:
                self.assertIn(repr(n), str(cm.exception))
            with self.assertRaises(UnknownNameError):
                reg.get("zz_nope")                                  # no default = raises (unlike dict.get)
        with self.assertRaises(ValueError) as cm:                   # the helpers keep their own ValueError messages
            self.d["get_normalizer"]("logret_zscore")
        self.assertIn("'window_scale'", str(cm.exception))

    def test_duplicate_registration_raises_for_every_registry(self):
        t, m, d = self.t, self.m, self.d
        with self.assertRaises(DuplicateNameError):
            t["register_task_type"]("regression", t["regression_task_loss"], ["mae"], supports_sample_weight=True,
                                    config_keys=("loss_fn", "huber_delta"))
        with self.assertRaises(DuplicateNameError):
            m["register_head_type"]("binary_classification")(lambda h, target, cfg: {})
        with self.assertRaises(DuplicateNameError):
            d["NORMALIZERS"].register("logret_std", d["LogretStd"](), version=1)
        d["register_sample_filter"]("zz_dup", lambda df: df.index == df.index)
        try:
            with self.assertRaises(DuplicateNameError):
                d["register_sample_filter"]("zz_dup", lambda df: df.index == df.index)
        finally:
            d["SAMPLE_FILTER_REGISTRY"].unregister("zz_dup")
        self.assertEqual(_fingerprint(t, m, d), PRE_MIGRATION_SHA256)       # nothing leaked into the registries

    def test_task_contract_is_checked_at_registration(self):
        reg, register = self.t["TASK_REGISTRY"], self.t["register_task_type"]
        ok = lambda y, o, t, s, sample_weight=None: (0.0, {})                # noqa: E731
        no_weight = lambda y, o, t, s: (0.0, {})                             # noqa: E731
        bad = [
            ("zz_a", dict(fn=no_weight, stat_keys=["x"], supports_sample_weight=True), "sample_weight"),
            ("zz_b", dict(fn=ok, stat_keys=["x", "x"]), "duplicates"),
            ("zz_c", dict(fn=ok, stat_keys=[1]), "stat_keys"),
            ("zz_d", dict(fn=ok, stat_keys=["x"], config_keys=("loss_weight",)), "common target keys"),
            ("zz_e", dict(fn=ok, stat_keys=["x"], supports_sample_weight=1), "bool"),
            ("zz_f", dict(fn="not callable", stat_keys=["x"]), "callable"),
        ]
        for name, kw, frag in bad:
            fn = kw.pop("fn")
            with self.assertRaises(ContractError, msg=name) as cm:
                register(name, fn, **kw)
            self.assertIn(frag, str(cm.exception))
            self.assertNotIn(name, reg)
        register("zz_ok", no_weight, ["x"])                                   # supports_sample_weight defaults to False
        try:
            self.assertEqual(reg["zz_ok"]["supports_sample_weight"], False)
        finally:
            reg.unregister("zz_ok")

    def test_head_contract_is_checked_at_registration(self):
        reg, register = self.m["HEAD_REGISTRY"], self.m["register_head_type"]
        with self.assertRaises(ContractError):
            register("zz_head")(lambda h: {})                                 # must accept (h, target, cfg)
        with self.assertRaises(ContractError):
            register("zz_head")("not callable")
        self.assertNotIn("zz_head", reg)

    def test_normalizer_contract_is_checked_at_registration(self):
        reg, cls = self.d["NORMALIZERS"], self.d["LogretStd"]
        with self.assertRaises(ContractError) as cm:
            reg.register("not_its_name", cls(), version=1)
        self.assertIn("its .name is 'logret_std'", str(cm.exception))
        with self.assertRaises(ContractError):
            reg.register("zz_plain", object(), version=1)
        self.assertEqual(list(reg), list(NORMS))

    def test_sample_filter_contract_is_checked_at_registration(self):
        reg = self.d["SAMPLE_FILTER_REGISTRY"]
        with self.assertRaises(TypeError):
            self.d["register_sample_filter"]("zz_s", "not callable")
        with self.assertRaises(ContractError):
            self.d["register_sample_filter"]("zz_s", lambda: True)             # takes no frame
        self.assertNotIn("zz_s", reg)

    def test_trainer_unknown_task_type_is_a_clear_error_not_a_bare_keyerror(self):
        with self.assertRaises(KeyError) as cm:
            self.t["TASK_REGISTRY"]["evidentail"]
        self.assertIn("Did you mean 'evidential'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

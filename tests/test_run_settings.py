"""RunSettings (workflow/settings.py): the one settings object of a main run.

What is pinned: the defaults are exactly the values main.ipynb's cells held before the object existed (tests/golden_run_settings.json,
read from the old cells' own text at commit 1b1b14f), unknown section/setting names raise everywhere, values are validated at
construction, the target-mode grammar agrees with retarget._parse_mode, the training folder name follows the settings like the old
inline expression, and a panel preset overrides exactly what the old PANEL_PRESET line did.
    python -m pytest tests/test_run_settings.py -q
"""
import dataclasses
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs", "research", "audit"))
import _nbload  # noqa: E402

GOLDEN = json.load(open(os.path.join(ROOT, "tests", "golden_run_settings.json"), encoding="utf-8"))
_NS = None


def _ns(only=("settings",)):
    global _NS
    if _NS is None:
        _NS = _nbload.workflow_package().load_into({"__name__": "t"}, only=only)
    return _NS


def _tup(v):
    """JSON lists -> tuples (settings normalise sequences to tuples)."""
    if isinstance(v, list):
        return tuple(_tup(x) for x in v)
    return v


class DefaultsAreTheOldNotebookValues(unittest.TestCase):
    def test_every_section_default_equals_the_recorded_notebook_literals(self):
        ns, d = _ns(), GOLDEN["defaults"]
        s = ns["RunSettings"]()
        for section, want in d.items():
            if section == "panel_presets":
                continue
            got = getattr(s, section).to_dict()
            want = dict(want)
            if section == "panel":
                want["baseline"] = "auto" if want["baseline"] == "model" else want["baseline"]   # "model" when RUN_MAIN_TRAINING (default)
            for k, v in want.items():
                self.assertEqual(_tup(got[k]) if isinstance(got[k], (list, tuple)) else got[k], _tup(v), f"{section}.{k}")
            # settings the old notebook did not have as cell globals are new fields; the old ones are all covered above
            extra = set(got) - set(want)
            self.assertLessEqual(extra, {"fit_verbose", "early_stopping_patience", "split_dates", "model_tfs", "group_freq",
                                         "mn_quantiles", "mn_universe"}, section)

    def test_panel_presets_are_the_recorded_ones(self):
        ns = _ns()
        old = GOLDEN["defaults"]["panel_presets"]["1h_s8"]
        new = ns["PANEL_PRESETS"]["1h_s8"]
        self.assertEqual({k[6:].lower() for k in old}, set(new))
        for k, v in old.items():
            self.assertEqual(_tup(new[k[6:].lower()]), _tup(v), k)

    def test_section_and_field_counts(self):
        ns = _ns()
        sections = ns["RunSettings"].SECTIONS
        self.assertEqual(list(sections), ["project", "data", "target", "model", "train", "evaluation", "panel"])
        self.assertEqual({n: len(c.field_names()) for n, c in sections.items()},
                         {"project": 1, "data": 7, "target": 3, "model": 2, "train": 14, "evaluation": 6, "panel": 20})


class Strictness(unittest.TestCase):
    def test_unknown_names_raise_everywhere(self):
        ns = _ns()
        RS = ns["RunSettings"]
        with self.assertRaises(TypeError):
            ns["TrainSettings"](epoch=3)                                    # constructor: Python's own error
        with self.assertRaises(ValueError) as cm:
            ns["TrainSettings"].from_dict({"epoch": 3})
        self.assertIn("epochs", str(cm.exception))                          # the message lists the known names
        with self.assertRaises(ValueError):
            RS.from_dict({"trainn": {}})
        with self.assertRaises(ValueError):
            RS.from_dict({"train": {"epoch": 3}})
        with self.assertRaises(ValueError):
            RS().updated({"train": {"epoch": 3}})
        with self.assertRaises(ValueError):
            RS().updated({"nope": {}})
        with self.assertRaises(ValueError):
            RS().train.updated(epoch=3)
        with self.assertRaises(TypeError):
            RS(train={"epochs": 3})                                          # a section must be a section object

    def test_frozen_and_updates_do_not_mutate(self):
        ns = _ns()
        s = ns["RunSettings"]()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.train.epochs = 1
        t = s.updated({"train": {"epochs": 5}, "target": {"target_mode": "relative"}})
        self.assertEqual((s.train.epochs, s.target.target_mode), (60, None))
        self.assertEqual((t.train.epochs, t.target.target_mode, t.train.batch_size), (5, "relative", 64))
        # dict-valued defaults are never shared between instances
        a, b = ns["TrainSettings"](), ns["TrainSettings"]()
        a.lambda_reg["end"] = 9.0
        self.assertEqual(b.lambda_reg["end"], 0.05)

    def test_roundtrip(self):
        ns = _ns()
        s = ns["RunSettings"]().updated({"data": {"model_tfs": ["1h", "4h"], "split_dates": {"train_end": "a", "val_end": "b"}},
                                         "panel": {"enabled": True, "k_eval": [5, None], "subset": {"last_days": 3, "coins": 2}}})
        self.assertEqual(ns["RunSettings"].from_dict(s.to_dict()), s)
        self.assertEqual(s.data.model_tfs, ("1h", "4h"))

    def test_values_are_validated(self):
        ns = _ns()
        bad = [("train", "epochs", 0), ("train", "epochs", True), ("train", "batch_size", -1), ("train", "run_dir", ""),
               ("train", "fit_verbose", 3), ("train", "early_stopping_mode", "avg"), ("data", "format", "csv"),
               ("data", "mmap", 1), ("data", "model_tfs", ()), ("data", "model_tfs", "1h"), ("data", "split_dates", {"train_end": "x"}),
               ("target", "target_mode", "nope"), ("target", "target_mode", "entry_range+relative"),
               ("target", "target_mode", "relative+relative"), ("target", "entry_close_reg", "pos"),
               ("model", "class_only", "yes"), ("evaluation", "calibration_method", "svm"),
               ("evaluation", "mn_quantiles", (0.9,)), ("panel", "preset", "9z"), ("panel", "variants", ()),
               ("panel", "overrides", {"trian": {}}), ("panel", "subset", {"days": 3}), ("panel", "k_draws", 0)]
        for section, name, value in bad:
            with self.assertRaises(ValueError, msg=f"{section}.{name}={value!r}"):
                ns["RunSettings"]().updated({section: {name: value}})

    def test_target_mode_grammar_agrees_with_retarget(self):
        ns = _ns()
        rt = _nbload.workflow_package().load_into({"__name__": "t"}, only=("retarget",))
        bases = list(rt["TARGET_MODES"])
        modes = bases + ["relative"] + [f"{b}+relative" for b in bases] + ["nope", "relative+relative", "", "return+", "+relative",
                                                                            "return+scaled"]
        for m in modes:
            try:
                rt["_parse_mode"](m)
                ok_old = True
            except ValueError:
                ok_old = False
            ok_new = ns["_target_mode_problem"](m) is None
            self.assertEqual(ok_new, ok_old, m)


class RunDir(unittest.TestCase):
    def test_folder_name_follows_the_settings(self):
        ns = _ns()
        S = ns["RunSettings"]
        base = "/content/drive/MyDrive/training_runs/crypto_model_v1"
        f = ns["run_dir_for"]
        self.assertEqual(f(S(), 1.0, ("1D",)), base + "_am")
        self.assertEqual(f(S(), 100.0, ("1h",)), base + "_s100_am")
        self.assertEqual(f(S().updated({"target": {"target_mode": "relative"}}), 100.0, ("1h",)), base + "_relative_s100_am")
        e = S().updated({"target": {"target_mode": "entry_range", "entry_close_reg": "range_pos"}})
        self.assertEqual(f(e, 1.0, ("1h",)), base + "_entry_range_range_pos_am")
        self.assertEqual(f(S().updated({"model": {"anti_memorization": False, "class_only": True}}), 1.0, ("1h", "4h")),
                         base + "_1h_4h_classonly")
        self.assertEqual(f(S().updated({"train": {"run_dir": "/x/run"}}), 2.5, ["1h", "4h", "1D"]), "/x/run_s2.5_am_1h_4h_1D")

    def test_equals_the_old_inline_expression_on_the_recorded_scenarios(self):
        ns = _ns()
        for name, sc in GOLDEN["scenarios"].items():
            a = sc["scenario"]
            s = ns["RunSettings"]().updated({
                "target": {"target_mode": a.get("target_mode"), "entry_close_reg": a.get("entry_close_reg", "abs_return")},
                "model": {"anti_memorization": a.get("anti_memorization", True), "class_only": a.get("class_only", False)},
                "train": {"run_dir": "<WORK>/run" if a.get("train") else GOLDEN["defaults"]["train"]["run_dir"]}})
            got = ns["run_dir_for"](s, sc["reg_target_scale"], sc["model_tfs"])      # the recorder trained in <work>/run
            self.assertEqual(got, sc["main_config"]["run"]["run_dir"], name)


class PanelPreset(unittest.TestCase):
    def test_applied_overrides_exactly_the_preset_fields(self):
        ns = _ns()
        P = ns["PanelSettings"]
        plain = P(enabled=True, variants=("A",), seeds=(7,), epochs=3, patience=2, group=None, k_eval=(5,), k_draws=9, baseline="model")
        self.assertIs(plain.applied(), plain)                                  # no preset: untouched
        p = plain.updated(preset="1h_s8").applied()
        self.assertEqual((p.variants, p.seeds, p.epochs, p.patience, p.group, p.k_eval, p.k_draws, p.baseline),
                         (("A_ic", "B_ic", "A_ic_k"), (0, 1), 50, 25, "auto", (5, 10, 20, None), 3, None))
        self.assertEqual((p.enabled, p.preset, p.batch_samples), (True, "1h_s8", 1024))   # what the preset does not name is kept
        self.assertEqual(p.applied(), p)                                       # idempotent
        self.assertEqual(p.updated(seeds=(3,)).applied().seeds, (0, 1))        # the preset wins when applied again (old order)


if __name__ == "__main__":
    unittest.main()

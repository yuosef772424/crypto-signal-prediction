#!/usr/bin/env python3
"""
PURPOSE:  The catalog of every setting a main run can take: each RunSettings field (all sections), each data-pipeline CONFIG key that passes through project.config_overrides, each MODEL_CONFIG key that passes through model.config_overrides, and the registered names (tasks, heads, normalizers, sample filters, price_norm_mode, target modes). Per key: path, type, accepted values (from the field's own validation or the registry), default, recommended value (workflow/presets.py RECOMMENDED_CHOICES, preset recommended_2026_11) and the source of that recommendation. Writes docs/research/settings_catalog.md (generated, never edited by hand) and docs/research/settings_catalog.json.
TAGS:     settings catalog, RunSettings, config_overrides, MODEL_CONFIG, registries, accepted values, recommended preset, recommended_2026_11, generated docs, unchecked setting, where do I set X
PITFALLS: Everything is read from the live code (dataclass fields, defaults, __post_init__ source, registries, CONFIG, MODEL_CONFIG), so the catalog cannot list a setting that does not exist; the ACCEPTED_NOTES table below is the only hand-written part (one line per RunSettings field, a missing line fails the test), and `validated` is derived from whether __post_init__ mentions the field at all (a field that never appears there accepts ANY value: such rows are listed under "Settings without a value check"). Importing data/model/trainer loads TensorFlow (~10 s). The md/json are compared by tests/test_settings_catalog.py: after adding a field run `python tools/settings_catalog.py`. dict-valued fields other than the two config_overrides are one row (their inner keys are validated by the trainer / panel code, not here).

Usage:  python tools/settings_catalog.py           (write docs/research/settings_catalog.md and .json)
        python tools/settings_catalog.py --check   (exit 1 if either file is stale)
"""
import argparse
import dataclasses
import inspect
import json
import os
import re
import sys
import typing

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from core.schema import ENTRY_CLOSE_REGS, TARGET_MODES  # noqa: E402

MD_PATH = os.path.join(ROOT, "docs", "research", "settings_catalog.md")
JSON_PATH = os.path.join(ROOT, "docs", "research", "settings_catalog.json")
PRESET_NAME = "recommended_2026_11"

#: One line per RunSettings field: what its validation accepts. "{...}" names are filled from the live constants (see _fill).
ACCEPTED_NOTES = {
    "project.config_overrides": "dict of data-pipeline CONFIG keys (table 'project.config_overrides'); an unknown key raises in update_config, values are not checked there",
    "data.filename_base": "non-empty str",
    "data.format": "one of {DATA_FORMATS}",
    "data.mmap": "bool",
    "data.local_dir": "None | False | path str",
    "data.path": "None | str",
    "data.model_tfs": "None | non-empty tuple of timeframe names (all must exist in the dataset; the first is the base timeframe)",
    "data.split_dates": "None | {'train_end': 'YYYY-MM-DD', 'val_end': 'YYYY-MM-DD'} (exactly these two keys)",
    "target.target_mode": "None | '<base>' | '<base>+relative' | 'relative' with base in {TARGET_MODES} ('+relative' refused on the NO_RELATIVE bases)",
    "target.entry_close_reg": "one of {ENTRY_CLOSE_REGS}",
    "target.group_freq": "None | duration str (e.g. '1D', '4h', '32h')",
    "target.suspended_targets": "None (old rule: close suspended unless entry_range) | tuple of CONFIG['targets'] names ('()' = train all; suspending every target raises in plan_model)",
    "model.anti_memorization": "bool",
    "model.class_only": "bool",
    "model.config_overrides": "dict of MODEL_CONFIG / anti-memorization keys (table 'model.config_overrides'); unknown key raises in plan_model; price_targets, head_types, enforce_order refused",
    "train.run_main_training": "bool",
    "train.run_dir": "non-empty str",
    "train.epochs": "positive int",
    "train.batch_size": "positive int",
    "train.train_mode": "str (not value-checked here; the trainer interprets it)",
    "train.seed": "None (trainer default 42) | int in [0, 2**32)",
    "train.early_stopping_patience": "None (trainer default) | positive int",
    "train.fit_verbose": "0 | 1 | 2",
    "train.use_uncertainty_weighting": "bool",
    "train.lambda_reg": "dict {'start','end','warmup_epochs','schedule'} (inner keys validated by the trainer config)",
    "train.lambda_calib": "dict {'start','end','warmup_epochs','schedule'} (inner keys validated by the trainer config)",
    "train.early_stopping_monitor": "str (a metric name of the trainer's history; not value-checked here)",
    "train.early_stopping_mode": "'min' | 'max'",
    "train.anti_memorization_trainer": "dict merged over the trainer settings (inner keys validated by build_config)",
    "train.anti_memorization_label_smoothing": "number in [0, 1)",
    "evaluation.calibrate_confidence": "bool",
    "evaluation.calibration_method": "one of {CALIBRATION_METHODS}",
    "evaluation.eval_all_rows": "bool",
    "evaluation.out_dir": "non-empty str",
    "evaluation.mn_quantiles": "non-empty sequence of numbers in (0, 0.5]",
    "evaluation.mn_universe": "None (all coins) | 'categories' | sequence of symbols",
    "evaluation.trivial_baseline": "bool",
    "panel.enabled": "bool",
    "panel.variants": "non-empty sequence of variant names ('A','A_ic','B','B_ic','A_ic_k'; names are checked by the panel code)",
    "panel.seeds": "non-empty sequence of ints",
    "panel.epochs": "None | positive int",
    "panel.batch_samples": "positive int",
    "panel.max_days": "positive int",
    "panel.baseline": "None | 'auto' | 'model' | path str",
    "panel.first_touch": "None | str (path of bracket_first_5.pkl)",
    "panel.subset": "None | {'last_days': int, 'coins': int}",
    "panel.run_root": "None | str",
    "panel.overrides": "dict with only the sections 'train' / 'model'",
    "panel.patience": "None | positive int",
    "panel.monitor": "non-empty str (val_loss | val_ic_asym | val_auc_low ...)",
    "panel.group": "None | 'auto' | duration str",
    "panel.allow_mixed_timestamps": "bool",
    "panel.min_coins": "None | positive int",
    "panel.max_coins": "None | positive int",
    "panel.k_eval": "None | sequence of int / None",
    "panel.k_draws": "positive int",
    "panel.preset": "None | one of {PANEL_PRESETS}",
}

#: Hand-written accepted values for the few data-pipeline CONFIG keys with a real check or a coupling worth stating.
CONFIG_NOTES = {
    "price_norm_mode": "one of {PRICE_NORM_MODES}; 'logret_*' refuse: reg_target_scale != 1, x_storage_dtype, cross_sectional_normalize, target modes other than the default, several timeframes (logret_sigma), regression heads (logret_sigma). Acts when the dataset is BUILT: apply_dataset_config overwrites it with the loaded file's own value",
    "stale_bars": "None | {'min_run': N >= 2} | {'min_run_hours': H > 0} (exactly one key). Acts when the dataset is built (stamped in the file)",
    "reg_target_mode": "'return' | 'window_scale' (see data/defaults.py comment)",
    "reg_target_scale": "number; the dataset file's own value wins at run time (apply_dataset_config)",
    "x_storage_dtype": "None | dtype name; refused by price_norm_mode='logret_std'",
    "sample_filters": "list of {'feature','op','value'} | {'expr'} | {'fn': <registered sample filter name>}; unregistered name raises",
}

_NO_CHECK = "not validated (any value is passed on as is)"


def _ns():
    """The workflow namespace with the settings + presets modules, and the data / model / trainer packages with their registries."""
    import data
    import model
    import trainer
    import workflow
    ns = {}
    workflow.load_into(ns, only=("settings", "presets"))
    return ns, data, model, trainer


def _fill(text, consts):
    return re.sub(r"\{(\w+)\}", lambda m: str(consts[m.group(1)]), text)


def _type_str(tp):
    return str(tp).replace("typing.", "").replace("<class '", "").replace("'>", "")


def _is_validated(cls, name):
    """True when the section's __post_init__ mentions the field (a field never mentioned accepts any value)."""
    src = inspect.getsource(cls.__post_init__)
    return bool(re.search(rf"\bself\.{re.escape(name)}\b|\"{re.escape(name)}\"|'{re.escape(name)}'", src))


def _default(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    return f.default_factory()


def _recommendation(choices):
    return {row["path"]: row for row in choices}


def _rec_columns(path, rec):
    row = rec.get(path)
    if path == "train.run_dir":                                    # derived by workflow.presets.preset_run_dir, not a fixed value
        return {"recommended": f"<run_dir>_{PRESET_NAME}", "evidence": "derived",
                "source": "workflow/presets.py preset_run_dir: the preset name is appended so a model of another setup is never resumed"}
    if row is None:
        subs = sorted(k[len(path) + 1:] for k in rec if k.startswith(path + "."))
        return {"recommended": f"= default, except keys {subs}" if subs else "= default", "evidence": "", "source": ""}
    return {"recommended": repr(row["value"]), "evidence": row["evidence"], "source": row["source"]}


def build():
    """The catalog as a plain dict (the JSON image)."""
    ns, data, model, trainer = _ns()
    rec = _recommendation(ns["RECOMMENDED_CHOICES"])
    consts = {"DATA_FORMATS": ns["DATA_FORMATS"], "CALIBRATION_METHODS": ns["CALIBRATION_METHODS"],
              "PANEL_PRESETS": sorted(ns["PANEL_PRESETS"]), "ENTRY_CLOSE_REGS": list(ENTRY_CLOSE_REGS),
              "TARGET_MODES": list(TARGET_MODES), "PRICE_NORM_MODES": list(data.PRICE_NORM_MODES)}
    settings_rows = []
    for section, cls in ns["RunSettings"].SECTIONS.items():
        for f in dataclasses.fields(cls):
            path = f"{section}.{f.name}"
            note = ACCEPTED_NOTES.get(path)
            validated = _is_validated(cls, f.name)
            settings_rows.append({
                "path": path, "section": section, "type": _type_str(f.type), "default": repr(_default(f)),
                "accepted": _fill(note, consts) if note else "MISSING NOTE: add the field to ACCEPTED_NOTES",
                "validated": validated, **_rec_columns(path, rec)})
    config_rows = []
    for key, value in data.CONFIG.items():
        path = f"project.config_overrides.{key}"
        note = CONFIG_NOTES.get(key)
        config_rows.append({"path": path, "type": type(value).__name__, "default": repr(value),
                            "accepted": _fill(note, consts) if note else _NO_CHECK, **_rec_columns(path, rec)})
    am = dict(model.ANTI_MEMORIZATION_CONFIG)
    model_rows = []
    for key in sorted(set(model.MODEL_CONFIG) | set(am)):
        path = f"model.config_overrides.{key}"
        base = model.MODEL_CONFIG.get(key, "(absent)")
        eff = am.get(key, base)
        owned = key in ("price_targets", "head_types", "enforce_order")
        model_rows.append({"path": path, "type": type(eff).__name__, "default": repr(base),
                           "default_with_anti_memorization": repr(eff),
                           "accepted": "refused: set by the run logic" if owned else "key must exist; the value is not checked here (the builder uses it as is)",
                           **_rec_columns(path, rec)})
    registries = {
        "trainer.TASK_REGISTRY": {"names": sorted(trainer.TASK_REGISTRY.names()), "used_by": "trainer config tasks (regression / evidential / classification)"},
        "model.HEAD_REGISTRY": {"names": sorted(model.HEAD_REGISTRY.names()), "used_by": "model.config_overrides.head_types entries registered here (nig_regression is built in)"},
        "data.NORMALIZERS": {"names": sorted(data.NORMALIZERS.names()), "used_by": "project.config_overrides.price_norm_mode (plus the built-in 'pct_change')"},
        "data.SAMPLE_FILTER_REGISTRY": {"names": sorted(data.SAMPLE_FILTER_REGISTRY.names()), "used_by": "project.config_overrides.sample_filters {'fn': name} (none registered by default)"},
        "data.PRICE_NORM_MODES": {"names": list(data.PRICE_NORM_MODES), "used_by": "project.config_overrides.price_norm_mode"},
        "core.schema.TARGET_MODES": {"names": list(TARGET_MODES), "used_by": "target.target_mode (optionally '+relative')"},
        "core.schema.ENTRY_CLOSE_REGS": {"names": list(ENTRY_CLOSE_REGS), "used_by": "target.entry_close_reg"},
    }
    return {
        "preset": PRESET_NAME,
        "settings": settings_rows,
        "config_keys": config_rows,
        "model_keys": model_rows,
        "registries": registries,
        "not_enabled": [dict(r) for r in ns["NOT_ENABLED"]],
        "conflicts": [{"between": list(c["between"]), "resolution": c["resolution"]} for c in ns["PRESET_CONFLICTS"]],
        "recommended_paths": sorted(rec),
    }


def _cell(text):
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(cat):
    L = ["<!-- GENERATED by tools/settings_catalog.py - do not edit by hand; regenerate with `python tools/settings_catalog.py` -->",
         "# Settings catalog", "",
         f"Every setting a main run takes, read from the live code. Preset **{cat['preset']}** (`workflow/presets.py`, `recommended_settings()`) "
         "is the recommended setup; `RunSettings()` defaults are unchanged. `evidence`: **proven** = a recorded result holds it, "
         "**new** = switched on by the owner but not measured yet (a run with several `new` choices measures the bundle; ablate afterwards).",
         "", "## RunSettings fields", ""]
    for section in dict.fromkeys(r["section"] for r in cat["settings"]):
        L += [f"### {section}", "", "| path | type | accepted values | default | recommended | evidence | source of the recommendation | value check |",
              "|---|---|---|---|---|---|---|---|"]
        for r in (x for x in cat["settings"] if x["section"] == section):
            L.append("| " + " | ".join(_cell(v) for v in (f"`{r['path']}`", r["type"], r["accepted"], f"`{r['default']}`",
                                                          r["recommended"] if r["recommended"].startswith("= default") or r["recommended"].startswith("<") else f"`{r['recommended']}`",
                                                          r["evidence"], r["source"], "yes" if r["validated"] else "**none**")) + " |")
        L.append("")
    unchecked = [r["path"] for r in cat["settings"] if not r["validated"]]
    L += ["## Settings without a value check", "",
          "Fields whose `__post_init__` never mentions them: any value is accepted at construction." if unchecked else "None: every field is mentioned by its section's `__post_init__`.", ""]
    L += [f"- `{p}`" for p in unchecked] + [""]
    rec_cfg = [r for r in cat["config_keys"] if not r["recommended"].startswith("= default")]
    L += ["## project.config_overrides (data-pipeline CONFIG keys)", "",
          f"{len(cat['config_keys'])} keys. An unknown key raises; values are mostly not checked. Keys with a recommendation first, the rest below with their defaults.", "",
          "| path | type | accepted values | default | recommended | evidence | source |", "|---|---|---|---|---|---|---|"]
    for r in rec_cfg + [x for x in cat["config_keys"] if x["recommended"].startswith("= default")]:
        L.append("| " + " | ".join(_cell(v) for v in (f"`{r['path']}`", r["type"], r["accepted"], f"`{r['default']}`",
                                                      r["recommended"] if r["recommended"].startswith("= default") or r["recommended"].startswith("<") else f"`{r['recommended']}`",
                                                      r["evidence"], r["source"])) + " |")
    L += ["", "## model.config_overrides (MODEL_CONFIG keys)", "",
          "`default` = MODEL_CONFIG; `with anti_memorization` = the value in force under the default `model.anti_memorization=True`. Overrides win over both.", "",
          "| path | type | accepted | default | with anti_memorization | recommended | evidence | source |", "|---|---|---|---|---|---|---|---|"]
    for r in cat["model_keys"]:
        L.append("| " + " | ".join(_cell(v) for v in (f"`{r['path']}`", r["type"], r["accepted"], f"`{r['default']}`",
                                                      f"`{r['default_with_anti_memorization']}`",
                                                      r["recommended"] if r["recommended"].startswith("= default") or r["recommended"].startswith("<") else f"`{r['recommended']}`",
                                                      r["evidence"], r["source"])) + " |")
    L += ["", "## Registered names", "", "| registry | names | used by |", "|---|---|---|"]
    for name, r in cat["registries"].items():
        L.append(f"| `{name}` | {_cell(', '.join(f'`{n}`' for n in r['names']) or '(none)')} | {_cell(r['used_by'])} |")
    L += ["", f"## Looked at and not enabled in {cat['preset']}", "", "| option | reason |", "|---|---|"]
    L += [f"| {_cell(r['path'])} | {_cell(r['reason'])} |" for r in cat["not_enabled"]]
    L += ["", "## Conflicts between choices", "", "| between | resolution |", "|---|---|"]
    L += [f"| {_cell(' vs '.join(c['between']))} | {_cell(c['resolution'])} |" for c in cat["conflicts"]]
    return "\n".join(L) + "\n"


def render_json(cat):
    return json.dumps(cat, indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/research/settings_catalog.md / .json are stale")
    a = ap.parse_args(argv)
    cat = build()
    outputs = {MD_PATH: render_markdown(cat), JSON_PATH: render_json(cat)}
    if a.check:
        stale = [os.path.relpath(p, ROOT) for p, text in outputs.items() if not os.path.exists(p) or open(p, encoding="utf-8").read() != text]
        if stale:
            print(f"settings_catalog --check: stale {stale}; run `python tools/settings_catalog.py`")
            return 1
        print(f"settings_catalog --check: OK ({len(cat['settings'])} settings, {len(cat['config_keys'])} config keys, {len(cat['model_keys'])} model keys)")
        return 0
    for p, text in outputs.items():
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("wrote", os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

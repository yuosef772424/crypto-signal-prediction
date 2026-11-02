"""
PURPOSE:  Named, ready-to-run RunSettings presets, separate from the RunSettings defaults (which keep the old behaviour): recommended_settings() = the "recommended_2026_11" setup, a table of choices where every row carries its evidence (a result that held, or a new option the owner switched on deliberately for this run) and its source; dataset_mismatches() tells which of the dataset-build options a preset names the loaded dataset was NOT built with.
TAGS:     preset, recommended_settings, recommended_2026_11, RECOMMENDED_CHOICES, NOT_ENABLED, PRESET_CONFLICTS, settings_with_preset, SETTINGS_PRESETS, preset_run_dir, dataset_mismatches, price_norm_mode, logret_std, stale_bars, use_instance_norm, suspended_targets, trivial_baseline, E-norm-001, E-norm-002, D-013, settings catalog
PITFALLS: RunSettings() defaults are NOT changed by anything here (a preset is chosen in one line: main.ipynb USE_RECOMMENDED, evaluate_trained_model --preset). evidence="proven" = a recorded result holds it; evidence="new" = not yet measured (cards pre-registered or no card), enabled by the owner on purpose, so the run is a measurement of the bundle and cannot attribute a gain to one option: ablate afterwards. project.config_overrides keys price_norm_mode / stale_bars act when the DATASET is BUILT (apply_dataset_config overwrites price_norm_mode with the loaded file's own): the preset only declares them, dataset_mismatches() shows if the loaded file differs. The training folder gets the preset name as a suffix (preset_run_dir) so a model of another setup is never resumed. The tools/settings_catalog.py test fails when a RunSettings field has no row here or in NOT_ENABLED. Executed into the shared workflow namespace by workflow/_loader.py (never imported on its own); uses only names of settings.py.
"""

RECOMMENDED_PRESET = "recommended_2026_11"

#: One row per choice: path = "<section>.<field>" or "<section>.<dict field>.<key>" (dict fields: project.config_overrides -> the data
#: pipeline CONFIG, model.config_overrides -> MODEL_CONFIG); evidence = "proven" | "new"; source = where the evidence / decision is.
RECOMMENDED_CHOICES = (
    # ── a) what held up before ─────────────────────────────────────────────────────────────────────────────────────────────────
    {"path": "model.anti_memorization", "value": True, "evidence": "proven",
     "source": "docs/research/anti_memorization_pr7.md section 8 (RL_gru measured); failure_registry F-0029, F-0030 (the old default encoder memorises). Same as the default: stated so the preset is complete"},
    {"path": "evaluation.calibrate_confidence", "value": True, "evidence": "proven",
     "source": "evaluation/trust_calibration.py (calibrators fitted on val only, never test); the report prints ECE before/after. Calibration fixes bias, not discrimination (the report says so). Same as the default"},
    {"path": "evaluation.calibration_method", "value": "isotonic", "evidence": "proven",
     "source": "the method main.ipynb has always used (no recorded comparison with platt: kept, not claimed better). Same as the default"},
    {"path": "train.early_stopping_monitor", "value": "val_raw_loss", "evidence": "proven",
     "source": "docs/research/anti_memorization_pr7.md 10-c (monitor the raw loss, not the Kendall-weighted one). hardening/bottlenecks.csv B-003 says the choice of the best epoch still needs a fixed quality metric (S06): not changed here. Same as the default"},
    {"path": "project.config_overrides.stale_bars", "value": {"min_run_hours": 168}, "evidence": "proven",
     "source": "docs/research/plan/STATUS.md Known issues: stale-bars-002 adopted trimmed data + stale_bars={'min_run_hours': 168} (1h train zero-target share 2.48% -> 1.87%); never drop y_close_reg == 0 rows; research/stale_bars/README.md. Acts when the dataset is built (dataset_mismatches checks the loaded file)"},
    # ── b) new options, not measured yet, switched on deliberately by the owner for this run ─────────────────────────────────────
    {"path": "project.config_overrides.price_norm_mode", "value": "logret_std", "evidence": "new",
     "source": "card docs/research/cards/E-norm-001.md (pre-registered; one valid seed per arm so far: model metric 0.649 vs 0.61-0.63, plan/STATUS.md Model line). Needs >= 3 seeds on one commit before it counts. Acts when the dataset is built. Not combinable with: target modes other than the default, reg_target_scale != 1, x_storage_dtype, cs_norm, logret_sigma heads"},
    {"path": "model.config_overrides.use_instance_norm", "value": False, "evidence": "new",
     "source": "card docs/research/cards/E-norm-002.md (pre-registered, NOT run; its pre-run gate passed: shape_and_level probe high 0.235 < 0.40). Only meaningful together with price_norm_mode='logret_std' (the data side already standardises); wins over anti_memorization's use_instance_norm=True because config_overrides are applied after it"},
    {"path": "target.suspended_targets", "value": (), "evidence": "new",
     "source": "docs/research/plan/decisions/D-013-default-changes-s03.md (owner): close is a trained target; step M3 compares high/low with and without it. () is also the RunSettings default since PR #62 (stated so the preset is complete). The training folder gets '_susp-none'. Tension with the closed finding that close DIRECTION carries no signal (F-0001, plan/STATUS.md T4): kept because market_neutral needs the close head and M3 measures its effect on high/low"},
    {"path": "evaluation.trivial_baseline", "value": True, "evidence": "new",
     "source": "PR #48 (opt-in column of the generalization report: AUC of the trivial rule 'close position in the last bar's range' and the model's uplift); F-0085 (high/low class AUC 0.74-0.78 is range mechanics, trivial rule 0.76-0.83). No result recorded yet"},
)

#: Candidate options looked at and NOT switched on, with the reason (so nobody re-derives it). path may name a field or a study.
NOT_ENABLED = (
    {"path": "target.target_mode", "reason": "relative / cross-sectional targets closed at economics: F-0031, F-0033; E-rel-001 not run and waits for the owner's decision on absolute vs relative; logret_std refuses non-default target modes"},
    {"path": "model.class_only", "reason": "classification-only heads: the direction/classification heads failed (F-0001, F-0085); the confidence and NIG heads are what the recommended run measures"},
    {"path": "project.config_overrides.sample_filters / features (rel_volume, zero_vol_share_w, min_liquidity_30d)", "reason": "E-liq-001 is pre-registered but its components are not built yet; gate-style features closed (F-0080, F-0081)"},
    {"path": "project.config_overrides.price_norm_mode='logret_sigma'", "reason": "refuses regression heads (raises by design, E-norm-001)"},
    {"path": "evaluation.mn_quantiles / mn_universe", "reason": "no result favours other values; defaults kept"},
    {"path": "train.batch_size / epochs / early_stopping_patience / lambda_reg / lambda_calib", "reason": "changing step size or schedules is a training experiment that needs its own card (hardening/profile_2026-11.md item 4, MODEL_DEV_PROTOCOL); not part of this preset"},
    {"path": "train.seed", "reason": "a run parameter, not a recommendation: one run is one seed; the protocol needs >= 3 seeds per arm"},
    {"path": "panel.*", "reason": "the cross-asset panel model closed for 1d (F-0035); disabled by default and not part of this run"},
    {"path": "data.* (filename_base, model_tfs, split_dates, path, ...)", "reason": "which dataset to load is the owner's; the preset only checks the options the dataset must have been built with (dataset_mismatches)"},
    {"path": "PREDICT_BATCH_SIZE (core.schema) and rf_max_samples (evaluation.patterns)", "reason": "not RunSettings fields; D-013 (PR #62, merged) made 2048 and 50 000 the defaults, so nothing to set here"},
    {"path": "research/natr_range E-natr-004 (rolling recalibration of the NATR range band)", "reason": "a separate study with its own tool (tools/natr_range.py), not a RunSettings option; Stage B waits for forward data (not before 2027-01-01)"},
)

#: Pairs of choices that pull in different directions and how each was resolved (proven wins; a pair that is only a hazard is listed too).
PRESET_CONFLICTS = (
    {"between": ("model.config_overrides.use_instance_norm=False (E-norm-002, new)", "model.anti_memorization=True (proven; its config sets use_instance_norm=True)"),
     "resolution": "both kept: anti_memorization is a bundle (small model, noise, EMA, level_passthrough) whose evidence is about the bundle on window_scale inputs; E-norm-002 changes the one key and its card names AM + logret_std as arm A. config_overrides win over AM by design (workflow.run.plan_model)"},
    {"between": ("target.suspended_targets=() (D-013, new)", "closed finding: close direction has no signal (F-0001, T4)"),
     "resolution": "kept by owner decision D-013 (market_neutral needs p_up_close; M3 measures high/low with and without close); the close head is trained, nothing is claimed about its direction"},
    {"between": ("price_norm_mode='logret_std' (new)", "dataset already built with another mode"),
     "resolution": "hazard, not a choice: the loaded file decides; dataset_mismatches(settings, dataset) lists it and the run folder suffix keeps models apart"},
)

#: Preset name -> function(base) -> RunSettings (see settings_with_preset).
SETTINGS_PRESETS = {}


def preset_run_dir(run_dir, name=RECOMMENDED_PRESET):
    """``run_dir`` with ``_<preset name>`` appended once, so a model trained under another setup is never resumed."""
    suffix = "_" + name
    return run_dir if run_dir.endswith(suffix) else run_dir + suffix


def recommended_settings(base=None):
    """RunSettings of the "recommended_2026_11" preset: ``base`` (default ``RunSettings()``) with every row of RECOMMENDED_CHOICES
    written over it and the training folder suffixed with the preset name. The defaults of RunSettings are not touched."""
    base = RunSettings() if base is None else base
    if not isinstance(base, RunSettings):
        raise TypeError(f"recommended_settings(base): a RunSettings is required, got {type(base).__name__}")
    changes = {}
    for row in RECOMMENDED_CHOICES:
        parts = row["path"].split(".")
        section, field = parts[0], parts[1]
        if len(parts) == 2:
            changes.setdefault(section, {})[field] = row["value"]
        else:
            current = changes.get(section, {}).get(field)
            if current is None:
                current = dict(getattr(getattr(base, section), field))
            current = dict(current)
            current[parts[2]] = row["value"]
            changes.setdefault(section, {})[field] = current
    changes.setdefault("train", {})["run_dir"] = preset_run_dir(base.train.run_dir)
    return base.updated(changes)


SETTINGS_PRESETS[RECOMMENDED_PRESET] = recommended_settings


def settings_with_preset(name, base=None):
    """The settings of preset ``name`` over ``base``; an unknown name raises (no silent default)."""
    if name not in SETTINGS_PRESETS:
        raise ValueError(f"unknown settings preset {name!r}; known: {sorted(SETTINGS_PRESETS)}")
    return SETTINGS_PRESETS[name](base)


def dataset_mismatches(settings, dataset):
    """Dataset-build options named in ``settings.project.config_overrides`` that the loaded ``dataset`` was not built with (a list of
    messages, empty = consistent). Only ``price_norm_mode`` (plus its reg_target_scale == 1 requirement) and ``stale_bars`` are checked: they are stamped in the
    dataset file and act at build time, where apply_dataset_config would silently take the file's own value."""
    over = settings.project.config_overrides
    problems = []
    if "price_norm_mode" in over:
        have = dataset.get("price_norm_mode", "window_scale")
        if have != over["price_norm_mode"]:
            problems.append(f"price_norm_mode: settings say {over['price_norm_mode']!r} but the dataset was built with {have!r} "
                            "(the dataset's value is what the run uses; rebuild the data to change it)")
    if str(over.get("price_norm_mode", "")).startswith("logret") and float(dataset.get("reg_target_scale", 1.0)) != 1.0:
        problems.append(f"price_norm_mode={over['price_norm_mode']!r} needs reg_target_scale == 1.0 but the dataset carries "
                        f"{dataset['reg_target_scale']!r} (e.g. the 1h_s8 files): the logret modes refuse it, use a dataset built with scale 1")
    if over.get("stale_bars") is not None and not dataset.get("stale_bars"):
        problems.append(f"stale_bars: settings say {over['stale_bars']!r} but the dataset carries no stale_bars stamp "
                        "(built without it; rebuild the data)")
    return problems

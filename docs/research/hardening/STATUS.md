# STATUS: hardening + execution plan 2026-11

> Read this first in every step session, then ONLY your step in `PLAN_2026-11.md`. Update your row in the same PR.
> Keep it under 60 lines. Append real obstacles to `friction_log.csv` (D-012) and update `bottlenecks.csv` (model-quality / runtime / development). No subagents (owner, 2026-11-02).

Done before this plan: audit PRs 0-4 (#57): `core/registry.py` + `tools/check_patterns.py`, batched evaluation predict,
strict `build_config`, persisted run identity + `TrainSettings.seed`.

| Step | Title | State | PR / job | Note |
|---|---|---|---|---|
| M0 | Colab runner: GPU + reliable publish (+ republish profile-005) | todo | | needs #62 merged |
| M1a | Switches 1: target_report + direct_wide input + raw-input probe | todo | | |
| M1b | Switches 2: selection metric, confidence target, NIG floor, interval scale | todo | | |
| M2 | One Colab run: baseline vs all switches on (`model-all-001`) | todo | | needs M0, M1a, M1b |
| M3 | Read M2, ablate or diagnose, fix | todo | | needs M2 |
| TF-final | settings catalog + recommended preset | PR open | #64 | last TF task: `tools/settings_catalog.py` -> `docs/research/settings_catalog.md/.json` (generated); `workflow/presets.py` `recommended_settings()` = `recommended_2026_11` (RunSettings defaults unchanged); `main.ipynb` `USE_RECOMMENDED`, `evaluate_trained_model --preset`; the preset is 5 proven + 5 new (unmeasured) choices, so M2 measures the bundle; bottlenecks B-016..B-020 |
| S01 | Registry adoption (PR 5) | merged | #60 | 4 registries = `core.registry.Registry` (dict-style reads via `view`); fingerprint sha256 pinned in `tests/test_registry_adoption.py`; `TASK_REGISTRY.get(x)` now needs a default |
| S02 | pandas 4 deprecation fix | merged | #61 | `core/timeframe.normalize_tf` (`1d`->`1D`, `1w`->`1W`, else unchanged) is the one fixer; used by 5 `data/` modules; Binance `1d` interval untouched (other vocabulary); `1H`/compound `1d12h` not covered |
| S03 | Run profiler + Colab `profile-001` | merged | #62 | `tools/profile_run.py`; baseline read: `profile-004` @ `d4cc437` (3 targets, batch 512, T4: 2693 s; chicks 776, train 611, verification 544, candle 484); fixes since (prediction batch 2048, pattern-forest cap `rf_max_samples`, failed-report output tails) are measured by `profile-005` (same config); numbers + ranked S04 candidates in `profile_2026-11.md`; defaults changed by owner: D-013; bottlenecks B-007..B-015 in `bottlenecks.csv`; `profile-005` result is on Drive only (the push failed: askpass.sh vanished from /tmp), republished by the `colab_runner` fix in M0; then S04 (one PR per bottleneck) |
| S04 | Bottleneck fixes | todo | | needs S03 results |
| S03b | Development-friction audit | todo | | goal 8 (D-012) |
| S05 | `stats/metrics.py` (PR 10) | todo | | |
| S06 | SelectionMetric + trainer audit | todo | | needs S05 |
| S07 | Health-audit Colab job `health-001` | todo | | |
| S08 | Health-audit analysis + cards | todo | | needs S07 results |
| S09 | PredictionTable (PR 11) | todo | | |
| S10 | Data roles + Calibrator (PR 6) | todo | | needs S01 |
| S11r | Uncertainty fix arms (T3) | todo | | needs S08, S10 |
| S12 | SplitPlan + label_end (PR 12) | todo | | |
| S13 | Rolling driver + staleness | todo | | needs S09, S12 |
| S14r | Rolling study run | todo | | needs S13 |
| S15 | Raw-feature injection seam | todo | | only if S08 supports it |
| S16r | E-arch-001 run | todo | | needs S15 |
| S17 | Slot panel design + card | todo | | |
| S18 | Slot panel data | todo | | needs S17 |
| S19 | Slot panel model | todo | | needs S18 |
| S20r | E-slot-001 run | todo | | needs S19 |
| S21 | HeadSpec (PR 7-8) | todo | | after T3/T5 |
| S22 | TargetCodec (PR 13) | todo | | needs S21, S09 |
| S23 | Model contract + plug-in architectures | todo | | needs S01 |
| S24a | Namespace -> imports: trainer | todo | | after S03b |
| S24b | Namespace -> imports: signal_eval | todo | | |
| S24c | Namespace -> imports: model | todo | | |
| S24d | Namespace -> imports: evaluation/workflow/discovery | todo | | |
| S24e | Namespace -> imports: data | todo | | |
| S25 | Friction review (repeat every ~5 steps) | todo | | |
| S26 | Market-neutral portfolio: realistic test (E-mn-001) | deferred | | owner: only when asked; improve the model first |

**Model track first (owner, 2026-11-02): M0 -> M1a -> M1b -> M2 -> M3**, all improvements as switches, one test run, then investigate (targets: meaningful confidence, uncertainty below the predicted range, no skill loss). S-steps wait unless an M-step needs them; S26 deferred.

States: todo / in progress / PR open / merged / waiting Colab / closed (rejected) / skipped (reason).

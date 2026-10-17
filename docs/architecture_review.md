# Architecture review: weaknesses vs. the three goals

Base commit: `0045d8b` (branch `claude/quirky-knuth-rpt5lg`). Scope: analysis only, no code changed.
Goals: **(G1) change propagation**: one change in one place applies everywhere. **(G2) readability/extensibility**: easy to
find X, see its dependencies, and add a feature/loss/head/target/study without touching unrelated code. **(G3) token cost**:
an AI agent can do a typical task by reading little.

**Method.** Throwaway scripts (kept out of the repo) used `ast`/`symtable` over the 116 shared-namespace modules to
compute free names, compared them with the definitions in every package, and built per-package dependency graphs (Tarjan
SCC for cycles). They also found exact and near-duplicate function bodies across all 1,263 top-level functions
(normalized AST hash plus `difflib` ≥ 0.85), used `difflib` on the loaders and `__init__`s, parsed the notebooks' JSON, and
timed first package access in subprocesses. Token estimates count Arabic characters at about 2 chars/token and other
text at about 4 chars/token (14% of the code-package text is Arabic). Every number below comes from those runs.
"Proved" means I reproduced the behaviour in a fresh interpreter.

**Already working well (keep):** maps are generated and checked (`build_map.py --check`, pre-commit hook, CI); module
cards are enforced by a ratchet; zone imports are checked (`check_deps.py`: 282 files, 0 errors); golden/byte-identical
tests exist (`tests/golden_pre_multi_tf.json`, `test_multi_tf.py`); real registries already exist for heads
(`model/heads.py:23`), task losses (`trainer/tasks.py:68`) and sample filters (`data/windows.py:43`); `cross_asset/` is
a conventional package with relative imports. The weaknesses below are mostly about what these aids **cannot see**.

---

## 1. Summary: weaknesses ranked by impact

| # | Weakness | Root cause (one line) | Goals | Sev. |
|---|---|---|---|---|
| W1 | The 7 "packages" are not Python modules. 116 files are `exec`'d into one dict and reach each other through **998 implicit free-name references** (61 of them cross packages). None of these show up in `check_deps`, in `maps/` DEPENDS, or in an IDE. | Cell extraction kept the `%run` notebook semantics (late binding, `globals()` patching) as the permanent architecture instead of as a migration step. | G1 G2 G3 | **High** |
| W2 | Configuration has several mutable "single sources of truth" that collide by name. Proved: in `main.ipynb`'s namespace `reset_config()` resets `CONFIG` to the **trainer's 6-key** `DEFAULT_CONFIG`. `workflow.CONFIG` is a separate copy of `data.CONFIG`. `update_config` silently accepts unknown keys. | One flat namespace per notebook with no namespacing, plus config as module-global dicts deep-merged without a schema. | G1 G2 | **High** |
| W3 | Lightweight consumers **copy library logic** instead of importing it: the `last_candles` schema (3 copies plus hard-coded indexes), `entry_range_to_prices` (2), `NIG_ALPHA_DEN_MIN` (2), `_auc` (2). Research and tools copy more: AUC ≥6, rank-IC ≥7, bootstrap ≥6, cluster-t ≥7, and the frozen **H07 rule in 6 copies with different normalisations**. | W1 makes library code all-or-nothing to load (seconds of load time, side effects, TF and gdown). There is no small importable tier for constants and stats, and studies may not import each other. | G1 | **High** |
| W4 | Orchestration lives in notebook cells, so other code drives it by **text-patching source**: `tools/evaluate_trained_model.py` relies on 21 source-text anchors. Tests `exec` `main.ipynb` cells found by text markers. `workflow/` modules read 12 names that exist only as notebook globals (`main_config`, `MODEL_TF`, `test_dict`, `model`...). | `main.ipynb` holds the run configuration (57 UPPERCASE globals) and the step sequence. There is no callable `run(settings)` API. | G1 G2 G3 | **High** |
| W5 | Pluggable concepts (features, targets) are spread across modules **without a registry**. A custom feature lives in 5 places with a silent normalization fallback. `infer_feature_columns` forward-references 10 column-list functions in 5 later modules. `entry_range` appears in 19 files. | The registries were built for heads and losses but never for features or target modes (PHILOSOPHY §8 defers this). Prefix-based classification stands in for registration. | G1 G2 G3 | Med-High |
| W6 | Loading a package has heavy side effects and fails all-or-nothing. `data.<x>` runs 135 self-tests (**38.6 s** vs 2.9 s without them). `trainer.<x>` runs a real training. `model.<x>` takes 22.2 s. `signal_eval.compute_ic` raises `ModuleNotFoundError: gdown` because of a dead module. | Notebook-cell self-tests were kept at module top level, and the lazy package loads every module. | G2 G3 | Medium |
| W7 | Infrastructure boilerplate is copied and already drifting: 7 `_loader.py` + 7 `__init__.py` (860 lines, 3 API variants), 9 repo-root-finder cells (8 variants), 64 files hacking `sys.path` (no `pyproject.toml`), 3 notebook executors, 2 fetch tools sharing about 15 identical functions. | Every extraction was a copy-and-rename of the previous one. There is no shared tier-0 helper because nothing is importable before `sys.path` is fixed. | G1 G3 | Medium |
| W8 | The navigation aids are blind and noisy. `maps/` shows **0 DEPENDS** for all 116 shared-namespace modules. The same boilerplate PITFALLS sentence appears in ≥96 cards. `tags.md` is 50 KB / 1,181 lines. `docs/` holds 57 md files totalling 738 KB, including a 43 KB root README. | Maps derive dependencies from `import` statements only (W1 removes them). Cards repeat package-level facts. Docs have no generated index or retention rule. | G3 G2 | Medium |

W1 is the **root of W3, W4, W6 and W8**. The fix order in §4 therefore starts by making the implicit graph visible and
checkable, and only then removes the shared namespace, one leaf module at a time.

---

## 2. Weaknesses in detail

### W1. Shared-namespace packages hide their dependency graph

**What.** `data/`, `model/`, `trainer/`, `evaluation/`, `signal_eval/`, `workflow/` and `discovery/` (116 modules) are
loaded by `<pkg>/_loader.py:load_into(ns)`, which `exec`s each file into one dict in a fixed `MODULES` order. Modules
use each other's names without importing them.

**Evidence (free-name analysis):**
- **998** (module, free name) pairs. 870 resolve to an earlier module of the same package, **41 to a LATER module**
  (data 31, evaluation 5, workflow 5; these work only because they are resolved at call time), **61 to another package**,
  and **26 to nothing in any package**: they are notebook globals.
- What the free names point to: 368 shared imports (`np`, `pd`... from `common.py`), 456 functions, 30 classes, 44
  constants, **70 references to mutable globals** (`CONFIG`, `LAST_COLUMNS`, `TASK_REGISTRY`, `TRAINER_REGISTRY`,
  `COINS_BY_CATEGORY`, `ANTI_MEMORIZATION_CONFIG`, ...).
- **73 of 116 modules have no import statement of their own.**
- Cross-package implicit edges: `workflow` → data 15, model 13, evaluation 10, trainer 10; `discovery` → signal_eval 8,
  data 5. For example, `workflow/diagnostics.py` uses 10 names from `model`/`trainer`, and `workflow/capacity.py` uses
  `_predict`, `_spearman` and `_unwrap_model` (private names) from `model/diagnostics.py`.
- Names that exist only as notebook globals: `workflow/` uses `main_config`, `MODEL_TF`, `MODEL_TFS`,
  `EVAL_TARGET_SPECS`, `PRICE_TARGETS`, `CHICKS_TARGETS`, `LAST_CLOSE_COL`, `MODEL_OVERRIDES`, `MODEL_SEQ_LEN`,
  `MODEL_N_FEATURES`, `model`, `test_dict` (for example `workflow/batches.py:21`, `workflow/diagnostics.py:38`,
  `workflow/capacity.py:146`, `workflow/permutation_control.py:29`, `workflow/reports.py:9`). `discovery/` uses
  `FEATURE_ORDER`, `DUMMY_DF` and `CATEGORY_HYPOTHESES` (`discovery/selftest.py:12`, `discovery/survey.py:47,103`).
- Cycles: in `data`, one strongly connected component of **10 modules** (`drive, runtime, heads, custom, features,
  market_context, cross_sectional_features, binance_client, funding_oi, phase2`). In `workflow`, `{capacity, diagnostics}`
  and `{candle_baseline, market_neutral, wiring_selftest, selective_eval, verification}`. model, trainer, evaluation,
  signal_eval and discovery are acyclic.
- Silent shadowing when several packages are merged into `main.ipynb`'s `globals()`: `DEFAULT_CONFIG` (data vs trainer,
  see W2), `_auc` (`model/diagnostics.py:94` vs `workflow/generalization.py:117`), and `NIG_ALPHA_DEN_MIN`
  (`model/nig_layers.py:52` vs `evaluation/targets.py:176`). After main cell 37 loads `generalization`, every model
  diagnostic uses workflow's `_auc`. The two are numerically equal today, but a fix to `model._auc` would not reach
  `main`. Ten module basenames repeat across packages (`common`×3, `diagnostics`×3, `selftests`×3, `config`, `heads`,
  `checkpoints`, `live`, `metrics`, `windows`, `selftest`), and `discovery/evaluation.py` shares its name with the
  `evaluation/` package.
- `tools/check_deps.py` only checks zone imports (code vs research/docs/tests) and reports "0 errors, 0 warnings": it
  cannot see any of the above.

**Root cause.** The notebook-to-package move preserved `%run` semantics so that results stay byte-identical. That was
correct as a *move*, but the move has no follow-up: no explicit imports, no per-module namespace, and no tool that
reconstructs the graph.

**Goals hurt.** G1: changing a name's definition affects unknown readers, and a later module, the notebook or a test
patch can silently redefine it. G2: "what does X depend on / who uses X" can only be answered by grepping all packages
plus notebooks. G3: an agent must read whole neighbouring modules or `common.py` to resolve names; the map cannot tell it.

**Fix.** (a) Now: a `tools/implicit_deps.py` that computes exactly this analysis. `build_map.py` should emit
`USES:`/`USED BY:` per module, and `check_deps.py` should fail on **new** cross-package implicit references, new
forward references, and new name collisions in the merged notebook namespace (a ratchet with an allowlist). (b) Then
turn modules into real modules leaf-first (§4, S9). 25 of the 116 modules already use only shared imports (trainer 9,
signal_eval 4, data 3, model 3, evaluation 3, workflow 2, discovery 1) and need only an import header.

### W2. Config: several "single sources of truth", mutable, colliding, unvalidated

**Evidence:**
- Two dicts named `DEFAULT_CONFIG`, each documented as "the single source of truth": `data/defaults.py:27` (100 keys)
  and `trainer/config.py:30` (6 sections). Only 1 key overlaps. Others: `model/config.py:20` `MODEL_CONFIG`,
  `model/config.py:68` `ANTI_MEMORIZATION_CONFIG`, `main.ipynb` cell 18 `main_config` plus `ANTI_MEMORIZATION_TRAINER`,
  **57 UPPERCASE run-setting globals** in `main.ipynb`, `cross_asset` presets, and `DEFAULT_CUSTOM_SETTINGS`
  (`data/custom.py:47`), which is merged with `DEFAULT_CONFIG["custom_settings"]` (`data/defaults.py:378`). So feature
  defaults live in two places.
- **Proved bug (latent):** load `data` and then `trainer` into one namespace, as `main.ipynb` does with `%run` at cells
  3 and 17, then call `reset_config()` (`data/runtime.py:54`). `CONFIG` now has **6 keys** (the trainer's) and has lost
  `higher_tf_mode` and the other 99 pipeline keys. `data/presets.py:351`'s self-test would raise `KeyError` in the same
  situation.
- **Proved:** `workflow/__init__.py:48-49` seeds itself from a **fresh** `data.load_into({})`, so
  `workflow.CONFIG is data.CONFIG` is `False`. `data.update_config(seed=12345)` is not visible in `workflow.CONFIG`.
  This breaks `data/runtime.py`'s own contract ("CONFIG is the same dict everywhere"). The lazy `workflow` package also
  lacks `build_model_fn` and `build_training_system`, which its modules call.
- **Proved:** `update_config(seed=12345)` added an unknown key with no error. `_deep_update` (`data/runtime.py:24`) has
  no schema, which contradicts CLAUDE.md's "unknown config keys raise". Only `trainer.build_config` validates.
- **133** `cfg.get(key, default)` calls in code packages (top: `trainer/system.py` 32, `data/split.py` 15,
  `data/heads.py` 15, `data/pipeline.py` 14, `data/checkpoints.py` 14). Each is a second, hidden default for a key that
  `DEFAULT_CONFIG` also defines. `data/sources.py:80` prints and **ignores** unknown categories.
- `main.ipynb` has 42 `globals()` uses (17 `globals().get(...)`). `workflow/retarget.py:162` reads
  `globals().get("PRICE_TARGETS")`, with a comment saying section 4 defines it later.

**Root cause.** Config is modelled as module-global mutable dicts in a namespace without packages. The notebook is the
config surface (W4). Validation was added per subsystem (trainer) rather than at one boundary.

**Goals hurt.** G1: a default changed in `DEFAULT_CONFIG` can be overridden by a `.get(k, other_default)` somewhere, and
`workflow` holds a stale copy. G2: no single place lists the run settings.

**Fix.** Rename the collision away now (S1). Add strict key validation in `update_config` and `load_config` (S6). Then
use frozen dataclass sections per subsystem (`PipelineConfig`, `TrainerConfig`, `ModelConfig`, `RunSettings`) built
from dicts with an unknown-key check, with defaults *only* in the dataclass, and replace `.get(k, d)` with attribute
access when a module is touched (ratchet: count may only go down).

### W3. Library logic copied into consumers instead of imported

**Evidence: library and tools code** (changing the canonical version does not propagate):

| Concept | Canonical | Copies |
|---|---|---|
| `last_candles` column schema | `data/windows.py:20` `LAST_COLUMNS` | `cross_asset/data.py:29` (`LC` dict), `tools/bracket_eval.py:27`; hard-coded indexes `cross_asset/train.py:369,384-385,411-412`, `cross_asset/selftest.py:59`, `docs/research/scripts/hourly_1h/base_data.py:8`, 3 audit scripts |
| entry_range inverse | `workflow/retarget.py:229` `entry_range_to_prices` | `cross_asset/data.py:419` (95% identical; parameter renamed) |
| NIG alpha floor | `model/nig_layers.py:52` | `evaluation/targets.py:176` |
| AUC | (none) | `model/diagnostics.py:94`, `workflow/generalization.py:117`, `docs/.../anti_memorization_benchmark.py:242`, `docs/.../entry_range_eval/lib.py:122`, `repro_05...:auc_pair`, `hourly_1h/exc_analysis.py:auc_in_q` |
| rank IC / Spearman | `signal_eval/core.py` `compute_ic` | `model/diagnostics.py:102`, `workflow/market_neutral.py:_rank_ic`, `cross_asset/report.py:94,110`, `research/edge_discovery/lib.py:36`, `docs/.../entry_range_eval/lib.py:88`, `anti_memorization_benchmark.py:251` |
| bootstrap (statistical) | (none; `signal_eval/bootstrap.py` is the Drive/gdown helper, despite the name) | `workflow/verification.py:12`, `workflow/candle_baseline.py:37`, `workflow/market_neutral.py:137`, `evaluation/tearsheet.py:217`, `research/h07_combos/01_combos.py:boot_dsharpe`, `research/h07_volsizing/01_volsizing.py:block_boot_diff` |
| notebook executor | `docs/research/audit/_nbload.py:15` | `tools/evaluate_trained_model.py:56`, `anti_memorization_benchmark.py:49`, plus `load_notebook_defs` ×6 in `docs/research/scripts/` |
| Binance fetch helpers | (none) | `tools/fetch_history_csv_concurrent.py` vs `tools/fetch_history_vision_colab.py`: about 15 functions identical or ≥ 0.97 similar (`request_with_retry`, `resolve_base_url`, `http_get_json`, `file_stats`, `update_registry`, ...) |

**Evidence: research.** The 43 `.py` files in `research/` import **no** repo package (0 `import data|model|...`).
`research/edge_discovery` has its own `lib.py`, `tsbt.py`, `features.py` and `events.py`. Within that one study,
`ctstat` ×4, `frame` ×3 (41-line exact copy in `31_wick_capture.py:17` / `32_breakout_stops.py:17`), `ema` ×3,
`cluster_t` ×2 and `dsr` ×2 are copied between numbered scripts. The **frozen H07 sizing rule** `0.02/std30` capped
at 3 exists in 6 copies with *different* normalisation:
`tools/h07_forward.py:56` (÷ number of live coins), `research/h07_combos/01_combos.py:41` (÷ live coins),
`research/edge_discovery/16_h07_oos.py:19` (÷ `len(cols)`), `15_h07_robust.py:16` and `14_holdout.py:34` (no division),
`07_trend.py:18`. Over the whole repo there are 18 groups of exactly duplicated function bodies (37 defs), and 71
function names are defined in more than one file.

**Root cause.** (1) W1/W6: getting `LAST_COLUMNS` "properly" means loading the whole pipeline: 2.9–38.6 s, pandas_ta,
self-tests, `sys.path` setup. Copying is cheaper. (2) The rule "a study never imports another study (lift shared code
into a package)" has no target package to lift into: there is no light `stats/` or `rules/` tier. (3) Research scripts
cannot `import` repo packages without `sys.path` hacks, because there is no `pyproject.toml`.

**Goals hurt.** G1, directly: a fix to the AUC, the schema or the H07 rule must be found and repeated by hand. The H07
copies already disagree.

**Fix.** A pure tier-0/1 layer: `core/schema.py`, `core/constants.py`, `stats/` (auc, rank_ic, bootstraps, cluster t,
DSR, Sharpe/turnover backtest primitives), and `rules/h07.py`. All are importable in milliseconds with only numpy and
pandas. Library copies are replaced (fix commits, with equality tests). **Recorded research scripts are not edited**
(reproducibility); new studies must import these modules, enforced by a check (§4, S4).

### W4. The run lives in notebook cells and is driven by text patching

**Evidence.**
- `tools/evaluate_trained_model.py:150-240` executes `main.ipynb` cell by cell and rewrites the source with **21
  string/regex anchors**, for example `"PANEL_MODE = False"`, `"# ── نهاية الإعدادات ──"`, `'"batch_size": \d+,'`,
  `"retarget_splits(train, val, test, mode=TARGET_MODE)"`, `"model.summary()"`. Changing a comment or a literal's
  formatting in `main.ipynb` breaks the tool, and no test pins most anchors.
- Tests run main-notebook cells: 11 `_cell("main.ipynb", <key|"text:...">)` calls in `tests/test_entry_range.py`,
  `test_reg_target_scale.py` and `test_real_price_modes.py`, plus text-marker scans in `test_workflow_packages.py:142,161-175`.
- Six legacy scripts (`docs/research/scripts/feature_screen.py:27`, `causal_decomposition.py:39`,
  `corrected_retest.py:37`, `holdout_momentum_orth_natr.py:37`, `mom_orth_fidelity.py:20`, ...) `exec` the code cells
  of `crypto_data_pipeline_v6.ipynb` as their API. Verified: this fails with `RuntimeError` unless the working
  directory is the repo root or `REPO_DIR` is set, because the runner cell searches for the repo.
- `main.ipynb`: 27 code cells, 657 code lines, 57 UPPERCASE settings, 42 `globals()` uses. Run settings such as
  `run_dir` are computed inline from six `globals().get(...)` flags (cell 18).

**Root cause.** The functions moved to `workflow/`, but the **composition** (the settings and the order of steps) stayed
in the notebook, so the notebook is the de-facto public API.

**Goals hurt.** G1: a new run option must be added to the notebook, the evaluation tool's anchors and the tests. G2:
`main` cannot be run or tested except by reading the notebook. G3: to touch evaluation, an agent reads `main.ipynb`
cells 3–24 (about 8.4k tokens) plus the 278-line tool.

**Fix.** `workflow/settings.py` (`RunSettings` frozen dataclass with today's defaults, which must reproduce current
behaviour) and `workflow/run.py` (`prepare_data(settings)`, `build(settings)`, `train(settings)`,
`evaluate(settings, model)`). Notebook cells become one-line calls; `evaluate_trained_model.py` builds `RunSettings`
from argparse. Golden check: the tool's output on the synthetic fixtures in `tests/test_entry_range.py` stays identical.

### W5. Features and targets have no registry

**Evidence.** Adding one custom feature means editing:
1. `DEFAULT_CUSTOM_SETTINGS` (`data/custom.py:47`);
2. a branch in the 280-line `add_custom_features` (`data/custom.py:177-456`);
3. a **parallel** name list in `custom_feature_names` (`data/custom.py:546-596`), which must match by hand;
4. a prefix in `FEATURE_KINDS` (`data/normalize.py:103-199`). An unknown prefix falls back to
   `DEFAULT_KIND = CUMULATIVE` **silently** (`data/normalize.py:88`), against the "no silent defaults" rule;
5. possibly `DEFAULT_CONFIG["custom_settings"]` (`data/defaults.py:378`).

The full feature list is assembled by `infer_feature_columns` (`data/features.py:90`), which forward-references 10
`*_columns()` functions in 5 later modules (funding_oi, phase2, market_context, cross_sectional_features). That is the
main cause of the 10-module cycle. Target modes: `entry_range` appears in **19 files** (114 occurrences): `workflow/retarget.py`
(25), `tests/test_entry_range.py` (23), `main.ipynb` (14), cross_asset ×3 files, evaluation ×2, workflow ×5, tools, docs.

**Root cause.** Registries were introduced where a notebook already had one (heads, task losses). Features grew one
`if s[...]` branch at a time. PHILOSOPHY §8 defers the feature registry "until the next normalization change".

**Goals hurt.** G1 (names, normalization and settings drift apart), G2 (adding a feature touches 3 files and a
1,374-line module), G3 (an agent reads about 30k tokens for this task; see §5).

**Fix.** `data/feature_registry.py`: `FeatureSpec(name, columns(settings), compute(df, settings), kind, settings_key,
default_settings)`. `custom_feature_names`, `FEATURE_KINDS` and `infer_feature_columns` become *derived* from the
registry. Unregistered columns raise, with a frozen allowlist of today's legacy prefixes so the outputs stay
byte-identical. Do the same for target modes (`TargetModeSpec(name, build_y, invert_to_prices, eval_spec)`), used by
`workflow/retarget.py`, `evaluation/targets.py` and `cross_asset`.

### W6. Import-time side effects and all-or-nothing loading

**Evidence (timed):** `import data; data.CONFIG` takes **38.6 s** and prints 157 lines (it runs
`run_pipeline_selftests()`, `data/selftests.py:2426`), against 2.9 s without the self-test module.
`import model; model.MODEL_CONFIG` takes **22.2 s** (`model/selftests.py:153`). `trainer.<name>` loads `smoke_test`,
which **runs a real tiny training** (`trainer/__init__.py:46`, `trainer/_loader.py:10`; not executed here).
`import signal_eval; signal_eval.compute_ic` **fails** with `ModuleNotFoundError: gdown` because `signal_eval/bootstrap.py:6`,
a dead module kept for name compatibility, imports gdown. Tests stub it (`tests/test_evaluation_packages.py:54`,
`tests/test_workflow_packages.py:183`). Code packages contain 27 module-level test calls. `data/custom.py` runs 14
self-tests at load, and lines 598–1374 (56% of the file) are tests. Of the 34,901 code-package lines, 4,338 are inside
self-test functions. `data/selftests.py` alone is 2,426 lines (about 37.6k tokens).

**Root cause.** Notebook cells showed "✅ tests passed" when run. The packages keep that behaviour on every load, and
`load_into` has no notion of optional modules (callers must know the right `exclude=` set; `_nbload` encodes it).

**Goals hurt.** G2 (a consumer cannot use one function cheaply, which pushes it to copy, see W3), G3 (large files mix
tests with code).

**Fix.** Self-tests stay as code but become opt-in. The lazy package excludes them by default. The runner notebook calls
`run_pipeline_selftests()` explicitly, so the notebook behaviour is unchanged. Move commits extract `custom.py`'s tests
to `data/selftests_custom.py`. The gdown import becomes lazy.

### W7. Copied infrastructure (loaders, root finders, path hacks)

**Evidence.** 7 `_loader.py` (448 lines) and 7 `__init__.py` (412 lines) are copy-renamed (pairwise similarity
0.57–0.99) and have already diverged into 3 `load_into` signatures: `(ns, exclude)`, `+exec_module`,
`(ns, only, exclude)`. `workflow/__init__.py` alone also seeds the data namespace. Repo-root finder: 9 cells in 8
notebooks, **8 distinct variants** (about 1.4–3 KB each). **64** `.py` files call `sys.path.insert/append` (docs 38,
tests 18, tools 8). There is no `pyproject.toml`. Notebook executors: 3 independent ones (see W3). Test helpers:
`_quiet` ×6 identical, `_cells` ×2, `_main_ns` ×2, `ohlcv` ×2.

**Root cause.** Each notebook extraction was done by copying the previous package. Nothing could be shared because
nothing is importable before the repo root is found.

**Goals hurt.** G1 (a loader fix must be made 7 times; the variants show this is already failing) and G3.

**Fix.** One `core/nsloader.py` (`load_into(package_dir, modules, ns, only=, exclude=)` and `lazy_package(globals(), ...)`).
Each `_loader.py` becomes a `MODULES` tuple plus one call. Add one `pyproject.toml` (`pip install -e .` in CI and Colab)
and one canonical bootstrap snippet. A test asserts that all runner cells contain the identical snippet, or
`tools/nb_cells.py` generates it.

### W8. Navigation aids blind to the real graph; documentation sprawl

**Evidence.**
- `maps/*.md`: **0 `DEPENDS` lines** in data, model, trainer, evaluation, signal_eval, workflow and discovery, because
  DEPENDS is computed from `import`s only.
- The same PITFALLS boilerplate ("Executed into the one shared ... namespace by .../_loader.py (never imported on its
  own) ...", about 170 chars) appears in ≥96 cards.
- `maps/` totals 297 KB (about 74k tokens). `maps/data.md` is about 9.3k tokens. `maps/tags.md` is 50 KB / 1,181 lines
  (about 12.6k tokens) and full of low-value tags (`- 136 tests:`, `- a.2:`, `- 2018-2023:`).
- The `h07` tag points to 6 modules with no canonical one.
- `docs/` holds 57 md files totalling 738 KB (largest: `pr7_external_sources.md` 169 KB, `hourly_1h.md` 61 KB). The
  "index" `docs/research/README.md` is 37 KB (about 9.1k tokens) and the root `README.md` is 43 KB (about 10.7k tokens).
- `CLAUDE.md` (1.6k tokens) spends 5 of its bullets explaining the per-package loading pattern.

**Root cause.** The map generator was designed for import-based code. Cards repeat facts that belong at package level.
Docs grow by addition (rule 2) with no generated index, size budget, or "superseded" marker.

**Goals hurt.** G3, mainly: PHILOSOPHY's own budget "map + one card + target symbol" is exceeded by 3–6×. G2: dependencies
are invisible.

**Fix.** S0 (USES/USED-BY from the implicit analysis). Emit package-level PITFALLS once per package. Give `tags.md`
a minimum tag quality (drop tags that equal symbol names already in the API list, and numeric tags), or split it per
zone. Add a size budget per `maps/<zone>.md` checked by `build_map --check`. Generate a one-line-per-doc
`docs/research/INDEX.md` from the docs' first heading and status, and shrink the hand-written READMEs to a screen each.

---

## 3. Target structure

### 3.1 Tiers (what fits this project)

```
T3  apps / orchestration   notebooks (*.ipynb runners), workflow/, discovery/, tools/ (CLIs), research/<study>/, tests/
                           may import T0-T2; research may not import another study; nothing imports T3
                           (exception: tests import anything)
T2  functional units       data/   model/   trainer/   evaluation/   signal_eval/   cross_asset/
                           may import T0, T1 and the T2 deps listed below; never T3
T1  utilities              stats/  (auc, rank_ic, spearman, bootstraps, cluster-t, DSR, sharpe/turnover, backtest
                           primitives)  rules/ (h07.py: the frozen research rules as functions)
                           pure numpy/pandas/scipy; may import T0 only
T0  core                   core/schema.py (LAST_COLUMNS, TS_COL, TARGET_COLUMNS, target-mode names),
                           core/constants.py (NIG_ALPHA_DEN_MIN, ...), core/config.py (strict dataclass-from-dict
                           helper), core/registry.py (Registry: register/get raises on unknown name),
                           core/paths.py (repo_root), core/nsloader.py (the ONE shared-namespace loader)
                           stdlib + numpy only; no internal imports; no import-time side effects
```

Allowed T2 → T2 edges (explicit and checked): `evaluation → data.schema` (via core), `cross_asset → core/stats`, and
otherwise none. `model` and `trainer` do not know `data`. Adapters between them (dataset → model inputs) live in T3
`workflow/`, which is the composition layer; that is already its role.

**Where config and registries live.** Each subsystem owns a frozen dataclass section (`data/config.py: PipelineConfig`,
`trainer/config.py: TrainerConfig`, `model/config.py: ModelConfig`, `workflow/settings.py: RunSettings`), built with
`core.config.from_dict(cls, d, strict=True)`. Defaults live **only** in the dataclass. A module-global `CONFIG` remains as
a legacy facade during migration, never in new code. Registries are `core.registry.Registry` instances owned by the
subsystem: `data.FEATURES`, `data.TARGET_MODES`, `data.SAMPLE_FILTERS`, `model.HEADS`, `trainer.TASKS`. Built-in
entries are registered at definition, and unknown names raise.

**Notebooks** import T0–T3 and contain no definitions, only `settings = RunSettings(...)` and step calls. **Research
studies** import T0–T2 (`stats`, `rules`, `data.schema`, ...) and must not re-define a function that exists in `stats/`
or `rules/` (check: name match plus AST similarity, a warning first and then an error for new studies).

**Enforcement.** `check_deps.py` gains a `TIERS` table and fails on any upward or sideways edge not in the allowlist,
counting both `import` edges **and** implicit shared-namespace edges from S0. `build_map.py` prints the tier in MAP.md.

### 3.2 What I deliberately do NOT adopt from the library model, and why

| Library practice | Not adopted because |
|---|---|
| Rewrite everything as clean importable modules in one go | Breaks byte-identical behaviour and test monkeypatching (`ns["mount_drive"] = ...`), and is unreviewable. Migrate leaf-first, on touch (PHILOSOPHY rule 6), behind the S0 ratchet. |
| Drop notebooks as entry points | Colab is the runtime and it saves `.ipynb` only. Notebooks stay as T3 runners: thin, no definitions. |
| Public API stability, deprecation cycles, semver, multiple distributions | Single owner, no external users. One `pyproject.toml` with an editable install is enough. Old names may be kept as aliases only where checkpoints or recorded results need them. |
| Registries and plugin entry points for everything | Only for things that really vary per experiment (features, target modes, heads, losses, sample filters). Not for metrics, plots or reports. |
| Remove global mutable state immediately | `CONFIG` mutation is wired into checkpoint fingerprints and notebooks. Freeze it at the boundary (dataclass built from `CONFIG`), forbid it in new code, and let the count fall. |
| Refactor or move legacy research scripts to use `stats/` | They are records of results. Editing them changes what was run. Mark them legacy and keep them out of the default map view; only new studies must import. |
| Delete load-time self-tests | They encode many past bugs. Keep them as opt-in code that runner notebooks call explicitly. |
| Deep package hierarchies (`csp.data.features.custom...`) | A flat set of top-level packages with tiers keeps import paths short for notebooks and agents. |

---

## 4. Migration plan (ordered, small, safe)

Each step is one commit type. Moves never mix with edits. Every step runs `build_map.py`, `check_deps.py` and pytest.
"Golden" means a byte-equality test exists, or is added, before the step.

| Step | Type | What | Proof of no behaviour change | Benefit | Risk | Enforce |
|---|---|---|---|---|---|---|
| **S0** | feat (tools) | `tools/implicit_deps.py`: the free-name/definition analysis used for this review. `build_map` adds `USES:`/`USED BY:` per shared-namespace module. `check_deps` fails on new cross-package implicit references, forward references and merged-namespace collisions (allowlist = today's 61 + 41 + 3). | Tools only. | Maps become truthful at once: G2, G3. Stops W1 from getting worse. | Low (false positives from dynamic `globals()`: 44 uses in 12 modules; allowlist them). | **check_deps, build_map** |
| **S1** | fix | Remove the `DEFAULT_CONFIG` collision: data binds its defaults under a non-colliding name (`PIPELINE_DEFAULT_CONFIG`; keep the `DEFAULT_CONFIG` alias for notebooks), and `reset_config` uses it. `workflow/__init__` seeds from `data`'s own namespace (shared `CONFIG`) instead of a fresh copy. | New tests: data+trainer merged namespace `reset_config()` keeps 100 keys; `workflow.CONFIG is data.CONFIG`. | Fixes a latent wrong-config run (W2). | Low. | S0 collision check |
| **S2** | refactor | `core/nsloader.py` plus `core/paths.py`; the 7 `_loader.py`/`__init__.py` become thin. One canonical runner bootstrap snippet, checked by test for identity across the 9 runner cells. Add `pyproject.toml`. | Existing package and runner tests (`test_*_packages.py`) already execute every loader path. | −700 lines; one place to fix loading (W7). | Low-medium (Colab path detection). | test |
| **S3** | move, then fix | `core/schema.py` (`LAST_COLUMNS`, `TS_COL`, `TARGET_COLUMNS`), `core/constants.py`. `data/windows.py` re-binds the same objects. Then fix commits point `cross_asset`, `tools/bracket_eval.py` and `evaluation/targets.py` at them. | Golden `golden_pre_multi_tf.json`, `test_multi_tf`; a test asserting `cross_asset` indexes equal `LAST_COLUMNS.index(...)`. | Schema and constants change in one place (W3). | Low. | S0: no new literal copies (lint: `LAST_COLUMNS =` defined once) |
| **S4** | move, then fix | `stats/` (auc, rank_ic, spearman, day and block bootstrap, cluster_t, dsr, sharpe/turnover) and `rules/h07.py` extracted **from `tools/h07_forward.py`** (the live, frozen version). The tool imports it. Library copies (`model._auc`, `workflow._auc`, `workflow._rank_ic`, bootstraps) delegate to `stats`. | Golden: `research/h07_forward/ledger*.csv` rebuilt byte-identically with `--dry-run`; numeric-equality tests old vs new for each stat (same tie and NaN rules). | Stats and H07 change once. New studies are cheap (W3). | Medium (tiny numeric differences such as NaN handling: compare first, and keep the old body if not identical). | check_deps rule: `research/*` new files may not define names that exist in `stats/` or `rules/` |
| **S5** | move + fix | Self-tests opt-in: `custom.py` tests → `data/selftests_custom.py` (move). Lazy packages exclude `selftests` and `smoke_test` by default. Runner notebooks call them explicitly. `signal_eval/bootstrap.py` imports gdown lazily. | The runner notebook still prints the same "✅" lines (test via `_nbload` runner execution). | `import data` drops from 38.6 s to about 3 s; `custom.py` drops from 1,374 to about 600 lines (W6). | Low. | test: package attribute access performs no I/O or training |
| **S6** | fix | Strict keys in `update_config`/`load_config` (unknown top-level and known-section keys raise; allowlist for documented free-form dicts). | Pytest plus all self-tests (they exercise `update_config` widely). | "No silent defaults" becomes true for the pipeline (W2). | Medium (old saved configs with stray keys: give `load_config` a `strict=False` escape hatch, logged). | test |
| **S7** | move, then refactor | `workflow/settings.py` (`RunSettings`: the 57 notebook globals with today's defaults) and `workflow/run.py` (step functions). `main.ipynb` cells call them. `evaluate_trained_model.py` builds `RunSettings` (no source patching). Tests call functions instead of `_cell("main.ipynb", ...)`. | Golden: the `evaluate_trained_model.py` outputs on `tests/test_entry_range.py` fixtures, and `run_wiring_selftest()`, identical before and after. | Removes 21 text anchors and the 12 notebook-only globals (W4). Main becomes testable. | Medium-high (largest step; split per notebook section). | S0: `workflow/*` may not read notebook-only names |
| **S8** | refactor | Feature registry (`data/feature_registry.py`), then target-mode registry. `FEATURE_KINDS`, `custom_feature_names` and `infer_feature_columns` are derived. An unknown prefix raises (legacy allowlist). | Golden: `feature_order`, `process_windows` output and the dataset hash on the golden fixture are byte-identical. | Adding a feature means 1 file plus 1 registry entry (W5). Breaks the 10-module data cycle. | Medium. | test: every produced column is registered |
| **S9** | refactor (continuous) | Convert shared-namespace modules into real modules **leaf-first**, using the S0 graph. Start with the 25 leaf modules (trainer 9, signal_eval 4, ...), then `model` and `signal_eval` (acyclic), and `data` last. `load_into` keeps exec'ing the same files for notebook compatibility until a package is fully converted. | Per module: package tests plus golden tests. Patch-sensitive tests are moved to patch the module attribute. | W1 shrinks monotonically; tier rules become checkable by imports alone. | Medium (monkeypatch semantics change per module; do one module per commit). | S0 ratchet counts only go down |
| **S10** | docs | Map diet: package-level PITFALLS once; `tags.md` quality filter; per-zone map size budget; generated `docs/research/INDEX.md`; mark superseded docs; shrink root README and the 5 CLAUDE.md loading bullets to one line plus a pointer. | Docs only. | Lower G3 cost everywhere (§5). | Low. | **build_map --check** (budgets) |

The first three to do: **S0** (visibility and ratchet: cheap, and the basis for everything else), **S1** (a real
correctness bug), **S3** (smallest change that removes copies of the most widely duplicated schema). S2 can run in
parallel with S3.

---

## 5. Token cost per typical task (estimates)

**Today** = the minimum an agent must read with perfect navigation: auto-loaded `CLAUDE.md` + `MAP.md` (3.3k), the
zone map, and the relevant symbol ranges (paths below). It excludes the extra greps that W1 forces to trace implicit
names, so real sessions cost more. **After** = estimate once the steps in brackets are done.

| Task | Today: what is read | Today | After (steps) | After: what is read |
|---|---|---|---|---|
| Add a custom feature | `maps/data.md` (9.3k); `data/custom.py` L1-175, 177-456, 546-640; `data/normalize.py` L1-230; `data/defaults.py` L370-390; `data/features.py` L90-140; `RESEARCH_RULES.md`; card template | **≈30.7k** / 9 files | **≈15k** (S0, S5, S8, S10) | slim zone map with USES; `feature_registry.py` entry pattern; one feature module; rules + card |
| Add a loss / task type | `maps/trainer.md`; `trainer/tasks.py`; `trainer/config.py`; `workflow/training_config.py`; `main.ipynb` cell 18 (`main_config`); `MODEL_DEV_PROTOCOL.md`; card | **≈20.5k** / 9 files | **≈11k** (S6, S7, S9, S10) | zone map; one loss module + `trainer.TASKS` registration; `TrainerConfig` section; protocol + card |
| Change normalization | `maps/data.md`; `data/normalize.py` (9.5k); `data/windows.py` L300-380; `data/pipeline.py` L1-120; `data/selftests.py` slices; `tests/test_pct_change_norm.py`; `tests/test_multi_tf.py` L1-200; normalization audit doc | **≈35.9k** / 10 files (+ grep across the 19 files naming normalization symbols) | **≈14k** (S0, S8, S10) | zone map with USED BY (callers listed); registry kinds; transform module; run, don't read, the golden test |
| New H07 research study | 3 research maps; `edge_discovery/REPORT.md` L1-60; `16_h07_oos.py`; `tsbt.py`; `tools/h07_forward.py` L1-90; `h07_volsizing/01_volsizing.py` + README + card; `RESEARCH_RULES.md`; card template. Then **re-type the rule and the stats** | **≈20.7k** / 14 files (+ writing copies) | **≈9k** (S4, S10) | `rules/h07.py` + `stats/` signatures from the map; one prior study README; rules + card |
| Evaluate a trained model | `maps/tools.md` + `maps/workflow.md`; `tools/evaluate_trained_model.py` (4.8k); `main.ipynb` cells 3-24 (8.4k); `workflow/retarget.py` L147-260; `workflow/reports.py`; `workflow/chicks_bridge.py`; `evaluation/full_analysis.py` L1-120 | **≈33.8k** / 10 files | **≈13k** (S7, S10) | zone map; `workflow/settings.py` + `workflow/run.py:evaluate`; a much smaller CLI; one evaluation entry module |

PHILOSOPHY's target ("map + one card + target symbol") would be about 6–10k tokens per task. Today's tasks run at
3–6× that. After S0–S10 they come within about 1.5×.

Fixed costs worth knowing: all of `maps/` is about 74k tokens (`tags.md` 12.6k, `data.md` 9.3k).
`data/selftests.py` is 37.6k. `data/custom.py` is 22.1k. `signal_discovery_lab.ipynb` code alone is 35.8k (1,813 code lines
in a "thin" runner with 0 defs). The root `README.md` is 10.7k and `docs/research/README.md` 9.1k.

---

### Not measured / limits
- Runtime of the full pytest suite, and of `trainer.<name>` access (it trains a model; inferred from
  `trainer/_loader.py:10` and `trainer/__init__.py:46`, not executed).
- Whether the 6 `load_notebook_defs` legacy scripts still give the same results when run from the repo root: only
  checked that they fail from another working directory.
- Free-name analysis is static: names reached only via `globals()[...]` or `exec` (44 `globals()` uses in 12 package
  modules, 42 in `main.ipynb`) are under-counted. Token figures are heuristic (chars/2 for Arabic, chars/4 otherwise).

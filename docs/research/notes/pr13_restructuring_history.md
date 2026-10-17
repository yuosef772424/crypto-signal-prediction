# PR #13 restructuring: history facts (evil merges, move+modify commits, verbatim extraction)

Facts read from `git show --stat`, `git diff` and `git merge-tree` on branch `claude/quirky-knuth-rpt5lg`. History is not
rewritten. Range inspected: `6ca9fbb..3301036` (50 commits, merge base with master `6ca9fbb`). The first 11 commits of that range
(H07 studies, registry rows, `tools/h07_forward.py`, plus the registry-tool changes `3092f7f` and `498cd8b`) reached master through PR #19 and are not part of the restructuring; their
registry rows are `F-0078`/`F-0079` on master (they are named `F-0064`/`F-0065` inside the old commit messages of this branch,
which stay as written). After the master merge, the branch diff against master contains the restructuring only.

## 1. Merge commits that carry content changes ("evil merges")

Seven merges are in the range. Five are clean (`1f1ec86`, `3874027`, `1a83e77`, `1b1b14f`, `9e46d35`: no file differs from both
parents, `git diff-tree --cc` is empty). Two had textual conflicts that were resolved by combining both sides, and so carry
content beyond what either parent had:

| merge | parents | what the resolution contains (not present as such in either parent) |
|---|---|---|
| `ce95c1a` evaluation/ + signal_eval/ | `3874027` + `e6ef3c3` | 12 files differ from git's own auto-merge (68 insertions, 270 deletions, mostly conflict markers and regenerated maps). Real content: `CLAUDE.md`, `PHILOSOPHY.md`, `README.md`, `docs/research/README.md` list the packages of both sides (model, trainer, evaluation, signal_eval); `tools/check_deps.py` `CODE_PACKAGES` and `tools/build_map.py` `RATCHET_PREFIXES` take the union of both lists; `docs/research/audit/_nbload.py` keeps the loaders of both sides (`load_model`, `load_trainer`, `load_evaluation`, `exec_evaluation_module`) under one `_repo_package` docstring; `tests/test_nig_calibration.py` loads the model through `_nbload.load_model` instead of exec-ing `model_v2 (1).ipynb` cells (it drops `json`, and re-imports `_nbload` directly); `MAP.md`, `maps/*` regenerated. |
| `25de666` workflow/ + discovery/ | `ce95c1a` + `ad6881e` | 20 files differ from git's auto-merge (105 insertions, 481 deletions, mostly conflict markers and regenerated maps). Real content: the same package-list unions in `CLAUDE.md`, `PHILOSOPHY.md`, `README.md`, `tools/check_deps.py`, `tools/build_map.py` (now nine code packages); `docs/research/model_diagnostics.md` points to `trainer/`, `model/` and `workflow/` modules; five legacy scripts (`docs/research/scripts/causal_decomposition.py`, `corrected_retest.py`, `feature_screen.py`, `holdout_momentum_orth_natr.py`, `mom_orth_fidelity.py`) load the axis from `signal_eval` and the lab from `discovery` (the two sides had each replaced one of the two `exec(load_notebook_defs(...))` lines); `tests/test_entry_range.py` addresses main cells by name (`chicks_bridge`, `text:RETURN_PRICE_TARGETS`) and loads chicks cell 18 through `_nbload.exec_evaluation_module`; `tests/test_model_diagnostics.py`, `tests/test_multi_tf.py` and `tests/test_reg_target_scale.py` use `_nbload.load_model/load_trainer/load_evaluation` plus `workflow` modules; `docs/research/audit/_nbload.py` and `r3_00c_gather_alignment.py` combine both sides; maps regenerated. |

`670886f` (refactor: drop the duplicate `_repo_package` helper left by the package merges) and `633b52a` (fix: lab runner and
workflow tests load code from the packages after the merges) are the follow-ups that repaired leftovers of these two merges. The
merge of this branch with master (`da18c55`) carries only conflict resolution: H07 paths and `failure_registry.csv` taken from
master, maps regenerated.

## 2. Commits that both move and modify files

Git records no renames here: a "move" is the creation of new package modules from notebook cells plus the notebook being cut down
to a thin runner (`<package>.load_into(globals())`). Six `move:` commits exist. Each one also modifies files that are not the moved
notebook (new files excluded):

| commit | moved | other modified files in the same commit |
|---|---|---|
| `30d1fbc` | `crypto_data_pipeline_v6.ipynb` -> `data/` (30 new files) | the notebook (runner edit), `docs/research/audit/_nbload.py` (+18 -5), `tools/build_map.py`, `tools/check_deps.py`, `MAP.md`, `maps/*` |
| `af3db7a` | `chicks_v4_5_input_output_patterns.ipynb` -> `evaluation/` (25 new files) | the notebook (+42 -5881 lines), `docs/research/audit/_nbload.py` (+28), `tests/test_entry_range.py`, `tests/test_nig_calibration.py`, `tests/test_reg_target_scale.py`, `tools/build_map.py`, `tools/check_deps.py`, `MAP.md`, `maps/*` |
| `3f1f20f` | `main.ipynb` function definitions -> `workflow/` (21 new files) | the notebook (+53 -860), `tools/build_map.py`, `tools/check_deps.py`, `MAP.md`, `maps/*` |
| `24f1f59` | `model_v2 (1).ipynb` -> `model/`, `trainer_framework_v2.ipynb` -> `trainer/` (37 new files) | the two notebooks, `tools/build_map.py`, `tools/check_deps.py`, `MAP.md`, `maps/*` |
| `f9a6796` | `signal_evaluation_axis (3).ipynb` -> `signal_eval/` (11 new files) | the notebook, `MAP.md`, `maps/*` |
| `7e67b31` | `signal_discovery_lab.ipynb` + `pandas_ta_full_survey.ipynb` -> `discovery/` (12 new files) | both notebooks, five `docs/research/scripts/*.py` (one line each), `MAP.md`, `maps/*` |

The edits to `tools/build_map.py` / `tools/check_deps.py` add the new package name to `RATCHET_PREFIXES` / `CODE_PACKAGES` (one
line each); the `tests/` and `docs/research/` edits redirect the cell loaders to the new packages. Maps are regenerated files.
No other commit in the range creates modules from notebook cells; later `refactor:` commits (`1f80b0c` lifting nested defs into
`workflow/` and `discovery/`, `f9c50fa`/`5852328` RunSettings) modify but do not move.

## 3. The moves are verbatim extractions (measured)

For each `move:` commit, every non-blank, non-magic (`!`, `%`) line of the code cells of the notebook at the parent commit was looked
up (stripped, as a multiset) in the `.py` files of the target package(s) at the move commit.

| commit | old code lines | not found in the new modules | what the missing lines are |
|---|---|---|---|
| `af3db7a` evaluation | 4717 | 0 | none |
| `30d1fbc` data | 10293 | 143 (1.4%) | comment lines and cells (installation notes, a removed `load_asse()` note) kept in the runner notebook |
| `24f1f59` model + trainer | 4008 | 37 (0.9%) | the commented example cells that stay in `trainer_framework_v2.ipynb` |
| `f9a6796` signal_eval | 912 | 33 (3.6%) | the self-test run cell and two docstrings that stay in the runner notebook |
| `3f1f20f` workflow | 2942 | 598 (20%) | by design only function definitions moved; bootstrap, call and report cells stay in `main.ipynb` |
| `7e67b31` discovery | 2629 | 1904 (72%) | by design only top-level definitions moved; the lab's runner cells stay in the notebook and its nested defs were lifted later in `1f80b0c` (a `refactor:`, not a move) |

Logic of the moved code is unchanged; the behavioral check is `tests/golden_run_settings.json` (recorded from the cells before
`RunSettings` existed): with the branch at `fdbad0d`, all six scenarios of `tests/test_run_steps.py` reproduce it with a maximum
absolute and relative difference of 0.0 over 253-328 float fingerprint leaves each.

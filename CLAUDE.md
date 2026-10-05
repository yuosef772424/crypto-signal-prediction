# Working Instructions for This Project

## Map
@MAP.md

Navigate MAP.md -> maps/<zone>.md -> read only the target symbol by line range (never whole large files). Full rules are in PHILOSOPHY.md (Arabic): read it on demand, do not import it.

## Commands
- `python tools/check_deps.py` - import direction between zones (fails on a violation).
- `python tools/build_map.py` - regenerate MAP.md and maps/ after adding or renaming modules or cards.
- `python tools/build_map.py --check` - fails if the maps are stale.
- `git config core.hooksPath .githooks` - once per clone (Claude Code sessions do it on SessionStart): the pre-commit hook regenerates and stages the maps and runs the structure checks, so a commit can't leave them stale.
- `data/` holds ALL data-preparation code (formerly the cells of `crypto_data_pipeline_v6.ipynb`, now a thin runner): edit the module, not the notebook. The modules run in ONE shared namespace (`data/_loader.py`: `data.load_into(ns)`), never `import data.<module>`; tests get it via `docs/research/audit/_nbload.load_pipeline()`.
- `python -m pytest tests/ -q` - full suite, CPU only, no Drive (needs `pip install -r requirements-ci.txt`).

## Working rules
- Zones and import direction: code packages (`cross_asset/`, `tools/`, `data/`) never import `research/`, `docs/` or `tests/`; a study never imports another study (lift shared code into a package).
- Add, don't modify: a new study is `research/<study>/` with a README; recorded results are never edited (add a corrected copy, keep the original marked).
- A new option's default must reproduce the old behavior exactly (checkpoints and past results stay valid).
- No silent defaults: unknown config keys or unregistered feature/head/loss names raise; pass config sections explicitly, not via a global.
- One change type per commit (`merge`/`move`/`refactor`/`fix`/`feat`/`exp`/`docs`); never move and modify together; never commit on `master`.
- Merge commits carry no content changes; any reorganization after a merge is a separate commit.
- Regenerate the maps and run the checks above before committing.
- Every experiment (features, targets, rules, data, models) follows `docs/research/RESEARCH_RULES.md`: localize the failure → test its cause → fix → validate the fix. A failed experiment is closed with a row in `docs/research/failure_registry.csv` (scope tags, failure level/location, verified cause, invariants, `reopen_if`). A new card must pass `python tools/experiment_registry.py check <card>`: no re-running a closed failure unless a listed reopen condition now holds.
- Model/training experiments follow `docs/research/MODEL_DEV_PROTOCOL.md`: an experiment card before the run (one change, predicted diagnostic movement, accept rule), the per-layer model health report after it, and no "architecture X is better" claim without beating seed noise AND the predicted diagnostic moving.
- Any script used more than once goes into `tools/` (or a model notebook section) with a card and a test — call it in later sessions instead of rebuilding it (`tools/nb_cells.py`, `tools/fetch_crypto_dataset.py`).

## Language
- **Reply to the user in Arabic.** Instructions, agent prompts and protocol documents may be in English (models follow English instructions more faithfully); translate on demand.

## Subagents
- **Default:** delegate large, self-contained tasks to a background agent — e.g. building a tool or package, long experiments, or reading and analysing many files or large run logs.
- **Goal: protect the main conversation's context, not reduce total usage.** An agent starts with an empty context and returns only a short report, so the main conversation stays light and focused on the core line (model, training, decisions) and is compacted later, losing fewer details. Total tokens may go up, not down, because the agent re-reads what it needs.
- **An agent's prompt carries only what it needs:**
  - the goal and the definition of done;
  - paths and branches;
  - constraints;
  - relevant prior results, so it doesn't redo settled experiments;
  - the report format expected from it.
- **Do not delegate:**
  - small tasks: a one-line edit, a quick question, reading a single file;
  - work that depends on a step-by-step discussion with the user;
  - work that needs most of the conversation's context — explaining it would cost more than doing it.
- **Working with running agents:**
  - never duplicate the work of an agent that is still running;
  - send it new requirements by message instead of spawning another agent;
  - relay a summary of its report to the user, not the full text.
- **Model choice (cost):** spawn subagents with `model: "sonnet"` by default — tasks are delegated with a clear spec, so the cheaper model suffices. Use a stronger model only for open-ended research or design work, and say why when you do.
- **Auditor/builder rounds** follow `docs/research/audit/PROTOCOL.md`; the auditor's independence from the builder's reasoning is its most important rule.

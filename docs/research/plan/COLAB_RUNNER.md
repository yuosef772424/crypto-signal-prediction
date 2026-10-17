# Colab runner: one click per experiment

The owner opens `colab_runner.ipynb` in Colab, chooses **Runtime > Run all**, and never edits a cell. The notebook reads the queued
job from branch `colab-queue`, runs it, and pushes the result bundle to branch `colab-results` (and to Drive), so nobody copies
output by hand. All logic is in `tools/colab_runner.py` (tests: `tests/test_colab_runner*.py`); the notebook only calls it.

```
planner/builder ──push job.json──▶ colab-queue ──(Run all)──▶ Colab: clone repo_ref, run entry, validate
                                                                     │
        reviewer: git fetch origin colab-results ◀── runs/<job_id>/ ─┴─▶ Drive: MyDrive/crypto_runs/<job_id>/
```

## Owner setup (once)
1. GitHub > Settings > Developer settings > Fine-grained tokens > new: repository access **only this repo**, permission **Contents: read and write** (nothing else), expiry 90 days.
2. Open `https://colab.research.google.com/github/yuosef772424/crypto-signal-prediction/blob/master/colab_runner.ipynb` (or the branch the notebook is on), then
   **Runtime > Run all** once and allow the Drive prompt. When asked, **paste the token** (it is not echoed). The notebook checks it with one
   request to the repo, creates `MyDrive/crypto/secrets/` and saves the token there as one line (mode 600 where Drive allows it).
3. From then on Run all reads the token silently. If GitHub rejects the saved token (expired or revoked, HTTP 401), the notebook says so and asks
   again; the file is overwritten only after the new token is accepted. Read the last block printed (job id, OK/FAIL, results commit).

## Queueing a job (planner or builder)
One commit on `colab-queue`, which holds only `job.json`. The code to run must already be pushed, and `repo_ref` must be a commit
that contains `tools/colab_runner.py` (the notebook downloads the runner from that commit).
```
git fetch origin colab-queue && git worktree add ../queue origin/colab-queue --detach
# edit ../queue/job.json: new job_id, repo_ref = full 40-hex SHA, entry, args, config ...
python tools/colab_runner.py check-job ../queue/job.json --repo .      # schema + repo_ref + entry exist
git -C ../queue commit -am "queue: <job_id>" && git -C ../queue push origin HEAD:colab-queue
```
Never touch `master` from this branch. Then ask the owner to click Run all.

`job.json` (unknown keys raise; every key is required except `rerun` and `pip`):
```json
{"job_id": "smoke-001", "repo_ref": "<40-hex commit SHA>", "entry": "tools/colab_smoke.py", "args": ["--sleep", "0"],
 "config": {"rows": 5, "seed": 7}, "max_minutes": 5, "needs_drive": false, "expected_outputs": ["smoke.csv"],
 "note": "free text", "rerun": false, "pip": []}
```
- `entry`: repo-relative `.py` under `tools/` or `research/`. It is run as `python <entry> --config <tmp.json> --out <dir> --max-minutes N <args>`
  from the repo root; `args` may not contain `--config`, `--out`, `--max-minutes`. `config` is written to the temp file verbatim.
- `max_minutes` 1..180: passed to the script; the runner kills it after `max_minutes` + 2 minutes (FAIL).
- `needs_drive`: when true the job gets `CSP_DRIVE_ROOT=/content/drive/MyDrive` (Drive is always mounted for the token);
  the job's own code reads Drive, e.g. `from colab_bridge import prepare_data` (`PYTHONPATH` has `<repo>/tools` and `<repo>`; `CSP_REPO_DIR` is the checkout).
- `expected_outputs`: relative paths that must exist (and be listed in the manifest) or the run is FAIL.
- `pip`: extra plain requirement specs (`scipy==1.15.0`), installed after `requirements-ci.txt`.
- A `job_id` that already has `runs/<job_id>/` on `colab-results` (OK or FAIL) is refused; queue a new id, or set `"rerun": true` and
  remove it afterwards (otherwise every Run all repeats the job).

The entry script follows the run contract (ROADMAP §Common contracts): `manifest.json` with `git_sha` (= `repo_ref`), `config`
(contains the job config), `input_fingerprint` (64-hex), `outputs` {relpath: sha256}; each output ≤ 300 KB. `colab_runner.write_manifest()` writes it;
`tools/colab_smoke.py` is the example. Validation recomputes every sha256 and fails on unlisted files.

## Reading results
```
git fetch origin colab-results
git show origin/colab-results:runs/<job_id>/bundle_summary.md      # status, problems, manifest, first lines of every CSV
git show origin/colab-results:runs/<job_id>/env.json               # python/packages/GPU, start/end UTC, exit code, queue commit
git log origin/colab-results --format='%h %s'
```
`runs/<job_id>/` holds the outputs, `run_log.txt` (last 200 KB), `env.json`, `bundle_summary.md`. A failed run keeps only the log,
env and summary (status FAIL). The builder copies an OK bundle verbatim under `research/<study>/results/<run_id>/` as an `exp` commit;
the reviewer re-checks the sha256 values. Drive keeps the same folder at `MyDrive/crypto_runs/<job_id>/`.

## Security notes
- The token lives only in the Drive file `MyDrive/crypto/secrets/github_token.txt`. It reaches git through a `GIT_ASKPASS` script that reads it from the environment of the git child
  process; it is never in a URL, a command line, `.git/config`, the notebook source, or the job's environment, and every printed line
  and exception is masked. The notebook's one HTTP bootstrap sends it in an `Authorization` header.
- Fine-grained, one-repo, Contents-only: a leaked token can rewrite this repo's branches (including `master` unless it is protected)
  but nothing else. Revoke it in GitHub settings if in doubt; use a short expiry; protect `master`.
- **Whoever can push to `colab-queue` runs code in the owner's Colab session with Drive mounted**, including read access to the token file.
  `repo_ref` pins the exact commit, so queue only commits from trusted sessions and glance at the first printed line (job id, SHA, entry).
- Pushes are plain commits (never force) to `colab-results`; it is separate from `master` and holds only small bundles. If the repo is
  public, results and logs are public: no raw data, no secrets in scripts.
- After a run, disconnect the runtime (Runtime > Disconnect and delete runtime) to drop the in-memory token.

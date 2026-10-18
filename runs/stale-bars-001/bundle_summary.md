# Bundle summary: stale-bars-001

- status: **FAIL**
- repo_ref: `de20d964c25b23b9b7ade30b5e91291efb7a0ee7`
- entry: `tools/stale_bars_colab_job.py `
- note: Stale/zero-target audit (claude/zero-target-fix de20d96): zero share per split OFF/ON, dead coins.
- exit code: None   duration: 46.5 s
- problems: 
  - validate: manifest.json missing: the entry script did not write one (run contract)

## last log lines
```
# colab_runner log for stale-bars-001 started 2026-10-18T00:51:26Z
== checkout ok: de20d964c25b23b9b7ade30b5e91291efb7a0ee7
== install -q --disable-pip-version-check -r requirements-ci.txt
== run: /usr/bin/python3 tools/stale_bars_colab_job.py --config /content/csp_runner/tmp/stale-bars-001.config.json --out /content/csp_runner/out/stale-bars-001 --max-minutes 40
```

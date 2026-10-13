"""
PURPOSE:  Logic behind the one-click Colab runner (colab_runner.ipynb): validate a queued job (job.json on branch
          colab-queue), check out exactly its repo_ref, run its entry script under a wall-clock guard, validate the run-contract
          outputs (manifest.json, sha256 recomputed, size limits), build the result bundle (runs/<job_id>/), push it to branch
          colab-results with a plain git commit and copy it to Drive. Stdlib only: the notebook downloads this one file first.
TAGS:     colab, runner, job queue, colab-queue, colab-results, job.json, manifest, run contract, sha256, bundle, token,
          GIT_ASKPASS, push results, one click, reproduction check
PITFALLS: The token is only ever put in the environment of git child processes (GIT_ASKPASS script reads it); it is never in a
          URL, an argv, .git/config or the job's environment, and every printed string / exception goes through mask().
          Job code still runs in the same VM with Drive mounted, so it could read the token file: queue only SHAs you trust.
          repo_ref must be a commit that already contains tools/colab_runner.py (the notebook downloads the runner from it).
          A finished job_id (runs/<job_id>/ on colab-results, OK or FAIL) is refused unless the job sets "rerun": true.

Usage (planner / builder, before queueing):
    python tools/colab_runner.py check-job job.json [--repo .]     # schema check + repo_ref exists locally

job.json schema (all keys required except rerun, pip):
    {"job_id": "h3-smoke-001", "repo_ref": "<40-hex commit SHA>", "entry": "tools/colab_smoke.py", "args": ["--sleep", "0"],
     "config": {"rows": 5, "seed": 7}, "max_minutes": 5, "needs_drive": false, "expected_outputs": ["smoke.csv"],
     "note": "free text", "rerun": false, "pip": ["scipy==1.15.0"]}
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

# --------------------------------------------------------------------------------------------------- constants
QUEUE_BRANCH = "colab-queue"
RESULTS_BRANCH = "colab-results"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive"
DEFAULT_DRIVE_RUNS = DEFAULT_DRIVE_ROOT + "/crypto_runs"
MAX_OUTPUT_BYTES = 300 * 1024          # per output file (run contract)
MAX_LOG_BYTES = 200 * 1024             # tail kept in run_log.txt
MAX_MINUTES_CAP = 180
RESERVED_NAMES = ("run_log.txt", "env.json", "bundle_summary.md")
RESERVED_FLAGS = ("--config", "--out", "--max-minutes")
MANIFEST_REQUIRED = ("git_sha", "config", "input_fingerprint", "outputs")
JOB_REQUIRED = ("job_id", "repo_ref", "entry", "args", "config", "max_minutes", "needs_drive", "expected_outputs", "note")
JOB_OPTIONAL = ("rerun", "pip")
ENV_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "tensorflow", "torch", "pyarrow", "matplotlib",
                "pandas-ta-classic")
CSV_HEAD_LINES = 12
SECRET_ENV_KEYS = ("GIT_ASKPASS", "COLAB_RUNNER_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")

_JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}")
_SHA40_RE = re.compile(r"[0-9a-fA-F]{40}")
_SHA64_RE = re.compile(r"[0-9a-f]{64}")
_ENTRY_RE = re.compile(r"(tools|research)(/[A-Za-z0-9_.-]+)+\.py")
_PIP_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?((==|~=|>=|<=|!=|<|>)[A-Za-z0-9.*+!_-]+)?")


class RunnerError(Exception):
    """Anything the runner refuses or cannot do; the message is already safe to print."""


class JobError(RunnerError):
    """job.json is not a valid job."""


# ------------------------------------------------------------------------------------------------ secrets / masking
def _variants(secret):
    s = secret
    forms = {s, urllib.parse.quote(s), urllib.parse.quote(s, safe=""), urllib.parse.quote_plus(s)}
    for raw in (s, "x-access-token:" + s):
        b = base64.b64encode(raw.encode()).decode()
        forms.update({b, b.rstrip("=")})
    return [f for f in forms if f]


def mask(text, secrets):
    """Replace every form of every secret (plain, url-encoded, basic-auth base64) with ***. Secrets under 6 chars are ignored."""
    values = secrets._values if isinstance(secrets, Secrets) else list(secrets or [])
    forms = set()
    for v in values:
        if isinstance(v, str) and len(v) >= 6:
            forms.update(_variants(v))
    out = str(text)
    for f in sorted(forms, key=len, reverse=True):
        out = out.replace(f, "***")
    return out


class Secrets:
    def __init__(self, *values):
        self._values = []
        for v in values:
            self.add(v)

    def add(self, value):
        if isinstance(value, str) and len(value) >= 6 and value not in self._values:
            self._values.append(value)

    def mask(self, text):
        return mask(text, self)


class guard:
    """`with guard(secrets): ...` turns any Exception into a RunnerError whose text is masked and whose chain is hidden."""

    def __init__(self, secrets):
        self.secrets = secrets

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if ev is None or not isinstance(ev, Exception):
            return False
        try:                                         # scrub the original too: it stays reachable as __context__
            ev.args = tuple(mask(a, self.secrets) if isinstance(a, str) else a for a in ev.args)
        except Exception:                            # noqa: BLE001
            pass
        text = mask(f"{type(ev).__name__}: {ev}" if not isinstance(ev, RunnerError) else str(ev), self.secrets)
        err = (type(ev) if isinstance(ev, RunnerError) else RunnerError)(text)
        err.__suppress_context__ = True
        raise err


def read_token(path):
    """Read the one-line token file. Errors never echo the file content."""
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as e:
        raise RunnerError(f"token file not readable at {path} ({type(e).__name__}); see docs/research/plan/COLAB_RUNNER.md") from None
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise RunnerError(f"token file must hold exactly one non-empty line (found {len(lines)})")
    token = lines[0].strip()
    if len(token) < 8 or any(c.isspace() for c in token):
        raise RunnerError("token file content does not look like a token (too short or contains whitespace)")
    return token


# ---------------------------------------------------------------------------------------------------- job schema
def _reject_duplicates(pairs):
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise JobError(f"job.json has duplicate key {k!r}")
        seen[k] = v
    return seen


def parse_job(text):
    try:
        obj = json.loads(text, object_pairs_hook=_reject_duplicates)
    except JobError:
        raise
    except ValueError as e:
        raise JobError(f"job.json is not valid JSON ({e})") from None
    return validate_job(obj)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _safe_rel(path):
    """A relative posix path with no empty, '.' or '..' segment and no backslash."""
    if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path or "\x00" in path or "\n" in path:
        return False
    return all(seg not in ("", ".", "..") for seg in path.split("/"))


def validate_job(obj):
    """Return a normalised copy of the job or raise JobError. Unknown keys raise; nothing is defaulted except rerun/pip."""
    if not isinstance(obj, dict):
        raise JobError("job.json must be a JSON object")
    unknown = sorted(set(obj) - set(JOB_REQUIRED) - set(JOB_OPTIONAL))
    if unknown:
        raise JobError(f"unknown job key(s): {unknown}")
    for key in JOB_REQUIRED:
        if key not in obj:
            raise JobError(f"missing key {key!r}")
    job = dict(obj)

    jid = job["job_id"]
    if not isinstance(jid, str) or not _JOB_ID_RE.fullmatch(jid) or ".." in jid:
        raise JobError("job_id: 3-64 chars of letters, digits, '.', '_', '-', starting with a letter or digit")
    ref = job["repo_ref"]
    if not isinstance(ref, str) or not _SHA40_RE.fullmatch(ref):
        raise JobError("repo_ref: must be a full 40-hex commit SHA (branch names and short SHAs are refused)")
    job["repo_ref"] = ref.lower()
    entry = job["entry"]
    if not isinstance(entry, str) or not _ENTRY_RE.fullmatch(entry) or not _safe_rel(entry):
        raise JobError("entry: must be a repo-relative .py path under tools/ or research/ (no '..', no absolute path)")
    args = job["args"]
    if not isinstance(args, list) or not all(isinstance(a, str) and "\n" not in a and "\x00" not in a for a in args):
        raise JobError("args: must be a list of single-line strings")
    for a in args:
        if a in RESERVED_FLAGS or a.startswith(tuple(f + "=" for f in RESERVED_FLAGS)):
            raise JobError(f"args: {a!r} is set by the runner and cannot be passed")
    cfg = job["config"]
    if not isinstance(cfg, dict):
        raise JobError("config: must be a JSON object")
    try:
        if len(json.dumps(cfg, allow_nan=False)) > 65536:
            raise JobError("config: larger than 64 KB")
    except ValueError:
        raise JobError("config: must be plain JSON (no NaN / Infinity)") from None
    mm = job["max_minutes"]
    if not _is_int(mm) or not 1 <= mm <= MAX_MINUTES_CAP:
        raise JobError(f"max_minutes: must be an integer in 1..{MAX_MINUTES_CAP}")
    if not isinstance(job["needs_drive"], bool):
        raise JobError("needs_drive: must be true or false")
    exp = job["expected_outputs"]
    if not isinstance(exp, list) or not all(_safe_rel(e) for e in exp) or len(set(exp)) != len(exp):
        raise JobError("expected_outputs: must be a list of unique relative paths (no '..', no absolute path)")
    if not isinstance(job["note"], str) or len(job["note"]) > 2000:
        raise JobError("note: must be a string of at most 2000 chars")
    job.setdefault("rerun", False)
    if not isinstance(job["rerun"], bool):
        raise JobError("rerun: must be true or false")
    job.setdefault("pip", [])
    if not isinstance(job["pip"], list) or not all(isinstance(p, str) and _PIP_RE.fullmatch(p) for p in job["pip"]):
        raise JobError("pip: must be a list of plain requirement specs like 'scipy==1.15.0' (no options, urls or extras syntax)")
    return job


# ------------------------------------------------------------------------------------------------ hashes / manifest
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_fingerprint(root, relpaths):
    """sha256 of the sorted '<relpath>\\t<size>' lines of the input files (run contract; empty list is allowed)."""
    lines = sorted(f"{str(p).replace(os.sep, '/')}\t{os.path.getsize(os.path.join(root, p))}" for p in relpaths)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def git_head(repo_dir):
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RunnerError(f"cannot read git HEAD in {repo_dir}")
    return r.stdout.strip()


def _walk_files(root):
    for base, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files + [d for d in dirs if os.path.islink(os.path.join(base, d))]):
            full = os.path.join(base, name)
            yield os.path.relpath(full, root).replace(os.sep, "/"), full


def write_manifest(out_dir, config, inputs=(), input_root="", row_counts=None, runtime_seconds=0.0, n_configurations=1,
                   repo_dir=None, extra=None):
    """Write <out_dir>/manifest.json per the run contract; every other file under out_dir becomes a hashed output."""
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs = {rel: sha256_file(full) for rel, full in _walk_files(out_dir) if rel != "manifest.json"}
    manifest = {"git_sha": git_head(repo_dir), "config": config, "input_fingerprint": input_fingerprint(input_root, inputs),
                "row_counts": row_counts or {}, "runtime_seconds": round(float(runtime_seconds), 3),
                "n_configurations": int(n_configurations), "outputs": outputs}
    if extra:
        manifest.update(extra)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return manifest


def validate_outputs(out_dir, job, repo_sha):
    """Run-contract check of <out_dir>. Returns a list of problems (empty = valid)."""
    if not os.path.isdir(out_dir):
        return [f"output directory missing: {out_dir}"]
    mpath = os.path.join(out_dir, "manifest.json")
    if not os.path.isfile(mpath):
        return ["manifest.json missing: the entry script did not write one (run contract)"]
    try:
        with open(mpath, encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as e:
        return [f"manifest.json is not valid JSON ({type(e).__name__})"]
    if not isinstance(manifest, dict):
        return ["manifest.json must be a JSON object"]
    problems = []
    for key in MANIFEST_REQUIRED:
        if key not in manifest:
            problems.append(f"manifest.json lacks required key {key!r}")
    if "git_sha" in manifest and manifest["git_sha"] != repo_sha:
        problems.append(f"manifest git_sha {manifest['git_sha']!r} != checked-out commit {repo_sha}")
    if "config" in manifest:
        mc = manifest["config"]
        if not isinstance(mc, dict):
            problems.append("manifest config must be an object")
        else:
            for k, v in job["config"].items():
                if k not in mc or mc[k] != v:
                    problems.append(f"manifest config differs from the job config at key {k!r}")
    if "input_fingerprint" in manifest and not (isinstance(manifest["input_fingerprint"], str)
                                                and _SHA64_RE.fullmatch(manifest["input_fingerprint"])):
        problems.append("manifest input_fingerprint must be a 64-hex sha256")
    outputs = manifest.get("outputs")
    listed = {}
    if "outputs" in manifest and not isinstance(outputs, dict):
        problems.append("manifest outputs must be an object {relpath: sha256}")
    elif isinstance(outputs, dict):
        real_out = os.path.realpath(out_dir)
        for rel, digest in sorted(outputs.items()):
            if not _safe_rel(rel):
                problems.append(f"output path {rel!r} escapes the output directory")
                continue
            if rel == "manifest.json":
                problems.append("manifest.json cannot list itself as an output")
                continue
            if rel in RESERVED_NAMES:
                problems.append(f"output name {rel!r} is reserved by the runner")
                continue
            full = os.path.join(out_dir, rel)
            if not os.path.realpath(full).startswith(real_out + os.sep):
                problems.append(f"output path {rel!r} escapes the output directory")
                continue
            if os.path.islink(full) or not os.path.isfile(full):
                problems.append(f"output {rel!r} is missing (listed in the manifest)")
                continue
            listed[rel] = digest
            size = os.path.getsize(full)
            if size > MAX_OUTPUT_BYTES:
                problems.append(f"output {rel!r} is {size} bytes (> {MAX_OUTPUT_BYTES // 1024} KB limit)")
            actual = sha256_file(full)
            if actual != digest:
                problems.append(f"output {rel!r}: sha256 mismatch (manifest {str(digest)[:12]}, recomputed {actual[:12]})")
    for rel, full in _walk_files(out_dir):
        if rel == "manifest.json":
            continue
        if os.path.islink(full):
            problems.append(f"{rel!r} is a symlink")
        elif rel not in listed and rel not in (outputs or {}):
            problems.append(f"{rel!r} is not listed in the manifest outputs")
    for rel in job["expected_outputs"]:
        if not os.path.isfile(os.path.join(out_dir, rel)):
            problems.append(f"expected output {rel!r} is missing")
        elif rel != "manifest.json" and isinstance(outputs, dict) and rel not in outputs:
            problems.append(f"expected output {rel!r} is not listed in the manifest")
    return problems


# --------------------------------------------------------------------------------------------- log / bundle / env
def tail_text(path, max_bytes):
    """The last <= max_bytes of a text file, cut at a line boundary, with a first line saying how much was dropped."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if size <= max_bytes:
            return fh.read().decode("utf-8", "replace")
        marker_len = len(f"[... log truncated: first {size} bytes dropped ...]\n".encode())
        fh.seek(size - (max_bytes - marker_len))
        data = fh.read()
    cut = data.find(b"\n")
    data = data[cut + 1:] if cut >= 0 else data
    marker = f"[... log truncated: first {size - len(data)} bytes dropped ...]\n"
    return marker + data.decode("utf-8", "replace")


def pip_versions(python, names):
    code = ("import importlib.metadata as m, json, sys\nout = {}\nfor n in sys.argv[1:]:\n    try:\n"
            "        out[n] = m.version(n)\n    except m.PackageNotFoundError:\n        out[n] = None\nprint(json.dumps(out))\n")
    try:
        r = subprocess.run([python, "-c", code, *names], capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return {n: None for n in names}


def gpu_info():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "none"
    except (OSError, subprocess.SubprocessError):
        return "none"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _csv_head(path, n=CSV_HEAD_LINES):
    lines, total = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            if len(lines) < n:
                lines.append(line.rstrip("\n")[:300])
    return lines, total


def build_bundle(dest, job, out_dir, log_path, env, status, problems, secrets=None):
    """Create <dest> (= runs/<job_id>/): outputs (OK only), run_log.txt, env.json, bundle_summary.md. Returns relative names."""
    secrets = secrets or Secrets()
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    names = []

    def put(rel, text):
        full = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(secrets.mask(text))
        names.append(rel)

    csvs = []
    if status == "OK":
        for rel, full in _walk_files(out_dir):
            if os.path.islink(full):
                continue
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(full, target)
            names.append(rel)
            if rel.lower().endswith(".csv"):
                csvs.append(rel)
    log_text = tail_text(log_path, MAX_LOG_BYTES) if os.path.isfile(log_path) else "(no log was written)\n"
    put("run_log.txt", log_text)
    put("env.json", json.dumps({"job_id": job["job_id"], "repo_ref": job["repo_ref"], **env, "status": status}, indent=1,
                               sort_keys=True) + "\n")
    summary = [f"# Bundle summary: {job['job_id']}", "", f"- status: **{status}**", f"- repo_ref: `{job['repo_ref']}`"]
    if "entry" in job:
        summary.append(f"- entry: `{job['entry']} {' '.join(job.get('args', []))}`".rstrip())
    if job.get("note"):
        summary.append(f"- note: {job['note'][:300]}")
    summary.append(f"- exit code: {env.get('exit_code')}   duration: {env.get('duration_s')} s")
    summary += ["- problems: " + ("none" if not problems else "")]
    summary += [f"  - {p}" for p in problems]
    if status == "OK":
        manifest_path = os.path.join(dest, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as fh:
                summary += ["", "## manifest.json", "```json", fh.read()[:20000].rstrip(), "```"]
        for rel in sorted(csvs):
            head, total = _csv_head(os.path.join(dest, rel))
            summary += ["", f"## {rel} (first {len(head)} of {total} lines)", "```", *head, "```"]
    else:
        tail = log_text.splitlines()[-40:]
        summary += ["", "## last log lines", "```", *[ln[:300] for ln in tail], "```"]
    put("bundle_summary.md", "\n".join(summary)[:150000] + "\n")
    return names


# --------------------------------------------------------------------------------------------------------- git
def github_url(repo):
    return f"https://github.com/{repo}.git"


def git_env(token, scratch_dir):
    """Environment for git child processes: GIT_ASKPASS script that reads the token from the environment."""
    os.makedirs(scratch_dir, exist_ok=True)
    script = os.path.join(scratch_dir, "askpass.sh")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write('#!/bin/sh\ncase "$1" in\n  Username*) printf \'%s\\n\' "x-access-token" ;;\n'
                 '  *) printf \'%s\\n\' "$COLAB_RUNNER_TOKEN" ;;\nesac\n')
    os.chmod(script, 0o700)
    env = dict(os.environ)
    env.update({"GIT_ASKPASS": script, "GIT_TERMINAL_PROMPT": "0", "COLAB_RUNNER_TOKEN": token, "GCM_INTERACTIVE": "never"})
    return env


def _git_label(args):
    i = 0
    while i < len(args) and args[i] == "-c":
        i += 2
    return args[i] if i < len(args) else "git"


def run_git(args, cwd=None, env=None, secrets=None, timeout=900):
    env = env if env is not None else dict(os.environ, GIT_TERMINAL_PROMPT="0")
    label = _git_label(args)
    try:
        r = subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RunnerError(f"git {label} timed out after {timeout} s") from None
    except OSError as e:
        raise RunnerError(f"git is not usable ({type(e).__name__})") from None
    if r.returncode != 0:
        tail = mask((r.stderr or r.stdout or "").strip()[-600:], secrets)
        raise RunnerError(f"git {label} failed (exit {r.returncode}): {tail}")
    return r.stdout.strip()


def clone_at(repo_url, sha, dest, env=None, secrets=None):
    """Clone and check out exactly <sha> (detached). Refuses anything that is not a full 40-hex SHA."""
    if not isinstance(sha, str) or not _SHA40_RE.fullmatch(sha):
        raise RunnerError("repo_ref must be a full 40-hex commit SHA")
    sha = sha.lower()
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    run_git(["clone", "--quiet", "--no-checkout", repo_url, dest], env=env, secrets=secrets)
    try:
        run_git(["checkout", "--quiet", "--detach", sha], cwd=dest, env=env, secrets=secrets)
    except RunnerError:
        try:                                          # not reachable from a branch: ask the server for it directly
            run_git(["fetch", "--quiet", "origin", sha], cwd=dest, env=env, secrets=secrets)
            run_git(["checkout", "--quiet", "--detach", sha], cwd=dest, env=env, secrets=secrets)
        except RunnerError:
            raise RunnerError(f"checkout of {sha} failed: commit not found in the repository") from None
    head = run_git(["rev-parse", "HEAD"], cwd=dest, env=env, secrets=secrets)
    if head != sha:
        raise RunnerError(f"checkout landed on {head}, expected {sha}")
    return dest


def check_entry(repo_dir, entry):
    path = os.path.join(repo_dir, entry)
    if not os.path.isfile(path):
        raise RunnerError(f"entry {entry} not found at the checked-out commit")
    real = os.path.realpath(path)
    allowed = [os.path.realpath(os.path.join(repo_dir, d)) + os.sep for d in ("tools", "research")]
    if not any(real.startswith(a) for a in allowed):
        raise RunnerError(f"entry {entry} resolves outside tools/ and research/")


def clone_branch(repo_url, branch, dest, env=None, secrets=None):
    """Shallow clone of one branch. Returns False (and clones nothing) when the branch does not exist on the remote."""
    r = subprocess.run(["git", "ls-remote", "--exit-code", "--heads", repo_url, branch], env=env or dict(os.environ),
                       capture_output=True, text=True, errors="replace", timeout=300)
    if r.returncode == 2:
        return False
    if r.returncode != 0:
        raise RunnerError(f"git ls-remote failed (exit {r.returncode}): {mask(r.stderr.strip()[-400:], secrets)}")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    run_git(["clone", "--quiet", "--depth", "1", "--branch", branch, repo_url, dest], env=env, secrets=secrets)
    return True


def results_has_job(repo_url, branch, job_id, workdir, env=None, secrets=None):
    dest = os.path.join(workdir, "results_probe")
    try:
        if not clone_branch(repo_url, branch, dest, env=env, secrets=secrets):
            return False
        return os.path.isfile(os.path.join(dest, "runs", job_id, "env.json"))
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def push_results(repo_url, branch, bundle_dir, job, status, workdir, env=None, secrets=None, attempts=4):
    """Commit <bundle_dir> as runs/<job_id>/ on <branch> (orphan when missing) and push without force; returns the commit SHA.
    A rejected push (someone else pushed meanwhile) is retried from a fresh clone, only the new folder is re-applied."""
    dest = os.path.join(workdir, "results_repo")
    last = None
    for attempt in range(attempts):
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        if not clone_branch(repo_url, branch, dest, env=env, secrets=secrets):
            os.makedirs(dest)
            run_git(["init", "--quiet"], cwd=dest, env=env, secrets=secrets)
            run_git(["symbolic-ref", "HEAD", f"refs/heads/{branch}"], cwd=dest, env=env, secrets=secrets)
            run_git(["remote", "add", "origin", repo_url], cwd=dest, env=env, secrets=secrets)
        target = os.path.join(dest, "runs", job["job_id"])
        if os.path.isdir(target):
            shutil.rmtree(target)
        shutil.copytree(bundle_dir, target)
        run_git(["add", "-A"], cwd=dest, env=env, secrets=secrets)
        message = (f"colab-results: {job['job_id']} {status}\n\njob_id: {job['job_id']}\nrepo_ref: {job['repo_ref']}\n"
                   f"status: {status}\n")
        run_git(["-c", "user.name=colab-runner", "-c", "user.email=colab-runner@users.noreply.github.com",
                 "-c", "commit.gpgsign=false", "commit", "--quiet", "--allow-empty", "-m", message], cwd=dest, env=env, secrets=secrets)
        try:
            run_git(["push", "--quiet", "origin", f"HEAD:refs/heads/{branch}"], cwd=dest, env=env, secrets=secrets)
        except RunnerError as e:
            last = e
            if attempt + 1 < attempts:
                time.sleep(attempt + 1)
                continue
            raise RunnerError(f"push to {branch} failed after {attempts} attempts: {e}") from None
        return run_git(["rev-parse", "HEAD"], cwd=dest, env=env, secrets=secrets)
    raise last  # pragma: no cover


# --------------------------------------------------------------------------------------------- running a process
def run_logged(cmd, cwd, env, log_path, wall_seconds, secrets=None, echo=print, echo_lines=400, append=False):
    """Run cmd, stream masked output to log_path (and echo the first echo_lines lines); SIGKILL the process group after
    wall_seconds. Returns (exit_code, timed_out)."""
    state = {"timed_out": False}
    with open(log_path, "ab" if append else "wb") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

        def kill():
            state["timed_out"] = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                proc.kill()

        timer = threading.Timer(wall_seconds, kill)
        timer.daemon = True
        timer.start()
        shown = 0
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = mask(raw.decode("utf-8", "replace"), secrets)
                log.write(line.encode("utf-8"))
                log.flush()
                if shown < echo_lines:
                    echo(line.rstrip("\n"))
                    shown += 1
                    if shown == echo_lines:
                        echo(f"... (echo stopped after {echo_lines} lines; the full log is in run_log.txt)")
        finally:
            timer.cancel()
            proc.stdout.close()
            code = proc.wait()
    return code, state["timed_out"]


def _default_pip(cmd, cwd, env, log_path, timeout, secrets, echo):
    code, timed_out = run_logged(cmd, cwd=cwd, env=env, log_path=log_path, wall_seconds=timeout, secrets=secrets, echo=echo,
                                 append=True)
    return 124 if timed_out else code


# ------------------------------------------------------------------------------------------------------ Runner
class Runner:
    """One job, one Runner. Steps never raise: a failure is recorded and later steps are skipped, so publish() can always
    push the log. Order: load, check_rerun, checkout, install, execute, publish."""

    def __init__(self, repo_url, token, workdir, queue_branch=QUEUE_BRANCH, results_branch=RESULTS_BRANCH,
                 drive_runs=DEFAULT_DRIVE_RUNS, drive_root=DEFAULT_DRIVE_ROOT, pip=None, echo=print, python=None,
                 grace_seconds=120, install_timeout=1800):
        if not isinstance(token, str) or len(token) < 8:
            raise RunnerError("a GitHub token is required")
        self.repo_url, self.workdir = repo_url, os.path.abspath(workdir)
        self.queue_branch, self.results_branch = queue_branch, results_branch
        self.drive_runs, self.drive_root = drive_runs, drive_root
        self.secrets = Secrets(token)
        self._scratch = tempfile.mkdtemp(prefix="colab-runner-")
        self.git = git_env(token, self._scratch)
        self.pip = pip or (lambda cmd, **k: _default_pip(cmd, secrets=self.secrets, echo=self._say, **k))
        self.echo, self.python = echo, python or sys.executable
        self.grace_seconds, self.install_timeout = grace_seconds, install_timeout
        self.job = self.stub = None
        self.problems, self.refused, self.phase = [], False, "init"
        self.queue_sha, self.repo_dir, self.out_dir, self.log_path = "", None, None, None
        self.exit_code, self.timed_out, self.run_seconds = None, False, None
        self.t_start = utc_now()
        self._rerun_checked = False
        os.makedirs(self.workdir, exist_ok=True)

    # -- helpers
    def _say(self, text):
        self.echo(self.secrets.mask(text))

    def _log(self, text):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(self.secrets.mask(text) + "\n")

    def _stop(self):
        return bool(self.problems) or self.refused

    def _step(self, phase, fn):
        if self._stop() or self.job is None:
            return None
        self.phase = phase
        try:
            with guard(self.secrets):
                return fn()
        except RunnerError as e:
            self.problems.append(f"{phase}: {e}")
            self._log(f"== {phase} FAILED: {e}")
            self._say(f"[{phase}] FAILED: {e}")
            return None

    def _job_env(self):
        env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV_KEYS}
        tools = os.path.join(self.repo_dir, "tools")
        env.update({"PYTHONUNBUFFERED": "1", "CSP_REPO_DIR": self.repo_dir, "CSP_JOB_ID": self.job["job_id"],
                    "PYTHONPATH": os.pathsep.join([tools, self.repo_dir] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))})
        if self.job["needs_drive"]:
            env["CSP_DRIVE_ROOT"] = self.drive_root
        return env

    def _open_log(self, job_id):
        os.makedirs(os.path.join(self.workdir, "logs"), exist_ok=True)
        self.log_path = os.path.join(self.workdir, "logs", f"{job_id}.log")
        with open(self.log_path, "w", encoding="utf-8") as fh:
            fh.write(f"# colab_runner log for {job_id} started {_iso(self.t_start)}\n")

    # -- steps
    def load(self, job_text, queue_sha=""):
        self.queue_sha = queue_sha
        try:
            self.job = parse_job(job_text)
            self._open_log(self.job["job_id"])
            self._say(f"job {self.job['job_id']} @ {self.job['repo_ref'][:12]}: {self.job['entry']}")
        except JobError as e:
            self.problems.append(f"load: {e}")
            self._say(f"[load] job rejected: {e}")
            try:
                raw = json.loads(job_text)
                jid, ref = raw.get("job_id"), raw.get("repo_ref")
                if isinstance(jid, str) and _JOB_ID_RE.fullmatch(jid) and ".." not in jid:
                    self.stub = {"job_id": jid, "repo_ref": ref.lower() if isinstance(ref, str) and _SHA40_RE.fullmatch(ref)
                                 else "n/a", "rerun": raw.get("rerun") is True, "note": "", "args": [], "entry": str(raw.get("entry"))[:200]}
                    self._open_log(jid)
                    self._log(f"== load FAILED: {e}")
            except (ValueError, AttributeError):
                pass
        return self.job

    def _finished(self, job_id):
        return results_has_job(self.repo_url, self.results_branch, job_id, self.workdir, env=self.git, secrets=self.secrets)

    def check_rerun(self):
        def go():
            self._rerun_checked = True
            if not self.job["rerun"] and self._finished(self.job["job_id"]):
                self._refuse(self.job["job_id"])
        self._step("check_rerun", go)

    def _refuse(self, job_id):
        self.refused = True
        self.problems.append(f"job {job_id} is already finished (runs/{job_id}/ exists on {self.results_branch}); "
                             f"queue a new job_id or set \"rerun\": true")
        self._say(f"[refused] {self.problems[-1]}")

    def checkout(self):
        def go():
            self.repo_dir = os.path.join(self.workdir, "repo")
            clone_at(self.repo_url, self.job["repo_ref"], self.repo_dir, env=self.git, secrets=self.secrets)
            check_entry(self.repo_dir, self.job["entry"])
            self._log(f"== checkout ok: {self.job['repo_ref']}")
            self._say("checked out " + self.job["repo_ref"])
        self._step("checkout", go)

    def install(self):
        def go():
            env = {k: v for k, v in os.environ.items() if k not in SECRET_ENV_KEYS}
            base = [self.python, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
            cmds = []
            if os.path.isfile(os.path.join(self.repo_dir, "requirements-ci.txt")):
                cmds.append(base + ["-r", "requirements-ci.txt"])
            else:
                self._log("== requirements-ci.txt not present at this commit: skipped")
            if self.job["pip"]:
                cmds.append(base + list(self.job["pip"]))
            for cmd in cmds:
                self._log("== " + " ".join(cmd[3:]))
                code = self.pip(cmd, cwd=self.repo_dir, env=env, log_path=self.log_path, timeout=self.install_timeout)
                if code != 0:
                    raise RunnerError(f"pip install failed (exit {code})")
        self._step("install", go)

    def execute(self):
        def go():
            job = self.job
            if job["needs_drive"] and not os.path.isdir(self.drive_root):
                raise RunnerError(f"needs_drive is true but Drive is not mounted at {self.drive_root}")
            tmp = os.path.join(self.workdir, "tmp")
            os.makedirs(tmp, exist_ok=True)
            cfg = os.path.join(tmp, f"{job['job_id']}.config.json")
            with open(cfg, "w", encoding="utf-8") as fh:
                json.dump(job["config"], fh)
            self.out_dir = os.path.join(self.workdir, "out", job["job_id"])
            shutil.rmtree(self.out_dir, ignore_errors=True)
            os.makedirs(self.out_dir)
            cmd = [self.python, job["entry"], "--config", cfg, "--out", self.out_dir, "--max-minutes", str(job["max_minutes"]),
                   *job["args"]]
            self._log("== run: " + " ".join(cmd))
            self._say("running: " + " ".join(cmd[1:]))
            t0 = time.time()
            self.exit_code, self.timed_out = run_logged(
                cmd, cwd=self.repo_dir, env=self._job_env(), log_path=self.log_path, append=True, secrets=self.secrets,
                wall_seconds=job["max_minutes"] * 60 + self.grace_seconds, echo=self._say)
            self.run_seconds = round(time.time() - t0, 1)
            if self.timed_out:
                raise RunnerError(f"killed by the wall-clock guard after {self.run_seconds} s (max_minutes {job['max_minutes']} "
                                  f"+ {self.grace_seconds} s grace): the script did not stop at --max-minutes")
            if self.exit_code != 0:
                raise RunnerError(f"exit code {self.exit_code}")
        self._step("execute", go)

    def _env_info(self, end):
        return {"python": sys.version.split()[0], "platform": platform.platform(),
                "packages": pip_versions(self.python, ENV_PACKAGES), "gpu": gpu_info(), "start_utc": _iso(self.t_start),
                "end_utc": _iso(end), "duration_s": round((end - self.t_start).total_seconds(), 1),
                "run_seconds": self.run_seconds, "exit_code": self.exit_code, "timed_out": self.timed_out,
                "queue_sha": self.queue_sha, "runner_sha256": sha256_file(os.path.abspath(__file__))}

    def _block(self, job_id, status, results_sha, duration, extra=()):
        lines = ["=" * 44, "COLAB RUNNER STATUS", f"job_id   : {job_id}", f"status   : {status}",
                 f"results  : {results_sha} ({self.results_branch})", f"duration : {duration} s", *extra, "=" * 44]
        for ln in lines:
            self._say(ln)

    def _result(self, status, job_id, results_sha, duration, message):
        return {"status": status, "job_id": job_id, "results_sha": results_sha, "duration_s": duration, "message": message}

    def publish(self):
        """Validate, bundle, copy to Drive, push to colab-results, print the status block. Returns a result dict."""
        try:
            return self._publish()
        finally:
            shutil.rmtree(self._scratch, ignore_errors=True)

    def _publish(self):
        end = utc_now()
        duration = round((end - self.t_start).total_seconds(), 1)
        job = self.job or self.stub
        if job is None:
            msg = "; ".join(self.problems) or "no usable job"
            self._block("-", "REFUSED", "-", duration, [f"reason   : {msg}", "nothing was pushed"])
            return self._result("REFUSED", None, "-", duration, msg)
        if not self.refused and not (self._rerun_checked or job.get("rerun")):
            try:
                with guard(self.secrets):
                    if self._finished(job["job_id"]):
                        self._refuse(job["job_id"])
            except RunnerError as e:
                self.problems.append(f"check_rerun: {e}")
        if self.refused:
            self._block(job["job_id"], "REFUSED", "-", duration, [f"reason   : {self.problems[-1]}", "nothing was pushed"])
            return self._result("REFUSED", job["job_id"], "-", duration, self.problems[-1])
        if self.job is not None and not self.problems:
            self.phase = "validate"
            problems = validate_outputs(self.out_dir, self.job, self.job["repo_ref"])
            self.problems += [f"validate: {p}" for p in problems]
        status = "OK" if not self.problems else "FAIL"
        bundle = os.path.join(self.workdir, "bundle", job["job_id"])
        log_path = self.log_path or os.path.join(self.workdir, "missing.log")
        extra, results_sha = [], "-"
        try:
            with guard(self.secrets):
                env = self._env_info(end)
                names = build_bundle(bundle, job, out_dir=self.out_dir or "", log_path=log_path, env=env, status=status,
                                     problems=self.problems, secrets=self.secrets)
                try:
                    dest = os.path.join(self.drive_runs, job["job_id"])
                    shutil.copytree(bundle, dest, dirs_exist_ok=True)
                    extra.append(f"drive    : {dest}")
                except OSError as e:
                    extra.append(f"drive    : copy failed ({type(e).__name__}); the bundle is still pushed to GitHub")
                results_sha = push_results(self.repo_url, self.results_branch, bundle, job, status, self.workdir,
                                           env=self.git, secrets=self.secrets)
        except RunnerError as e:
            self.problems.append(f"publish: {e}")
            status = "FAIL"
            extra.append(f"push     : FAILED ({e})")
            if os.path.isdir(os.path.join(self.drive_runs, job["job_id"])):
                extra.append("the bundle is saved on Drive only")
        for p in self.problems:
            extra.append(f"problem  : {p}")
        self._block(job["job_id"], status, results_sha, duration, extra)
        return self._result(status, job["job_id"], results_sha, duration, "; ".join(self.problems))


# ---------------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Colab runner helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cj = sub.add_parser("check-job", help="validate a job.json (schema + repo_ref exists in a local clone)")
    cj.add_argument("file")
    cj.add_argument("--repo", default=".", help="local clone used to confirm that repo_ref exists and holds the entry")
    ns = ap.parse_args(argv)
    with open(ns.file, encoding="utf-8") as fh:
        job = parse_job(fh.read())
    ref = job["repo_ref"]
    if subprocess.run(["git", "cat-file", "-e", ref + "^{commit}"], cwd=ns.repo, capture_output=True).returncode != 0:
        raise SystemExit(f"repo_ref {ref} is not a commit in {ns.repo} (fetch first, or push it)")
    for path in (job["entry"], "tools/colab_runner.py"):
        if subprocess.run(["git", "cat-file", "-e", f"{ref}:{path}"], cwd=ns.repo, capture_output=True).returncode != 0:
            raise SystemExit(f"{path} does not exist at {ref}")
    print(f"OK: job {job['job_id']} @ {ref[:12]} entry {job['entry']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

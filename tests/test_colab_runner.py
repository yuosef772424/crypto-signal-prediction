"""tools/colab_runner.py (one-click Colab experiment runner: job validation, output validation, bundle, token masking,
refuse rules, git fetch/push against local bare remotes) and tools/colab_smoke.py. Offline: no Drive, no network.
    python -m pytest tests/test_colab_runner.py -q
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import colab_runner as cr  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"
# Built at runtime so the CI secret scan never sees a token-shaped literal in the source.
TOKEN = "gh" + "p_" + "Zq9" * 12


def good_job(**over):
    job = {"job_id": "smoke-001", "repo_ref": SHA, "entry": "tools/colab_smoke.py", "args": ["--sleep", "0"],
           "config": {"rows": 5, "seed": 7}, "max_minutes": 5, "needs_drive": False,
           "expected_outputs": ["smoke.csv"], "note": "smoke"}
    job.update(over)
    return job


# ----------------------------------------------------------------------------------------------- job validation
def test_valid_job_is_normalised():
    job = cr.validate_job(good_job(repo_ref=SHA.upper()))
    assert job["repo_ref"] == SHA and job["rerun"] is False and job["pip"] == []


@pytest.mark.parametrize("ref", ["main", "abc1234", SHA[:-1], SHA + "0", "g" * 40, "", 5, None, SHA + "\n"])
def test_repo_ref_must_be_full_sha(ref):
    with pytest.raises(cr.JobError, match="repo_ref"):
        cr.validate_job(good_job(repo_ref=ref))


@pytest.mark.parametrize("entry", [
    "docs/research/x.py", "../tools/x.py", "tools/../docs/x.py", "/etc/passwd", "/tools/x.py", "tools\\x.py",
    "tools/x.sh", "tools/", "tools", "research/../../x.py", "tools//x.py", "./tools/x.py", "tools/./x.py",
    "toolsx/x.py", "", None, 7, "tools/x.py\n", "tools/ x.py"])
def test_entry_must_be_py_under_tools_or_research(entry):
    with pytest.raises(cr.JobError, match="entry"):
        cr.validate_job(good_job(entry=entry))


def test_entry_under_research_is_fine():
    assert cr.validate_job(good_job(entry="research/h99/run.py"))["entry"] == "research/h99/run.py"


def test_unknown_and_missing_keys_raise():
    with pytest.raises(cr.JobError, match="unknown"):
        cr.validate_job(good_job(surprise=1))
    for key in ("job_id", "repo_ref", "entry", "args", "config", "max_minutes", "needs_drive", "expected_outputs", "note"):
        job = good_job()
        del job[key]
        with pytest.raises(cr.JobError, match=key):
            cr.validate_job(job)
    with pytest.raises(cr.JobError):
        cr.validate_job(["not", "a", "dict"])


@pytest.mark.parametrize("field,value", [
    ("job_id", "a/b"), ("job_id", ".."), ("job_id", "x"), ("job_id", "a b"), ("job_id", "-lead"), ("job_id", "é" * 5),
    ("args", "--rows 5"), ("args", ["ok", 5]), ("args", ["--config", "x"]), ("args", ["--out=/tmp/x"]),
    ("args", ["--max-minutes", "9"]), ("args", ["a\nb"]),
    ("config", []), ("config", "x"), ("config", {"k": float("nan")}),
    ("max_minutes", 0), ("max_minutes", -3), ("max_minutes", 100000), ("max_minutes", True), ("max_minutes", "5"),
    ("max_minutes", 2.5), ("needs_drive", "yes"), ("needs_drive", 1),
    ("expected_outputs", "a.csv"), ("expected_outputs", ["/abs.csv"]), ("expected_outputs", ["../x.csv"]),
    ("expected_outputs", ["a.csv", "a.csv"]), ("expected_outputs", [3]),
    ("note", 5), ("rerun", "true"), ("pip", "numpy"), ("pip", ["--index-url=http://evil"]), ("pip", ["-e ."]),
    ("pip", ["numpy; rm -rf"]), ("pip", ["git+https://x/y.git"])])
def test_bad_field_values_raise(field, value):
    with pytest.raises(cr.JobError, match=field):
        cr.validate_job(good_job(**{field: value}))


def test_rerun_and_pip_are_accepted():
    job = cr.validate_job(good_job(rerun=True, pip=["scipy==1.15.0", "tqdm"]))
    assert job["rerun"] is True and job["pip"] == ["scipy==1.15.0", "tqdm"]


def test_parse_job_rejects_bad_json_and_duplicate_keys():
    assert cr.parse_job(json.dumps(good_job()))["job_id"] == "smoke-001"
    with pytest.raises(cr.JobError, match="JSON"):
        cr.parse_job("{nope")
    with pytest.raises(cr.JobError, match="duplicate"):
        cr.parse_job('{"job_id": "aaa", "job_id": "bbb"}')


# ------------------------------------------------------------------------------------------------- token masking
def test_mask_covers_plain_urlencoded_and_basic_auth_forms():
    odd = TOKEN + "+/=x y"
    b64 = base64.b64encode(("x-access-token:" + odd).encode()).decode()
    text = f"a {odd} b {urllib.parse.quote(odd)} c {b64} d {TOKEN}"
    out = cr.mask(text, [odd, TOKEN])
    for leak in (odd, urllib.parse.quote(odd), b64, TOKEN):
        assert leak not in out
    assert out.count("***") >= 4


def test_mask_ignores_short_or_empty_secrets():
    assert cr.mask("hello world", ["", None, "o"]) == "hello world"


def test_guard_masks_exception_text_and_context():
    secrets = cr.Secrets(TOKEN)
    with pytest.raises(cr.RunnerError) as info:
        with cr.guard(secrets):
            raise ValueError("clone failed for https://x:" + TOKEN + "@host")
    assert TOKEN not in str(info.value) and "***" in str(info.value)
    assert info.value.__suppress_context__ is True
    assert TOKEN not in str(info.value.__context__)


def test_guard_passes_clean_blocks_and_keyboard_interrupt():
    secrets = cr.Secrets(TOKEN)
    with cr.guard(secrets):
        pass
    with pytest.raises(KeyboardInterrupt):
        with cr.guard(secrets):
            raise KeyboardInterrupt()


def test_read_token_single_line(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(TOKEN + "\n")
    assert cr.read_token(str(p)) == TOKEN
    p.write_text("  " + TOKEN + "  ")
    assert cr.read_token(str(p)) == TOKEN


@pytest.mark.parametrize("content", ["", "\n", "short", TOKEN + "\nsecond line", "two words here ok"])
def test_read_token_rejects_bad_files_without_echoing(tmp_path, content):
    p = tmp_path / "t.txt"
    p.write_text(content)
    with pytest.raises(cr.RunnerError) as info:
        cr.read_token(str(p))
    assert TOKEN not in str(info.value) and "second line" not in str(info.value)


def test_read_token_missing_file(tmp_path):
    with pytest.raises(cr.RunnerError, match="token file"):
        cr.read_token(str(tmp_path / "nope.txt"))


def test_git_env_uses_askpass_and_never_puts_token_in_argv(tmp_path):
    env = cr.git_env(TOKEN, str(tmp_path))
    assert env["GIT_TERMINAL_PROMPT"] == "0" and os.access(env["GIT_ASKPASS"], os.X_OK)
    user = subprocess.run([env["GIT_ASKPASS"], "Username for 'https://github.com': "], env=env, capture_output=True, text=True)
    pw = subprocess.run([env["GIT_ASKPASS"], "Password for 'https://x@github.com': "], env=env, capture_output=True, text=True)
    assert user.stdout.strip() == "x-access-token" and pw.stdout.strip() == TOKEN
    with open(env["GIT_ASKPASS"], encoding="utf-8") as fh:
        assert TOKEN not in fh.read()            # the script reads the environment, it does not embed the token
    assert cr.github_url("o/r") == "https://github.com/o/r.git"


# ---------------------------------------------------------------------------------- manifest + output validation
def make_run(tmp_path, repo_sha=SHA, **over):
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    (out / "a.csv").write_text("x,y\n1,2\n3,4\n")
    sub = out / "sub"
    sub.mkdir(exist_ok=True)
    (sub / "b.csv").write_text("k\n1\n")
    manifest = {"git_sha": repo_sha, "config": {"rows": 5, "seed": 7, "extra_default": 1},
                "input_fingerprint": cr.input_fingerprint(str(tmp_path), []), "row_counts": {"a.csv": 2},
                "runtime_seconds": 0.1, "n_configurations": 1,
                "outputs": {"a.csv": cr.sha256_file(str(out / "a.csv")), "sub/b.csv": cr.sha256_file(str(sub / "b.csv"))}}
    manifest.update(over)
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out, manifest


def rewrite_manifest(out, **changes):
    m = json.loads((out / "manifest.json").read_text())
    m.update(changes)
    (out / "manifest.json").write_text(json.dumps(m))


def job_for(**over):
    return cr.validate_job(good_job(expected_outputs=["a.csv", "sub/b.csv"], **over))


def test_valid_outputs_pass(tmp_path):
    out, _ = make_run(tmp_path)
    assert cr.validate_outputs(str(out), job_for(), SHA) == []


def test_missing_or_broken_manifest(tmp_path):
    out, _ = make_run(tmp_path)
    (out / "manifest.json").write_text("{nope")
    assert any("manifest.json" in p for p in cr.validate_outputs(str(out), job_for(), SHA))
    os.remove(out / "manifest.json")
    assert any("manifest.json" in p and "missing" in p for p in cr.validate_outputs(str(out), job_for(), SHA))
    assert any("missing" in p for p in cr.validate_outputs(str(tmp_path / "does-not-exist"), job_for(), SHA))


@pytest.mark.parametrize("key", ["git_sha", "config", "input_fingerprint", "outputs"])
def test_manifest_required_keys(tmp_path, key):
    out, manifest = make_run(tmp_path)
    del manifest[key]
    (out / "manifest.json").write_text(json.dumps(manifest))
    assert any(key in p for p in cr.validate_outputs(str(out), job_for(), SHA))


def test_git_sha_and_config_must_match_the_job(tmp_path):
    out, _ = make_run(tmp_path)
    rewrite_manifest(out, git_sha="f" * 40)
    assert any("git_sha" in p for p in cr.validate_outputs(str(out), job_for(), SHA))
    out, _ = make_run(tmp_path)
    rewrite_manifest(out, config={"rows": 6, "seed": 7})
    assert any("config" in p for p in cr.validate_outputs(str(out), job_for(), SHA))


def test_bad_input_fingerprint(tmp_path):
    out, _ = make_run(tmp_path)
    rewrite_manifest(out, input_fingerprint="not-a-sha")
    assert any("input_fingerprint" in p for p in cr.validate_outputs(str(out), job_for(), SHA))


def test_sha256_mismatch_and_missing_listed_file(tmp_path):
    out, _ = make_run(tmp_path)
    (out / "a.csv").write_text("x,y\n1,3\n")                           # changed after the manifest was written
    assert any("a.csv" in p and "sha256" in p for p in cr.validate_outputs(str(out), job_for(), SHA))
    out, _ = make_run(tmp_path)
    os.remove(out / "sub" / "b.csv")
    assert any("sub/b.csv" in p for p in cr.validate_outputs(str(out), job_for(), SHA))


def test_unlisted_file_is_a_problem(tmp_path):
    out, _ = make_run(tmp_path)
    (out / "stray.txt").write_text("hi")
    assert any("stray.txt" in p and "not listed" in p for p in cr.validate_outputs(str(out), job_for(), SHA))


def test_oversize_output(tmp_path):
    out, _ = make_run(tmp_path)
    big = out / "big.csv"
    big.write_bytes(b"x" * (cr.MAX_OUTPUT_BYTES + 1))
    m = json.loads((out / "manifest.json").read_text())
    m["outputs"]["big.csv"] = cr.sha256_file(str(big))
    (out / "manifest.json").write_text(json.dumps(m))
    assert any("big.csv" in p and "300" in p for p in cr.validate_outputs(str(out), job_for(), SHA))
    big.write_bytes(b"x" * cr.MAX_OUTPUT_BYTES)                          # exactly at the limit is fine
    m["outputs"]["big.csv"] = cr.sha256_file(str(big))
    (out / "manifest.json").write_text(json.dumps(m))
    assert cr.validate_outputs(str(out), job_for(), SHA) == []


def test_expected_outputs_must_exist(tmp_path):
    out, _ = make_run(tmp_path)
    job = cr.validate_job(good_job(expected_outputs=["a.csv", "never.csv"]))
    assert any("never.csv" in p and "expected" in p for p in cr.validate_outputs(str(out), job, SHA))


def test_manifest_cannot_point_outside_out_dir(tmp_path):
    out, manifest = make_run(tmp_path)
    secret = tmp_path / "outside.txt"
    secret.write_text("x")
    manifest["outputs"]["../outside.txt"] = cr.sha256_file(str(secret))
    (out / "manifest.json").write_text(json.dumps(manifest))
    assert any("outside.txt" in p for p in cr.validate_outputs(str(out), job_for(), SHA))


def test_symlink_and_reserved_names_rejected(tmp_path):
    out, _ = make_run(tmp_path)
    target = tmp_path / "t.txt"
    target.write_text("zz")
    os.symlink(target, out / "link.txt")
    assert any("link.txt" in p for p in cr.validate_outputs(str(out), job_for(), SHA))
    out, manifest = make_run(tmp_path)
    (out / "link.txt").unlink()
    (out / "env.json").write_text("{}")
    manifest["outputs"]["env.json"] = cr.sha256_file(str(out / "env.json"))
    (out / "manifest.json").write_text(json.dumps(manifest))
    assert any("env.json" in p and "reserved" in p for p in cr.validate_outputs(str(out), job_for(), SHA))


def test_input_fingerprint_is_order_independent_and_size_sensitive(tmp_path):
    (tmp_path / "a.csv").write_text("12345")
    (tmp_path / "b.csv").write_text("1")
    f1 = cr.input_fingerprint(str(tmp_path), ["a.csv", "b.csv"])
    assert f1 == cr.input_fingerprint(str(tmp_path), ["b.csv", "a.csv"]) and len(f1) == 64
    (tmp_path / "b.csv").write_text("12")
    assert cr.input_fingerprint(str(tmp_path), ["a.csv", "b.csv"]) != f1
    assert len(cr.input_fingerprint(str(tmp_path), [])) == 64


def test_write_manifest_round_trips_through_validation(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    sha = init_repo(repo, {"f.txt": "x"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "r.csv").write_text("a\n1\n")
    cr.write_manifest(str(out), config={"rows": 5, "seed": 7}, repo_dir=str(repo), row_counts={"r.csv": 1})
    job = cr.validate_job(good_job(expected_outputs=["r.csv"]))
    assert cr.validate_outputs(str(out), job, sha) == []
    m = json.loads((out / "manifest.json").read_text())
    assert m["git_sha"] == sha and list(m["outputs"]) == ["r.csv"] and m["n_configurations"] == 1


# ------------------------------------------------------------------------------------------------ log + bundle
def test_tail_text_is_bounded_and_marks_truncation(tmp_path):
    p = tmp_path / "log.txt"
    p.write_text("".join(f"line {i:06d}\n" for i in range(40000)))
    assert p.stat().st_size > cr.MAX_LOG_BYTES
    tail = cr.tail_text(str(p), cr.MAX_LOG_BYTES)
    assert len(tail.encode()) <= cr.MAX_LOG_BYTES and tail.endswith("line 039999\n") and "truncated" in tail.splitlines()[0]
    small = tmp_path / "s.txt"
    small.write_text("a\nb\n")
    assert cr.tail_text(str(small), cr.MAX_LOG_BYTES) == "a\nb\n"


def test_build_bundle_ok(tmp_path):
    out, _ = make_run(tmp_path)
    log = tmp_path / "raw.log"
    log.write_text("hello\nleak " + TOKEN + "\n")
    dest = tmp_path / "runs" / "smoke-001"
    env = {"exit_code": 0, "python": "3.x"}
    names = cr.build_bundle(str(dest), job_for(), out_dir=str(out), log_path=str(log), env=env, status="OK", problems=[],
                            secrets=cr.Secrets(TOKEN))
    assert {"manifest.json", "a.csv", "sub/b.csv", "run_log.txt", "env.json", "bundle_summary.md"} <= set(names)
    assert TOKEN not in (dest / "run_log.txt").read_text() and "hello" in (dest / "run_log.txt").read_text()
    assert json.loads((dest / "env.json").read_text())["status"] == "OK"
    summary = (dest / "bundle_summary.md").read_text()
    assert "smoke-001" in summary and "OK" in summary and "x,y" in summary and "git_sha" in summary and "sub/b.csv" in summary
    assert (dest / "sub" / "b.csv").read_text() == "k\n1\n"


def test_build_bundle_csv_head_is_limited(tmp_path):
    out, manifest = make_run(tmp_path)
    (out / "a.csv").write_text("h\n" + "".join(f"row{i}\n" for i in range(500)))
    manifest["outputs"]["a.csv"] = cr.sha256_file(str(out / "a.csv"))
    (out / "manifest.json").write_text(json.dumps(manifest))
    log = tmp_path / "raw.log"
    log.write_text("x\n")
    dest = tmp_path / "bundle"
    cr.build_bundle(str(dest), job_for(), out_dir=str(out), log_path=str(log), env={}, status="OK", problems=[])
    summary = (dest / "bundle_summary.md").read_text()
    assert "row5" in summary and "row499" not in summary


def test_build_bundle_fail_has_no_outputs_but_keeps_log_and_problems(tmp_path):
    out, _ = make_run(tmp_path)
    log = tmp_path / "raw.log"
    log.write_text("boom\n")
    dest = tmp_path / "bundle"
    names = cr.build_bundle(str(dest), job_for(), out_dir=str(out), log_path=str(log), env={"exit_code": 3}, status="FAIL",
                            problems=["exit code 3"])
    assert set(names) == {"run_log.txt", "env.json", "bundle_summary.md"}
    assert "FAIL" in (dest / "bundle_summary.md").read_text() and "exit code 3" in (dest / "bundle_summary.md").read_text()


def test_pip_versions_reports_installed_packages():
    v = cr.pip_versions(sys.executable, ["pytest", "definitely-not-installed-pkg"])
    assert v["pytest"] and v["definitely-not-installed-pkg"] is None


# -------------------------------------------------------------------------------------------- running the job
def test_run_logged_streams_masks_and_enforces_the_wall_clock(tmp_path):
    log = tmp_path / "log.txt"
    lines = []
    t0 = time.time()
    code, timed_out = cr.run_logged(
        [sys.executable, "-c", f"import time,sys; print('hello {TOKEN}', flush=True); time.sleep(60)"],
        cwd=str(tmp_path), env=dict(os.environ), log_path=str(log), wall_seconds=1.5, secrets=cr.Secrets(TOKEN),
        echo=lines.append)
    assert timed_out is True and code != 0 and time.time() - t0 < 30
    text = log.read_text()
    assert "hello" in text and TOKEN not in text and all(TOKEN not in line for line in lines)


def test_run_logged_normal_exit_and_echo_cap(tmp_path):
    log = tmp_path / "log.txt"
    lines = []
    code, timed_out = cr.run_logged([sys.executable, "-c", "print('\\n'.join(str(i) for i in range(2000)))"],
                                    cwd=str(tmp_path), env=dict(os.environ), log_path=str(log), wall_seconds=60,
                                    echo=lines.append, echo_lines=50)
    assert (code, timed_out) == (0, False)
    assert len(log.read_text().splitlines()) == 2000 and len(lines) <= 52


# -------------------------------------------------------------------------------- git against local bare remotes
def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args], cwd=cwd,
                          check=True, capture_output=True, text=True).stdout.strip()


def init_repo(path, files, branch="master"):
    git(str(path), "init", "-q", "-b", branch)
    for rel, content in files.items():
        f = os.path.join(str(path), rel)
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(content)
    git(str(path), "add", "-A")
    git(str(path), "commit", "-q", "-m", "init")
    return git(str(path), "rev-parse", "HEAD")


def make_remote(tmp_path, files):
    """A local repo with the real runner + smoke script, cloned bare: stands in for GitHub."""
    src = tmp_path / "src"
    src.mkdir()
    base = {"tools/colab_runner.py": open(os.path.join(ROOT, "tools", "colab_runner.py"), encoding="utf-8").read(),
            "tools/colab_smoke.py": open(os.path.join(ROOT, "tools", "colab_smoke.py"), encoding="utf-8").read(),
            "requirements-ci.txt": "# none\n"}
    base.update(files)
    sha = init_repo(src, base)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(src), str(bare)], check=True)
    return str(bare), sha, src


def test_clone_at_checks_out_exactly_the_sha_detached(tmp_path):
    remote, sha1, src = make_remote(tmp_path, {})
    (src / "later.txt").write_text("later")
    git(str(src), "add", "-A")
    git(str(src), "commit", "-q", "-m", "second")
    subprocess.run(["git", "push", "-q", remote, "master"], cwd=src, check=True)
    dest = tmp_path / "work" / "repo"
    cr.clone_at(remote, sha1, str(dest))
    assert git(str(dest), "rev-parse", "HEAD") == sha1 and not (dest / "later.txt").exists()
    assert subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=dest).returncode != 0          # detached


def test_clone_at_unknown_sha_fails_cleanly(tmp_path):
    remote, _, _ = make_remote(tmp_path, {})
    with pytest.raises(cr.RunnerError, match="checkout"):
        cr.clone_at(remote, "a" * 40, str(tmp_path / "w"))


def test_clone_at_refuses_non_sha(tmp_path):
    remote, _, _ = make_remote(tmp_path, {})
    with pytest.raises(cr.RunnerError, match="repo_ref"):
        cr.clone_at(remote, "master", str(tmp_path / "w"))


def test_check_entry(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    init_repo(repo, {"tools/ok.py": "x=1\n", "docs/bad.py": "x=1\n", "research/s/run.py": "x=1\n"})
    cr.check_entry(str(repo), "tools/ok.py")
    cr.check_entry(str(repo), "research/s/run.py")
    with pytest.raises(cr.RunnerError, match="not found"):
        cr.check_entry(str(repo), "tools/missing.py")
    os.symlink(repo / "docs" / "bad.py", repo / "tools" / "link.py")                  # symlink escaping tools/
    with pytest.raises(cr.RunnerError, match="outside"):
        cr.check_entry(str(repo), "tools/link.py")


def test_push_results_creates_orphan_branch_then_appends(tmp_path):
    remote, _, _ = make_remote(tmp_path, {})
    bundle = tmp_path / "b1"
    bundle.mkdir()
    (bundle / "env.json").write_text("{}")
    job = cr.validate_job(good_job())
    sha1 = cr.push_results(remote, "colab-results", str(bundle), job, "OK", str(tmp_path / "w1"))
    heads = git(remote, "for-each-ref", "--format=%(refname:short)", "refs/heads")
    assert "colab-results" in heads.split() and git(remote, "rev-parse", "colab-results") == sha1
    assert git(remote, "rev-list", "--max-parents=0", "colab-results") == sha1                    # orphan: no master history
    msg = git(remote, "log", "-1", "--format=%B", "colab-results")
    assert "smoke-001" in msg and SHA in msg and "OK" in msg
    assert git(remote, "show", "colab-results:runs/smoke-001/env.json") == "{}"
    assert "colab-results" not in git(remote, "branch", "--contains", "master").split()
    job2 = cr.validate_job(good_job(job_id="smoke-002"))
    sha2 = cr.push_results(remote, "colab-results", str(bundle), job2, "FAIL", str(tmp_path / "w2"))
    assert git(remote, "rev-parse", "colab-results~1") == sha1 and sha1 != sha2
    assert git(remote, "show", "colab-results:runs/smoke-001/env.json") == "{}"                  # earlier run kept


def test_push_results_retries_after_a_competing_push(tmp_path, monkeypatch):
    remote, _, _ = make_remote(tmp_path, {})
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "env.json").write_text("{}")
    cr.push_results(remote, "colab-results", str(bundle), cr.validate_job(good_job(job_id="first-1")), "OK",
                    str(tmp_path / "w0"))
    real_clone, state = cr.clone_branch, {"raced": False}

    def racing_clone(*a, **k):
        result = real_clone(*a, **k)
        if not state["raced"]:                                                    # someone else pushes right after our clone
            state["raced"] = True
            other = tmp_path / "other"
            subprocess.run(["git", "clone", "-q", "--branch", "colab-results", remote, str(other)], check=True)
            (other / "runs" / "rival-1").mkdir(parents=True)
            (other / "runs" / "rival-1" / "env.json").write_text("{}")
            git(str(other), "add", "-A")
            git(str(other), "commit", "-q", "-m", "rival")
            git(str(other), "push", "-q", "origin", "colab-results")
        return result

    monkeypatch.setattr(cr, "clone_branch", racing_clone)
    cr.push_results(remote, "colab-results", str(bundle), cr.validate_job(good_job(job_id="mine-001")), "OK",
                    str(tmp_path / "w1"))
    assert git(remote, "show", "colab-results:runs/rival-1/env.json") == "{}"
    assert git(remote, "show", "colab-results:runs/mine-001/env.json") == "{}"


def test_results_has_job_and_refuse_rules(tmp_path):
    remote, _, _ = make_remote(tmp_path, {})
    assert cr.results_has_job(remote, "colab-results", "smoke-001", str(tmp_path / "w0")) is False      # branch missing
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "env.json").write_text("{}")
    cr.push_results(remote, "colab-results", str(bundle), cr.validate_job(good_job()), "OK", str(tmp_path / "w1"))
    assert cr.results_has_job(remote, "colab-results", "smoke-001", str(tmp_path / "w2")) is True
    assert cr.results_has_job(remote, "colab-results", "smoke-002", str(tmp_path / "w3")) is False


# ------------------------------------------------------------------------------- end to end on a local remote
def make_runner(tmp_path, remote, **kw):
    kw.setdefault("pip", lambda cmd, **k: 0)                    # no network: the install step is stubbed
    return cr.Runner(repo_url=remote, token=TOKEN, workdir=str(tmp_path / "work"), drive_runs=str(tmp_path / "drive" / "runs"),
                     echo=lambda *a: None, **kw)


def run_all(runner, job):
    runner.load(json.dumps(job), queue_sha="q" * 40)
    runner.check_rerun()
    runner.checkout()
    runner.install()
    runner.execute()
    return runner.publish()


def test_end_to_end_ok_then_refuse_then_rerun(tmp_path):
    remote, sha, _ = make_remote(tmp_path, {})
    job = good_job(repo_ref=sha, expected_outputs=["smoke.csv", "manifest.json"])
    res = run_all(make_runner(tmp_path, remote), job)
    assert res["status"] == "OK" and len(res["results_sha"]) == 40 and res["duration_s"] >= 0
    run_dir = "runs/smoke-001/"
    assert git(remote, "rev-parse", "colab-results") == res["results_sha"]
    for name in ("manifest.json", "smoke.csv", "run_log.txt", "env.json", "bundle_summary.md"):
        git(remote, "cat-file", "-e", f"colab-results:{run_dir}{name}")
    env = json.loads(git(remote, "show", f"colab-results:{run_dir}env.json"))
    assert env["exit_code"] == 0 and env["status"] == "OK" and env["repo_ref"] == sha and env["queue_sha"] == "q" * 40
    assert {"python", "start_utc", "end_utc", "duration_s", "gpu", "packages"} <= set(env)
    assert os.path.isfile(tmp_path / "drive" / "runs" / "smoke-001" / "manifest.json")
    assert TOKEN not in git(remote, "show", f"colab-results:{run_dir}run_log.txt")
    assert "master" in git(remote, "for-each-ref", "--format=%(refname:short)", "refs/heads")

    before = git(remote, "rev-parse", "colab-results")
    refused = make_runner(tmp_path, remote)
    res2 = run_all(refused, job)                                                  # same job_id, already finished
    assert refused.refused is True and res2["status"] == "REFUSED"
    assert git(remote, "rev-parse", "colab-results") == before                  # nothing pushed

    res3 = run_all(make_runner(tmp_path, remote), dict(job, rerun=True))
    assert res3["status"] == "OK" and res3["results_sha"] != before, res3["message"]


def test_end_to_end_failing_job_still_pushes_fail_log(tmp_path):
    crash = "import sys\nprint('about to fail')\nsys.exit(3)\n"
    remote, sha, _ = make_remote(tmp_path, {"tools/crash.py": crash})
    res = run_all(make_runner(tmp_path, remote), good_job(job_id="crash-001", repo_ref=sha, entry="tools/crash.py",
                                                          args=[], expected_outputs=[]))
    assert res["status"] == "FAIL" and len(res["results_sha"]) == 40
    env = json.loads(git(remote, "show", "colab-results:runs/crash-001/env.json"))
    assert env["exit_code"] == 3 and env["status"] == "FAIL"
    assert "about to fail" in git(remote, "show", "colab-results:runs/crash-001/run_log.txt")
    assert "exit code 3" in git(remote, "show", "colab-results:runs/crash-001/bundle_summary.md")


def test_end_to_end_job_without_manifest_fails_validation(tmp_path):
    quiet = "print('done, but no manifest')\n"
    remote, sha, _ = make_remote(tmp_path, {"tools/quiet.py": quiet})
    res = run_all(make_runner(tmp_path, remote), good_job(job_id="quiet-001", repo_ref=sha, entry="tools/quiet.py",
                                                          args=[], expected_outputs=[]))
    assert res["status"] == "FAIL"
    assert "manifest.json" in git(remote, "show", "colab-results:runs/quiet-001/bundle_summary.md")


def test_end_to_end_unknown_sha_pushes_fail(tmp_path):
    remote, _, _ = make_remote(tmp_path, {})
    res = run_all(make_runner(tmp_path, remote), good_job(job_id="ghost-001", repo_ref="b" * 40))
    assert res["status"] == "FAIL"
    assert "checkout" in git(remote, "show", "colab-results:runs/ghost-001/bundle_summary.md")


def test_end_to_end_invalid_job_with_usable_id_pushes_fail_and_nothing_runs(tmp_path):
    remote, sha, _ = make_remote(tmp_path, {})
    res = run_all(make_runner(tmp_path, remote), good_job(job_id="evil-001", repo_ref=sha, entry="docs/research/x.py"))
    assert res["status"] == "FAIL"
    assert "entry" in git(remote, "show", "colab-results:runs/evil-001/bundle_summary.md")
    assert not os.path.exists(tmp_path / "work" / "repo")                         # never cloned


def test_end_to_end_unusable_job_is_refused_without_push(tmp_path):
    remote, _, _ = make_remote(tmp_path, {})
    r = make_runner(tmp_path, remote)
    r.load("{not json", queue_sha="")
    r.check_rerun(), r.checkout(), r.install(), r.execute()
    res = r.publish()
    assert res["status"] == "REFUSED" and "JSON" in res["message"]
    assert "colab-results" not in git(remote, "for-each-ref", "--format=%(refname:short)", "refs/heads").split()


def test_job_environment_never_contains_the_token(tmp_path):
    probe = ("import os, json, sys\nsys.path.insert(0, os.path.join(os.path.dirname(__file__)))\nimport colab_runner as cr\n"
             "import argparse\np = argparse.ArgumentParser()\np.add_argument('--config'); p.add_argument('--out')\n"
             "p.add_argument('--max-minutes')\na = p.parse_args()\nos.makedirs(a.out, exist_ok=True)\n"
             "open(os.path.join(a.out, 'env_seen.txt'), 'w').write(json.dumps(sorted(os.environ)))\n"
             "cr.write_manifest(a.out, config=json.load(open(a.config)))\n")
    remote, sha, _ = make_remote(tmp_path, {"tools/probe.py": probe})
    res = run_all(make_runner(tmp_path, remote), good_job(job_id="probe-001", repo_ref=sha, entry="tools/probe.py", args=[],
                                                          config={}, expected_outputs=["env_seen.txt"]))
    assert res["status"] == "OK", res
    seen = json.loads(git(remote, "show", "colab-results:runs/probe-001/env_seen.txt"))
    assert not {"GIT_ASKPASS", "COLAB_RUNNER_TOKEN"} & set(seen)
    assert "CSP_REPO_DIR" in seen and ("CSP_DRIVE_ROOT" in seen) is False


# ----------------------------------------------------------------------------------------------- smoke script
def test_smoke_script_writes_a_valid_contract_bundle(tmp_path):
    out = tmp_path / "o"
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"rows": 4, "seed": 3}))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "colab_smoke.py"), "--config", str(cfg), "--out", str(out),
                        "--max-minutes", "1", "--sleep", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = (out / "smoke.csv").read_text().splitlines()
    assert rows[0] == "i,value" and len(rows) == 5
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    job = cr.validate_job(good_job(config={"rows": 4, "seed": 3}, expected_outputs=["smoke.csv"]))
    assert cr.validate_outputs(str(out), job, head.stdout.strip() if head.returncode == 0 else json.loads(
        (out / "manifest.json").read_text())["git_sha"]) == []


def test_smoke_script_rejects_unknown_config_keys(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"rows": 4, "typo": 1}))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "colab_smoke.py"), "--config", str(cfg), "--out",
                        str(tmp_path / "o"), "--max-minutes", "1"], capture_output=True, text=True)
    assert r.returncode != 0 and "typo" in r.stderr


def test_smoke_script_is_deterministic(tmp_path):
    outs = []
    for name in ("a", "b"):
        cfg = tmp_path / f"{name}.json"
        cfg.write_text(json.dumps({"rows": 6, "seed": 11}))
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "colab_smoke.py"), "--config", str(cfg), "--out",
                        str(tmp_path / name), "--max-minutes", "1"], check=True, capture_output=True)
        outs.append(cr.sha256_file(str(tmp_path / name / "smoke.csv")))
    assert outs[0] == outs[1]
    shutil.rmtree(tmp_path / "a")


# ----------------------------------------------------------------------------------------------------- check-job CLI
def test_check_job_cli(tmp_path, capsys):
    repo = tmp_path / "r"
    repo.mkdir()
    sha = init_repo(repo, {"tools/colab_runner.py": "x=1\n", "tools/colab_smoke.py": "x=1\n"})
    jf = tmp_path / "job.json"
    jf.write_text(json.dumps(good_job(repo_ref=sha)))
    assert cr.main(["check-job", str(jf), "--repo", str(repo)]) == 0 and "OK: job smoke-001" in capsys.readouterr().out
    jf.write_text(json.dumps(good_job(repo_ref="c" * 40)))
    with pytest.raises(SystemExit, match="not a commit"):
        cr.main(["check-job", str(jf), "--repo", str(repo)])
    jf.write_text(json.dumps(good_job(repo_ref=sha, entry="tools/nothere.py")))
    with pytest.raises(SystemExit, match="does not exist"):
        cr.main(["check-job", str(jf), "--repo", str(repo)])
    jf.write_text(json.dumps(good_job(repo_ref=sha, entry="docs/x.py")))
    with pytest.raises(cr.JobError):
        cr.main(["check-job", str(jf), "--repo", str(repo)])

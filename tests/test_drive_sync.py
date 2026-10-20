"""tools/drive_sync.py (GitHub ref -> Drive folder: overwrite + keep, no delete, no backup, SYNCED_FROM.txt). Offline: the clone
step is injected; one test clones a local bare repo (branch, tag, short SHA).
    python -m pytest tests/test_drive_sync.py -q
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import drive_sync as ds  # noqa: E402

LINE = "0123456789abcdef0123456789abcdef01234567 2026-10-20 10:00:00 +0000 feat: example"


def put(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def read(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture
def world(tmp_path):
    """A fake clone (with .git/) and a fake Drive folder holding old, unchanged and Drive-only files."""
    clone, drive = tmp_path / "clone", tmp_path / "drive"
    put(clone, "new.py", "new")
    put(clone, "pkg/changed.py", "from github")
    put(clone, "same.py", "same")
    put(clone, "same_size.py", "aaaa")
    put(clone, ".git/config", "git internals")
    put(drive, "pkg/changed.py", "old drive copy")
    put(drive, "same.py", "same")
    put(drive, "same_size.py", "bbbb")
    put(drive, "models/best.keras", "weights")
    put(drive, "secrets/github_token.txt", "token")
    put(drive, "SYNCED_FROM.txt", "old record\n")
    return clone, drive, "SYNCED_FROM.txt"


def run(world, **kw):
    clone, drive, _ = world
    return ds.sync("master", str(drive), clone=lambda ref, work: (str(clone), LINE), echo=lambda *_: None, **kw)


def test_plan_sync_is_pure_and_sorted():
    plan = ds.plan_sync({"a": 1, "b": 2, "c": 3, "d": 4}, {"b": 2, "c": 9, "d": 4, "z": 1}, lambda rel: rel == "d")
    assert plan.new == ["a"] and plan.updated == ["b", "c"] and plan.unchanged == ["d"] and plan.kept == ["z"]
    assert plan.counts() == {"new": 1, "updated": 2, "unchanged": 1, "kept": 1}


def test_content_is_compared_only_for_equal_sizes():
    asked = []
    ds.plan_sync({"a": 1, "b": 2}, {"a": 5, "b": 2}, lambda rel: asked.append(rel) or True)
    assert asked == ["b"]


def test_overwrite_keep_new_unchanged_and_git_skipped(world):
    clone, drive, _ = world
    plan, line = run(world)
    assert plan.new == ["new.py"] and plan.updated == ["pkg/changed.py", "same_size.py"] and plan.unchanged == ["same.py"]
    assert plan.kept == ["models/best.keras", "secrets/github_token.txt"]
    assert read(drive, "pkg/changed.py") == "from github" and read(drive, "same_size.py") == "aaaa"
    assert read(drive, "new.py") == "new"
    assert read(drive, "models/best.keras") == "weights" and read(drive, "secrets/github_token.txt") == "token"
    assert not os.path.exists(os.path.join(drive, ".git"))


def test_no_backup_and_no_temp_files_left(world):
    _, drive, _ = world
    run(world)
    names = [n for _, _, fs in os.walk(drive) for n in fs]
    assert not any(n.endswith((".zip", ".bak", ".drive_sync_tmp")) for n in names)


def test_synced_from_has_sha_date_subject_and_second_run_is_all_unchanged(world):
    _, drive, name = world
    run(world)
    assert read(drive, name) == LINE + "\n"
    plan, _ = run(world)
    assert plan.new == [] and plan.updated == [] and len(plan.unchanged) == 4      # SYNCED_FROM.txt is not counted as kept
    assert plan.kept == ["models/best.keras", "secrets/github_token.txt"]


def test_dry_run_writes_nothing(world):
    _, drive, name = world
    plan, _ = run(world, dry_run=True)
    assert plan.new == ["new.py"] and not os.path.exists(os.path.join(drive, "new.py"))
    assert read(drive, "pkg/changed.py") == "old drive copy" and read(drive, name) == "old record\n"


def test_missing_drive_parent_and_bad_commit_line_raise(tmp_path, world):
    clone, _, _ = world
    with pytest.raises(ds.SyncError, match="mount Drive"):
        ds.sync("master", str(tmp_path / "nodrive" / "crypto"), clone=lambda r, w: (str(clone), LINE))
    with pytest.raises(ds.SyncError, match="unexpected commit line"):
        ds.sync("master", str(tmp_path / "d2"), clone=lambda r, w: (str(clone), "not a commit"))
    assert not os.path.exists(tmp_path / "d2")


def test_file_folder_clash_is_refused_before_any_write(tmp_path):
    clone, drive = tmp_path / "c", tmp_path / "d"
    put(clone, "x/y.py", "1")
    put(clone, "ok.py", "1")
    put(drive, "x", "a file where GitHub has a folder")
    with pytest.raises(ds.SyncError, match="folder in GitHub but a file"):
        ds.sync("master", str(drive), clone=lambda r, w: (str(clone), LINE), echo=lambda *_: None)
    assert not os.path.exists(drive / "ok.py")


def test_resolve_token_prefers_env_then_userdata_and_never_raises():
    class UD:
        @staticmethod
        def get(key):
            return "from-userdata"

    class Broken:
        @staticmethod
        def get(key):
            raise RuntimeError("no secret")

    assert ds.resolve_token({"GITHUB_TOKEN": "envtok"}, UD) == "envtok"
    assert ds.resolve_token({}, UD) == "from-userdata"
    assert ds.resolve_token({}, Broken) is None


def test_ref_kinds():
    assert ds.is_sha_ref("abc1234") and ds.is_sha_ref("a" * 40)
    assert not ds.is_sha_ref("master") and not ds.is_sha_ref("v1.0") and not ds.is_sha_ref("abc12")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_real_clone_of_a_local_repo_for_branch_tag_and_short_sha(tmp_path):
    repo = tmp_path / "origin"
    put(repo, "a.py", "v1")
    git(repo, "init", "-q", "-b", "master")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "first commit")
    first = git(repo, "rev-parse", "HEAD")
    git(repo, "tag", "v1")
    put(repo, "a.py", "v2")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "second commit")
    url = repo.as_uri()
    for ref, want, subject in (("master", "v2", "second commit"), ("v1", "v1", "first commit"), (first[:8], "v1", "first commit")):
        drive = tmp_path / f"drive_{ref}"
        ds.sync(ref, str(drive), repo_url=url, echo=lambda *_: None)
        assert read(drive, "a.py") == want
        assert read(drive, "SYNCED_FROM.txt").rstrip().endswith(subject)
        assert len(read(drive, "SYNCED_FROM.txt").split()[0]) == 40
        assert not os.path.exists(drive / ".git")

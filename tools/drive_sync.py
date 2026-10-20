#!/usr/bin/env python3
"""
PURPOSE:  Sync a GitHub ref (branch, tag or SHA) of the repo into the Drive folder the notebooks run from: files present in
          GitHub OVERWRITE the Drive copy, files only in Drive are KEPT (nothing is ever deleted), no backup is made. Writes
          SYNCED_FROM.txt (`<full sha> <commit date> <subject>`) and prints the sha, so every run records its commit.
TAGS:     drive, sync, github, colab, SYNCED_FROM, overwrite keep, no delete, no backup, update drive from github, clone ref,
          GITHUB_TOKEN, bootstrap
PITFALLS: Never deletes and never backs up (the old rsync --delete + zip cell is gone): a file removed from GitHub stays in Drive
          until you remove it by hand. Changed = different size or different sha256 (mtime is ignored: Drive and git disagree on
          it). `.git/` is skipped. SYNCED_FROM.txt is written last, so a failed copy leaves the previous record. The token
          (env GITHUB_TOKEN, else Colab userdata) is only given to git child processes via GIT_ASKPASS and is masked in every
          message. A hex ref of 7-40 chars is treated as a commit SHA (full clone + checkout), anything else as branch/tag
          (depth 1). Reuses the git/secret helpers of tools/colab_runner.py. Runs with only the standard library.

Usage:
    python tools/drive_sync.py --ref master --dest /content/drive/MyDrive/crypto [--dry-run]
    from drive_sync import sync; sync("master", "/content/drive/MyDrive/crypto")      # tools/ on sys.path
Bootstrap from a fresh Colab: docs/research/plan/DRIVE_SYNC.md.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colab_runner as cr  # noqa: E402  (git env, run_git, Secrets: one implementation of token handling)

REPO_URL = "https://github.com/yuosef772424/crypto-signal-prediction.git"
DEFAULT_DEST = "/content/drive/MyDrive/crypto"
SYNC_FILE = "SYNCED_FROM.txt"
SKIP_DIRS = frozenset({".git"})
TOKEN_ENV = "GITHUB_TOKEN"
COMMIT_FORMAT = "%H %ci %s"
_SHA_RE = re.compile(r"[0-9a-fA-F]{7,40}")
_LIST_LIMIT = 40


class SyncError(cr.RunnerError):
    """The sync cannot run or would be unsafe; the message is already safe to print."""


@dataclass
class Plan:
    """Which relative paths are copied (new, updated), left alone (unchanged) or only in the destination (kept)."""
    new: list = field(default_factory=list)
    updated: list = field(default_factory=list)
    unchanged: list = field(default_factory=list)
    kept: list = field(default_factory=list)

    def counts(self):
        return {k: len(getattr(self, k)) for k in ("new", "updated", "unchanged", "kept")}


# ------------------------------------------------------------------------------------------------------- pure logic
def plan_sync(src_sizes, dst_sizes, same_content):
    """File plan from {relpath: size} of the source and of the destination.

    ``same_content(rel)`` is asked only for paths present on both sides with equal size (different size = updated).
    Nothing here touches the disk; every list is sorted."""
    plan = Plan()
    for rel in sorted(src_sizes):
        if rel not in dst_sizes:
            plan.new.append(rel)
        elif src_sizes[rel] != dst_sizes[rel] or not same_content(rel):
            plan.updated.append(rel)
        else:
            plan.unchanged.append(rel)
    plan.kept = sorted(set(dst_sizes) - set(src_sizes))
    return plan


def sync_record(commit_line):
    """Content of SYNCED_FROM.txt for the `git log` line ``<sha> <date> <subject>``."""
    if not _SHA_RE.match(commit_line.split(" ", 1)[0]):
        raise SyncError(f"unexpected commit line: {commit_line!r}")
    return commit_line.strip() + "\n"


def is_sha_ref(ref):
    return bool(_SHA_RE.fullmatch(ref))


# ------------------------------------------------------------------------------------------------------------- I/O
def scan(root, skip_names=()):
    """{relpath (posix): size} of every file below ``root``, skipping SKIP_DIRS and the names in ``skip_names``."""
    sizes = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), root).replace(os.sep, "/")
            if rel not in skip_names:
                sizes[rel] = os.path.getsize(os.path.join(base, name))
    return sizes


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_token(environ=None, userdata=None):
    """Token from the environment, else from Colab userdata (``userdata`` = module with .get, injectable); None when absent."""
    environ = os.environ if environ is None else environ
    if environ.get(TOKEN_ENV):
        return environ[TOKEN_ENV]
    if userdata is None:
        try:
            from google.colab import userdata  # type: ignore
        except ImportError:
            return None
    try:
        return userdata.get(TOKEN_ENV) or None
    except Exception:                                  # noqa: BLE001  (Colab raises its own errors for a missing/denied secret)
        return None


def clone_ref(ref, workdir, repo_url=REPO_URL, token=None):
    """Clone ``ref`` into ``workdir/src`` and return ``(src_dir, commit_line)``; a SHA gets a full clone + checkout."""
    secrets = cr.Secrets(token)
    with cr.guard(secrets):
        env = cr.git_env(token, os.path.join(workdir, "askpass")) if token else None
        src = os.path.join(workdir, "src")
        if is_sha_ref(ref):
            cr.run_git(["clone", "--quiet", "--no-checkout", repo_url, src], env=env, secrets=secrets)
            cr.run_git(["checkout", "--quiet", "--detach", ref], cwd=src, env=env, secrets=secrets)
        else:
            cr.run_git(["clone", "--quiet", "--depth", "1", "--branch", ref, repo_url, src], env=env, secrets=secrets)
        return src, cr.run_git(["log", "-1", f"--format={COMMIT_FORMAT}"], cwd=src, env=env, secrets=secrets)


def _copy_atomic(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = os.path.join(os.path.dirname(dst), f".{os.path.basename(dst)}.drive_sync_tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _check_no_type_clash(src_dir, dest, rels):
    for rel in rels:
        target = os.path.join(dest, rel)
        if os.path.isdir(target):
            raise SyncError(f"{rel}: a file in GitHub but a folder in the destination; move it by hand (nothing is deleted)")
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            if os.path.isfile(os.path.join(dest, *parts[:i])):
                raise SyncError(f"{'/'.join(parts[:i])}: a folder in GitHub but a file in the destination; move it by hand")


def sync(ref, dest, repo_url=REPO_URL, token=None, clone=None, dry_run=False, echo=print):
    """Sync ``ref`` of the repo into ``dest`` (overwrite + keep, no delete, no backup). Returns ``(plan, commit_line)``.

    ``clone(ref, workdir) -> (src_dir, commit_line)`` is injectable (tests); the default clones ``repo_url`` with ``token``."""
    if not os.path.isdir(os.path.dirname(os.path.abspath(dest))):
        raise SyncError(f"parent of {dest} does not exist: mount Drive first (drive.mount('/content/drive'))")
    clone = clone or (lambda r, w: clone_ref(r, w, repo_url=repo_url, token=token))
    with tempfile.TemporaryDirectory(prefix="drive_sync_") as work:
        src, line = clone(ref, work)
        sync_record(line)                                            # validates the shape before anything is written
        src_sizes = scan(src)
        dst_sizes = scan(dest, skip_names=(SYNC_FILE,)) if os.path.isdir(dest) else {}
        plan = plan_sync(src_sizes, dst_sizes,
                         lambda rel: sha256_file(os.path.join(src, rel)) == sha256_file(os.path.join(dest, rel)))
        _check_no_type_clash(src, dest, plan.new + plan.updated)
        if not dry_run:
            os.makedirs(dest, exist_ok=True)
            for rel in plan.new + plan.updated:
                _copy_atomic(os.path.join(src, rel), os.path.join(dest, rel))
            with open(os.path.join(dest, SYNC_FILE), "w", encoding="utf-8") as fh:
                fh.write(sync_record(line))
    c = plan.counts()
    for label in ("new", "updated"):
        for rel in getattr(plan, label)[:_LIST_LIMIT]:
            echo(f"  {label:8s} {rel}")
        if len(getattr(plan, label)) > _LIST_LIMIT:
            echo(f"  ... {len(getattr(plan, label)) - _LIST_LIMIT} more {label}")
    echo(f"{'DRY RUN - nothing written: ' if dry_run else ''}new {c['new']} | updated {c['updated']} | unchanged {c['unchanged']} "
         f"| kept only in Drive {c['kept']}")
    echo(f"SYNCED_FROM {line}")
    return plan, line


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sync a GitHub ref into the Drive folder (overwrite + keep, no delete, no backup).")
    ap.add_argument("--ref", required=True, help="branch, tag, or full/short commit SHA")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"Drive folder (default {DEFAULT_DEST})")
    ap.add_argument("--repo-url", default=REPO_URL)
    ap.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = ap.parse_args(argv)
    try:
        sync(args.ref, args.dest, repo_url=args.repo_url, token=resolve_token(), dry_run=args.dry_run)
    except cr.RunnerError as e:
        print(f"drive_sync: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

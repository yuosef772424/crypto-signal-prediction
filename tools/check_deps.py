#!/usr/bin/env python3
"""
PURPOSE:  Enforce the import direction of PHILOSOPHY.md section 2 with `ast` (modules are never imported): code packages must not import research/docs/tests, and a research study must not import another study; path-based coupling is reported as warnings.
TAGS:     dependencies, imports, import direction, zones, architecture, lint, ci, sys.path, اتجاه الاعتماد
PITFALLS: Also hosts the shared repo helpers (Repo, zone_of, file_imports, classify) that tools/build_map.py reuses for DEPENDS. Bare imports (`import lib`) are resolved by a repo-wide module-name index, so only unambiguous names (one zone) are classified. Existing violations are never fixed here: they go in ALLOWLIST below, with a reason.

Usage: python tools/check_deps.py        (exit 1 on any non-allowlisted error)
"""
from __future__ import annotations

import argparse
import ast
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE_PACKAGES = ("cross_asset", "tools", "data", "evaluation", "signal_eval")
CODE_FORBIDDEN = ("research", "docs", "tests")  # code packages must not import these
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".claude"}
ZONE_HEADS = CODE_PACKAGES + ("tests", "docs", "research")

# Calls whose string arguments are inspected for path-style coupling.
SYSPATH_CALLS = {"sys.path.insert", "sys.path.append", "sys.path.extend"}
EXEC_CALLS = {"exec", "execfile", "run_path", "run_module", "load_notebook", "spec_from_file_location"}

# (path, rule) -> reason. Existing violations are allowlisted here, not "fixed" (PHILOSOPHY.md rule 6).
# Keep each entry commented with the reason it exists and when it can go.
ALLOWLIST: dict[tuple[str, str], str] = {
    # ("tools/example.py", "code-must-not-import-research"): "reason, and when it can be removed",
}


# --------------------------------------------------------------------------- repo helpers (shared)
def list_repo_files(root: Path = ROOT) -> list[str]:
    """Files in the git index (posix, relative, sorted); falls back to a walk outside git.
    Index only, not untracked files: the maps must match what CI checks out, so a local scratch file never
    enters them (a new module is mapped once it is `git add`-ed, which the pre-commit hook guarantees)."""
    files: list[str] = []
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached"],
            cwd=root, capture_output=True, check=True,
        ).stdout.decode("utf-8", "surrogateescape")
        files = [f for f in out.split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        for p in root.rglob("*"):
            if p.is_file():
                files.append(p.relative_to(root).as_posix())
    keep = []
    for f in files:
        if any(part in SKIP_DIRS for part in f.split("/")):
            continue
        if (root / f).is_file():
            keep.append(f)
    return sorted(set(keep))


def zone_of(rel: str) -> str:
    """root | cross_asset | tools | data | tests | docs | research/<study> | research | other."""
    parts = rel.split("/")
    if len(parts) == 1:
        return "root"
    head = parts[0]
    if head in CODE_PACKAGES or head in ("tests", "docs"):
        return head
    if head == "research":
        return f"research/{parts[1]}" if len(parts) >= 3 else "research"
    return "other"


class Repo:
    """Index of the repo's files, used to resolve imports to zones."""

    def __init__(self, root: Path = ROOT, files: list[str] | None = None):
        self.root = root
        self.files = files if files is not None else list_repo_files(root)
        self.py_files = [f for f in self.files if f.endswith(".py")]
        fileset = set(self.files)
        self._dir_names: dict[str, set[str]] = {}
        for f in self.files:
            parts = f.split("/")
            for i in range(len(parts) - 1):  # every ancestor directory is a name in its parent
                self._dir_names.setdefault("/".join(parts[:i]), set()).add(parts[i])
            if f.endswith(".py"):
                self._dir_names.setdefault("/".join(parts[:-1]), set()).add(parts[-1][:-3])
        # bare module name -> zones providing it (only dirs that are not regular packages)
        self.bare_index: dict[str, set[str]] = {}
        for f in self.py_files:
            d = posixpath.dirname(f)
            if d and posixpath.join(d, "__init__.py") in fileset:
                continue
            stem = posixpath.basename(f)[:-3]
            if stem == "__init__":
                continue
            self.bare_index.setdefault(stem, set()).add(zone_of(f))

    def own_dir_names(self, rel: str) -> set[str]:
        return self._dir_names.get(posixpath.dirname(rel), set())

    def classify(self, dotted: str, rel: str) -> str | None:
        """Zone targeted by a dotted module name, as seen from file `rel` (None = external/own/unknown)."""
        parts = dotted.split(".")
        head = parts[0]
        if not head:
            return None
        if head in CODE_PACKAGES or head in ("tests", "docs"):
            return head
        if head == "research":
            return f"research/{parts[1]}" if len(parts) > 1 else "research"
        if head in self.own_dir_names(rel) or head in sys.stdlib_module_names:
            return None
        zones = self.bare_index.get(head)
        if zones and len(zones) == 1:
            z = next(iter(zones))
            return None if z == zone_of(rel) else z
        return None


def _package_parts(rel: str) -> list[str]:
    return rel.split("/")[:-1]


def file_imports(tree: ast.AST, rel: str) -> list[tuple[int, str, str]]:
    """(line, dotted target, display text) for every import statement (also nested ones)."""
    out: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((node.lineno, a.name, f"import {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = _package_parts(rel)
                keep = len(pkg) - (node.level - 1)
                if keep < 0:
                    continue
                base_parts = pkg[:keep] + (node.module.split(".") if node.module else [])
                base = ".".join(base_parts)
                dots = "." * node.level + (node.module or "")
            else:
                base = node.module or ""
                dots = base
            for a in node.names:
                shown = f"from {dots} import {a.name}"
                if base in ("", "research"):  # `from research import study` / `from . import mod`
                    target = f"{base}.{a.name}".strip(".")
                else:
                    target = base
                out.append((node.lineno, target, shown))
    return sorted(out)


def dotted_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _string_segments(call: ast.Call) -> list[str]:
    consts = [n for n in ast.walk(call) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    consts.sort(key=lambda n: (n.lineno, n.col_offset))
    segs: list[str] = []
    for c in consts:
        segs.extend(s for s in re.split(r"[\\/]", c.value) if s)
    return segs


def _zones_in_segments(segs: list[str]) -> list[str]:
    zones = []
    for i, s in enumerate(segs):
        if s == "research":
            nxt = segs[i + 1] if i + 1 < len(segs) else ""
            zones.append(f"research/{nxt}" if nxt and nxt not in (".", "..") else "research")
        elif s in ("docs", "tests", "cross_asset", "tools"):
            zones.append(s)
    return zones


def notebook_cells(path: Path) -> tuple[list[tuple[int, str, ast.AST]], list[str], int, int, int]:
    """Parse a notebook's code cells with `ast`.

    Lines starting with `%` or `!` are dropped first; a cell that still fails to parse is skipped
    and counted. Returns (parsed [(cell index, code, tree)], markdown sources, n_cells, n_code, skipped).
    The cell index is the 0-based index into all cells (the one tests/_cell uses).
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    parsed: list[tuple[int, str, ast.AST]] = []
    md: list[str] = []
    n_code = skipped = 0
    cells = nb.get("cells", [])
    for idx, c in enumerate(cells):
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        if c.get("cell_type") == "markdown":
            md.append(src)
        elif c.get("cell_type") == "code":
            n_code += 1
            code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("%", "!")))
            if not code.strip():
                continue
            try:
                parsed.append((idx, code, ast.parse(code)))
            except (SyntaxError, ValueError):
                skipped += 1
    return parsed, md, len(cells), n_code, skipped


# --------------------------------------------------------------------------- the check
def violation_for(own: str, target: str | None) -> str | None:
    """Rule id broken by `own` zone importing `target` zone, else None."""
    if not target or target == own:
        return None
    if own in CODE_PACKAGES and target.split("/")[0] in CODE_FORBIDDEN:
        return f"code-must-not-import-{target.split('/')[0]}"
    if own.startswith("research/") and target.startswith("research/"):
        return "study-must-not-import-other-study"
    return None


def check_file(repo: Repo, rel: str) -> list[tuple[int, str, str, str]]:
    """Return (line, severity, rule, text) for one file."""
    own = zone_of(rel)
    if own not in CODE_PACKAGES and not own.startswith("research/"):
        return []
    try:
        tree = ast.parse((repo.root / rel).read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as e:
        return [(e.lineno or 1, "warning", "syntax-error", f"cannot parse: {e.msg}")]
    found: list[tuple[int, str, str, str]] = []
    for line, dotted, shown in file_imports(tree, rel):
        rule = violation_for(own, repo.classify(dotted, rel))
        if rule:
            found.append((line, "error", rule, shown))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted_name(node.func)
        short = name.rsplit(".", 1)[-1]
        kind = None
        if name in SYSPATH_CALLS:
            kind = "sys-path-coupling"
        elif short in EXEC_CALLS:
            kind = "exec-coupling"
        elif short == "import_module":
            for a in node.args[:1]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    rule = violation_for(own, repo.classify(a.value, rel))
                    if rule:
                        found.append((node.lineno, "warning", "dynamic-import", f"import_module({a.value!r})"))
            continue
        if not kind:
            continue
        for z in _zones_in_segments(_string_segments(node)):
            if violation_for(own, z):
                found.append((node.lineno, "warning", kind, f"{name}(... {z} ...)"))
                break
    return sorted(set(found))


def check_notebook(repo: Repo, rel: str) -> list[tuple[str, int, str, str, str]]:
    """Warnings (never errors) for root notebooks whose code imports research/ or docs/."""
    try:
        parsed = notebook_cells(repo.root / rel)[0]
    except (ValueError, OSError):
        return []
    found = []
    for idx, _code, tree in parsed:
        for line, dotted, shown in file_imports(tree, rel):
            z = repo.classify(dotted, rel)
            if z and z.split("/")[0] in ("research", "docs"):
                found.append((f"{rel}#cell{idx}", line, "warning", f"notebook-imports-{z.split('/')[0]}", shown))
    return sorted(set(found))


def run(repo: Repo) -> tuple[list[str], int, int, int, int]:
    lines: list[str] = []
    errors = warnings = allowed = 0
    used: set[tuple[str, str]] = set()
    for rel in repo.py_files:
        for line, sev, rule, text in check_file(repo, rel):
            key = (rel, rule)
            if key in ALLOWLIST:
                used.add(key)
                allowed += 1
                continue
            lines.append(f"{rel}:{line}: {sev}[{rule}] — {text}")
            if sev == "error":
                errors += 1
            else:
                warnings += 1
    notebooks = [f for f in repo.files if "/" not in f and f.endswith(".ipynb")]
    for rel in notebooks:
        for where, line, sev, rule, text in check_notebook(repo, rel):
            lines.append(f"{where}:{line}: {sev}[{rule}] — {text}")
            warnings += 1
    for key in sorted(set(ALLOWLIST) - used):
        lines.append(f"{key[0]}:1: warning[stale-allowlist] — no violation of {key[1]} any more; remove the entry")
        warnings += 1
    return lines, len(repo.py_files) + len(notebooks), errors, warnings, allowed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check import direction between repo zones (PHILOSOPHY.md section 2).")
    ap.add_argument("--root", default=str(ROOT), help="repo root (default: the repo containing this script)")
    args = ap.parse_args(argv)
    repo = Repo(Path(args.root).resolve())
    lines, n_files, errors, warnings, allowed = run(repo)
    for ln in lines:
        print(ln)
    print(f"check_deps: {n_files} files (.py + root notebooks), {errors} errors, {warnings} warnings, {allowed} allowlisted")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

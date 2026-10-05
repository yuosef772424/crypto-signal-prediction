#!/usr/bin/env python3
"""
PURPOSE:  Reconstruct the IMPLICIT dependency graph of the shared-namespace packages (data, model, trainer, evaluation, signal_eval, workflow, discovery) with `symtable`/`ast`: per module, the free names it uses and which module defines each (same package earlier or later, another package, or nothing = a notebook global), plus top-level names defined in two modules; gates NEW findings against tools/implicit_deps_allowlist.txt.
TAGS:     implicit dependencies, shared namespace, free names, symtable, forward reference, cross-package, notebook globals, duplicate definitions, shadowing, allowlist ratchet, uses, used by, load_into, MODULES
PITFALLS: Modules are never imported or executed; MODULES order comes from each <pkg>/_loader.py (literal tuple). Names bound by `import numpy as np` etc. are `import` kind: shown as "shared imports" and never gated; names imported FROM a repo package (e.g. `from core.schema import X`) are `alias` kind: real definitions for the reader (other modules may use them implicitly) but never duplicate-conflicts. Dynamic access (`globals()[...]`, `globals().get`) is invisible to the analysis; it is only counted. Stdlib only (CI `structure` job installs nothing). The allowlist records today's findings and may only shrink: a finding that is gone must be removed (`--prune-allowlist`), a new one fails tools/check_deps.py.

Usage:
  python tools/implicit_deps.py                     per-package summary (counts per kind)
  python tools/implicit_deps.py <module>            uses / used-by of one module (path `data/windows.py` or `data/windows`)
  python tools/implicit_deps.py --findings          the gated findings, in allowlist format
  python tools/implicit_deps.py --write-allowlist   (re)write tools/implicit_deps_allowlist.txt = exactly today's findings
  python tools/implicit_deps.py --prune-allowlist   drop stale allowlist entries only (never adds)
"""
from __future__ import annotations

import argparse
import ast
import builtins
import symtable
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Packages whose modules are exec'd into one namespace dict by <pkg>/_loader.py (order = the merge order of main.ipynb).
SHARED_PACKAGES = ("data", "model", "trainer", "evaluation", "signal_eval", "workflow", "discovery")
ALLOWLIST_PATH = "tools/implicit_deps_allowlist.txt"

#: Finding kinds that fail the check when not allowlisted (the others are informational).
GATED_REF_KINDS = ("cross", "forward", "notebook")
GATED_KINDS = GATED_REF_KINDS + ("dup",)
_BUILTINS = frozenset(dir(builtins)) | {"__name__", "__file__", "__doc__", "__builtins__", "__package__", "__spec__"}


@dataclass
class ModInfo:
    path: str  # data/windows.py
    pkg: str
    name: str  # windows
    index: int  # position in the package MODULES
    defs: dict[str, str] = field(default_factory=dict)  # name -> def | class | assign | alias | import
    free: dict[str, int] = field(default_factory=dict)  # free name -> first line of use
    dynamic: int = 0  # number of globals() calls (invisible to this analysis)


@dataclass(frozen=True)
class Ref:
    """One free name of `module` and where it resolves.
    kind: same (earlier module, same package) | import (shared import of an EARLIER module, e.g. np from common) |
          forward (LATER module, same package; also for imports) | cross (other package; also for imports) |
          notebook (no definition anywhere = a notebook global)."""
    module: str
    name: str
    kind: str
    target: str  # defining module path ('' for notebook)
    line: int

    @property
    def key(self) -> str:
        return f"{self.kind} {self.module} {self.name}"


@dataclass
class Analysis:
    root: Path
    mods: dict[str, ModInfo]  # path -> info, in package/load order
    refs: list[Ref]

    def uses(self, path: str) -> list[Ref]:
        return [r for r in self.refs if r.module == path]

    def used_by(self, path: str) -> dict[str, list[str]]:
        """Modules that use a name defined in `path` (real definitions only; shared imports excluded): module -> names."""
        out: dict[str, list[str]] = {}
        for r in self.refs:
            if r.target == path and r.kind in ("same", "forward", "cross"):
                out.setdefault(r.module, []).append(r.name)
        return {m: sorted(set(n)) for m, n in sorted(out.items())}


# --------------------------------------------------------------------------- reading the loaders
def package_modules(root: Path, pkg: str) -> list[str]:
    """MODULES tuple of <pkg>/_loader.py (read with ast, never imported); [] if the package has no loader."""
    loader = root / pkg / "_loader.py"
    if not loader.is_file():
        return []
    tree = ast.parse(loader.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODULES" for t in node.targets):
            return list(ast.literal_eval(node.value))
    return []


def _repo_packages(root: Path) -> set[str]:
    return {p.parent.name for p in root.glob("*/__init__.py")}


# --------------------------------------------------------------------------- per-module analysis
def _bound_kinds(body: list[ast.stmt], repo_pkgs: set[str]) -> dict[str, str]:
    """name -> kind for names bound by the module's top-level statements (also inside if/try/for/with, not in defs)."""
    out: dict[str, str] = {}

    def target_names(t: ast.AST):
        if isinstance(t, ast.Name):
            yield t.id
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                yield from target_names(e)
        elif isinstance(t, ast.Starred):
            yield from target_names(t.value)

    def walk(stmts: list[ast.stmt]) -> None:
        for n in stmts:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[n.name] = "def"
            elif isinstance(n, ast.ClassDef):
                out[n.name] = "class"
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    for nm in target_names(t):
                        out[nm] = "assign"
            elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                for nm in target_names(n.target):
                    out.setdefault(nm, "assign")
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                top = (n.module or "").split(".")[0] if isinstance(n, ast.ImportFrom) else None
                for a in n.names:
                    if isinstance(n, ast.Import):
                        nm = a.asname or a.name.split(".")[0]
                        kind = "alias" if a.name.split(".")[0] in repo_pkgs else "import"
                    else:
                        nm = a.asname or a.name
                        kind = "alias" if (n.level == 0 and top in repo_pkgs) else "import"
                    if nm != "*":
                        out[nm] = kind
            else:
                for attr in ("body", "orelse", "finalbody"):
                    walk(getattr(n, attr, []) or [])
                for h in getattr(n, "handlers", []) or []:
                    walk(h.body)
                if isinstance(n, (ast.For, ast.AsyncFor)):
                    for nm in target_names(n.target):
                        out.setdefault(nm, "assign")
                if isinstance(n, (ast.With, ast.AsyncWith)):
                    for it in n.items:
                        if it.optional_vars is not None:
                            for nm in target_names(it.optional_vars):
                                out.setdefault(nm, "assign")

    walk(body)
    return out


def analyse_source(source: str, filename: str, repo_pkgs: frozenset[str] | set[str] = frozenset()) -> tuple[dict[str, str], dict[str, int], int]:
    """(defs, free names -> first line, number of globals() calls) of one module's source."""
    tree = ast.parse(source, filename)
    kinds = _bound_kinds(tree.body, set(repo_pkgs))
    table = symtable.symtable(source, filename, "exec")
    defs: dict[str, str] = {}
    candidates: set[str] = set()

    def visit(t: symtable.SymbolTable, is_module: bool) -> None:
        for s in t.get_symbols():
            nm = s.get_name()
            if is_module:
                if s.is_assigned() or s.is_imported() or s.is_namespace():
                    defs[nm] = kinds.get(nm, "assign")
                elif s.is_referenced():
                    candidates.add(nm)
            else:
                if s.is_global():
                    if s.is_assigned() and s.is_declared_global():
                        defs.setdefault(nm, "assign")  # `global X; X = ...` inside a function defines a module name
                    if s.is_referenced():
                        candidates.add(nm)
        for c in t.get_children():
            visit(c, False)

    visit(table, True)
    free_names = {n for n in candidates if n not in defs and n not in _BUILTINS}
    first: dict[str, int] = {}
    n_dynamic = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in free_names:
            first[n.id] = min(first.get(n.id, n.lineno), n.lineno)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "globals":
            n_dynamic += 1
    return defs, {n: first.get(n, 1) for n in sorted(free_names)}, n_dynamic


def analyse(root: Path = ROOT) -> Analysis:
    root = Path(root)
    repo_pkgs = _repo_packages(root)
    mods: dict[str, ModInfo] = {}
    for pkg in SHARED_PACKAGES:
        for i, name in enumerate(package_modules(root, pkg)):
            f = root / pkg / f"{name}.py"
            if not f.is_file():
                continue
            rel = f"{pkg}/{name}.py"
            defs, free, dyn = analyse_source(f.read_text(encoding="utf-8"), rel, repo_pkgs)
            mods[rel] = ModInfo(rel, pkg, name, i, defs, free, dyn)
    index: dict[str, list[ModInfo]] = {}
    for m in mods.values():
        for nm in m.defs:
            index.setdefault(nm, []).append(m)
    pkg_order = {p: i for i, p in enumerate(SHARED_PACKAGES)}
    refs: list[Ref] = []
    for m in mods.values():
        for nm, line in m.free.items():
            cands = index.get(nm, [])
            same = [c for c in cands if c.pkg == m.pkg]
            earlier = [c for c in same if c.index < m.index]
            later = [c for c in same if c.index > m.index]
            if earlier:
                tgt = earlier[-1]  # the last earlier definition is the one in force at call time
                kind = "import" if tgt.defs[nm] == "import" else "same"
            elif later:
                tgt = later[0]
                kind = "forward"
            elif cands:
                tgt = sorted(cands, key=lambda c: (pkg_order[c.pkg], c.index))[0]
                kind = "cross"
            else:
                refs.append(Ref(m.path, nm, "notebook", "", line))
                continue
            refs.append(Ref(m.path, nm, kind, tgt.path, line))
    return Analysis(root, mods, refs)


# --------------------------------------------------------------------------- findings and the allowlist
def duplicates(an: Analysis) -> dict[str, list[str]]:
    """Top-level names defined (def/class/assign, not imports or aliases) in two or more modules of the shared namespaces."""
    by_name: dict[str, list[str]] = {}
    for m in an.mods.values():
        for nm, kind in m.defs.items():
            if kind in ("def", "class", "assign") and not (nm.startswith("__") and nm.endswith("__")):
                by_name.setdefault(nm, []).append(m.path)
    return {nm: sorted(ps) for nm, ps in sorted(by_name.items()) if len(ps) >= 2}


def findings(an: Analysis) -> dict[str, str]:
    """Gated findings: allowlist key -> human text (key is what the allowlist stores, text adds the target/line)."""
    out: dict[str, str] = {}
    for r in an.refs:
        if r.kind in GATED_REF_KINDS:
            what = {"cross": f"uses {r.name} defined in another package ({r.target})",
                    "forward": f"uses {r.name} defined in a LATER module ({r.target})",
                    "notebook": f"uses {r.name}, defined by no module (a notebook global)"}[r.kind]
            out[r.key] = f"{r.module}:{r.line}: {what}"
    for nm, paths in duplicates(an).items():
        out[f"dup {nm} : {', '.join(paths)}"] = f"{nm} is defined in {len(paths)} modules: {', '.join(paths)}"
    return out


def _target_note(an: Analysis, key: str) -> str:
    kind = key.split(" ", 1)[0]
    if kind in GATED_REF_KINDS:
        for r in an.refs:
            if r.key == key:
                return f" -> {r.target}" if r.target else ""
    return ""


def read_allowlist(root: Path = ROOT) -> list[str]:
    p = Path(root) / ALLOWLIST_PATH
    if not p.is_file():
        return []
    keys = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            keys.append(ln.split(" -> ", 1)[0].strip())
    return keys


def allowlist_text(an: Analysis, keys: list[str]) -> str:
    head = [
        "# Implicit-dependency debt of the shared-namespace packages (tools/implicit_deps.py), as found when S0 landed.",
        "# One finding per line: `<kind> <module> <name> [-> <defining module>]` for cross / forward / notebook references,",
        "# `dup <name> : <modules>` for a name defined in several modules. This list may only shrink: tools/check_deps.py",
        "# fails on a finding that is not listed AND on a listed finding that no longer exists (python tools/implicit_deps.py",
        "# --prune-allowlist removes those). Never add a line by hand to silence a new dependency: fix the dependency.",
        "",
    ]
    order = {k: i for i, k in enumerate(GATED_KINDS)}
    body = [k + _target_note(an, k) for k in sorted(keys, key=lambda k: (order[k.split(" ", 1)[0]], k))]
    return "\n".join(head + body) + "\n"


def check(root: Path = ROOT) -> tuple[list[str], int]:
    """(problems, number allowlisted). A problem is a new finding or a stale allowlist entry."""
    an = analyse(root)
    found = findings(an)
    allow = set(read_allowlist(root))
    problems = [f"{text}  [new implicit dependency: import it explicitly, or move the definition]"
                for key, text in found.items() if key not in allow]
    problems += [f"{ALLOWLIST_PATH}: stale entry (finding is gone; remove it, e.g. `python tools/implicit_deps.py "
                 f"--prune-allowlist`): {key}" for key in sorted(allow - set(found))]
    return sorted(problems), len(allow & set(found))


# --------------------------------------------------------------------------- compact view for maps/<pkg>.md
def short_module(path: str, from_pkg: str) -> str:
    pkg, name = path[:-3].split("/", 1)
    return name if pkg == from_pkg else f"{pkg}/{name}"


def map_lines(an: Analysis, path: str, max_used_by: int = 10) -> list[str]:
    """`- USES:` / `- NOTEBOOK-GLOBALS:` / `- USED BY:` lines for one module (module names only; `*` = defined in a LATER
    module, `pkg/mod` = another package; shared imports such as np/pd are omitted)."""
    m = an.mods.get(path)
    if m is None:
        return []
    uses: dict[str, bool] = {}
    nb: list[str] = []
    for r in an.uses(path):
        if r.kind in ("same", "forward", "cross"):
            label = short_module(r.target, m.pkg)
            uses[label] = uses.get(label, False) or r.kind == "forward"
        elif r.kind == "notebook":
            nb.append(r.name)
    out = []
    if uses:
        out.append("- USES: " + ", ".join(k + ("*" if fwd else "") for k, fwd in sorted(uses.items())))
    if nb:
        out.append("- NOTEBOOK-GLOBALS: " + ", ".join(sorted(nb)))
    users = [short_module(p, m.pkg) for p in an.used_by(path) if p != path]
    if users:
        shown = ", ".join(users[:max_used_by]) + (f" +{len(users) - max_used_by}" if len(users) > max_used_by else "")
        out.append(f"- USED BY: {shown}")
    return out


LEGEND = ("implicit deps (shared namespace, tools/implicit_deps.py): USES / USED BY list module names; `*` = defined in a "
          "LATER module, `pkg/mod` = another package; shared imports (np, pd, ...) omitted; NOTEBOOK-GLOBALS = names no module defines")


# --------------------------------------------------------------------------- CLI
def _summary(an: Analysis) -> str:
    kinds = ("same", "import", "forward", "cross", "notebook")
    L = []
    for pkg in SHARED_PACKAGES:
        ms = [m for m in an.mods.values() if m.pkg == pkg]
        if not ms:
            continue
        counts = {k: sum(1 for r in an.refs if r.kind == k and an.mods[r.module].pkg == pkg) for k in kinds}
        dyn = sum(m.dynamic for m in ms)
        L.append(f"{pkg:12} {len(ms):3} modules  " + "  ".join(f"{k}={v}" for k, v in counts.items()) + f"  globals()-calls={dyn}")
    d = duplicates(an)
    L.append(f"duplicate top-level definitions: {len(d)} names")
    return "\n".join(L)


def _resolve_module(an: Analysis, arg: str) -> str | None:
    cand = arg if arg.endswith(".py") else arg + ".py"
    if cand in an.mods:
        return cand
    hits = [p for p in an.mods if p.endswith("/" + cand)]
    return hits[0] if len(hits) == 1 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Implicit dependencies of the shared-namespace packages.")
    ap.add_argument("module", nargs="?", help="module to show (data/windows.py, data/windows or a unique bare name)")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--findings", action="store_true", help="print the gated findings in allowlist format")
    ap.add_argument("--write-allowlist", action="store_true", help=f"rewrite {ALLOWLIST_PATH} = exactly today's findings")
    ap.add_argument("--prune-allowlist", action="store_true", help="remove stale allowlist entries (never adds one)")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    an = analyse(root)
    if args.write_allowlist or args.prune_allowlist:
        keys = sorted(findings(an))
        if args.prune_allowlist:
            keys = sorted(set(keys) & set(read_allowlist(root)))
        (root / ALLOWLIST_PATH).write_text(allowlist_text(an, keys), encoding="utf-8", newline="\n")
        print(f"implicit_deps: wrote {ALLOWLIST_PATH} ({len(keys)} entries)")
        return 0
    if args.findings:
        for k in sorted(findings(an)):
            print(k + _target_note(an, k))
        return 0
    if args.module:
        path = _resolve_module(an, args.module)
        if path is None:
            print(f"unknown or ambiguous module {args.module!r}", file=sys.stderr)
            return 2
        m = an.mods[path]
        print(f"{path}: {len(m.defs)} top-level names, {len(m.free)} free names, {m.dynamic} globals() calls")
        for r in an.uses(path):
            print(f"  L{r.line:<5} {r.kind:12} {r.name}" + (f"  <- {r.target}" if r.target else ""))
        for mod, names in an.used_by(path).items():
            print(f"  used by {mod}: {', '.join(names)}")
        return 0
    print(_summary(an))
    problems, allowed = check(root)
    print(f"gated findings: {len(findings(an))} ({allowed} allowlisted, {len(problems)} problems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

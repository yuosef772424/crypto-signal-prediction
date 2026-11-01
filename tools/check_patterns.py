#!/usr/bin/env python3
"""
PURPOSE:  Generic pattern RATCHET (hardening audit C8): each rule = (name, matcher, file globs, exclusions); the per-rule per-file match counts of TODAY are frozen in tools/pattern_baseline.json, a count above the baseline fails (a new instance of an anti-pattern), a count below it also fails until `--update-baseline` records the shrink; the baseline can only shrink, never grow. Rules (AST matchers; each documents the audit class it guards): output-name-literal (C1), hlc-target-tuple (C1), config-get-default (C2), target-mode-branch (C4), seed-literal (C3), metric-def-outside-stats (C7), per-asset-model-call-loop (C5).
TAGS:     pattern ratchet, baseline, anti-pattern guard, ast, lint, ci, output name literal, y_ target head names, HLC tuple, .get default, config default, mode name branch, seed literal, metric definition, per-asset predict loop, hardening, frozen counts, only shrinks
PITFALLS: Counts are per rule per FILE (no line numbers) so unrelated edits above a finding never change the baseline. Moving a finding from file A to file B fails twice (B grew); that is intended: migrate, do not relocate. `--update-baseline` REFUSES to record growth (a new violation is fixed, never baselined by hand), it only lowers counts, drops files that reached zero, and creates the baseline when absent (the one-time freeze). A rule added later gets its frozen counts through `--freeze-rule <name>` (adds that rule only, refuses if the rule is already in the baseline). Matchers are static heuristics, not proofs (receiver names, name tokens): they over-count rather than under-count on purpose, and a rule changes only together with a re-freeze reviewed in the PR. Scope = .py files of the code packages (core, cross_asset, tools, data, model, trainer, evaluation, signal_eval, workflow, discovery) from `git ls-files` plus untracked non-ignored files; tests/, research/, docs/ and notebooks are never scanned, and this file is excluded from its own rules; seed-literal also skips in-package self-test modules (*selftest*, *smoke_test*). Docstrings are not counted as string literals. Stdlib only (CI `structure` job installs nothing). Wired into tools/check_deps.py (so CI and the pre-commit hook run it).

Usage:
  python tools/check_patterns.py                      check against tools/pattern_baseline.json (exit 1 on growth or stale entry)
  python tools/check_patterns.py --list [<rule>]      every match with file:line (to find what to migrate)
  python tools/check_patterns.py --summary            per-rule totals (baseline vs now)
  python tools/check_patterns.py --update-baseline    record SHRINKS only (refuses growth); creates the baseline if absent
  python tools/check_patterns.py --freeze-rule <name> add one new rule's current counts to the baseline
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = "tools/pattern_baseline.json"
BASELINE_VERSION = 1

CODE_PACKAGES = ("core", "stats", "cross_asset", "tools", "data", "model", "trainer", "evaluation", "signal_eval", "workflow",
                 "discovery")
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".claude"}
_ALL_CODE = tuple(f"{p}/**/*.py" for p in CODE_PACKAGES) + tuple(f"{p}/*.py" for p in CODE_PACKAGES)

#: (line, short text) of one match
Match = tuple[int, str]


@dataclass(frozen=True)
class Rule:
    """One ratchet rule. Exactly one of `matcher` (AST: (tree, source) -> matches) or `regex` (per source line) is set."""
    name: str
    guards: str                       # the audit class this rule fences (e.g. "C1")
    purpose: str
    globs: tuple[str, ...] = _ALL_CODE
    exclude: tuple[str, ...] = ()
    matcher: Callable[[ast.AST, str], list[Match]] | None = None
    regex: str | None = None

    def __post_init__(self):
        if (self.matcher is None) == (self.regex is None):
            raise ValueError(f"rule {self.name!r}: set exactly one of matcher / regex")

    def applies_to(self, rel: str) -> bool:
        return (any(_glob(rel, g) for g in self.globs) and not any(_glob(rel, g) for g in self.exclude))

    def find(self, rel: str, source: str) -> list[Match]:
        if self.regex is not None:
            rx = re.compile(self.regex)
            return [(i, ln.strip()[:80]) for i, ln in enumerate(source.splitlines(), 1) if rx.search(ln)]
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            return []
        assert self.matcher is not None
        return sorted(self.matcher(tree, source))


def _glob(rel: str, pattern: str) -> bool:
    """fnmatch where `**/` also matches zero directories (so `pkg/**/*.py` matches `pkg/x.py`)."""
    if fnmatch.fnmatchcase(rel, pattern):
        return True
    if "**/" in pattern:
        return fnmatch.fnmatchcase(rel, pattern.replace("**/", "")) or fnmatch.fnmatchcase(rel, pattern.replace("**/", "*/"))
    return False


# --------------------------------------------------------------------------- AST helpers
def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    elif isinstance(node, ast.Call):  # a.b().c -> keep the tail readable
        parts.append(_dotted(node.func) + "()")
    return ".".join(reversed(parts))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _string_templates(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, text) of every non-docstring string literal; an f-string is ONE item with `{}` for each placeholder, and
    `"..{x}..".format` / `%s` style placeholders are normalised to `{}` too."""
    doc = _docstring_nodes(tree)
    inside_fstring: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                inside_fstring.add(id(v))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        text = None
        if isinstance(node, ast.JoinedStr):
            text = "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}" for v in node.values)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in doc or id(node) in inside_fstring:
                continue
            text = node.value
        if text is None:
            continue
        text = re.sub(r"\{[^{}]*\}", "{}", text)
        text = re.sub(r"%\(?\w*\)?[sd]", "{}", text)
        out.append((node.lineno, text))
    return out


# --------------------------------------------------------------------------- matchers
_HEAD_SUFFIXES = ("class_logits", "nu", "alpha", "beta", "confidence", "epistemic", "aleatoric")
_OUTPUT_KEY_RX = re.compile(r"(?:^|[^A-Za-z0-9])y_(?:\{\}|[A-Za-z]+)_(?:%s)(?:$|[^A-Za-z0-9])" % "|".join(_HEAD_SUFFIXES))


def match_output_name_literal(tree: ast.AST, source: str) -> list[Match]:
    """A string literal / f-string that spells a model output key (`y_{t}_class_logits`, `y_high_nu`, ...) or contains
    `_class_logits` (string slicing / `.endswith("_class_logits")`)."""
    out = []
    for line, text in _string_templates(tree):
        if _OUTPUT_KEY_RX.search(text) or "_class_logits" in text:
            out.append((line, text[:60]))
    return out


_HLC = frozenset({"high", "low", "close"})


def match_hlc_target_tuple(tree: ast.AST, source: str) -> list[Match]:
    """A literal tuple/list/set of exactly the three strings high, low, close (any order): core.schema.TARGET_COLUMNS."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)) and len(node.elts) == 3 \
                and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts) \
                and {e.value for e in node.elts} == _HLC:
            out.append((node.lineno, "(" + ",".join(e.value for e in node.elts) + ")"))
    return out


_CONFIG_RECEIVER_RX = re.compile(r"(cfg|config|conf|settings|params|opts|options|kwargs|section)", re.I)


def _receiver_names(recv: ast.AST) -> list[str]:
    """Names that describe a `.get` receiver: its last segment; for `x[...]` the base's last segment and a string key; for a
    chained `cfg.get("a", {}).get(...)` the inner receiver."""
    def last(n: ast.AST) -> str:
        return n.attr if isinstance(n, ast.Attribute) else n.id if isinstance(n, ast.Name) else ""
    names = [last(recv)]
    if isinstance(recv, ast.Subscript):
        names.append(last(recv.value))
        if isinstance(recv.slice, ast.Constant) and isinstance(recv.slice.value, str):
            names.append(recv.slice.value)
    elif isinstance(recv, ast.Call) and isinstance(recv.func, ast.Attribute):
        names.extend(_receiver_names(recv.func.value))
    return [n for n in names if n]


def match_config_get_default(tree: ast.AST, source: str) -> list[Match]:
    """`<config-like>.get("key", default)` with a non-None default: a default living at the call site instead of in the
    one config schema. The receiver counts as config-like when its last name segment contains cfg/config/conf/settings/
    params/opts/options/kwargs/section (a heuristic: it over-counts rather than misses)."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
            continue
        default = node.args[1] if len(node.args) >= 2 else next((k.value for k in node.keywords if k.arg == "default"), None)
        if default is None or (isinstance(default, ast.Constant) and default.value is None):
            continue
        if not node.args or not (isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            continue
        if any(_CONFIG_RECEIVER_RX.search(n) for n in _receiver_names(node.func.value)):
            out.append((node.lineno, f".get({node.args[0].value!r}, ...)"))
    return out


#: core.schema.TARGET_MODES names + "+relative". The distinctive ones always count; the common words (scaled, magnitude,
#: return, relative) count only when compared with a mode-like variable (name contains "mode" or "base").
_MODE_DISTINCT = frozenset({"return_close", "volnorm", "entry_range"})
_MODE_COMMON = frozenset({"scaled", "magnitude", "return", "relative"})
_MODE_VAR_RX = re.compile(r"(mode|base)", re.I)


def _mode_const(node: ast.AST, other: ast.AST) -> str | None:
    values: list[str] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        values = [node.value]
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    for v in values:
        base = v.split("+", 1)[0]
        if base in _MODE_DISTINCT:
            return v
        if (base in _MODE_COMMON or v == "return+relative") and _MODE_VAR_RX.search(ast.unparse(other)):
            return v
    return None


def match_target_mode_branch(tree: ast.AST, source: str) -> list[Match]:
    """A comparison or `startswith/endswith` against a target-mode NAME (`mode == "entry_range"`, `base in ("scaled", ..)`,
    `mode.startswith("entry_range")`): branching on a mode name outside the one component that owns the modes.
    `scaled` / `magnitude` / `return` / `relative` count only when compared with a mode-like variable."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sides = [node.left] + list(node.comparators)
            for i, s in enumerate(sides):
                if not isinstance(node.ops[min(i, len(node.ops) - 1)], (ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Is, ast.IsNot)):
                    continue
                other = sides[0] if i else sides[1]
                hit = _mode_const(s, other)
                if hit:
                    out.append((node.lineno, f"{ast.unparse(other)[:30]} vs {hit!r}"))
                    break
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("startswith", "endswith") \
                and node.args:
            arg = node.args[0]
            consts = [arg] if isinstance(arg, ast.Constant) else list(getattr(arg, "elts", []))
            for c in consts:
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and (c.value.lstrip("+").split("+", 1)[0] in _MODE_DISTINCT or c.value == "+relative"):
                    out.append((node.lineno, f".{node.func.attr}({c.value!r})"))
                    break
    return out


_SEED_FUNCS = frozenset({"default_rng", "set_random_seed", "set_seed", "manual_seed", "seed", "RandomState", "Random",
                         "SeedSequence"})


def _is_int_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool)


def match_seed_literal(tree: ast.AST, source: str) -> list[Match]:
    """A seed fixed as an integer literal: `default_rng(42)`, `set_random_seed(0)`, `np.random.seed(1)`, `RandomState(7)`,
    or any call with `seed=<int literal>`. Seeds must come from one place (a setting), not from call sites."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        short = name.rsplit(".", 1)[-1]
        if short in _SEED_FUNCS and node.args and _is_int_literal(node.args[0]):
            out.append((node.lineno, f"{name}({ast.unparse(node.args[0])})"))
            continue
        for kw in node.keywords:
            if kw.arg == "seed" and _is_int_literal(kw.value):
                out.append((node.lineno, f"{name}(seed={ast.unparse(kw.value)})"))
                break
    return out


_METRIC_TOKENS = ("auc", "roc_auc", "rank_ic", "ic", "spearman", "pearson", "wilson", "pinball", "ece")
_METRIC_NAME_RX = re.compile(r"(?:^|_)(?:%s)(?:_|$)" % "|".join(_METRIC_TOKENS))


def match_metric_def_outside_stats(tree: ast.AST, source: str) -> list[Match]:
    """A function (any nesting, incl. methods) whose name has a metric token (`auc`, `roc_auc`, `rank_ic`, `ic`, `spearman`,
    `pearson`, `wilson`, `pinball`, `ece`) between underscores (test_*/t_* helpers skipped): one definition of each belongs in a future `stats/`."""
    return [(n.lineno, f"def {n.name}") for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith(("test_", "t_"))
            and _METRIC_NAME_RX.search(n.name.lower())]


_ASSET_ITER_RX = re.compile(r"(asset|coin|symbol|test_dict|per_asset|ticker)", re.I)
_MODEL_NAMES = frozenset({"model", "model_tf", "self.model", "self._model", "self.model_tf"})


def _model_call(call: ast.Call) -> str | None:
    name = _dotted(call.func)
    short = name.rsplit(".", 1)[-1]
    if short.startswith("predict"):
        return name
    if name in _MODEL_NAMES:
        return name
    return None


def match_per_asset_model_call_loop(tree: ast.AST, source: str) -> list[Match]:
    """A `for` loop (or comprehension) over assets/coins whose body calls a model (`*predict*(...)` or `model(...)`):
    one model call per asset instead of one pooled call sliced by asset bounds."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            head = f"{ast.unparse(node.target)} in {ast.unparse(node.iter)}"
            body: Iterable[ast.AST] = node.body + node.orelse
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            head = " ".join(f"{ast.unparse(g.target)} in {ast.unparse(g.iter)}" for g in node.generators)
            body = [node]
        else:
            continue
        if not _ASSET_ITER_RX.search(head):
            continue
        for stmt in body:
            hit = next((_model_call(c) for c in ast.walk(stmt) if isinstance(c, ast.Call) and _model_call(c)), None)
            if hit:
                out.append((node.lineno, f"for {head[:40]}: {hit}(...)"))
                break
    return out


# --------------------------------------------------------------------------- the rules
_NOT_SELF = ("tools/check_patterns.py",)
_SELFTESTS = ("*selftest*", "*smoke_test*")  # in-package self-tests: fixtures with fixed seeds are legitimate
_CONFIG_PKGS = tuple(f"{p}/**/*.py" for p in ("cross_asset", "data", "model", "trainer", "evaluation", "signal_eval",
                                              "workflow", "discovery")) \
    + tuple(f"{p}/*.py" for p in ("cross_asset", "data", "model", "trainer", "evaluation", "signal_eval", "workflow", "discovery"))

RULES: tuple[Rule, ...] = (
    Rule("output-name-literal", "C1",
         "Model output keys (`y_{t}_class_logits`, `..._nu/_alpha/_beta/_confidence/_epistemic/_aleatoric`) are spelled "
         "in many files; one owner (core/heads.py, excluded from this rule) formats them. Counts string literals/f-strings.",
         exclude=_NOT_SELF + ("core/heads.py",), matcher=match_output_name_literal),
    Rule("hlc-target-tuple", "C1",
         "The literal ('high','low','close') repeated instead of core.schema.TARGET_COLUMNS.",
         exclude=_NOT_SELF + ("core/schema.py",), matcher=match_hlc_target_tuple),
    Rule("config-get-default", "C2",
         "`cfg.get('key', default)` in config-consuming packages: defaults belong in ONE schema, a call-site default "
         "is a silent default (a typo'd key is accepted and the dead key hides).",
         globs=_CONFIG_PKGS, exclude=_NOT_SELF, matcher=match_config_get_default),
    Rule("target-mode-branch", "C4",
         "`mode == '<target mode>'` / `startswith('entry_range')` outside the component that owns target modes "
         "(a future TargetCodec): every new mode otherwise edits N modules.",
         exclude=_NOT_SELF + ("core/schema.py", "stats/target_codec.py"), matcher=match_target_mode_branch),
    Rule("seed-literal", "C3",
         "Integer seed literals at call sites (`default_rng(42)`, `set_random_seed(0)`, `seed=7`): seeds must be one "
         "setting threaded from one place (multi-seed runs, run identity). In-package self-tests / smoke tests are not scanned.",
         exclude=_NOT_SELF + _SELFTESTS, matcher=match_seed_literal),
    Rule("metric-def-outside-stats", "C7",
         "Metric functions (auc / rank-IC / spearman / pearson / wilson / pinball / ece) defined outside a future "
         "`stats/`: one definition each, with one tie/NaN policy.",
         exclude=_NOT_SELF + ("stats/*.py", "stats/**/*.py"), matcher=match_metric_def_outside_stats),
    Rule("per-asset-model-call-loop", "C5",
         "A model call inside a loop over assets (evaluation cost O(N coins) model calls); use one pooled predict "
         "sliced by asset bounds (workflow/pooling.py).",
         exclude=_NOT_SELF, matcher=match_per_asset_model_call_loop),
)
RULES_BY_NAME = {r.name: r for r in RULES}


# --------------------------------------------------------------------------- scan
def list_files(root: Path) -> list[str]:
    """Posix relative paths of tracked + untracked-not-ignored files (so a brand-new file is scanned before `git add`);
    walks the tree outside git."""
    files: list[str] = []
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
                             capture_output=True, check=True).stdout.decode("utf-8", "surrogateescape")
        files = [f for f in out.split("\0") if f]
    except (OSError, subprocess.CalledProcessError):
        files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    return sorted({f for f in files if f.endswith(".py") and not any(part in SKIP_DIRS for part in f.split("/"))
                   and (root / f).is_file()})


def scan(root: Path = ROOT, rules: Iterable[Rule] = RULES, files: list[str] | None = None) -> dict[str, dict[str, list[Match]]]:
    """rule name -> file -> matches (only files with at least one match)."""
    rules = tuple(rules)
    files = list_files(root) if files is None else files
    found: dict[str, dict[str, list[Match]]] = {r.name: {} for r in rules}
    for rel in files:
        applicable = [r for r in rules if r.applies_to(rel)]
        if not applicable:
            continue
        source = (root / rel).read_text(encoding="utf-8", errors="replace")
        for r in applicable:
            m = r.find(rel, source)
            if m:
                found[r.name][rel] = m
    return found


def counts_of(found: dict[str, dict[str, list[Match]]]) -> dict[str, dict[str, int]]:
    return {rule: {f: len(m) for f, m in sorted(files.items())} for rule, files in found.items()}


# --------------------------------------------------------------------------- baseline
def read_baseline(root: Path = ROOT) -> dict[str, dict[str, int]] | None:
    p = root / BASELINE_PATH
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("version") != BASELINE_VERSION:
        raise ValueError(f"{BASELINE_PATH}: unsupported version {data.get('version')!r}")
    return {rule: {f: int(n) for f, n in files.items()} for rule, files in data["rules"].items()}


def format_baseline(counts: dict[str, dict[str, int]]) -> str:
    body = {"version": BASELINE_VERSION,
            "_comment": "Pattern ratchet baseline (tools/check_patterns.py): per-rule per-file match counts of the day the rule "
                        "was frozen. Counts may only shrink: `python tools/check_patterns.py --update-baseline` records shrinks "
                        "and refuses growth. Never edit by hand to silence a new finding: fix the finding.",
            "rules": {r: {f: n for f, n in sorted(files.items()) if n > 0} for r, files in sorted(counts.items())}}
    return json.dumps(body, indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def write_baseline(root: Path, counts: dict[str, dict[str, int]]) -> None:
    path = root / BASELINE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_baseline(counts), encoding="utf-8")


def compare(counts: dict[str, dict[str, int]], baseline: dict[str, dict[str, int]] | None,
            rules: Iterable[Rule] = RULES) -> tuple[list[str], list[str]]:
    """(growth problems, stale problems). Growth = more matches than the baseline allows (a missing file/rule counts as 0).
    Stale = fewer than the baseline records (run --update-baseline)."""
    baseline = baseline or {}
    growth: list[str] = []
    stale: list[str] = []
    for r in rules:
        now = counts.get(r.name, {})
        base = baseline.get(r.name, {})
        for f in sorted(set(now) | set(base)):
            n, b = now.get(f, 0), base.get(f, 0)
            if n > b:
                growth.append(f"{f}: [{r.name}] {n} match(es), baseline allows {b} (audit class {r.guards}): {r.purpose.split(':')[0]}")
            elif n < b:
                stale.append(f"{f}: [{r.name}] {n} match(es) but baseline records {b}: record the shrink with "
                             f"`python tools/check_patterns.py --update-baseline`")
    return growth, stale


def shrink_only(counts: dict[str, dict[str, int]], baseline: dict[str, dict[str, int]]) -> tuple[dict[str, dict[str, int]], list[str]]:
    """The new baseline = min(now, old) per rule per file; the refusal messages are the growth (files/rules not in the old
    baseline are growth too). A rule absent from the old baseline is carried over unchanged only by --freeze-rule."""
    new: dict[str, dict[str, int]] = {}
    refused: list[str] = []
    for rule, old in baseline.items():
        now = counts.get(rule, {})
        new[rule] = {}
        for f in sorted(set(now) | set(old)):
            n, b = now.get(f, 0), old.get(f, 0)
            if n > b:
                refused.append(f"{f}: [{rule}] {n} > baseline {b}")
            lowered = min(n, b)
            if lowered > 0:
                new[rule][f] = lowered
    return new, refused


# --------------------------------------------------------------------------- check (used by tools/check_deps.py)
def check(root: Path = ROOT, rules: Iterable[Rule] = RULES) -> tuple[list[str], int]:
    """(problems, number of baselined matches). A problem is growth over the baseline or a stale (shrunk) baseline entry.
    A rule that is missing from the baseline file has an implicit empty baseline: every match is growth."""
    rules = tuple(rules)
    counts = counts_of(scan(root, rules))
    baseline = read_baseline(root)
    growth, stale = compare(counts, baseline, rules)
    problems = [f"{g}  [new anti-pattern instance: fix it, do not baseline it]" for g in growth]
    problems += [f"{BASELINE_PATH}: {s}" for s in stale]
    if baseline is None and not growth:
        problems.append(f"{BASELINE_PATH}: missing (create it once with `python tools/check_patterns.py --update-baseline`)")
    allowed = sum(min(n, (baseline or {}).get(r.name, {}).get(f, 0)) for r in rules for f, n in counts.get(r.name, {}).items())
    return sorted(problems), allowed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pattern ratchet: counts of anti-pattern matches may only shrink.")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--list", nargs="?", const="*", metavar="RULE", help="print every match (optionally of one rule)")
    ap.add_argument("--summary", action="store_true", help="per-rule totals, baseline vs now")
    ap.add_argument("--update-baseline", action="store_true", help="record shrinks only; refuses growth")
    ap.add_argument("--freeze-rule", metavar="RULE", help="add one new rule's current counts to the baseline")
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()

    if args.list is not None:
        names = [r.name for r in RULES] if args.list == "*" else [args.list]
        unknown = [n for n in names if n not in RULES_BY_NAME]
        if unknown:
            print(f"unknown rule {unknown}; known: {sorted(RULES_BY_NAME)}", file=sys.stderr)
            return 2
        found = scan(root, [RULES_BY_NAME[n] for n in names])
        for n in names:
            for f, ms in sorted(found[n].items()):
                for line, text in ms:
                    print(f"{f}:{line}: [{n}] {text}")
        return 0

    counts = counts_of(scan(root))
    baseline = read_baseline(root)

    if args.summary:
        for r in RULES:
            now = sum(counts[r.name].values())
            base = sum((baseline or {}).get(r.name, {}).values())
            print(f"{r.name:28s} {r.guards:3s} baseline {base:5d}  now {now:5d}  files {len(counts[r.name])}")
        return 0

    if args.freeze_rule:
        if args.freeze_rule not in RULES_BY_NAME:
            print(f"unknown rule {args.freeze_rule!r}; known: {sorted(RULES_BY_NAME)}", file=sys.stderr)
            return 2
        if baseline is not None and args.freeze_rule in baseline:
            print(f"refused: rule {args.freeze_rule!r} is already in the baseline (use --update-baseline to shrink it)",
                  file=sys.stderr)
            return 1
        merged = dict(baseline or {})
        merged[args.freeze_rule] = counts[args.freeze_rule]
        write_baseline(root, merged)
        print(f"froze {args.freeze_rule}: {sum(counts[args.freeze_rule].values())} matches in {len(counts[args.freeze_rule])} files")
        return 0

    if args.update_baseline:
        if baseline is None:  # the one-time freeze of every rule at today's counts
            write_baseline(root, counts)
            print(f"created {BASELINE_PATH}: " + ", ".join(f"{r}={sum(c.values())}" for r, c in sorted(counts.items())))
            return 0
        missing = [r.name for r in RULES if r.name not in baseline]
        if missing:
            print(f"refused: rules not in the baseline yet: {missing}; freeze each with --freeze-rule <name>", file=sys.stderr)
            return 1
        new, refused = shrink_only(counts, baseline)
        if refused:
            for ln in refused:
                print(f"refused (growth): {ln}", file=sys.stderr)
            print("baseline NOT updated: fix the new matches first (the baseline only shrinks).", file=sys.stderr)
            return 1
        write_baseline(root, new)
        print("baseline updated (shrinks only): " + ", ".join(f"{r}={sum(c.values())}" for r, c in sorted(new.items())))
        return 0

    problems, allowed = check(root)
    for p in problems:
        print(f"error[pattern-ratchet] — {p}")
    print(f"check_patterns: {len(RULES)} rules, {len(problems)} problems, {allowed} baselined matches")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""
PURPOSE:  Failure registry and no-repeat check for every experiment in the project. A failed experiment is closed with its
          scope (mechanism, timeframe, target, model class...), the level/location where it failed, the verified cause
          and the explicit reopen conditions (R1, R2...). A new experiment card must declare, for every closed failure that
          covers it, either which reopen condition now holds (`reopens: F-0003:R1`) or why it is not covered
          (`not_covered_by: F-0003=reason`). `check` fails otherwise, so a failure is never re-run by accident.
TAGS:     failure registry, negative results, no-repeat, reopen conditions, experiment card, research protocol, ledger
PITFALLS: Coverage is decided by tag overlap (mechanism AND timeframe AND target AND model_class, `any` matches all), so
          tags must come from the vocabulary in the registry rows - a new spelling silently escapes coverage; `search`
          before writing a card. Registry rows are append-only (protocol): close/reopen by adding a row, never editing.

Usage:
    python tools/experiment_registry.py validate                     # registry schema + every card in the repo
    python tools/experiment_registry.py search momentum 1h           # rows whose tags/claim contain all words
    python tools/experiment_registry.py check path/to/card.md        # one card against the registry
    python tools/experiment_registry.py covering path/to/card.md     # which closed failures cover this card
"""
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "docs", "research", "failure_registry.csv")
CARD_GLOBS = ("docs/research/cards/*.md", "research/*/cards/*.md")

COLUMNS = ["id", "source", "date", "claim", "mechanism", "timeframe", "target", "model_class", "universe", "period",
           "cost_model", "failure_level", "failure_location", "evidence", "verified_cause", "cause_test", "invariants",
           "reopen_if", "status", "supersedes"]
TAG_FIELDS = ("mechanism", "timeframe", "target", "model_class")
LEVELS = {"data", "representation", "optimization", "head", "generalization", "economics", "implementation"}
STATUSES = {"closed", "closed-unverified-cause", "reopened"}
ID_RE = re.compile(r"^F-\d{4}$")
REOPEN_RE = re.compile(r"^(R\d+):\s*\S")


def tags(value):
    """'a; b, c' -> {'a','b','c'} (lower-case, trimmed)."""
    return {t.strip().lower() for t in re.split(r"[;,]", value or "") if t.strip()}


def reopen_clauses(value):
    """'R1: maker cost <= 2bp | R2: new data source' -> {'R1': '...', 'R2': '...'}."""
    out = {}
    for part in (value or "").split("|"):
        part = part.strip()
        m = REOPEN_RE.match(part)
        if m:
            out[m.group(1)] = part[len(m.group(1)) + 1:].strip()
    return out


def load_registry(path=REGISTRY):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_registry(rows, header=None):
    errors = []
    if header is not None and header != COLUMNS:
        errors.append(f"registry header must be exactly {COLUMNS}")
    seen = set()
    for i, r in enumerate(rows, start=2):
        rid = r.get("id", "")
        where = f"row {i} ({rid})"
        if not ID_RE.match(rid):
            errors.append(f"{where}: id must look like F-0001")
        if rid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(rid)
        for col in ("source", "claim", "evidence", "invariants", "reopen_if", "status", "failure_level"):
            if not (r.get(col) or "").strip():
                errors.append(f"{where}: '{col}' is empty")
        for col in TAG_FIELDS:
            if not tags(r.get(col)):
                errors.append(f"{where}: '{col}' needs at least one tag (or 'any')")
        if r.get("failure_level") and r["failure_level"].strip() not in LEVELS:
            errors.append(f"{where}: failure_level must be one of {sorted(LEVELS)}")
        if r.get("status") and r["status"].strip() not in STATUSES:
            errors.append(f"{where}: status must be one of {sorted(STATUSES)}")
        if r.get("status", "").strip() == "closed" and not (r.get("verified_cause") or "").strip():
            errors.append(f"{where}: status 'closed' needs verified_cause (else 'closed-unverified-cause')")
        if r.get("reopen_if") and not reopen_clauses(r["reopen_if"]):
            errors.append(f"{where}: reopen_if must be 'R1: condition | R2: ...'")
        for sid in tags(r.get("supersedes")):
            if sid.upper() not in seen:
                errors.append(f"{where}: supersedes unknown/later id {sid.upper()}")
    return errors


def parse_card(path):
    """Front matter between the first two '---' lines: 'key: a, b' or 'key: [a, b]'."""
    text = open(path, encoding="utf-8").read()
    m = re.match(r"\s*---\n(.*?)\n---", text, re.S)
    if not m:
        raise ValueError(f"{path}: no front matter (--- ... ---) at the top")
    meta = {}
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{path}: bad front-matter line {line!r}")
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip().strip("[]").strip()
    return meta


def _overlap(a, b):
    return "any" in a or "any" in b or bool(a & b)


def covering(meta, rows):
    """Active closed failures whose scope overlaps the card on every tag field."""
    superseded = set().union(*[tags(r.get("supersedes")) for r in rows]) if rows else set()
    out = []
    for r in rows:
        if r["status"].strip() == "reopened" or r["id"].lower() in superseded:
            continue
        if all(_overlap(tags(meta.get(f)), tags(r.get(f))) for f in TAG_FIELDS):
            out.append(r)
    return out


def check_card(path, rows):
    meta = parse_card(path)
    errors = [f"{path}: '{f}' is required in the front matter" for f in TAG_FIELDS if not tags(meta.get(f))]
    if errors:
        return errors
    by_id = {r["id"]: r for r in rows}
    reopens = {}
    for item in tags(meta.get("reopens")):
        fid, _, clause = item.upper().partition(":")
        reopens[fid] = clause
    not_covered = {}
    for item in (meta.get("not_covered_by") or "").split(";"):
        if item.strip():
            fid, _, why = item.partition("=")
            not_covered[fid.strip().upper()] = why.strip()
    for fid, clause in reopens.items():
        if fid not in by_id:
            errors.append(f"{path}: reopens unknown {fid}")
        elif clause not in reopen_clauses(by_id[fid]["reopen_if"]):
            errors.append(f"{path}: {fid} has no reopen condition {clause or '(missing :R#)'} — "
                          f"available: {reopen_clauses(by_id[fid]['reopen_if'])}")
    for fid, why in not_covered.items():
        if fid not in by_id:
            errors.append(f"{path}: not_covered_by unknown {fid}")
        elif len(why) < 15:
            errors.append(f"{path}: not_covered_by {fid} needs a real reason (which scope/invariant differs)")
    for r in covering(meta, rows):
        if r["id"] not in reopens and r["id"] not in not_covered:
            errors.append(f"{path}: covered by closed failure {r['id']} ({r['claim'][:80]}) — declare "
                          f"'reopens: {r['id']}:R#' with a condition from its reopen_if, or 'not_covered_by: "
                          f"{r['id']}=<why the scope differs>'")
    return errors


def all_cards():
    out = []
    for g in CARD_GLOBS:
        out += sorted(glob.glob(os.path.join(ROOT, g)))
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    rows = load_registry()
    if cmd == "validate":
        header = None
        if os.path.exists(REGISTRY):
            with open(REGISTRY, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
        errors = validate_registry(rows, header)
        for card in all_cards():
            try:
                errors += check_card(card, rows)
            except ValueError as e:
                errors.append(str(e))
        print("\n".join(errors) if errors else f"registry OK: {len(rows)} rows, {len(all_cards())} cards")
        return 1 if errors else 0
    if cmd == "search":
        words = [w.lower() for w in args]
        for r in rows:
            hay = " ".join(str(v) for v in r.values()).lower()
            if all(w in hay for w in words):
                print(f"{r['id']} [{r['status']}] {r['claim'][:90]}\n    level={r['failure_level']} "
                      f"where={r['failure_location']}\n    reopen_if: {r['reopen_if']}")
        return 0
    if cmd == "covering":
        for r in covering(parse_card(args[0]), rows):
            print(f"{r['id']} {r['claim'][:100]}\n    reopen_if: {r['reopen_if']}")
        return 0
    if cmd == "check":
        errors = check_card(args[0], rows)
        print("\n".join(errors) if errors else "card OK")
        return 1 if errors else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())

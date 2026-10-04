"""
PURPOSE:  Edit notebook code/markdown cells as plain files and write them back without reformatting the .ipynb: list cells,
          extract chosen cells to <dir>/<nb>_<i>.py, inject them back, insert new cells. Preserves the notebook's own JSON
          layout (compact or indent=1) and non-ASCII text, so diffs show only the edited cells.
TAGS:     notebook, ipynb, cells, extract, inject, edit notebook, json format, round-trip
PITFALLS: Cell indices are 0-based (same as MAP.md `cell N`) and shift after `insert`; inject only cells you extracted from
          the same version of the notebook. `check` must print OK before editing a notebook this tool has not seen.

Usage:
    python tools/nb_cells.py list    NB.ipynb [--grep TEXT]
    python tools/nb_cells.py check   NB.ipynb                      # load -> dump is byte-identical?
    python tools/nb_cells.py extract NB.ipynb DIR 20 24            # -> DIR/<stem>_20.py, DIR/<stem>_24.py
    python tools/nb_cells.py inject  NB.ipynb DIR 20 24            # DIR/<stem>_<i>.py -> cell i
    python tools/nb_cells.py insert  NB.ipynb AFTER {code|markdown} FILE [--id CELL_ID]
"""
import argparse
import json
import sys
from pathlib import Path


def _load(path):
    raw = Path(path).read_text(encoding="utf-8")
    return raw, json.loads(raw)


def _dump(nb, raw):
    """Same layout the file already uses: compact (no newline after '{') or indent=1 (Jupyter's default)."""
    if raw.startswith("{\n"):
        out = json.dumps(nb, ensure_ascii=False, indent=1)
        return out + "\n" if raw.endswith("\n") else out
    return json.dumps(nb, ensure_ascii=False, separators=(",", ":"))


def _write(path, nb, raw):
    Path(path).write_text(_dump(nb, raw), encoding="utf-8")


def _cell_file(directory, nb_path, i):
    stem = Path(nb_path).stem.replace(" ", "_")
    return Path(directory) / f"{stem}_{i}.py"


def _new_cell(kind, src, cell_id):
    lines = src.splitlines(keepends=True)
    if kind == "code":
        return {"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {"id": cell_id},
                "outputs": [], "source": lines}
    return {"cell_type": "markdown", "id": cell_id, "metadata": {"id": cell_id}, "source": lines}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("nb"); p.add_argument("--grep")
    p = sub.add_parser("check"); p.add_argument("nb")
    for name in ("extract", "inject"):
        p = sub.add_parser(name); p.add_argument("nb"); p.add_argument("dir"); p.add_argument("cells", nargs="+", type=int)
    p = sub.add_parser("insert"); p.add_argument("nb"); p.add_argument("after", type=int)
    p.add_argument("kind", choices=("code", "markdown")); p.add_argument("file"); p.add_argument("--id")
    a = ap.parse_args(argv)

    raw, nb = _load(a.nb)
    cells = nb["cells"]
    if a.cmd == "list":
        for i, c in enumerate(cells):
            src = "".join(c["source"])
            if a.grep and a.grep not in src:
                continue
            first = next((ln for ln in src.splitlines() if ln.strip()), "")
            print(f"{i:3d} {c['cell_type']:8s} {len(src.splitlines()):5d} lines | {first[:100]}")
    elif a.cmd == "check":
        ok = _dump(nb, raw) == raw
        print("OK: round-trips byte-identically" if ok else "DIFFERS: do not edit with this tool")
        return 0 if ok else 1
    elif a.cmd == "extract":
        Path(a.dir).mkdir(parents=True, exist_ok=True)
        for i in a.cells:
            f = _cell_file(a.dir, a.nb, i)
            f.write_text("".join(cells[i]["source"]), encoding="utf-8")
            print(f)
    elif a.cmd == "inject":
        for i in a.cells:
            text = _cell_file(a.dir, a.nb, i).read_text(encoding="utf-8")
            # keep the field type: some notebooks (main) store source as one string, not a list of lines
            cells[i]["source"] = text if isinstance(cells[i]["source"], str) else text.splitlines(keepends=True)
        _write(a.nb, nb, raw)
        print(f"injected {a.cells} into {a.nb}")
    elif a.cmd == "insert":
        cell_id = a.id or f"cell{len(cells)}{abs(hash(a.file)) % 10**6}"
        if any(c.get("id") == cell_id for c in cells):
            raise SystemExit(f"cell id {cell_id!r} already exists")
        cells.insert(a.after + 1, _new_cell(a.kind, Path(a.file).read_text(encoding="utf-8").rstrip("\n"), cell_id))
        _write(a.nb, nb, raw)
        print(f"inserted {a.kind} cell {cell_id!r} at index {a.after + 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
PURPOSE:  AST-based loader of notebook definitions (functions, classes, imports, literal assignments) without running example cells; used to bring signal_evaluation_axis into the lab namespace.
TAGS:     load_notebook_defs, _notebook_code, ast, notebook definitions, signal_evaluation_axis, safe load
PITFALLS: Imports json/ast/warnings itself. Executing the returned code (exec(compile(load_notebook_defs(path), ...))) defines the axis functions in the CALLER's globals. Other agents may turn signal_evaluation_axis into a package; this loader then no longer applies to it. Executed into the notebook's shared namespace by discovery/_loader.py (never imported on its own): names from the axis/pipeline notebooks resolve at call time. Extracted verbatim from signal_discovery_lab.ipynb cell 2 (section 1).
"""
import json, ast, warnings


def _notebook_code(path):
    nb = json.load(open(path, encoding="utf-8"))
    return "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def load_notebook_defs(path):
    """يستخرج تعريفات الدوال/الأصناف والاستيرادات والقيم الحرفية فقط من دفتر
    — يتجاهل خلايا الأمثلة/السائقة (متغيّرات تفاعلية غير معرَّفة، أو استدعاءات
    شبكية حقيقية). آمن لتحميل signal_evaluation_axis دون تشغيله بالكامل."""
    code_text = _notebook_code(path)
    code_text = "\n".join(l for l in code_text.split("\n")
                          if not l.strip().startswith(("%", "!")))
    tree = ast.parse(code_text)
    keep_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    literal_types = (ast.Tuple, ast.List, ast.Constant, ast.Dict, ast.Set)
    kept, futures = [], []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            futures.append(node)
        elif isinstance(node, keep_types):
            kept.append(node)
        elif isinstance(node, ast.Assign) and isinstance(node.value, literal_types):
            kept.append(node)
    mod = ast.Module(body=futures[:1] + kept, type_ignores=[])
    ast.fix_missing_locations(mod)
    return ast.unparse(mod)

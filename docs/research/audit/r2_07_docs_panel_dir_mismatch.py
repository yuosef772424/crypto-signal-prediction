"""Repro R2-07: the Colab instructions name a results folder that the notebook never creates.

main.ipynb cell 17 builds run_dir = ".../crypto_model_v1" + ("_"+TARGET_MODE) + ("_s{REG_TARGET_SCALE:g}" if scale != 1) + "_am",
and cell 38 (panel) writes to run_dir + "_panel". For the documented run (TARGET_MODE=None, REG_TARGET_SCALE=100,
ANTI_MEMORIZATION=True) that is .../crypto_model_v1_s100_am_panel, but docs/research/hourly_1h.md tells the operator to
`cd .../crypto_model_v1_am_panel && zip ...` (the `_s100` part, added by the scale change, is missing) and to delete the
old `..._panel` folder by name. The `cd` fails, so the results are not zipped (and a stale-folder cleanup can hit the wrong
directory).

Expected (after fix): the path in the docs equals the path the notebook computes.
Actual (b14b9bb): docs .../crypto_model_v1_am_panel vs notebook .../crypto_model_v1_s100_am_panel.

Run:  python docs/research/audit/r2_07_docs_panel_dir_mismatch.py   (<1 s)
"""
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
nb = json.load(open(os.path.join(ROOT, "main.ipynb")))
cell = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and '"run_dir": "/content/drive' in "".join(c["source"]))
m = re.search(r'"run_dir": (.*?),\n\s*"epochs"', cell, re.S)
expr = "(" + m.group(1) + ")"
env = {"TARGET_MODE": None, "REG_TARGET_SCALE": 100.0, "_am": True}
env["globals"] = lambda: env
run_dir = eval(expr, env)                                       # the notebook's own expression, documented settings
panel_dir = run_dir + "_panel"
docs = open(os.path.join(ROOT, "docs", "research", "hourly_1h.md"), encoding="utf-8").read()
doc_dir = re.search(r"cd (/content/drive/MyDrive/training_runs/\S+?) &&", docs).group(1)
print(f"notebook: {panel_dir}\ndocs    : {doc_dir}")
assert panel_dir == doc_dir, "DEFECT: docs point to a folder the notebook does not create"
print("PASS")
